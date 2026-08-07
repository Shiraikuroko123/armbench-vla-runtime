"""Component-level LIBERO Cartesian action adapter for the Panda guard path."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from armbench.vla.types import ActionChunk, DROID_ACTION_DIM, VLAObservation

FloatArray = NDArray[np.float64]
LIBERO_ACTION_DIM = 7
LIBERO_ACTION_SPACE_ID = "libero.ee_delta_pose_gripper.v1"
LIBERO_CONTROLLER_SEMANTICS_ID = (
    "libero-f78abd68.robosuite-1.4.1.osc-pose.v1"
)
PANDA_KINEMATIC_CONTROL_POINT_ID = "mujoco-menagerie.hand-body-origin.v1"
LIBERO_CONTROL_DT_S = 0.05
LIBERO_TRANSLATION_DELTA_SCALE_M = 0.05
LIBERO_ROTATION_DELTA_SCALE_RAD = 0.50


class PandaCartesianKinematics(Protocol):
    """Minimal kinematic surface required by the adapter."""

    dof: int
    lower_limits: FloatArray
    upper_limits: FloatArray
    velocity_limits: FloatArray

    def validate_configuration(self, q: ArrayLike) -> FloatArray: ...

    def within_limits(self, q: ArrayLike, *, atol: float = 1e-12) -> bool: ...

    def hand_pose(self, q: ArrayLike) -> tuple[FloatArray, FloatArray]: ...

    def hand_jacobian(self, q: ArrayLike) -> FloatArray: ...


@dataclass(frozen=True)
class CartesianAdapterConfig:
    """Source-attested LIBERO OSC_POSE mapping from actions to a hand twist.

    The default scale, clip, frame, and gripper semantics match LIBERO commit
    f78abd68 with robosuite 1.4.1. The local differential-IK velocity adapter is
    still not dynamically equivalent to robosuite's torque-level OSC controller.
    """

    control_dt_s: float = LIBERO_CONTROL_DT_S
    translation_delta_scale_m: float = LIBERO_TRANSLATION_DELTA_SCALE_M
    rotation_delta_scale_rad: float = LIBERO_ROTATION_DELTA_SCALE_RAD
    damping: float = 0.05
    normalized_action_limit: float = 1.0
    joint_velocity_limit_scale: float = 0.50
    joint_limit_margin_rad: float = 0.02
    action_frame: str = "base"

    def __post_init__(self) -> None:
        positive = {
            "control_dt_s": self.control_dt_s,
            "translation_delta_scale_m": self.translation_delta_scale_m,
            "rotation_delta_scale_rad": self.rotation_delta_scale_rad,
            "damping": self.damping,
            "normalized_action_limit": self.normalized_action_limit,
            "joint_velocity_limit_scale": self.joint_velocity_limit_scale,
        }
        if any(not np.isfinite(value) or value <= 0.0 for value in positive.values()):
            raise ValueError("adapter scales, timing, damping, and limits must be finite and positive")
        if self.joint_velocity_limit_scale > 1.0:
            raise ValueError("joint_velocity_limit_scale cannot exceed 1")
        if (
            not np.isfinite(self.joint_limit_margin_rad)
            or self.joint_limit_margin_rad < 0.0
        ):
            raise ValueError("joint_limit_margin_rad must be finite and nonnegative")
        if self.action_frame not in {"base", "tool"}:
            raise ValueError("action_frame must be 'base' or 'tool'")


@dataclass(frozen=True)
class CartesianAdapterStep:
    index: int
    input_clipped: bool
    velocity_scale: float
    desired_twist: FloatArray
    achieved_twist: FloatArray
    residual_norm: float
    minimum_singular_value: float
    joint_velocity: FloatArray
    q_before: FloatArray
    q_after: FloatArray


@dataclass(frozen=True)
class CartesianAdapterResult:
    action_space_id: str
    chunk: ActionChunk
    predicted_positions: FloatArray
    steps: tuple[CartesianAdapterStep, ...]
    adapter_latency_ms: float

    @property
    def clipped_input_steps(self) -> int:
        return sum(step.input_clipped for step in self.steps)

    @property
    def velocity_limited_steps(self) -> int:
        return sum(step.velocity_scale < 1.0 - 1e-12 for step in self.steps)

    @property
    def max_residual_norm(self) -> float:
        return max(step.residual_norm for step in self.steps)

    def metrics(self) -> dict[str, object]:
        return {
            "scope": "component_cartesian_adapter_only",
            "action_space_id": self.action_space_id,
            "controller_semantics_id": LIBERO_CONTROLLER_SEMANTICS_ID,
            "kinematic_control_point_id": PANDA_KINEMATIC_CONTROL_POINT_ID,
            "horizon": self.chunk.horizon,
            "source": self.chunk.source,
            "clipped_input_steps": self.clipped_input_steps,
            "velocity_limited_steps": self.velocity_limited_steps,
            "max_residual_norm": self.max_residual_norm,
            "minimum_singular_value": min(
                step.minimum_singular_value for step in self.steps
            ),
            "adapter_latency_ms": self.adapter_latency_ms,
        }


class PandaCartesianActionAdapter:
    """Map normalized LIBERO-style Cartesian deltas to Panda joint velocities."""

    def __init__(
        self,
        robot: PandaCartesianKinematics,
        config: CartesianAdapterConfig = CartesianAdapterConfig(),
    ) -> None:
        if robot.dof != 7:
            raise ValueError("Panda Cartesian adapter requires seven arm joints")
        self.robot = robot
        self.config = config
        lower = np.asarray(robot.lower_limits, dtype=float)
        upper = np.asarray(robot.upper_limits, dtype=float)
        velocity = np.asarray(robot.velocity_limits, dtype=float)
        if lower.shape != (7,) or upper.shape != (7,) or velocity.shape != (7,):
            raise ValueError("robot limits must be finite seven-vectors")
        if not np.all(np.isfinite(np.concatenate((lower, upper, velocity)))):
            raise ValueError("robot limits must be finite seven-vectors")
        if np.any(lower + config.joint_limit_margin_rad >= upper - config.joint_limit_margin_rad):
            raise ValueError("joint limit margin leaves an empty feasible interval")

    def _desired_twist(self, action: FloatArray, q: FloatArray) -> FloatArray:
        normalized = np.clip(
            action[:6],
            -self.config.normalized_action_limit,
            self.config.normalized_action_limit,
        ) / self.config.normalized_action_limit
        delta = np.concatenate(
            (
                normalized[:3] * self.config.translation_delta_scale_m,
                normalized[3:] * self.config.rotation_delta_scale_rad,
            )
        )
        twist = delta / self.config.control_dt_s
        if self.config.action_frame == "tool":
            _, rotation = self.robot.hand_pose(q)
            twist = np.concatenate(
                (rotation @ twist[:3], rotation @ twist[3:])
            )
        return twist

    def _velocity_scale(self, q: FloatArray, velocity: FloatArray) -> float:
        bounds = (
            np.asarray(self.robot.velocity_limits, dtype=float)
            * self.config.joint_velocity_limit_scale
        )
        scale = 1.0
        moving = np.abs(velocity) > 1e-15
        if np.any(moving):
            scale = min(scale, float(np.min(bounds[moving] / np.abs(velocity[moving]))))

        lower = np.asarray(self.robot.lower_limits, dtype=float) + self.config.joint_limit_margin_rad
        upper = np.asarray(self.robot.upper_limits, dtype=float) - self.config.joint_limit_margin_rad
        for index, value in enumerate(velocity):
            if value > 1e-15:
                available = max(0.0, (upper[index] - q[index]) / self.config.control_dt_s)
                scale = min(scale, available / value)
            elif value < -1e-15:
                available = max(0.0, (q[index] - lower[index]) / self.config.control_dt_s)
                scale = min(scale, available / -value)
        return float(np.clip(scale, 0.0, 1.0))

    def adapt(
        self,
        actions: ArrayLike,
        q_start: ArrayLike,
        *,
        source: str,
        observation_sequence_id: int,
        inference_latency_ms: float,
        received_at_s: float | None = None,
    ) -> CartesianAdapterResult:
        """Adapt one finite Hx7 chunk while preserving timing provenance."""

        started = time.perf_counter()
        values = np.asarray(actions, dtype=float)
        if values.ndim != 2 or values.shape[1] != LIBERO_ACTION_DIM:
            raise ValueError("LIBERO actions must have shape (horizon, 7)")
        if values.shape[0] == 0 or not np.all(np.isfinite(values)):
            raise ValueError("LIBERO actions must be nonempty and finite")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a nonempty string")
        if observation_sequence_id < 0:
            raise ValueError("observation_sequence_id must be nonnegative")
        if not np.isfinite(inference_latency_ms) or inference_latency_ms < 0.0:
            raise ValueError("inference_latency_ms must be finite and nonnegative")

        q = self.robot.validate_configuration(q_start).copy()
        if not self.robot.within_limits(q):
            raise ValueError("q_start violates robot joint limits")
        output = np.zeros((values.shape[0], DROID_ACTION_DIM), dtype=float)
        predicted = [q.copy()]
        records: list[CartesianAdapterStep] = []
        limit = self.config.normalized_action_limit

        for index, action in enumerate(values):
            q_before = q.copy()
            desired_twist = self._desired_twist(action, q)
            jacobian = np.asarray(self.robot.hand_jacobian(q), dtype=float)
            if jacobian.shape != (6, 7) or not np.all(np.isfinite(jacobian)):
                raise ValueError("robot hand Jacobian must be a finite 6x7 matrix")
            regularized = (
                jacobian @ jacobian.T
                + self.config.damping**2 * np.eye(6, dtype=float)
            )
            joint_velocity = jacobian.T @ np.linalg.solve(
                regularized, desired_twist
            )
            velocity_scale = self._velocity_scale(q, joint_velocity)
            joint_velocity = joint_velocity * velocity_scale
            q = q + self.config.control_dt_s * joint_velocity
            if not self.robot.within_limits(q, atol=1e-10):
                raise RuntimeError("adapter produced a joint-limit violation")

            achieved_twist = jacobian @ joint_velocity
            singular_values = np.linalg.svd(jacobian, compute_uv=False)
            input_clipped = bool(np.any(np.abs(action) > limit + 1e-12))
            normalized_gripper = float(np.clip(action[6] / limit, -1.0, 1.0))
            # LIBERO: -1=open, +1=closed. Local Panda: 0=closed, 1=open.
            gripper = (1.0 - normalized_gripper) / 2.0
            output[index, :7] = joint_velocity
            output[index, 7] = gripper
            predicted.append(q.copy())
            records.append(
                CartesianAdapterStep(
                    index=index,
                    input_clipped=input_clipped,
                    velocity_scale=velocity_scale,
                    desired_twist=desired_twist.copy(),
                    achieved_twist=achieved_twist.copy(),
                    residual_norm=float(np.linalg.norm(desired_twist - achieved_twist)),
                    minimum_singular_value=float(np.min(singular_values)),
                    joint_velocity=joint_velocity.copy(),
                    q_before=q_before,
                    q_after=q.copy(),
                )
            )

        chunk = ActionChunk(
            actions=output,
            source=f"{source}|panda_cartesian_adapter",
            observation_sequence_id=observation_sequence_id,
            inference_latency_ms=float(inference_latency_ms),
            received_at_s=(time.monotonic() if received_at_s is None else received_at_s),
        )
        return CartesianAdapterResult(
            action_space_id=LIBERO_ACTION_SPACE_ID,
            chunk=chunk,
            predicted_positions=np.asarray(predicted),
            steps=tuple(records),
            adapter_latency_ms=(time.perf_counter() - started) * 1000.0,
        )


def run_cartesian_adapter_smoke() -> dict[str, object]:
    """Run a deterministic CPU-only adapter and guard integration check."""

    from armbench.mujoco_sim import MuJoCoCollisionChecker, MuJoCoPanda
    from armbench.mujoco_sim.scenarios import mujoco_scenarios
    from armbench.vla.guard import ActionChunkGuard, GuardConfig

    robot = MuJoCoPanda.create(obstacles=())
    q_start = mujoco_scenarios()["free_space"].start
    config = CartesianAdapterConfig()
    actions = np.zeros((10, LIBERO_ACTION_DIM), dtype=float)
    actions[:, 0] = 0.10
    actions[:, 6] = -1.0
    captured_at_s = 100.0
    adapter = PandaCartesianActionAdapter(robot, config)
    adapted = adapter.adapt(
        actions,
        q_start,
        source="scripted_libero_cartesian",
        observation_sequence_id=0,
        inference_latency_ms=40.0,
        received_at_s=captured_at_s + 0.04,
    )
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    observation = VLAObservation(
        exterior_image=image,
        wrist_image=image,
        joint_position=q_start,
        gripper_position=np.array([1.0]),
        prompt="component adapter smoke",
        sequence_id=0,
        captured_at_s=captured_at_s,
    )
    checker = MuJoCoCollisionChecker(robot, resolution=0.02)
    guarded = ActionChunkGuard(
        checker,
        GuardConfig(
            control_dt_s=config.control_dt_s,
            deadline_ms=200.0,
            joint_velocity_clip_rad_s=float(np.max(robot.velocity_limits)),
        ),
    ).guard(q_start, 1.0, observation, adapted.chunk)
    start_hand = robot.hand_position(q_start)
    end_hand = robot.hand_position(guarded.predicted_positions[-1])
    displacement = end_hand - start_hand
    passed = bool(
        guarded.safe_after_guard
        and checker.path_is_valid(guarded.predicted_positions)
        and displacement[0] > 1e-4
        and np.all(np.isfinite(guarded.guarded_actions))
    )
    return {
        "passed": passed,
        "scope": "scripted_cartesian_adapter_component_only",
        "policy_checkpoint_used": False,
        "adapter": adapted.metrics(),
        "guard": guarded.metrics(),
        "hand_displacement_m": displacement.tolist(),
        "limitations": [
            "No pi0.5 inference or LIBERO task-success claim.",
            "LIBERO action semantics are source-attested; local differential IK is not torque-OSC equivalence.",
            "The local control point is the Menagerie hand-body origin, not the robosuite grip_site.",
            "Collision checking is resolution-bounded edge sampling, not continuous certification.",
        ],
    }
