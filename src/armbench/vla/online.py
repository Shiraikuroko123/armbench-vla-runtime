"""Receding-horizon VLA execution against live MuJoCo state and cameras."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
from time import monotonic

import imageio.v2 as imageio
import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from armbench.mujoco_sim.benchmark import inflate_obstacles
from armbench.mujoco_sim.collision import MuJoCoCollisionChecker
from armbench.mujoco_sim.execution import DEFAULT_KD, DEFAULT_KP
from armbench.mujoco_sim.model import MuJoCoPanda, VLA_EXTERNAL_CAMERA
from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.vla.guard import ActionChunkGuard, GuardConfig
from armbench.vla.observation import MuJoCoDroidObservationBuilder
from armbench.vla.policy import ActionChunkPolicy
from armbench.vla.runtime import VLARuntimeSupervisor
from armbench.vla.types import ActionChunk, VLAObservation

FloatArray = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]
OBSERVATION_THUMBNAIL_SHAPE = (16, 16, 3)


def _image_sha256(image: UInt8Array) -> str:
    return hashlib.sha256(image.tobytes(order="C")).hexdigest()


def _mean_abs_image_delta(
    image: UInt8Array, previous: UInt8Array | None
) -> float | None:
    if previous is None:
        return None
    difference = image.astype(np.int16) - previous.astype(np.int16)
    return float(np.mean(np.abs(difference)))


def _image_thumbnail(image: UInt8Array) -> UInt8Array:
    height, width, channels = image.shape
    target_height, target_width, target_channels = OBSERVATION_THUMBNAIL_SHAPE
    if channels != target_channels:
        raise ValueError("observation image must have three channels")
    if height % target_height or width % target_width:
        raise ValueError("observation image cannot be evenly downsampled")
    blocks = image.reshape(
        target_height,
        height // target_height,
        target_width,
        width // target_width,
        channels,
    )
    return np.rint(blocks.mean(axis=(1, 3))).astype(np.uint8)


class _OnlineVideoRecorder:
    def __init__(
        self,
        model: mujoco.MjModel,
        path: Path | None,
        *,
        fps: int,
        render_size: tuple[int, int],
    ) -> None:
        self.path = path
        self.fps = fps
        self._next_frame_time: float | None = None
        self._renderer: mujoco.Renderer | None = None
        self._writer: object | None = None
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        width, height = render_size
        self._renderer = mujoco.Renderer(
            model, height=height, width=width
        )
        try:
            self._writer = imageio.get_writer(
                str(path),
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=None,
            )
        except Exception:
            self._renderer.close()
            self._renderer = None
            raise

    def __enter__(self) -> "_OnlineVideoRecorder":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def capture(self, data: mujoco.MjData) -> None:
        if self._renderer is None or self._writer is None:
            return
        if self._next_frame_time is None:
            self._next_frame_time = float(data.time)
        while data.time + 1e-12 >= self._next_frame_time:
            self._renderer.update_scene(data, camera=VLA_EXTERNAL_CAMERA)
            self._writer.append_data(self._renderer.render())
            self._next_frame_time += 1.0 / self.fps

    def close(self) -> None:
        writer = self._writer
        self._writer = None
        if writer is not None:
            writer.close()
        renderer = self._renderer
        self._renderer = None
        if renderer is not None:
            renderer.close()


@dataclass(frozen=True)
class OnlineExecutionConfig:
    action_dt_s: float = 1.0 / 15.0
    controller_dt_s: float = 0.01
    warmup_s: float = 0.1
    hold_s: float = 0.3
    goal_tolerance_rad: float = 0.05
    max_extra_actions: int = 15
    max_policy_queries: int | None = None
    kp: tuple[float, ...] = tuple(float(value) for value in DEFAULT_KP)
    kd: tuple[float, ...] = tuple(float(value) for value in DEFAULT_KD)

    def __post_init__(self) -> None:
        timing = np.asarray(
            [
                self.action_dt_s,
                self.controller_dt_s,
                self.warmup_s,
                self.hold_s,
                self.goal_tolerance_rad,
            ],
            dtype=float,
        )
        if (
            not np.all(np.isfinite(timing))
            or self.action_dt_s <= 0.0
            or self.controller_dt_s <= 0.0
            or self.warmup_s < 0.0
            or self.hold_s < 0.0
            or self.goal_tolerance_rad < 0.0
            or self.max_extra_actions < 0
            or (
                self.max_policy_queries is not None
                and self.max_policy_queries <= 0
            )
        ):
            raise ValueError("online execution timing/tolerance is invalid")
        for name, values in (("kp", self.kp), ("kd", self.kd)):
            array = np.asarray(values, dtype=float)
            if (
                array.shape != (7,)
                or not np.all(np.isfinite(array))
                or np.any(array < 0.0)
            ):
                raise ValueError(f"{name} must contain seven nonnegative gains")


@dataclass(frozen=True)
class OnlineFaultConfig:
    """Deterministic faults injected between observation and action dispatch."""

    state_jump_query: int | None = None
    state_jump_rad: tuple[float, ...] = (0.0,) * 7

    def __post_init__(self) -> None:
        jump = np.asarray(self.state_jump_rad, dtype=float)
        if jump.shape != (7,) or not np.all(np.isfinite(jump)):
            raise ValueError("state_jump_rad must contain seven finite offsets")
        if self.state_jump_query is not None and self.state_jump_query < 0:
            raise ValueError("state_jump_query must be nonnegative")
        enabled = bool(np.any(np.abs(jump) > 0.0))
        if enabled != (self.state_jump_query is not None):
            raise ValueError(
                "state jump requires both a query index and a nonzero offset"
            )

    @property
    def enabled(self) -> bool:
        return self.state_jump_query is not None

    def offset_for_query(self, query_index: int) -> FloatArray:
        if query_index == self.state_jump_query:
            return np.asarray(self.state_jump_rad, dtype=float).copy()
        return np.zeros(7, dtype=float)


@dataclass(frozen=True)
class OnlineChunkRecord:
    query_index: int
    sequence_id: int
    action_offset: int
    executed_horizon: int
    observation_q: FloatArray
    dispatch_q: FloatArray
    actual_q_after: FloatArray
    decision_status: str
    policy_source: str | None
    runtime_fallback: bool
    failure_stage: str | None
    guard_fallback: bool
    fallback_reason: str | None
    deadline_exceeded: bool
    state_mismatch_exceeded: bool
    state_mismatch_rad: float
    fault_injected: bool
    injected_state_jump_rad: FloatArray
    exterior_image_sha256: str
    wrist_image_sha256: str
    exterior_frame_delta_mean_abs: float | None
    wrist_frame_delta_mean_abs: float | None
    exterior_thumbnail: UInt8Array
    wrist_thumbnail: UInt8Array
    policy_latency_ms: float
    client_inference_latency_ms: float
    validated_policy_response: bool
    server_timing: dict[str, float]
    simulated_inference_wait_s: float
    executed_interventions: int
    planned_interventions: int
    supervisor_latency_ms: float
    raw_actions: FloatArray | None
    guarded_actions: FloatArray
    predicted_positions: FloatArray
    action_reasons: tuple[str, ...]
    action_scales: FloatArray
    action_interventions: tuple[bool, ...]

    def metrics(self) -> dict[str, object]:
        return {
            "query_index": self.query_index,
            "sequence_id": self.sequence_id,
            "action_offset": self.action_offset,
            "executed_horizon": self.executed_horizon,
            "decision_status": self.decision_status,
            "policy_source": self.policy_source,
            "runtime_fallback": self.runtime_fallback,
            "failure_stage": self.failure_stage,
            "guard_fallback": self.guard_fallback,
            "fallback_reason": self.fallback_reason,
            "deadline_exceeded": self.deadline_exceeded,
            "state_mismatch_exceeded": self.state_mismatch_exceeded,
            "state_mismatch_rad": self.state_mismatch_rad,
            "fault_injected": self.fault_injected,
            "injected_state_jump_rad": self.injected_state_jump_rad.tolist(),
            "exterior_image_sha256": self.exterior_image_sha256,
            "wrist_image_sha256": self.wrist_image_sha256,
            "exterior_frame_delta_mean_abs": (
                self.exterior_frame_delta_mean_abs
            ),
            "wrist_frame_delta_mean_abs": self.wrist_frame_delta_mean_abs,
            "policy_latency_ms": self.policy_latency_ms,
            "client_inference_latency_ms": self.client_inference_latency_ms,
            "validated_policy_response": self.validated_policy_response,
            "server_timing": dict(self.server_timing),
            "simulated_inference_wait_s": self.simulated_inference_wait_s,
            "executed_interventions": self.executed_interventions,
            "planned_interventions": self.planned_interventions,
            "supervisor_latency_ms": self.supervisor_latency_ms,
            "raw_action_available": self.raw_actions is not None,
            "observation_q": self.observation_q.tolist(),
            "dispatch_q": self.dispatch_q.tolist(),
            "actual_q_after": self.actual_q_after.tolist(),
        }


@dataclass(frozen=True)
class OnlineEpisodeResult:
    scenario: str
    execution_horizon: int
    payload_mass: float
    policy_source: str
    action_steps: int
    policy_queries: int
    termination_reason: str
    task_success: bool
    physical_safe: bool
    final_goal_error_rad: float
    rmse_rad: float
    max_tracking_error_rad: float
    torque_saturation_count: int
    obstacle_contact_steps: int
    self_contact_steps: int
    joint_limit_violation_steps: int
    runtime_fallback_chunks: int
    guard_fallback_chunks: int
    deadline_chunks: int
    state_mismatch_chunks: int
    fault_injections: int
    executed_interventions: int
    simulated_inference_wait_s: float
    video_path: str | None
    times: FloatArray
    desired_positions: FloatArray
    actual_positions: FloatArray
    chunks: tuple[OnlineChunkRecord, ...]
    first_exterior_image: UInt8Array
    first_wrist_image: UInt8Array
    last_exterior_image: UInt8Array
    last_wrist_image: UInt8Array

    def metrics(self) -> dict[str, object]:
        end_to_end_latencies = np.asarray(
            [record.policy_latency_ms for record in self.chunks], dtype=float
        )
        client_latencies = np.asarray(
            [record.client_inference_latency_ms for record in self.chunks],
            dtype=float,
        )
        exterior_deltas = [
            record.exterior_frame_delta_mean_abs
            for record in self.chunks
            if record.exterior_frame_delta_mean_abs is not None
        ]
        wrist_deltas = [
            record.wrist_frame_delta_mean_abs
            for record in self.chunks
            if record.wrist_frame_delta_mean_abs is not None
        ]
        return {
            "scenario": self.scenario,
            "execution_horizon": self.execution_horizon,
            "payload_mass": self.payload_mass,
            "policy_source": self.policy_source,
            "action_steps": self.action_steps,
            "policy_queries": self.policy_queries,
            "termination_reason": self.termination_reason,
            "task_success": self.task_success,
            "physical_safe": self.physical_safe,
            "safe_task_success": self.task_success and self.physical_safe,
            "final_goal_error_rad": self.final_goal_error_rad,
            "rmse_rad": self.rmse_rad,
            "max_tracking_error_rad": self.max_tracking_error_rad,
            "torque_saturation_count": self.torque_saturation_count,
            "obstacle_contact_steps": self.obstacle_contact_steps,
            "self_contact_steps": self.self_contact_steps,
            "joint_limit_violation_steps": self.joint_limit_violation_steps,
            "runtime_fallback_chunks": self.runtime_fallback_chunks,
            "guard_fallback_chunks": self.guard_fallback_chunks,
            "deadline_chunks": self.deadline_chunks,
            "state_mismatch_chunks": self.state_mismatch_chunks,
            "fault_injections": self.fault_injections,
            "executed_interventions": self.executed_interventions,
            "simulated_inference_wait_s": self.simulated_inference_wait_s,
            "camera_audit_queries": len(self.chunks),
            "unique_exterior_observation_hashes": len(
                {record.exterior_image_sha256 for record in self.chunks}
            ),
            "unique_wrist_observation_hashes": len(
                {record.wrist_image_sha256 for record in self.chunks}
            ),
            "min_exterior_frame_delta_mean_abs": (
                min(exterior_deltas) if exterior_deltas else None
            ),
            "min_wrist_frame_delta_mean_abs": (
                min(wrist_deltas) if wrist_deltas else None
            ),
            "video_path": self.video_path,
            "mean_policy_latency_ms": float(np.mean(end_to_end_latencies)),
            "p95_policy_latency_ms": float(
                np.percentile(end_to_end_latencies, 95)
            ),
            "max_policy_latency_ms": float(np.max(end_to_end_latencies)),
            "mean_client_inference_latency_ms": float(
                np.mean(client_latencies)
            ),
            "p95_client_inference_latency_ms": float(
                np.percentile(client_latencies, 95)
            ),
        }


class ReferenceActionChunkPolicy:
    """Deterministic non-learned policy used to isolate online runtime behavior."""

    def __init__(
        self,
        reference_positions: ArrayLike,
        *,
        action_dt_s: float = 1.0 / 15.0,
        action_horizon: int = 15,
        velocity_limit_rad_s: float = 1.0,
        latency_ms: float = 0.0,
        latency_schedule_ms: Sequence[float] | None = None,
    ) -> None:
        positions = np.asarray(reference_positions, dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[1] != 7
            or len(positions) < 2
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError("reference_positions must be a finite Nx7 path")
        if (
            not np.all(
                np.isfinite(
                    [action_dt_s, velocity_limit_rad_s, latency_ms]
                )
            )
            or action_dt_s <= 0.0
            or action_horizon <= 0
            or velocity_limit_rad_s <= 0.0
            or latency_ms < 0.0
        ):
            raise ValueError("reference policy timing/limits are invalid")
        if latency_schedule_ms is None:
            schedule = (float(latency_ms),)
        else:
            schedule = tuple(float(value) for value in latency_schedule_ms)
            if latency_ms != 0.0:
                raise ValueError(
                    "use either latency_ms or latency_schedule_ms, not both"
                )
            if not schedule or any(
                not np.isfinite(value) or value < 0.0 for value in schedule
            ):
                raise ValueError(
                    "latency schedule must be finite, nonnegative, and nonempty"
                )
        self.reference_positions = positions.copy()
        self.action_dt_s = float(action_dt_s)
        self.action_horizon = int(action_horizon)
        self.velocity_limit_rad_s = float(velocity_limit_rad_s)
        self.latency_schedule_ms = schedule
        self._query_index = 0

    def infer(self, observation: VLAObservation) -> ActionChunk:
        latency_ms = self.latency_schedule_ms[
            self._query_index % len(self.latency_schedule_ms)
        ]
        self._query_index += 1
        cursor = min(observation.sequence_id, len(self.reference_positions) - 1)
        predicted_q = observation.joint_position.copy()
        actions = np.zeros((self.action_horizon, 8), dtype=float)
        for index in range(self.action_horizon):
            target_index = min(
                cursor + index + 1, len(self.reference_positions) - 1
            )
            target = self.reference_positions[target_index]
            velocity = np.clip(
                (target - predicted_q) / self.action_dt_s,
                -self.velocity_limit_rad_s,
                self.velocity_limit_rad_s,
            )
            actions[index, :7] = velocity
            actions[index, 7] = float(observation.gripper_position[0])
            predicted_q = predicted_q + self.action_dt_s * velocity
        return ActionChunk(
            actions=actions,
            source="scripted_non_learned_reference",
            observation_sequence_id=observation.sequence_id,
            inference_latency_ms=latency_ms,
            received_at_s=(
                observation.captured_at_s + latency_ms / 1000.0
            ),
        )


def _gain_array(values: tuple[float, ...], label: str) -> FloatArray:
    result = np.asarray(values, dtype=float)
    if result.shape != (7,) or np.any(result < 0.0):
        raise ValueError(f"{label} must contain seven nonnegative gains")
    return result


def run_online_episode(
    scenario_name: str,
    policy: ActionChunkPolicy,
    reference_positions: ArrayLike,
    *,
    execution_horizon: int,
    payload_mass: float = 0.0,
    clearance_m: float = 0.02,
    collision_resolution_rad: float = 0.02,
    guard_config: GuardConfig = GuardConfig(),
    execution_config: OnlineExecutionConfig = OnlineExecutionConfig(),
    fault_config: OnlineFaultConfig = OnlineFaultConfig(),
    prompt: str | None = None,
    video_path: Path | None = None,
    video_fps: int = 30,
    render_size: tuple[int, int] = (640, 480),
) -> OnlineEpisodeResult:
    """Execute action prefixes and recapture actual state/images every query."""

    if execution_horizon <= 0 or execution_horizon > 15:
        raise ValueError("execution_horizon must be within [1, 15]")
    if (
        not np.isfinite(payload_mass)
        or not np.isfinite(clearance_m)
        or payload_mass < 0.0
        or clearance_m < 0.0
    ):
        raise ValueError("payload and clearance must be nonnegative")
    if not np.isclose(guard_config.control_dt_s, execution_config.action_dt_s):
        raise ValueError("guard and online action periods must match")
    if (
        video_fps <= 0
        or len(render_size) != 2
        or any(value <= 0 for value in render_size)
    ):
        raise ValueError("online video settings must be positive")
    references = np.asarray(reference_positions, dtype=float)
    if (
        references.ndim != 2
        or references.shape[1] != 7
        or len(references) < 2
        or not np.all(np.isfinite(references))
    ):
        raise ValueError("reference_positions must be a finite Nx7 path")
    scenarios = mujoco_scenarios()
    if scenario_name not in scenarios:
        raise ValueError(f"unknown MuJoCo scenario: {scenario_name}")
    scenario = scenarios[scenario_name]
    task_prompt = prompt or f"move the gripper to the {scenario.name} goal"

    guard_robot = MuJoCoPanda.create(
        obstacles=inflate_obstacles(scenario.obstacles, clearance_m)
    )
    guard = ActionChunkGuard(
        MuJoCoCollisionChecker(
            guard_robot, resolution=collision_resolution_rad
        ),
        guard_config,
    )
    supervisor = VLARuntimeSupervisor(policy, guard)
    robot = MuJoCoPanda.create(
        obstacles=scenario.obstacles,
        payload_mass=payload_mass,
        torque_control=True,
        vla_cameras=True,
        goal_marker=guard_robot.hand_position(references[-1]),
    )
    data = mujoco.MjData(robot.model)
    robot.set_configuration(data, references[0])
    data.ctrl[7] = 255.0
    arm_qpos = robot.arm_qpos_addresses
    arm_dofs = robot.arm_dof_addresses
    physics_dt = float(robot.model.opt.timestep)
    controller_stride = execution_config.controller_dt_s / physics_dt
    if not np.isclose(controller_stride, round(controller_stride)):
        raise ValueError(
            "controller_dt_s must be an integer multiple of the MuJoCo timestep"
        )
    kp = _gain_array(execution_config.kp, "kp")
    kd = _gain_array(execution_config.kd, "kd")
    next_controller_time = 0.0
    times: list[float] = []
    desired_trace: list[FloatArray] = []
    actual_trace: list[FloatArray] = []
    torque_saturation_count = 0
    obstacle_contact_steps = 0
    self_contact_steps = 0
    joint_limit_violation_steps = 0
    online_recorder: _OnlineVideoRecorder | None = None

    def apply_control(desired_q: FloatArray, desired_dq: FloatArray) -> None:
        nonlocal torque_saturation_count
        current_q = data.qpos[arm_qpos].copy()
        current_dq = data.qvel[arm_dofs].copy()
        requested = (
            kp * (desired_q - current_q)
            + kd * (desired_dq - current_dq)
            + data.qfrc_bias[arm_dofs]
        )
        applied = np.clip(requested, -robot.force_limits, robot.force_limits)
        torque_saturation_count += int(
            np.count_nonzero(np.abs(requested - applied) > 1e-10)
        )
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[arm_dofs] = applied

    def step_physics() -> None:
        nonlocal obstacle_contact_steps
        nonlocal self_contact_steps
        nonlocal joint_limit_violation_steps
        mujoco.mj_step(robot.model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            raise RuntimeError("MuJoCo state became non-finite")
        q = data.qpos[arm_qpos]
        if np.any(q < robot.lower_limits - 1e-6) or np.any(
            q > robot.upper_limits + 1e-6
        ):
            joint_limit_violation_steps += 1
        if robot.obstacle_contacts(data):
            obstacle_contact_steps += 1
        if robot.self_contacts(data):
            self_contact_steps += 1
        if online_recorder is not None:
            online_recorder.capture(data)

    warmup_end = data.time + execution_config.warmup_s
    while data.time + 0.5 * physics_dt < warmup_end:
        if data.time + 1e-12 >= next_controller_time:
            apply_control(references[0], np.zeros(7, dtype=float))
            next_controller_time += execution_config.controller_dt_s
        step_physics()

    action_offset = 0
    action_limit = len(references) - 1 + execution_config.max_extra_actions
    records: list[OnlineChunkRecord] = []
    first_exterior: UInt8Array | None = None
    first_wrist: UInt8Array | None = None
    last_exterior: UInt8Array | None = None
    last_wrist: UInt8Array | None = None
    last_command = np.zeros(7, dtype=float)
    last_gripper = 1.0
    hold_target = references[-1].copy()
    termination_reason = "action_limit"

    with (
        MuJoCoDroidObservationBuilder(robot) as builder,
        _OnlineVideoRecorder(
            robot.model,
            video_path,
            fps=video_fps,
            render_size=render_size,
        ) as active_recorder,
    ):
        online_recorder = active_recorder
        online_recorder.capture(data)
        while action_offset < action_limit:
            if (
                execution_config.max_policy_queries is not None
                and len(records) >= execution_config.max_policy_queries
            ):
                termination_reason = "query_budget"
                hold_target = data.qpos[arm_qpos].copy()
                break
            guard.synchronize_velocity(last_command)
            observation = builder.capture(
                data,
                prompt=task_prompt,
                sequence_id=action_offset,
            )
            exterior_delta = _mean_abs_image_delta(
                observation.exterior_image, last_exterior
            )
            wrist_delta = _mean_abs_image_delta(
                observation.wrist_image, last_wrist
            )
            if first_exterior is None:
                first_exterior = observation.exterior_image.copy()
                first_wrist = observation.wrist_image.copy()
            last_exterior = observation.exterior_image.copy()
            last_wrist = observation.wrist_image.copy()
            q_before = data.qpos[arm_qpos].copy()
            dispatch_q = q_before.copy()
            policy_latency_ms = 0.0
            client_inference_latency_ms = 0.0
            validated_policy_response = False
            server_timing: dict[str, float] = {}
            simulated_wait_s = 0.0
            injected_state_jump = np.zeros(7, dtype=float)
            query_index = len(records)

            policy_call_started = monotonic()

            def advance_for_inference_wait(
                wait_ms: float,
            ) -> tuple[FloatArray, float]:
                nonlocal dispatch_q
                nonlocal next_controller_time
                nonlocal simulated_wait_s
                wait_start = float(data.time)
                wait_end = wait_start + wait_ms / 1000.0
                hold_q = data.qpos[arm_qpos].copy()
                while data.time + 0.5 * physics_dt < wait_end:
                    if data.time + 1e-12 >= next_controller_time:
                        desired_q = hold_q
                        desired_dq = np.zeros(7, dtype=float)
                        data.ctrl[7] = 255.0 * last_gripper
                        apply_control(desired_q, desired_dq)
                        times.append(float(data.time))
                        desired_trace.append(desired_q.copy())
                        actual_trace.append(data.qpos[arm_qpos].copy())
                        next_controller_time += execution_config.controller_dt_s
                    step_physics()
                simulated_wait_s = float(data.time) - wait_start
                dispatch_q = data.qpos[arm_qpos].copy()
                if simulated_wait_s > 0.0:
                    guard.synchronize_velocity(np.zeros(7, dtype=float))
                gripper = float(
                    np.clip(
                        np.mean(data.qpos[robot.finger_qpos_addresses]) / 0.04,
                        0.0,
                        1.0,
                    )
                )
                return dispatch_q, gripper

            def advance_during_inference(
                chunk: ActionChunk,
            ) -> tuple[FloatArray, float]:
                nonlocal dispatch_q
                nonlocal policy_latency_ms
                nonlocal client_inference_latency_ms
                nonlocal validated_policy_response
                nonlocal server_timing
                nonlocal injected_state_jump
                policy_latency_ms = chunk.age_ms(observation)
                client_inference_latency_ms = chunk.inference_latency_ms
                validated_policy_response = True
                server_timing = dict(chunk.server_timing)
                dispatch_q, gripper = advance_for_inference_wait(
                    policy_latency_ms
                )
                injected_state_jump = fault_config.offset_for_query(query_index)
                if np.any(np.abs(injected_state_jump) > 0.0):
                    jumped_q = data.qpos[arm_qpos].copy() + injected_state_jump
                    robot.validate_configuration(jumped_q)
                    data.qpos[arm_qpos] = jumped_q
                    data.qvel[arm_dofs] = 0.0
                    mujoco.mj_forward(robot.model, data)
                dispatch_q = data.qpos[arm_qpos].copy()
                return dispatch_q, gripper

            def advance_after_policy_failure() -> tuple[FloatArray, float]:
                nonlocal policy_latency_ms
                nonlocal client_inference_latency_ms
                policy_latency_ms = max(
                    0.0, (monotonic() - observation.captured_at_s) * 1000.0
                )
                client_inference_latency_ms = max(
                    0.0, (monotonic() - policy_call_started) * 1000.0
                )
                return advance_for_inference_wait(policy_latency_ms)

            decision = supervisor.infer_and_guard(
                q_before,
                float(observation.gripper_position[0]),
                observation,
                on_policy_response=advance_during_inference,
                on_policy_failure=advance_after_policy_failure,
            )
            execute_count = min(execution_horizon, action_limit - action_offset)
            selected = decision.actions[:execute_count]
            predicted = decision.predicted_positions[: execute_count + 1]
            chunk_start = float(data.time)
            action_clock = (
                chunk_start + execute_count * execution_config.action_dt_s
            )

            while data.time + 0.5 * physics_dt < action_clock:
                if data.time + 1e-12 >= next_controller_time:
                    local_time = float(
                        np.clip(
                            data.time - chunk_start,
                            0.0,
                            action_clock - chunk_start,
                        )
                    )
                    local_index = min(
                        int(local_time / execution_config.action_dt_s),
                        execute_count - 1,
                    )
                    phase = np.clip(
                        (
                            local_time
                            - local_index * execution_config.action_dt_s
                        )
                        / execution_config.action_dt_s,
                        0.0,
                        1.0,
                    )
                    desired_q = (
                        predicted[local_index]
                        + phase
                        * (predicted[local_index + 1] - predicted[local_index])
                    )
                    desired_dq = selected[local_index, :7]
                    data.ctrl[7] = 255.0 * float(selected[local_index, 7])
                    apply_control(desired_q, desired_dq)
                    times.append(float(data.time))
                    desired_trace.append(desired_q.copy())
                    actual_trace.append(data.qpos[arm_qpos].copy())
                    next_controller_time += execution_config.controller_dt_s
                step_physics()

            guard_result = decision.guard_result
            executed_interventions = (
                sum(
                    step.intervened
                    for step in guard_result.steps[:execute_count]
                )
                if guard_result is not None
                else execute_count
            )
            planned_interventions = (
                guard_result.intervention_steps
                if guard_result is not None
                else len(decision.actions)
            )
            if guard_result is not None:
                action_reasons = tuple(
                    step.reason for step in guard_result.steps
                )
                action_scales = np.asarray(
                    [step.scale for step in guard_result.steps], dtype=float
                )
                action_interventions = tuple(
                    step.intervened for step in guard_result.steps
                )
            else:
                fallback_reason = (
                    f"runtime_fallback:{decision.failure.stage}"
                    if decision.failure is not None
                    else "runtime_fallback"
                )
                action_reasons = (fallback_reason,) * len(decision.actions)
                action_scales = np.zeros(len(decision.actions), dtype=float)
                action_interventions = (True,) * len(decision.actions)
            records.append(
                OnlineChunkRecord(
                    query_index=query_index,
                    sequence_id=observation.sequence_id,
                    action_offset=action_offset,
                    executed_horizon=execute_count,
                    observation_q=observation.joint_position.copy(),
                    dispatch_q=dispatch_q.copy(),
                    actual_q_after=data.qpos[arm_qpos].copy(),
                    decision_status=decision.status,
                    policy_source=decision.policy_source,
                    runtime_fallback=decision.used_runtime_fallback,
                    failure_stage=(
                        decision.failure.stage if decision.failure else None
                    ),
                    guard_fallback=(
                        guard_result.fallback_reason is not None
                        if guard_result is not None
                        else False
                    ),
                    fallback_reason=(
                        guard_result.fallback_reason
                        if guard_result is not None
                        else None
                    ),
                    deadline_exceeded=(
                        guard_result.deadline_exceeded
                        if guard_result is not None
                        else False
                    ),
                    state_mismatch_exceeded=(
                        guard_result.state_mismatch_exceeded
                        if guard_result is not None
                        else False
                    ),
                    state_mismatch_rad=(
                        guard_result.state_mismatch_rad
                        if guard_result is not None
                        else float(
                            np.max(
                                np.abs(dispatch_q - observation.joint_position)
                            )
                        )
                    ),
                    fault_injected=bool(
                        np.any(np.abs(injected_state_jump) > 0.0)
                    ),
                    injected_state_jump_rad=injected_state_jump.copy(),
                    exterior_image_sha256=_image_sha256(
                        observation.exterior_image
                    ),
                    wrist_image_sha256=_image_sha256(
                        observation.wrist_image
                    ),
                    exterior_frame_delta_mean_abs=exterior_delta,
                    wrist_frame_delta_mean_abs=wrist_delta,
                    exterior_thumbnail=_image_thumbnail(
                        observation.exterior_image
                    ),
                    wrist_thumbnail=_image_thumbnail(
                        observation.wrist_image
                    ),
                    policy_latency_ms=policy_latency_ms,
                    client_inference_latency_ms=client_inference_latency_ms,
                    validated_policy_response=validated_policy_response,
                    server_timing=server_timing,
                    simulated_inference_wait_s=simulated_wait_s,
                    executed_interventions=executed_interventions,
                    planned_interventions=planned_interventions,
                    supervisor_latency_ms=decision.supervisor_latency_ms,
                    raw_actions=(
                        decision.raw_actions.copy()
                        if decision.raw_actions is not None
                        else None
                    ),
                    guarded_actions=decision.actions.copy(),
                    predicted_positions=decision.predicted_positions.copy(),
                    action_reasons=action_reasons,
                    action_scales=action_scales,
                    action_interventions=action_interventions,
                )
            )
            action_offset += execute_count
            last_command = selected[-1, :7].copy()
            last_gripper = float(selected[-1, 7])
            if decision.used_runtime_fallback:
                hold_target = data.qpos[arm_qpos].copy()
                termination_reason = (
                    f"runtime_fallback:{decision.failure.stage}"
                    if decision.failure is not None
                    else "runtime_fallback"
                )
                break
            if guard_result is not None and guard_result.fallback_latched:
                hold_target = data.qpos[arm_qpos].copy()
                termination_reason = (
                    f"guard_fallback:{guard_result.fallback_reason}"
                )
                break
            if (
                action_offset >= len(references) - 1
                and np.max(np.abs(data.qpos[arm_qpos] - references[-1]))
                <= execution_config.goal_tolerance_rad
                and np.max(np.abs(data.qvel[arm_dofs])) <= 0.05
            ):
                termination_reason = "goal_reached"
                break

        hold_end = data.time + execution_config.hold_s
        while data.time + 0.5 * physics_dt < hold_end:
            if data.time + 1e-12 >= next_controller_time:
                desired_q = hold_target
                desired_dq = np.zeros(7, dtype=float)
                data.ctrl[7] = 255.0 * last_gripper
                apply_control(desired_q, desired_dq)
                times.append(float(data.time))
                desired_trace.append(desired_q.copy())
                actual_trace.append(data.qpos[arm_qpos].copy())
                next_controller_time += execution_config.controller_dt_s
            step_physics()

    if first_exterior is None or first_wrist is None:
        raise RuntimeError("online episode captured no observations")
    if last_exterior is None or last_wrist is None:
        raise RuntimeError("online episode has no final observation")
    desired_array = np.asarray(desired_trace)
    actual_array = np.asarray(actual_trace)
    errors = actual_array - desired_array
    final_goal_error = float(
        np.max(np.abs(data.qpos[arm_qpos] - references[-1]))
    )
    physical_safe = bool(
        obstacle_contact_steps == 0
        and self_contact_steps == 0
        and joint_limit_violation_steps == 0
    )
    policy_sources = {record.policy_source for record in records}
    policy_source = (
        next(iter(policy_sources))
        if len(policy_sources) == 1 and None not in policy_sources
        else "mixed_or_unavailable"
    )
    return OnlineEpisodeResult(
        scenario=scenario_name,
        execution_horizon=execution_horizon,
        payload_mass=payload_mass,
        policy_source=str(policy_source),
        action_steps=action_offset,
        policy_queries=len(records),
        termination_reason=termination_reason,
        task_success=final_goal_error <= execution_config.goal_tolerance_rad,
        physical_safe=physical_safe,
        final_goal_error_rad=final_goal_error,
        rmse_rad=float(np.sqrt(np.mean(errors**2))),
        max_tracking_error_rad=float(np.max(np.abs(errors))),
        torque_saturation_count=torque_saturation_count,
        obstacle_contact_steps=obstacle_contact_steps,
        self_contact_steps=self_contact_steps,
        joint_limit_violation_steps=joint_limit_violation_steps,
        runtime_fallback_chunks=sum(record.runtime_fallback for record in records),
        guard_fallback_chunks=sum(record.guard_fallback for record in records),
        deadline_chunks=sum(record.deadline_exceeded for record in records),
        state_mismatch_chunks=sum(
            record.state_mismatch_exceeded for record in records
        ),
        fault_injections=sum(record.fault_injected for record in records),
        executed_interventions=sum(
            record.executed_interventions for record in records
        ),
        simulated_inference_wait_s=sum(
            record.simulated_inference_wait_s for record in records
        ),
        video_path=str(video_path.resolve()) if video_path is not None else None,
        times=np.asarray(times),
        desired_positions=desired_array,
        actual_positions=actual_array,
        chunks=tuple(records),
        first_exterior_image=first_exterior,
        first_wrist_image=first_wrist,
        last_exterior_image=last_exterior,
        last_wrist_image=last_wrist,
    )
