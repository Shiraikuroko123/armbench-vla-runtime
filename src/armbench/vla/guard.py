"""Training-free runtime validation and repair for DROID action chunks."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

from armbench.mujoco_sim.collision import MuJoCoCollisionChecker
from armbench.vla.types import ActionChunk, VLAObservation

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GuardConfig:
    control_dt_s: float = 1.0 / 15.0
    deadline_ms: float = 200.0
    max_state_mismatch_rad: float = 0.05
    joint_velocity_clip_rad_s: float = 1.0
    joint_acceleration_clip_rad_s2: float = 15.0
    latch_on_deadline: bool = True
    latch_on_state_mismatch: bool = True
    backtracking_scales: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0)

    def __post_init__(self) -> None:
        if self.control_dt_s <= 0.0 or self.deadline_ms < 0.0:
            raise ValueError("guard timing parameters are invalid")
        if (
            self.max_state_mismatch_rad < 0.0
            or not np.isfinite(self.max_state_mismatch_rad)
        ):
            raise ValueError("max_state_mismatch_rad must be finite and nonnegative")
        if self.joint_velocity_clip_rad_s <= 0.0:
            raise ValueError("joint_velocity_clip_rad_s must be positive")
        if (
            self.joint_acceleration_clip_rad_s2 <= 0.0
            or not np.isfinite(self.joint_acceleration_clip_rad_s2)
        ):
            raise ValueError(
                "joint_acceleration_clip_rad_s2 must be finite and positive"
            )
        if not self.backtracking_scales or self.backtracking_scales[-1] != 0.0:
            raise ValueError("backtracking scales must end with a hold action")
        if any(
            first < second or not 0.0 <= first <= 1.0
            for first, second in zip(
                self.backtracking_scales, self.backtracking_scales[1:]
            )
        ):
            raise ValueError("backtracking scales must descend within [0, 1]")


@dataclass(frozen=True)
class GuardStep:
    index: int
    raw_safe: bool
    repaired_safe: bool
    intervened: bool
    scale: float
    reason: str
    raw_acceleration_safe: bool
    selected_acceleration_safe: bool
    acceleration_limited: bool
    max_raw_acceleration_rad_s2: float
    max_selected_acceleration_rad_s2: float
    holding: bool
    q_before: FloatArray
    q_after: FloatArray


@dataclass(frozen=True)
class GuardResult:
    source: str
    deadline_exceeded: bool
    deadline_latched: bool
    state_mismatch_exceeded: bool
    state_mismatch_latched: bool
    state_mismatch_rad: float
    fallback_latched: bool
    fallback_reason: str | None
    end_to_end_latency_ms: float
    guarded_actions: FloatArray
    predicted_positions: FloatArray
    steps: tuple[GuardStep, ...]
    guard_latency_ms: float

    @property
    def unsafe_raw_steps(self) -> int:
        return sum(not step.raw_safe for step in self.steps)

    @property
    def intervention_steps(self) -> int:
        return sum(step.intervened for step in self.steps)

    @property
    def hold_steps(self) -> int:
        return sum(step.holding and step.scale == 0.0 for step in self.steps)

    @property
    def slew_limited_steps(self) -> int:
        return sum(step.acceleration_limited for step in self.steps)

    @property
    def acceleration_override_steps(self) -> int:
        return sum(not step.selected_acceleration_safe for step in self.steps)

    @property
    def max_raw_acceleration_rad_s2(self) -> float:
        return max(step.max_raw_acceleration_rad_s2 for step in self.steps)

    @property
    def max_guarded_acceleration_rad_s2(self) -> float:
        return max(step.max_selected_acceleration_rad_s2 for step in self.steps)

    @property
    def safe_after_guard(self) -> bool:
        return all(
            step.repaired_safe and step.selected_acceleration_safe
            for step in self.steps
        )

    def metrics(self) -> dict[str, object]:
        return {
            "source": self.source,
            "deadline_exceeded": self.deadline_exceeded,
            "deadline_latched": self.deadline_latched,
            "state_mismatch_exceeded": self.state_mismatch_exceeded,
            "state_mismatch_latched": self.state_mismatch_latched,
            "state_mismatch_rad": self.state_mismatch_rad,
            "fallback_latched": self.fallback_latched,
            "fallback_reason": self.fallback_reason,
            "end_to_end_latency_ms": self.end_to_end_latency_ms,
            "guard_latency_ms": self.guard_latency_ms,
            "horizon": len(self.steps),
            "unsafe_raw_steps": self.unsafe_raw_steps,
            "intervention_steps": self.intervention_steps,
            "hold_steps": self.hold_steps,
            "slew_limited_steps": self.slew_limited_steps,
            "acceleration_override_steps": self.acceleration_override_steps,
            "max_raw_acceleration_rad_s2": self.max_raw_acceleration_rad_s2,
            "max_guarded_acceleration_rad_s2": (
                self.max_guarded_acceleration_rad_s2
            ),
            "safe_after_guard": self.safe_after_guard,
        }


class ActionChunkGuard:
    """Check action lookahead and backtrack unsafe joint velocities in rad/s."""

    def __init__(
        self,
        checker: MuJoCoCollisionChecker,
        config: GuardConfig = GuardConfig(),
    ) -> None:
        self.checker = checker
        self.config = config
        self._deadline_latched = False
        self._state_mismatch_latched = False
        self._previous_velocity = np.zeros(7, dtype=float)

    def reset(self, previous_joint_velocity: ArrayLike | None = None) -> None:
        """Clear latched fallbacks after an explicit runtime resynchronization."""

        self._deadline_latched = False
        self._state_mismatch_latched = False
        if previous_joint_velocity is None:
            self._previous_velocity = np.zeros(7, dtype=float)
            return
        velocity = np.asarray(previous_joint_velocity, dtype=float)
        if velocity.shape != (7,) or not np.all(np.isfinite(velocity)):
            raise ValueError("previous_joint_velocity must be a finite 7-vector")
        if np.any(np.abs(velocity) > self._velocity_limits() + 1e-12):
            raise ValueError("previous_joint_velocity exceeds configured limits")
        self._previous_velocity = velocity.copy()

    def _velocity_limits(self) -> FloatArray:
        return np.minimum(
            self.checker.robot.velocity_limits,
            self.config.joint_velocity_clip_rad_s,
        )

    def _candidate_failure(self, q_start: FloatArray, q_end: FloatArray) -> str | None:
        failure = self.checker.configuration_failure(q_end)
        if failure is not None:
            return failure
        if not self.checker.edge_is_valid(q_start, q_end):
            return "invalid_edge"
        return None

    def _integrate(self, q: FloatArray, joint_velocity: FloatArray) -> FloatArray:
        return q + self.config.control_dt_s * joint_velocity

    def guard(
        self,
        q_start: ArrayLike,
        gripper_position: float,
        observation: VLAObservation,
        chunk: ActionChunk,
    ) -> GuardResult:
        started = perf_counter()
        q = self.checker.robot.validate_configuration(q_start).copy()
        if not 0.0 <= gripper_position <= 1.0:
            raise ValueError("gripper_position must be normalized to [0, 1]")
        end_to_end_latency = chunk.age_ms(observation)
        deadline_exceeded = end_to_end_latency > self.config.deadline_ms
        state_mismatch_rad = float(
            np.max(np.abs(q - observation.joint_position))
        )
        state_mismatch_exceeded = (
            state_mismatch_rad > self.config.max_state_mismatch_rad
        )
        if deadline_exceeded and self.config.latch_on_deadline:
            self._deadline_latched = True
        if state_mismatch_exceeded and self.config.latch_on_state_mismatch:
            self._state_mismatch_latched = True
        deadline_latched = self._deadline_latched
        state_mismatch_latched = self._state_mismatch_latched
        fallback_latched = deadline_latched or state_mismatch_latched
        fallback_active = (
            deadline_exceeded or state_mismatch_exceeded or fallback_latched
        )
        if deadline_exceeded:
            fallback_reason = "deadline"
        elif state_mismatch_exceeded:
            fallback_reason = "state_mismatch"
        elif deadline_latched:
            fallback_reason = "deadline_latched"
        elif state_mismatch_latched:
            fallback_reason = "state_mismatch_latched"
        else:
            fallback_reason = None
        guarded = np.zeros_like(chunk.actions)
        positions = [q.copy()]
        records: list[GuardStep] = []
        current_gripper = float(gripper_position)
        previous_velocity = self._previous_velocity.copy()
        velocity_limits = self._velocity_limits()
        max_velocity_delta = (
            self.config.joint_acceleration_clip_rad_s2
            * self.config.control_dt_s
        )

        for index, raw_action in enumerate(chunk.actions):
            q_before = q.copy()
            raw_velocity = raw_action[:7]
            raw_gripper = float(raw_action[7])
            finite = bool(np.all(np.isfinite(raw_action)))
            bounded = finite and bool(
                np.all(np.abs(raw_velocity) <= velocity_limits + 1e-12)
            )
            gripper_bounded = finite and 0.0 <= raw_gripper <= 1.0
            max_raw_acceleration = (
                float(
                    np.max(np.abs(raw_velocity - previous_velocity))
                    / self.config.control_dt_s
                )
                if finite
                else float("inf")
            )
            raw_acceleration_safe = (
                finite
                and max_raw_acceleration
                <= self.config.joint_acceleration_clip_rad_s2 + 1e-12
            )
            raw_candidate = (
                self._integrate(q, raw_velocity) if finite else q.copy()
            )
            if not finite or not bounded or not gripper_bounded:
                raw_failure = "nonfinite_or_action_bounds"
            elif not raw_acceleration_safe:
                raw_failure = "joint_acceleration_limit"
            else:
                raw_failure = self._candidate_failure(q, raw_candidate)
            raw_safe = raw_failure is None and not fallback_active

            selected_scale = 0.0
            if fallback_active:
                selected_reason = str(fallback_reason)
            else:
                selected_reason = str(raw_failure)
            selected_velocity = np.zeros(7, dtype=float)
            selected_q = q.copy()
            selected_gripper = current_gripper
            repaired_safe = True
            acceleration_limited = False
            if not fallback_active and finite:
                clipped_velocity = np.clip(
                    raw_velocity,
                    -velocity_limits,
                    velocity_limits,
                )
                clipped_gripper = float(np.clip(raw_gripper, 0.0, 1.0))
                slew_lower = previous_velocity - max_velocity_delta
                slew_upper = previous_velocity + max_velocity_delta
                for scale in self.config.backtracking_scales:
                    scaled_velocity = clipped_velocity * scale
                    candidate_velocity = np.clip(
                        scaled_velocity, slew_lower, slew_upper
                    )
                    candidate_velocity = np.clip(
                        candidate_velocity, -velocity_limits, velocity_limits
                    )
                    candidate_q = self._integrate(q, candidate_velocity)
                    failure = self._candidate_failure(q, candidate_q)
                    if failure is None:
                        selected_scale = float(scale)
                        selected_velocity = candidate_velocity
                        selected_q = candidate_q
                        selected_gripper = clipped_gripper
                        acceleration_limited = not np.allclose(
                            candidate_velocity, scaled_velocity, atol=1e-12
                        )
                        if raw_safe and scale == 1.0 and not acceleration_limited:
                            selected_reason = "accepted"
                        elif (
                            raw_failure == "joint_acceleration_limit"
                            and scale == 1.0
                        ):
                            selected_reason = "slew_rate_repaired"
                        elif (
                            raw_failure == "nonfinite_or_action_bounds"
                            and scale == 1.0
                        ):
                            selected_reason = "action_bounds_repaired"
                        else:
                            selected_reason = f"backtracked:{raw_failure}"
                        break
                else:
                    repaired_safe = False
                    selected_reason = "no_safe_fallback"

            max_selected_acceleration = float(
                np.max(np.abs(selected_velocity - previous_velocity))
                / self.config.control_dt_s
            )
            selected_acceleration_safe = (
                max_selected_acceleration
                <= self.config.joint_acceleration_clip_rad_s2 + 1e-12
            )

            guarded[index, :7] = selected_velocity
            guarded[index, 7] = selected_gripper
            q = selected_q
            current_gripper = selected_gripper
            previous_velocity = selected_velocity.copy()
            positions.append(q.copy())
            records.append(
                GuardStep(
                    index=index,
                    raw_safe=raw_safe,
                    repaired_safe=repaired_safe,
                    intervened=not raw_safe or selected_scale != 1.0,
                    scale=selected_scale,
                    reason=selected_reason,
                    raw_acceleration_safe=raw_acceleration_safe,
                    selected_acceleration_safe=selected_acceleration_safe,
                    acceleration_limited=acceleration_limited,
                    max_raw_acceleration_rad_s2=max_raw_acceleration,
                    max_selected_acceleration_rad_s2=(
                        max_selected_acceleration
                    ),
                    holding=bool(np.all(np.abs(selected_velocity) <= 1e-12)),
                    q_before=q_before,
                    q_after=q.copy(),
                )
            )
        self._previous_velocity = previous_velocity.copy()
        return GuardResult(
            source=chunk.source,
            deadline_exceeded=deadline_exceeded,
            deadline_latched=deadline_latched,
            state_mismatch_exceeded=state_mismatch_exceeded,
            state_mismatch_latched=state_mismatch_latched,
            state_mismatch_rad=state_mismatch_rad,
            fallback_latched=fallback_latched,
            fallback_reason=fallback_reason,
            end_to_end_latency_ms=end_to_end_latency,
            guarded_actions=guarded,
            predicted_positions=np.asarray(positions),
            steps=tuple(records),
            guard_latency_ms=(perf_counter() - started) * 1000.0,
        )
