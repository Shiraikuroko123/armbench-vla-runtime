"""Trajectory-level VLA action repair with a terminal braking invariant."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

from armbench.mujoco_sim.collision import MuJoCoCollisionChecker
from armbench.vla.guard import GuardConfig
from armbench.vla.types import ActionChunk, VLAObservation


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BrakingRepairConfig:
    """Bound a conservative whole-chunk scale search.

    The wall-clock deadline is a measured software budget, not an operating
    system scheduling guarantee. The scale/evaluation limit is deterministic.
    """

    selection_deadline_ms: float = 20.0
    trajectory_scales: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0)
    max_scale_evaluations: int = 5
    max_terminal_brake_steps: int = 8

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.selection_deadline_ms)
            or self.selection_deadline_ms <= 0.0
        ):
            raise ValueError("selection_deadline_ms must be finite and positive")
        if (
            not self.trajectory_scales
            or self.trajectory_scales[-1] != 0.0
            or any(
                first <= second or not 0.0 <= first <= 1.0
                for first, second in zip(
                    self.trajectory_scales,
                    self.trajectory_scales[1:],
                )
            )
        ):
            raise ValueError(
                "trajectory_scales must strictly descend from at most 1 to 0"
            )
        if (
            type(self.max_scale_evaluations) is not int
            or self.max_scale_evaluations <= 0
            or self.max_scale_evaluations > len(self.trajectory_scales)
        ):
            raise ValueError("max_scale_evaluations is outside the scale set")
        if (
            type(self.max_terminal_brake_steps) is not int
            or self.max_terminal_brake_steps <= 0
        ):
            raise ValueError("max_terminal_brake_steps must be positive")


@dataclass(frozen=True)
class TrajectoryScaleCandidate:
    scale: float
    feasible: bool
    failure_step: int | None
    failure_reason: str | None
    actions: FloatArray
    positions: FloatArray
    terminal_braking_positions: FloatArray
    terminal_brake_steps: int
    max_acceleration_rad_s2: float
    collision_edge_checks: int

    def __post_init__(self) -> None:
        for value in (
            self.actions,
            self.positions,
            self.terminal_braking_positions,
        ):
            value.setflags(write=False)


@dataclass(frozen=True)
class BrakingRepairResult:
    source: str
    response_deadline_exceeded: bool
    state_mismatch_exceeded: bool
    state_mismatch_rad: float
    fallback_reason: str | None
    selected_scale: float
    selected_candidate_feasible: bool
    evaluated_scales: tuple[float, ...]
    selection_deadline_ms: float
    selection_deadline_exceeded: bool
    repaired_actions: FloatArray
    predicted_positions: FloatArray
    terminal_braking_positions: FloatArray
    terminal_brake_steps: int
    intervention_steps: int
    max_repaired_acceleration_rad_s2: float
    collision_edge_checks: int
    repair_latency_ms: float

    @property
    def safe_after_repair(self) -> bool:
        return self.selected_candidate_feasible

    def metrics(self) -> dict[str, object]:
        return {
            "source": self.source,
            "scope": "trajectory_scale_repair_with_terminal_braking",
            "response_deadline_exceeded": self.response_deadline_exceeded,
            "state_mismatch_exceeded": self.state_mismatch_exceeded,
            "state_mismatch_rad": self.state_mismatch_rad,
            "fallback_reason": self.fallback_reason,
            "selected_scale": self.selected_scale,
            "selected_candidate_feasible": self.selected_candidate_feasible,
            "evaluated_scales": list(self.evaluated_scales),
            "selection_deadline_ms": self.selection_deadline_ms,
            "selection_deadline_exceeded": self.selection_deadline_exceeded,
            "horizon": len(self.repaired_actions),
            "terminal_brake_steps": self.terminal_brake_steps,
            "intervention_steps": self.intervention_steps,
            "max_repaired_acceleration_rad_s2": (
                self.max_repaired_acceleration_rad_s2
            ),
            "collision_edge_checks": self.collision_edge_checks,
            "repair_latency_ms": self.repair_latency_ms,
            "safe_after_repair": self.safe_after_repair,
        }


class BrakingTrajectoryGuard:
    """Repair a complete chunk before execution and preserve a safe stop path."""

    def __init__(
        self,
        checker: MuJoCoCollisionChecker,
        guard_config: GuardConfig,
        repair_config: BrakingRepairConfig = BrakingRepairConfig(),
    ) -> None:
        self.checker = checker
        self.guard_config = guard_config
        self.repair_config = repair_config
        self._previous_velocity = np.zeros(7, dtype=float)
        required_brake_steps = int(
            ceil(
                float(np.max(self._velocity_limits()))
                / self._maximum_velocity_delta()
            )
        )
        if repair_config.max_terminal_brake_steps < required_brake_steps:
            raise ValueError(
                "max_terminal_brake_steps cannot stop the configured velocity limit"
            )

    def reset(self, previous_joint_velocity: ArrayLike | None = None) -> None:
        if previous_joint_velocity is None:
            self._previous_velocity = np.zeros(7, dtype=float)
            return
        value = np.asarray(previous_joint_velocity, dtype=float)
        if value.shape != (7,) or not np.all(np.isfinite(value)):
            raise ValueError("previous_joint_velocity must be a finite 7-vector")
        if np.any(np.abs(value) > self._velocity_limits() + 1e-12):
            raise ValueError("previous_joint_velocity exceeds configured limits")
        self._previous_velocity = value.copy()

    def _velocity_limits(self) -> FloatArray:
        return np.minimum(
            self.checker.robot.velocity_limits,
            self.guard_config.joint_velocity_clip_rad_s,
        )

    def _maximum_velocity_delta(self) -> float:
        return (
            self.guard_config.joint_acceleration_clip_rad_s2
            * self.guard_config.control_dt_s
        )

    def _candidate_failure(
        self, q_start: FloatArray, q_end: FloatArray
    ) -> str | None:
        failure = self.checker.configuration_failure(q_end)
        if failure is not None:
            return failure
        if not self.checker.edge_is_valid(q_start, q_end):
            return "invalid_edge"
        return None

    def _scaled_rollout(
        self,
        q_start: FloatArray,
        gripper_position: float,
        chunk: ActionChunk,
        scale: float,
    ) -> TrajectoryScaleCandidate:
        q = q_start.copy()
        previous_velocity = self._previous_velocity.copy()
        velocity_limits = self._velocity_limits()
        velocity_delta = self._maximum_velocity_delta()
        actions = np.zeros_like(chunk.actions)
        positions = [q.copy()]
        current_gripper = float(gripper_position)
        maximum_acceleration = 0.0
        edge_checks = 0

        start_failure = self.checker.configuration_failure(q)
        if start_failure is not None:
            return TrajectoryScaleCandidate(
                scale=scale,
                feasible=False,
                failure_step=0,
                failure_reason=f"invalid_start:{start_failure}",
                actions=actions,
                positions=np.asarray(positions),
                terminal_braking_positions=np.asarray([q]),
                terminal_brake_steps=0,
                max_acceleration_rad_s2=float("inf"),
                collision_edge_checks=edge_checks,
            )

        for index, raw_action in enumerate(chunk.actions):
            target_velocity = np.clip(
                raw_action[:7] * scale,
                -velocity_limits,
                velocity_limits,
            )
            selected_velocity = np.clip(
                target_velocity,
                previous_velocity - velocity_delta,
                previous_velocity + velocity_delta,
            )
            selected_velocity = np.clip(
                selected_velocity,
                -velocity_limits,
                velocity_limits,
            )
            acceleration = float(
                np.max(np.abs(selected_velocity - previous_velocity))
                / self.guard_config.control_dt_s
            )
            maximum_acceleration = max(maximum_acceleration, acceleration)
            q_next = (
                q
                + self.guard_config.control_dt_s
                * selected_velocity
            )
            failure = self._candidate_failure(q, q_next)
            edge_checks += 1
            if failure is not None:
                return TrajectoryScaleCandidate(
                    scale=scale,
                    feasible=False,
                    failure_step=index,
                    failure_reason=failure,
                    actions=actions,
                    positions=np.asarray(positions),
                    terminal_braking_positions=np.asarray([q]),
                    terminal_brake_steps=0,
                    max_acceleration_rad_s2=maximum_acceleration,
                    collision_edge_checks=edge_checks,
                )
            selected_gripper = (
                current_gripper
                if scale == 0.0
                else float(np.clip(raw_action[7], 0.0, 1.0))
            )
            actions[index, :7] = selected_velocity
            actions[index, 7] = selected_gripper
            positions.append(q_next.copy())
            q = q_next
            previous_velocity = selected_velocity
            current_gripper = selected_gripper

        braking_positions = [q.copy()]
        brake_steps = 0
        while (
            np.max(np.abs(previous_velocity)) > 1e-12
            and brake_steps < self.repair_config.max_terminal_brake_steps
        ):
            selected_velocity = np.sign(previous_velocity) * np.maximum(
                np.abs(previous_velocity) - velocity_delta,
                0.0,
            )
            acceleration = float(
                np.max(np.abs(selected_velocity - previous_velocity))
                / self.guard_config.control_dt_s
            )
            maximum_acceleration = max(maximum_acceleration, acceleration)
            q_next = (
                q
                + self.guard_config.control_dt_s
                * selected_velocity
            )
            failure = self._candidate_failure(q, q_next)
            edge_checks += 1
            if failure is not None:
                return TrajectoryScaleCandidate(
                    scale=scale,
                    feasible=False,
                    failure_step=len(chunk.actions) + brake_steps,
                    failure_reason=f"terminal_braking:{failure}",
                    actions=actions,
                    positions=np.asarray(positions),
                    terminal_braking_positions=np.asarray(braking_positions),
                    terminal_brake_steps=brake_steps,
                    max_acceleration_rad_s2=maximum_acceleration,
                    collision_edge_checks=edge_checks,
                )
            q = q_next
            previous_velocity = selected_velocity
            braking_positions.append(q.copy())
            brake_steps += 1

        stopped = bool(np.max(np.abs(previous_velocity)) <= 1e-12)
        return TrajectoryScaleCandidate(
            scale=scale,
            feasible=stopped,
            failure_step=None if stopped else len(chunk.actions) + brake_steps,
            failure_reason=None if stopped else "terminal_brake_horizon",
            actions=actions,
            positions=np.asarray(positions),
            terminal_braking_positions=np.asarray(braking_positions),
            terminal_brake_steps=brake_steps,
            max_acceleration_rad_s2=maximum_acceleration,
            collision_edge_checks=edge_checks,
        )

    def repair(
        self,
        q_start: ArrayLike,
        gripper_position: float,
        observation: VLAObservation,
        chunk: ActionChunk,
    ) -> BrakingRepairResult:
        started = perf_counter()
        q = self.checker.robot.validate_configuration(q_start).copy()
        if not self.checker.robot.within_limits(q):
            raise ValueError("q_start violates robot joint limits")
        if not 0.0 <= gripper_position <= 1.0:
            raise ValueError("gripper_position must be normalized to [0, 1]")
        end_to_end_latency_ms = chunk.age_ms(observation)
        response_deadline_exceeded = (
            end_to_end_latency_ms > self.guard_config.deadline_ms
        )
        state_mismatch_rad = float(
            np.max(np.abs(q - observation.joint_position))
        )
        state_mismatch_exceeded = (
            state_mismatch_rad > self.guard_config.max_state_mismatch_rad
        )
        if response_deadline_exceeded:
            fallback_reason = "response_deadline"
        elif state_mismatch_exceeded:
            fallback_reason = "state_mismatch"
        else:
            fallback_reason = None

        # Establish a verified stop/hold candidate before spending time on
        # higher-progress alternatives.
        fallback = self._scaled_rollout(q, gripper_position, chunk, 0.0)
        evaluated_candidates = [fallback]
        selected = fallback
        selection_deadline_exceeded = (
            (perf_counter() - started) * 1000.0
            > self.repair_config.selection_deadline_ms
        )
        if fallback_reason is None and fallback.feasible:
            for scale in self.repair_config.trajectory_scales:
                if scale == 0.0:
                    continue
                if len(evaluated_candidates) >= self.repair_config.max_scale_evaluations:
                    break
                elapsed_ms = (perf_counter() - started) * 1000.0
                if elapsed_ms >= self.repair_config.selection_deadline_ms:
                    selection_deadline_exceeded = True
                    fallback_reason = "selection_deadline"
                    break
                candidate = self._scaled_rollout(
                    q,
                    gripper_position,
                    chunk,
                    scale,
                )
                evaluated_candidates.append(candidate)
                elapsed_ms = (perf_counter() - started) * 1000.0
                if elapsed_ms > self.repair_config.selection_deadline_ms:
                    selection_deadline_exceeded = True
                    fallback_reason = "selection_deadline"
                    break
                if candidate.feasible:
                    selected = candidate
                    break
        elif not fallback.feasible and fallback_reason is None:
            fallback_reason = "no_feasible_stop"

        if not selected.feasible:
            actions = np.zeros_like(chunk.actions)
            actions[:, 7] = gripper_position
            positions = np.repeat(q[None, :], len(chunk.actions) + 1, axis=0)
            selected = TrajectoryScaleCandidate(
                scale=0.0,
                feasible=False,
                failure_step=fallback.failure_step,
                failure_reason=fallback.failure_reason,
                actions=actions,
                positions=positions,
                terminal_braking_positions=np.asarray([q]),
                terminal_brake_steps=0,
                max_acceleration_rad_s2=float(
                    np.max(np.abs(self._previous_velocity))
                    / self.guard_config.control_dt_s
                ),
                collision_edge_checks=sum(
                    candidate.collision_edge_checks
                    for candidate in evaluated_candidates
                ),
            )

        intervention_steps = int(
            np.count_nonzero(
                np.any(
                    np.abs(selected.actions - chunk.actions) > 1e-12,
                    axis=1,
                )
            )
        )
        self._previous_velocity = selected.actions[-1, :7].copy()
        repair_latency_ms = (perf_counter() - started) * 1000.0
        return BrakingRepairResult(
            source=chunk.source,
            response_deadline_exceeded=response_deadline_exceeded,
            state_mismatch_exceeded=state_mismatch_exceeded,
            state_mismatch_rad=state_mismatch_rad,
            fallback_reason=fallback_reason,
            selected_scale=selected.scale,
            selected_candidate_feasible=selected.feasible,
            evaluated_scales=tuple(
                candidate.scale for candidate in evaluated_candidates
            ),
            selection_deadline_ms=self.repair_config.selection_deadline_ms,
            selection_deadline_exceeded=selection_deadline_exceeded,
            repaired_actions=selected.actions,
            predicted_positions=selected.positions,
            terminal_braking_positions=selected.terminal_braking_positions,
            terminal_brake_steps=selected.terminal_brake_steps,
            intervention_steps=intervention_steps,
            max_repaired_acceleration_rad_s2=(
                selected.max_acceleration_rad_s2
            ),
            collision_edge_checks=sum(
                candidate.collision_edge_checks
                for candidate in evaluated_candidates
            ),
            repair_latency_ms=repair_latency_ms,
        )
