"""Reusable inverse-dynamics workspace for synchronized Panda stops.

The reference braking function deliberately rebuilds every resource on every
call and checks every sampled state plus every adjacent edge.  A synchronized
constant-acceleration stop follows one monotone scalar parameter along a
single joint-space line segment.  One continuous certificate for that segment
therefore covers every sampled state and every inter-sample edge.

This validator keeps the inverse-dynamics ``MjData`` and actuator effort limits
alive across calls.  It remains lock-protected because MuJoCo data and the
collision checker are mutable workspaces.
"""

from __future__ import annotations

import threading

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from armbench.mujoco_sim.continuous_collision import (
    ContinuousMuJoCoCollisionChecker,
)
from armbench.mujoco_sim.dynamics_braking import (
    PANDA_DOF,
    BrakingCollisionChecker,
    DynamicsBrakingConfig,
    DynamicsBrakingResult,
    _arm_actuator_effort_limits,
    _candidate_stop,
    _failure_result,
    _readonly_vector,
)
from armbench.mujoco_sim.model import MuJoCoPanda


class PersistentDynamicsBrakingValidator:
    """Validate repeated stops with cached MuJoCo and actuator workspaces."""

    def __init__(
        self,
        robot: MuJoCoPanda,
        collision_checker: BrakingCollisionChecker,
        config: DynamicsBrakingConfig = DynamicsBrakingConfig(),
    ) -> None:
        if collision_checker.robot is not robot:
            raise ValueError("collision checker must use the same Panda instance")
        if robot.dof != PANDA_DOF:
            raise ValueError("dynamics braking requires a seven-DoF Panda")
        self.robot = robot
        self.collision_checker = collision_checker
        self.config = config
        self._lock = threading.Lock()
        self._inverse_data = mujoco.MjData(robot.model)
        self._validation_calls = 0
        self._whole_stop_edges_checked = 0
        try:
            self._effort_lower, self._effort_upper = _arm_actuator_effort_limits(
                robot, config.actuator_force_limit_scale
            )
            self._effort_limit_failure: str | None = None
        except RuntimeError as error:
            self._effort_lower = None
            self._effort_upper = None
            self._effort_limit_failure = str(error)

    @property
    def validation_calls(self) -> int:
        return self._validation_calls

    @property
    def whole_stop_edges_checked(self) -> int:
        return self._whole_stop_edges_checked

    def validate(self, q: ArrayLike, qvel: ArrayLike) -> DynamicsBrakingResult:
        """Generate and validate one stop while holding the workspace lock."""

        with self._lock:
            self._validation_calls += 1
            return self._validate_locked(q, qvel)

    def _failure(
        self,
        *,
        reason: str,
        failure_sample_index: int | None,
        times: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
        efforts: list[np.ndarray],
        required_stop_time_s: float,
        max_torque_ratio: float | None,
    ) -> DynamicsBrakingResult:
        return _failure_result(
            reason=reason,
            failure_sample_index=failure_sample_index,
            times=times,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            efforts=efforts,
            lower=self._effort_lower,
            upper=self._effort_upper,
            required_stop_time_s=required_stop_time_s,
            max_torque_ratio=max_torque_ratio,
            inter_sample_edges_checked=self.config.check_inter_sample_edges,
        )

    def _validate_locked(
        self, q: ArrayLike, qvel: ArrayLike
    ) -> DynamicsBrakingResult:
        position = self.robot.validate_configuration(q).copy()
        velocity = _readonly_vector(qvel, PANDA_DOF, "qvel").copy()
        required_time, times, positions, velocities, accelerations = (
            _candidate_stop(position, velocity, self.config)
        )
        if self._effort_limit_failure is not None:
            return self._failure(
                reason=self._effort_limit_failure,
                failure_sample_index=None,
                times=times,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                efforts=[],
                required_stop_time_s=required_time,
                max_torque_ratio=None,
            )

        tolerance = self.config.state_tolerance
        initial_position_violation = (
            position < self.robot.lower_limits - tolerance
        ) | (position > self.robot.upper_limits + tolerance)
        if np.any(initial_position_violation):
            joint = int(np.flatnonzero(initial_position_violation)[0]) + 1
            return self._failure(
                reason=f"joint_position_limit:joint{joint}",
                failure_sample_index=0,
                times=times,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                efforts=[],
                required_stop_time_s=required_time,
                max_torque_ratio=None,
            )
        initial_velocity_violation = (
            np.abs(velocity) > self.robot.velocity_limits + tolerance
        )
        if np.any(initial_velocity_violation):
            joint = int(np.flatnonzero(initial_velocity_violation)[0]) + 1
            return self._failure(
                reason=f"joint_velocity_limit:joint{joint}",
                failure_sample_index=0,
                times=times,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                efforts=[],
                required_stop_time_s=required_time,
                max_torque_ratio=None,
            )

        initial_collision = self.collision_checker.configuration_failure(position)
        if initial_collision is not None:
            return self._failure(
                reason=initial_collision,
                failure_sample_index=0,
                times=times,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                efforts=[],
                required_stop_time_s=required_time,
                max_torque_ratio=None,
            )
        if required_time > self.config.max_stop_time_s + tolerance:
            return self._failure(
                reason="stop_time_limit",
                failure_sample_index=None,
                times=times,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                efforts=[],
                required_stop_time_s=required_time,
                max_torque_ratio=None,
            )

        # Cheap state checks retain the reference function's sample ordering.
        acceleration_limits = self.config.acceleration_limits
        for index, (sample_q, sample_qvel, sample_qacc) in enumerate(
            zip(positions, velocities, accelerations)
        ):
            position_violation = (
                sample_q < self.robot.lower_limits - tolerance
            ) | (sample_q > self.robot.upper_limits + tolerance)
            if np.any(position_violation):
                joint = int(np.flatnonzero(position_violation)[0]) + 1
                return self._failure(
                    reason=f"joint_position_limit:joint{joint}",
                    failure_sample_index=index,
                    times=times,
                    positions=positions,
                    velocities=velocities,
                    accelerations=accelerations,
                    efforts=[],
                    required_stop_time_s=required_time,
                    max_torque_ratio=None,
                )
            velocity_violation = (
                np.abs(sample_qvel) > self.robot.velocity_limits + tolerance
            )
            if np.any(velocity_violation):
                joint = int(np.flatnonzero(velocity_violation)[0]) + 1
                return self._failure(
                    reason=f"joint_velocity_limit:joint{joint}",
                    failure_sample_index=index,
                    times=times,
                    positions=positions,
                    velocities=velocities,
                    accelerations=accelerations,
                    efforts=[],
                    required_stop_time_s=required_time,
                    max_torque_ratio=None,
                )
            acceleration_violation = (
                np.abs(sample_qacc) > acceleration_limits + tolerance
            )
            if np.any(acceleration_violation):
                joint = int(np.flatnonzero(acceleration_violation)[0]) + 1
                return self._failure(
                    reason=f"joint_acceleration_limit:joint{joint}",
                    failure_sample_index=index,
                    times=times,
                    positions=positions,
                    velocities=velocities,
                    accelerations=accelerations,
                    efforts=[],
                    required_stop_time_s=required_time,
                    max_torque_ratio=None,
                )

        collision_failure = self._stop_collision_failure(positions)
        if collision_failure is not None:
            reason, failure_index = collision_failure
            return self._failure(
                reason=reason,
                failure_sample_index=failure_index,
                times=times,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                efforts=[],
                required_stop_time_s=required_time,
                max_torque_ratio=None,
            )

        effort_lower = self._effort_lower
        effort_upper = self._effort_upper
        if effort_lower is None or effort_upper is None:
            raise RuntimeError("cached actuator effort limits are unavailable")
        efforts: list[np.ndarray] = []
        maximum_ratio: float | None = None
        for index, (sample_q, sample_qvel, sample_qacc) in enumerate(
            zip(positions, velocities, accelerations)
        ):
            self.robot.set_configuration(
                self._inverse_data, sample_q, forward=False
            )
            self._inverse_data.qvel[self.robot.arm_dof_addresses] = sample_qvel
            self._inverse_data.qacc[:] = 0.0
            self._inverse_data.qacc[self.robot.arm_dof_addresses] = sample_qacc
            self._inverse_data.qfrc_applied[:] = 0.0
            self._inverse_data.xfrc_applied[:] = 0.0
            try:
                mujoco.mj_inverse(self.robot.model, self._inverse_data)
            except Exception as error:
                return self._failure(
                    reason=f"inverse_dynamics_error:{type(error).__name__}",
                    failure_sample_index=index,
                    times=times,
                    positions=positions,
                    velocities=velocities,
                    accelerations=accelerations,
                    efforts=efforts,
                    required_stop_time_s=required_time,
                    max_torque_ratio=maximum_ratio,
                )
            effort = self._inverse_data.qfrc_inverse[
                self.robot.arm_dof_addresses
            ].copy()
            if not np.all(np.isfinite(effort)):
                return self._failure(
                    reason="inverse_dynamics_nonfinite",
                    failure_sample_index=index,
                    times=times,
                    positions=positions,
                    velocities=velocities,
                    accelerations=accelerations,
                    efforts=efforts,
                    required_stop_time_s=required_time,
                    max_torque_ratio=maximum_ratio,
                )
            efforts.append(effort)
            directional_limit = np.where(
                effort >= 0.0, effort_upper, -effort_lower
            )
            sample_ratio = float(np.max(np.abs(effort) / directional_limit))
            maximum_ratio = (
                sample_ratio
                if maximum_ratio is None
                else max(maximum_ratio, sample_ratio)
            )
            force_violation = (effort < effort_lower - tolerance) | (
                effort > effort_upper + tolerance
            )
            if np.any(force_violation):
                joint = int(np.flatnonzero(force_violation)[0]) + 1
                return self._failure(
                    reason=f"actuator_force_limit:joint{joint}",
                    failure_sample_index=index,
                    times=times,
                    positions=positions,
                    velocities=velocities,
                    accelerations=accelerations,
                    efforts=efforts,
                    required_stop_time_s=required_time,
                    max_torque_ratio=maximum_ratio,
                )

        if (
            np.max(np.abs(velocities[-1]))
            > self.config.terminal_velocity_tolerance_rad_s
        ):
            return self._failure(
                reason="terminal_velocity_nonzero",
                failure_sample_index=len(times) - 1,
                times=times,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                efforts=efforts,
                required_stop_time_s=required_time,
                max_torque_ratio=maximum_ratio,
            )
        return DynamicsBrakingResult(
            validated=True,
            failure_reason=None,
            failure_sample_index=None,
            times_s=times,
            positions_rad=positions,
            velocities_rad_s=velocities,
            accelerations_rad_s2=accelerations,
            inverse_dynamics_efforts=np.asarray(efforts, dtype=float),
            actuator_effort_lower_limits=effort_lower,
            actuator_effort_upper_limits=effort_upper,
            required_stop_time_s=required_time,
            evaluated_samples=len(efforts),
            max_torque_ratio=maximum_ratio,
            inter_sample_edges_checked=self.config.check_inter_sample_edges,
        )

    def _stop_collision_failure(
        self, positions: np.ndarray
    ) -> tuple[str, int | None] | None:
        if self.config.check_inter_sample_edges and len(positions) > 1:
            self._whole_stop_edges_checked += 1
            if isinstance(
                self.collision_checker, ContinuousMuJoCoCollisionChecker
            ):
                certificate = self.collision_checker.edge_certificate(
                    positions[0], positions[-1]
                )
                if not certificate.certified_safe:
                    return "inter_sample_collision", None
            elif not self.collision_checker.edge_is_valid(
                positions[0], positions[-1]
            ):
                return "inter_sample_collision", None
            return None

        for index, sample_q in enumerate(positions[1:], start=1):
            failure = self.collision_checker.configuration_failure(sample_q)
            if failure is not None:
                return failure, index
        return None


__all__ = ["PersistentDynamicsBrakingValidator"]
