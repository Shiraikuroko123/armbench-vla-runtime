"""Fail-closed inverse-dynamics validation for Panda stopping trajectories.

This module checks whether a sampled emergency stop is kinematically and
dynamically feasible for the compiled MuJoCo Panda model.  It is a trajectory
feasibility check, not a hard-real-time or hardware safety certification.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import ceil
from typing import Final

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from armbench.mujoco_sim.collision import MuJoCoCollisionChecker
from armbench.mujoco_sim.continuous_collision import (
    ContinuousMuJoCoCollisionChecker,
)
from armbench.mujoco_sim.model import MuJoCoPanda

FloatArray = NDArray[np.float64]

PANDA_DOF: Final = 7
INVERSE_DYNAMICS_METHOD: Final = "mujoco.mj_inverse.v1"
EFFORT_LIMIT_METHOD: Final = "mjmodel.actuator_forcerange_times_gear.v1"
BrakingCollisionChecker = (
    MuJoCoCollisionChecker | ContinuousMuJoCoCollisionChecker
)


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _readonly_vector(value: ArrayLike, length: int, label: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{label} must be a finite numeric vector")
    result = np.asarray(raw, dtype=float)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite vector with length {length}")
    result = result.copy()
    result.flags.writeable = False
    return result


def _readonly_array(
    value: ArrayLike, shape_tail: tuple[int, ...], label: str
) -> FloatArray:
    result = np.asarray(value, dtype=float)
    if result.ndim != len(shape_tail) + 1 or result.shape[1:] != shape_tail:
        raise ValueError(f"{label} has an invalid shape")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain only finite values")
    result = result.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class DynamicsBrakingConfig:
    """Numerical and physical limits for one sampled Panda stop."""

    sample_dt_s: float = 0.01
    joint_acceleration_limits_rad_s2: tuple[float, ...] = (5.0,) * PANDA_DOF
    max_stop_time_s: float = 2.0
    actuator_force_limit_scale: float = 1.0
    state_tolerance: float = 1e-9
    terminal_velocity_tolerance_rad_s: float = 1e-9
    check_inter_sample_edges: bool = True

    def __post_init__(self) -> None:
        raw_acceleration = np.asarray(self.joint_acceleration_limits_rad_s2)
        if raw_acceleration.dtype.kind not in {"i", "u", "f"}:
            raise ValueError("joint acceleration limits must be numeric")
        acceleration = np.asarray(raw_acceleration, dtype=float)
        if (
            acceleration.shape != (PANDA_DOF,)
            or not np.all(np.isfinite(acceleration))
            or np.any(acceleration <= 0.0)
        ):
            raise ValueError(
                "joint acceleration limits must be seven finite positive values"
            )
        sample_dt = _finite_float(self.sample_dt_s, "sample_dt_s")
        max_stop_time = _finite_float(self.max_stop_time_s, "max_stop_time_s")
        force_scale = _finite_float(
            self.actuator_force_limit_scale, "actuator_force_limit_scale"
        )
        state_tolerance = _finite_float(self.state_tolerance, "state_tolerance")
        terminal_tolerance = _finite_float(
            self.terminal_velocity_tolerance_rad_s,
            "terminal_velocity_tolerance_rad_s",
        )
        if sample_dt <= 0.0 or max_stop_time <= 0.0:
            raise ValueError(
                "braking sample period and maximum stop time must be positive"
            )
        if not 0.0 < force_scale <= 1.0:
            raise ValueError("actuator force limit scale must be within (0, 1]")
        if state_tolerance < 0.0 or terminal_tolerance < 0.0:
            raise ValueError("braking tolerances must be finite and nonnegative")
        if type(self.check_inter_sample_edges) is not bool:
            raise ValueError("check_inter_sample_edges must be a boolean")
        object.__setattr__(
            self,
            "joint_acceleration_limits_rad_s2",
            tuple(float(value) for value in acceleration),
        )
        object.__setattr__(self, "sample_dt_s", sample_dt)
        object.__setattr__(self, "max_stop_time_s", max_stop_time)
        object.__setattr__(self, "actuator_force_limit_scale", force_scale)
        object.__setattr__(self, "state_tolerance", state_tolerance)
        object.__setattr__(
            self, "terminal_velocity_tolerance_rad_s", terminal_tolerance
        )

    @property
    def acceleration_limits(self) -> FloatArray:
        result = np.asarray(self.joint_acceleration_limits_rad_s2, dtype=float)
        result.flags.writeable = False
        return result


@dataclass(frozen=True)
class DynamicsBrakingResult:
    """Candidate stop and the prefix that passed every registered check."""

    validated: bool
    failure_reason: str | None
    failure_sample_index: int | None
    times_s: FloatArray
    positions_rad: FloatArray
    velocities_rad_s: FloatArray
    accelerations_rad_s2: FloatArray
    inverse_dynamics_efforts: FloatArray
    actuator_effort_lower_limits: FloatArray | None
    actuator_effort_upper_limits: FloatArray | None
    required_stop_time_s: float
    evaluated_samples: int
    max_torque_ratio: float | None
    inter_sample_edges_checked: bool

    def __post_init__(self) -> None:
        times = _readonly_vector(self.times_s, len(self.times_s), "times_s")
        if len(times) == 0 or not np.isclose(times[0], 0.0):
            raise ValueError("braking times must be nonempty and begin at zero")
        if len(times) > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError("braking times must be strictly increasing")
        positions = _readonly_array(self.positions_rad, (PANDA_DOF,), "positions_rad")
        velocities = _readonly_array(
            self.velocities_rad_s, (PANDA_DOF,), "velocities_rad_s"
        )
        accelerations = _readonly_array(
            self.accelerations_rad_s2,
            (PANDA_DOF,),
            "accelerations_rad_s2",
        )
        efforts = _readonly_array(
            self.inverse_dynamics_efforts,
            (PANDA_DOF,),
            "inverse_dynamics_efforts",
        )
        if not (len(positions) == len(velocities) == len(accelerations) == len(times)):
            raise ValueError("braking state arrays must share the time dimension")
        if len(efforts) != self.evaluated_samples:
            raise ValueError("evaluated sample count must match inverse-dynamics rows")
        if not 0 <= self.evaluated_samples <= len(times):
            raise ValueError(
                "evaluated sample count is outside the candidate trajectory"
            )
        if (self.actuator_effort_lower_limits is None) != (
            self.actuator_effort_upper_limits is None
        ):
            raise ValueError("actuator effort limits must both be present or absent")
        lower: FloatArray | None = None
        upper: FloatArray | None = None
        if self.actuator_effort_lower_limits is not None:
            lower = _readonly_vector(
                self.actuator_effort_lower_limits,
                PANDA_DOF,
                "actuator_effort_lower_limits",
            )
            upper = _readonly_vector(
                self.actuator_effort_upper_limits,
                PANDA_DOF,
                "actuator_effort_upper_limits",
            )
            if np.any(lower >= 0.0) or np.any(upper <= 0.0) or np.any(lower >= upper):
                raise ValueError("actuator effort limits must straddle zero")
        if self.validated != (self.failure_reason is None):
            raise ValueError("validated flag and failure reason disagree")
        if self.validated and self.failure_sample_index is not None:
            raise ValueError("validated result cannot identify a failure sample")
        if not self.validated and self.failure_reason is None:
            raise ValueError("failed result must identify a reason")
        if (
            not np.isfinite(self.required_stop_time_s)
            or self.required_stop_time_s < 0.0
        ):
            raise ValueError("required stop time must be finite and nonnegative")
        if self.max_torque_ratio is not None and (
            not np.isfinite(self.max_torque_ratio) or self.max_torque_ratio < 0.0
        ):
            raise ValueError("maximum torque ratio must be finite and nonnegative")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "positions_rad", positions)
        object.__setattr__(self, "velocities_rad_s", velocities)
        object.__setattr__(self, "accelerations_rad_s2", accelerations)
        object.__setattr__(self, "inverse_dynamics_efforts", efforts)
        object.__setattr__(self, "actuator_effort_lower_limits", lower)
        object.__setattr__(self, "actuator_effort_upper_limits", upper)

    @property
    def stop_time_s(self) -> float:
        return float(self.times_s[-1])

    @property
    def joint_stopping_displacement_rad(self) -> FloatArray:
        result = np.abs(self.positions_rad[-1] - self.positions_rad[0])
        result.flags.writeable = False
        return result

    @property
    def stopping_distance_l2_rad(self) -> float:
        return float(np.linalg.norm(self.positions_rad[-1] - self.positions_rad[0]))

    @property
    def max_joint_stopping_distance_rad(self) -> float:
        return float(np.max(self.joint_stopping_displacement_rad))

    @property
    def max_absolute_effort(self) -> FloatArray | None:
        if len(self.inverse_dynamics_efforts) == 0:
            return None
        result = np.max(np.abs(self.inverse_dynamics_efforts), axis=0)
        result.flags.writeable = False
        return result

    def metrics(self) -> dict[str, object]:
        maximum_effort = self.max_absolute_effort
        return {
            "schema_version": "armbench.dynamics_braking_result.v1",
            "validated": self.validated,
            "failure_reason": self.failure_reason,
            "failure_sample_index": self.failure_sample_index,
            "candidate_samples": len(self.times_s),
            "evaluated_samples": self.evaluated_samples,
            "required_stop_time_s": self.required_stop_time_s,
            "stop_time_s": self.stop_time_s,
            "joint_stopping_displacement_rad": (
                self.joint_stopping_displacement_rad.tolist()
            ),
            "stopping_distance_l2_rad": self.stopping_distance_l2_rad,
            "max_joint_stopping_distance_rad": (self.max_joint_stopping_distance_rad),
            "max_absolute_inverse_dynamics_effort": (
                None if maximum_effort is None else maximum_effort.tolist()
            ),
            "max_torque_ratio": self.max_torque_ratio,
            "actuator_effort_lower_limits": (
                None
                if self.actuator_effort_lower_limits is None
                else self.actuator_effort_lower_limits.tolist()
            ),
            "actuator_effort_upper_limits": (
                None
                if self.actuator_effort_upper_limits is None
                else self.actuator_effort_upper_limits.tolist()
            ),
            "inverse_dynamics_method": INVERSE_DYNAMICS_METHOD,
            "effort_limit_method": EFFORT_LIMIT_METHOD,
            "collision_check": (
                "sample_and_inter_sample_edge"
                if self.inter_sample_edges_checked
                else "sample_only"
            ),
            "scope": "sampled_inverse_dynamics_feasibility_not_hardware_certification",
        }


def _arm_actuator_effort_limits(
    robot: MuJoCoPanda,
    scale: float,
) -> tuple[FloatArray, FloatArray]:
    """Map force-limited joint actuators to generalized hinge efforts."""

    model = robot.model
    lower: list[float] = []
    upper: list[float] = []
    joint_transmission = int(mujoco.mjtTrn.mjTRN_JOINT)
    for joint_id in robot.arm_joint_ids:
        matches = np.flatnonzero(
            (model.actuator_trntype == joint_transmission)
            & (model.actuator_trnid[:, 0] == int(joint_id))
        )
        if len(matches) != 1:
            raise RuntimeError("actuator_mapping_not_one_to_one")
        actuator_id = int(matches[0])
        if not bool(model.actuator_forcelimited[actuator_id]):
            raise RuntimeError("actuator_force_limit_unavailable")
        gear = float(model.actuator_gear[actuator_id, 0])
        force_range = np.asarray(model.actuator_forcerange[actuator_id], dtype=float)
        if (
            not np.isfinite(gear)
            or abs(gear) <= 1e-15
            or force_range.shape != (2,)
            or not np.all(np.isfinite(force_range))
            or force_range[0] >= force_range[1]
        ):
            raise RuntimeError("actuator_force_limit_invalid")
        generalized = np.sort(force_range * gear * scale)
        lower.append(float(generalized[0]))
        upper.append(float(generalized[1]))
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def _candidate_stop(
    q: FloatArray,
    qvel: FloatArray,
    config: DynamicsBrakingConfig,
) -> tuple[float, FloatArray, FloatArray, FloatArray, FloatArray]:
    acceleration_limits = config.acceleration_limits
    minimum_stop_time = float(np.max(np.abs(qvel) / acceleration_limits))
    if np.max(np.abs(qvel)) <= config.terminal_velocity_tolerance_rad_s:
        times = np.array([0.0], dtype=float)
        positions = q[None, :].copy()
        velocities = qvel[None, :].copy()
        accelerations = np.zeros((1, PANDA_DOF), dtype=float)
        return 0.0, times, positions, velocities, accelerations

    if minimum_stop_time > config.max_stop_time_s + config.state_tolerance:
        times = np.array([0.0], dtype=float)
        positions = q[None, :].copy()
        velocities = qvel[None, :].copy()
        accelerations = np.zeros((1, PANDA_DOF), dtype=float)
        return minimum_stop_time, times, positions, velocities, accelerations

    steps = max(1, int(ceil(minimum_stop_time / config.sample_dt_s)))
    stop_time = steps * config.sample_dt_s
    times = np.linspace(0.0, stop_time, steps + 1, dtype=float)
    acceleration = -qvel / stop_time
    positions = (
        q[None, :]
        + times[:, None] * qvel[None, :]
        + 0.5 * times[:, None] ** 2 * acceleration[None, :]
    )
    velocities = qvel[None, :] + times[:, None] * acceleration[None, :]
    velocities[-1] = 0.0
    accelerations = np.repeat(acceleration[None, :], steps + 1, axis=0)
    accelerations[-1] = 0.0
    return minimum_stop_time, times, positions, velocities, accelerations


def _failure_result(
    *,
    reason: str,
    failure_sample_index: int | None,
    times: FloatArray,
    positions: FloatArray,
    velocities: FloatArray,
    accelerations: FloatArray,
    efforts: list[FloatArray],
    lower: FloatArray | None,
    upper: FloatArray | None,
    required_stop_time_s: float,
    max_torque_ratio: float | None,
    inter_sample_edges_checked: bool,
) -> DynamicsBrakingResult:
    return DynamicsBrakingResult(
        validated=False,
        failure_reason=reason,
        failure_sample_index=failure_sample_index,
        times_s=times,
        positions_rad=positions,
        velocities_rad_s=velocities,
        accelerations_rad_s2=accelerations,
        inverse_dynamics_efforts=np.asarray(efforts, dtype=float).reshape(
            (-1, PANDA_DOF)
        ),
        actuator_effort_lower_limits=lower,
        actuator_effort_upper_limits=upper,
        required_stop_time_s=required_stop_time_s,
        evaluated_samples=len(efforts),
        max_torque_ratio=max_torque_ratio,
        inter_sample_edges_checked=inter_sample_edges_checked,
    )


def generate_dynamics_validated_brake(
    robot: MuJoCoPanda,
    collision_checker: BrakingCollisionChecker,
    q: ArrayLike,
    qvel: ArrayLike,
    config: DynamicsBrakingConfig = DynamicsBrakingConfig(),
) -> DynamicsBrakingResult:
    """Generate a synchronized acceleration-limited stop and validate it.

    Contract-invalid inputs raise ``ValueError``.  Physical infeasibility and
    unavailable actuator limits return ``validated=False`` so a caller can
    enter its hold or emergency-stop boundary without executing the candidate.
    """

    if collision_checker.robot is not robot:
        raise ValueError("collision checker must use the same Panda instance")
    if robot.dof != PANDA_DOF:
        raise ValueError("dynamics braking requires a seven-DoF Panda")
    position = robot.validate_configuration(q).copy()
    velocity = _readonly_vector(qvel, PANDA_DOF, "qvel").copy()
    required_time, times, positions, velocities, accelerations = _candidate_stop(
        position, velocity, config
    )

    try:
        effort_lower, effort_upper = _arm_actuator_effort_limits(
            robot, config.actuator_force_limit_scale
        )
    except RuntimeError as error:
        return _failure_result(
            reason=str(error),
            failure_sample_index=None,
            times=times,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            efforts=[],
            lower=None,
            upper=None,
            required_stop_time_s=required_time,
            max_torque_ratio=None,
            inter_sample_edges_checked=config.check_inter_sample_edges,
        )

    initial_position_violation = (
        position < robot.lower_limits - config.state_tolerance
    ) | (position > robot.upper_limits + config.state_tolerance)
    if np.any(initial_position_violation):
        joint = int(np.flatnonzero(initial_position_violation)[0]) + 1
        return _failure_result(
            reason=f"joint_position_limit:joint{joint}",
            failure_sample_index=0,
            times=times,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            efforts=[],
            lower=effort_lower,
            upper=effort_upper,
            required_stop_time_s=required_time,
            max_torque_ratio=None,
            inter_sample_edges_checked=config.check_inter_sample_edges,
        )
    initial_velocity_violation = (
        np.abs(velocity) > robot.velocity_limits + config.state_tolerance
    )
    if np.any(initial_velocity_violation):
        joint = int(np.flatnonzero(initial_velocity_violation)[0]) + 1
        return _failure_result(
            reason=f"joint_velocity_limit:joint{joint}",
            failure_sample_index=0,
            times=times,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            efforts=[],
            lower=effort_lower,
            upper=effort_upper,
            required_stop_time_s=required_time,
            max_torque_ratio=None,
            inter_sample_edges_checked=config.check_inter_sample_edges,
        )
    initial_collision = collision_checker.configuration_failure(position)
    if initial_collision is not None:
        return _failure_result(
            reason=initial_collision,
            failure_sample_index=0,
            times=times,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            efforts=[],
            lower=effort_lower,
            upper=effort_upper,
            required_stop_time_s=required_time,
            max_torque_ratio=None,
            inter_sample_edges_checked=config.check_inter_sample_edges,
        )

    if required_time > config.max_stop_time_s + config.state_tolerance:
        return _failure_result(
            reason="stop_time_limit",
            failure_sample_index=None,
            times=times,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            efforts=[],
            lower=effort_lower,
            upper=effort_upper,
            required_stop_time_s=required_time,
            max_torque_ratio=None,
            inter_sample_edges_checked=config.check_inter_sample_edges,
        )

    inverse_data = mujoco.MjData(robot.model)
    efforts: list[FloatArray] = []
    maximum_ratio: float | None = None
    acceleration_limits = config.acceleration_limits
    tolerance = config.state_tolerance

    for index, (sample_q, sample_qvel, sample_qacc) in enumerate(
        zip(positions, velocities, accelerations)
    ):
        below = sample_q < robot.lower_limits - tolerance
        above = sample_q > robot.upper_limits + tolerance
        if np.any(below | above):
            joint = int(np.flatnonzero(below | above)[0]) + 1
            return _failure_result(
                reason=f"joint_position_limit:joint{joint}",
                failure_sample_index=index,
                times=times,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                efforts=efforts,
                lower=effort_lower,
                upper=effort_upper,
                required_stop_time_s=required_time,
                max_torque_ratio=maximum_ratio,
                inter_sample_edges_checked=config.check_inter_sample_edges,
            )
        velocity_violation = np.abs(sample_qvel) > robot.velocity_limits + tolerance
        if np.any(velocity_violation):
            joint = int(np.flatnonzero(velocity_violation)[0]) + 1
            return _failure_result(
                reason=f"joint_velocity_limit:joint{joint}",
                failure_sample_index=index,
                times=times,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                efforts=efforts,
                lower=effort_lower,
                upper=effort_upper,
                required_stop_time_s=required_time,
                max_torque_ratio=maximum_ratio,
                inter_sample_edges_checked=config.check_inter_sample_edges,
            )
        acceleration_violation = np.abs(sample_qacc) > acceleration_limits + tolerance
        if np.any(acceleration_violation):
            joint = int(np.flatnonzero(acceleration_violation)[0]) + 1
            return _failure_result(
                reason=f"joint_acceleration_limit:joint{joint}",
                failure_sample_index=index,
                times=times,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                efforts=efforts,
                lower=effort_lower,
                upper=effort_upper,
                required_stop_time_s=required_time,
                max_torque_ratio=maximum_ratio,
                inter_sample_edges_checked=config.check_inter_sample_edges,
            )

        collision_failure = collision_checker.configuration_failure(sample_q)
        if collision_failure is not None:
            return _failure_result(
                reason=collision_failure,
                failure_sample_index=index,
                times=times,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                efforts=efforts,
                lower=effort_lower,
                upper=effort_upper,
                required_stop_time_s=required_time,
                max_torque_ratio=maximum_ratio,
                inter_sample_edges_checked=config.check_inter_sample_edges,
            )
        if (
            config.check_inter_sample_edges
            and index > 0
            and not collision_checker.edge_is_valid(positions[index - 1], sample_q)
        ):
            return _failure_result(
                reason="inter_sample_collision",
                failure_sample_index=index,
                times=times,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                efforts=efforts,
                lower=effort_lower,
                upper=effort_upper,
                required_stop_time_s=required_time,
                max_torque_ratio=maximum_ratio,
                inter_sample_edges_checked=config.check_inter_sample_edges,
            )

        robot.set_configuration(inverse_data, sample_q, forward=False)
        inverse_data.qvel[robot.arm_dof_addresses] = sample_qvel
        inverse_data.qacc[:] = 0.0
        inverse_data.qacc[robot.arm_dof_addresses] = sample_qacc
        inverse_data.qfrc_applied[:] = 0.0
        inverse_data.xfrc_applied[:] = 0.0
        try:
            mujoco.mj_inverse(robot.model, inverse_data)
        except Exception as error:
            return _failure_result(
                reason=f"inverse_dynamics_error:{type(error).__name__}",
                failure_sample_index=index,
                times=times,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                efforts=efforts,
                lower=effort_lower,
                upper=effort_upper,
                required_stop_time_s=required_time,
                max_torque_ratio=maximum_ratio,
                inter_sample_edges_checked=config.check_inter_sample_edges,
            )
        effort = inverse_data.qfrc_inverse[robot.arm_dof_addresses].copy()
        if not np.all(np.isfinite(effort)):
            return _failure_result(
                reason="inverse_dynamics_nonfinite",
                failure_sample_index=index,
                times=times,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                efforts=efforts,
                lower=effort_lower,
                upper=effort_upper,
                required_stop_time_s=required_time,
                max_torque_ratio=maximum_ratio,
                inter_sample_edges_checked=config.check_inter_sample_edges,
            )
        efforts.append(effort)
        directional_limit = np.where(effort >= 0.0, effort_upper, -effort_lower)
        ratios = np.abs(effort) / directional_limit
        sample_ratio = float(np.max(ratios))
        maximum_ratio = (
            sample_ratio if maximum_ratio is None else max(maximum_ratio, sample_ratio)
        )
        force_violation = (effort < effort_lower - tolerance) | (
            effort > effort_upper + tolerance
        )
        if np.any(force_violation):
            joint = int(np.flatnonzero(force_violation)[0]) + 1
            return _failure_result(
                reason=f"actuator_force_limit:joint{joint}",
                failure_sample_index=index,
                times=times,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                efforts=efforts,
                lower=effort_lower,
                upper=effort_upper,
                required_stop_time_s=required_time,
                max_torque_ratio=maximum_ratio,
                inter_sample_edges_checked=config.check_inter_sample_edges,
            )

    if np.max(np.abs(velocities[-1])) > config.terminal_velocity_tolerance_rad_s:
        return _failure_result(
            reason="terminal_velocity_nonzero",
            failure_sample_index=len(times) - 1,
            times=times,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            efforts=efforts,
            lower=effort_lower,
            upper=effort_upper,
            required_stop_time_s=required_time,
            max_torque_ratio=maximum_ratio,
            inter_sample_edges_checked=config.check_inter_sample_edges,
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
        inter_sample_edges_checked=config.check_inter_sample_edges,
    )


def run_dynamics_braking_smoke() -> dict[str, object]:
    """Run one deterministic CPU-only inverse-dynamics braking check."""

    from armbench.mujoco_sim.scenarios import mujoco_scenarios

    scenario = mujoco_scenarios()["free_space"]
    robot = MuJoCoPanda.create(obstacles=())
    checker = ContinuousMuJoCoCollisionChecker(robot)
    velocity = np.array([0.20, -0.10, 0.05, 0.0, 0.0, 0.0, 0.0])
    result = generate_dynamics_validated_brake(
        robot,
        checker,
        scenario.start,
        velocity,
        DynamicsBrakingConfig(joint_acceleration_limits_rad_s2=(2.0,) * PANDA_DOF),
    )
    report = result.metrics()
    report["input_joint_velocity_rad_s"] = velocity.tolist()
    report["collision_checker"] = type(checker).__name__
    report["continuous_collision_edges"] = True
    report["policy_checkpoint_used"] = False
    report["hard_realtime_claim"] = False
    return report


def main() -> int:
    """Print the deterministic smoke report for ``python -m`` use."""

    report = run_dynamics_braking_smoke()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["validated"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
