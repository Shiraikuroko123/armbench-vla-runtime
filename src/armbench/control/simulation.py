"""Tracking experiment with delayed observations and command saturation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from armbench.collision import CollisionChecker
from armbench.control.joint_plant import JointPlant, JointState
from armbench.postprocess.time_parameterization import Trajectory

FloatArray = NDArray[np.float64]


class Controller(Protocol):
    name: str

    def command(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        reference_position: ArrayLike,
        reference_velocity: ArrayLike,
    ) -> FloatArray: ...


@dataclass(frozen=True)
class TrackingResult:
    controller: str
    delay_ms: int
    inertia_scale: float
    times: FloatArray
    desired_positions: FloatArray
    actual_positions: FloatArray
    commands: FloatArray
    rmse: float
    per_joint_rmse: FloatArray
    max_error: float
    settling_time_s: float | None
    saturation_count: int
    invalid_state_samples: int
    joint_limit_violation_samples: int
    collision_violation_samples: int
    invalid_edge_intervals: int

    def metrics(self) -> dict[str, object]:
        return {
            "controller": self.controller,
            "delay_ms": self.delay_ms,
            "inertia_scale": self.inertia_scale,
            "rmse": self.rmse,
            "per_joint_rmse": self.per_joint_rmse.tolist(),
            "max_error": self.max_error,
            "settling_time_s": self.settling_time_s,
            "saturation_count": self.saturation_count,
            "invalid_state_samples": self.invalid_state_samples,
            "joint_limit_violation_samples": self.joint_limit_violation_samples,
            "collision_violation_samples": self.collision_violation_samples,
            "invalid_edge_intervals": self.invalid_edge_intervals,
        }


def simulate_tracking(
    trajectory: Trajectory,
    controller: Controller,
    rng: np.random.Generator,
    *,
    delay_ms: int = 0,
    inertia_scale: float = 1.0,
    acceleration_limits: float | ArrayLike = 10.0,
    damping: float = 0.15,
    measurement_noise_std: float = 0.001,
    process_noise_std: float = 0.0,
    hold_time_s: float = 1.0,
    settling_threshold: float = 0.03,
    checker: CollisionChecker | None = None,
) -> TrackingResult:
    if delay_ms < 0 or hold_time_s < 0.0 or settling_threshold <= 0.0:
        raise ValueError("invalid delay, hold time, or settling threshold")
    if measurement_noise_std < 0.0:
        raise ValueError("measurement noise cannot be negative")
    dt = float(np.median(np.diff(trajectory.times)))
    delay_steps_float = (delay_ms / 1000.0) / dt
    delay_steps = int(round(delay_steps_float))
    if not np.isclose(delay_steps_float, delay_steps, atol=1e-9):
        raise ValueError("delay_ms must be an integer multiple of the control period")
    limits = np.broadcast_to(
        np.asarray(acceleration_limits, dtype=float), (trajectory.dof,)
    ).copy()
    if np.any(limits <= 0.0):
        raise ValueError("acceleration limits must be positive")

    final_time = trajectory.duration + hold_time_s
    times = np.arange(0.0, final_time + 0.5 * dt, dt)
    desired_positions, desired_velocities = trajectory.sample(times)
    actual_positions = np.empty_like(desired_positions)
    commands = np.empty_like(desired_positions)
    plant = JointPlant.create(
        trajectory.dof,
        dt=dt,
        inertia_scale=inertia_scale,
        damping=damping,
        process_noise_std=process_noise_std,
    )
    state = JointState(desired_positions[0].copy(), np.zeros(trajectory.dof))
    history = [state]
    saturation_count = 0
    invalid_state_samples = 0
    joint_limit_violation_samples = 0
    collision_violation_samples = 0
    invalid_edge_intervals = 0

    for index, _ in enumerate(times):
        actual_positions[index] = state.position
        observed_index = max(0, len(history) - 1 - delay_steps)
        observed = history[observed_index]
        observed_position = observed.position + rng.normal(
            0.0, measurement_noise_std, size=trajectory.dof
        )
        observed_velocity = observed.velocity + rng.normal(
            0.0, measurement_noise_std, size=trajectory.dof
        )
        requested = controller.command(
            observed_position,
            observed_velocity,
            desired_positions[index],
            desired_velocities[index],
        )
        clipped = np.clip(requested, -limits, limits)
        saturation_count += int(np.count_nonzero(np.abs(requested - clipped) > 1e-12))
        commands[index] = clipped
        if checker is not None:
            failure = checker.configuration_failure(state.position)
            if failure is not None:
                invalid_state_samples += 1
                if failure in {"joint_limit", "invalid_configuration"}:
                    joint_limit_violation_samples += 1
                elif failure.startswith("collision:"):
                    collision_violation_samples += 1
            if index > 0 and not checker.edge_is_valid(
                actual_positions[index - 1], state.position
            ):
                invalid_edge_intervals += 1
        if index + 1 < len(times):
            state = plant.step(state, clipped, rng)
            history.append(state)

    errors = actual_positions - desired_positions
    per_joint_rmse = np.sqrt(np.mean(errors**2, axis=0))
    rmse = float(np.sqrt(np.mean(errors**2)))
    max_error = float(np.max(np.abs(errors)))
    after_motion = np.flatnonzero(times >= trajectory.duration - 1e-12)
    settling_time: float | None = None
    for index in after_motion:
        if np.all(np.max(np.abs(errors[index:]), axis=1) <= settling_threshold):
            settling_time = float(times[index] - trajectory.duration)
            break
    return TrackingResult(
        controller=controller.name,
        delay_ms=delay_ms,
        inertia_scale=float(inertia_scale),
        times=times,
        desired_positions=desired_positions,
        actual_positions=actual_positions,
        commands=commands,
        rmse=rmse,
        per_joint_rmse=per_joint_rmse,
        max_error=max_error,
        settling_time_s=settling_time,
        saturation_count=saturation_count,
        invalid_state_samples=invalid_state_samples,
        joint_limit_violation_samples=joint_limit_violation_samples,
        collision_violation_samples=collision_violation_samples,
        invalid_edge_intervals=invalid_edge_intervals,
    )
