"""Decoupled double-integrator joint plant with load and viscous damping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class JointState:
    position: FloatArray
    velocity: FloatArray


@dataclass(frozen=True)
class JointPlant:
    dt: float
    inertia_scale: FloatArray
    damping: FloatArray
    process_noise_std: float = 0.0

    @classmethod
    def create(
        cls,
        dof: int,
        *,
        dt: float = 0.02,
        inertia_scale: float | ArrayLike = 1.0,
        damping: float | ArrayLike = 0.15,
        process_noise_std: float = 0.0,
    ) -> "JointPlant":
        if dt <= 0.0 or process_noise_std < 0.0:
            raise ValueError("invalid plant time step or noise")
        inertia = np.broadcast_to(np.asarray(inertia_scale, dtype=float), (dof,)).copy()
        damping_array = np.broadcast_to(np.asarray(damping, dtype=float), (dof,)).copy()
        if np.any(inertia <= 0.0) or np.any(damping_array < 0.0):
            raise ValueError("inertia must be positive and damping nonnegative")
        return cls(float(dt), inertia, damping_array, float(process_noise_std))

    def step(
        self,
        state: JointState,
        command: ArrayLike,
        rng: np.random.Generator,
    ) -> JointState:
        u = np.asarray(command, dtype=float)
        if u.shape != state.position.shape:
            raise ValueError("command shape must match state")
        acceleration = (u - self.damping * state.velocity) / self.inertia_scale
        if self.process_noise_std > 0.0:
            acceleration = acceleration + rng.normal(
                0.0, self.process_noise_std, size=acceleration.shape
            )
        next_position = (
            state.position
            + self.dt * state.velocity
            + 0.5 * self.dt**2 * acceleration
        )
        next_velocity = state.velocity + self.dt * acceleration
        return JointState(next_position, next_velocity)

