"""PD and discrete finite-state LQR tracking controllers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def solve_discrete_lqr(
    a: FloatArray,
    b: FloatArray,
    q: FloatArray,
    r: FloatArray,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 10_000,
) -> FloatArray:
    """Solve the infinite-horizon discrete Riccati equation by iteration."""

    p = q.copy()
    for _ in range(max_iterations):
        gain = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
        next_p = q + a.T @ p @ a - a.T @ p @ b @ gain
        if float(np.max(np.abs(next_p - p))) <= tolerance:
            return np.linalg.solve(r + b.T @ next_p @ b, b.T @ next_p @ a)
        p = next_p
    raise RuntimeError("discrete Riccati iteration did not converge")


@dataclass(frozen=True)
class PDController:
    kp: FloatArray
    kd: FloatArray
    name: str = "pd"

    @classmethod
    def create(
        cls, dof: int, *, kp: float | ArrayLike = 45.0, kd: float | ArrayLike = 13.0
    ) -> "PDController":
        proportional = np.broadcast_to(np.asarray(kp, dtype=float), (dof,)).copy()
        derivative = np.broadcast_to(np.asarray(kd, dtype=float), (dof,)).copy()
        if np.any(proportional <= 0.0) or np.any(derivative < 0.0):
            raise ValueError("PD gains must be nonnegative with positive kp")
        return cls(proportional, derivative)

    def command(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        reference_position: ArrayLike,
        reference_velocity: ArrayLike,
    ) -> FloatArray:
        return self.kp * (
            np.asarray(reference_position) - np.asarray(position)
        ) + self.kd * (np.asarray(reference_velocity) - np.asarray(velocity))


@dataclass(frozen=True)
class DiscreteLQR:
    gain: FloatArray
    dof: int
    name: str = "lqr"

    @classmethod
    def create(
        cls,
        dof: int,
        *,
        dt: float = 0.02,
        position_weight: float = 80.0,
        velocity_weight: float = 3.0,
        effort_weight: float = 0.25,
    ) -> "DiscreteLQR":
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        a = np.array([[1.0, dt], [0.0, 1.0]])
        b = np.array([[0.5 * dt**2], [dt]])
        q = np.diag([position_weight, velocity_weight])
        r = np.array([[effort_weight]])
        gain = solve_discrete_lqr(a, b, q, r).reshape(2)
        return cls(gain, dof)

    def command(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        reference_position: ArrayLike,
        reference_velocity: ArrayLike,
    ) -> FloatArray:
        position_error = np.asarray(position) - np.asarray(reference_position)
        velocity_error = np.asarray(velocity) - np.asarray(reference_velocity)
        return -(self.gain[0] * position_error + self.gain[1] * velocity_error)

