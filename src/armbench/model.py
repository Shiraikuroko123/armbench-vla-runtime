"""Pure NumPy kinematics and limits for the seven-joint benchmark arm."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


# Standard-DH rows: theta offset, alpha, a, d. The table matches the Panda
# model used by the pinned PythonRobotics baseline; limits come from Franka's
# published control parameter documentation. Angles are radians, lengths m.
PANDA_DH = np.array(
    [
        [0.0, np.pi / 2.0, 0.0, 0.333],
        [0.0, -np.pi / 2.0, 0.0, 0.0],
        [0.0, np.pi / 2.0, 0.0825, 0.316],
        [0.0, -np.pi / 2.0, -0.0825, 0.0],
        [0.0, np.pi / 2.0, 0.0, 0.384],
        [0.0, np.pi / 2.0, 0.088, 0.0],
        [0.0, 0.0, 0.0, 0.107],
    ],
    dtype=float,
)

PANDA_LOWER = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=float,
)
PANDA_UPPER = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=float,
)
PANDA_VELOCITY = np.array(
    [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], dtype=float
)


@dataclass(frozen=True)
class RobotModel:
    """Serial revolute arm represented with standard DH parameters."""

    dh: FloatArray
    lower_limits: FloatArray
    upper_limits: FloatArray
    velocity_limits: FloatArray
    name: str = "franka_panda_7dof"

    def __post_init__(self) -> None:
        dh = np.asarray(self.dh, dtype=float)
        lower = np.asarray(self.lower_limits, dtype=float)
        upper = np.asarray(self.upper_limits, dtype=float)
        velocity = np.asarray(self.velocity_limits, dtype=float)
        if dh.ndim != 2 or dh.shape[1] != 4:
            raise ValueError("dh must have shape (n_joints, 4)")
        if lower.shape != (dh.shape[0],) or upper.shape != lower.shape:
            raise ValueError("joint limits must match the number of DH rows")
        if velocity.shape != lower.shape or np.any(velocity <= 0.0):
            raise ValueError("velocity limits must be positive per-joint values")
        if np.any(lower >= upper):
            raise ValueError("every lower joint limit must be below its upper limit")
        object.__setattr__(self, "dh", dh.copy())
        object.__setattr__(self, "lower_limits", lower.copy())
        object.__setattr__(self, "upper_limits", upper.copy())
        object.__setattr__(self, "velocity_limits", velocity.copy())

    @classmethod
    def panda(cls) -> "RobotModel":
        return cls(PANDA_DH, PANDA_LOWER, PANDA_UPPER, PANDA_VELOCITY)

    @property
    def dof(self) -> int:
        return int(self.dh.shape[0])

    def validate_configuration(self, q: ArrayLike) -> FloatArray:
        configuration = np.asarray(q, dtype=float)
        if configuration.shape != (self.dof,):
            raise ValueError(f"configuration must have shape ({self.dof},)")
        if not np.all(np.isfinite(configuration)):
            raise ValueError("configuration must contain only finite values")
        return configuration

    def within_limits(self, q: ArrayLike, *, atol: float = 1e-12) -> bool:
        configuration = self.validate_configuration(q)
        return bool(
            np.all(configuration >= self.lower_limits - atol)
            and np.all(configuration <= self.upper_limits + atol)
        )

    def sample(self, rng: np.random.Generator) -> FloatArray:
        return rng.uniform(self.lower_limits, self.upper_limits)

    def forward_points(self, q: ArrayLike) -> FloatArray:
        """Return base and successive joint-frame origins with shape (n+1, 3)."""

        configuration = self.validate_configuration(q)
        transform = np.eye(4, dtype=float)
        points = [transform[:3, 3].copy()]
        for angle, (offset, alpha, a, d) in zip(configuration, self.dh):
            theta = angle + offset
            st, ct = np.sin(theta), np.cos(theta)
            sa, ca = np.sin(alpha), np.cos(alpha)
            link_transform = np.array(
                [
                    [ct, -st * ca, st * sa, a * ct],
                    [st, ct * ca, -ct * sa, a * st],
                    [0.0, sa, ca, d],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            transform = transform @ link_transform
            points.append(transform[:3, 3].copy())
        return np.asarray(points)

