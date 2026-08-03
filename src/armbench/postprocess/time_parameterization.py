"""First-order, per-joint velocity-limited path timing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Trajectory:
    times: FloatArray
    positions: FloatArray
    velocities: FloatArray
    segment_durations: FloatArray

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    @property
    def dof(self) -> int:
        return int(self.positions.shape[1])

    def sample(self, query_times: ArrayLike) -> tuple[FloatArray, FloatArray]:
        query = np.asarray(query_times, dtype=float)
        if query.ndim != 1:
            raise ValueError("query_times must be one-dimensional")
        clipped = np.clip(query, self.times[0], self.times[-1])
        positions = np.column_stack(
            [
                np.interp(clipped, self.times, self.positions[:, joint])
                for joint in range(self.dof)
            ]
        )
        indices = np.searchsorted(self.times, clipped, side="right") - 1
        indices = np.clip(indices, 0, len(self.times) - 1)
        velocities = self.velocities[indices].copy()
        velocities[query >= self.times[-1]] = 0.0
        return positions, velocities


def time_parameterize(
    path: list[ArrayLike] | tuple[ArrayLike, ...],
    velocity_limits: ArrayLike,
    *,
    control_dt: float = 0.02,
    speed_scale: float = 0.35,
) -> Trajectory:
    """Interpolate a path using the slowest joint in each segment.

    This enforces velocity bounds only. Acceleration and jerk are intentionally
    outside the first-order benchmark and are disclosed in the project report.
    """

    if control_dt <= 0.0:
        raise ValueError("control_dt must be positive")
    if not 0.0 < speed_scale <= 1.0:
        raise ValueError("speed_scale must be in (0, 1]")
    points = np.asarray(path, dtype=float)
    limits = np.asarray(velocity_limits, dtype=float) * speed_scale
    if points.ndim != 2 or len(points) < 2:
        raise ValueError("path must contain at least two configurations")
    if limits.shape != (points.shape[1],) or np.any(limits <= 0.0):
        raise ValueError("velocity_limits must be positive and match path width")

    keep = np.concatenate(([True], np.any(np.abs(np.diff(points, axis=0)) > 1e-12, axis=1)))
    points = points[keep]
    if len(points) < 2:
        raise ValueError("path must contain at least two distinct configurations")
    deltas = np.diff(points, axis=0)
    durations = np.max(np.abs(deltas) / limits, axis=1)
    durations = np.maximum(durations, control_dt)
    boundaries = np.concatenate(([0.0], np.cumsum(durations)))
    total = float(boundaries[-1])
    times = np.arange(0.0, total, control_dt)
    if len(times) == 0 or total - times[-1] > 1e-12:
        times = np.append(times, total)
    else:
        times[-1] = total

    segment_indices = np.searchsorted(boundaries[1:], times, side="right")
    segment_indices = np.clip(segment_indices, 0, len(durations) - 1)
    fractions = (
        (times - boundaries[segment_indices]) / durations[segment_indices]
    )
    fractions = np.clip(fractions, 0.0, 1.0)
    positions = points[segment_indices] + fractions[:, None] * deltas[segment_indices]
    velocities = deltas[segment_indices] / durations[segment_indices, None]
    positions[-1] = points[-1]
    velocities[-1] = 0.0
    return Trajectory(times, positions, velocities, durations)

