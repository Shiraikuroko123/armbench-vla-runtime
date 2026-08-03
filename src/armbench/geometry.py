"""Geometry primitives used by collision validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Sphere:
    center: FloatArray
    radius: float
    label: str = "obstacle"

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("sphere center must be a finite three-vector")
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("sphere radius must be positive")
        object.__setattr__(self, "center", center.copy())
        object.__setattr__(self, "radius", float(self.radius))

    def to_dict(self) -> dict[str, object]:
        return {
            "center": self.center.tolist(),
            "radius": self.radius,
            "label": self.label,
        }


def closest_point_on_segment(
    point: ArrayLike, segment_start: ArrayLike, segment_end: ArrayLike
) -> FloatArray:
    """Return the closest point on a closed 3-D line segment."""

    p = np.asarray(point, dtype=float)
    a = np.asarray(segment_start, dtype=float)
    b = np.asarray(segment_end, dtype=float)
    if p.shape != (3,) or a.shape != (3,) or b.shape != (3,):
        raise ValueError("point and segment endpoints must be three-vectors")
    direction = b - a
    squared_length = float(direction @ direction)
    if squared_length <= np.finfo(float).eps:
        return a.copy()
    projection = float((p - a) @ direction) / squared_length
    return a + np.clip(projection, 0.0, 1.0) * direction


def point_to_segment_distance(
    point: ArrayLike, segment_start: ArrayLike, segment_end: ArrayLike
) -> float:
    closest = closest_point_on_segment(point, segment_start, segment_end)
    return float(np.linalg.norm(np.asarray(point, dtype=float) - closest))


def segment_intersects_sphere(
    segment_start: ArrayLike,
    segment_end: ArrayLike,
    sphere: Sphere,
    *,
    link_radius: float = 0.0,
    safety_margin: float = 0.0,
) -> bool:
    if link_radius < 0.0 or safety_margin < 0.0:
        raise ValueError("link radius and safety margin cannot be negative")
    threshold = sphere.radius + link_radius + safety_margin
    distance = point_to_segment_distance(sphere.center, segment_start, segment_end)
    return bool(distance <= threshold + 1e-12)

