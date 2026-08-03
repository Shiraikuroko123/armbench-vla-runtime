"""Versioned, deterministic benchmark scenarios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from armbench.geometry import Sphere

FloatArray = NDArray[np.float64]
SCENARIO_VERSION = "1.0"


@dataclass(frozen=True)
class Scenario:
    name: str
    start: FloatArray
    goal: FloatArray
    obstacles: tuple[Sphere, ...]
    description: str
    version: str = SCENARIO_VERSION

    def __post_init__(self) -> None:
        start = np.asarray(self.start, dtype=float)
        goal = np.asarray(self.goal, dtype=float)
        if start.shape != goal.shape or start.ndim != 1:
            raise ValueError("start and goal must be same-length vectors")
        object.__setattr__(self, "start", start.copy())
        object.__setattr__(self, "goal", goal.copy())
        object.__setattr__(self, "obstacles", tuple(self.obstacles))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "start": self.start.tolist(),
            "goal": self.goal.tolist(),
            "obstacles": [obstacle.to_dict() for obstacle in self.obstacles],
        }


_START = np.array([0.0, -0.75, 0.0, -2.25, 0.0, 1.55, 0.75])
_GOAL = np.array([1.2, 0.35, -0.8, -1.35, 0.7, 2.25, -0.65])


def benchmark_scenarios() -> dict[str, Scenario]:
    """Return fresh scenario objects so callers cannot mutate shared arrays."""

    scenarios = (
        Scenario(
            name="free_space",
            start=_START,
            goal=_GOAL,
            obstacles=(),
            description="Limit-respecting sanity check without workspace obstacles.",
        ),
        Scenario(
            name="single_block",
            start=_START,
            goal=_GOAL,
            obstacles=(
                Sphere(np.array([0.33, 0.095, 0.625]), 0.09, "center_block"),
            ),
            description="One central sphere blocks simple joint interpolation.",
        ),
        Scenario(
            name="narrow_passage",
            start=_START,
            goal=_GOAL,
            obstacles=(
                Sphere(np.array([0.33, 0.095, 0.625]), 0.09, "gate_center"),
                Sphere(np.array([0.26, -0.12, 0.68]), 0.10, "gate_lower"),
                Sphere(np.array([0.22, 0.29, 0.66]), 0.10, "gate_upper"),
            ),
            description="Three spheres form a deliberately constrained passage.",
        ),
    )
    return {scenario.name: scenario for scenario in scenarios}
