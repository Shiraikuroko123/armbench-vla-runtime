"""Planner result types and shared path metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class PlanStatus(str, Enum):
    SUCCESS = "success"
    START_IN_COLLISION = "start_in_collision"
    GOAL_IN_COLLISION = "goal_in_collision"
    TIMEOUT = "timeout"
    ITERATION_LIMIT = "iteration_limit"


@dataclass
class PlanResult:
    planner: str
    status: PlanStatus
    elapsed_s: float
    iterations: int
    nodes: int
    collision_queries: int
    edge_queries: int
    path: list[FloatArray] = field(default_factory=list)
    detail: str = ""

    @property
    def success(self) -> bool:
        return self.status is PlanStatus.SUCCESS

    @property
    def path_length(self) -> float | None:
        if not self.success or len(self.path) < 2:
            return None
        return path_length(self.path)


def path_length(path: list[FloatArray] | tuple[FloatArray, ...]) -> float:
    if len(path) < 2:
        return 0.0
    points = np.asarray(path, dtype=float)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

