"""Shared planner building blocks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from numpy.typing import NDArray
import numpy as np

FloatArray = NDArray[np.float64]


class ExtendStatus(str, Enum):
    TRAPPED = "trapped"
    ADVANCED = "advanced"
    REACHED = "reached"


@dataclass
class TreeNode:
    q: FloatArray
    parent: int | None
    cost: float = 0.0


def nearest_index(nodes: list[TreeNode], target: FloatArray) -> int:
    distances = [float(np.sum((node.q - target) ** 2)) for node in nodes]
    return int(np.argmin(distances))


def steer(q_from: FloatArray, q_to: FloatArray, step_size: float) -> FloatArray:
    difference = q_to - q_from
    distance = float(np.linalg.norm(difference))
    if distance <= step_size:
        return q_to.copy()
    return q_from + (step_size / distance) * difference


def trace_to_root(nodes: list[TreeNode], index: int) -> list[FloatArray]:
    reversed_path: list[FloatArray] = []
    current: int | None = index
    while current is not None:
        reversed_path.append(nodes[current].q.copy())
        current = nodes[current].parent
    return list(reversed(reversed_path))

