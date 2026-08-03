"""Random shortcut smoothing with collision revalidation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike

from armbench.collision import CollisionChecker
from armbench.result import path_length


@dataclass(frozen=True)
class ShortcutResult:
    path: list[np.ndarray]
    attempts: int
    accepted: int
    elapsed_s: float
    original_length: float
    smoothed_length: float

    @property
    def reduction_ratio(self) -> float:
        if self.original_length == 0.0:
            return 0.0
        return 1.0 - self.smoothed_length / self.original_length


def shortcut_path(
    path: list[ArrayLike] | tuple[ArrayLike, ...],
    checker: CollisionChecker,
    rng: np.random.Generator,
    *,
    attempts: int = 100,
) -> ShortcutResult:
    if attempts < 0:
        raise ValueError("shortcut attempts cannot be negative")
    working = [checker.robot.validate_configuration(q).copy() for q in path]
    if not working:
        raise ValueError("path cannot be empty")
    if not checker.path_is_valid(working):
        raise ValueError("shortcut input path is not collision-free")
    original_length = path_length(working)
    started_at = perf_counter()
    accepted = 0
    attempted = 0
    for _ in range(attempts):
        if len(working) < 3:
            break
        i, j = sorted(rng.choice(len(working), size=2, replace=False).tolist())
        if j <= i + 1:
            continue
        attempted += 1
        if checker.edge_is_valid(working[i], working[j]):
            working = working[: i + 1] + working[j:]
            accepted += 1
    if not checker.path_is_valid(working):
        raise RuntimeError("shortcut smoothing produced an invalid path")
    return ShortcutResult(
        path=working,
        attempts=attempted,
        accepted=accepted,
        elapsed_s=perf_counter() - started_at,
        original_length=original_length,
        smoothed_length=path_length(working),
    )

