"""Conservative bounding-sphere pruning for continuous Panda edge checks.

MuJoCo publishes ``geom_rbound`` as a bounding-sphere radius.  At one
configuration, the distance between two sphere surfaces is therefore a lower
bound on the corresponding geometry distance.  Over a joint-space interval,
subtracting the existing per-geometry motion bound keeps that lower bound
valid for every configuration in the interval.

This checker only skips an exact ``mj_geomDistance`` call when that lower
bound is already above the registered clearance.  Ambiguous pairs still use
the reference checker and an ambiguous interval still fails closed.
"""

from __future__ import annotations

import numpy as np

from armbench.mujoco_sim.continuous_collision import (
    ContinuousCollisionConfig,
    ContinuousCollisionPair,
    ContinuousMuJoCoCollisionChecker,
    _CertificateState,
)
from armbench.mujoco_sim.model import MuJoCoPanda


class BroadPhaseContinuousMuJoCoCollisionChecker(
    ContinuousMuJoCoCollisionChecker
):
    """Prune exact pair distances with conservative world-space sphere bounds."""

    def __init__(
        self,
        robot: MuJoCoPanda,
        config: ContinuousCollisionConfig = ContinuousCollisionConfig(),
    ) -> None:
        super().__init__(robot, config)
        self._refresh_pair_arrays()
        self.broad_phase_pair_tests = 0
        self.broad_phase_pruned_pairs = 0

    def set_pairs(self, pairs: tuple[ContinuousCollisionPair, ...]) -> None:
        if not pairs:
            raise ValueError("broad-phase checker requires collision pairs")
        self.pairs = pairs
        self._refresh_pair_arrays()

    def reset_metrics(self) -> None:
        self.broad_phase_pair_tests = 0
        self.broad_phase_pruned_pairs = 0

    @property
    def broad_phase_prune_rate(self) -> float:
        if self.broad_phase_pair_tests == 0:
            return 0.0
        return self.broad_phase_pruned_pairs / self.broad_phase_pair_tests

    def _refresh_pair_arrays(self) -> None:
        self._pair_geom1 = np.asarray(
            [pair.geom1 for pair in self.pairs], dtype=np.int32
        )
        self._pair_geom2 = np.asarray(
            [pair.geom2 for pair in self.pairs], dtype=np.int32
        )
        model = self.robot.model
        self._pair_radius_sums = (
            model.geom_rbound[self._pair_geom1]
            + model.geom_rbound[self._pair_geom2]
        ).astype(float, copy=True)
        self._pair_joint_motion_radii_m = (
            self.geom_joint_motion_radii_m[self._pair_geom1]
            + self.geom_joint_motion_radii_m[self._pair_geom2]
        ).astype(float, copy=True)
        if (
            self._pair_radius_sums.shape != (len(self.pairs),)
            or not np.all(np.isfinite(self._pair_radius_sums))
            or np.any(self._pair_radius_sums < 0.0)
            or self._pair_joint_motion_radii_m.shape
            != (len(self.pairs), self.robot.dof)
            or not np.all(np.isfinite(self._pair_joint_motion_radii_m))
            or np.any(self._pair_joint_motion_radii_m < 0.0)
        ):
            raise RuntimeError("MuJoCo geometry bounds are invalid")

    def _sphere_distance_lower_bounds(self) -> np.ndarray:
        first = self.data.geom_xpos[self._pair_geom1]
        second = self.data.geom_xpos[self._pair_geom2]
        return np.linalg.norm(first - second, axis=1) - self._pair_radius_sums

    def _configuration_collision(
        self,
        q: np.ndarray,
        state: _CertificateState,
    ) -> bool | None:
        self.robot.set_configuration(self.data, q)
        lower_bounds = self._sphere_distance_lower_bounds()
        threshold = self.config.clearance_m + self.config.distance_tolerance_m
        candidates = np.flatnonzero(lower_bounds <= threshold)
        self.broad_phase_pair_tests += len(self.pairs)
        self.broad_phase_pruned_pairs += len(self.pairs) - len(candidates)
        for index in candidates:
            pair = self.pairs[int(index)]
            distance = self._pair_distance(pair, state)
            if distance is None:
                return None
            if distance <= threshold:
                state.collision_pair = pair.label
                state.failure_reason = f"collision:{pair.label}"
                return True
        return False

    def _interval_status(
        self,
        start: np.ndarray,
        end: np.ndarray,
        depth: int,
        state: _CertificateState,
    ) -> bool | None:
        state.subintervals += 1
        state.maximum_depth = max(state.maximum_depth, depth)
        midpoint = 0.5 * (start + end)
        self.robot.set_configuration(self.data, midpoint)
        lower_bounds = self._sphere_distance_lower_bounds()
        delta = np.abs(end - start)
        uncertain = False
        collision_threshold = (
            self.config.clearance_m + self.config.distance_tolerance_m
        )
        motion_bounds = 0.5 * (self._pair_joint_motion_radii_m @ delta)
        if len(motion_bounds):
            state.maximum_motion_bound_m = max(
                state.maximum_motion_bound_m, float(np.max(motion_bounds))
            )
        candidates = np.flatnonzero(
            lower_bounds <= collision_threshold + motion_bounds
        )
        self.broad_phase_pair_tests += len(self.pairs)
        self.broad_phase_pruned_pairs += len(self.pairs) - len(candidates)
        for index in candidates:
            pair = self.pairs[int(index)]
            distance = self._pair_distance(pair, state)
            if distance is None:
                return None
            if distance <= collision_threshold:
                state.collision_pair = pair.label
                state.failure_reason = f"collision:{pair.label}"
                return False
            if distance <= collision_threshold + motion_bounds[index]:
                uncertain = True
        if not uncertain:
            return True
        if depth >= self.config.max_depth:
            state.failure_reason = "maximum_subdivision_depth"
            return None
        if float(np.max(delta)) <= self.config.minimum_interval_rad:
            state.failure_reason = "minimum_interval_indeterminate"
            return None
        left = self._interval_status(start, midpoint, depth + 1, state)
        if left is not True:
            return left
        return self._interval_status(midpoint, end, depth + 1, state)


__all__ = ["BroadPhaseContinuousMuJoCoCollisionChecker"]
