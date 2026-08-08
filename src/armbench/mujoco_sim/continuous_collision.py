"""Conservative continuous certificates for linear joint-space Panda edges."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from armbench.mujoco_sim.model import MuJoCoPanda


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ContinuousCollisionConfig:
    clearance_m: float = 0.0
    distance_tolerance_m: float = 1e-8
    distance_upper_bound_m: float = 10.0
    max_depth: int = 16
    minimum_interval_rad: float = 1e-6
    max_pair_evaluations: int = 250_000
    include_static_obstacles: bool = True
    include_self_collision: bool = True

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.clearance_m)
            or self.clearance_m < 0.0
            or not np.isfinite(self.distance_tolerance_m)
            or self.distance_tolerance_m < 0.0
            or not np.isfinite(self.distance_upper_bound_m)
            or self.distance_upper_bound_m <= self.clearance_m
            or not np.isfinite(self.minimum_interval_rad)
            or self.minimum_interval_rad <= 0.0
        ):
            raise ValueError("continuous collision distances are invalid")
        if type(self.max_depth) is not int or self.max_depth < 0:
            raise ValueError("max_depth must be a nonnegative integer")
        if (
            type(self.max_pair_evaluations) is not int
            or self.max_pair_evaluations <= 0
        ):
            raise ValueError("max_pair_evaluations must be a positive integer")
        if not self.include_static_obstacles and not self.include_self_collision:
            raise ValueError("at least one collision pair class must be enabled")


@dataclass(frozen=True)
class ContinuousCollisionPair:
    geom1: int
    geom2: int
    kind: str
    label: str


@dataclass(frozen=True)
class ContinuousCollisionCertificate:
    status: str
    reason: str
    certified_safe: bool
    collision_pair: str | None
    pair_evaluations: int
    subintervals_evaluated: int
    maximum_depth_reached: int
    minimum_sampled_distance_m: float | None
    maximum_interval_motion_bound_m: float
    latency_ms: float

    def metrics(self) -> dict[str, object]:
        return {
            "scope": "continuous_linear_joint_edge_distance_bound_certificate",
            "status": self.status,
            "reason": self.reason,
            "certified_safe": self.certified_safe,
            "collision_pair": self.collision_pair,
            "pair_evaluations": self.pair_evaluations,
            "subintervals_evaluated": self.subintervals_evaluated,
            "maximum_depth_reached": self.maximum_depth_reached,
            "minimum_sampled_distance_m": self.minimum_sampled_distance_m,
            "maximum_interval_motion_bound_m": (
                self.maximum_interval_motion_bound_m
            ),
            "latency_ms": self.latency_ms,
        }


@dataclass
class _CertificateState:
    pair_evaluations: int = 0
    subintervals: int = 0
    maximum_depth: int = 0
    minimum_distance_m: float = float("inf")
    maximum_motion_bound_m: float = 0.0
    collision_pair: str | None = None
    failure_reason: str | None = None


class ContinuousMuJoCoCollisionChecker:
    """Certify a linear joint edge or reject it as collision/indeterminate.

    A returned safe edge is covered continuously under the compiled MuJoCo
    geometry and the declared joint-linear interpolation. An indeterminate edge
    is fail-closed and never exposed as safe.
    """

    def __init__(
        self,
        robot: MuJoCoPanda,
        config: ContinuousCollisionConfig = ContinuousCollisionConfig(),
    ) -> None:
        self.robot = robot
        self.config = config
        self.data = mujoco.MjData(robot.model)
        self.geom_joint_motion_radii_m = self._geom_joint_motion_radii()
        self.pairs = self._collision_pairs()
        if not self.pairs:
            raise ValueError("continuous checker has no enabled collision pairs")

    def _is_descendant(self, body_id: int, ancestor_id: int) -> bool:
        current = int(body_id)
        while current != 0:
            if current == ancestor_id:
                return True
            current = int(self.robot.model.body_parentid[current])
        return False

    def _geom_joint_motion_radii(self) -> FloatArray:
        model = self.robot.model
        result = np.zeros((model.ngeom, self.robot.dof), dtype=float)
        for geom_id in self.robot.robot_geom_ids:
            geom_body = int(model.geom_bodyid[geom_id])
            for column, joint_id in enumerate(self.robot.arm_joint_ids):
                if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                    raise ValueError("continuous checker currently requires hinge joints")
                joint_body = int(model.jnt_bodyid[joint_id])
                if not self._is_descendant(geom_body, joint_body):
                    continue
                bound = (
                    float(np.linalg.norm(model.jnt_pos[joint_id]))
                    + float(np.linalg.norm(model.geom_pos[geom_id]))
                    + float(model.geom_rbound[geom_id])
                )
                current = geom_body
                while current != joint_body:
                    bound += float(np.linalg.norm(model.body_pos[current]))
                    current = int(model.body_parentid[current])
                result[int(geom_id), column] = bound
        if not np.all(np.isfinite(result)) or np.any(result < 0.0):
            raise RuntimeError("failed to derive finite geometry motion bounds")
        return result

    def _body_graph_distance(self, first: int, second: int) -> int:
        first_depth: dict[int, int] = {}
        current = first
        depth = 0
        while True:
            first_depth[current] = depth
            if current == 0:
                break
            parent = int(self.robot.model.body_parentid[current])
            current = self._collision_body_id(parent)
            depth += 1
        current = second
        depth = 0
        while current not in first_depth:
            parent = int(self.robot.model.body_parentid[current])
            current = self._collision_body_id(parent)
            depth += 1
        return depth + first_depth[current]

    def _collision_body_id(self, body_id: int) -> int:
        """Return MuJoCo's welded body representative for contact filtering."""

        return int(self.robot.model.body_weldid[body_id])

    def _excluded_body_pairs(self) -> set[tuple[int, int]]:
        excluded: set[tuple[int, int]] = set()
        for raw_signature in self.robot.model.exclude_signature:
            signature = int(raw_signature)
            first = signature >> 16
            second = signature & 0xFFFF
            excluded.add(tuple(sorted((first, second))))
        return excluded

    def _masks_compatible(self, first: int, second: int) -> bool:
        model = self.robot.model
        return bool(
            (int(model.geom_contype[first]) & int(model.geom_conaffinity[second]))
            or (
                int(model.geom_contype[second])
                & int(model.geom_conaffinity[first])
            )
        )

    def _geom_label(self, geom_id: int) -> str:
        name = mujoco.mj_id2name(
            self.robot.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        )
        return name or f"{self.robot.body_name_for_geom(geom_id)}:geom_{geom_id}"

    def _collision_pairs(self) -> tuple[ContinuousCollisionPair, ...]:
        model = self.robot.model
        active_robot_geoms = sorted(
            int(geom_id)
            for geom_id in self.robot.robot_geom_ids
            if int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        )
        pairs: list[ContinuousCollisionPair] = []
        if self.config.include_static_obstacles:
            for obstacle_id, obstacle_label in sorted(
                self.robot.obstacle_geom_labels.items()
            ):
                for robot_geom in active_robot_geoms:
                    if not self._masks_compatible(obstacle_id, robot_geom):
                        continue
                    pairs.append(
                        ContinuousCollisionPair(
                            geom1=int(obstacle_id),
                            geom2=robot_geom,
                            kind="static_obstacle",
                            label=(
                                f"static:{obstacle_label}:"
                                f"{self.robot.body_name_for_geom(robot_geom)}"
                            ),
                        )
                    )
        if self.config.include_self_collision:
            excluded = self._excluded_body_pairs()
            for index, first in enumerate(active_robot_geoms):
                first_body = self._collision_body_id(
                    int(model.geom_bodyid[first])
                )
                for second in active_robot_geoms[index + 1 :]:
                    second_body = self._collision_body_id(
                        int(model.geom_bodyid[second])
                    )
                    body_pair = tuple(sorted((first_body, second_body)))
                    if (
                        first_body == second_body
                        or self._body_graph_distance(first_body, second_body) <= 1
                        or body_pair in excluded
                        or not self._masks_compatible(first, second)
                    ):
                        continue
                    pairs.append(
                        ContinuousCollisionPair(
                            geom1=first,
                            geom2=second,
                            kind="self_collision",
                            label=(
                                f"self:{self._geom_label(first)}:"
                                f"{self._geom_label(second)}"
                            ),
                        )
                    )
        return tuple(pairs)

    def _pair_distance(
        self,
        pair: ContinuousCollisionPair,
        state: _CertificateState,
    ) -> float | None:
        if state.pair_evaluations >= self.config.max_pair_evaluations:
            state.failure_reason = "pair_evaluation_budget"
            return None
        fromto = np.empty(6, dtype=float)
        distance = float(
            mujoco.mj_geomDistance(
                self.robot.model,
                self.data,
                pair.geom1,
                pair.geom2,
                self.config.distance_upper_bound_m,
                fromto,
            )
        )
        state.pair_evaluations += 1
        if not np.isfinite(distance):
            state.failure_reason = "nonfinite_geometry_distance"
            return None

        # MuJoCo 3.11 can report signed zero for separated mesh pairs while
        # returning distinct closest points. Preserve real contacts first;
        # otherwise the witness segment is the usable separation distance.
        if abs(distance) <= self.config.distance_tolerance_m:
            matching_contact_distances = [
                float(contact.dist)
                for contact in self.data.contact[: self.data.ncon]
                if {int(contact.geom1), int(contact.geom2)}
                == {pair.geom1, pair.geom2}
            ]
            if matching_contact_distances:
                distance = min(matching_contact_distances)
            else:
                witness_distance = float(
                    np.linalg.norm(fromto[:3] - fromto[3:])
                )
                if not np.isfinite(witness_distance):
                    state.failure_reason = "nonfinite_geometry_witness"
                    return None
                if witness_distance > self.config.distance_tolerance_m:
                    distance = witness_distance
        state.minimum_distance_m = min(state.minimum_distance_m, distance)
        return distance

    def _configuration_collision(
        self,
        q: FloatArray,
        state: _CertificateState,
    ) -> bool | None:
        self.robot.set_configuration(self.data, q)
        for pair in self.pairs:
            distance = self._pair_distance(pair, state)
            if distance is None:
                return None
            if distance <= self.config.clearance_m + self.config.distance_tolerance_m:
                state.collision_pair = pair.label
                state.failure_reason = f"collision:{pair.label}"
                return True
        return False

    def _interval_status(
        self,
        start: FloatArray,
        end: FloatArray,
        depth: int,
        state: _CertificateState,
    ) -> bool | None:
        state.subintervals += 1
        state.maximum_depth = max(state.maximum_depth, depth)
        midpoint = 0.5 * (start + end)
        self.robot.set_configuration(self.data, midpoint)
        delta = np.abs(end - start)
        uncertain = False
        for pair in self.pairs:
            distance = self._pair_distance(pair, state)
            if distance is None:
                return None
            if distance <= self.config.clearance_m + self.config.distance_tolerance_m:
                state.collision_pair = pair.label
                state.failure_reason = f"collision:{pair.label}"
                return False
            motion_bound = 0.5 * float(
                np.dot(
                    self.geom_joint_motion_radii_m[pair.geom1]
                    + self.geom_joint_motion_radii_m[pair.geom2],
                    delta,
                )
            )
            state.maximum_motion_bound_m = max(
                state.maximum_motion_bound_m, motion_bound
            )
            if (
                distance
                <= self.config.clearance_m
                + self.config.distance_tolerance_m
                + motion_bound
            ):
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

    def edge_certificate(
        self, q_start: ArrayLike, q_end: ArrayLike
    ) -> ContinuousCollisionCertificate:
        started_at = perf_counter()
        state = _CertificateState()
        try:
            start = self.robot.validate_configuration(q_start).copy()
            end = self.robot.validate_configuration(q_end).copy()
        except ValueError:
            state.failure_reason = "invalid_configuration"
            result: bool | None = None
        else:
            if not self.robot.within_limits(start) or not self.robot.within_limits(end):
                state.failure_reason = "joint_limit"
                result = None
            else:
                start_collision = self._configuration_collision(start, state)
                if start_collision is None:
                    result = None
                elif start_collision:
                    result = False
                else:
                    end_collision = self._configuration_collision(end, state)
                    if end_collision is None:
                        result = None
                    elif end_collision:
                        result = False
                    else:
                        result = self._interval_status(start, end, 0, state)
        if result is True:
            status = "certified_safe"
            reason = "all_pair_distance_bounds_positive"
        elif result is False:
            status = "collision"
            reason = state.failure_reason or "collision"
        else:
            status = "indeterminate"
            reason = state.failure_reason or "certificate_failed"
        return ContinuousCollisionCertificate(
            status=status,
            reason=reason,
            certified_safe=result is True,
            collision_pair=state.collision_pair,
            pair_evaluations=state.pair_evaluations,
            subintervals_evaluated=state.subintervals,
            maximum_depth_reached=state.maximum_depth,
            minimum_sampled_distance_m=(
                None
                if not np.isfinite(state.minimum_distance_m)
                else state.minimum_distance_m
            ),
            maximum_interval_motion_bound_m=state.maximum_motion_bound_m,
            latency_ms=(perf_counter() - started_at) * 1000.0,
        )

    def configuration_failure(self, q: ArrayLike) -> str | None:
        try:
            configuration = self.robot.validate_configuration(q)
        except ValueError:
            return "invalid_configuration"
        if not self.robot.within_limits(configuration):
            return "joint_limit"
        state = _CertificateState()
        collision = self._configuration_collision(configuration, state)
        if collision is None:
            return state.failure_reason or "indeterminate_configuration"
        return state.failure_reason if collision else None

    def configuration_is_valid(self, q: ArrayLike) -> bool:
        return self.configuration_failure(q) is None

    def edge_is_valid(self, q_start: ArrayLike, q_end: ArrayLike) -> bool:
        return self.edge_certificate(q_start, q_end).certified_safe

    def path_is_valid(self, path: Iterable[ArrayLike]) -> bool:
        configurations = [self.robot.validate_configuration(q) for q in path]
        if not configurations:
            return False
        if len(configurations) == 1:
            return self.configuration_is_valid(configurations[0])
        return all(
            self.edge_is_valid(start, end)
            for start, end in zip(configurations[:-1], configurations[1:])
        )


def run_continuous_collision_smoke() -> dict[str, object]:
    """Exercise safe, static-collision, and self-collision edge decisions."""

    from armbench.mujoco_sim.scenarios import mujoco_scenarios

    free = mujoco_scenarios()["free_space"]
    free_robot = MuJoCoPanda.create(obstacles=())
    free_checker = ContinuousMuJoCoCollisionChecker(free_robot)
    small_end = free.start.copy()
    small_end[:3] += [0.01, -0.01, 0.005]
    safe = free_checker.edge_certificate(free.start, small_end)

    blocked_scenario = mujoco_scenarios()["single_block"]
    blocked_robot = MuJoCoPanda.create(obstacles=blocked_scenario.obstacles)
    blocked_checker = ContinuousMuJoCoCollisionChecker(blocked_robot)
    blocked = blocked_checker.edge_certificate(
        blocked_scenario.start, blocked_scenario.goal
    )

    self_start = np.array(
        [
            2.013896901192687,
            0.8514937392438364,
            2.128626643260985,
            -0.7308358555657621,
            1.5703065944221684,
            1.5832426543430034,
            0.5959999237674181,
        ]
    )
    self_end = np.array(
        [
            2.044760389840342,
            -1.7175851740339965,
            2.558238709554596,
            -2.9380304930478514,
            -2.797519848117953,
            1.1846938393238287,
            -2.8554195536676996,
        ]
    )
    self_collision = free_checker.edge_certificate(self_start, self_end)
    return {
        "passed": bool(
            safe.certified_safe
            and not blocked.certified_safe
            and not self_collision.certified_safe
            and self_collision.collision_pair is not None
        ),
        "safe_edge": safe.metrics(),
        "static_blocked_edge": blocked.metrics(),
        "self_collision_edge": self_collision.metrics(),
        "claim_boundary": (
            "continuous only for linear joint interpolation and compiled MuJoCo "
            "geometry; indeterminate edges fail closed"
        ),
    }
