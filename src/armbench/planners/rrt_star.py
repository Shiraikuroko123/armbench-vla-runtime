"""Bounded RRT* baseline that reports time to the first feasible solution."""

from __future__ import annotations

from math import log
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike

from armbench.collision import CollisionChecker
from armbench.planners.base import TreeNode, nearest_index, steer, trace_to_root
from armbench.result import PlanResult, PlanStatus


class RRTStar:
    """RRT* rewiring with a first-solution stopping rule.

    This keeps the latency comparison with RRT-Connect interpretable. It is not
    an asymptotic-quality experiment because it does not continue optimizing
    after finding a feasible goal connection.
    """

    name = "rrt_star"

    def __init__(
        self,
        checker: CollisionChecker,
        rng: np.random.Generator,
        *,
        step_size: float = 0.35,
        max_iterations: int = 4_000,
        timeout_s: float = 3.0,
        goal_sample_rate: float = 0.1,
        goal_connection_radius: float = 0.6,
        rewire_radius: float = 2.5,
    ) -> None:
        if step_size <= 0.0 or goal_connection_radius <= 0.0:
            raise ValueError("planner distances must be positive")
        if max_iterations <= 0 or timeout_s < 0.0:
            raise ValueError("invalid iteration or timeout bound")
        if not 0.0 <= goal_sample_rate <= 1.0:
            raise ValueError("goal_sample_rate must be in [0, 1]")
        self.checker = checker
        self.rng = rng
        self.step_size = float(step_size)
        self.max_iterations = int(max_iterations)
        self.timeout_s = float(timeout_s)
        self.goal_sample_rate = float(goal_sample_rate)
        self.goal_connection_radius = float(goal_connection_radius)
        self.rewire_radius = float(rewire_radius)

    def sample(self, goal: np.ndarray) -> np.ndarray:
        if self.rng.random() < self.goal_sample_rate:
            return goal.copy()
        return self.checker.robot.sample(self.rng)

    def _near_indices(self, nodes: list[TreeNode], q: np.ndarray) -> list[int]:
        count = len(nodes) + 1
        dimension = self.checker.robot.dof
        radius = self.rewire_radius * (log(max(count, 2)) / count) ** (1.0 / dimension)
        radius = max(self.step_size * 1.25, min(radius, self.rewire_radius))
        return [
            index
            for index, node in enumerate(nodes)
            if float(np.linalg.norm(node.q - q)) <= radius
        ]

    @staticmethod
    def _propagate_costs(nodes: list[TreeNode], parent_index: int) -> None:
        queue = [parent_index]
        while queue:
            parent = queue.pop()
            for index, node in enumerate(nodes):
                if node.parent == parent:
                    node.cost = nodes[parent].cost + float(
                        np.linalg.norm(node.q - nodes[parent].q)
                    )
                    queue.append(index)

    def _result(
        self,
        status: PlanStatus,
        started_at: float,
        iterations: int,
        nodes: int,
        *,
        path: list[np.ndarray] | None = None,
        detail: str = "",
    ) -> PlanResult:
        return PlanResult(
            planner=self.name,
            status=status,
            elapsed_s=perf_counter() - started_at,
            iterations=iterations,
            nodes=nodes,
            collision_queries=self.checker.stats.configuration_queries,
            edge_queries=self.checker.stats.edge_queries,
            path=[] if path is None else path,
            detail=detail,
        )

    def plan(self, start: ArrayLike, goal: ArrayLike) -> PlanResult:
        started_at = perf_counter()
        self.checker.reset_stats()
        start_q = self.checker.robot.validate_configuration(start).copy()
        goal_q = self.checker.robot.validate_configuration(goal).copy()
        start_failure = self.checker.configuration_failure(start_q)
        if start_failure is not None:
            return self._result(
                PlanStatus.START_IN_COLLISION,
                started_at,
                0,
                0,
                detail=start_failure,
            )
        goal_failure = self.checker.configuration_failure(goal_q)
        if goal_failure is not None:
            return self._result(
                PlanStatus.GOAL_IN_COLLISION,
                started_at,
                0,
                0,
                detail=goal_failure,
            )
        if self.timeout_s == 0.0:
            return self._result(PlanStatus.TIMEOUT, started_at, 0, 1)
        deadline = started_at + self.timeout_s
        if self.checker.edge_is_valid(start_q, goal_q):
            return self._result(
                PlanStatus.SUCCESS,
                started_at,
                0,
                2,
                path=[start_q, goal_q],
                detail="direct_connection",
            )

        nodes = [TreeNode(start_q, None, 0.0)]
        for iteration in range(1, self.max_iterations + 1):
            if perf_counter() >= deadline:
                return self._result(
                    PlanStatus.TIMEOUT, started_at, iteration - 1, len(nodes)
                )
            target = self.sample(goal_q)
            nearest = nearest_index(nodes, target)
            candidate = steer(nodes[nearest].q, target, self.step_size)
            if not self.checker.edge_is_valid(nodes[nearest].q, candidate):
                continue

            near_indices = self._near_indices(nodes, candidate)
            if nearest not in near_indices:
                near_indices.append(nearest)
            parent = nearest
            best_cost = nodes[nearest].cost + float(
                np.linalg.norm(candidate - nodes[nearest].q)
            )
            for near in near_indices:
                candidate_cost = nodes[near].cost + float(
                    np.linalg.norm(candidate - nodes[near].q)
                )
                if candidate_cost + 1e-12 < best_cost and self.checker.edge_is_valid(
                    nodes[near].q, candidate
                ):
                    parent = near
                    best_cost = candidate_cost

            nodes.append(TreeNode(candidate, parent, best_cost))
            new_index = len(nodes) - 1
            for near in near_indices:
                if near == parent:
                    continue
                rewired_cost = best_cost + float(np.linalg.norm(nodes[near].q - candidate))
                if rewired_cost + 1e-12 < nodes[near].cost and self.checker.edge_is_valid(
                    candidate, nodes[near].q
                ):
                    nodes[near].parent = new_index
                    nodes[near].cost = rewired_cost
                    self._propagate_costs(nodes, near)

            if (
                float(np.linalg.norm(candidate - goal_q))
                <= self.goal_connection_radius
                and self.checker.edge_is_valid(candidate, goal_q)
            ):
                path = trace_to_root(nodes, new_index) + [goal_q]
                if not self.checker.path_is_valid(path):
                    raise RuntimeError("RRT* reconstructed an invalid path")
                return self._result(
                    PlanStatus.SUCCESS,
                    started_at,
                    iteration,
                    len(nodes) + 1,
                    path=path,
                )

        return self._result(
            PlanStatus.ITERATION_LIMIT,
            started_at,
            self.max_iterations,
            len(nodes),
        )

