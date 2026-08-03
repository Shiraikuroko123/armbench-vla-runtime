"""Deterministic, bounded bidirectional RRT-Connect."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike

from armbench.collision import CollisionChecker
from armbench.planners.base import (
    ExtendStatus,
    TreeNode,
    nearest_index,
    steer,
    trace_to_root,
)
from armbench.result import PlanResult, PlanStatus


class RRTConnect:
    name = "rrt_connect"

    def __init__(
        self,
        checker: CollisionChecker,
        rng: np.random.Generator,
        *,
        step_size: float = 0.35,
        max_iterations: int = 2_000,
        timeout_s: float = 3.0,
        goal_sample_rate: float = 0.05,
    ) -> None:
        if step_size <= 0.0:
            raise ValueError("step_size must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if timeout_s < 0.0:
            raise ValueError("timeout_s cannot be negative")
        if not 0.0 <= goal_sample_rate <= 1.0:
            raise ValueError("goal_sample_rate must be in [0, 1]")
        self.checker = checker
        self.rng = rng
        self.step_size = float(step_size)
        self.max_iterations = int(max_iterations)
        self.timeout_s = float(timeout_s)
        self.goal_sample_rate = float(goal_sample_rate)

    def sample(self, opposite_root: np.ndarray) -> np.ndarray:
        if self.rng.random() < self.goal_sample_rate:
            return opposite_root.copy()
        return self.checker.robot.sample(self.rng)

    @staticmethod
    def nearest(nodes: list[TreeNode], target: np.ndarray) -> int:
        return nearest_index(nodes, target)

    def steer(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        return steer(q_from, q_to, self.step_size)

    def edge_is_valid(self, q_from: np.ndarray, q_to: np.ndarray) -> bool:
        return self.checker.edge_is_valid(q_from, q_to)

    def extend(
        self, nodes: list[TreeNode], target: np.ndarray
    ) -> tuple[ExtendStatus, int]:
        near_index = self.nearest(nodes, target)
        near = nodes[near_index].q
        if np.allclose(near, target, atol=1e-12, rtol=0.0):
            return ExtendStatus.REACHED, near_index
        candidate = self.steer(near, target)
        if not self.edge_is_valid(near, candidate):
            return ExtendStatus.TRAPPED, near_index
        nodes.append(
            TreeNode(
                q=candidate,
                parent=near_index,
                cost=nodes[near_index].cost + float(np.linalg.norm(candidate - near)),
            )
        )
        new_index = len(nodes) - 1
        if np.allclose(candidate, target, atol=1e-12, rtol=0.0):
            return ExtendStatus.REACHED, new_index
        return ExtendStatus.ADVANCED, new_index

    def connect(
        self, nodes: list[TreeNode], target: np.ndarray, deadline: float
    ) -> tuple[ExtendStatus, int, bool]:
        status = ExtendStatus.ADVANCED
        index = 0
        while status is ExtendStatus.ADVANCED:
            if perf_counter() >= deadline:
                return status, index, True
            status, index = self.extend(nodes, target)
        return status, index, False

    @staticmethod
    def trace_path(
        tree_a: list[TreeNode],
        index_a: int,
        tree_b: list[TreeNode],
        index_b: int,
        *,
        tree_a_root_is_start: bool,
    ) -> list[np.ndarray]:
        if tree_a_root_is_start:
            start_branch = trace_to_root(tree_a, index_a)
            goal_branch = trace_to_root(tree_b, index_b)
        else:
            start_branch = trace_to_root(tree_b, index_b)
            goal_branch = trace_to_root(tree_a, index_a)
        connection_to_goal = list(reversed(goal_branch))
        return start_branch + connection_to_goal[1:]

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
            return self._result(PlanStatus.TIMEOUT, started_at, 0, 2)

        deadline = started_at + self.timeout_s
        if self.edge_is_valid(start_q, goal_q):
            return self._result(
                PlanStatus.SUCCESS,
                started_at,
                0,
                2,
                path=[start_q, goal_q],
                detail="direct_connection",
            )

        tree_a = [TreeNode(start_q, None)]
        tree_b = [TreeNode(goal_q, None)]
        tree_a_root_is_start = True

        for iteration in range(1, self.max_iterations + 1):
            if perf_counter() >= deadline:
                return self._result(
                    PlanStatus.TIMEOUT,
                    started_at,
                    iteration - 1,
                    len(tree_a) + len(tree_b),
                )
            target = self.sample(tree_b[0].q)
            status_a, index_a = self.extend(tree_a, target)
            if status_a is not ExtendStatus.TRAPPED:
                status_b, index_b, timed_out = self.connect(
                    tree_b, tree_a[index_a].q, deadline
                )
                if timed_out:
                    return self._result(
                        PlanStatus.TIMEOUT,
                        started_at,
                        iteration,
                        len(tree_a) + len(tree_b),
                    )
                if status_b is ExtendStatus.REACHED:
                    path = self.trace_path(
                        tree_a,
                        index_a,
                        tree_b,
                        index_b,
                        tree_a_root_is_start=tree_a_root_is_start,
                    )
                    if not np.allclose(path[0], start_q) or not np.allclose(
                        path[-1], goal_q
                    ):
                        raise RuntimeError("RRT-Connect reconstructed a reversed path")
                    if not self.checker.path_is_valid(path):
                        raise RuntimeError("RRT-Connect reconstructed an invalid path")
                    return self._result(
                        PlanStatus.SUCCESS,
                        started_at,
                        iteration,
                        len(tree_a) + len(tree_b),
                        path=path,
                    )
            tree_a, tree_b = tree_b, tree_a
            tree_a_root_is_start = not tree_a_root_is_start

        return self._result(
            PlanStatus.ITERATION_LIMIT,
            started_at,
            self.max_iterations,
            len(tree_a) + len(tree_b),
        )

