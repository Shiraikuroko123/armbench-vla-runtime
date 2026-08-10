"""Persistent OSQP projection for repeated Panda action chunks.

The reference projector creates a solver for every action step. This variant
prebuilds the fixed seven-dimensional problem once and updates only the linear
objective and box bounds. It preserves the same result contract and resets the
primal and dual warm start on every solve so prior chunks cannot silently alter
the initial solver state.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from time import perf_counter

import numpy as np
import osqp
from scipy import sparse

from armbench.mujoco_sim.collision import MuJoCoCollisionChecker
from armbench.mujoco_sim.model import MuJoCoPanda
from armbench.vla.qp_projection import (
    QPLinearConstraint,
    QPProjectionConfig,
    QPProjectionResult,
    QPProjectionStep,
)
from armbench.vla.types import ActionChunk


PANDA_DOF = 7


@dataclass(frozen=True)
class _SolveResult:
    solved: bool
    velocity: np.ndarray
    status: str
    iterations: int
    latency_ms: float
    budget_exceeded: bool


class PersistentQPActionProjector:
    """Project chunks with one prebuilt, lock-protected OSQP workspace."""

    def __init__(
        self,
        robot: MuJoCoPanda,
        checker: MuJoCoCollisionChecker | None,
        config: QPProjectionConfig = QPProjectionConfig(),
    ) -> None:
        if robot.dof != PANDA_DOF:
            raise ValueError("persistent projector requires a seven-DoF robot")
        if checker is not None and checker.robot is not robot:
            raise ValueError("collision checker and projector must share one robot")
        lower = robot.lower_limits + config.joint_limit_margin_rad
        upper = robot.upper_limits - config.joint_limit_margin_rad
        if np.any(lower >= upper):
            raise ValueError("joint limit margin leaves an empty feasible range")
        self.robot = robot
        self.checker = checker
        self.config = config
        self._lock = threading.Lock()
        self._solve_calls = 0

        weight = config.tracking_weight + config.smoothness_weight
        hessian = sparse.eye(PANDA_DOF, format="csc") * (2.0 * weight)
        matrix = sparse.eye(PANDA_DOF, format="csc")
        self._solver = osqp.OSQP()
        self._solver.setup(
            P=hessian,
            q=np.zeros(PANDA_DOF),
            A=matrix,
            l=np.full(PANDA_DOF, -np.inf),
            u=np.full(PANDA_DOF, np.inf),
            verbose=False,
            eps_abs=config.eps_abs,
            eps_rel=config.eps_rel,
            max_iter=config.max_iterations,
            polishing=False,
            warm_starting=True,
            adaptive_rho=False,
            time_limit=config.step_budget_ms / 1000.0,
        )

    @property
    def velocity_limits(self) -> np.ndarray:
        scaled = self.robot.velocity_limits * self.config.joint_velocity_limit_scale
        absolute = self.config.absolute_joint_velocity_limits_rad_s
        if absolute is None:
            return scaled
        return np.minimum(scaled, np.asarray(absolute, dtype=float))

    @property
    def solve_calls(self) -> int:
        return self._solve_calls

    def _box_bounds(
        self, q: np.ndarray, previous_velocity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        dt = self.config.control_dt_s
        velocity_delta = self.config.joint_acceleration_limit_rad_s2 * dt
        lower = np.maximum.reduce(
            (
                -self.velocity_limits,
                previous_velocity - velocity_delta,
                (
                    self.robot.lower_limits
                    + self.config.joint_limit_margin_rad
                    - q
                )
                / dt,
            )
        )
        upper = np.minimum.reduce(
            (
                self.velocity_limits,
                previous_velocity + velocity_delta,
                (
                    self.robot.upper_limits
                    - self.config.joint_limit_margin_rad
                    - q
                )
                / dt,
            )
        )
        return lower, upper

    def _solve(
        self,
        target_velocity: np.ndarray,
        previous_velocity: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> _SolveResult:
        if np.any(lower > upper + 1e-12):
            return _SolveResult(
                False,
                np.zeros(PANDA_DOF),
                "inconsistent_box_bounds",
                0,
                0.0,
                False,
            )
        gradient = -2.0 * (
            self.config.tracking_weight * target_velocity
            + self.config.smoothness_weight * previous_velocity
        )
        started = perf_counter()
        try:
            self._solver.update(q=gradient, l=lower, u=upper)
            self._solver.warm_start(
                x=np.clip(previous_velocity, lower, upper),
                y=np.zeros(PANDA_DOF),
            )
            result = self._solver.solve(raise_error=False)
        except Exception as error:
            latency_ms = (perf_counter() - started) * 1000.0
            return _SolveResult(
                False,
                np.zeros(PANDA_DOF),
                f"solver_exception:{type(error).__name__}",
                0,
                latency_ms,
                latency_ms > self.config.step_budget_ms,
            )
        latency_ms = (perf_counter() - started) * 1000.0
        self._solve_calls += 1
        status = str(result.info.status).lower().replace(" ", "_")
        budget_exceeded = latency_ms > self.config.step_budget_ms
        success = (
            int(result.info.status_val) in {1, 2}
            and result.x is not None
            and np.asarray(result.x).shape == (PANDA_DOF,)
            and np.all(np.isfinite(result.x))
            and not budget_exceeded
        )
        velocity = (
            np.clip(np.asarray(result.x, dtype=float), lower, upper)
            if success
            else np.zeros(PANDA_DOF)
        )
        return _SolveResult(
            success,
            velocity,
            status,
            int(result.info.iter),
            latency_ms,
            budget_exceeded,
        )

    def _collision_safe(self, q: np.ndarray, q_next: np.ndarray) -> bool:
        if self.checker is None:
            return True
        return self.checker.edge_is_valid(q, q_next)

    def project_chunk(
        self,
        q_start: np.ndarray,
        chunk: ActionChunk,
        *,
        previous_velocity: np.ndarray | None = None,
        linear_constraints: tuple[QPLinearConstraint, ...] = (),
    ) -> QPProjectionResult:
        """Project a complete Hx8 chunk without rebuilding the solver."""

        if not isinstance(linear_constraints, tuple):
            raise TypeError("linear_constraints must be a tuple")
        if linear_constraints:
            raise ValueError(
                "persistent projector supports the registered box constraints only"
            )
        with self._lock:
            return self._project_locked(q_start, chunk, previous_velocity)

    def _project_locked(
        self,
        q_start: np.ndarray,
        chunk: ActionChunk,
        previous_velocity: np.ndarray | None,
    ) -> QPProjectionResult:
        started = perf_counter()
        q = self.robot.validate_configuration(q_start).copy()
        if not self.robot.within_limits(q):
            raise ValueError("q_start violates robot joint limits")
        if previous_velocity is None:
            previous = np.zeros(PANDA_DOF, dtype=float)
        else:
            previous = np.asarray(previous_velocity, dtype=float)
            if (
                previous.shape != (PANDA_DOF,)
                or not np.all(np.isfinite(previous))
                or np.any(np.abs(previous) > self.velocity_limits + 1e-12)
            ):
                raise ValueError("previous_velocity is invalid")
            previous = previous.copy()

        output = np.zeros_like(chunk.actions)
        positions = [q.copy()]
        steps: list[QPProjectionStep] = []
        feasible = True
        failure_step: int | None = None
        failure_reason: str | None = None
        for index, raw_action in enumerate(chunk.actions):
            q_before = q.copy()
            raw_velocity = np.asarray(raw_action[:PANDA_DOF], dtype=float)
            lower, upper = self._box_bounds(q, previous)
            primary = self._solve(raw_velocity, previous, lower, upper)
            selected = primary
            fallback_used = False
            reason = "projected"
            collision_checked = primary.solved
            collision_safe = False
            if primary.solved:
                q_candidate = q + self.config.control_dt_s * primary.velocity
                collision_safe = self._collision_safe(q, q_candidate)
            else:
                q_candidate = q.copy()
            if not primary.solved or not collision_safe:
                fallback_used = True
                reason = (
                    "primary_solver_failed"
                    if not primary.solved
                    else "primary_collision_rejected"
                )
                selected = self._solve(
                    np.zeros(PANDA_DOF), previous, lower, upper
                )
                collision_checked = selected.solved
                if selected.solved:
                    q_candidate = q + self.config.control_dt_s * selected.velocity
                    collision_safe = self._collision_safe(q, q_candidate)
                else:
                    q_candidate = q.copy()
                    collision_safe = False
            if not selected.solved or not collision_safe:
                feasible = False
                failure_step = index
                failure_reason = (
                    "fallback_collision_rejected"
                    if selected.solved
                    else f"fallback_solver_failed:{selected.status}"
                )
                steps.append(
                    QPProjectionStep(
                        index=index,
                        status="rejected",
                        reason=failure_reason,
                        fallback_used=fallback_used,
                        solver_status=selected.status,
                        solver_iterations=selected.iterations,
                        solve_latency_ms=primary.latency_ms
                        + (selected.latency_ms if fallback_used else 0.0),
                        budget_exceeded=primary.budget_exceeded
                        or selected.budget_exceeded,
                        collision_checked=collision_checked,
                        collision_safe=collision_safe,
                        raw_velocity=raw_velocity.copy(),
                        projected_velocity=np.zeros(PANDA_DOF),
                        q_before=q_before,
                        q_after=q.copy(),
                    )
                )
                break
            output[index, :PANDA_DOF] = selected.velocity
            output[index, PANDA_DOF] = float(np.clip(raw_action[PANDA_DOF], 0.0, 1.0))
            q = q_candidate
            previous = selected.velocity.copy()
            positions.append(q.copy())
            steps.append(
                QPProjectionStep(
                    index=index,
                    status="accepted",
                    reason=reason,
                    fallback_used=fallback_used,
                    solver_status=selected.status,
                    solver_iterations=selected.iterations,
                    solve_latency_ms=primary.latency_ms
                    + (selected.latency_ms if fallback_used else 0.0),
                    budget_exceeded=primary.budget_exceeded
                    or selected.budget_exceeded,
                    collision_checked=collision_checked,
                    collision_safe=collision_safe,
                    raw_velocity=raw_velocity.copy(),
                    projected_velocity=selected.velocity.copy(),
                    q_before=q_before,
                    q_after=q.copy(),
                )
            )
        return QPProjectionResult(
            feasible=feasible,
            failure_step=failure_step,
            failure_reason=failure_reason,
            projected_actions=output,
            predicted_positions=np.asarray(positions),
            steps=tuple(steps),
            total_latency_ms=(perf_counter() - started) * 1000.0,
        )


__all__ = ["PersistentQPActionProjector"]
