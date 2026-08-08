"""Convex joint-velocity projection with fail-closed collision validation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray
import osqp
from scipy import sparse

from armbench.mujoco_sim.collision import MuJoCoCollisionChecker
from armbench.mujoco_sim.model import MuJoCoPanda
from armbench.vla.types import ActionChunk


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class QPProjectionConfig:
    control_dt_s: float = 0.05
    joint_velocity_limit_scale: float = 0.5
    absolute_joint_velocity_limits_rad_s: tuple[float, ...] | None = None
    joint_acceleration_limit_rad_s2: float = 15.0
    joint_limit_margin_rad: float = 0.02
    tracking_weight: float = 1.0
    smoothness_weight: float = 0.05
    step_budget_ms: float = 20.0
    eps_abs: float = 1e-6
    eps_rel: float = 1e-6
    max_iterations: int = 4000

    def __post_init__(self) -> None:
        positive = (
            self.control_dt_s,
            self.joint_velocity_limit_scale,
            self.joint_acceleration_limit_rad_s2,
            self.tracking_weight,
            self.step_budget_ms,
            self.eps_abs,
            self.eps_rel,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("QP timing, limits, weights, and tolerances must be positive")
        if self.joint_velocity_limit_scale > 1.0:
            raise ValueError("joint_velocity_limit_scale cannot exceed one")
        if self.absolute_joint_velocity_limits_rad_s is not None:
            raw_limits = np.asarray(self.absolute_joint_velocity_limits_rad_s)
            if raw_limits.dtype.kind not in {"i", "u", "f"}:
                raise ValueError("absolute joint velocity limits must be numeric")
            limits = np.asarray(raw_limits, dtype=float)
            if (
                limits.shape != (7,)
                or not np.all(np.isfinite(limits))
                or np.any(limits <= 0.0)
            ):
                raise ValueError(
                    "absolute joint velocity limits must be seven positive values"
                )
            object.__setattr__(
                self,
                "absolute_joint_velocity_limits_rad_s",
                tuple(float(value) for value in limits),
            )
        if (
            not np.isfinite(self.joint_limit_margin_rad)
            or self.joint_limit_margin_rad < 0.0
            or not np.isfinite(self.smoothness_weight)
            or self.smoothness_weight < 0.0
        ):
            raise ValueError("QP margin and smoothness weight are invalid")
        if type(self.max_iterations) is not int or self.max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")


@dataclass(frozen=True)
class QPLinearConstraint:
    matrix: FloatArray
    lower: FloatArray
    upper: FloatArray
    label: str

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        if (
            matrix.ndim != 2
            or matrix.shape[1] != 7
            or matrix.shape[0] == 0
            or lower.shape != (matrix.shape[0],)
            or upper.shape != (matrix.shape[0],)
            or not np.all(np.isfinite(matrix))
            or np.any(np.isnan(lower))
            or np.any(np.isnan(upper))
            or np.any(lower > upper)
            or not isinstance(self.label, str)
            or not self.label.strip()
        ):
            raise ValueError("QP linear constraint is invalid")
        copied_matrix = matrix.copy()
        copied_lower = lower.copy()
        copied_upper = upper.copy()
        copied_matrix.flags.writeable = False
        copied_lower.flags.writeable = False
        copied_upper.flags.writeable = False
        object.__setattr__(self, "matrix", copied_matrix)
        object.__setattr__(self, "lower", copied_lower)
        object.__setattr__(self, "upper", copied_upper)


@dataclass(frozen=True)
class QPProjectionStep:
    index: int
    status: str
    reason: str
    fallback_used: bool
    solver_status: str
    solver_iterations: int
    solve_latency_ms: float
    budget_exceeded: bool
    collision_checked: bool
    collision_safe: bool
    raw_velocity: FloatArray
    projected_velocity: FloatArray
    q_before: FloatArray
    q_after: FloatArray

    def __post_init__(self) -> None:
        for value in (
            self.raw_velocity,
            self.projected_velocity,
            self.q_before,
            self.q_after,
        ):
            value.flags.writeable = False


@dataclass(frozen=True)
class QPProjectionResult:
    feasible: bool
    failure_step: int | None
    failure_reason: str | None
    projected_actions: FloatArray
    predicted_positions: FloatArray
    steps: tuple[QPProjectionStep, ...]
    total_latency_ms: float

    def __post_init__(self) -> None:
        self.projected_actions.flags.writeable = False
        self.predicted_positions.flags.writeable = False

    @property
    def intervention_steps(self) -> int:
        return sum(
            not np.allclose(
                step.raw_velocity,
                step.projected_velocity,
                atol=1e-8,
                rtol=0.0,
            )
            for step in self.steps
        )

    @property
    def fallback_steps(self) -> int:
        return sum(step.fallback_used for step in self.steps)

    @property
    def solve_p95_ms(self) -> float:
        if not self.steps:
            return 0.0
        return float(np.percentile([step.solve_latency_ms for step in self.steps], 95))

    def metrics(self) -> dict[str, object]:
        return {
            "scope": "component_qp_joint_velocity_projection",
            "solver": "OSQP",
            "feasible": self.feasible,
            "failure_step": self.failure_step,
            "failure_reason": self.failure_reason,
            "horizon": len(self.steps),
            "intervention_steps": self.intervention_steps,
            "fallback_steps": self.fallback_steps,
            "budget_exceeded_steps": sum(
                step.budget_exceeded for step in self.steps
            ),
            "solve_p95_ms": self.solve_p95_ms,
            "solve_max_ms": max(
                (step.solve_latency_ms for step in self.steps), default=0.0
            ),
            "total_latency_ms": self.total_latency_ms,
        }


@dataclass(frozen=True)
class _SolveResult:
    solved: bool
    velocity: FloatArray
    status: str
    iterations: int
    latency_ms: float
    budget_exceeded: bool


class QPActionProjector:
    """Project a complete action chunk through hard kinematic constraints."""

    def __init__(
        self,
        robot: MuJoCoPanda,
        checker: MuJoCoCollisionChecker | None,
        config: QPProjectionConfig = QPProjectionConfig(),
    ) -> None:
        if robot.dof != 7:
            raise ValueError("QPActionProjector requires a seven-DoF robot")
        if checker is not None and checker.robot is not robot:
            raise ValueError("collision checker and projector must share one robot")
        lower = robot.lower_limits + config.joint_limit_margin_rad
        upper = robot.upper_limits - config.joint_limit_margin_rad
        if np.any(lower >= upper):
            raise ValueError("joint limit margin leaves an empty feasible range")
        self.robot = robot
        self.checker = checker
        self.config = config

    @property
    def velocity_limits(self) -> FloatArray:
        scaled = self.robot.velocity_limits * self.config.joint_velocity_limit_scale
        absolute = self.config.absolute_joint_velocity_limits_rad_s
        if absolute is None:
            return scaled
        return np.minimum(scaled, np.asarray(absolute, dtype=float))

    def _box_bounds(
        self,
        q: FloatArray,
        previous_velocity: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
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
        target_velocity: FloatArray,
        previous_velocity: FloatArray,
        lower: FloatArray,
        upper: FloatArray,
        constraints: tuple[QPLinearConstraint, ...],
    ) -> _SolveResult:
        if np.any(lower > upper + 1e-12):
            return _SolveResult(
                False,
                np.zeros(7),
                "inconsistent_box_bounds",
                0,
                0.0,
                False,
            )
        weight = self.config.tracking_weight + self.config.smoothness_weight
        hessian = sparse.eye(7, format="csc") * (2.0 * weight)
        gradient = -2.0 * (
            self.config.tracking_weight * target_velocity
            + self.config.smoothness_weight * previous_velocity
        )
        matrices = [sparse.eye(7, format="csc")]
        lowers = [lower]
        uppers = [upper]
        for constraint in constraints:
            matrices.append(sparse.csc_matrix(constraint.matrix))
            lowers.append(constraint.lower)
            uppers.append(constraint.upper)
        matrix = sparse.vstack(matrices, format="csc")
        lower_bound = np.concatenate(lowers)
        upper_bound = np.concatenate(uppers)
        solver = osqp.OSQP()
        started = perf_counter()
        try:
            solver.setup(
                P=hessian,
                q=gradient,
                A=matrix,
                l=lower_bound,
                u=upper_bound,
                verbose=False,
                eps_abs=self.config.eps_abs,
                eps_rel=self.config.eps_rel,
                max_iter=self.config.max_iterations,
                polishing=False,
                warm_starting=False,
                time_limit=self.config.step_budget_ms / 1000.0,
            )
            solved = solver.solve(raise_error=False)
        except Exception as error:
            latency_ms = (perf_counter() - started) * 1000.0
            return _SolveResult(
                False,
                np.zeros(7),
                f"solver_exception:{type(error).__name__}",
                0,
                latency_ms,
                latency_ms > self.config.step_budget_ms,
            )
        latency_ms = (perf_counter() - started) * 1000.0
        status = str(solved.info.status).lower().replace(" ", "_")
        budget_exceeded = latency_ms > self.config.step_budget_ms
        success = (
            int(solved.info.status_val) in {1, 2}
            and solved.x is not None
            and np.asarray(solved.x).shape == (7,)
            and np.all(np.isfinite(solved.x))
            and not budget_exceeded
        )
        velocity = np.asarray(solved.x, dtype=float) if success else np.zeros(7)
        return _SolveResult(
            success,
            velocity,
            status,
            int(solved.info.iter),
            latency_ms,
            budget_exceeded,
        )

    def _collision_safe(self, q: FloatArray, q_next: FloatArray) -> bool:
        if self.checker is None:
            return True
        return self.checker.edge_is_valid(q, q_next)

    def project_chunk(
        self,
        q_start: ArrayLike,
        chunk: ActionChunk,
        *,
        previous_velocity: ArrayLike | None = None,
        linear_constraints: tuple[QPLinearConstraint, ...] = (),
    ) -> QPProjectionResult:
        started = perf_counter()
        q = self.robot.validate_configuration(q_start).copy()
        if not self.robot.within_limits(q):
            raise ValueError("q_start violates robot joint limits")
        if previous_velocity is None:
            previous = np.zeros(7, dtype=float)
        else:
            previous = np.asarray(previous_velocity, dtype=float)
            if previous.shape != (7,) or not np.all(np.isfinite(previous)):
                raise ValueError("previous_velocity must be a finite seven-vector")
            if np.any(np.abs(previous) > self.velocity_limits + 1e-12):
                raise ValueError("previous_velocity exceeds projector limits")
            previous = previous.copy()
        if not isinstance(linear_constraints, tuple) or any(
            not isinstance(item, QPLinearConstraint) for item in linear_constraints
        ):
            raise TypeError("linear_constraints must be a tuple of QPLinearConstraint")
        output = np.zeros_like(chunk.actions)
        positions = [q.copy()]
        steps: list[QPProjectionStep] = []
        feasible = True
        failure_step: int | None = None
        failure_reason: str | None = None

        for index, raw_action in enumerate(chunk.actions):
            q_before = q.copy()
            raw_velocity = np.asarray(raw_action[:7], dtype=float)
            lower, upper = self._box_bounds(q, previous)
            primary = self._solve(
                raw_velocity,
                previous,
                lower,
                upper,
                linear_constraints,
            )
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
                    np.zeros(7),
                    previous,
                    lower,
                    upper,
                    linear_constraints,
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
                        solve_latency_ms=primary.latency_ms + (
                            selected.latency_ms if fallback_used else 0.0
                        ),
                        budget_exceeded=(
                            primary.budget_exceeded or selected.budget_exceeded
                        ),
                        collision_checked=collision_checked,
                        collision_safe=collision_safe,
                        raw_velocity=raw_velocity.copy(),
                        projected_velocity=np.zeros(7),
                        q_before=q_before,
                        q_after=q.copy(),
                    )
                )
                break
            selected_gripper = float(np.clip(raw_action[7], 0.0, 1.0))
            output[index, :7] = selected.velocity
            output[index, 7] = selected_gripper
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
                    solve_latency_ms=primary.latency_ms + (
                        selected.latency_ms if fallback_used else 0.0
                    ),
                    budget_exceeded=(
                        primary.budget_exceeded or selected.budget_exceeded
                    ),
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


def run_qp_projection_smoke() -> dict[str, object]:
    """Run a deterministic CPU projection without claiming task efficacy."""

    from armbench.mujoco_sim.scenarios import mujoco_scenarios

    scenario = mujoco_scenarios()["free_space"]
    robot = MuJoCoPanda.create(obstacles=scenario.obstacles)
    checker = MuJoCoCollisionChecker(robot, resolution=0.02)
    actions = np.zeros((64, 8), dtype=float)
    rng = np.random.default_rng(20260808)
    actions[:, :7] = rng.normal(0.0, 2.0, size=(64, 7))
    actions[:, 7] = 0.7
    chunk = ActionChunk(
        actions=actions,
        source="deterministic_nonlearned_qp_smoke",
        observation_sequence_id=0,
        inference_latency_ms=0.0,
        received_at_s=100.0,
    )
    projector = QPActionProjector(robot, checker)
    result = projector.project_chunk(scenario.start, chunk)
    return {
        "passed": result.feasible,
        **result.metrics(),
        "all_positions_within_limits": all(
            robot.within_limits(q) for q in result.predicted_positions
        ),
    }
