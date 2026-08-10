"""Optimized CPU implementation of the integrated Panda assurance contract."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike

from armbench.mujoco_sim.broad_phase_continuous_collision import (
    BroadPhaseContinuousMuJoCoCollisionChecker,
)
from armbench.mujoco_sim.continuous_collision import (
    ContinuousCollisionCertificate,
)
from armbench.mujoco_sim.dynamics_braking import DynamicsBrakingResult
from armbench.mujoco_sim.model import MuJoCoPanda
from armbench.mujoco_sim.persistent_dynamics_braking import (
    PersistentDynamicsBrakingValidator,
)
from armbench.vla.integrated_panda_guard import (
    ACTION_DIM,
    PANDA_DOF,
    IntegratedPandaDecision,
    IntegratedPandaGuardConfig,
    IntegratedPandaSupervisor,
    _finite_number,
    _readonly_vector,
)
from armbench.vla.persistent_qp_projection import PersistentQPActionProjector
from armbench.vla.qp_projection import QPProjectionConfig, QPProjectionResult
from armbench.vla.types import ActionChunk


class OptimizedIntegratedPandaSupervisor(IntegratedPandaSupervisor):
    """Preserve the supervisor contract while reusing fixed CPU workspaces."""

    def __init__(
        self,
        robot: MuJoCoPanda,
        checker: BroadPhaseContinuousMuJoCoCollisionChecker,
        config: IntegratedPandaGuardConfig = IntegratedPandaGuardConfig(),
    ) -> None:
        if robot.dof != PANDA_DOF:
            raise ValueError("integrated Panda supervisor requires seven joints")
        if checker.robot is not robot:
            raise ValueError("supervisor and continuous checker must share one Panda")
        if not isinstance(checker, BroadPhaseContinuousMuJoCoCollisionChecker):
            raise TypeError("optimized supervisor requires the broad-phase checker")
        self.robot = robot
        self.checker = checker
        self.config = config
        self.projector = PersistentQPActionProjector(
            robot,
            None,
            QPProjectionConfig(
                control_dt_s=config.control_dt_s,
                joint_velocity_limit_scale=1.0,
                absolute_joint_velocity_limits_rad_s=(
                    config.joint_velocity_limits_rad_s
                ),
                joint_acceleration_limit_rad_s2=(
                    config.joint_acceleration_limit_rad_s2
                ),
                joint_limit_margin_rad=config.joint_limit_margin_rad,
                step_budget_ms=config.qp_step_budget_ms,
            ),
        )
        self.braking_validator = PersistentDynamicsBrakingValidator(
            robot, checker, config.braking
        )

    def optimization_metrics(self) -> dict[str, int | float]:
        """Return cumulative workspace and broad-phase counters."""

        return {
            "persistent_qp_solve_calls": self.projector.solve_calls,
            "persistent_brake_validation_calls": (
                self.braking_validator.validation_calls
            ),
            "whole_stop_edges_checked": (
                self.braking_validator.whole_stop_edges_checked
            ),
            "broad_phase_pair_tests": self.checker.broad_phase_pair_tests,
            "broad_phase_pruned_pairs": self.checker.broad_phase_pruned_pairs,
            "broad_phase_prune_rate": self.checker.broad_phase_prune_rate,
            "safe_configuration_cache_hits": (
                self.checker.safe_configuration_cache_hits
            ),
            "safe_configuration_cache_size": (
                self.checker.safe_configuration_cache_size
            ),
        }

    def _fallback(
        self,
        *,
        started_at: float,
        q: np.ndarray,
        velocity: np.ndarray,
        response_age_ms: float,
        mismatch: float,
        reason: str,
        stage: str,
        failure_index: int | None,
        stage_latencies: dict[str, float],
        qp_result: QPProjectionResult | None,
        edge_certificates: list[ContinuousCollisionCertificate],
        braking_certificates: list[DynamicsBrakingResult],
    ) -> IntegratedPandaDecision:
        fallback_started = perf_counter()
        fallback = self.braking_validator.validate(q, velocity)
        stage_latencies["fallback_brake"] = (
            perf_counter() - fallback_started
        ) * 1000.0
        if fallback.validated:
            stationary = bool(
                np.max(np.abs(velocity))
                <= self.config.stationary_velocity_tolerance_rad_s
            )
            status = "hold" if stationary else "verified_brake"
        else:
            status = "unrecoverable_stop"
        return IntegratedPandaDecision(
            status=status,
            reason=reason,
            failure_stage=stage,
            failure_index=failure_index,
            response_age_ms=response_age_ms,
            state_mismatch_rad=mismatch,
            total_latency_ms=self._elapsed_ms(started_at),
            stage_latencies_ms=stage_latencies,
            executable_actions=np.empty((0, ACTION_DIM), dtype=float),
            predicted_positions=np.asarray([q], dtype=float),
            qp_result=qp_result,
            edge_certificates=tuple(edge_certificates),
            braking_certificates=tuple(braking_certificates),
            fallback_brake=fallback,
        )

    def supervise(
        self,
        q: ArrayLike,
        qvel: ArrayLike,
        chunk: ActionChunk,
        *,
        observed_q: ArrayLike | None = None,
        response_age_ms: float = 0.0,
    ) -> IntegratedPandaDecision:
        """Return one atomic decision using persistent CPU workspaces."""

        started_at = perf_counter()
        position = self.robot.validate_configuration(q).copy()
        velocity = _readonly_vector(qvel, "qvel").copy()
        age = _finite_number(response_age_ms, "response_age_ms")
        if age < 0.0:
            raise ValueError("response_age_ms cannot be negative")
        observation_position = (
            position
            if observed_q is None
            else self.robot.validate_configuration(observed_q)
        )
        mismatch = float(np.max(np.abs(position - observation_position)))
        stage_latencies: dict[str, float] = {}
        edges: list[ContinuousCollisionCertificate] = []
        brakes: list[DynamicsBrakingResult] = []

        if age > self.config.response_deadline_ms:
            return self._fallback(
                started_at=started_at,
                q=position,
                velocity=velocity,
                response_age_ms=age,
                mismatch=mismatch,
                reason="response_deadline_exceeded_before_supervision",
                stage="deadline",
                failure_index=None,
                stage_latencies=stage_latencies,
                qp_result=None,
                edge_certificates=edges,
                braking_certificates=brakes,
            )
        if mismatch > self.config.max_state_mismatch_rad:
            return self._fallback(
                started_at=started_at,
                q=position,
                velocity=velocity,
                response_age_ms=age,
                mismatch=mismatch,
                reason="observation_state_mismatch",
                stage="state_alignment",
                failure_index=None,
                stage_latencies=stage_latencies,
                qp_result=None,
                edge_certificates=edges,
                braking_certificates=brakes,
            )

        qp_started = perf_counter()
        try:
            projection = self.projector.project_chunk(
                position, chunk, previous_velocity=velocity
            )
        except ValueError as error:
            stage_latencies["qp_projection"] = (
                perf_counter() - qp_started
            ) * 1000.0
            return self._fallback(
                started_at=started_at,
                q=position,
                velocity=velocity,
                response_age_ms=age,
                mismatch=mismatch,
                reason=f"qp_input_rejected:{error}",
                stage="qp_projection",
                failure_index=None,
                stage_latencies=stage_latencies,
                qp_result=None,
                edge_certificates=edges,
                braking_certificates=brakes,
            )
        stage_latencies["qp_projection"] = (
            perf_counter() - qp_started
        ) * 1000.0
        if not projection.feasible:
            return self._fallback(
                started_at=started_at,
                q=position,
                velocity=velocity,
                response_age_ms=age,
                mismatch=mismatch,
                reason=f"qp_projection_failed:{projection.failure_reason}",
                stage="qp_projection",
                failure_index=projection.failure_step,
                stage_latencies=stage_latencies,
                qp_result=projection,
                edge_certificates=edges,
                braking_certificates=brakes,
            )
        timing_failure = self._timing_failure(started_at, age)
        if timing_failure is not None:
            stage, reason = timing_failure
            return self._fallback(
                started_at=started_at,
                q=position,
                velocity=velocity,
                response_age_ms=age,
                mismatch=mismatch,
                reason=reason,
                stage=stage,
                failure_index=None,
                stage_latencies=stage_latencies,
                qp_result=projection,
                edge_certificates=edges,
                braking_certificates=brakes,
            )

        collision_started = perf_counter()
        for index, (q_before, q_after) in enumerate(
            zip(
                projection.predicted_positions[:-1],
                projection.predicted_positions[1:],
            )
        ):
            certificate = self.checker.edge_certificate(q_before, q_after)
            edges.append(certificate)
            if not certificate.certified_safe:
                stage_latencies["continuous_collision"] = (
                    perf_counter() - collision_started
                ) * 1000.0
                return self._fallback(
                    started_at=started_at,
                    q=position,
                    velocity=velocity,
                    response_age_ms=age,
                    mismatch=mismatch,
                    reason=f"continuous_edge_rejected:{certificate.reason}",
                    stage="continuous_collision",
                    failure_index=index,
                    stage_latencies=stage_latencies,
                    qp_result=projection,
                    edge_certificates=edges,
                    braking_certificates=brakes,
                )
            timing_failure = self._timing_failure(started_at, age)
            if timing_failure is not None:
                stage_latencies["continuous_collision"] = (
                    perf_counter() - collision_started
                ) * 1000.0
                stage, reason = timing_failure
                return self._fallback(
                    started_at=started_at,
                    q=position,
                    velocity=velocity,
                    response_age_ms=age,
                    mismatch=mismatch,
                    reason=reason,
                    stage=stage,
                    failure_index=index,
                    stage_latencies=stage_latencies,
                    qp_result=projection,
                    edge_certificates=edges,
                    braking_certificates=brakes,
                )
        stage_latencies["continuous_collision"] = (
            perf_counter() - collision_started
        ) * 1000.0

        braking_started = perf_counter()
        for index, (q_after, action) in enumerate(
            zip(
                projection.predicted_positions[1:],
                projection.projected_actions,
            )
        ):
            result = self.braking_validator.validate(
                q_after, action[:PANDA_DOF]
            )
            brakes.append(result)
            if not result.validated:
                stage_latencies["dynamics_braking"] = (
                    perf_counter() - braking_started
                ) * 1000.0
                return self._fallback(
                    started_at=started_at,
                    q=position,
                    velocity=velocity,
                    response_age_ms=age,
                    mismatch=mismatch,
                    reason=f"braking_invariant_failed:{result.failure_reason}",
                    stage="dynamics_braking",
                    failure_index=index,
                    stage_latencies=stage_latencies,
                    qp_result=projection,
                    edge_certificates=edges,
                    braking_certificates=brakes,
                )
            timing_failure = self._timing_failure(started_at, age)
            if timing_failure is not None:
                stage_latencies["dynamics_braking"] = (
                    perf_counter() - braking_started
                ) * 1000.0
                stage, reason = timing_failure
                return self._fallback(
                    started_at=started_at,
                    q=position,
                    velocity=velocity,
                    response_age_ms=age,
                    mismatch=mismatch,
                    reason=reason,
                    stage=stage,
                    failure_index=index,
                    stage_latencies=stage_latencies,
                    qp_result=projection,
                    edge_certificates=edges,
                    braking_certificates=brakes,
                )
        stage_latencies["dynamics_braking"] = (
            perf_counter() - braking_started
        ) * 1000.0
        return IntegratedPandaDecision(
            status="accepted",
            reason="qp_continuous_collision_and_braking_invariant_passed",
            failure_stage=None,
            failure_index=None,
            response_age_ms=age,
            state_mismatch_rad=mismatch,
            total_latency_ms=self._elapsed_ms(started_at),
            stage_latencies_ms=stage_latencies,
            executable_actions=projection.projected_actions,
            predicted_positions=projection.predicted_positions,
            qp_result=projection,
            edge_certificates=tuple(edges),
            braking_certificates=tuple(brakes),
            fallback_brake=None,
        )


__all__ = ["OptimizedIntegratedPandaSupervisor"]
