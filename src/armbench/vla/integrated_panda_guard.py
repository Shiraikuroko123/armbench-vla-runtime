"""Task-level Panda action assurance composed from existing CPU components.

The supervisor projects one joint-velocity action chunk, certifies every
projected edge against static and registered self-collision geometry, and
checks that a dynamics-feasible stop remains available after every action.
It is a synchronous, best-effort reference implementation. It does not claim
hard-real-time scheduling or physical-robot safety certification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from armbench.mujoco_sim.continuous_collision import (
    ContinuousCollisionCertificate,
    ContinuousMuJoCoCollisionChecker,
)
from armbench.mujoco_sim.dynamics_braking import (
    DynamicsBrakingConfig,
    DynamicsBrakingResult,
    generate_dynamics_validated_brake,
)
from armbench.mujoco_sim.model import MuJoCoPanda
from armbench.vla.qp_projection import (
    QPActionProjector,
    QPProjectionConfig,
    QPProjectionResult,
)
from armbench.vla.types import ActionChunk


FloatArray = NDArray[np.float64]
PANDA_DOF = 7
ACTION_DIM = 8
DECISION_STATUSES = (
    "accepted",
    "verified_brake",
    "hold",
    "unrecoverable_stop",
)


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _readonly_array(value: ArrayLike, shape_tail: tuple[int, ...]) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail:
        raise ValueError("integrated guard array has an invalid shape")
    if not np.all(np.isfinite(array)):
        raise ValueError("integrated guard array must be finite")
    result = array.copy()
    result.flags.writeable = False
    return result


def _readonly_vector(value: ArrayLike, label: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{label} must be a numeric vector")
    vector = np.asarray(raw, dtype=float)
    if vector.shape != (PANDA_DOF,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must be a finite seven-vector")
    result = vector.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class IntegratedPandaGuardConfig:
    """Timing and constraint contract for one supervised action chunk."""

    control_dt_s: float = 0.05
    response_deadline_ms: float = 1000.0
    supervision_budget_ms: float = 1000.0
    max_state_mismatch_rad: float = 0.05
    joint_velocity_limits_rad_s: tuple[float, ...] = (1.0,) * PANDA_DOF
    joint_acceleration_limit_rad_s2: float = 5.0
    joint_limit_margin_rad: float = 0.02
    qp_step_budget_ms: float = 100.0
    stationary_velocity_tolerance_rad_s: float = 1e-9
    braking: DynamicsBrakingConfig = field(
        default_factory=lambda: DynamicsBrakingConfig(
            sample_dt_s=0.01,
            joint_acceleration_limits_rad_s2=(5.0,) * PANDA_DOF,
            max_stop_time_s=2.0,
            actuator_force_limit_scale=0.8,
            check_inter_sample_edges=True,
        )
    )

    def __post_init__(self) -> None:
        positive_labels = (
            (self.control_dt_s, "control_dt_s"),
            (self.response_deadline_ms, "response_deadline_ms"),
            (self.supervision_budget_ms, "supervision_budget_ms"),
            (
                self.joint_acceleration_limit_rad_s2,
                "joint_acceleration_limit_rad_s2",
            ),
            (self.qp_step_budget_ms, "qp_step_budget_ms"),
        )
        for value, label in positive_labels:
            if _finite_number(value, label) <= 0.0:
                raise ValueError(f"{label} must be positive")
        for value, label in (
            (self.max_state_mismatch_rad, "max_state_mismatch_rad"),
            (self.joint_limit_margin_rad, "joint_limit_margin_rad"),
            (
                self.stationary_velocity_tolerance_rad_s,
                "stationary_velocity_tolerance_rad_s",
            ),
        ):
            if _finite_number(value, label) < 0.0:
                raise ValueError(f"{label} must be nonnegative")
        raw_limits = np.asarray(self.joint_velocity_limits_rad_s)
        if raw_limits.dtype.kind not in {"i", "u", "f"}:
            raise ValueError("joint velocity limits must be numeric")
        limits = np.asarray(raw_limits, dtype=float)
        if (
            limits.shape != (PANDA_DOF,)
            or not np.all(np.isfinite(limits))
            or np.any(limits <= 0.0)
        ):
            raise ValueError("joint velocity limits must be seven positive values")
        object.__setattr__(
            self,
            "joint_velocity_limits_rad_s",
            tuple(float(value) for value in limits),
        )


@dataclass(frozen=True)
class IntegratedPandaDecision:
    """Atomic supervisor output: a complete plan, complete stop, or no motion."""

    status: str
    reason: str
    failure_stage: str | None
    failure_index: int | None
    response_age_ms: float
    state_mismatch_rad: float
    total_latency_ms: float
    stage_latencies_ms: Mapping[str, float]
    executable_actions: FloatArray
    predicted_positions: FloatArray
    qp_result: QPProjectionResult | None
    edge_certificates: tuple[ContinuousCollisionCertificate, ...]
    braking_certificates: tuple[DynamicsBrakingResult, ...]
    fallback_brake: DynamicsBrakingResult | None

    def __post_init__(self) -> None:
        if self.status not in DECISION_STATUSES:
            raise ValueError("integrated guard decision status is invalid")
        if not self.reason:
            raise ValueError("integrated guard decision requires a reason")
        if self.failure_index is not None and self.failure_index < 0:
            raise ValueError("failure index cannot be negative")
        for value in (
            self.response_age_ms,
            self.state_mismatch_rad,
            self.total_latency_ms,
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("integrated guard metrics must be nonnegative")
        latencies = {str(key): float(value) for key, value in self.stage_latencies_ms.items()}
        if any(not np.isfinite(value) or value < 0.0 for value in latencies.values()):
            raise ValueError("stage latencies must be finite and nonnegative")
        actions = _readonly_array(self.executable_actions, (ACTION_DIM,))
        positions = _readonly_array(self.predicted_positions, (PANDA_DOF,))
        if self.status == "accepted":
            if self.qp_result is None or len(actions) == 0:
                raise ValueError("accepted decision requires a complete QP plan")
            if len(positions) != len(actions) + 1:
                raise ValueError("accepted actions and predicted positions disagree")
            if len(self.edge_certificates) != len(actions):
                raise ValueError("accepted plan requires one edge certificate per action")
            if len(self.braking_certificates) != len(actions):
                raise ValueError("accepted plan requires one braking certificate per action")
            if not all(item.certified_safe for item in self.edge_certificates):
                raise ValueError("accepted plan contains an unsafe edge")
            if not all(item.validated for item in self.braking_certificates):
                raise ValueError("accepted plan lacks a braking invariant")
        elif len(actions) != 0:
            raise ValueError("rejected policy chunks cannot expose partial actions")
        if self.status in {"verified_brake", "hold"}:
            if self.fallback_brake is None or not self.fallback_brake.validated:
                raise ValueError("fallback status requires a fully validated brake")
        if self.status == "unrecoverable_stop" and (
            self.fallback_brake is None or self.fallback_brake.validated
        ):
            raise ValueError("unrecoverable stop requires a failed brake certificate")
        object.__setattr__(self, "stage_latencies_ms", latencies)
        object.__setattr__(self, "executable_actions", actions)
        object.__setattr__(self, "predicted_positions", positions)

    @property
    def policy_actions_executable(self) -> bool:
        return self.status == "accepted"

    @property
    def fallback_covered(self) -> bool:
        if self.status == "accepted":
            return bool(self.braking_certificates) and all(
                result.validated for result in self.braking_certificates
            )
        return self.fallback_brake is not None and self.fallback_brake.validated

    @property
    def intervention_steps(self) -> int:
        return self.qp_result.intervention_steps if self.qp_result else 0

    @property
    def conservative_rejection(self) -> bool:
        return self.failure_stage == "continuous_collision" and any(
            certificate.status == "indeterminate"
            for certificate in self.edge_certificates
        )

    def metrics(self) -> dict[str, object]:
        minimum_distance = [
            item.minimum_sampled_distance_m
            for item in self.edge_certificates
            if item.minimum_sampled_distance_m is not None
        ]
        torque_ratios = [
            item.max_torque_ratio
            for item in self.braking_certificates
            if item.max_torque_ratio is not None
        ]
        if self.fallback_brake is not None and self.fallback_brake.max_torque_ratio is not None:
            torque_ratios.append(self.fallback_brake.max_torque_ratio)
        return {
            "schema_version": "armbench.integrated_panda_decision.v1",
            "scope": "synchronous_cpu_reference_not_hard_realtime_or_hardware_certification",
            "status": self.status,
            "reason": self.reason,
            "failure_stage": self.failure_stage,
            "failure_index": self.failure_index,
            "policy_actions_executable": self.policy_actions_executable,
            "policy_action_count": len(self.executable_actions),
            "response_age_ms": self.response_age_ms,
            "state_mismatch_rad": self.state_mismatch_rad,
            "total_latency_ms": self.total_latency_ms,
            "stage_latencies_ms": dict(self.stage_latencies_ms),
            "qp_feasible": self.qp_result.feasible if self.qp_result else None,
            "qp_intervention_steps": self.intervention_steps,
            "qp_fallback_steps": self.qp_result.fallback_steps if self.qp_result else 0,
            "continuous_edges_checked": len(self.edge_certificates),
            "continuous_all_safe": bool(self.edge_certificates)
            and all(item.certified_safe for item in self.edge_certificates),
            "continuous_minimum_sampled_distance_m": (
                min(minimum_distance) if minimum_distance else None
            ),
            "continuous_pair_evaluations": sum(
                item.pair_evaluations for item in self.edge_certificates
            ),
            "conservative_rejection": self.conservative_rejection,
            "braking_boundaries_checked": len(self.braking_certificates),
            "braking_invariant_complete": self.fallback_covered,
            "fallback_validated": (
                self.fallback_brake.validated
                if self.fallback_brake is not None
                else None
            ),
            "fallback_failure_reason": (
                self.fallback_brake.failure_reason
                if self.fallback_brake is not None
                else None
            ),
            "max_inverse_dynamics_torque_ratio": (
                max(torque_ratios) if torque_ratios else None
            ),
        }


class IntegratedPandaSupervisor:
    """Compose QP, continuous geometry, and inverse-dynamics stop checks."""

    def __init__(
        self,
        robot: MuJoCoPanda,
        checker: ContinuousMuJoCoCollisionChecker,
        config: IntegratedPandaGuardConfig = IntegratedPandaGuardConfig(),
    ) -> None:
        if robot.dof != PANDA_DOF:
            raise ValueError("integrated Panda supervisor requires seven joints")
        if checker.robot is not robot:
            raise ValueError("supervisor and continuous checker must share one Panda")
        self.robot = robot
        self.checker = checker
        self.config = config
        self.projector = QPActionProjector(
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

    def _elapsed_ms(self, started_at: float) -> float:
        return (perf_counter() - started_at) * 1000.0

    def _timing_failure(
        self, started_at: float, response_age_ms: float
    ) -> tuple[str, str] | None:
        elapsed = self._elapsed_ms(started_at)
        if response_age_ms + elapsed > self.config.response_deadline_ms:
            return "deadline", "response_deadline_exceeded_during_supervision"
        if elapsed > self.config.supervision_budget_ms:
            return "supervision_budget", "supervision_budget_exceeded"
        return None

    def _fallback(
        self,
        *,
        started_at: float,
        q: FloatArray,
        velocity: FloatArray,
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
        fallback = generate_dynamics_validated_brake(
            self.robot,
            self.checker,
            q,
            velocity,
            self.config.braking,
        )
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
        """Return one atomic action decision from a frozen measured state."""

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
                position,
                chunk,
                previous_velocity=velocity,
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
            zip(projection.predicted_positions[:-1], projection.predicted_positions[1:])
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
            zip(projection.predicted_positions[1:], projection.projected_actions)
        ):
            result = generate_dynamics_validated_brake(
                self.robot,
                self.checker,
                q_after,
                action[:PANDA_DOF],
                self.config.braking,
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


def run_integrated_panda_guard_smoke() -> dict[str, object]:
    """Run one deterministic accepted chunk on the local Panda model."""

    from armbench.mujoco_sim.continuous_collision import ContinuousCollisionConfig
    from armbench.mujoco_sim.scenarios import mujoco_scenarios

    scenario = mujoco_scenarios()["free_space"]
    robot = MuJoCoPanda.create(obstacles=())
    checker = ContinuousMuJoCoCollisionChecker(
        robot,
        ContinuousCollisionConfig(
            include_static_obstacles=False,
            include_self_collision=True,
        ),
    )
    actions = np.zeros((3, ACTION_DIM), dtype=float)
    actions[:, :PANDA_DOF] = np.array(
        [0.08, -0.04, 0.03, 0.0, 0.02, 0.0, -0.01]
    )
    actions[:, 7] = 0.5
    chunk = ActionChunk(
        actions=actions,
        source="deterministic_nonlearned_integrated_guard_smoke",
        observation_sequence_id=0,
        inference_latency_ms=0.0,
        received_at_s=100.0,
    )
    supervisor = IntegratedPandaSupervisor(
        robot,
        checker,
        IntegratedPandaGuardConfig(
            response_deadline_ms=5000.0,
            supervision_budget_ms=5000.0,
            qp_step_budget_ms=500.0,
        ),
    )
    decision = supervisor.supervise(
        scenario.start,
        np.zeros(PANDA_DOF),
        chunk,
        observed_q=scenario.start,
    )
    return {"passed": decision.status == "accepted", **decision.metrics()}
