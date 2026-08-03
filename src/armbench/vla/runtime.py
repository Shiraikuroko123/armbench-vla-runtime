"""Fail-closed orchestration around a VLA policy and action guard."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

from armbench.vla.guard import ActionChunkGuard, GuardResult
from armbench.vla.policy import ActionChunkPolicy
from armbench.vla.types import ActionChunk, DROID_ACTION_DIM, VLAObservation

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RuntimeFailure:
    stage: str
    error_type: str
    message: str


@dataclass(frozen=True)
class RuntimeDecision:
    """Auditable command decision produced by the runtime supervisor."""

    status: str
    actions: FloatArray
    predicted_positions: FloatArray
    observation_sequence_id: int
    policy_source: str | None
    raw_actions: FloatArray | None
    failure: RuntimeFailure | None
    runtime_failure_latched: bool
    guard_result: GuardResult | None
    supervisor_latency_ms: float

    @property
    def used_runtime_fallback(self) -> bool:
        return self.status == "runtime_fallback"

    def metrics(self) -> dict[str, object]:
        return {
            "status": self.status,
            "used_runtime_fallback": self.used_runtime_fallback,
            "runtime_failure_latched": self.runtime_failure_latched,
            "observation_sequence_id": self.observation_sequence_id,
            "policy_source": self.policy_source,
            "failure_stage": self.failure.stage if self.failure else None,
            "failure_type": self.failure.error_type if self.failure else None,
            "failure_message": self.failure.message if self.failure else None,
            "supervisor_latency_ms": self.supervisor_latency_ms,
            "guard": self.guard_result.metrics() if self.guard_result else None,
        }


class VLARuntimeSupervisor:
    """Convert policy/guard failures into a latched, provenance-safe hold."""

    def __init__(
        self,
        policy: ActionChunkPolicy,
        guard: ActionChunkGuard,
        *,
        fallback_horizon: int = 15,
        latch_on_failure: bool = True,
    ) -> None:
        if fallback_horizon <= 0:
            raise ValueError("fallback_horizon must be positive")
        self.policy = policy
        self.guard = guard
        self.fallback_horizon = int(fallback_horizon)
        self.latch_on_failure = bool(latch_on_failure)
        self._latched_failure: RuntimeFailure | None = None

    def reset(self, previous_joint_velocity: ArrayLike | None = None) -> None:
        """Resynchronize command state, then clear all runtime fallbacks."""

        self.guard.reset(previous_joint_velocity=previous_joint_velocity)
        self._latched_failure = None

    @staticmethod
    def _failure(stage: str, error: Exception) -> RuntimeFailure:
        message = str(error).strip() or "no error message"
        return RuntimeFailure(
            stage=stage,
            error_type=type(error).__name__,
            message=message[:500],
        )

    def _hold_decision(
        self,
        q_start: ArrayLike,
        gripper_position: float,
        observation: VLAObservation,
        failure: RuntimeFailure,
        *,
        policy_source: str | None,
        raw_actions: FloatArray | None,
        started: float,
    ) -> RuntimeDecision:
        q = self.guard.checker.robot.validate_configuration(q_start)
        if not 0.0 <= gripper_position <= 1.0:
            raise ValueError("gripper_position must be normalized to [0, 1]")
        if self.latch_on_failure and self._latched_failure is None:
            self._latched_failure = failure
        actions = np.zeros(
            (self.fallback_horizon, DROID_ACTION_DIM), dtype=float
        )
        actions[:, 7] = gripper_position
        positions = np.repeat(q[None, :], self.fallback_horizon + 1, axis=0)
        return RuntimeDecision(
            status="runtime_fallback",
            actions=actions,
            predicted_positions=positions,
            observation_sequence_id=observation.sequence_id,
            policy_source=policy_source,
            raw_actions=raw_actions,
            failure=failure,
            runtime_failure_latched=self._latched_failure is not None,
            guard_result=None,
            supervisor_latency_ms=(perf_counter() - started) * 1000.0,
        )

    def infer_and_guard(
        self,
        q_start: ArrayLike,
        gripper_position: float,
        observation: VLAObservation,
        *,
        on_policy_response: (
            Callable[[ActionChunk], tuple[ArrayLike, float]] | None
        ) = None,
        on_policy_failure: (
            Callable[[], tuple[ArrayLike, float]] | None
        ) = None,
    ) -> RuntimeDecision:
        started = perf_counter()
        if self._latched_failure is not None:
            failure = RuntimeFailure(
                stage="runtime_latched",
                error_type=self._latched_failure.error_type,
                message=(
                    f"latched after {self._latched_failure.stage}: "
                    f"{self._latched_failure.message}"
                ),
            )
            return self._hold_decision(
                q_start,
                gripper_position,
                observation,
                failure,
                policy_source=None,
                raw_actions=None,
                started=started,
            )
        try:
            chunk = self.policy.infer(observation)
        except Exception as error:
            fallback_q = q_start
            fallback_gripper = gripper_position
            if on_policy_failure is not None:
                try:
                    fallback_q, fallback_gripper = on_policy_failure()
                except Exception as dispatch_error:
                    return self._hold_decision(
                        q_start,
                        gripper_position,
                        observation,
                        self._failure("response_dispatch", dispatch_error),
                        policy_source=None,
                        raw_actions=None,
                        started=started,
                    )
            return self._hold_decision(
                fallback_q,
                fallback_gripper,
                observation,
                self._failure("policy_inference", error),
                policy_source=None,
                raw_actions=None,
                started=started,
            )
        dispatch_q = q_start
        dispatch_gripper = gripper_position
        if on_policy_response is not None:
            try:
                dispatch_q, dispatch_gripper = on_policy_response(chunk)
            except Exception as error:
                return self._hold_decision(
                    q_start,
                    gripper_position,
                    observation,
                    self._failure("response_dispatch", error),
                    policy_source=chunk.source,
                    raw_actions=chunk.actions,
                    started=started,
                )
        try:
            result = self.guard.guard(
                dispatch_q, dispatch_gripper, observation, chunk
            )
        except Exception as error:
            return self._hold_decision(
                dispatch_q,
                dispatch_gripper,
                observation,
                self._failure("guard_validation", error),
                policy_source=getattr(chunk, "source", None),
                raw_actions=getattr(chunk, "actions", None),
                started=started,
            )
        if not result.safe_after_guard:
            error = RuntimeError(
                "guard could not satisfy every configured action constraint"
            )
            return self._hold_decision(
                dispatch_q,
                dispatch_gripper,
                observation,
                self._failure("guard_assurance", error),
                policy_source=chunk.source,
                raw_actions=chunk.actions,
                started=started,
            )
        return RuntimeDecision(
            status="guarded",
            actions=result.guarded_actions,
            predicted_positions=result.predicted_positions,
            observation_sequence_id=observation.sequence_id,
            policy_source=chunk.source,
            raw_actions=chunk.actions,
            failure=None,
            runtime_failure_latched=False,
            guard_result=result,
            supervisor_latency_ms=(perf_counter() - started) * 1000.0,
        )
