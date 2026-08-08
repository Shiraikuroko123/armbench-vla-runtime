"""Non-blocking execution boundary for the integrated Panda supervisor.

The integrated supervisor deliberately performs expensive CPU checks over a
complete action chunk.  Running those checks on the actuator thread would stop
the control clock.  This module therefore owns the supervisor on a dedicated
worker thread and publishes its result through an atomic generation-aware gate.

The gate never exposes a prefix from a rejected chunk.  It also rechecks age
and state alignment when a completed plan reaches the control side, because a
plan can become stale while continuous-collision and braking certificates are
being computed.  This is best-effort Python scheduling, not hard real time.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import threading
import time

import numpy as np
from numpy.typing import ArrayLike, NDArray

from armbench.vla.integrated_panda_guard import (
    IntegratedPandaDecision,
    IntegratedPandaSupervisor,
)
from armbench.vla.types import ActionChunk


FloatArray = NDArray[np.float64]
PANDA_DOF = 7
PANDA_ACTION_DIM = 8


def _finite_monotonic() -> float:
    value = float(time.monotonic())
    if not math.isfinite(value):
        raise ValueError("monotonic clock returned a non-finite value")
    return value


def _readonly_vector(value: ArrayLike, *, label: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{label} must be numeric")
    vector = np.asarray(raw, dtype=float)
    if vector.shape != (PANDA_DOF,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must be a finite seven-vector")
    result = vector.copy()
    result.flags.writeable = False
    return result


def _readonly_actions(value: ArrayLike) -> FloatArray:
    actions = np.asarray(value, dtype=float)
    if (
        actions.ndim != 2
        or actions.shape[1] != PANDA_ACTION_DIM
        or not np.all(np.isfinite(actions))
    ):
        raise ValueError("published actions must be a finite Hx8 array")
    result = actions.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class AssuranceSubmission:
    """Receipt returned without waiting for a full supervisor decision."""

    request_id: int
    generation: int
    observation_sequence_id: int
    submitted_at_s: float
    replaced_request_id: int | None


@dataclass(frozen=True)
class _AssuranceRequest:
    request_id: int
    generation: int
    observation_sequence_id: int
    submitted_at_s: float
    q: FloatArray
    qvel: FloatArray
    observed_q: FloatArray
    response_age_ms: float
    chunk: ActionChunk


@dataclass(frozen=True)
class AssuranceOutcome:
    """One complete supervisor result or one explicit worker failure."""

    request: _AssuranceRequest
    started_at_s: float
    finished_at_s: float
    worker_thread_id: int
    decision: IntegratedPandaDecision | None
    failure_type: str | None
    failure_message: str | None

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.started_at_s)
            or not np.isfinite(self.finished_at_s)
            or self.finished_at_s < self.started_at_s
        ):
            raise ValueError("assurance outcome timestamps are invalid")
        if (self.decision is None) == (self.failure_type is None):
            raise ValueError("assurance outcome must contain one result or one failure")
        if self.failure_type is None and self.failure_message is not None:
            raise ValueError("successful assurance outcome cannot contain a failure")

    @property
    def succeeded(self) -> bool:
        return self.decision is not None

    @property
    def worker_latency_ms(self) -> float:
        return (self.finished_at_s - self.started_at_s) * 1000.0

    def metrics(self) -> dict[str, object]:
        return {
            "request_id": self.request.request_id,
            "generation": self.request.generation,
            "observation_sequence_id": self.request.observation_sequence_id,
            "succeeded": self.succeeded,
            "worker_thread_id": self.worker_thread_id,
            "worker_latency_ms": self.worker_latency_ms,
            "decision_status": (
                None if self.decision is None else self.decision.status
            ),
            "failure_type": self.failure_type,
            "failure_message": self.failure_message,
        }


class LatestIntegratedPandaWorker:
    """Run complete action assurance away from the control tick thread.

    One request may be executing and at most one newer request remains pending.
    Replacing a pending request is safe because no decision from it has been
    published.  An in-flight computation cannot be cancelled by Python and its
    eventual outcome is filtered by request id and reset generation.
    """

    def __init__(
        self,
        supervisor: IntegratedPandaSupervisor,
        *,
        max_outcomes: int = 32,
    ) -> None:
        if max_outcomes <= 0:
            raise ValueError("max_outcomes must be positive")
        self.supervisor = supervisor
        self._max_outcomes = max_outcomes
        self._condition = threading.Condition()
        self._pending: _AssuranceRequest | None = None
        self._outcomes: deque[AssuranceOutcome] = deque()
        self._closed = False
        self._next_request_id = 0
        self._submitted = 0
        self._started = 0
        self._completed = 0
        self._failed = 0
        self._superseded_pending = 0
        self._cancelled_pending = 0
        self._dropped_outcomes = 0
        self._worker_thread_id: int | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="armbench-integrated-panda-worker",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        *,
        generation: int,
        observation_sequence_id: int,
        q: ArrayLike,
        qvel: ArrayLike,
        observed_q: ArrayLike,
        response_age_ms: float,
        chunk: ActionChunk,
    ) -> AssuranceSubmission:
        if type(generation) is not int or generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        if type(observation_sequence_id) is not int or observation_sequence_id < 0:
            raise ValueError("observation sequence must be a nonnegative integer")
        if chunk.observation_sequence_id != observation_sequence_id:
            raise ValueError("action chunk and observation sequence do not match")
        if not np.isfinite(response_age_ms) or response_age_ms < 0.0:
            raise ValueError("response age must be finite and nonnegative")

        position = _readonly_vector(q, label="q")
        velocity = _readonly_vector(qvel, label="qvel")
        observed = _readonly_vector(observed_q, label="observed_q")
        submitted_at_s = _finite_monotonic()
        with self._condition:
            if self._closed:
                raise RuntimeError("integrated Panda worker is closed")
            request_id = self._next_request_id
            self._next_request_id += 1
            replaced_request_id = None
            if self._pending is not None:
                replaced_request_id = self._pending.request_id
                self._superseded_pending += 1
            self._pending = _AssuranceRequest(
                request_id=request_id,
                generation=generation,
                observation_sequence_id=observation_sequence_id,
                submitted_at_s=submitted_at_s,
                q=position,
                qvel=velocity,
                observed_q=observed,
                response_age_ms=float(response_age_ms),
                chunk=chunk,
            )
            self._submitted += 1
            self._condition.notify_all()
        return AssuranceSubmission(
            request_id=request_id,
            generation=generation,
            observation_sequence_id=observation_sequence_id,
            submitted_at_s=submitted_at_s,
            replaced_request_id=replaced_request_id,
        )

    def drain(self) -> tuple[AssuranceOutcome, ...]:
        with self._condition:
            outcomes = tuple(self._outcomes)
            self._outcomes.clear()
        return outcomes

    def metrics(self) -> dict[str, object]:
        with self._condition:
            return {
                "submitted": self._submitted,
                "started": self._started,
                "completed": self._completed,
                "failed": self._failed,
                "superseded_pending": self._superseded_pending,
                "cancelled_pending": self._cancelled_pending,
                "dropped_outcomes": self._dropped_outcomes,
                "worker_thread_id": self._worker_thread_id,
                "worker_alive": self._thread.is_alive(),
                "closed": self._closed,
            }

    def close(self, *, timeout_s: float = 10.0) -> bool:
        if not np.isfinite(timeout_s) or timeout_s < 0.0:
            raise ValueError("timeout_s must be finite and nonnegative")
        with self._condition:
            if self._closed:
                return not self._thread.is_alive()
            self._closed = True
            if self._pending is not None:
                self._pending = None
                self._cancelled_pending += 1
            self._condition.notify_all()
        self._thread.join(timeout_s)
        return not self._thread.is_alive()

    def __enter__(self) -> "LatestIntegratedPandaWorker":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _run(self) -> None:
        with self._condition:
            self._worker_thread_id = threading.get_ident()
            self._condition.notify_all()
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._closed and self._pending is None:
                    return
                request = self._pending
                self._pending = None
                self._started += 1
            assert request is not None

            started_at_s = _finite_monotonic()
            decision: IntegratedPandaDecision | None = None
            failure_type: str | None = None
            failure_message: str | None = None
            try:
                decision = self.supervisor.supervise(
                    request.q,
                    request.qvel,
                    request.chunk,
                    observed_q=request.observed_q,
                    response_age_ms=request.response_age_ms,
                )
            except Exception as error:
                failure_type = type(error).__name__
                failure_message = (str(error).strip() or "no error message")[:500]
            finished_at_s = max(started_at_s, _finite_monotonic())
            outcome = AssuranceOutcome(
                request=request,
                started_at_s=started_at_s,
                finished_at_s=finished_at_s,
                worker_thread_id=threading.get_ident(),
                decision=decision,
                failure_type=failure_type,
                failure_message=failure_message,
            )
            with self._condition:
                if len(self._outcomes) >= self._max_outcomes:
                    self._outcomes.popleft()
                    self._dropped_outcomes += 1
                self._outcomes.append(outcome)
                self._completed += 1
                self._failed += int(not outcome.succeeded)
                self._condition.notify_all()


@dataclass(frozen=True)
class AtomicAssuranceDecision:
    """Control-side publication of either one whole plan or no policy motion."""

    status: str
    reason: str
    request_id: int
    generation: int
    observation_sequence_id: int
    response_age_ms: float
    activation_state_mismatch_rad: float
    policy_actions: FloatArray
    supervisor_status: str | None
    fallback_validated: bool

    def __post_init__(self) -> None:
        if self.status not in {
            "execute",
            "verified_brake",
            "hold",
            "unrecoverable_stop",
        }:
            raise ValueError("atomic assurance status is invalid")
        if not self.reason:
            raise ValueError("atomic assurance decision requires a reason")
        if not np.isfinite(self.response_age_ms) or self.response_age_ms < 0.0:
            raise ValueError("atomic assurance response age is invalid")
        if (
            not np.isfinite(self.activation_state_mismatch_rad)
            or self.activation_state_mismatch_rad < 0.0
        ):
            raise ValueError("atomic assurance state mismatch is invalid")
        actions = _readonly_actions(self.policy_actions)
        if self.status == "execute" and len(actions) == 0:
            raise ValueError("execute requires a complete policy plan")
        if self.status != "execute" and len(actions) != 0:
            raise ValueError("non-execute decisions cannot expose policy actions")
        object.__setattr__(self, "policy_actions", actions)

    @property
    def policy_actions_executable(self) -> bool:
        return self.status == "execute"

    def metrics(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "request_id": self.request_id,
            "generation": self.generation,
            "observation_sequence_id": self.observation_sequence_id,
            "response_age_ms": self.response_age_ms,
            "activation_state_mismatch_rad": (
                self.activation_state_mismatch_rad
            ),
            "policy_actions_executable": self.policy_actions_executable,
            "policy_action_count": len(self.policy_actions),
            "supervisor_status": self.supervisor_status,
            "fallback_validated": self.fallback_validated,
        }


class AtomicPandaPlanGate:
    """Reject stale worker results across time, motion, ordering, and reset."""

    def __init__(self, supervisor: IntegratedPandaSupervisor) -> None:
        self.config = supervisor.config
        self._generation = 0
        self._last_committed_request_id = -1

    @property
    def generation(self) -> int:
        return self._generation

    def reset(self) -> int:
        """Invalidate every in-flight result from the previous runtime epoch."""

        self._generation += 1
        self._last_committed_request_id = -1
        return self._generation

    def _hold(
        self,
        outcome: AssuranceOutcome,
        *,
        reason: str,
        response_age_ms: float,
        mismatch: float,
    ) -> AtomicAssuranceDecision:
        return AtomicAssuranceDecision(
            status="hold",
            reason=reason,
            request_id=outcome.request.request_id,
            generation=self._generation,
            observation_sequence_id=outcome.request.observation_sequence_id,
            response_age_ms=response_age_ms,
            activation_state_mismatch_rad=mismatch,
            policy_actions=np.empty((0, PANDA_ACTION_DIM), dtype=float),
            supervisor_status=(
                None if outcome.decision is None else outcome.decision.status
            ),
            fallback_validated=False,
        )

    def commit(
        self,
        outcome: AssuranceOutcome,
        *,
        q_now: ArrayLike,
        now_s: float | None = None,
    ) -> AtomicAssuranceDecision:
        """Publish a complete plan only if it is still valid at activation."""

        activation_time_s = _finite_monotonic() if now_s is None else float(now_s)
        if not np.isfinite(activation_time_s):
            raise ValueError("activation time must be finite")
        current_q = _readonly_vector(q_now, label="q_now")
        elapsed_since_submission_ms = max(
            0.0, (activation_time_s - outcome.request.submitted_at_s) * 1000.0
        )
        response_age_ms = (
            outcome.request.response_age_ms + elapsed_since_submission_ms
        )
        mismatch = float(np.max(np.abs(current_q - outcome.request.q)))

        if outcome.request.generation != self._generation:
            return self._hold(
                outcome,
                reason="reset_generation_mismatch",
                response_age_ms=response_age_ms,
                mismatch=mismatch,
            )
        if outcome.request.request_id <= self._last_committed_request_id:
            return self._hold(
                outcome,
                reason="replayed_or_out_of_order_assurance_result",
                response_age_ms=response_age_ms,
                mismatch=mismatch,
            )
        self._last_committed_request_id = outcome.request.request_id
        if not outcome.succeeded or outcome.decision is None:
            return self._hold(
                outcome,
                reason=f"assurance_worker_failed:{outcome.failure_type}",
                response_age_ms=response_age_ms,
                mismatch=mismatch,
            )
        decision = outcome.decision
        if decision.status != "accepted":
            # A stale policy response can intentionally produce a verified
            # brake/hold decision.  Its age is the reason for that fallback,
            # not a reason to erase the already computed supervisor status.
            # State drift is still checked because the brake certificate is
            # tied to the measured request state.
            if mismatch > self.config.max_state_mismatch_rad:
                return self._hold(
                    outcome,
                    reason="state_changed_during_assurance",
                    response_age_ms=response_age_ms,
                    mismatch=mismatch,
                )
            return AtomicAssuranceDecision(
                status=decision.status,
                reason=decision.reason,
                request_id=outcome.request.request_id,
                generation=self._generation,
                observation_sequence_id=outcome.request.observation_sequence_id,
                response_age_ms=response_age_ms,
                activation_state_mismatch_rad=mismatch,
                policy_actions=np.empty((0, PANDA_ACTION_DIM), dtype=float),
                supervisor_status=decision.status,
                fallback_validated=decision.fallback_covered,
            )
        if response_age_ms > self.config.response_deadline_ms:
            return self._hold(
                outcome,
                reason="response_deadline_exceeded_before_activation",
                response_age_ms=response_age_ms,
                mismatch=mismatch,
            )
        if mismatch > self.config.max_state_mismatch_rad:
            return self._hold(
                outcome,
                reason="state_changed_during_assurance",
                response_age_ms=response_age_ms,
                mismatch=mismatch,
            )

        return AtomicAssuranceDecision(
            status="execute",
            reason=decision.reason,
            request_id=outcome.request.request_id,
            generation=self._generation,
            observation_sequence_id=outcome.request.observation_sequence_id,
            response_age_ms=response_age_ms,
            activation_state_mismatch_rad=mismatch,
            policy_actions=decision.executable_actions,
            supervisor_status=decision.status,
            fallback_validated=decision.fallback_covered,
        )


__all__ = [
    "AssuranceOutcome",
    "AssuranceSubmission",
    "AtomicAssuranceDecision",
    "AtomicPandaPlanGate",
    "LatestIntegratedPandaWorker",
]
