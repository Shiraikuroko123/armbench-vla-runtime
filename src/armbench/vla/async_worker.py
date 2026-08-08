"""Latest-only worker for running blocking VLA policy calls off-loop."""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from armbench.vla.policy import ActionChunkPolicy
from armbench.vla.types import ActionChunk, VLAObservation


def _finite_clock_value(clock: Callable[[], float]) -> float:
    value = float(clock())
    if not math.isfinite(value):
        raise ValueError("monotonic clock returned a non-finite value")
    return value


def _elapsed_ms(start_s: float, end_s: float) -> float:
    elapsed = (end_s - start_s) * 1000.0
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("monotonic clock moved backwards")
    return elapsed


@dataclass(frozen=True)
class PolicySubmission:
    request_id: int
    observation_sequence_id: int
    submitted_at_s: float
    replaced_request_id: int | None


@dataclass(frozen=True)
class PolicyOutcome:
    request_id: int
    observation: VLAObservation
    submitted_at_s: float
    started_at_s: float
    finished_at_s: float
    worker_thread_id: int
    chunk: ActionChunk | None
    failure_type: str | None = None
    failure_message: str | None = None
    worker_process_id: int | None = None

    def __post_init__(self) -> None:
        timing = np.asarray(
            [self.submitted_at_s, self.started_at_s, self.finished_at_s],
            dtype=float,
        )
        if (
            self.request_id < 0
            or not np.all(np.isfinite(timing))
            or self.started_at_s < self.submitted_at_s
            or self.finished_at_s < self.started_at_s
            or (
                self.worker_process_id is not None
                and self.worker_process_id <= 0
            )
        ):
            raise ValueError("policy outcome timing is invalid")
        has_chunk = self.chunk is not None
        has_failure = self.failure_type is not None
        if has_chunk == has_failure:
            raise ValueError("policy outcome must contain one chunk or one failure")
        if has_failure and not (self.failure_message or "").strip():
            raise ValueError("policy failure must include a message")

    @property
    def succeeded(self) -> bool:
        return self.chunk is not None

    @property
    def queue_wait_ms(self) -> float:
        return _elapsed_ms(self.submitted_at_s, self.started_at_s)

    @property
    def worker_latency_ms(self) -> float:
        return _elapsed_ms(self.started_at_s, self.finished_at_s)

    def metrics(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "observation_sequence_id": self.observation.sequence_id,
            "succeeded": self.succeeded,
            "queue_wait_ms": self.queue_wait_ms,
            "worker_latency_ms": self.worker_latency_ms,
            "worker_thread_id": self.worker_thread_id,
            "worker_process_id": self.worker_process_id,
            "policy_source": self.chunk.source if self.chunk is not None else None,
            "failure_type": self.failure_type,
            "failure_message": self.failure_message,
        }


@dataclass(frozen=True)
class _PolicyRequest:
    request_id: int
    observation: VLAObservation
    submitted_at_s: float


class LatestPolicyWorker:
    """Run blocking policy calls off the control thread with a latest-only queue.

    One request may be in flight and one pending observation is retained. A new
    submission replaces only the pending request; Python cannot cancel a policy
    call that is already executing.
    """

    def __init__(
        self,
        policy: ActionChunkPolicy,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_outcomes: int = 32,
        thread_name: str = "armbench-policy-worker",
    ) -> None:
        if max_outcomes <= 0:
            raise ValueError("max_outcomes must be positive")
        self.policy = policy
        self._clock = clock
        self._max_outcomes = int(max_outcomes)
        self._condition = threading.Condition()
        self._pending: _PolicyRequest | None = None
        self._outcomes: deque[PolicyOutcome] = deque()
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
            name=thread_name,
            daemon=True,
        )
        self._thread.start()

    def submit(self, observation: VLAObservation) -> PolicySubmission:
        if not isinstance(observation, VLAObservation):
            raise TypeError("observation must be a VLAObservation")
        submitted_at = _finite_clock_value(self._clock)
        with self._condition:
            if self._closed:
                raise RuntimeError("policy worker is closed")
            request_id = self._next_request_id
            self._next_request_id += 1
            replaced = self._pending.request_id if self._pending is not None else None
            if replaced is not None:
                self._superseded_pending += 1
            self._pending = _PolicyRequest(request_id, observation, submitted_at)
            self._submitted += 1
            self._condition.notify_all()
        return PolicySubmission(
            request_id=request_id,
            observation_sequence_id=observation.sequence_id,
            submitted_at_s=submitted_at,
            replaced_request_id=replaced,
        )

    def drain(self) -> tuple[PolicyOutcome, ...]:
        """Return completed outcomes without waiting for policy inference."""

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
                "pending": self._pending is not None,
                "worker_thread_id": self._worker_thread_id,
                "worker_alive": self._thread.is_alive(),
                "closed": self._closed,
            }

    def close(self, *, timeout_s: float = 2.0) -> bool:
        if not math.isfinite(timeout_s) or timeout_s < 0.0:
            raise ValueError("timeout_s must be finite and nonnegative")
        with self._condition:
            self._closed = True
            if self._pending is not None:
                self._cancelled_pending += 1
                self._pending = None
            self._condition.notify_all()
        self._thread.join(timeout_s)
        return not self._thread.is_alive()

    def __enter__(self) -> "LatestPolicyWorker":
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
            started_at = _finite_clock_value(self._clock)
            chunk: ActionChunk | None = None
            failure_type: str | None = None
            failure_message: str | None = None
            try:
                candidate = self.policy.infer(request.observation)
                if not isinstance(candidate, ActionChunk):
                    raise TypeError("policy did not return an ActionChunk")
                if (
                    candidate.observation_sequence_id
                    != request.observation.sequence_id
                ):
                    raise ValueError(
                        "policy response sequence does not match its observation"
                    )
                chunk = candidate
            except Exception as error:
                failure_type = type(error).__name__
                failure_message = (str(error).strip() or "no error message")[:500]
            finished_at = _finite_clock_value(self._clock)
            if finished_at < started_at:
                chunk = None
                failure_type = "ClockRegressionError"
                failure_message = "monotonic clock moved backwards during inference"
                finished_at = started_at
            outcome = PolicyOutcome(
                request_id=request.request_id,
                observation=request.observation,
                submitted_at_s=request.submitted_at_s,
                started_at_s=started_at,
                finished_at_s=finished_at,
                worker_thread_id=threading.get_ident(),
                chunk=chunk,
                failure_type=failure_type,
                failure_message=failure_message,
                worker_process_id=os.getpid(),
            )
            with self._condition:
                if len(self._outcomes) >= self._max_outcomes:
                    self._outcomes.popleft()
                    self._dropped_outcomes += 1
                self._outcomes.append(outcome)
                self._completed += 1
                self._failed += not outcome.succeeded
                self._condition.notify_all()
