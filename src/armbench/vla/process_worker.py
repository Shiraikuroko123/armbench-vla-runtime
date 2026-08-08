"""Spawn-safe policy worker with independently ticking parent control time."""

from __future__ import annotations

from dataclasses import dataclass
import math
import multiprocessing as mp
import os
import pickle
from queue import Empty, Full
import threading
import time
from typing import Protocol

import numpy as np

from armbench.vla.async_worker import PolicyOutcome, PolicySubmission
from armbench.vla.policy import ActionChunkPolicy
from armbench.vla.types import ActionChunk, DROID_ACTION_DIM, VLAObservation


class ActionChunkPolicyFactory(Protocol):
    """Pickle-safe callable that constructs all policy resources in the child."""

    def __call__(self) -> ActionChunkPolicy: ...


@dataclass(frozen=True)
class _ProcessRequest:
    request_id: int
    observation: VLAObservation
    submitted_at_s: float


@dataclass(frozen=True)
class _StartupResult:
    succeeded: bool
    process_id: int
    failure_type: str | None = None
    failure_message: str | None = None


def _finite_monotonic() -> float:
    value = float(time.monotonic())
    if not math.isfinite(value):
        raise ValueError("monotonic clock returned a non-finite value")
    return value


def _increment(value: object, amount: int = 1) -> None:
    lock = value.get_lock()
    with lock:
        value.value += amount


def _publish_outcome(outcome_queue: object, outcome: PolicyOutcome, dropped: object) -> None:
    try:
        outcome_queue.put_nowait(outcome)
        return
    except Full:
        pass
    try:
        outcome_queue.get_nowait()
    except Empty:
        pass
    else:
        _increment(dropped)
    try:
        outcome_queue.put_nowait(outcome)
    except Full:
        _increment(dropped)


def _process_main(
    policy_factory: ActionChunkPolicyFactory,
    request_queue: object,
    outcome_queue: object,
    startup_queue: object,
    stop_event: object,
    started_count: object,
    completed_count: object,
    failed_count: object,
    dropped_count: object,
) -> None:
    process_id = os.getpid()
    try:
        policy = policy_factory()
        if not hasattr(policy, "infer"):
            raise TypeError("policy factory did not return an ActionChunkPolicy")
    except Exception as error:
        startup_queue.put(
            _StartupResult(
                succeeded=False,
                process_id=process_id,
                failure_type=type(error).__name__,
                failure_message=(str(error).strip() or "no error message")[:500],
            )
        )
        return
    startup_queue.put(_StartupResult(succeeded=True, process_id=process_id))
    try:
        while not stop_event.is_set():
            try:
                request = request_queue.get(timeout=0.05)
            except Empty:
                continue
            if request is None:
                return
            if not isinstance(request, _ProcessRequest):
                continue
            _increment(started_count)
            started_at_s = _finite_monotonic()
            chunk: ActionChunk | None = None
            failure_type: str | None = None
            failure_message: str | None = None
            try:
                candidate = policy.infer(request.observation)
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
            finished_at_s = _finite_monotonic()
            if finished_at_s < started_at_s:
                chunk = None
                failure_type = "ClockRegressionError"
                failure_message = "monotonic clock moved backwards during inference"
                finished_at_s = started_at_s
            try:
                outcome = PolicyOutcome(
                    request_id=request.request_id,
                    observation=request.observation,
                    submitted_at_s=request.submitted_at_s,
                    started_at_s=started_at_s,
                    finished_at_s=finished_at_s,
                    worker_thread_id=threading.get_ident(),
                    worker_process_id=process_id,
                    chunk=chunk,
                    failure_type=failure_type,
                    failure_message=failure_message,
                )
            except Exception as error:
                now_s = max(request.submitted_at_s, _finite_monotonic())
                outcome = PolicyOutcome(
                    request_id=request.request_id,
                    observation=request.observation,
                    submitted_at_s=request.submitted_at_s,
                    started_at_s=now_s,
                    finished_at_s=now_s,
                    worker_thread_id=threading.get_ident(),
                    worker_process_id=process_id,
                    chunk=None,
                    failure_type=type(error).__name__,
                    failure_message=(str(error).strip() or "no error message")[:500],
                )
            _publish_outcome(outcome_queue, outcome, dropped_count)
            _increment(completed_count)
            if not outcome.succeeded:
                _increment(failed_count)
    finally:
        close = getattr(policy, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


class ProcessPolicyWorker:
    """Run blocking inference in a spawned child with a latest-only input slot."""

    def __init__(
        self,
        policy_factory: ActionChunkPolicyFactory,
        *,
        context_name: str = "spawn",
        max_outcomes: int = 32,
        startup_timeout_s: float = 10.0,
    ) -> None:
        if max_outcomes <= 0:
            raise ValueError("max_outcomes must be positive")
        if not math.isfinite(startup_timeout_s) or startup_timeout_s <= 0.0:
            raise ValueError("startup_timeout_s must be finite and positive")
        try:
            pickle.dumps(policy_factory)
        except Exception as error:
            raise TypeError("policy_factory must be pickle-safe") from error
        context = mp.get_context(context_name)
        self._context_name = context_name
        self._request_queue = context.Queue(maxsize=1)
        self._outcome_queue = context.Queue(maxsize=max_outcomes)
        self._startup_queue = context.Queue(maxsize=1)
        self._stop_event = context.Event()
        self._started_count = context.Value("q", 0)
        self._completed_count = context.Value("q", 0)
        self._failed_count = context.Value("q", 0)
        self._dropped_count = context.Value("q", 0)
        self._lock = threading.Lock()
        self._closed = False
        self._next_request_id = 0
        self._submitted = 0
        self._superseded_pending = 0
        self._cancelled_pending = 0
        self._process = context.Process(
            target=_process_main,
            args=(
                policy_factory,
                self._request_queue,
                self._outcome_queue,
                self._startup_queue,
                self._stop_event,
                self._started_count,
                self._completed_count,
                self._failed_count,
                self._dropped_count,
            ),
            name="armbench-policy-process",
            daemon=True,
        )
        self._process.start()
        try:
            startup = self._startup_queue.get(timeout=startup_timeout_s)
        except Empty as error:
            self._stop_event.set()
            self._process.terminate()
            self._process.join(1.0)
            raise TimeoutError("policy process did not report startup") from error
        if not isinstance(startup, _StartupResult) or not startup.succeeded:
            self._process.join(1.0)
            failure_type = getattr(startup, "failure_type", "StartupError")
            failure_message = getattr(startup, "failure_message", "unknown error")
            raise RuntimeError(
                f"policy process startup failed: {failure_type}: {failure_message}"
            )
        self._process_id = startup.process_id

    def submit(self, observation: VLAObservation) -> PolicySubmission:
        if not isinstance(observation, VLAObservation):
            raise TypeError("observation must be a VLAObservation")
        submitted_at_s = _finite_monotonic()
        with self._lock:
            if self._closed:
                raise RuntimeError("policy worker is closed")
            request_id = self._next_request_id
            self._next_request_id += 1
            replaced: int | None = None
            try:
                pending = self._request_queue.get_nowait()
            except Empty:
                pass
            else:
                if isinstance(pending, _ProcessRequest):
                    replaced = pending.request_id
                    self._superseded_pending += 1
            request = _ProcessRequest(request_id, observation, submitted_at_s)
            try:
                self._request_queue.put(request, timeout=0.25)
            except Full as error:
                raise RuntimeError("policy process request slot remained full") from error
            self._submitted += 1
        return PolicySubmission(
            request_id=request_id,
            observation_sequence_id=observation.sequence_id,
            submitted_at_s=submitted_at_s,
            replaced_request_id=replaced,
        )

    def drain(self) -> tuple[PolicyOutcome, ...]:
        outcomes: list[PolicyOutcome] = []
        while True:
            try:
                outcome = self._outcome_queue.get_nowait()
            except Empty:
                break
            if isinstance(outcome, PolicyOutcome):
                outcomes.append(outcome)
        return tuple(outcomes)

    def metrics(self) -> dict[str, object]:
        return {
            "submitted": self._submitted,
            "started": self._started_count.value,
            "completed": self._completed_count.value,
            "failed": self._failed_count.value,
            "superseded_pending": self._superseded_pending,
            "cancelled_pending": self._cancelled_pending,
            "dropped_outcomes": self._dropped_count.value,
            "worker_process_id": self._process_id,
            "worker_alive": self._process.is_alive(),
            "closed": self._closed,
            "process_start_method": self._context_name,
        }

    def close(self, *, timeout_s: float = 2.0) -> bool:
        if not math.isfinite(timeout_s) or timeout_s < 0.0:
            raise ValueError("timeout_s must be finite and nonnegative")
        with self._lock:
            if self._closed:
                return not self._process.is_alive()
            self._closed = True
            try:
                pending = self._request_queue.get_nowait()
            except Empty:
                pass
            else:
                if isinstance(pending, _ProcessRequest):
                    self._cancelled_pending += 1
            self._stop_event.set()
            try:
                self._request_queue.put_nowait(None)
            except Full:
                pass
        self._process.join(timeout_s)
        stopped_cleanly = not self._process.is_alive()
        if not stopped_cleanly:
            self._process.terminate()
            self._process.join(min(1.0, max(0.1, timeout_s)))
        self._request_queue.close()
        self._outcome_queue.close()
        self._startup_queue.close()
        return stopped_cleanly

    def __enter__(self) -> "ProcessPolicyWorker":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


@dataclass(frozen=True)
class DelayedProcessPolicyFactory:
    """Pickle-safe non-learning policy factory used by the CPU acceptance."""

    latency_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.latency_s) or self.latency_s < 0.0:
            raise ValueError("latency_s must be finite and nonnegative")

    def __call__(self) -> ActionChunkPolicy:
        return _DelayedProcessPolicy(self.latency_s)


class _DelayedProcessPolicy:
    def __init__(self, latency_s: float) -> None:
        self.latency_s = latency_s

    def infer(self, observation: VLAObservation) -> ActionChunk:
        started_at_s = _finite_monotonic()
        time.sleep(self.latency_s)
        received_at_s = _finite_monotonic()
        actions = np.zeros((15, DROID_ACTION_DIM), dtype=float)
        actions[:, 0] = np.arange(15, dtype=float) * 0.01
        actions[:, 7] = float(observation.gripper_position[0])
        return ActionChunk(
            actions=actions,
            source="delayed_scripted_process_smoke",
            observation_sequence_id=observation.sequence_id,
            inference_latency_ms=(received_at_s - started_at_s) * 1000.0,
            received_at_s=received_at_s,
        )
