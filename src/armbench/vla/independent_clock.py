"""Independent-clock, latest-only inference harness.

The older LIBERO evaluator in :mod:`integrations.openpi.libero_runtime`
performs a blocking policy call and then advances the environment to catch up
with the measured delay.  That is useful for a deterministic comparison, but
it cannot demonstrate that a controller keeps ticking while inference is in
flight.  This module provides a deliberately small runtime boundary for that
question:

* the parent owns the environment and advances it at wall-clock ticks;
* a spawned child owns the injectable, potentially blocking provider;
* one pending request is retained (a newer observation supersedes it), while
  an already-running request is allowed to finish;
* every request and control tick is represented in the returned artifact.

The environment and provider contracts are intentionally generic.  A LIBERO
adapter can turn its observation mapping into a picklable payload and turn a
selected action vector back into the environment's native action type without
changing the scheduler below.  No model, simulator, or GPU is imported here;
``run_independent_clock_smoke`` therefore runs on a CPU-only checkout.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import multiprocessing as mp
import os
import pickle
from queue import Empty, Full
import time
import threading
from typing import Any, Protocol

import numpy as np


class IndependentClockEnvironment(Protocol):
    """Minimal parent-owned environment contract.

    ``observe`` must be non-blocking with respect to policy inference.  The
    return value only needs to be picklable because it crosses the process
    boundary.  ``step`` may return either Gym's four-tuple or five-tuple.
    """

    def reset(self) -> Any: ...

    def observe(self) -> Any: ...

    def step(self, action: np.ndarray) -> Any: ...


class IndependentClockProviderFactory(Protocol):
    """Pickle-safe callable that constructs a provider inside the child."""

    def __call__(self) -> Any: ...


@dataclass(frozen=True)
class IndependentClockConfig:
    """Wall-clock scheduler and deadline parameters for one run."""

    control_period_s: float = 0.01
    action_period_s: float = 0.05
    deadline_s: float = 0.20
    max_ticks: int = 40
    action_dim: int = 8
    submit_every_ticks: int = 1
    startup_timeout_s: float = 10.0
    shutdown_timeout_s: float = 2.0
    boundary_tolerance_s: float = 1e-9

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.control_period_s,
                self.action_period_s,
                self.deadline_s,
                self.startup_timeout_s,
                self.shutdown_timeout_s,
                self.boundary_tolerance_s,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("independent-clock timing values must be finite")
        if self.control_period_s <= 0.0:
            raise ValueError("control_period_s must be positive")
        if self.action_period_s <= 0.0:
            raise ValueError("action_period_s must be positive")
        if self.deadline_s < 0.0:
            raise ValueError("deadline_s must be nonnegative")
        if self.max_ticks <= 0:
            raise ValueError("max_ticks must be positive")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.submit_every_ticks <= 0:
            raise ValueError("submit_every_ticks must be positive")
        if self.startup_timeout_s <= 0.0:
            raise ValueError("startup_timeout_s must be positive")
        if self.shutdown_timeout_s < 0.0:
            raise ValueError("shutdown_timeout_s must be nonnegative")
        if self.boundary_tolerance_s < 0.0:
            raise ValueError("boundary_tolerance_s must be nonnegative")


@dataclass(frozen=True)
class IndependentClockSubmission:
    """Submission metadata returned by :class:`IndependentClockWorker`."""

    request_id: int
    observation_sequence_id: int
    submitted_at_s: float
    replaced_request_id: int | None
    parent_process_id: int
    worker_process_id: int

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "observation_sequence_id": self.observation_sequence_id,
            "submitted_at_s": self.submitted_at_s,
            "replaced_request_id": self.replaced_request_id,
            "parent_process_id": self.parent_process_id,
            "worker_process_id": self.worker_process_id,
        }


@dataclass(frozen=True)
class _ProcessRequest:
    request_id: int
    observation: Any
    observation_sequence_id: int
    captured_at_s: float
    submitted_at_s: float


@dataclass(frozen=True)
class _StartupMessage:
    succeeded: bool
    worker_process_id: int
    failure_type: str | None = None
    failure_message: str | None = None


@dataclass(frozen=True)
class _WorkerMessage:
    """A started/completed event emitted by the spawned child."""

    kind: str
    request_id: int
    observation_sequence_id: int
    worker_process_id: int
    at_s: float
    started_at_s: float | None = None
    completed_at_s: float | None = None
    actions: np.ndarray | None = None
    source: str | None = None
    failure_type: str | None = None
    failure_message: str | None = None


def _finite_monotonic() -> float:
    value = float(time.monotonic())
    if not math.isfinite(value):
        raise ValueError("monotonic clock returned a non-finite value")
    return value


def _safe_message(value: object) -> str:
    return (str(value).strip() or "no error message")[:500]


def _increment_shared(value: object) -> None:
    lock = value.get_lock()
    with lock:
        value.value += 1


def _emit(queue: object, message: _WorkerMessage, dropped: object) -> None:
    """Publish a lifecycle event without allowing a full audit queue to hang.

    The queue is sized generously for a normal episode.  If a caller submits
    pathological amounts of work, dropping an event is preferable to
    blocking the inference process forever; the parent exposes the queue-drop
    count in its metrics so the artifact cannot silently look complete.
    """

    try:
        queue.put_nowait(message)
    except Full:
        # There is no safe way to block a policy call here.  Keep an explicit
        # count so an artifact can fail closed if a caller overflows the audit
        # channel.
        _increment_shared(dropped)


def _provider_infer(provider: Any, observation: Any) -> Any:
    infer = getattr(provider, "infer", None)
    if callable(infer):
        return infer(observation)
    if callable(provider):
        return provider(observation)
    raise TypeError("provider must expose infer(observation) or be callable")


def _normalise_provider_result(
    result: Any,
    *,
    action_dim: int,
) -> tuple[np.ndarray, str]:
    """Accept an ndarray, ``{"actions": ...}``, or an ActionChunk-like value."""

    source = "provider"
    candidate = result
    if isinstance(result, Mapping):
        if "actions" not in result:
            raise ValueError("provider mapping must contain an 'actions' field")
        candidate = result["actions"]
        if result.get("source") is not None:
            source = str(result["source"])
    elif hasattr(result, "actions"):
        candidate = getattr(result, "actions")
        if getattr(result, "source", None) is not None:
            source = str(getattr(result, "source"))
    try:
        actions = np.asarray(candidate, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("provider actions must be numeric") from error
    if (
        actions.ndim != 2
        or actions.shape[1] != action_dim
        or actions.shape[0] <= 0
        or not np.all(np.isfinite(actions))
    ):
        raise ValueError(
            "provider actions must be finite with shape (horizon, %d)"
            % action_dim
        )
    return np.ascontiguousarray(actions), source


def _worker_main(
    provider_factory: IndependentClockProviderFactory,
    request_queue: object,
    message_queue: object,
    startup_queue: object,
    stop_event: object,
    action_dim: int,
    dropped_messages: object,
) -> None:
    worker_pid = os.getpid()
    try:
        provider = provider_factory()
    except Exception as error:
        startup_queue.put(
            _StartupMessage(
                False,
                worker_pid,
                type(error).__name__,
                _safe_message(error),
            )
        )
        return
    startup_queue.put(_StartupMessage(True, worker_pid))
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
            started_at = _finite_monotonic()
            _emit(
                message_queue,
                _WorkerMessage(
                    kind="started",
                    request_id=request.request_id,
                    observation_sequence_id=request.observation_sequence_id,
                    worker_process_id=worker_pid,
                    at_s=started_at,
                    started_at_s=started_at,
                ),
                dropped_messages,
            )
            actions: np.ndarray | None = None
            source: str | None = None
            failure_type: str | None = None
            failure_message: str | None = None
            try:
                result = _provider_infer(provider, request.observation)
                actions, source = _normalise_provider_result(
                    result, action_dim=action_dim
                )
            except Exception as error:
                failure_type = type(error).__name__
                failure_message = _safe_message(error)
            finished_at = _finite_monotonic()
            if finished_at < started_at:
                actions = None
                source = None
                failure_type = "ClockRegressionError"
                failure_message = "monotonic clock moved backwards during inference"
                finished_at = started_at
            _emit(
                message_queue,
                _WorkerMessage(
                    kind="completed",
                    request_id=request.request_id,
                    observation_sequence_id=request.observation_sequence_id,
                    worker_process_id=worker_pid,
                    at_s=finished_at,
                    started_at_s=started_at,
                    completed_at_s=finished_at,
                    actions=actions,
                    source=source,
                    failure_type=failure_type,
                    failure_message=failure_message,
                ),
                dropped_messages,
            )
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


class IndependentClockWorker:
    """Spawned provider worker with one latest-only pending mailbox slot."""

    def __init__(
        self,
        provider_factory: IndependentClockProviderFactory,
        *,
        action_dim: int,
        context_name: str = "spawn",
        max_messages: int = 4096,
        startup_timeout_s: float = 10.0,
    ) -> None:
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if max_messages <= 0:
            raise ValueError("max_messages must be positive")
        if not math.isfinite(startup_timeout_s) or startup_timeout_s <= 0.0:
            raise ValueError("startup_timeout_s must be finite and positive")
        try:
            pickle.dumps(provider_factory)
        except Exception as error:
            raise TypeError("provider_factory must be pickle-safe") from error
        context = mp.get_context(context_name)
        self._context_name = context_name
        self._parent_process_id = os.getpid()
        self._request_queue = context.Queue(maxsize=1)
        self._message_queue = context.Queue(maxsize=max_messages)
        self._startup_queue = context.Queue(maxsize=1)
        self._stop_event = context.Event()
        self._dropped_messages = context.Value("q", 0)
        self._lock = threading.Lock()
        self._closed = False
        self._next_request_id = 0
        self._submitted = 0
        self._superseded = 0
        self._queue_dropped = 0
        self._process = context.Process(
            target=_worker_main,
            args=(
                provider_factory,
                self._request_queue,
                self._message_queue,
                self._startup_queue,
                self._stop_event,
                int(action_dim),
                self._dropped_messages,
            ),
            name="armbench-independent-clock-provider",
            daemon=True,
        )
        self._process.start()
        try:
            startup = self._startup_queue.get(timeout=startup_timeout_s)
        except Empty as error:
            self._stop_event.set()
            self._process.terminate()
            self._process.join(1.0)
            raise TimeoutError("provider process did not report startup") from error
        if not isinstance(startup, _StartupMessage) or not startup.succeeded:
            self._process.join(1.0)
            failure_type = getattr(startup, "failure_type", "StartupError")
            failure_message = getattr(startup, "failure_message", "unknown error")
            raise RuntimeError(
                "provider process startup failed: %s: %s"
                % (failure_type, failure_message)
            )
        self._worker_process_id = startup.worker_process_id

    @property
    def parent_process_id(self) -> int:
        return self._parent_process_id

    @property
    def worker_process_id(self) -> int:
        return self._worker_process_id

    def submit(
        self,
        observation: Any,
        *,
        observation_sequence_id: int,
        captured_at_s: float,
        submitted_at_s: float | None = None,
    ) -> IndependentClockSubmission:
        if observation_sequence_id < 0:
            raise ValueError("observation_sequence_id must be nonnegative")
        if not math.isfinite(float(captured_at_s)):
            raise ValueError("captured_at_s must be finite")
        submitted = (
            _finite_monotonic() if submitted_at_s is None else float(submitted_at_s)
        )
        if not math.isfinite(submitted):
            raise ValueError("submitted_at_s must be finite")
        # Fail before queueing if the environment object cannot cross spawn.
        try:
            pickle.dumps(observation)
        except Exception as error:
            raise TypeError("observation must be pickle-safe") from error
        with self._lock:
            if self._closed:
                raise RuntimeError("provider worker is closed")
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
                    self._superseded += 1
            request = _ProcessRequest(
                request_id=request_id,
                observation=observation,
                observation_sequence_id=int(observation_sequence_id),
                captured_at_s=float(captured_at_s),
                submitted_at_s=submitted,
            )
            try:
                self._request_queue.put(request, timeout=0.5)
            except Full as error:
                raise RuntimeError("provider request mailbox remained full") from error
            self._submitted += 1
        return IndependentClockSubmission(
            request_id=request_id,
            observation_sequence_id=int(observation_sequence_id),
            submitted_at_s=submitted,
            replaced_request_id=replaced,
            parent_process_id=self._parent_process_id,
            worker_process_id=self._worker_process_id,
        )

    def drain(self) -> tuple[_WorkerMessage, ...]:
        messages: list[_WorkerMessage] = []
        while True:
            try:
                message = self._message_queue.get_nowait()
            except Empty:
                break
            if isinstance(message, _WorkerMessage):
                messages.append(message)
            else:
                self._queue_dropped += 1
        return tuple(messages)

    def metrics(self) -> dict[str, object]:
        return {
            "submitted": self._submitted,
            "superseded": self._superseded,
            "queue_dropped": self._queue_dropped + self._dropped_messages.value,
            "worker_process_id": self._worker_process_id,
            "parent_process_id": self._parent_process_id,
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
            self._stop_event.set()
            try:
                self._request_queue.put_nowait(None)
            except Full:
                # The child sees stop_event after finishing its in-flight call.
                pass
        self._process.join(timeout_s)
        stopped_cleanly = not self._process.is_alive()
        if not stopped_cleanly:
            self._process.terminate()
            self._process.join(min(1.0, max(0.1, timeout_s)))
        self._request_queue.close()
        self._message_queue.close()
        self._startup_queue.close()
        return stopped_cleanly

    def __enter__(self) -> "IndependentClockWorker":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


@dataclass
class RequestLifecycle:
    """Auditable lifecycle record for one submitted observation."""

    request_id: int
    observation_sequence_id: int
    captured_at_s: float
    submitted_at_s: float
    parent_process_id: int
    worker_process_id: int
    started_at_s: float | None = None
    completed_at_s: float | None = None
    superseded_at_s: float | None = None
    response_age_ms: float | None = None
    response_status: str = "submitted"
    source: str | None = None
    failure_type: str | None = None
    failure_message: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "observation_sequence_id": self.observation_sequence_id,
            "captured_at_s": self.captured_at_s,
            "submitted_at_s": self.submitted_at_s,
            "started_at_s": self.started_at_s,
            "completed_at_s": self.completed_at_s,
            "superseded_at_s": self.superseded_at_s,
            "response_age_ms": self.response_age_ms,
            "response_status": self.response_status,
            "source": self.source,
            "failure_type": self.failure_type,
            "failure_message": self.failure_message,
            "parent_process_id": self.parent_process_id,
            "worker_process_id": self.worker_process_id,
        }


@dataclass(frozen=True)
class ControlTickRecord:
    """One parent control tick, including deadline and suffix disposition."""

    tick_index: int
    scheduled_at_s: float
    tick_started_at_s: float
    observation_sequence_id: int
    submitted_request_id: int | None
    response_request_id: int | None
    response_age_ms: float | None
    deadline_ms: float
    status: str
    reason: str
    stale_prefix_steps: int
    stale_suffix_steps: int
    action_index: int | None
    action: tuple[float, ...]
    environment_done: bool
    parent_process_id: int
    worker_process_id: int

    def to_dict(self) -> dict[str, object]:
        return {
            "tick_index": self.tick_index,
            "scheduled_at_s": self.scheduled_at_s,
            "tick_started_at_s": self.tick_started_at_s,
            "observation_sequence_id": self.observation_sequence_id,
            "submitted_request_id": self.submitted_request_id,
            "response_request_id": self.response_request_id,
            "response_age_ms": self.response_age_ms,
            "deadline_ms": self.deadline_ms,
            "status": self.status,
            "reason": self.reason,
            "stale_prefix_steps": self.stale_prefix_steps,
            "stale_suffix_steps": self.stale_suffix_steps,
            # Alias used by the existing LIBERO artifact vocabulary.
            "available_suffix_steps": self.stale_suffix_steps,
            "action_index": self.action_index,
            "action": list(self.action),
            "environment_done": self.environment_done,
            "parent_process_id": self.parent_process_id,
            "worker_process_id": self.worker_process_id,
        }


@dataclass(frozen=True)
class IndependentClockResult:
    """Immutable snapshot returned by :func:`run_independent_clock`."""

    schema_version: str
    termination_reason: str
    parent_process_id: int
    worker_process_id: int
    worker_stopped: bool
    environment_steps: int
    ticks: tuple[ControlTickRecord, ...]
    requests: tuple[RequestLifecycle, ...]
    worker_metrics: Mapping[str, object]
    environment_done: bool
    tick_overruns: int

    @property
    def submitted(self) -> int:
        return len(self.requests)

    @property
    def started(self) -> int:
        return sum(record.started_at_s is not None for record in self.requests)

    @property
    def completed(self) -> int:
        return sum(record.completed_at_s is not None for record in self.requests)

    @property
    def superseded(self) -> int:
        return sum(record.superseded_at_s is not None for record in self.requests)

    @property
    def holds(self) -> int:
        return sum(record.status == "hold" for record in self.ticks)

    @property
    def executes(self) -> int:
        return sum(record.status == "execute" for record in self.ticks)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "termination_reason": self.termination_reason,
            "parent_process_id": self.parent_process_id,
            "worker_process_id": self.worker_process_id,
            "worker_stopped": self.worker_stopped,
            "environment_steps": self.environment_steps,
            "environment_done": self.environment_done,
            "tick_overruns": self.tick_overruns,
            "metrics": {
                "submitted": self.submitted,
                "started": self.started,
                "completed": self.completed,
                "superseded": self.superseded,
                "holds": self.holds,
                "executes": self.executes,
            },
            "worker": dict(self.worker_metrics),
            "requests": [record.to_dict() for record in self.requests],
            "ticks": [record.to_dict() for record in self.ticks],
        }

    @property
    def passed(self) -> bool:
        """Whether the scheduler smoke invariants (not task success) hold."""

        return bool(
            self.worker_stopped
            and self.parent_process_id != self.worker_process_id
            and self.environment_steps == len(self.ticks)
            and self.submitted > 0
            and self.started > 0
        )


@dataclass
class _ActiveResponse:
    request_id: int
    captured_at_s: float
    completed_at_s: float
    actions: np.ndarray
    source: str


def _stale_prefix_steps(age_s: float, config: IndependentClockConfig) -> int:
    if age_s <= config.boundary_tolerance_s:
        return 0
    return max(
        0,
        int(
            math.ceil(
                (age_s - config.boundary_tolerance_s) / config.action_period_s
            )
        ),
    )


def _parse_step_result(result: Any) -> tuple[bool, Any]:
    if not isinstance(result, tuple):
        raise TypeError("environment.step must return a tuple")
    if len(result) == 4:
        observation, _reward, done, _info = result
    elif len(result) == 5:
        observation, _reward, terminated, truncated, _info = result
        done = bool(terminated) or bool(truncated)
    else:
        raise ValueError("environment.step must return a 4- or 5-tuple")
    return bool(done), observation


def run_independent_clock(
    environment: IndependentClockEnvironment,
    provider_factory: IndependentClockProviderFactory,
    *,
    config: IndependentClockConfig = IndependentClockConfig(),
    observation_builder: Callable[[Any, int, float], Any] | None = None,
    action_adapter: Callable[[np.ndarray], Any] | None = None,
    hold_action: Sequence[float] | np.ndarray | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> IndependentClockResult:
    """Run a parent-ticking environment against an asynchronous provider.

    ``observation_builder(raw, sequence_id, captured_at_s)`` is called in the
    parent and should return the provider payload.  ``action_adapter`` maps a
    selected vector to the environment's native action type; by default the
    NumPy vector is passed through.  The injected clock is intended for
    instrumentation only; it must be a monotonic wall clock because child
    timestamps come from :func:`time.monotonic`.
    """

    if not isinstance(config, IndependentClockConfig):
        raise TypeError("config must be an IndependentClockConfig")
    if not callable(getattr(environment, "reset", None)):
        raise TypeError("environment must provide reset()")
    if not callable(getattr(environment, "observe", None)):
        raise TypeError("environment must provide observe()")
    if not callable(getattr(environment, "step", None)):
        raise TypeError("environment must provide step(action)")
    if observation_builder is None:
        def observation_builder(raw: Any, _sequence: int, _captured: float) -> Any:
            return raw

    if action_adapter is None:
        def action_adapter(action: np.ndarray) -> Any:
            return action

    if hold_action is None:
        hold = np.zeros(config.action_dim, dtype=np.float64)
    else:
        hold = np.asarray(hold_action, dtype=np.float64)
        if (
            hold.shape != (config.action_dim,)
            or not np.all(np.isfinite(hold))
        ):
            raise ValueError(
                "hold_action must be finite with shape (%d,)" % config.action_dim
            )
        hold = hold.copy()

    parent_pid = os.getpid()
    worker = IndependentClockWorker(
        provider_factory,
        action_dim=config.action_dim,
        startup_timeout_s=config.startup_timeout_s,
    )
    requests: dict[int, RequestLifecycle] = {}
    ticks: list[ControlTickRecord] = []
    active: _ActiveResponse | None = None
    latest_response_id = -1
    last_response_id: int | None = None
    last_response_age_ms: float | None = None
    last_stale_prefix = 0
    last_stale_suffix = 0
    hold_reason = "no_policy_response"
    done = False
    tick_overruns = 0
    termination_reason = "max_ticks"
    sequence_id = 0
    next_tick = float(clock())
    if not math.isfinite(next_tick):
        worker.close(timeout_s=config.shutdown_timeout_s)
        raise ValueError("clock returned a non-finite value")

    def process_messages(now_s: float) -> None:
        nonlocal active, latest_response_id, last_response_id
        nonlocal last_response_age_ms, last_stale_prefix, last_stale_suffix
        nonlocal hold_reason
        for message in worker.drain():
            record = requests.get(message.request_id)
            if record is None:
                # A message that arrived after an aborted submission is still
                # useful evidence; retain a synthetic request row.
                record = RequestLifecycle(
                    request_id=message.request_id,
                    observation_sequence_id=message.observation_sequence_id,
                    captured_at_s=message.started_at_s or message.at_s,
                    submitted_at_s=message.started_at_s or message.at_s,
                    parent_process_id=parent_pid,
                    worker_process_id=message.worker_process_id,
                )
                requests[message.request_id] = record
            if message.kind == "started":
                if record.started_at_s is None:
                    record.started_at_s = message.started_at_s or message.at_s
                    record.worker_process_id = message.worker_process_id
                continue
            if message.kind != "completed":
                continue
            record.started_at_s = message.started_at_s or record.started_at_s
            record.completed_at_s = message.completed_at_s or message.at_s
            record.worker_process_id = message.worker_process_id
            record.source = message.source
            record.failure_type = message.failure_type
            record.failure_message = message.failure_message
            completed_at = record.completed_at_s
            if completed_at is None:
                continue
            record.response_age_ms = max(
                0.0, (completed_at - record.captured_at_s) * 1000.0
            )
            if record.superseded_at_s is not None:
                record.response_status = "completed_after_superseded"
                continue
            if message.request_id <= latest_response_id:
                record.response_status = "stale_response"
                continue
            latest_response_id = message.request_id
            last_response_id = message.request_id
            last_response_age_ms = max(
                0.0, (now_s - record.captured_at_s) * 1000.0
            )
            last_stale_prefix = _stale_prefix_steps(
                max(0.0, now_s - record.captured_at_s), config
            )
            if message.failure_type is not None or message.actions is None:
                record.response_status = "failed"
                last_stale_suffix = 0
                active = None
                hold_reason = "policy_failure"
                continue
            age_s = max(0.0, (now_s - record.captured_at_s))
            prefix = _stale_prefix_steps(age_s, config)
            suffix = int(message.actions.shape[0]) - prefix
            last_stale_prefix = prefix
            last_stale_suffix = max(0, suffix)
            if age_s > config.deadline_s + config.boundary_tolerance_s:
                record.response_status = "deadline_exceeded"
                active = None
                hold_reason = "deadline_exceeded"
            elif suffix <= 0:
                record.response_status = "stale_suffix_exhausted"
                active = None
                hold_reason = "stale_suffix_exhausted"
            else:
                record.response_status = "accepted"
                active = _ActiveResponse(
                    request_id=message.request_id,
                    captured_at_s=record.captured_at_s,
                    completed_at_s=completed_at,
                    actions=message.actions,
                    source=message.source or "provider",
                )
                hold_reason = ""

    try:
        environment.reset()
        for tick_index in range(config.max_ticks):
            now = float(clock())
            if not math.isfinite(now):
                raise ValueError("clock returned a non-finite value")
            if now < next_tick:
                sleeper(next_tick - now)
                now = float(clock())
            if now > next_tick + config.control_period_s:
                tick_overruns += 1
            tick_started = now
            process_messages(now)
            raw_observation = environment.observe()
            captured_at = float(clock())
            if not math.isfinite(captured_at):
                raise ValueError("clock returned a non-finite observation time")
            provider_observation = observation_builder(
                raw_observation, sequence_id, captured_at
            )
            submitted_id: int | None = None
            if tick_index % config.submit_every_ticks == 0:
                submission = worker.submit(
                    provider_observation,
                    observation_sequence_id=sequence_id,
                    captured_at_s=captured_at,
                    submitted_at_s=captured_at,
                )
                submitted_id = submission.request_id
                requests[submission.request_id] = RequestLifecycle(
                    request_id=submission.request_id,
                    observation_sequence_id=sequence_id,
                    captured_at_s=captured_at,
                    submitted_at_s=submission.submitted_at_s,
                    parent_process_id=parent_pid,
                    worker_process_id=submission.worker_process_id,
                )
                if submission.replaced_request_id is not None:
                    replaced = requests.get(submission.replaced_request_id)
                    if replaced is not None:
                        replaced.superseded_at_s = captured_at
                        replaced.response_status = "superseded"
            # A very fast provider may have completed between the first drain
            # and the submission.  Drain once more before selecting a command.
            decision_now = float(clock())
            process_messages(decision_now)

            status = "hold"
            reason = hold_reason or "no_policy_response"
            response_id: int | None = (
                last_response_id if hold_reason != "no_policy_response" else None
            )
            response_age_ms: float | None = (
                last_response_age_ms if response_id is not None else None
            )
            stale_prefix = last_stale_prefix if response_id is not None else 0
            stale_suffix = last_stale_suffix if response_id is not None else 0
            action_index: int | None = None
            action = hold
            if active is not None:
                response_id = active.request_id
                age_s = max(0.0, decision_now - active.captured_at_s)
                response_age_ms = age_s * 1000.0
                stale_prefix = _stale_prefix_steps(age_s, config)
                stale_suffix = max(0, int(active.actions.shape[0]) - stale_prefix)
                last_response_age_ms = response_age_ms
                last_stale_prefix = stale_prefix
                last_stale_suffix = stale_suffix
                if age_s > config.deadline_s + config.boundary_tolerance_s:
                    active = None
                    hold_reason = "deadline_exceeded"
                    reason = hold_reason
                    action_index = None
                elif stale_suffix <= 0:
                    active = None
                    hold_reason = "stale_suffix_exhausted"
                    reason = hold_reason
                    action_index = stale_prefix
                else:
                    action_index = min(stale_prefix, int(active.actions.shape[0]) - 1)
                    action = np.asarray(active.actions[action_index], dtype=np.float64)
                    status = "execute"
                    reason = "fresh_suffix_available"
            if status == "hold":
                action = hold
            action_for_env = action_adapter(np.asarray(action, dtype=np.float64).copy())
            step_result = environment.step(action_for_env)
            done, _next_raw = _parse_step_result(step_result)
            ticks.append(
                ControlTickRecord(
                    tick_index=tick_index,
                    scheduled_at_s=next_tick,
                    tick_started_at_s=tick_started,
                    observation_sequence_id=sequence_id,
                    submitted_request_id=submitted_id,
                    response_request_id=response_id,
                    response_age_ms=response_age_ms,
                    deadline_ms=config.deadline_s * 1000.0,
                    status=status,
                    reason=reason,
                    stale_prefix_steps=stale_prefix,
                    stale_suffix_steps=stale_suffix,
                    action_index=action_index,
                    action=tuple(float(value) for value in action),
                    environment_done=done,
                    parent_process_id=parent_pid,
                    worker_process_id=worker.worker_process_id,
                )
            )
            sequence_id += 1
            if done:
                termination_reason = "environment_done"
                break
            next_tick += config.control_period_s
            # If a tick overran, discard missed periods instead of stepping the
            # environment repeatedly in a blocking catch-up loop.
            current = float(clock())
            if current > next_tick:
                next_tick = current
        else:
            termination_reason = "max_ticks"
    except Exception:
        termination_reason = "runtime_error"
        raise
    finally:
        # Capture events that reached the parent just after the final control
        # tick.  We do not wait for a new inference here: an unfinished call is
        # deliberately left as ``submitted``/``started`` in the artifact.
        try:
            process_messages(float(clock()))
        except Exception:
            # Preserve the original episode outcome if an injected diagnostic
            # clock fails during shutdown.
            pass
        # ``close`` closes the multiprocessing queues, so all messages that
        # can affect a control decision must have been drained during the
        # parent loop.  A final close intentionally leaves an in-flight request
        # auditable as ``submitted``/``started`` rather than inventing a
        # completion timestamp after the control clock has stopped.
        worker_stopped = worker.close(timeout_s=config.shutdown_timeout_s)

    return IndependentClockResult(
        schema_version="armbench.independent_clock.v1",
        termination_reason=termination_reason,
        parent_process_id=parent_pid,
        worker_process_id=worker.worker_process_id,
        worker_stopped=worker_stopped,
        environment_steps=len(ticks),
        ticks=tuple(ticks),
        requests=tuple(requests[index] for index in sorted(requests)),
        worker_metrics=worker.metrics(),
        environment_done=done,
        tick_overruns=tick_overruns,
    )


class _FakeClockEnvironment:
    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.step_count = 0
        self.actions: list[np.ndarray] = []

    def reset(self) -> None:
        self.step_count = 0
        self.actions.clear()

    def observe(self) -> dict[str, int]:
        return {"step": self.step_count}

    def step(self, action: np.ndarray) -> tuple[dict[str, int], float, bool, dict[str, object]]:
        self.actions.append(np.asarray(action, dtype=np.float64).copy())
        self.step_count += 1
        done = self.step_count >= self.max_steps
        return self.observe(), 0.0, done, {}


@dataclass(frozen=True)
class _FakeProviderFactory:
    latency_s: float
    action_dim: int
    horizon: int

    def __post_init__(self) -> None:
        if self.latency_s < 0.0 or not math.isfinite(self.latency_s):
            raise ValueError("fake provider latency must be finite and nonnegative")
        if self.action_dim <= 0 or self.horizon <= 0:
            raise ValueError("fake provider dimensions must be positive")

    def __call__(self) -> "_FakeProvider":
        return _FakeProvider(self.latency_s, self.action_dim, self.horizon)


class _FakeProvider:
    def __init__(self, latency_s: float, action_dim: int, horizon: int) -> None:
        self.latency_s = latency_s
        self.action_dim = action_dim
        self.horizon = horizon

    def infer(self, observation: Mapping[str, object]) -> Mapping[str, object]:
        time.sleep(self.latency_s)
        sequence_id = int(observation["sequence_id"])
        actions = np.zeros((self.horizon, self.action_dim), dtype=np.float64)
        actions[:, 0] = float(sequence_id)
        return {"actions": actions, "source": "cpu_fake_provider"}


def run_independent_clock_smoke(
    *,
    policy_latency_ms: float = 40.0,
    control_period_ms: float = 5.0,
    action_period_ms: float = 20.0,
    deadline_ms: float = 120.0,
    max_ticks: int = 20,
    action_dim: int = 1,
) -> dict[str, object]:
    """Run a CPU-only fake environment/provider smoke and return JSON data."""

    timing = np.asarray(
        [
            policy_latency_ms,
            control_period_ms,
            action_period_ms,
            deadline_ms,
        ],
        dtype=float,
    )
    if (
        not np.all(np.isfinite(timing))
        or policy_latency_ms < 0.0
        or control_period_ms <= 0.0
        or action_period_ms <= 0.0
        or deadline_ms < 0.0
        or max_ticks <= 0
        or action_dim <= 0
    ):
        raise ValueError("independent-clock smoke timing values are invalid")
    config = IndependentClockConfig(
        control_period_s=control_period_ms / 1000.0,
        action_period_s=action_period_ms / 1000.0,
        deadline_s=deadline_ms / 1000.0,
        max_ticks=max_ticks,
        action_dim=action_dim,
    )
    environment = _FakeClockEnvironment(max_ticks)

    def build_observation(raw: Mapping[str, int], sequence: int, captured: float) -> dict[str, object]:
        del captured
        return {"step": int(raw["step"]), "sequence_id": sequence}

    result = run_independent_clock(
        environment,
        _FakeProviderFactory(
            policy_latency_ms / 1000.0,
            action_dim,
            max(4, int(math.ceil(deadline_ms / max(action_period_ms, 1e-9))) + 2),
        ),
        config=config,
        observation_builder=build_observation,
    )
    report = result.to_dict()
    report["passed"] = result.passed
    report["scope"] = "cpu_fake_environment_and_spawned_provider"
    report["fake_environment"] = {
        "steps": environment.step_count,
        "executed_actions": len(environment.actions),
    }
    return report


__all__ = [
    "ControlTickRecord",
    "IndependentClockConfig",
    "IndependentClockEnvironment",
    "IndependentClockProviderFactory",
    "IndependentClockResult",
    "IndependentClockSubmission",
    "IndependentClockWorker",
    "RequestLifecycle",
    "run_independent_clock",
    "run_independent_clock_smoke",
]
