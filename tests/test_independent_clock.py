from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import time
from typing import Any

import numpy as np
import pytest

from armbench.vla.independent_clock import (
    AGE_ALIGNED_SUFFIX,
    IndependentClockConfig,
    IndependentClockWorker,
    RESPONSE_RELATIVE_CHUNK,
    run_independent_clock,
    run_independent_clock_smoke,
)


class _FakeEnvironment:
    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.steps = 0
        self.actions: list[np.ndarray] = []

    def reset(self) -> None:
        self.steps = 0
        self.actions.clear()

    def observe(self) -> dict[str, int]:
        return {"step": self.steps}

    def step(self, action: np.ndarray):
        self.actions.append(np.asarray(action, dtype=float).copy())
        self.steps += 1
        return self.observe(), 0.0, self.steps >= self.max_steps, {}


@dataclass(frozen=True)
class _DelayedFactory:
    latency_s: float = 0.03

    def __call__(self) -> "_DelayedProvider":
        return _DelayedProvider(self.latency_s)


class _DelayedProvider:
    def __init__(self, latency_s: float) -> None:
        self.latency_s = latency_s

    def infer(self, observation: dict[str, int]) -> dict[str, object]:
        time.sleep(self.latency_s)
        sequence = float(observation["sequence_id"])
        actions = np.zeros((8, 1), dtype=float)
        actions[:, 0] = sequence * 100.0 + np.arange(8, dtype=float)
        return {"actions": actions, "source": "test_delayed_provider"}


@dataclass(frozen=True)
class _BlockingFactory:
    entered: Any
    release: Any

    def __call__(self) -> "_BlockingProvider":
        return _BlockingProvider(self.entered, self.release)


class _BlockingProvider:
    def __init__(self, entered: Any, release: Any) -> None:
        self.entered = entered
        self.release = release

    def infer(self, observation: dict[str, int]) -> dict[str, object]:
        self.entered.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("test provider release was not signalled")
        sequence = float(observation["sequence_id"])
        return {
            "actions": np.full((8, 1), sequence, dtype=float),
            "source": "test_blocking_provider",
        }


@dataclass(frozen=True)
class _FailingFactory:
    def __call__(self) -> "_FailingProvider":
        return _FailingProvider()


class _FailingProvider:
    def infer(self, observation: dict[str, int]) -> dict[str, object]:
        del observation
        raise RuntimeError("scripted provider failure")


def _builder(raw: dict[str, int], sequence: int, captured_at_s: float):
    del captured_at_s
    return {"step": raw["step"], "sequence_id": sequence}


def _incrementing_hold(previous: np.ndarray) -> np.ndarray:
    return previous + 0.25


def test_spawned_worker_records_latest_only_supersession() -> None:
    context = mp.get_context("spawn")
    manager = context.Manager()
    entered = manager.Event()
    release = manager.Event()
    worker = IndependentClockWorker(_BlockingFactory(entered, release), action_dim=1)
    try:
        first = worker.submit(
            {"sequence_id": 0},
            observation_sequence_id=0,
            captured_at_s=time.monotonic(),
        )
        assert entered.wait(timeout=2.0)
        second = worker.submit(
            {"sequence_id": 1},
            observation_sequence_id=1,
            captured_at_s=time.monotonic(),
        )
        third = worker.submit(
            {"sequence_id": 2},
            observation_sequence_id=2,
            captured_at_s=time.monotonic(),
        )
        release.set()
        deadline = time.monotonic() + 2.0
        messages = []
        while time.monotonic() < deadline:
            messages.extend(worker.drain())
            if any(message.kind == "completed" for message in messages):
                break
            time.sleep(0.005)
        metrics = worker.metrics()
    finally:
        release.set()
        assert worker.close()
        manager.shutdown()

    completed = [message for message in messages if message.kind == "completed"]
    assert completed
    assert first.replaced_request_id is None
    assert second.replaced_request_id is None
    assert third.replaced_request_id == second.request_id
    assert metrics["superseded"] >= 1
    assert completed[0].worker_process_id != worker.parent_process_id


def test_independent_clock_parent_keeps_stepping_and_records_suffix() -> None:
    environment = _FakeEnvironment(max_steps=100)
    result = run_independent_clock(
        environment,
        _DelayedFactory(0.03),
        config=IndependentClockConfig(
            control_period_s=0.005,
            action_period_s=0.1,
            deadline_s=1.0,
            max_ticks=100,
            action_dim=1,
        ),
        observation_builder=_builder,
    )

    assert result.passed
    assert result.parent_process_id != result.worker_process_id
    assert result.environment_steps == 100
    assert result.superseded >= 1
    assert result.started >= 1
    assert result.completed >= 1
    assert result.holds >= 1
    assert result.executes >= 1
    executed = [tick for tick in result.ticks if tick.status == "execute"]
    assert executed
    assert all(tick.stale_suffix_steps > 0 for tick in executed)
    assert all(tick.response_age_ms is not None for tick in executed)
    assert all(tick.deadline_ms == 1000.0 for tick in result.ticks)
    assert len(environment.actions) == result.environment_steps
    completed = [request for request in result.requests if request.actions is not None]
    assert completed
    assert len(completed[0].actions) == 8


def test_response_relative_baseline_starts_at_chunk_head() -> None:
    aligned_environment = _FakeEnvironment(max_steps=30)
    aligned = run_independent_clock(
        aligned_environment,
        _DelayedFactory(0.02),
        config=IndependentClockConfig(
            control_period_s=0.01,
            action_period_s=0.50,
            deadline_s=2.0,
            max_ticks=100,
            action_dim=1,
            submit_every_ticks=100,
            action_selection_mode=AGE_ALIGNED_SUFFIX,
        ),
        observation_builder=_builder,
    )
    relative_environment = _FakeEnvironment(max_steps=30)
    relative = run_independent_clock(
        relative_environment,
        _DelayedFactory(0.02),
        config=IndependentClockConfig(
            control_period_s=0.01,
            action_period_s=0.50,
            deadline_s=2.0,
            max_ticks=100,
            action_dim=1,
            submit_every_ticks=100,
            action_selection_mode=RESPONSE_RELATIVE_CHUNK,
        ),
        observation_builder=_builder,
    )

    first_aligned = next(tick for tick in aligned.ticks if tick.status == "execute")
    first_relative = next(tick for tick in relative.ticks if tick.status == "execute")
    assert first_aligned.action_index is not None
    assert first_aligned.action_index >= 1
    assert first_aligned.reason == "fresh_suffix_available"
    assert first_relative.action_index == 0
    assert first_relative.reason == "response_relative_chunk_available"
    assert first_relative.stale_prefix_steps >= 1
    assert first_relative.action == (0.0,)


def test_independent_clock_rejects_unknown_selection_mode() -> None:
    with pytest.raises(ValueError, match="action_selection_mode"):
        IndependentClockConfig(action_selection_mode="unknown")


def test_independent_clock_dynamic_hold_receives_previous_command() -> None:
    environment = _FakeEnvironment(max_steps=4)
    result = run_independent_clock(
        environment,
        _DelayedFactory(0.5),
        config=IndependentClockConfig(
            control_period_s=0.005,
            action_period_s=0.01,
            deadline_s=1.0,
            max_ticks=4,
            action_dim=1,
        ),
        observation_builder=_builder,
        hold_action=_incrementing_hold,
    )

    assert result.holds == 4
    np.testing.assert_allclose(
        [action[0] for action in environment.actions],
        [0.25, 0.50, 0.75, 1.00],
    )
    assert [tick.action for tick in result.ticks] == [
        (0.25,),
        (0.50,),
        (0.75,),
        (1.00,),
    ]


def test_independent_clock_deadline_fails_closed() -> None:
    environment = _FakeEnvironment(max_steps=60)
    result = run_independent_clock(
        environment,
        _DelayedFactory(0.02),
        config=IndependentClockConfig(
            control_period_s=0.005,
            action_period_s=0.01,
            deadline_s=0.005,
            max_ticks=60,
            action_dim=1,
            submit_every_ticks=60,
        ),
        observation_builder=_builder,
    )

    assert result.passed
    assert result.holds == len(result.ticks)
    assert result.completed >= 1
    assert any(
        request.response_status == "deadline_exceeded" for request in result.requests
    )
    assert any(tick.reason == "deadline_exceeded" for tick in result.ticks)
    assert all(np.allclose(action, 0.0) for action in environment.actions)


def test_independent_clock_provider_failure_is_auditable() -> None:
    environment = _FakeEnvironment(max_steps=60)
    result = run_independent_clock(
        environment,
        _FailingFactory(),
        config=IndependentClockConfig(
            control_period_s=0.005,
            action_period_s=0.01,
            deadline_s=1.0,
            max_ticks=60,
            action_dim=1,
            submit_every_ticks=60,
        ),
        observation_builder=_builder,
    )

    assert result.passed
    assert result.completed >= 1
    failed = [record for record in result.requests if record.failure_type]
    assert failed
    assert failed[0].response_status == "failed"
    assert failed[0].failure_message == "scripted provider failure"
    assert all(tick.status == "hold" for tick in result.ticks)


def test_cpu_independent_clock_smoke_schema_and_scope() -> None:
    report = run_independent_clock_smoke(
        policy_latency_ms=25.0,
        control_period_ms=5.0,
        action_period_ms=10.0,
        deadline_ms=500.0,
        max_ticks=60,
    )

    assert report["passed"]
    assert report["scope"] == "cpu_fake_environment_and_spawned_provider"
    assert report["schema_version"] == "armbench.independent_clock.v1"
    metrics = report["metrics"]
    assert metrics["submitted"] >= metrics["started"] >= 1
    assert metrics["superseded"] >= 1
    assert metrics["holds"] >= 1
    assert metrics["executes"] >= 1
    assert report["worker"]["process_start_method"] == "spawn"
