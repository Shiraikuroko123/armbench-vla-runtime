from __future__ import annotations

from dataclasses import dataclass
import os
import time

import numpy as np
import pytest

from armbench.vla.async_smoke import run_process_runtime_smoke
from armbench.vla.process_worker import ProcessPolicyWorker
from armbench.vla.types import ActionChunk, VLAObservation


def _observation(sequence_id: int) -> VLAObservation:
    return VLAObservation(
        exterior_image=np.zeros((224, 224, 3), dtype=np.uint8),
        wrist_image=np.zeros((224, 224, 3), dtype=np.uint8),
        joint_position=np.zeros(7),
        gripper_position=np.array([1.0]),
        prompt="process worker test",
        sequence_id=sequence_id,
        captured_at_s=time.monotonic(),
    )


class _Policy:
    def __init__(self, latency_s: float) -> None:
        self.latency_s = latency_s

    def infer(self, observation: VLAObservation) -> ActionChunk:
        started = time.monotonic()
        time.sleep(self.latency_s)
        received = time.monotonic()
        actions = np.zeros((5, 8), dtype=float)
        actions[:, 0] = observation.sequence_id
        actions[:, 7] = 1.0
        return ActionChunk(
            actions=actions,
            source="process_test",
            observation_sequence_id=observation.sequence_id,
            inference_latency_ms=(received - started) * 1000.0,
            received_at_s=received,
        )


@dataclass(frozen=True)
class _Factory:
    latency_s: float = 0.0

    def __call__(self) -> _Policy:
        return _Policy(self.latency_s)


@dataclass(frozen=True)
class _FailingFactory:
    def __call__(self) -> _Policy:
        raise ConnectionError("factory could not connect")


def _collect(worker: ProcessPolicyWorker, count: int) -> list[object]:
    deadline = time.monotonic() + 5.0
    outcomes: list[object] = []
    while len(outcomes) < count and time.monotonic() < deadline:
        outcomes.extend(worker.drain())
        if len(outcomes) < count:
            time.sleep(0.01)
    return outcomes


def test_process_worker_runs_in_a_different_process() -> None:
    worker = ProcessPolicyWorker(_Factory())
    try:
        worker.submit(_observation(4))
        outcomes = _collect(worker, 1)
        metrics = worker.metrics()
    finally:
        assert worker.close()

    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.succeeded
    assert outcome.worker_process_id != os.getpid()
    assert metrics["process_start_method"] == "spawn"


def test_process_worker_reports_factory_startup_failure() -> None:
    with pytest.raises(RuntimeError, match="factory could not connect"):
        ProcessPolicyWorker(_FailingFactory())


def test_process_smoke_keeps_parent_control_clock_ticking() -> None:
    report = run_process_runtime_smoke(
        policy_latency_ms=80.0,
        control_period_ms=5.0,
        action_period_ms=50.0,
        deadline_ms=200.0,
    )

    assert report["passed"]
    assert report["scope"] == "scripted_process_isolation_and_dispatch_only"
    assert report["control"]["ticks_during_inference"] >= 2
    assert report["policy"]["worker_process_id"] != report["control"]["process_id"]
    assert report["command"]["status"] == "execute"


def test_process_smoke_deadline_fails_closed() -> None:
    report = run_process_runtime_smoke(
        policy_latency_ms=80.0,
        control_period_ms=5.0,
        action_period_ms=50.0,
        deadline_ms=20.0,
    )

    assert report["passed"]
    assert report["dispatch_update"]["reason"] == "deadline_exceeded"
    assert report["command"]["status"] == "hold"
