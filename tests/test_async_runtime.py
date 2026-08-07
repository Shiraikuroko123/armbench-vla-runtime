from __future__ import annotations

import threading
import time

import numpy as np

from armbench.vla.async_runtime import (
    AsyncChunkDispatcher,
    AsyncDispatchConfig,
    LatestPolicyWorker,
    PolicyOutcome,
    run_async_runtime_smoke,
)
from armbench.vla.types import ActionChunk, VLAObservation


def _observation(sequence_id: int, captured_at_s: float | None = None) -> VLAObservation:
    return VLAObservation(
        exterior_image=np.zeros((224, 224, 3), dtype=np.uint8),
        wrist_image=np.zeros((224, 224, 3), dtype=np.uint8),
        joint_position=np.zeros(7),
        gripper_position=np.array([1.0]),
        prompt="test",
        sequence_id=sequence_id,
        captured_at_s=time.monotonic() if captured_at_s is None else captured_at_s,
    )


def _chunk(observation: VLAObservation, received_at_s: float) -> ActionChunk:
    actions = np.zeros((15, 8), dtype=float)
    actions[:, 0] = np.arange(15)
    actions[:, 7] = 1.0
    return ActionChunk(
        actions=actions,
        source="test",
        observation_sequence_id=observation.sequence_id,
        inference_latency_ms=max(
            0.0, (received_at_s - observation.captured_at_s) * 1000.0
        ),
        received_at_s=received_at_s,
    )


def _outcome(
    request_id: int,
    observation: VLAObservation,
    finished_at_s: float,
    *,
    failed: bool = False,
) -> PolicyOutcome:
    return PolicyOutcome(
        request_id=request_id,
        observation=observation,
        submitted_at_s=observation.captured_at_s,
        started_at_s=observation.captured_at_s,
        finished_at_s=finished_at_s,
        worker_thread_id=999,
        chunk=None if failed else _chunk(observation, finished_at_s),
        failure_type="RuntimeError" if failed else None,
        failure_message="policy failed" if failed else None,
    )


class _BlockingPolicy:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls: list[int] = []

    def infer(self, observation: VLAObservation) -> ActionChunk:
        self.calls.append(observation.sequence_id)
        if len(self.calls) == 1:
            self.started.set()
            if not self.release.wait(2.0):
                raise TimeoutError("test did not release policy")
        return _chunk(observation, time.monotonic())


class _FailingPolicy:
    def infer(self, observation: VLAObservation) -> ActionChunk:
        del observation
        raise ConnectionError("scripted transport failure")


def _collect(worker: LatestPolicyWorker, count: int) -> list[PolicyOutcome]:
    deadline = time.monotonic() + 2.0
    outcomes: list[PolicyOutcome] = []
    while len(outcomes) < count and time.monotonic() < deadline:
        outcomes.extend(worker.drain())
        if len(outcomes) < count:
            time.sleep(0.005)
    return outcomes


def test_blocking_policy_does_not_block_control_side_polling() -> None:
    policy = _BlockingPolicy()
    worker = LatestPolicyWorker(policy)
    try:
        worker.submit(_observation(0))
        assert policy.started.wait(1.0)
        for _ in range(20):
            assert worker.drain() == ()
        assert worker.metrics()["worker_alive"]
        policy.release.set()
        outcomes = _collect(worker, 1)
    finally:
        policy.release.set()
        assert worker.close()

    assert len(outcomes) == 1
    assert outcomes[0].succeeded
    assert outcomes[0].worker_thread_id != threading.get_ident()


def test_latest_pending_observation_replaces_older_pending_request() -> None:
    policy = _BlockingPolicy()
    worker = LatestPolicyWorker(policy)
    try:
        worker.submit(_observation(0))
        assert policy.started.wait(1.0)
        second = worker.submit(_observation(1))
        third = worker.submit(_observation(2))
        assert second.replaced_request_id is None
        assert third.replaced_request_id == second.request_id
        policy.release.set()
        outcomes = _collect(worker, 2)
    finally:
        policy.release.set()
        assert worker.close()

    assert policy.calls == [0, 2]
    assert [outcome.observation.sequence_id for outcome in outcomes] == [0, 2]
    assert worker.metrics()["superseded_pending"] == 1


def test_worker_converts_policy_exception_to_auditable_outcome() -> None:
    worker = LatestPolicyWorker(_FailingPolicy())
    try:
        worker.submit(_observation(8))
        outcomes = _collect(worker, 1)
    finally:
        assert worker.close()

    assert len(outcomes) == 1
    assert not outcomes[0].succeeded
    assert outcomes[0].failure_type == "ConnectionError"
    assert outcomes[0].failure_message == "scripted transport failure"
    assert worker.metrics()["failed"] == 1


def test_dispatcher_tracks_measured_age_and_deadline() -> None:
    observation = _observation(4, captured_at_s=10.0)
    outcome = _outcome(7, observation, finished_at_s=10.08)
    dispatcher = AsyncChunkDispatcher(
        AsyncDispatchConfig(action_period_s=0.05, deadline_s=0.2)
    )
    hold = np.zeros(8)

    update = dispatcher.publish(outcome, now_s=10.08)
    at_response = dispatcher.select(hold, now_s=10.08)
    at_boundary = dispatcher.select(hold, now_s=10.10)
    after_boundary = dispatcher.select(hold, now_s=10.1001)
    expired = dispatcher.select(hold, now_s=10.201)

    assert update.status == "accepted"
    assert update.action_offset == 2
    assert at_response.action_index == 2
    assert at_response.action[0] == 2.0
    assert at_boundary.action_index == 2
    assert after_boundary.action_index == 3
    assert expired.holding
    assert expired.reason == "deadline_exceeded"


def test_newer_policy_failure_clears_active_chunk_fail_closed() -> None:
    dispatcher = AsyncChunkDispatcher(
        AsyncDispatchConfig(action_period_s=0.05, deadline_s=0.5)
    )
    first_observation = _observation(1, captured_at_s=20.0)
    failed_observation = _observation(2, captured_at_s=20.1)
    dispatcher.publish(_outcome(1, first_observation, 20.01), now_s=20.01)

    update = dispatcher.publish(
        _outcome(2, failed_observation, 20.11, failed=True), now_s=20.11
    )
    decision = dispatcher.select(np.zeros(8), now_s=20.11)

    assert update.status == "rejected"
    assert update.reason == "policy_failure"
    assert decision.holding
    assert decision.reason == "policy_failure"
    assert decision.request_id == 2
    assert decision.observation_sequence_id == 2


def test_superseded_response_cannot_replace_newer_active_chunk() -> None:
    dispatcher = AsyncChunkDispatcher(
        AsyncDispatchConfig(action_period_s=0.05, deadline_s=0.5)
    )
    newer = _observation(2, captured_at_s=30.0)
    older = _observation(1, captured_at_s=30.0)
    dispatcher.publish(_outcome(2, newer, 30.01), now_s=30.01)

    update = dispatcher.publish(_outcome(1, older, 30.02), now_s=30.02)
    decision = dispatcher.select(np.zeros(8), now_s=30.02)

    assert update.reason == "superseded_response"
    assert decision.request_id == 2
    assert not decision.holding


def test_local_async_smoke_proves_separate_policy_and_control_threads() -> None:
    report = run_async_runtime_smoke(
        policy_latency_ms=40.0,
        control_period_ms=5.0,
        action_period_ms=50.0,
        deadline_ms=100.0,
    )

    assert report["passed"]
    assert report["scope"] == "scripted_threading_and_dispatch_only"
    assert report["control"]["ticks_during_inference"] >= 2
    assert report["policy"]["worker_thread_id"] != report["control"]["thread_id"]
    assert report["command"]["status"] == "execute"


def test_local_async_smoke_treats_deadline_rejection_as_expected_hold() -> None:
    report = run_async_runtime_smoke(
        policy_latency_ms=80.0,
        control_period_ms=5.0,
        action_period_ms=50.0,
        deadline_ms=20.0,
    )

    assert report["passed"]
    assert report["dispatch_update"]["reason"] == "deadline_exceeded"
    assert report["command"]["status"] == "hold"
    assert report["command"]["request_id"] == 0
