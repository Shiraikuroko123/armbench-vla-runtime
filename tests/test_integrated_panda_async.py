from __future__ import annotations

import threading
import time

import numpy as np

from armbench.mujoco_sim import MuJoCoPanda
from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.vla.integrated_panda_async import (
    AtomicPandaPlanGate,
    LatestIntegratedPandaWorker,
)
from armbench.vla.integrated_panda_guard import (
    IntegratedPandaGuardConfig,
    IntegratedPandaSupervisor,
)
from armbench.vla.integrated_panda_task import make_integrated_task_checker
from armbench.vla.types import ActionChunk


def _supervisor() -> tuple[IntegratedPandaSupervisor, np.ndarray]:
    scenario = mujoco_scenarios()["free_space"]
    robot = MuJoCoPanda.create(obstacles=())
    checker, _ = make_integrated_task_checker(robot)
    supervisor = IntegratedPandaSupervisor(
        robot,
        checker,
        IntegratedPandaGuardConfig(
            response_deadline_ms=20_000.0,
            supervision_budget_ms=20_000.0,
            qp_step_budget_ms=500.0,
        ),
    )
    return supervisor, scenario.start


def _chunk(sequence_id: int = 0) -> ActionChunk:
    actions = np.zeros((2, 8), dtype=float)
    actions[:, 0] = 0.02
    actions[:, 7] = 1.0
    return ActionChunk(
        actions=actions,
        source="scripted_nonlearned_async_assurance_test",
        observation_sequence_id=sequence_id,
        inference_latency_ms=0.0,
    )


def _await_outcome(
    worker: LatestIntegratedPandaWorker,
    *,
    timeout_s: float = 20.0,
) -> tuple[object, int]:
    deadline = time.monotonic() + timeout_s
    ticks = 0
    while time.monotonic() < deadline:
        outcomes = worker.drain()
        if outcomes:
            return outcomes[0], ticks
        ticks += 1
        time.sleep(0.001)
    raise TimeoutError("integrated Panda worker test timed out")


def test_integrated_supervision_does_not_block_control_thread() -> None:
    supervisor, start = _supervisor()
    gate = AtomicPandaPlanGate(supervisor)
    control_thread_id = threading.get_ident()
    worker = LatestIntegratedPandaWorker(supervisor)
    try:
        worker.submit(
            generation=gate.generation,
            observation_sequence_id=0,
            q=start,
            qvel=np.zeros(7),
            observed_q=start,
            response_age_ms=0.0,
            chunk=_chunk(),
        )
        outcome, ticks = _await_outcome(worker)
        decision = gate.commit(outcome, q_now=start)
    finally:
        assert worker.close()

    assert outcome.succeeded
    assert outcome.worker_thread_id != control_thread_id
    assert ticks > 0
    assert decision.status == "execute"
    assert decision.policy_actions_executable
    assert len(decision.policy_actions) == 2
    assert decision.fallback_validated


def test_reset_invalidates_an_inflight_complete_plan() -> None:
    supervisor, start = _supervisor()
    gate = AtomicPandaPlanGate(supervisor)
    worker = LatestIntegratedPandaWorker(supervisor)
    try:
        worker.submit(
            generation=gate.generation,
            observation_sequence_id=0,
            q=start,
            qvel=np.zeros(7),
            observed_q=start,
            response_age_ms=0.0,
            chunk=_chunk(),
        )
        outcome, _ = _await_outcome(worker)
        gate.reset()
        decision = gate.commit(outcome, q_now=start)
    finally:
        assert worker.close()

    assert decision.status == "hold"
    assert decision.reason == "reset_generation_mismatch"
    assert len(decision.policy_actions) == 0


def test_activation_rechecks_state_after_background_assurance() -> None:
    supervisor, start = _supervisor()
    gate = AtomicPandaPlanGate(supervisor)
    worker = LatestIntegratedPandaWorker(supervisor)
    try:
        worker.submit(
            generation=gate.generation,
            observation_sequence_id=0,
            q=start,
            qvel=np.zeros(7),
            observed_q=start,
            response_age_ms=0.0,
            chunk=_chunk(),
        )
        outcome, _ = _await_outcome(worker)
        moved = start.copy()
        moved[0] += supervisor.config.max_state_mismatch_rad + 0.01
        decision = gate.commit(outcome, q_now=moved)
    finally:
        assert worker.close()

    assert decision.status == "hold"
    assert decision.reason == "state_changed_during_assurance"
    assert len(decision.policy_actions) == 0


def test_activation_rechecks_deadline_after_background_assurance() -> None:
    supervisor, start = _supervisor()
    gate = AtomicPandaPlanGate(supervisor)
    worker = LatestIntegratedPandaWorker(supervisor)
    try:
        worker.submit(
            generation=gate.generation,
            observation_sequence_id=0,
            q=start,
            qvel=np.zeros(7),
            observed_q=start,
            response_age_ms=0.0,
            chunk=_chunk(),
        )
        outcome, _ = _await_outcome(worker)
        expired_at = (
            outcome.request.submitted_at_s
            + supervisor.config.response_deadline_ms / 1000.0
            + 0.001
        )
        decision = gate.commit(outcome, q_now=start, now_s=expired_at)
    finally:
        assert worker.close()

    assert outcome.succeeded
    assert decision.status == "hold"
    assert decision.reason == "response_deadline_exceeded_before_activation"
    assert len(decision.policy_actions) == 0


def test_committed_result_cannot_be_replayed() -> None:
    supervisor, start = _supervisor()
    gate = AtomicPandaPlanGate(supervisor)
    worker = LatestIntegratedPandaWorker(supervisor)
    try:
        worker.submit(
            generation=gate.generation,
            observation_sequence_id=0,
            q=start,
            qvel=np.zeros(7),
            observed_q=start,
            response_age_ms=0.0,
            chunk=_chunk(),
        )
        outcome, _ = _await_outcome(worker)
        assert gate.commit(outcome, q_now=start).status == "execute"
        replay = gate.commit(outcome, q_now=start)
    finally:
        assert worker.close()

    assert replay.status == "hold"
    assert replay.reason == "replayed_or_out_of_order_assurance_result"
    assert len(replay.policy_actions) == 0
