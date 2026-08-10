from __future__ import annotations

import time

import numpy as np

from armbench.mujoco_sim.broad_phase_continuous_collision import (
    BroadPhaseContinuousMuJoCoCollisionChecker,
)
from armbench.mujoco_sim.model import MuJoCoPanda
from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.vla.integrated_panda_async import (
    AtomicPandaPlanGate,
    LatestIntegratedPandaWorker,
)
from armbench.vla.integrated_panda_guard import IntegratedPandaGuardConfig
from armbench.vla.optimized_integrated_panda_guard import (
    OptimizedIntegratedPandaSupervisor,
)
from armbench.vla.types import ActionChunk


def _supervisor() -> tuple[
    MuJoCoPanda, OptimizedIntegratedPandaSupervisor
]:
    robot = MuJoCoPanda.create(obstacles=())
    checker = BroadPhaseContinuousMuJoCoCollisionChecker(robot)
    supervisor = OptimizedIntegratedPandaSupervisor(
        robot,
        checker,
        IntegratedPandaGuardConfig(
            response_deadline_ms=5000.0,
            supervision_budget_ms=5000.0,
            qp_step_budget_ms=500.0,
        ),
    )
    return robot, supervisor


def _chunk() -> ActionChunk:
    actions = np.zeros((3, 8), dtype=float)
    actions[:, :7] = np.array(
        [0.08, -0.04, 0.03, 0.0, 0.02, 0.0, -0.01]
    )
    actions[:, 7] = 0.5
    return ActionChunk(
        actions=actions,
        source="optimized_integrated_guard_test",
        observation_sequence_id=7,
        inference_latency_ms=0.0,
    )


def test_optimized_supervisor_accepts_complete_safe_chunk() -> None:
    _, supervisor = _supervisor()
    start = mujoco_scenarios()["free_space"].start

    decision = supervisor.supervise(
        start, np.zeros(7), _chunk(), observed_q=start
    )

    assert decision.status == "accepted"
    assert len(decision.executable_actions) == 3
    assert len(decision.edge_certificates) == 3
    assert len(decision.braking_certificates) == 3
    assert all(item.validated for item in decision.braking_certificates)
    metrics = supervisor.optimization_metrics()
    assert metrics["persistent_qp_solve_calls"] == 3
    assert metrics["persistent_brake_validation_calls"] == 3
    assert metrics["whole_stop_edges_checked"] == 3
    assert metrics["broad_phase_pruned_pairs"] > 0


def test_optimized_supervisor_deadline_fails_closed_without_policy_prefix() -> None:
    _, supervisor = _supervisor()
    start = mujoco_scenarios()["free_space"].start

    decision = supervisor.supervise(
        start,
        np.zeros(7),
        _chunk(),
        observed_q=start,
        response_age_ms=6000.0,
    )

    assert decision.status == "hold"
    assert decision.failure_stage == "deadline"
    assert len(decision.executable_actions) == 0
    assert decision.fallback_brake is not None
    assert decision.fallback_brake.validated is True


def test_optimized_supervisor_is_compatible_with_async_atomic_gate() -> None:
    _, supervisor = _supervisor()
    start = mujoco_scenarios()["free_space"].start
    gate = AtomicPandaPlanGate(supervisor)
    with LatestIntegratedPandaWorker(supervisor) as worker:
        submission = worker.submit(
            generation=gate.generation,
            observation_sequence_id=7,
            q=start,
            qvel=np.zeros(7),
            observed_q=start,
            response_age_ms=0.0,
            chunk=_chunk(),
        )
        outcomes = ()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            outcomes = worker.drain()
            if outcomes:
                break
            time.sleep(0.001)
    assert submission.request_id == 0
    assert len(outcomes) == 1

    atomic = gate.commit(outcomes[0], q_now=start)

    assert atomic.status == "execute"
    assert len(atomic.policy_actions) == 3
