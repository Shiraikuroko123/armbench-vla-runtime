from __future__ import annotations

import numpy as np
import pytest

from armbench.mujoco_sim.continuous_collision import (
    ContinuousCollisionConfig,
    ContinuousMuJoCoCollisionChecker,
)
from armbench.mujoco_sim.model import MuJoCoPanda
from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.vla.integrated_panda_guard import (
    IntegratedPandaGuardConfig,
    IntegratedPandaSupervisor,
    run_integrated_panda_guard_smoke,
)
from armbench.vla.types import ActionChunk


SELF_START = np.array(
    [
        2.013896901192687,
        0.8514937392438364,
        2.128626643260985,
        -0.7308358555657621,
        1.5703065944221684,
        1.5832426543430034,
        0.5959999237674181,
    ]
)
SELF_END = np.array(
    [
        2.044760389840342,
        -1.7175851740339965,
        2.558238709554596,
        -2.9380304930478514,
        -2.797519848117953,
        1.1846938393238287,
        -2.8554195536676996,
    ]
)


def _chunk(actions: np.ndarray) -> ActionChunk:
    return ActionChunk(
        actions=actions,
        source="scripted_nonlearned_integrated_guard_test",
        observation_sequence_id=0,
        inference_latency_ms=0.0,
        received_at_s=1.0,
    )


def _free_supervisor(
    config: IntegratedPandaGuardConfig | None = None,
) -> tuple[IntegratedPandaSupervisor, np.ndarray]:
    scenario = mujoco_scenarios()["free_space"]
    robot = MuJoCoPanda.create(obstacles=())
    checker = ContinuousMuJoCoCollisionChecker(
        robot,
        ContinuousCollisionConfig(
            include_static_obstacles=False,
            include_self_collision=True,
        ),
    )
    return IntegratedPandaSupervisor(
        robot,
        checker,
        config
        or IntegratedPandaGuardConfig(
            response_deadline_ms=5000.0,
            supervision_budget_ms=5000.0,
            qp_step_budget_ms=500.0,
        ),
    ), scenario.start


def test_integrated_smoke_accepts_only_complete_certified_plan() -> None:
    report = run_integrated_panda_guard_smoke()

    assert report["passed"] is True
    assert report["status"] == "accepted"
    assert report["continuous_all_safe"] is True
    assert report["braking_invariant_complete"] is True
    assert report["braking_boundaries_checked"] == 3


def test_expired_response_with_motion_returns_complete_verified_brake() -> None:
    supervisor, start = _free_supervisor()
    actions = np.zeros((2, 8), dtype=float)
    decision = supervisor.supervise(
        start,
        np.array([0.10, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0]),
        _chunk(actions),
        response_age_ms=5001.0,
    )

    assert decision.status == "verified_brake"
    assert decision.failure_stage == "deadline"
    assert decision.fallback_brake is not None
    assert decision.fallback_brake.validated
    assert len(decision.executable_actions) == 0
    np.testing.assert_allclose(decision.fallback_brake.velocities_rad_s[-1], 0.0)


def test_state_mismatch_is_rejected_before_qp_and_holds_when_stationary() -> None:
    supervisor, start = _free_supervisor()
    actions = np.zeros((2, 8), dtype=float)
    observed = start.copy()
    observed[0] += 0.2

    decision = supervisor.supervise(
        start,
        np.zeros(7),
        _chunk(actions),
        observed_q=observed,
    )

    assert decision.status == "hold"
    assert decision.failure_stage == "state_alignment"
    assert decision.qp_result is None
    assert decision.stage_latencies_ms.keys() == {"fallback_brake"}


def test_intermediate_self_collision_rejects_entire_projected_chunk() -> None:
    robot = MuJoCoPanda.create(obstacles=())
    checker = ContinuousMuJoCoCollisionChecker(robot)
    supervisor = IntegratedPandaSupervisor(
        robot,
        checker,
        IntegratedPandaGuardConfig(
            control_dt_s=1.0,
            response_deadline_ms=20_000.0,
            supervision_budget_ms=20_000.0,
            joint_velocity_limits_rad_s=(2.6,) * 7,
            joint_acceleration_limit_rad_s2=100.0,
            qp_step_budget_ms=500.0,
        ),
    )
    actions = np.zeros((2, 8), dtype=float)
    actions[:, :7] = (SELF_END - SELF_START) / 2.0
    actions[:, 7] = 0.5

    decision = supervisor.supervise(
        SELF_START,
        np.zeros(7),
        _chunk(actions),
        observed_q=SELF_START,
    )

    assert decision.status == "hold"
    assert decision.failure_stage == "continuous_collision"
    assert decision.qp_result is not None
    assert decision.qp_result.feasible
    assert len(decision.executable_actions) == 0
    assert any(not item.certified_safe for item in decision.edge_certificates)


def test_failed_fallback_never_exposes_partial_brake_or_policy_actions() -> None:
    supervisor, _ = _free_supervisor()
    q = supervisor.robot.upper_limits - 0.001
    qvel = np.zeros(7)
    qvel[0] = 0.2

    decision = supervisor.supervise(
        q,
        qvel,
        _chunk(np.zeros((2, 8))),
        response_age_ms=5001.0,
    )

    assert decision.status == "unrecoverable_stop"
    assert decision.fallback_brake is not None
    assert not decision.fallback_brake.validated
    assert decision.fallback_brake.failure_reason == "joint_position_limit:joint1"
    assert len(decision.executable_actions) == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"control_dt_s": 0.0},
        {"response_deadline_ms": True},
        {"supervision_budget_ms": float("nan")},
        {"joint_velocity_limits_rad_s": (1.0,) * 6},
        {"joint_velocity_limits_rad_s": ("1.0",) * 7},
    ],
)
def test_integrated_guard_rejects_invalid_config(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        IntegratedPandaGuardConfig(**kwargs)
