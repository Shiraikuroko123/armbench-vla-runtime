from __future__ import annotations

import numpy as np
import pytest

from armbench.mujoco_sim import MuJoCoCollisionChecker, MuJoCoPanda
from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.vla.qp_projection import (
    QPActionProjector,
    QPLinearConstraint,
    QPProjectionConfig,
    run_qp_projection_smoke,
)
from armbench.vla.types import ActionChunk


def _chunk(actions: np.ndarray) -> ActionChunk:
    return ActionChunk(
        actions=actions,
        source="qp_test",
        observation_sequence_id=0,
        inference_latency_ms=0.0,
        received_at_s=1.0,
    )


def _projector(
    config: QPProjectionConfig = QPProjectionConfig(step_budget_ms=100.0),
    *,
    check_collision: bool = True,
) -> tuple[QPActionProjector, np.ndarray]:
    scenario = mujoco_scenarios()["free_space"]
    robot = MuJoCoPanda.create(obstacles=scenario.obstacles)
    checker = MuJoCoCollisionChecker(robot, resolution=0.02)
    return QPActionProjector(
        robot, checker if check_collision else None, config
    ), scenario.start


def test_qp_config_and_constraint_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        QPProjectionConfig(control_dt_s=0.0)
    with pytest.raises(ValueError, match="invalid"):
        QPLinearConstraint(
            matrix=np.ones((1, 6)),
            lower=np.array([0.0]),
            upper=np.array([1.0]),
            label="wrong width",
        )


def test_qp_projects_long_chunk_into_position_velocity_and_acceleration_bounds() -> None:
    config = QPProjectionConfig(
        control_dt_s=0.05,
        joint_velocity_limit_scale=0.5,
        joint_acceleration_limit_rad_s2=8.0,
        step_budget_ms=100.0,
    )
    projector, start = _projector(config, check_collision=False)
    actions = np.zeros((1000, 8), dtype=float)
    phase = np.linspace(0.0, 40.0 * np.pi, 1000, endpoint=False)
    offsets = np.linspace(0.0, np.pi, 7)
    actions[:, :7] = 3.0 * np.sin(phase[:, None] + offsets[None, :])
    actions[:, 7] = 0.5

    result = projector.project_chunk(start, _chunk(actions))

    assert result.feasible
    velocities = result.projected_actions[:, :7]
    assert np.all(np.abs(velocities) <= projector.velocity_limits + 1e-7)
    acceleration = np.diff(
        np.vstack((np.zeros((1, 7)), velocities)), axis=0
    ) / config.control_dt_s
    assert np.max(np.abs(acceleration)) <= 8.0 + 1e-5
    assert all(projector.robot.within_limits(q) for q in result.predicted_positions)
    assert result.intervention_steps > 0
    assert result.solve_p95_ms < config.step_budget_ms


def test_qp_applies_explicit_per_joint_velocity_limits() -> None:
    absolute_limits = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
    config = QPProjectionConfig(
        control_dt_s=0.05,
        joint_velocity_limit_scale=1.0,
        absolute_joint_velocity_limits_rad_s=absolute_limits,
        joint_acceleration_limit_rad_s2=100.0,
        step_budget_ms=100.0,
    )
    projector, start = _projector(config, check_collision=False)
    actions = np.full((4, 8), 5.0, dtype=float)
    actions[:, 7] = 0.5

    result = projector.project_chunk(start, _chunk(actions))

    assert result.feasible
    np.testing.assert_allclose(projector.velocity_limits, absolute_limits)
    assert np.all(
        np.abs(result.projected_actions[:, :7])
        <= np.asarray(absolute_limits)[None, :] + 1e-7
    )
    assert result.intervention_steps == 4


@pytest.mark.parametrize(
    "limits",
    [
        (1.0,) * 6,
        (1.0,) * 6 + (0.0,),
        (1.0,) * 6 + (float("nan"),),
        ("1.0",) * 7,
    ],
)
def test_qp_rejects_invalid_explicit_velocity_limits(limits: object) -> None:
    with pytest.raises(ValueError, match="velocity limits"):
        QPProjectionConfig(absolute_joint_velocity_limits_rad_s=limits)


def test_qp_enforces_coupled_linear_constraint() -> None:
    projector, start = _projector()
    actions = np.ones((3, 8), dtype=float)
    actions[:, 7] = 0.5
    coupled = QPLinearConstraint(
        matrix=np.ones((1, 7)),
        lower=np.array([-np.inf]),
        upper=np.array([0.10]),
        label="sum_velocity_cap",
    )

    result = projector.project_chunk(
        start,
        _chunk(actions),
        linear_constraints=(coupled,),
    )

    assert result.feasible
    assert np.all(np.sum(result.projected_actions[:, :7], axis=1) <= 0.10001)
    assert result.intervention_steps == 3


def test_qp_rejects_infeasible_constraint_fail_closed() -> None:
    projector, start = _projector()
    actions = np.zeros((2, 8), dtype=float)
    impossible = QPLinearConstraint(
        matrix=np.ones((1, 7)),
        lower=np.array([100.0]),
        upper=np.array([np.inf]),
        label="impossible_progress",
    )

    result = projector.project_chunk(
        start,
        _chunk(actions),
        linear_constraints=(impossible,),
    )

    assert not result.feasible
    assert result.failure_step == 0
    assert result.failure_reason is not None
    assert "solver_failed" in result.failure_reason
    np.testing.assert_allclose(result.projected_actions, 0.0)


def test_qp_rejects_state_that_cannot_brake_before_joint_margin() -> None:
    config = QPProjectionConfig(
        control_dt_s=0.05,
        joint_acceleration_limit_rad_s2=1.0,
        step_budget_ms=100.0,
    )
    projector, _ = _projector(config)
    near_upper = (
        projector.robot.upper_limits - config.joint_limit_margin_rad - 0.001
    )
    previous = np.minimum(np.full(7, 0.5), projector.velocity_limits)
    actions = np.zeros((1, 8), dtype=float)

    result = projector.project_chunk(
        near_upper,
        _chunk(actions),
        previous_velocity=previous,
    )

    assert not result.feasible
    assert result.failure_reason == "fallback_solver_failed:inconsistent_box_bounds"


def test_qp_smoke_reports_solver_and_limits() -> None:
    report = run_qp_projection_smoke()

    assert report["passed"]
    assert report["solver"] == "OSQP"
    assert report["all_positions_within_limits"]
