from __future__ import annotations

import numpy as np
import pytest

from armbench.mujoco_sim.model import MuJoCoPanda
from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.vla.persistent_qp_projection import PersistentQPActionProjector
from armbench.vla.qp_projection import (
    QPActionProjector,
    QPLinearConstraint,
    QPProjectionConfig,
)
from armbench.vla.types import ActionChunk


def _chunk(actions: np.ndarray) -> ActionChunk:
    return ActionChunk(
        actions=actions,
        source="persistent_qp_test",
        observation_sequence_id=0,
        inference_latency_ms=0.0,
    )


def _projectors() -> tuple[
    MuJoCoPanda, QPActionProjector, PersistentQPActionProjector
]:
    robot = MuJoCoPanda.create(obstacles=())
    config = QPProjectionConfig(
        absolute_joint_velocity_limits_rad_s=(1.0,) * 7,
        joint_acceleration_limit_rad_s2=5.0,
        step_budget_ms=500.0,
    )
    return (
        robot,
        QPActionProjector(robot, None, config),
        PersistentQPActionProjector(robot, None, config),
    )


def test_persistent_projector_matches_reference_constraints_and_solution() -> None:
    robot, reference, persistent = _projectors()
    start = mujoco_scenarios()["free_space"].start
    rng = np.random.default_rng(20260810)
    actions = np.zeros((10, 8), dtype=float)
    actions[:, :7] = rng.normal(0.0, 0.8, size=(10, 7))
    actions[:, 7] = np.linspace(-0.2, 1.2, 10)
    chunk = _chunk(actions)

    expected = reference.project_chunk(start, chunk)
    actual = persistent.project_chunk(start, chunk)

    assert expected.feasible is True
    assert actual.feasible is True
    assert persistent.solve_calls == 10
    np.testing.assert_allclose(
        actual.projected_actions, expected.projected_actions, atol=2e-4, rtol=0.0
    )
    np.testing.assert_allclose(
        actual.predicted_positions, expected.predicted_positions, atol=1e-5, rtol=0.0
    )
    assert np.max(np.abs(actual.projected_actions[:, :7])) <= 1.0 + 1e-9
    acceleration = np.diff(
        np.vstack((np.zeros((1, 7)), actual.projected_actions[:, :7])), axis=0
    ) / persistent.config.control_dt_s
    assert np.max(np.abs(acceleration)) <= 5.0 + 1e-6
    assert robot.within_limits(actual.predicted_positions[-1])


def test_persistent_projector_resets_warm_start_between_chunks() -> None:
    _, _, persistent = _projectors()
    start = mujoco_scenarios()["free_space"].start
    actions = np.zeros((4, 8), dtype=float)
    actions[:, 0] = 0.4
    actions[:, 7] = 0.5
    chunk = _chunk(actions)

    first = persistent.project_chunk(start, chunk)
    persistent.project_chunk(start, _chunk(-actions))
    repeated = persistent.project_chunk(start, chunk)

    np.testing.assert_allclose(
        first.projected_actions, repeated.projected_actions, atol=1e-10, rtol=0.0
    )
    np.testing.assert_allclose(
        first.predicted_positions, repeated.predicted_positions, atol=1e-10, rtol=0.0
    )


def test_persistent_projector_rejects_dynamic_constraint_matrix() -> None:
    _, _, persistent = _projectors()
    start = mujoco_scenarios()["free_space"].start
    actions = np.zeros((2, 8), dtype=float)
    constraint = QPLinearConstraint(
        matrix=np.eye(7),
        lower=-np.ones(7),
        upper=np.ones(7),
        label="test",
    )

    with pytest.raises(ValueError, match="registered box constraints"):
        persistent.project_chunk(
            start, _chunk(actions), linear_constraints=(constraint,)
        )


@pytest.mark.parametrize("previous", [np.ones(6), np.full(7, np.nan)])
def test_persistent_projector_rejects_invalid_previous_velocity(
    previous: np.ndarray,
) -> None:
    _, _, persistent = _projectors()
    start = mujoco_scenarios()["free_space"].start

    with pytest.raises(ValueError, match="previous_velocity"):
        persistent.project_chunk(
            start,
            _chunk(np.zeros((2, 8))),
            previous_velocity=previous,
        )
