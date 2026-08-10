from __future__ import annotations

import numpy as np
import pytest

from armbench.mujoco_sim.continuous_collision import (
    ContinuousMuJoCoCollisionChecker,
)
from armbench.mujoco_sim.dynamics_braking import (
    DynamicsBrakingConfig,
    generate_dynamics_validated_brake,
)
from armbench.mujoco_sim.model import MuJoCoPanda
from armbench.mujoco_sim.persistent_dynamics_braking import (
    PersistentDynamicsBrakingValidator,
)
from armbench.mujoco_sim.scenarios import mujoco_scenarios


class _CountingChecker(ContinuousMuJoCoCollisionChecker):
    def __init__(self, robot: MuJoCoPanda) -> None:
        super().__init__(robot)
        self.configuration_calls = 0
        self.edge_calls = 0

    def configuration_failure(self, q: np.ndarray) -> str | None:
        self.configuration_calls += 1
        return super().configuration_failure(q)

    def edge_certificate(self, q_start: np.ndarray, q_end: np.ndarray):
        self.edge_calls += 1
        return super().edge_certificate(q_start, q_end)


def _config(**overrides: object) -> DynamicsBrakingConfig:
    values: dict[str, object] = {
        "sample_dt_s": 0.01,
        "joint_acceleration_limits_rad_s2": (2.0,) * 7,
        "max_stop_time_s": 1.0,
    }
    values.update(overrides)
    return DynamicsBrakingConfig(**values)


def test_persistent_brake_matches_reference_safe_stop() -> None:
    robot = MuJoCoPanda.create(obstacles=())
    checker = ContinuousMuJoCoCollisionChecker(robot)
    config = _config()
    q = mujoco_scenarios()["free_space"].start
    qvel = np.array([0.20, -0.10, 0.05, 0.0, 0.0, 0.0, 0.0])
    expected = generate_dynamics_validated_brake(
        robot, checker, q, qvel, config
    )
    validator = PersistentDynamicsBrakingValidator(robot, checker, config)

    actual = validator.validate(q, qvel)

    assert actual.validated == expected.validated is True
    assert actual.failure_reason == expected.failure_reason
    assert actual.evaluated_samples == expected.evaluated_samples
    assert actual.max_torque_ratio == pytest.approx(expected.max_torque_ratio)
    np.testing.assert_allclose(actual.times_s, expected.times_s, atol=0.0)
    np.testing.assert_allclose(actual.positions_rad, expected.positions_rad)
    np.testing.assert_allclose(actual.velocities_rad_s, expected.velocities_rad_s)
    np.testing.assert_allclose(
        actual.accelerations_rad_s2, expected.accelerations_rad_s2
    )
    np.testing.assert_allclose(
        actual.inverse_dynamics_efforts,
        expected.inverse_dynamics_efforts,
        atol=1e-12,
    )


def test_persistent_brake_certifies_one_whole_stop_edge() -> None:
    robot = MuJoCoPanda.create(obstacles=())
    checker = _CountingChecker(robot)
    validator = PersistentDynamicsBrakingValidator(robot, checker, _config())
    q = mujoco_scenarios()["free_space"].start
    qvel = np.array([0.20, -0.10, 0.05, 0.0, 0.0, 0.0, 0.0])

    result = validator.validate(q, qvel)

    assert result.validated is True
    assert len(result.times_s) == 11
    assert checker.configuration_calls == 1
    assert checker.edge_calls == 1
    assert validator.whole_stop_edges_checked == 1


def test_persistent_brake_reuses_workspace_deterministically() -> None:
    robot = MuJoCoPanda.create(obstacles=())
    checker = ContinuousMuJoCoCollisionChecker(robot)
    validator = PersistentDynamicsBrakingValidator(robot, checker, _config())
    q = mujoco_scenarios()["free_space"].start
    qvel = np.array([0.15, -0.08, 0.03, 0.0, 0.0, 0.0, 0.0])

    first = validator.validate(q, qvel)
    second = validator.validate(q, qvel)

    assert first.validated == second.validated is True
    assert validator.validation_calls == 2
    np.testing.assert_array_equal(
        first.inverse_dynamics_efforts, second.inverse_dynamics_efforts
    )


def test_persistent_brake_preserves_joint_limit_rejection() -> None:
    robot = MuJoCoPanda.create(obstacles=())
    checker = ContinuousMuJoCoCollisionChecker(robot)
    config = _config()
    q = mujoco_scenarios()["free_space"].start.copy()
    q[0] = robot.upper_limits[0] - 0.001
    qvel = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    expected = generate_dynamics_validated_brake(
        robot, checker, q, qvel, config
    )
    validator = PersistentDynamicsBrakingValidator(robot, checker, config)

    actual = validator.validate(q, qvel)

    assert actual.validated == expected.validated is False
    assert actual.failure_reason == expected.failure_reason
    assert actual.failure_sample_index == expected.failure_sample_index


def test_persistent_brake_rejects_checker_for_another_robot() -> None:
    robot = MuJoCoPanda.create(obstacles=())
    other = MuJoCoPanda.create(obstacles=())
    checker = ContinuousMuJoCoCollisionChecker(other)

    with pytest.raises(ValueError, match="same Panda"):
        PersistentDynamicsBrakingValidator(robot, checker)


@pytest.mark.parametrize("velocity", [["0.0"] * 7, [False] * 7])
def test_persistent_brake_rejects_coerced_velocity_vectors(
    velocity: object,
) -> None:
    robot = MuJoCoPanda.create(obstacles=())
    checker = ContinuousMuJoCoCollisionChecker(robot)
    validator = PersistentDynamicsBrakingValidator(robot, checker)

    with pytest.raises(ValueError, match="numeric vector"):
        validator.validate(mujoco_scenarios()["free_space"].start, velocity)
