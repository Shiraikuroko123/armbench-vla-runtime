from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from armbench.mujoco_sim import MuJoCoCollisionChecker, MuJoCoPanda
from armbench.mujoco_sim.dynamics_braking import (
    DynamicsBrakingConfig,
    generate_dynamics_validated_brake,
    run_dynamics_braking_smoke,
)
from armbench.mujoco_sim.continuous_collision import (
    ContinuousMuJoCoCollisionChecker,
)
from armbench.mujoco_sim.model import default_panda_scene_path
from armbench.mujoco_sim.scenarios import mujoco_scenarios


@pytest.fixture(scope="module")
def menagerie_path() -> Path:
    try:
        return default_panda_scene_path()
    except FileNotFoundError as error:
        pytest.skip(str(error))


@pytest.fixture(scope="module")
def free_space_pair(
    menagerie_path: Path,
) -> tuple[MuJoCoPanda, MuJoCoCollisionChecker]:
    robot = MuJoCoPanda.create(scene_path=menagerie_path, obstacles=())
    return robot, MuJoCoCollisionChecker(robot, resolution=0.01)


def _config(**overrides: object) -> DynamicsBrakingConfig:
    values: dict[str, object] = {
        "sample_dt_s": 0.01,
        "joint_acceleration_limits_rad_s2": (2.0,) * 7,
        "max_stop_time_s": 1.0,
    }
    values.update(overrides)
    return DynamicsBrakingConfig(**values)


def test_inverse_dynamics_validated_stop_is_acceleration_limited(
    free_space_pair: tuple[MuJoCoPanda, MuJoCoCollisionChecker],
) -> None:
    robot, checker = free_space_pair
    q = mujoco_scenarios()["free_space"].start
    qvel = np.array([0.20, -0.10, 0.05, 0.0, 0.0, 0.0, 0.0])

    result = generate_dynamics_validated_brake(robot, checker, q, qvel, _config())

    assert result.validated
    assert result.failure_reason is None
    assert result.stop_time_s == pytest.approx(0.1)
    assert result.required_stop_time_s == pytest.approx(0.1)
    assert result.evaluated_samples == len(result.times_s) == 11
    np.testing.assert_allclose(result.velocities_rad_s[-1], 0.0, atol=1e-15)
    np.testing.assert_allclose(
        np.diff(result.velocities_rad_s, axis=0) / 0.01,
        result.accelerations_rad_s2[:-1],
        atol=1e-12,
    )
    assert np.max(np.abs(result.accelerations_rad_s2)) <= 2.0 + 1e-12
    np.testing.assert_allclose(
        result.positions_rad[-1], q + 0.5 * qvel * result.stop_time_s
    )
    np.testing.assert_allclose(
        result.joint_stopping_displacement_rad,
        np.abs(0.5 * qvel * result.stop_time_s),
    )
    assert result.max_torque_ratio is not None
    assert result.max_torque_ratio < 1.0
    assert not result.positions_rad.flags.writeable
    assert not result.inverse_dynamics_efforts.flags.writeable


def test_recorded_effort_matches_direct_mujoco_inverse_dynamics(
    free_space_pair: tuple[MuJoCoPanda, MuJoCoCollisionChecker],
) -> None:
    robot, checker = free_space_pair
    q = mujoco_scenarios()["free_space"].start
    qvel = np.array([0.20, -0.10, 0.05, 0.0, 0.0, 0.0, 0.0])
    result = generate_dynamics_validated_brake(robot, checker, q, qvel, _config())
    data = mujoco.MjData(robot.model)
    robot.set_configuration(data, result.positions_rad[0], forward=False)
    data.qvel[robot.arm_dof_addresses] = result.velocities_rad_s[0]
    data.qacc[robot.arm_dof_addresses] = result.accelerations_rad_s2[0]

    mujoco.mj_inverse(robot.model, data)

    np.testing.assert_allclose(
        result.inverse_dynamics_efforts[0],
        data.qfrc_inverse[robot.arm_dof_addresses],
        atol=1e-12,
    )
    assert result.actuator_effort_lower_limits is not None
    assert result.actuator_effort_upper_limits is not None
    np.testing.assert_allclose(result.actuator_effort_lower_limits, -robot.force_limits)
    np.testing.assert_allclose(result.actuator_effort_upper_limits, robot.force_limits)


def test_zero_velocity_still_checks_gravity_effort_and_collision_state(
    free_space_pair: tuple[MuJoCoPanda, MuJoCoCollisionChecker],
) -> None:
    robot, checker = free_space_pair
    q = mujoco_scenarios()["free_space"].start

    result = generate_dynamics_validated_brake(
        robot, checker, q, np.zeros(7), _config()
    )

    assert result.validated
    assert result.stop_time_s == 0.0
    assert result.evaluated_samples == 1
    np.testing.assert_allclose(result.joint_stopping_displacement_rad, 0.0)
    assert np.max(np.abs(result.inverse_dynamics_efforts[0])) > 0.0


def test_stop_can_require_continuous_inter_sample_collision_certificates(
    menagerie_path: Path,
) -> None:
    robot = MuJoCoPanda.create(scene_path=menagerie_path, obstacles=())
    checker = ContinuousMuJoCoCollisionChecker(robot)
    q = mujoco_scenarios()["free_space"].start

    result = generate_dynamics_validated_brake(
        robot,
        checker,
        q,
        np.array([0.10, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0]),
        _config(),
    )

    assert result.validated
    assert result.inter_sample_edges_checked


def test_initial_velocity_limit_violation_fails_closed(
    free_space_pair: tuple[MuJoCoPanda, MuJoCoCollisionChecker],
) -> None:
    robot, checker = free_space_pair
    qvel = np.zeros(7)
    qvel[0] = robot.velocity_limits[0] + 0.01

    result = generate_dynamics_validated_brake(
        robot,
        checker,
        mujoco_scenarios()["free_space"].start,
        qvel,
        _config(),
    )

    assert not result.validated
    assert result.failure_reason == "joint_velocity_limit:joint1"
    assert result.failure_sample_index == 0
    assert result.evaluated_samples == 0
    assert result.max_torque_ratio is None


def test_stop_time_limit_is_rejected_before_large_candidate_allocation(
    free_space_pair: tuple[MuJoCoPanda, MuJoCoCollisionChecker],
) -> None:
    robot, checker = free_space_pair
    result = generate_dynamics_validated_brake(
        robot,
        checker,
        mujoco_scenarios()["free_space"].start,
        np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        _config(
            joint_acceleration_limits_rad_s2=(1e-9,) * 7,
            max_stop_time_s=0.1,
        ),
    )

    assert not result.validated
    assert result.failure_reason == "stop_time_limit"
    assert result.required_stop_time_s == pytest.approx(1e8)
    assert len(result.times_s) == 1
    assert result.evaluated_samples == 0


def test_predicted_joint_limit_crossing_fails_before_execution(
    free_space_pair: tuple[MuJoCoPanda, MuJoCoCollisionChecker],
) -> None:
    robot, checker = free_space_pair
    q = mujoco_scenarios()["free_space"].start.copy()
    q[0] = robot.upper_limits[0] - 0.001
    assert checker.configuration_is_valid(q)

    result = generate_dynamics_validated_brake(
        robot,
        checker,
        q,
        np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        _config(),
    )

    assert not result.validated
    assert result.failure_reason == "joint_position_limit:joint1"
    assert result.failure_sample_index is not None
    assert result.failure_sample_index > 0


def test_collision_at_any_sample_fails_closed(menagerie_path: Path) -> None:
    scenario = mujoco_scenarios()["single_block"]
    robot = MuJoCoPanda.create(scene_path=menagerie_path, obstacles=scenario.obstacles)
    checker = MuJoCoCollisionChecker(robot, resolution=0.01)
    colliding = scenario.start + 0.5 * (scenario.goal - scenario.start)

    result = generate_dynamics_validated_brake(
        robot, checker, colliding, np.zeros(7), _config()
    )

    assert not result.validated
    assert result.failure_reason is not None
    assert result.failure_reason.startswith("collision:center_block:")
    assert result.failure_sample_index == 0
    assert result.evaluated_samples == 0


def test_inverse_dynamics_force_limit_violation_reports_ratio(
    free_space_pair: tuple[MuJoCoPanda, MuJoCoCollisionChecker],
) -> None:
    robot, checker = free_space_pair
    result = generate_dynamics_validated_brake(
        robot,
        checker,
        mujoco_scenarios()["free_space"].start,
        np.array([0.20, -0.10, 0.05, 0.0, 0.0, 0.0, 0.0]),
        _config(actuator_force_limit_scale=0.1),
    )

    assert not result.validated
    assert result.failure_reason == "actuator_force_limit:joint4"
    assert result.failure_sample_index == 0
    assert result.evaluated_samples == 1
    assert result.max_torque_ratio is not None
    assert result.max_torque_ratio > 1.0


def test_missing_actuator_force_limit_fails_closed(menagerie_path: Path) -> None:
    robot = MuJoCoPanda.create(scene_path=menagerie_path, obstacles=())
    robot.model.actuator_forcelimited[0] = False
    checker = MuJoCoCollisionChecker(robot, resolution=0.01)

    result = generate_dynamics_validated_brake(
        robot,
        checker,
        mujoco_scenarios()["free_space"].start,
        np.zeros(7),
        _config(),
    )

    assert not result.validated
    assert result.failure_reason == "actuator_force_limit_unavailable"
    assert result.actuator_effort_lower_limits is None
    assert result.actuator_effort_upper_limits is None
    assert result.metrics()["actuator_effort_lower_limits"] is None


def test_rejects_a_collision_checker_for_another_robot(
    menagerie_path: Path,
) -> None:
    robot = MuJoCoPanda.create(scene_path=menagerie_path, obstacles=())
    other = MuJoCoPanda.create(scene_path=menagerie_path, obstacles=())
    checker = MuJoCoCollisionChecker(other)

    with pytest.raises(ValueError, match="same Panda"):
        generate_dynamics_validated_brake(
            robot,
            checker,
            mujoco_scenarios()["free_space"].start,
            np.zeros(7),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_dt_s": 0.0},
        {"sample_dt_s": "0.01"},
        {"max_stop_time_s": True},
        {"joint_acceleration_limits_rad_s2": (1.0,) * 6},
        {"joint_acceleration_limits_rad_s2": (1.0,) * 6 + (float("nan"),)},
        {"joint_acceleration_limits_rad_s2": ("1.0",) * 7},
        {"actuator_force_limit_scale": 0.0},
        {"actuator_force_limit_scale": 1.1},
        {"check_inter_sample_edges": 1},
    ],
)
def test_invalid_braking_configuration_is_rejected(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        DynamicsBrakingConfig(**kwargs)


@pytest.mark.parametrize("velocity", [["0.0"] * 7, [False] * 7])
def test_braking_rejects_coerced_velocity_vectors(
    free_space_pair: tuple[MuJoCoPanda, MuJoCoCollisionChecker],
    velocity: object,
) -> None:
    robot, checker = free_space_pair

    with pytest.raises(ValueError, match="numeric vector"):
        generate_dynamics_validated_brake(
            robot,
            checker,
            mujoco_scenarios()["free_space"].start,
            velocity,
        )


def test_cpu_smoke_reports_scope_and_dynamics_metrics() -> None:
    report = run_dynamics_braking_smoke()

    assert report["validated"] is True
    assert report["inverse_dynamics_method"] == "mujoco.mj_inverse.v1"
    assert report["max_torque_ratio"] < 1.0
    assert report["collision_checker"] == "ContinuousMuJoCoCollisionChecker"
    assert report["continuous_collision_edges"] is True
    assert report["policy_checkpoint_used"] is False
    assert report["hard_realtime_claim"] is False
    assert report["scope"] == (
        "sampled_inverse_dynamics_feasibility_not_hardware_certification"
    )
