import numpy as np
import pytest

from armbench.collision import CollisionChecker
from armbench.control import DiscreteLQR, PDController, simulate_tracking
from armbench.model import RobotModel
from armbench.planners import RRTConnect
from armbench.postprocess import shortcut_path, time_parameterize
from armbench.scenario import benchmark_scenarios


def test_shortcut_never_increases_length_and_revalidates() -> None:
    robot = RobotModel.panda()
    scenario = benchmark_scenarios()["single_block"]
    checker = CollisionChecker(robot, scenario.obstacles)
    planned = RRTConnect(checker, np.random.default_rng(2), timeout_s=2.0).plan(
        scenario.start, scenario.goal
    )
    assert planned.success

    result = shortcut_path(
        planned.path, checker, np.random.default_rng(102), attempts=100
    )

    assert result.smoothed_length <= result.original_length + 1e-12
    assert checker.path_is_valid(result.path)
    np.testing.assert_allclose(result.path[0], scenario.start)
    np.testing.assert_allclose(result.path[-1], scenario.goal)


def test_time_parameterization_respects_scaled_joint_velocity_limits() -> None:
    path = [np.zeros(2), np.array([1.0, -0.5]), np.array([1.5, 0.5])]
    limits = np.array([1.0, 2.0])
    trajectory = time_parameterize(path, limits, control_dt=0.01, speed_scale=0.5)

    assert trajectory.times[0] == 0.0
    np.testing.assert_allclose(trajectory.positions[0], path[0])
    np.testing.assert_allclose(trajectory.positions[-1], path[-1])
    assert np.max(np.abs(trajectory.velocities[:, 0])) <= 0.5 + 1e-12
    assert np.max(np.abs(trajectory.velocities[:, 1])) <= 1.0 + 1e-12


@pytest.mark.parametrize("controller_name", ["pd", "lqr"])
def test_controller_tracks_one_joint_ramp(controller_name: str) -> None:
    trajectory = time_parameterize(
        [np.array([0.0]), np.array([0.5])],
        np.array([1.0]),
        control_dt=0.02,
        speed_scale=0.25,
    )
    if controller_name == "pd":
        controller = PDController.create(1)
    else:
        controller = DiscreteLQR.create(1, dt=0.02)

    result = simulate_tracking(
        trajectory,
        controller,
        np.random.default_rng(0),
        delay_ms=0,
        measurement_noise_std=0.0,
        hold_time_s=1.0,
    )

    assert result.rmse < 0.08
    assert abs(result.actual_positions[-1, 0] - 0.5) < 0.03


def test_discrete_lqr_closed_loop_is_stable() -> None:
    dt = 0.02
    controller = DiscreteLQR.create(1, dt=dt)
    a = np.array([[1.0, dt], [0.0, 1.0]])
    b = np.array([[0.5 * dt**2], [dt]])
    eigenvalues = np.linalg.eigvals(a - b @ controller.gain[None, :])

    assert np.all(np.abs(eigenvalues) < 1.0)

