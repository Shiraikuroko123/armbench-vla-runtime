import numpy as np
import pytest

from armbench.collision import CollisionChecker
from armbench.geometry import Sphere
from armbench.model import RobotModel
from armbench.planners import RRTConnect, RRTStar
from armbench.result import PlanStatus
from armbench.scenario import benchmark_scenarios


@pytest.mark.parametrize("planner_class", [RRTConnect, RRTStar])
def test_free_space_returns_exact_endpoints(planner_class: type) -> None:
    robot = RobotModel.panda()
    scenario = benchmark_scenarios()["free_space"]
    checker = CollisionChecker(robot, scenario.obstacles)
    planner = planner_class(checker, np.random.default_rng(0), timeout_s=1.0)

    result = planner.plan(scenario.start, scenario.goal)

    assert result.status is PlanStatus.SUCCESS
    np.testing.assert_allclose(result.path[0], scenario.start)
    np.testing.assert_allclose(result.path[-1], scenario.goal)
    assert checker.path_is_valid(result.path)


def test_rrt_connect_is_path_deterministic_for_a_fixed_seed() -> None:
    robot = RobotModel.panda()
    scenario = benchmark_scenarios()["single_block"]
    paths = []
    for _ in range(2):
        checker = CollisionChecker(robot, scenario.obstacles)
        planner = RRTConnect(
            checker, np.random.default_rng(7), max_iterations=1_000, timeout_s=2.0
        )
        result = planner.plan(scenario.start, scenario.goal)
        assert result.success
        paths.append(np.asarray(result.path))

    np.testing.assert_allclose(paths[0], paths[1], atol=0.0, rtol=0.0)


def test_rrt_star_finds_single_block_path() -> None:
    robot = RobotModel.panda()
    scenario = benchmark_scenarios()["single_block"]
    checker = CollisionChecker(robot, scenario.obstacles)
    planner = RRTStar(
        checker, np.random.default_rng(0), max_iterations=2_000, timeout_s=2.0
    )

    result = planner.plan(scenario.start, scenario.goal)

    assert result.success
    assert checker.path_is_valid(result.path)


def test_start_and_goal_collision_have_distinct_statuses() -> None:
    robot = RobotModel(
        np.array([[0.0, 0.0, 1.0, 0.0]]),
        np.array([-np.pi]),
        np.array([np.pi]),
        np.array([1.0]),
    )
    checker = CollisionChecker(
        robot,
        (Sphere(np.array([0.5, 0.0, 0.0]), 0.05, "blocking"),),
        link_radius=0.0,
        safety_margin=0.0,
    )

    start_invalid = RRTConnect(checker, np.random.default_rng(0)).plan(
        [0.0], [np.pi / 2.0]
    )
    assert start_invalid.status is PlanStatus.START_IN_COLLISION

    goal_invalid = RRTConnect(checker, np.random.default_rng(0)).plan(
        [np.pi / 2.0], [0.0]
    )
    assert goal_invalid.status is PlanStatus.GOAL_IN_COLLISION


def test_zero_deadline_returns_timeout_before_search() -> None:
    robot = RobotModel.panda()
    scenario = benchmark_scenarios()["free_space"]
    planner = RRTConnect(
        CollisionChecker(robot), np.random.default_rng(0), timeout_s=0.0
    )

    assert planner.plan(scenario.start, scenario.goal).status is PlanStatus.TIMEOUT

