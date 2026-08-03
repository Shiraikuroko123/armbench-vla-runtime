import numpy as np

from armbench.collision import CollisionChecker
from armbench.geometry import Sphere
from armbench.model import RobotModel
from armbench.scenario import benchmark_scenarios


def one_link_robot() -> RobotModel:
    return RobotModel(
        dh=np.array([[0.0, 0.0, 1.0, 0.0]]),
        lower_limits=np.array([-np.pi]),
        upper_limits=np.array([np.pi]),
        velocity_limits=np.array([1.0]),
        name="one_link_test_arm",
    )


def test_panda_forward_kinematics_shape_and_fixed_base_height() -> None:
    robot = RobotModel.panda()
    scenario = benchmark_scenarios()["free_space"]
    points = robot.forward_points(scenario.start)

    assert points.shape == (8, 3)
    np.testing.assert_allclose(points[0], [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(points[1], [0.0, 0.0, 0.333], atol=1e-12)


def test_joint_limits_are_checked_per_joint() -> None:
    robot = RobotModel.panda()
    valid = (robot.lower_limits + robot.upper_limits) / 2.0
    invalid = valid.copy()
    invalid[3] = 0.0

    assert robot.within_limits(valid)
    assert not robot.within_limits(invalid)


def test_link_midpoint_collision_that_endpoint_only_check_would_miss() -> None:
    robot = one_link_robot()
    sphere = Sphere(np.array([0.5, 0.08, 0.0]), 0.03, "mid_link")
    checker = CollisionChecker(
        robot, (sphere,), link_radius=0.03, safety_margin=0.02
    )
    points = robot.forward_points([0.0])

    assert all(np.linalg.norm(point - sphere.center) > sphere.radius for point in points)
    assert checker.configuration_failure([0.0]) == "collision:mid_link:link_0"


def test_edge_interpolation_finds_collision_between_safe_endpoints() -> None:
    robot = one_link_robot()
    sphere = Sphere(np.array([0.5, 0.0, 0.0]), 0.05, "sweep")
    checker = CollisionChecker(
        robot,
        (sphere,),
        link_radius=0.01,
        safety_margin=0.0,
        resolution=0.05,
    )
    start = np.array([-np.pi / 2.0])
    goal = np.array([np.pi / 2.0])

    assert checker.configuration_is_valid(start)
    assert checker.configuration_is_valid(goal)
    assert not checker.edge_is_valid(start, goal)


def test_all_benchmark_endpoints_are_valid_and_obstacle_paths_are_blocked() -> None:
    robot = RobotModel.panda()
    for name, scenario in benchmark_scenarios().items():
        checker = CollisionChecker(robot, scenario.obstacles)
        assert checker.configuration_is_valid(scenario.start), name
        assert checker.configuration_is_valid(scenario.goal), name
        if name != "free_space":
            assert not checker.edge_is_valid(scenario.start, scenario.goal), name

