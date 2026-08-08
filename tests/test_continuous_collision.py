from __future__ import annotations

import numpy as np

from armbench.mujoco_sim import MuJoCoCollisionChecker, MuJoCoPanda
from armbench.mujoco_sim.continuous_collision import (
    ContinuousCollisionConfig,
    ContinuousMuJoCoCollisionChecker,
    run_continuous_collision_smoke,
)
from armbench.mujoco_sim.scenarios import mujoco_scenarios


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


def test_small_free_space_edge_receives_continuous_certificate() -> None:
    scenario = mujoco_scenarios()["free_space"]
    checker = ContinuousMuJoCoCollisionChecker(MuJoCoPanda.create(obstacles=()))
    end = scenario.start.copy()
    end[:3] += [0.01, -0.01, 0.005]

    certificate = checker.edge_certificate(scenario.start, end)

    assert certificate.certified_safe
    assert certificate.status == "certified_safe"
    assert certificate.pair_evaluations > 0
    assert certificate.subintervals_evaluated > 0


def test_continuous_checker_rejects_intermediate_self_collision() -> None:
    robot = MuJoCoPanda.create(obstacles=())
    sampled = MuJoCoCollisionChecker(robot, resolution=0.01)
    checker = ContinuousMuJoCoCollisionChecker(robot)

    assert sampled.configuration_is_valid(SELF_START)
    assert sampled.configuration_is_valid(SELF_END)
    certificate = checker.edge_certificate(SELF_START, SELF_END)

    assert not certificate.certified_safe
    assert certificate.status == "collision"
    assert certificate.collision_pair is not None
    assert certificate.collision_pair.startswith("self:")


def test_unproven_edge_fails_closed_at_depth_limit() -> None:
    scenario = mujoco_scenarios()["free_space"]
    checker = ContinuousMuJoCoCollisionChecker(
        MuJoCoPanda.create(obstacles=()),
        ContinuousCollisionConfig(max_depth=0),
    )

    certificate = checker.edge_certificate(scenario.start, scenario.goal)

    assert not certificate.certified_safe
    assert certificate.status == "indeterminate"
    assert certificate.reason == "maximum_subdivision_depth"


def test_continuous_safe_edges_are_not_rejected_by_dense_sampling() -> None:
    scenario = mujoco_scenarios()["free_space"]
    continuous_robot = MuJoCoPanda.create(obstacles=())
    dense_robot = MuJoCoPanda.create(obstacles=())
    continuous = ContinuousMuJoCoCollisionChecker(continuous_robot)
    dense = MuJoCoCollisionChecker(dense_robot, resolution=0.001)
    rng = np.random.default_rng(20260808)
    certified = 0
    for _ in range(12):
        start = scenario.start + rng.normal(0.0, 0.02, size=7)
        end = start + rng.normal(0.0, 0.04, size=7)
        start = np.clip(
            start,
            continuous_robot.lower_limits + 1e-3,
            continuous_robot.upper_limits - 1e-3,
        )
        end = np.clip(
            end,
            continuous_robot.lower_limits + 1e-3,
            continuous_robot.upper_limits - 1e-3,
        )
        certificate = continuous.edge_certificate(start, end)
        if certificate.certified_safe:
            certified += 1
            assert dense.edge_is_valid(start, end)
    assert certified > 0


def test_continuous_collision_smoke_covers_static_and_self_pairs() -> None:
    report = run_continuous_collision_smoke()

    assert report["passed"]
    assert report["safe_edge"]["certified_safe"]
    assert not report["static_blocked_edge"]["certified_safe"]
    assert report["self_collision_edge"]["collision_pair"].startswith("self:")
