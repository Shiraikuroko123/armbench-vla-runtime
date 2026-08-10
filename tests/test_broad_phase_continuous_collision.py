from __future__ import annotations

import numpy as np

from armbench.mujoco_sim.broad_phase_continuous_collision import (
    BroadPhaseContinuousMuJoCoCollisionChecker,
)
from armbench.mujoco_sim.continuous_collision import (
    ContinuousMuJoCoCollisionChecker,
)
from armbench.mujoco_sim.model import MuJoCoPanda
from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.vla.integrated_panda_task import (
    FIXED_GRIPPER_ALLOWED_BODY_PAIR,
)


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

# MuJoCo 3.11's mesh distance can change sign after a sequence of unrelated
# mesh queries at this edge.  The world-space bounding spheres remain disjoint
# throughout, so this fixture protects the independent broad-phase proof from
# being changed merely to reproduce that order-sensitive legacy result.
ORDER_SENSITIVE_START = np.array(
    [
        -0.05600463276513281,
        -0.6249999884901319,
        -0.0037666235341182806,
        -2.1351759403695127,
        -0.0077373487107266145,
        1.5961440897914543,
        0.8513987112435305,
    ]
)
ORDER_SENSITIVE_END = np.array(
    [
        -0.07070152075400343,
        -0.5749999571849913,
        -0.0032124681862945696,
        -2.0925744274677203,
        -0.008973900812106455,
        1.6024438354620965,
        0.8900147171325068,
    ]
)


def _remove_fixed_gripper_pair(checker: object) -> None:
    retained = []
    for pair in checker.pairs:
        bodies = frozenset(
            (
                checker.robot.body_name_for_geom(pair.geom1),
                checker.robot.body_name_for_geom(pair.geom2),
            )
        )
        if pair.kind == "self_collision" and bodies == FIXED_GRIPPER_ALLOWED_BODY_PAIR:
            continue
        retained.append(pair)
    if isinstance(checker, BroadPhaseContinuousMuJoCoCollisionChecker):
        checker.set_pairs(tuple(retained))
    else:
        checker.pairs = tuple(retained)


def _checkers(
    scenario_name: str,
) -> tuple[
    ContinuousMuJoCoCollisionChecker,
    BroadPhaseContinuousMuJoCoCollisionChecker,
]:
    scenario = mujoco_scenarios()[scenario_name]
    reference_robot = MuJoCoPanda.create(obstacles=scenario.obstacles)
    broad_robot = MuJoCoPanda.create(obstacles=scenario.obstacles)
    reference = ContinuousMuJoCoCollisionChecker(reference_robot)
    broad = BroadPhaseContinuousMuJoCoCollisionChecker(broad_robot)
    _remove_fixed_gripper_pair(reference)
    _remove_fixed_gripper_pair(broad)
    return reference, broad


def test_broad_phase_matches_registered_safe_and_collision_edges() -> None:
    free_reference, free_broad = _checkers("free_space")
    free = mujoco_scenarios()["free_space"]
    small_end = free.start.copy()
    small_end[:3] += np.array([0.01, -0.01, 0.005])

    safe_reference = free_reference.edge_certificate(free.start, small_end)
    safe_broad = free_broad.edge_certificate(free.start, small_end)
    self_reference = free_reference.edge_certificate(SELF_START, SELF_END)
    self_broad = free_broad.edge_certificate(SELF_START, SELF_END)

    assert safe_reference.certified_safe is True
    assert safe_broad.certified_safe is True
    assert self_reference.certified_safe is False
    assert self_broad.certified_safe is False
    assert self_broad.status == self_reference.status
    assert free_broad.broad_phase_pruned_pairs > 0
    assert safe_broad.pair_evaluations < safe_reference.pair_evaluations


def test_broad_phase_never_marks_reference_rejection_safe_on_seeded_edges() -> None:
    rng = np.random.default_rng(20260810)
    for scenario_name in ("free_space", "single_block", "narrow_gate"):
        reference, broad = _checkers(scenario_name)
        scenario = mujoco_scenarios()[scenario_name]
        for _ in range(12):
            first = scenario.start + rng.normal(0.0, 0.12, size=7)
            second = first + rng.normal(0.0, 0.08, size=7)
            first = np.clip(
                first,
                reference.robot.lower_limits + 0.03,
                reference.robot.upper_limits - 0.03,
            )
            second = np.clip(
                second,
                reference.robot.lower_limits + 0.03,
                reference.robot.upper_limits - 0.03,
            )
            expected = reference.edge_certificate(first, second)
            actual = broad.edge_certificate(first, second)

            assert not (actual.certified_safe and not expected.certified_safe)
            if expected.certified_safe:
                assert actual.certified_safe


def test_broad_phase_metric_reset_does_not_change_decision() -> None:
    reference, broad = _checkers("free_space")
    scenario = mujoco_scenarios()["free_space"]
    end = scenario.start.copy()
    end[0] += 0.02
    expected = reference.edge_certificate(scenario.start, end)
    first = broad.edge_certificate(scenario.start, end)

    assert broad.broad_phase_pair_tests > 0
    broad.reset_metrics()
    repeated = broad.edge_certificate(scenario.start, end)

    assert broad.broad_phase_pair_tests > 0
    assert first.status == repeated.status == expected.status
    assert broad.safe_configuration_cache_hits >= 2


def test_changing_registered_pairs_invalidates_safe_configuration_cache() -> None:
    _, broad = _checkers("free_space")
    scenario = mujoco_scenarios()["free_space"]
    end = scenario.start.copy()
    end[0] += 0.02
    assert broad.edge_certificate(scenario.start, end).certified_safe
    assert broad.safe_configuration_cache_size > 0

    broad.set_pairs(broad.pairs)

    assert broad.safe_configuration_cache_size == 0


def test_broad_phase_regression_for_order_sensitive_mesh_distance() -> None:
    reference, broad = _checkers("free_space")

    legacy = reference.edge_certificate(
        ORDER_SENSITIVE_START, ORDER_SENSITIVE_END
    )
    certified = broad.edge_certificate(
        ORDER_SENSITIVE_START, ORDER_SENSITIVE_END
    )

    assert legacy.status == "collision"
    assert legacy.collision_pair == "self:link6:geom_50:left_finger:geom_68"
    assert certified.certified_safe is True
    assert broad.broad_phase_pruned_pairs > 0

    # Dense contact sampling is a regression oracle in addition to the
    # continuous bounding-sphere certificate used for the actual decision.
    for fraction in np.linspace(0.0, 1.0, 257):
        q = ORDER_SENSITIVE_START + fraction * (
            ORDER_SENSITIVE_END - ORDER_SENSITIVE_START
        )
        broad.robot.set_configuration(broad.data, q)
        assert broad.data.ncon == 0
