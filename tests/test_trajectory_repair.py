from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from armbench.mujoco_sim import MuJoCoCollisionChecker, MuJoCoPanda
from armbench.mujoco_sim.model import default_panda_scene_path
from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.vla.cartesian_adapter import PandaCartesianActionAdapter
from armbench.vla.guard import ActionChunkGuard, GuardConfig
from armbench.vla.pi05_archive_replay import (
    ValidatedPi05Archive,
    validate_pi05_source_archive,
)
from armbench.vla.trajectory_repair import (
    BrakingRepairConfig,
    BrakingTrajectoryGuard,
)
from armbench.vla.types import DROID_IMAGE_SHAPE, VLAObservation


SOURCE = Path(
    "evidence/pi05_rtc_overlap_primary_v3_seed_20260807_001/evaluation"
)


@pytest.fixture(scope="module")
def archive() -> ValidatedPi05Archive:
    try:
        default_panda_scene_path()
    except FileNotFoundError as error:
        pytest.skip(str(error))
    return validate_pi05_source_archive(SOURCE)


def _case(
    archive: ValidatedPi05Archive,
    *,
    source_index: int,
    scenario_name: str,
    latency_ms: float | None = None,
) -> tuple[
    MuJoCoPanda,
    MuJoCoCollisionChecker,
    np.ndarray,
    VLAObservation,
    object,
    GuardConfig,
]:
    scenario = mujoco_scenarios()[scenario_name]
    robot = MuJoCoPanda.create(obstacles=scenario.obstacles)
    checker = MuJoCoCollisionChecker(robot, resolution=0.02)
    query = archive.transition_queries[source_index]
    measured_latency = (
        float(query["inference_latency_ms"])
        if latency_ms is None
        else latency_ms
    )
    captured_at_s = 1000.0
    chunk = PandaCartesianActionAdapter(robot).adapt(
        archive.arrays["response_actions"][source_index],
        scenario.start,
        source="registered_conflict_test",
        observation_sequence_id=source_index,
        inference_latency_ms=measured_latency,
        received_at_s=captured_at_s + measured_latency / 1000.0,
    ).chunk
    image = np.zeros(DROID_IMAGE_SHAPE, dtype=np.uint8)
    observation = VLAObservation(
        exterior_image=image,
        wrist_image=image,
        joint_position=scenario.start,
        gripper_position=np.array([1.0]),
        prompt="registered braking repair test",
        sequence_id=source_index,
        captured_at_s=captured_at_s,
    )
    config = GuardConfig(
        control_dt_s=0.05,
        deadline_ms=200.0,
        joint_velocity_clip_rad_s=float(np.max(robot.velocity_limits)),
    )
    return robot, checker, scenario.start, observation, chunk, config


def test_repair_config_rejects_unbounded_or_unsorted_search() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        BrakingRepairConfig(selection_deadline_ms=0.0)
    with pytest.raises(ValueError, match="strictly descend"):
        BrakingRepairConfig(trajectory_scales=(1.0, 0.5, 0.5, 0.0))
    with pytest.raises(ValueError, match="outside the scale set"):
        BrakingRepairConfig(max_scale_evaluations=6)


def test_registered_legacy_conflict_has_a_braking_invariant_repair(
    archive: ValidatedPi05Archive,
) -> None:
    robot, checker, q_start, observation, chunk, guard_config = _case(
        archive,
        source_index=2605,
        scenario_name="narrow_gate",
    )
    legacy = ActionChunkGuard(checker, guard_config).guard(
        q_start,
        1.0,
        observation,
        chunk,
    )
    repaired = BrakingTrajectoryGuard(
        checker,
        guard_config,
        BrakingRepairConfig(selection_deadline_ms=1000.0),
    ).repair(q_start, 1.0, observation, chunk)

    assert legacy.safe_after_guard is False
    assert legacy.acceleration_override_steps == 1
    assert repaired.safe_after_repair is True
    assert repaired.selected_scale == pytest.approx(0.25)
    assert repaired.max_repaired_acceleration_rad_s2 <= 15.0 + 1e-12
    assert repaired.terminal_brake_steps > 0
    assert checker.path_is_valid(repaired.predicted_positions)
    assert checker.path_is_valid(repaired.terminal_braking_positions)
    assert all(robot.within_limits(q) for q in repaired.predicted_positions)


def test_late_response_uses_prevalidated_hold(
    archive: ValidatedPi05Archive,
) -> None:
    _, checker, q_start, observation, chunk, guard_config = _case(
        archive,
        source_index=2605,
        scenario_name="narrow_gate",
        latency_ms=240.0,
    )
    result = BrakingTrajectoryGuard(
        checker,
        guard_config,
        BrakingRepairConfig(selection_deadline_ms=1000.0),
    ).repair(q_start, 1.0, observation, chunk)

    assert result.response_deadline_exceeded is True
    assert result.fallback_reason == "response_deadline"
    assert result.selected_scale == 0.0
    assert result.safe_after_repair is True
    np.testing.assert_allclose(result.repaired_actions[:, :7], 0.0)
    np.testing.assert_allclose(result.repaired_actions[:, 7], 1.0)
    np.testing.assert_allclose(
        result.predicted_positions,
        np.repeat(q_start[None, :], len(chunk.actions) + 1, axis=0),
    )


def test_guard_requires_enough_terminal_braking_steps(
    archive: ValidatedPi05Archive,
) -> None:
    _, checker, _, _, _, guard_config = _case(
        archive,
        source_index=2605,
        scenario_name="free_space",
    )

    with pytest.raises(ValueError, match="cannot stop"):
        BrakingTrajectoryGuard(
            checker,
            guard_config,
            BrakingRepairConfig(max_terminal_brake_steps=1),
        )
