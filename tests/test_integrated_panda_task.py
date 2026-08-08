from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest

from armbench.mujoco_sim.model import MuJoCoPanda
from armbench.vla.integrated_panda_task import (
    FIXED_GRIPPER_ALLOWED_BODY_PAIR,
    IntegratedPandaTaskConfig,
    _write_manifest,
    make_integrated_task_checker,
    run_integrated_panda_tasks,
    validate_integrated_panda_tasks,
)


@pytest.fixture(scope="module")
def task_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return run_integrated_panda_tasks(
        tmp_path_factory.mktemp("integrated_task") / "artifact",
        IntegratedPandaTaskConfig(profiles=("free_space_short_smoke",)),
    )


def test_task_checker_only_removes_registered_fixed_gripper_pair() -> None:
    robot = MuJoCoPanda.create(obstacles=())
    checker, excluded = make_integrated_task_checker(robot)

    assert excluded > 0
    assert any(pair.kind == "self_collision" for pair in checker.pairs)
    for pair in checker.pairs:
        bodies = frozenset(
            (
                robot.body_name_for_geom(pair.geom1),
                robot.body_name_for_geom(pair.geom2),
            )
        )
        assert bodies != FIXED_GRIPPER_ALLOWED_BODY_PAIR


def test_integrated_task_reruns_guard_physics_and_trace(
    task_artifact: Path,
) -> None:
    result = validate_integrated_panda_tasks(task_artifact)

    assert result["valid"] is True
    assert result["cases"] == 1
    assert result["safe_task_success"] == 1
    assert result["target_reached"] == 1
    assert result["physical_safe"] == 1
    assert result["continuous_edges_checked"] > 0
    assert result["braking_boundaries_checked"] > 0
    assert "closed_loop_mujoco_physics_rerun" in result["checks"]


def test_integrated_task_rejects_resigned_trace_tampering(
    task_artifact: Path,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "tampered_trace"
    shutil.copytree(task_artifact, copied)
    trace_path = copied / "traces" / "free_space_short_smoke.npz"
    with np.load(trace_path, allow_pickle=False) as archive:
        values = {key: np.asarray(archive[key]).copy() for key in archive.files}
    values["actual_positions"][0, 0] += 0.1
    np.savez_compressed(trace_path, **values)
    _write_manifest(copied)

    with pytest.raises(ValueError, match="trace mismatch"):
        validate_integrated_panda_tasks(copied)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"profiles": ("missing",)},
        {"planner_max_iterations": 0},
        {"collision_resolution_rad": 0.0},
        {"settle_action_steps": True},
        {"goal_tolerance_rad": -0.1},
    ],
)
def test_integrated_task_rejects_invalid_config(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        IntegratedPandaTaskConfig(**kwargs)
