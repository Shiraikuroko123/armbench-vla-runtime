import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from armbench.mujoco_sim import MuJoCoCollisionChecker, MuJoCoPanda
from armbench.mujoco_sim.benchmark import execute_mujoco_benchmark, inflate_obstacles
from armbench.mujoco_sim.execution import execute_trajectory
from armbench.mujoco_sim.model import ARM_JOINT_NAMES, default_panda_scene_path
from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.mujoco_sim.viewer import load_pose_sequence
from armbench.postprocess import time_parameterize


@pytest.fixture(scope="module")
def menagerie_path() -> Path:
    try:
        return default_panda_scene_path()
    except FileNotFoundError as error:
        pytest.skip(str(error))


def test_official_panda_model_mapping(menagerie_path: Path) -> None:
    robot = MuJoCoPanda.create(scene_path=menagerie_path)

    assert robot.model.nq == 9
    assert robot.model.nv == 9
    assert robot.model.nu == 8
    assert tuple(
        mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id))
        for joint_id in robot.arm_joint_ids
    ) == ARM_JOINT_NAMES
    np.testing.assert_allclose(
        robot.lower_limits,
        [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    )


@pytest.mark.parametrize("scenario_name", ["single_block", "narrow_gate"])
@pytest.mark.parametrize("clearance_m", [0.0, 0.02])
def test_scenario_endpoints_are_valid_and_direct_edge_is_blocked(
    menagerie_path: Path, scenario_name: str, clearance_m: float
) -> None:
    scenario = mujoco_scenarios()[scenario_name]
    robot = MuJoCoPanda.create(
        scene_path=menagerie_path,
        obstacles=inflate_obstacles(scenario.obstacles, clearance_m),
    )
    checker = MuJoCoCollisionChecker(robot, resolution=0.05)

    assert checker.configuration_is_valid(scenario.start)
    assert checker.configuration_is_valid(scenario.goal)
    assert not checker.edge_is_valid(scenario.start, scenario.goal)


def test_mesh_contact_identifies_blocking_body(menagerie_path: Path) -> None:
    scenario = mujoco_scenarios()["single_block"]
    robot = MuJoCoPanda.create(
        scene_path=menagerie_path, obstacles=scenario.obstacles
    )
    checker = MuJoCoCollisionChecker(robot, resolution=0.02)
    failures = [
        checker.configuration_failure(
            scenario.start + fraction * (scenario.goal - scenario.start)
        )
        for fraction in np.linspace(0.0, 1.0, 101)
    ]

    collisions = [failure for failure in failures if failure and failure.startswith("collision:")]
    assert collisions
    assert any("center_block" in failure for failure in collisions)


def test_short_torque_control_execution_reaches_goal(menagerie_path: Path) -> None:
    scenario = mujoco_scenarios()["free_space"]
    goal = scenario.start.copy()
    goal[:2] += [0.10, 0.05]
    robot = MuJoCoPanda.create(scene_path=menagerie_path, torque_control=True)
    trajectory = time_parameterize(
        [scenario.start, goal],
        robot.velocity_limits,
        control_dt=0.01,
        speed_scale=0.1,
    )

    result = execute_trajectory(
        robot,
        trajectory,
        delay_ms=0,
        control_dt=0.01,
        kp=5.0,
        kd=2.0,
        warmup_s=0.1,
        hold_s=0.4,
    )

    assert result.safe_success
    assert result.final_goal_error <= 0.05
    assert np.all(np.isfinite(result.actual_positions))


def test_offscreen_render_is_nonblank(menagerie_path: Path) -> None:
    scenario = mujoco_scenarios()["single_block"]
    robot = MuJoCoPanda.create(
        scene_path=menagerie_path, obstacles=scenario.obstacles
    )
    data = mujoco.MjData(robot.model)
    robot.set_configuration(data, scenario.start)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.22, 0.08, 0.58]
    camera.distance = 1.55
    camera.azimuth = 135.0
    camera.elevation = -22.0
    renderer = mujoco.Renderer(robot.model, height=240, width=320)
    try:
        renderer.update_scene(data, camera=camera)
        frame = renderer.render()
    finally:
        renderer.close()

    assert frame.shape == (240, 320, 3)
    assert float(frame.std()) > 10.0
    assert np.unique(frame.reshape(-1, 3), axis=0).shape[0] > 100


def test_minimal_mujoco_benchmark_writes_reproducible_artifact(
    menagerie_path: Path, tmp_path: Path
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "configs" / "mujoco_benchmark.json").open(
        encoding="utf-8"
    ) as handle:
        config = json.load(handle)
    config["scenarios"] = ["single_block"]
    config["seeds"] = [0]
    config["planning"]["clearances_m"] = [0.0]
    config["planning"]["planners"] = {
        "rrt_connect": config["planning"]["planners"]["rrt_connect"]
    }
    config["collision_consistency"]["scenarios"] = ["single_block"]
    config["collision_consistency"]["samples"] = 5
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    run_directory = execute_mujoco_benchmark(
        config_path,
        tmp_path / "results",
        run_id="test_run",
        skip_execution=True,
        make_videos=False,
    )

    expected = {
        "aggregate.json",
        "collision_samples.csv",
        "config.json",
        "environment.json",
        "planning_per_trial.csv",
        "run.log",
        "scenario_validation.json",
        "summary.md",
    }
    assert expected.issubset(path.name for path in run_directory.iterdir())


def test_saved_trace_can_be_loaded_for_interactive_debugging(tmp_path: Path) -> None:
    positions = np.arange(21, dtype=float).reshape(3, 7) / 10.0
    times = np.array([2.0, 2.1, 2.2])
    trace_path = tmp_path / "trace.npz"
    np.savez_compressed(trace_path, actual_positions=positions, times=times)

    loaded_positions, loaded_times, selected = load_pose_sequence(trace_path)

    np.testing.assert_allclose(loaded_positions, positions)
    np.testing.assert_allclose(loaded_times, [0.0, 0.1, 0.2])
    assert selected == "actual_positions"
