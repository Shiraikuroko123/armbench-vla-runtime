"""Interactive inspection of Panda poses and recorded ArmBench trajectories."""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from numpy.typing import NDArray

from armbench.mujoco_sim.benchmark import inflate_obstacles
from armbench.mujoco_sim.model import MuJoCoPanda
from armbench.mujoco_sim.scenarios import mujoco_scenarios

FloatArray = NDArray[np.float64]
TRACE_ARRAY_KEYS = (
    "actual_positions",
    "desired_positions",
    "smoothed",
    "raw",
    "raw_positions",
    "legacy_positions",
    "repair_positions",
)


def load_pose_sequence(
    trace_path: Path,
    *,
    array_key: str = "auto",
    episode: int = 0,
) -> tuple[FloatArray, FloatArray | None, str]:
    if not trace_path.is_file():
        raise FileNotFoundError(f"trajectory file not found: {trace_path}")
    with np.load(trace_path, allow_pickle=False) as trace:
        if array_key == "auto":
            selected = next((key for key in TRACE_ARRAY_KEYS if key in trace), None)
            if selected is None:
                raise ValueError(
                    f"trajectory contains none of the supported arrays: {TRACE_ARRAY_KEYS}"
                )
        else:
            if array_key not in TRACE_ARRAY_KEYS:
                raise ValueError(f"unsupported trajectory array: {array_key}")
            if array_key not in trace:
                raise ValueError(f"trajectory does not contain array {array_key!r}")
            selected = array_key
        positions = np.asarray(trace[selected], dtype=float)
        times = np.asarray(trace["times"], dtype=float) if "times" in trace else None
    if episode < 0:
        raise ValueError("trajectory episode must be nonnegative")
    if positions.ndim == 3:
        if episode >= len(positions):
            raise IndexError(
                f"episode {episode} is outside trajectory batch of size {len(positions)}"
            )
        positions = positions[episode]
        if times is not None and times.ndim == 2:
            times = times[episode]
    elif episode != 0:
        raise IndexError("a two-dimensional trajectory only contains episode 0")
    if positions.ndim != 2 or positions.shape[1] != 7 or len(positions) == 0:
        raise ValueError("trajectory positions must have shape (samples, 7)")
    if not np.all(np.isfinite(positions)):
        raise ValueError("trajectory positions contain non-finite values")
    if times is not None:
        if times.shape != (len(positions),) or np.any(np.diff(times) < 0.0):
            raise ValueError("trajectory times must be monotonic and match positions")
        times = times - times[0]
    return positions, times, selected


def inspect_configuration(
    robot: MuJoCoPanda, q: FloatArray
) -> tuple[mujoco.MjData, dict[str, object]]:
    data = mujoco.MjData(robot.model)
    robot.set_configuration(data, q)
    hand_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    obstacle_contacts = [
        {"obstacle": obstacle, "body": body}
        for _, _, obstacle, body in robot.obstacle_contacts(data)
    ]
    self_contacts = [
        {"first_body": first, "second_body": second}
        for _, first, second in robot.self_contacts(data)
    ]
    record = {
        "configuration": q.tolist(),
        "within_joint_limits": robot.within_limits(q),
        "hand_position_m": data.xpos[hand_id].tolist(),
        "obstacle_contacts": obstacle_contacts,
        "self_contacts": self_contacts,
    }
    return data, record


def launch_trajectory_viewer(
    *,
    scenario_name: str,
    pose_name: str = "start",
    clearance_m: float = 0.0,
    payload_mass: float = 0.0,
    trace_path: Path | None = None,
    array_key: str = "auto",
    episode: int = 0,
    frame: int = -1,
    play: bool = False,
    playback_speed: float = 1.0,
    loop: bool = False,
) -> dict[str, object]:
    scenarios = mujoco_scenarios()
    if scenario_name not in scenarios:
        raise ValueError(f"unknown scenario: {scenario_name}")
    if pose_name not in {"start", "goal"}:
        raise ValueError("pose_name must be 'start' or 'goal'")
    if playback_speed <= 0.0:
        raise ValueError("playback_speed must be positive")
    scenario = scenarios[scenario_name]
    positions: FloatArray | None = None
    times: FloatArray | None = None
    selected_array: str | None = None
    if trace_path is None:
        q = scenario.start if pose_name == "start" else scenario.goal
    else:
        positions, times, selected_array = load_pose_sequence(
            trace_path,
            array_key=array_key,
            episode=episode,
        )
        resolved_frame = frame if frame >= 0 else len(positions) - 1
        if resolved_frame >= len(positions):
            raise IndexError(
                f"frame {resolved_frame} is outside trajectory of length {len(positions)}"
            )
        q = positions[resolved_frame]
    robot = MuJoCoPanda.create(
        obstacles=inflate_obstacles(scenario.obstacles, clearance_m),
        payload_mass=payload_mass,
    )
    data, record = inspect_configuration(robot, q)
    record.update(
        {
            "scenario": scenario.name,
            "clearance_mm": clearance_m * 1000.0,
            "payload_mass": payload_mass,
            "trace_path": str(trace_path.resolve()) if trace_path else None,
            "trace_array": selected_array,
            "trace_episode": episode if trace_path else None,
        }
    )
    with mujoco.viewer.launch_passive(robot.model, data) as viewer:
        started = time.perf_counter()
        while viewer.is_running():
            if play and positions is not None:
                elapsed = (time.perf_counter() - started) * playback_speed
                if times is None:
                    sample_time = elapsed * 30.0
                    index = int(sample_time)
                    duration = float(len(positions))
                else:
                    duration = float(times[-1])
                    sample_time = elapsed
                    index = int(np.searchsorted(times, sample_time, side="right") - 1)
                if duration > 0.0 and sample_time >= duration:
                    if loop:
                        started = time.perf_counter()
                        index = 0
                    else:
                        index = len(positions) - 1
                robot.set_configuration(data, positions[max(0, index)])
            viewer.sync()
            time.sleep(1.0 / 60.0)
    return record
