"""Reproducible OpenPI-contract action-chunk guard benchmark and probe."""

from __future__ import annotations

import csv
from importlib.metadata import PackageNotFoundError, version
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import socket
from typing import Callable, Sequence

import imageio.v2 as imageio
import mujoco
import numpy as np

from armbench.benchmark import environment_metadata
from armbench.mujoco_sim.benchmark import inflate_obstacles
from armbench.mujoco_sim.collision import MuJoCoCollisionChecker
from armbench.mujoco_sim.execution import execute_trajectory
from armbench.mujoco_sim.model import MuJoCoPanda
from armbench.mujoco_sim.scenarios import MUJOCO_SCENARIO_VERSION, mujoco_scenarios
from armbench.planners import RRTConnect
from armbench.postprocess import shortcut_path, time_parameterize
from armbench.postprocess.time_parameterization import Trajectory
from armbench.vla.guard import ActionChunkGuard, GuardConfig
from armbench.vla.observation import MuJoCoDroidObservationBuilder
from armbench.vla.policy import OpenPIPolicyClient, ScriptedActionChunkPolicy

OPENPI_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
LogFunction = Callable[[str], None]


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def load_vla_config(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "schema_version",
        "scenario_version",
        "openpi_contract",
        "scenarios",
        "prompts",
        "guard",
        "safe_stream",
        "direct_stream",
        "conditions",
        "modes",
        "execution",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"VLA config is missing keys: {sorted(missing)}")
    if config["scenario_version"] != MUJOCO_SCENARIO_VERSION:
        raise ValueError("VLA scenario version does not match the MuJoCo scenarios")
    contract = dict(config["openpi_contract"])
    if contract["upstream_commit"] != OPENPI_COMMIT:
        raise ValueError("OpenPI commit does not match the pinned client contract")
    if contract["action_horizon"] != 15 or contract["action_dim"] != 8:
        raise ValueError("pi05_droid contract must use 15x8 action chunks")
    known = mujoco_scenarios()
    unknown = set(config["scenarios"]).difference(known)
    if unknown:
        raise ValueError(f"unknown VLA scenarios: {sorted(unknown)}")
    if set(config["modes"]) != {"unguarded", "guarded"}:
        raise ValueError("VLA benchmark must compare unguarded and guarded modes")
    condition_names = [str(item["name"]) for item in config["conditions"]]
    if len(condition_names) != len(set(condition_names)):
        raise ValueError("VLA condition names must be unique")
    for raw_condition in config["conditions"]:
        condition = dict(raw_condition)
        if condition.get("stream") not in {"planner_safe", "direct_unsafe"}:
            raise ValueError("VLA condition has an unknown action stream")
        timing_keys = {
            key
            for key in ("latency_ms", "latency_schedule_ms")
            if key in condition
        }
        if len(timing_keys) != 1:
            raise ValueError(
                "each VLA condition needs exactly one latency value or schedule"
            )
        values = (
            [condition["latency_ms"]]
            if "latency_ms" in condition
            else list(condition["latency_schedule_ms"])
        )
        if not values or any(
            not np.isfinite(float(value)) or float(value) < 0.0
            for value in values
        ):
            raise ValueError("VLA latency values must be finite and nonnegative")
    return config


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            default=lambda item: item.tolist()
            if isinstance(item, np.ndarray)
            else float(item),
        )
        handle.write("\n")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _guard_config(config: dict[str, object]) -> GuardConfig:
    raw = dict(config["guard"])
    return GuardConfig(
        control_dt_s=1.0 / float(dict(config["openpi_contract"])["control_hz"]),
        deadline_ms=float(raw["deadline_ms"]),
        joint_velocity_clip_rad_s=float(raw["joint_velocity_clip_rad_s"]),
        latch_on_deadline=bool(raw["latch_on_deadline"]),
        backtracking_scales=tuple(float(value) for value in raw["backtracking_scales"]),
    )


def _pad_actions(actions: np.ndarray, horizon: int) -> np.ndarray:
    remainder = len(actions) % horizon
    if remainder == 0:
        return actions
    padding = np.zeros((horizon - remainder, 8), dtype=float)
    padding[:, 7] = actions[-1, 7]
    return np.vstack([actions, padding])


def _latency_schedule(
    condition: dict[str, object],
    chunk_count: int,
) -> list[float]:
    base = (
        [float(condition["latency_ms"])]
        if "latency_ms" in condition
        else [float(value) for value in condition["latency_schedule_ms"]]
    )
    return [base[index % len(base)] for index in range(chunk_count)]


def _positions_to_actions(
    robot: MuJoCoPanda,
    positions: np.ndarray,
    guard_config: GuardConfig,
) -> np.ndarray:
    velocities = np.diff(positions, axis=0) / guard_config.control_dt_s
    velocity_bounds = np.minimum(
        robot.velocity_limits,
        guard_config.joint_velocity_clip_rad_s,
    )
    if np.any(np.abs(velocities) > velocity_bounds + 1e-9):
        raise RuntimeError("reference stream exceeds DROID joint-velocity bounds")
    actions = np.zeros((len(velocities), 8), dtype=float)
    actions[:, :7] = velocities
    actions[:, 7] = 1.0
    return actions


def _safe_stream(
    scenario_name: str,
    config: dict[str, object],
    guard_config: GuardConfig,
) -> np.ndarray:
    scenario = mujoco_scenarios()[scenario_name]
    raw_guard = dict(config["guard"])
    planning_robot = MuJoCoPanda.create(
        obstacles=inflate_obstacles(
            scenario.obstacles, float(raw_guard["clearance_m"])
        )
    )
    checker = MuJoCoCollisionChecker(
        planning_robot, resolution=float(raw_guard["collision_resolution_rad"])
    )
    safe = dict(config["safe_stream"])
    planner = RRTConnect(
        checker,
        np.random.default_rng(int(safe["planning_seed"])),
        **dict(safe["planner"]),
    )
    plan = planner.plan(scenario.start, scenario.goal)
    if not plan.success:
        raise RuntimeError(f"safe VLA stream planning failed: {plan.status.value}")
    path = shortcut_path(
        plan.path,
        checker,
        np.random.default_rng(int(safe["planning_seed"]) + 300_000),
        attempts=int(safe["shortcut_attempts"]),
    ).path
    timed = time_parameterize(
        path,
        planning_robot.velocity_limits,
        control_dt=guard_config.control_dt_s,
        speed_scale=float(safe["speed_scale"]),
    )
    sample_times = np.arange(
        0.0,
        np.ceil(timed.duration / guard_config.control_dt_s)
        * guard_config.control_dt_s
        + 0.5 * guard_config.control_dt_s,
        guard_config.control_dt_s,
    )
    positions, _ = timed.sample(sample_times)
    positions[-1] = scenario.goal
    return _positions_to_actions(planning_robot, positions, guard_config)


def _direct_stream(
    scenario_name: str,
    robot: MuJoCoPanda,
    config: dict[str, object],
    guard_config: GuardConfig,
) -> np.ndarray:
    scenario = mujoco_scenarios()[scenario_name]
    steps = int(dict(config["direct_stream"])["steps"])
    positions = np.linspace(scenario.start, scenario.goal, steps + 1)
    return _positions_to_actions(robot, positions, guard_config)


def _integrate_actions(
    robot: MuJoCoPanda,
    q_start: np.ndarray,
    actions: np.ndarray,
    guard_config: GuardConfig,
) -> np.ndarray:
    positions = [np.asarray(q_start, dtype=float).copy()]
    for action in actions:
        velocity_bounds = np.minimum(
            robot.velocity_limits,
            guard_config.joint_velocity_clip_rad_s,
        )
        velocity = np.clip(action[:7], -velocity_bounds, velocity_bounds)
        positions.append(positions[-1] + guard_config.control_dt_s * velocity)
    return np.asarray(positions)


def _trajectory(positions: np.ndarray, dt: float) -> Trajectory:
    times = np.arange(len(positions), dtype=float) * dt
    velocities = np.zeros_like(positions)
    velocities[:-1] = np.diff(positions, axis=0) / dt
    return Trajectory(
        times=times,
        positions=positions,
        velocities=velocities,
        segment_durations=np.full(len(positions) - 1, dt),
    )


def _render_case_requested(
    render_cases: Sequence[dict[str, object]],
    scenario: str,
    condition: str,
    mode: str,
) -> bool:
    return any(
        str(case["scenario"]) == scenario
        and str(case["condition"]) == condition
        and str(case["mode"]) == mode
        for case in render_cases
    )


def _protect_stream(
    scenario_name: str,
    prompt: str,
    actions: np.ndarray,
    latency_schedule_ms: Sequence[float],
    mode: str,
    config: dict[str, object],
    observation_directory: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[dict[str, object]],
    list[dict[str, object]],
    str,
    str,
]:
    scenario = mujoco_scenarios()[scenario_name]
    raw_guard = dict(config["guard"])
    guard_config = _guard_config(config)
    guard_robot = MuJoCoPanda.create(
        obstacles=inflate_obstacles(
            scenario.obstacles, float(raw_guard["clearance_m"])
        )
    )
    checker = MuJoCoCollisionChecker(
        guard_robot, resolution=float(raw_guard["collision_resolution_rad"])
    )
    guard = ActionChunkGuard(checker, guard_config)
    observation_robot = MuJoCoPanda.create(
        obstacles=scenario.obstacles,
        vla_cameras=True,
        goal_marker=guard_robot.hand_position(scenario.goal),
    )
    observation_data = mujoco.MjData(observation_robot.model)
    output_actions: list[np.ndarray] = []
    output_positions = [scenario.start.copy()]
    guard_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    first_external = ""
    first_wrist = ""
    horizon = int(dict(config["openpi_contract"])["action_horizon"])
    padded = _pad_actions(actions, horizon)
    scripted_chunks = [
        padded[begin : begin + horizon]
        for begin in range(0, len(padded), horizon)
    ]
    if len(latency_schedule_ms) != len(scripted_chunks):
        raise ValueError("latency schedule must match the action chunk count")
    policy = ScriptedActionChunkPolicy(
        scripted_chunks,
        latencies_ms=latency_schedule_ms,
    )
    with MuJoCoDroidObservationBuilder(observation_robot) as builder:
        for chunk_index in range(len(scripted_chunks)):
            q = output_positions[-1]
            observation_robot.set_configuration(observation_data, q)
            observation = builder.capture(
                observation_data,
                prompt=prompt,
                sequence_id=chunk_index,
                captured_at_s=1000.0 + chunk_index,
            )
            if chunk_index == 0:
                observation_directory.mkdir(parents=True, exist_ok=True)
                external_path = observation_directory / f"{scenario_name}_external.png"
                wrist_path = observation_directory / f"{scenario_name}_wrist.png"
                imageio.imwrite(external_path, observation.exterior_image)
                imageio.imwrite(wrist_path, observation.wrist_image)
                first_external = str(external_path)
                first_wrist = str(wrist_path)
            chunk = policy.infer(observation)
            raw_chunk = chunk.actions
            chunk_latency_ms = chunk.age_ms(observation)
            if mode == "guarded":
                result = guard.guard(
                    q,
                    float(observation.gripper_position[0]),
                    observation,
                    chunk,
                )
                selected = result.guarded_actions
                positions = result.predicted_positions
                guard_rows.append(
                    {
                        "chunk": chunk_index,
                        **result.metrics(),
                    }
                )
                for step, raw_action, guarded_action in zip(
                    result.steps,
                    raw_chunk,
                    selected,
                    strict=True,
                ):
                    action_rows.append(
                        {
                            "chunk": chunk_index,
                            "action": step.index,
                            "source": chunk.source,
                            "deadline_exceeded": result.deadline_exceeded,
                            "fallback_latched": result.fallback_latched,
                            "raw_safe": step.raw_safe,
                            "repaired_safe": step.repaired_safe,
                            "intervened": step.intervened,
                            "scale": step.scale,
                            "reason": step.reason,
                            "raw_action": json.dumps(raw_action.tolist()),
                            "executed_action": json.dumps(
                                guarded_action.tolist()
                            ),
                            "q_before": json.dumps(step.q_before.tolist()),
                            "q_after": json.dumps(step.q_after.tolist()),
                        }
                    )
            else:
                selected = raw_chunk
                positions = _integrate_actions(
                    guard_robot, q, selected, guard_config
                )
                guard_rows.append(
                    {
                        "chunk": chunk_index,
                        "source": chunk.source,
                        "deadline_exceeded": chunk_latency_ms
                        > guard_config.deadline_ms,
                        "fallback_latched": False,
                        "end_to_end_latency_ms": chunk_latency_ms,
                        "guard_latency_ms": 0.0,
                        "horizon": horizon,
                        "unsafe_raw_steps": None,
                        "intervention_steps": 0,
                        "hold_steps": 0,
                        "safe_after_guard": None,
                    }
                )
                for action_index, (raw_action, q_before, q_after) in enumerate(
                    zip(raw_chunk, positions[:-1], positions[1:], strict=True)
                ):
                    action_rows.append(
                        {
                            "chunk": chunk_index,
                            "action": action_index,
                            "source": chunk.source,
                            "deadline_exceeded": chunk.age_ms(observation)
                            > guard_config.deadline_ms,
                            "fallback_latched": False,
                            "raw_safe": None,
                            "repaired_safe": None,
                            "intervened": False,
                            "scale": 1.0,
                            "reason": "guard_disabled",
                            "raw_action": json.dumps(raw_action.tolist()),
                            "executed_action": json.dumps(raw_action.tolist()),
                            "q_before": json.dumps(q_before.tolist()),
                            "q_after": json.dumps(q_after.tolist()),
                        }
                    )
            output_actions.append(selected)
            output_positions.extend(positions[1:])
    return (
        np.vstack(output_actions),
        np.asarray(output_positions),
        guard_rows,
        action_rows,
        first_external,
        first_wrist,
    )


def _summary(rows: list[dict[str, object]], contract: dict[str, object]) -> str:
    lines = [
        "# OpenPI-contract VLA action guard benchmark",
        "",
        "![VLA runtime benchmark overview](overview.png)",
        "",
        "**Policy provenance:** scripted non-learned action streams. No pi0 or "
        "pi0.5 checkpoint was used in this local run.",
        "",
        f"The interface matches `{contract['model_config']}` at OpenPI commit "
        f"`{contract['upstream_commit']}`: two 224x224 RGB images, 8-D state, "
        "language prompt, and 15x8 action chunks.",
        "",
        "| Scenario | Condition | Mode | Interventions | Deadline chunks | "
        "Guard P95 ms | Contacts | Task | Safe |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['condition']} | {row['mode']} | "
            f"{row['intervention_steps']} | {row['deadline_chunks']} | "
            f"{float(row['guard_latency_p95_ms']):.3f} | "
            f"{row['obstacle_contact_steps']} | {row['task_success']} | "
            f"{row['physical_safe']} |"
        )
    lines.extend(
        [
            "",
            "This artifact validates the policy/runtime contract and guard logic. "
            "It is not evidence of learned-policy task performance. Use "
            "`armbench vla-probe` with an official remote OpenPI server before "
            "making a pi0/pi0.5 inference claim.",
            "",
            "A deadline miss latches hold until an explicit runtime reset; later "
            "fresh chunks do not silently resume an open-loop stream.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_overview(
    path: Path,
    run_directory: Path,
    rows: list[dict[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    representative = rows[0]
    exterior = imageio.imread(run_directory / str(representative["external_image"]))
    wrist = imageio.imread(run_directory / str(representative["wrist_image"]))
    fault_rows = [
        row for row in rows if row["condition"] == "fresh_collision_fault"
    ]
    guarded_rows = [row for row in rows if row["mode"] == "guarded"]

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    axes[0, 0].imshow(exterior)
    axes[0, 0].set_title("Exterior RGB input (224 x 224)")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(wrist)
    axes[0, 1].set_title("Wrist RGB input (224 x 224)")
    axes[0, 1].axis("off")

    fault_labels = [
        f"{row['scenario']}\n{row['mode']}" for row in fault_rows
    ]
    fault_colors = [
        "#c93f36" if row["mode"] == "unguarded" else "#27824b"
        for row in fault_rows
    ]
    axes[1, 0].bar(
        np.arange(len(fault_rows)),
        [int(row["obstacle_contact_steps"]) for row in fault_rows],
        color=fault_colors,
    )
    axes[1, 0].set_xticks(np.arange(len(fault_rows)), fault_labels)
    axes[1, 0].set_ylabel("MuJoCo contact steps")
    axes[1, 0].set_title("Injected collision fault")
    axes[1, 0].grid(axis="y", alpha=0.25)

    intervention_labels = [
        f"{row['scenario']}\n{row['condition'].replace('_', ' ')}"
        for row in guarded_rows
    ]
    intervention_rates = [
        100.0 * int(row["intervention_steps"]) / int(row["action_steps"])
        for row in guarded_rows
    ]
    axes[1, 1].barh(
        np.arange(len(guarded_rows)),
        intervention_rates,
        color="#426d91",
    )
    axes[1, 1].set_yticks(np.arange(len(guarded_rows)), intervention_labels)
    axes[1, 1].set_xlabel("Guarded actions (%)")
    axes[1, 1].set_xlim(0.0, 105.0)
    axes[1, 1].set_title("Runtime intervention rate")
    axes[1, 1].grid(axis="x", alpha=0.25)

    figure.suptitle(
        "ArmBench OpenPI-contract runtime assurance",
        fontsize=16,
    )
    figure.text(
        0.5,
        0.015,
        "Policy source: scripted non-learned streams; no pi0/pi0.5 checkpoint used",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def execute_vla_guard_benchmark(
    config_path: Path,
    output_root: Path,
    *,
    run_id: str | None = None,
    scenarios: Sequence[str] | None = None,
    make_videos: bool = True,
) -> Path:
    config = load_vla_config(config_path)
    if scenarios is not None:
        unknown = set(scenarios).difference(mujoco_scenarios())
        if unknown:
            raise ValueError(f"unknown VLA scenarios: {sorted(unknown)}")
        config["scenarios"] = list(scenarios)
    resolved_id = run_id or datetime.now(timezone.utc).strftime(
        "vla_guard_%Y%m%dT%H%M%SZ"
    )
    run_directory = output_root / resolved_id
    metadata = environment_metadata(Path(__file__).resolve().parents[3])
    metadata["packages"].update(
        {
            "imageio": _package_version("imageio"),
            "msgpack": _package_version("msgpack"),
            "mujoco": _package_version("mujoco"),
            "openpi-client": _package_version("openpi-client"),
            "websockets": _package_version("websockets"),
        }
    )
    metadata["vla"] = {
        **dict(config["openpi_contract"]),
        "benchmark_policy_provenance": "scripted_non_learned",
    }
    run_directory.mkdir(parents=True, exist_ok=False)
    log_lines: list[str] = []

    def log(message: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        log_lines.append(line)
        print(message, flush=True)

    try:
        _write_json(run_directory / "config.json", config)
        _write_json(run_directory / "environment.json", metadata)
        guard_config = _guard_config(config)
        execution_config = dict(config["execution"])
        render_cases = list(execution_config["render_cases"])
        rows: list[dict[str, object]] = []
        all_guard_rows: list[dict[str, object]] = []
        all_action_rows: list[dict[str, object]] = []
        for scenario_name in config["scenarios"]:
            scenario = mujoco_scenarios()[str(scenario_name)]
            prompt = str(dict(config["prompts"])[scenario.name])
            physical_robot = MuJoCoPanda.create(obstacles=scenario.obstacles)
            safe_actions = _safe_stream(scenario.name, config, guard_config)
            direct_actions = _direct_stream(
                scenario.name, physical_robot, config, guard_config
            )
            streams = {
                "planner_safe": safe_actions,
                "direct_unsafe": direct_actions,
            }
            for raw_condition in config["conditions"]:
                condition = dict(raw_condition)
                condition_name = str(condition["name"])
                stream_name = str(condition["stream"])
                raw_actions = streams[stream_name]
                horizon = int(
                    dict(config["openpi_contract"])["action_horizon"]
                )
                chunk_count = int(np.ceil(len(raw_actions) / horizon))
                latency_schedule_ms = _latency_schedule(condition, chunk_count)
                raw_positions = _integrate_actions(
                    physical_robot, scenario.start, raw_actions, guard_config
                )
                raw_checker = MuJoCoCollisionChecker(physical_robot, resolution=0.02)
                raw_kinematic_valid = raw_checker.path_is_valid(raw_positions)
                for mode in config["modes"]:
                    mode_name = str(mode)
                    (
                        actions,
                        positions,
                        guard_rows,
                        action_rows,
                        external,
                        wrist,
                    ) = _protect_stream(
                        scenario.name,
                        prompt,
                        raw_actions,
                        latency_schedule_ms,
                        mode_name,
                        config,
                        run_directory / "observations",
                    )
                    for guard_row in guard_rows:
                        all_guard_rows.append(
                            {
                                "scenario": scenario.name,
                                "condition": condition_name,
                                "mode": mode_name,
                                **guard_row,
                            }
                        )
                    for action_row in action_rows:
                        all_action_rows.append(
                            {
                                "scenario": scenario.name,
                                "condition": condition_name,
                                "mode": mode_name,
                                **action_row,
                            }
                        )
                    case_name = f"{scenario.name}__{condition_name}__{mode_name}"
                    video_path = None
                    if make_videos and _render_case_requested(
                        render_cases, scenario.name, condition_name, mode_name
                    ):
                        video_path = run_directory / "videos" / f"{case_name}.mp4"
                    execution_robot = MuJoCoPanda.create(
                        obstacles=scenario.obstacles,
                        torque_control=True,
                        goal_marker=physical_robot.hand_position(scenario.goal),
                    )
                    execution_result = execute_trajectory(
                        execution_robot,
                        _trajectory(positions, guard_config.control_dt_s),
                        control_dt=float(execution_config["controller_dt_s"]),
                        kp=execution_config["kp"],
                        kd=execution_config["kd"],
                        warmup_s=float(execution_config["warmup_s"]),
                        hold_s=float(execution_config["hold_s"]),
                        goal_tolerance=float(execution_config["goal_tolerance_rad"]),
                        video_path=video_path,
                        video_fps=int(execution_config["video_fps"]),
                    )
                    task_goal_error = float(
                        np.max(
                            np.abs(
                                execution_result.actual_positions[-1] - scenario.goal
                            )
                        )
                    )
                    task_success = task_goal_error <= float(
                        execution_config["goal_tolerance_rad"]
                    )
                    physical_safe = bool(
                        execution_result.obstacle_contact_steps == 0
                        and execution_result.self_contact_steps == 0
                        and execution_result.joint_limit_violation_steps == 0
                    )
                    intervention_steps = sum(
                        int(row["intervention_steps"] or 0) for row in guard_rows
                    )
                    deadline_chunks = sum(
                        bool(row["deadline_exceeded"]) for row in guard_rows
                    )
                    fallback_latched_chunks = sum(
                        bool(row["fallback_latched"]) for row in guard_rows
                    )
                    guard_latencies = [
                        float(row["guard_latency_ms"]) for row in guard_rows
                    ]
                    np.savez_compressed(
                        run_directory / f"{case_name}.npz",
                        raw_actions=raw_actions,
                        executed_actions=actions,
                        predicted_positions=positions,
                        actual_positions=execution_result.actual_positions,
                        desired_positions=execution_result.desired_positions,
                        times=execution_result.times,
                    )
                    row = {
                        "scenario": scenario.name,
                        "condition": condition_name,
                        "stream": stream_name,
                        "policy_source": "scripted_non_learned",
                        "actual_openpi_inference": False,
                        "mode": mode_name,
                        "latency_profile_ms": json.dumps(
                            condition.get(
                                "latency_schedule_ms",
                                [condition.get("latency_ms")],
                            )
                        ),
                        "inference_latency_p50_ms": float(
                            np.percentile(latency_schedule_ms, 50)
                        ),
                        "inference_latency_p95_ms": float(
                            np.percentile(latency_schedule_ms, 95)
                        ),
                        "inference_latency_max_ms": max(latency_schedule_ms),
                        "chunks": len(guard_rows),
                        "action_steps": len(actions),
                        "raw_kinematic_valid": raw_kinematic_valid,
                        "executed_kinematic_valid": MuJoCoCollisionChecker(
                            physical_robot, resolution=0.02
                        ).path_is_valid(positions),
                        "intervention_steps": intervention_steps,
                        "deadline_chunks": deadline_chunks,
                        "fallback_latched_chunks": fallback_latched_chunks,
                        "guard_latency_p50_ms": float(
                            np.percentile(guard_latencies, 50)
                        ),
                        "guard_latency_p95_ms": float(
                            np.percentile(guard_latencies, 95)
                        ),
                        "task_goal_error_rad": task_goal_error,
                        "task_success": task_success,
                        "physical_safe": physical_safe,
                        "safe_task_success": bool(task_success and physical_safe),
                        "rmse_rad": execution_result.rmse,
                        "obstacle_contact_steps": execution_result.obstacle_contact_steps,
                        "max_contact_force_n": execution_result.max_contact_force_n,
                        "joint_limit_violation_steps": execution_result.joint_limit_violation_steps,
                        "self_contact_steps": execution_result.self_contact_steps,
                        "external_image": str(
                            Path(external).relative_to(run_directory)
                        ),
                        "wrist_image": str(Path(wrist).relative_to(run_directory)),
                        "video_path": (
                            str(video_path.relative_to(run_directory))
                            if video_path is not None
                            else None
                        ),
                    }
                    rows.append(row)
                    log(
                        f"{case_name}: task={task_success} safe={physical_safe} "
                        f"contacts={execution_result.obstacle_contact_steps} "
                        f"interventions={intervention_steps}"
                    )
        _write_csv(run_directory / "per_case.csv", rows)
        _write_csv(run_directory / "per_chunk.csv", all_guard_rows)
        _write_csv(run_directory / "per_action.csv", all_action_rows)
        _write_json(run_directory / "aggregate.json", rows)
        _write_overview(run_directory / "overview.png", run_directory, rows)
        (run_directory / "summary.md").write_text(
            _summary(rows, dict(config["openpi_contract"])),
            encoding="utf-8",
            newline="\n",
        )
        log("VLA guard benchmark complete")
    finally:
        (run_directory / "run.log").write_text(
            "\n".join(log_lines) + ("\n" if log_lines else ""),
            encoding="utf-8",
            newline="\n",
        )
    return run_directory


def execute_openpi_probe(
    config_path: Path,
    output_directory: Path,
    *,
    host: str,
    port: int,
    scenario_name: str,
    prompt: str,
    api_key: str | None = None,
    connect_timeout_s: float = 3.0,
) -> Path:
    config = load_vla_config(config_path)
    if scenario_name not in mujoco_scenarios():
        raise ValueError(f"unknown VLA scenario: {scenario_name}")
    if output_directory.exists():
        raise FileExistsError(f"output directory already exists: {output_directory}")
    try:
        with socket.create_connection((host, port), timeout=connect_timeout_s):
            pass
    except OSError as error:
        raise ConnectionError(
            f"OpenPI server is not reachable at {host}:{port}"
        ) from error
    scenario = mujoco_scenarios()[scenario_name]
    reference_robot = MuJoCoPanda.create(obstacles=scenario.obstacles)
    observation_robot = MuJoCoPanda.create(
        obstacles=scenario.obstacles,
        vla_cameras=True,
        goal_marker=reference_robot.hand_position(scenario.goal),
    )
    data = mujoco.MjData(observation_robot.model)
    observation_robot.set_configuration(data, scenario.start)
    with MuJoCoDroidObservationBuilder(observation_robot) as builder:
        observation = builder.capture(data, prompt=prompt, sequence_id=0)
    client = OpenPIPolicyClient(host=host, port=port, api_key=api_key)
    chunk = client.infer(observation)
    raw_guard = dict(config["guard"])
    guard_robot = MuJoCoPanda.create(
        obstacles=inflate_obstacles(
            scenario.obstacles, float(raw_guard["clearance_m"])
        )
    )
    result = ActionChunkGuard(
        MuJoCoCollisionChecker(
            guard_robot, resolution=float(raw_guard["collision_resolution_rad"])
        ),
        _guard_config(config),
    ).guard(
        scenario.start,
        float(observation.gripper_position[0]),
        observation,
        chunk,
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    imageio.imwrite(output_directory / "exterior.png", observation.exterior_image)
    imageio.imwrite(output_directory / "wrist.png", observation.wrist_image)
    np.savez_compressed(
        output_directory / "openpi_probe.npz",
        raw_actions=chunk.actions,
        guarded_actions=result.guarded_actions,
        predicted_positions=result.predicted_positions,
        joint_position=observation.joint_position,
    )
    _write_json(
        output_directory / "probe.json",
        {
            "actual_openpi_inference": True,
            "openpi_commit": OPENPI_COMMIT,
            "server": f"{host}:{port}",
            "server_metadata": client.server_metadata,
            "scenario": scenario_name,
            "prompt": prompt,
            "action_shape": list(chunk.actions.shape),
            "policy_source": chunk.source,
            "inference_latency_ms": chunk.inference_latency_ms,
            "server_timing": chunk.server_timing,
            "guard": result.metrics(),
        },
    )
    return output_directory
