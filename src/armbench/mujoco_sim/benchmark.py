"""Reproducible planning and rigid-body experiments for the Menagerie Panda."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Callable, Iterable, Sequence

import imageio.v2 as imageio
import numpy as np

from armbench.benchmark import environment_metadata
from armbench.geometry import Sphere, point_to_segment_distance
from armbench.mujoco_sim.collision import MuJoCoCollisionChecker
from armbench.mujoco_sim.execution import execute_trajectory
from armbench.mujoco_sim.model import MENAGERIE_COMMIT, MuJoCoPanda
from armbench.mujoco_sim.scenarios import MUJOCO_SCENARIO_VERSION, mujoco_scenarios
from armbench.planners import RRTConnect, RRTStar
from armbench.postprocess import shortcut_path, time_parameterize
from armbench.result import PlanResult, path_length
from armbench.scenario import Scenario

LogFunction = Callable[[str], None]


def load_mujoco_config(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "schema_version",
        "scenario_version",
        "scenarios",
        "seeds",
        "planning",
        "execution",
        "collision_consistency",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"MuJoCo config is missing keys: {sorted(missing)}")
    if config["scenario_version"] != MUJOCO_SCENARIO_VERSION:
        raise ValueError(
            f"scenario_version must be {MUJOCO_SCENARIO_VERSION!r}, "
            f"got {config['scenario_version']!r}"
        )
    known = mujoco_scenarios()
    unknown = set(config["scenarios"]).difference(known)
    if unknown:
        raise ValueError(f"unknown MuJoCo scenarios: {sorted(unknown)}")
    seeds = [int(seed) for seed in config["seeds"]]
    if not seeds or any(seed < 0 for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique nonnegative integers")
    planning = dict(config["planning"])
    clearances = [float(value) for value in planning["clearances_m"]]
    if not clearances or any(value < 0.0 for value in clearances):
        raise ValueError("planning clearances must be nonnegative")
    planners = dict(planning["planners"])
    if not planners or set(planners).difference({"rrt_connect", "rrt_star"}):
        raise ValueError("planning must configure rrt_connect and/or rrt_star")
    execution = dict(config["execution"])
    execution_scenarios = set(execution["scenarios"])
    if not execution_scenarios.issubset(known):
        raise ValueError("execution contains an unknown scenario")
    profile_names = [str(profile["name"]) for profile in execution["profiles"]]
    if not profile_names or len(profile_names) != len(set(profile_names)):
        raise ValueError("execution profile names must be unique")
    return config


def inflate_obstacles(
    obstacles: Sequence[Sphere], clearance_m: float
) -> tuple[Sphere, ...]:
    if clearance_m < 0.0:
        raise ValueError("clearance cannot be negative")
    return tuple(
        Sphere(obstacle.center, obstacle.radius + clearance_m, obstacle.label)
        for obstacle in obstacles
    )


def validate_mujoco_scenarios(
    config: dict[str, object],
) -> list[dict[str, object]]:
    scenarios = mujoco_scenarios()
    planning = dict(config["planning"])
    resolution = float(planning["collision_resolution_rad"])
    records: list[dict[str, object]] = []
    for name in config["scenarios"]:
        scenario = scenarios[str(name)]
        for clearance in planning["clearances_m"]:
            clearance_m = float(clearance)
            robot = MuJoCoPanda.create(
                obstacles=inflate_obstacles(scenario.obstacles, clearance_m)
            )
            checker = MuJoCoCollisionChecker(robot, resolution=resolution)
            start_valid = checker.configuration_is_valid(scenario.start)
            goal_valid = checker.configuration_is_valid(scenario.goal)
            direct_valid = checker.edge_is_valid(scenario.start, scenario.goal)
            expected_direct = scenario.name == "free_space"
            records.append(
                {
                    "scenario": scenario.name,
                    "clearance_mm": int(round(clearance_m * 1000.0)),
                    "start_valid": start_valid,
                    "goal_valid": goal_valid,
                    "direct_edge_valid": direct_valid,
                    "passed": bool(
                        start_valid
                        and goal_valid
                        and direct_valid == expected_direct
                    ),
                }
            )
    return records


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
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


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 0.0
    z = 1.959964
    rate = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (rate + z**2 / (2.0 * trials)) / denominator
    radius = z * np.sqrt(
        rate * (1.0 - rate) / trials + z**2 / (4.0 * trials**2)
    ) / denominator
    return float(max(0.0, center - radius)), float(min(1.0, center + radius))


def _mean(values: Iterable[float]) -> float | None:
    data = list(values)
    return None if not data else float(np.mean(data))


def _planner(
    name: str,
    checker: MuJoCoCollisionChecker,
    seed: int,
    parameters: dict[str, object],
) -> RRTConnect | RRTStar:
    rng = np.random.default_rng(seed)
    if name == "rrt_connect":
        return RRTConnect(checker, rng, **parameters)
    if name == "rrt_star":
        return RRTStar(checker, rng, **parameters)
    raise ValueError(f"unknown planner: {name}")


def _run_planning(
    config: dict[str, object], run_directory: Path, log: LogFunction
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    planning = dict(config["planning"])
    resolution = float(planning["collision_resolution_rad"])
    attempts = int(planning["shortcut_attempts"])
    scenarios = mujoco_scenarios()
    rows: list[dict[str, object]] = []
    for scenario_name in config["scenarios"]:
        scenario = scenarios[str(scenario_name)]
        exact_robot = MuJoCoPanda.create(obstacles=scenario.obstacles)
        exact_checker = MuJoCoCollisionChecker(exact_robot, resolution=resolution)
        for clearance in planning["clearances_m"]:
            clearance_m = float(clearance)
            planning_robot = MuJoCoPanda.create(
                obstacles=inflate_obstacles(scenario.obstacles, clearance_m)
            )
            checker = MuJoCoCollisionChecker(planning_robot, resolution=resolution)
            for planner_name, raw_parameters in dict(planning["planners"]).items():
                parameters = dict(raw_parameters)
                for seed in config["seeds"]:
                    seed_value = int(seed)
                    planner = _planner(
                        str(planner_name), checker, seed_value, parameters
                    )
                    result = planner.plan(scenario.start, scenario.goal)
                    smoothed_path: list[np.ndarray] = []
                    smoothed_length: float | None = None
                    exact_mesh_valid: bool | None = None
                    if result.success:
                        shortcut = shortcut_path(
                            result.path,
                            checker,
                            np.random.default_rng(seed_value + 100_000),
                            attempts=attempts,
                        )
                        smoothed_path = shortcut.path
                        smoothed_length = shortcut.smoothed_length
                        exact_mesh_valid = exact_checker.path_is_valid(smoothed_path)
                        path_file = (
                            run_directory
                            / "paths"
                            / (
                                f"{scenario.name}__{round(clearance_m * 1000):03d}mm"
                                f"__{planner_name}__seed_{seed_value:03d}.npz"
                            )
                        )
                        path_file.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            path_file,
                            raw=np.asarray(result.path),
                            smoothed=np.asarray(smoothed_path),
                        )
                    rows.append(
                        {
                            "scenario": scenario.name,
                            "clearance_mm": int(round(clearance_m * 1000.0)),
                            "planner": str(planner_name),
                            "seed": seed_value,
                            "status": result.status.value,
                            "elapsed_ms": result.elapsed_s * 1000.0,
                            "iterations": result.iterations,
                            "nodes": result.nodes,
                            "configuration_queries": result.collision_queries,
                            "edge_queries": result.edge_queries,
                            "raw_path_length_rad": result.path_length,
                            "smoothed_path_length_rad": smoothed_length,
                            "exact_mesh_valid": exact_mesh_valid,
                            "detail": result.detail,
                        }
                    )
                    log(
                        "planning "
                        f"{scenario.name} clearance={clearance_m * 1000:.0f}mm "
                        f"{planner_name} seed={seed_value}: {result.status.value} "
                        f"{result.elapsed_s * 1000:.1f}ms"
                    )
    groups: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (str(row["scenario"]), int(row["clearance_mm"]), str(row["planner"]))
        ].append(row)
    aggregates: list[dict[str, object]] = []
    for (scenario, clearance_mm, planner_name), trials in sorted(groups.items()):
        successful = [row for row in trials if row["status"] == "success"]
        lower, upper = _wilson_interval(len(successful), len(trials))
        latencies = np.asarray([float(row["elapsed_ms"]) for row in trials])
        aggregates.append(
            {
                "scenario": scenario,
                "clearance_mm": clearance_mm,
                "planner": planner_name,
                "trials": len(trials),
                "successes": len(successful),
                "success_rate": len(successful) / len(trials),
                "wilson_95_low": lower,
                "wilson_95_high": upper,
                "latency_p50_ms": float(np.percentile(latencies, 50)),
                "latency_p95_ms": float(np.percentile(latencies, 95)),
                "raw_path_length_mean_rad": _mean(
                    float(row["raw_path_length_rad"]) for row in successful
                ),
                "smoothed_path_length_mean_rad": _mean(
                    float(row["smoothed_path_length_rad"]) for row in successful
                ),
                "all_successful_paths_exact_mesh_valid": bool(
                    successful
                    and all(row["exact_mesh_valid"] is True for row in successful)
                ),
            }
        )
    return rows, aggregates


def _capsule_collision(
    robot: MuJoCoPanda,
    q: np.ndarray,
    obstacles: Sequence[Sphere],
    capsule_radius_m: float,
) -> bool:
    points = robot.forward_points(q)
    for obstacle in obstacles:
        for start, end in zip(points[:-1], points[1:]):
            if point_to_segment_distance(obstacle.center, start, end) <= (
                obstacle.radius + capsule_radius_m
            ):
                return True
    return False


def _run_collision_consistency(
    config: dict[str, object], log: LogFunction
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    collision_config = dict(config["collision_consistency"])
    samples = int(collision_config["samples"])
    seed = int(collision_config["seed"])
    capsule_radius = float(collision_config["capsule_radius_m"])
    if samples <= 0 or capsule_radius < 0.0:
        raise ValueError("collision consistency parameters are invalid")
    rows: list[dict[str, object]] = []
    aggregates: list[dict[str, object]] = []
    scenarios = mujoco_scenarios()
    for scenario_index, scenario_name in enumerate(collision_config["scenarios"]):
        scenario = scenarios[str(scenario_name)]
        robot = MuJoCoPanda.create(obstacles=scenario.obstacles)
        checker = MuJoCoCollisionChecker(robot)
        rng = np.random.default_rng(seed + scenario_index)
        counts = {"true_safe": 0, "true_collision": 0, "false_safe": 0, "false_collision": 0}
        for sample_index in range(samples):
            q = robot.sample(rng)
            robot.set_configuration(checker.data, q)
            exact_collision = bool(robot.obstacle_contacts(checker.data))
            capsule_collision = _capsule_collision(
                robot, q, scenario.obstacles, capsule_radius
            )
            if exact_collision and capsule_collision:
                outcome = "true_collision"
            elif not exact_collision and not capsule_collision:
                outcome = "true_safe"
            elif exact_collision:
                outcome = "false_safe"
            else:
                outcome = "false_collision"
            counts[outcome] += 1
            row: dict[str, object] = {
                "scenario": scenario.name,
                "sample": sample_index,
                "exact_mesh_collision": exact_collision,
                "capsule_collision": capsule_collision,
                "outcome": outcome,
            }
            row.update({f"q{joint + 1}": float(value) for joint, value in enumerate(q)})
            rows.append(row)
        aggregates.append(
            {
                "scenario": scenario.name,
                "samples": samples,
                "capsule_radius_mm": capsule_radius * 1000.0,
                **counts,
                "disagreement_rate": (
                    counts["false_safe"] + counts["false_collision"]
                )
                / samples,
            }
        )
        log(f"collision consistency {scenario.name}: {counts}")
    return rows, aggregates


def _reference_path(
    scenario: Scenario,
    clearance_m: float,
    payload_mass: float,
    seed: int,
    planning: dict[str, object],
) -> tuple[list[np.ndarray], PlanResult]:
    robot = MuJoCoPanda.create(
        obstacles=inflate_obstacles(scenario.obstacles, clearance_m),
        payload_mass=payload_mass,
    )
    checker = MuJoCoCollisionChecker(
        robot, resolution=float(planning["collision_resolution_rad"])
    )
    planner = RRTConnect(
        checker,
        np.random.default_rng(seed),
        **dict(planning["planners"]["rrt_connect"]),
    )
    result = planner.plan(scenario.start, scenario.goal)
    if not result.success:
        raise RuntimeError(
            f"reference planning failed for {scenario.name}, "
            f"clearance={clearance_m}, payload={payload_mass}: {result.status.value}"
        )
    shortcut = shortcut_path(
        result.path,
        checker,
        np.random.default_rng(seed + 200_000),
        attempts=int(planning["shortcut_attempts"]),
    )
    return shortcut.path, result


def _render_requested(
    cases: Sequence[dict[str, object]],
    *,
    scenario: str,
    profile: str,
    delay_ms: int,
    payload_mass: float,
) -> bool:
    return any(
        str(case["scenario"]) == scenario
        and str(case["profile"]) == profile
        and int(case["delay_ms"]) == delay_ms
        and np.isclose(float(case["payload_mass"]), payload_mass)
        for case in cases
    )


def _save_video_poster(video_path: Path) -> Path:
    reader = imageio.get_reader(str(video_path))
    try:
        count = reader.count_frames()
        frame = reader.get_data(max(0, count // 2))
    finally:
        reader.close()
    poster = video_path.with_suffix(".png")
    imageio.imwrite(poster, frame)
    return poster


def _run_execution(
    config: dict[str, object],
    run_directory: Path,
    log: LogFunction,
    *,
    make_videos: bool,
) -> list[dict[str, object]]:
    execution = dict(config["execution"])
    planning = dict(config["planning"])
    scenarios = mujoco_scenarios()
    reference_seed = int(execution["planning_seed"])
    render_cases = list(execution.get("render_cases", []))
    path_cache: dict[tuple[str, float, float], tuple[list[np.ndarray], PlanResult]] = {}
    rows: list[dict[str, object]] = []
    velocity_limits = MuJoCoPanda.create().velocity_limits
    for scenario_name in execution["scenarios"]:
        scenario = scenarios[str(scenario_name)]
        for raw_profile in execution["profiles"]:
            profile = dict(raw_profile)
            profile_name = str(profile["name"])
            clearance_m = float(profile["clearance_m"])
            for payload in execution["payload_masses"]:
                payload_mass = float(payload)
                cache_key = (scenario.name, clearance_m, payload_mass)
                if cache_key not in path_cache:
                    path_cache[cache_key] = _reference_path(
                        scenario,
                        clearance_m,
                        payload_mass,
                        reference_seed,
                        planning,
                    )
                path, plan_result = path_cache[cache_key]
                trajectory = time_parameterize(
                    path,
                    velocity_limits,
                    control_dt=float(execution["control_dt_s"]),
                    speed_scale=float(profile["speed_scale"]),
                )
                for delay in execution["delays_ms"]:
                    delay_ms = int(delay)
                    case_name = (
                        f"{scenario.name}__{profile_name}__delay_{delay_ms:03d}ms"
                        f"__payload_{payload_mass:.1f}kg"
                    )
                    video_path = None
                    if make_videos and _render_requested(
                        render_cases,
                        scenario=scenario.name,
                        profile=profile_name,
                        delay_ms=delay_ms,
                        payload_mass=payload_mass,
                    ):
                        video_path = run_directory / "videos" / f"{case_name}.mp4"
                    execution_robot = MuJoCoPanda.create(
                        obstacles=scenario.obstacles,
                        payload_mass=payload_mass,
                        torque_control=True,
                    )
                    result = execute_trajectory(
                        execution_robot,
                        trajectory,
                        delay_ms=delay_ms,
                        control_dt=float(execution["control_dt_s"]),
                        kp=profile["kp"],
                        kd=profile["kd"],
                        warmup_s=float(execution["warmup_s"]),
                        hold_s=float(execution["hold_s"]),
                        goal_tolerance=float(execution["goal_tolerance_rad"]),
                        feedback_mode=str(profile.get("feedback_mode", "delayed")),
                        video_path=video_path,
                        video_fps=int(execution["video_fps"]),
                    )
                    trace_path = run_directory / "traces" / f"{case_name}.npz"
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        trace_path,
                        times=result.times,
                        desired_positions=result.desired_positions,
                        actual_positions=result.actual_positions,
                        applied_torques=result.applied_torques,
                    )
                    metrics = result.metrics()
                    poster_path: Path | None = None
                    if video_path is not None:
                        poster_path = _save_video_poster(video_path)
                    rows.append(
                        {
                            "scenario": scenario.name,
                            "profile": profile_name,
                            "clearance_mm": int(round(clearance_m * 1000.0)),
                            "speed_scale": float(profile["speed_scale"]),
                            "delay_ms": delay_ms,
                            "payload_mass": payload_mass,
                            "trajectory_duration_s": trajectory.duration,
                            "path_length_rad": path_length(path),
                            "planning_elapsed_ms": plan_result.elapsed_s * 1000.0,
                            **metrics,
                            "video_path": (
                                str(video_path.relative_to(run_directory))
                                if video_path is not None
                                else None
                            ),
                            "poster_path": (
                                str(poster_path.relative_to(run_directory))
                                if poster_path is not None
                                else None
                            ),
                            "trace_path": str(trace_path.relative_to(run_directory)),
                        }
                    )
                    log(
                        f"execution {case_name}: safe={result.safe_success} "
                        f"rmse={result.rmse:.4f} contact_steps="
                        f"{result.obstacle_contact_steps}"
                    )
    return rows


def _markdown_summary(
    planning: list[dict[str, object]],
    execution: list[dict[str, object]],
    consistency: list[dict[str, object]],
) -> str:
    lines = [
        "# MuJoCo Panda benchmark summary",
        "",
        "This artifact uses the pinned MuJoCo Menagerie Panda model. Planning "
        "contacts use compiled collision meshes; execution uses 2 ms rigid-body "
        "physics and torque-limited joint PD control.",
        "",
        "## Planning",
        "",
        "| Scenario | Clearance | Planner | Success | P50 ms | P95 ms |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in planning:
        lines.append(
            f"| {row['scenario']} | {row['clearance_mm']} mm | {row['planner']} | "
            f"{row['successes']}/{row['trials']} | "
            f"{float(row['latency_p50_ms']):.1f} | "
            f"{float(row['latency_p95_ms']):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Physics execution",
            "",
            "| Scenario | Profile | Delay | Payload | RMSE rad | Contact steps | "
            "Max force N | Safe |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if execution:
        for row in execution:
            lines.append(
                f"| {row['scenario']} | {row['profile']} | {row['delay_ms']} ms | "
                f"{float(row['payload_mass']):.1f} kg | {float(row['rmse']):.4f} | "
                f"{row['obstacle_contact_steps']} | "
                f"{float(row['max_contact_force_n']):.2f} | "
                f"{row['safe_success']} |"
            )
    else:
        lines.append("| skipped | - | - | - | - | - | - | - |")
    lines.extend(
        [
            "",
            "## Capsule approximation versus mesh contacts",
            "",
            "| Scenario | Samples | False safe | False collision | Disagreement |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in consistency:
        lines.append(
            f"| {row['scenario']} | {row['samples']} | {row['false_safe']} | "
            f"{row['false_collision']} | "
            f"{100.0 * float(row['disagreement_rate']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "`safe_success` requires final joint error within tolerance, zero "
            "environment contact steps, zero self-contact steps, and zero "
            "joint-limit violation steps. This is simulation evidence, not "
            "real-robot validation.",
            "",
        ]
    )
    return "\n".join(lines)


def execute_mujoco_benchmark(
    config_path: Path,
    output_root: Path,
    *,
    run_id: str | None = None,
    planning_seeds: Sequence[int] | None = None,
    collision_samples: int | None = None,
    skip_execution: bool = False,
    make_videos: bool = True,
) -> Path:
    config = load_mujoco_config(config_path)
    if planning_seeds is not None:
        if not planning_seeds or any(seed < 0 for seed in planning_seeds):
            raise ValueError("planning seeds must be nonnegative")
        config["seeds"] = [int(seed) for seed in planning_seeds]
    if collision_samples is not None:
        if collision_samples <= 0:
            raise ValueError("collision_samples must be positive")
        config["collision_consistency"]["samples"] = collision_samples
    resolved_run_id = run_id or datetime.now(timezone.utc).strftime(
        "mujoco_%Y%m%dT%H%M%SZ"
    )
    run_directory = output_root / resolved_run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    log_lines: list[str] = []

    def log(message: str) -> None:
        stamp = datetime.now(timezone.utc).isoformat()
        line = f"[{stamp}] {message}"
        log_lines.append(line)
        print(message, flush=True)

    try:
        validation = validate_mujoco_scenarios(config)
        if not all(record["passed"] for record in validation):
            raise RuntimeError("MuJoCo scenario validation failed")
        _write_json(run_directory / "config.json", config)
        metadata = environment_metadata(Path(__file__).resolve().parents[3])
        metadata["experiment"] = {
            "backend": "mujoco",
            "menagerie_commit": MENAGERIE_COMMIT,
            "scenario_version": MUJOCO_SCENARIO_VERSION,
        }
        metadata["packages"].update(
            {
                "mujoco": version("mujoco"),
                "imageio": version("imageio"),
                "imageio-ffmpeg": version("imageio-ffmpeg"),
            }
        )
        _write_json(run_directory / "environment.json", metadata)
        _write_json(run_directory / "scenario_validation.json", validation)
        planning_rows, planning_aggregate = _run_planning(
            config, run_directory, log
        )
        collision_rows, collision_aggregate = _run_collision_consistency(
            config, log
        )
        execution_rows = (
            []
            if skip_execution
            else _run_execution(
                config, run_directory, log, make_videos=make_videos
            )
        )
        _write_csv(run_directory / "planning_per_trial.csv", planning_rows)
        _write_csv(run_directory / "collision_samples.csv", collision_rows)
        if execution_rows:
            _write_csv(run_directory / "execution_per_trial.csv", execution_rows)
        aggregate = {
            "planning": planning_aggregate,
            "execution": execution_rows,
            "collision_consistency": collision_aggregate,
        }
        _write_json(run_directory / "aggregate.json", aggregate)
        (run_directory / "summary.md").write_text(
            _markdown_summary(
                planning_aggregate, execution_rows, collision_aggregate
            ),
            encoding="utf-8",
            newline="\n",
        )
        log("benchmark complete")
    finally:
        (run_directory / "run.log").write_text(
            "\n".join(log_lines) + ("\n" if log_lines else ""),
            encoding="utf-8",
            newline="\n",
        )
    return run_directory
