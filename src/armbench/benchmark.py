"""Reproducible planning and tracking benchmark orchestration."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from armbench.collision import CollisionChecker
from armbench.control import DiscreteLQR, PDController, simulate_tracking
from armbench.geometry import Sphere, point_to_segment_distance
from armbench.model import RobotModel
from armbench.planners import RRTConnect, RRTStar
from armbench.postprocess import shortcut_path, time_parameterize
from armbench.result import PlanResult
from armbench.scenario import SCENARIO_VERSION, Scenario, benchmark_scenarios
from armbench.visualization import (
    plot_control_summary,
    plot_planning_summary,
    plot_scene_path,
    plot_tracking_trace,
)

LogFunction = Callable[[str], None]


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"schema_version", "scenarios", "seeds", "collision", "planners"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"configuration is missing keys: {sorted(missing)}")
    if config.get("scenario_version") != SCENARIO_VERSION:
        raise ValueError(
            f"config scenario_version must be {SCENARIO_VERSION!r}, "
            f"got {config.get('scenario_version')!r}"
        )
    known_scenarios = benchmark_scenarios()
    unknown = set(config["scenarios"]).difference(known_scenarios)
    if unknown:
        raise ValueError(f"unknown scenarios: {sorted(unknown)}")
    return config


def parse_seed_spec(specification: str) -> list[int]:
    """Parse `0:30`, `0,7,19`, or a single integer."""

    text = specification.strip()
    if ":" in text:
        fields = [int(field) if field else None for field in text.split(":")]
        if len(fields) not in (2, 3):
            raise ValueError("seed range must be start:stop or start:stop:step")
        start = 0 if fields[0] is None else fields[0]
        if fields[1] is None:
            raise ValueError("seed range requires a stop value")
        step = 1 if len(fields) == 2 or fields[2] is None else fields[2]
        seeds = list(range(start, fields[1], step))
    else:
        seeds = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("at least one nonnegative seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed list cannot contain duplicates")
    return seeds


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def _git_metadata(repository: Path) -> dict[str, object]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "no-commit"
    try:
        status = run("status", "--short")
    except (subprocess.CalledProcessError, FileNotFoundError):
        status = "unavailable"
    return {"commit": commit, "dirty": bool(status), "status_short": status.splitlines()}


def environment_metadata(repository: Path) -> dict[str, object]:
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": {
            "armbench": _package_version("armbench"),
            "numpy": _package_version("numpy"),
            "matplotlib": _package_version("matplotlib"),
            "pytest": _package_version("pytest"),
        },
        "git": _git_metadata(repository),
    }


def _checker(
    robot: RobotModel, scenario: Scenario, collision_config: dict[str, object]
) -> CollisionChecker:
    return CollisionChecker(
        robot,
        scenario.obstacles,
        link_radius=float(collision_config["link_radius"]),
        safety_margin=float(collision_config["safety_margin"]),
        resolution=float(collision_config["resolution"]),
    )


def _planner(
    name: str,
    checker: CollisionChecker,
    rng: np.random.Generator,
    planner_configs: dict[str, object],
) -> RRTConnect | RRTStar:
    parameters = dict(planner_configs[name])
    if name == "rrt_connect":
        return RRTConnect(checker, rng, **parameters)
    if name == "rrt_star":
        return RRTStar(checker, rng, **parameters)
    raise ValueError(f"unknown planner: {name}")


def _wilson_interval(successes: int, trials: int, z: float = 1.959964) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 0.0
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


def aggregate_planning(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["scenario"]), str(row["planner"]))].append(row)
    aggregate: list[dict[str, object]] = []
    for (scenario, planner), trials in groups.items():
        successful = [row for row in trials if row["status"] == "success"]
        lower, upper = _wilson_interval(len(successful), len(trials))
        latencies = np.asarray([float(row["elapsed_ms"]) for row in trials])
        aggregate.append(
            {
                "scenario": scenario,
                "planner": planner,
                "trials": len(trials),
                "successes": len(successful),
                "success_rate": len(successful) / len(trials),
                "success_wilson95_low": lower,
                "success_wilson95_high": upper,
                "latency_p50_ms": float(np.percentile(latencies, 50)),
                "latency_p95_ms": float(np.percentile(latencies, 95)),
                "path_length_mean": _mean(
                    float(row["path_length"]) for row in successful
                ),
                "smoothed_length_mean": _mean(
                    float(row["smoothed_length"]) for row in successful
                ),
                "collision_queries_mean": _mean(
                    float(row["collision_queries"]) for row in trials
                ),
                "edge_queries_mean": _mean(
                    float(row["edge_queries"]) for row in trials
                ),
            }
        )
    return aggregate


def run_planning_benchmark(
    config: dict[str, object],
    run_directory: Path,
    seeds: Sequence[int],
    *,
    make_figures: bool,
    log: LogFunction,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    robot = RobotModel.panda()
    scenarios = benchmark_scenarios()
    collision_config = dict(config["collision"])
    planner_configs = dict(config["planners"])
    shortcut_config = dict(config.get("shortcut", {"attempts": 100}))
    rows: list[dict[str, object]] = []
    figure_groups: set[tuple[str, str]] = set()
    selected_scenarios = [str(name) for name in config["scenarios"]]
    planner_names = list(planner_configs)
    total = len(selected_scenarios) * len(planner_names) * len(seeds)
    completed = 0

    for scenario_name in selected_scenarios:
        scenario = scenarios[scenario_name]
        for planner_name in planner_names:
            log(f"planning: {scenario_name} / {planner_name} / {len(seeds)} seeds")
            for seed in seeds:
                checker = _checker(robot, scenario, collision_config)
                planner = _planner(
                    planner_name,
                    checker,
                    np.random.default_rng(seed),
                    planner_configs,
                )
                result = planner.plan(scenario.start, scenario.goal)
                completed += 1
                row: dict[str, object] = {
                    "scenario": scenario_name,
                    "planner": planner_name,
                    "seed": seed,
                    "status": result.status.value,
                    "detail": result.detail,
                    "elapsed_ms": result.elapsed_s * 1000.0,
                    "iterations": result.iterations,
                    "nodes": result.nodes,
                    "collision_queries": result.collision_queries,
                    "edge_queries": result.edge_queries,
                    "waypoints": len(result.path) if result.success else None,
                    "path_length": result.path_length,
                    "smoothed_waypoints": None,
                    "smoothed_length": None,
                    "shortcut_ms": None,
                    "shortcut_accepted": None,
                    "verified": False,
                }
                stem = f"{scenario_name}__{planner_name}__seed_{seed:03d}"
                if result.success:
                    smoothing_checker = _checker(robot, scenario, collision_config)
                    smoothing_rng = np.random.default_rng(
                        np.random.SeedSequence([seed, 0x5A17])
                    )
                    smoothed = shortcut_path(
                        result.path,
                        smoothing_checker,
                        smoothing_rng,
                        attempts=int(shortcut_config["attempts"]),
                    )
                    verified = smoothing_checker.path_is_valid(smoothed.path)
                    if not verified:
                        raise RuntimeError(f"post-processed path failed verification: {stem}")
                    row.update(
                        {
                            "smoothed_waypoints": len(smoothed.path),
                            "smoothed_length": smoothed.smoothed_length,
                            "shortcut_ms": smoothed.elapsed_s * 1000.0,
                            "shortcut_accepted": smoothed.accepted,
                            "verified": True,
                        }
                    )
                    _write_json(
                        run_directory / "paths" / f"{stem}.json",
                        {
                            "scenario": scenario.to_dict(),
                            "planner": planner_name,
                            "seed": seed,
                            "raw_path": [q.tolist() for q in result.path],
                            "smoothed_path": [q.tolist() for q in smoothed.path],
                        },
                    )
                    group = (scenario_name, planner_name)
                    if make_figures and group not in figure_groups:
                        plot_scene_path(
                            robot,
                            scenario,
                            smoothed.path,
                            run_directory / "figures" / f"path__{scenario_name}__{planner_name}.png",
                            title=f"{scenario_name} | {planner_name} | seed {seed}",
                        )
                        figure_groups.add(group)
                else:
                    _write_json(
                        run_directory / "failures" / f"{stem}.json",
                        {
                            "scenario": scenario.to_dict(),
                            "planner": planner_name,
                            "seed": seed,
                            "status": result.status.value,
                            "detail": result.detail,
                            "elapsed_ms": result.elapsed_s * 1000.0,
                            "iterations": result.iterations,
                            "nodes": result.nodes,
                        },
                    )
                rows.append(row)
                if completed % 10 == 0 or completed == total:
                    log(f"planning progress: {completed}/{total}")

    aggregate = aggregate_planning(rows)
    _write_csv(run_directory / "per_trial.csv", rows)
    if make_figures:
        plot_planning_summary(aggregate, run_directory / "figures" / "planning_summary.png")
    return rows, aggregate


def aggregate_control(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (str(row["controller"]), int(row["delay_ms"]), float(row["inertia_scale"]))
        ].append(row)
    aggregate: list[dict[str, object]] = []
    for (controller, delay, inertia), trials in groups.items():
        rmse = np.asarray([float(row["rmse"]) for row in trials])
        max_error = np.asarray([float(row["max_error"]) for row in trials])
        settled = [
            float(row["settling_time_s"])
            for row in trials
            if row["settling_time_s"] is not None
        ]
        aggregate.append(
            {
                "controller": controller,
                "delay_ms": delay,
                "inertia_scale": inertia,
                "trials": len(trials),
                "rmse_mean": float(np.mean(rmse)),
                "rmse_p95": float(np.percentile(rmse, 95)),
                "max_error_p95": float(np.percentile(max_error, 95)),
                "settling_rate": len(settled) / len(trials),
                "settling_time_mean_s": _mean(settled),
                "saturation_count_mean": _mean(
                    float(row["saturation_count"]) for row in trials
                ),
                "invalid_state_samples_mean": _mean(
                    float(row["invalid_state_samples"]) for row in trials
                ),
                "joint_limit_violation_samples_mean": _mean(
                    float(row["joint_limit_violation_samples"]) for row in trials
                ),
                "collision_violation_samples_mean": _mean(
                    float(row["collision_violation_samples"]) for row in trials
                ),
                "invalid_edge_intervals_mean": _mean(
                    float(row["invalid_edge_intervals"]) for row in trials
                ),
            }
        )
    return aggregate


def run_control_benchmark(
    config: dict[str, object],
    run_directory: Path,
    control_seeds: Sequence[int],
    *,
    make_figures: bool,
    log: LogFunction,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    control_config = dict(config["control"])
    robot = RobotModel.panda()
    scenario = benchmark_scenarios()[str(control_config["scenario"])]
    collision_config = dict(config["collision"])
    checker = _checker(robot, scenario, collision_config)
    planning_seed = int(control_config["planning_seed"])
    planner_parameters = dict(config["planners"])["rrt_connect"]
    planned = RRTConnect(
        checker, np.random.default_rng(planning_seed), **dict(planner_parameters)
    ).plan(scenario.start, scenario.goal)
    if not planned.success:
        raise RuntimeError("control reference planner did not find a path")
    smoothed = shortcut_path(
        planned.path,
        _checker(robot, scenario, collision_config),
        np.random.default_rng(np.random.SeedSequence([planning_seed, 0xC017])),
        attempts=int(dict(config["shortcut"])["attempts"]),
    )
    trajectory_config = dict(config["trajectory"])
    trajectory = time_parameterize(
        smoothed.path,
        robot.velocity_limits,
        control_dt=float(trajectory_config["control_dt"]),
        speed_scale=float(trajectory_config["speed_scale"]),
    )
    _write_json(
        run_directory / "control_reference.json",
        {
            "scenario": scenario.to_dict(),
            "planning_seed": planning_seed,
            "smoothed_path": [q.tolist() for q in smoothed.path],
            "trajectory_duration_s": trajectory.duration,
            "sample_count": len(trajectory.times),
            "segment_durations": trajectory.segment_durations.tolist(),
        },
    )

    dt = float(trajectory_config["control_dt"])
    gains = dict(control_config["gains"])
    controllers = (
        PDController.create(robot.dof, **dict(gains["pd"])),
        DiscreteLQR.create(robot.dof, dt=dt, **dict(gains["lqr"])),
    )
    delays = [int(value) for value in control_config["delays_ms"]]
    inertias = [float(value) for value in control_config["inertia_scales"]]
    rows: list[dict[str, object]] = []
    total = len(controllers) * len(delays) * len(inertias) * len(control_seeds)
    completed = 0
    for controller_index, controller in enumerate(controllers):
        for delay in delays:
            for inertia_index, inertia in enumerate(inertias):
                for seed in control_seeds:
                    rng = np.random.default_rng(
                        np.random.SeedSequence(
                            [seed, controller_index, delay, inertia_index, 0x7A4C]
                        )
                    )
                    result = simulate_tracking(
                        trajectory,
                        controller,
                        rng,
                        delay_ms=delay,
                        inertia_scale=inertia,
                        acceleration_limits=control_config["acceleration_limits"],
                        damping=float(control_config["damping"]),
                        measurement_noise_std=float(
                            control_config["measurement_noise_std"]
                        ),
                        process_noise_std=float(control_config["process_noise_std"]),
                        hold_time_s=float(control_config["hold_time_s"]),
                        settling_threshold=float(control_config["settling_threshold"]),
                        checker=_checker(robot, scenario, collision_config),
                    )
                    metrics = result.metrics()
                    row = {
                        "controller": result.controller,
                        "delay_ms": result.delay_ms,
                        "inertia_scale": result.inertia_scale,
                        "seed": seed,
                        "rmse": result.rmse,
                        "per_joint_rmse": json.dumps(
                            result.per_joint_rmse.tolist(), separators=(",", ":")
                        ),
                        "max_error": result.max_error,
                        "settling_time_s": result.settling_time_s,
                        "saturation_count": result.saturation_count,
                        "invalid_state_samples": result.invalid_state_samples,
                        "joint_limit_violation_samples": (
                            result.joint_limit_violation_samples
                        ),
                        "collision_violation_samples": (
                            result.collision_violation_samples
                        ),
                        "invalid_edge_intervals": result.invalid_edge_intervals,
                    }
                    rows.append(row)
                    if seed == control_seeds[0]:
                        stem = (
                            f"{controller.name}__delay_{delay:03d}ms__"
                            f"load_{inertia:g}__seed_{seed:03d}"
                        )
                        np.savez_compressed(
                            run_directory / "control_traces" / f"{stem}.npz",
                            times=result.times,
                            desired_positions=result.desired_positions,
                            actual_positions=result.actual_positions,
                            commands=result.commands,
                        )
                        if (
                            make_figures
                            and delay == max(delays)
                            and inertia == max(inertias)
                        ):
                            plot_tracking_trace(
                                result,
                                trajectory.duration,
                                run_directory / "figures" / f"tracking__{controller.name}.png",
                            )
                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        log(f"control progress: {completed}/{total}")
    aggregate = aggregate_control(rows)
    _write_csv(run_directory / "control_per_trial.csv", rows)
    if make_figures:
        plot_control_summary(aggregate, run_directory / "figures" / "control_summary.png")
    return rows, aggregate


def _most_unique_link_midpoint(
    robot: RobotModel, target: np.ndarray, other: np.ndarray
) -> np.ndarray:
    target_points = robot.forward_points(target)
    other_points = robot.forward_points(other)
    candidates = 0.5 * (target_points[:-1] + target_points[1:])
    clearances = [
        min(
            point_to_segment_distance(candidate, start, end)
            for start, end in zip(other_points[:-1], other_points[1:])
        )
        for candidate in candidates
    ]
    return candidates[int(np.argmax(clearances))]


def run_failure_diagnostics(
    config: dict[str, object], run_directory: Path
) -> list[dict[str, object]]:
    robot = RobotModel.panda()
    scenario = benchmark_scenarios()["free_space"]
    collision_config = dict(config["collision"])
    radius = 0.01
    start_center = _most_unique_link_midpoint(robot, scenario.start, scenario.goal)
    goal_center = _most_unique_link_midpoint(robot, scenario.goal, scenario.start)
    cases = [
        (
            "invalid_start",
            "start_in_collision",
            (Sphere(start_center, radius, "diagnostic_start"),),
            1.0,
        ),
        (
            "invalid_goal",
            "goal_in_collision",
            (Sphere(goal_center, radius, "diagnostic_goal"),),
            1.0,
        ),
        ("zero_deadline", "timeout", (), 0.0),
    ]
    diagnostics: list[dict[str, object]] = []
    for name, expected, obstacles, timeout in cases:
        diagnostic_scenario = Scenario(
            name=name,
            start=scenario.start,
            goal=scenario.goal,
            obstacles=obstacles,
            description="Synthetic failure-mode diagnostic, excluded from benchmark metrics.",
        )
        checker = _checker(robot, diagnostic_scenario, collision_config)
        result = RRTConnect(
            checker,
            np.random.default_rng(0),
            max_iterations=10,
            timeout_s=timeout,
        ).plan(diagnostic_scenario.start, diagnostic_scenario.goal)
        record = {
            "name": name,
            "expected_status": expected,
            "actual_status": result.status.value,
            "passed": result.status.value == expected,
            "detail": result.detail,
            "scenario": diagnostic_scenario.to_dict(),
        }
        diagnostics.append(record)
        _write_json(run_directory / "failures" / f"diagnostic__{name}.json", record)
    if not all(bool(record["passed"]) for record in diagnostics):
        raise RuntimeError("one or more failure-mode diagnostics failed")
    return diagnostics


def _summary_markdown(
    planning: list[dict[str, object]], control: list[dict[str, object]]
) -> str:
    lines = [
        "# Benchmark summary",
        "",
        "Planning latency percentiles include both successful and failed trials; path",
        "statistics use successful trials only. Success intervals are Wilson 95% CIs.",
        "",
        "| Scenario | Planner | Success (95% CI) | P50 ms | P95 ms | Mean path | Mean smoothed |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in planning:
        path = row["path_length_mean"]
        smooth = row["smoothed_length_mean"]
        lines.append(
            f"| {row['scenario']} | {row['planner']} | "
            f"{float(row['success_rate']):.1%} "
            f"[{float(row['success_wilson95_low']):.1%}, "
            f"{float(row['success_wilson95_high']):.1%}] | "
            f"{float(row['latency_p50_ms']):.1f} | "
            f"{float(row['latency_p95_ms']):.1f} | "
            f"{'n/a' if path is None else f'{float(path):.3f}'} | "
            f"{'n/a' if smooth is None else f'{float(smooth):.3f}'} |"
        )
    if control:
        lines.extend(
            [
                "",
                "| Controller | Delay ms | Load | RMSE mean | RMSE P95 | Collision samples | Limit samples | Invalid edges |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in control:
            lines.append(
                f"| {row['controller']} | {row['delay_ms']} | "
                f"{float(row['inertia_scale']):.2f} | "
                f"{float(row['rmse_mean']):.4f} | {float(row['rmse_p95']):.4f} | "
                f"{float(row['collision_violation_samples_mean']):.1f} | "
                f"{float(row['joint_limit_violation_samples_mean']):.1f} | "
                f"{float(row['invalid_edge_intervals_mean']):.1f} |"
            )
    return "\n".join(lines) + "\n"


def execute_benchmark(
    config_path: Path,
    output_root: Path,
    *,
    run_id: str | None = None,
    planning_seeds: Sequence[int] | None = None,
    control_seeds: Sequence[int] | None = None,
    skip_control: bool = False,
    make_figures: bool = True,
) -> Path:
    config = load_config(config_path)
    planning_seeds = list(config["seeds"] if planning_seeds is None else planning_seeds)
    if not planning_seeds:
        raise ValueError("planning seed list cannot be empty")
    control_config = dict(config.get("control", {}))
    control_seeds = list(
        control_config.get("seeds", [0]) if control_seeds is None else control_seeds
    )
    if not skip_control and not control_config:
        raise ValueError("control configuration is required unless --skip-control is used")
    identifier = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not identifier or any(character in identifier for character in '<>:"/\\|?*'):
        raise ValueError("run_id contains characters that are invalid in a directory name")
    run_directory = output_root / identifier
    run_directory.mkdir(parents=True, exist_ok=False)
    for child in ("paths", "figures", "failures", "control_traces"):
        (run_directory / child).mkdir()

    log_path = run_directory / "run.log"

    def log(message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")

    snapshot = dict(config)
    snapshot["resolved_planning_seeds"] = list(planning_seeds)
    snapshot["resolved_control_seeds"] = list(control_seeds)
    snapshot["source_config"] = str(config_path.resolve())
    _write_json(run_directory / "config.json", snapshot)
    repository = Path(__file__).resolve().parents[2]
    _write_json(run_directory / "environment.json", environment_metadata(repository))
    log(f"run started: {identifier}")
    diagnostics = run_failure_diagnostics(config, run_directory)
    log("failure diagnostics: 3/3 passed")
    _, planning_aggregate = run_planning_benchmark(
        config,
        run_directory,
        planning_seeds,
        make_figures=make_figures,
        log=log,
    )
    control_aggregate: list[dict[str, object]] = []
    if not skip_control:
        _, control_aggregate = run_control_benchmark(
            config,
            run_directory,
            control_seeds,
            make_figures=make_figures,
            log=log,
        )
    aggregate = {
        "planning": planning_aggregate,
        "control": control_aggregate,
        "failure_diagnostics": diagnostics,
    }
    _write_json(run_directory / "aggregate.json", aggregate)
    summary_path = run_directory / "summary.md"
    temporary_summary = summary_path.with_suffix(".md.tmp")
    with temporary_summary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(_summary_markdown(planning_aggregate, control_aggregate))
    os.replace(temporary_summary, summary_path)
    log("run completed")
    return run_directory
