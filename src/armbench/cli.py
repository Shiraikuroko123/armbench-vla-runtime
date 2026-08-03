"""Command-line interface for validation and reproducible experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from armbench.benchmark import execute_benchmark, load_config, parse_seed_spec
from armbench.collision import CollisionChecker
from armbench.model import RobotModel
from armbench.mujoco_sim.benchmark import (
    execute_mujoco_benchmark,
    load_mujoco_config,
    validate_mujoco_scenarios,
)
from armbench.scenario import benchmark_scenarios
from armbench.vla.benchmark import (
    execute_openpi_probe,
    execute_vla_guard_benchmark,
    load_vla_config,
)
from armbench.vla.online_benchmark import execute_vla_online_benchmark


def _validate(config_path: Path) -> int:
    config = load_config(config_path)
    robot = RobotModel.panda()
    collision = dict(config["collision"])
    records = []
    for name in config["scenarios"]:
        scenario = benchmark_scenarios()[str(name)]
        checker = CollisionChecker(
            robot,
            scenario.obstacles,
            link_radius=float(collision["link_radius"]),
            safety_margin=float(collision["safety_margin"]),
            resolution=float(collision["resolution"]),
        )
        start_valid = checker.configuration_is_valid(scenario.start)
        goal_valid = checker.configuration_is_valid(scenario.goal)
        direct_valid = checker.edge_is_valid(scenario.start, scenario.goal)
        expected_direct = scenario.name == "free_space"
        records.append(
            {
                "scenario": scenario.name,
                "start_valid": start_valid,
                "goal_valid": goal_valid,
                "direct_edge_valid": direct_valid,
                "passed": start_valid and goal_valid and direct_valid == expected_direct,
            }
        )
    print(json.dumps(records, indent=2, ensure_ascii=False))
    return 0 if all(record["passed"] for record in records) else 1


def _mujoco_validate(config_path: Path) -> int:
    records = validate_mujoco_scenarios(load_mujoco_config(config_path))
    print(json.dumps(records, indent=2, ensure_ascii=False))
    return 0 if all(record["passed"] for record in records) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="armbench",
        description="VLA action runtime assurance and Panda physics benchmark",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate config and scenario geometry")
    validate.add_argument(
        "--config", type=Path, default=Path("configs/benchmark.json")
    )

    run = subparsers.add_parser("run", help="run planning and control experiments")
    run.add_argument("--config", type=Path, default=Path("configs/benchmark.json"))
    run.add_argument("--output-root", type=Path, default=Path("results"))
    run.add_argument("--run-id")
    run.add_argument(
        "--seeds", help="planning seeds as start:stop or comma-separated integers"
    )
    run.add_argument(
        "--control-seeds", help="control seeds as start:stop or comma-separated integers"
    )
    run.add_argument(
        "--quick",
        action="store_true",
        help="use planning seeds 0:3 and control seeds 0:2",
    )
    run.add_argument("--skip-control", action="store_true")
    run.add_argument("--no-figures", action="store_true")

    mujoco_validate = subparsers.add_parser(
        "mujoco-validate",
        help="validate Menagerie Panda endpoints and direct-edge protocol",
    )
    mujoco_validate.add_argument(
        "--config", type=Path, default=Path("configs/mujoco_benchmark.json")
    )

    mujoco_run = subparsers.add_parser(
        "mujoco-run", help="run mesh planning and MuJoCo rigid-body experiments"
    )
    mujoco_run.add_argument(
        "--config", type=Path, default=Path("configs/mujoco_benchmark.json")
    )
    mujoco_run.add_argument("--output-root", type=Path, default=Path("results"))
    mujoco_run.add_argument("--run-id")
    mujoco_run.add_argument(
        "--seeds", help="planning seeds as start:stop or comma-separated integers"
    )
    mujoco_run.add_argument(
        "--quick",
        action="store_true",
        help="use seeds 0:2, 40 collision samples, and skip physics execution",
    )
    mujoco_run.add_argument("--skip-execution", action="store_true")
    mujoco_run.add_argument("--no-videos", action="store_true")

    mujoco_view = subparsers.add_parser(
        "mujoco-view", help="inspect a Panda pose or replay a saved trajectory"
    )
    mujoco_view.add_argument(
        "--scenario",
        choices=("free_space", "single_block", "narrow_gate"),
        default="single_block",
    )
    mujoco_view.add_argument("--pose", choices=("start", "goal"), default="start")
    mujoco_view.add_argument("--clearance-mm", type=float, default=0.0)
    mujoco_view.add_argument("--payload", type=float, default=0.0)
    mujoco_view.add_argument("--trace", type=Path)
    mujoco_view.add_argument(
        "--array",
        choices=("auto", "actual_positions", "desired_positions", "smoothed", "raw"),
        default="auto",
    )
    mujoco_view.add_argument("--frame", type=int, default=-1)
    mujoco_view.add_argument("--play", action="store_true")
    mujoco_view.add_argument("--speed", type=float, default=1.0)
    mujoco_view.add_argument("--loop", action="store_true")

    vla_guard = subparsers.add_parser(
        "vla-guard-run",
        help="benchmark OpenPI-compatible action chunks with runtime assurance",
    )
    vla_guard.add_argument(
        "--config", type=Path, default=Path("configs/vla_guard_benchmark.json")
    )
    vla_guard.add_argument("--output-root", type=Path, default=Path("results"))
    vla_guard.add_argument("--run-id")
    vla_guard.add_argument(
        "--scenarios",
        nargs="+",
        choices=("single_block", "narrow_gate"),
        help="optional scenario subset",
    )
    vla_guard.add_argument(
        "--quick",
        action="store_true",
        help="run single_block without MP4 rendering",
    )
    vla_guard.add_argument("--no-videos", action="store_true")

    vla_online = subparsers.add_parser(
        "vla-online-run",
        help="compare receding-horizon VLA execution with live MuJoCo feedback",
    )
    vla_online.add_argument(
        "--config", type=Path, default=Path("configs/vla_guard_benchmark.json")
    )
    vla_online.add_argument("--output-root", type=Path, default=Path("results"))
    vla_online.add_argument("--run-id")
    vla_online.add_argument(
        "--scenarios",
        nargs="+",
        choices=("single_block", "narrow_gate"),
    )
    vla_online.add_argument(
        "--horizons", nargs="+", type=int, choices=range(1, 16)
    )
    vla_online.add_argument("--payloads", nargs="+", type=float)
    vla_online.add_argument(
        "--quick",
        action="store_true",
        help="run single_block at horizons 1 and 15 with zero payload",
    )

    vla_probe = subparsers.add_parser(
        "vla-probe",
        help="send one MuJoCo observation to a real remote OpenPI server",
    )
    vla_probe.add_argument(
        "--config", type=Path, default=Path("configs/vla_guard_benchmark.json")
    )
    vla_probe.add_argument("--output-directory", type=Path, required=True)
    vla_probe.add_argument("--host", default="localhost")
    vla_probe.add_argument("--port", type=int, default=8000)
    vla_probe.add_argument(
        "--scenario",
        choices=("single_block", "narrow_gate"),
        default="single_block",
    )
    vla_probe.add_argument(
        "--prompt",
        help="language instruction; defaults to the configured scenario prompt",
    )
    vla_probe.add_argument(
        "--api-key-env",
        default="OPENPI_API_KEY",
        help="environment variable containing an optional server API key",
    )
    vla_probe.add_argument("--connect-timeout-s", type=float, default=3.0)
    return parser


def main(arguments: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(arguments)
    if args.command == "validate":
        return _validate(args.config)
    if args.command == "mujoco-validate":
        return _mujoco_validate(args.config)
    if args.command == "mujoco-run":
        if args.quick and args.seeds:
            parser.error("--quick cannot be combined with --seeds")
        seeds = [0, 1] if args.quick else (
            parse_seed_spec(args.seeds) if args.seeds else None
        )
        output = execute_mujoco_benchmark(
            args.config,
            args.output_root,
            run_id=args.run_id,
            planning_seeds=seeds,
            collision_samples=40 if args.quick else None,
            skip_execution=args.skip_execution or args.quick,
            make_videos=not args.no_videos,
        )
        print(f"results: {output.resolve()}")
        return 0
    if args.command == "mujoco-view":
        from armbench.mujoco_sim.viewer import launch_trajectory_viewer

        record = launch_trajectory_viewer(
            scenario_name=args.scenario,
            pose_name=args.pose,
            clearance_m=args.clearance_mm / 1000.0,
            payload_mass=args.payload,
            trace_path=args.trace,
            array_key=args.array,
            frame=args.frame,
            play=args.play,
            playback_speed=args.speed,
            loop=args.loop,
        )
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return 0
    if args.command == "vla-guard-run":
        if args.quick and args.scenarios:
            parser.error("--quick cannot be combined with --scenarios")
        output = execute_vla_guard_benchmark(
            args.config,
            args.output_root,
            run_id=args.run_id,
            scenarios=["single_block"] if args.quick else args.scenarios,
            make_videos=not (args.no_videos or args.quick),
        )
        print(f"results: {output.resolve()}")
        return 0
    if args.command == "vla-online-run":
        if args.quick and (args.scenarios or args.horizons or args.payloads):
            parser.error(
                "--quick cannot be combined with scenarios, horizons, or payloads"
            )
        output = execute_vla_online_benchmark(
            args.config,
            args.output_root,
            run_id=args.run_id,
            scenarios=["single_block"] if args.quick else args.scenarios,
            execution_horizons=[1, 15] if args.quick else args.horizons,
            payload_masses=[0.0] if args.quick else args.payloads,
        )
        print(f"results: {output.resolve()}")
        return 0
    if args.command == "vla-probe":
        config = load_vla_config(args.config)
        prompt = args.prompt or str(dict(config["prompts"])[args.scenario])
        output = execute_openpi_probe(
            args.config,
            args.output_directory,
            host=args.host,
            port=args.port,
            scenario_name=args.scenario,
            prompt=prompt,
            api_key=os.environ.get(args.api_key_env),
            connect_timeout_s=args.connect_timeout_s,
        )
        print(f"results: {output.resolve()}")
        return 0
    planning_seeds = parse_seed_spec(args.seeds) if args.seeds else None
    control_seeds = (
        parse_seed_spec(args.control_seeds) if args.control_seeds else None
    )
    if args.quick:
        if planning_seeds is not None or control_seeds is not None:
            parser.error("--quick cannot be combined with explicit seed options")
        planning_seeds = [0, 1, 2]
        control_seeds = [0, 1]
    output = execute_benchmark(
        args.config,
        args.output_root,
        run_id=args.run_id,
        planning_seeds=planning_seeds,
        control_seeds=control_seeds,
        skip_control=args.skip_control,
        make_figures=not args.no_figures,
    )
    print(f"results: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
