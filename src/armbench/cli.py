"""Command-line interface for validation and reproducible experiments."""

from __future__ import annotations

import argparse
import json
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
        description="Seven-joint planning, control, and MuJoCo physics benchmark",
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
