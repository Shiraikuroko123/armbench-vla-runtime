"""Command-line interface for validation and reproducible experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from armbench.benchmark import execute_benchmark, load_config, parse_seed_spec
from armbench.collision import CollisionChecker
from armbench.environment import (
    collect_environment_report,
    format_environment_report,
)
from armbench.model import RobotModel
from armbench.mujoco_sim.benchmark import (
    execute_mujoco_benchmark,
    load_mujoco_config,
    validate_mujoco_scenarios,
)
from armbench.mujoco_sim.swept_audit import (
    SweptAuditConfig,
    run_swept_collision_audit,
    validate_swept_collision_audit,
)
from armbench.scenario import benchmark_scenarios
from armbench.vla.async_runtime import run_async_runtime_smoke
from armbench.vla.async_panda import ASYNC_PANDA_MODES
from armbench.vla.async_panda_benchmark import (
    AsyncPandaCondition,
    execute_async_panda_benchmark,
    validate_async_panda_artifact,
)
from armbench.vla.cartesian_adapter import run_cartesian_adapter_smoke
from armbench.vla.pi05_archive_replay import (
    ArchiveReplayConfig,
    execute_pi05_archive_replay,
    validate_pi05_replay_artifact,
)
from armbench.vla.pi05_braking_repair import (
    Pi05BrakingComparisonConfig,
    execute_pi05_braking_comparison,
    validate_pi05_braking_comparison,
)
from armbench.vla.provider_contract import (
    run_provider_contract_audit,
    validate_frozen_provider_bundle,
    validate_provider_contract_audit,
)
from armbench.vla.benchmark import (
    execute_openpi_probe,
    execute_vla_guard_benchmark,
    load_vla_config,
)
from armbench.vla.artifact import validate_online_artifact
from armbench.vla.fault_matrix import execute_loopback_fault_matrix
from armbench.vla.online_benchmark import (
    execute_openpi_online_run,
    execute_vla_online_benchmark,
)
from armbench.vla.loopback import (
    LOOPBACK_FAULT_MODES,
    execute_openpi_loopback_run,
)
from armbench.vla.probe_comparison import (
    execute_recorded_probe_comparison,
    validate_recorded_probe_comparison,
)
from armbench.vla.probe_batch_comparison import (
    execute_recorded_probe_batch_comparison,
    validate_recorded_probe_batch_comparison,
)
from armbench.vla.request_replay import load_recorded_openpi_request
from armbench.vla.replay_probe import (
    execute_recorded_openpi_probe,
    validate_recorded_openpi_probe,
)
from armbench.vla.probe_sweep import (
    execute_recorded_openpi_probe_sweep,
    validate_recorded_openpi_probe_sweep,
)


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
        description="VLA action-chunk runtime evaluation and Panda physics benchmarks",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor", help="check the local CPU runtime and Panda model installation"
    )
    doctor.add_argument(
        "--require-vla",
        action="store_true",
        help="treat the optional OpenPI client as a required dependency",
    )
    doctor.add_argument(
        "--json", action="store_true", help="emit a machine-readable report"
    )

    async_smoke = subparsers.add_parser(
        "vla-async-smoke",
        help="verify that policy inference does not block a local control loop",
    )
    async_smoke.add_argument("--policy-latency-ms", type=float, default=160.0)
    async_smoke.add_argument("--control-period-ms", type=float, default=10.0)
    async_smoke.add_argument(
        "--action-period-ms", type=float, default=1000.0 / 15.0
    )
    async_smoke.add_argument("--deadline-ms", type=float, default=200.0)

    async_panda = subparsers.add_parser(
        "vla-panda-async-run",
        help="run a wall-clock asynchronous policy/repair/Panda fault matrix",
    )
    async_panda.add_argument(
        "--config", type=Path, default=Path("configs/vla_guard_benchmark.json")
    )
    async_panda.add_argument("--output-directory", type=Path, required=True)
    async_panda.add_argument(
        "--scenario",
        choices=("free_space", "single_block", "narrow_gate"),
        default="single_block",
    )
    async_panda.add_argument(
        "--modes",
        nargs="+",
        choices=ASYNC_PANDA_MODES,
        default=list(ASYNC_PANDA_MODES),
    )
    async_panda.add_argument(
        "--latencies-ms",
        nargs="+",
        type=float,
        help="run only fixed-latency conditions instead of the default fault matrix",
    )
    async_panda.add_argument("--max-reference-steps", type=int)
    async_panda.add_argument("--extra-action-steps", type=int, default=15)
    async_panda.add_argument(
        "--runtime-clearance-mm",
        type=float,
        help=(
            "runtime collision inflation; defaults to the configured planning "
            "clearance (pass 0 for a true-geometry ablation)"
        ),
    )
    async_panda.add_argument(
        "--deadline-ms",
        type=float,
        help="override the configured response deadline for this matrix",
    )
    async_panda.add_argument("--seed", type=int, default=20260807)
    async_panda.add_argument(
        "--quick",
        action="store_true",
        help="run 20 reference steps at 0/80/240 ms across all selected modes",
    )
    async_panda.add_argument(
        "--videos",
        action="store_true",
        help="render each measured trace to MP4 after timing collection",
    )

    async_panda_validate = subparsers.add_parser(
        "vla-panda-async-validate",
        help="recompute an asynchronous Panda artifact from traces and events",
    )
    async_panda_validate.add_argument("directory", type=Path)

    subparsers.add_parser(
        "vla-panda-adapter-smoke",
        help="verify the CPU-only LIBERO Cartesian to Panda guard bridge",
    )
    provider_audit = subparsers.add_parser(
        "vla-provider-audit",
        help="audit a CPU-only second-provider ABI and semantic gate",
    )
    provider_audit.add_argument("--output-directory", type=Path, required=True)
    provider_audit_validate = subparsers.add_parser(
        "vla-provider-audit-validate",
        help="replay and validate a provider-contract audit artifact",
    )
    provider_audit_validate.add_argument("directory", type=Path)
    provider_bundle_validate = subparsers.add_parser(
        "vla-provider-bundle-validate",
        help="validate one frozen action-provider bundle",
    )
    provider_bundle_validate.add_argument("directory", type=Path)

    archive_replay = subparsers.add_parser(
        "vla-panda-archive-replay",
        help="replay hash-verified frozen pi0.5 responses through the Panda guard",
    )
    archive_replay.add_argument("source_directory", type=Path)
    archive_replay.add_argument("--output-directory", type=Path, required=True)
    archive_replay.add_argument("--chunks", type=int, default=90)
    archive_replay.add_argument("--selection-seed", type=int, default=20260807)
    archive_replay.add_argument(
        "--scenarios",
        nargs="+",
        choices=("free_space", "single_block", "narrow_gate"),
        default=("free_space", "single_block", "narrow_gate"),
    )
    archive_replay.add_argument("--deadline-ms", type=float, default=200.0)
    archive_replay.add_argument(
        "--collision-resolution-rad", type=float, default=0.02
    )

    archive_replay_validate = subparsers.add_parser(
        "vla-panda-archive-replay-validate",
        help="verify a derived frozen-response Panda replay artifact",
    )
    archive_replay_validate.add_argument("directory", type=Path)
    archive_replay_validate.add_argument("--source-directory", type=Path)

    braking_repair = subparsers.add_parser(
        "vla-panda-braking-repair",
        help="compare greedy and braking-invariant guards on frozen pi0.5 chunks",
    )
    braking_repair.add_argument("source_directory", type=Path)
    braking_repair.add_argument("--output-directory", type=Path, required=True)
    braking_repair.add_argument("--chunks", type=int, default=90)
    braking_repair.add_argument("--selection-seed", type=int, default=20260807)
    braking_repair.add_argument(
        "--scenarios",
        nargs="+",
        choices=("free_space", "single_block", "narrow_gate"),
        default=("free_space", "single_block", "narrow_gate"),
    )
    braking_repair.add_argument(
        "--response-deadline-ms", type=float, default=200.0
    )
    braking_repair.add_argument(
        "--repair-selection-deadline-ms", type=float, default=20.0
    )
    braking_repair.add_argument(
        "--collision-resolution-rad", type=float, default=0.02
    )

    braking_repair_validate = subparsers.add_parser(
        "vla-panda-braking-repair-validate",
        help="verify a paired braking-repair report and trajectory archive",
    )
    braking_repair_validate.add_argument("directory", type=Path)
    braking_repair_validate.add_argument("--source-directory", type=Path)

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

    swept_audit = subparsers.add_parser(
        "mujoco-swept-audit",
        help="compare clearance-backed swept checks with a dense sampled oracle",
    )
    swept_audit.add_argument("--output-directory", type=Path, required=True)
    swept_audit.add_argument(
        "--scenarios",
        nargs="+",
        choices=("free_space", "single_block", "narrow_gate"),
        default=("free_space", "single_block", "narrow_gate"),
    )
    swept_audit.add_argument("--samples-per-scenario", type=int, default=24)
    swept_audit.add_argument("--seed", type=int, default=20260808)
    swept_audit.add_argument("--clearance-mm", type=float, default=20.0)
    swept_audit.add_argument("--sampled-resolution-rad", type=float, default=0.05)
    swept_audit.add_argument("--dense-resolution-rad", type=float, default=0.002)
    swept_audit.add_argument("--quick", action="store_true")
    swept_validate = subparsers.add_parser(
        "mujoco-swept-validate",
        help="validate a preserved swept-collision audit",
    )
    swept_validate.add_argument("directory", type=Path)

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
        choices=(
            "auto",
            "actual_positions",
            "desired_positions",
            "smoothed",
            "raw",
            "raw_positions",
            "legacy_positions",
            "repair_positions",
        ),
        default="auto",
    )
    mujoco_view.add_argument("--episode", type=int, default=0)
    mujoco_view.add_argument("--frame", type=int, default=-1)
    mujoco_view.add_argument("--play", action="store_true")
    mujoco_view.add_argument("--speed", type=float, default=1.0)
    mujoco_view.add_argument("--loop", action="store_true")

    vla_guard = subparsers.add_parser(
        "vla-guard-run",
        help="evaluate OpenPI-compatible action chunks with runtime validation",
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
    vla_online_latency = vla_online.add_mutually_exclusive_group()
    vla_online_latency.add_argument(
        "--policy-latency-ms",
        type=float,
        help="synthetic inference delay that advances MuJoCo under pose hold",
    )
    vla_online_latency.add_argument(
        "--policy-latency-schedule-ms",
        nargs="+",
        type=float,
        help="repeating synthetic per-query inference latency schedule",
    )
    vla_online.add_argument(
        "--state-jump-joint",
        type=int,
        choices=range(1, 8),
        help="inject a synthetic dispatch-state jump on this 1-based joint",
    )
    vla_online.add_argument(
        "--state-jump-rad",
        type=float,
        help="signed synthetic state jump magnitude in radians",
    )
    vla_online.add_argument(
        "--state-jump-query",
        type=int,
        default=None,
        help="zero-based policy query for the jump (default: 0)",
    )
    vla_online.add_argument(
        "--freeze-camera",
        choices=("exterior", "wrist", "both"),
        help="replay the preceding frame from the selected camera input",
    )
    vla_online.add_argument(
        "--freeze-camera-query",
        type=int,
        help="zero-based observation cycle for camera replay (must be > 0)",
    )
    vla_online.add_argument(
        "--quick",
        action="store_true",
        help="run single_block at horizons 1 and 15 with zero payload",
    )
    vla_online.add_argument(
        "--videos",
        action="store_true",
        help="record live MuJoCo MP4 for every selected online episode",
    )
    vla_online.add_argument(
        "--save-observations",
        action="store_true",
        help="store both full 224x224 policy images for every query",
    )

    vla_openpi_run = subparsers.add_parser(
        "vla-openpi-run",
        help="run bounded remote OpenPI inference in the live MuJoCo loop",
    )
    vla_openpi_run.add_argument(
        "--config", type=Path, default=Path("configs/vla_guard_benchmark.json")
    )
    vla_openpi_run.add_argument(
        "--output-directory", type=Path, required=True
    )
    vla_openpi_run.add_argument("--host", default="localhost")
    vla_openpi_run.add_argument("--port", type=int, default=8000)
    vla_openpi_run.add_argument(
        "--scenario",
        choices=("single_block", "narrow_gate"),
        default="single_block",
    )
    vla_openpi_run.add_argument(
        "--horizon", type=int, choices=range(1, 16), default=5
    )
    vla_openpi_run.add_argument("--payload", type=float, default=0.0)
    vla_openpi_run.add_argument(
        "--max-policy-queries", type=int, default=10
    )
    vla_openpi_run.add_argument("--prompt")
    vla_openpi_run.add_argument("--api-key-env", default="OPENPI_API_KEY")
    vla_openpi_run.add_argument("--connect-timeout-s", type=float, default=3.0)
    vla_openpi_run.add_argument("--inference-timeout-s", type=float, default=1.0)
    vla_openpi_run.add_argument(
        "--video",
        action="store_true",
        help="record the live MuJoCo episode as MP4",
    )
    vla_openpi_run.add_argument(
        "--save-observations",
        action="store_true",
        help="store both full 224x224 policy images for every query",
    )

    vla_loopback = subparsers.add_parser(
        "vla-loopback-run",
        help="exercise the OpenPI wire path with a non-learned local server",
    )
    vla_loopback.add_argument(
        "--config", type=Path, default=Path("configs/vla_guard_benchmark.json")
    )
    vla_loopback.add_argument(
        "--output-directory", type=Path, required=True
    )
    vla_loopback.add_argument(
        "--scenario",
        choices=("single_block", "narrow_gate"),
        default="single_block",
    )
    vla_loopback.add_argument(
        "--horizon", type=int, choices=range(1, 16), default=5
    )
    vla_loopback.add_argument("--payload", type=float, default=0.0)
    vla_loopback.add_argument("--max-policy-queries", type=int, default=3)
    vla_loopback.add_argument("--prompt")
    vla_loopback.add_argument(
        "--fault-mode",
        choices=LOOPBACK_FAULT_MODES,
        default="none",
        help="inject one deterministic server response/transport fault",
    )
    vla_loopback.add_argument(
        "--fault-query",
        type=int,
        default=0,
        help="zero-based request index at which to inject the fault",
    )
    vla_loopback.add_argument(
        "--fault-delay-ms",
        type=float,
        default=250.0,
        help="server delay used by the timeout fault",
    )
    vla_loopback.add_argument(
        "--inference-timeout-s",
        type=float,
        default=0.1,
        help="bounded client receive timeout",
    )
    vla_loopback.add_argument(
        "--video",
        action="store_true",
        help="record the live MuJoCo episode as MP4",
    )
    vla_loopback.add_argument(
        "--save-observations",
        action="store_true",
        help="store both full 224x224 policy images for every request",
    )

    vla_loopback_matrix = subparsers.add_parser(
        "vla-loopback-matrix",
        help="run a matched OpenPI nominal/response/transport fault matrix",
    )
    vla_loopback_matrix.add_argument(
        "--config", type=Path, default=Path("configs/vla_guard_benchmark.json")
    )
    vla_loopback_matrix.add_argument(
        "--output-directory", type=Path, required=True
    )
    vla_loopback_matrix.add_argument(
        "--scenario",
        choices=("single_block", "narrow_gate"),
        default="single_block",
    )
    vla_loopback_matrix.add_argument(
        "--horizon", type=int, choices=range(1, 16), default=1
    )
    vla_loopback_matrix.add_argument("--payload", type=float, default=0.0)
    vla_loopback_matrix.add_argument(
        "--fault-modes",
        nargs="+",
        choices=LOOPBACK_FAULT_MODES,
        default=list(LOOPBACK_FAULT_MODES),
    )
    vla_loopback_matrix.add_argument(
        "--fault-delay-ms", type=float, default=250.0
    )
    vla_loopback_matrix.add_argument(
        "--inference-timeout-s", type=float, default=0.1
    )
    vla_loopback_matrix.add_argument("--videos", action="store_true")
    vla_loopback_matrix.add_argument(
        "--no-save-observations",
        action="store_true",
        help="omit full per-request camera arrays from child artifacts",
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
    vla_probe.add_argument("--inference-timeout-s", type=float, default=1.0)

    vla_artifact = subparsers.add_parser(
        "vla-artifact-validate",
        help="cross-check a schema-v5 online VLA artifact",
    )
    vla_artifact.add_argument("directory", type=Path)
    vla_artifact.add_argument(
        "--decode-videos",
        action="store_true",
        help="decode and inspect the first frame of every recorded MP4",
    )

    vla_request = subparsers.add_parser(
        "vla-request-inspect",
        help="reconstruct and hash one recorded OpenPI DROID request",
    )
    vla_request.add_argument("directory", type=Path)
    vla_request.add_argument("--query", type=int, default=0)
    vla_request.add_argument("--scenario")
    vla_request.add_argument("--payload", type=float)
    vla_request.add_argument(
        "--horizon", type=int, choices=range(1, 16)
    )

    vla_recorded_probe = subparsers.add_parser(
        "vla-recorded-probe",
        help="query an OpenPI server with one exact recorded request",
    )
    vla_recorded_probe.add_argument(
        "--config", type=Path, default=Path("configs/vla_guard_benchmark.json")
    )
    vla_recorded_probe.add_argument("artifact_directory", type=Path)
    vla_recorded_probe.add_argument(
        "--output-directory", type=Path, required=True
    )
    vla_recorded_probe.add_argument("--host", default="localhost")
    vla_recorded_probe.add_argument("--port", type=int, default=8000)
    vla_recorded_probe.add_argument("--query", type=int, default=0)
    vla_recorded_probe.add_argument("--scenario")
    vla_recorded_probe.add_argument("--payload", type=float)
    vla_recorded_probe.add_argument(
        "--horizon", type=int, choices=range(1, 16)
    )
    vla_recorded_probe.add_argument(
        "--api-key-env", default="OPENPI_API_KEY"
    )
    vla_recorded_probe.add_argument(
        "--connect-timeout-s", type=float, default=3.0
    )
    vla_recorded_probe.add_argument(
        "--inference-timeout-s", type=float, default=1.0
    )
    vla_recorded_probe_sweep = subparsers.add_parser(
        "vla-recorded-probe-sweep",
        help="collect multiple exact recorded requests from one OpenPI server",
    )
    vla_recorded_probe_sweep.add_argument(
        "--config", type=Path, default=Path("configs/vla_guard_benchmark.json")
    )
    vla_recorded_probe_sweep.add_argument("artifact_directory", type=Path)
    vla_recorded_probe_sweep.add_argument(
        "--output-directory", type=Path, required=True
    )
    vla_recorded_probe_sweep.add_argument(
        "--queries",
        default="0",
        help="query indices as 0:10, 0,2,4, or one integer",
    )
    vla_recorded_probe_sweep.add_argument("--host", default="localhost")
    vla_recorded_probe_sweep.add_argument("--port", type=int, default=8000)
    vla_recorded_probe_sweep.add_argument("--scenario")
    vla_recorded_probe_sweep.add_argument("--payload", type=float)
    vla_recorded_probe_sweep.add_argument(
        "--horizon", type=int, choices=range(1, 16)
    )
    vla_recorded_probe_sweep.add_argument(
        "--api-key-env", default="OPENPI_API_KEY"
    )
    vla_recorded_probe_sweep.add_argument(
        "--connect-timeout-s", type=float, default=3.0
    )
    vla_recorded_probe_sweep.add_argument(
        "--inference-timeout-s", type=float, default=1.0
    )
    vla_recorded_probe_sweep.add_argument(
        "--policy-provenance", default="remote_server_unverified"
    )
    vla_recorded_probe_sweep_validate = subparsers.add_parser(
        "vla-recorded-probe-sweep-validate",
        help="cross-check a recorded OpenPI probe sweep artifact",
    )
    vla_recorded_probe_sweep_validate.add_argument("directory", type=Path)
    vla_recorded_probe_validate = subparsers.add_parser(
        "vla-recorded-probe-validate",
        help="cross-check a fixed-request OpenPI probe artifact",
    )
    vla_recorded_probe_validate.add_argument("directory", type=Path)
    vla_recorded_probe_compare = subparsers.add_parser(
        "vla-recorded-probe-compare",
        help="compare two validated responses to the same recorded request",
    )
    vla_recorded_probe_compare.add_argument("left_directory", type=Path)
    vla_recorded_probe_compare.add_argument("right_directory", type=Path)
    vla_recorded_probe_compare.add_argument(
        "--output-directory", type=Path, required=True
    )
    vla_recorded_probe_compare.add_argument("--left-label", default="left")
    vla_recorded_probe_compare.add_argument("--right-label", default="right")
    vla_recorded_probe_compare_validate = subparsers.add_parser(
        "vla-recorded-probe-compare-validate",
        help="recompute a paired recorded-probe comparison artifact",
    )
    vla_recorded_probe_compare_validate.add_argument("directory", type=Path)
    vla_recorded_probe_batch_compare = subparsers.add_parser(
        "vla-recorded-probe-batch-compare",
        help="pair and summarize two directories of recorded probe artifacts",
    )
    vla_recorded_probe_batch_compare.add_argument("left_root", type=Path)
    vla_recorded_probe_batch_compare.add_argument("right_root", type=Path)
    vla_recorded_probe_batch_compare.add_argument(
        "--output-directory", type=Path, required=True
    )
    vla_recorded_probe_batch_compare.add_argument(
        "--left-label", default="left"
    )
    vla_recorded_probe_batch_compare.add_argument(
        "--right-label", default="right"
    )
    vla_recorded_probe_batch_compare_validate = subparsers.add_parser(
        "vla-recorded-probe-batch-compare-validate",
        help="recompute a recorded-probe cohort comparison artifact",
    )
    vla_recorded_probe_batch_compare_validate.add_argument(
        "directory", type=Path
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(arguments)
    if args.command == "doctor":
        report = collect_environment_report(require_vla=args.require_vla)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(format_environment_report(report))
        return 0 if report.ready else 1
    if args.command == "vla-async-smoke":
        report = run_async_runtime_smoke(
            policy_latency_ms=args.policy_latency_ms,
            control_period_ms=args.control_period_ms,
            action_period_ms=args.action_period_ms,
            deadline_ms=args.deadline_ms,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if bool(report["passed"]) else 1
    if args.command == "vla-panda-async-run":
        if args.quick and (
            args.latencies_ms is not None
            or args.max_reference_steps is not None
            or args.extra_action_steps != 15
        ):
            parser.error(
                "--quick cannot be combined with latency/reference step overrides"
            )
        latencies = (
            [0.0, 80.0, 240.0]
            if args.quick
            else args.latencies_ms
        )
        if latencies is not None and (
            len(set(latencies)) != len(latencies)
            or any(
                not float(value).is_integer() or value < 0.0
                for value in latencies
            )
        ):
            parser.error("fixed latencies must be unique nonnegative integers")
        conditions = (
            tuple(
                AsyncPandaCondition(
                    name=f"fixed_{int(value):03d}ms",
                    latency_schedule_ms=(float(value),),
                )
                for value in latencies
            )
            if latencies is not None
            else None
        )
        output = execute_async_panda_benchmark(
            args.config,
            args.output_directory,
            scenario_name=args.scenario,
            modes=args.modes,
            conditions=conditions,
            max_reference_steps=(20 if args.quick else args.max_reference_steps),
            extra_action_steps=(5 if args.quick else args.extra_action_steps),
            seed=args.seed,
            make_videos=args.videos,
            runtime_clearance_m=(
                None
                if args.runtime_clearance_mm is None
                else args.runtime_clearance_mm / 1000.0
            ),
            response_deadline_ms=args.deadline_ms,
        )
        result = validate_async_panda_artifact(output)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"results: {output.resolve()}")
        return 0
    if args.command == "vla-panda-async-validate":
        result = validate_async_panda_artifact(args.directory)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "vla-panda-adapter-smoke":
        report = run_cartesian_adapter_smoke()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if bool(report["passed"]) else 1
    if args.command == "vla-provider-audit":
        output = run_provider_contract_audit(args.output_directory)
        result = validate_provider_contract_audit(output)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"results: {output.resolve()}")
        return 0
    if args.command == "vla-provider-audit-validate":
        result = validate_provider_contract_audit(args.directory)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "vla-provider-bundle-validate":
        result = validate_frozen_provider_bundle(args.directory)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "vla-panda-archive-replay":
        output = execute_pi05_archive_replay(
            args.source_directory,
            args.output_directory,
            ArchiveReplayConfig(
                chunk_count=args.chunks,
                selection_seed=args.selection_seed,
                scenarios=tuple(args.scenarios),
                deadline_ms=args.deadline_ms,
                collision_resolution_rad=args.collision_resolution_rad,
            ),
        )
        result = validate_pi05_replay_artifact(
            output, source_directory=args.source_directory
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"results: {output.resolve()}")
        return 0
    if args.command == "vla-panda-archive-replay-validate":
        result = validate_pi05_replay_artifact(
            args.directory, source_directory=args.source_directory
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "vla-panda-braking-repair":
        output = execute_pi05_braking_comparison(
            args.source_directory,
            args.output_directory,
            Pi05BrakingComparisonConfig(
                chunk_count=args.chunks,
                selection_seed=args.selection_seed,
                scenarios=tuple(args.scenarios),
                response_deadline_ms=args.response_deadline_ms,
                repair_selection_deadline_ms=(
                    args.repair_selection_deadline_ms
                ),
                collision_resolution_rad=args.collision_resolution_rad,
            ),
        )
        result = validate_pi05_braking_comparison(
            output, source_directory=args.source_directory
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"results: {output.resolve()}")
        return 0
    if args.command == "vla-panda-braking-repair-validate":
        result = validate_pi05_braking_comparison(
            args.directory, source_directory=args.source_directory
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "validate":
        return _validate(args.config)
    if args.command == "mujoco-validate":
        return _mujoco_validate(args.config)
    if args.command == "mujoco-swept-audit":
        if args.quick and args.samples_per_scenario != 24:
            parser.error("--quick cannot be combined with --samples-per-scenario")
        output = run_swept_collision_audit(
            args.output_directory,
            config=SweptAuditConfig(
                scenarios=tuple(args.scenarios),
                samples_per_scenario=(8 if args.quick else args.samples_per_scenario),
                seed=args.seed,
                clearance_m=args.clearance_mm / 1000.0,
                sampled_resolution_rad=args.sampled_resolution_rad,
                dense_resolution_rad=args.dense_resolution_rad,
            ),
        )
        result = validate_swept_collision_audit(output)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"results: {output.resolve()}")
        return 0
    if args.command == "mujoco-swept-validate":
        result = validate_swept_collision_audit(args.directory)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
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
            episode=args.episode,
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
        if (args.state_jump_joint is None) != (args.state_jump_rad is None):
            parser.error(
                "--state-jump-joint and --state-jump-rad must be used together"
            )
        if args.state_jump_query is not None and args.state_jump_joint is None:
            parser.error("--state-jump-query requires a configured state jump")
        if (args.freeze_camera is None) != (
            args.freeze_camera_query is None
        ):
            parser.error(
                "--freeze-camera and --freeze-camera-query must be used together"
            )
        output = execute_vla_online_benchmark(
            args.config,
            args.output_root,
            run_id=args.run_id,
            scenarios=["single_block"] if args.quick else args.scenarios,
            execution_horizons=[1, 15] if args.quick else args.horizons,
            payload_masses=[0.0] if args.quick else args.payloads,
            policy_latency_ms=args.policy_latency_ms,
            policy_latency_schedule_ms=args.policy_latency_schedule_ms,
            state_jump_query=args.state_jump_query,
            state_jump_joint=args.state_jump_joint,
            state_jump_rad=args.state_jump_rad,
            camera_freeze_query=args.freeze_camera_query,
            camera_freeze_target=args.freeze_camera,
            make_videos=args.videos,
            record_full_observations=args.save_observations,
        )
        print(f"results: {output.resolve()}")
        return 0
    if args.command == "vla-openpi-run":
        if args.max_policy_queries <= 0:
            parser.error("--max-policy-queries must be positive")
        config = load_vla_config(args.config)
        prompt = args.prompt or str(dict(config["prompts"])[args.scenario])
        output = execute_openpi_online_run(
            args.config,
            args.output_directory,
            host=args.host,
            port=args.port,
            scenario_name=args.scenario,
            execution_horizon=args.horizon,
            payload_mass=args.payload,
            max_policy_queries=args.max_policy_queries,
            prompt=prompt,
            api_key=os.environ.get(args.api_key_env),
            connect_timeout_s=args.connect_timeout_s,
            inference_timeout_s=args.inference_timeout_s,
            make_video=args.video,
            record_full_observations=args.save_observations,
        )
        print(f"results: {output.resolve()}")
        return 0
    if args.command == "vla-loopback-run":
        if args.max_policy_queries <= 0:
            parser.error("--max-policy-queries must be positive")
        config = load_vla_config(args.config)
        prompt = args.prompt or str(dict(config["prompts"])[args.scenario])
        output = execute_openpi_loopback_run(
            args.config,
            args.output_directory,
            scenario_name=args.scenario,
            execution_horizon=args.horizon,
            payload_mass=args.payload,
            max_policy_queries=args.max_policy_queries,
            prompt=prompt,
            make_video=args.video,
            fault_mode=args.fault_mode,
            fault_request_index=args.fault_query,
            fault_delay_ms=args.fault_delay_ms,
            inference_timeout_s=args.inference_timeout_s,
            record_full_observations=args.save_observations,
        )
        print(f"results: {output.resolve()}")
        return 0
    if args.command == "vla-loopback-matrix":
        output = execute_loopback_fault_matrix(
            args.config,
            args.output_directory,
            scenario_name=args.scenario,
            execution_horizon=args.horizon,
            payload_mass=args.payload,
            fault_modes=args.fault_modes,
            fault_delay_ms=args.fault_delay_ms,
            inference_timeout_s=args.inference_timeout_s,
            make_videos=args.videos,
            record_full_observations=not args.no_save_observations,
        )
        matrix = json.loads((output / "manifest.json").read_text("utf-8"))
        print(json.dumps(matrix, indent=2, ensure_ascii=False))
        print(f"results: {output.resolve()}")
        return 0 if bool(matrix["matrix_passed"]) else 1
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
            inference_timeout_s=args.inference_timeout_s,
        )
        print(f"results: {output.resolve()}")
        return 0
    if args.command == "vla-artifact-validate":
        result = validate_online_artifact(
            args.directory, decode_videos=args.decode_videos
        )
        print(json.dumps(result.metrics(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "vla-request-inspect":
        request = load_recorded_openpi_request(
            args.directory,
            query_index=args.query,
            scenario=args.scenario,
            payload_mass=args.payload,
            execution_horizon=args.horizon,
        )
        print(json.dumps(request.metrics(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "vla-recorded-probe":
        output = execute_recorded_openpi_probe(
            args.config,
            args.artifact_directory,
            args.output_directory,
            host=args.host,
            port=args.port,
            query_index=args.query,
            scenario=args.scenario,
            payload_mass=args.payload,
            execution_horizon=args.horizon,
            api_key=os.environ.get(args.api_key_env),
            connect_timeout_s=args.connect_timeout_s,
            inference_timeout_s=args.inference_timeout_s,
        )
        print(f"results: {output.resolve()}")
        return 0
    if args.command == "vla-recorded-probe-sweep":
        output = execute_recorded_openpi_probe_sweep(
            args.config,
            args.artifact_directory,
            args.output_directory,
            host=args.host,
            port=args.port,
            query_indices=parse_seed_spec(args.queries),
            scenario=args.scenario,
            payload_mass=args.payload,
            execution_horizon=args.horizon,
            api_key=os.environ.get(args.api_key_env),
            connect_timeout_s=args.connect_timeout_s,
            inference_timeout_s=args.inference_timeout_s,
            policy_provenance=args.policy_provenance,
        )
        manifest = json.loads(
            (output / "manifest.json").read_text(encoding="utf-8")
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"results: {output.resolve()}")
        return 0 if bool(manifest["sweep_complete"]) else 1
    if args.command == "vla-recorded-probe-sweep-validate":
        result = validate_recorded_openpi_probe_sweep(args.directory)
        print(json.dumps(result.metrics(), indent=2, ensure_ascii=False))
        return 0 if result.sweep_complete else 1
    if args.command == "vla-recorded-probe-validate":
        result = validate_recorded_openpi_probe(args.directory)
        print(json.dumps(result.metrics(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "vla-recorded-probe-compare":
        output = execute_recorded_probe_comparison(
            args.left_directory,
            args.right_directory,
            args.output_directory,
            left_label=args.left_label,
            right_label=args.right_label,
        )
        comparison = json.loads(
            (output / "comparison.json").read_text(encoding="utf-8")
        )
        print(json.dumps(comparison, indent=2, ensure_ascii=False))
        print(f"results: {output.resolve()}")
        return 0
    if args.command == "vla-recorded-probe-compare-validate":
        result = validate_recorded_probe_comparison(args.directory)
        print(json.dumps(result.metrics(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "vla-recorded-probe-batch-compare":
        output = execute_recorded_probe_batch_comparison(
            args.left_root,
            args.right_root,
            args.output_directory,
            left_label=args.left_label,
            right_label=args.right_label,
        )
        batch = json.loads(
            (output / "batch.json").read_text(encoding="utf-8")
        )
        print(json.dumps(batch, indent=2, ensure_ascii=False))
        print(f"results: {output.resolve()}")
        return 0
    if args.command == "vla-recorded-probe-batch-compare-validate":
        result = validate_recorded_probe_batch_comparison(args.directory)
        print(json.dumps(result.metrics(), indent=2, ensure_ascii=False))
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
