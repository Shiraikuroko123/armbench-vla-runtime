"""Run an auditable true independent-clock pi0.5-LIBERO pilot."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from integrations.openpi.libero_independent_clock import (
    IndependentLiberoRequestBuilder,
    OpenPIIndependentClockProviderFactory,
    SCHEMA_VERSION,
    run_libero_independent_clock_episode,
)
from integrations.openpi.libero_runtime_eval import (
    DEFAULT_CHECKPOINT,
    DEFAULT_POLICY_CONFIG,
    LIBERO_CONTROL_FREQUENCY_HZ,
    LIBERO_CONTROL_PERIOD_MS,
    LIBERO_ENV_RESOLUTION,
    OPENPI_COMMIT,
    SERVER_ATTESTATION_SCHEMA_VERSION,
    SUITE_MAX_STEPS,
    SUITE_TASK_COUNTS,
    BoundedOpenPIClient,
    _make_libero_environment,
    _parse_int_selection,
    _validate_server_attestation,
    _validate_server_launch_args,
    _write_video,
)
from integrations.openpi.measured_age_libero_eval import (
    validate_policy_sampling_server_contract,
)
from integrations.openpi.serve_policy_attested import policy_sampling_contract


RUNTIME_SOURCE_FILES = (
    "src/armbench/vla/independent_clock.py",
    "integrations/openpi/libero_independent_clock.py",
    "integrations/openpi/libero_independent_clock_eval.py",
    "integrations/openpi/validate_libero_independent_clock.py",
    "integrations/openpi/libero_runtime.py",
    "integrations/openpi/libero_runtime_eval.py",
    "integrations/openpi/serve_policy_attested.py",
    "integrations/openpi/compose.libero-independent-clock.yml",
)

EPISODE_FIELDS = (
    "schema_version",
    "episode_id",
    "task_suite",
    "task_id",
    "episode_index",
    "task_description",
    "seed",
    "task_success",
    "termination_reason",
    "initial_state_sha256",
    "stabilization_steps",
    "task_steps",
    "wall_time_s",
    "parent_process_id",
    "worker_process_id",
    "worker_stopped",
    "submitted_requests",
    "started_requests",
    "completed_requests",
    "superseded_requests",
    "accepted_responses",
    "failed_responses",
    "deadline_exceeded_responses",
    "hold_ticks",
    "execute_ticks",
    "ticks_during_inference",
    "tick_overruns",
    "response_age_p50_ms",
    "response_age_p95_ms",
    "response_age_max_ms",
    "video_path",
)


@dataclass(frozen=True)
class ExperimentCell:
    task_suite: str
    task_id: int
    episode_index: int

    @property
    def episode_id(self) -> str:
        return "%s__task_%03d__episode_%02d" % (
            self.task_suite,
            self.task_id,
            self.episode_index,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_suite": self.task_suite,
            "task_id": self.task_id,
            "episode_index": self.episode_index,
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("artifact JSON cannot contain non-finite floats")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return repr(value)


def _write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(
    path: pathlib.Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    temporary.replace(path)


def _command_output(
    command: Sequence[str], cwd: pathlib.Path | None = None
) -> str | None:
    try:
        completed = subprocess.run(
            list(command),
            cwd=None if cwd is None else str(cwd),
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or completed.stderr.strip() or None


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentiles(values: Sequence[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.percentile(array, 50)),
        float(np.percentile(array, 95)),
        float(np.max(array)),
    )


def ticks_during_inference(runtime: Mapping[str, Any]) -> int:
    """Count control ticks inside a recorded child inference interval."""

    intervals = []
    for request in runtime["requests"]:
        started = request.get("started_at_s")
        completed = request.get("completed_at_s")
        if started is not None and completed is not None:
            intervals.append((float(started), float(completed)))
    return sum(
        any(start <= float(tick["tick_started_at_s"]) <= end for start, end in intervals)
        for tick in runtime["ticks"]
    )


def episode_row(
    cell: ExperimentCell,
    task_description: str,
    seed: int,
    result: Mapping[str, Any],
    wall_time_s: float,
    video_path: str | None,
) -> dict[str, Any]:
    runtime = result["runtime"]
    requests = runtime["requests"]
    metrics = runtime["metrics"]
    ages = [
        float(request["response_age_ms"])
        for request in requests
        if request.get("response_age_ms") is not None
    ]
    p50, p95, maximum = _percentiles(ages)
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": cell.episode_id,
        "task_suite": cell.task_suite,
        "task_id": cell.task_id,
        "episode_index": cell.episode_index,
        "task_description": task_description,
        "seed": seed,
        "task_success": bool(result["task_success"]),
        "termination_reason": result["termination_reason"],
        "initial_state_sha256": result["initial_state_sha256"],
        "stabilization_steps": int(result["stabilization_steps"]),
        "task_steps": int(result["task_steps"]),
        "wall_time_s": float(wall_time_s),
        "parent_process_id": int(runtime["parent_process_id"]),
        "worker_process_id": int(runtime["worker_process_id"]),
        "worker_stopped": bool(runtime["worker_stopped"]),
        "submitted_requests": int(metrics["submitted"]),
        "started_requests": int(metrics["started"]),
        "completed_requests": int(metrics["completed"]),
        "superseded_requests": int(metrics["superseded"]),
        "accepted_responses": sum(
            request["response_status"] == "accepted" for request in requests
        ),
        "failed_responses": sum(
            request.get("failure_type") is not None for request in requests
        ),
        "deadline_exceeded_responses": sum(
            request["response_status"] == "deadline_exceeded"
            for request in requests
        ),
        "hold_ticks": int(metrics["holds"]),
        "execute_ticks": int(metrics["executes"]),
        "ticks_during_inference": ticks_during_inference(runtime),
        "tick_overruns": int(runtime["tick_overruns"]),
        "response_age_p50_ms": p50,
        "response_age_p95_ms": p95,
        "response_age_max_ms": maximum,
        "video_path": video_path,
    }


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]], planned_rollouts: int
) -> dict[str, Any]:
    completed = len(rows)
    successes = sum(bool(row["task_success"]) for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "planned_rollouts": planned_rollouts,
        "completed_rollouts": completed,
        "complete": completed == planned_rollouts,
        "task_successes": successes,
        "task_success_rate": None if not completed else successes / completed,
        "total_control_ticks": sum(int(row["task_steps"]) for row in rows),
        "total_ticks_during_inference": sum(
            int(row["ticks_during_inference"]) for row in rows
        ),
        "episodes_with_inference_overlap": sum(
            int(row["ticks_during_inference"]) > 0 for row in rows
        ),
        "total_holds": sum(int(row["hold_ticks"]) for row in rows),
        "total_executes": sum(int(row["execute_ticks"]) for row in rows),
        "total_deadline_exceeded_responses": sum(
            int(row["deadline_exceeded_responses"]) for row in rows
        ),
        "total_failed_responses": sum(int(row["failed_responses"]) for row in rows),
    }


def summary_markdown(
    rows: Sequence[Mapping[str, Any]], aggregate: Mapping[str, Any]
) -> str:
    success_rate = aggregate["task_success_rate"]
    rate_text = "n/a" if success_rate is None else "%.1f%%" % (100.0 * success_rate)
    return (
        "# pi0.5-LIBERO independent-clock pilot\n\n"
        "- Completed: %d/%d\n"
        "- Official LIBERO task success: %d/%d (%s)\n"
        "- Control ticks while policy inference was in flight: %d\n"
        "- Episodes proving inference/simulation overlap: %d/%d\n"
        "- Execute / hold ticks: %d / %d\n"
        "- Deadline-exceeded / failed responses: %d / %d\n\n"
        "This is a simulation pilot with the official attested `pi05_libero` "
        "checkpoint. It is not a hardware result, hard-real-time guarantee, or "
        "safety certificate.\n"
        % (
            aggregate["completed_rollouts"],
            aggregate["planned_rollouts"],
            aggregate["task_successes"],
            aggregate["completed_rollouts"],
            rate_text,
            aggregate["total_ticks_during_inference"],
            aggregate["episodes_with_inference_overlap"],
            len(rows),
            aggregate["total_executes"],
            aggregate["total_holds"],
            aggregate["total_deadline_exceeded_responses"],
            aggregate["total_failed_responses"],
        )
    )


def _write_derived(
    output: pathlib.Path,
    rows: Sequence[Mapping[str, Any]],
    planned_rollouts: int,
) -> None:
    _write_json(output / "per_episode.json", list(rows))
    _write_csv(output / "per_episode.csv", rows, EPISODE_FIELDS)
    aggregate = aggregate_rows(rows, planned_rollouts)
    _write_json(output / "aggregate.json", aggregate)
    (output / "summary.md").write_text(
        summary_markdown(rows, aggregate), encoding="utf-8"
    )
    _write_json(
        output / "progress.json",
        {
            "schema_version": SCHEMA_VERSION,
            "planned_rollouts": planned_rollouts,
            "completed_rollouts": len(rows),
            "complete": len(rows) == planned_rollouts,
            "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )


def _snapshot_sources(root: pathlib.Path, output: pathlib.Path) -> dict[str, str]:
    hashes = {}
    snapshot = output / "provenance" / "armbench_source"
    for relative in RUNTIME_SOURCE_FILES:
        source = root / pathlib.PurePosixPath(relative)
        if not source.is_file():
            raise FileNotFoundError("required runtime source is missing: %s" % relative)
        destination = snapshot / pathlib.PurePosixPath(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        hashes[relative] = _sha256(destination)
    return hashes


def _capture_environment(
    args: argparse.Namespace,
    openpi_root: pathlib.Path,
    armbench_root: pathlib.Path,
    metadata: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    packages = {}
    for package in ("numpy", "imageio", "openpi-client", "libero"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "openpi_git_commit": _command_output(("git", "rev-parse", "HEAD"), openpi_root),
        "armbench_git_commit": _command_output(("git", "rev-parse", "HEAD"), armbench_root),
        "armbench_git_status": _command_output(
            ("git", "status", "--porcelain"), armbench_root
        )
        or "",
        "nvidia_smi": _command_output(
            (
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            )
        ),
        "packages": packages,
        "server_metadata": _json_safe(metadata),
        "runtime_source_sha256": dict(source_hashes),
        "arguments": _json_safe(vars(args)),
        "selected_environment_variables": {
            name: os.environ.get(name)
            for name in (
                "MUJOCO_GL",
                "MUJOCO_EGL_DEVICE_ID",
                "NVIDIA_VISIBLE_DEVICES",
                "OPENPI_DATA_HOME",
                "ARMBENCH_SERVER_ARGS",
            )
        },
    }


def resolved_protocol(
    args: argparse.Namespace, cells: Sequence[ExperimentCell]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "research_question": (
            "Can an attested pi0.5-LIBERO policy run with simulation and blocking "
            "inference on independent clocks while every deadline, stale suffix, "
            "hold, action, and task outcome remains independently auditable?"
        ),
        "openpi_commit": args.expected_openpi_commit,
        "policy_config": DEFAULT_POLICY_CONFIG,
        "checkpoint": args.checkpoint,
        "server_launch_args": args.server_launch_args,
        "server_attestation_schema": SERVER_ATTESTATION_SCHEMA_VERSION,
        "runtime": {
            "controller_model": "parent_simulation_spawned_blocking_provider",
            "control_period_ms": args.control_period_ms,
            "action_period_ms": args.control_period_ms,
            "deadline_ms": args.deadline_ms,
            "submit_every_ticks": args.submit_every_ticks,
            "mailbox": "latest_only_single_pending_observation",
            "stale_prefix_rule": "ceil(observation_age/control_period)",
            "deadline_disposition": "hold",
            "horizon_exhaustion_disposition": "hold",
            "hold_semantics": "zero Cartesian motion, preserve last gripper command",
        },
        "official_protocol": {
            "environment_render_resolution": [
                LIBERO_ENV_RESOLUTION,
                LIBERO_ENV_RESOLUTION,
            ],
            "request_image_resolution": [224, 224],
            "camera_rotation_degrees": 180,
            "state_dimension": 8,
            "action_dimension": 7,
            "action_horizon": 10,
            "control_frequency_hz": LIBERO_CONTROL_FREQUENCY_HZ,
            "task_success_source": "LIBERO environment done",
            "stabilization_steps": args.num_steps_wait,
        },
        "policy_sampling": {
            **policy_sampling_contract(),
            "seed": args.seed,
            "pairing_key": [
                "task_suite",
                "task_id",
                "episode_index",
                "submit_every_ticks",
            ],
            "query_index": "observation_sequence_id",
        },
        "matrix": {
            "planned_rollouts": len(cells),
            "cells": [cell.to_dict() for cell in cells],
        },
        "task_step_override": args.max_task_steps,
        "suite_task_step_limits": SUITE_MAX_STEPS,
        "video_mode": args.video_mode,
        "limitations": [
            "The pilot is not an official aggregate LIBERO leaderboard score unless all required suite episodes are run.",
            "Process separation and measured overlap do not establish an OS hard-real-time guarantee.",
            "LIBERO simulation success is not hardware deployment or safety certification.",
            "The deadline rule is training-free temporal selection, not dynamics-aware action repair.",
        ],
    }


def _write_manifest(output: pathlib.Path) -> None:
    files = {}
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "manifest.json" or path.suffix == ".tmp":
            continue
        relative = path.relative_to(output).as_posix()
        files[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    _write_json(
        output / "manifest.json",
        {"schema_version": SCHEMA_VERSION, "files": files},
    )


def _prepare_output(path: pathlib.Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("output directory must be absent or empty: %s" % path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "episodes").mkdir(exist_ok=True)
    (path / "videos").mkdir(exist_ok=True)


def resolve_cells(args: argparse.Namespace) -> list[ExperimentCell]:
    task_ids = _parse_int_selection(
        args.task_ids, SUITE_TASK_COUNTS[args.task_suite], "task_ids"
    )
    episode_indices = _parse_int_selection(args.episode_indices, 50, "episode_indices")
    return [
        ExperimentCell(args.task_suite, task_id, episode_index)
        for task_id in task_ids
        for episode_index in episode_indices
    ]


def execute(args: argparse.Namespace, cells: Sequence[ExperimentCell]) -> int:
    from libero.libero import benchmark

    output = pathlib.Path(args.output_dir).resolve()
    openpi_root = pathlib.Path(args.openpi_root).resolve()
    armbench_root = pathlib.Path(args.armbench_root).resolve()
    _prepare_output(output)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    try:
        actual_commit = _command_output(("git", "rev-parse", "HEAD"), openpi_root)
        if actual_commit != args.expected_openpi_commit:
            raise RuntimeError(
                "OpenPI commit mismatch: expected %s, got %s"
                % (args.expected_openpi_commit, actual_commit)
            )
        source_hashes = _snapshot_sources(armbench_root, output)
        probe = BoundedOpenPIClient(
            args.host,
            args.port,
            startup_timeout_s=args.server_startup_timeout_s,
            inference_timeout_s=args.inference_timeout_s,
        )
        try:
            metadata = probe.get_server_metadata()
        finally:
            probe.close()
        server_source = armbench_root / "integrations/openpi/serve_policy_attested.py"
        _validate_server_attestation(metadata, args, _sha256(server_source))
        validate_policy_sampling_server_contract(metadata)
        _write_json(output / "resolved_protocol.json", resolved_protocol(args, cells))
        _write_json(
            output / "environment.json",
            _capture_environment(
                args, openpi_root, armbench_root, metadata, source_hashes
            ),
        )
        _write_derived(output, rows, len(cells))

        suite = benchmark.get_benchmark_dict()[args.task_suite]()
        cells_by_task: dict[int, list[ExperimentCell]] = {}
        for cell in cells:
            cells_by_task.setdefault(cell.task_id, []).append(cell)
        for task_id, task_cells in cells_by_task.items():
            task = suite.get_task(task_id)
            task_description = str(task.language)
            initial_states = suite.get_task_init_states(task_id)
            environment = _make_libero_environment(task, args.seed)
            try:
                for cell in task_cells:
                    if cell.episode_index >= len(initial_states):
                        raise IndexError("episode index exceeds available initial states")
                    environment.seed(args.seed)
                    request_builder = IndependentLiberoRequestBuilder(
                        task_description=task_description,
                        task_suite=cell.task_suite,
                        task_id=cell.task_id,
                        episode_index=cell.episode_index,
                        seed=args.seed,
                        replan_steps=args.submit_every_ticks,
                    )
                    provider_factory = OpenPIIndependentClockProviderFactory(
                        host=args.host,
                        port=args.port,
                        startup_timeout_s=args.server_startup_timeout_s,
                        inference_timeout_s=args.inference_timeout_s,
                    )
                    started = time.perf_counter()
                    result = run_libero_independent_clock_episode(
                        environment,
                        provider_factory,
                        initial_states[cell.episode_index],
                        request_builder,
                        control_period_s=args.control_period_ms / 1000.0,
                        deadline_s=args.deadline_ms / 1000.0,
                        max_task_steps=(
                            args.max_task_steps
                            if args.max_task_steps is not None
                            else SUITE_MAX_STEPS[args.task_suite]
                        ),
                        num_steps_wait=args.num_steps_wait,
                        submit_every_ticks=args.submit_every_ticks,
                        startup_timeout_s=args.server_startup_timeout_s,
                        shutdown_timeout_s=args.shutdown_timeout_s,
                        record_video=args.video_mode != "none",
                    )
                    wall_time_s = time.perf_counter() - started
                    episode_directory = output / "episodes" / cell.episode_id
                    episode_directory.mkdir(parents=True, exist_ok=False)
                    np.save(
                        episode_directory / "initial_state.npy",
                        np.asarray(initial_states[cell.episode_index]),
                        allow_pickle=False,
                    )
                    payload = result.to_dict()
                    _write_json(episode_directory / "runtime.json", payload)
                    video_path = None
                    should_write_video = args.video_mode == "all" or (
                        args.video_mode == "failures" and not result.task_success
                    )
                    if should_write_video:
                        video_path = "videos/%s.mp4" % cell.episode_id
                        _write_video(
                            output / video_path,
                            result.replay_frames,
                            fps=int(round(1000.0 / args.control_period_ms)),
                        )
                    row = episode_row(
                        cell,
                        task_description,
                        args.seed,
                        payload,
                        wall_time_s,
                        video_path,
                    )
                    rows.append(row)
                    _write_derived(output, rows, len(cells))
            finally:
                environment.close()
    except Exception as error:
        errors.append(
            {
                "type": type(error).__name__,
                "message": str(error),
                "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        )
        _write_json(output / "run_error.json", {"errors": errors})

    aggregate = aggregate_rows(rows, len(cells))
    overlap_complete = (
        aggregate["episodes_with_inference_overlap"] == len(rows) and bool(rows)
    )
    integrity_errors = []
    if len(rows) != len(cells):
        integrity_errors.append("completed rollout count does not match the matrix")
    if errors:
        integrity_errors.append("run_error.json records a runtime failure")
    if rows and not overlap_complete:
        integrity_errors.append("one or more episodes lack measured inference overlap")
    _write_json(
        output / "integrity.json",
        {
            "schema_version": SCHEMA_VERSION,
            "valid": not integrity_errors,
            "errors": integrity_errors,
            "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )
    _write_manifest(output)
    return 0 if not integrity_errors else 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "run"):
        child = subparsers.add_parser(command, allow_abbrev=False)
        child.add_argument("--task-suite", choices=tuple(SUITE_TASK_COUNTS), default="libero_spatial")
        child.add_argument("--task-ids", default="0")
        child.add_argument("--episode-indices", default="0")
    run = subparsers.choices["run"]
    run.add_argument("--output-dir", required=True)
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8000)
    run.add_argument("--openpi-root", default="/app")
    run.add_argument("--armbench-root", default="/armbench")
    run.add_argument("--expected-openpi-commit", default=OPENPI_COMMIT)
    run.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    run.add_argument(
        "--server-launch-args",
        default=os.environ.get("ARMBENCH_SERVER_ARGS") or os.environ.get("SERVER_ARGS"),
    )
    run.add_argument("--server-startup-timeout-s", type=float, default=1200.0)
    run.add_argument("--inference-timeout-s", type=float, default=600.0)
    run.add_argument("--shutdown-timeout-s", type=float, default=5.0)
    run.add_argument("--control-period-ms", type=float, default=LIBERO_CONTROL_PERIOD_MS)
    run.add_argument("--deadline-ms", type=float, default=200.0)
    run.add_argument("--submit-every-ticks", type=int, default=1)
    run.add_argument("--num-steps-wait", type=int, default=10)
    run.add_argument("--max-task-steps", type=int)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--video-mode", choices=("none", "failures", "all"), default="failures")
    run.add_argument("--allow-unattested-server", action="store_true")
    return parser


def _validate_run_arguments(args: argparse.Namespace) -> None:
    if args.port <= 0 or args.port > 65535:
        raise ValueError("port must be in [1, 65535]")
    for name in (
        "server_startup_timeout_s",
        "inference_timeout_s",
        "control_period_ms",
        "deadline_ms",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("%s must be finite and positive" % name)
    if args.num_steps_wait != 10:
        raise ValueError("official LIBERO evaluation requires 10 stabilization steps")
    if args.submit_every_ticks <= 0:
        raise ValueError("submit_every_ticks must be positive")
    if args.max_task_steps is not None and args.max_task_steps <= 0:
        raise ValueError("max_task_steps must be positive")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if args.allow_unattested_server:
        raise ValueError("independent-clock scored pilots require server attestation")
    _validate_server_launch_args(args.server_launch_args, args.checkpoint)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        cells = resolve_cells(args)
        if args.command == "plan":
            print(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "planned_rollouts": len(cells),
                        "cells": [cell.to_dict() for cell in cells],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        _validate_run_arguments(args)
        return execute(args, cells)
    except Exception as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
