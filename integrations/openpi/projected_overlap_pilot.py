"""Plan, run, and validate the paired pi0.5 projected-overlap pilot."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import math
import os
import pathlib
import platform
import re
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from integrations.openpi.libero_runtime_eval import (
    BoundedOpenPIClient,
    SUITE_MAX_STEPS,
    SUITE_TASK_COUNTS,
    _make_libero_environment,
    _write_video,
)
from integrations.openpi.projected_overlap_runtime import (
    OVERLAP_UNCONDITIONED,
    PROJECTED_OVERLAP,
    VALID_OVERLAP_METHODS,
    OverlapRuntimeConfig,
    run_overlap_episode,
)
from integrations.openpi.serve_policy_attested import (
    ATTESTATION_SCHEMA_VERSION,
    DEFAULT_CHECKPOINT,
    OPENPI_COMMIT,
    OPENPI_PROJECTED_CONDITIONING_COMMIT,
    POLICY_CONDITIONING_METADATA_FIELD,
    POLICY_SAMPLING_METADATA_FIELD,
    POLICY_SAMPLING_SCORED_NAMESPACE,
    build_policy_sampling_control,
    policy_conditioning_contract,
    policy_sampling_contract,
)


SCHEMA_VERSION = "armbench.pi05_projected_overlap_pilot.v1"
DEFAULT_CHECKPOINT_CONTENT_SHA256 = (
    "9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5"
)
SOURCE_FILES = (
    "integrations/openpi/projected_overlap_runtime.py",
    "integrations/openpi/projected_overlap_pilot.py",
    "integrations/openpi/realtime_chunking.py",
    "integrations/openpi/serve_policy_attested.py",
    "integrations/openpi/patches/projected_conditioning_v1.json",
)


class PilotValidationError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class PilotCell:
    task_suite: str
    task_id: int
    episode_index: int
    method: str
    condition_order: int
    execute_horizon: int
    inference_delay_steps: int

    @property
    def pair_id(self) -> str:
        return "%s__task_%02d__episode_%02d" % (
            self.task_suite,
            self.task_id,
            self.episode_index,
        )

    @property
    def episode_id(self) -> str:
        return "%s__%s" % (self.pair_id, self.method)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **dataclasses.asdict(self),
            "pair_id": self.pair_id,
            "episode_id": self.episode_id,
        }


def build_cells(
    task_suite: str,
    task_ids: Sequence[int],
    episode_indices: Sequence[int],
    *,
    execute_horizon: int,
    inference_delay_steps: int,
) -> List[PilotCell]:
    cells = []
    pair_index = 0
    for task_id in task_ids:
        for episode_index in episode_indices:
            order = (
                VALID_OVERLAP_METHODS
                if pair_index % 2 == 0
                else tuple(reversed(VALID_OVERLAP_METHODS))
            )
            for condition_order, method in enumerate(order):
                cells.append(
                    PilotCell(
                        task_suite=task_suite,
                        task_id=int(task_id),
                        episode_index=int(episode_index),
                        method=method,
                        condition_order=condition_order,
                        execute_horizon=execute_horizon,
                        inference_delay_steps=inference_delay_steps,
                    )
                )
            pair_index += 1
    return cells


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


def _write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _json_safe(value), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: pathlib.Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            PilotValidationError("nonfinite JSON token: %s" % token)
        ),
    )


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(
    command: Sequence[str], cwd: Optional[pathlib.Path] = None
) -> str:
    completed = subprocess.run(
        list(command),
        cwd=None if cwd is None else str(cwd),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.rstrip("\r\n")


def _prepare_output(path: pathlib.Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("output directory must be absent or empty")
    path.mkdir(parents=True, exist_ok=True)
    (path / "videos").mkdir(exist_ok=True)


def _write_root_manifest(output: pathlib.Path) -> None:
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    canonical = json.dumps(
        files, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    _write_json(
        output / "manifest.json",
        {
            "schema_version": "armbench.root_manifest.v1",
            "files": files,
            "files_sha256": hashlib.sha256(canonical).hexdigest(),
        },
    )


def _mcnemar_exact_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(wins, losses) + 1)
    ) / float(2**discordant)
    return min(1.0, 2.0 * tail)


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None]
    return None if not finite else float(np.mean(finite))


def summarize(
    episodes: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_pair: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for episode in episodes:
        by_pair.setdefault(str(episode["pair_id"]), {})[
            str(episode["method"])
        ] = episode
    valid_pairs = [
        pair
        for pair in by_pair.values()
        if set(pair) == set(VALID_OVERLAP_METHODS)
    ]
    wins = sum(
        bool(pair[PROJECTED_OVERLAP]["success"])
        and not bool(pair[OVERLAP_UNCONDITIONED]["success"])
        for pair in valid_pairs
    )
    losses = sum(
        bool(pair[OVERLAP_UNCONDITIONED]["success"])
        and not bool(pair[PROJECTED_OVERLAP]["success"])
        for pair in valid_pairs
    )
    methods = {}
    for method in VALID_OVERLAP_METHODS:
        selected_episodes = [row for row in episodes if row["method"] == method]
        selected_queries = [
            row
            for row in queries
            if row["method"] == method and not row["bootstrap"]
        ]
        methods[method] = {
            "rollouts": len(selected_episodes),
            "successes": sum(bool(row["success"]) for row in selected_episodes),
            "success_rate": (
                None
                if not selected_episodes
                else sum(bool(row["success"]) for row in selected_episodes)
                / float(len(selected_episodes))
            ),
            "mean_policy_queries": _mean(
                float(row["policy_queries"]) for row in selected_episodes
            ),
            "mean_seam_motion_l2": _mean(
                row.get("seam_motion_l2") for row in selected_queries
            ),
            "mean_seam_gripper_abs": _mean(
                row.get("seam_gripper_abs") for row in selected_queries
            ),
            "max_model_residual": (
                None
                if method != PROJECTED_OVERLAP or not selected_queries
                else max(float(row["max_model_residual"]) for row in selected_queries)
            ),
        }
    reference_rate = methods[OVERLAP_UNCONDITIONED]["success_rate"]
    projected_rate = methods[PROJECTED_OVERLAP]["success_rate"]
    return {
        "schema_version": SCHEMA_VERSION,
        "pilot_only": True,
        "paired_rollouts": len(valid_pairs),
        "methods": methods,
        "projected_minus_unconditioned_success_rate": (
            None
            if reference_rate is None or projected_rate is None
            else projected_rate - reference_rate
        ),
        "projected_wins": wins,
        "unconditioned_wins": losses,
        "ties": len(valid_pairs) - wins - losses,
        "mcnemar_exact_p": _mcnemar_exact_p(wins, losses),
    }


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    methods = summary["methods"]
    lines = [
        "# pi0.5 projected-overlap pilot",
        "",
        "This is a paired pilot, not a confirmatory efficacy claim. Both methods use "
        "the same fixed-width overlap scheduler and keyed policy noise.",
        "",
        "| Method | Success | Rate | Mean queries | Mean motion seam | Mean gripper seam | Max residual |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in VALID_OVERLAP_METHODS:
        row = methods[method]
        lines.append(
            "| %s | %d/%d | %.3f | %.2f | %s | %s | %s |"
            % (
                method,
                row["successes"],
                row["rollouts"],
                row["success_rate"],
                row["mean_policy_queries"],
                "n/a"
                if row["mean_seam_motion_l2"] is None
                else "%.6f" % row["mean_seam_motion_l2"],
                "n/a"
                if row["mean_seam_gripper_abs"] is None
                else "%.6f" % row["mean_seam_gripper_abs"],
                "n/a"
                if row["max_model_residual"] is None
                else "%.3g" % row["max_model_residual"],
            )
        )
    lines.extend(
        [
            "",
            "- Paired rollouts: %d" % summary["paired_rollouts"],
            "- Success difference: %.3f"
            % summary["projected_minus_unconditioned_success_rate"],
            "- Wins/losses/ties: %d/%d/%d"
            % (
                summary["projected_wins"],
                summary["unconditioned_wins"],
                summary["ties"],
            ),
            "- Exact McNemar p: %.6g" % summary["mcnemar_exact_p"],
            "",
        ]
    )
    return "\n".join(lines)


def _validate_server_metadata(
    metadata: Mapping[str, Any], armbench_root: pathlib.Path
) -> Mapping[str, Any]:
    attestation = metadata.get("armbench_server_attestation")
    if not isinstance(attestation, Mapping):
        raise PilotValidationError("server attestation is missing")
    if attestation.get("schema_version") != ATTESTATION_SCHEMA_VERSION:
        raise PilotValidationError("server attestation schema mismatch")
    expected = {
        "openpi_commit": OPENPI_PROJECTED_CONDITIONING_COMMIT,
        "openpi_upstream_base_commit": OPENPI_COMMIT,
        "openpi_tracked_clean": True,
        "checkpoint_uri": DEFAULT_CHECKPOINT,
        "checkpoint_content_sha256": DEFAULT_CHECKPOINT_CONTENT_SHA256,
    }
    for key, value in expected.items():
        if attestation.get(key) != value:
            raise PilotValidationError("server attestation mismatch: %s" % key)
    extension = _load_json(
        armbench_root
        / "integrations/openpi/patches/projected_conditioning_v1.json"
    )
    if attestation.get("openpi_extension_files") != extension.get(
        "production_source_sha256"
    ):
        raise PilotValidationError("server extension source hashes mismatch")
    server_source = armbench_root / "integrations/openpi/serve_policy_attested.py"
    if attestation.get("server_source_sha256") != _sha256_file(server_source):
        raise PilotValidationError("server source hash mismatch")
    if metadata.get(POLICY_SAMPLING_METADATA_FIELD) != policy_sampling_contract():
        raise PilotValidationError("policy sampling contract mismatch")
    if metadata.get(POLICY_CONDITIONING_METADATA_FIELD) != policy_conditioning_contract():
        raise PilotValidationError("policy conditioning contract mismatch")
    return attestation


def _protocol(args: argparse.Namespace, cells: Sequence[PilotCell]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "pilot_only": True,
        "research_question": (
            "Under the same fixed-width overlap scheduler and explicit pi0.5 noise, "
            "does projected flow conditioning improve closed-loop task success or action seams?"
        ),
        "policy_config": "pi05_libero",
        "checkpoint": DEFAULT_CHECKPOINT,
        "checkpoint_content_sha256": DEFAULT_CHECKPOINT_CONTENT_SHA256,
        "openpi_upstream_commit": OPENPI_COMMIT,
        "openpi_extension_commit": OPENPI_PROJECTED_CONDITIONING_COMMIT,
        "task_suite": args.task_suite,
        "task_ids": sorted({cell.task_id for cell in cells}),
        "episode_indices": sorted({cell.episode_index for cell in cells}),
        "methods": list(VALID_OVERLAP_METHODS),
        "execute_horizon": args.execute_horizon,
        "inference_delay_steps": args.inference_delay_steps,
        "action_horizon": 10,
        "control_period_ms": 50.0,
        "fixed_delay_ms": args.inference_delay_steps * 50.0,
        "sampling_seed": args.sampling_seed,
        "pairing_key_fields": [
            "task_suite",
            "task_id",
            "episode_index",
            "execute_horizon",
            "query_index",
        ],
        "bootstrap_rule": "query 0 is unconditioned in both methods",
        "scheduler": "old[:d] + new[d:E], then new[E:H] + zeros(E)",
        "matrix": [cell.to_dict() for cell in cells],
        "planned_rollouts": len(cells),
        "video_mode": args.video_mode,
        "limitations": [
            "The blocking evaluator simulates fixed-step overlap after response return; it is not an OS-real-time loop.",
            "This pilot is not powered for a publication claim.",
            "Projected flow inpainting is not RTC pseudoinverse guidance.",
        ],
    }


def execute(args: argparse.Namespace, cells: Sequence[PilotCell]) -> int:
    output = pathlib.Path(args.output_dir).resolve()
    openpi_root = pathlib.Path(args.openpi_root).resolve()
    armbench_root = pathlib.Path(args.armbench_root).resolve()
    _prepare_output(output)
    if _command_output(("git", "rev-parse", "HEAD"), openpi_root) != OPENPI_PROJECTED_CONDITIONING_COMMIT:
        raise RuntimeError("OpenPI extension commit mismatch")
    if _command_output(("git", "status", "--porcelain"), openpi_root):
        raise RuntimeError("OpenPI extension worktree must be clean")
    armbench_commit = _command_output(("git", "rev-parse", "HEAD"), armbench_root)
    if _command_output(("git", "status", "--porcelain"), armbench_root):
        raise RuntimeError("ArmBench worktree must be clean")

    protocol = _protocol(args, cells)
    _write_json(output / "resolved_protocol.json", protocol)
    _write_json(
        output / "progress.json",
        {"planned": len(cells), "completed": 0, "complete": False},
    )
    client = BoundedOpenPIClient(
        args.host,
        args.port,
        startup_timeout_s=args.server_startup_timeout_s,
        inference_timeout_s=args.inference_timeout_s,
    )
    episodes: List[Dict[str, Any]] = []
    queries: List[Dict[str, Any]] = []
    try:
        metadata = client.get_server_metadata()
        attestation = _validate_server_metadata(metadata, armbench_root)
        source_hashes = {
            relative: _sha256_file(armbench_root / relative)
            for relative in SOURCE_FILES
        }
        _write_json(
            output / "environment.json",
            {
                "schema_version": SCHEMA_VERSION,
                "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "python": sys.version,
                "platform": platform.platform(),
                "command": sys.argv,
                "armbench_commit": armbench_commit,
                "armbench_status": "",
                "openpi_commit": OPENPI_PROJECTED_CONDITIONING_COMMIT,
                "openpi_status": "",
                "source_sha256": source_hashes,
                "server_attestation": attestation,
                "nvidia_smi": _command_output(
                    (
                        "nvidia-smi",
                        "--query-gpu=name,driver_version,memory.total",
                        "--format=csv,noheader",
                    )
                ),
            },
        )

        from libero.libero import benchmark

        benchmark_class = benchmark.get_benchmark_dict()[args.task_suite]
        task_suite = benchmark_class()
        cells_by_task: Dict[int, List[PilotCell]] = {}
        for cell in cells:
            cells_by_task.setdefault(cell.task_id, []).append(cell)
        for task_id, task_cells in cells_by_task.items():
            task = task_suite.get_task(task_id)
            description = str(task.language)
            initial_states = task_suite.get_task_init_states(task_id)
            environment = _make_libero_environment(task, args.environment_seed)
            try:
                for cell in task_cells:
                    environment.seed(args.environment_seed)

                    def sampling_builder(query_index: int, cell: PilotCell = cell):
                        return build_policy_sampling_control(
                            POLICY_SAMPLING_SCORED_NAMESPACE,
                            args.sampling_seed,
                            [
                                cell.task_suite,
                                cell.task_id,
                                cell.episode_index,
                                cell.execute_horizon,
                            ],
                            query_index,
                        )

                    config = OverlapRuntimeConfig(
                        method=cell.method,
                        execute_horizon=cell.execute_horizon,
                        inference_delay_steps=cell.inference_delay_steps,
                        action_horizon=10,
                        max_task_steps=(
                            args.max_task_steps
                            if args.max_task_steps is not None
                            else SUITE_MAX_STEPS[args.task_suite]
                        ),
                        num_steps_wait=10,
                        resize_size=224,
                        record_video=args.video_mode != "none",
                    )
                    logging.info("Starting %s", cell.episode_id)
                    started = time.perf_counter()
                    result = run_overlap_episode(
                        environment,
                        client,
                        initial_states[cell.episode_index],
                        description,
                        config,
                        sampling_control_builder=sampling_builder,
                    )
                    wall_time_s = time.perf_counter() - started
                    video_path = None
                    write_video = args.video_mode == "all" or (
                        args.video_mode == "failures" and not result.success
                    )
                    if write_video:
                        if not result.replay_frames:
                            raise RuntimeError("video was required but no frames were captured")
                        video_path = "videos/%s__%s.mp4" % (
                            re.sub(r"[^A-Za-z0-9_.-]+", "_", cell.episode_id),
                            "success" if result.success else "failure",
                        )
                        _write_video(output / video_path, result.replay_frames)
                    episode = {
                        "schema_version": SCHEMA_VERSION,
                        **cell.to_dict(),
                        "task_description": description,
                        **{
                            key: value
                            for key, value in result.to_dict().items()
                            if key != "query_records"
                        },
                        "wall_time_s": wall_time_s,
                        "video_path": video_path,
                    }
                    episodes.append(episode)
                    for record in result.query_records:
                        queries.append(
                            {
                                "schema_version": SCHEMA_VERSION,
                                **cell.to_dict(),
                                **record.to_dict(),
                            }
                        )
                    _write_json(output / "episodes.json", episodes)
                    _write_json(output / "queries.json", queries)
                    _write_json(
                        output / "progress.json",
                        {
                            "planned": len(cells),
                            "completed": len(episodes),
                            "complete": False,
                        },
                    )
                    if result.failure_type is not None:
                        raise RuntimeError(
                            "runtime failure in %s: %s"
                            % (cell.episode_id, result.failure_message)
                        )
            finally:
                environment.close()
    finally:
        client.close()

    summary = summarize(episodes, queries)
    _write_json(output / "summary.json", summary)
    (output / "summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    _write_json(
        output / "progress.json",
        {"planned": len(cells), "completed": len(episodes), "complete": True},
    )
    _write_root_manifest(output)
    validate_artifact(output)
    return 0


def validate_artifact(output: pathlib.Path) -> Dict[str, Any]:
    output = output.resolve()
    root_manifest = _load_json(output / "manifest.json")
    files = root_manifest.get("files")
    if not isinstance(files, list):
        raise PilotValidationError("root manifest file inventory is missing")
    recorded_paths = [str(item.get("path")) for item in files]
    actual_paths = sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    if recorded_paths != actual_paths:
        raise PilotValidationError("root manifest inventory is incomplete")
    for item in files:
        path = output / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise PilotValidationError("artifact file size mismatch: %s" % item["path"])
        if _sha256_file(path) != item["sha256"]:
            raise PilotValidationError("artifact file hash mismatch: %s" % item["path"])
    canonical = json.dumps(
        files, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    if hashlib.sha256(canonical).hexdigest() != root_manifest.get("files_sha256"):
        raise PilotValidationError("root manifest aggregate hash mismatch")

    protocol = _load_json(output / "resolved_protocol.json")
    progress = _load_json(output / "progress.json")
    episodes = _load_json(output / "episodes.json")
    queries = _load_json(output / "queries.json")
    summary = _load_json(output / "summary.json")
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise PilotValidationError("protocol schema mismatch")
    planned = int(protocol["planned_rollouts"])
    if progress != {"planned": planned, "completed": planned, "complete": True}:
        raise PilotValidationError("pilot progress is incomplete")
    if len(episodes) != planned:
        raise PilotValidationError("episode count does not match the plan")
    planned_ids = [row["episode_id"] for row in protocol["matrix"]]
    if [row["episode_id"] for row in episodes] != planned_ids:
        raise PilotValidationError("episode order or identity mismatch")
    if protocol.get("video_mode") == "all":
        for episode in episodes:
            video_path = episode.get("video_path")
            if not isinstance(video_path, str) or not (output / video_path).is_file():
                raise PilotValidationError("all-video protocol omitted an episode video")

    queries_by_episode: Dict[str, List[Mapping[str, Any]]] = {}
    for query in queries:
        queries_by_episode.setdefault(str(query["episode_id"]), []).append(query)
    for episode in episodes:
        selected = queries_by_episode.get(str(episode["episode_id"]), [])
        if [row["query_index"] for row in selected] != list(range(len(selected))):
            raise PilotValidationError("query indices are not contiguous")
        if not selected or selected[0]["bootstrap"] is not True:
            raise PilotValidationError("every episode requires one bootstrap query")
        if any(row["executed_steps"] != episode["execute_horizon"] for row in selected[:-1]):
            raise PilotValidationError("nonfinal query did not advance exactly E steps")
        for row in selected:
            if not row.get("sampling_key_sha256") or not row.get("sampling_noise_sha256"):
                raise PilotValidationError("scored query omitted sampling audit")
            conditioned = episode["method"] == PROJECTED_OVERLAP and not row["bootstrap"]
            fields = (
                "condition_raw_actions_sha256",
                "condition_model_actions_sha256",
                "condition_mask_sha256",
                "max_model_residual",
            )
            if conditioned:
                if any(row.get(field) is None for field in fields):
                    raise PilotValidationError("projected query omitted conditioning audit")
                if float(row["max_model_residual"]) >= 1e-6:
                    raise PilotValidationError("projected query residual failed")
            elif any(row.get(field) is not None for field in fields):
                raise PilotValidationError("unconditioned query contains conditioning audit")

    by_pair_and_query: Dict[Tuple[str, int], Dict[str, Mapping[str, Any]]] = {}
    for query in queries:
        by_pair_and_query.setdefault(
            (str(query["pair_id"]), int(query["query_index"])), {}
        )[str(query["method"])] = query
    for methods in by_pair_and_query.values():
        if set(methods) == set(VALID_OVERLAP_METHODS):
            if (
                methods[OVERLAP_UNCONDITIONED]["sampling_key_sha256"]
                != methods[PROJECTED_OVERLAP]["sampling_key_sha256"]
            ):
                raise PilotValidationError("paired methods used different sampling keys")
    recomputed = summarize(episodes, queries)
    if summary != recomputed:
        raise PilotValidationError("pilot summary is not reproducible")
    return summary


def _parse_selection(value: str, upper: int, name: str) -> List[int]:
    if value == "all":
        return list(range(upper))
    result = []
    for token in value.split(","):
        try:
            number = int(token)
        except ValueError as exc:
            raise ValueError("%s must be comma-separated integers or all" % name) from exc
        if number < 0 or number >= upper or number in result:
            raise ValueError("%s contains an invalid or duplicate value" % name)
        result.append(number)
    if not result:
        raise ValueError("%s cannot be empty" % name)
    return result


def _add_matrix_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task-suite", choices=tuple(SUITE_TASK_COUNTS), default="libero_10")
    parser.add_argument("--task-ids", default="all")
    parser.add_argument("--episode-indices", default="0,1")
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--inference-delay-steps", type=int, default=4)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", allow_abbrev=False)
    _add_matrix_args(plan)
    run = commands.add_parser("run", allow_abbrev=False)
    _add_matrix_args(run)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8000)
    run.add_argument("--openpi-root", default="/app")
    run.add_argument("--armbench-root", default="/armbench")
    run.add_argument("--sampling-seed", type=int, default=20260805)
    run.add_argument("--environment-seed", type=int, default=7)
    run.add_argument("--max-task-steps", type=int)
    run.add_argument("--video-mode", choices=("none", "failures", "all"), default="failures")
    run.add_argument("--server-startup-timeout-s", type=float, default=1200.0)
    run.add_argument("--inference-timeout-s", type=float, default=600.0)
    validate = commands.add_parser("validate", allow_abbrev=False)
    validate.add_argument("output_dir")
    return parser


def _resolve_cells(args: argparse.Namespace) -> List[PilotCell]:
    if args.execute_horizon != 5 or args.inference_delay_steps != 4:
        raise ValueError("the frozen pilot requires H=10, E=5, d=4")
    return build_cells(
        args.task_suite,
        _parse_selection(
            args.task_ids, SUITE_TASK_COUNTS[args.task_suite], "task_ids"
        ),
        _parse_selection(args.episode_indices, 50, "episode_indices"),
        execute_horizon=args.execute_horizon,
        inference_delay_steps=args.inference_delay_steps,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "validate":
        summary = validate_artifact(pathlib.Path(args.output_dir))
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
        return 0
    cells = _resolve_cells(args)
    if args.command == "plan":
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "planned_rollouts": len(cells),
                    "paired_rollouts": len(cells) // 2,
                    "matrix": [cell.to_dict() for cell in cells],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.port <= 0 or args.port > 65535:
        parser.error("port must be within [1, 65535]")
    if args.max_task_steps is not None and args.max_task_steps <= 0:
        parser.error("max-task-steps must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return execute(args, cells)


if __name__ == "__main__":
    raise SystemExit(main())
