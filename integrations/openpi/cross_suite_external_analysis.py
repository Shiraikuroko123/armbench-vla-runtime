"""Strict read-only analysis for the frozen pi0.5 cross-suite validation."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import io
import json
import pathlib
import re
import shutil
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from integrations.openpi.latency_aligned_analysis import (
    ASYNC_UNGUARDED,
    LATENCY_ALIGNED,
    Episode,
    PairKey,
    _holm_adjust,
    _mcnemar_exact_p,
    _paired_bootstrap_interval,
    _wilson_interval,
)
from integrations.openpi.libero_compose_run import validate_run_manifest


ANALYSIS_SCHEMA_VERSION = "armbench.pi05_cross_suite_external_analysis.v1"
SOURCE_SCHEMA_VERSION = "armbench.pi05_libero_async.v1"
ROOT_SCHEMA_VERSION = "armbench.pi05_libero_container_run.v1"
FROZEN_SUITES = ("libero_object", "libero_goal", "libero_10")
FROZEN_RUN_IDS = (
    "pi05_libero_object_alignment_external_001",
    "pi05_libero_goal_alignment_external_001",
    "pi05_libero_10_alignment_external_001",
)
FROZEN_TASK_IDS = tuple(range(10))
FROZEN_EPISODE_INDICES = tuple(range(5))
FROZEN_MODES = (ASYNC_UNGUARDED, LATENCY_ALIGNED)
FROZEN_REPLAN_STEPS = 5
FROZEN_LATENCY_STEPS = 4
FROZEN_SEED = 7
FROZEN_BOOTSTRAP_RESAMPLES = 10_000
FROZEN_ARMBENCH_RUN_COMMIT = "92ff977fb830505118f7a522ed4a8d91b3a02965"
FROZEN_ALIGNMENT_IMPLEMENTATION_COMMIT = (
    "cccbe351a1a4523c65d01eff2997580f7ca83649"
)
FROZEN_OPENPI_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
FROZEN_POLICY_CONFIG = "pi05_libero"
FROZEN_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
FROZEN_CHECKPOINT_CONTENT_SHA256 = (
    "9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5"
)
EMPTY_GIT_DIFF_SHA256 = hashlib.sha256(b"").hexdigest()
SOURCE_CSV = "evaluation/per_episode.csv"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_UINT = re.compile(r"0|[1-9][0-9]*\Z")
_REQUIRED_COLUMNS = frozenset(
    (
        "schema_version", "episode_id", "pair_id", "condition_order",
        "task_suite", "task_id", "episode_index", "task_description", "mode",
        "replan_steps", "latency_steps", "injected_latency_ms",
        "fixed_refresh_interval", "seed", "success", "initial_state_sha256",
        "policy_queries", "failure_category", "failure_type", "failure_message",
        "video_required",
        "video_path",
        "video_error_type",
        "video_error_message",
    )
)
_OFFICIAL_SUITE_TASK_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
_STEP_BUDGET_SEMANTICS = (
    "Injected delay steps consume the same environment-step budget as task actions."
)


@dataclasses.dataclass(frozen=True)
class SourceEvidence:
    suite: str
    run_id: str
    root: pathlib.Path
    csv_data: bytes
    csv_sha256: str
    csv_bytes: int
    root_manifest_sha256: str
    evaluation_manifest_sha256: str
    protocol_sha256: str
    environment_sha256: str
    independently_validated_files: int
    videos_verified: int
    complete: bool


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object_bytes(data: bytes, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError("%s is not valid UTF-8 JSON: %s" % (label, exc)) from exc
    if not isinstance(value, dict):
        raise ValueError("%s must contain a JSON object" % label)
    return value


def _strict_int(value: Any, label: str) -> int:
    text = str(value)
    if not _UINT.fullmatch(text):
        raise ValueError("%s must be a canonical nonnegative integer" % label)
    return int(text)


def _strict_bool(value: Any, label: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError("%s must be exactly True or False" % label)


def _check_manifest_record(
    manifest: Mapping[str, Any], relative: str, data: bytes, label: str
) -> None:
    files = manifest.get("files")
    record = files.get(relative) if isinstance(files, Mapping) else None
    if not isinstance(record, Mapping):
        raise ValueError("%s does not protect %s" % (label, relative))
    if record.get("bytes") != len(data):
        raise ValueError("%s byte count mismatch for %s" % (label, relative))
    if record.get("sha256") != _sha256_bytes(data):
        raise ValueError("%s SHA-256 mismatch for %s" % (label, relative))


def _safe_relative_path(value: Any) -> Optional[pathlib.PurePosixPath]:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    candidate = pathlib.PurePosixPath(value)
    if (
        candidate.is_absolute()
        or candidate.as_posix() != value
        or any(part in ("", ".", "..") for part in candidate.parts)
    ):
        return None
    return candidate


def _snapshot_regular_file(path: pathlib.Path, relative: str) -> Tuple[int, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("manifest-bound file must be regular: %s" % relative)
    before = path.stat()
    if before.st_size <= 0:
        raise ValueError("manifest-bound file is empty: %s" % relative)
    digest = _sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("manifest-bound file changed while hashing: %s" % relative)
    return after.st_size, digest


def _check_manifest_file_record(
    manifest: Mapping[str, Any],
    relative: str,
    size: int,
    digest: str,
    label: str,
) -> None:
    files = manifest.get("files")
    record = files.get(relative) if isinstance(files, Mapping) else None
    if not isinstance(record, Mapping):
        raise ValueError("%s does not protect %s" % (label, relative))
    if record.get("bytes") != size:
        raise ValueError("%s byte count mismatch for %s" % (label, relative))
    if record.get("sha256") != digest:
        raise ValueError("%s SHA-256 mismatch for %s" % (label, relative))


def _video_paths(csv_data: bytes, suite: str) -> List[pathlib.PurePosixPath]:
    try:
        text = csv_data.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("%s per_episode.csv is not UTF-8" % suite) from exc
    with io.StringIO(text, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        missing = {
            "video_path",
            "video_error_type",
            "video_error_message",
        } - set(fields)
        if missing:
            raise ValueError(
                "%s CSV is missing video fields: %s"
                % (suite, ", ".join(sorted(missing)))
            )
        paths: List[pathlib.PurePosixPath] = []
        for number, row in enumerate(reader, 2):
            if row.get("video_error_type") or row.get("video_error_message"):
                raise ValueError(
                    "%s row %d records a video encoding error" % (suite, number)
                )
            value = row.get("video_path")
            safe = _safe_relative_path(value)
            if safe is None or not safe.parts or safe.parts[0] != "videos":
                raise ValueError("%s row %d has unsafe or empty video_path" % (suite, number))
            paths.append(safe)
    if len(paths) != 100:
        raise ValueError("%s has %d video paths, expected 100" % (suite, len(paths)))
    if len(set(paths)) != 100:
        raise ValueError("%s video_path values must be unique per episode" % suite)
    return paths


def _strict_full_validation(root: pathlib.Path, phase: str) -> Mapping[str, Any]:
    validation = validate_run_manifest(root)
    if (
        not isinstance(validation, Mapping)
        or validation.get("valid") is not True
        or validation.get("complete") is not True
    ):
        errors = validation.get("errors", []) if isinstance(validation, Mapping) else []
        raise ValueError(
            "%s strict full run validation failed%s"
            % (phase, ((": " + "; ".join(map(str, list(errors)[:3]))) if errors else ""))
        )
    files_checked = validation.get("files_checked")
    if isinstance(files_checked, bool) or not isinstance(files_checked, int) or files_checked <= 0:
        raise ValueError("%s strict full run validation returned no file count" % phase)
    return validation


def _read_manifest_bound(
    path: pathlib.Path,
    relative: str,
    evaluation_manifest: Mapping[str, Any],
    root_manifest: Mapping[str, Any],
) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("manifest-bound file must be regular: %s" % relative)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError("cannot snapshot %s: %s" % (relative, exc)) from exc
    _check_manifest_record(evaluation_manifest, relative, data, "evaluation manifest")
    _check_manifest_record(root_manifest, "evaluation/" + relative, data, "root manifest")
    return data


def _require_exact(value: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for field, wanted in expected.items():
        observed = value.get(field)
        if type(observed) is not type(wanted) or observed != wanted:
            raise ValueError("%s mismatch: %s" % (label, field))


def _validate_identity(
    protocol: Mapping[str, Any], environment: Mapping[str, Any], suite: str
) -> None:
    _require_exact(
        protocol,
        {
            "schema_version": SOURCE_SCHEMA_VERSION,
            "openpi_commit": FROZEN_OPENPI_COMMIT,
            "policy_config": FROZEN_POLICY_CONFIG,
            "declared_checkpoint": FROZEN_CHECKPOINT,
            "checkpoint_provenance": "server_attestation_with_checkpoint_content_sha256",
            "runtime_failure_policy": "abort_formal_run",
            "seed": FROZEN_SEED,
            "bootstrap_resamples": FROZEN_BOOTSTRAP_RESAMPLES,
        },
        "frozen protocol",
    )
    _require_exact(
        environment,
        {
            "armbench_git_commit": FROZEN_ARMBENCH_RUN_COMMIT,
            "armbench_git_status": "",
            "armbench_git_diff_sha256": EMPTY_GIT_DIFF_SHA256,
            "openpi_git_commit": FROZEN_OPENPI_COMMIT,
        },
        "frozen environment",
    )
    arguments = environment.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("frozen environment has no arguments object")
    _require_exact(
        arguments,
        {
            "video_mode": "all",
            "command": "run",
            "allow_commit_mismatch": False,
            "allow_unattested_server": False,
            "checkpoint": FROZEN_CHECKPOINT,
            "expected_openpi_commit": FROZEN_OPENPI_COMMIT,
            "max_task_steps": None,
            "continue_after_runtime_failure": False,
            "resize_size": 224,
            "num_steps_wait": 10,
            "max_requeries": 2,
            "inference_timeout_s": 600.0,
            "position_threshold_m": 0.01,
            "orientation_threshold_rad": 0.1,
            "gripper_threshold": 0.05,
            "fixed_refresh_interval": None,
            "task_suite": suite,
            "modes": "async_unguarded,latency_aligned",
            "task_ids": "all",
            "episode_indices": "0:5",
            "replan_steps": "5",
            "latency_steps": "4",
            "seed": FROZEN_SEED,
            "bootstrap_resamples": FROZEN_BOOTSTRAP_RESAMPLES,
            "max_consecutive_infrastructure_failures": 3,
            "server_startup_timeout_s": 1200.0,
        },
        "frozen arguments",
    )
    episode_budget = protocol.get("episode_budget")
    if not isinstance(episode_budget, Mapping):
        raise ValueError("frozen protocol has no episode_budget object")
    _require_exact(
        episode_budget,
        {
            "official_suite_task_steps": _OFFICIAL_SUITE_TASK_STEPS,
            "stabilization_steps": 10,
            "task_steps_override": None,
        },
        "frozen episode budget",
    )
    mechanism = protocol.get("experimental_mechanism")
    if not isinstance(mechanism, Mapping):
        raise ValueError("frozen protocol has no experimental_mechanism object")
    _require_exact(
        mechanism,
        {
            "fixed_refresh_interval": None,
            "step_budget": _STEP_BUDGET_SEMANTICS,
        },
        "frozen budget semantics",
    )
    thresholds = protocol.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("frozen protocol has no thresholds object")
    _require_exact(
        thresholds,
        {
            "position_m": 0.01,
            "orientation_rad": 0.1,
            "gripper_linf": 0.05,
            "max_requeries": 2,
        },
        "frozen thresholds",
    )
    timeouts = protocol.get("timeouts")
    if not isinstance(timeouts, Mapping):
        raise ValueError("frozen protocol has no timeouts object")
    _require_exact(
        timeouts,
        {"server_startup_s": 1200.0, "policy_inference_s": 600.0},
        "frozen timeouts",
    )
    official = protocol.get("official_protocol")
    if not isinstance(official, Mapping):
        raise ValueError("frozen protocol has no official_protocol object")
    _require_exact(
        official,
        {
            "action_dimension": 7,
            "camera_rotation_degrees": 180,
            "control_frequency_hz": 20,
            "control_period_ms": 50.0,
            "environment_render_resolution": [256, 256],
            "resize": [224, 224],
            "state_dimension": 8,
            "task_success_source": "LIBERO environment done",
            "video_playback_fps": 10,
        },
        "frozen official protocol",
    )
    metadata = environment.get("server_metadata")
    attestation = (
        metadata.get("armbench_server_attestation")
        if isinstance(metadata, Mapping) else None
    )
    if not isinstance(attestation, Mapping):
        raise ValueError("frozen environment has no server attestation")
    _require_exact(
        attestation,
        {
            "policy_config": FROZEN_POLICY_CONFIG,
            "checkpoint_uri": FROZEN_CHECKPOINT,
            "openpi_commit": FROZEN_OPENPI_COMMIT,
            "checkpoint_content_sha256": FROZEN_CHECKPOINT_CONTENT_SHA256,
            "action_horizon": 10,
        },
        "frozen checkpoint attestation",
    )


def _validate_matrix(
    protocol: Mapping[str, Any], progress: Mapping[str, Any], suite: str
) -> None:
    matrix = protocol.get("matrix")
    if not isinstance(matrix, Mapping):
        raise ValueError("resolved protocol has no matrix")
    expected = {
        "task_suites": [suite],
        "task_ids": list(FROZEN_TASK_IDS),
        "episode_indices": list(FROZEN_EPISODE_INDICES),
        "modes": list(FROZEN_MODES),
        "replan_steps": [FROZEN_REPLAN_STEPS],
        "latency_steps": [FROZEN_LATENCY_STEPS],
        "matched_condition_groups": 50,
        "rollouts": 100,
    }
    _require_exact(matrix, expected, "frozen matrix")
    _require_exact(
        progress,
        {"planned_rollouts": 100, "completed_rollouts": 100, "complete": True},
        "frozen progress",
    )


def validate_frozen_source(path: pathlib.Path, suite: str, run_id: str) -> SourceEvidence:
    raw = pathlib.Path(path)
    if raw.is_symlink():
        raise ValueError("run artifact must not be a symbolic link")
    root = raw.resolve()
    if not root.is_dir():
        raise ValueError("run artifact does not exist: %s" % root)
    if root.name != run_id:
        raise ValueError("run artifact order/name mismatch: expected %s" % run_id)
    validation = _strict_full_validation(root, "pre-snapshot")
    files_checked = int(validation["files_checked"])
    evaluation = root / "evaluation"
    root_manifest_path = root / "manifest.json"
    evaluation_manifest_path = evaluation / "manifest.json"
    if root_manifest_path.is_symlink() or evaluation_manifest_path.is_symlink():
        raise ValueError("run manifests must be regular files")
    try:
        root_manifest_data = root_manifest_path.read_bytes()
        evaluation_manifest_data = evaluation_manifest_path.read_bytes()
    except OSError as exc:
        raise ValueError("run manifests cannot be snapshotted: %s" % exc) from exc
    root_manifest = _json_object_bytes(root_manifest_data, "root manifest")
    evaluation_manifest = _json_object_bytes(
        evaluation_manifest_data, "evaluation manifest"
    )
    if root_manifest.get("schema_version") != ROOT_SCHEMA_VERSION or root_manifest.get("complete") is not True:
        raise ValueError("root manifest is not a complete recognized run")
    if evaluation_manifest.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError("evaluation manifest schema is not recognized")
    _check_manifest_record(
        root_manifest,
        "evaluation/manifest.json",
        evaluation_manifest_data,
        "root manifest",
    )
    csv_data = _read_manifest_bound(
        evaluation / "per_episode.csv", "per_episode.csv", evaluation_manifest, root_manifest
    )
    protocol_data = _read_manifest_bound(
        evaluation / "resolved_protocol.json", "resolved_protocol.json",
        evaluation_manifest, root_manifest,
    )
    environment_data = _read_manifest_bound(
        evaluation / "environment.json", "environment.json",
        evaluation_manifest, root_manifest,
    )
    progress_data = _read_manifest_bound(
        evaluation / "progress.json", "progress.json", evaluation_manifest, root_manifest
    )
    try:
        protocol = json.loads(protocol_data.decode("utf-8"))
        environment = json.loads(environment_data.decode("utf-8"))
        progress = json.loads(progress_data.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError("manifest-bound protocol metadata is invalid: %s" % exc) from exc
    if not all(isinstance(item, dict) for item in (protocol, environment, progress)):
        raise ValueError("manifest-bound protocol metadata must contain objects")
    _validate_identity(protocol, environment, suite)
    _validate_matrix(protocol, progress, suite)
    # Recheck immutable snapshots after parsing, closing the validation/read race.
    _check_manifest_record(evaluation_manifest, "per_episode.csv", csv_data, "evaluation manifest")
    _check_manifest_record(root_manifest, SOURCE_CSV, csv_data, "root manifest")
    videos = _video_paths(csv_data, suite)
    for relative in videos:
        relative_text = relative.as_posix()
        video = evaluation.joinpath(*relative.parts)
        video_size, video_sha256 = _snapshot_regular_file(video, relative_text)
        _check_manifest_file_record(
            evaluation_manifest,
            relative_text,
            video_size,
            video_sha256,
            "evaluation manifest",
        )
        _check_manifest_file_record(
            root_manifest,
            "evaluation/" + relative_text,
            video_size,
            video_sha256,
            "root manifest",
        )
    post_validation = _strict_full_validation(root, "post-snapshot")
    if post_validation.get("files_checked") != files_checked:
        raise ValueError("strict full validation file count changed during snapshot")
    try:
        root_manifest_after = root_manifest_path.read_bytes()
        evaluation_manifest_after = evaluation_manifest_path.read_bytes()
    except OSError as exc:
        raise ValueError("run manifests cannot be rechecked: %s" % exc) from exc
    if root_manifest_after != root_manifest_data:
        raise ValueError("root manifest changed during source snapshot")
    if evaluation_manifest_after != evaluation_manifest_data:
        raise ValueError("evaluation manifest changed during source snapshot")
    return SourceEvidence(
        suite=suite, run_id=run_id, root=root, csv_data=csv_data,
        csv_sha256=_sha256_bytes(csv_data), csv_bytes=len(csv_data),
        root_manifest_sha256=_sha256_bytes(root_manifest_data),
        evaluation_manifest_sha256=_sha256_bytes(evaluation_manifest_data),
        protocol_sha256=_sha256_bytes(protocol_data),
        environment_sha256=_sha256_bytes(environment_data),
        independently_validated_files=files_checked,
        videos_verified=len(videos),
        complete=True,
    )


def _parse_episodes(source: SourceEvidence) -> List[Episode]:
    try:
        text = source.csv_data.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("%s per_episode.csv is not UTF-8" % source.suite) from exc
    rows: List[Episode] = []
    with io.StringIO(text, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if len(fields) != len(set(fields)):
            raise ValueError("%s CSV has duplicate headers" % source.suite)
        missing = sorted(_REQUIRED_COLUMNS - set(fields))
        if missing:
            raise ValueError("%s CSV is missing columns: %s" % (source.suite, ", ".join(missing)))
        for number, raw in enumerate(reader, 2):
            label = "%s row %d" % (source.suite, number)
            if raw.get("schema_version") != SOURCE_SCHEMA_VERSION:
                raise ValueError("%s schema mismatch" % label)
            mode = str(raw.get("mode", ""))
            if mode not in FROZEN_MODES:
                raise ValueError("%s unexpected mode" % label)
            if raw.get("fixed_refresh_interval", "") != "":
                raise ValueError("%s mixes fixed refresh" % label)
            task_suite = str(raw.get("task_suite", ""))
            task_id = _strict_int(raw.get("task_id"), label + ".task_id")
            episode_index = _strict_int(raw.get("episode_index"), label + ".episode_index")
            replan = _strict_int(raw.get("replan_steps"), label + ".replan_steps")
            latency = _strict_int(raw.get("latency_steps"), label + ".latency_steps")
            seed = _strict_int(raw.get("seed"), label + ".seed")
            if (task_suite != source.suite or task_id not in FROZEN_TASK_IDS
                    or episode_index not in FROZEN_EPISODE_INDICES
                    or replan != FROZEN_REPLAN_STEPS
                    or latency != FROZEN_LATENCY_STEPS or seed != FROZEN_SEED):
                raise ValueError("%s is outside the frozen matrix" % label)
            try:
                latency_ms = float(str(raw.get("injected_latency_ms", "")))
            except ValueError as exc:
                raise ValueError("%s injected latency is invalid" % label) from exc
            if not np.isfinite(latency_ms) or latency_ms != 200.0:
                raise ValueError("%s injected latency must be 200 ms" % label)
            initial_hash = str(raw.get("initial_state_sha256", ""))
            if not _SHA256.fullmatch(initial_hash):
                raise ValueError("%s initial state hash is invalid" % label)
            success = _strict_bool(raw.get("success"), label + ".success")
            failure_category = str(raw.get("failure_category", ""))
            failure_type = str(raw.get("failure_type", ""))
            if success and (failure_category or failure_type):
                raise ValueError("%s success conflicts with runtime failure" % label)
            if not _strict_bool(raw.get("video_required"), label + ".video_required"):
                raise ValueError("%s violates video_mode=all" % label)
            rows.append(Episode(
                episode_id=str(raw.get("episode_id", "")), pair_id=str(raw.get("pair_id", "")),
                condition_order=_strict_int(raw.get("condition_order"), label + ".condition_order"),
                task_description=str(raw.get("task_description", "")), mode=mode,
                key=PairKey(task_suite, task_id, episode_index, replan, latency, seed, initial_hash),
                success=success,
                policy_queries=_strict_int(raw.get("policy_queries"), label + ".policy_queries"),
                failure_category=failure_category, failure_type=failure_type,
                failure_message=str(raw.get("failure_message", "")), video_required=True,
            ))
    if len(rows) != 100:
        raise ValueError("%s CSV contains %d rows, expected 100" % (source.suite, len(rows)))
    return rows


def _pair_episodes(source: SourceEvidence) -> List[Tuple[PairKey, Episode, Episode]]:
    episodes = _parse_episodes(source)
    expected = {(task, episode) for task in FROZEN_TASK_IDS for episode in FROZEN_EPISODE_INDICES}
    grouped: Dict[Tuple[int, int], Dict[str, Episode]] = {}
    ids: set[str] = set()
    orders: set[int] = set()
    for episode in episodes:
        if episode.episode_id in ids:
            raise ValueError("duplicate episode_id: %s" % episode.episode_id)
        if episode.condition_order in orders:
            raise ValueError("duplicate condition_order: %d" % episode.condition_order)
        ids.add(episode.episode_id)
        orders.add(episode.condition_order)
        key = (episode.key.task_id, episode.key.episode_index)
        if episode.mode in grouped.setdefault(key, {}):
            raise ValueError("duplicate mode for %r" % (key,))
        grouped[key][episode.mode] = episode
    if set(grouped) != expected:
        raise ValueError("%s task/episode matrix is incomplete" % source.suite)
    if sorted(orders) != list(range(100)):
        raise ValueError("%s condition_order must be contiguous" % source.suite)
    pairs = []
    for key in sorted(expected):
        modes = grouped[key]
        if set(modes) != set(FROZEN_MODES):
            raise ValueError("%s pair %r mode mismatch" % (source.suite, key))
        baseline, aligned = modes[ASYNC_UNGUARDED], modes[LATENCY_ALIGNED]
        if baseline.key.initial_state_sha256 != aligned.key.initial_state_sha256:
            raise ValueError("initial_state_sha256 mismatch for %r" % (key,))
        if baseline.pair_id != aligned.pair_id or baseline.task_description != aligned.task_description:
            raise ValueError("pair metadata mismatch for %r" % (key,))
        if abs(baseline.condition_order - aligned.condition_order) != 1:
            raise ValueError("pair modes are not adjacent for %r" % (key,))
        pairs.append((baseline.key, baseline, aligned))
    ordered = sorted(pairs, key=lambda item: min(item[1].condition_order, item[2].condition_order))
    first_order: Optional[Tuple[str, str]] = None
    for index, (_, baseline, aligned) in enumerate(ordered):
        block = sorted((baseline, aligned), key=lambda row: row.condition_order)
        if [row.condition_order for row in block] != [2 * index, 2 * index + 1]:
            raise ValueError("%s pair ordering is not adjacent" % source.suite)
        mode_order = tuple(row.mode for row in block)
        first_order = mode_order if first_order is None else first_order
        expected_order = first_order if index % 2 == 0 else tuple(reversed(first_order))
        if mode_order != expected_order:
            raise ValueError("%s does not follow alternating mode order" % source.suite)
    return pairs


def _failure_breakdown(episodes: Iterable[Episode], field: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for episode in episodes:
        if episode.runtime_failure:
            value = str(getattr(episode, field)) or "unspecified"
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _summary_row(
    pairs: Sequence[Tuple[PairKey, Episode, Episode]], suite: str, inferential: bool
) -> Dict[str, Any]:
    baseline = [item[1] for item in pairs]
    aligned = [item[2] for item in pairs]
    differences = [int(right.success) - int(left.success) for _, left, right in pairs]
    baseline_successes = sum(row.success for row in baseline)
    aligned_successes = sum(row.success for row in aligned)
    baseline_ci = _wilson_interval(baseline_successes, len(baseline))
    aligned_ci = _wilson_interval(aligned_successes, len(aligned))
    bootstrap = _paired_bootstrap_interval(differences, FROZEN_BOOTSTRAP_RESAMPLES)
    aligned_wins = sum(value > 0 for value in differences)
    baseline_wins = sum(value < 0 for value in differences)
    row: Dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_scope": "suite_confirmatory" if inferential else "pooled_descriptive_only_no_p_value",
        "analysis_role": "confirmatory_suite" if inferential else "descriptive_only",
        "task_suite": suite,
        "paired_n": len(pairs),
        "async_unguarded_successes": baseline_successes,
        "async_unguarded_success_rate": baseline_successes / len(baseline),
        "async_unguarded_wilson95_low": baseline_ci[0],
        "async_unguarded_wilson95_high": baseline_ci[1],
        "latency_aligned_successes": aligned_successes,
        "latency_aligned_success_rate": aligned_successes / len(aligned),
        "latency_aligned_wilson95_low": aligned_ci[0],
        "latency_aligned_wilson95_high": aligned_ci[1],
        "aligned_minus_async_success_rate_difference": float(np.mean(differences)),
        "paired_bootstrap95_low": bootstrap[0],
        "paired_bootstrap95_high": bootstrap[1],
        "aligned_wins": aligned_wins,
        "async_unguarded_wins": baseline_wins,
        "ties": sum(value == 0 for value in differences),
        "async_unguarded_mean_policy_queries": float(np.mean([row.policy_queries for row in baseline])),
        "latency_aligned_mean_policy_queries": float(np.mean([row.policy_queries for row in aligned])),
        "aligned_minus_async_mean_policy_queries": float(np.mean([
            right.policy_queries - left.policy_queries for _, left, right in pairs
        ])),
        "async_unguarded_runtime_failures": sum(row.runtime_failure for row in baseline),
        "latency_aligned_runtime_failures": sum(row.runtime_failure for row in aligned),
        "async_unguarded_failure_categories": _failure_breakdown(baseline, "failure_category"),
        "latency_aligned_failure_categories": _failure_breakdown(aligned, "failure_category"),
        "async_unguarded_failure_types": _failure_breakdown(baseline, "failure_type"),
        "latency_aligned_failure_types": _failure_breakdown(aligned, "failure_type"),
    }
    if inferential:
        row["mcnemar_exact_p"] = _mcnemar_exact_p(aligned_wins, baseline_wins)
        row["mcnemar_holm_p"] = None
        row["holm_reject_alpha_0_05"] = None
    return row


def build_analysis(sources: Sequence[SourceEvidence]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if tuple(source.suite for source in sources) != FROZEN_SUITES:
        raise ValueError("sources must be supplied in frozen suite order")
    suite_pairs = [_pair_episodes(source) for source in sources]
    suite_rows = [_summary_row(pairs, source.suite, True) for source, pairs in zip(sources, suite_pairs)]
    _holm_adjust(suite_rows)
    for row in suite_rows:
        row["holm_reject_alpha_0_05"] = row["mcnemar_holm_p"] <= 0.05
    task_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    runtime_failures: List[Dict[str, Any]] = []
    for source, pairs in zip(sources, suite_pairs):
        for task_id in FROZEN_TASK_IDS:
            selected = [pair for pair in pairs if pair[0].task_id == task_id]
            row = _summary_row(selected, source.suite, False)
            row["analysis_scope"] = "task_descriptive_only_no_inference"
            row["task_id"] = task_id
            task_rows.append(row)
        for key, baseline, aligned in pairs:
            pair_rows.append({
                **key.to_dict(), "pair_id": baseline.pair_id,
                "async_unguarded_success": baseline.success,
                "latency_aligned_success": aligned.success,
                "aligned_minus_async_success": int(aligned.success) - int(baseline.success),
                "async_unguarded_policy_queries": baseline.policy_queries,
                "latency_aligned_policy_queries": aligned.policy_queries,
                "async_unguarded_runtime_failure": baseline.runtime_failure,
                "latency_aligned_runtime_failure": aligned.runtime_failure,
            })
            for episode in (baseline, aligned):
                if episode.runtime_failure:
                    runtime_failures.append({
                        **episode.key.to_dict(), "episode_id": episode.episode_id,
                        "mode": episode.mode, "success": episode.success,
                        "failure_category": episode.failure_category,
                        "failure_type": episode.failure_type,
                        "failure_message": episode.failure_message,
                    })
    pooled = _summary_row([pair for pairs in suite_pairs for pair in pairs], "all_external_suites", False)
    suite_acceptance = []
    for source, row in zip(sources, suite_rows):
        suite_runtime_failures = (
            row["async_unguarded_runtime_failures"]
            + row["latency_aligned_runtime_failures"]
        )
        complete = source.complete
        zero_runtime_failures = suite_runtime_failures == 0
        videos_complete = source.videos_verified == 100
        suite_acceptance.append(
            {
                "task_suite": source.suite,
                "complete": complete,
                "runtime_failures": suite_runtime_failures,
                "zero_runtime_failures": zero_runtime_failures,
                "videos_expected": 100,
                "videos_verified": source.videos_verified,
                "passed": complete and zero_runtime_failures and videos_complete,
            }
        )
    acceptance_passed = len(suite_acceptance) == 3 and all(
        row["passed"] for row in suite_acceptance
    )
    analysis = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_type": "frozen_cross_suite_external_validation_itt",
        "sources": [
            {
                "task_suite": source.suite, "run_id": source.run_id,
                "per_episode_csv": SOURCE_CSV,
                "per_episode_csv_sha256": source.csv_sha256,
                "per_episode_csv_bytes": source.csv_bytes,
                "root_manifest_sha256": source.root_manifest_sha256,
                "evaluation_manifest_sha256": source.evaluation_manifest_sha256,
                "resolved_protocol_sha256": source.protocol_sha256,
                "environment_sha256": source.environment_sha256,
                "independently_validated_files": source.independently_validated_files,
                "videos_verified": source.videos_verified,
            } for source in sources
        ],
        "frozen_identity": {
            "armbench_run_commit": FROZEN_ARMBENCH_RUN_COMMIT,
            "temporal_alignment_implementation_commit": FROZEN_ALIGNMENT_IMPLEMENTATION_COMMIT,
            "openpi_commit": FROZEN_OPENPI_COMMIT, "policy_config": FROZEN_POLICY_CONFIG,
            "checkpoint": FROZEN_CHECKPOINT,
            "checkpoint_content_sha256": FROZEN_CHECKPOINT_CONTENT_SHA256,
            "action_horizon": 10, "video_mode": "all",
        },
        "protocol": {
            "suite_order": list(FROZEN_SUITES), "task_ids": list(FROZEN_TASK_IDS),
            "episode_indices": list(FROZEN_EPISODE_INDICES), "modes": list(FROZEN_MODES),
            "replan_steps": FROZEN_REPLAN_STEPS, "latency_steps": FROZEN_LATENCY_STEPS,
            "injected_latency_ms": 200.0, "seed": FROZEN_SEED,
            "rollouts_per_suite": 100, "pairs_per_suite": 50,
        },
        "bootstrap": {
            "method": "paired percentile bootstrap of mean (latency_aligned - async_unguarded) success",
            "seed": 20260804, "resamples": FROZEN_BOOTSTRAP_RESAMPLES,
            "confidence_level": 0.95, "inferential_role": "descriptive_only",
        },
        "multiplicity": {
            "method": "Holm step-down correction", "hypotheses": 3,
            "family": "three frozen suite-level exact two-sided McNemar tests",
            "confirmatory_decision_statistic": "mcnemar_holm_p",
        },
        "itt": {"rollouts": 300, "pairs": 150, "runtime_failures_retained": len(runtime_failures)},
        "acceptance": {
            "passed": acceptance_passed,
            "criteria": (
                "all three suites complete, exactly 100 manifest-verified videos "
                "per suite, and zero runtime/infrastructure failures"
            ),
            "complete_suites": sum(row["complete"] for row in suite_acceptance),
            "videos_expected": 300,
            "videos_verified": sum(row["videos_verified"] for row in suite_acceptance),
            "runtime_failures": len(runtime_failures),
            "zero_runtime_failures": len(runtime_failures) == 0,
            "suites": suite_acceptance,
        },
        "suite_results": suite_rows,
        "pooled_descriptive": pooled,
        "runtime_failures": runtime_failures,
        "claim_boundary": (
            "External validation of training-free temporal alignment under deterministic 200 ms "
            "LIBERO delay; pooled and task rows are descriptive, with no pooled p value."
        ),
    }
    return analysis, task_rows, pair_rows


def _csv_value(value: Any) -> Any:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _write_csv(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows({field: _csv_value(row[field]) for field in fields} for row in rows)


def _summary(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# pi0.5 cross-suite external validation", "",
        "Manifest-bound, read-only intention-to-treat analysis at frozen 200 ms delay.", "",
        "- Rollouts/pairs: `%d/%d`" % (analysis["itt"]["rollouts"], analysis["itt"]["pairs"]),
        "- Runtime failures retained in ITT: `%d`" % analysis["itt"]["runtime_failures_retained"],
        "- Formal acceptance: **%s**" % ("PASSED" if analysis["acceptance"]["passed"] else "FAILED"),
        "- Acceptance requires all three suites complete, 100/100 verified videos per suite, and zero runtime failures",
        "- Confirmatory family: three suite-level exact two-sided McNemar tests with Holm correction",
        "- Bootstrap intervals, task rows, and the pooled 150-pair row are descriptive only",
        "- No pooled p value is computed or reported", "",
        "| Suite | Async success (Wilson 95%) | Aligned success (Wilson 95%) | Difference (bootstrap 95%) | Wins/losses/ties | McNemar raw/Holm |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in analysis["suite_results"]:
        lines.append(
            "| %s | %d/50 (%.3f [%.3f, %.3f]) | %d/50 (%.3f [%.3f, %.3f]) | %.3f [%.3f, %.3f] | %d/%d/%d | %.4g/%.4g |"
            % (row["task_suite"], row["async_unguarded_successes"], row["async_unguarded_success_rate"],
               row["async_unguarded_wilson95_low"], row["async_unguarded_wilson95_high"],
               row["latency_aligned_successes"], row["latency_aligned_success_rate"],
               row["latency_aligned_wilson95_low"], row["latency_aligned_wilson95_high"],
               row["aligned_minus_async_success_rate_difference"], row["paired_bootstrap95_low"],
               row["paired_bootstrap95_high"], row["aligned_wins"], row["async_unguarded_wins"],
               row["ties"], row["mcnemar_exact_p"], row["mcnemar_holm_p"])
        )
    pooled = analysis["pooled_descriptive"]
    lines.extend(("", "Pooled descriptive difference: `%.3f` (paired bootstrap 95%% `[%.3f, %.3f]`); no pooled significance test." % (
        pooled["aligned_minus_async_success_rate_difference"], pooled["paired_bootstrap95_low"], pooled["paired_bootstrap95_high"]),
        "", str(analysis["claim_boundary"]), ""))
    return "\n".join(lines)


def validate_analysis_manifest(root: pathlib.Path) -> Dict[str, Any]:
    expected_files = {
        "analysis.json",
        "suite_results.csv",
        "task_descriptives.csv",
        "pooled_descriptive.csv",
        "per_pair.csv",
        "summary.md",
    }
    errors: List[str] = []
    try:
        raw_output = pathlib.Path(root)
        if raw_output.is_symlink():
            errors.append("analysis directory must not be a symbolic link")
        output = raw_output.resolve()
        if not output.is_dir():
            errors.append("analysis directory does not exist")
            return {"valid": False, "errors": errors, "files_checked": 0}
        manifest_path = output / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            errors.append("analysis manifest is missing or not a regular file")
            return {"valid": False, "errors": errors, "files_checked": 0}
        manifest = _json_object_bytes(manifest_path.read_bytes(), "analysis manifest")
    except (OSError, ValueError) as exc:
        errors.append("analysis manifest is malformed: %s" % exc)
        return {"valid": False, "errors": errors, "files_checked": 0}
    if manifest.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        errors.append("analysis manifest schema mismatch")
    records = manifest.get("files")
    if not isinstance(records, Mapping):
        errors.append("analysis manifest files must be an object")
        records = {}
    actual = {}
    try:
        for path in sorted(output.iterdir()):
            if path.is_symlink():
                errors.append("symbolic links are not accepted: %s" % path.name)
            elif path.is_file() and path.name != "manifest.json":
                actual[path.name] = path
            elif path.name != "manifest.json":
                errors.append("unexpected non-file output: %s" % path.name)
    except OSError as exc:
        errors.append("cannot enumerate analysis output: %s" % exc)
    if set(actual) != expected_files:
        errors.append("analysis output set mismatch")
    if set(records) != expected_files:
        errors.append("analysis manifest must protect the fixed six outputs")
    for name in sorted(set(actual) & set(records)):
        record = records[name]
        if not isinstance(record, Mapping):
            errors.append("invalid analysis manifest record: %s" % name)
            continue
        try:
            size = actual[name].stat().st_size
            digest = _sha256_file(actual[name])
        except OSError as exc:
            errors.append("cannot validate analysis output %s: %s" % (name, exc))
            continue
        if record.get("bytes") != size or record.get("sha256") != digest:
            errors.append("analysis manifest digest mismatch: %s" % name)
    analysis: Any = None
    analysis_path = actual.get("analysis.json")
    if analysis_path is not None:
        try:
            analysis = _json_object_bytes(analysis_path.read_bytes(), "analysis.json")
        except (OSError, ValueError) as exc:
            errors.append("analysis.json is malformed: %s" % exc)
    manifest_sources = manifest.get("sources")
    analysis_sources = analysis.get("sources") if isinstance(analysis, Mapping) else None
    if not isinstance(manifest_sources, list) or len(manifest_sources) != 3:
        errors.append("analysis manifest must bind exactly three sources")
    elif not isinstance(analysis_sources, list) or len(analysis_sources) != 3:
        errors.append("analysis.json must describe exactly three sources")
    else:
        expected_sources = []
        for index, source in enumerate(analysis_sources):
            if not isinstance(source, Mapping):
                errors.append("analysis.json source %d is not an object" % index)
                continue
            if source.get("task_suite") != FROZEN_SUITES[index]:
                errors.append("analysis.json source %d suite order mismatch" % index)
            if source.get("run_id") != FROZEN_RUN_IDS[index]:
                errors.append("analysis.json source %d run_id mismatch" % index)
            for field in (
                "per_episode_csv_sha256",
                "root_manifest_sha256",
                "evaluation_manifest_sha256",
                "resolved_protocol_sha256",
                "environment_sha256",
            ):
                digest = source.get(field)
                if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                    errors.append("analysis.json source %d %s is invalid" % (index, field))
            expected_sources.append(
                {
                    "task_suite": source.get("task_suite"),
                    "run_id": source.get("run_id"),
                    "per_episode_csv_sha256": source.get("per_episode_csv_sha256"),
                    "root_manifest_sha256": source.get("root_manifest_sha256"),
                    "evaluation_manifest_sha256": source.get(
                        "evaluation_manifest_sha256"
                    ),
                    "resolved_protocol_sha256": source.get(
                        "resolved_protocol_sha256"
                    ),
                    "environment_sha256": source.get("environment_sha256"),
                }
            )
        if len(expected_sources) == 3 and manifest_sources != expected_sources:
            errors.append("analysis manifest source hashes do not match analysis.json")
    return {"valid": not errors, "errors": errors, "files_checked": len(actual)}


def write_analysis(
    sources: Sequence[SourceEvidence], analysis: Mapping[str, Any],
    task_rows: Sequence[Mapping[str, Any]], pair_rows: Sequence[Mapping[str, Any]],
    output_directory: pathlib.Path,
) -> pathlib.Path:
    output = pathlib.Path(output_directory).resolve()
    for source in sources:
        if output == source.root or source.root in output.parents:
            raise ValueError("output directory must be outside every frozen input artifact")
    if output.exists():
        raise FileExistsError("output directory already exists: %s" % output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(tempfile.mkdtemp(prefix=".%s.tmp-" % output.name, dir=str(output.parent)))
    try:
        _write_json(staging / "analysis.json", analysis)
        _write_csv(staging / "suite_results.csv", analysis["suite_results"])
        _write_csv(staging / "task_descriptives.csv", task_rows)
        _write_csv(staging / "pooled_descriptive.csv", [analysis["pooled_descriptive"]])
        _write_csv(staging / "per_pair.csv", pair_rows)
        (staging / "summary.md").write_text(_summary(analysis), encoding="utf-8", newline="\n")
        files = {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in sorted(staging.iterdir()) if path.is_file()
        }
        _write_json(staging / "manifest.json", {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "sources": [
                {
                    "task_suite": source.suite,
                    "run_id": source.run_id,
                    "per_episode_csv_sha256": source.csv_sha256,
                    "root_manifest_sha256": source.root_manifest_sha256,
                    "evaluation_manifest_sha256": source.evaluation_manifest_sha256,
                    "resolved_protocol_sha256": source.protocol_sha256,
                    "environment_sha256": source.environment_sha256,
                }
                for source in sources
            ],
            "files": files,
        })
        check = validate_analysis_manifest(staging)
        if check["valid"] is not True:
            raise ValueError("self-validation failed: %s" % "; ".join(check["errors"]))
        if output.exists():
            raise FileExistsError("output directory already exists: %s" % output)
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def execute_analysis(run_artifacts: Sequence[pathlib.Path], output_directory: pathlib.Path) -> pathlib.Path:
    if len(run_artifacts) != 3:
        raise ValueError("exactly three run artifacts are required in frozen order")
    sources = [
        validate_frozen_source(path, suite, run_id)
        for path, suite, run_id in zip(run_artifacts, FROZEN_SUITES, FROZEN_RUN_IDS)
    ]
    analysis, task_rows, pair_rows = build_analysis(sources)
    return write_analysis(sources, analysis, task_rows, pair_rows, output_directory)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("run_artifacts", nargs=3, type=pathlib.Path, help="Object, Goal, and LIBERO-10 run directories, in that order")
    parser.add_argument("--output-directory", required=True, type=pathlib.Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        output = execute_analysis(args.run_artifacts, args.output_directory)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"schema_version": ANALYSIS_SCHEMA_VERSION, "valid": True, "output_directory": str(output)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
