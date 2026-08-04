"""Fail-closed cross-validator for finalized pi0.5-LIBERO artifacts.

The evaluator is responsible for producing an artifact.  This module treats that
artifact as untrusted input: it verifies its manifest, parses raw CSV values with
strict types, reconstructs the registered matrix, and recomputes every reported
statistic from the raw episode and query records.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import pathlib
import re
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from integrations.openpi.libero_runtime import (
    ASYNC_UNGUARDED,
    FIXED_REFRESH,
    LATENCY_ALIGNED,
    STATE_GUARD,
    VALID_MODES,
)
from integrations.openpi.libero_runtime_eval import (
    CONTROL_COMPARISON_FIELDS,
    CONTRAST_FIELDS,
    DEFAULT_POLICY_CONFIG,
    EPISODE_FIELDS,
    LIBERO_CONTROL_PERIOD_MS,
    PI05_LIBERO_ACTION_HORIZON,
    QUERY_FIELDS,
    RUNTIME_SOURCE_FILES,
    SCHEMA_VERSION,
    SERVER_ATTESTATION_SCHEMA_VERSION,
    SUITE_TASK_COUNTS,
    ExperimentCell,
    _summary_markdown,
    aggregate_episodes,
    artifact_integrity_errors,
    intervention_control_comparisons,
    matched_condition_contrasts,
    paired_comparisons,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
_INTEGER = re.compile(r"0|[1-9][0-9]*\Z")
_CORE_FILES = frozenset(
    (
        "per_episode.csv",
        "per_query.csv",
        "aggregate.json",
        "aggregate.csv",
        "paired_comparisons.json",
        "paired_comparisons.csv",
        "intervention_control_comparisons.json",
        "intervention_control_comparisons.csv",
        "condition_contrasts.json",
        "condition_contrasts.csv",
        "resolved_protocol.json",
        "environment.json",
        "integrity.json",
        "progress.json",
        "summary.md",
    )
)
_FLOAT_ABS_TOLERANCE = 1e-10
_FLOAT_REL_TOLERANCE = 1e-10


@dataclasses.dataclass(frozen=True)
class ValidationReport:
    artifact: str
    valid: bool
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...]
    checks: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "armbench.libero_artifact_validation.v1",
            "artifact": self.artifact,
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "checks": list(self.checks),
        }


class _Collector:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.checks: List[str] = []

    def error(self, section: str, message: str) -> None:
        if len(self.errors) < 200:
            self.errors.append("[%s] %s" % (section, message))
        elif len(self.errors) == 200:
            self.errors.append("[validator] further errors were suppressed")

    def warning(self, section: str, message: str) -> None:
        self.warnings.append("[%s] %s" % (section, message))

    def checked(self, name: str) -> None:
        self.checks.append(name)


def _reject_json_constant(value: str) -> Any:
    raise ValueError("non-finite JSON number: %s" % value)


def _strict_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def _read_json(path: pathlib.Path, section: str, collector: _Collector) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        collector.error(section, "cannot read strict JSON %s: %s" % (path.name, exc))
        return None


def _safe_relative_path(value: Any) -> Optional[pathlib.PurePosixPath]:
    if not isinstance(value, str) or not value:
        return None
    if "\\" in value or "\x00" in value or ":" in value:
        return None
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or str(path) != value:
        return None
    if any(part in ("", ".", "..") for part in path.parts):
        return None
    return path


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _validate_manifest(
    root: pathlib.Path, collector: _Collector
) -> Optional[Mapping[str, Any]]:
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path, "manifest", collector)
    if not isinstance(manifest, Mapping):
        collector.error("manifest", "manifest.json must contain an object")
        return None
    if manifest.get("schema_version") != SCHEMA_VERSION:
        collector.error(
            "manifest",
            "schema_version is %r, expected %r"
            % (manifest.get("schema_version"), SCHEMA_VERSION),
        )
    records = manifest.get("files")
    if not isinstance(records, Mapping):
        collector.error("manifest", "files must be an object")
        return manifest

    root_resolved = root.resolve()
    safe_records: Dict[str, Mapping[str, Any]] = {}
    normalized: Dict[str, str] = {}
    for relative, record in records.items():
        safe = _safe_relative_path(relative)
        if safe is None:
            collector.error("manifest", "unsafe or non-canonical path: %r" % relative)
            continue
        canonical = safe.as_posix()
        key = canonical.casefold()
        if key in normalized:
            collector.error(
                "manifest",
                "duplicate normalized path: %r and %r" % (normalized[key], relative),
            )
            continue
        normalized[key] = str(relative)
        candidate = root.joinpath(*safe.parts)
        try:
            candidate.resolve().relative_to(root_resolved)
        except (OSError, ValueError):
            collector.error("manifest", "path escapes artifact root: %s" % relative)
            continue
        if candidate.is_symlink():
            collector.error("manifest", "symbolic links are not accepted: %s" % relative)
            continue
        if not isinstance(record, Mapping):
            collector.error("manifest", "record for %s must be an object" % relative)
            continue
        safe_records[canonical] = record

    actual_files: Dict[str, pathlib.Path] = {}
    try:
        for path in root.rglob("*"):
            if not path.is_file() or path.name == "manifest.json" or path.suffix == ".tmp":
                continue
            relative = path.relative_to(root).as_posix()
            actual_files[relative] = path
    except OSError as exc:
        collector.error("manifest", "cannot enumerate artifact files: %s" % exc)

    missing_entries = sorted(set(actual_files) - set(safe_records))
    absent_files = sorted(set(safe_records) - set(actual_files))
    if missing_entries:
        collector.error(
            "manifest", "files omitted from manifest: %s" % ", ".join(missing_entries)
        )
    if absent_files:
        collector.error(
            "manifest", "manifest references absent files: %s" % ", ".join(absent_files)
        )
    missing_core = sorted(_CORE_FILES - set(safe_records))
    if missing_core:
        collector.error(
            "manifest", "required artifact files are missing: %s" % ", ".join(missing_core)
        )

    for relative in sorted(set(safe_records) & set(actual_files)):
        record = safe_records[relative]
        expected_bytes = record.get("bytes")
        expected_hash = record.get("sha256")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
        ):
            collector.error("manifest", "invalid byte count for %s" % relative)
        elif actual_files[relative].stat().st_size != expected_bytes:
            collector.error("manifest", "byte count mismatch for %s" % relative)
        if not isinstance(expected_hash, str) or not _SHA256.fullmatch(expected_hash):
            collector.error("manifest", "invalid SHA-256 for %s" % relative)
        elif _sha256(actual_files[relative]) != expected_hash:
            collector.error("manifest", "SHA-256 mismatch for %s" % relative)
    collector.checked("manifest coverage, sizes, hashes, and path safety")
    return manifest


def _read_csv_rows(
    path: pathlib.Path,
    expected_fields: Sequence[str],
    section: str,
    collector: _Collector,
) -> Optional[List[Dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        collector.error(section, "cannot read %s: %s" % (path.name, exc))
        return None
    if not rows:
        collector.error(section, "%s has no header" % path.name)
        return None
    header = rows[0]
    if header != list(expected_fields):
        collector.error(
            section,
            "%s header mismatch: got %r, expected %r"
            % (path.name, header, list(expected_fields)),
        )
        return None
    parsed: List[Dict[str, str]] = []
    for index, values in enumerate(rows[1:], start=2):
        if not values and not expected_fields:
            continue
        if len(values) != len(header):
            collector.error(
                section,
                "%s row %d has %d fields, expected %d"
                % (path.name, index, len(values), len(header)),
            )
            continue
        parsed.append(dict(zip(header, values)))
    return parsed


def _parse_bool(value: str, field: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError("%s must be exactly True or False" % field)


def _parse_int(value: str, field: str, positive: bool = False) -> int:
    if not isinstance(value, str) or not _INTEGER.fullmatch(value):
        raise ValueError("%s must be a canonical nonnegative integer" % field)
    parsed = int(value)
    if positive and parsed <= 0:
        raise ValueError("%s must be positive" % field)
    return parsed


def _parse_optional_int(value: str, field: str, positive: bool = False) -> Optional[int]:
    if value == "":
        return None
    return _parse_int(value, field, positive=positive)


def _parse_float(
    value: str, field: str, optional: bool = False, nonnegative: bool = True
) -> Optional[float]:
    if optional and value == "":
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("%s must be a finite number" % field)
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError("%s must be a finite number" % field)
    if not math.isfinite(parsed) or (nonnegative and parsed < 0.0):
        raise ValueError("%s must be finite and nonnegative" % field)
    return parsed


def _required_string(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("%s must be a nonempty trimmed string" % field)
    return value


def _optional_string(value: str) -> Optional[str]:
    return None if value == "" else value


_EPISODE_INTS = frozenset(
    (
        "condition_order",
        "task_id",
        "episode_index",
        "replan_steps",
        "latency_steps",
        "seed",
        "environment_steps",
        "task_action_steps",
        "latency_action_steps",
        "policy_queries",
        "accepted_chunks",
        "rejected_chunks",
        "stale_chunks_executed",
        "stale_action_steps",
        "interventions",
    )
)
_EPISODE_OPTIONAL_FLOATS = frozenset(
    (
        "inference_latency_p50_ms",
        "inference_latency_p95_ms",
        "policy_inference_latency_p50_ms",
        "policy_inference_latency_p95_ms",
        "server_inference_latency_p50_ms",
        "server_inference_latency_p95_ms",
    )
)
_EPISODE_OPTIONAL_STRINGS = frozenset(
    (
        "failure_category",
        "failure_type",
        "failure_message",
        "video_path",
        "video_error_type",
        "video_error_message",
    )
)


def _parse_episode(raw: Mapping[str, str]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError("schema_version mismatch")
    row["schema_version"] = raw["schema_version"]
    for field in _EPISODE_INTS:
        row[field] = _parse_int(
            raw[field], field, positive=field == "replan_steps"
        )
    for field in _EPISODE_OPTIONAL_FLOATS:
        row[field] = _parse_float(raw[field], field, optional=True)
    row["injected_latency_ms"] = _parse_float(
        raw["injected_latency_ms"], "injected_latency_ms"
    )
    row["fixed_refresh_interval"] = _parse_optional_int(
        raw["fixed_refresh_interval"], "fixed_refresh_interval", positive=True
    )
    row["wall_time_s"] = _parse_float(raw["wall_time_s"], "wall_time_s")
    row["success"] = _parse_bool(raw["success"], "success")
    row["video_required"] = _parse_bool(raw["video_required"], "video_required")
    for field in _EPISODE_OPTIONAL_STRINGS:
        row[field] = _optional_string(raw[field])
    for field in (
        "episode_id",
        "pair_id",
        "task_suite",
        "task_description",
        "mode",
        "termination_reason",
        "initial_state_sha256",
    ):
        row[field] = _required_string(raw[field], field)
    if row["mode"] not in VALID_MODES:
        raise ValueError("unknown mode: %s" % row["mode"])
    if row["task_suite"] not in SUITE_TASK_COUNTS:
        raise ValueError("unknown task_suite: %s" % row["task_suite"])
    if not _SHA256.fullmatch(row["initial_state_sha256"]):
        raise ValueError("initial_state_sha256 is not a lowercase SHA-256")
    if row["seed"] >= 2 ** 32:
        raise ValueError("seed must be below 2**32")
    if row["success"] and row["failure_type"]:
        raise ValueError("successful episode cannot record failure_type")
    if row["stale_action_steps"] > row["task_action_steps"]:
        raise ValueError("stale_action_steps exceeds task_action_steps")
    if row["interventions"] != row["rejected_chunks"]:
        raise ValueError("interventions must equal rejected_chunks")
    return row


_QUERY_INTS = frozenset(
    (
        "task_id",
        "episode_index",
        "replan_steps",
        "latency_steps",
        "query_index",
        "observation_step",
        "response_step",
        "injected_latency_steps_requested",
        "injected_latency_steps_executed",
        "action_chunk_steps",
    )
)
_QUERY_OPTIONAL_FLOATS = frozenset(
    (
        "policy_inference_latency_ms",
        "server_inference_latency_ms",
        "position_mismatch_m",
        "orientation_mismatch_rad",
        "gripper_mismatch_linf",
    )
)


def _parse_query(raw: Mapping[str, str]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError("schema_version mismatch")
    row["schema_version"] = raw["schema_version"]
    for field in _QUERY_INTS:
        row[field] = _parse_int(
            raw[field], field, positive=field == "replan_steps"
        )
    row["inference_latency_ms"] = _parse_float(
        raw["inference_latency_ms"], "inference_latency_ms"
    )
    for field in _QUERY_OPTIONAL_FLOATS:
        row[field] = _parse_float(raw[field], field, optional=True)
    row["fixed_refresh_interval"] = _parse_optional_int(
        raw["fixed_refresh_interval"], "fixed_refresh_interval", positive=True
    )
    row["accepted"] = _parse_bool(raw["accepted"], "accepted")
    for field in (
        "episode_id",
        "pair_id",
        "task_suite",
        "mode",
        "decision",
    ):
        row[field] = _required_string(raw[field], field)
    row["rejection_reasons"] = raw["rejection_reasons"]
    for field in ("error_stage", "error_type", "error_message"):
        row[field] = _optional_string(raw[field])
    if row["mode"] not in VALID_MODES:
        raise ValueError("unknown mode: %s" % row["mode"])
    if row["task_suite"] not in SUITE_TASK_COUNTS:
        raise ValueError("unknown task_suite: %s" % row["task_suite"])
    if row["response_step"] < row["observation_step"]:
        raise ValueError("response_step precedes observation_step")
    if (
        row["injected_latency_steps_executed"]
        > row["injected_latency_steps_requested"]
    ):
        raise ValueError("executed injected latency exceeds requested latency")
    return row


def _parse_raw_rows(
    root: pathlib.Path, collector: _Collector
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
    episode_raw = _read_csv_rows(
        root / "per_episode.csv", EPISODE_FIELDS, "raw episodes", collector
    )
    query_raw = _read_csv_rows(
        root / "per_query.csv", QUERY_FIELDS, "raw queries", collector
    )
    if episode_raw is None or query_raw is None:
        return None, None
    episodes: List[Dict[str, Any]] = []
    queries: List[Dict[str, Any]] = []
    for index, raw in enumerate(episode_raw, start=2):
        try:
            episodes.append(_parse_episode(raw))
        except (KeyError, ValueError) as exc:
            collector.error("raw episodes", "row %d: %s" % (index, exc))
    for index, raw in enumerate(query_raw, start=2):
        try:
            queries.append(_parse_query(raw))
        except (KeyError, ValueError) as exc:
            collector.error("raw queries", "row %d: %s" % (index, exc))
    if len(episodes) != len(episode_raw) or len(queries) != len(query_raw):
        return None, None
    if not episodes:
        collector.error("raw episodes", "finalized artifact contains no episodes")
        return None, None
    collector.checked("strict raw episode and query parsing")
    return episodes, queries


def _json_int(value: Any, field: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("%s must be a nonnegative integer" % field)
    if positive and value <= 0:
        raise ValueError("%s must be positive" % field)
    return value


def _matrix_values(
    matrix: Mapping[str, Any],
    field: str,
    kind: str,
) -> List[Any]:
    value = matrix.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError("matrix.%s must be a nonempty list" % field)
    if len(value) != len({repr(item) for item in value}):
        raise ValueError("matrix.%s contains duplicates" % field)
    if kind == "string":
        if any(not isinstance(item, str) or not item for item in value):
            raise ValueError("matrix.%s must contain strings" % field)
    else:
        for item in value:
            _json_int(item, "matrix.%s" % field, positive=kind == "positive")
    if value != sorted(value):
        raise ValueError("matrix.%s must be sorted" % field)
    return list(value)


def _resolve_expected_cells(
    protocol: Any, collector: _Collector
) -> Optional[Tuple[List[ExperimentCell], List[str], int]]:
    if not isinstance(protocol, Mapping):
        collector.error("protocol", "resolved_protocol.json must contain an object")
        return None
    if protocol.get("schema_version") != SCHEMA_VERSION:
        collector.error("protocol", "schema_version mismatch")
    matrix = protocol.get("matrix")
    try:
        if not isinstance(matrix, Mapping):
            raise ValueError("matrix must be an object")
        if matrix.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("matrix.schema_version mismatch")
        suites = _matrix_values(matrix, "task_suites", "string")
        task_ids = _matrix_values(matrix, "task_ids", "nonnegative")
        episode_indices = _matrix_values(matrix, "episode_indices", "nonnegative")
        modes = _matrix_values(matrix, "modes", "string")
        horizons = _matrix_values(matrix, "replan_steps", "positive")
        delays = _matrix_values(matrix, "latency_steps", "nonnegative")
        if any(suite not in SUITE_TASK_COUNTS for suite in suites):
            raise ValueError("matrix contains an unknown LIBERO suite")
        if any(mode not in VALID_MODES for mode in modes):
            raise ValueError("matrix contains an unknown runtime mode")
        if any(value > PI05_LIBERO_ACTION_HORIZON for value in horizons):
            raise ValueError("matrix replan_steps exceeds policy action horizon")
        seed = _json_int(protocol.get("seed"), "seed")
        if seed >= 2 ** 32:
            raise ValueError("seed must be below 2**32")
        bootstrap = _json_int(
            protocol.get("bootstrap_resamples"), "bootstrap_resamples", positive=True
        )
    except ValueError as exc:
        collector.error("protocol", str(exc))
        return None

    cells: List[ExperimentCell] = []
    order = 0
    for suite in suites:
        for task_id in task_ids:
            if task_id >= SUITE_TASK_COUNTS[suite]:
                collector.error(
                    "protocol", "task_id %d is outside %s" % (task_id, suite)
                )
            for episode_index in episode_indices:
                for horizon in horizons:
                    for delay in delays:
                        pair_id = "%s/task_%03d/episode_%03d/h_%02d/l_%03d" % (
                            suite,
                            task_id,
                            episode_index,
                            horizon,
                            delay,
                        )
                        for mode in modes:
                            cells.append(
                                ExperimentCell(
                                    task_suite=suite,
                                    task_id=task_id,
                                    episode_index=episode_index,
                                    mode=mode,
                                    replan_steps=horizon,
                                    latency_steps=delay,
                                    pair_id=pair_id,
                                    condition_order=order,
                                )
                            )
                            order += 1
    expected_rollouts = len(cells)
    expected_pairs = expected_rollouts // len(modes)
    try:
        if _json_int(matrix.get("rollouts"), "matrix.rollouts") != expected_rollouts:
            raise ValueError("matrix.rollouts does not match its dimensions")
        if (
            _json_int(matrix.get("paired_conditions"), "matrix.paired_conditions")
            != expected_pairs
        ):
            raise ValueError("matrix.paired_conditions does not match its dimensions")
        if (
            _json_int(
                matrix.get("matched_condition_groups"),
                "matrix.matched_condition_groups",
            )
            != expected_pairs
        ):
            raise ValueError(
                "matrix.matched_condition_groups does not match its dimensions"
            )
    except ValueError as exc:
        collector.error("protocol", str(exc))
    collector.checked("registered matrix reconstruction")
    return cells, modes, bootstrap


def _close(left: Any, right: Any) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=_FLOAT_REL_TOLERANCE,
        abs_tol=_FLOAT_ABS_TOLERANCE,
    )


def _validate_raw_consistency(
    root: pathlib.Path,
    episodes: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    cells: Sequence[ExperimentCell],
    modes: Sequence[str],
    collector: _Collector,
) -> None:
    episode_ids = [str(row["episode_id"]) for row in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        collector.error("raw consistency", "episode_id values are not unique")
    orders = [int(row["condition_order"]) for row in episodes]
    if len(orders) != len(set(orders)):
        collector.error("raw consistency", "condition_order values are not unique")
    if sorted(orders) != list(range(len(episodes))):
        collector.error("raw consistency", "condition_order is not contiguous from zero")

    expected: Dict[str, ExperimentCell] = {cell.episode_id: cell for cell in cells}
    actual: Dict[str, Mapping[str, Any]] = {
        str(row["episode_id"]): row for row in episodes
    }
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing:
        collector.error("matrix", "missing planned episodes: %s" % ", ".join(missing))
    if unexpected:
        collector.error("matrix", "unexpected episodes: %s" % ", ".join(unexpected))
    seed = protocol.get("seed")
    mechanism = protocol.get("experimental_mechanism")
    refresh_interval = (
        mechanism.get("fixed_refresh_interval")
        if isinstance(mechanism, Mapping)
        else None
    )
    if FIXED_REFRESH in modes:
        if (
            isinstance(refresh_interval, bool)
            or not isinstance(refresh_interval, int)
            or refresh_interval <= 0
        ):
            collector.error(
                "protocol",
                "fixed_refresh mode requires a positive registered fixed_refresh_interval",
            )
    elif refresh_interval is not None and (
        isinstance(refresh_interval, bool)
        or not isinstance(refresh_interval, int)
        or refresh_interval <= 0
    ):
        collector.error("protocol", "fixed_refresh_interval must be null or positive")
    for episode_id in sorted(set(expected) & set(actual)):
        cell = expected[episode_id]
        row = actual[episode_id]
        expected_fields = {
            "pair_id": cell.pair_id,
            "task_suite": cell.task_suite,
            "task_id": cell.task_id,
            "episode_index": cell.episode_index,
            "mode": cell.mode,
            "replan_steps": cell.replan_steps,
            "latency_steps": cell.latency_steps,
            "fixed_refresh_interval": (
                refresh_interval if cell.mode == FIXED_REFRESH else None
            ),
            "seed": seed,
        }
        for field, value in expected_fields.items():
            if row[field] != value:
                collector.error(
                    "matrix", "%s %s=%r, expected %r" % (episode_id, field, row[field], value)
                )
        expected_latency = cell.latency_steps * LIBERO_CONTROL_PERIOD_MS
        if not _close(row["injected_latency_ms"], expected_latency):
            collector.error("matrix", "%s injected_latency_ms mismatch" % episode_id)

    by_pair: Dict[str, List[Mapping[str, Any]]] = {}
    for row in episodes:
        by_pair.setdefault(str(row["pair_id"]), []).append(row)
    expected_pair_order: List[str] = []
    for cell in cells:
        if cell.pair_id not in expected_pair_order:
            expected_pair_order.append(cell.pair_id)
    first_mode_order: Optional[Tuple[str, ...]] = None
    for pair_index, pair_id in enumerate(expected_pair_order):
        rows = sorted(by_pair.get(pair_id, []), key=lambda row: int(row["condition_order"]))
        if len(rows) != len(modes):
            collector.error(
                "pairing", "%s has %d modes, expected %d" % (pair_id, len(rows), len(modes))
            )
            continue
        actual_modes = tuple(str(row["mode"]) for row in rows)
        if set(actual_modes) != set(modes):
            collector.error("pairing", "%s mode set does not match protocol" % pair_id)
        pair_orders = sorted(int(row["condition_order"]) for row in rows)
        expected_orders = list(
            range(pair_index * len(modes), (pair_index + 1) * len(modes))
        )
        if pair_orders != expected_orders:
            collector.error("pairing", "%s is not an adjacent condition block" % pair_id)
        if len(modes) > 1:
            if first_mode_order is None:
                first_mode_order = actual_modes
            desired = first_mode_order if pair_index % 2 == 0 else tuple(reversed(first_mode_order))
            if actual_modes != desired:
                collector.error("pairing", "%s does not follow alternating mode order" % pair_id)
        hashes = {str(row["initial_state_sha256"]) for row in rows}
        seeds = {int(row["seed"]) for row in rows}
        descriptions = {str(row["task_description"]) for row in rows}
        if len(hashes) != 1 or len(seeds) != 1 or len(descriptions) != 1:
            collector.error("pairing", "%s paired initial condition mismatch" % pair_id)

    by_episode: Dict[str, List[Mapping[str, Any]]] = {}
    for query in queries:
        episode_id = str(query["episode_id"])
        by_episode.setdefault(episode_id, []).append(query)
        episode = actual.get(episode_id)
        if episode is None:
            collector.error("raw queries", "query references unknown episode %s" % episode_id)
            continue
        for field in (
            "pair_id",
            "task_suite",
            "task_id",
            "episode_index",
            "mode",
            "replan_steps",
            "latency_steps",
            "fixed_refresh_interval",
        ):
            if query[field] != episode[field]:
                collector.error(
                    "raw queries", "%s query %s mismatch" % (episode_id, field)
                )
        if query["injected_latency_steps_requested"] != episode["latency_steps"]:
            collector.error("raw queries", "%s requested latency mismatch" % episode_id)

    for episode in episodes:
        episode_id = str(episode["episode_id"])
        episode_queries = by_episode.get(episode_id, [])
        if len(episode_queries) != int(episode["policy_queries"]):
            collector.error("raw queries", "%s policy_queries count mismatch" % episode_id)
        indices = [int(query["query_index"]) for query in episode_queries]
        if indices != list(range(len(indices))):
            collector.error(
                "raw queries",
                "%s query_index is not contiguous, unique, and in file order"
                % episode_id,
            )
        accepted = sum(bool(query["accepted"]) for query in episode_queries)
        rejected = sum(
            query["decision"]
            in ("rejected_state_mismatch", "rejected_fixed_refresh")
            for query in episode_queries
        )
        stale_chunks = sum(
            bool(query["accepted"])
            and int(query["injected_latency_steps_executed"]) > 0
            for query in episode_queries
        )
        latency_steps = sum(
            int(query["injected_latency_steps_executed"]) for query in episode_queries
        )
        for field, value in (
            ("accepted_chunks", accepted),
            ("rejected_chunks", rejected),
            ("stale_chunks_executed", stale_chunks),
            ("latency_action_steps", latency_steps),
        ):
            if int(episode[field]) != value:
                collector.error("raw queries", "%s %s mismatch" % (episode_id, field))
        latency_specs = (
            ("inference_latency", "inference_latency_ms"),
            ("policy_inference_latency", "policy_inference_latency_ms"),
            ("server_inference_latency", "server_inference_latency_ms"),
        )
        for prefix, query_field in latency_specs:
            values = [
                float(query[query_field])
                for query in episode_queries
                if query[query_field] is not None
            ]
            expected_p50 = float(np.percentile(values, 50)) if values else None
            expected_p95 = float(np.percentile(values, 95)) if values else None
            for suffix, expected_value in (("p50_ms", expected_p50), ("p95_ms", expected_p95)):
                actual_value = episode["%s_%s" % (prefix, suffix)]
                if expected_value is None:
                    if actual_value is not None:
                        collector.error("raw queries", "%s %s percentile should be blank" % (episode_id, prefix))
                elif actual_value is None or not _close(actual_value, expected_value):
                    collector.error("raw queries", "%s %s percentile mismatch" % (episode_id, prefix))

        thresholds = protocol.get("thresholds")
        if isinstance(thresholds, Mapping):
            try:
                position_threshold = float(thresholds["position_m"])
                orientation_threshold = float(thresholds["orientation_rad"])
                gripper_threshold = float(thresholds["gripper_linf"])
                max_requeries = thresholds["max_requeries"]
                if any(
                    not math.isfinite(value) or value < 0.0
                    for value in (
                        position_threshold,
                        orientation_threshold,
                        gripper_threshold,
                    )
                ):
                    raise ValueError("mismatch thresholds must be finite and nonnegative")
                if (
                    isinstance(max_requeries, bool)
                    or not isinstance(max_requeries, int)
                    or max_requeries < 0
                ):
                    raise ValueError("max_requeries must be a nonnegative integer")
            except (KeyError, TypeError, ValueError) as exc:
                collector.error("protocol", "invalid mismatch thresholds: %s" % exc)
            else:
                accepted_chunks = 0
                consecutive_rejections = 0
                episode_mode = str(episode["mode"])
                decisions_checkable = not (
                    episode_mode == FIXED_REFRESH
                    and (
                        isinstance(refresh_interval, bool)
                        or not isinstance(refresh_interval, int)
                        or refresh_interval <= 0
                    )
                )
                for query_position, query in enumerate(episode_queries):
                    if not decisions_checkable:
                        break
                    decision = str(query["decision"])
                    if query["error_stage"] is not None:
                        if query["accepted"] or not decision.endswith("_error"):
                            collector.error(
                                "decision semantics",
                                "%s query %d has inconsistent error decision"
                                % (episode_id, query["query_index"]),
                            )
                        continue
                    if decision in (
                        "success_during_inference_delay",
                        "step_budget_exhausted_during_delay",
                    ):
                        if query["accepted"] or query["rejection_reasons"]:
                            collector.error(
                                "decision semantics",
                                "%s query %d has inconsistent terminal-delay decision"
                                % (episode_id, query["query_index"]),
                            )
                        continue
                    mismatch_values = (
                        query["position_mismatch_m"],
                        query["orientation_mismatch_rad"],
                        query["gripper_mismatch_linf"],
                    )
                    if any(value is None for value in mismatch_values):
                        collector.error(
                            "decision semantics",
                            "%s query %d lacks mismatch measurements"
                            % (episode_id, query["query_index"]),
                        )
                        continue
                    mismatch_reasons = []
                    if float(mismatch_values[0]) > position_threshold:
                        mismatch_reasons.append("position_mismatch")
                    if float(mismatch_values[1]) > orientation_threshold:
                        mismatch_reasons.append("orientation_mismatch")
                    if float(mismatch_values[2]) > gripper_threshold:
                        mismatch_reasons.append("gripper_mismatch")
                    if episode_mode == ASYNC_UNGUARDED:
                        expected_decision = "accepted_unguarded"
                        expected_accepted = True
                        expected_reasons = "|".join(mismatch_reasons)
                    elif episode_mode == LATENCY_ALIGNED:
                        expected_decision = "accepted_latency_aligned"
                        expected_accepted = True
                        expected_reasons = "|".join(mismatch_reasons)
                    elif episode_mode == STATE_GUARD:
                        expected_accepted = not mismatch_reasons
                        expected_decision = (
                            "accepted_state_guard"
                            if expected_accepted
                            else "rejected_state_mismatch"
                        )
                        expected_reasons = "|".join(mismatch_reasons)
                    else:
                        scheduled = (
                            consecutive_rejections == 0
                            and (accepted_chunks + 1) % int(refresh_interval) == 0
                        )
                        expected_accepted = not scheduled
                        expected_decision = (
                            "accepted_fixed_refresh"
                            if expected_accepted
                            else "rejected_fixed_refresh"
                        )
                        expected_reasons = (
                            "|".join(mismatch_reasons)
                            if expected_accepted
                            else "scheduled_refresh"
                        )
                    if (
                        query["accepted"] is not expected_accepted
                        or decision != expected_decision
                        or query["rejection_reasons"] != expected_reasons
                    ):
                        collector.error(
                            "decision semantics",
                            "%s query %d does not follow %s rejection policy"
                            % (episode_id, query["query_index"], episode_mode),
                        )
                    if expected_accepted:
                        accepted_chunks += 1
                        consecutive_rejections = 0
                    else:
                        consecutive_rejections += 1
                        if consecutive_rejections > max_requeries and (
                            query_position != len(episode_queries) - 1
                            or episode["termination_reason"] != "max_requeries_exceeded"
                        ):
                            collector.error(
                                "decision semantics",
                                "%s exceeds the bounded requery policy" % episode_id,
                            )
                if episode_mode == FIXED_REFRESH and max_requeries < 1:
                    collector.error(
                        "protocol", "fixed_refresh requires max_requeries >= 1"
                    )
        else:
            collector.error("protocol", "thresholds must be an object")

        video_path = episode.get("video_path")
        if video_path:
            safe = _safe_relative_path(video_path)
            if safe is None or not str(video_path).startswith("videos/"):
                collector.error("video", "%s has unsafe video_path" % episode_id)
            else:
                path = root.joinpath(*safe.parts)
                if not path.is_file() or path.stat().st_size <= 0:
                    collector.error("video", "%s video is missing or empty" % episode_id)
    collector.checked("matrix, pair, query, percentile, and video consistency")


def _compare_values(
    actual: Any,
    expected: Any,
    path: str,
    collector: _Collector,
) -> None:
    if isinstance(expected, bool):
        if type(actual) is not bool or actual != expected:
            collector.error("recomputation", "%s mismatch: got %r expected %r" % (path, actual, expected))
        return
    if expected is None:
        if actual is not None:
            collector.error("recomputation", "%s mismatch: got %r expected null" % (path, actual))
        return
    if isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            collector.error("recomputation", "%s is not numeric" % path)
        elif not math.isfinite(float(actual)) or not _close(actual, expected):
            collector.error("recomputation", "%s mismatch: got %r expected %r" % (path, actual, expected))
        return
    if isinstance(expected, int):
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
            collector.error("recomputation", "%s mismatch: got %r expected %r" % (path, actual, expected))
        return
    if isinstance(expected, str):
        if not isinstance(actual, str) or actual != expected:
            collector.error("recomputation", "%s mismatch: got %r expected %r" % (path, actual, expected))
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            collector.error("recomputation", "%s must be a list" % path)
            return
        if len(actual) != len(expected):
            collector.error("recomputation", "%s length mismatch" % path)
        for index, (left, right) in enumerate(zip(actual, expected)):
            _compare_values(left, right, "%s[%d]" % (path, index), collector)
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            collector.error("recomputation", "%s must be an object" % path)
            return
        if set(actual) != set(expected):
            collector.error(
                "recomputation",
                "%s keys mismatch: got %r expected %r"
                % (path, sorted(actual), sorted(expected)),
            )
        for key in sorted(set(actual) & set(expected)):
            _compare_values(actual[key], expected[key], "%s.%s" % (path, key), collector)
        return
    if actual != expected:
        collector.error("recomputation", "%s mismatch" % path)


def _csv_value(value: Any) -> str:
    return "" if value is None else str(value)


def _validate_report_file_pair(
    root: pathlib.Path,
    stem: str,
    expected: Sequence[Mapping[str, Any]],
    fixed_fields: Optional[Sequence[str]],
    collector: _Collector,
) -> None:
    actual_json = _read_json(root / (stem + ".json"), stem, collector)
    _compare_values(actual_json, list(expected), stem + ".json", collector)
    fields: Sequence[str]
    if fixed_fields is not None:
        fields = fixed_fields
    elif expected:
        fields = tuple(expected[0].keys())
    else:
        fields = ()
    actual_csv = _read_csv_rows(
        root / (stem + ".csv"), fields, stem, collector
    )
    if actual_csv is None:
        return
    if len(actual_csv) != len(expected):
        collector.error(
            "recomputation",
            "%s.csv row count mismatch: got %d expected %d"
            % (stem, len(actual_csv), len(expected)),
        )
    for index, (actual, wanted) in enumerate(zip(actual_csv, expected)):
        for field in fields:
            wanted_text = _csv_value(wanted[field])
            if actual[field] != wanted_text:
                collector.error(
                    "recomputation",
                    "%s.csv[%d].%s mismatch: got %r expected %r"
                    % (stem, index, field, actual[field], wanted_text),
                )


def _validate_recomputed_reports(
    root: pathlib.Path,
    episodes: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    bootstrap_resamples: int,
    collector: _Collector,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    try:
        aggregate = aggregate_episodes(episodes, queries)
        comparisons = paired_comparisons(episodes, bootstrap_resamples)
        control_comparisons = intervention_control_comparisons(
            episodes, bootstrap_resamples
        )
        contrasts = matched_condition_contrasts(episodes, bootstrap_resamples)
    except Exception as exc:
        collector.error("recomputation", "statistics could not be recomputed: %s" % exc)
        return [], [], []
    _validate_report_file_pair(root, "aggregate", aggregate, None, collector)
    _validate_report_file_pair(root, "paired_comparisons", comparisons, None, collector)
    _validate_report_file_pair(
        root,
        "intervention_control_comparisons",
        control_comparisons,
        CONTROL_COMPARISON_FIELDS,
        collector,
    )
    _validate_report_file_pair(
        root, "condition_contrasts", contrasts, CONTRAST_FIELDS, collector
    )
    collector.checked(
        "aggregate, paired, intervention-control, and registered contrast recomputation"
    )
    return aggregate, comparisons, control_comparisons


def _validate_progress_integrity_summary(
    root: pathlib.Path,
    episodes: Sequence[Mapping[str, Any]],
    cells: Sequence[ExperimentCell],
    aggregate: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    control_comparisons: Sequence[Mapping[str, Any]],
    collector: _Collector,
) -> None:
    expected_errors = artifact_integrity_errors(root, episodes, cells)
    integrity = _read_json(root / "integrity.json", "integrity", collector)
    if isinstance(integrity, Mapping):
        expected_keys = {"schema_version", "valid", "errors", "checked_at_utc"}
        if set(integrity) != expected_keys:
            collector.error("integrity", "integrity.json keys do not match schema")
        if integrity.get("schema_version") != SCHEMA_VERSION:
            collector.error("integrity", "schema_version mismatch")
        if type(integrity.get("valid")) is not bool:
            collector.error("integrity", "valid must be a JSON boolean")
        elif integrity.get("valid") != (not expected_errors):
            collector.error("integrity", "valid contradicts independently checked errors")
        if integrity.get("errors") != expected_errors:
            collector.error("integrity", "errors contradict independent integrity check")
    else:
        collector.error("integrity", "integrity.json must contain an object")
    if expected_errors:
        collector.error("integrity", "artifact is structurally invalid: %s" % "; ".join(expected_errors))

    planned = len(cells)
    complete = len(episodes) == planned and not expected_errors
    progress = _read_json(root / "progress.json", "progress", collector)
    if isinstance(progress, Mapping):
        expected_keys = {
            "schema_version",
            "planned_rollouts",
            "completed_rollouts",
            "complete",
            "updated_at_utc",
        }
        if set(progress) != expected_keys:
            collector.error("progress", "progress.json keys do not match schema")
        for field, expected_value in (
            ("schema_version", SCHEMA_VERSION),
            ("planned_rollouts", planned),
            ("completed_rollouts", len(episodes)),
            ("complete", complete),
        ):
            if progress.get(field) != expected_value or (
                field == "complete" and type(progress.get(field)) is not bool
            ):
                collector.error("progress", "%s mismatch" % field)
    else:
        collector.error("progress", "progress.json must contain an object")

    expected_summary = _summary_markdown(
        episodes, aggregate, comparisons, control_comparisons, planned
    )
    expected_summary += "\n## Artifact integrity\n\n"
    expected_summary += "- Valid: %s\n" % ("yes" if not expected_errors else "no")
    for error in expected_errors:
        expected_summary += "- ERROR: %s\n" % error
    try:
        actual_summary = (root / "summary.md").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        collector.error("summary", "cannot read summary.md: %s" % exc)
    else:
        if actual_summary != expected_summary:
            collector.error("summary", "summary.md is not the deterministic raw-data summary")
    collector.checked("integrity, progress, and deterministic summary")


def _validate_attestation(
    root: pathlib.Path,
    protocol: Any,
    environment: Any,
    collector: _Collector,
) -> None:
    if not isinstance(protocol, Mapping) or not isinstance(environment, Mapping):
        collector.error("attestation", "protocol and environment must be objects")
        return
    if environment.get("schema_version") != SCHEMA_VERSION:
        collector.error("environment", "schema_version mismatch")
    provenance = protocol.get("checkpoint_provenance")
    if provenance not in (
        "server_attestation_with_checkpoint_content_sha256",
        "launcher_declaration_only",
    ):
        collector.error("attestation", "unknown checkpoint_provenance declaration")
        return
    arguments = environment.get("arguments")
    if not isinstance(arguments, Mapping):
        collector.error("attestation", "environment.arguments must be an object")
        arguments = {}
    formal = provenance == "server_attestation_with_checkpoint_content_sha256"
    expected_bypass = not formal
    if arguments.get("allow_unattested_server") is not expected_bypass:
        collector.error("attestation", "diagnostic attestation bypass declaration mismatch")
    for argument_field, protocol_field in (
        ("checkpoint", "declared_checkpoint"),
        ("expected_openpi_commit", "openpi_commit"),
        ("server_launch_args", "server_launch_args"),
    ):
        if arguments.get(argument_field) != protocol.get(protocol_field):
            collector.error("attestation", "%s does not match protocol" % argument_field)

    commit = protocol.get("openpi_commit")
    if not isinstance(commit, str) or not _GIT_SHA.fullmatch(commit):
        collector.error("attestation", "protocol openpi_commit is not a lowercase Git SHA")
    if environment.get("openpi_git_commit") != commit:
        collector.error("attestation", "captured OpenPI commit does not match protocol")
    if protocol.get("policy_config") != DEFAULT_POLICY_CONFIG:
        collector.error("attestation", "protocol policy_config mismatch")

    metadata = environment.get("server_metadata")
    attestation = metadata.get("armbench_server_attestation") if isinstance(metadata, Mapping) else None
    if not formal:
        collector.warning(
            "attestation",
            "artifact explicitly uses diagnostic launcher-only checkpoint provenance",
        )
    elif not isinstance(attestation, Mapping):
        collector.error("attestation", "formal artifact has no server attestation")
    else:
        expected = {
            "schema_version": SERVER_ATTESTATION_SCHEMA_VERSION,
            "policy_loaded": True,
            "policy_config": protocol.get("policy_config"),
            "checkpoint_uri": protocol.get("declared_checkpoint"),
            "openpi_commit": protocol.get("openpi_commit"),
            "openpi_tracked_clean": True,
            "openpi_tracked_status": "",
            "openpi_submodules_clean": True,
            "action_horizon": PI05_LIBERO_ACTION_HORIZON,
        }
        for field, value in expected.items():
            if attestation.get(field) != value or (
                isinstance(value, bool) and type(attestation.get(field)) is not bool
            ):
                collector.error("attestation", "%s mismatch" % field)
        for field in ("checkpoint_content_sha256", "server_source_sha256"):
            value = attestation.get(field)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                collector.error("attestation", "%s is not a lowercase SHA-256" % field)
        for field in ("checkpoint_file_count", "checkpoint_total_bytes"):
            value = attestation.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                collector.error("attestation", "%s must be a positive integer" % field)

    source_hashes = environment.get("runtime_source_sha256")
    if not isinstance(source_hashes, Mapping):
        collector.error("provenance", "runtime_source_sha256 must be an object")
    else:
        for required in RUNTIME_SOURCE_FILES[:2]:
            if required not in source_hashes:
                collector.error("provenance", "missing required runtime source hash: %s" % required)
        for relative, digest in source_hashes.items():
            safe = _safe_relative_path(relative)
            if safe is None or not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                collector.error("provenance", "invalid runtime source record: %r" % relative)
                continue
            snapshot = root / "provenance" / "armbench_source"
            snapshot = snapshot.joinpath(*safe.parts)
            if not snapshot.is_file():
                collector.error("provenance", "runtime source snapshot missing: %s" % relative)
            elif _sha256(snapshot) != digest:
                collector.error("provenance", "runtime source snapshot hash mismatch: %s" % relative)
        if formal:
            server_source = "integrations/openpi/serve_policy_attested.py"
            server_digest = source_hashes.get(server_source)
            if not isinstance(server_digest, str) or not _SHA256.fullmatch(
                server_digest
            ):
                collector.error(
                    "provenance", "formal artifact is missing the attested server source hash"
                )
            elif isinstance(attestation, Mapping) and attestation.get(
                "server_source_sha256"
            ) != server_digest:
                collector.error(
                    "provenance", "server attestation does not match the source snapshot"
                )
    diff_hash = environment.get("armbench_git_diff_sha256")
    if not isinstance(diff_hash, str) or not _SHA256.fullmatch(diff_hash):
        collector.error("provenance", "armbench_git_diff_sha256 is invalid")
    collector.checked("checkpoint, server, commit, and source attestation")


def validate_artifact(path: pathlib.Path) -> ValidationReport:
    """Validate a finalized artifact without trusting its derived reports."""

    root = pathlib.Path(path).resolve()
    collector = _Collector()
    if not root.is_dir():
        collector.error("artifact", "artifact directory does not exist: %s" % root)
        return ValidationReport(
            str(root), False, tuple(collector.errors), (), tuple(collector.checks)
        )

    try:
        _validate_manifest(root, collector)
        protocol = _read_json(root / "resolved_protocol.json", "protocol", collector)
        environment = _read_json(root / "environment.json", "environment", collector)
        episodes, queries = _parse_raw_rows(root, collector)
        resolved = _resolve_expected_cells(protocol, collector)
        _validate_attestation(root, protocol, environment, collector)
        if episodes is not None and queries is not None and resolved is not None:
            cells, modes, bootstrap = resolved
            _validate_raw_consistency(
                root, episodes, queries, protocol, cells, modes, collector
            )
            aggregate, comparisons, control_comparisons = _validate_recomputed_reports(
                root, episodes, queries, bootstrap, collector
            )
            _validate_progress_integrity_summary(
                root,
                episodes,
                cells,
                aggregate,
                comparisons,
                control_comparisons,
                collector,
            )
    except Exception as exc:  # Fail closed even for a new malformed-input edge case.
        collector.error(
            "validator", "unexpected validation failure: %s: %s" % (type(exc).__name__, exc)
        )
    return ValidationReport(
        artifact=str(root),
        valid=not collector.errors,
        errors=tuple(collector.errors),
        warnings=tuple(collector.warnings),
        checks=tuple(collector.checks),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=pathlib.Path, help="Finalized artifact directory")
    parser.add_argument(
        "--json", action="store_true", help="Print a machine-readable validation report"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    report = validate_artifact(args.artifact)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print("VALID" if report.valid else "INVALID", report.artifact)
        for error in report.errors:
            print("ERROR", error)
        for warning in report.warnings:
            print("WARNING", warning)
        if report.valid:
            print("Checked: %s" % "; ".join(report.checks))
    return 0 if report.valid else 2


if __name__ == "__main__":
    sys.exit(main())
