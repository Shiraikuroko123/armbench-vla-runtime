"""Independent fail-closed validator for measured-age pi0.5 artifacts.

The evaluator and its timing helpers are intentionally not imported.  This
module treats every artifact byte as untrusted and reconstructs the registered
matrix, monotonic-clock arithmetic, temporal-alignment decisions, keyed jitter,
episode counters, and derived summaries from fixed v2 contracts.
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "armbench.pi05_libero_measured_age.v2"
VALID_MODES = ("async_unguarded", "latency_aligned")
MEASURED_WALL = "measured_wall"

# Copied from measured_age_libero_eval.py.  Importing the producer would make a
# producer regression invisible to this validator.
WARMUP_FIELDS = (
    "schema_version", "warmup_index", "scored", "client_session_id",
    "checkpoint_content_sha256", "observation_captured_monotonic_ns",
    "policy_call_started_monotonic_ns", "policy_call_finished_monotonic_ns",
    "response_ready_monotonic_ns", "observation_age_ms",
    "inference_latency_ms", "action_chunk_steps", "action_dimension",
    "accepted", "error_type", "error_message",
)
EPISODE_FIELDS = (
    "schema_version", "episode_id", "pair_id", "condition_order",
    "task_suite", "task_id", "episode_index", "task_description", "mode",
    "latency_source", "replan_steps", "latency_steps", "seed", "success",
    "termination_reason", "initial_state_sha256", "environment_steps",
    "task_action_steps", "latency_action_steps", "policy_queries",
    "accepted_chunks", "rejected_chunks", "stale_chunks_executed",
    "stale_action_steps", "interventions", "deadline_misses",
    "horizon_overruns", "age_refreshes", "fallback_hold_steps",
    "simulated_catchup_steps", "observation_age_p50_ms",
    "observation_age_p95_ms", "observation_age_max_ms",
    "inference_latency_p50_ms", "inference_latency_p95_ms", "wall_time_s",
    "failure_category", "failure_type", "failure_message", "video_required",
    "video_path", "video_error_type", "video_error_message",
)
QUERY_FIELDS = (
    "schema_version", "episode_id", "pair_id", "condition_order",
    "task_suite", "task_id", "episode_index", "mode", "latency_source",
    "replan_steps", "latency_steps", "query_index", "observation_step",
    "response_step", "observation_captured_monotonic_ns",
    "policy_call_started_monotonic_ns", "policy_call_finished_monotonic_ns",
    "response_ready_monotonic_ns", "clock_trace_complete",
    "observation_age_ms", "inference_latency_ms",
    "response_delivery_elapsed_ms", "policy_inference_latency_ms",
    "server_inference_latency_ms", "response_jitter_requested_ms",
    "jitter_key_sha256", "completed_controller_steps",
    "simulated_catchup_steps", "action_chunk_steps", "measured_stale_steps",
    "action_offset_steps", "selected_stop_step", "available_suffix_steps",
    "deadline_exceeded", "horizon_overrun", "age_refresh_index",
    "fallback_hold_steps", "alignment_disposition", "alignment_reason",
    "accepted", "decision", "rejection_reasons", "position_mismatch_m",
    "orientation_mismatch_rad", "gripper_mismatch_linf", "error_stage",
    "error_type", "error_message",
)
RUNTIME_SOURCE_FILES = (
    "integrations/openpi/libero_runtime.py",
    "integrations/openpi/libero_runtime_eval.py",
    "integrations/openpi/measured_age_libero_eval.py",
    "integrations/openpi/validate_measured_age_artifact.py",
    "integrations/openpi/measured_age_compose_run.py",
    "integrations/openpi/deadline_alignment.py",
    "integrations/openpi/preflight.py",
    "integrations/openpi/compose.libero-runtime.yml",
    "integrations/openpi/compose.libero-measured-age.yml",
    "integrations/openpi/serve_policy_attested.py",
)

_CORE_FILES = frozenset(
    {
        "resolved_protocol.json", "environment.json", "warmup_queries.csv",
        "per_episode.csv", "per_query.csv", "progress.json", "summary.json",
        "summary.md", "integrity.json",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_UINT = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_FINITE_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")
_FLOAT_TOL = 1e-9


@dataclasses.dataclass(frozen=True)
class ValidationReport:
    artifact: str
    valid: bool
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...]
    checks: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "armbench.measured_age_artifact_validation.v1",
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
        if len(self.errors) < 250:
            self.errors.append("[%s] %s" % (section, message))
        elif len(self.errors) == 250:
            self.errors.append("[validator] further errors suppressed")

    def checked(self, message: str) -> None:
        self.checks.append(message)


def _strict_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate JSON key: %s" % key)
        output[key] = value
    return output


def _reject_constant(value: str) -> Any:
    raise ValueError("non-finite JSON value: %s" % value)


def _read_json(path: pathlib.Path, section: str, c: _Collector) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        c.error(section, "cannot read strict JSON %s: %s" % (path.name, exc))
        return None


def _safe_path(value: Any) -> Optional[pathlib.PurePosixPath]:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        return None
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        return None
    if any(part in ("", ".", "..") for part in path.parts):
        return None
    return path


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_manifest(root: pathlib.Path, c: _Collector) -> Optional[Mapping[str, Any]]:
    value = _read_json(root / "manifest.json", "manifest", c)
    if not isinstance(value, Mapping):
        c.error("manifest", "manifest.json must contain an object")
        return None
    if value.get("schema_version") != SCHEMA_VERSION:
        c.error("manifest", "schema_version mismatch")
    records = value.get("files")
    if not isinstance(records, Mapping):
        c.error("manifest", "files must be an object")
        return value

    safe: Dict[str, Mapping[str, Any]] = {}
    folded: Dict[str, str] = {}
    root_resolved = root.resolve()
    for raw_path, record in records.items():
        relative = _safe_path(raw_path)
        if relative is None:
            c.error("manifest", "unsafe or non-canonical path: %r" % raw_path)
            continue
        canonical = relative.as_posix()
        key = canonical.casefold()
        if key in folded:
            c.error("manifest", "duplicate normalized path: %s" % canonical)
            continue
        folded[key] = canonical
        candidate = root.joinpath(*relative.parts)
        try:
            candidate.resolve().relative_to(root_resolved)
        except (OSError, ValueError):
            c.error("manifest", "path escapes artifact: %s" % canonical)
            continue
        if candidate.is_symlink():
            c.error("manifest", "symbolic link is forbidden: %s" % canonical)
        if not isinstance(record, Mapping):
            c.error("manifest", "record must be an object: %s" % canonical)
            continue
        safe[canonical] = record

    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json" and path.suffix != ".tmp"
    }
    omitted = sorted(set(actual) - set(safe))
    absent = sorted(set(safe) - set(actual))
    if omitted:
        c.error("manifest", "files omitted from manifest: %s" % ", ".join(omitted))
    if absent:
        c.error("manifest", "manifest references absent files: %s" % ", ".join(absent))
    missing = sorted(_CORE_FILES - set(safe))
    if missing:
        c.error("manifest", "required files missing: %s" % ", ".join(missing))
    for relative in sorted(set(actual) & set(safe)):
        record = safe[relative]
        byte_count = record.get("bytes")
        digest = record.get("sha256")
        if type(byte_count) is not int or byte_count < 0:
            c.error("manifest", "invalid byte count for %s" % relative)
        elif actual[relative].stat().st_size != byte_count:
            c.error("manifest", "byte count mismatch for %s" % relative)
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            c.error("manifest", "invalid SHA-256 for %s" % relative)
        elif _file_sha256(actual[relative]) != digest:
            c.error("manifest", "SHA-256 mismatch for %s" % relative)
    c.checked("manifest path safety, complete coverage, byte counts, and hashes")
    return value


def _read_csv(
    path: pathlib.Path, fields: Sequence[str], section: str, c: _Collector
) -> Optional[List[Dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            raw = list(csv.reader(handle, strict=True))
    except (OSError, UnicodeError, csv.Error) as exc:
        c.error(section, "cannot read %s: %s" % (path.name, exc))
        return None
    if not raw or raw[0] != list(fields):
        c.error(section, "%s exact header mismatch" % path.name)
        return None
    output = []
    for number, values in enumerate(raw[1:], start=2):
        if len(values) != len(fields):
            c.error(section, "%s row %d field-count mismatch" % (path.name, number))
            continue
        output.append(dict(zip(fields, values)))
    return output


def _uint(value: str, field: str, positive: bool = False) -> int:
    if not _UINT.fullmatch(value):
        raise ValueError("%s must be a canonical unsigned integer" % field)
    result = int(value)
    if positive and result == 0:
        raise ValueError("%s must be positive" % field)
    return result


def _opt_uint(value: str, field: str) -> Optional[int]:
    return None if value == "" else _uint(value, field)


def _bool(value: str, field: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError("%s must be exactly True or False" % field)


def _float(value: str, field: str, optional: bool = False) -> Optional[float]:
    if optional and value == "":
        return None
    if not value or value != value.strip() or not _FINITE_DECIMAL.fullmatch(value):
        raise ValueError("%s must be a canonical finite float" % field)
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError("%s must be a canonical finite float" % field) from exc
    if not math.isfinite(result) or result < 0.0 or value.startswith("+"):
        raise ValueError("%s must be finite and nonnegative" % field)
    return result


def _required(value: str, field: str) -> str:
    if not value or value != value.strip():
        raise ValueError("%s must be nonempty and trimmed" % field)
    return value


def _parse_warmup(raw: Mapping[str, str]) -> Dict[str, Any]:
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError("schema_version mismatch")
    row: Dict[str, Any] = dict(raw)
    for field in (
        "warmup_index", "observation_captured_monotonic_ns",
        "policy_call_started_monotonic_ns", "policy_call_finished_monotonic_ns",
        "response_ready_monotonic_ns", "action_chunk_steps", "action_dimension",
    ):
        row[field] = _uint(raw[field], field)
    for field in ("observation_age_ms", "inference_latency_ms"):
        row[field] = _float(raw[field], field)
    row["scored"] = _bool(raw["scored"], "scored")
    row["accepted"] = _bool(raw["accepted"], "accepted")
    row["client_session_id"] = _required(raw["client_session_id"], "client_session_id")
    digest = _required(raw["checkpoint_content_sha256"], "checkpoint_content_sha256")
    if not _SHA256.fullmatch(digest):
        raise ValueError("checkpoint_content_sha256 is invalid")
    return row


_EPISODE_INTS = frozenset(
    {
        "condition_order", "task_id", "episode_index", "replan_steps",
        "latency_steps", "seed", "environment_steps", "task_action_steps",
        "latency_action_steps", "policy_queries", "accepted_chunks",
        "rejected_chunks", "stale_chunks_executed", "stale_action_steps",
        "interventions", "deadline_misses", "horizon_overruns", "age_refreshes",
        "fallback_hold_steps", "simulated_catchup_steps",
    }
)
_EPISODE_FLOATS = frozenset(
    {
        "observation_age_p50_ms", "observation_age_p95_ms",
        "observation_age_max_ms", "inference_latency_p50_ms",
        "inference_latency_p95_ms",
    }
)


def _parse_episode(raw: Mapping[str, str]) -> Dict[str, Any]:
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError("schema_version mismatch")
    row: Dict[str, Any] = dict(raw)
    for field in _EPISODE_INTS:
        row[field] = _uint(raw[field], field, positive=field == "replan_steps")
    for field in _EPISODE_FLOATS:
        row[field] = _float(raw[field], field, optional=True)
    row["wall_time_s"] = _float(raw["wall_time_s"], "wall_time_s")
    row["success"] = _bool(raw["success"], "success")
    row["video_required"] = _bool(raw["video_required"], "video_required")
    for field in (
        "episode_id", "pair_id", "task_suite", "task_description", "mode",
        "latency_source", "termination_reason", "initial_state_sha256",
    ):
        row[field] = _required(raw[field], field)
    if row["mode"] not in VALID_MODES or row["latency_source"] != MEASURED_WALL:
        raise ValueError("invalid measured-age mode or latency_source")
    if row["latency_steps"] != 0:
        raise ValueError("latency_steps must be zero")
    if not _SHA256.fullmatch(row["initial_state_sha256"]):
        raise ValueError("initial_state_sha256 is invalid")
    return row


_QUERY_INTS = frozenset(
    {
        "condition_order", "task_id", "episode_index", "replan_steps",
        "latency_steps", "query_index", "observation_step", "response_step",
        "completed_controller_steps", "simulated_catchup_steps",
        "action_chunk_steps", "measured_stale_steps", "action_offset_steps",
        "selected_stop_step", "available_suffix_steps", "age_refresh_index",
        "fallback_hold_steps",
    }
)
_QUERY_NS = frozenset(
    {
        "observation_captured_monotonic_ns", "policy_call_started_monotonic_ns",
        "policy_call_finished_monotonic_ns", "response_ready_monotonic_ns",
    }
)
_QUERY_FLOATS = frozenset(
    {
        "observation_age_ms", "inference_latency_ms",
        "response_delivery_elapsed_ms", "response_jitter_requested_ms",
    }
)
_QUERY_OPTIONAL_FLOATS = frozenset(
    {
        "policy_inference_latency_ms", "server_inference_latency_ms",
        "position_mismatch_m", "orientation_mismatch_rad", "gripper_mismatch_linf",
    }
)


def _parse_query(raw: Mapping[str, str]) -> Dict[str, Any]:
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError("schema_version mismatch")
    row: Dict[str, Any] = dict(raw)
    complete = _bool(raw["clock_trace_complete"], "clock_trace_complete")
    row["clock_trace_complete"] = complete
    for field in _QUERY_INTS:
        row[field] = _uint(raw[field], field, positive=field == "replan_steps")
    for field in _QUERY_NS:
        row[field] = _opt_uint(raw[field], field)
    for field in _QUERY_FLOATS:
        row[field] = _float(raw[field], field, optional=not complete)
    for field in _QUERY_OPTIONAL_FLOATS:
        row[field] = _float(raw[field], field, optional=True)
    for field in ("deadline_exceeded", "horizon_overrun", "accepted"):
        row[field] = _bool(raw[field], field)
    for field in (
        "episode_id", "pair_id", "task_suite", "mode", "latency_source",
        "alignment_disposition", "alignment_reason", "decision",
    ):
        row[field] = _required(raw[field], field)
    if row["mode"] not in VALID_MODES or row["latency_source"] != MEASURED_WALL:
        raise ValueError("invalid measured-age mode or latency_source")
    if row["latency_steps"] != 0:
        raise ValueError("latency_steps must be zero")
    if not _SHA256.fullmatch(raw["jitter_key_sha256"]):
        raise ValueError("jitter_key_sha256 is invalid")
    return row


def _parse_rows(root: pathlib.Path, c: _Collector) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    specs = (
        ("warmup_queries.csv", WARMUP_FIELDS, "warmup", _parse_warmup),
        ("per_episode.csv", EPISODE_FIELDS, "episodes", _parse_episode),
        ("per_query.csv", QUERY_FIELDS, "queries", _parse_query),
    )
    parsed_sets: List[List[Dict[str, Any]]] = []
    for filename, fields, section, parser in specs:
        raw_rows = _read_csv(root / filename, fields, section, c) or []
        parsed = []
        for number, raw in enumerate(raw_rows, start=2):
            try:
                parsed.append(parser(raw))
            except (KeyError, ValueError) as exc:
                c.error(section, "row %d: %s" % (number, exc))
        if len(parsed) != len(raw_rows):
            c.error(section, "one or more rows failed canonical parsing")
        parsed_sets.append(parsed)
    c.checked("exact CSV headers and canonical finite scalar parsing")
    return parsed_sets[0], parsed_sets[1], parsed_sets[2]


def _json_number(value: Any, name: str, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be numeric" % name)
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError("%s must be finite and nonnegative" % name)
    return result


def _json_uint(value: Any, name: str, positive: bool = False) -> int:
    if type(value) is not int or value < 0 or (positive and value == 0):
        raise ValueError("%s must be a canonical unsigned integer" % name)
    return value


@dataclasses.dataclass(frozen=True)
class _Protocol:
    control_period_ms: float
    completed_rounding: str
    offset_rounding: str
    tolerance_ms: float
    deadline_ms: Optional[float]
    max_refreshes: int
    jitter_seed: int
    jitter_candidates: Tuple[float, ...]
    registered_cells: Tuple[Mapping[str, Any], ...]
    warmup_queries: int
    matrix: Mapping[str, Any]


def _protocol(value: Any, c: _Collector) -> Optional[_Protocol]:
    try:
        if not isinstance(value, Mapping) or value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("protocol schema_version mismatch")
        temporal = value["temporal_alignment"]
        jitter = value["jitter"]
        warmup = value["warmup"]
        cells = value["registered_cells"]
        if not all(isinstance(item, Mapping) for item in (temporal, jitter, warmup)):
            raise ValueError("protocol sections must be objects")
        if not isinstance(cells, list) or not cells:
            raise ValueError("registered_cells must be a nonempty array")
        if temporal.get("latency_source") != MEASURED_WALL or temporal.get("latency_steps") != 0:
            raise ValueError("temporal alignment must use measured_wall and zero fixed steps")
        completed = temporal.get("completed_step_rounding")
        rounding = temporal.get("action_offset_rounding")
        if completed != "floor" or rounding not in ("ceil", "floor"):
            raise ValueError("invalid registered rounding")
        deadline_raw = temporal.get("deadline_ms")
        deadline = None if deadline_raw is None else _json_number(deadline_raw, "deadline_ms")
        expected_jitter_contract = {
            "generator": "sha256_first_u64_mod_v1",
            "pairing_key_fields": ["task_suite", "task_id", "episode_index", "replan_steps"],
            "query_index_field": "query_index",
            "mode_in_key": False,
            "payload_fields": ["seed", "pairing_key", "query_index"],
            "json_encoding": "utf-8 canonical sorted compact ASCII",
        }
        for key, expected in expected_jitter_contract.items():
            if jitter.get(key) != expected:
                raise ValueError("jitter.%s contract mismatch" % key)
        candidates_raw = jitter.get("candidates_ms")
        if not isinstance(candidates_raw, list) or not candidates_raw:
            raise ValueError("jitter candidates must be nonempty")
        candidates = tuple(_json_number(item, "jitter candidate") for item in candidates_raw)
        if len(candidates) != len(set(candidates)):
            raise ValueError("jitter candidates must be unique")
        if warmup.get("scored") is not False or warmup.get("same_checkpoint_and_action_contract") is not True:
            raise ValueError("warmup must be unscored and use the scored action contract")
        result = _Protocol(
            control_period_ms=_json_number(temporal["control_period_ms"], "control_period_ms"),
            completed_rounding=str(completed),
            offset_rounding=str(rounding),
            tolerance_ms=_json_number(temporal["boundary_tolerance_ms"], "boundary_tolerance_ms"),
            deadline_ms=deadline,
            max_refreshes=_json_uint(temporal["max_age_refreshes"], "max_age_refreshes"),
            jitter_seed=_json_uint(jitter["seed"], "jitter.seed"),
            jitter_candidates=candidates,
            registered_cells=tuple(cells),
            warmup_queries=_json_uint(warmup["queries"], "warmup.queries", positive=True),
            matrix=value["matrix"],
        )
        if not isinstance(result.matrix, Mapping):
            raise ValueError("matrix must be an object")
        if result.control_period_ms <= 0.0:
            raise ValueError("control_period_ms must be positive")
    except (KeyError, TypeError, ValueError) as exc:
        c.error("protocol", str(exc))
        return None
    c.checked("fixed measured-age protocol, matrix, rounding, refresh, and jitter contract")
    return result


def _close(left: Any, right: Any, tolerance: float = _FLOAT_TOL) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _completed(age: float, period: float, tolerance: float) -> int:
    return int(math.floor((age + tolerance) / period))


def _stale(age: float, period: float, rounding: str, tolerance: float) -> int:
    if rounding == "floor":
        return _completed(age, period, tolerance)
    if age <= tolerance:
        return 0
    return int(math.ceil((age - tolerance) / period))


def _jitter(row: Mapping[str, Any], protocol: _Protocol) -> Tuple[str, float]:
    payload = json.dumps(
        {
            "pairing_key": [
                row["task_suite"], int(row["task_id"]), int(row["episode_index"]),
                int(row["replan_steps"]),
            ],
            "query_index": int(row["query_index"]),
            "seed": protocol.jitter_seed,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    index = int.from_bytes(digest[:8], "big") % len(protocol.jitter_candidates)
    return digest.hex(), protocol.jitter_candidates[index]


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _validate_warmup(
    warmups: Sequence[Mapping[str, Any]], queries: Sequence[Mapping[str, Any]],
    protocol: _Protocol, c: _Collector,
) -> None:
    if len(warmups) != protocol.warmup_queries:
        c.error("warmup", "row count does not match registered warmup queries")
    if [row["warmup_index"] for row in warmups] != list(range(len(warmups))):
        c.error("warmup", "warmup_index must be contiguous in file order")
    last_ready = None
    sessions = set()
    checkpoints = set()
    for row in warmups:
        if row["scored"] or not row["accepted"]:
            c.error("warmup", "warmup rows must be unscored accepted calls")
        times = [
            row["observation_captured_monotonic_ns"], row["policy_call_started_monotonic_ns"],
            row["policy_call_finished_monotonic_ns"], row["response_ready_monotonic_ns"],
        ]
        if times != sorted(times):
            c.error("warmup", "monotonic clock order is invalid")
        age = (times[3] - times[0]) / 1_000_000.0
        inference = (times[2] - times[1]) / 1_000_000.0
        if not _close(row["observation_age_ms"], age, 1e-6) or not _close(row["inference_latency_ms"], inference, 1e-6):
            c.error("warmup", "clock-derived latency mismatch")
        last_ready = times[3]
        sessions.add(row["client_session_id"])
        checkpoints.add(row["checkpoint_content_sha256"])
    if len(sessions) > 1 or len(checkpoints) > 1:
        c.error("warmup", "warmups do not share one session/checkpoint")
    scored_starts = [row["observation_captured_monotonic_ns"] for row in queries if row["clock_trace_complete"]]
    if last_ready is not None and scored_starts and last_ready > min(scored_starts):
        c.error("warmup", "warmup did not finish before scored observations")
    c.checked("unscored attested warmup precedes every scored query")


def _registered_cell_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("episode_id"), row.get("pair_id"), row.get("condition_order"),
        row.get("task_suite"), row.get("task_id"), row.get("episode_index"),
        row.get("mode"), row.get("replan_steps"), row.get("latency_source"),
        row.get("latency_steps"),
    )


def _validate_queries(
    episodes: Sequence[Mapping[str, Any]], queries: Sequence[Mapping[str, Any]],
    protocol: _Protocol, c: _Collector,
) -> None:
    episode_map = {row["episode_id"]: row for row in episodes}
    if len(episode_map) != len(episodes):
        c.error("matrix", "episode_id values must be unique")
    registered = {_registered_cell_key(row) for row in protocol.registered_cells}
    actual = {_registered_cell_key(row) for row in episodes}
    if len(registered) != len(protocol.registered_cells):
        c.error("matrix", "registered_cells contains duplicate cells")
    if len(episodes) != len(protocol.registered_cells) or registered != actual:
        c.error("matrix", "per_episode rows do not exactly cover registered_cells")
    expected_matrix = {
        "schema_version": SCHEMA_VERSION,
        "rollouts": len(protocol.registered_cells),
        "paired_conditions": len({row.get("pair_id") for row in protocol.registered_cells}),
        "matched_condition_groups": len({row.get("pair_id") for row in protocol.registered_cells}),
        "task_suites": sorted({row.get("task_suite") for row in protocol.registered_cells}),
        "task_ids": sorted({row.get("task_id") for row in protocol.registered_cells}),
        "episode_indices": sorted({row.get("episode_index") for row in protocol.registered_cells}),
        "modes": sorted({row.get("mode") for row in protocol.registered_cells}),
        "replan_steps": sorted({row.get("replan_steps") for row in protocol.registered_cells}),
        "latency_sources": [MEASURED_WALL],
        "latency_steps": [0],
    }
    if dict(protocol.matrix) != expected_matrix:
        c.error("matrix", "matrix summary does not exactly match registered_cells")
    orders = [row.get("condition_order") for row in protocol.registered_cells]
    if orders != list(range(len(orders))):
        c.error("matrix", "registered condition_order must be contiguous in file order")

    by_pair: Dict[str, List[Mapping[str, Any]]] = {}
    for episode in episodes:
        by_pair.setdefault(str(episode["pair_id"]), []).append(episode)
    for pair_id, rows in by_pair.items():
        if {row["mode"] for row in rows} != set(VALID_MODES) or len(rows) != 2:
            c.error("pairing", "%s is not one baseline/candidate pair" % pair_id)
        invariants = {
            (row["task_suite"], row["task_id"], row["episode_index"], row["replan_steps"],
             row["seed"], row["initial_state_sha256"], row["task_description"])
            for row in rows
        }
        if len(invariants) != 1:
            c.error("pairing", "%s paired initial condition mismatch" % pair_id)

    by_episode: Dict[str, List[Mapping[str, Any]]] = {}
    paired_jitter: Dict[Tuple[str, int], Dict[str, Tuple[str, float]]] = {}
    for row in queries:
        episode = episode_map.get(row["episode_id"])
        if episode is None:
            c.error("queries", "query references unknown episode %s" % row["episode_id"])
            continue
        by_episode.setdefault(str(row["episode_id"]), []).append(row)
        for field in (
            "pair_id", "condition_order", "task_suite", "task_id", "episode_index",
            "mode", "latency_source", "replan_steps", "latency_steps",
        ):
            if row[field] != episode[field]:
                c.error("queries", "%s query %s mismatch" % (row["episode_id"], field))

        if not row["clock_trace_complete"]:
            c.error("timing", "%s query %d has incomplete clock trace" % (row["episode_id"], row["query_index"]))
            continue
        times = [row[field] for field in (
            "observation_captured_monotonic_ns", "policy_call_started_monotonic_ns",
            "policy_call_finished_monotonic_ns", "response_ready_monotonic_ns",
        )]
        if any(value is None for value in times) or times != sorted(times):
            c.error("timing", "%s query %d raw ns ordering mismatch" % (row["episode_id"], row["query_index"]))
            continue
        age = (times[3] - times[0]) / 1_000_000.0
        inference = (times[2] - times[1]) / 1_000_000.0
        delivery = (times[3] - times[2]) / 1_000_000.0
        for field, expected in (
            ("observation_age_ms", age), ("inference_latency_ms", inference),
            ("response_delivery_elapsed_ms", delivery),
        ):
            if not _close(row[field], expected, 1e-6):
                c.error("timing", "%s query %d %s mismatch" % (row["episode_id"], row["query_index"], field))

        completed = _completed(age, protocol.control_period_ms, protocol.tolerance_ms)
        stale = _stale(age, protocol.control_period_ms, protocol.offset_rounding, protocol.tolerance_ms)
        stop = stale + row["replan_steps"]
        available = max(0, row["action_chunk_steps"] - stale)
        deadline = protocol.deadline_ms is not None and age > protocol.deadline_ms + protocol.tolerance_ms
        horizon = stop > row["action_chunk_steps"]
        for field, expected in (
            ("completed_controller_steps", completed),
            ("simulated_catchup_steps", row["response_step"] - row["observation_step"]),
            ("measured_stale_steps", stale), ("selected_stop_step", stop),
            ("available_suffix_steps", available), ("deadline_exceeded", deadline),
            ("horizon_overrun", horizon),
        ):
            if row[field] != expected:
                c.error("alignment", "%s query %d %s mismatch" % (row["episode_id"], row["query_index"], field))
        if row["simulated_catchup_steps"] > completed:
            c.error("alignment", "%s query %d catch-up exceeds completed ticks" % (row["episode_id"], row["query_index"]))

        expected_hash, expected_jitter = _jitter(row, protocol)
        if row["jitter_key_sha256"] != expected_hash:
            c.error("jitter", "%s query %d jitter key SHA-256 mismatch" % (row["episode_id"], row["query_index"]))
        if not _close(row["response_jitter_requested_ms"], expected_jitter):
            c.error("jitter", "%s query %d jitter value mismatch" % (row["episode_id"], row["query_index"]))
        paired_jitter.setdefault((str(row["pair_id"]), int(row["query_index"])), {})[str(row["mode"])] = (expected_hash, float(row["response_jitter_requested_ms"]))

        plan_disposition = "execute"
        if deadline or horizon:
            plan_disposition = "hold_refresh" if row["age_refresh_index"] < protocol.max_refreshes else "fail_closed"
        reason = (
            "deadline_and_horizon_overrun" if deadline and horizon else
            "deadline_exceeded" if deadline else "horizon_overrun" if horizon else
            "fresh_suffix_available"
        )
        terminal_before_dispatch = row["decision"] in (
            "success_during_inference_delay",
            "step_budget_exhausted_during_delay",
        )
        if terminal_before_dispatch:
            expected_offset = stale if row["mode"] == "latency_aligned" else 0
            if (
                row["accepted"]
                or row["action_offset_steps"] != expected_offset
                or row["fallback_hold_steps"] != 0
                or row["rejection_reasons"] != ""
                or row["alignment_disposition"] != "terminal_before_dispatch"
                or row["alignment_reason"] != row["decision"]
            ):
                c.error("alignment", "%s query %d terminal-before-dispatch semantics mismatch" % (row["episode_id"], row["query_index"]))
        elif row["mode"] == "async_unguarded":
            if row["action_offset_steps"] != 0 or not row["accepted"] or row["decision"] != "accepted_unguarded":
                c.error("alignment", "%s query %d unguarded dispatch mismatch" % (row["episode_id"], row["query_index"]))
            if row["alignment_disposition"] != "not_applied" or row["alignment_reason"] != "async_unguarded":
                c.error("alignment", "%s query %d unguarded alignment labels mismatch" % (row["episode_id"], row["query_index"]))
            if row["fallback_hold_steps"] != 0:
                c.error("alignment", "%s query %d unguarded fallback hold is forbidden" % (row["episode_id"], row["query_index"]))
        else:
            if row["action_offset_steps"] != stale:
                c.error("alignment", "%s query %d aligned offset mismatch" % (row["episode_id"], row["query_index"]))
            if row["alignment_disposition"] != plan_disposition or row["alignment_reason"] != reason:
                c.error("alignment", "%s query %d disposition/reason mismatch" % (row["episode_id"], row["query_index"]))
            if plan_disposition == "execute":
                if not row["accepted"] or row["decision"] != "accepted_measured_latency_aligned" or row["fallback_hold_steps"] != 0:
                    c.error("alignment", "%s query %d accepted suffix semantics mismatch" % (row["episode_id"], row["query_index"]))
            else:
                expected_decision = "rejected_%s_%s" % (reason, plan_disposition)
                if row["accepted"] or row["decision"] != expected_decision or row["rejection_reasons"] != reason:
                    c.error("alignment", "%s query %d rejection semantics mismatch" % (row["episode_id"], row["query_index"]))
                if row["fallback_hold_steps"] not in (0, 1):
                    c.error("alignment", "%s query %d invalid fallback hold count" % (row["episode_id"], row["query_index"]))

    for key, modes in paired_jitter.items():
        if set(modes) == set(VALID_MODES) and len(set(modes.values())) != 1:
            c.error("jitter", "%s query %d differs across paired modes" % key)

    for episode in episodes:
        episode_id = str(episode["episode_id"])
        rows = by_episode.get(episode_id, [])
        indices = [row["query_index"] for row in rows]
        if indices != list(range(len(rows))):
            c.error("queries", "%s query_index is not contiguous in file order" % episode_id)
        if len(rows) != episode["policy_queries"]:
            c.error("episodes", "%s policy_queries mismatch" % episode_id)
        terminal_rows = [
            row for row in rows if row["decision"] in (
                "success_during_inference_delay",
                "step_budget_exhausted_during_delay",
            )
        ]
        if terminal_rows:
            terminal = terminal_rows[-1]
            if len(terminal_rows) != 1 or terminal is not rows[-1]:
                c.error("episodes", "%s terminal-before-dispatch query must be unique and last" % episode_id)
            expected_success = terminal["decision"] == "success_during_inference_delay"
            expected_termination = "success_during_inference_delay" if expected_success else "step_limit"
            if episode["success"] is not expected_success or episode["termination_reason"] != expected_termination:
                c.error("episodes", "%s terminal-before-dispatch episode outcome mismatch" % episode_id)
        if episode["mode"] == "latency_aligned":
            refresh = 0
            for row in rows:
                if row["age_refresh_index"] != refresh:
                    c.error("refresh", "%s query %d refresh index mismatch" % (episode_id, row["query_index"]))
                if row["alignment_disposition"] == "hold_refresh":
                    refresh += 1
                    if refresh > protocol.max_refreshes:
                        c.error("refresh", "%s exceeded max_age_refreshes" % episode_id)
                elif row["alignment_disposition"] == "execute":
                    refresh = 0
                elif row["alignment_disposition"] == "fail_closed" and row is not rows[-1]:
                    c.error("refresh", "%s continued after fail_closed" % episode_id)
        accepted = sum(bool(row["accepted"]) for row in rows)
        rejected = sum(str(row["alignment_disposition"]) in ("hold_refresh", "fail_closed") for row in rows)
        counters = {
            "accepted_chunks": accepted,
            "rejected_chunks": rejected,
            "interventions": rejected,
            "deadline_misses": sum(bool(row["deadline_exceeded"]) for row in rows),
            "horizon_overruns": sum(bool(row["horizon_overrun"]) for row in rows),
            "age_refreshes": sum(row["alignment_disposition"] == "hold_refresh" for row in rows),
            "fallback_hold_steps": sum(int(row["fallback_hold_steps"]) for row in rows),
            "simulated_catchup_steps": sum(int(row["simulated_catchup_steps"]) for row in rows),
            "latency_action_steps": sum(int(row["simulated_catchup_steps"]) for row in rows),
            "stale_chunks_executed": sum(bool(row["accepted"]) and int(row["measured_stale_steps"]) > 0 for row in rows),
        }
        for field, expected in counters.items():
            if episode[field] != expected:
                c.error("episodes", "%s %s mismatch" % (episode_id, field))
        ages = [
            float(row["observation_age_ms"])
            for row in rows
            if row["observation_age_ms"] is not None
        ]
        inference = [
            float(row["inference_latency_ms"])
            for row in rows
            if row["inference_latency_ms"] is not None
        ]
        derived = {
            "observation_age_p50_ms": _percentile(ages, 50),
            "observation_age_p95_ms": _percentile(ages, 95),
            "observation_age_max_ms": max(ages) if ages else None,
            "inference_latency_p50_ms": _percentile(inference, 50),
            "inference_latency_p95_ms": _percentile(inference, 95),
        }
        for field, expected in derived.items():
            if expected is None:
                if episode[field] is not None:
                    c.error("episodes", "%s %s should be blank" % (episode_id, field))
            elif episode[field] is None or not _close(episode[field], expected):
                c.error("episodes", "%s %s percentile mismatch" % (episode_id, field))
        video = episode.get("video_path")
        if episode["video_required"] and not video:
            c.error("video", "%s requires a video" % episode_id)
        if video:
            safe = _safe_path(video)
            if safe is None or not str(video).startswith("videos/"):
                c.error("video", "%s has unsafe video path" % episode_id)
            else:
                path = pathlib.Path(c._root).joinpath(*safe.parts)  # type: ignore[attr-defined]
                if not path.is_file() or path.stat().st_size == 0:
                    c.error("video", "%s video is missing or empty" % episode_id)
    c.checked("raw clocks, timing arithmetic, alignment, refresh bounds, paired jitter, episodes, pairs, matrix, and videos")


def _summary_expected(
    episodes: Sequence[Mapping[str, Any]], queries: Sequence[Mapping[str, Any]],
    warmups: Sequence[Mapping[str, Any]], protocol: _Protocol,
) -> Dict[str, Any]:
    aggregate = []
    for mode in VALID_MODES:
        mode_episodes = [row for row in episodes if row["mode"] == mode]
        mode_queries = [row for row in queries if row["mode"] == mode]
        ages = [
            float(row["observation_age_ms"])
            for row in mode_queries
            if row["observation_age_ms"] is not None
        ]
        successes = sum(bool(row["success"]) for row in mode_episodes)
        aggregate.append(
            {
                "mode": mode,
                "rollouts": len(mode_episodes),
                "successes": successes,
                "success_rate": successes / float(len(mode_episodes)) if mode_episodes else None,
                "policy_queries": len(mode_queries),
                "mean_policy_queries": (
                    sum(int(row["policy_queries"]) for row in mode_episodes)
                    / float(len(mode_episodes)) if mode_episodes else None
                ),
                "observation_age_p50_ms": _percentile(ages, 50),
                "observation_age_p95_ms": _percentile(ages, 95),
                "observation_age_max_ms": max(ages) if ages else None,
                "deadline_misses": sum(int(row["deadline_misses"]) for row in mode_episodes),
                "horizon_overruns": sum(int(row["horizon_overruns"]) for row in mode_episodes),
                "age_refreshes": sum(int(row["age_refreshes"]) for row in mode_episodes),
                "fallback_hold_steps": sum(int(row["fallback_hold_steps"]) for row in mode_episodes),
            }
        )
    pairs: Dict[str, Dict[str, bool]] = {}
    for row in episodes:
        pairs.setdefault(str(row["pair_id"]), {})[str(row["mode"])] = bool(row["success"])
    complete = [value for value in pairs.values() if set(value) == set(VALID_MODES)]
    async_successes = sum(value["async_unguarded"] for value in complete)
    aligned_successes = sum(value["latency_aligned"] for value in complete)
    paired = {
        "pairs": len(complete),
        "async_successes": async_successes,
        "aligned_successes": aligned_successes,
        "candidate_wins": sum(value["latency_aligned"] and not value["async_unguarded"] for value in complete),
        "reference_wins": sum(not value["latency_aligned"] and value["async_unguarded"] for value in complete),
        "ties": sum(value["latency_aligned"] == value["async_unguarded"] for value in complete),
        "success_rate_difference": (
            (aligned_successes - async_successes) / float(len(complete)) if complete else None
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "planned_rollouts": len(protocol.registered_cells),
        "completed_rollouts": len(episodes),
        "valid": len(episodes) == len(protocol.registered_cells),
        "complete": len(episodes) == len(protocol.registered_cells),
        "warmup_queries_planned": protocol.warmup_queries,
        "warmup_queries_completed": len(warmups),
        "warmup_queries_valid": sum(bool(row["accepted"]) for row in warmups),
        "aggregate": aggregate,
        "paired": paired,
    }


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# pi0.5-LIBERO measured-age evaluation",
        "",
        "- Schema: `%s`" % summary["schema_version"],
        "- Planned/completed rollouts: %d/%d"
        % (summary["planned_rollouts"], summary["completed_rollouts"]),
        "- Non-scoring warm-up queries: %d" % summary["warmup_queries_completed"],
        "- Complete: %s" % ("yes" if summary["complete"] else "no"),
        "",
        "| Mode | Success | Queries | Age P95 ms | Deadline misses | Horizon overruns |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["aggregate"]:
        age_p95 = "n/a" if row["observation_age_p95_ms"] is None else "%.3f" % row["observation_age_p95_ms"]
        lines.append(
            "| %s | %d/%d | %d | %s | %d | %d |"
            % (
                row["mode"], row["successes"], row["rollouts"],
                row["policy_queries"], age_p95, row["deadline_misses"],
                row["horizon_overruns"],
            )
        )
    lines.extend(
        [
            "",
            "This artifact uses blocking inference plus post-response simulator catch-up. "
            "It is not an OS hard-real-time or real-robot result.",
            "",
        ]
    )
    return "\n".join(lines)


def _compare_subset(actual: Any, expected: Any, path: str, c: _Collector) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            c.error("summary", "%s must be an object" % path)
            return
        for key, value in expected.items():
            if key not in actual:
                c.error("summary", "%s.%s is missing" % (path, key))
            else:
                _compare_subset(actual[key], value, "%s.%s" % (path, key), c)
    elif isinstance(expected, float):
        if not _close(actual, expected):
            c.error("summary", "%s mismatch" % path)
    elif actual != expected:
        c.error("summary", "%s mismatch" % path)


def _validate_derived(
    root: pathlib.Path, episodes: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]], warmups: Sequence[Mapping[str, Any]],
    protocol: _Protocol, c: _Collector,
) -> None:
    expected = _summary_expected(episodes, queries, warmups, protocol)
    summary = _read_json(root / "summary.json", "summary", c)
    if summary != expected:
        c.error("summary", "summary.json does not exactly equal raw-data recomputation")
        _compare_subset(summary, expected, "summary.json", c)
    progress = _read_json(root / "progress.json", "progress", c)
    if isinstance(progress, Mapping):
        checks = {
            "schema_version": SCHEMA_VERSION,
            "planned_rollouts": len(protocol.registered_cells),
            "completed_rollouts": len(episodes),
            "warmup_queries_required": protocol.warmup_queries,
            "warmup_queries_completed": len(warmups),
            "complete": len(episodes) == len(protocol.registered_cells),
        }
        _compare_subset(progress, checks, "progress.json", c)
    integrity = _read_json(root / "integrity.json", "integrity", c)
    if isinstance(integrity, Mapping):
        _compare_subset(
            integrity,
            {"schema_version": SCHEMA_VERSION, "valid": True, "errors": []},
            "integrity.json", c,
        )
    try:
        markdown = (root / "summary.md").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        c.error("summary", "cannot read summary.md: %s" % exc)
    else:
        expected_markdown = _summary_markdown(expected)
        if markdown != expected_markdown:
            c.error("summary", "summary.md does not equal deterministic recomputation")
    c.checked("summary, pair outcome, progress, integrity, and Markdown consistency")


def _validate_source_provenance(
    root: pathlib.Path, environment: Any, c: _Collector
) -> None:
    if not isinstance(environment, Mapping):
        c.error("provenance", "environment.json must contain an object")
        return
    recorded = environment.get("runtime_source_sha256")
    if not isinstance(recorded, Mapping):
        c.error("provenance", "environment.runtime_source_sha256 must be an object")
        return
    if set(recorded) != set(RUNTIME_SOURCE_FILES):
        c.error("provenance", "runtime source hash set does not match frozen v2 source set")
    for relative in RUNTIME_SOURCE_FILES:
        expected = recorded.get(relative)
        if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
            c.error("provenance", "invalid runtime source hash for %s" % relative)
            continue
        snapshot = root / "provenance" / "armbench_source" / pathlib.PurePosixPath(relative)
        if not snapshot.is_file():
            c.error("provenance", "missing runtime source snapshot: %s" % relative)
        elif _file_sha256(snapshot) != expected:
            c.error("provenance", "runtime source snapshot hash mismatch: %s" % relative)
    c.checked("environment source hashes close over every frozen provenance copy")


def validate_artifact(path: pathlib.Path) -> ValidationReport:
    root = pathlib.Path(path).resolve()
    c = _Collector()
    c._root = root  # type: ignore[attr-defined]
    if not root.is_dir():
        c.error("artifact", "artifact directory does not exist: %s" % root)
        return ValidationReport(str(root), False, tuple(c.errors), (), ())
    try:
        _validate_manifest(root, c)
        protocol_value = _read_json(root / "resolved_protocol.json", "protocol", c)
        environment = _read_json(root / "environment.json", "environment", c)
        protocol = _protocol(protocol_value, c)
        warmups, episodes, queries = _parse_rows(root, c)
        if isinstance(environment, Mapping) and environment.get("schema_version") != SCHEMA_VERSION:
            c.error("environment", "schema_version mismatch")
        _validate_source_provenance(root, environment, c)
        if protocol is not None:
            _validate_warmup(warmups, queries, protocol, c)
            _validate_queries(episodes, queries, protocol, c)
            _validate_derived(root, episodes, queries, warmups, protocol, c)
    except Exception as exc:  # Fail closed for malformed-input edges.
        c.error("validator", "unexpected failure: %s: %s" % (type(exc).__name__, exc))
    return ValidationReport(
        artifact=str(root), valid=not c.errors, errors=tuple(c.errors),
        warnings=tuple(c.warnings), checks=tuple(c.checks),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=pathlib.Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    report = validate_artifact(args.artifact)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print("VALID" if report.valid else "INVALID", report.artifact)
        for error in report.errors:
            print("ERROR", error)
        if report.valid:
            print("Checked: %s" % "; ".join(report.checks))
    return 0 if report.valid else 2


if __name__ == "__main__":
    sys.exit(main())
