"""Independent fail-closed validator for pi0.5-LIBERO clock artifacts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import pathlib
import re
from typing import Any, Mapping, Sequence

import numpy as np

from integrations.openpi.libero_independent_clock import (
    AGE_ALIGNED_SUFFIX,
    RESPONSE_RELATIVE_CHUNK,
    SCHEMA_VERSION,
    VALID_ACTION_SELECTION_MODES,
    canonical_action_chunk_sha256,
)
from integrations.openpi.libero_independent_clock_eval import (
    EPISODE_FIELDS,
    RUNTIME_SOURCE_FILES,
    ExperimentCell,
    aggregate_rows,
    episode_row,
    summary_markdown,
)
from integrations.openpi.libero_runtime import initial_state_digest
from integrations.openpi.libero_runtime_eval import (
    DEFAULT_POLICY_CONFIG,
    SERVER_ATTESTATION_SCHEMA_VERSION,
)
from integrations.openpi.serve_policy_attested import (
    POLICY_SAMPLING_GENERATOR,
    POLICY_SAMPLING_SCORED_NAMESPACE,
    build_policy_sampling_control,
    policy_sampling_contract,
    policy_sampling_noise,
    policy_sampling_noise_sha256,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class ValidationReport:
    artifact: str
    valid: bool
    errors: tuple[str, ...]
    checks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "armbench.pi05_libero_independent_clock_validation.v1",
            "artifact": self.artifact,
            "valid": self.valid,
            "errors": list(self.errors),
            "checks": list(self.checks),
        }


class _Collector:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.checks: list[str] = []

    def error(self, section: str, message: str) -> None:
        if len(self.errors) < 200:
            self.errors.append("[%s] %s" % (section, message))

    def checked(self, value: str) -> None:
        self.checks.append(value)


def _reject_constant(value: str) -> Any:
    raise ValueError("non-finite JSON constant: %s" % value)


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def _read_json(path: pathlib.Path, collector: _Collector) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        collector.error(path.name, "cannot read strict JSON: %s" % error)
        return None


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: Any) -> pathlib.PurePosixPath | None:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        return None
    path = pathlib.PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        return None
    return path


def _validate_manifest(root: pathlib.Path, collector: _Collector) -> None:
    manifest = _read_json(root / "manifest.json", collector)
    if not isinstance(manifest, Mapping):
        collector.error("manifest", "manifest must be an object")
        return
    if manifest.get("schema_version") != SCHEMA_VERSION:
        collector.error("manifest", "schema version mismatch")
    records = manifest.get("files")
    if not isinstance(records, Mapping):
        collector.error("manifest", "files must be an object")
        return
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json" and path.suffix != ".tmp"
    }
    if set(records) != set(actual):
        collector.error("manifest", "manifest file coverage mismatch")
    for relative, record in records.items():
        safe = _safe_relative(relative)
        if safe is None or not isinstance(record, Mapping):
            collector.error("manifest", "unsafe or invalid record: %r" % relative)
            continue
        path = actual.get(relative)
        if path is None:
            continue
        if path.is_symlink():
            collector.error("manifest", "symbolic link is not accepted: %s" % relative)
        if record.get("bytes") != path.stat().st_size:
            collector.error("manifest", "byte count mismatch: %s" % relative)
        digest = record.get("sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            collector.error("manifest", "invalid SHA-256: %s" % relative)
        elif digest != _sha256(path):
            collector.error("manifest", "SHA-256 mismatch: %s" % relative)
    collector.checked("manifest coverage, sizes, hashes, and path safety")


def _close(left: Any, right: Any, tolerance: float = 1e-8) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=tolerance, abs_tol=tolerance
        )
    except (TypeError, ValueError):
        return False


def _expected_sampling_audit(key_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": policy_sampling_contract()["schema_version"],
        "namespace": POLICY_SAMPLING_SCORED_NAMESPACE,
        "key_sha256": key_sha256,
        "noise_sha256": policy_sampling_noise_sha256(policy_sampling_noise(key_sha256)),
        "generator": POLICY_SAMPLING_GENERATOR,
    }


def _stale_prefix(age_ms: float, period_ms: float) -> int:
    if age_ms <= 1e-6:
        return 0
    return max(0, int(math.ceil((age_ms - 1e-6) / period_ms)))


def _response_relative_index(age_ms: float, period_ms: float) -> int:
    if age_ms <= 1e-6:
        return 0
    return max(0, int(math.floor((age_ms + 1e-6) / period_ms)))


def _validate_runtime(
    payload: Mapping[str, Any],
    cell: ExperimentCell,
    seed: int,
    period_ms: float,
    deadline_ms: float,
    submit_every_ticks: int,
    action_selection_mode: str,
    require_policy_input_sha256: bool,
    collector: _Collector,
) -> None:
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        collector.error(cell.episode_id, "runtime must be an object")
        return
    requests = runtime.get("requests")
    ticks = runtime.get("ticks")
    metrics = runtime.get("metrics")
    if not isinstance(requests, list) or not isinstance(ticks, list):
        collector.error(cell.episode_id, "requests and ticks must be arrays")
        return
    if not isinstance(metrics, Mapping):
        collector.error(cell.episode_id, "metrics must be an object")
        return
    if runtime.get("worker_stopped") is not True:
        collector.error(cell.episode_id, "runtime worker_stopped must be true")
    worker = runtime.get("worker")
    if not isinstance(worker, Mapping):
        collector.error(cell.episode_id, "runtime worker metrics must be an object")
    else:
        if worker.get("closed") is not True:
            collector.error(cell.episode_id, "runtime worker closed must be true")
        if worker.get("worker_alive") is not False:
            collector.error(cell.episode_id, "runtime worker_alive must be false")
        queue_dropped = worker.get("queue_dropped")
        if (
            not isinstance(queue_dropped, int)
            or isinstance(queue_dropped, bool)
            or queue_dropped != 0
        ):
            collector.error(cell.episode_id, "runtime worker queue_dropped must be zero")
    parent_pid = runtime.get("parent_process_id")
    worker_pid = runtime.get("worker_process_id")
    if (
        not isinstance(parent_pid, int)
        or not isinstance(worker_pid, int)
        or parent_pid == worker_pid
    ):
        collector.error(
            cell.episode_id, "parent and worker PIDs are not distinct integers"
        )
    if runtime.get("environment_steps") != len(ticks):
        collector.error(cell.episode_id, "environment step count does not match ticks")
    if payload.get("task_steps") != len(ticks):
        collector.error(cell.episode_id, "task step count does not match ticks")

    request_by_id = {}
    for index, request in enumerate(requests):
        if not isinstance(request, Mapping) or request.get("request_id") != index:
            collector.error(cell.episode_id, "request IDs must be contiguous")
            continue
        request_by_id[index] = request
        captured = request.get("captured_at_s")
        submitted = request.get("submitted_at_s")
        if not _close(captured, submitted):
            collector.error(
                cell.episode_id, "request capture/submission timestamps differ"
            )
        started = request.get("started_at_s")
        completed = request.get("completed_at_s")
        if started is not None and float(started) < float(submitted):
            collector.error(cell.episode_id, "request starts before submission")
        if completed is not None:
            if started is None or float(completed) < float(started):
                collector.error(
                    cell.episode_id, "request completion ordering is invalid"
                )
            expected_age = (float(completed) - float(captured)) * 1000.0
            if not _close(request.get("response_age_ms"), expected_age, 1e-7):
                collector.error(cell.episode_id, "request response age mismatch")

        actions = request.get("actions")
        metadata = request.get("response_metadata")
        if actions is None:
            if completed is not None and request.get("failure_type") is None:
                collector.error(cell.episode_id, "successful completion lacks actions")
            continue
        try:
            action_array = np.asarray(actions, dtype=np.float64)
        except (TypeError, ValueError):
            collector.error(cell.episode_id, "request actions are not numeric")
            continue
        if action_array.shape != (10, 7) or not np.all(np.isfinite(action_array)):
            collector.error(
                cell.episode_id, "request actions do not have finite (10, 7) shape"
            )
            continue
        if not isinstance(metadata, Mapping):
            collector.error(cell.episode_id, "completed actions lack response metadata")
            continue
        if metadata.get("action_chunk_sha256") != canonical_action_chunk_sha256(
            action_array
        ):
            collector.error(cell.episode_id, "action chunk SHA-256 mismatch")
        if require_policy_input_sha256:
            policy_input_sha256 = metadata.get("policy_input_sha256")
            if not isinstance(policy_input_sha256, str) or not _SHA256.fullmatch(
                policy_input_sha256
            ):
                collector.error(
                    cell.episode_id, "policy input SHA-256 is missing or invalid"
                )
        sequence_id = request.get("observation_sequence_id")
        control = build_policy_sampling_control(
            POLICY_SAMPLING_SCORED_NAMESPACE,
            seed,
            (cell.task_suite, cell.task_id, cell.episode_index, submit_every_ticks),
            int(sequence_id),
        )
        expected_audit = _expected_sampling_audit(str(control["key_sha256"]))
        if metadata.get("policy_sampling") != expected_audit:
            collector.error(cell.episode_id, "policy sampling audit mismatch")

    expected_metrics = {
        "submitted": len(requests),
        "started": sum(request.get("started_at_s") is not None for request in requests),
        "completed": sum(
            request.get("completed_at_s") is not None for request in requests
        ),
        "superseded": sum(
            request.get("superseded_at_s") is not None for request in requests
        ),
        "holds": sum(
            tick.get("status") == "hold" for tick in ticks if isinstance(tick, Mapping)
        ),
        "executes": sum(
            tick.get("status") == "execute"
            for tick in ticks
            if isinstance(tick, Mapping)
        ),
    }
    if dict(metrics) != expected_metrics:
        collector.error(cell.episode_id, "runtime metrics do not recompute")

    previous_action = np.asarray([0.0] * 6 + [-1.0], dtype=np.float64)
    done_indices = []
    for index, tick in enumerate(ticks):
        if not isinstance(tick, Mapping) or tick.get("tick_index") != index:
            collector.error(cell.episode_id, "tick IDs must be contiguous")
            continue
        if tick.get("observation_sequence_id") != index:
            collector.error(cell.episode_id, "tick observation sequence mismatch")
        try:
            action = np.asarray(tick.get("action"), dtype=np.float64)
        except (TypeError, ValueError):
            collector.error(cell.episode_id, "tick action is not numeric")
            continue
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            collector.error(cell.episode_id, "tick action is not finite shape (7,)")
            continue
        status = tick.get("status")
        if status == "hold":
            if not np.allclose(action[:6], 0.0) or not _close(
                action[-1], previous_action[-1]
            ):
                collector.error(
                    cell.episode_id, "hold does not preserve gripper and stop motion"
                )
        elif status == "execute":
            response_id = tick.get("response_request_id")
            request = request_by_id.get(response_id)
            if request is None or request.get("actions") is None:
                collector.error(
                    cell.episode_id, "execute tick references no action chunk"
                )
            else:
                age = float(tick.get("response_age_ms"))
                prefix = _stale_prefix(age, period_ms)
                suffix = max(0, 10 - prefix)
                if age > deadline_ms + 1e-6:
                    collector.error(
                        cell.episode_id, "execute tick violates deadline rule"
                    )
                if action_selection_mode == AGE_ALIGNED_SUFFIX:
                    expected_index = prefix
                    if suffix <= 0:
                        collector.error(
                            cell.episode_id,
                            "age-aligned execute tick has no fresh suffix",
                        )
                    expected_reason = "fresh_suffix_available"
                elif action_selection_mode == RESPONSE_RELATIVE_CHUNK:
                    completed_at = float(request["completed_at_s"])
                    captured_at = float(request["captured_at_s"])
                    decision_at = captured_at + age / 1000.0
                    expected_index = _response_relative_index(
                        max(0.0, decision_at - completed_at) * 1000.0,
                        period_ms,
                    )
                    if expected_index >= 10:
                        collector.error(
                            cell.episode_id,
                            "response-relative execute tick uses an exhausted chunk",
                        )
                    expected_reason = "response_relative_chunk_available"
                else:
                    expected_index = -1
                    expected_reason = ""
                    collector.error(cell.episode_id, "unknown action selection mode")
                if tick.get("action_index") != expected_index:
                    collector.error(cell.episode_id, "execute action index mismatch")
                if (
                    tick.get("stale_prefix_steps") != prefix
                    or tick.get("stale_suffix_steps") != suffix
                ):
                    collector.error(
                        cell.episode_id, "execute stale prefix/suffix mismatch"
                    )
                if tick.get("reason") != expected_reason:
                    collector.error(
                        cell.episode_id, "execute reason does not match selection mode"
                    )
                if 0 <= expected_index < 10:
                    expected_action = np.asarray(
                        request["actions"][expected_index], dtype=np.float64
                    )
                    if not np.allclose(action, expected_action, rtol=0.0, atol=1e-12):
                        collector.error(
                            cell.episode_id,
                            "executed action differs from recorded chunk",
                        )
        else:
            collector.error(cell.episode_id, "unknown tick status")
        previous_action = action
        if tick.get("environment_done") is True:
            done_indices.append(index)
    if done_indices and done_indices != [len(ticks) - 1]:
        collector.error(
            cell.episode_id, "environment done must occur only on the final tick"
        )
    expected_success = bool(done_indices)
    if payload.get("task_success") is not expected_success:
        collector.error(cell.episode_id, "task success does not match environment done")
    expected_reason = (
        "task_success" if expected_success else runtime.get("termination_reason")
    )
    if payload.get("termination_reason") != expected_reason:
        collector.error(cell.episode_id, "termination reason mismatch")


def _validate_provenance(
    root: pathlib.Path,
    protocol: Mapping[str, Any],
    environment: Mapping[str, Any],
    collector: _Collector,
) -> None:
    if environment.get("schema_version") != SCHEMA_VERSION:
        collector.error("provenance", "environment schema mismatch")
    if environment.get("openpi_git_commit") != protocol.get("openpi_commit"):
        collector.error("provenance", "OpenPI commit mismatch")
    hashes = environment.get("runtime_source_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(RUNTIME_SOURCE_FILES):
        collector.error("provenance", "runtime source hash coverage mismatch")
    else:
        for relative in RUNTIME_SOURCE_FILES:
            snapshot = (
                root
                / "provenance"
                / "armbench_source"
                / pathlib.PurePosixPath(relative)
            )
            digest = hashes.get(relative)
            if not snapshot.is_file() or digest != _sha256(snapshot):
                collector.error("provenance", "source snapshot mismatch: %s" % relative)
    metadata = environment.get("server_metadata")
    attestation = (
        metadata.get("armbench_server_attestation")
        if isinstance(metadata, Mapping)
        else None
    )
    if not isinstance(attestation, Mapping):
        collector.error("attestation", "server attestation is missing")
    else:
        expected = {
            "schema_version": SERVER_ATTESTATION_SCHEMA_VERSION,
            "policy_loaded": True,
            "policy_config": DEFAULT_POLICY_CONFIG,
            "checkpoint_uri": protocol.get("checkpoint"),
            "openpi_commit": protocol.get("openpi_commit"),
            "openpi_tracked_clean": True,
            "openpi_submodules_clean": True,
            "action_horizon": 10,
        }
        for key, value in expected.items():
            if attestation.get(key) != value:
                collector.error("attestation", "%s mismatch" % key)
        for key in ("checkpoint_content_sha256", "server_source_sha256"):
            if not isinstance(attestation.get(key), str) or not _SHA256.fullmatch(
                attestation[key]
            ):
                collector.error("attestation", "%s is not a SHA-256" % key)
    sampling = (
        metadata.get("armbench_policy_sampling_contract")
        if isinstance(metadata, Mapping)
        else None
    )
    if sampling != policy_sampling_contract():
        collector.error("attestation", "policy sampling server contract mismatch")
    collector.checked("checkpoint, sampling, commit, and runtime source provenance")


def _read_csv(path: pathlib.Path, collector: _Collector) -> list[dict[str, str]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != EPISODE_FIELDS:
                collector.error("CSV", "per_episode.csv header mismatch")
            return list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        collector.error("CSV", "cannot read per_episode.csv: %s" % error)
        return []


def validate_artifact(path: pathlib.Path) -> ValidationReport:
    root = path.resolve()
    collector = _Collector()
    if not root.is_dir():
        collector.error("artifact", "directory does not exist: %s" % root)
        return ValidationReport(str(root), False, tuple(collector.errors), ())
    try:
        _validate_manifest(root, collector)
        protocol = _read_json(root / "resolved_protocol.json", collector)
        environment = _read_json(root / "environment.json", collector)
        rows = _read_json(root / "per_episode.json", collector)
        if not isinstance(protocol, Mapping) or not isinstance(environment, Mapping):
            collector.error("artifact", "protocol and environment must be objects")
            return ValidationReport(
                str(root), False, tuple(collector.errors), tuple(collector.checks)
            )
        if protocol.get("schema_version") != SCHEMA_VERSION:
            collector.error("protocol", "schema mismatch")
        matrix = protocol.get("matrix")
        cell_values = matrix.get("cells") if isinstance(matrix, Mapping) else None
        if not isinstance(cell_values, list):
            collector.error("matrix", "matrix cells must be an array")
            cell_values = []
        cells = [
            ExperimentCell(
                str(value["task_suite"]),
                int(value["task_id"]),
                int(value["episode_index"]),
            )
            for value in cell_values
            if isinstance(value, Mapping)
        ]
        if any(cell.to_dict() != value for cell, value in zip(cells, cell_values)):
            collector.error("matrix", "matrix cell IDs are not canonical")
        if not isinstance(rows, list) or len(rows) != len(cells):
            collector.error("matrix", "episode row count does not match matrix")
            rows = [] if not isinstance(rows, list) else rows
        _validate_provenance(root, protocol, environment, collector)
        runtime_config = protocol.get("runtime")
        sampling_config = protocol.get("policy_sampling")
        if not isinstance(runtime_config, Mapping) or not isinstance(
            sampling_config, Mapping
        ):
            collector.error("protocol", "runtime and sampling configs must be objects")
            runtime_config = {}
            sampling_config = {}
        seed = int(sampling_config.get("seed", -1))
        period_ms = float(runtime_config.get("control_period_ms", float("nan")))
        deadline_ms = float(runtime_config.get("deadline_ms", float("nan")))
        submit_every = int(runtime_config.get("submit_every_ticks", -1))
        action_selection_mode = str(
            runtime_config.get("action_selection_mode", AGE_ALIGNED_SUFFIX)
        )
        if action_selection_mode not in VALID_ACTION_SELECTION_MODES:
            collector.error("protocol", "unknown action selection mode")
        require_policy_input_sha256 = (
            runtime_config.get("policy_input_audit")
            == "canonical_pi05_libero_request_sha256_v1"
        )
        recomputed_rows = []
        for cell, recorded_row in zip(cells, rows):
            episode_directory = root / "episodes" / cell.episode_id
            payload = _read_json(episode_directory / "runtime.json", collector)
            try:
                initial_state = np.load(
                    episode_directory / "initial_state.npy", allow_pickle=False
                )
            except (OSError, ValueError) as error:
                collector.error(
                    cell.episode_id, "cannot load initial state: %s" % error
                )
                continue
            if not isinstance(payload, Mapping) or not isinstance(
                recorded_row, Mapping
            ):
                collector.error(cell.episode_id, "runtime or episode row is invalid")
                continue
            if recorded_row.get("worker_stopped") is not True:
                collector.error(cell.episode_id, "per-episode worker_stopped must be true")
            digest = initial_state_digest(initial_state)
            if payload.get("initial_state_sha256") != digest:
                collector.error(cell.episode_id, "initial state digest mismatch")
            _validate_runtime(
                payload,
                cell,
                seed,
                period_ms,
                deadline_ms,
                submit_every,
                action_selection_mode,
                require_policy_input_sha256,
                collector,
            )
            video_path = recorded_row.get("video_path")
            if video_path:
                safe = _safe_relative(video_path)
                if safe is None or not (root / safe).is_file():
                    collector.error(cell.episode_id, "video path is unsafe or missing")
            expected_row = episode_row(
                cell,
                str(recorded_row.get("task_description", "")),
                seed,
                payload,
                float(recorded_row.get("wall_time_s", 0.0)),
                video_path,
            )
            if expected_row != dict(recorded_row):
                collector.error(cell.episode_id, "per-episode row does not recompute")
            recomputed_rows.append(expected_row)
        collector.checked(
            "initial states, request lifecycles, action chunks, ticks, and outcomes"
        )

        expected_aggregate = aggregate_rows(recomputed_rows, len(cells))
        if _read_json(root / "aggregate.json", collector) != expected_aggregate:
            collector.error("aggregate", "aggregate.json does not recompute")
        try:
            actual_summary = (root / "summary.md").read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            collector.error("summary", "cannot read summary: %s" % error)
        else:
            if actual_summary != summary_markdown(recomputed_rows, expected_aggregate):
                collector.error("summary", "summary does not recompute")
        csv_rows = _read_csv(root / "per_episode.csv", collector)
        if len(csv_rows) != len(recomputed_rows):
            collector.error("CSV", "row count mismatch")
        for csv_row, expected in zip(csv_rows, recomputed_rows):
            for field in EPISODE_FIELDS:
                expected_text = "" if expected[field] is None else str(expected[field])
                if csv_row.get(field) != expected_text:
                    collector.error("CSV", "%s mismatch" % field)
                    break
        progress = _read_json(root / "progress.json", collector)
        if not isinstance(progress, Mapping) or (
            progress.get("planned_rollouts") != len(cells)
            or progress.get("completed_rollouts") != len(recomputed_rows)
            or progress.get("complete") is not (len(recomputed_rows) == len(cells))
        ):
            collector.error("progress", "progress does not match raw episodes")
        integrity = _read_json(root / "integrity.json", collector)
        if (
            not isinstance(integrity, Mapping)
            or integrity.get("valid") is not True
            or integrity.get("errors") != []
        ):
            collector.error(
                "integrity", "artifact producer did not report a valid complete run"
            )
        if (root / "run_error.json").exists():
            collector.error("integrity", "run_error.json is present")
        if any(int(row["ticks_during_inference"]) <= 0 for row in recomputed_rows):
            collector.error(
                "overlap", "an episode has no control tick during inference"
            )
        collector.checked(
            "derived rows, aggregate, summary, progress, and overlap proof"
        )
    except Exception as error:
        collector.error(
            "validator", "unexpected failure: %s: %s" % (type(error).__name__, error)
        )
    return ValidationReport(
        str(root),
        not collector.errors,
        tuple(collector.errors),
        tuple(collector.checks),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=pathlib.Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
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
    raise SystemExit(main())
