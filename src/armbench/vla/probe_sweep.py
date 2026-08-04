"""Bounded collection of exact recorded requests from one OpenPI server."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

from armbench.benchmark import environment_metadata
from armbench.vla.benchmark import _write_csv, _write_json
from armbench.vla.replay_probe import (
    execute_recorded_openpi_probe,
    validate_recorded_openpi_probe,
)
from armbench.vla.request_replay import load_recorded_openpi_request


RECORDED_OPENPI_PROBE_SWEEP_ARTIFACT_TYPE = (
    "armbench_recorded_openpi_probe_sweep_v1"
)


class RecordedProbeSweepValidationError(ValueError):
    """Raised when a recorded probe sweep artifact is inconsistent."""


@dataclass(frozen=True)
class RecordedProbeSweepValidationResult:
    directory: str
    planned_queries: int
    successful_queries: int
    failed_queries: int
    sweep_complete: bool
    successful_action_sha256: tuple[str, ...]
    artifact_sha256: str
    checks: tuple[str, ...]

    def metrics(self) -> dict[str, object]:
        return {
            "directory": self.directory,
            "planned_queries": self.planned_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "sweep_complete": self.sweep_complete,
            "successful_action_sha256": list(self.successful_action_sha256),
            "artifact_sha256": self.artifact_sha256,
            "checks": list(self.checks),
            "valid": True,
        }


def _validation_require(condition: bool, message: str) -> None:
    if not condition:
        raise RecordedProbeSweepValidationError(message)


def _valid_sha256(value: object, label: str) -> str:
    _validation_require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"invalid {label}",
    )
    return value


def _csv_bool(value: object, label: str) -> bool:
    if value in (True, "True", "true", "1"):
        return True
    if value in (False, "False", "false", "0"):
        return False
    raise RecordedProbeSweepValidationError(f"invalid {label}")


def _bounded_error_message(error: Exception, limit: int = 240) -> str:
    message = " ".join(str(error).split())
    if not message:
        message = error.__class__.__name__
    return message[:limit]


def _manifest(
    source_artifact: Path,
    query_indices: list[int],
    request_hashes: list[str],
    rows: list[dict[str, object]],
    *,
    server: str,
    policy_provenance: str,
    scenario: str | None,
    payload_mass: float | None,
    execution_horizon: int | None,
) -> dict[str, object]:
    successful = sum(row["status"] == "success" for row in rows)
    failed = len(rows) - successful
    return {
        "artifact_type": RECORDED_OPENPI_PROBE_SWEEP_ARTIFACT_TYPE,
        "source_artifact": str(source_artifact.resolve()),
        "server": server,
        "policy_provenance": policy_provenance,
        "selection": {
            "scenario": scenario,
            "payload_mass": payload_mass,
            "execution_horizon": execution_horizon,
            "query_indices": query_indices,
            "request_payload_sha256": request_hashes,
        },
        "attempted_queries": len(rows),
        "successful_queries": successful,
        "failed_queries": failed,
        "planned_queries": len(query_indices),
        "sweep_complete": len(rows) == len(query_indices) and failed == 0,
        "remote_policy_responses_validated": successful,
        "checkpoint_identity_verified": False,
        "physics_executed": False,
        "physical_safe": None,
    }


def _summary(manifest: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Recorded OpenPI probe sweep",
            "",
            f"- Server: `{manifest['server']}`",
            f"- Policy provenance: `{manifest['policy_provenance']}`",
            f"- Planned queries: `{manifest['planned_queries']}`",
            f"- Successful queries: `{manifest['successful_queries']}`",
            f"- Failed queries: `{manifest['failed_queries']}`",
            f"- Sweep complete: `{str(manifest['sweep_complete']).lower()}`",
            "- Checkpoint identity verified by protocol: `false`",
            "- Physics executed: `false`",
            "",
            "Each query uses a separately validated recorded DROID request and "
            "an isolated client connection. Successful children are fixed-input "
            "inference probes, not task rollouts or physical-safety results.",
            "",
            "Policy provenance is user-supplied. Preserve the server launch log "
            "and checkpoint checksum before naming a model in results.",
            "",
        ]
    )


def validate_recorded_openpi_probe_sweep(
    directory: Path,
) -> RecordedProbeSweepValidationResult:
    """Cross-check sweep accounting, failures, and successful child probes."""

    root = directory.resolve()
    _validation_require(
        root.is_dir(), f"artifact directory does not exist: {root}"
    )
    manifest_path = root / "manifest.json"
    csv_path = root / "per_query.csv"
    environment_path = root / "environment.json"
    summary_path = root / "summary.md"
    log_path = root / "sweep.log"
    for path in (
        manifest_path,
        csv_path,
        environment_path,
        summary_path,
        log_path,
    ):
        _validation_require(
            path.is_file() and path.stat().st_size > 0,
            f"missing file: {path}",
        )
    try:
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
        environment_value = json.loads(
            environment_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RecordedProbeSweepValidationError(
            "probe sweep contains invalid JSON"
        ) from error
    _validation_require(
        isinstance(manifest_value, dict), "sweep manifest must be a mapping"
    )
    _validation_require(
        isinstance(environment_value, dict),
        "sweep environment must be a mapping",
    )
    manifest = dict(manifest_value)
    environment = dict(environment_value)
    _validation_require(
        manifest.get("artifact_type")
        == RECORDED_OPENPI_PROBE_SWEEP_ARTIFACT_TYPE,
        "unexpected probe sweep artifact type",
    )
    _validation_require(
        manifest.get("checkpoint_identity_verified") is False,
        "probe sweep must not claim checkpoint attestation",
    )
    _validation_require(
        manifest.get("physics_executed") is False,
        "probe sweep must not claim physics execution",
    )
    _validation_require(
        manifest.get("physical_safe") is None,
        "probe sweep must not claim a physical-safety outcome",
    )
    for name in ("server", "policy_provenance"):
        value = manifest.get(name)
        _validation_require(
            isinstance(value, str) and bool(value.strip()),
            f"sweep {name} must be nonempty",
        )
    selection = manifest.get("selection")
    _validation_require(
        isinstance(selection, dict), "sweep selection must be a mapping"
    )
    query_values = selection.get("query_indices")
    request_values = selection.get("request_payload_sha256")
    _validation_require(
        isinstance(query_values, list) and bool(query_values),
        "sweep query selection must be a nonempty list",
    )
    _validation_require(
        all(isinstance(value, int) and value >= 0 for value in query_values)
        and len(query_values) == len(set(query_values)),
        "sweep query indices must be unique and nonnegative",
    )
    _validation_require(
        isinstance(request_values, list)
        and len(request_values) == len(query_values),
        "sweep request hash count mismatch",
    )
    request_hashes = [
        _valid_sha256(value, f"request hash at index {index}")
        for index, value in enumerate(request_values)
    ]
    planned_queries = len(query_values)
    _validation_require(
        manifest.get("planned_queries") == planned_queries,
        "planned query count mismatch",
    )

    try:
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as error:
        raise RecordedProbeSweepValidationError(
            f"invalid per-query CSV: {csv_path}"
        ) from error
    expected_fields = {
        "query_index",
        "request_payload_sha256",
        "artifact_directory",
        "status",
        "probe_validated",
        "action_sha256",
        "client_inference_latency_ms",
        "guard_safe_after",
        "probe_elapsed_ms",
        "failure_type",
        "failure_message",
    }
    _validation_require(len(rows) == planned_queries, "per-query row count mismatch")
    _validation_require(
        bool(rows) and set(rows[0]) == expected_fields,
        "per-query CSV fields mismatch",
    )
    successful = 0
    failed = 0
    action_hashes: list[str] = []
    probe_root = (root / "probes").resolve()
    for index, row in enumerate(rows):
        try:
            query_index = int(row["query_index"])
            elapsed_ms = float(row["probe_elapsed_ms"])
        except (KeyError, TypeError, ValueError) as error:
            raise RecordedProbeSweepValidationError(
                f"invalid numeric field at query row {index}"
            ) from error
        _validation_require(
            query_index == query_values[index],
            f"query index mismatch at row {index}",
        )
        _validation_require(
            row.get("request_payload_sha256") == request_hashes[index],
            f"request hash mismatch at row {index}",
        )
        _validation_require(
            np.isfinite(elapsed_ms) and elapsed_ms >= 0.0,
            f"invalid probe elapsed time at row {index}",
        )
        child_value = row.get("artifact_directory")
        _validation_require(
            isinstance(child_value, str) and bool(child_value),
            f"missing probe child at row {index}",
        )
        child_directory = (root / child_value).resolve()
        _validation_require(
            child_directory.is_relative_to(probe_root),
            f"probe child escapes output directory at row {index}",
        )
        status = row.get("status")
        if status == "success":
            successful += 1
            _validation_require(
                _csv_bool(
                    row.get("probe_validated"),
                    f"probe validation flag at row {index}",
                ),
                f"successful probe is not validated at row {index}",
            )
            _validation_require(
                not row.get("failure_type") and not row.get("failure_message"),
                f"successful probe contains failure fields at row {index}",
            )
            validation = validate_recorded_openpi_probe(child_directory)
            _validation_require(
                validation.request_payload_sha256 == request_hashes[index],
                f"child request hash mismatch at row {index}",
            )
            _validation_require(
                row.get("action_sha256") == validation.action_sha256,
                f"child action hash mismatch at row {index}",
            )
            try:
                latency_ms = float(row["client_inference_latency_ms"])
            except (KeyError, TypeError, ValueError) as error:
                raise RecordedProbeSweepValidationError(
                    f"invalid client latency at row {index}"
                ) from error
            _validation_require(
                np.isfinite(latency_ms) and latency_ms >= 0.0,
                f"invalid client latency at row {index}",
            )
            _validation_require(
                _csv_bool(
                    row.get("guard_safe_after"),
                    f"guard safety at row {index}",
                )
                is validation.guard_safe_after,
                f"child guard safety mismatch at row {index}",
            )
            action_hashes.append(validation.action_sha256)
        elif status == "failure":
            failed += 1
            _validation_require(
                not _csv_bool(
                    row.get("probe_validated"),
                    f"probe validation flag at row {index}",
                ),
                f"failed probe is marked validated at row {index}",
            )
            _validation_require(
                not row.get("action_sha256")
                and not row.get("client_inference_latency_ms")
                and not row.get("guard_safe_after"),
                f"failed probe contains successful response fields at row {index}",
            )
            _validation_require(
                bool(row.get("failure_type"))
                and bool(row.get("failure_message")),
                f"failed probe lacks root-cause fields at row {index}",
            )
            if child_directory.exists():
                try:
                    validate_recorded_openpi_probe(child_directory)
                except ValueError:
                    pass
                else:
                    raise RecordedProbeSweepValidationError(
                        f"failed row {index} points to a valid probe artifact"
                    )
        else:
            raise RecordedProbeSweepValidationError(
                f"invalid status at query row {index}: {status!r}"
            )

    expected_complete = failed == 0
    expected_counts: dict[str, object] = {
        "attempted_queries": planned_queries,
        "successful_queries": successful,
        "failed_queries": failed,
        "remote_policy_responses_validated": successful,
        "sweep_complete": expected_complete,
    }
    for name, expected in expected_counts.items():
        _validation_require(
            manifest.get(name) == expected,
            f"sweep manifest mismatch: {name}",
        )
    probe_environment = environment.get("recorded_openpi_probe_sweep")
    _validation_require(
        isinstance(probe_environment, dict),
        "sweep environment metadata is missing",
    )
    expected_environment: dict[str, object] = {
        "artifact_type": RECORDED_OPENPI_PROBE_SWEEP_ARTIFACT_TYPE,
        "server": manifest["server"],
        "policy_provenance": manifest["policy_provenance"],
        "planned_queries": planned_queries,
        "successful_queries": successful,
        "manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "per_query_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "checkpoint_identity_verified": False,
        "physics_executed": False,
    }
    for name, expected in expected_environment.items():
        _validation_require(
            probe_environment.get(name) == expected,
            f"sweep environment mismatch: {name}",
        )
    summary = summary_path.read_text(encoding="utf-8")
    _validation_require(
        f"Successful queries: `{successful}`" in summary
        and f"Failed queries: `{failed}`" in summary,
        "sweep summary counts mismatch",
    )
    _validation_require(
        "Checkpoint identity verified by protocol: `false`" in summary,
        "sweep summary checkpoint claim mismatch",
    )
    _validation_require(
        "Physics executed: `false`" in summary,
        "sweep summary physics claim mismatch",
    )
    log = log_path.read_text(encoding="utf-8")
    for query_index in query_values:
        _validation_require(
            f"query {query_index} " in log,
            f"sweep log is missing query {query_index}",
        )
    artifact_hash = hashlib.sha256()
    for path in (
        manifest_path,
        csv_path,
        environment_path,
        summary_path,
        log_path,
    ):
        artifact_hash.update(path.name.encode("utf-8"))
        artifact_hash.update(path.read_bytes())
    for action_hash in action_hashes:
        artifact_hash.update(action_hash.encode("ascii"))
    return RecordedProbeSweepValidationResult(
        directory=str(root),
        planned_queries=planned_queries,
        successful_queries=successful,
        failed_queries=failed,
        sweep_complete=expected_complete,
        successful_action_sha256=tuple(action_hashes),
        artifact_sha256=artifact_hash.hexdigest(),
        checks=(
            "claim_boundaries",
            "selection_contract",
            "query_accounting",
            "successful_children",
            "failure_root_causes",
            "environment_hashes",
            "summary_counts",
            "sweep_log",
        ),
    )


def execute_recorded_openpi_probe_sweep(
    config_path: Path,
    artifact_directory: Path,
    output_directory: Path,
    *,
    host: str,
    port: int,
    query_indices: Sequence[int],
    scenario: str | None = None,
    payload_mass: float | None = None,
    execution_horizon: int | None = None,
    api_key: str | None = None,
    connect_timeout_s: float = 3.0,
    inference_timeout_s: float = 1.0,
    policy_provenance: str = "remote_server_unverified",
) -> Path:
    """Collect a bounded cohort while preserving per-request failures."""

    if output_directory.exists():
        raise FileExistsError(
            f"output directory already exists: {output_directory}"
        )
    selected_queries = [int(value) for value in query_indices]
    if (
        not selected_queries
        or any(value < 0 for value in selected_queries)
        or len(selected_queries) != len(set(selected_queries))
    ):
        raise ValueError("query indices must be unique and nonnegative")
    if not policy_provenance.strip():
        raise ValueError("policy_provenance must be nonempty")
    requests = [
        load_recorded_openpi_request(
            artifact_directory,
            query_index=query_index,
            scenario=scenario,
            payload_mass=payload_mass,
            execution_horizon=execution_horizon,
        )
        for query_index in selected_queries
    ]
    request_hashes = [request.packed_payload_sha256 for request in requests]
    server = f"{host}:{port}"
    output_directory.mkdir(parents=True, exist_ok=False)
    probe_root = output_directory / "probes"
    probe_root.mkdir()
    rows: list[dict[str, object]] = []
    log_lines: list[str] = []
    for query_index, request_hash in zip(
        selected_queries, request_hashes, strict=True
    ):
        child_name = f"query_{query_index:04d}_{request_hash[:12]}"
        child_directory = probe_root / child_name
        started = perf_counter()
        row: dict[str, object] = {
            "query_index": query_index,
            "request_payload_sha256": request_hash,
            "artifact_directory": f"probes/{child_name}",
            "status": "failure",
            "probe_validated": False,
            "action_sha256": "",
            "client_inference_latency_ms": "",
            "guard_safe_after": "",
            "probe_elapsed_ms": 0.0,
            "failure_type": "",
            "failure_message": "",
        }
        log_lines.append(f"starting query {query_index} {request_hash}")
        try:
            execute_recorded_openpi_probe(
                config_path,
                artifact_directory,
                child_directory,
                host=host,
                port=port,
                query_index=query_index,
                scenario=scenario,
                payload_mass=payload_mass,
                execution_horizon=execution_horizon,
                api_key=api_key,
                connect_timeout_s=connect_timeout_s,
                inference_timeout_s=inference_timeout_s,
                policy_provenance=policy_provenance,
            )
            validation = validate_recorded_openpi_probe(child_directory)
            response_value = json.loads(
                (child_directory / "response.json").read_text(encoding="utf-8")
            )
            if not isinstance(response_value, dict):
                raise ValueError("probe response must be a JSON mapping")
            if validation.request_payload_sha256 != request_hash:
                raise ValueError("probe request hash changed after preflight")
            row.update(
                {
                    "status": "success",
                    "probe_validated": True,
                    "action_sha256": validation.action_sha256,
                    "client_inference_latency_ms": float(
                        response_value["client_inference_latency_ms"]
                    ),
                    "guard_safe_after": validation.guard_safe_after,
                }
            )
            log_lines.append(
                f"completed query {query_index} action "
                f"{validation.action_sha256}"
            )
        except Exception as error:
            row["failure_type"] = error.__class__.__name__
            row["failure_message"] = _bounded_error_message(error)
            log_lines.append(
                f"failed query {query_index} {error.__class__.__name__}: "
                f"{row['failure_message']}"
            )
        row["probe_elapsed_ms"] = (perf_counter() - started) * 1000.0
        rows.append(row)
        current_manifest = _manifest(
            artifact_directory,
            selected_queries,
            request_hashes,
            rows,
            server=server,
            policy_provenance=policy_provenance,
            scenario=scenario,
            payload_mass=payload_mass,
            execution_horizon=execution_horizon,
        )
        _write_csv(output_directory / "per_query.csv", rows)
        _write_json(output_directory / "manifest.json", current_manifest)
        (output_directory / "sweep.log").write_text(
            "\n".join(log_lines) + "\n", encoding="utf-8", newline="\n"
        )

    manifest = _manifest(
        artifact_directory,
        selected_queries,
        request_hashes,
        rows,
        server=server,
        policy_provenance=policy_provenance,
        scenario=scenario,
        payload_mass=payload_mass,
        execution_horizon=execution_horizon,
    )
    metadata = environment_metadata(Path(__file__).resolve().parents[3])
    metadata["recorded_openpi_probe_sweep"] = {
        "artifact_type": RECORDED_OPENPI_PROBE_SWEEP_ARTIFACT_TYPE,
        "server": server,
        "policy_provenance": policy_provenance,
        "planned_queries": len(selected_queries),
        "successful_queries": manifest["successful_queries"],
        "manifest_sha256": hashlib.sha256(
            (output_directory / "manifest.json").read_bytes()
        ).hexdigest(),
        "per_query_csv_sha256": hashlib.sha256(
            (output_directory / "per_query.csv").read_bytes()
        ).hexdigest(),
        "checkpoint_identity_verified": False,
        "physics_executed": False,
    }
    _write_json(output_directory / "environment.json", metadata)
    (output_directory / "summary.md").write_text(
        _summary(manifest), encoding="utf-8", newline="\n"
    )
    return output_directory
