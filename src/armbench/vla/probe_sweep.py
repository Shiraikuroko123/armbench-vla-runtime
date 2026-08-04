"""Bounded collection of exact recorded requests from one OpenPI server."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Sequence

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
