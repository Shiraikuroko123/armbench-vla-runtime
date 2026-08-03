"""Reproducible OpenPI loopback response/transport fault matrix."""

from __future__ import annotations

from collections.abc import Sequence
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from armbench.vla.artifact import validate_online_artifact
from armbench.vla.benchmark import _write_csv, _write_json
from armbench.vla.loopback import (
    LOOPBACK_FAULT_MODES,
    execute_openpi_loopback_run,
)


def _json_mapping(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    return dict(value)


def _aggregate_row(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"expected exactly one aggregate row: {path}")
    if not isinstance(value[0], dict):
        raise ValueError(f"aggregate row is not a mapping: {path}")
    return dict(value[0])


def _first_chunk(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"fault-matrix case must have one chunk row: {path}")
    return rows[0]


def _case_passed(
    fault_mode: str,
    row: dict[str, object],
    audit: dict[str, object],
    *,
    request_hashes_match: bool,
) -> bool:
    safety_violations = sum(
        int(row[field])
        for field in (
            "obstacle_contact_steps",
            "self_contact_steps",
            "joint_limit_violation_steps",
        )
    )
    common = (
        int(row["policy_queries"]) == 1
        and int(audit["request_count"]) == 1
        and bool(row["physical_safe"])
        and safety_violations == 0
        and request_hashes_match
    )
    if fault_mode == "none":
        return bool(
            common
            and bool(row["remote_policy_response_validated"])
            and int(row["validated_remote_chunks"]) == 1
            and int(row["runtime_fallback_chunks"]) == 0
            and int(audit["fault_injected_count"]) == 0
        )
    return bool(
        common
        and not bool(row["remote_policy_response_validated"])
        and int(row["validated_remote_chunks"]) == 0
        and int(row["runtime_fallback_chunks"]) == 1
        and row["termination_reason"] == "runtime_fallback:policy_inference"
        and int(audit["fault_injected_count"]) == 1
    )


def _write_overview(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["fault_mode"]).replace("_", "\n") for row in rows]
    positions = np.arange(len(rows), dtype=float)
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    axes[0].bar(
        positions - width / 2,
        [int(row["validated_remote_chunks"]) for row in rows],
        width,
        label="Validated remote chunks",
        color="#27864f",
    )
    axes[0].bar(
        positions + width / 2,
        [int(row["runtime_fallback_chunks"]) for row in rows],
        width,
        label="Runtime fallback chunks",
        color="#bf4a3f",
    )
    axes[0].set_xticks(positions, labels)
    axes[0].set_yticks([0, 1])
    axes[0].set_ylim(0.0, 1.25)
    axes[0].set_title("Response authority")
    axes[0].set_ylabel("Chunk count")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].bar(
        positions - width / 2,
        [float(row["p95_client_inference_latency_ms"]) for row in rows],
        width,
        label="Client call",
        color="#3677a8",
    )
    axes[1].bar(
        positions + width / 2,
        [float(row["p95_policy_latency_ms"]) for row in rows],
        width,
        label="Capture-to-dispatch",
        color="#d29b31",
    )
    axes[1].set_xticks(positions, labels)
    axes[1].set_title("Observed latency")
    axes[1].set_ylabel("Milliseconds")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)

    fault_rows = [row for row in rows if row["fault_mode"] != "none"]
    passed = sum(bool(row["case_passed"]) for row in fault_rows)
    violations = sum(int(row["safety_violation_steps"]) for row in rows)
    figure.suptitle("ArmBench OpenPI wire fault matrix", fontsize=15)
    figure.text(
        0.5,
        0.015,
        f"Scripted non-learned loopback | fail-closed {passed}/{len(fault_rows)} "
        f"faults | safety violations {violations}",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.94))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _summary(rows: list[dict[str, object]], matrix_passed: bool) -> str:
    fault_rows = [row for row in rows if row["fault_mode"] != "none"]
    fail_closed = sum(bool(row["case_passed"]) for row in fault_rows)
    violations = sum(int(row["safety_violation_steps"]) for row in rows)
    lines = [
        "# OpenPI wire fault matrix",
        "",
        "This matrix uses an ephemeral scripted non-learned server. It sends "
        "real DROID requests through the official OpenPI MessagePack/WebSocket "
        "client; it does not run pi0/pi0.5 inference.",
        "",
        f"- Matrix passed: `{str(matrix_passed).lower()}`",
        f"- Injected faults that failed closed: `{fail_closed}/{len(fault_rows)}`",
        f"- Total physical safety violation steps: `{violations}`",
        "",
        "| Fault | Requests | Valid chunks | Fallbacks | Client failure | Safe | "
        "Violation steps | Hashes match | Case passed |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['fault_mode']} | {row['request_count']} | "
            f"{row['validated_remote_chunks']} | "
            f"{row['runtime_fallback_chunks']} | "
            f"{row['failure_type'] or '-'} | {row['physical_safe']} | "
            f"{row['safety_violation_steps']} | "
            f"{row['request_hashes_match']} | {row['case_passed']} |"
        )
    lines.extend(
        [
            "",
            "`case_passed` requires one server-audited request, matching camera "
            "hashes, zero physical safety violations, and the expected response "
            "authority. Fault cases must have zero validated remote chunks and "
            "one policy-inference runtime fallback.",
            "",
            "These are deterministic fault injections, not estimates of server "
            "availability or certified safety guarantees.",
            "",
        ]
    )
    return "\n".join(lines)


def execute_loopback_fault_matrix(
    config_path: Path,
    output_directory: Path,
    *,
    scenario_name: str = "single_block",
    execution_horizon: int = 1,
    payload_mass: float = 0.0,
    fault_modes: Sequence[str] = LOOPBACK_FAULT_MODES,
    fault_delay_ms: float = 250.0,
    inference_timeout_s: float = 0.1,
    make_videos: bool = False,
    record_full_observations: bool = True,
) -> Path:
    """Run matched nominal/fault loopbacks and write a combined report."""

    if output_directory.exists():
        raise FileExistsError(
            f"output directory already exists: {output_directory}"
        )
    selected_modes = [str(mode) for mode in fault_modes]
    if not selected_modes or len(selected_modes) != len(set(selected_modes)):
        raise ValueError("fault matrix modes must be unique and nonempty")
    unknown = set(selected_modes).difference(LOOPBACK_FAULT_MODES)
    if unknown:
        raise ValueError(f"unknown loopback fault modes: {sorted(unknown)}")
    output_directory.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    log_lines: list[str] = []
    for fault_mode in selected_modes:
        case_directory = output_directory / fault_mode
        log_lines.append(f"starting {fault_mode}")
        execute_openpi_loopback_run(
            config_path,
            case_directory,
            scenario_name=scenario_name,
            execution_horizon=execution_horizon,
            payload_mass=payload_mass,
            max_policy_queries=1,
            make_video=make_videos,
            fault_mode=fault_mode,
            fault_request_index=0,
            fault_delay_ms=fault_delay_ms,
            inference_timeout_s=inference_timeout_s,
            record_full_observations=record_full_observations,
        )
        validation = validate_online_artifact(
            case_directory, decode_videos=make_videos
        )
        aggregate = _aggregate_row(case_directory / "aggregate.json")
        audit = _json_mapping(case_directory / "loopback_server.json")
        chunk = _first_chunk(case_directory / "per_chunk.csv")
        requests = audit.get("requests")
        if not isinstance(requests, list) or len(requests) != 1:
            raise ValueError(f"invalid loopback request audit: {case_directory}")
        request = requests[0]
        if not isinstance(request, dict):
            raise ValueError(f"invalid loopback request row: {case_directory}")
        request_hashes_match = (
            request.get("exterior_image_sha256")
            == chunk["exterior_image_sha256"]
            and request.get("wrist_image_sha256")
            == chunk["wrist_image_sha256"]
        )
        safety_violations = sum(
            int(aggregate[field])
            for field in (
                "obstacle_contact_steps",
                "self_contact_steps",
                "joint_limit_violation_steps",
            )
        )
        failure_types = aggregate.get("runtime_failure_types", [])
        if not isinstance(failure_types, list):
            raise ValueError("runtime_failure_types must be a list")
        matrix_row: dict[str, object] = {
            "fault_mode": fault_mode,
            "artifact_directory": fault_mode,
            "request_count": int(audit["request_count"]),
            "request_hashes_match": request_hashes_match,
            "policy_queries": int(aggregate["policy_queries"]),
            "validated_remote_chunks": int(
                aggregate["validated_remote_chunks"]
            ),
            "runtime_fallback_chunks": int(
                aggregate["runtime_fallback_chunks"]
            ),
            "failure_type": "|".join(str(value) for value in failure_types),
            "termination_reason": str(aggregate["termination_reason"]),
            "task_success": bool(aggregate["task_success"]),
            "physical_safe": bool(aggregate["physical_safe"]),
            "safety_violation_steps": safety_violations,
            "p95_client_inference_latency_ms": float(
                aggregate["p95_client_inference_latency_ms"]
            ),
            "p95_policy_latency_ms": float(
                aggregate["p95_policy_latency_ms"]
            ),
            "full_observation_frames": validation.full_observation_frames,
            "artifact_sha256": validation.aggregate_sha256,
        }
        matrix_row["case_passed"] = _case_passed(
            fault_mode,
            aggregate,
            audit,
            request_hashes_match=request_hashes_match,
        )
        rows.append(matrix_row)
        log_lines.append(
            f"completed {fault_mode}: passed={matrix_row['case_passed']}"
        )

    matrix_passed = all(bool(row["case_passed"]) for row in rows)
    _write_csv(output_directory / "matrix.csv", rows)
    _write_json(output_directory / "matrix.json", rows)
    matrix_sha256 = hashlib.sha256(
        (output_directory / "matrix.json").read_bytes()
    ).hexdigest()
    _write_overview(output_directory / "overview.png", rows)
    (output_directory / "summary.md").write_text(
        _summary(rows, matrix_passed), encoding="utf-8", newline="\n"
    )
    manifest = {
        "artifact_type": "armbench_openpi_loopback_fault_matrix_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario_name,
        "execution_horizon": execution_horizon,
        "payload_mass_kg": payload_mass,
        "fault_modes": selected_modes,
        "fault_delay_ms": fault_delay_ms,
        "inference_timeout_s": inference_timeout_s,
        "make_videos": bool(make_videos),
        "record_full_observations": bool(record_full_observations),
        "case_count": len(rows),
        "fault_case_count": sum(mode != "none" for mode in selected_modes),
        "passed_case_count": sum(bool(row["case_passed"]) for row in rows),
        "matrix_passed": matrix_passed,
        "matrix_sha256": matrix_sha256,
    }
    _write_json(output_directory / "manifest.json", manifest)
    log_lines.append(f"matrix_passed={matrix_passed}")
    (output_directory / "run.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8", newline="\n"
    )
    return output_directory
