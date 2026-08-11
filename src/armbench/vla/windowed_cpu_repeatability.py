"""Fresh-process repeatability audit for the certified-window replay."""

from __future__ import annotations

import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from armbench.vla.optimized_cpu_repeatability import (
    _as_bytes,
    _first_json_document,
    _start_cpu_load,
    _stop_cpu_load,
)
from armbench.vla.serialization import (
    canonical_json,
    has_exact_fields,
    is_sha256,
    sha256_bytes,
    sha256_file,
    strict_json_load,
    write_json,
)
from armbench.vla.windowed_cpu_replay import (
    PROFILE_IDS,
    PROTOCOL_RELATIVE as WINDOW_PROTOCOL_RELATIVE,
    SCHEMA as WINDOW_SCHEMA,
    _input_inventory,
    validate_windowed_cpu_replay,
)


SCHEMA = "armbench.pi05_windowed_cpu_repeatability.v1"
SUMMARY_SCHEMA = "armbench.pi05_windowed_cpu_repeatability_summary.v1"
TRIAL_SCHEMA = "armbench.pi05_windowed_cpu_repeatability_trial.v1"
MANIFEST_SCHEMA = "armbench.pi05_windowed_cpu_repeatability_manifest.v1"
PROTOCOL_SCHEMA = "armbench.pi05_windowed_cpu_repeatability_protocol.v1"
SCOPE = "frozen_pi05_panda_windowed_cpu_repeatability_audit"
PROTOCOL_RELATIVE = Path(
    "docs/research/pi05_windowed_cpu_repeatability_protocol_20260811.json"
)
CONDITIONS = ("idle", "cpu_load")
PROFILE_METRICS = (
    "cases",
    "execute",
    "hold",
    "constraint_satisfied_candidates",
    "unsafe_windows_published",
    "partial_windows_exposed",
    "source_chunk_prefix_publications",
    "software_budget_exceeded",
    "response_deadline_exceeded",
    "p50_supervisor_latency_ms",
    "p95_supervisor_latency_ms",
    "maximum_supervisor_latency_ms",
    "p50_worker_latency_ms",
    "p95_worker_latency_ms",
    "maximum_worker_latency_ms",
)
TRIAL_FIELDS = {
    "schema_version",
    "trial_id",
    "condition",
    "replicate",
    "load_workers",
    "started_at_utc",
    "finished_at_utc",
    "wall_time_ms",
    "returncode",
    "timed_out",
    "valid",
    "command",
    "cwd",
    "environment",
    "stdout_file",
    "stdout_sha256",
    "stderr_file",
    "stderr_sha256",
    "artifact_directory",
    "artifact_manifest_sha256",
    "artifact_summary_sha256",
    "child_result",
    "summary",
}
ARTIFACT_FILES = {
    "protocol.json",
    "provenance.json",
    "trials.json",
    "summary.json",
    "summary.md",
}
CLAIM_BOUNDARY = [
    "Each trial launches a fresh Python process; the load condition is bounded host contention.",
    "Frozen responses are replayed and no pi0.5 checkpoint inference is rerun.",
    "The result is descriptive timing and publication evidence, not hard real time or physical safety.",
    "H=1 preserves window atomicity but changes the source-chunk publication contract.",
]


@dataclass(frozen=True)
class WindowedCPURepeatabilityConfig:
    """Fixed idle/load trial matrix and child replay settings."""

    baseline_repeats: int = 3
    load_repeats: int = 3
    load_workers: int = 4
    chunks: int = 30
    supervision_budget_ms: float = 20.0
    response_deadline_ms: float = 200.0
    qp_step_budget_ms: float = 5.0
    worker_timeout_s: float = 30.0
    trial_timeout_s: float = 240.0

    def __post_init__(self) -> None:
        for value, label, minimum in (
            (self.baseline_repeats, "baseline_repeats", 0),
            (self.load_repeats, "load_repeats", 0),
            (self.load_workers, "load_workers", 0),
            (self.chunks, "chunks", 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{label} must be an integer >= {minimum}")
        if self.baseline_repeats + self.load_repeats <= 0:
            raise ValueError("at least one repeat is required")
        if self.load_repeats and self.load_workers <= 0:
            raise ValueError("load_workers must be positive with load repeats")
        for value, label in (
            (self.supervision_budget_ms, "supervision_budget_ms"),
            (self.response_deadline_ms, "response_deadline_ms"),
            (self.qp_step_budget_ms, "qp_step_budget_ms"),
            (self.worker_timeout_s, "worker_timeout_s"),
            (self.trial_timeout_s, "trial_timeout_s"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np_is_finite_positive(value)
            ):
                raise ValueError(f"{label} must be finite and positive")

    @property
    def protocol_conformant(self) -> bool:
        return self == WindowedCPURepeatabilityConfig()


def np_is_finite_positive(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return numeric > 0.0 and numeric == numeric and numeric != float("inf")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _protocol_path() -> Path:
    return _project_root() / PROTOCOL_RELATIVE


def _window_protocol_path() -> Path:
    return _project_root() / WINDOW_PROTOCOL_RELATIVE


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]


def _write_manifest(root: Path) -> None:
    files = _inventory(root)
    write_json(
        root / "manifest.json",
        {
            "schema_version": MANIFEST_SCHEMA,
            "files": files,
            "inventory_sha256": sha256_bytes(canonical_json(files)),
        },
    )


def _validate_root_manifest(root: Path) -> str:
    manifest = strict_json_load(root / "manifest.json")
    expected = _inventory(root)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if not (
        has_exact_fields(
            manifest, {"schema_version", "files", "inventory_sha256"}
        )
        and manifest["schema_version"] == MANIFEST_SCHEMA
        and manifest["files"] == expected
        and actual == {item["path"] for item in expected}
        and is_sha256(manifest["inventory_sha256"])
        and manifest["inventory_sha256"]
        == sha256_bytes(canonical_json(expected))
    ):
        raise ValueError("windowed repeatability manifest mismatch")
    return str(manifest["inventory_sha256"])


def _source_binding(source: Path) -> dict[str, str]:
    inventory_sha, manifest_sha = _input_inventory(source)
    return {
        "manifest_sha256": manifest_sha,
        "inventory_sha256": inventory_sha,
        "summary_sha256": sha256_file(source / "summary.json"),
    }


def _implementation_hashes() -> dict[str, str]:
    root = _project_root()
    paths = {
        "src/armbench/vla/windowed_cpu_repeatability.py": root
        / "src/armbench/vla/windowed_cpu_repeatability.py",
        "src/armbench/vla/windowed_cpu_replay.py": root
        / "src/armbench/vla/windowed_cpu_replay.py",
        PROTOCOL_RELATIVE.as_posix(): _protocol_path(),
        WINDOW_PROTOCOL_RELATIVE.as_posix(): _window_protocol_path(),
    }
    return {label: sha256_file(path) for label, path in paths.items()}


def _child_command(
    source: Path,
    artifact: Path,
    config: WindowedCPURepeatabilityConfig,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "armbench",
        "vla-panda-windowed-replay",
        str(source),
        "--output-directory",
        str(artifact),
        "--chunks",
        str(config.chunks),
        "--supervision-budget-ms",
        str(config.supervision_budget_ms),
        "--response-deadline-ms",
        str(config.response_deadline_ms),
        "--qp-step-budget-ms",
        str(config.qp_step_budget_ms),
        "--worker-timeout-s",
        str(config.worker_timeout_s),
    ]


def _summary_snapshot(summary: Mapping[str, Any]) -> dict[str, Any]:
    by_profile = summary.get("by_profile")
    if not isinstance(by_profile, Mapping) or set(by_profile) != set(PROFILE_IDS):
        raise ValueError("windowed child summary profiles are invalid")
    snapshot: dict[str, Any] = {"by_profile": {}}
    for profile in PROFILE_IDS:
        values = by_profile[profile]
        if not isinstance(values, Mapping):
            raise ValueError("windowed child profile summary is invalid")
        snapshot["by_profile"][profile] = {
            key: values[key] for key in PROFILE_METRICS
        }
    snapshot["paired_comparison"] = summary.get("paired_comparison")
    if not isinstance(snapshot["paired_comparison"], Mapping):
        raise ValueError("windowed child comparison is invalid")
    return snapshot


def _stats(values: Sequence[float | int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "min": min(numeric),
        "max": max(numeric),
        "mean": statistics.fmean(numeric),
        "stdev": statistics.pstdev(numeric) if len(numeric) > 1 else 0.0,
    }


def _condition_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if bool(record.get("valid"))]
    profiles: dict[str, Any] = {}
    for profile in PROFILE_IDS:
        runs = [
            record["summary"]["by_profile"][profile]
            for record in valid
            if isinstance(record.get("summary"), Mapping)
        ]
        values: dict[str, Any] = {"trials": len(runs)}
        for key in PROFILE_METRICS:
            values[key] = _stats([run[key] for run in runs]) if runs else None
        values["execute_counts"] = [int(run["execute"]) for run in runs]
        values["unsafe_windows_all_zero"] = bool(runs) and all(
            int(run["unsafe_windows_published"]) == 0 for run in runs
        )
        values["partial_windows_all_zero"] = bool(runs) and all(
            int(run["partial_windows_exposed"]) == 0 for run in runs
        )
        profiles[profile] = values
    return {
        "trials": len(records),
        "valid_trials": len(valid),
        "failed_trials": len(records) - len(valid),
        "wall_time_ms": _stats(
            [float(record["wall_time_ms"]) for record in records]
        )
        if records
        else None,
        "profiles": profiles,
    }


def _summary(
    records: Sequence[Mapping[str, Any]],
    config: WindowedCPURepeatabilityConfig,
    source: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA,
        "scope": SCOPE,
        "configuration": asdict(config),
        "protocol_conformant": config.protocol_conformant,
        "source": dict(source),
        "all_trials_valid": all(bool(record.get("valid")) for record in records),
        "conditions": {
            condition: _condition_summary(
                [record for record in records if record.get("condition") == condition]
            )
            for condition in CONDITIONS
        },
        "claim_boundary": list(CLAIM_BOUNDARY),
    }


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Windowed Panda CPU repeatability audit",
        "",
        (
            "Each row is a fresh process running the paired H=10/H=1 replay. "
            "The CPU-load condition is bounded host contention."
        ),
        "",
        (
            "| Condition | Profile | Trials | Execute counts | P95 supervisor "
            "(mean +/- stdev) | Unsafe all zero | Partial all zero |"
        ),
        "| --- | --- | ---: | --- | ---: | --- | --- |",
    ]
    for condition in CONDITIONS:
        section = summary["conditions"][condition]
        for profile in PROFILE_IDS:
            values = section["profiles"][profile]
            p95 = values["p95_supervisor_latency_ms"]
            p95_text = (
                f"{p95['mean']:.3f} +/- {p95['stdev']:.3f} ms"
                if values["trials"]
                else "n/a"
            )
            lines.append(
                f"| `{condition}` | `{profile}` | {values['trials']} | "
                f"{values['execute_counts']} | {p95_text} | "
                f"{values['unsafe_windows_all_zero']} | "
                f"{values['partial_windows_all_zero']} |"
            )
    lines.extend(
        [
            "",
            f"All trials valid: **{summary['all_trials_valid']}**.",
            "",
            (
                "This is descriptive repeatability evidence. It does not "
                "establish hard real-time behavior, task success, or physical "
                "robot safety."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _trial_record(
    *,
    condition: str,
    replicate: int,
    load_workers: int,
    source: Path,
    trial_directory: Path,
    config: WindowedCPURepeatabilityConfig,
) -> dict[str, Any]:
    artifact = trial_directory / "artifact"
    stdout_path = trial_directory / "stdout.log"
    stderr_path = trial_directory / "stderr.log"
    command = _child_command(source, artifact, config)
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    started = _utc_now()
    load_processes = _start_cpu_load(load_workers, config.trial_timeout_s + 10.0)
    start = time.perf_counter()
    timed_out = False
    returncode: int | None = None
    stdout = b""
    stderr = b""
    try:
        try:
            completed = subprocess.run(
                command,
                cwd=_project_root(),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=config.trial_timeout_s,
                check=False,
            )
            returncode = completed.returncode
            stdout = _as_bytes(completed.stdout)
            stderr = _as_bytes(completed.stderr)
        except subprocess.TimeoutExpired as error:
            timed_out = True
            stdout = _as_bytes(error.stdout)
            stderr = _as_bytes(error.stderr)
            returncode = -1
    finally:
        _stop_cpu_load(load_processes)
    finished = _utc_now()
    wall_time_ms = (time.perf_counter() - start) * 1000.0
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    child_result = _first_json_document(stdout)
    child_summary: dict[str, Any] | None = None
    if returncode == 0 and (artifact / "summary.json").is_file():
        child_summary = _summary_snapshot(strict_json_load(artifact / "summary.json"))
    valid = bool(returncode == 0 and not timed_out and child_summary is not None)
    record = {
        "schema_version": TRIAL_SCHEMA,
        "trial_id": f"{condition}_{replicate:03d}",
        "condition": condition,
        "replicate": replicate,
        "load_workers": load_workers,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "wall_time_ms": wall_time_ms,
        "returncode": returncode,
        "timed_out": timed_out,
        "valid": valid,
        "command": command,
        "cwd": str(_project_root()),
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "overrides": {"PYTHONHASHSEED": "0"},
        },
        "stdout_file": stdout_path.relative_to(trial_directory.parent.parent).as_posix(),
        "stdout_sha256": sha256_bytes(stdout),
        "stderr_file": stderr_path.relative_to(trial_directory.parent.parent).as_posix(),
        "stderr_sha256": sha256_bytes(stderr),
        "artifact_directory": (
            artifact.relative_to(trial_directory.parent.parent).as_posix()
            if artifact.is_dir()
            else None
        ),
        "artifact_manifest_sha256": (
            sha256_file(artifact / "manifest.json")
            if (artifact / "manifest.json").is_file()
            else None
        ),
        "artifact_summary_sha256": (
            sha256_file(artifact / "summary.json")
            if (artifact / "summary.json").is_file()
            else None
        ),
        "child_result": child_result,
        "summary": child_summary,
    }
    write_json(trial_directory / "trial.json", record)
    return record


def execute_windowed_cpu_repeatability(
    input_directory: Path,
    output_directory: Path,
    config: WindowedCPURepeatabilityConfig = WindowedCPURepeatabilityConfig(),
) -> Path:
    """Run idle and bounded-load fresh-process trials."""

    source = input_directory.resolve()
    output = output_directory.resolve()
    if output.exists():
        raise FileExistsError(
            f"windowed repeatability output already exists: {output}"
        )
    source_binding = _source_binding(source)
    output.mkdir(parents=True)
    (output / "trials").mkdir()
    records: list[dict[str, Any]] = []
    for condition, repeats, workers in (
        ("idle", config.baseline_repeats, 0),
        ("cpu_load", config.load_repeats, config.load_workers),
    ):
        for replicate in range(1, repeats + 1):
            trial_directory = output / "trials" / f"{condition}_{replicate:03d}"
            trial_directory.mkdir()
            records.append(
                _trial_record(
                    condition=condition,
                    replicate=replicate,
                    load_workers=workers,
                    source=source,
                    trial_directory=trial_directory,
                    config=config,
                )
            )
    write_json(
        output / "trials.json",
        {"schema_version": SCHEMA, "trials": records},
    )
    summary = _summary(records, config, source_binding)
    write_json(output / "summary.json", summary)
    (output / "summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    shutil.copyfile(_protocol_path(), output / "protocol.json")
    source_provenance = strict_json_load(source / "provenance.json")
    write_json(
        output / "provenance.json",
        {
            "schema_version": SCHEMA,
            "scope": SCOPE,
            "source": source_binding,
            "source_root_manifest_sha256": source_provenance["source"][
                "root_manifest_sha256"
            ],
            "window_schema": WINDOW_SCHEMA,
            "window_protocol_sha256": sha256_file(_window_protocol_path()),
            "implementation_sha256": _implementation_hashes(),
            "claim_boundary": list(CLAIM_BOUNDARY),
        },
    )
    _write_manifest(output)
    validate_windowed_cpu_repeatability(output, source)
    return output


def _config_from_json(value: object) -> WindowedCPURepeatabilityConfig:
    expected = set(asdict(WindowedCPURepeatabilityConfig()))
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("windowed repeatability configuration is invalid")
    try:
        config = WindowedCPURepeatabilityConfig(**value)
    except (TypeError, ValueError) as error:
        raise ValueError("windowed repeatability configuration is invalid") from error
    if asdict(config) != dict(value):
        raise ValueError("windowed repeatability configuration is not canonical")
    return config


def _validate_provenance(root: Path, source: Path) -> None:
    provenance = strict_json_load(root / "provenance.json")
    source_binding = _source_binding(source)
    source_provenance = strict_json_load(source / "provenance.json")
    expected = {
        "schema_version": SCHEMA,
        "scope": SCOPE,
        "source": source_binding,
        "source_root_manifest_sha256": source_provenance["source"][
            "root_manifest_sha256"
        ],
        "window_schema": WINDOW_SCHEMA,
        "window_protocol_sha256": sha256_file(_window_protocol_path()),
        "implementation_sha256": _implementation_hashes(),
        "claim_boundary": list(CLAIM_BOUNDARY),
    }
    if provenance != expected:
        raise ValueError("windowed repeatability provenance mismatch")


def _safe_relative(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} is invalid")
    relative = Path(value)
    path = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or root not in path.parents:
        raise ValueError(f"{label} escapes artifact root")
    return path


def _validate_trial_record(
    root: Path,
    source: Path,
    record: Mapping[str, Any],
    config: WindowedCPURepeatabilityConfig,
) -> None:
    if set(record) != TRIAL_FIELDS or record.get("schema_version") != TRIAL_SCHEMA:
        raise ValueError("windowed trial fields are invalid")
    condition = record.get("condition")
    if condition not in CONDITIONS:
        raise ValueError("windowed trial condition is invalid")
    replicate = record.get("replicate")
    if type(replicate) is not int or replicate <= 0:
        raise ValueError("windowed trial replicate is invalid")
    expected_id = f"{condition}_{replicate:03d}"
    if record.get("trial_id") != expected_id:
        raise ValueError("windowed trial identity is invalid")
    for label, field in (
        ("stdout_file", "stdout_sha256"),
        ("stderr_file", "stderr_sha256"),
    ):
        path = _safe_relative(root, record[label], label)
        digest = record[field]
        if not is_sha256(digest) or sha256_file(path) != digest:
            raise ValueError(f"{label} binding mismatch")
    artifact_value = record.get("artifact_directory")
    if not bool(record.get("valid")):
        if artifact_value is not None or record.get("summary") is not None:
            raise ValueError("invalid trial contains an artifact")
        return
    artifact = _safe_relative(root, artifact_value, "artifact_directory")
    if not artifact.is_dir():
        raise ValueError("windowed trial artifact is missing")
    manifest_digest = record.get("artifact_manifest_sha256")
    summary_digest = record.get("artifact_summary_sha256")
    if (
        not is_sha256(manifest_digest)
        or not is_sha256(summary_digest)
        or sha256_file(artifact / "manifest.json") != manifest_digest
        or sha256_file(artifact / "summary.json") != summary_digest
    ):
        raise ValueError("windowed child artifact binding mismatch")
    child = validate_windowed_cpu_replay(artifact, source)
    if child.get("valid") is not True:
        raise ValueError("windowed child artifact is invalid")
    expected_snapshot = _summary_snapshot(strict_json_load(artifact / "summary.json"))
    if record.get("summary") != expected_snapshot:
        raise ValueError("windowed child summary binding mismatch")
    if int(record.get("returncode", -1)) != 0 or bool(record.get("timed_out")):
        raise ValueError("valid windowed trial has a failed process")


def validate_windowed_cpu_repeatability(
    directory: Path, input_directory: Path
) -> dict[str, Any]:
    """Validate all trial logs, child artifacts, and recomputed statistics."""

    root = directory.resolve()
    source = input_directory.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"windowed repeatability directory not found: {root}")
    inventory_sha = _validate_root_manifest(root)
    if (root / "protocol.json").read_bytes() != _protocol_path().read_bytes():
        raise ValueError("windowed repeatability protocol copy mismatch")
    protocol = strict_json_load(root / "protocol.json")
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("windowed repeatability protocol schema mismatch")
    summary = strict_json_load(root / "summary.json")
    if not isinstance(summary, Mapping):
        raise ValueError("windowed repeatability summary is invalid")
    config = _config_from_json(summary.get("configuration"))
    _validate_provenance(root, source)
    trials_document = strict_json_load(root / "trials.json")
    if not (
        isinstance(trials_document, Mapping)
        and trials_document.get("schema_version") == SCHEMA
        and isinstance(trials_document.get("trials"), list)
    ):
        raise ValueError("windowed trials document is invalid")
    records = trials_document["trials"]
    expected_count = config.baseline_repeats + config.load_repeats
    if len(records) != expected_count:
        raise ValueError("windowed trial count is invalid")
    expected_ids = {
        f"{condition}_{replicate:03d}"
        for condition, repeats in (
            ("idle", config.baseline_repeats),
            ("cpu_load", config.load_repeats),
        )
        for replicate in range(1, repeats + 1)
    }
    if {record.get("trial_id") for record in records} != expected_ids:
        raise ValueError("windowed trial identities are invalid")
    for record in records:
        _validate_trial_record(root, source, record, config)
    expected_summary = _summary(records, config, _source_binding(source))
    if summary != expected_summary:
        raise ValueError("windowed repeatability summary mismatch")
    if (root / "summary.md").read_text(encoding="utf-8") != _summary_markdown(
        summary
    ):
        raise ValueError("windowed repeatability Markdown mismatch")
    return {
        "valid": True,
        "trials": len(records),
        "all_trials_valid": summary["all_trials_valid"],
        "inventory_sha256": inventory_sha,
        "checks": [
            "recursive_manifest_and_exact_file_set",
            "protocol_and_source_bindings",
            "fresh_trial_logs_and_child_artifacts",
            "child_window_validators_recomputed",
            "condition_statistics_recomputed",
            "summary_and_markdown_recomputed",
        ],
    }


__all__ = [
    "WindowedCPURepeatabilityConfig",
    "execute_windowed_cpu_repeatability",
    "validate_windowed_cpu_repeatability",
]
