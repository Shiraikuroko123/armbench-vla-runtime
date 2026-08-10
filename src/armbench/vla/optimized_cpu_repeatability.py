"""Cold-process repeatability audit for the optimized CPU assurance replay.

The v0.2.0 replay is deliberately kept unchanged.  This module launches that
replay in a fresh child process for every trial, records the complete child
artifact and process evidence, and recomputes condition-level summaries from
the recorded files.  A small controlled CPU-contention condition is included
as a diagnostic; it is not a real-time or hardware-safety certification.
"""

from __future__ import annotations

import json
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

from armbench.vla.optimized_cpu_replay import (
    PROFILE_IDS,
    _input_inventory,
    validate_optimized_cpu_replay,
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


SCHEMA = "armbench.pi05_optimized_cpu_repeatability.v1"
SUMMARY_SCHEMA = "armbench.pi05_optimized_cpu_repeatability_summary.v1"
PROVENANCE_SCHEMA = "armbench.pi05_optimized_cpu_repeatability_provenance.v1"
MANIFEST_SCHEMA = "armbench.pi05_optimized_cpu_repeatability_manifest.v1"
TRIAL_SCHEMA = "armbench.pi05_optimized_cpu_repeatability_trial.v1"
SCOPE = "frozen_pi05_optimized_cpu_repeatability_audit"
PROTOCOL_RELATIVE = Path(
    "docs/research/pi05_optimized_cpu_repeatability_protocol_20260811.json"
)
_CONDITIONS = ("idle", "cpu_load")
_PROFILE_METRICS = (
    "cases",
    "execute",
    "hold",
    "constraint_satisfied_candidates",
    "unsafe_plans_published",
    "partial_prefixes_exposed",
    "software_budget_exceeded",
    "response_deadline_exceeded",
    "p50_supervisor_latency_ms",
    "p95_supervisor_latency_ms",
    "maximum_supervisor_latency_ms",
    "p50_worker_latency_ms",
    "p95_worker_latency_ms",
    "maximum_worker_latency_ms",
)
_CLAIM_BOUNDARY = [
    "Frozen official responses are replayed; the pi0.5 checkpoint is not executed.",
    "LIBERO actions are cross-controller Panda inputs, not native Panda policy outputs.",
    "Independent cases do not provide closed-loop policy feedback or task success.",
    "Python CPU timing is best effort and is not a hard-real-time guarantee.",
    "MuJoCo geometry and inverse dynamics are not physical-robot safety certification.",
    "A repeatability audit cannot establish robot-level safety or generalization.",
]

# Four independent workers give a visible but bounded host-contention condition
# on the 20-thread development machine.  The process is terminated by the
# parent as soon as the scored replay finishes.
_CPU_LOAD_PROGRAM = r"""
import sys
import time

duration = float(sys.argv[1])
state = (int(sys.argv[2]) + 1) * 0.6180339887498949
deadline = time.perf_counter() + duration
while time.perf_counter() < deadline:
    state = (state * 1.000000119 + 0.000000731) % 1000003.0
"""


@dataclass(frozen=True)
class OptimizedCPURepeatabilityConfig:
    """Fixed trial matrix and child replay configuration."""

    baseline_repeats: int = 3
    load_repeats: int = 3
    load_workers: int = 4
    chunks: int = 30
    operational_budget_ms: float = 20.0
    diagnostic_budget_ms: float = 100.0
    response_deadline_ms: float = 200.0
    qp_step_budget_ms: float = 5.0
    worker_timeout_s: float = 30.0
    trial_timeout_s: float = 180.0

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
            raise ValueError("load_workers must be positive when load repeats are used")
        for value, label in (
            (self.operational_budget_ms, "operational_budget_ms"),
            (self.diagnostic_budget_ms, "diagnostic_budget_ms"),
            (self.response_deadline_ms, "response_deadline_ms"),
            (self.qp_step_budget_ms, "qp_step_budget_ms"),
            (self.worker_timeout_s, "worker_timeout_s"),
            (self.trial_timeout_s, "trial_timeout_s"),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{label} must be finite and positive")
            if value <= 0 or value != value or value == float("inf"):
                raise ValueError(f"{label} must be finite and positive")
        if self.diagnostic_budget_ms <= self.operational_budget_ms:
            raise ValueError("diagnostic budget must exceed operational budget")

    @property
    def protocol_conformant(self) -> bool:
        return self == OptimizedCPURepeatabilityConfig()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _protocol_path() -> Path:
    return _project_root() / PROTOCOL_RELATIVE


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _root_inventory(root: Path) -> list[dict[str, Any]]:
    """Inventory every file except the repeatability root manifest itself."""

    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "manifest.json"
    ]


def _write_manifest(root: Path) -> None:
    files = _root_inventory(root)
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
    expected = _root_inventory(root)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / "manifest.json"
    }
    files = manifest.get("files") if isinstance(manifest, Mapping) else None
    if not (
        has_exact_fields(manifest, {"schema_version", "files", "inventory_sha256"})
        and manifest["schema_version"] == MANIFEST_SCHEMA
        and isinstance(files, list)
        and files == expected
        and actual == {item["path"] for item in expected}
        and is_sha256(manifest["inventory_sha256"])
        and manifest["inventory_sha256"] == sha256_bytes(canonical_json(expected))
    ):
        raise ValueError("repeatability manifest mismatch")
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
        "src/armbench/vla/optimized_cpu_repeatability.py": root
        / "src/armbench/vla/optimized_cpu_repeatability.py",
        PROTOCOL_RELATIVE.as_posix(): _protocol_path(),
    }
    return {label: sha256_file(path) for label, path in paths.items()}


def _child_command(
    source: Path, artifact: Path, config: OptimizedCPURepeatabilityConfig
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "armbench",
        "vla-panda-optimized-replay",
        str(source),
        "--output-directory",
        str(artifact),
        "--chunks",
        str(config.chunks),
        "--operational-budget-ms",
        str(config.operational_budget_ms),
        "--diagnostic-budget-ms",
        str(config.diagnostic_budget_ms),
        "--response-deadline-ms",
        str(config.response_deadline_ms),
        "--qp-step-budget-ms",
        str(config.qp_step_budget_ms),
        "--worker-timeout-s",
        str(config.worker_timeout_s),
    ]


def _start_cpu_load(workers: int, duration_s: float) -> list[subprocess.Popen[bytes]]:
    if workers <= 0:
        return []
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    processes: list[subprocess.Popen[bytes]] = []
    try:
        for index in range(workers):
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        _CPU_LOAD_PROGRAM,
                        str(duration_s),
                        str(index),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=environment,
                )
            )
    except OSError:
        _stop_cpu_load(processes)
        raise
    return processes


def _stop_cpu_load(processes: Sequence[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)


def _as_bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    return value if isinstance(value, bytes) else value.encode("utf-8", "replace")


def _first_json_document(value: bytes) -> object | None:
    """Extract the first JSON object from CLI output followed by a results line."""

    text = value.decode("utf-8", "replace")
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            document, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return document
    return None


def _summary_snapshot(summary: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(summary, Mapping):
        raise ValueError("optimized replay summary is not an object")
    by_profile = summary.get("by_profile")
    if not isinstance(by_profile, Mapping) or set(by_profile) != set(PROFILE_IDS):
        raise ValueError("optimized replay summary profiles are invalid")
    snapshot: dict[str, Any] = {"by_profile": {}, "overall": summary.get("overall")}
    if not isinstance(snapshot["overall"], Mapping):
        raise ValueError("optimized replay summary overall section is invalid")
    for profile in PROFILE_IDS:
        values = by_profile[profile]
        if not isinstance(values, Mapping):
            raise ValueError("optimized replay profile summary is invalid")
        snapshot["by_profile"][profile] = {
            key: values[key] for key in _PROFILE_METRICS
        }
    snapshot["go_no_go"] = summary.get("go_no_go")
    if not isinstance(snapshot["go_no_go"], Mapping):
        raise ValueError("optimized replay go/no-go section is invalid")
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
    if not records:
        return {
            "trials": 0,
            "valid_trials": 0,
            "failed_trials": 0,
            "wall_time_ms": None,
            "profiles": {
                profile: {
                    "trials": 0,
                    **{key: None for key in _PROFILE_METRICS},
                    "execute_counts": [],
                    "unsafe_publications_all_zero": False,
                    "partial_prefixes_all_zero": False,
                }
                for profile in PROFILE_IDS
            },
            "go_no_go_counts": {"go": 0, "no_go": 0},
        }
    valid = [record for record in records if bool(record.get("valid"))]
    profiles: dict[str, Any] = {}
    for profile in PROFILE_IDS:
        profile_runs = [
            record["summary"]["by_profile"][profile]
            for record in valid
            if isinstance(record.get("summary"), Mapping)
        ]
        values: dict[str, Any] = {"trials": len(profile_runs)}
        for key in _PROFILE_METRICS:
            values[key] = _stats([run[key] for run in profile_runs]) if profile_runs else None
        values["execute_counts"] = [int(run["execute"]) for run in profile_runs]
        values["unsafe_publications_all_zero"] = bool(profile_runs) and all(
            int(run["unsafe_plans_published"]) == 0 for run in profile_runs
        )
        values["partial_prefixes_all_zero"] = bool(profile_runs) and all(
            int(run["partial_prefixes_exposed"]) == 0 for run in profile_runs
        )
        profiles[profile] = values
    return {
        "trials": len(records),
        "valid_trials": len(valid),
        "failed_trials": len(records) - len(valid),
        "wall_time_ms": _stats([float(record["wall_time_ms"]) for record in records]),
        "profiles": profiles,
        "go_no_go_counts": {
            decision: sum(
                record.get("summary", {}).get("go_no_go", {}).get("decision")
                == decision
                for record in valid
            )
            for decision in ("go", "no_go")
        },
    }


def _summary(
    records: Sequence[Mapping[str, Any]],
    config: OptimizedCPURepeatabilityConfig,
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
            for condition in _CONDITIONS
        },
        "claim_boundary": list(_CLAIM_BOUNDARY),
    }


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Optimized CPU repeatability audit",
        "",
        "Each row is one fresh Python process running the frozen 180-case optimized replay.",
        "The CPU-load condition is a bounded host-contention diagnostic.",
        "",
        "| Condition | Profile | Trials | Execute counts | P95 supervisor (mean +/- stdev) | P95 worker (mean +/- stdev) | Unsafe all zero | Prefixes all zero |",
        "| --- | --- | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for condition in _CONDITIONS:
        section = summary["conditions"][condition]
        for profile in PROFILE_IDS:
            values = section["profiles"][profile]
            p95_supervisor = values["p95_supervisor_latency_ms"]
            p95_worker = values["p95_worker_latency_ms"]
            if values["trials"]:
                supervisor_text = (
                    f"{p95_supervisor['mean']:.3f} +/- {p95_supervisor['stdev']:.3f} ms"
                )
                worker_text = (
                    f"{p95_worker['mean']:.3f} +/- {p95_worker['stdev']:.3f} ms"
                )
            else:
                supervisor_text = worker_text = "n/a"
            lines.append(
                f"| `{condition}` | `{profile}` | {values['trials']} | "
                f"{values['execute_counts']} | {supervisor_text} | {worker_text} | "
                f"{values['unsafe_publications_all_zero']} | "
                f"{values['partial_prefixes_all_zero']} |"
            )
    lines.extend(
        [
            "",
            f"All trials valid: **{summary['all_trials_valid']}**.",
            "",
            "The descriptive repeatability result does not establish hard real-time behavior, task success, or physical-robot safety.",
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
    config: OptimizedCPURepeatabilityConfig,
) -> dict[str, Any]:
    artifact = trial_directory / "artifact"
    stdout_path = trial_directory / "stdout.log"
    stderr_path = trial_directory / "stderr.log"
    command = _child_command(source, artifact, config)
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    started = _utc_now()
    load_processes = _start_cpu_load(
        load_workers, config.trial_timeout_s + 10.0
    )
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
    summary: dict[str, Any] | None = None
    child_result = _first_json_document(stdout)
    if returncode == 0 and (artifact / "summary.json").is_file():
        summary = _summary_snapshot(strict_json_load(artifact / "summary.json"))
    valid = bool(returncode == 0 and not timed_out and summary is not None)
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
        "summary": summary,
    }
    write_json(trial_directory / "trial.json", record)
    return record


def execute_optimized_cpu_repeatability(
    input_directory: Path,
    output_directory: Path,
    config: OptimizedCPURepeatabilityConfig = OptimizedCPURepeatabilityConfig(),
) -> Path:
    """Run the fixed idle/load matrix in fresh child processes."""

    source = input_directory.resolve()
    output = output_directory.resolve()
    if output.exists():
        raise FileExistsError(f"repeatability output already exists: {output}")
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
    write_json(output / "trials.json", {"schema_version": SCHEMA, "trials": records})
    write_json(output / "summary.json", _summary(records, config, source_binding))
    (output / "summary.md").write_text(
        _summary_markdown(strict_json_load(output / "summary.json")),
        encoding="utf-8",
    )
    write_json(
        output / "provenance.json",
        {
            "schema_version": PROVENANCE_SCHEMA,
            "scope": SCOPE,
            "source": {
                "directory": str(input_directory),
                **source_binding,
            },
            "protocol": {
                "schema_version": "armbench.pi05_optimized_cpu_repeatability_protocol.v1",
                "sha256": sha256_file(_protocol_path()),
                "conformant": config.protocol_conformant,
            },
            "implementation_sha256": _implementation_hashes(),
            "environment": {
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
            },
            "claim_boundary": list(_CLAIM_BOUNDARY),
        },
    )
    shutil.copyfile(_protocol_path(), output / "protocol.json")
    _write_manifest(output)
    validate_optimized_cpu_repeatability(output, source)
    return output


def _parse_trials(root: Path) -> list[dict[str, Any]]:
    value = strict_json_load(root / "trials.json")
    if not has_exact_fields(value, {"schema_version", "trials"}) or value[
        "schema_version"
    ] != SCHEMA:
        raise ValueError("repeatability trials schema mismatch")
    trials = value["trials"]
    if not isinstance(trials, list) or not trials:
        raise ValueError("repeatability trials are empty")
    return trials


def _validate_trial_record(
    root: Path,
    source: Path,
    record: Mapping[str, Any],
) -> None:
    expected_fields = {
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
    if not has_exact_fields(record, expected_fields):
        raise ValueError("repeatability trial fields are invalid")
    condition = record["condition"]
    if (
        record["schema_version"] != TRIAL_SCHEMA
        or condition not in _CONDITIONS
        or type(record["replicate"]) is not int
        or record["replicate"] <= 0
        or type(record["load_workers"]) is not int
        or record["load_workers"] < 0
        or not isinstance(record["command"], list)
        or not isinstance(record["environment"], Mapping)
        or not isinstance(record["timed_out"], bool)
        or not isinstance(record["valid"], bool)
        or not isinstance(record["wall_time_ms"], (int, float))
        or record["wall_time_ms"] < 0
        or not is_sha256(record["stdout_sha256"])
        or not is_sha256(record["stderr_sha256"])
    ):
        raise ValueError("repeatability trial metadata is invalid")
    for key in ("stdout_file", "stderr_file"):
        path = root / str(record[key])
        relative = Path(str(record[key]))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_file()
            or sha256_file(path) != record[f"{key[:-5]}_sha256"]
        ):
            raise ValueError(f"repeatability {key} binding mismatch")
    artifact_name = record["artifact_directory"]
    summary = record["summary"]
    if bool(record["valid"]):
        if not isinstance(artifact_name, str) or not isinstance(summary, Mapping):
            raise ValueError("valid repeatability trial lacks artifact")
        artifact = root / artifact_name
        relative_artifact = Path(artifact_name)
        if (
            relative_artifact.is_absolute()
            or ".." in relative_artifact.parts
            or not artifact.is_dir()
        ):
            raise ValueError("repeatability trial artifact is missing")
        if sha256_file(artifact / "manifest.json") != record["artifact_manifest_sha256"]:
            raise ValueError("repeatability child manifest binding mismatch")
        if sha256_file(artifact / "summary.json") != record["artifact_summary_sha256"]:
            raise ValueError("repeatability child summary binding mismatch")
        child_summary = _summary_snapshot(strict_json_load(artifact / "summary.json"))
        if canonical_json(child_summary) != canonical_json(summary):
            raise ValueError("repeatability child summary snapshot mismatch")
        child_result = validate_optimized_cpu_replay(artifact, source)
        if not child_result.get("valid"):
            raise ValueError("repeatability child artifact did not validate")
    elif summary is not None:
        raise ValueError("failed repeatability trial contains a summary")


def validate_optimized_cpu_repeatability(
    directory: Path, input_directory: Path
) -> dict[str, Any]:
    """Recompute every trial binding and the condition-level summary."""

    root = directory.resolve()
    source = input_directory.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"repeatability directory not found: {root}")
    inventory_sha = _validate_root_manifest(root)
    protocol = strict_json_load(root / "protocol.json")
    if (
        (root / "protocol.json").read_bytes() != _protocol_path().read_bytes()
        or protocol.get("schema_version")
        != "armbench.pi05_optimized_cpu_repeatability_protocol.v1"
    ):
        raise ValueError("repeatability protocol copy mismatch")
    provenance = strict_json_load(root / "provenance.json")
    source_binding = _source_binding(source)
    if not (
        provenance.get("schema_version") == PROVENANCE_SCHEMA
        and provenance.get("scope") == SCOPE
        and provenance.get("source", {}).get("manifest_sha256")
        == source_binding["manifest_sha256"]
        and provenance.get("source", {}).get("inventory_sha256")
        == source_binding["inventory_sha256"]
        and provenance.get("source", {}).get("summary_sha256")
        == source_binding["summary_sha256"]
        and provenance.get("protocol", {}).get("sha256")
        == sha256_file(_protocol_path())
        and provenance.get("implementation_sha256") == _implementation_hashes()
        and canonical_json(provenance.get("claim_boundary"))
        == canonical_json(_CLAIM_BOUNDARY)
    ):
        raise ValueError("repeatability provenance mismatch")
    trials = _parse_trials(root)
    seen: set[str] = set()
    for record in trials:
        trial_id = record.get("trial_id")
        if not isinstance(trial_id, str) or trial_id in seen:
            raise ValueError("repeatability trial identities are invalid")
        seen.add(trial_id)
        _validate_trial_record(root, source, record)
    summary = strict_json_load(root / "summary.json")
    configuration = summary.get("configuration") if isinstance(summary, Mapping) else None
    if not has_exact_fields(configuration, set(asdict(OptimizedCPURepeatabilityConfig()))):
        raise ValueError("repeatability configuration is invalid")
    try:
        config = OptimizedCPURepeatabilityConfig(**configuration)
    except (TypeError, ValueError) as error:
        raise ValueError("repeatability configuration is invalid") from error
    expected = _summary(trials, config, source_binding)
    if canonical_json(summary) != canonical_json(expected):
        raise ValueError("repeatability summary is not reproducible")
    if (root / "summary.md").read_text(encoding="utf-8") != _summary_markdown(summary):
        raise ValueError("repeatability Markdown summary is not reproducible")
    expected_count = config.baseline_repeats + config.load_repeats
    if len(trials) != expected_count:
        raise ValueError("repeatability trial count does not match configuration")
    return {
        "valid": True,
        "scope": SCOPE,
        "trials": len(trials),
        "valid_trials": sum(bool(record["valid"]) for record in trials),
        "idle_trials": sum(record["condition"] == "idle" for record in trials),
        "cpu_load_trials": sum(record["condition"] == "cpu_load" for record in trials),
        "all_trials_valid": bool(summary["all_trials_valid"]),
        "manifest_inventory_sha256": inventory_sha,
        "checks": [
            "recursive_manifest_and_exact_file_set",
            "protocol_and_source_bindings",
            "child_replay_artifacts_revalidated",
            "trial_log_hashes",
            "condition_summary_recomputed",
            "claim_boundaries_recomputed",
        ],
    }


__all__ = [
    "OptimizedCPURepeatabilityConfig",
    "execute_optimized_cpu_repeatability",
    "validate_optimized_cpu_repeatability",
    "_condition_summary",
    "_summary",
    "_summary_markdown",
]
