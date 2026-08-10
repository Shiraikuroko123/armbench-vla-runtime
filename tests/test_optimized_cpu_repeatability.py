from __future__ import annotations

from pathlib import Path

import pytest

from armbench.cli import build_parser
from armbench.vla.optimized_cpu_repeatability import (
    PROFILE_IDS,
    TRIAL_SCHEMA,
    OptimizedCPURepeatabilityConfig,
    _summary,
    _summary_markdown,
    _summary_snapshot,
    _validate_trial_record,
)
from armbench.vla.serialization import sha256_file, strict_json_load


def _record(condition: str, replicate: int, execute: int) -> dict[str, object]:
    source = Path(__file__).resolve().parents[1] / "reports" / "pi05_optimized_cpu_replay_180_001"
    summary = _summary_snapshot(strict_json_load(source / "summary.json"))
    for profile in PROFILE_IDS:
        summary["by_profile"][profile]["execute"] = execute
    return {
        "condition": condition,
        "replicate": replicate,
        "valid": True,
        "summary": summary,
        "wall_time_ms": 100.0 + replicate,
    }


def test_repeatability_summary_recomputes_condition_statistics() -> None:
    config = OptimizedCPURepeatabilityConfig(
        baseline_repeats=1,
        load_repeats=1,
        load_workers=1,
        chunks=1,
    )
    records = [_record("idle", 1, 1), _record("cpu_load", 1, 0)]
    summary = _summary(
        records,
        config,
        {
            "manifest_sha256": "a" * 64,
            "inventory_sha256": "b" * 64,
            "summary_sha256": "c" * 64,
        },
    )

    assert summary["all_trials_valid"] is True
    assert summary["conditions"]["idle"]["profiles"]["operational_20ms"][
        "execute_counts"
    ] == [1]
    assert summary["conditions"]["cpu_load"]["profiles"]["operational_20ms"][
        "execute_counts"
    ] == [0]
    assert "repeatability" in _summary_markdown(summary)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"baseline_repeats": -1},
        {"load_repeats": -1},
        {"load_repeats": 1, "load_workers": 0},
        {"baseline_repeats": 0, "load_repeats": 0},
        {"chunks": 0},
        {"trial_timeout_s": 0.0},
        {"diagnostic_budget_ms": 20.0},
    ],
)
def test_repeatability_configuration_fails_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        OptimizedCPURepeatabilityConfig(**kwargs)


def test_repeatability_parser_exposes_matrix_controls() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "vla-panda-optimized-repeatability",
            "input",
            "--output-directory",
            "output",
            "--baseline-repeats",
            "2",
            "--load-workers",
            "3",
        ]
    )
    assert args.input_directory == Path("input")
    assert args.output_directory == Path("output")
    assert args.baseline_repeats == 2
    assert args.load_workers == 3


def test_repeatability_rejects_tampered_trial_log(tmp_path: Path) -> None:
    trial = tmp_path / "trials" / "idle_001"
    trial.mkdir(parents=True)
    stdout = trial / "stdout.log"
    stderr = trial / "stderr.log"
    stdout.write_bytes(b"ok\n")
    stderr.write_bytes(b"")
    record = {
        "schema_version": TRIAL_SCHEMA,
        "trial_id": "idle_001",
        "condition": "idle",
        "replicate": 1,
        "load_workers": 0,
        "started_at_utc": "2026-08-11T00:00:00.000Z",
        "finished_at_utc": "2026-08-11T00:00:00.001Z",
        "wall_time_ms": 1.0,
        "returncode": 1,
        "timed_out": False,
        "valid": False,
        "command": ["python", "-m", "armbench"],
        "cwd": str(tmp_path),
        "environment": {},
        "stdout_file": "trials/idle_001/stdout.log",
        "stdout_sha256": sha256_file(stdout),
        "stderr_file": "trials/idle_001/stderr.log",
        "stderr_sha256": sha256_file(stderr),
        "artifact_directory": None,
        "artifact_manifest_sha256": None,
        "artifact_summary_sha256": None,
        "child_result": None,
        "summary": None,
    }
    _validate_trial_record(tmp_path, tmp_path, record)
    stdout.write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="stdout_file binding"):
        _validate_trial_record(tmp_path, tmp_path, record)
