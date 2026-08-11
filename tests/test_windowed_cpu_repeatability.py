from __future__ import annotations

from pathlib import Path

import pytest

from armbench.cli import build_parser
from armbench.vla.windowed_cpu_repeatability import (
    WindowedCPURepeatabilityConfig,
    _condition_summary,
    _summary,
)


def _record(condition: str, replicate: int, execute: int) -> dict[str, object]:
    profiles = {}
    for profile in ("full_chunk_h10", "certified_window_h1"):
        profiles[profile] = {
            "cases": 3,
            "execute": execute,
            "hold": 3 - execute,
            "constraint_satisfied_candidates": 3,
            "unsafe_windows_published": 0,
            "partial_windows_exposed": 0,
            "source_chunk_prefix_publications": execute,
            "software_budget_exceeded": 3 - execute,
            "response_deadline_exceeded": 0,
            "p50_supervisor_latency_ms": 2.0,
            "p95_supervisor_latency_ms": 3.0,
            "maximum_supervisor_latency_ms": 4.0,
            "p50_worker_latency_ms": 1.0,
            "p95_worker_latency_ms": 2.0,
            "maximum_worker_latency_ms": 3.0,
        }
    return {
        "trial_id": f"{condition}_{replicate:03d}",
        "condition": condition,
        "replicate": replicate,
        "valid": True,
        "summary": {
            "by_profile": profiles,
            "paired_comparison": {
                "window_minus_full_execute": 0,
                "window_minus_full_constraint_candidates": 0,
                "window_is_repeatability_candidate": True,
                "source_chunk_atomicity_changed": True,
                "window_atomicity_preserved": True,
            },
        },
        "wall_time_ms": 100.0 + replicate,
    }


def test_condition_summary_recomputes_counts() -> None:
    records = [_record("idle", 1, 3)]
    summary = _condition_summary(records)
    assert summary["valid_trials"] == 1
    assert summary["profiles"]["certified_window_h1"]["execute_counts"] == [3]
    assert summary["profiles"]["full_chunk_h10"]["unsafe_windows_all_zero"]


def test_repeatability_summary_recomputes_condition_statistics() -> None:
    config = WindowedCPURepeatabilityConfig(
        baseline_repeats=1,
        load_repeats=1,
        load_workers=1,
        chunks=1,
    )
    records = [_record("idle", 1, 3), _record("cpu_load", 1, 0)]
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
    assert summary["conditions"]["idle"]["profiles"]["certified_window_h1"][
        "execute_counts"
    ] == [3]
    assert summary["conditions"]["cpu_load"]["profiles"]["certified_window_h1"][
        "execute_counts"
    ] == [0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"baseline_repeats": -1},
        {"load_repeats": -1},
        {"load_repeats": 1, "load_workers": 0},
        {"baseline_repeats": 0, "load_repeats": 0},
        {"chunks": 0},
        {"trial_timeout_s": 0.0},
        {"response_deadline_ms": "200"},
    ],
)
def test_repeatability_configuration_fails_closed(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        WindowedCPURepeatabilityConfig(**kwargs)


def test_repeatability_parser_exposes_matrix_controls() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "vla-panda-windowed-repeatability",
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
