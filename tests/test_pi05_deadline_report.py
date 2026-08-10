from __future__ import annotations

import json
import pathlib

import pytest

from scripts import build_pi05_deadline_report as report


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _seed8_artifacts() -> list[pathlib.Path]:
    return [
        PROJECT_ROOT
        / "evidence"
        / "pi05_libero_spatial_deadline150_seed8_40_20260810_001",
        PROJECT_ROOT
        / "evidence"
        / "pi05_libero_spatial_deadline175_seed8_40_20260810_001",
    ]


def test_wilson_interval_handles_boundary_counts() -> None:
    lower, upper = report.wilson_interval(0, 40)
    assert lower == pytest.approx(0.0)
    assert upper == pytest.approx(0.0876216, rel=1e-5)

    lower, upper = report.wilson_interval(37, 40)
    assert lower == pytest.approx(0.8014, rel=1e-4)
    assert upper == pytest.approx(0.9742, rel=1e-4)


def test_report_recomputes_tick_level_and_response_level_metrics() -> None:
    summary = report.build_summary(_seed8_artifacts())
    cells = {cell["deadline_ms"]: cell for cell in summary["cells"]}

    assert summary["artifact_count"] == 2
    assert summary["total_rollouts"] == 80
    assert cells[150.0]["task_successes"] == 0
    assert cells[150.0]["deadline_hold_ticks"] == 6338
    assert cells[150.0]["response_level_deadline_rejections"] == 2
    assert cells[175.0]["task_successes"] == 37
    assert cells[175.0]["deadline_hold_ticks"] == 501
    assert cells[175.0]["response_level_deadline_rejections"] == 0

    [comparison] = summary["adjacent_deadline_comparisons"]
    assert comparison["task_success_rate_difference"] == pytest.approx(0.925)
    assert comparison["execute_duty_cycle_difference"] == pytest.approx(
        4060 / 4681 - 2342 / 8800
    )


def test_rendered_report_is_deterministic_and_checkable(tmp_path: pathlib.Path) -> None:
    summary = report.build_summary(_seed8_artifacts())
    first = report.render_outputs(summary)
    second = report.render_outputs(summary)
    assert first == second

    output = tmp_path / "report"
    report.write_outputs(output, first)
    report.check_outputs(output, second)
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved["artifact_count"] == 2

    (output / "summary.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        report.check_outputs(output, second)
