from __future__ import annotations

import copy
import csv
import hashlib
import json
import pathlib

import pytest

from integrations.openpi import measured_age_dashboard as dashboard
from integrations.openpi.measured_age_analysis import (
    ANALYSIS_SCHEMA_VERSION,
    ANALYZER_SOURCE,
    VALIDATOR_SOURCE,
)
from integrations.openpi.validate_measured_age_artifact import (
    EPISODE_FIELDS,
    SCHEMA_VERSION as SOURCE_SCHEMA_VERSION,
    ValidationReport,
)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: pathlib.Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: pathlib.Path, fields, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _refresh_manifest(root: pathlib.Path, schema: str) -> None:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json" and path.suffix != ".tmp":
            files[path.relative_to(root).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    _write_json(root / "manifest.json", {"schema_version": schema, "files": files})


def _pair() -> dict:
    row = {
        "pair_id": "libero_spatial/task_000/episode_000/h_05/l_000",
        "task_suite": "libero_spatial",
        "task_id": 0,
        "episode_index": 0,
        "replan_steps": 5,
        "seed": 7,
        "initial_state_sha256": "a" * 64,
        "async_success": False,
        "aligned_success": True,
        "success_difference": 1,
    }
    for field in dashboard.BURDEN_FIELDS:
        row["async_%s" % field] = 1 if field == "policy_queries" else 0
        row["aligned_%s" % field] = 1 if field == "policy_queries" else 0
        row["%s_difference" % field] = 0
    return row


def _episode(pair: dict, mode: str, video_path: str) -> dict:
    row = {field: "" for field in EPISODE_FIELDS}
    row.update(
        {
            "schema_version": SOURCE_SCHEMA_VERSION,
            "episode_id": "%s/%s" % (pair["pair_id"], mode),
            "pair_id": pair["pair_id"],
            "condition_order": 0 if mode == "async_unguarded" else 1,
            "task_suite": pair["task_suite"],
            "task_id": pair["task_id"],
            "episode_index": pair["episode_index"],
            "task_description": "pick up the black bowl",
            "mode": mode,
            "latency_source": "measured_wall",
            "replan_steps": pair["replan_steps"],
            "latency_steps": 0,
            "seed": pair["seed"],
            "success": pair["%s_success" % ("async" if mode == "async_unguarded" else "aligned")],
            "initial_state_sha256": pair["initial_state_sha256"],
            "policy_queries": 1,
            "video_required": True,
            "video_path": video_path,
        }
    )
    return row


def _analysis(source: pathlib.Path, pair: dict, checks) -> dict:
    project_root = pathlib.Path(dashboard.__file__).resolve().parents[2]
    timing = []
    for mode, p95, maximum in (
        ("async_unguarded", 142.0, 180.0),
        ("latency_aligned", 145.0, 190.0),
    ):
        timing.append(
            {
                "mode": mode,
                "queries": 1,
                "observation_age_ms": {"p50": p95, "p95": p95, "max": maximum},
                "inference_latency_ms": {"p50": 80.0, "p95": 80.0, "max": 80.0},
                "deadline_misses": 0,
                "deadline_miss_rate_per_query": 0.0,
                "horizon_overruns": 0,
                "horizon_overrun_rate_per_query": 0.0,
                "hold_refresh_queries": 0,
                "fail_closed_queries": 0,
            }
        )
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "implementation": {
            "analyzer_source": ANALYZER_SOURCE,
            "analyzer_sha256": _sha256(project_root / ANALYZER_SOURCE),
            "validator_source": VALIDATOR_SOURCE,
            "validator_sha256": _sha256(project_root / VALIDATOR_SOURCE),
            "python_version": "fixture",
            "numpy_version": "fixture",
        },
        "source": {
            "artifact": str(source.resolve()),
            "source_schema_version": SOURCE_SCHEMA_VERSION,
            "source_manifest_sha256": _sha256(source / "manifest.json"),
            "per_episode_sha256": _sha256(source / "per_episode.csv"),
            "per_query_sha256": _sha256(source / "per_query.csv"),
            "validator_schema_version": "armbench.measured_age_artifact_validation.v1",
            "validator_checks": list(checks),
        },
        "cohort": {
            "rollouts": 2,
            "pairs": 1,
            "queries": 2,
            "task_suites": ["libero_spatial"],
            "replan_steps": [5],
            "age_rounding": "ceil",
            "deadline_ms": 250.0,
        },
        "success": {
            "async_unguarded": {
                "successes": 0, "rollouts": 1, "rate": 0.0,
                "wilson95_low": 0.0, "wilson95_high": 0.793,
            },
            "latency_aligned": {
                "successes": 1, "rollouts": 1, "rate": 1.0,
                "wilson95_low": 0.207, "wilson95_high": 1.0,
            },
            "paired": {
                "pairs": 1, "rate_difference": 1.0,
                "paired_bootstrap95_low": 1.0, "paired_bootstrap95_high": 1.0,
                "candidate_wins": 1, "reference_wins": 0, "ties": 0,
                "mcnemar_exact_two_sided_p": 1.0,
            },
        },
        "timing": timing,
        "runtime_burden": [],
        "statistics": {
            "confidence_level": 0.95,
            "success_rate_interval": "Wilson score interval",
            "paired_difference_interval": "paired percentile bootstrap of the mean",
            "paired_test": "exact two-sided McNemar binomial test",
            "bootstrap_seed": 19,
            "bootstrap_resamples": 500,
            "multiplicity_adjustment": "none; one prespecified success comparison",
        },
        "claim_boundary": (
            "Simulation-only measured-age analysis. Inference remains blocking and "
            "controller catch-up is simulated after response arrival."
        ),
    }


@pytest.fixture
def dashboard_fixture(tmp_path: pathlib.Path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "videos").mkdir()
    (source / "videos" / "async.mp4").write_bytes(b"async-video")
    (source / "videos" / "aligned.mp4").write_bytes(b"aligned-video")
    pair = _pair()
    _write_csv(
        source / "per_episode.csv",
        EPISODE_FIELDS,
        [
            _episode(pair, "async_unguarded", "videos/async.mp4"),
            _episode(pair, "latency_aligned", "videos/aligned.mp4"),
        ],
    )
    (source / "per_query.csv").write_text("query\n0\n", encoding="utf-8")
    _refresh_manifest(source, SOURCE_SCHEMA_VERSION)

    checks = ("manifest coverage and hashes", "paired measured-age cohort")
    report = ValidationReport(str(source), True, (), (), checks)
    analysis = _analysis(source, pair, checks)
    analysis_root = tmp_path / "analysis"
    analysis_root.mkdir()
    _write_json(analysis_root / "analysis.json", analysis)
    _write_csv(analysis_root / "per_pair.csv", dashboard.PER_PAIR_FIELDS, [pair])
    (analysis_root / "summary.md").write_text("# fixture\n", encoding="utf-8")
    _refresh_manifest(analysis_root, ANALYSIS_SCHEMA_VERSION)

    monkeypatch.setattr(dashboard, "validate_artifact", lambda _root: report)
    monkeypatch.setattr(
        dashboard,
        "analyze_artifact",
        lambda _root, bootstrap_resamples, bootstrap_seed: (
            copy.deepcopy(analysis), [copy.deepcopy(pair)]
        ),
    )
    return source, analysis_root, analysis, pair


def test_valid_fixture_builds_offline_paired_video_dashboard(
    dashboard_fixture, tmp_path: pathlib.Path
) -> None:
    source, analysis_root, analysis, _pair_row = dashboard_fixture
    output = tmp_path / "report" / "index.html"

    result = dashboard.build_dashboard(source, analysis_root, output)

    assert result["rollouts"] == 2
    assert result["pairs"] == 1
    assert result["queries"] == 2
    assert result["videos_verified"] == 2
    html = output.read_text(encoding="utf-8")
    assert "Measured-Age VLA Pilot" in html
    assert "PILOT / SIMULATION / BLOCKING INFERENCE" in html
    assert "async.mp4" in html and "aligned.mp4" in html
    assert analysis["source"]["per_episode_sha256"] in html
    assert "Age P95" in html and "Deadline" in html and "Hold-refresh" in html


def test_exact_frozen_matrix_is_labeled_confirmatory() -> None:
    pairs = [
        {
            "task_suite": "libero_spatial",
            "task_id": task_id,
            "episode_index": episode_index,
            "replan_steps": 5,
            "seed": 7,
        }
        for task_id in range(10)
        for episode_index in range(5, 17)
    ]

    presentation = dashboard._evidence_presentation(pairs)

    assert presentation == {
        "stage": "confirmatory",
        "title": "Measured-Age VLA Confirmatory Study",
        "scope_label": "CONFIRMATORY / SIMULATION / BLOCKING INFERENCE",
    }
    pairs[0] = {**pairs[0], "episode_index": 0}
    assert dashboard._evidence_presentation(pairs)["stage"] == "pilot"


def test_invalid_source_fails_before_output(
    dashboard_fixture, tmp_path: pathlib.Path, monkeypatch
) -> None:
    source, analysis_root, _analysis_value, _pair_row = dashboard_fixture
    monkeypatch.setattr(
        dashboard,
        "validate_artifact",
        lambda _root: ValidationReport(str(source), False, ("tampered",), (), ()),
    )
    output = tmp_path / "index.html"

    with pytest.raises(ValueError, match="failed independent validation"):
        dashboard.build_dashboard(source, analysis_root, output)
    assert not output.exists()


def test_source_csv_tamper_breaks_analysis_binding(
    dashboard_fixture, tmp_path: pathlib.Path
) -> None:
    source, analysis_root, _analysis_value, _pair_row = dashboard_fixture
    with (source / "per_episode.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")

    with pytest.raises(ValueError, match="source hash mismatch: per_episode_sha256"):
        dashboard.build_dashboard(source, analysis_root, tmp_path / "index.html")


def test_resigned_analysis_tamper_fails_fresh_recomputation(
    dashboard_fixture, tmp_path: pathlib.Path
) -> None:
    source, analysis_root, _analysis_value, _pair_row = dashboard_fixture
    recorded = json.loads((analysis_root / "analysis.json").read_text(encoding="utf-8"))
    recorded["success"]["latency_aligned"]["successes"] = 0
    _write_json(analysis_root / "analysis.json", recorded)
    _refresh_manifest(analysis_root, ANALYSIS_SCHEMA_VERSION)

    with pytest.raises(ValueError, match="disagrees with fresh source recomputation"):
        dashboard.build_dashboard(source, analysis_root, tmp_path / "index.html")


def test_historical_validator_snapshot_preserves_portable_recomputation(
    dashboard_fixture, tmp_path: pathlib.Path
) -> None:
    source, _analysis_root, analysis, _pair_row = dashboard_fixture
    frozen_validator = (
        source
        / "provenance"
        / "armbench_source"
        / pathlib.PurePosixPath(dashboard.VALIDATOR_SOURCE)
    )
    frozen_validator.parent.mkdir(parents=True)
    frozen_validator.write_bytes(b"historical-validator-snapshot\n")
    _refresh_manifest(source, SOURCE_SCHEMA_VERSION)

    recorded = copy.deepcopy(analysis)
    recorded["source"]["source_manifest_sha256"] = _sha256(
        source / "manifest.json"
    )
    recorded["source"]["validator_checks"] = ["historical strict checks"]
    recorded["implementation"]["validator_sha256"] = _sha256(frozen_validator)
    current_report = {
        "schema_version": recorded["source"]["validator_schema_version"],
        "checks": ["current strict checks", "new compatible check"],
    }

    verified = dashboard._validate_source_bindings(
        source, recorded, current_report
    )
    assert verified["validator_sha256"] == _sha256(frozen_validator)

    recomputed = copy.deepcopy(recorded)
    recomputed["source"]["validator_checks"] = current_report["checks"]
    project_root = pathlib.Path(dashboard.__file__).resolve().parents[2]
    recomputed["implementation"]["validator_sha256"] = _sha256(
        project_root / dashboard.VALIDATOR_SOURCE
    )
    assert dashboard._portable_analysis_equal(recorded, recomputed)


def test_missing_or_empty_video_fails_closed(
    dashboard_fixture, tmp_path: pathlib.Path
) -> None:
    source, analysis_root, _analysis_value, _pair_row = dashboard_fixture
    (source / "videos" / "aligned.mp4").write_bytes(b"")

    with pytest.raises(ValueError, match="required video is missing or empty"):
        dashboard.build_dashboard(source, analysis_root, tmp_path / "index.html")


def test_video_reference_falls_back_to_file_uri_across_windows_drives(
    dashboard_fixture, tmp_path: pathlib.Path, monkeypatch
) -> None:
    source, _analysis_root, _analysis_value, _pair_row = dashboard_fixture

    def cross_drive(*_args, **_kwargs):
        raise ValueError("other drive")

    monkeypatch.setattr(dashboard.os.path, "relpath", cross_drive)

    reference = dashboard._relative_video(
        source, tmp_path, "videos/aligned.mp4"
    )

    assert reference == (source / "videos/aligned.mp4").resolve().as_uri()


@pytest.mark.parametrize("mutation", ["extra", "bad_sha", "omitted"])
def test_analysis_manifest_coverage_and_hash_fail_closed(
    dashboard_fixture, tmp_path: pathlib.Path, mutation: str
) -> None:
    source, analysis_root, _analysis_value, _pair_row = dashboard_fixture
    manifest_path = analysis_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "extra":
        (analysis_root / "unregistered.txt").write_text("extra", encoding="utf-8")
    elif mutation == "bad_sha":
        manifest["files"]["summary.md"]["sha256"] = "0" * 64
        _write_json(manifest_path, manifest)
    else:
        del manifest["files"]["summary.md"]
        _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="coverage mismatch|SHA-256 mismatch"):
        dashboard.build_dashboard(source, analysis_root, tmp_path / "index.html")


def test_cli_reports_invalid_and_does_not_open_browser(
    dashboard_fixture, tmp_path: pathlib.Path, monkeypatch, capsys
) -> None:
    source, analysis_root, _analysis_value, _pair_row = dashboard_fixture
    monkeypatch.setattr(
        dashboard,
        "validate_artifact",
        lambda _root: ValidationReport(str(source), False, ("tampered",), (), ()),
    )
    opened = []
    monkeypatch.setattr(dashboard.webbrowser, "open", opened.append)

    code = dashboard.main(
        [str(source), str(analysis_root), "--output", str(tmp_path / "index.html"), "--open"]
    )

    assert code == 2
    assert json.loads(capsys.readouterr().out)["valid"] is False
    assert opened == []


def test_cli_opens_browser_only_after_success(
    dashboard_fixture, tmp_path: pathlib.Path, monkeypatch, capsys
) -> None:
    source, analysis_root, _analysis_value, _pair_row = dashboard_fixture
    output = tmp_path / "index.html"
    opened = []
    monkeypatch.setattr(dashboard.webbrowser, "open", opened.append)

    code = dashboard.main(
        [str(source), str(analysis_root), "--output", str(output), "--open"]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
    assert opened == [output.resolve().as_uri()]
