from __future__ import annotations

import csv
import hashlib
import json
import pathlib

import pytest

import integrations.openpi.cross_suite_dashboard as dashboard
from integrations.openpi.cross_suite_external_analysis import (
    ANALYSIS_SCHEMA_VERSION,
    FROZEN_RUN_IDS,
    FROZEN_SUITES,
)


def _write_json(path: pathlib.Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: pathlib.Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_record(path: pathlib.Path):
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _write_analysis_manifest(analysis_root: pathlib.Path) -> None:
    analysis = json.loads((analysis_root / "analysis.json").read_text(encoding="utf-8"))
    source_fields = (
        "task_suite",
        "run_id",
        "per_episode_csv_sha256",
        "root_manifest_sha256",
        "evaluation_manifest_sha256",
        "resolved_protocol_sha256",
        "environment_sha256",
    )
    files = {
        path.name: _manifest_record(path)
        for path in sorted(analysis_root.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    _write_json(
        analysis_root / "manifest.json",
        {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "sources": [
                {field: source[field] for field in source_fields}
                for source in analysis["sources"]
            ],
            "files": files,
        },
    )


def _fixture(tmp_path: pathlib.Path):
    evidence = tmp_path / "external"
    analysis_root = evidence / "analysis"
    sources = []
    pair_rows = []
    task_rows = []
    suite_rows = []
    run_roots = []
    for suite, run_id in zip(FROZEN_SUITES, FROZEN_RUN_IDS):
        run = evidence / run_id
        evaluation = run / "evaluation"
        videos = evaluation / "videos"
        videos.mkdir(parents=True)
        episode_rows = []
        order = 0
        for task_id in range(10):
            task_rows.append(
                {
                    "task_suite": suite,
                    "task_id": task_id,
                    "async_unguarded_successes": 2,
                    "latency_aligned_successes": 4,
                    "aligned_minus_async_success_rate_difference": 0.4,
                    "async_unguarded_mean_policy_queries": 22.0,
                    "latency_aligned_mean_policy_queries": 14.0,
                }
            )
            for episode_index in range(5):
                pair_id = "%s/task_%03d/episode_%03d/h_05/l_004" % (
                    suite,
                    task_id,
                    episode_index,
                )
                async_success = episode_index < 2
                aligned_success = episode_index < 4
                pair_rows.append(
                    {
                        "task_suite": suite,
                        "task_id": task_id,
                        "episode_index": episode_index,
                        "pair_id": pair_id,
                        "async_unguarded_success": async_success,
                        "latency_aligned_success": aligned_success,
                        "async_unguarded_policy_queries": 20 + episode_index,
                        "latency_aligned_policy_queries": 12 + episode_index,
                    }
                )
                for mode, success, queries in (
                    ("async_unguarded", async_success, 20 + episode_index),
                    ("latency_aligned", aligned_success, 12 + episode_index),
                ):
                    video_name = "%s__task_%03d__episode_%03d__%s.mp4" % (
                        suite,
                        task_id,
                        episode_index,
                        mode,
                    )
                    (videos / video_name).write_bytes(
                        (pair_id + "/" + mode).encode("ascii")
                    )
                    episode_rows.append(
                        {
                            "pair_id": pair_id,
                            "mode": mode,
                            "task_suite": suite,
                            "task_id": task_id,
                            "episode_index": episode_index,
                            "task_description": "fixture task %d" % task_id,
                            "success": success,
                            "policy_queries": queries,
                            "termination_reason": "task_success" if success else "step_limit",
                            "video_required": True,
                            "video_path": "videos/" + video_name,
                            "video_error_type": "",
                            "video_error_message": "",
                            "condition_order": order,
                        }
                    )
                    order += 1
        _write_csv(evaluation / "per_episode.csv", episode_rows)
        _write_json(evaluation / "resolved_protocol.json", {"suite": suite})
        _write_json(evaluation / "environment.json", {"run_id": run_id})
        _write_json(evaluation / "manifest.json", {"fixture": "evaluation"})
        _write_json(run / "manifest.json", {"fixture": "root", "complete": True})
        sources.append(
            {
                "task_suite": suite,
                "run_id": run_id,
                "independently_validated_files": 131,
                "videos_verified": 100,
                "per_episode_csv_sha256": _sha256(evaluation / "per_episode.csv"),
                "root_manifest_sha256": _sha256(run / "manifest.json"),
                "evaluation_manifest_sha256": _sha256(evaluation / "manifest.json"),
                "resolved_protocol_sha256": _sha256(evaluation / "resolved_protocol.json"),
                "environment_sha256": _sha256(evaluation / "environment.json"),
            }
        )
        suite_rows.append(
            {
                "task_suite": suite,
                "paired_n": 50,
                "async_unguarded_successes": 20,
                "latency_aligned_successes": 40,
                "async_unguarded_success_rate": 0.4,
                "latency_aligned_success_rate": 0.8,
                "aligned_minus_async_success_rate_difference": 0.4,
                "paired_bootstrap95_low": 0.2,
                "paired_bootstrap95_high": 0.6,
                "aligned_wins": 20,
                "async_unguarded_wins": 0,
                "ties": 30,
                "mcnemar_exact_p": 0.000001,
                "mcnemar_holm_p": 0.000003,
                "holm_reject_alpha_0_05": True,
                "async_unguarded_mean_policy_queries": 22.0,
                "latency_aligned_mean_policy_queries": 14.0,
            }
        )
        run_roots.append(run)

    analysis = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "sources": sources,
        "suite_results": suite_rows,
        "pooled_descriptive": {
            "analysis_scope": "pooled_descriptive_only_no_p_value",
            "async_unguarded_successes": 60,
            "latency_aligned_successes": 120,
            "aligned_minus_async_success_rate_difference": 0.4,
            "paired_bootstrap95_low": 0.3,
            "paired_bootstrap95_high": 0.5,
            "async_unguarded_mean_policy_queries": 22.0,
            "latency_aligned_mean_policy_queries": 14.0,
        },
        "acceptance": {
            "passed": True,
            "videos_verified": 300,
            "runtime_failures": 0,
        },
        "itt": {"rollouts": 300, "pairs": 150},
        "frozen_identity": {
            "policy_config": "pi05_libero",
            "checkpoint_content_sha256": "a" * 64,
            "armbench_run_commit": "b" * 40,
            "temporal_alignment_implementation_commit": "c" * 40,
            "openpi_commit": "d" * 40,
        },
        "claim_boundary": "deterministic 200 ms delay; no pooled p value",
    }
    _write_json(analysis_root / "analysis.json", analysis)
    _write_csv(analysis_root / "suite_results.csv", suite_rows)
    _write_csv(analysis_root / "task_descriptives.csv", task_rows)
    _write_csv(analysis_root / "pooled_descriptive.csv", [analysis["pooled_descriptive"]])
    _write_csv(analysis_root / "per_pair.csv", pair_rows)
    (analysis_root / "summary.md").write_text("# fixture\n", encoding="utf-8")
    _write_analysis_manifest(analysis_root)
    return run_roots, analysis_root


@pytest.fixture
def valid_source_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dashboard,
        "validate_run_manifest",
        lambda _root: {
            "valid": True,
            "complete": True,
            "errors": [],
            "files_checked": 131,
        },
    )


def test_builds_manifest_bound_cross_suite_dashboard(
    tmp_path: pathlib.Path, valid_source_validation: None
) -> None:
    runs, analysis = _fixture(tmp_path)
    output = tmp_path / "reports" / "index.html"

    result = dashboard.build_dashboard(runs, analysis, output)

    assert result == {
        "output": str(output.resolve()),
        "rollouts": 300,
        "matched_pairs": 150,
        "videos_verified": 300,
        "suites": 3,
        "tasks": 30,
    }
    html = output.read_text(encoding="utf-8")
    assert "Cross-Suite Temporal Alignment" in html
    assert '"pairs":[' in html
    assert html.count('"pairId":') == 150
    assert html.count('"video":') == 300
    assert '"scope":"descriptive_only_no_p_value"' in html
    assert "pooled significance claim" in html
    assert "a" * 64 in html
    assert "b" * 40 in html
    assert _sha256(analysis / "manifest.json") in html


def test_rejects_tampered_analysis_before_rendering(
    tmp_path: pathlib.Path, valid_source_validation: None
) -> None:
    runs, analysis = _fixture(tmp_path)
    with (analysis / "per_pair.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    output = tmp_path / "reports" / "index.html"

    with pytest.raises(ValueError, match="cross-suite analysis failed validation"):
        dashboard.build_dashboard(runs, analysis, output)

    assert not output.exists()


def test_rejects_source_hash_mismatch_and_missing_video(
    tmp_path: pathlib.Path, valid_source_validation: None
) -> None:
    runs, analysis = _fixture(tmp_path)
    with (runs[0] / "manifest.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="root manifest SHA-256 mismatch"):
        dashboard.build_dashboard(runs, analysis, tmp_path / "hash.html")

    runs, analysis = _fixture(tmp_path / "missing")
    video = next((runs[2] / "evaluation" / "videos").iterdir())
    video.unlink()
    with pytest.raises(ValueError, match="required video is missing or empty"):
        dashboard.build_dashboard(runs, analysis, tmp_path / "missing.html")


def test_rejects_manifest_bound_video_encoding_error(
    tmp_path: pathlib.Path, valid_source_validation: None
) -> None:
    runs, analysis_root = _fixture(tmp_path)
    source_csv = runs[1] / "evaluation" / "per_episode.csv"
    with source_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows[0]["video_error_type"] = "RuntimeError"
    rows[0]["video_error_message"] = "encoder failed"
    with source_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    analysis_path = analysis_root / "analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["sources"][1]["per_episode_csv_sha256"] = _sha256(source_csv)
    _write_json(analysis_path, analysis)
    _write_analysis_manifest(analysis_root)

    with pytest.raises(ValueError, match="records a video encoding error"):
        dashboard.build_dashboard(runs, analysis_root, tmp_path / "video-error.html")
