from __future__ import annotations

import csv
import json
import pathlib

import pytest

import integrations.openpi.alignment_acceptance as acceptance


def _write_csv(path: pathlib.Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: pathlib.Path):
    run_root = tmp_path / "run"
    analysis_root = tmp_path / "analysis"
    evaluation = run_root / "evaluation"
    videos = evaluation / "videos"
    videos.mkdir(parents=True)
    analysis_root.mkdir()
    (videos / "baseline.mp4").write_bytes(b"baseline-video")
    (videos / "method.mp4").write_bytes(b"method-video")

    checkpoint_sha = "a" * 64
    openpi_commit = "b" * 40
    (run_root / "checkpoint_attestation.json").write_text(
        json.dumps(
            {
                "policy_loaded": True,
                "policy_config": "pi05_libero",
                "checkpoint_uri": "gs://openpi-assets/checkpoints/pi05_libero",
                "checkpoint_content_sha256": checkpoint_sha,
                "checkpoint_file_count": 16,
                "openpi_commit": openpi_commit,
                "openpi_tracked_clean": True,
                "action_horizon": 10,
            }
        ),
        encoding="utf-8",
    )
    (analysis_root / "analysis.json").write_text(
        json.dumps(
            {
                "frozen_identity": {
                    "policy_config": "pi05_libero",
                    "checkpoint": "gs://openpi-assets/checkpoints/pi05_libero",
                    "checkpoint_content_sha256": checkpoint_sha,
                    "openpi_commit": openpi_commit,
                    "armbench_run_commit": "c" * 40,
                },
                "itt": {
                    "rollouts": 300,
                    "pairs": 150,
                    "runtime_failures_retained": 0,
                },
                "latency_strata": [
                    {
                        "analysis_role": "primary",
                        "injected_latency_ms": 200.0,
                        "paired_n": 50,
                        "async_unguarded_successes": 18,
                        "latency_aligned_successes": 50,
                        "aligned_minus_async_success_rate_difference": 0.64,
                        "paired_bootstrap95_low": 0.50,
                        "paired_bootstrap95_high": 0.76,
                        "mcnemar_holm_p": 1.4e-9,
                    }
                ],
                "claim_boundary": "Simulation-only deterministic-delay evidence.",
            }
        ),
        encoding="utf-8",
    )
    pair_id = "libero_spatial/task_000/episode_001/h_05/l_004"
    _write_csv(
        analysis_root / "per_pair.csv",
        [
            {
                "pair_id": pair_id,
                "task_id": "0",
                "episode_index": "1",
                "latency_steps": "4",
                "async_unguarded_success": "False",
                "latency_aligned_success": "True",
                "async_unguarded_runtime_failure": "False",
                "latency_aligned_runtime_failure": "False",
            }
        ],
    )
    _write_csv(
        evaluation / "per_episode.csv",
        [
            {
                "pair_id": pair_id,
                "mode": "async_unguarded",
                "injected_latency_ms": "200.0",
                "task_description": "pick and place",
                "success": "False",
                "policy_queries": "22",
                "video_path": "videos/baseline.mp4",
            },
            {
                "pair_id": pair_id,
                "mode": "latency_aligned",
                "injected_latency_ms": "200.0",
                "task_description": "pick and place",
                "success": "True",
                "policy_queries": "13",
                "video_path": "videos/method.mp4",
            },
        ],
    )
    return run_root, analysis_root


def test_builds_review_report_from_validated_sources(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root, analysis_root = _fixture(tmp_path)
    output = tmp_path / "report" / "index.html"

    def validated_dashboard(_run, _analysis, dashboard_output):
        dashboard_output.parent.mkdir(parents=True)
        dashboard_output.write_text("validated", encoding="utf-8")
        return {
            "output": str(dashboard_output),
            "rollouts": 300,
            "matched_pairs": 150,
            "videos_verified": 300,
            "files_checked": 331,
        }

    monkeypatch.setattr(acceptance, "build_dashboard", validated_dashboard)

    report = acceptance.build_alignment_acceptance(
        run_root, analysis_root, output
    )

    assert report["valid"] is True
    assert report["pi05_identity"]["policy_config"] == "pi05_libero"
    assert report["formal_result"]["baseline_successes"] == 18
    assert report["formal_result"]["method_successes"] == 50
    assert report["evidence"] == {
        "root_and_nested_validation": "valid / complete",
        "protected_files_checked": 331,
        "rollouts": 300,
        "matched_pairs": 150,
        "videos_verified": 300,
        "runtime_failures_retained": 0,
    }
    representative = report["demo"]["representative_pair"]
    assert representative["baseline"]["success"] is False
    assert representative["method"]["success"] is True
    assert pathlib.Path(representative["method"]["video"]).is_file()


def test_rejects_analysis_attestation_identity_mismatch(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root, analysis_root = _fixture(tmp_path)
    analysis_path = analysis_root / "analysis.json"
    value = json.loads(analysis_path.read_text(encoding="utf-8"))
    value["frozen_identity"]["checkpoint_content_sha256"] = "f" * 64
    analysis_path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(
        acceptance,
        "build_dashboard",
        lambda *_args: {
            "videos_verified": 300,
            "files_checked": 331,
        },
    )

    with pytest.raises(ValueError, match="checkpoint_content_sha256"):
        acceptance.build_alignment_acceptance(
            run_root, analysis_root, tmp_path / "index.html"
        )


def test_cli_no_open_does_not_launch_browser(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = {
        "schema_version": acceptance.REPORT_SCHEMA_VERSION,
        "valid": True,
        "demo": {"dashboard_uri": "file:///validated/index.html"},
    }
    monkeypatch.setattr(
        acceptance, "build_alignment_acceptance", lambda *_args: report
    )
    monkeypatch.setattr(
        acceptance.webbrowser,
        "open",
        lambda _uri: (_ for _ in ()).throw(AssertionError("browser opened")),
    )

    assert acceptance.main(["--no-open"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["valid"] is True
