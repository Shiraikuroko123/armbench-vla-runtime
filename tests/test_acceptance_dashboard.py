from __future__ import annotations

import csv
import json
import pathlib

import pytest

import integrations.openpi.acceptance_dashboard as dashboard


def _write_csv(path: pathlib.Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: pathlib.Path):
    run_root = tmp_path / "evidence" / "sample_run" / "run"
    analysis_root = run_root.parent / "analysis"
    evaluation = run_root / "evaluation"
    videos = evaluation / "videos"
    videos.mkdir(parents=True)
    (videos / "async.mp4").write_bytes(b"async-video")
    (videos / "aligned.mp4").write_bytes(b"aligned-video")

    pair_id = "libero_spatial/task_000/episode_000/h_05/l_004"
    pair_rows = [
        {
            "pair_id": pair_id,
            "task_id": "0",
            "episode_index": "0",
            "latency_steps": "4",
            "async_unguarded_success": "False",
            "latency_aligned_success": "True",
            "async_unguarded_policy_queries": "23",
            "latency_aligned_policy_queries": "13",
        }
    ]
    _write_csv(analysis_root / "per_pair.csv", pair_rows)

    episode_rows = []
    for order, (mode, success, video) in enumerate(
        (
            ("async_unguarded", "False", "videos/async.mp4"),
            ("latency_aligned", "True", "videos/aligned.mp4"),
        )
    ):
        episode_rows.append(
            {
                "pair_id": pair_id,
                "mode": mode,
                "task_description": "put the bowl on the plate",
                "injected_latency_ms": "200.0",
                "success": success,
                "termination_reason": "task_success" if success == "True" else "max_steps",
                "video_required": "True",
                "video_path": video,
                "condition_order": str(order),
            }
        )
    _write_csv(evaluation / "per_episode.csv", episode_rows)

    analysis = {
        "claim_boundary": "deterministic injected delay only",
        "itt": {"rollouts": 2, "pairs": 1},
        "runtime_failures": [],
        "frozen_identity": {
            "policy_config": "pi05_libero",
            "checkpoint_content_sha256": "a" * 64,
            "armbench_run_commit": "b" * 40,
        },
        "source": {"root_manifest_sha256": "c" * 64},
        "latency_strata": [
            {
                "injected_latency_ms": 200.0,
                "analysis_role": "primary",
                "paired_n": 1,
                "async_unguarded_successes": 0,
                "latency_aligned_successes": 1,
                "aligned_minus_async_success_rate_difference": 1.0,
                "paired_bootstrap95_low": 1.0,
                "paired_bootstrap95_high": 1.0,
                "mcnemar_holm_p": 1.0,
                "async_unguarded_mean_policy_queries": 23.0,
                "latency_aligned_mean_policy_queries": 13.0,
            }
        ],
    }
    analysis_root.mkdir(parents=True, exist_ok=True)
    (analysis_root / "analysis.json").write_text(
        json.dumps(analysis), encoding="utf-8"
    )
    return run_root, analysis_root


def test_build_dashboard_verifies_and_embeds_paired_videos(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root, analysis_root = _fixture(tmp_path)
    monkeypatch.setattr(
        dashboard,
        "validate_run_manifest",
        lambda _root: {
            "valid": True,
            "complete": True,
            "errors": [],
            "files_checked": 9,
        },
    )
    output = tmp_path / "reports" / "acceptance" / "index.html"

    result = dashboard.build_dashboard(run_root, analysis_root, output)

    assert result == {
        "output": str(output.resolve()),
        "rollouts": 2,
        "matched_pairs": 1,
        "videos_verified": 2,
        "files_checked": 9,
    }
    html = output.read_text(encoding="utf-8")
    assert "ArmBench VLA 运行时验收台" in html
    assert '"outcome":"aligned_win"' in html
    assert "async.mp4" in html
    assert "aligned.mp4" in html
    assert "put the bowl on the plate" in html


def test_build_dashboard_rejects_missing_required_video(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root, analysis_root = _fixture(tmp_path)
    (run_root / "evaluation" / "videos" / "async.mp4").unlink()
    monkeypatch.setattr(
        dashboard,
        "validate_run_manifest",
        lambda _root: {
            "valid": True,
            "complete": True,
            "errors": [],
            "files_checked": 8,
        },
    )

    with pytest.raises(ValueError, match="required video is missing or empty"):
        dashboard.build_dashboard(
            run_root, analysis_root, tmp_path / "reports" / "index.html"
        )


def test_build_dashboard_rejects_invalid_source_before_rendering(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root, analysis_root = _fixture(tmp_path)
    monkeypatch.setattr(
        dashboard,
        "validate_run_manifest",
        lambda _root: {
            "valid": False,
            "complete": False,
            "errors": ["manifest hash mismatch"],
            "files_checked": 8,
        },
    )

    with pytest.raises(ValueError, match="manifest hash mismatch"):
        dashboard.build_dashboard(
            run_root, analysis_root, tmp_path / "reports" / "index.html"
        )
