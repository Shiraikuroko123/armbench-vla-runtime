from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from integrations.openpi.libero_independent_clock import (
    AGE_ALIGNED_SUFFIX,
    RESPONSE_RELATIVE_CHUNK,
)
from scripts import build_pi05_selection_report as report


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _artifact(
    seed: int,
    mode: str,
    *,
    task_ids: range = range(2),
    episode_indices: range = range(4, 5),
    armbench_commit: str = "c" * 40,
) -> dict:
    records = {}
    for task_id in task_ids:
        for episode_index in episode_indices:
            identity = ("libero_spatial", seed, task_id, episode_index)
            records[identity] = {
                "task_success": mode == AGE_ALIGNED_SUFFIX or task_id == 1,
                "initial_state_sha256": _digest(
                    f"state-{seed}-{task_id}-{episode_index}"
                ),
                "policy_input_sha256": _digest(
                    f"input-{seed}-{task_id}-{episode_index}"
                ),
                "sampling_key_sha256": _digest(f"key-{seed}-{task_id}-{episode_index}"),
                "sampling_noise_sha256": _digest(
                    f"noise-{seed}-{task_id}-{episode_index}"
                ),
                "action_chunk_sha256": _digest(
                    f"action-{seed}-{task_id}-{episode_index}"
                ),
                "execute_ticks": 8 if mode == AGE_ALIGNED_SUFFIX else 7,
                "control_ticks": 10,
                "hold_reasons": {
                    "no_policy_response": 2 if mode == AGE_ALIGNED_SUFFIX else 3
                },
                "action_indices": {"0": 3, "1": 5},
            }
    return {
        "artifact_id": f"seed{seed}-{mode}",
        "evaluation": pathlib.Path(f"seed{seed}-{mode}/evaluation"),
        "manifest_sha256": _digest(f"manifest-{seed}-{mode}"),
        "mode": mode,
        "seed": seed,
        "checkpoint": "gs://openpi-assets/checkpoints/pi05_libero",
        "checkpoint_content_sha256": "a" * 64,
        "openpi_commit": "b" * 40,
        "armbench_commit": armbench_commit,
        "runtime_source_sha256": {"runtime.py": "d" * 64},
        "control_period_ms": 50.0,
        "deadline_ms": 175.0,
        "submit_every_ticks": 1,
        "provider_failures": 0,
        "deadline_exceeded_responses": 0,
        "episodes_with_inference_overlap": len(records),
        "records": records,
    }


def _patch_artifacts(monkeypatch: pytest.MonkeyPatch) -> list[pathlib.Path]:
    artifacts = {
        pathlib.Path(f"seed{seed}-{mode}"): _artifact(seed, mode)
        for seed in (7, 8)
        for mode in (AGE_ALIGNED_SUFFIX, RESPONSE_RELATIVE_CHUNK)
    }

    def load(path: pathlib.Path) -> dict:
        return artifacts[path]

    monkeypatch.setattr(report, "_load_artifact", load)
    return list(artifacts)


def _patch_frozen_artifacts(monkeypatch: pytest.MonkeyPatch) -> list[pathlib.Path]:
    artifacts = {
        pathlib.Path(f"frozen-seed{seed}-{mode}"): _artifact(
            seed,
            mode,
            task_ids=range(10),
            episode_indices=range(4, 8),
            armbench_commit=report.FROZEN_240_RUNTIME_COMMIT,
        )
        for seed in report.FROZEN_240_SEEDS
        for mode in (AGE_ALIGNED_SUFFIX, RESPONSE_RELATIVE_CHUNK)
    }
    monkeypatch.setattr(report, "_load_artifact", artifacts.__getitem__)
    return list(artifacts)


def test_mcnemar_exact_handles_no_and_one_sided_discordance() -> None:
    assert report.mcnemar_exact(0, 0) == 1.0
    assert report.mcnemar_exact(2, 0) == pytest.approx(0.5)
    assert report.mcnemar_exact(5, 5) == 1.0
    with pytest.raises(ValueError, match="nonnegative"):
        report.mcnemar_exact(-1, 0)


def test_selection_report_requires_and_summarizes_query0_pairing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _patch_artifacts(monkeypatch)
    summary = report.build_summary(paths)

    assert summary["artifact_count"] == 4
    assert summary["pair_count"] == 4
    assert summary["total_rollouts"] == 8
    assert summary["pairing_gate"]["valid"] is True
    assert summary["modes"][AGE_ALIGNED_SUFFIX]["successes"] == 4
    assert summary["modes"][RESPONSE_RELATIVE_CHUNK]["successes"] == 2
    assert len(summary["task_seed_blocks"]) == 4
    assert summary["block_robustness"]["block_count"] == 4
    assert summary["block_robustness"]["success_rate_difference"][
        "point_estimate"
    ] == pytest.approx(0.5)
    assert summary["modes"][AGE_ALIGNED_SUFFIX]["episodes_with_inference_overlap"] == 4
    assert summary["modes"][AGE_ALIGNED_SUFFIX]["provider_failures"] == 0
    assert summary["paired_success"] == {
        "both_success": 2,
        "age_aligned_only": 2,
        "response_relative_only": 0,
        "both_failure": 0,
        "success_rate_difference": 0.5,
        "mcnemar_exact_two_sided_p": 0.5,
    }


def test_smoke_profile_preserves_single_pair_use_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = {
        pathlib.Path(mode): _artifact(
            7, mode, task_ids=range(1), episode_indices=range(4, 5)
        )
        for mode in (AGE_ALIGNED_SUFFIX, RESPONSE_RELATIVE_CHUNK)
    }
    monkeypatch.setattr(report, "_load_artifact", artifacts.__getitem__)

    summary = report.build_summary(list(artifacts), profile=report.SMOKE_PROFILE)

    assert summary["pair_count"] == 1
    assert summary["total_rollouts"] == 2
    assert "frozen_matrix_gate" not in summary


def test_frozen_240_profile_accepts_only_registered_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = report.build_summary(
        _patch_frozen_artifacts(monkeypatch), profile=report.FROZEN_240_PROFILE
    )

    assert summary["artifact_count"] == 6
    assert summary["pair_count"] == 120
    assert summary["total_rollouts"] == 240
    assert summary["scoring_profile"] == report.FROZEN_240_PROFILE
    assert summary["frozen_matrix_gate"]["valid"] is True
    assert summary["frozen_matrix_gate"]["pairs_per_seed"] == 40
    assert [row["pairs"] for row in summary["seed_summaries"]] == [40, 40, 40]
    assert len(summary["task_seed_blocks"]) == 30
    assert summary["block_robustness"]["block_count"] == 30


def test_frozen_240_profile_rejects_wrong_runtime_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _patch_frozen_artifacts(monkeypatch)
    original = report._load_artifact

    def wrong_commit(path: pathlib.Path) -> dict:
        artifact = original(path)
        artifact["armbench_commit"] = "f" * 40
        return artifact

    monkeypatch.setattr(report, "_load_artifact", wrong_commit)
    with pytest.raises(ValueError, match="requires ArmBench runtime commit"):
        report.build_summary(paths, profile=report.FROZEN_240_PROFILE)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_seed", "exactly seeds 7/8/9"),
        ("wrong_episode_matrix", "episode matrix mismatch"),
    ],
)
def test_frozen_240_profile_rejects_incomplete_registration(
    monkeypatch: pytest.MonkeyPatch, mutation: str, message: str
) -> None:
    paths = _patch_frozen_artifacts(monkeypatch)
    if mutation == "missing_seed":
        paths = [path for path in paths if "seed9" not in path.name]
    else:
        original = report._load_artifact

        def wrong_matrix(path: pathlib.Path) -> dict:
            artifact = original(path)
            if artifact["seed"] == 8 and artifact["mode"] == AGE_ALIGNED_SUFFIX:
                artifact["records"].pop(("libero_spatial", 8, 9, 7))
            return artifact

        monkeypatch.setattr(report, "_load_artifact", wrong_matrix)

    with pytest.raises(ValueError, match=message):
        report.build_summary(paths, profile=report.FROZEN_240_PROFILE)


def test_selection_report_rejects_pairing_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _patch_artifacts(monkeypatch)
    original = report._load_artifact

    def mismatched(path: pathlib.Path) -> dict:
        artifact = original(path)
        if artifact["seed"] == 7 and artifact["mode"] == RESPONSE_RELATIVE_CHUNK:
            identity = next(iter(artifact["records"]))
            artifact["records"][identity]["policy_input_sha256"] = "f" * 64
        return artifact

    monkeypatch.setattr(report, "_load_artifact", mismatched)
    with pytest.raises(ValueError, match="policy_input_sha256"):
        report.build_summary(paths)


def test_selection_report_outputs_are_deterministic_and_checkable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    summary = report.build_summary(_patch_artifacts(monkeypatch))
    first = report.render_outputs(summary)
    second = report.render_outputs(summary)
    assert first == second

    output = tmp_path / "selection-report"
    report.write_outputs(output, first)
    report.check_outputs(output, second)
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved["pair_count"] == 4

    (output / "pairs.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        report.check_outputs(output, second)
