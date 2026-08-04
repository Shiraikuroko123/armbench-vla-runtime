from __future__ import annotations

import csv
import hashlib
import io
import json
import pathlib

import pytest

import integrations.openpi.measured_age_confirmatory_analysis as module
from integrations.openpi.measured_age_confirmatory_analysis import (
    ANALYSIS_SCHEMA_VERSION,
    AnalysisError,
    _condition_first_from_episode_bytes,
    analyze_artifact,
    analyze_pairs,
    generate_report,
    main,
    validate_report,
)
from integrations.openpi.validate_measured_age_artifact import EPISODE_FIELDS


def _pairs(
    *,
    episodes_per_task: int = 2,
    episode_start: int = 0,
    all_candidate_wins: bool = False,
):
    rows = []
    first = {}
    for task_id in range(10):
        for episode_index in range(episode_start, episode_start + episodes_per_task):
            pair_id = "libero_spatial/task_%03d/episode_%03d/h_05/l_000" % (
                task_id,
                episode_index,
            )
            if all_candidate_wins:
                async_success, aligned_success = False, True
            elif episode_index == 0:
                async_success, aligned_success = False, True
            elif task_id < 5:
                async_success, aligned_success = False, True
            elif task_id < 8:
                async_success, aligned_success = True, True
            else:
                async_success, aligned_success = True, False
            rows.append(
                {
                    "pair_id": pair_id,
                    "task_suite": "libero_spatial",
                    "task_id": task_id,
                    "episode_index": episode_index,
                    "async_success": async_success,
                    "aligned_success": aligned_success,
                    "success_difference": int(aligned_success) - int(async_success),
                }
            )
            first[pair_id] = (
                "async_unguarded"
                if (task_id * episodes_per_task + episode_index - episode_start) % 2 == 0
                else "latency_aligned"
            )
    return rows, first


def _episode_csv(pairs, first_by_pair) -> bytes:
    rows = []
    order = 0
    for pair in pairs:
        first = first_by_pair[pair["pair_id"]]
        second = (
            "latency_aligned" if first == "async_unguarded" else "async_unguarded"
        )
        for mode in (first, second):
            row = {field: "" for field in EPISODE_FIELDS}
            row.update(
                {
                    "pair_id": pair["pair_id"],
                    "mode": mode,
                    "condition_order": str(order),
                }
            )
            rows.append(row)
            order += 1
    with io.StringIO(newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPISODE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return handle.getvalue().encode("utf-8")


def test_all_candidate_wins_have_exact_task_results() -> None:
    pairs, first = _pairs(episodes_per_task=1, all_candidate_wins=True)
    result = analyze_pairs(pairs, first)
    assert result["primary"]["candidate_wins"] == 10
    assert result["primary"]["reference_wins"] == 0
    assert result["primary"]["exact_p"] == 0.001953125
    assert result["primary"]["risk_difference"] == 1.0
    bootstrap = result["secondary"]["task_cluster_bootstrap"]
    assert bootstrap["resamples"] == 10_000
    assert bootstrap["seed"] == 20260805
    assert bootstrap["percentile95_low"] == 1.0
    assert bootstrap["percentile95_high"] == 1.0
    sign_flip = result["secondary"]["exact_task_sign_flip"]
    assert sign_flip["enumerated_assignments"] == 1024
    assert sign_flip["extreme_assignments"] == 2
    assert sign_flip["exact_p"] == 2 / 1024.0
    assert all(
        row["risk_difference"] == 1.0
        for row in result["secondary"]["leave_one_task_out"]
    )


def test_task_rows_loto_and_condition_first_are_complete() -> None:
    pairs, first = _pairs()
    result = analyze_pairs(pairs, first)
    assert result["cohort"]["tasks"] == 10
    assert result["cohort"]["pairs"] == 20
    assert result["cohort"]["condition_first_counts"] == {
        "async_unguarded": 10,
        "latency_aligned": 10,
    }
    assert len(result["secondary"]["per_task"]) == 10
    assert len(result["secondary"]["leave_one_task_out"]) == 10
    assert len(result["secondary"]["condition_first"]) == 2
    task_zero = result["secondary"]["per_task"][0]
    assert task_zero["candidate_wins"] == 2
    assert task_zero["reference_wins"] == 0
    assert task_zero["risk_difference"] == 1.0
    omitted_zero = result["secondary"]["leave_one_task_out"][0]
    assert omitted_zero["omitted_task_id"] == 0
    assert omitted_zero["remaining_tasks"] == 9
    assert omitted_zero["remaining_pairs"] == 18


def test_cluster_bootstrap_is_reproducible() -> None:
    pairs, first = _pairs()
    first_result = analyze_pairs(pairs, first)
    second_result = analyze_pairs(list(reversed(pairs)), first)
    assert (
        first_result["secondary"]["task_cluster_bootstrap"]
        == second_result["secondary"]["task_cluster_bootstrap"]
    )


def test_condition_first_is_recovered_from_episode_order() -> None:
    pairs, first = _pairs(episodes_per_task=1)
    assert _condition_first_from_episode_bytes(_episode_csv(pairs, first)) == first


@pytest.mark.parametrize("mutation", ["missing_task", "unbalanced", "one_stratum"])
def test_invalid_confirmatory_matrices_fail(mutation: str) -> None:
    pairs, first = _pairs()
    if mutation == "missing_task":
        pairs = [row for row in pairs if row["task_id"] != 9]
        first = {row["pair_id"]: first[row["pair_id"]] for row in pairs}
    elif mutation == "unbalanced":
        removed = pairs.pop()
        first.pop(removed["pair_id"])
    else:
        first = {pair_id: "async_unguarded" for pair_id in first}
    with pytest.raises(AnalysisError):
        analyze_pairs(pairs, first)


def _base_analysis(pairs, snapshots):
    wins = sum(row["success_difference"] > 0 for row in pairs)
    losses = sum(row["success_difference"] < 0 for row in pairs)
    return {
        "schema_version": module.BASE_ANALYSIS_SCHEMA_VERSION,
        "source": {
            "artifact": "artifact",
            "source_schema_version": "armbench.pi05_libero_measured_age.v2",
            "source_manifest_sha256": hashlib.sha256(snapshots["manifest.json"]).hexdigest(),
            "per_episode_sha256": hashlib.sha256(snapshots["per_episode.csv"]).hexdigest(),
            "per_query_sha256": hashlib.sha256(snapshots["per_query.csv"]).hexdigest(),
            "validator_schema_version": "validator.v1",
            "validator_checks": ["strict"],
        },
        "success": {
            "paired": {
                "mcnemar_exact_two_sided_p": module.base_analysis._mcnemar_exact_p(
                    wins, losses
                )
            }
        },
    }


def test_artifact_analysis_reuses_strict_base_validation(monkeypatch, tmp_path) -> None:
    pairs, first = _pairs(episodes_per_task=12, episode_start=5)
    snapshots = {
        "manifest.json": b"manifest",
        "per_episode.csv": _episode_csv(pairs, first),
        "per_query.csv": b"queries",
    }
    inherited = _base_analysis(pairs, snapshots)
    calls = {"base": 0, "stable": 0}

    def fake_base(*args, **kwargs):
        calls["base"] += 1
        return inherited, pairs

    def fake_stable(*args, **kwargs):
        calls["stable"] += 1
        return snapshots, {"schema_version": "validator.v1"}

    monkeypatch.setattr(module.base_analysis, "analyze_artifact", fake_base)
    monkeypatch.setattr(module.base_analysis, "_stable_source", fake_stable)
    result = analyze_artifact(tmp_path)
    assert result["valid"] is True
    assert result["primary"]["exact_p"] == inherited["success"]["paired"][
        "mcnemar_exact_two_sided_p"
    ]
    assert calls == {"base": 1, "stable": 1}


def test_artifact_analysis_rejects_nonfrozen_pilot_cohort(monkeypatch, tmp_path) -> None:
    pairs, first = _pairs()
    snapshots = {
        "manifest.json": b"manifest",
        "per_episode.csv": _episode_csv(pairs, first),
        "per_query.csv": b"queries",
    }
    inherited = _base_analysis(pairs, snapshots)
    monkeypatch.setattr(
        module.base_analysis,
        "analyze_artifact",
        lambda *args, **kwargs: (inherited, pairs),
    )
    monkeypatch.setattr(
        module.base_analysis,
        "_stable_source",
        lambda *args, **kwargs: (snapshots, {"schema_version": "validator.v1"}),
    )

    with pytest.raises(AnalysisError, match="120 pairs"):
        analyze_artifact(tmp_path)


def _analysis_for_report():
    pairs, first = _pairs()
    core = analyze_pairs(pairs, first)
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "valid": True,
        "implementation": {},
        "source": {"source_manifest_sha256": "a" * 64},
        **core,
        "statistics": {},
        "claim_boundary": "Simulation-only test fixture.",
    }


def test_report_is_transactional_and_tamper_evident(monkeypatch, tmp_path) -> None:
    analysis = _analysis_for_report()
    monkeypatch.setattr(module, "analyze_artifact", lambda _artifact: analysis)
    output = tmp_path / "report"
    assert generate_report(tmp_path / "source", output) == analysis
    assert validate_report(output) == {"valid": True, "errors": []}
    assert set(path.name for path in output.iterdir()) == set(
        module.EXPECTED_OUTPUT_FILES
    ) | {"manifest.json"}
    with pytest.raises(AnalysisError, match="already exists"):
        generate_report(tmp_path / "source", output)
    with (output / "per_task.csv").open("a", encoding="utf-8") as handle:
        handle.write("tamper\n")
    invalid = validate_report(output)
    assert invalid["valid"] is False
    assert any("per_task.csv" in error for error in invalid["errors"])


def test_cli_validation_exit_codes(monkeypatch, tmp_path, capsys) -> None:
    analysis = _analysis_for_report()
    monkeypatch.setattr(module, "analyze_artifact", lambda _artifact: analysis)
    output = tmp_path / "report"
    assert main(
        ["analyze", str(tmp_path / "source"), "--output-directory", str(output)]
    ) == 0
    assert capsys.readouterr().out.strip() == "VALID"
    assert main(["validate", str(output), "--json"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed == {"errors": [], "valid": True}
    (output / "summary.md").write_text("changed", encoding="utf-8")
    assert main(["validate", str(output)]) == 2
    assert capsys.readouterr().out.strip() == "INVALID"
