from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

import integrations.openpi.rtc_overlap_analysis as analysis_module
from integrations.openpi import rtc_overlap_pilot
from integrations.openpi.rtc_overlap_analysis import (
    ANALYSIS_SCHEMA_VERSION,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    AnalysisError,
    analyze_artifact,
    generate_report,
    main,
)


METHODS = tuple(rtc_overlap_pilot.V2_OVERLAP_METHODS)
BASELINE = rtc_overlap_pilot.OVERLAP_UNCONDITIONED
PROJECTED = rtc_overlap_pilot.PROJECTED_OVERLAP
RTC = rtc_overlap_pilot.RTC_GUIDED_OVERLAP


def _write_json(path: pathlib.Path, value, *, allow_nan: bool = False) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=allow_nan) + "\n",
        encoding="utf-8",
    )


def _artifact(root: pathlib.Path) -> pathlib.Path:
    root.mkdir()
    cells = rtc_overlap_pilot.build_cells(
        "libero_10",
        list(range(10)),
        [0, 1],
        execute_horizon=5,
        inference_delay_steps=4,
    )
    protocol = {
        "schema_version": rtc_overlap_pilot.SCHEMA_VERSION,
        "planned_rollouts": 60,
        "matrix": [cell.to_dict() for cell in cells],
    }
    episodes = []
    queries = []
    for cell in cells:
        baseline_motion = 0.20 + 0.01 * cell.task_id + 0.001 * cell.episode_index
        baseline_gripper = 0.10 + 0.002 * cell.task_id
        motion_offset = {BASELINE: 0.0, PROJECTED: -0.05, RTC: -0.025}[cell.method]
        gripper_offset = {BASELINE: 0.0, PROJECTED: -0.04, RTC: -0.03}[cell.method]
        success = True
        if cell.method == BASELINE and (cell.task_id, cell.episode_index) == (0, 0):
            success = False
        if cell.method == PROJECTED and (cell.task_id, cell.episode_index) == (1, 0):
            success = False
        state_hash = hashlib.sha256(cell.pair_id.encode("ascii")).hexdigest()
        identity = {
            "schema_version": rtc_overlap_pilot.SCHEMA_VERSION,
            **cell.to_dict(),
        }
        episodes.append(
            {
                **identity,
                "success": success,
                "policy_queries": 2,
                "initial_state_sha256": state_hash,
                "wall_time_s": 1.0,
            }
        )
        queries.extend(
            [
                {
                    **identity,
                    "query_index": 0,
                    "bootstrap": True,
                    "seam_motion_l2": None,
                    "seam_gripper_abs": None,
                },
                {
                    **identity,
                    "query_index": 1,
                    "bootstrap": False,
                    "seam_motion_l2": baseline_motion + motion_offset,
                    "seam_gripper_abs": baseline_gripper + gripper_offset,
                },
            ]
        )
    values = {
        "resolved_protocol.json": protocol,
        "progress.json": {"planned": 60, "completed": 60, "complete": True},
        "episodes.json": episodes,
        "queries.json": queries,
        "environment.json": {"schema_version": rtc_overlap_pilot.SCHEMA_VERSION},
        "summary.json": {"schema_version": rtc_overlap_pilot.SCHEMA_VERSION},
        "transition_descriptor.json": {"schema_version": "test.transition.v1"},
        "manifest.json": {"schema_version": "test.manifest.v1"},
    }
    for relative, value in values.items():
        _write_json(root / relative, value)
    return root


@pytest.fixture
def artifact(tmp_path: pathlib.Path, monkeypatch):
    calls = []

    def validate(path: pathlib.Path):
        calls.append(pathlib.Path(path).resolve())
        return {"schema_version": "test.validator.v1", "valid": True}

    monkeypatch.setattr(rtc_overlap_pilot, "validate_artifact", validate)
    return _artifact(tmp_path / "pilot"), calls


def test_complete_matrix_analysis_is_paired_deterministic_and_task_blocked(
    artifact,
) -> None:
    root, calls = artifact
    first, rows = analyze_artifact(root)
    second, second_rows = analyze_artifact(root)

    assert first == second
    assert rows == second_rows
    assert len(calls) == 4
    assert len(rows) == 20
    assert first["schema_version"] == ANALYSIS_SCHEMA_VERSION
    assert first["source"]["artifact"] == "pilot"
    assert first["statistics"]["bootstrap_resamples"] == BOOTSTRAP_RESAMPLES
    assert first["statistics"]["bootstrap_seed"] == BOOTSTRAP_SEED
    assert first["success"]["methods"][BASELINE]["successes"] == 19
    assert first["success"]["methods"][PROJECTED]["successes"] == 19
    assert first["success"]["methods"][RTC]["successes"] == 20
    projected = first["success"]["contrasts_vs_unconditioned"][PROJECTED]
    assert (projected["candidate_wins"], projected["candidate_losses"], projected["ties"]) == (1, 1, 18)
    assert projected["mcnemar_exact_two_sided_p"] == 1.0
    assert projected["holm_adjusted_p"] == 1.0
    motion = first["seam"]["seam_motion_l2"]["contrasts_vs_unconditioned"]
    assert motion[PROJECTED]["paired_mean_difference"] == pytest.approx(-0.05)
    assert motion[PROJECTED]["paired_median_difference"] == pytest.approx(-0.05)
    assert motion[RTC]["paired_mean_difference"] == pytest.approx(-0.025)
    assert len(first["per_task"]) == 10
    assert len(first["leave_one_task_out"]) == 10
    assert [row["condition_order"] for row in first["condition_order"]["strata"]] == [0, 1, 2]


def test_report_is_transactional_manifest_bound_and_cli_succeeds(
    artifact, tmp_path: pathlib.Path, capsys
) -> None:
    root, _ = artifact
    output = tmp_path / "analysis"
    assert main([str(root), "--output-directory", str(output), "--json"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["schema_version"] == ANALYSIS_SCHEMA_VERSION
    assert {path.name for path in output.iterdir()} == {
        "analysis.json",
        "per_episode.csv",
        "summary.md",
        "manifest.json",
    }
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {"analysis.json", "per_episode.csv", "summary.md"}
    for relative, record in manifest["files"].items():
        payload = (output / relative).read_bytes()
        assert record == {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nonfinite"])
def test_incomplete_duplicate_and_nonfinite_evidence_fail_closed(
    artifact, mutation: str
) -> None:
    root, _ = artifact
    episodes_path = root / "episodes.json"
    episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        episodes.pop()
    elif mutation == "duplicate":
        episodes[-1] = dict(episodes[0])
    else:
        episodes[0]["wall_time_s"] = float("nan")
    _write_json(episodes_path, episodes, allow_nan=mutation == "nonfinite")

    with pytest.raises(AnalysisError):
        analyze_artifact(root)


def test_duplicate_json_key_fails_closed(artifact) -> None:
    root, _ = artifact
    path = root / "episodes.json"
    payload = path.read_text(encoding="utf-8")
    payload = payload.replace(
        '"condition_order": 0,',
        '"condition_order": 0, "condition_order": 0,',
        1,
    )
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(AnalysisError, match="duplicate JSON key"):
        analyze_artifact(root)


def test_unequal_query_counts_remain_episode_weighted(artifact) -> None:
    root, _ = artifact
    episodes_path = root / "episodes.json"
    queries_path = root / "queries.json"
    episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    episode_id = "libero_10__task_00__episode_00__%s" % BASELINE
    episode = next(row for row in episodes if row["episode_id"] == episode_id)
    episode["policy_queries"] = 3
    extra = dict(
        next(
            row
            for row in queries
            if row["episode_id"] == episode_id and row["query_index"] == 1
        )
    )
    extra.update(
        {
            "query_index": 2,
            "seam_motion_l2": 1.0,
            "seam_gripper_abs": 0.5,
        }
    )
    queries.append(extra)
    _write_json(episodes_path, episodes)
    _write_json(queries_path, queries)

    analysis, rows = analyze_artifact(root)
    baseline = analysis["seam"]["seam_motion_l2"]["methods"][BASELINE]
    # The first episode contributes mean([0.2, 1.0]) once, not two pooled queries.
    assert baseline["mean"] == pytest.approx(0.2655)
    assert baseline["mean"] != pytest.approx(5.91 / 21.0)
    assert baseline["episodes"] == 20
    assert baseline["scored_transitions"] == 21
    assert rows[0]["%s_scored_transition_queries" % BASELINE] == 2
    assert rows[0]["%s_scored_transition_queries" % PROJECTED] == 1


def test_triplet_initial_state_mismatch_fails_closed(artifact) -> None:
    root, _ = artifact
    path = root / "episodes.json"
    episodes = json.loads(path.read_text(encoding="utf-8"))
    target = next(
        row
        for row in episodes
        if row["task_id"] == 0
        and row["episode_index"] == 0
        and row["method"] == RTC
    )
    target["initial_state_sha256"] = "f" * 64
    _write_json(path, episodes)

    with pytest.raises(AnalysisError, match="does not share one initial state"):
        analyze_artifact(root)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({PROJECTED: 0.01, RTC: 0.04}, {PROJECTED: 0.02, RTC: 0.04}),
        ({PROJECTED: 0.80, RTC: 1.00}, {PROJECTED: 1.0, RTC: 1.0}),
        ({PROJECTED: 0.00, RTC: 0.00}, {PROJECTED: 0.0, RTC: 0.0}),
        ({PROJECTED: 0.03, RTC: 0.03}, {PROJECTED: 0.06, RTC: 0.06}),
    ],
)
def test_holm_adjustment_is_monotone_and_bounded(raw, expected) -> None:
    adjusted = analysis_module._holm_adjust(raw)

    assert adjusted == expected
    ordered = sorted(raw, key=lambda method: raw[method])
    ordered_adjusted = [adjusted[method] for method in ordered]
    assert ordered_adjusted == sorted(ordered_adjusted)
    assert all(raw[method] <= adjusted[method] <= 1.0 for method in raw)


def test_independent_validator_runs_before_source_parsing(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    root = tmp_path / "invalid"
    root.mkdir()
    (root / "episodes.json").write_text("not JSON", encoding="utf-8")
    calls = []

    def reject(path: pathlib.Path):
        calls.append(path)
        raise rtc_overlap_pilot.PilotValidationError("manifest mismatch")

    monkeypatch.setattr(rtc_overlap_pilot, "validate_artifact", reject)
    with pytest.raises(AnalysisError, match="failed independent validation"):
        analyze_artifact(root)
    assert len(calls) == 1


def test_existing_output_is_never_overwritten(
    artifact, tmp_path: pathlib.Path
) -> None:
    root, calls = artifact
    output = tmp_path / "analysis"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(AnalysisError, match="already exists"):
        generate_report(root, output)
    assert calls == []
    assert sentinel.read_text(encoding="utf-8") == "keep"
