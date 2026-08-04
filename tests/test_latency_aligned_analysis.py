from __future__ import annotations

import csv
import hashlib
import json
import pathlib

import pytest

import integrations.openpi.latency_aligned_analysis as analysis_module
from integrations.openpi.latency_aligned_analysis import (
    ANALYSIS_SCHEMA_VERSION,
    ASYNC_UNGUARDED,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    EMPTY_GIT_DIFF_SHA256,
    FROZEN_ARMBENCH_RUN_COMMIT,
    FROZEN_CHECKPOINT,
    FROZEN_CHECKPOINT_CONTENT_SHA256,
    FROZEN_OPENPI_COMMIT,
    FROZEN_POLICY_CONFIG,
    LATENCY_ALIGNED,
    _holm_adjust,
    execute_analysis,
    main,
)


FIELDS = (
    "schema_version",
    "episode_id",
    "pair_id",
    "condition_order",
    "task_suite",
    "task_id",
    "episode_index",
    "task_description",
    "mode",
    "replan_steps",
    "latency_steps",
    "injected_latency_ms",
    "fixed_refresh_interval",
    "seed",
    "success",
    "initial_state_sha256",
    "policy_queries",
    "failure_category",
    "failure_type",
    "failure_message",
    "video_required",
)


@pytest.fixture(autouse=True)
def _stub_independent_full_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        analysis_module,
        "validate_run_manifest",
        lambda _root: {
            "valid": True,
            "complete": True,
            "errors": [],
            "files_checked": 42,
        },
    )


def _source_rows():
    rows = []
    order = 0
    success_limits = {0: (25, 30), 2: (20, 30), 4: (10, 40)}
    for task_id in range(10):
        for episode_index in range(5):
            pair_index = task_id * 5 + episode_index
            initial_hash = hashlib.sha256(
                ("%d/%d" % (task_id, episode_index)).encode("ascii")
            ).hexdigest()
            for latency_steps in (0, 2, 4):
                pair_id = (
                    "libero_spatial/task_%03d/episode_%03d/h_05/l_%03d"
                    % (task_id, episode_index, latency_steps)
                )
                limits = success_limits[latency_steps]
                mode_order = (
                    (ASYNC_UNGUARDED, LATENCY_ALIGNED)
                    if order // 2 % 2 == 0
                    else (LATENCY_ALIGNED, ASYNC_UNGUARDED)
                )
                for mode in mode_order:
                    mode_index = 0 if mode == ASYNC_UNGUARDED else 1
                    success = pair_index < limits[mode_index]
                    failure = (
                        latency_steps == 2
                        and task_id == 4
                        and episode_index == 0
                        and mode == ASYNC_UNGUARDED
                    )
                    rows.append(
                        {
                            "schema_version": "armbench.pi05_libero_async.v1",
                            "episode_id": "%s__%s" % (pair_id, mode),
                            "pair_id": pair_id,
                            "condition_order": order,
                            "task_suite": "libero_spatial",
                            "task_id": task_id,
                            "episode_index": episode_index,
                            "task_description": "task %d" % task_id,
                            "mode": mode,
                            "replan_steps": 5,
                            "latency_steps": latency_steps,
                            "injected_latency_ms": latency_steps * 50.0,
                            "fixed_refresh_interval": "",
                            "seed": 7,
                            "success": success,
                            "initial_state_sha256": initial_hash,
                            "policy_queries": 10 + latency_steps + mode_index * 2,
                            "failure_category": "policy_timeout" if failure else "",
                            "failure_type": "TimeoutError" if failure else "",
                            "failure_message": "timed out" if failure else "",
                            "video_required": True,
                        }
                    )
                    order += 1
    return rows


def _write_json(path: pathlib.Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_source(
    root: pathlib.Path,
    rows=None,
    *,
    video_mode: str = "all",
    armbench_commit: str = FROZEN_ARMBENCH_RUN_COMMIT,
    checkpoint_sha256: str = FROZEN_CHECKPOINT_CONTENT_SHA256,
) -> pathlib.Path:
    run = root / "run"
    evaluation = run / "evaluation"
    evaluation.mkdir(parents=True)
    source = evaluation / "per_episode.csv"
    selected = list(_source_rows() if rows is None else rows)
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(selected)
    def record(path: pathlib.Path):
        return {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    _write_json(
        evaluation / "progress.json",
        {
            "planned_rollouts": 300,
            "completed_rollouts": 300,
            "complete": True,
        },
    )
    _write_json(evaluation / "integrity.json", {"valid": True})
    _write_json(run / "finalization.json", {"complete": True})
    _write_json(run / "artifact_validation.json", {"valid": True})
    _write_json(
        evaluation / "resolved_protocol.json",
        {
            "schema_version": "armbench.pi05_libero_async.v1",
            "openpi_commit": FROZEN_OPENPI_COMMIT,
            "policy_config": FROZEN_POLICY_CONFIG,
            "declared_checkpoint": FROZEN_CHECKPOINT,
            "checkpoint_provenance": (
                "server_attestation_with_checkpoint_content_sha256"
            ),
            "runtime_failure_policy": "abort_formal_run",
            "seed": 7,
            "bootstrap_resamples": 10000,
            "matrix": {
                "task_suites": ["libero_spatial"],
                "task_ids": list(range(10)),
                "episode_indices": list(range(5)),
                "modes": [ASYNC_UNGUARDED, LATENCY_ALIGNED],
                "replan_steps": [5],
                "latency_steps": [0, 2, 4],
                "matched_condition_groups": 150,
                "rollouts": 300,
            },
        },
    )
    _write_json(
        evaluation / "environment.json",
        {
            "armbench_git_commit": armbench_commit,
            "armbench_git_status": "",
            "armbench_git_diff_sha256": EMPTY_GIT_DIFF_SHA256,
            "openpi_git_commit": FROZEN_OPENPI_COMMIT,
            "arguments": {
                "allow_unattested_server": False,
                "video_mode": video_mode,
            },
            "server_metadata": {
                "armbench_server_attestation": {
                    "policy_config": FROZEN_POLICY_CONFIG,
                    "checkpoint_uri": FROZEN_CHECKPOINT,
                    "openpi_commit": FROZEN_OPENPI_COMMIT,
                    "checkpoint_content_sha256": checkpoint_sha256,
                }
            },
        },
    )
    evaluation_files = {
        relative: record(evaluation / relative)
        for relative in (
            "per_episode.csv",
            "progress.json",
            "integrity.json",
            "resolved_protocol.json",
            "environment.json",
        )
    }
    _write_json(
        evaluation / "manifest.json",
        {
            "schema_version": "armbench.pi05_libero_async.v1",
            "files": evaluation_files,
        },
    )
    root_files = {
        "evaluation/" + relative: record(evaluation / relative)
        for relative in (
            "per_episode.csv",
            "progress.json",
            "integrity.json",
            "resolved_protocol.json",
            "environment.json",
            "manifest.json",
        )
    }
    root_files.update(
        {
            relative: record(run / relative)
            for relative in ("finalization.json", "artifact_validation.json")
        }
    )
    _write_json(
        run / "manifest.json",
        {
            "schema_version": "armbench.pi05_libero_container_run.v1",
            "complete": True,
            "files": root_files,
        },
    )
    return source


def test_cli_writes_deterministic_stratified_itt_analysis_read_only(
    tmp_path: pathlib.Path, capsys
) -> None:
    source = _write_source(tmp_path / "source")
    before_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    before_mtime = source.stat().st_mtime_ns
    output = tmp_path / "analysis"

    assert main([str(source), "--output-directory", str(output)]) == 0

    assert json.loads(capsys.readouterr().out)["valid"] is True
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before_hash
    assert source.stat().st_mtime_ns == before_mtime
    analysis = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["schema_version"] == ANALYSIS_SCHEMA_VERSION
    assert analysis["source"]["per_episode_csv"] == "evaluation/per_episode.csv"
    assert analysis["contrast"] == {
        "comparison_mode": LATENCY_ALIGNED,
        "difference_direction": "latency_aligned - async_unguarded",
        "reference_mode": ASYNC_UNGUARDED,
    }
    assert analysis["bootstrap"] == {
        "confidence_level": 0.95,
        "method": (
            "paired percentile bootstrap of mean "
            "(latency_aligned - async_unguarded) success"
        ),
        "inferential_role": (
            "descriptive marginal uncertainty; confirmatory decisions use "
            "the Holm-adjusted exact McNemar tests"
        ),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
    }
    assert analysis["itt"] == {
        "pairs": 150,
        "rollouts": 300,
        "runtime_failures_retained": 1,
    }
    assert analysis["source"]["independently_validated_files"] == 42
    assert analysis["protocol"]["primary_latency_steps"] == 4
    assert analysis["protocol"]["secondary_latency_steps"] == [0, 2]
    by_delay = {row["latency_steps"]: row for row in analysis["latency_strata"]}
    assert set(by_delay) == {0, 2, 4}
    assert by_delay[0]["async_unguarded_successes"] == 25
    assert by_delay[0]["latency_aligned_successes"] == 30
    assert by_delay[0]["aligned_minus_async_success_rate_difference"] == pytest.approx(0.1)
    assert by_delay[0]["aligned_wins"] == 5
    assert by_delay[0]["mcnemar_exact_p"] == pytest.approx(0.0625)
    assert by_delay[2]["async_unguarded_runtime_failures"] == 1
    assert by_delay[2]["aligned_wins"] == 10
    assert by_delay[2]["async_unguarded_n"] == 50
    assert by_delay[2]["aligned_minus_async_mean_policy_queries"] == 2.0
    assert by_delay[4]["latency_aligned_successes"] == 40
    assert by_delay[4]["analysis_role"] == "primary"
    assert by_delay[0]["analysis_role"] == "prespecified_secondary"
    assert by_delay[4]["mcnemar_holm_p"] <= by_delay[4]["mcnemar_exact_p"] * 3
    assert 0.0 < by_delay[0]["async_unguarded_wilson95_low"] < 0.5
    assert len(analysis["runtime_failures"]) == 1
    assert analysis["runtime_failures"][0]["failure_type"] == "TimeoutError"
    with (output / "per_pair.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 150
    with (output / "latency_strata.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 3
    with (output / "task_latency_descriptives.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        task_rows = list(csv.DictReader(handle))
    assert len(task_rows) == 30
    assert all(row["paired_n"] == "5" for row in task_rows)
    assert all(
        row["analysis_scope"] == "descriptive_only_no_task_level_inference"
        for row in task_rows
    )
    assert not any(
        "mcnemar" in field or "bootstrap" in field or field.endswith("_p")
        for field in task_rows[0]
    )
    assert {
        (int(row["task_id"]), int(row["latency_steps"])) for row in task_rows
    } == {(task_id, latency) for task_id in range(10) for latency in (0, 2, 4)}
    assert analysis["task_latency_descriptives"] == {
        "artifact": "task_latency_descriptives.csv",
        "inference": "none",
        "rows": 30,
        "scope": "task x latency; five matched conditions per row",
    }
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert "Runtime failures retained in ITT: `1`" in summary
    assert "Difference direction: `latency_aligned - async_unguarded`" in summary
    assert "Primary comparison: delay `4`" in summary
    assert "Holm-adjusted exact McNemar" in summary
    assert "descriptive only; no task-level significance tests" in summary
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_per_episode_csv_sha256"] == before_hash
    assert set(manifest["files"]) == {
        "analysis.json",
        "latency_strata.csv",
        "per_pair.csv",
        "summary.md",
        "task_latency_descriptives.csv",
    }


def test_outputs_are_byte_identical_across_source_roots(
    tmp_path: pathlib.Path,
) -> None:
    source_a = _write_source(tmp_path / "source-a")
    source_b = _write_source(tmp_path / "source-b")
    output_a = tmp_path / "analysis-a"
    output_b = tmp_path / "analysis-b"

    execute_analysis(source_a, output_a)
    execute_analysis(source_b, output_b)

    expected_files = {
        "analysis.json",
        "latency_strata.csv",
        "manifest.json",
        "per_pair.csv",
        "summary.md",
        "task_latency_descriptives.csv",
    }
    assert {path.name for path in output_a.iterdir()} == expected_files
    assert {path.name for path in output_b.iterdir()} == expected_files
    for name in sorted(expected_files):
        assert (output_a / name).read_bytes() == (output_b / name).read_bytes()


def test_holm_adjustment_is_monotone_and_bounded() -> None:
    rows = [
        {"mcnemar_exact_p": 0.04},
        {"mcnemar_exact_p": 0.01},
        {"mcnemar_exact_p": 0.03},
    ]

    _holm_adjust(rows)

    ordered = sorted(rows, key=lambda row: row["mcnemar_exact_p"])
    adjusted = [row["mcnemar_holm_p"] for row in ordered]
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert adjusted == sorted(adjusted)
    assert all(0.0 <= value <= 1.0 for value in adjusted)

    capped = [{"mcnemar_exact_p": value} for value in (0.8, 0.9, 1.0)]
    _holm_adjust(capped)
    assert [row["mcnemar_holm_p"] for row in capped] == [1.0, 1.0, 1.0]


@pytest.mark.parametrize("mutation, message", [
    ("missing", "contains 299 rows"),
    ("duplicate", "duplicate episode_id"),
    ("seed", "outside the frozen matrix"),
    ("hash", "initial_state_sha256 mismatch"),
    ("mode", "unexpected mode"),
    ("order", "does not follow alternating mode order"),
    ("video", "violates frozen video_mode=all coverage"),
])
def test_analysis_rejects_missing_duplicate_and_mixed_protocol_rows(
    tmp_path: pathlib.Path, mutation: str, message: str
) -> None:
    rows = _source_rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = dict(rows[-2])
    elif mutation == "seed":
        rows[0]["seed"] = 8
    elif mutation == "hash":
        rows[1]["initial_state_sha256"] = "f" * 64
    elif mutation == "order":
        rows[2]["condition_order"], rows[3]["condition_order"] = (
            rows[3]["condition_order"],
            rows[2]["condition_order"],
        )
    elif mutation == "video":
        rows[0]["video_required"] = False
    else:
        rows[0]["mode"] = "state_guard"
    source = _write_source(tmp_path / mutation, rows)

    with pytest.raises(ValueError, match=message):
        execute_analysis(source, tmp_path / (mutation + "-output"))


def test_analysis_requires_complete_frozen_source_and_external_output(
    tmp_path: pathlib.Path,
) -> None:
    source = _write_source(tmp_path / "source")
    root_manifest = source.parent.parent / "manifest.json"
    value = json.loads(root_manifest.read_text(encoding="utf-8"))
    value["complete"] = False
    _write_json(root_manifest, value)

    with pytest.raises(ValueError, match="root manifest is not complete"):
        execute_analysis(source, tmp_path / "output")

    value["complete"] = True
    _write_json(root_manifest, value)
    with pytest.raises(ValueError, match="outside the frozen input artifact"):
        execute_analysis(source, source.parent.parent / "analysis")


def test_analysis_rejects_manifest_bound_file_tampering(
    tmp_path: pathlib.Path,
) -> None:
    source = _write_source(tmp_path / "source")
    progress = source.parent / "progress.json"
    value = json.loads(progress.read_text(encoding="utf-8"))
    value["planned_rollouts"] = 301
    _write_json(progress, value)

    with pytest.raises(ValueError, match="SHA-256 mismatch for progress.json"):
        execute_analysis(source, tmp_path / "output")

    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "source_kwargs,message",
    [
        ({"video_mode": "failures"}, "requires video_mode=all"),
        ({"armbench_commit": "f" * 40}, "armbench_git_commit"),
        ({"checkpoint_sha256": "f" * 64}, "checkpoint_content_sha256"),
    ],
)
def test_analysis_rejects_nonfrozen_provenance(
    tmp_path: pathlib.Path, source_kwargs, message: str
) -> None:
    source = _write_source(tmp_path / "source", **source_kwargs)

    with pytest.raises(ValueError, match=message):
        execute_analysis(source, tmp_path / "output")


def test_analysis_requires_fresh_full_run_validation(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_source(tmp_path / "source")
    monkeypatch.setattr(
        analysis_module,
        "validate_run_manifest",
        lambda _root: {
            "valid": False,
            "complete": False,
            "errors": ["file is not protected by manifest: rogue.log"],
            "files_checked": 0,
        },
    )

    with pytest.raises(
        ValueError, match="independent full run validation failed.*rogue.log"
    ):
        execute_analysis(source, tmp_path / "output")

    assert not (tmp_path / "output").exists()


def test_analysis_uses_manifest_checked_csv_snapshot(
    tmp_path: pathlib.Path,
) -> None:
    source = _write_source(tmp_path / "source")
    evidence = analysis_module.validate_frozen_source(source)
    frozen_sha256 = evidence.csv_sha256

    source.write_text("replaced after validation\n", encoding="utf-8")
    analysis, task_rows, pair_rows = analysis_module.build_analysis(evidence)

    assert analysis["source"]["per_episode_csv_sha256"] == frozen_sha256
    assert analysis["itt"] == {
        "pairs": 150,
        "rollouts": 300,
        "runtime_failures_retained": 1,
    }
    assert len(task_rows) == 30
    assert len(pair_rows) == 150


def test_analysis_output_is_atomic_on_write_failure(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_source(tmp_path / "source")
    output = tmp_path / "analysis"
    write_csv = analysis_module._write_csv

    def fail_on_per_pair(path, rows):
        if path.name == "per_pair.csv":
            raise OSError("injected write failure")
        write_csv(path, rows)

    monkeypatch.setattr(analysis_module, "_write_csv", fail_on_per_pair)

    with pytest.raises(OSError, match="injected write failure"):
        execute_analysis(source, output)

    assert not output.exists()
    assert list(tmp_path.glob(".analysis.tmp-*")) == []
