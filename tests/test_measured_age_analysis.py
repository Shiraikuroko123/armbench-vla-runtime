from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from integrations.openpi.measured_age_analysis import (
    ANALYSIS_SCHEMA_VERSION,
    AnalysisError,
    analyze_artifact,
    generate_report,
    main,
)
from integrations.openpi.measured_age_libero_eval import (
    EPISODE_FIELDS,
    QUERY_FIELDS,
    RUNTIME_SOURCE_FILES,
    SCHEMA_VERSION,
    WARMUP_FIELDS,
    write_artifacts,
)
from integrations.openpi.validate_measured_age_artifact import validate_artifact


def _jitter_hash(task_id: int, episode_index: int, query_index: int = 0) -> str:
    payload = json.dumps(
        {
            "pairing_key": ["libero_spatial", task_id, episode_index, 5],
            "query_index": query_index,
            "seed": 7,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cell(pair_index: int, mode: str, condition_order: int):
    pair_id = "libero_spatial/task_%03d/episode_%03d/h_05/l_000" % (
        pair_index,
        pair_index,
    )
    return {
        "condition_order": condition_order,
        "episode_id": "%s/%s" % (pair_id, mode),
        "pair_id": pair_id,
        "task_suite": "libero_spatial",
        "task_id": pair_index,
        "episode_index": pair_index,
        "mode": mode,
        "replan_steps": 5,
        "latency_source": "measured_wall",
        "latency_steps": 0,
    }


def _artifact(root: pathlib.Path) -> pathlib.Path:
    root.mkdir()
    cells = [
        _cell(pair_index, mode, pair_index * 2 + mode_index)
        for pair_index in range(2)
        for mode_index, mode in enumerate(("async_unguarded", "latency_aligned"))
    ]
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "temporal_alignment": {
            "latency_source": "measured_wall",
            "latency_steps": 0,
            "control_period_ms": 50.0,
            "completed_step_rounding": "floor",
            "action_offset_rounding": "ceil",
            "boundary_tolerance_ms": 1e-9,
            "deadline_ms": 250.0,
            "max_age_refreshes": 2,
            "controller_model": "post_response_catchup_simulation",
        },
        "jitter": {
            "generator": "sha256_first_u64_mod_v1",
            "seed": 7,
            "candidates_ms": [0.0],
            "pairing_key_fields": ["task_suite", "task_id", "episode_index", "replan_steps"],
            "query_index_field": "query_index",
            "mode_in_key": False,
            "payload_fields": ["seed", "pairing_key", "query_index"],
            "json_encoding": "utf-8 canonical sorted compact ASCII",
        },
        "warmup": {
            "queries": 1,
            "scored": False,
            "same_checkpoint_and_action_contract": True,
        },
        "matrix": {
            "schema_version": SCHEMA_VERSION,
            "rollouts": 4,
            "paired_conditions": 2,
            "matched_condition_groups": 2,
            "task_suites": ["libero_spatial"],
            "task_ids": [0, 1],
            "episode_indices": [0, 1],
            "modes": ["async_unguarded", "latency_aligned"],
            "replan_steps": [5],
            "latency_sources": ["measured_wall"],
            "latency_steps": [0],
        },
        "registered_cells": cells,
        "seed": 7,
    }
    checkpoint = "c" * 64
    warmup = {field: "" for field in WARMUP_FIELDS}
    warmup.update(
        {
            "schema_version": SCHEMA_VERSION,
            "warmup_index": 0,
            "scored": False,
            "client_session_id": "session-analysis-001",
            "checkpoint_content_sha256": checkpoint,
            "observation_captured_monotonic_ns": 100_000_000,
            "policy_call_started_monotonic_ns": 105_000_000,
            "policy_call_finished_monotonic_ns": 135_000_000,
            "response_ready_monotonic_ns": 140_000_000,
            "observation_age_ms": 40.0,
            "inference_latency_ms": 30.0,
            "action_chunk_steps": 10,
            "action_dimension": 7,
            "accepted": True,
            "error_type": None,
            "error_message": None,
        }
    )

    episodes = []
    queries = []
    ages = {
        (0, "async_unguarded"): 120.0,
        (0, "latency_aligned"): 160.0,
        (1, "async_unguarded"): 140.0,
        (1, "latency_aligned"): 180.0,
    }
    successes = {
        (0, "async_unguarded"): False,
        (0, "latency_aligned"): True,
        (1, "async_unguarded"): True,
        (1, "latency_aligned"): True,
    }
    for cell in cells:
        pair_index = int(cell["task_id"])
        mode = str(cell["mode"])
        age = ages[(pair_index, mode)]
        completed = int(age // 50.0)
        stale = int((age + 49.999999999) // 50.0)
        base_ns = 1_000_000_000 + int(cell["condition_order"]) * 1_000_000_000
        query = {field: "" for field in QUERY_FIELDS}
        query.update(cell)
        query.update(
            {
                "schema_version": SCHEMA_VERSION,
                "query_index": 0,
                "observation_step": 10,
                "response_step": 10 + completed,
                "observation_captured_monotonic_ns": base_ns,
                "policy_call_started_monotonic_ns": base_ns + 5_000_000,
                "policy_call_finished_monotonic_ns": base_ns + 35_000_000,
                "response_ready_monotonic_ns": base_ns + int(age * 1_000_000),
                "clock_trace_complete": True,
                "observation_age_ms": age,
                "inference_latency_ms": 30.0,
                "response_delivery_elapsed_ms": age - 35.0,
                "policy_inference_latency_ms": 25.0,
                "server_inference_latency_ms": 20.0,
                "response_jitter_requested_ms": 0.0,
                "jitter_key_sha256": _jitter_hash(pair_index, pair_index),
                "completed_controller_steps": completed,
                "simulated_catchup_steps": completed,
                "action_chunk_steps": 10,
                "measured_stale_steps": stale,
                "action_offset_steps": stale if mode == "latency_aligned" else 0,
                "selected_stop_step": stale + 5,
                "available_suffix_steps": 10 - stale,
                "deadline_exceeded": False,
                "horizon_overrun": False,
                "age_refresh_index": 0,
                "fallback_hold_steps": 0,
                "alignment_disposition": "execute" if mode == "latency_aligned" else "not_applied",
                "alignment_reason": "fresh_suffix_available" if mode == "latency_aligned" else "async_unguarded",
                "accepted": True,
                "decision": "accepted_measured_latency_aligned" if mode == "latency_aligned" else "accepted_unguarded",
                "position_mismatch_m": 0.0,
                "orientation_mismatch_rad": 0.0,
                "gripper_mismatch_linf": 0.0,
                "error_stage": None,
                "error_type": None,
                "error_message": None,
            }
        )
        queries.append(query)

        episode = {field: "" for field in EPISODE_FIELDS}
        episode.update(cell)
        success = successes[(pair_index, mode)]
        episode.update(
            {
                "schema_version": SCHEMA_VERSION,
                "task_description": "analysis task %d" % pair_index,
                "seed": 7,
                "success": success,
                "termination_reason": "task_success" if success else "step_limit",
                "initial_state_sha256": ("a" if pair_index == 0 else "b") * 64,
                "environment_steps": 15 + completed,
                "task_action_steps": 5,
                "latency_action_steps": completed,
                "policy_queries": 1,
                "accepted_chunks": 1,
                "rejected_chunks": 0,
                "stale_chunks_executed": 1,
                "stale_action_steps": 5,
                "interventions": 0,
                "deadline_misses": 0,
                "horizon_overruns": 0,
                "age_refreshes": 0,
                "fallback_hold_steps": 0,
                "simulated_catchup_steps": completed,
                "observation_age_p50_ms": age,
                "observation_age_p95_ms": age,
                "observation_age_max_ms": age,
                "inference_latency_p50_ms": 30.0,
                "inference_latency_p95_ms": 30.0,
                "wall_time_s": 1.0,
                "video_required": False,
                "failure_category": None,
                "failure_type": None,
                "failure_message": None,
                "video_path": None,
                "video_error_type": None,
                "video_error_message": None,
            }
        )
        episodes.append(episode)

    source_hashes = {}
    for index, relative in enumerate(RUNTIME_SOURCE_FILES):
        snapshot = root / "provenance" / "armbench_source" / pathlib.PurePosixPath(relative)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(("analysis-source-%d\n" % index).encode("ascii"))
        source_hashes[relative] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    environment = {
        "schema_version": SCHEMA_VERSION,
        "runtime_source_sha256": source_hashes,
        "client_session_id": "session-analysis-001",
    }
    integrity = write_artifacts(
        root,
        protocol,
        environment,
        [warmup],
        episodes,
        queries,
        planned_rollouts=4,
        complete=True,
    )
    assert integrity["valid"] is True, integrity["errors"]
    report = validate_artifact(root)
    assert report.valid, report.errors
    return root


@pytest.fixture
def artifact(tmp_path: pathlib.Path) -> pathlib.Path:
    return _artifact(tmp_path / "measured")


def _refresh_manifest(root: pathlib.Path) -> None:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json" and path.suffix != ".tmp":
            files[path.relative_to(root).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    (root / "manifest.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "files": files}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_valid_paired_analysis_recomputes_effects_and_runtime_metrics(
    artifact: pathlib.Path,
) -> None:
    analysis, pairs = analyze_artifact(
        artifact, bootstrap_resamples=1_000, bootstrap_seed=19
    )

    assert analysis["schema_version"] == ANALYSIS_SCHEMA_VERSION
    assert analysis["cohort"] == {
        "rollouts": 4,
        "pairs": 2,
        "queries": 4,
        "task_suites": ["libero_spatial"],
        "replan_steps": [5],
        "age_rounding": "ceil",
        "deadline_ms": 250.0,
    }
    assert analysis["success"]["async_unguarded"]["rate"] == 0.5
    assert analysis["success"]["latency_aligned"]["rate"] == 1.0
    paired = analysis["success"]["paired"]
    assert paired["rate_difference"] == 0.5
    assert (paired["candidate_wins"], paired["reference_wins"], paired["ties"]) == (1, 0, 1)
    assert paired["mcnemar_exact_two_sided_p"] == 1.0
    timing = {row["mode"]: row for row in analysis["timing"]}
    assert timing["async_unguarded"]["observation_age_ms"]["p95"] == 139.0
    assert timing["latency_aligned"]["observation_age_ms"]["max"] == 180.0
    assert all(row["deadline_misses"] == 0 for row in analysis["timing"])
    policy = next(row for row in analysis["runtime_burden"] if row["metric"] == "policy_queries")
    assert policy["paired_mean_difference"] == 0.0
    assert len(pairs) == 2


def test_report_is_transactional_manifest_bound_and_cli_succeeds(
    artifact: pathlib.Path, tmp_path: pathlib.Path, capsys
) -> None:
    output = tmp_path / "analysis"
    assert main(
        [
            str(artifact), "--output-directory", str(output),
            "--bootstrap-resamples", "500", "--bootstrap-seed", "23", "--json",
        ]
    ) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["schema_version"] == ANALYSIS_SCHEMA_VERSION
    assert {path.name for path in output.iterdir()} == {
        "analysis.json", "per_pair.csv", "summary.md", "manifest.json"
    }
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {"analysis.json", "per_pair.csv", "summary.md"}
    for relative, record in manifest["files"].items():
        payload = (output / relative).read_bytes()
        assert record == {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def test_unresigned_source_tamper_fails_closed(
    artifact: pathlib.Path, tmp_path: pathlib.Path, capsys
) -> None:
    with (artifact / "per_episode.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")

    with pytest.raises(AnalysisError, match="failed independent validation"):
        analyze_artifact(artifact, bootstrap_resamples=100)
    output = tmp_path / "must-not-exist"
    assert main(
        [str(artifact), "--output-directory", str(output), "--bootstrap-resamples", "100", "--json"]
    ) == 2
    assert json.loads(capsys.readouterr().out)["valid"] is False
    assert not output.exists()


def test_resigned_incomplete_artifact_fails_closed(artifact: pathlib.Path) -> None:
    progress_path = artifact / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["complete"] = False
    progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest(artifact)

    with pytest.raises(AnalysisError, match="failed independent validation"):
        analyze_artifact(artifact, bootstrap_resamples=100)


def test_existing_output_is_never_overwritten(
    artifact: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    output = tmp_path / "analysis"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(AnalysisError, match="already exists"):
        generate_report(artifact, output, bootstrap_resamples=100)
    assert sentinel.read_text(encoding="utf-8") == "keep"
