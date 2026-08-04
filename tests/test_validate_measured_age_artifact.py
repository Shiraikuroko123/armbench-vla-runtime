from __future__ import annotations

import csv
import hashlib
import json
import pathlib

import pytest

from integrations.openpi.validate_measured_age_artifact import (
    EPISODE_FIELDS,
    QUERY_FIELDS,
    RUNTIME_SOURCE_FILES,
    SCHEMA_VERSION,
    WARMUP_FIELDS,
    main,
    validate_artifact,
)


def _write_csv(path: pathlib.Path, fields, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: pathlib.Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_json(path: pathlib.Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _jitter(pairing, query_index: int, seed: int, candidates):
    payload = json.dumps(
        {"pairing_key": list(pairing), "query_index": query_index, "seed": seed},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return digest.hex(), candidates[int.from_bytes(digest[:8], "big") % len(candidates)]


def _refresh_manifest(root: pathlib.Path) -> None:
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json" or path.suffix == ".tmp":
            continue
        files[path.relative_to(root).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    _write_json(root / "manifest.json", {"schema_version": SCHEMA_VERSION, "files": files})


def _artifact(root: pathlib.Path) -> pathlib.Path:
    root.mkdir()
    candidates = [40.0]
    seed = 7
    pairing = ("libero_spatial", 0, 0, 5)
    jitter_hash, jitter_value = _jitter(pairing, 0, seed, candidates)
    checkpoint = "b" * 64

    warmup = {
        "schema_version": SCHEMA_VERSION,
        "warmup_index": 0,
        "scored": False,
        "client_session_id": "session-001",
        "checkpoint_content_sha256": checkpoint,
        "observation_captured_monotonic_ns": 100_000_000,
        "policy_call_started_monotonic_ns": 110_000_000,
        "policy_call_finished_monotonic_ns": 150_000_000,
        "response_ready_monotonic_ns": 160_000_000,
        "observation_age_ms": 60.0,
        "inference_latency_ms": 40.0,
        "action_chunk_steps": 10,
        "action_dimension": 7,
        "accepted": True,
        "error_type": "",
        "error_message": "",
    }
    _write_csv(root / "warmup_queries.csv", WARMUP_FIELDS, [warmup])

    modes = ("async_unguarded", "latency_aligned")
    episodes = []
    queries = []
    cells = []
    for order, mode in enumerate(modes):
        episode_id = "libero_spatial/task_000/episode_000/%s/h_05/l_000" % mode
        pair_id = "libero_spatial/task_000/episode_000/h_05/l_000"
        cells.append(
            {
                "condition_order": order,
                "episode_id": episode_id,
                "pair_id": pair_id,
                "task_suite": "libero_spatial",
                "task_id": 0,
                "episode_index": 0,
                "mode": mode,
                "replan_steps": 5,
                "latency_source": "measured_wall",
                "latency_steps": 0,
            }
        )
        success = mode == "latency_aligned"
        episodes.append(
            {
                "schema_version": SCHEMA_VERSION,
                "episode_id": episode_id,
                "pair_id": pair_id,
                "condition_order": order,
                "task_suite": "libero_spatial",
                "task_id": 0,
                "episode_index": 0,
                "task_description": "pick up the black bowl",
                "mode": mode,
                "latency_source": "measured_wall",
                "replan_steps": 5,
                "latency_steps": 0,
                "seed": seed,
                "success": success,
                "termination_reason": "task_success" if success else "step_limit",
                "initial_state_sha256": "a" * 64,
                "environment_steps": 17,
                "task_action_steps": 5,
                "latency_action_steps": 2,
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
                "simulated_catchup_steps": 2,
                "observation_age_p50_ms": 120.0,
                "observation_age_p95_ms": 120.0,
                "observation_age_max_ms": 120.0,
                "inference_latency_p50_ms": 40.0,
                "inference_latency_p95_ms": 40.0,
                "wall_time_s": 1.0,
                "failure_category": "",
                "failure_type": "",
                "failure_message": "",
                "video_required": False,
                "video_path": "",
                "video_error_type": "",
                "video_error_message": "",
            }
        )
        queries.append(
            {
                "schema_version": SCHEMA_VERSION,
                "episode_id": episode_id,
                "pair_id": pair_id,
                "condition_order": order,
                "task_suite": "libero_spatial",
                "task_id": 0,
                "episode_index": 0,
                "mode": mode,
                "latency_source": "measured_wall",
                "replan_steps": 5,
                "latency_steps": 0,
                "query_index": 0,
                "observation_step": 10,
                "response_step": 12,
                "observation_captured_monotonic_ns": 1_000_000_000 + order * 1_000_000_000,
                "policy_call_started_monotonic_ns": 1_010_000_000 + order * 1_000_000_000,
                "policy_call_finished_monotonic_ns": 1_050_000_000 + order * 1_000_000_000,
                "response_ready_monotonic_ns": 1_120_000_000 + order * 1_000_000_000,
                "clock_trace_complete": True,
                "observation_age_ms": 120.0,
                "inference_latency_ms": 40.0,
                "response_delivery_elapsed_ms": 70.0,
                "policy_inference_latency_ms": 35.0,
                "server_inference_latency_ms": 30.0,
                "response_jitter_requested_ms": jitter_value,
                "jitter_key_sha256": jitter_hash,
                "completed_controller_steps": 2,
                "simulated_catchup_steps": 2,
                "action_chunk_steps": 10,
                "measured_stale_steps": 3,
                "action_offset_steps": 0 if mode == "async_unguarded" else 3,
                "selected_stop_step": 8,
                "available_suffix_steps": 7,
                "deadline_exceeded": False,
                "horizon_overrun": False,
                "age_refresh_index": 0,
                "fallback_hold_steps": 0,
                "alignment_disposition": "not_applied" if mode == "async_unguarded" else "execute",
                "alignment_reason": "async_unguarded" if mode == "async_unguarded" else "fresh_suffix_available",
                "accepted": True,
                "decision": "accepted_unguarded" if mode == "async_unguarded" else "accepted_measured_latency_aligned",
                "rejection_reasons": "",
                "position_mismatch_m": 0.0,
                "orientation_mismatch_rad": 0.0,
                "gripper_mismatch_linf": 0.0,
                "error_stage": "",
                "error_type": "",
                "error_message": "",
            }
        )
    _write_csv(root / "per_episode.csv", EPISODE_FIELDS, episodes)
    _write_csv(root / "per_query.csv", QUERY_FIELDS, queries)

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
            "seed": seed,
            "candidates_ms": candidates,
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
            "rollouts": 2,
            "paired_conditions": 1,
            "matched_condition_groups": 1,
            "task_suites": ["libero_spatial"],
            "task_ids": [0],
            "episode_indices": [0],
            "modes": ["async_unguarded", "latency_aligned"],
            "replan_steps": [5],
            "latency_sources": ["measured_wall"],
            "latency_steps": [0],
        },
        "registered_cells": cells,
        "seed": seed,
    }
    _write_json(root / "resolved_protocol.json", protocol)
    source_hashes = {}
    for index, relative in enumerate(RUNTIME_SOURCE_FILES):
        snapshot = root / "provenance" / "armbench_source" / pathlib.PurePosixPath(relative)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(("source-%d\n" % index).encode("ascii"))
        source_hashes[relative] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    _write_json(
        root / "environment.json",
        {"schema_version": SCHEMA_VERSION, "runtime_source_sha256": source_hashes},
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "planned_rollouts": 2,
        "completed_rollouts": 2,
        "valid": True,
        "complete": True,
        "warmup_queries_planned": 1,
        "warmup_queries_completed": 1,
        "warmup_queries_valid": 1,
        "aggregate": [
            {
                "mode": "async_unguarded", "rollouts": 1, "successes": 0,
                "success_rate": 0.0, "policy_queries": 1,
                "mean_policy_queries": 1.0, "observation_age_p50_ms": 120.0,
                "observation_age_p95_ms": 120.0, "observation_age_max_ms": 120.0,
                "deadline_misses": 0, "horizon_overruns": 0,
                "age_refreshes": 0, "fallback_hold_steps": 0,
            },
            {
                "mode": "latency_aligned", "rollouts": 1, "successes": 1,
                "success_rate": 1.0, "policy_queries": 1,
                "mean_policy_queries": 1.0, "observation_age_p50_ms": 120.0,
                "observation_age_p95_ms": 120.0, "observation_age_max_ms": 120.0,
                "deadline_misses": 0, "horizon_overruns": 0,
                "age_refreshes": 0, "fallback_hold_steps": 0,
            },
        ],
        "paired": {
            "pairs": 1, "async_successes": 0, "aligned_successes": 1,
            "candidate_wins": 1, "reference_wins": 0, "ties": 0,
            "success_rate_difference": 1.0,
        },
    }
    _write_json(root / "summary.json", summary)
    _write_json(
        root / "progress.json",
        {
            "schema_version": SCHEMA_VERSION, "planned_rollouts": 2,
            "completed_rollouts": 2, "warmup_queries_required": 1,
            "warmup_queries_completed": 1, "complete": True,
        },
    )
    _write_json(
        root / "integrity.json",
        {"schema_version": SCHEMA_VERSION, "valid": True, "errors": []},
    )
    (root / "summary.md").write_text(
        "# pi0.5-LIBERO measured-age evaluation\n\n"
        "- Schema: `%s`\n"
        "- Planned/completed rollouts: 2/2\n"
        "- Non-scoring warm-up queries: 1\n"
        "- Complete: yes\n\n"
        "| Mode | Success | Queries | Age P95 ms | Deadline misses | Horizon overruns |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: |\n"
        "| async_unguarded | 0/1 | 1 | 120.000 | 0 | 0 |\n"
        "| latency_aligned | 1/1 | 1 | 120.000 | 0 | 0 |\n\n"
        "This artifact uses blocking inference plus post-response simulator catch-up. "
        "It is not an OS hard-real-time or real-robot result.\n"
        % SCHEMA_VERSION,
        encoding="utf-8",
    )
    _refresh_manifest(root)
    return root


@pytest.fixture
def artifact(tmp_path: pathlib.Path) -> pathlib.Path:
    return _artifact(tmp_path / "artifact")


def test_valid_artifact_and_cli_json(artifact: pathlib.Path, capsys) -> None:
    report = validate_artifact(artifact)
    assert report.valid, "\n".join(report.errors)
    assert main([str(artifact), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True


def _tamper_csv(root: pathlib.Path, filename: str, field: str, value: str) -> None:
    path = root / filename
    fields, rows = _read_csv(path)
    rows[0][field] = value
    _write_csv(path, fields, rows)
    _refresh_manifest(root)


def test_resigned_age_tamper_is_recomputed(artifact: pathlib.Path) -> None:
    _tamper_csv(artifact, "per_query.csv", "observation_age_ms", "121.0")
    report = validate_artifact(artifact)
    assert not report.valid
    assert any("observation_age_ms mismatch" in error for error in report.errors)


def test_resigned_offset_tamper_is_recomputed(artifact: pathlib.Path) -> None:
    path = artifact / "per_query.csv"
    fields, rows = _read_csv(path)
    rows[1]["action_offset_steps"] = "2"
    _write_csv(path, fields, rows)
    _refresh_manifest(artifact)
    report = validate_artifact(artifact)
    assert not report.valid
    assert any("aligned offset mismatch" in error for error in report.errors)


def test_resigned_jitter_tamper_is_recomputed(artifact: pathlib.Path) -> None:
    _tamper_csv(artifact, "per_query.csv", "jitter_key_sha256", "c" * 64)
    report = validate_artifact(artifact)
    assert not report.valid
    assert any("jitter key SHA-256 mismatch" in error for error in report.errors)


def test_manifest_detects_unresigned_tamper(artifact: pathlib.Path) -> None:
    with (artifact / "summary.md").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    report = validate_artifact(artifact)
    assert not report.valid
    assert any("SHA-256 mismatch for summary.md" in error for error in report.errors)


def test_terminal_during_catchup_is_valid_and_not_an_intervention(
    artifact: pathlib.Path,
) -> None:
    query_path = artifact / "per_query.csv"
    query_fields, queries = _read_csv(query_path)
    for query in queries:
        query["accepted"] = "False"
        query["decision"] = "success_during_inference_delay"
        query["alignment_disposition"] = "terminal_before_dispatch"
        query["alignment_reason"] = "success_during_inference_delay"
    _write_csv(query_path, query_fields, queries)

    episode_path = artifact / "per_episode.csv"
    episode_fields, episodes = _read_csv(episode_path)
    for episode in episodes:
        episode["success"] = "True"
        episode["termination_reason"] = "success_during_inference_delay"
        episode["accepted_chunks"] = "0"
        episode["stale_chunks_executed"] = "0"
        episode["stale_action_steps"] = "0"
        episode["task_action_steps"] = "0"
    _write_csv(episode_path, episode_fields, episodes)

    summary_path = artifact / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for row in summary["aggregate"]:
        row["successes"] = 1
        row["success_rate"] = 1.0
    summary["paired"].update(
        {
            "async_successes": 1,
            "aligned_successes": 1,
            "candidate_wins": 0,
            "reference_wins": 0,
            "ties": 1,
            "success_rate_difference": 0.0,
        }
    )
    _write_json(summary_path, summary)
    markdown = (artifact / "summary.md").read_text(encoding="utf-8")
    markdown = markdown.replace(
        "| async_unguarded | 0/1 |", "| async_unguarded | 1/1 |"
    )
    (artifact / "summary.md").write_text(markdown, encoding="utf-8")
    _refresh_manifest(artifact)

    report = validate_artifact(artifact)

    assert report.valid, "\n".join(report.errors)


def test_incomplete_error_clock_fails_closed_without_validator_exception(
    artifact: pathlib.Path,
) -> None:
    path = artifact / "per_query.csv"
    fields, rows = _read_csv(path)
    rows[0]["clock_trace_complete"] = "False"
    rows[0]["response_ready_monotonic_ns"] = ""
    rows[0]["observation_age_ms"] = ""
    rows[0]["response_delivery_elapsed_ms"] = ""
    rows[0]["error_stage"] = "response_delivery"
    rows[0]["error_type"] = "TimeoutError"
    rows[0]["error_message"] = "timed out"
    _write_csv(path, fields, rows)
    _refresh_manifest(artifact)

    report = validate_artifact(artifact)

    assert not report.valid
    assert any("incomplete clock trace" in error for error in report.errors)
    assert not any("unexpected failure" in error for error in report.errors)
