from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import pathlib

import pytest

from integrations.openpi.libero_runtime import (
    ASYNC_UNGUARDED,
    LATENCY_ALIGNED,
    MEASURED_WALL_LATENCY,
    EpisodeResult,
    QueryRecord,
    StateMismatch,
)
from integrations.openpi.libero_runtime_eval import ExperimentCell
from integrations.openpi.measured_age_libero_eval import (
    EPISODE_FIELDS,
    QUERY_FIELDS,
    SCHEMA_VERSION,
    WARMUP_FIELDS,
    artifact_errors,
    jitter_key_sha256,
    jitter_value_ms,
    main,
    make_runtime_config,
    resolved_protocol,
    result_rows,
    write_artifacts,
)


def _args(**updates):
    values = {
        "expected_openpi_commit": "a" * 40,
        "checkpoint": "gs://openpi-assets/checkpoints/pi05_libero",
        "server_launch_args": "--env LIBERO",
        "allow_unattested_server": False,
        "resize_size": 224,
        "num_steps_wait": 10,
        "control_period_ms": 50.0,
        "age_rounding": "ceil",
        "deadline_ms": 250.0,
        "max_age_refreshes": 2,
        "seed": 7,
        "warmup_queries": 3,
        "warmup_task_id": 0,
        "warmup_episode_index": 49,
        "task_suite": "libero_spatial",
        "max_task_steps": None,
        "server_startup_timeout_s": 1200.0,
        "inference_timeout_s": 600.0,
        "video_mode": "none",
    }
    values.update(updates)
    return argparse.Namespace(**values)


def _cell(mode: str, order: int) -> ExperimentCell:
    return ExperimentCell(
        task_suite="libero_spatial",
        task_id=0,
        episode_index=0,
        mode=mode,
        replan_steps=5,
        latency_steps=0,
        pair_id="libero_spatial/task_000/episode_000/h_05/l_000",
        condition_order=order,
    )


def _result(cell: ExperimentCell, jitter_ms: float) -> EpisodeResult:
    query = QueryRecord(
        query_index=0,
        observation_step=10,
        response_step=12,
        inference_latency_ms=35.0,
        injected_latency_steps_requested=0,
        injected_latency_steps_executed=0,
        action_chunk_steps=10,
        accepted=True,
        decision=(
            "accepted_unguarded"
            if cell.mode == ASYNC_UNGUARDED
            else "accepted_measured_latency_aligned"
        ),
        rejection_reasons=(),
        mismatch=StateMismatch(0.0, 0.0, 0.0),
        latency_source=MEASURED_WALL_LATENCY,
        observation_age_ms=120.0,
        response_jitter_ms=jitter_ms,
        measured_stale_steps=3,
        action_offset_steps=3 if cell.mode == LATENCY_ALIGNED else 0,
        available_suffix_steps=7,
        deadline_exceeded=False,
        horizon_overrun=False,
        age_refresh_index=0,
        fallback_hold_steps=0,
        observation_captured_monotonic_ns=1_000_000_000,
        policy_call_started_monotonic_ns=1_005_000_000,
        policy_call_finished_monotonic_ns=1_040_000_000,
        response_ready_monotonic_ns=1_120_000_000,
        response_delivery_elapsed_ms=80.0,
        simulated_catchup_steps=2,
        selected_stop_step=8,
        alignment_disposition="execute",
        alignment_reason="fresh_suffix_available",
    )
    return EpisodeResult(
        success=True,
        termination_reason="task_success",
        initial_state_sha256="b" * 64,
        environment_steps=20,
        task_action_steps=8,
        latency_action_steps=2,
        policy_queries=1,
        accepted_chunks=1,
        rejected_chunks=0,
        stale_chunks_executed=1,
        stale_action_steps=5,
        interventions=0,
        query_records=[query],
        replay_frames=[],
    )


def _warmups():
    rows = []
    for index in range(3):
        start = 100_000_000 + index * 20_000_000
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "warmup_index": index,
                "scored": False,
                "client_session_id": "session",
                "checkpoint_content_sha256": "c" * 64,
                "observation_captured_monotonic_ns": start,
                "policy_call_started_monotonic_ns": start + 1_000_000,
                "policy_call_finished_monotonic_ns": start + 10_000_000,
                "response_ready_monotonic_ns": start + 10_000_000,
                "observation_age_ms": 10.0,
                "inference_latency_ms": 9.0,
                "action_chunk_steps": 10,
                "action_dimension": 7,
                "accepted": True,
                "error_type": None,
                "error_message": None,
            }
        )
    return rows


def test_plan_contract_is_measured_only_and_has_three_unscored_warmups(capsys):
    assert main(["plan", "--task-ids", "0", "--episode-indices", "0"]) == 0
    plan = json.loads(capsys.readouterr().out)

    assert plan["schema_version"] == SCHEMA_VERSION
    assert plan["matrix"]["rollouts"] == 2
    assert plan["matrix"]["paired_conditions"] == 1
    assert plan["matrix"]["latency_sources"] == [MEASURED_WALL_LATENCY]
    assert plan["matrix"]["latency_steps"] == [0]
    assert plan["temporal_alignment"]["completed_step_rounding"] == "floor"
    assert plan["temporal_alignment"]["action_offset_rounding"] == "ceil"
    assert plan["warmup"] == {
        "queries": 3,
        "task_suite": "libero_spatial",
        "task_id": 0,
        "episode_index": 49,
        "scored": False,
        "required_before_scoring": True,
    }
    assert len(plan["registered_cells"]) == 2


def test_plan_resolves_forty_rollout_pilot(capsys):
    assert (
        main(
            [
                "plan",
                "--task-ids",
                "all",
                "--episode-indices",
                "0:2",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["matrix"]["rollouts"] == 40
    assert plan["matrix"]["matched_condition_groups"] == 20
    assert len(plan["registered_cells"]) == 40


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["plan", "--modes", "state_guard"], "only allows modes"),
        (["plan", "--warmup-queries", "0"], "at least 1"),
        (["plan", "--jitter-values-ms", "0,nan"], "finite"),
    ],
)
def test_plan_rejects_non_measured_or_unfrozen_inputs(arguments, message, capsys):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_keyed_jitter_excludes_mode_and_rows_retain_raw_timing():
    baseline = _cell(ASYNC_UNGUARDED, 0)
    aligned = _cell(LATENCY_ALIGNED, 1)
    candidates = (0.0, 40.0, 80.0, 160.0)

    assert jitter_key_sha256(baseline, 7, 0) == jitter_key_sha256(aligned, 7, 0)
    assert jitter_value_ms(baseline, 7, 0, candidates) == jitter_value_ms(
        aligned, 7, 0, candidates
    )
    jitter = jitter_value_ms(aligned, 7, 0, candidates)
    episode, rows = result_rows(
        aligned,
        _result(aligned, jitter),
        "test task",
        7,
        1.0,
        candidates,
        50.0,
    )
    query = rows[0]

    assert tuple(query) == QUERY_FIELDS
    assert tuple(episode) == EPISODE_FIELDS
    assert query["clock_trace_complete"] is True
    assert query["observation_captured_monotonic_ns"] == 1_000_000_000
    assert query["policy_call_started_monotonic_ns"] == 1_005_000_000
    assert query["response_delivery_elapsed_ms"] == 80.0
    assert query["completed_controller_steps"] == 2
    assert query["simulated_catchup_steps"] == 2
    assert query["measured_stale_steps"] == 3
    assert query["action_offset_steps"] == 3
    assert query["selected_stop_step"] == 8
    assert episode["simulated_catchup_steps"] == 2

    _, baseline_rows = result_rows(
        baseline,
        _result(baseline, jitter),
        "test task",
        7,
        1.0,
        candidates,
        50.0,
    )
    assert baseline_rows[0]["alignment_disposition"] == "not_applied"
    assert baseline_rows[0]["alignment_reason"] == "async_unguarded"


@pytest.mark.parametrize(
    ("mode", "decision"),
    [
        (ASYNC_UNGUARDED, "success_during_inference_delay"),
        (LATENCY_ALIGNED, "step_budget_exhausted_during_delay"),
    ],
)
def test_catchup_terminal_query_is_not_mislabeled_as_a_dispatch(mode, decision):
    cell = _cell(mode, 0)
    result = _result(cell, 0.0)
    result.query_records[0] = dataclasses.replace(
        result.query_records[0],
        accepted=False,
        decision=decision,
    )
    result.accepted_chunks = 0
    result.stale_chunks_executed = 0
    result.stale_action_steps = 0

    _, rows = result_rows(
        cell,
        result,
        "test task",
        7,
        1.0,
        (0.0,),
        50.0,
    )

    assert rows[0]["alignment_disposition"] == "terminal_before_dispatch"
    assert rows[0]["alignment_reason"] == decision
    assert rows[0]["accepted"] is False


def test_runtime_config_wiring_is_measured_wall_and_has_no_fixed_delay():
    args = _args(
        control_period_ms=40.0,
        age_rounding="floor",
        deadline_ms=180.0,
        max_age_refreshes=4,
        max_task_steps=123,
        video_mode="all",
    )
    config = make_runtime_config(args, _cell(LATENCY_ALIGNED, 0))

    assert config.latency_source == MEASURED_WALL_LATENCY
    assert config.latency_steps == 0
    assert config.control_period_ms == 40.0
    assert config.age_rounding == "floor"
    assert config.deadline_ms == 180.0
    assert config.max_age_refreshes == 4
    assert config.max_task_steps == 123
    assert config.record_video is True


def test_writer_emits_v2_manifest_bound_complete_artifact(tmp_path):
    cells = [_cell(ASYNC_UNGUARDED, 0), _cell(LATENCY_ALIGNED, 1)]
    args = _args()
    candidates = (0.0, 40.0, 80.0, 160.0)
    protocol = resolved_protocol(args, cells, candidates)
    episodes = []
    queries = []
    for cell in cells:
        jitter = jitter_value_ms(cell, args.seed, 0, candidates)
        episode, query_rows = result_rows(
            cell,
            _result(cell, jitter),
            "test task",
            args.seed,
            1.0,
            candidates,
            args.control_period_ms,
        )
        episodes.append(episode)
        queries.extend(query_rows)
    environment = {
        "schema_version": SCHEMA_VERSION,
        "client_session_id": "session",
    }

    integrity = write_artifacts(
        tmp_path,
        protocol,
        environment,
        _warmups(),
        episodes,
        queries,
        planned_rollouts=2,
        complete=True,
    )

    assert integrity["valid"] is True
    expected = {
        "resolved_protocol.json",
        "environment.json",
        "warmup_queries.csv",
        "per_episode.csv",
        "per_query.csv",
        "progress.json",
        "summary.json",
        "summary.md",
        "integrity.json",
        "manifest.json",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    assert (tmp_path / "videos").is_dir()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == SCHEMA_VERSION
    for relative, record in manifest["files"].items():
        assert hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest() == record[
            "sha256"
        ]
    with (tmp_path / "per_query.csv").open(encoding="utf-8", newline="") as handle:
        assert all(row["schema_version"] == SCHEMA_VERSION for row in csv.DictReader(handle))
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["valid"] is True
    assert summary["warmup_queries_planned"] == 3
    assert summary["warmup_queries_completed"] == 3
    assert summary["paired"]["pairs"] == 1
    assert summary["paired"]["ties"] == 1


def test_error_query_finalizes_invalid_instead_of_crashing_writer(tmp_path):
    cell = _cell(LATENCY_ALIGNED, 0)
    candidates = (0.0,)
    failed_query = QueryRecord(
        query_index=0,
        observation_step=10,
        response_step=10,
        inference_latency_ms=5.0,
        injected_latency_steps_requested=0,
        injected_latency_steps_executed=0,
        action_chunk_steps=0,
        accepted=False,
        decision="policy_response_validation_error",
        rejection_reasons=(),
        mismatch=None,
        error_stage="policy_response_validation",
        error_type="ValueError",
        error_message="bad shape",
        latency_source=MEASURED_WALL_LATENCY,
        observation_age_ms=10.0,
        response_jitter_ms=0.0,
        observation_captured_monotonic_ns=1_000_000_000,
        policy_call_started_monotonic_ns=1_001_000_000,
        policy_call_finished_monotonic_ns=1_005_000_000,
        response_ready_monotonic_ns=1_010_000_000,
        response_delivery_elapsed_ms=5.0,
    )
    result = EpisodeResult(
        success=False,
        termination_reason="invalid_policy_response",
        initial_state_sha256="b" * 64,
        environment_steps=10,
        task_action_steps=0,
        latency_action_steps=0,
        policy_queries=1,
        accepted_chunks=0,
        rejected_chunks=0,
        stale_chunks_executed=0,
        stale_action_steps=0,
        interventions=0,
        query_records=[failed_query],
        replay_frames=[],
        failure_category="policy_contract",
        failure_type="ValueError",
        failure_message="bad shape",
    )
    episode, queries = result_rows(
        cell, result, "test task", 7, 1.0, candidates, 50.0
    )
    protocol = resolved_protocol(_args(warmup_queries=3), [cell], candidates)

    errors = artifact_errors(
        protocol, _warmups(), [episode], queries, planned_rollouts=1, complete=True
    )
    assert any("records runtime error" in error for error in errors)
    integrity = write_artifacts(
        tmp_path,
        protocol,
        {"schema_version": SCHEMA_VERSION},
        _warmups(),
        [episode],
        queries,
        planned_rollouts=1,
        complete=True,
    )
    assert integrity["valid"] is False
    assert (tmp_path / "manifest.json").is_file()


def test_failed_warmup_finalizes_invalid_with_zero_scored_rollouts(tmp_path):
    cell = _cell(ASYNC_UNGUARDED, 0)
    protocol = resolved_protocol(_args(), [cell], (0.0,))
    warmups = _warmups()
    warmups[1]["accepted"] = False
    warmups[1]["error_type"] = "TimeoutError"
    warmups[1]["error_message"] = "warmup timed out"
    warmups = warmups[:2]

    integrity = write_artifacts(
        tmp_path,
        protocol,
        {"schema_version": SCHEMA_VERSION},
        warmups,
        [],
        [],
        planned_rollouts=1,
        complete=False,
    )

    assert integrity["valid"] is False
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed_rollouts"] == 0
    assert summary["warmup_queries_planned"] == 3
    assert summary["warmup_queries_completed"] == 2
    assert summary["warmup_queries_valid"] == 1
