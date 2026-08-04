from __future__ import annotations

import hashlib
import json
import logging

import numpy as np
import pytest

import integrations.openpi.libero_runtime_eval as runtime_eval

from integrations.openpi.libero_runtime import (
    ASYNC_UNGUARDED,
    FIXED_REFRESH,
    STATE_GUARD,
    EpisodeResult,
    QueryRecord,
    StateMismatch,
)
from integrations.openpi.libero_runtime_eval import (
    BoundedOpenPIClient,
    DEFAULT_CHECKPOINT,
    ExperimentCell,
    LIBERO_CONTROL_PERIOD_MS,
    LIBERO_ENV_RESOLUTION,
    _validate_server_launch_args,
    _validate_server_attestation,
    aggregate_episodes,
    artifact_integrity_errors,
    build_matrix,
    episode_rows,
    execute_benchmark,
    intervention_control_comparisons,
    main,
    matrix_plan,
    matched_condition_contrasts,
    paired_comparisons,
    snapshot_runtime_sources,
    write_run_artifacts,
)


def _result(success: bool, mode: str) -> EpisodeResult:
    mismatch = StateMismatch(0.02 if mode == STATE_GUARD else 0.01, 0.0, 0.0)
    query = QueryRecord(
        query_index=0,
        observation_step=10,
        response_step=12,
        inference_latency_ms=25.0,
        injected_latency_steps_requested=2,
        injected_latency_steps_executed=2,
        action_chunk_steps=10,
        accepted=mode == ASYNC_UNGUARDED,
        decision=(
            "accepted_unguarded"
            if mode == ASYNC_UNGUARDED
            else "rejected_state_mismatch"
        ),
        rejection_reasons=() if mode == ASYNC_UNGUARDED else ("position_mismatch",),
        mismatch=mismatch,
        policy_inference_latency_ms=20.0,
        server_inference_latency_ms=22.0,
    )
    return EpisodeResult(
        success=success,
        termination_reason="task_success" if success else "step_limit",
        initial_state_sha256="a" * 64,
        environment_steps=20,
        task_action_steps=18,
        latency_action_steps=2,
        policy_queries=1,
        accepted_chunks=1 if mode == ASYNC_UNGUARDED else 0,
        rejected_chunks=0 if mode == ASYNC_UNGUARDED else 1,
        stale_chunks_executed=1 if mode == ASYNC_UNGUARDED else 0,
        stale_action_steps=1 if mode == ASYNC_UNGUARDED else 0,
        interventions=0 if mode == ASYNC_UNGUARDED else 1,
        query_records=[query],
        replay_frames=[],
    )


def _cell(
    mode: str,
    order: int,
    *,
    replan_steps: int = 5,
    latency_steps: int = 2,
) -> ExperimentCell:
    return ExperimentCell(
        task_suite="libero_spatial",
        task_id=0,
        episode_index=0,
        mode=mode,
        replan_steps=replan_steps,
        latency_steps=latency_steps,
        pair_id="libero_spatial/task_000/episode_000/h_%02d/l_%03d"
        % (replan_steps, latency_steps),
        condition_order=order,
    )


def test_matrix_keeps_pairs_adjacent_and_alternates_mode_order() -> None:
    cells = build_matrix(
        "libero_spatial",
        task_ids=[0],
        episode_indices=[0, 1],
        modes=[ASYNC_UNGUARDED, STATE_GUARD],
        replan_steps=[5],
        latency_steps=[0, 2],
    )

    assert matrix_plan(cells)["rollouts"] == 8
    for offset in range(0, len(cells), 2):
        assert cells[offset].pair_id == cells[offset + 1].pair_id
    assert [cell.mode for cell in cells[:4]] == [
        ASYNC_UNGUARDED,
        STATE_GUARD,
        STATE_GUARD,
        ASYNC_UNGUARDED,
    ]
    assert LIBERO_ENV_RESOLUTION == 256
    assert LIBERO_CONTROL_PERIOD_MS == 50.0


def test_fixed_refresh_plan_requires_explicit_interval(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["plan", "--modes", FIXED_REFRESH])
    assert error.value.code == 2

    assert (
        main(
            [
                "plan",
                "--modes",
                "%s,%s,%s" % (ASYNC_UNGUARDED, STATE_GUARD, FIXED_REFRESH),
                "--fixed-refresh-interval",
                "4",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["rollouts"] == 3
    assert plan["matched_condition_groups"] == 1
    assert plan["fixed_refresh_interval"] == 4


def test_server_launch_provenance_requires_matching_libero_checkpoint() -> None:
    _validate_server_launch_args("--env LIBERO", DEFAULT_CHECKPOINT)
    _validate_server_launch_args(
        "policy:checkpoint --policy.config pi05_libero "
        "--policy.dir gs://openpi-assets/checkpoints/pi05_libero",
        DEFAULT_CHECKPOINT,
    )

    with np.testing.assert_raises_regex(ValueError, "provenance"):
        _validate_server_launch_args(None, DEFAULT_CHECKPOINT)
    with np.testing.assert_raises_regex(ValueError, "different path"):
        _validate_server_launch_args("--env LIBERO", "custom_checkpoint")


def test_server_attestation_is_strictly_bound_to_run_arguments() -> None:
    class Args:
        allow_unattested_server = False
        checkpoint = DEFAULT_CHECKPOINT
        expected_openpi_commit = "15a9616a00943ada6c20a0f158e3adb39df2ccac"

    attestation = {
        "schema_version": "armbench.openpi_server_attestation.v1",
        "policy_loaded": True,
        "policy_config": "pi05_libero",
        "checkpoint_uri": DEFAULT_CHECKPOINT,
        "openpi_commit": Args.expected_openpi_commit,
        "openpi_tracked_clean": True,
        "openpi_tracked_status": "",
        "openpi_submodules_clean": True,
        "action_horizon": 10,
        "checkpoint_content_sha256": "a" * 64,
        "server_source_sha256": "b" * 64,
        "checkpoint_file_count": 4,
        "checkpoint_total_bytes": 100,
    }

    assert (
        _validate_server_attestation(
            {"armbench_server_attestation": attestation}, Args(), "b" * 64
        )
        == attestation
    )
    bad = dict(attestation, policy_config="wrong")
    with np.testing.assert_raises_regex(ValueError, "policy_config"):
        _validate_server_attestation(
            {"armbench_server_attestation": bad}, Args(), "b" * 64
        )

    with np.testing.assert_raises_regex(ValueError, "server_source_sha256"):
        _validate_server_attestation(
            {"armbench_server_attestation": attestation}, Args(), "c" * 64
        )


def test_bounded_openpi_client_records_metadata_and_applies_recv_timeout() -> None:
    class Codec:
        def pack(self, value):
            return b"request"

        def unpack(self, value):
            if value == b"metadata":
                return {"model": "test_pi05_libero"}
            return {"actions": np.zeros((5, 7))}

    class Connection:
        def __init__(self):
            self.received_timeouts = []
            self.sent = []
            self.closed = False
            self.responses = [b"metadata", b"response"]

        def recv(self, timeout):
            self.received_timeouts.append(timeout)
            return self.responses.pop(0)

        def send(self, value):
            self.sent.append(value)

        def close(self):
            self.closed = True

    connection = Connection()
    client = BoundedOpenPIClient(
        "localhost",
        8000,
        startup_timeout_s=10.0,
        inference_timeout_s=3.0,
        connect_fn=lambda *args, **kwargs: connection,
        codec=Codec(),
    )

    assert client.get_server_metadata() == {"model": "test_pi05_libero"}
    assert client.infer({"test": True})["actions"].shape == (5, 7)
    assert connection.sent == [b"request"]
    assert connection.received_timeouts[-1] == 3.0
    client.close()
    assert connection.closed


def test_bounded_openpi_client_closes_connection_after_timeout() -> None:
    class Codec:
        def pack(self, value):
            return b"request"

        def unpack(self, value):
            return {"model": "test"}

    class Connection:
        def __init__(self):
            self.calls = 0
            self.closed = False

        def recv(self, timeout):
            self.calls += 1
            if self.calls == 1:
                return b"metadata"
            raise TimeoutError("slow policy")

        def send(self, value):
            return None

        def close(self):
            self.closed = True

    connection = Connection()
    client = BoundedOpenPIClient(
        "localhost",
        8000,
        startup_timeout_s=10.0,
        inference_timeout_s=0.1,
        connect_fn=lambda *args, **kwargs: connection,
        codec=Codec(),
    )

    with np.testing.assert_raises_regex(TimeoutError, "slow policy"):
        client.infer({"test": True})
    assert client.broken
    assert connection.closed
    with np.testing.assert_raises_regex(ConnectionError, "prior protocol failure"):
        client.infer({"test": True})


def test_aggregate_and_paired_effect_use_episode_as_statistical_unit() -> None:
    baseline_row, baseline_queries = episode_rows(
        _cell(ASYNC_UNGUARDED, 0),
        _result(False, ASYNC_UNGUARDED),
        "test task",
        7,
        1.0,
        None,
    )
    guard_row, guard_queries = episode_rows(
        _cell(STATE_GUARD, 1),
        _result(True, STATE_GUARD),
        "test task",
        7,
        1.0,
        None,
    )
    episodes = [baseline_row, guard_row]
    queries = baseline_queries + guard_queries

    aggregate = aggregate_episodes(episodes, queries)
    overall = [row for row in aggregate if row["scope"] == "selected_tasks"]
    assert len(overall) == 2
    assert {row["policy_inference_latency_p50_ms"] for row in overall} == {20.0}
    assert {row["server_inference_latency_p95_ms"] for row in overall} == {22.0}
    comparison = [
        row
        for row in paired_comparisons(episodes, bootstrap_resamples=100)
        if row["scope"] == "selected_tasks"
    ][0]
    assert comparison["paired_episodes"] == 1
    assert comparison["success_rate_difference"] == 1.0
    assert comparison["guard_wins"] == 1
    assert comparison["mcnemar_exact_p"] == 1.0
    assert comparison["mcnemar_holm_p"] == 1.0


def test_intervention_control_reports_effect_and_cost_matching() -> None:
    fixed_row, _ = episode_rows(
        _cell(FIXED_REFRESH, 0),
        _result(False, FIXED_REFRESH),
        "test task",
        7,
        1.0,
        None,
        fixed_refresh_interval=4,
    )
    guard_row, _ = episode_rows(
        _cell(STATE_GUARD, 1),
        _result(True, STATE_GUARD),
        "test task",
        7,
        1.0,
        None,
        fixed_refresh_interval=4,
    )
    fixed_row["policy_queries"] = 2
    assert fixed_row["fixed_refresh_interval"] == 4
    assert guard_row["fixed_refresh_interval"] is None

    selected = [
        row
        for row in intervention_control_comparisons(
            [fixed_row, guard_row], bootstrap_resamples=100
        )
        if row["scope"] == "selected_tasks"
    ][0]

    assert selected["reference_mode"] == FIXED_REFRESH
    assert selected["candidate_mode"] == STATE_GUARD
    assert selected["fixed_refresh_interval"] == 4
    assert selected["success_rate_difference"] == 1.0
    assert selected["mean_policy_query_difference"] == -1.0
    assert selected["policy_query_count_matched_pairs"] == 0
    assert selected["intervention_count_matched_pairs"] == 1


def test_condition_contrasts_compare_delay_against_zero() -> None:
    reference_row, _ = episode_rows(
        _cell(ASYNC_UNGUARDED, 0, latency_steps=0),
        _result(True, ASYNC_UNGUARDED),
        "test task",
        7,
        1.0,
        None,
    )
    delayed_row, _ = episode_rows(
        _cell(ASYNC_UNGUARDED, 1, latency_steps=2),
        _result(False, ASYNC_UNGUARDED),
        "test task",
        7,
        1.0,
        None,
    )

    contrasts = matched_condition_contrasts(
        [reference_row, delayed_row], bootstrap_resamples=100
    )

    assert len(contrasts) == 1
    assert contrasts[0]["contrast_type"] == "delay_vs_zero"
    assert contrasts[0]["matched_pairs"] == 1
    assert contrasts[0]["success_rate_difference"] == -1.0


def test_artifact_writer_retains_failures_and_hashes_every_file(tmp_path) -> None:
    baseline_row, query_rows = episode_rows(
        _cell(ASYNC_UNGUARDED, 0),
        _result(False, ASYNC_UNGUARDED),
        "test task",
        7,
        1.25,
        None,
    )
    baseline_row["failure_type"] = "ConnectionError"
    baseline_row["failure_category"] = "policy_transport_or_server"
    baseline_row["failure_message"] = "test disconnect"
    protocol = {"schema_version": "test", "checkpoint": "test_only"}
    environment = {"schema_version": "test", "gpu": "fake"}

    write_run_artifacts(
        tmp_path,
        [baseline_row],
        query_rows,
        protocol,
        environment,
        planned_rollouts=1,
        bootstrap_resamples=100,
        final=True,
    )

    required = {
        "manifest.json",
        "resolved_protocol.json",
        "per_episode.csv",
        "per_query.csv",
        "aggregate.json",
        "aggregate.csv",
        "paired_comparisons.json",
        "paired_comparisons.csv",
        "intervention_control_comparisons.json",
        "intervention_control_comparisons.csv",
        "environment.json",
        "progress.json",
        "summary.md",
    }
    assert required <= {path.name for path in tmp_path.iterdir()}
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    for relative, record in manifest["files"].items():
        assert hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest() == record["sha256"]
    assert "runtime/contract failures retained: 1" in (tmp_path / "summary.md").read_text(
        encoding="utf-8"
    )


def test_runtime_source_snapshot_preserves_exact_bytes(tmp_path) -> None:
    armbench_root = tmp_path / "armbench"
    source_root = armbench_root / "integrations" / "openpi"
    source_root.mkdir(parents=True)
    (source_root / "libero_runtime.py").write_bytes(b"runtime source\n")
    (source_root / "libero_runtime_eval.py").write_bytes(b"evaluator source\n")
    output = tmp_path / "output"

    snapshot_runtime_sources(armbench_root, output)

    assert (
        output
        / "provenance"
        / "armbench_source"
        / "integrations"
        / "openpi"
        / "libero_runtime.py"
    ).read_bytes() == b"runtime source\n"


def test_integrity_rejects_mismatched_pair_and_missing_required_video(tmp_path) -> None:
    baseline_row, _ = episode_rows(
        _cell(ASYNC_UNGUARDED, 0),
        _result(False, ASYNC_UNGUARDED),
        "test task",
        7,
        1.0,
        None,
        video_required=True,
    )
    guard_row, _ = episode_rows(
        _cell(STATE_GUARD, 1),
        _result(True, STATE_GUARD),
        "test task",
        8,
        1.0,
        None,
    )

    errors = artifact_integrity_errors(
        tmp_path,
        [baseline_row, guard_row],
        [_cell(ASYNC_UNGUARDED, 0), _cell(STATE_GUARD, 1)],
    )

    assert any("matched condition mismatch" in error for error in errors)
    assert any("required video missing" in error for error in errors)


def test_integrity_rejects_recorded_runtime_error(tmp_path) -> None:
    (tmp_path / "run_error.json").write_text(
        json.dumps({"errors": [{"stage": "environment_close"}]}),
        encoding="utf-8",
    )

    errors = artifact_integrity_errors(tmp_path, [])

    assert any("runtime/teardown errors recorded" in error for error in errors)


def test_infrastructure_failure_is_excluded_from_efficacy_pair() -> None:
    baseline_row, baseline_queries = episode_rows(
        _cell(ASYNC_UNGUARDED, 0),
        _result(False, ASYNC_UNGUARDED),
        "test task",
        7,
        1.0,
        None,
    )
    guard_row, guard_queries = episode_rows(
        _cell(STATE_GUARD, 1),
        _result(True, STATE_GUARD),
        "test task",
        7,
        1.0,
        None,
    )
    baseline_row["failure_type"] = "ConnectionError"
    baseline_row["failure_category"] = "policy_transport_or_server"
    rows = [baseline_row, guard_row]

    baseline_aggregate = [
        row
        for row in aggregate_episodes(rows, baseline_queries + guard_queries)
        if row["scope"] == "selected_tasks" and row["mode"] == ASYNC_UNGUARDED
    ][0]
    assert baseline_aggregate["rollouts"] == 1
    assert baseline_aggregate["eligible_rollouts"] == 0
    assert baseline_aggregate["success_rate"] == 0.0
    assert baseline_aggregate["per_protocol_success_rate"] is None

    comparison = [
        row
        for row in paired_comparisons(rows, bootstrap_resamples=100)
        if row["scope"] == "selected_tasks"
    ][0]
    assert comparison["candidate_pairs"] == 1
    assert comparison["paired_episodes"] == 1
    assert comparison["per_protocol_pairs"] == 0
    assert comparison["excluded_infrastructure_pairs"] == 1
    assert comparison["success_rate_difference"] == 1.0
    assert comparison["per_protocol_success_rate_difference"] is None


def test_execute_benchmark_finalizes_unhandled_failure_artifacts(
    tmp_path, monkeypatch
) -> None:
    class Args:
        output_dir = str(tmp_path / "run")
        openpi_root = str(tmp_path / "openpi")
        armbench_root = str(tmp_path / "armbench")
        expected_openpi_commit = "a" * 40
        allow_commit_mismatch = False
        host = "localhost"
        port = 8000
        server_startup_timeout_s = 1.0
        inference_timeout_s = 1.0
        seed = 7

    class Client:
        instances = []

        def __init__(self, *args, **kwargs):
            self.closed = False
            self.instances.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(runtime_eval, "_command_output", lambda *args, **kwargs: "a" * 40)
    monkeypatch.setattr(runtime_eval, "snapshot_runtime_sources", lambda *args: None)
    monkeypatch.setattr(runtime_eval, "BoundedOpenPIClient", Client)

    def fail(*args, **kwargs):
        raise RuntimeError("bad attestation")

    monkeypatch.setattr(runtime_eval, "_execute_connected_benchmark", fail)

    assert execute_benchmark(Args(), []) == 2
    assert Client.instances[0].closed
    output = tmp_path / "run"
    run_error = json.loads((output / "run_error.json").read_text(encoding="utf-8"))
    assert run_error["errors"] == [
        {
            "stage": "connected_benchmark",
            "type": "RuntimeError",
            "message": "bad attestation",
        }
    ]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert {"run.log", "run_error.json"} <= set(manifest["files"])
    for relative, record in manifest["files"].items():
        assert hashlib.sha256((output / relative).read_bytes()).hexdigest() == record["sha256"]

    log_before = (output / "run.log").read_bytes()
    logging.getLogger().warning("must not mutate finalized run log")
    assert (output / "run.log").read_bytes() == log_before
