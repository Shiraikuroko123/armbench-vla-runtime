from __future__ import annotations

import csv
import hashlib
import json
import pathlib

import pytest

from integrations.openpi.libero_runtime import (
    ASYNC_UNGUARDED,
    FIXED_REFRESH,
    STATE_GUARD,
    EpisodeResult,
    QueryRecord,
    StateMismatch,
)
from integrations.openpi.libero_runtime_eval import (
    DEFAULT_CHECKPOINT,
    DEFAULT_POLICY_CONFIG,
    OPENPI_COMMIT,
    RUNTIME_SOURCE_FILES,
    SCHEMA_VERSION,
    SERVER_ATTESTATION_SCHEMA_VERSION,
    build_matrix,
    episode_rows,
    matrix_plan,
    write_run_artifacts,
)
from integrations.openpi.validate_libero_artifact import main, validate_artifact


def _result(mode: str, latency_steps: int, success: bool) -> EpisodeResult:
    query = QueryRecord(
        query_index=0,
        observation_step=10,
        response_step=10 + latency_steps,
        inference_latency_ms=25.0 + latency_steps,
        injected_latency_steps_requested=latency_steps,
        injected_latency_steps_executed=latency_steps,
        action_chunk_steps=10,
        accepted=True,
        decision=(
            "accepted_unguarded"
            if mode == ASYNC_UNGUARDED
            else (
                "accepted_state_guard"
                if mode == STATE_GUARD
                else "accepted_fixed_refresh"
            )
        ),
        rejection_reasons=(),
        mismatch=StateMismatch(0.0, 0.0, 0.0),
        policy_inference_latency_ms=20.0 + latency_steps,
        server_inference_latency_ms=22.0 + latency_steps,
    )
    return EpisodeResult(
        success=success,
        termination_reason="task_success" if success else "step_limit",
        initial_state_sha256="a" * 64,
        environment_steps=12,
        task_action_steps=12 - latency_steps,
        latency_action_steps=latency_steps,
        policy_queries=1,
        accepted_chunks=1,
        rejected_chunks=0,
        stale_chunks_executed=1 if latency_steps else 0,
        stale_action_steps=1 if latency_steps else 0,
        interventions=0,
        query_records=[query],
        replay_frames=[],
    )


def _make_artifact(root: pathlib.Path) -> pathlib.Path:
    cells = build_matrix(
        "libero_spatial",
        task_ids=[0],
        episode_indices=[0],
        modes=[ASYNC_UNGUARDED, STATE_GUARD, FIXED_REFRESH],
        replan_steps=[5],
        latency_steps=[0, 2],
    )
    episodes = []
    queries = []
    for cell in cells:
        success = cell.mode == STATE_GUARD or cell.latency_steps == 0
        episode, episode_queries = episode_rows(
            cell,
            _result(cell.mode, cell.latency_steps, success),
            "pick up the black bowl",
            seed=7,
            wall_time_s=1.0 + cell.condition_order,
            video_path=None,
            fixed_refresh_interval=2 if cell.mode == FIXED_REFRESH else None,
        )
        episodes.append(episode)
        queries.extend(episode_queries)

    protocol = {
        "schema_version": SCHEMA_VERSION,
        "openpi_commit": OPENPI_COMMIT,
        "policy_config": DEFAULT_POLICY_CONFIG,
        "declared_checkpoint": DEFAULT_CHECKPOINT,
        "server_launch_args": "--env LIBERO",
        "checkpoint_provenance": "server_attestation_with_checkpoint_content_sha256",
        "official_protocol": {"control_period_ms": 50.0},
        "experimental_mechanism": {"fixed_refresh_interval": 2},
        "thresholds": {
            "position_m": 0.01,
            "orientation_rad": 0.10,
            "gripper_linf": 0.05,
            "max_requeries": 2,
        },
        "matrix": matrix_plan(cells),
        "seed": 7,
        "bootstrap_resamples": 100,
    }
    attestation = {
        "schema_version": SERVER_ATTESTATION_SCHEMA_VERSION,
        "policy_loaded": True,
        "policy_config": DEFAULT_POLICY_CONFIG,
        "checkpoint_uri": DEFAULT_CHECKPOINT,
        "openpi_commit": OPENPI_COMMIT,
        "openpi_tracked_clean": True,
        "openpi_tracked_status": "",
        "openpi_submodules_clean": True,
        "action_horizon": 10,
        "checkpoint_content_sha256": "b" * 64,
        "server_source_sha256": "c" * 64,
        "checkpoint_file_count": 4,
        "checkpoint_total_bytes": 1024,
    }
    source_hashes = {}
    for index, relative in enumerate(RUNTIME_SOURCE_FILES):
        snapshot = root / "provenance" / "armbench_source" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(("source-%d\n" % index).encode("ascii"))
        source_hashes[relative] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    attestation["server_source_sha256"] = source_hashes[
        "integrations/openpi/serve_policy_attested.py"
    ]
    environment = {
        "schema_version": SCHEMA_VERSION,
        "openpi_git_commit": OPENPI_COMMIT,
        "armbench_git_diff_sha256": "d" * 64,
        "runtime_source_sha256": source_hashes,
        "server_metadata": {"armbench_server_attestation": attestation},
        "arguments": {
            "allow_unattested_server": False,
            "checkpoint": DEFAULT_CHECKPOINT,
            "expected_openpi_commit": OPENPI_COMMIT,
            "server_launch_args": "--env LIBERO",
        },
    }
    write_run_artifacts(
        root,
        episodes,
        queries,
        protocol,
        environment,
        planned_rollouts=len(cells),
        bootstrap_resamples=100,
        final=True,
        expected_cells=cells,
    )
    return root


@pytest.fixture
def artifact(tmp_path: pathlib.Path) -> pathlib.Path:
    return _make_artifact(tmp_path / "artifact")


def _read_csv(path: pathlib.Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _write_csv(path: pathlib.Path, rows, fields) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _refresh_manifest(root: pathlib.Path) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json" or path.suffix == ".tmp":
            continue
        files[path.relative_to(root).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest["files"] = files
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_valid_artifact_recomputes_all_claims_and_cli_succeeds(
    artifact: pathlib.Path, capsys
) -> None:
    report = validate_artifact(artifact)

    assert report.valid, "\n".join(report.errors)
    assert not report.warnings
    assert main([str(artifact), "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is True


def test_manifest_detects_unresigned_file_tampering(artifact: pathlib.Path) -> None:
    with (artifact / "summary.md").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")

    report = validate_artifact(artifact)

    assert not report.valid
    assert any("SHA-256 mismatch for summary.md" in error for error in report.errors)


def test_resigned_raw_success_tampering_breaks_recomputed_claims(
    artifact: pathlib.Path,
) -> None:
    path = artifact / "per_episode.csv"
    rows, fields = _read_csv(path)
    rows[0]["success"] = "False" if rows[0]["success"] == "True" else "True"
    _write_csv(path, rows, fields)
    _refresh_manifest(artifact)

    report = validate_artifact(artifact)

    assert not report.valid
    assert any("aggregate.json" in error for error in report.errors)
    assert any("summary.md" in error for error in report.errors)


@pytest.mark.parametrize(
    "stem",
    [
        "aggregate",
        "paired_comparisons",
        "intervention_control_comparisons",
    ],
)
def test_resigned_derived_json_tampering_is_recomputed(
    artifact: pathlib.Path, stem: str
) -> None:
    path = artifact / (stem + ".json")
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows[0]["schema_version"] = "forged"
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest(artifact)

    report = validate_artifact(artifact)

    assert not report.valid
    assert any(stem + ".json" in error for error in report.errors)


def test_missing_pair_member_is_detected_even_with_resigned_manifest(
    artifact: pathlib.Path,
) -> None:
    episode_path = artifact / "per_episode.csv"
    episodes, episode_fields = _read_csv(episode_path)
    removed_id = episodes[-1]["episode_id"]
    _write_csv(episode_path, episodes[:-1], episode_fields)
    query_path = artifact / "per_query.csv"
    queries, query_fields = _read_csv(query_path)
    _write_csv(
        query_path,
        [row for row in queries if row["episode_id"] != removed_id],
        query_fields,
    )
    _refresh_manifest(artifact)

    report = validate_artifact(artifact)

    assert not report.valid
    assert any("missing planned episodes" in error for error in report.errors)
    assert any("has 2 modes, expected 3" in error for error in report.errors)


def test_query_index_and_policy_query_count_are_cross_checked(
    artifact: pathlib.Path,
) -> None:
    path = artifact / "per_query.csv"
    rows, fields = _read_csv(path)
    rows[0]["query_index"] = "3"
    _write_csv(path, rows, fields)
    _refresh_manifest(artifact)

    report = validate_artifact(artifact)

    assert not report.valid
    assert any("query_index is not contiguous" in error for error in report.errors)


def test_query_count_must_equal_episode_declaration(artifact: pathlib.Path) -> None:
    path = artifact / "per_query.csv"
    rows, fields = _read_csv(path)
    _write_csv(path, rows[1:], fields)
    _refresh_manifest(artifact)

    report = validate_artifact(artifact)

    assert not report.valid
    assert any("policy_queries count mismatch" in error for error in report.errors)


def test_three_mode_order_must_reverse_between_condition_groups(
    artifact: pathlib.Path,
) -> None:
    path = artifact / "per_episode.csv"
    rows, fields = _read_csv(path)
    second_pair = rows[3:6]
    by_mode = {row["mode"]: row for row in second_pair}
    by_mode[ASYNC_UNGUARDED]["condition_order"] = "3"
    by_mode[STATE_GUARD]["condition_order"] = "4"
    by_mode[FIXED_REFRESH]["condition_order"] = "5"
    _write_csv(path, rows, fields)
    _refresh_manifest(artifact)

    report = validate_artifact(artifact)

    assert not report.valid
    assert any("alternating mode order" in error for error in report.errors)


def test_fixed_refresh_decision_schedule_is_independently_checked(
    artifact: pathlib.Path,
) -> None:
    path = artifact / "per_query.csv"
    rows, fields = _read_csv(path)
    fixed = next(row for row in rows if row["mode"] == FIXED_REFRESH)
    fixed["decision"] = "rejected_fixed_refresh"
    fixed["accepted"] = "False"
    fixed["rejection_reasons"] = "scheduled_refresh"
    _write_csv(path, rows, fields)
    _refresh_manifest(artifact)

    report = validate_artifact(artifact)

    assert not report.valid
    assert any("does not follow fixed_refresh rejection policy" in error for error in report.errors)


def test_raw_booleans_are_not_parsed_by_truthiness(artifact: pathlib.Path) -> None:
    path = artifact / "per_episode.csv"
    rows, fields = _read_csv(path)
    rows[0]["success"] = "1"
    _write_csv(path, rows, fields)
    _refresh_manifest(artifact)

    report = validate_artifact(artifact)

    assert not report.valid
    assert any("must be exactly True or False" in error for error in report.errors)


def test_resigned_summary_count_tampering_is_detected(artifact: pathlib.Path) -> None:
    path = artifact / "summary.md"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("Completed rollouts: 6", "Completed rollouts: 600"), encoding="utf-8")
    _refresh_manifest(artifact)

    report = validate_artifact(artifact)

    assert not report.valid
    assert any("deterministic raw-data summary" in error for error in report.errors)


def test_integrity_cannot_falsely_claim_validity(artifact: pathlib.Path) -> None:
    path = artifact / "integrity.json"
    integrity = json.loads(path.read_text(encoding="utf-8"))
    integrity["errors"] = ["fabricated independent check"]
    integrity["valid"] = True
    path.write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_manifest(artifact)

    report = validate_artifact(artifact)

    assert not report.valid
    assert any("errors contradict independent integrity check" in error for error in report.errors)


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("checkpoint_uri", "gs://forged/checkpoint", "checkpoint_uri mismatch"),
        ("policy_config", "forged_config", "policy_config mismatch"),
        (
            "checkpoint_content_sha256",
            "not-a-sha256",
            "checkpoint_content_sha256 is not a lowercase SHA-256",
        ),
        ("openpi_submodules_clean", False, "openpi_submodules_clean mismatch"),
        (
            "server_source_sha256",
            "e" * 64,
            "server attestation does not match the source snapshot",
        ),
    ],
)
def test_attestation_mismatch_is_detected_after_resigning(
    artifact: pathlib.Path, field: str, value: str, expected_error: str
) -> None:
    path = artifact / "environment.json"
    environment = json.loads(path.read_text(encoding="utf-8"))
    environment["server_metadata"]["armbench_server_attestation"][field] = value
    path.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_manifest(artifact)

    report = validate_artifact(artifact)

    assert not report.valid
    assert any(expected_error in error for error in report.errors)


def test_explicit_diagnostic_bypass_is_valid_but_warned(artifact: pathlib.Path) -> None:
    protocol_path = artifact / "resolved_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["checkpoint_provenance"] = "launcher_declaration_only"
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    environment_path = artifact / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["arguments"]["allow_unattested_server"] = True
    environment["server_metadata"].pop("armbench_server_attestation")
    environment_path.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_manifest(artifact)

    report = validate_artifact(artifact)

    assert report.valid, "\n".join(report.errors)
    assert any("diagnostic launcher-only" in warning for warning in report.warnings)


def test_manifest_rejects_path_traversal_entry(artifact: pathlib.Path) -> None:
    path = artifact / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"]["../escape.txt"] = {"bytes": 0, "sha256": "0" * 64}
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = validate_artifact(artifact)

    assert not report.valid
    assert any("unsafe or non-canonical path" in error for error in report.errors)


def test_cli_returns_nonzero_for_invalid_artifact(
    artifact: pathlib.Path, capsys
) -> None:
    (artifact / "aggregate.json").write_text("[]\n", encoding="utf-8")

    assert main([str(artifact)]) == 2
    assert capsys.readouterr().out.startswith("INVALID")
