from __future__ import annotations

from argparse import Namespace
import json
import pathlib

import numpy as np

from integrations.openpi.libero_independent_clock import (
    AGE_ALIGNED_SUFFIX,
    RESPONSE_RELATIVE_CHUNK,
    SCHEMA_VERSION,
    canonical_action_chunk_sha256,
)
from integrations.openpi.libero_independent_clock_eval import (
    ExperimentCell,
    _snapshot_sources,
    _write_derived,
    _write_json,
    _write_manifest,
    episode_row,
    resolve_cells,
    resolved_protocol,
)
from integrations.openpi.libero_runtime import initial_state_digest
from integrations.openpi.libero_runtime_eval import (
    DEFAULT_CHECKPOINT,
    OPENPI_COMMIT,
    SERVER_ATTESTATION_SCHEMA_VERSION,
)
from integrations.openpi.serve_policy_attested import (
    POLICY_SAMPLING_GENERATOR,
    build_policy_sampling_control,
    policy_sampling_contract,
    policy_sampling_noise,
    policy_sampling_noise_sha256,
)
from integrations.openpi.validate_libero_independent_clock import validate_artifact


def _args(**overrides) -> Namespace:
    values = {
        "task_suite": "libero_spatial",
        "task_ids": "0",
        "episode_indices": "0",
        "expected_openpi_commit": OPENPI_COMMIT,
        "checkpoint": DEFAULT_CHECKPOINT,
        "server_launch_args": (
            "--policy-config pi05_libero --checkpoint " + DEFAULT_CHECKPOINT
        ),
        "control_period_ms": 50.0,
        "deadline_ms": 200.0,
        "submit_every_ticks": 2,
        "action_selection_mode": AGE_ALIGNED_SUFFIX,
        "num_steps_wait": 10,
        "max_task_steps": 2,
        "seed": 7,
        "video_mode": "none",
    }
    values.update(overrides)
    return Namespace(**values)


def _sampling_audit(cell: ExperimentCell, submit_every: int) -> dict:
    control = build_policy_sampling_control(
        "scored",
        7,
        (cell.task_suite, cell.task_id, cell.episode_index, submit_every),
        0,
    )
    key = control["key_sha256"]
    return {
        "schema_version": policy_sampling_contract()["schema_version"],
        "namespace": "scored",
        "key_sha256": key,
        "noise_sha256": policy_sampling_noise_sha256(policy_sampling_noise(key)),
        "generator": POLICY_SAMPLING_GENERATOR,
    }


def _runtime_payload(
    cell: ExperimentCell,
    action_selection_mode: str = AGE_ALIGNED_SUFFIX,
) -> dict:
    actions = np.zeros((10, 7), dtype=np.float64)
    actions[:, 0] = np.arange(10, dtype=np.float64) / 100.0
    actions[:, -1] = -1.0
    request = {
        "request_id": 0,
        "observation_sequence_id": 0,
        "captured_at_s": 1.0,
        "submitted_at_s": 1.0,
        "started_at_s": 1.01,
        "completed_at_s": 1.02,
        "superseded_at_s": None,
        "response_age_ms": 20.0,
        "response_status": "accepted",
        "source": "official_openpi_pi05_libero",
        "actions": actions.tolist(),
        "response_metadata": {
            "action_chunk_sha256": canonical_action_chunk_sha256(actions),
            "policy_input_sha256": "c" * 64,
            "policy_sampling": _sampling_audit(cell, 2),
            "policy_inference_latency_ms": 10.0,
            "server_inference_latency_ms": 9.0,
        },
        "failure_type": None,
        "failure_message": None,
        "parent_process_id": 100,
        "worker_process_id": 200,
    }
    hold = [0.0] * 6 + [-1.0]
    ticks = [
        {
            "tick_index": 0,
            "scheduled_at_s": 1.0,
            "tick_started_at_s": 1.015,
            "observation_sequence_id": 0,
            "submitted_request_id": 0,
            "response_request_id": None,
            "response_age_ms": None,
            "deadline_ms": 200.0,
            "status": "hold",
            "reason": "no_policy_response",
            "stale_prefix_steps": 0,
            "stale_suffix_steps": 0,
            "available_suffix_steps": 0,
            "action_index": None,
            "action": hold,
            "environment_done": False,
            "parent_process_id": 100,
            "worker_process_id": 200,
        },
        {
            "tick_index": 1,
            "scheduled_at_s": 1.05,
            "tick_started_at_s": 1.025,
            "observation_sequence_id": 1,
            "submitted_request_id": None,
            "response_request_id": 0,
            "response_age_ms": 25.0,
            "deadline_ms": 200.0,
            "status": "execute",
            "reason": (
                "fresh_suffix_available"
                if action_selection_mode == AGE_ALIGNED_SUFFIX
                else "response_relative_chunk_available"
            ),
            "stale_prefix_steps": 1,
            "stale_suffix_steps": 9,
            "available_suffix_steps": 9,
            "action_index": (1 if action_selection_mode == AGE_ALIGNED_SUFFIX else 0),
            "action": actions[
                1 if action_selection_mode == AGE_ALIGNED_SUFFIX else 0
            ].tolist(),
            "environment_done": True,
            "parent_process_id": 100,
            "worker_process_id": 200,
        },
    ]
    state = np.asarray([0.125, 0.25], dtype=np.float64)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_success": True,
        "termination_reason": "task_success",
        "initial_state_sha256": initial_state_digest(state),
        "stabilization_steps": 10,
        "task_steps": 2,
        "cumulative_reward": 1.0,
        "final_info": {},
        "runtime": {
            "schema_version": "armbench.independent_clock.v1",
            "termination_reason": "environment_done",
            "parent_process_id": 100,
            "worker_process_id": 200,
            "worker_stopped": True,
            "environment_steps": 2,
            "environment_done": True,
            "tick_overruns": 0,
            "metrics": {
                "submitted": 1,
                "started": 1,
                "completed": 1,
                "superseded": 0,
                "holds": 1,
                "executes": 1,
            },
            "worker": {"queue_dropped": 0},
            "requests": [request],
            "ticks": ticks,
        },
    }


def _write_valid_artifact(
    root: pathlib.Path,
    action_selection_mode: str = AGE_ALIGNED_SUFFIX,
) -> None:
    project_root = pathlib.Path(__file__).resolve().parents[1]
    cell = ExperimentCell("libero_spatial", 0, 0)
    args = _args(action_selection_mode=action_selection_mode)
    root.mkdir()
    (root / "episodes" / cell.episode_id).mkdir(parents=True)
    (root / "videos").mkdir()
    source_hashes = _snapshot_sources(project_root, root)
    protocol = resolved_protocol(args, [cell])
    payload = _runtime_payload(cell, action_selection_mode)
    state = np.asarray([0.125, 0.25], dtype=np.float64)
    np.save(
        root / "episodes" / cell.episode_id / "initial_state.npy",
        state,
        allow_pickle=False,
    )
    _write_json(root / "episodes" / cell.episode_id / "runtime.json", payload)
    row = episode_row(
        cell,
        "pick up the black bowl",
        7,
        payload,
        wall_time_s=0.1,
        video_path=None,
    )
    _write_json(root / "resolved_protocol.json", protocol)
    _write_json(
        root / "environment.json",
        {
            "schema_version": SCHEMA_VERSION,
            "openpi_git_commit": OPENPI_COMMIT,
            "runtime_source_sha256": source_hashes,
            "server_metadata": {
                "armbench_server_attestation": {
                    "schema_version": SERVER_ATTESTATION_SCHEMA_VERSION,
                    "policy_loaded": True,
                    "policy_config": "pi05_libero",
                    "checkpoint_uri": DEFAULT_CHECKPOINT,
                    "openpi_commit": OPENPI_COMMIT,
                    "openpi_tracked_clean": True,
                    "openpi_submodules_clean": True,
                    "action_horizon": 10,
                    "checkpoint_content_sha256": "a" * 64,
                    "server_source_sha256": "b" * 64,
                },
                "armbench_policy_sampling_contract": policy_sampling_contract(),
            },
        },
    )
    _write_derived(root, [row], 1)
    _write_json(
        root / "integrity.json",
        {"schema_version": SCHEMA_VERSION, "valid": True, "errors": []},
    )
    _write_manifest(root)


def test_plan_resolves_forty_rollouts() -> None:
    cells = resolve_cells(_args(task_ids="all", episode_indices="0:4"))

    assert len(cells) == 40
    assert len({cell.episode_id for cell in cells}) == 40


def test_independent_validator_accepts_and_rejects_tampering(tmp_path) -> None:
    artifact = tmp_path / "artifact"
    _write_valid_artifact(artifact)

    assert validate_artifact(artifact).valid

    runtime_path = next(artifact.glob("episodes/*/runtime.json"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["runtime"]["ticks"][1]["action"][0] = 999.0
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    _write_manifest(artifact)

    report = validate_artifact(artifact)
    assert not report.valid
    assert any("executed action differs" in error for error in report.errors)


def test_response_relative_artifact_is_validated_and_action_index_is_bound(
    tmp_path,
) -> None:
    artifact = tmp_path / "response-relative"
    _write_valid_artifact(artifact, RESPONSE_RELATIVE_CHUNK)

    assert validate_artifact(artifact).valid

    runtime_path = next(artifact.glob("episodes/*/runtime.json"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["runtime"]["ticks"][1]["action_index"] = 1
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    _write_manifest(artifact)

    report = validate_artifact(artifact)
    assert not report.valid
    assert any("execute action index mismatch" in error for error in report.errors)
