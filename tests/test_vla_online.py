from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import threading
import time

import imageio.v2 as imageio
import numpy as np
import pytest
from openpi_client import msgpack_numpy
from websockets.sync.server import serve

from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.vla.artifact import (
    ArtifactValidationError,
    validate_online_artifact,
)
from armbench.vla.guard import GuardConfig
from armbench.vla.online import (
    OnlineExecutionConfig,
    OnlineFaultConfig,
    ReferenceActionChunkPolicy,
    run_online_episode,
)
from armbench.vla.online_benchmark import (
    execute_openpi_online_run,
    execute_vla_online_benchmark,
)
from armbench.vla.loopback import (
    LOOPBACK_POLICY_PROVENANCE,
    execute_openpi_loopback_run,
)
from armbench.vla.observation_guard import (
    ObservationGuardConfig,
    ObservationRejectedError,
    VLAObservationGuard,
)
from armbench.vla.types import ActionChunk, VLAObservation


def _observation(q: np.ndarray, *, sequence_id: int) -> VLAObservation:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    return VLAObservation(
        exterior_image=image,
        wrist_image=image,
        joint_position=q,
        gripper_position=np.array([1.0]),
        prompt="follow the collision-free reference",
        sequence_id=sequence_id,
        captured_at_s=100.0,
    )


def _textured_observation(
    q: np.ndarray,
    *,
    sequence_id: int,
    captured_at_s: float,
) -> VLAObservation:
    rows, columns = np.indices((224, 224))
    image = np.stack(
        [rows % 256, columns % 256, (rows + columns) % 256], axis=-1
    ).astype(np.uint8)
    return VLAObservation(
        exterior_image=image,
        wrist_image=np.flip(image, axis=1),
        joint_position=q,
        gripper_position=np.array([1.0]),
        prompt="follow the collision-free reference",
        sequence_id=sequence_id,
        captured_at_s=captured_at_s,
    )


def test_observation_guard_rejects_blank_and_replayed_frames() -> None:
    start = mujoco_scenarios()["single_block"].start
    guard = VLAObservationGuard(
        ObservationGuardConfig(
            min_image_std=5.0, motion_threshold_rad=0.001
        )
    )
    blank = _observation(start, sequence_id=0)
    with pytest.raises(ObservationRejectedError) as blank_error:
        guard.check(blank)
    assert set(blank_error.value.check.reasons) == {
        "exterior_image_low_information",
        "wrist_image_low_information",
    }

    first = _textured_observation(
        start, sequence_id=1, captured_at_s=101.0
    )
    assert guard.check(first).healthy
    stationary = _textured_observation(
        start, sequence_id=2, captured_at_s=102.0
    )
    stationary_check = guard.check(stationary)
    assert stationary_check.exterior_frame_replayed
    assert stationary_check.healthy

    moved = start.copy()
    moved[0] += 0.01
    replayed = _textured_observation(
        moved, sequence_id=3, captured_at_s=103.0
    )
    with pytest.raises(ObservationRejectedError) as replay_error:
        guard.check(replayed)
    assert set(replay_error.value.check.reasons) == {
        "exterior_frame_replayed_during_motion",
        "wrist_frame_replayed_during_motion",
    }

    with pytest.raises(ValueError, match="thresholds"):
        ObservationGuardConfig(min_image_std=-1.0)


def test_reference_policy_replans_chunk_from_observed_state() -> None:
    start = mujoco_scenarios()["single_block"].start
    reference = np.repeat(start[None, :], 5, axis=0)
    reference[:, 0] += np.arange(5) * 0.01
    policy = ReferenceActionChunkPolicy(
        reference,
        action_dt_s=0.1,
        action_horizon=3,
        velocity_limit_rad_s=1.0,
    )
    observed = start.copy()
    observed[0] += 0.005

    chunk = policy.infer(_observation(observed, sequence_id=1))

    assert chunk.source == "scripted_non_learned_reference"
    assert chunk.actions.shape == (3, 8)
    np.testing.assert_allclose(chunk.actions[0, 0], 0.15)

    with pytest.raises(ValueError, match="timing/limits"):
        ReferenceActionChunkPolicy(reference, latency_ms=float("nan"))
    with pytest.raises(ValueError, match="timing/tolerance"):
        OnlineExecutionConfig(action_dt_s=float("nan"))


def test_reference_policy_repeats_deterministic_latency_schedule() -> None:
    start = mujoco_scenarios()["single_block"].start
    reference = np.repeat(start[None, :], 3, axis=0)
    policy = ReferenceActionChunkPolicy(
        reference,
        latency_schedule_ms=[0.0, 40.0, 80.0],
    )

    ages = [
        policy.infer(_observation(start, sequence_id=index)).age_ms(
            _observation(start, sequence_id=index)
        )
        for index in range(4)
    ]

    assert ages == pytest.approx([0.0, 40.0, 80.0, 0.0])
    with pytest.raises(ValueError, match="either latency"):
        ReferenceActionChunkPolicy(
            reference,
            latency_ms=1.0,
            latency_schedule_ms=[2.0],
        )
    with pytest.raises(ValueError, match="latency schedule"):
        ReferenceActionChunkPolicy(reference, latency_schedule_ms=[])


def test_online_episode_reobserves_actual_mujoco_state_each_action() -> None:
    start = mujoco_scenarios()["single_block"].start
    reference = np.repeat(start[None, :], 5, axis=0)
    reference[:, 0] += np.arange(5) * 0.005
    action_dt = 0.1
    policy = ReferenceActionChunkPolicy(
        reference,
        action_dt_s=action_dt,
        velocity_limit_rad_s=0.5,
    )

    result = run_online_episode(
        "single_block",
        policy,
        reference,
        execution_horizon=1,
        clearance_m=0.0,
        guard_config=GuardConfig(
            control_dt_s=action_dt,
            deadline_ms=200.0,
            joint_velocity_clip_rad_s=0.5,
            joint_acceleration_clip_rad_s2=15.0,
        ),
        execution_config=OnlineExecutionConfig(
            action_dt_s=action_dt,
            warmup_s=0.01,
            hold_s=0.01,
            max_extra_actions=0,
        ),
    )

    assert result.policy_queries == 4
    assert result.termination_reason == "action_limit"
    assert result.action_steps == 4
    assert [record.sequence_id for record in result.chunks] == [0, 1, 2, 3]
    for previous, current in zip(result.chunks, result.chunks[1:]):
        np.testing.assert_allclose(
            current.observation_q, previous.actual_q_after, atol=1e-12
        )
    feedback_errors = [
        np.max(np.abs(record.observation_q - reference[record.action_offset]))
        for record in result.chunks[1:]
    ]
    assert max(feedback_errors) > 1e-6
    assert result.first_exterior_image.shape == (224, 224, 3)
    assert result.last_wrist_image.shape == (224, 224, 3)
    assert float(result.first_exterior_image.std()) > 10.0
    assert result.policy_source == "scripted_non_learned_reference"
    assert result.runtime_fallback_chunks == 0
    assert result.chunks[0].raw_actions is not None
    assert result.chunks[0].raw_actions.shape == (15, 8)
    assert result.chunks[0].guarded_actions.shape == (15, 8)
    assert result.chunks[0].server_timing == {}
    assert len(result.chunks[0].exterior_image_sha256) == 64
    assert len(result.chunks[0].wrist_image_sha256) == 64
    assert result.chunks[0].exterior_frame_delta_mean_abs is None
    assert result.chunks[0].wrist_frame_delta_mean_abs is None
    assert result.chunks[0].exterior_thumbnail.shape == (16, 16, 3)
    assert result.chunks[0].wrist_thumbnail.shape == (16, 16, 3)
    assert all(
        record.exterior_frame_delta_mean_abs is not None
        for record in result.chunks[1:]
    )
    assert len({record.exterior_image_sha256 for record in result.chunks}) > 1
    assert len(result.chunks[0].action_reasons) == 15
    assert result.chunks[0].predicted_positions.shape == (16, 7)
    assert result.physical_safe


def test_online_episode_records_nonblank_live_video(tmp_path: Path) -> None:
    start = mujoco_scenarios()["single_block"].start
    reference = np.repeat(start[None, :], 4, axis=0)
    reference[:, 0] += np.arange(4) * 0.01
    video_path = tmp_path / "online.mp4"
    result = run_online_episode(
        "single_block",
        ReferenceActionChunkPolicy(
            reference,
            action_dt_s=0.1,
            velocity_limit_rad_s=0.5,
        ),
        reference,
        execution_horizon=1,
        clearance_m=0.0,
        guard_config=GuardConfig(
            control_dt_s=0.1,
            joint_velocity_clip_rad_s=0.5,
            joint_acceleration_clip_rad_s2=15.0,
        ),
        execution_config=OnlineExecutionConfig(
            action_dt_s=0.1,
            warmup_s=0.01,
            hold_s=0.01,
            max_extra_actions=0,
        ),
        video_path=video_path,
        video_fps=10,
        render_size=(320, 240),
    )

    assert result.video_path == str(video_path.resolve())
    assert video_path.stat().st_size > 3000
    reader = imageio.get_reader(video_path)
    try:
        frame = reader.get_data(0)
    finally:
        reader.close()
    assert frame.shape == (240, 320, 3)
    assert float(frame.std()) > 10.0


def test_online_episode_advances_physics_during_stale_inference() -> None:
    start = mujoco_scenarios()["single_block"].start
    reference = np.repeat(start[None, :], 4, axis=0)
    reference[:, 0] += np.arange(4) * 0.005
    action_dt = 0.1
    policy = ReferenceActionChunkPolicy(
        reference,
        action_dt_s=action_dt,
        velocity_limit_rad_s=0.5,
        latency_ms=250.0,
    )

    result = run_online_episode(
        "single_block",
        policy,
        reference,
        execution_horizon=1,
        clearance_m=0.0,
        guard_config=GuardConfig(
            control_dt_s=action_dt,
            deadline_ms=200.0,
            joint_velocity_clip_rad_s=0.5,
            joint_acceleration_clip_rad_s2=15.0,
        ),
        execution_config=OnlineExecutionConfig(
            action_dt_s=action_dt,
            warmup_s=0.01,
            hold_s=0.0,
            max_extra_actions=0,
        ),
    )

    assert result.policy_queries == 1
    assert result.action_steps == 1
    assert result.simulated_inference_wait_s == pytest.approx(0.25, abs=0.003)
    record = result.chunks[0]
    assert record.policy_latency_ms == pytest.approx(250.0)
    assert record.simulated_inference_wait_s == pytest.approx(0.25, abs=0.003)
    assert record.deadline_exceeded
    assert record.executed_interventions == 1
    assert result.physical_safe
    assert result.obstacle_contact_steps == 0
    assert result.termination_reason == "guard_fallback:deadline"


def test_online_episode_applies_latency_schedule_until_deadline() -> None:
    start = mujoco_scenarios()["single_block"].start
    reference = np.repeat(start[None, :], 5, axis=0)
    reference[:, 0] += np.arange(5) * 0.01
    action_dt = 0.1
    policy = ReferenceActionChunkPolicy(
        reference,
        action_dt_s=action_dt,
        velocity_limit_rad_s=0.5,
        latency_schedule_ms=[0.0, 250.0],
    )

    result = run_online_episode(
        "single_block",
        policy,
        reference,
        execution_horizon=1,
        clearance_m=0.0,
        guard_config=GuardConfig(
            control_dt_s=action_dt,
            deadline_ms=200.0,
            joint_velocity_clip_rad_s=0.5,
            joint_acceleration_clip_rad_s2=15.0,
        ),
        execution_config=OnlineExecutionConfig(
            action_dt_s=action_dt,
            warmup_s=0.01,
            hold_s=0.01,
            max_extra_actions=0,
        ),
    )

    assert result.policy_queries == 2
    assert [record.policy_latency_ms for record in result.chunks] == (
        pytest.approx([0.0, 250.0])
    )
    assert result.deadline_chunks == 1
    assert result.guard_fallback_chunks == 1
    assert result.simulated_inference_wait_s == pytest.approx(0.25, abs=0.003)
    assert result.termination_reason == "guard_fallback:deadline"
    assert result.physical_safe


def test_online_episode_latches_on_injected_dispatch_state_jump() -> None:
    start = mujoco_scenarios()["single_block"].start
    reference = np.repeat(start[None, :], 4, axis=0)
    reference[:, 0] += np.arange(4) * 0.005
    action_dt = 0.1
    policy = ReferenceActionChunkPolicy(
        reference,
        action_dt_s=action_dt,
        velocity_limit_rad_s=0.5,
    )

    result = run_online_episode(
        "single_block",
        policy,
        reference,
        execution_horizon=1,
        clearance_m=0.0,
        guard_config=GuardConfig(
            control_dt_s=action_dt,
            deadline_ms=200.0,
            max_state_mismatch_rad=0.05,
            joint_velocity_clip_rad_s=0.5,
            joint_acceleration_clip_rad_s2=15.0,
        ),
        execution_config=OnlineExecutionConfig(
            action_dt_s=action_dt,
            warmup_s=0.01,
            hold_s=0.01,
            max_extra_actions=0,
        ),
        fault_config=OnlineFaultConfig(
            state_jump_query=0,
            state_jump_rad=(0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
    )

    assert result.policy_queries == 1
    assert result.fault_injections == 1
    assert result.state_mismatch_chunks == 1
    assert result.guard_fallback_chunks == 1
    assert result.deadline_chunks == 0
    assert result.runtime_fallback_chunks == 0
    assert not result.task_success
    assert result.physical_safe
    assert result.termination_reason == "guard_fallback:state_mismatch"
    record = result.chunks[0]
    assert record.fault_injected
    assert record.guard_fallback
    assert record.fallback_reason == "state_mismatch"
    assert set(record.action_reasons) == {"state_mismatch"}
    assert all(record.action_interventions)
    assert record.state_mismatch_rad == pytest.approx(0.08, abs=1e-9)
    np.testing.assert_allclose(
        record.dispatch_q - record.observation_q,
        np.array([0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        atol=1e-9,
    )


def test_online_fault_config_rejects_partial_or_invalid_state_jump() -> None:
    with pytest.raises(ValueError, match="both a query index"):
        OnlineFaultConfig(state_jump_query=0)
    with pytest.raises(ValueError, match="both a query index"):
        OnlineFaultConfig(state_jump_rad=(0.1,) + (0.0,) * 6)
    with pytest.raises(ValueError, match="seven finite"):
        OnlineFaultConfig(state_jump_query=0, state_jump_rad=(0.1,))
    with pytest.raises(ValueError, match="both a query index and target"):
        OnlineFaultConfig(camera_freeze_query=1)
    with pytest.raises(ValueError, match="greater than zero"):
        OnlineFaultConfig(
            camera_freeze_query=0, camera_freeze_target="both"
        )


def test_online_episode_rejects_frozen_cameras_before_policy_query() -> None:
    start = mujoco_scenarios()["single_block"].start
    reference = np.repeat(start[None, :], 11, axis=0)
    reference[:, 0] += np.arange(11) * 0.01
    action_dt = 0.1
    result = run_online_episode(
        "single_block",
        ReferenceActionChunkPolicy(
            reference,
            action_dt_s=action_dt,
            velocity_limit_rad_s=0.5,
        ),
        reference,
        execution_horizon=5,
        clearance_m=0.0,
        guard_config=GuardConfig(
            control_dt_s=action_dt,
            joint_velocity_clip_rad_s=0.5,
            joint_acceleration_clip_rad_s2=15.0,
        ),
        observation_guard_config=ObservationGuardConfig(
            min_image_std=5.0, motion_threshold_rad=0.0001
        ),
        execution_config=OnlineExecutionConfig(
            action_dt_s=action_dt,
            warmup_s=0.01,
            hold_s=0.01,
            max_extra_actions=0,
        ),
        fault_config=OnlineFaultConfig(
            camera_freeze_query=1,
            camera_freeze_target="both",
        ),
    )

    assert result.observation_cycles == 2
    assert result.policy_queries == 1
    assert result.observation_rejection_chunks == 1
    assert result.camera_freeze_injections == 1
    assert result.fault_injections == 1
    assert result.runtime_fallback_chunks == 1
    assert result.termination_reason == "runtime_fallback:observation_validation"
    assert result.physical_safe
    first, rejected = result.chunks
    assert first.policy_inference_attempted
    assert not rejected.policy_inference_attempted
    assert rejected.camera_freeze_injected
    assert rejected.camera_freeze_target == "both"
    assert rejected.observation_healthy is False
    assert set(rejected.observation_failure_reasons) == {
        "exterior_frame_replayed_during_motion",
        "wrist_frame_replayed_during_motion",
    }
    assert rejected.exterior_frame_replayed
    assert rejected.wrist_frame_replayed
    assert rejected.exterior_image_sha256 == first.exterior_image_sha256
    assert rejected.wrist_image_sha256 == first.wrist_image_sha256
    assert rejected.exterior_frame_delta_mean_abs == 0.0
    assert rejected.wrist_frame_delta_mean_abs == 0.0
    assert rejected.raw_actions is None


def test_online_episode_stops_at_policy_query_budget() -> None:
    start = mujoco_scenarios()["single_block"].start
    reference = np.repeat(start[None, :], 8, axis=0)
    reference[:, 0] += np.arange(8) * 0.02
    policy = ReferenceActionChunkPolicy(
        reference,
        action_dt_s=0.1,
        velocity_limit_rad_s=0.5,
    )

    result = run_online_episode(
        "single_block",
        policy,
        reference,
        execution_horizon=1,
        clearance_m=0.0,
        guard_config=GuardConfig(
            control_dt_s=0.1,
            joint_velocity_clip_rad_s=0.5,
            joint_acceleration_clip_rad_s2=15.0,
        ),
        execution_config=OnlineExecutionConfig(
            action_dt_s=0.1,
            warmup_s=0.01,
            hold_s=0.01,
            max_extra_actions=0,
            max_policy_queries=2,
        ),
    )

    assert result.policy_queries == 2
    assert result.action_steps == 2
    assert result.termination_reason == "query_budget"
    assert result.physical_safe
    assert not result.task_success

    with pytest.raises(ValueError, match="timing/tolerance"):
        OnlineExecutionConfig(max_policy_queries=0)


def test_online_episode_advances_physics_when_policy_times_out() -> None:
    class SlowFailingPolicy:
        def infer(self, observation: VLAObservation) -> ActionChunk:
            time.sleep(0.03)
            raise TimeoutError("synthetic timeout")

    start = mujoco_scenarios()["single_block"].start
    reference = np.repeat(start[None, :], 3, axis=0)
    reference[-1, 0] += 0.2
    result = run_online_episode(
        "single_block",
        SlowFailingPolicy(),
        reference,
        execution_horizon=1,
        clearance_m=0.0,
        guard_config=GuardConfig(control_dt_s=0.1),
        execution_config=OnlineExecutionConfig(
            action_dt_s=0.1,
            warmup_s=0.01,
            hold_s=0.01,
            max_extra_actions=0,
        ),
    )

    assert result.policy_queries == 1
    assert result.termination_reason == "runtime_fallback:policy_inference"
    record = result.chunks[0]
    assert not record.validated_policy_response
    assert record.failure_stage == "policy_inference"
    assert record.client_inference_latency_ms >= 25.0
    assert record.policy_latency_ms >= record.client_inference_latency_ms
    assert record.simulated_inference_wait_s >= 0.025
    assert result.physical_safe


def test_online_benchmark_writes_auditable_artifact(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (project_root / "configs" / "vla_guard_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    config["scenarios"] = ["single_block"]
    config["online"]["execution_horizons"] = [15]
    config["online"]["payload_masses_kg"] = [0.0]
    config["online"]["warmup_s"] = 0.01
    config["online"]["hold_s"] = 0.01
    config["online"]["max_extra_actions"] = 0
    config_path = tmp_path / "online_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    run_directory = execute_vla_online_benchmark(
        config_path,
        tmp_path / "results",
        run_id="online_test",
        make_videos=True,
    )

    rows = json.loads((run_directory / "aggregate.json").read_text("utf-8"))
    environment = json.loads(
        (run_directory / "environment.json").read_text("utf-8")
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["online_physics_feedback"] is True
    assert row["artifact_schema_version"] == 5
    assert row["camera_recapture_per_query"] is True
    assert row["remote_policy_response_validated"] is False
    assert row["checkpoint_identity_verified"] is False
    assert row["policy_source"] == "scripted_non_learned_reference"
    assert row["policy_queries"] > 1
    assert row["camera_audit_queries"] == row["policy_queries"]
    assert row["observation_cycles"] == row["policy_queries"]
    assert row["observation_rejection_chunks"] == 0
    assert row["unique_exterior_observation_hashes"] > 1
    assert row["unique_wrist_observation_hashes"] > 1
    assert row["fault_injections"] == 0
    assert row["state_mismatch_chunks"] == 0
    assert row["policy_latency_schedule_ms"] == [0.0]
    assert row["termination_reason"] in {"goal_reached", "action_limit"}
    assert environment["packages"]["mujoco"] == "3.11.0"
    assert environment["packages"]["websockets"] == "16.1.1"
    assert environment["vla_online"]["openpi_contract"]["model_config"] == (
        "pi05_droid"
    )
    assert environment["vla_online"]["artifact_schema_version"] == 5
    assert environment["vla_online"]["camera_observation_audit"] == {
        "frame_delta": "mean_abs_uint8",
        "full_frame_hash": "sha256",
        "thumbnail_shape": [16, 16, 3],
    }
    assert (run_directory / row["external_image"]).is_file()
    assert (run_directory / row["trace"]).is_file()
    assert row["video_path"] is not None
    assert (run_directory / row["video_path"]).stat().st_size > 3000
    with np.load(run_directory / row["trace"]) as trace:
        query_count = row["policy_queries"]
        assert trace["raw_action_chunks"].shape[1:] == (15, 8)
        assert trace["guarded_action_chunks"].shape[1:] == (15, 8)
        assert trace["predicted_position_chunks"].shape[1:] == (16, 7)
        assert trace["policy_inference_attempted"].shape == (query_count,)
        assert trace["policy_inference_attempted"].all()
        assert trace["observation_healthy"].all()
        assert not trace["camera_freeze_injected"].any()
        assert not trace["exterior_frame_replayed"].any()
        assert not trace["wrist_frame_replayed"].any()
        assert trace["exterior_image_sha256"].shape == (query_count,)
        assert trace["wrist_image_sha256"].shape == (query_count,)
        assert trace["exterior_image_thumbnails"].shape == (
            query_count,
            16,
            16,
            3,
        )
        assert trace["wrist_image_thumbnails"].shape == (
            query_count,
            16,
            16,
            3,
        )
        assert np.isnan(trace["exterior_frame_delta_mean_abs"][0])
        first_exterior = imageio.imread(run_directory / row["external_image"])
        assert str(trace["exterior_image_sha256"][0]) == hashlib.sha256(
            first_exterior.tobytes(order="C")
        ).hexdigest()
    with (run_directory / "per_chunk.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        chunk_rows = list(csv.DictReader(handle))
    assert len(chunk_rows) == row["policy_queries"]
    assert all(len(item["exterior_image_sha256"]) == 64 for item in chunk_rows)
    assert chunk_rows[0]["exterior_frame_delta_mean_abs"] == ""
    assert all(item["exterior_frame_delta_mean_abs"] for item in chunk_rows[1:])
    with (run_directory / "per_action.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        action_rows = list(csv.DictReader(handle))
    assert len(action_rows) == row["policy_queries"] * 15
    assert sum(item["executed"] == "True" for item in action_rows) == row[
        "action_steps"
    ]
    assert {item["reason"] for item in action_rows}
    assert (run_directory / "overview.png").stat().st_size > 10_000
    summary = (run_directory / "summary.md").read_text("utf-8")
    assert "No pi0/pi0.5 checkpoint" in summary
    assert "recaptures both 224x224 cameras" in summary
    assert "State mismatches" in summary
    assert "Termination" in summary
    assert "Repeating synthetic latency profile" in summary
    validation = validate_online_artifact(run_directory, decode_videos=True)
    assert validation.episodes == 1
    assert validation.observation_cycles == row["observation_cycles"]
    assert validation.policy_queries == row["policy_queries"]
    assert validation.videos_decoded == 1
    assert len(validation.aggregate_sha256) == 64

    chunk_rows[0]["exterior_image_sha256"] = "0" * 64
    with (run_directory / "per_chunk.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(chunk_rows[0]))
        writer.writeheader()
        writer.writerows(chunk_rows)
    with pytest.raises(ArtifactValidationError, match="exterior hash trace"):
        validate_online_artifact(run_directory)


def test_remote_openpi_online_run_uses_network_policy_in_feedback_loop(
    tmp_path: Path,
) -> None:
    requests: list[dict[str, object]] = []

    def handler(websocket: object) -> None:
        websocket.send(
            msgpack_numpy.packb(
                {"model_config": "pi05_droid", "checkpoint": "test_only"}
            )
        )
        for _ in range(2):
            request = msgpack_numpy.unpackb(websocket.recv())
            requests.append(dict(request))
            actions = np.zeros((15, 8), dtype=float)
            actions[:, 7] = 1.0
            websocket.send(
                msgpack_numpy.packb(
                    {
                        "actions": actions,
                        "server_timing": {"infer_ms": 2.5},
                    }
                )
            )

    server = serve(
        handler,
        "127.0.0.1",
        0,
        compression=None,
        max_size=None,
    )
    port = int(server.socket.getsockname()[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    project_root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (project_root / "configs" / "vla_guard_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    config["guard"]["deadline_ms"] = 5000.0
    config["online"]["warmup_s"] = 0.01
    config["online"]["hold_s"] = 0.01
    config["online"]["max_extra_actions"] = 0
    config_path = tmp_path / "remote_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_directory = tmp_path / "remote_online"
    try:
        execute_openpi_online_run(
            config_path,
            output_directory,
            host="127.0.0.1",
            port=port,
            scenario_name="single_block",
            execution_horizon=1,
            max_policy_queries=2,
            connect_timeout_s=0.5,
            inference_timeout_s=0.5,
        )
    finally:
        server.shutdown()
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert len(requests) == 2
    assert set(requests[0]) == {
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
        "observation/joint_position",
        "observation/gripper_position",
        "prompt",
    }
    assert requests[0]["observation/exterior_image_1_left"].shape == (
        224,
        224,
        3,
    )
    rows = json.loads((output_directory / "aggregate.json").read_text("utf-8"))
    row = rows[0]
    assert row["remote_inference_attempted"] is True
    assert row["artifact_schema_version"] == 5
    assert row["remote_policy_response_validated"] is True
    assert row["checkpoint_identity_verified"] is False
    assert row["validated_remote_chunks"] == 2
    assert row["policy_source"] == "openpi_remote"
    assert row["policy_provenance"] == "remote_server_unverified"
    assert row["policy_queries"] == 2
    assert row["termination_reason"] == "query_budget"
    assert row["physical_safe"] is True
    environment = json.loads(
        (output_directory / "environment.json").read_text("utf-8")
    )
    assert environment["vla_online"]["remote_openpi_transport"] is True
    assert environment["vla_online"]["server_metadata"]["model_config"] == (
        "pi05_droid"
    )
    with np.load(output_directory / row["trace"]) as trace:
        assert trace["raw_action_chunks"].shape == (2, 15, 8)
        assert trace["guarded_action_chunks"].shape == (2, 15, 8)
    with (output_directory / "per_action.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        action_rows = list(csv.DictReader(handle))
    assert len(action_rows) == 30
    assert sum(item["executed"] == "True" for item in action_rows) == 2
    assert all(item["raw_action"] for item in action_rows)
    summary = (output_directory / "summary.md").read_text("utf-8")
    assert "Remote policy response validated: `true`" in summary
    assert "Checkpoint identity verified by protocol: `false`" in summary
    assert "integration result" in summary


def test_remote_openpi_online_run_does_not_count_invalid_reply(
    tmp_path: Path,
) -> None:
    def handler(websocket: object) -> None:
        websocket.send(msgpack_numpy.packb({"model_config": "bad_test"}))
        websocket.recv()
        websocket.send(
            msgpack_numpy.packb({"actions": np.zeros((1, 8), dtype=float)})
        )

    server = serve(
        handler,
        "127.0.0.1",
        0,
        compression=None,
        max_size=None,
    )
    port = int(server.socket.getsockname()[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    project_root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (project_root / "configs" / "vla_guard_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    config["guard"]["deadline_ms"] = 5000.0
    config["online"]["warmup_s"] = 0.01
    config["online"]["hold_s"] = 0.01
    config_path = tmp_path / "invalid_remote_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_directory = tmp_path / "invalid_remote_online"
    try:
        execute_openpi_online_run(
            config_path,
            output_directory,
            host="127.0.0.1",
            port=port,
            scenario_name="single_block",
            execution_horizon=1,
            max_policy_queries=1,
            connect_timeout_s=0.5,
            inference_timeout_s=0.5,
        )
    finally:
        server.shutdown()
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    row = json.loads(
        (output_directory / "aggregate.json").read_text("utf-8")
    )[0]
    assert row["remote_inference_attempted"] is True
    assert row["remote_policy_response_validated"] is False
    assert row["checkpoint_identity_verified"] is False
    assert row["validated_remote_chunks"] == 0
    assert row["runtime_fallback_chunks"] == 1
    assert row["termination_reason"] == "runtime_fallback:policy_inference"
    with (output_directory / "per_action.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        action_rows = list(csv.DictReader(handle))
    assert {item["reason"] for item in action_rows} == {
        "runtime_fallback:policy_inference"
    }
    assert all(not item["raw_action"] for item in action_rows)
    summary = (output_directory / "summary.md").read_text("utf-8")
    assert "Remote policy response validated: `false`" in summary


def test_loopback_cli_backend_exercises_complete_remote_policy_path(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (project_root / "configs" / "vla_guard_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    config["guard"]["deadline_ms"] = 5000.0
    config["online"]["warmup_s"] = 0.01
    config["online"]["hold_s"] = 0.01
    config["online"]["max_extra_actions"] = 0
    config_path = tmp_path / "loopback_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_directory = tmp_path / "loopback_online"

    execute_openpi_loopback_run(
        config_path,
        output_directory,
        scenario_name="single_block",
        execution_horizon=1,
        max_policy_queries=2,
    )

    row = json.loads(
        (output_directory / "aggregate.json").read_text("utf-8")
    )[0]
    assert row["remote_policy_response_validated"] is True
    assert row["checkpoint_identity_verified"] is False
    assert row["policy_provenance"] == LOOPBACK_POLICY_PROVENANCE
    assert row["policy_source"] == "openpi_remote"
    assert row["policy_queries"] == 2
    assert row["observation_cycles"] == 2
    assert row["termination_reason"] == "query_budget"
    assert row["physical_safe"] is True
    environment = json.loads(
        (output_directory / "environment.json").read_text("utf-8")
    )
    server_metadata = environment["vla_online"]["server_metadata"]
    assert server_metadata["armbench_loopback"] is True
    assert server_metadata["checkpoint_identity_verified"] is False
    assert server_metadata["policy_source"] == LOOPBACK_POLICY_PROVENANCE
    audit = json.loads(
        (output_directory / "loopback_server.json").read_text("utf-8")
    )
    assert audit["request_count"] == 2
    assert audit["checkpoint_identity_verified"] is False
    assert len(audit["requests"]) == 2
    with (output_directory / "per_chunk.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        chunks = list(csv.DictReader(handle))
    assert [item["policy_inference_attempted"] for item in chunks] == [
        "True",
        "True",
    ]
    assert audit["requests"][0]["exterior_image_sha256"] == chunks[0][
        "exterior_image_sha256"
    ]
    summary = (output_directory / "summary.md").read_text("utf-8")
    assert f"Policy provenance: `{LOOPBACK_POLICY_PROVENANCE}`" in summary
    assert "No learned checkpoint produced these actions" in summary


@pytest.mark.parametrize(
    ("fault_mode", "server_outcome"),
    [
        ("malformed_shape", "malformed_action_shape"),
        ("nonfinite", "nonfinite_action_chunk"),
        ("disconnect", "connection_closed_before_response"),
        ("timeout", "response_delayed_past_client_deadline"),
    ],
)
def test_loopback_wire_faults_fail_closed_with_auditable_artifact(
    tmp_path: Path,
    fault_mode: str,
    server_outcome: str,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (project_root / "configs" / "vla_guard_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    config["guard"]["deadline_ms"] = 5000.0
    config["online"]["warmup_s"] = 0.01
    config["online"]["hold_s"] = 0.01
    config["online"]["max_extra_actions"] = 0
    config_path = tmp_path / f"loopback_{fault_mode}_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_directory = tmp_path / f"loopback_{fault_mode}"

    execute_openpi_loopback_run(
        config_path,
        output_directory,
        scenario_name="single_block",
        execution_horizon=1,
        max_policy_queries=1,
        fault_mode=fault_mode,
        fault_request_index=0,
        fault_delay_ms=120.0,
        inference_timeout_s=0.05,
    )

    row = json.loads(
        (output_directory / "aggregate.json").read_text("utf-8")
    )[0]
    assert row["observation_cycles"] == 1
    assert row["policy_queries"] == 1
    assert row["validated_remote_chunks"] == 0
    assert row["remote_policy_response_validated"] is False
    assert row["runtime_fallback_chunks"] == 1
    assert row["termination_reason"] == "runtime_fallback:policy_inference"
    assert row["task_success"] is False
    assert row["physical_safe"] is True
    assert row["obstacle_contact_steps"] == 0
    assert row["self_contact_steps"] == 0
    assert row["joint_limit_violation_steps"] == 0

    with (output_directory / "per_chunk.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        chunks = list(csv.DictReader(handle))
    assert len(chunks) == 1
    assert chunks[0]["policy_inference_attempted"] == "True"
    assert chunks[0]["validated_policy_response"] == "False"
    assert chunks[0]["raw_action_available"] == "False"
    assert chunks[0]["failure_stage"] == "policy_inference"

    audit = json.loads(
        (output_directory / "loopback_server.json").read_text("utf-8")
    )
    assert audit["fault_mode"] == fault_mode
    assert audit["fault_request_index"] == 0
    assert audit["fault_injected_count"] == 1
    assert audit["request_count"] == 1
    assert audit["requests"][0]["injected_fault"] == fault_mode
    assert audit["requests"][0]["server_outcome"] == server_outcome
    summary = (output_directory / "summary.md").read_text("utf-8")
    assert f"Loopback fault injection: `{fault_mode}`" in summary

    validation = validate_online_artifact(output_directory)
    assert validation.observation_cycles == 1
    assert validation.policy_queries == 1
    assert validation.action_rows == 15
