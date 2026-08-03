from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.vla.guard import GuardConfig
from armbench.vla.online import (
    OnlineExecutionConfig,
    ReferenceActionChunkPolicy,
    run_online_episode,
)
from armbench.vla.online_benchmark import execute_vla_online_benchmark
from armbench.vla.types import VLAObservation


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
    assert result.physical_safe


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
    )

    rows = json.loads((run_directory / "aggregate.json").read_text("utf-8"))
    environment = json.loads(
        (run_directory / "environment.json").read_text("utf-8")
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["online_physics_feedback"] is True
    assert row["camera_recapture_per_query"] is True
    assert row["actual_openpi_inference"] is False
    assert row["policy_source"] == "scripted_non_learned_reference"
    assert row["policy_queries"] > 1
    assert environment["packages"]["mujoco"] == "3.11.0"
    assert environment["packages"]["websockets"] == "16.1.1"
    assert environment["vla_online"]["openpi_contract"]["model_config"] == (
        "pi05_droid"
    )
    assert (run_directory / row["external_image"]).is_file()
    assert (run_directory / row["trace"]).is_file()
    assert (run_directory / "overview.png").stat().st_size > 10_000
    summary = (run_directory / "summary.md").read_text("utf-8")
    assert "No pi0/pi0.5 checkpoint" in summary
    assert "recaptures both 224x224 cameras" in summary
