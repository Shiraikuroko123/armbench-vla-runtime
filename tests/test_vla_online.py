from __future__ import annotations

import numpy as np

from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.vla.guard import GuardConfig
from armbench.vla.online import (
    OnlineExecutionConfig,
    ReferenceActionChunkPolicy,
    run_online_episode,
)
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
