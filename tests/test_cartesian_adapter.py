from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from armbench.mujoco_sim import MuJoCoPanda
from armbench.mujoco_sim.model import default_panda_scene_path
from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.vla.cartesian_adapter import (
    CartesianAdapterConfig,
    PandaCartesianActionAdapter,
    run_cartesian_adapter_smoke,
)


@pytest.fixture(scope="module")
def robot() -> MuJoCoPanda:
    try:
        scene_path: Path = default_panda_scene_path()
    except FileNotFoundError as error:
        pytest.skip(str(error))
    return MuJoCoPanda.create(scene_path=scene_path, obstacles=())


def test_hand_jacobian_translation_matches_finite_difference(
    robot: MuJoCoPanda,
) -> None:
    q = mujoco_scenarios()["free_space"].start
    jacobian = robot.hand_jacobian(q)
    epsilon = 1e-6
    numerical = np.zeros((3, 7), dtype=float)
    for joint in range(7):
        offset = np.zeros(7, dtype=float)
        offset[joint] = epsilon
        numerical[:, joint] = (
            robot.hand_position(q + offset) - robot.hand_position(q - offset)
        ) / (2.0 * epsilon)

    assert jacobian.shape == (6, 7)
    np.testing.assert_allclose(jacobian[:3], numerical, atol=1e-7, rtol=1e-6)


def test_hand_pose_rotation_is_orthonormal(robot: MuJoCoPanda) -> None:
    q = mujoco_scenarios()["free_space"].start
    position, rotation = robot.hand_pose(q)

    assert position.shape == (3,)
    assert rotation.shape == (3, 3)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)


def test_zero_cartesian_actions_hold_joints_and_map_gripper(
    robot: MuJoCoPanda,
) -> None:
    q = mujoco_scenarios()["free_space"].start
    actions = np.zeros((3, 7), dtype=float)
    actions[:, 6] = [-1.0, 0.0, 1.0]

    result = PandaCartesianActionAdapter(robot).adapt(
        actions,
        q,
        source="test",
        observation_sequence_id=2,
        inference_latency_ms=15.0,
        received_at_s=10.0,
    )

    np.testing.assert_allclose(result.chunk.actions[:, :7], 0.0, atol=1e-12)
    np.testing.assert_allclose(result.chunk.actions[:, 7], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(
        result.predicted_positions,
        np.repeat(q[None, :], 4, axis=0),
        atol=1e-12,
    )


def test_adapter_reduces_cartesian_twist_residual(robot: MuJoCoPanda) -> None:
    q = mujoco_scenarios()["free_space"].start
    actions = np.zeros((1, 7), dtype=float)
    actions[0, 0] = 0.1
    actions[0, 6] = 1.0

    step = PandaCartesianActionAdapter(robot).adapt(
        actions,
        q,
        source="test",
        observation_sequence_id=0,
        inference_latency_ms=10.0,
        received_at_s=20.0,
    ).steps[0]

    assert step.residual_norm < np.linalg.norm(step.desired_twist)
    assert step.achieved_twist[0] > 0.0
    assert robot.within_limits(step.q_after)


def test_adapter_clips_input_and_respects_joint_velocity_limits(
    robot: MuJoCoPanda,
) -> None:
    q = mujoco_scenarios()["free_space"].start
    config = CartesianAdapterConfig(joint_velocity_limit_scale=0.1)
    actions = np.full((2, 7), 10.0, dtype=float)

    result = PandaCartesianActionAdapter(robot, config).adapt(
        actions,
        q,
        source="test",
        observation_sequence_id=0,
        inference_latency_ms=10.0,
        received_at_s=30.0,
    )

    assert result.clipped_input_steps == 2
    assert result.velocity_limited_steps >= 1
    limits = robot.velocity_limits * config.joint_velocity_limit_scale
    assert np.all(np.abs(result.chunk.actions[:, :7]) <= limits + 1e-12)
    assert all(robot.within_limits(position) for position in result.predicted_positions)


def test_tool_frame_rotates_the_requested_twist(robot: MuJoCoPanda) -> None:
    q = mujoco_scenarios()["free_space"].start
    actions = np.zeros((1, 7), dtype=float)
    actions[0, 0] = 0.1
    _, rotation = robot.hand_pose(q)
    config = CartesianAdapterConfig(action_frame="tool")

    step = PandaCartesianActionAdapter(robot, config).adapt(
        actions,
        q,
        source="test",
        observation_sequence_id=0,
        inference_latency_ms=10.0,
        received_at_s=40.0,
    ).steps[0]

    expected_local_delta = np.array(
        [
            0.1 * config.translation_delta_scale_m / config.control_dt_s,
            0.0,
            0.0,
        ]
    )
    np.testing.assert_allclose(
        step.desired_twist[:3], rotation @ expected_local_delta, atol=1e-12
    )


@pytest.mark.parametrize(
    "actions, message",
    [
        (np.zeros((2, 8)), "shape"),
        (np.zeros((0, 7)), "nonempty"),
        (np.full((1, 7), np.nan), "finite"),
    ],
)
def test_adapter_rejects_invalid_action_chunks(
    robot: MuJoCoPanda,
    actions: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PandaCartesianActionAdapter(robot).adapt(
            actions,
            mujoco_scenarios()["free_space"].start,
            source="test",
            observation_sequence_id=0,
            inference_latency_ms=10.0,
        )


def test_cartesian_adapter_smoke_passes_with_explicit_scope() -> None:
    report = run_cartesian_adapter_smoke()

    assert report["passed"]
    assert report["scope"] == "scripted_cartesian_adapter_component_only"
    assert report["policy_checkpoint_used"] is False
    assert report["adapter"]["action_space_id"] == "libero.ee_delta_pose_gripper.v1"
    assert report["guard"]["safe_after_guard"]
    assert report["hand_displacement_m"][0] > 0.0
