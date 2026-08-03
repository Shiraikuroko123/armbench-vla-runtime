from __future__ import annotations

import json
from pathlib import Path
import socket
import threading
import time

import mujoco
import numpy as np
import pytest
from openpi_client import msgpack_numpy
from websockets.sync.server import serve

from armbench.mujoco_sim import MuJoCoCollisionChecker, MuJoCoPanda
from armbench.mujoco_sim.benchmark import inflate_obstacles
from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.vla.guard import ActionChunkGuard, GuardConfig
from armbench.vla.benchmark import execute_vla_guard_benchmark
from armbench.vla.observation import MuJoCoDroidObservationBuilder
from armbench.vla.policy import OpenPIPolicyClient, ScriptedActionChunkPolicy
from armbench.vla.runtime import VLARuntimeSupervisor
from armbench.vla.types import ActionChunk, VLAObservation


def _observation(q: np.ndarray, *, captured_at_s: float = 100.0) -> VLAObservation:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    return VLAObservation(
        exterior_image=image,
        wrist_image=image,
        joint_position=q,
        gripper_position=np.array([1.0]),
        prompt="move around the obstacle",
        sequence_id=7,
        captured_at_s=captured_at_s,
    )


class _FakeBackend:
    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    def get_server_metadata(self) -> dict[str, object]:
        return {"model": "pi05_droid_test"}

    def infer(self, observation: dict[str, object]) -> dict[str, object]:
        self.request = observation
        return {
            "actions": np.zeros((15, 8)),
            "server_timing": {"infer_ms": 12.5},
        }


def test_openpi_wrapper_uses_official_droid_keys() -> None:
    backend = _FakeBackend()
    client = OpenPIPolicyClient(backend=backend)
    observation = _observation(np.zeros(7), captured_at_s=0.0)

    chunk = client.infer(observation)

    assert backend.request is not None
    assert set(backend.request) == {
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
        "observation/joint_position",
        "observation/gripper_position",
        "prompt",
    }
    assert chunk.actions.shape == (15, 8)
    assert chunk.source == "openpi_remote"
    assert chunk.server_timing == {"infer_ms": 12.5}
    assert client.server_metadata == {"model": "pi05_droid_test"}


def test_openpi_wrapper_rejects_wrong_action_shape() -> None:
    backend = _FakeBackend()
    backend.infer = lambda observation: {"actions": np.zeros((10, 8))}  # type: ignore[method-assign]
    client = OpenPIPolicyClient(backend=backend)

    with pytest.raises(ValueError, match="shape mismatch"):
        client.infer(_observation(np.zeros(7)))


def test_runtime_supervisor_holds_nonfinite_openpi_response() -> None:
    backend = _FakeBackend()
    backend.infer = lambda observation: {  # type: ignore[method-assign]
        "actions": np.full((15, 8), np.nan)
    }
    scenario = mujoco_scenarios()["single_block"]
    robot = MuJoCoPanda.create(obstacles=())
    supervisor = VLARuntimeSupervisor(
        OpenPIPolicyClient(backend=backend),
        ActionChunkGuard(MuJoCoCollisionChecker(robot)),
    )

    decision = supervisor.infer_and_guard(
        scenario.start, 1.0, _observation(scenario.start)
    )

    assert decision.used_runtime_fallback
    assert decision.failure is not None
    assert decision.failure.stage == "policy_inference"
    assert decision.failure.error_type == "ValueError"
    np.testing.assert_allclose(decision.actions[:, :7], 0.0)


def test_scripted_policy_fails_closed_when_chunks_are_exhausted() -> None:
    observation = _observation(np.zeros(7))
    policy = ScriptedActionChunkPolicy([np.ones((1, 8))])

    first = policy.infer(observation)
    with pytest.raises(RuntimeError, match="chunks exhausted"):
        policy.infer(observation)

    np.testing.assert_allclose(first.actions, 1.0)
    policy.reset()
    np.testing.assert_allclose(policy.infer(observation).actions, 1.0)


def test_scripted_policy_repeats_only_with_explicit_opt_in() -> None:
    observation = _observation(np.zeros(7))
    policy = ScriptedActionChunkPolicy(
        [np.ones((1, 8))], repeat_last=True
    )

    policy.infer(observation)
    repeated = policy.infer(observation)

    np.testing.assert_allclose(repeated.actions, 1.0)


def test_scripted_policy_rejects_bad_contract_at_construction() -> None:
    with pytest.raises(ValueError, match="finite nonempty Nx8"):
        ScriptedActionChunkPolicy([np.full((1, 8), np.nan)])
    with pytest.raises(ValueError, match="latencies"):
        ScriptedActionChunkPolicy(
            [np.zeros((1, 8))], latencies_ms=[-1.0]
        )


def test_guard_config_rejects_nonfinite_limits() -> None:
    with pytest.raises(ValueError, match="timing"):
        GuardConfig(deadline_ms=float("nan"))
    with pytest.raises(ValueError, match="velocity"):
        GuardConfig(joint_velocity_clip_rad_s=float("nan"))


def test_openpi_wire_protocol_round_trip() -> None:
    seen: dict[str, object] = {}

    def handler(websocket: object) -> None:
        websocket.send(msgpack_numpy.packb({"model": "fake_pi05_droid"}))
        request = msgpack_numpy.unpackb(websocket.recv())
        seen.update(request)
        websocket.send(
            msgpack_numpy.packb(
                {
                    "actions": np.zeros((15, 8)),
                    "server_timing": {"infer_ms": 1.0},
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
    try:
        with OpenPIPolicyClient(host="127.0.0.1", port=port) as client:
            chunk = client.infer(_observation(np.zeros(7)))
            metadata = client.server_metadata
    finally:
        server.shutdown()
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert seen["prompt"] == "move around the obstacle"
    assert chunk.actions.shape == (15, 8)
    assert metadata == {"model": "fake_pi05_droid"}


def test_openpi_connection_refusal_is_bounded() -> None:
    reservation = socket.socket()
    reservation.bind(("127.0.0.1", 0))
    port = int(reservation.getsockname()[1])
    reservation.close()

    started = time.monotonic()
    with pytest.raises(OSError):
        OpenPIPolicyClient(
            host="127.0.0.1",
            port=port,
            connect_timeout_s=0.1,
            inference_timeout_s=0.1,
        )

    assert time.monotonic() - started < 1.0


def test_openpi_inference_timeout_closes_transport() -> None:
    request_seen = threading.Event()
    release_server = threading.Event()

    def handler(websocket: object) -> None:
        websocket.send(msgpack_numpy.packb({"model": "slow_test"}))
        websocket.recv()
        request_seen.set()
        release_server.wait(timeout=1.0)

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
    client: OpenPIPolicyClient | None = None
    try:
        client = OpenPIPolicyClient(
            host="127.0.0.1",
            port=port,
            connect_timeout_s=0.2,
            inference_timeout_s=0.05,
        )
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            client.infer(_observation(np.zeros(7)))
        assert request_seen.wait(timeout=0.5)
        assert time.monotonic() - started < 0.75
        with pytest.raises(ConnectionError, match="closed"):
            client.infer(_observation(np.zeros(7)))
    finally:
        release_server.set()
        if client is not None:
            client.close()
        server.shutdown()
        thread.join(timeout=2.0)

    assert not thread.is_alive()


def test_mujoco_builder_captures_nonblank_droid_observation() -> None:
    scenario = mujoco_scenarios()["narrow_gate"]
    reference_robot = MuJoCoPanda.create(obstacles=scenario.obstacles)
    robot = MuJoCoPanda.create(
        obstacles=scenario.obstacles,
        vla_cameras=True,
        goal_marker=reference_robot.hand_position(scenario.goal),
    )
    data = mujoco.MjData(robot.model)
    robot.set_configuration(data, scenario.start)

    with MuJoCoDroidObservationBuilder(robot) as builder:
        observation = builder.capture(
            data,
            prompt="move the gripper to the other side of the red obstacles",
            sequence_id=3,
        )

    assert observation.exterior_image.shape == (224, 224, 3)
    assert observation.wrist_image.shape == (224, 224, 3)
    assert observation.exterior_image.dtype == np.uint8
    assert float(observation.exterior_image.std()) > 10.0
    assert float(observation.wrist_image.std()) > 10.0
    for image in (observation.exterior_image, observation.wrist_image):
        pixels = image.astype(float)
        red_pixels = (
            (pixels[:, :, 0] > 1.4 * pixels[:, :, 1])
            & (pixels[:, :, 0] > 1.2 * pixels[:, :, 2])
            & (pixels[:, :, 0] > 70.0)
        )
        green_pixels = (
            (pixels[:, :, 1] > 1.4 * pixels[:, :, 0])
            & (pixels[:, :, 1] > 1.15 * pixels[:, :, 2])
            & (pixels[:, :, 1] > 70.0)
        )
        assert int(red_pixels.sum()) > 50
        assert int(green_pixels.sum()) > 20
    np.testing.assert_allclose(observation.joint_position, scenario.start)
    assert observation.to_openpi_droid()["prompt"] == observation.prompt


def _direct_action_chunk(robot: MuJoCoPanda, steps: int = 15) -> np.ndarray:
    scenario = mujoco_scenarios()["single_block"]
    direction = scenario.goal - scenario.start
    joint_velocity = direction / np.max(np.abs(direction))
    assert np.max(np.abs(joint_velocity)) <= 1.0
    actions = np.zeros((steps, 8))
    actions[:, :7] = joint_velocity
    actions[:, 7] = 1.0
    return actions


def test_guard_intervenes_before_direct_path_collision() -> None:
    scenario = mujoco_scenarios()["single_block"]
    robot = MuJoCoPanda.create(
        obstacles=inflate_obstacles(scenario.obstacles, 0.02)
    )
    checker = MuJoCoCollisionChecker(robot, resolution=0.02)
    observation = _observation(scenario.start)
    chunk = ActionChunk(
        actions=_direct_action_chunk(robot),
        source="scripted_non_learned",
        observation_sequence_id=observation.sequence_id,
        inference_latency_ms=20.0,
        received_at_s=observation.captured_at_s + 0.02,
    )
    guard = ActionChunkGuard(
        checker,
        GuardConfig(joint_velocity_clip_rad_s=1.0, deadline_ms=200.0),
    )

    result = guard.guard(scenario.start, 1.0, observation, chunk)

    assert result.unsafe_raw_steps > 0
    assert result.intervention_steps > 0
    assert result.hold_steps > 0
    assert result.safe_after_guard
    assert checker.path_is_valid(result.predicted_positions)


def test_guard_holds_entire_stale_chunk() -> None:
    scenario = mujoco_scenarios()["single_block"]
    robot = MuJoCoPanda.create(obstacles=scenario.obstacles)
    checker = MuJoCoCollisionChecker(robot)
    observation = _observation(scenario.start)
    chunk = ActionChunk(
        actions=np.zeros((15, 8)),
        source="openpi_remote",
        observation_sequence_id=observation.sequence_id,
        inference_latency_ms=250.0,
        received_at_s=observation.captured_at_s + 0.25,
    )

    result = ActionChunkGuard(
        checker, GuardConfig(deadline_ms=200.0)
    ).guard(scenario.start, 1.0, observation, chunk)

    assert result.deadline_exceeded
    assert result.fallback_latched
    assert result.hold_steps == 15
    assert result.intervention_steps == 15
    np.testing.assert_allclose(
        result.predicted_positions,
        np.repeat(scenario.start[None, :], 16, axis=0),
    )


def test_deadline_fallback_stays_latched_until_explicit_reset() -> None:
    scenario = mujoco_scenarios()["single_block"]
    robot = MuJoCoPanda.create(obstacles=scenario.obstacles)
    guard = ActionChunkGuard(
        MuJoCoCollisionChecker(robot),
        GuardConfig(deadline_ms=200.0, latch_on_deadline=True),
    )
    stale_observation = _observation(scenario.start, captured_at_s=100.0)
    stale_chunk = ActionChunk(
        actions=np.zeros((15, 8)),
        source="openpi_remote",
        observation_sequence_id=stale_observation.sequence_id,
        inference_latency_ms=250.0,
        received_at_s=100.25,
    )
    guard.guard(scenario.start, 1.0, stale_observation, stale_chunk)

    fresh_observation = _observation(scenario.start, captured_at_s=101.0)
    fresh_chunk = ActionChunk(
        actions=np.zeros((15, 8)),
        source="openpi_remote",
        observation_sequence_id=fresh_observation.sequence_id,
        inference_latency_ms=10.0,
        received_at_s=101.01,
    )
    latched = guard.guard(scenario.start, 1.0, fresh_observation, fresh_chunk)
    assert not latched.deadline_exceeded
    assert latched.fallback_latched
    assert latched.hold_steps == 15
    assert {step.reason for step in latched.steps} == {"deadline_latched"}

    guard.reset()
    recovered = guard.guard(scenario.start, 1.0, fresh_observation, fresh_chunk)
    assert not recovered.fallback_latched
    assert recovered.hold_steps == 0


def test_state_mismatch_holds_and_latches_until_resynchronization() -> None:
    scenario = mujoco_scenarios()["single_block"]
    robot = MuJoCoPanda.create(obstacles=scenario.obstacles)
    guard = ActionChunkGuard(
        MuJoCoCollisionChecker(robot),
        GuardConfig(
            max_state_mismatch_rad=0.05,
            latch_on_state_mismatch=True,
        ),
    )
    observation = _observation(scenario.start)
    chunk = ActionChunk(
        actions=np.zeros((15, 8)),
        source="openpi_remote",
        observation_sequence_id=observation.sequence_id,
        inference_latency_ms=10.0,
        received_at_s=observation.captured_at_s + 0.01,
    )
    actual_q = scenario.start.copy()
    actual_q[0] += 0.06

    mismatched = guard.guard(actual_q, 1.0, observation, chunk)

    assert mismatched.state_mismatch_exceeded
    assert mismatched.state_mismatch_latched
    assert mismatched.state_mismatch_rad == pytest.approx(0.06)
    assert mismatched.fallback_reason == "state_mismatch"
    assert mismatched.hold_steps == 15
    assert {step.reason for step in mismatched.steps} == {"state_mismatch"}

    synchronized_observation = _observation(actual_q, captured_at_s=101.0)
    synchronized_chunk = ActionChunk(
        actions=np.zeros((15, 8)),
        source="openpi_remote",
        observation_sequence_id=synchronized_observation.sequence_id,
        inference_latency_ms=10.0,
        received_at_s=101.01,
    )
    latched = guard.guard(
        actual_q, 1.0, synchronized_observation, synchronized_chunk
    )
    assert not latched.state_mismatch_exceeded
    assert latched.state_mismatch_latched
    assert latched.fallback_reason == "state_mismatch_latched"
    assert latched.hold_steps == 15

    with pytest.raises(ValueError, match="exceeds configured limits"):
        guard.reset(previous_joint_velocity=np.full(7, 2.0))
    still_latched = guard.guard(
        actual_q, 1.0, synchronized_observation, synchronized_chunk
    )
    assert still_latched.fallback_reason == "state_mismatch_latched"

    guard.reset()
    recovered = guard.guard(
        actual_q, 1.0, synchronized_observation, synchronized_chunk
    )
    assert not recovered.fallback_latched
    assert recovered.fallback_reason is None
    assert recovered.hold_steps == 0


def test_guard_limits_velocity_slew_with_cross_chunk_state() -> None:
    scenario = mujoco_scenarios()["single_block"]
    robot = MuJoCoPanda.create(obstacles=())
    guard = ActionChunkGuard(
        MuJoCoCollisionChecker(robot),
        GuardConfig(
            control_dt_s=0.1,
            joint_velocity_clip_rad_s=1.0,
            joint_acceleration_clip_rad_s2=2.0,
        ),
    )
    observation = _observation(scenario.start)
    first_actions = np.zeros((2, 8))
    first_actions[:, 7] = 1.0
    first_actions[0, 0] = 1.0
    first_actions[1, 0] = -1.0
    first_chunk = ActionChunk(
        actions=first_actions,
        source="openpi_remote",
        observation_sequence_id=observation.sequence_id,
        inference_latency_ms=10.0,
        received_at_s=observation.captured_at_s + 0.01,
    )

    first = guard.guard(scenario.start, 1.0, observation, first_chunk)

    np.testing.assert_allclose(first.guarded_actions[:, 0], [0.2, 0.0])
    assert first.slew_limited_steps == 2
    assert first.acceleration_override_steps == 0
    assert first.max_guarded_acceleration_rad_s2 == pytest.approx(2.0)
    assert {step.reason for step in first.steps} == {"slew_rate_repaired"}
    assert first.safe_after_guard

    second_observation = _observation(
        first.predicted_positions[-1], captured_at_s=101.0
    )
    second_actions = np.zeros((1, 8))
    second_actions[0, 0] = 1.0
    second_actions[0, 7] = 1.0
    second_chunk = ActionChunk(
        actions=second_actions,
        source="openpi_remote",
        observation_sequence_id=second_observation.sequence_id,
        inference_latency_ms=10.0,
        received_at_s=101.01,
    )

    second = guard.guard(
        first.predicted_positions[-1],
        1.0,
        second_observation,
        second_chunk,
    )
    assert second.guarded_actions[0, 0] == pytest.approx(0.2)

    guard.reset(previous_joint_velocity=np.full(7, 0.6))
    reset_observation = _observation(
        second.predicted_positions[-1], captured_at_s=102.0
    )
    reset_chunk_actions = -np.ones((1, 8))
    reset_chunk_actions[:, 7] = 1.0
    reset_chunk = ActionChunk(
        actions=reset_chunk_actions,
        source="openpi_remote",
        observation_sequence_id=reset_observation.sequence_id,
        inference_latency_ms=10.0,
        received_at_s=102.01,
    )
    after_reset = guard.guard(
        second.predicted_positions[-1],
        1.0,
        reset_observation,
        reset_chunk,
    )
    np.testing.assert_allclose(after_reset.guarded_actions[0, :7], 0.4)


def test_guard_applies_robot_velocity_limits_to_executed_action() -> None:
    scenario = mujoco_scenarios()["single_block"]
    robot = MuJoCoPanda.create(obstacles=())
    guard = ActionChunkGuard(
        MuJoCoCollisionChecker(robot),
        GuardConfig(
            joint_velocity_clip_rad_s=3.0,
            joint_acceleration_clip_rad_s2=100.0,
        ),
    )
    observation = _observation(scenario.start)
    actions = np.zeros((1, 8))
    actions[0, 0] = 2.5
    actions[0, 7] = 1.0
    chunk = ActionChunk(
        actions=actions,
        source="openpi_remote",
        observation_sequence_id=observation.sequence_id,
        inference_latency_ms=10.0,
        received_at_s=observation.captured_at_s + 0.01,
    )

    result = guard.guard(scenario.start, 1.0, observation, chunk)

    assert result.guarded_actions[0, 0] == pytest.approx(
        robot.velocity_limits[0]
    )
    assert result.steps[0].reason == "action_bounds_repaired"
    assert result.intervention_steps == 1


class _FailingPolicy:
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, observation: VLAObservation) -> ActionChunk:
        self.calls += 1
        raise ConnectionError("policy server disconnected")


def test_runtime_supervisor_converts_policy_failure_to_latched_hold() -> None:
    scenario = mujoco_scenarios()["single_block"]
    robot = MuJoCoPanda.create(obstacles=())
    policy = _FailingPolicy()
    supervisor = VLARuntimeSupervisor(
        policy,
        ActionChunkGuard(MuJoCoCollisionChecker(robot)),
    )
    observation = _observation(scenario.start)

    failed = supervisor.infer_and_guard(scenario.start, 1.0, observation)

    assert failed.used_runtime_fallback
    assert failed.policy_source is None
    assert failed.failure is not None
    assert failed.failure.stage == "policy_inference"
    assert failed.failure.error_type == "ConnectionError"
    assert failed.runtime_failure_latched
    assert failed.actions.shape == (15, 8)
    np.testing.assert_allclose(failed.actions[:, :7], 0.0)
    np.testing.assert_allclose(failed.actions[:, 7], 1.0)
    np.testing.assert_allclose(
        failed.predicted_positions,
        np.repeat(scenario.start[None, :], 16, axis=0),
    )

    latched = supervisor.infer_and_guard(scenario.start, 1.0, observation)
    assert latched.failure is not None
    assert latched.failure.stage == "runtime_latched"
    assert policy.calls == 1

    supervisor.reset()
    supervisor.infer_and_guard(scenario.start, 1.0, observation)
    assert policy.calls == 2


def test_runtime_supervisor_catches_policy_guard_contract_mismatch() -> None:
    scenario = mujoco_scenarios()["single_block"]
    robot = MuJoCoPanda.create(obstacles=())

    class WrongSequencePolicy:
        def infer(self, observation: VLAObservation) -> ActionChunk:
            return ActionChunk(
                actions=np.zeros((15, 8)),
                source="contract_fault",
                observation_sequence_id=observation.sequence_id + 1,
                inference_latency_ms=1.0,
                received_at_s=observation.captured_at_s + 0.001,
            )

    supervisor = VLARuntimeSupervisor(
        WrongSequencePolicy(),
        ActionChunkGuard(MuJoCoCollisionChecker(robot)),
    )

    decision = supervisor.infer_and_guard(
        scenario.start, 1.0, _observation(scenario.start)
    )

    assert decision.used_runtime_fallback
    assert decision.policy_source == "contract_fault"
    assert decision.failure is not None
    assert decision.failure.stage == "guard_validation"
    assert decision.failure.error_type == "ValueError"


def test_runtime_supervisor_preserves_successful_policy_provenance() -> None:
    scenario = mujoco_scenarios()["single_block"]
    robot = MuJoCoPanda.create(obstacles=())
    actions = np.zeros((15, 8))
    actions[:, 7] = 1.0
    supervisor = VLARuntimeSupervisor(
        ScriptedActionChunkPolicy([actions]),
        ActionChunkGuard(MuJoCoCollisionChecker(robot)),
    )

    decision = supervisor.infer_and_guard(
        scenario.start, 1.0, _observation(scenario.start)
    )

    assert not decision.used_runtime_fallback
    assert decision.status == "guarded"
    assert decision.policy_source == "scripted_non_learned"
    assert decision.failure is None
    assert decision.guard_result is not None
    assert decision.metrics()["guard"] is not None


def test_vla_benchmark_writes_honest_reproducible_artifact(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "configs" / "vla_guard_benchmark.json").open(
        encoding="utf-8"
    ) as handle:
        config = json.load(handle)
    config["scenarios"] = ["single_block"]
    config["conditions"] = [
        {
            "name": "fresh_collision_fault",
            "stream": "direct_unsafe",
            "latency_ms": 50.0,
        }
    ]
    config["direct_stream"]["steps"] = 30
    config["execution"]["warmup_s"] = 0.01
    config["execution"]["hold_s"] = 0.01
    config["execution"]["render_cases"] = []
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    run_directory = execute_vla_guard_benchmark(
        config_path,
        tmp_path / "results",
        run_id="test_vla",
        make_videos=False,
    )

    rows = json.loads((run_directory / "aggregate.json").read_text("utf-8"))
    environment = json.loads(
        (run_directory / "environment.json").read_text("utf-8")
    )
    assert {row["mode"] for row in rows} == {"unguarded", "guarded"}
    assert all(row["policy_source"] == "scripted_non_learned" for row in rows)
    assert all(row["remote_policy_response_validated"] is False for row in rows)
    assert all(row["checkpoint_identity_verified"] is False for row in rows)
    assert environment["packages"]["mujoco"] == mujoco.__version__
    assert environment["packages"]["openpi-client"] != "not-installed"
    guarded = next(row for row in rows if row["mode"] == "guarded")
    assert guarded["intervention_steps"] > 0
    assert guarded["max_guarded_acceleration_rad_s2"] <= 15.0 + 1e-9
    assert guarded["acceleration_override_steps"] == 0
    assert guarded["executed_kinematic_valid"] is True
    assert (run_directory / guarded["external_image"]).is_file()
    assert (run_directory / guarded["wrist_image"]).is_file()
    assert (run_directory / "overview.png").stat().st_size > 10_000
    action_rows = (run_directory / "per_action.csv").read_text("utf-8")
    assert "backtracked:" in action_rows
    assert "guard_disabled" in action_rows
    assert "No pi0 or pi0.5 checkpoint was used" in (
        run_directory / "summary.md"
    ).read_text("utf-8")
