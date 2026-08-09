from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from armbench.mujoco_sim import MuJoCoCollisionChecker, MuJoCoPanda
from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.vla.async_panda import (
    AsyncPandaConfig,
    ScriptedPolicyFaults,
    SleepingReferenceActionChunkPolicy,
    _verified_braking_actions,
    run_async_panda_episode,
)
from armbench.vla.async_panda_benchmark import (
    AsyncPandaCondition,
    execute_async_panda_benchmark,
    validate_async_panda_artifact,
)
from armbench.vla.pi05_archive_replay import _write_root_manifest
from armbench.vla.types import ActionChunk, VLAObservation


def _short_reference(steps: int = 8) -> np.ndarray:
    start = mujoco_scenarios()["free_space"].start
    goal = start + np.array([0.08, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0])
    return np.linspace(start, goal, steps + 1)


def _config(*, steps: int = 8) -> AsyncPandaConfig:
    return AsyncPandaConfig(
        action_period_s=0.05,
        control_period_s=0.01,
        response_deadline_s=0.25,
        warmup_s=0.01,
        settle_s=0.1,
        max_action_steps=steps,
        action_horizon=6,
        clearance_m=0.0,
    )


class _InjectedActionPolicy:
    policy_source = "injected_test_policy"

    def __init__(self) -> None:
        self.calls = 0

    def infer(self, observation: VLAObservation) -> ActionChunk:
        self.calls += 1
        actions = np.zeros((6, 8), dtype=float)
        actions[:, 7] = float(observation.gripper_position[0])
        return ActionChunk(
            actions=actions,
            source=self.policy_source,
            observation_sequence_id=observation.sequence_id,
            inference_latency_ms=0.0,
        )


def test_async_panda_accepts_injected_policy_factory_and_preserves_metadata() -> None:
    holder: list[_InjectedActionPolicy] = []

    def factory() -> _InjectedActionPolicy:
        policy = _InjectedActionPolicy()
        holder.append(policy)
        return policy

    result = run_async_panda_episode(
        "free_space",
        "unguarded",
        _short_reference(),
        policy_factory=factory,
        config=_config(),
    )

    assert holder and holder[0].calls > 0
    assert result.scripted_policy is False
    assert result.policy_checkpoint_executed is False
    assert result.policy_source == _InjectedActionPolicy.policy_source
    metrics = result.metrics()
    assert metrics["scripted_policy"] is False
    assert metrics["policy_checkpoint_executed"] is False


def test_async_panda_rejects_conflicting_policy_injection_arguments() -> None:
    with pytest.raises(ValueError, match="either policy or policy_factory"):
        run_async_panda_episode(
            "free_space",
            "unguarded",
            _short_reference(steps=1),
            policy=_InjectedActionPolicy(),
            policy_factory=_InjectedActionPolicy,
            config=_config(steps=1),
        )


def test_async_panda_configuration_rejects_invalid_faults() -> None:
    with pytest.raises(ValueError, match="deadline"):
        AsyncPandaConfig(response_deadline_s=0.01)
    with pytest.raises(ValueError, match="configuration"):
        AsyncPandaConfig(observation_prewarm_timeout_s=0.0)
    with pytest.raises(ValueError, match="fault"):
        ScriptedPolicyFaults(latency_schedule_ms=())
    with pytest.raises(ValueError, match="fault"):
        ScriptedPolicyFaults(drop_probability=1.1)


def test_terminal_chunk_retains_feedback_after_stale_prefix() -> None:
    reference = _short_reference(steps=3)
    config = _config(steps=3)
    policy = SleepingReferenceActionChunkPolicy(reference, config)
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    observed = reference[-1].copy()
    observed[0] -= 0.05
    observation = VLAObservation(
        exterior_image=image,
        wrist_image=image,
        joint_position=observed,
        gripper_position=np.array([1.0]),
        prompt="reach the terminal reference",
        sequence_id=len(reference) - 1,
    )

    chunk = policy.infer(observation)

    assert chunk.actions[0, 0] > 0.0
    np.testing.assert_allclose(chunk.actions[:, 0], chunk.actions[0, 0])


def test_wall_clock_panda_loop_handles_stale_response_off_policy_thread() -> None:
    result = run_async_panda_episode(
        "free_space",
        "braking_invariant",
        _short_reference(),
        config=_config(),
        policy_faults=ScriptedPolicyFaults(latency_schedule_ms=(40.0,)),
    )

    metrics = result.metrics()
    assert metrics["separate_policy_thread"] is True
    assert result.policy_worker_thread_id != result.control_thread_id
    assert result.observation_worker_thread_id != result.control_thread_id
    assert result.control_ticks_during_inference > 0
    # Camera rendering is intentionally part of measured observation age. On
    # slower CI runners the first response can therefore miss the 250 ms
    # deadline; both outcomes must remain fail-closed and observable.
    assert result.accepted_responses > 0 or result.deadline_rejections > 0
    assert result.physical_safe
    assert result.abrupt_stop_violations == 0
    assert result.repair_selection_deadline_exceedances == 0

    if result.accepted_responses > 0:
        accepted = [
            event
            for event in result.events
            if event["event"] == "policy_outcome"
            and event["dispatch_status"] == "accepted"
        ]
        prepared = [
            event for event in result.events if event["event"] == "plan_prepared"
        ]
        assert accepted
        assert prepared
        assert all("selection_deadline_exceeded" in event for event in prepared)
        assert all(int(event["action_offset"]) > 0 for event in accepted)
        assert all(int(event["base_action_index"]) > 0 for event in prepared)
    else:
        assert result.deadline_rejections > 0
        assert result.hold_boundaries > 0


def test_policy_drop_fails_closed_without_stopping_control_ticks() -> None:
    # Leave enough measured time for a software EGL renderer to deliver the
    # first observation on shared CI. A shorter episode can correctly remain
    # in hold:no_policy_response without ever exercising the injected drop.
    steps = 20
    result = run_async_panda_episode(
        "free_space",
        "braking_invariant",
        _short_reference(steps=steps),
        config=_config(steps=steps),
        policy_faults=ScriptedPolicyFaults(
            latency_schedule_ms=(20.0,), drop_probability=1.0
        ),
    )

    assert result.observation_frames_completed > 0
    assert result.policy_failures > 0
    assert result.accepted_responses == 0
    assert result.hold_boundaries == result.action_boundaries
    assert result.physical_safe
    assert result.abrupt_stop_violations == 0
    assert len(result.actual_wall_times_s) > result.action_boundaries


def test_emergency_brake_validates_complete_acceleration_limited_stop() -> None:
    scenario = mujoco_scenarios()["free_space"]
    robot = MuJoCoPanda.create()
    checker = MuJoCoCollisionChecker(robot, resolution=0.02)
    config = _config(steps=2)
    previous = np.array([0.6, -0.4, 0.2, 0.0, 0.0, 0.0, 0.0])

    actions, safe, _ = _verified_braking_actions(
        scenario.start,
        1.0,
        previous,
        checker,
        config,
    )

    assert safe
    assert len(actions) > 0
    velocities = np.vstack([previous, actions[:, :7]])
    accelerations = np.abs(np.diff(velocities, axis=0)) / config.action_period_s
    assert np.max(accelerations) <= config.joint_acceleration_limit_rad_s2 + 1e-12
    np.testing.assert_allclose(actions[-1, :7], 0.0)


@pytest.fixture(scope="module")
def async_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("async_panda") / "artifact"
    return execute_async_panda_benchmark(
        Path("configs/vla_guard_benchmark.json"),
        root,
        scenario_name="free_space",
        modes=("braking_invariant",),
        conditions=(AsyncPandaCondition("fixed_000ms", (0.0,)),),
        max_reference_steps=4,
        extra_action_steps=2,
    )


def test_async_benchmark_writes_recomputable_manifest_artifact(
    async_artifact: Path,
) -> None:
    validation = validate_async_panda_artifact(async_artifact)

    assert validation["valid"]
    assert validation["cases"] == 1
    assert set(path.name for path in async_artifact.iterdir()) >= {
        "events.jsonl",
        "manifest.json",
        "per_case.csv",
        "provenance.json",
        "summary.json",
        "summary.md",
        "traces",
    }
    summary = json.loads((async_artifact / "summary.json").read_text("utf-8"))
    provenance = json.loads(
        (async_artifact / "provenance.json").read_text("utf-8")
    )
    assert summary["policy_checkpoint_executed"] is False
    assert summary["scripted_policy"] is True
    assert summary["panda_closed_loop_executed"] is True
    assert summary["hard_realtime_claim"] is False
    assert (
        summary["overall"]["repair_selection_deadline_exceedances"] >= 0
    )
    assert provenance["matrix"]["runtime_clearance_source"] == (
        "planning_clearance"
    )
    assert provenance["matrix"]["runtime_clearance_m"] == (
        provenance["matrix"]["planning_clearance_m"]
    )
    assert provenance["collision_validation"]["method"] == (
        "clearance_backed_swept_static_obstacle_subdivision"
    )


def test_async_benchmark_records_explicit_zero_runtime_clearance(
    tmp_path: Path,
) -> None:
    artifact = execute_async_panda_benchmark(
        Path("configs/vla_guard_benchmark.json"),
        tmp_path / "zero_clearance",
        scenario_name="free_space",
        modes=("braking_invariant",),
        conditions=(AsyncPandaCondition("fixed_000ms", (0.0,)),),
        max_reference_steps=1,
        extra_action_steps=0,
        runtime_clearance_m=0.0,
    )
    provenance = json.loads((artifact / "provenance.json").read_text("utf-8"))

    assert provenance["matrix"]["runtime_clearance_source"] == (
        "explicit_override"
    )
    assert provenance["matrix"]["runtime_clearance_m"] == 0.0
    assert provenance["collision_validation"]["method"] == (
        "resolution_bounded_joint_space_sampling"
    )
    assert validate_async_panda_artifact(artifact)["valid"]


def test_async_validator_accepts_frozen_v2_collision_provenance(
    async_artifact: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "legacy_v2"
    shutil.copytree(async_artifact, copied)
    provenance_path = copied / "provenance.json"
    provenance = json.loads(provenance_path.read_text("utf-8"))
    provenance["schema_version"] = (
        "armbench.async_panda_closed_loop_provenance.v2"
    )
    provenance.pop("collision_validation")
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_root_manifest(copied)

    assert validate_async_panda_artifact(copied)["valid"]


def test_resigned_csv_metric_tamper_is_rejected(
    async_artifact: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "tampered"
    shutil.copytree(async_artifact, copied)
    csv_path = copied / "per_case.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = tuple(rows[0])
    rows[0]["tracking_rmse_rad"] = str(
        float(rows[0]["tracking_rmse_rad"]) + 0.1
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _write_root_manifest(copied)

    with pytest.raises(ValueError, match="trace-derived metric mismatch"):
        validate_async_panda_artifact(copied)
