from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from integrations.openpi.libero_independent_clock import (
    AGE_ALIGNED_SUFFIX,
    IndependentLiberoRequestBuilder,
    canonical_policy_input_sha256,
    canonical_action_chunk_sha256,
    libero_dynamic_hold,
    run_libero_independent_clock_episode,
)
from integrations.openpi.libero_independent_clock_eval import ExperimentCell
from integrations.openpi.libero_runtime import LIBERO_DUMMY_ACTION
from integrations.openpi.serve_policy_attested import (
    POLICY_SAMPLING_GENERATOR,
    POLICY_SAMPLING_REQUEST_FIELD,
    build_policy_sampling_control,
    policy_sampling_contract,
    policy_sampling_noise,
    policy_sampling_noise_sha256,
)
from integrations.openpi.validate_libero_independent_clock import (
    _Collector,
    _validate_runtime,
)


def _observation(position_x: float = 0.0) -> dict[str, np.ndarray]:
    rows, columns = np.indices((224, 224))
    image = np.stack(
        [rows % 256, columns % 256, (rows + columns) % 256], axis=-1
    ).astype(np.uint8)
    return {
        "agentview_image": image,
        "robot0_eye_in_hand_image": np.flip(image, axis=1),
        "robot0_eef_pos": np.asarray([position_x, 0.0, 0.0]),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.asarray([0.02, -0.02]),
    }


class _FakeLiberoEnvironment:
    def __init__(self, success_x: float = 0.15) -> None:
        self.success_x = success_x
        self.position_x = 0.0
        self.actions: list[np.ndarray] = []

    def reset(self) -> dict[str, np.ndarray]:
        self.position_x = 0.0
        self.actions = []
        return _observation()

    def set_init_state(self, initial_state: Any) -> dict[str, np.ndarray]:
        self.position_x = float(np.asarray(initial_state)[0])
        return _observation(self.position_x)

    def step(self, action: list[float]):
        action_array = np.asarray(action, dtype=np.float64)
        self.actions.append(action_array.copy())
        self.position_x += float(action_array[0])
        done = self.position_x >= self.success_x
        return _observation(self.position_x), float(done), done, {}


@dataclass(frozen=True)
class _DelayedLiberoFactory:
    latency_s: float

    def __call__(self) -> "_DelayedLiberoProvider":
        return _DelayedLiberoProvider(self.latency_s)


class _DelayedLiberoProvider:
    def __init__(self, latency_s: float) -> None:
        self.latency_s = latency_s

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        time.sleep(self.latency_s)
        sampling = request[POLICY_SAMPLING_REQUEST_FIELD]
        actions = np.zeros((10, 7), dtype=np.float64)
        actions[:, 0] = 0.10
        actions[:, -1] = -1.0
        key_sha256 = sampling["key_sha256"]
        return {
            "actions": actions,
            "source": "cpu_test_pi05_fixture",
            "response_metadata": {
                "policy_sampling": {
                    "schema_version": policy_sampling_contract()["schema_version"],
                    "namespace": "scored",
                    "key_sha256": key_sha256,
                    "noise_sha256": policy_sampling_noise_sha256(
                        policy_sampling_noise(key_sha256)
                    ),
                    "generator": POLICY_SAMPLING_GENERATOR,
                },
                "action_chunk_sha256": canonical_action_chunk_sha256(actions),
            },
        }


def _request_builder() -> IndependentLiberoRequestBuilder:
    return IndependentLiberoRequestBuilder(
        task_description="pick up the black bowl",
        task_suite="libero_spatial",
        task_id=0,
        episode_index=2,
        seed=7,
    )


def test_request_builder_matches_official_contract_and_keyed_sampling() -> None:
    builder = _request_builder()
    first = builder(_observation(), 3, 1.25)
    second = builder(_observation(), 3, 99.0)

    assert set(first) == {
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
        POLICY_SAMPLING_REQUEST_FIELD,
    }
    assert first["prompt"] == "pick up the black bowl"
    assert first[POLICY_SAMPLING_REQUEST_FIELD] == second[POLICY_SAMPLING_REQUEST_FIELD]
    assert first[POLICY_SAMPLING_REQUEST_FIELD] == build_policy_sampling_control(
        "scored",
        7,
        ("libero_spatial", 0, 2, 1),
        3,
    )
    assert canonical_policy_input_sha256(first) == canonical_policy_input_sha256(second)


def test_libero_dynamic_hold_stops_motion_and_preserves_gripper() -> None:
    first = libero_dynamic_hold(np.zeros(7, dtype=np.float64))
    previous = np.asarray([0.1, -0.2, 0.3, 0.4, 0.5, 0.6, 0.75])
    later = libero_dynamic_hold(previous)

    np.testing.assert_array_equal(first, LIBERO_DUMMY_ACTION)
    np.testing.assert_allclose(later[:6], 0.0)
    assert later[-1] == 0.75


def test_true_independent_clock_libero_episode_ticks_during_inference() -> None:
    environment = _FakeLiberoEnvironment()
    result = run_libero_independent_clock_episode(
        environment,
        _DelayedLiberoFactory(0.025),
        np.asarray([0.0]),
        _request_builder(),
        control_period_s=0.05,
        deadline_s=1.0,
        max_task_steps=40,
        num_steps_wait=10,
        startup_timeout_s=5.0,
        shutdown_timeout_s=1.0,
    )

    assert result.task_success
    assert result.termination_reason == "task_success"
    assert result.stabilization_steps == 10
    assert result.task_steps == result.runtime.environment_steps
    assert result.runtime.parent_process_id != result.runtime.worker_process_id
    assert result.runtime.holds >= 2
    assert result.runtime.executes >= 2
    assert result.runtime.submitted >= result.runtime.completed >= 1
    np.testing.assert_allclose(environment.actions[:10], [LIBERO_DUMMY_ACTION] * 10)
    completed = [
        request for request in result.runtime.requests if request.actions is not None
    ]
    assert completed
    assert completed[0].source == "cpu_test_pi05_fixture"
    assert len(completed[0].actions) == 10
    metadata = completed[0].response_metadata
    assert metadata is not None
    assert len(metadata["policy_sampling"]["key_sha256"]) == 64

    collector = _Collector()
    _validate_runtime(
        result.to_dict(),
        ExperimentCell("libero_spatial", 0, 2),
        seed=7,
        period_ms=50.0,
        deadline_ms=1000.0,
        submit_every_ticks=1,
        action_selection_mode=AGE_ALIGNED_SUFFIX,
        require_policy_input_sha256=False,
        collector=collector,
    )
    assert collector.errors == []


def test_action_chunk_hash_is_shape_bound_and_deterministic() -> None:
    actions = np.arange(70, dtype=np.float64).reshape(10, 7)

    assert canonical_action_chunk_sha256(actions) == canonical_action_chunk_sha256(
        actions.copy()
    )
    assert canonical_action_chunk_sha256(actions) != canonical_action_chunk_sha256(
        actions + 1.0
    )
