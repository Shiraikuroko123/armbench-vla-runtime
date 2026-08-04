from __future__ import annotations

import numpy as np
import pytest

from integrations.openpi.libero_runtime import (
    ASYNC_UNGUARDED,
    FIXED_REFRESH,
    STATE_GUARD,
    PolicyResponseError,
    RuntimeConfig,
    build_libero_request,
    run_episode,
    validate_action_chunk,
)


def _observation(position_x: float = 0.0) -> dict:
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


class FakeEnvironment:
    def __init__(self, *, drift_per_step: float = 0.0, success_x: float = 999.0):
        self.drift_per_step = drift_per_step
        self.success_x = success_x
        self.position_x = 0.0
        self.actions = []
        self.initial_states = []

    def reset(self):
        self.position_x = 0.0
        self.actions = []
        return _observation(self.position_x)

    def set_init_state(self, initial_state):
        state = np.asarray(initial_state, dtype=np.float64).copy()
        self.initial_states.append(state)
        self.position_x = float(state[0])
        return _observation(self.position_x)

    def step(self, action):
        action_array = np.asarray(action, dtype=np.float64)
        self.actions.append(action_array.copy())
        self.position_x += float(action_array[0]) + self.drift_per_step
        done = self.position_x >= self.success_x
        return _observation(self.position_x), float(done), done, {}


class ConstantPolicy:
    def __init__(self, action_x: float = 0.01, horizon: int = 8):
        self.actions = np.zeros((horizon, 7), dtype=np.float64)
        self.actions[:, 0] = action_x
        self.actions[:, -1] = -1.0
        self.requests = []

    def infer(self, request):
        self.requests.append(request)
        return {
            "actions": self.actions.copy(),
            "policy_timing": {"infer_ms": 12.5},
            "server_timing": {"infer_ms": 14.0},
        }


def _config(mode: str, **overrides) -> RuntimeConfig:
    values = {
        "mode": mode,
        "replan_steps": 1,
        "latency_steps": 2,
        "max_task_steps": 8,
        "num_steps_wait": 0,
        "position_threshold_m": 0.005,
        "orientation_threshold_rad": 0.05,
        "gripper_threshold": 0.01,
    }
    values.update(overrides)
    return RuntimeConfig(**values)


def test_build_request_matches_official_pi05_libero_contract() -> None:
    observation = _observation()
    request = build_libero_request(observation, "pick up the bowl", 224)

    assert set(request) == {
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
    }
    assert request["observation/image"].shape == (224, 224, 3)
    assert request["observation/image"].dtype == np.uint8
    np.testing.assert_array_equal(
        request["observation/image"][0, 0],
        observation["agentview_image"][-1, -1],
    )
    np.testing.assert_allclose(
        request["observation/state"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, -0.02],
    )
    assert request["prompt"] == "pick up the bowl"


def test_paired_modes_receive_byte_identical_initial_states() -> None:
    initial_state = np.asarray([0.125, 2.0, 3.0], dtype=np.float64)
    results = []
    environments = []
    for mode in (ASYNC_UNGUARDED, STATE_GUARD):
        environment = FakeEnvironment()
        environments.append(environment)
        results.append(
            run_episode(
                environment,
                ConstantPolicy(action_x=0.0),
                initial_state,
                "test task",
                _config(mode, latency_steps=0, max_task_steps=1),
            )
        )

    assert results[0].initial_state_sha256 == results[1].initial_state_sha256
    assert environments[0].initial_states[0].tobytes() == initial_state.tobytes()
    assert environments[1].initial_states[0].tobytes() == initial_state.tobytes()


def test_unguarded_mode_executes_stale_chunk_after_latency_steps() -> None:
    result = run_episode(
        FakeEnvironment(),
        ConstantPolicy(action_x=0.01),
        np.asarray([0.0]),
        "test task",
        _config(ASYNC_UNGUARDED),
    )

    assert result.latency_action_steps > 0
    assert result.stale_chunks_executed > 0
    assert result.stale_action_steps > 0
    second_query = result.query_records[1]
    assert second_query.accepted
    assert second_query.decision == "accepted_unguarded"
    assert second_query.mismatch.position_m == pytest.approx(0.02)
    assert second_query.policy_inference_latency_ms == pytest.approx(12.5)
    assert second_query.server_inference_latency_ms == pytest.approx(14.0)


def test_state_guard_rejects_mismatch_then_requeries_from_hold() -> None:
    result = run_episode(
        FakeEnvironment(),
        ConstantPolicy(action_x=0.01),
        np.asarray([0.0]),
        "test task",
        _config(STATE_GUARD, max_task_steps=10),
    )

    rejected = [record for record in result.query_records if not record.accepted]
    assert rejected
    assert rejected[0].decision == "rejected_state_mismatch"
    assert rejected[0].rejection_reasons == ("position_mismatch",)
    assert result.interventions == result.rejected_chunks
    accepted_after_rejection = result.query_records[
        result.query_records.index(rejected[0]) + 1
    ]
    assert accepted_after_rejection.accepted
    assert accepted_after_rejection.mismatch.position_m == pytest.approx(0.0)


def test_guard_has_bounded_requery_failure() -> None:
    result = run_episode(
        FakeEnvironment(drift_per_step=0.01),
        ConstantPolicy(action_x=0.0),
        np.asarray([0.0]),
        "test task",
        _config(STATE_GUARD, max_requeries=1, max_task_steps=20),
    )

    assert not result.success
    assert result.termination_reason == "max_requeries_exceeded"
    assert result.policy_queries == 2
    assert result.rejected_chunks == 2


def test_fixed_refresh_reuses_hold_and_requery_path_without_state_trigger() -> None:
    result = run_episode(
        FakeEnvironment(),
        ConstantPolicy(action_x=0.01),
        np.asarray([0.0]),
        "test task",
        _config(
            FIXED_REFRESH,
            fixed_refresh_interval=2,
            position_threshold_m=999.0,
            max_task_steps=12,
        ),
    )

    rejected = [
        record
        for record in result.query_records
        if record.decision == "rejected_fixed_refresh"
    ]
    assert rejected
    assert rejected[0].rejection_reasons == ("scheduled_refresh",)
    assert rejected[0].mismatch.position_m < 999.0
    accepted_after_rejection = result.query_records[
        result.query_records.index(rejected[0]) + 1
    ]
    assert accepted_after_rejection.decision == "accepted_fixed_refresh"
    assert accepted_after_rejection.mismatch.position_m == pytest.approx(0.0)
    assert result.interventions == result.rejected_chunks


@pytest.mark.parametrize(
    "response, expected_message",
    [
        ({"actions": np.zeros((8, 8))}, "shape"),
        ({"actions": np.full((8, 7), np.nan)}, "finite"),
        ({"wrong": np.zeros((8, 7))}, "actions"),
    ],
)
def test_invalid_policy_chunks_fail_closed_and_remain_recorded(
    response, expected_message
) -> None:
    class InvalidPolicy:
        def infer(self, request):
            return response

    result = run_episode(
        FakeEnvironment(),
        InvalidPolicy(),
        np.asarray([0.0]),
        "test task",
        _config(ASYNC_UNGUARDED),
    )

    assert not result.success
    assert result.termination_reason == "invalid_policy_response"
    assert result.policy_queries == 1
    assert result.query_records[0].decision == "policy_response_validation_error"
    assert result.failure_category == "policy_contract"
    assert result.query_records[0].error_stage == "policy_response_validation"
    assert result.query_records[0].error_type == PolicyResponseError.__name__
    assert expected_message in result.query_records[0].error_message


def test_action_validator_rejects_short_chunks() -> None:
    with pytest.raises(PolicyResponseError, match="returned 2 actions"):
        validate_action_chunk({"actions": np.zeros((2, 7))}, replan_steps=5)


def test_runtime_config_rejects_invalid_experiment_conditions() -> None:
    with pytest.raises(ValueError, match="mode"):
        _config("unknown")
    with pytest.raises(ValueError, match="latency_steps"):
        _config(ASYNC_UNGUARDED, latency_steps=-1)
    with pytest.raises(ValueError, match="position_threshold_m"):
        _config(STATE_GUARD, position_threshold_m=float("nan"))
    with pytest.raises(ValueError, match="fixed_refresh_interval"):
        _config(FIXED_REFRESH)
    with pytest.raises(ValueError, match="max_requeries"):
        _config(FIXED_REFRESH, fixed_refresh_interval=2, max_requeries=0)
