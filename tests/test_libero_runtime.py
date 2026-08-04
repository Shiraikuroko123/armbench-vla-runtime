from __future__ import annotations

import numpy as np
import pytest

from integrations.openpi.libero_runtime import (
    ASYNC_UNGUARDED,
    FIXED_REFRESH,
    LATENCY_ALIGNED,
    MEASURED_WALL_LATENCY,
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


def test_latency_aligned_mode_discards_delayed_action_prefix() -> None:
    environment = FakeEnvironment()

    class IndexedPolicy(ConstantPolicy):
        def __init__(self) -> None:
            super().__init__(action_x=0.0, horizon=8)
            self.actions[:, 0] = np.arange(1, 9, dtype=np.float64) / 100.0

    result = run_episode(
        environment,
        IndexedPolicy(),
        np.asarray([0.0]),
        "test task",
        _config(
            LATENCY_ALIGNED,
            replan_steps=2,
            latency_steps=2,
            max_task_steps=4,
        ),
    )

    assert result.query_records[0].decision == "accepted_latency_aligned"
    np.testing.assert_allclose(
        [action[0] for action in environment.actions[:4]],
        [0.0, 0.0, 0.03, 0.04],
    )


def test_latency_aligned_mode_fails_closed_for_short_chunk() -> None:
    result = run_episode(
        FakeEnvironment(),
        ConstantPolicy(horizon=2),
        np.asarray([0.0]),
        "test task",
        _config(LATENCY_ALIGNED, replan_steps=1, latency_steps=2),
    )

    assert not result.success
    assert result.termination_reason == "invalid_policy_response"
    assert result.query_records[0].error_type == PolicyResponseError.__name__
    assert "required_action_steps=3" in result.query_records[0].error_message


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
    with pytest.raises(
        PolicyResponseError, match="returned 2 actions.*required_action_steps=5"
    ):
        validate_action_chunk({"actions": np.zeros((2, 7))}, required_steps=5)


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


class FakeMonotonicClock:
    def __init__(self) -> None:
        self.now_s = 0.0

    def __call__(self) -> float:
        return self.now_s

    def advance_ms(self, milliseconds: float) -> None:
        self.now_s += milliseconds / 1000.0

    def sleep(self, seconds: float) -> None:
        self.now_s += seconds


class TimedIndexedPolicy(ConstantPolicy):
    def __init__(self, clock: FakeMonotonicClock, inference_ms: float) -> None:
        super().__init__(action_x=0.0, horizon=10)
        self.clock = clock
        self.inference_ms = inference_ms
        self.actions[:, 0] = np.arange(1, 11, dtype=np.float64) / 100.0

    def infer(self, request):
        self.clock.advance_ms(self.inference_ms)
        return super().infer(request)


class SequencedTimedPolicy(TimedIndexedPolicy):
    def __init__(
        self, clock: FakeMonotonicClock, inference_schedule_ms: list[float]
    ) -> None:
        super().__init__(clock, inference_ms=0.0)
        self.inference_schedule_ms = list(inference_schedule_ms)

    def infer(self, request):
        index = len(self.requests)
        self.clock.advance_ms(self.inference_schedule_ms[index])
        return ConstantPolicy.infer(self, request)


def test_measured_wall_alignment_uses_observed_age_not_fixed_delay() -> None:
    clock = FakeMonotonicClock()
    environment = FakeEnvironment()
    result = run_episode(
        environment,
        TimedIndexedPolicy(clock, inference_ms=120.0),
        np.asarray([0.0]),
        "test task",
        _config(
            LATENCY_ALIGNED,
            latency_steps=0,
            latency_source=MEASURED_WALL_LATENCY,
            control_period_ms=50.0,
            age_rounding="ceil",
            replan_steps=2,
            max_task_steps=4,
        ),
        clock=clock,
        sleeper=clock.sleep,
    )

    first = result.query_records[0]
    assert first.decision == "accepted_measured_latency_aligned"
    assert first.observation_age_ms == pytest.approx(120.0)
    assert first.observation_captured_monotonic_ns == 0
    assert first.policy_call_started_monotonic_ns == 0
    assert first.policy_call_finished_monotonic_ns == 120_000_000
    assert first.response_ready_monotonic_ns == 120_000_000
    assert first.response_delivery_elapsed_ms == pytest.approx(0.0)
    assert first.simulated_catchup_steps == 2
    assert first.measured_stale_steps == 3
    assert first.action_offset_steps == 3
    assert first.selected_stop_step == 5
    assert first.alignment_disposition == "execute"
    assert first.alignment_reason == "fresh_suffix_available"
    assert first.injected_latency_steps_executed == 0
    assert result.latency_action_steps == 2
    np.testing.assert_allclose(
        [action[0] for action in environment.actions],
        [0.0, 0.0, 0.04, 0.05],
    )


def test_measured_wall_age_includes_seedable_delivery_jitter() -> None:
    clock = FakeMonotonicClock()
    result = run_episode(
        FakeEnvironment(),
        TimedIndexedPolicy(clock, inference_ms=40.0),
        np.asarray([0.0]),
        "test task",
        _config(
            LATENCY_ALIGNED,
            latency_steps=0,
            latency_source=MEASURED_WALL_LATENCY,
            replan_steps=1,
            max_task_steps=3,
        ),
        clock=clock,
        response_jitter_ms=lambda _: 80.0,
        sleeper=clock.sleep,
    )

    first = result.query_records[0]
    assert first.inference_latency_ms == pytest.approx(40.0)
    assert first.response_jitter_ms == pytest.approx(80.0)
    assert first.response_delivery_elapsed_ms == pytest.approx(80.0)
    assert first.response_ready_monotonic_ns == 120_000_000
    assert first.observation_age_ms == pytest.approx(120.0)
    assert first.measured_stale_steps == 3


def test_measured_horizon_overrun_fails_closed_without_executing_suffix() -> None:
    clock = FakeMonotonicClock()
    environment = FakeEnvironment()
    result = run_episode(
        environment,
        TimedIndexedPolicy(clock, inference_ms=251.0),
        np.asarray([0.0]),
        "test task",
        _config(
            LATENCY_ALIGNED,
            latency_steps=0,
            latency_source=MEASURED_WALL_LATENCY,
            replan_steps=5,
            max_task_steps=20,
            max_age_refreshes=0,
        ),
        clock=clock,
        sleeper=clock.sleep,
    )

    assert not result.success
    assert result.termination_reason == "stale_horizon_refresh_exhausted"
    assert result.horizon_overruns == 1
    assert result.accepted_chunks == 0
    assert result.rejected_chunks == 1
    assert result.query_records[0].decision == (
        "rejected_horizon_overrun_fail_closed"
    )
    assert len(environment.actions) == 6
    np.testing.assert_allclose(
        [action[0] for action in environment.actions],
        np.zeros(6),
    )
    assert result.fallback_hold_steps == 1
    assert result.query_records[0].fallback_hold_steps == 1


def test_measured_horizon_overrun_holds_and_accepts_fresh_retry() -> None:
    clock = FakeMonotonicClock()
    environment = FakeEnvironment()
    result = run_episode(
        environment,
        SequencedTimedPolicy(clock, [251.0, 120.0, 120.0]),
        np.asarray([0.0]),
        "test task",
        _config(
            LATENCY_ALIGNED,
            latency_steps=0,
            latency_source=MEASURED_WALL_LATENCY,
            replan_steps=5,
            max_task_steps=13,
            max_age_refreshes=1,
        ),
        clock=clock,
        sleeper=clock.sleep,
    )

    assert result.query_records[0].decision == (
        "rejected_horizon_overrun_hold_refresh"
    )
    assert result.query_records[1].decision == (
        "accepted_measured_latency_aligned"
    )
    assert result.age_refreshes == 1
    assert result.horizon_overruns == 1
    np.testing.assert_allclose(
        [action[0] for action in environment.actions[:13]],
        [0.0] * 8 + [0.04, 0.05, 0.06, 0.07, 0.08],
    )


def test_measured_refresh_budget_resets_after_an_accepted_chunk() -> None:
    clock = FakeMonotonicClock()
    result = run_episode(
        FakeEnvironment(),
        SequencedTimedPolicy(clock, [251.0, 120.0, 251.0, 120.0]),
        np.asarray([0.0]),
        "test task",
        _config(
            LATENCY_ALIGNED,
            latency_steps=0,
            latency_source=MEASURED_WALL_LATENCY,
            replan_steps=5,
            max_task_steps=26,
            max_age_refreshes=1,
        ),
        clock=clock,
        sleeper=clock.sleep,
    )

    rejected = [record for record in result.query_records if not record.accepted]
    assert [record.decision for record in rejected] == [
        "rejected_horizon_overrun_hold_refresh",
        "rejected_horizon_overrun_hold_refresh",
    ]
    assert [record.age_refresh_index for record in rejected] == [0, 0]
    assert result.age_refreshes == 2


def test_fail_closed_sends_hold_after_a_nonzero_previous_action() -> None:
    clock = FakeMonotonicClock()
    environment = FakeEnvironment()
    result = run_episode(
        environment,
        SequencedTimedPolicy(clock, [40.0, 251.0]),
        np.asarray([0.0]),
        "test task",
        _config(
            LATENCY_ALIGNED,
            latency_steps=0,
            latency_source=MEASURED_WALL_LATENCY,
            replan_steps=1,
            deadline_ms=200.0,
            max_task_steps=20,
            max_age_refreshes=0,
        ),
        clock=clock,
        sleeper=clock.sleep,
    )

    assert result.query_records[0].accepted
    assert result.query_records[1].decision == (
        "rejected_deadline_exceeded_fail_closed"
    )
    assert environment.actions[-2][0] == pytest.approx(0.02)
    assert environment.actions[-1][0] == pytest.approx(0.0)
    assert result.query_records[1].fallback_hold_steps == 1


def test_measured_deadline_can_reject_even_when_suffix_fits() -> None:
    clock = FakeMonotonicClock()
    result = run_episode(
        FakeEnvironment(),
        TimedIndexedPolicy(clock, inference_ms=201.0),
        np.asarray([0.0]),
        "test task",
        _config(
            LATENCY_ALIGNED,
            latency_steps=0,
            latency_source=MEASURED_WALL_LATENCY,
            replan_steps=1,
            deadline_ms=200.0,
            max_age_refreshes=0,
            max_task_steps=20,
        ),
        clock=clock,
        sleeper=clock.sleep,
    )

    assert result.termination_reason == "deadline_refresh_exhausted"
    assert result.deadline_misses == 1
    assert not result.query_records[0].horizon_overrun
    assert result.query_records[0].deadline_exceeded


def test_measured_async_records_would_overrun_but_executes_offset_zero() -> None:
    clock = FakeMonotonicClock()
    result = run_episode(
        FakeEnvironment(),
        TimedIndexedPolicy(clock, inference_ms=251.0),
        np.asarray([0.0]),
        "test task",
        _config(
            ASYNC_UNGUARDED,
            latency_steps=0,
            latency_source=MEASURED_WALL_LATENCY,
            replan_steps=5,
            max_task_steps=10,
        ),
        clock=clock,
        sleeper=clock.sleep,
    )

    record = result.query_records[0]
    assert record.accepted
    assert record.horizon_overrun
    assert record.available_suffix_steps == 4
    assert record.action_offset_steps == 0
    assert result.horizon_overruns == 1


def test_measured_wall_protocol_rejects_ambiguous_fixed_delay() -> None:
    with pytest.raises(ValueError, match="requires latency_steps=0"):
        _config(
            LATENCY_ALIGNED,
            latency_source=MEASURED_WALL_LATENCY,
            latency_steps=1,
        )


def test_invalid_response_jitter_is_recorded_as_latency_failure() -> None:
    clock = FakeMonotonicClock()
    result = run_episode(
        FakeEnvironment(),
        TimedIndexedPolicy(clock, inference_ms=40.0),
        np.asarray([0.0]),
        "test task",
        _config(
            LATENCY_ALIGNED,
            latency_steps=0,
            latency_source=MEASURED_WALL_LATENCY,
        ),
        clock=clock,
        response_jitter_ms=lambda _: float("nan"),
        sleeper=clock.sleep,
    )

    assert not result.success
    assert result.termination_reason == "invalid_latency_measurement"
    assert result.failure_category == "latency_measurement"
    assert result.query_records[0].error_stage == "response_delivery"


@pytest.mark.parametrize(
    "clock",
    [
        lambda: float("nan"),
        lambda: (_ for _ in ()).throw(RuntimeError("clock unavailable")),
    ],
)
def test_invalid_observation_clock_is_a_latency_failure(clock) -> None:
    result = run_episode(
        FakeEnvironment(),
        ConstantPolicy(),
        np.asarray([0.0]),
        "test task",
        _config(
            LATENCY_ALIGNED,
            latency_steps=0,
            latency_source=MEASURED_WALL_LATENCY,
        ),
        clock=clock,
    )

    assert result.termination_reason == "invalid_latency_measurement"
    assert result.failure_category == "latency_measurement"
    assert result.query_records[0].error_stage == "latency_measurement"


def test_backward_clock_is_recorded_without_masking_the_original_failure() -> None:
    values = iter((1.0, 1.0, 0.5))
    result = run_episode(
        FakeEnvironment(),
        ConstantPolicy(),
        np.asarray([0.0]),
        "test task",
        _config(
            LATENCY_ALIGNED,
            latency_steps=0,
            latency_source=MEASURED_WALL_LATENCY,
        ),
        clock=lambda: next(values),
    )

    assert result.termination_reason == "invalid_latency_measurement"
    assert result.query_records[0].error_type == "ValueError"
    assert "moved backwards" in result.query_records[0].error_message
