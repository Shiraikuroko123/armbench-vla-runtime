from __future__ import annotations

import numpy as np

from integrations.openpi.projected_overlap_runtime import (
    OVERLAP_UNCONDITIONED,
    PROJECTED_OVERLAP,
    RTC_GUIDED_OVERLAP,
    OverlapRuntimeConfig,
    run_overlap_episode,
)
from integrations.openpi.serve_policy_attested import (
    POLICY_CONDITIONING_REQUEST_FIELD,
    POLICY_CONDITIONING_RESPONSE_FIELD,
    POLICY_CONDITIONING_TRACE_SCHEMA_VERSION,
    POLICY_SAMPLING_GENERATOR,
    POLICY_SAMPLING_RESPONSE_FIELD,
    POLICY_SAMPLING_SCHEMA_VERSION,
    POLICY_GUIDANCE_REQUEST_FIELD,
    POLICY_GUIDANCE_RESPONSE_FIELD,
    POLICY_GUIDANCE_TRACE_SCHEMA_VERSION,
    build_policy_sampling_control,
    policy_sampling_noise,
    policy_sampling_noise_sha256,
)


def _observation():
    return {
        "agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
    }


class RecordingEnvironment:
    def __init__(self):
        self.actions = []

    def reset(self):
        return None

    def set_init_state(self, _state):
        return _observation()

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float64))
        return _observation(), 0.0, False, {}


class IndexedPolicy:
    def __init__(self):
        self.requests = []

    def infer(self, request):
        query = len(self.requests)
        self.requests.append(request)
        actions = np.zeros((10, 7), dtype=np.float32)
        actions[:, 0] = np.arange(1 + 10 * query, 11 + 10 * query)
        response = {
            "actions": actions,
            "policy_timing": {"infer_ms": 1.0},
        }
        control = request.get(POLICY_CONDITIONING_REQUEST_FIELD)
        if control is not None:
            response[POLICY_CONDITIONING_RESPONSE_FIELD] = {
                "schema_version": POLICY_CONDITIONING_TRACE_SCHEMA_VERSION,
                "method": "projected_flow_inpainting",
                "inference_delay": control["inference_delay"],
                "execute_horizon": control["execute_horizon"],
                "raw_actions_sha256": control["raw_actions_sha256"],
                "model_actions_sha256": "a" * 64,
                "mask_sha256": control["mask_sha256"],
                "max_model_residual": 0.0,
            }
        guidance = request.get(POLICY_GUIDANCE_REQUEST_FIELD)
        if guidance is not None:
            response[POLICY_GUIDANCE_RESPONSE_FIELD] = {
                "schema_version": POLICY_GUIDANCE_TRACE_SCHEMA_VERSION,
                "method": "rtc_pseudoinverse_guidance",
                "inference_delay": guidance["inference_delay"],
                "execute_horizon": guidance["execute_horizon"],
                "schedule": guidance["schedule"],
                "max_guidance_weight": guidance["max_guidance_weight"],
                "guidance_active": True,
                "guided_steps": int(
                    np.count_nonzero(guidance["guidance_weights"])
                ),
                "raw_actions_sha256": guidance["raw_actions_sha256"],
                "model_actions_sha256": "b" * 64,
                "weights_sha256": guidance["weights_sha256"],
                "max_weighted_model_residual": 0.2,
                "weighted_model_rmse": 0.05,
            }
        return response


def _config(method):
    return OverlapRuntimeConfig(
        method=method,
        execute_horizon=5,
        inference_delay_steps=4,
        max_task_steps=10,
        num_steps_wait=0,
        resize_size=4,
    )


def _v2_config(method):
    return OverlapRuntimeConfig(
        method=method,
        execute_horizon=5,
        inference_delay_steps=4,
        max_task_steps=10,
        num_steps_wait=0,
        resize_size=4,
        bootstrap_reference_only=True,
    )


def test_unconditioned_overlap_advances_exactly_e_steps_per_query() -> None:
    environment = RecordingEnvironment()
    result = run_overlap_episode(
        environment,
        IndexedPolicy(),
        np.array([0.0]),
        "task",
        _config(OVERLAP_UNCONDITIONED),
    )

    np.testing.assert_array_equal(
        [action[0] for action in environment.actions],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 15],
    )
    assert [record.executed_steps for record in result.query_records] == [5, 5]
    assert result.query_records[1].old_prefix_steps == 4
    assert result.query_records[1].new_suffix_steps == 1
    assert result.query_records[1].decision == "accepted_unconditioned_overlap"


def test_projected_overlap_conditions_shifted_reference_after_bootstrap() -> None:
    environment = RecordingEnvironment()
    policy = IndexedPolicy()
    result = run_overlap_episode(
        environment,
        policy,
        np.array([0.0]),
        "task",
        _config(PROJECTED_OVERLAP),
    )

    assert POLICY_CONDITIONING_REQUEST_FIELD not in policy.requests[0]
    control = policy.requests[1][POLICY_CONDITIONING_REQUEST_FIELD]
    np.testing.assert_array_equal(
        control["condition_actions"][:, 0],
        [6, 7, 8, 9, 10, 0, 0, 0, 0, 0],
    )
    np.testing.assert_array_equal(
        control["condition_mask"],
        [True, True, True, True, False, False, False, False, False, False],
    )
    assert result.bootstrap_queries == 1
    assert result.conditioned_queries == 1
    assert result.query_records[1].max_model_residual == 0.0
    assert result.query_records[1].decision == "accepted_projected_overlap"


def test_projected_overlap_fails_closed_when_trace_is_missing() -> None:
    class MissingTracePolicy(IndexedPolicy):
        def infer(self, request):
            response = super().infer(request)
            response.pop(POLICY_CONDITIONING_RESPONSE_FIELD, None)
            return response

    result = run_overlap_episode(
        RecordingEnvironment(),
        MissingTracePolicy(),
        np.array([0.0]),
        "task",
        _config(PROJECTED_OVERLAP),
    )

    assert not result.success
    assert result.termination_reason == "overlap_query_failure"
    assert result.failure_stage == "policy_response_validation"
    assert "trace is missing" in result.failure_message


def test_scored_sampling_control_is_audited_per_query() -> None:
    class SamplingPolicy(IndexedPolicy):
        def infer(self, request):
            response = super().infer(request)
            control = request["armbench_policy_sampling"]
            response[POLICY_SAMPLING_RESPONSE_FIELD] = {
                "schema_version": POLICY_SAMPLING_SCHEMA_VERSION,
                "namespace": control["namespace"],
                "key_sha256": control["key_sha256"],
                "noise_sha256": policy_sampling_noise_sha256(
                    policy_sampling_noise(control["key_sha256"])
                ),
                "generator": POLICY_SAMPLING_GENERATOR,
            }
            return response

    def sampling(query_index):
        return build_policy_sampling_control(
            "scored", 7, ["libero_10", 0, 0, 5], query_index
        )

    result = run_overlap_episode(
        RecordingEnvironment(),
        SamplingPolicy(),
        np.array([0.0]),
        "task",
        _config(OVERLAP_UNCONDITIONED),
        sampling_control_builder=sampling,
    )

    assert all(record.sampling_key_sha256 for record in result.query_records)
    assert all(record.sampling_noise_sha256 for record in result.query_records)


def test_v2_bootstrap_builds_reference_without_executing_actions() -> None:
    environment = RecordingEnvironment()
    policy = IndexedPolicy()

    result = run_overlap_episode(
        environment,
        policy,
        np.array([0.0]),
        "task",
        _v2_config(RTC_GUIDED_OVERLAP),
    )

    assert POLICY_GUIDANCE_REQUEST_FIELD not in policy.requests[0]
    guidance = policy.requests[1][POLICY_GUIDANCE_REQUEST_FIELD]
    np.testing.assert_array_equal(
        guidance["guidance_actions"][:, 0],
        np.arange(1, 11),
    )
    np.testing.assert_allclose(
        guidance["guidance_weights"],
        [1.0, 1.0, 1.0, 1.0, 0.18877034, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        [action[0] for action in environment.actions],
        [1, 2, 3, 4, 15, 16, 17, 18, 19, 25],
    )
    assert result.query_records[0].executed_steps == 0
    assert result.query_records[0].decision == "bootstrap_reference_only"
    assert result.query_records[1].decision == "accepted_rtc_guided_overlap"
    assert result.query_records[1].weighted_model_rmse == 0.05
    assert result.bootstrap_queries == 1
    assert result.conditioned_queries == 2
    assert len(result.transition_records) == 2


def test_rtc_guided_overlap_fails_closed_when_trace_is_missing() -> None:
    class MissingTracePolicy(IndexedPolicy):
        def infer(self, request):
            response = super().infer(request)
            response.pop(POLICY_GUIDANCE_RESPONSE_FIELD, None)
            return response

    result = run_overlap_episode(
        RecordingEnvironment(),
        MissingTracePolicy(),
        np.array([0.0]),
        "task",
        _v2_config(RTC_GUIDED_OVERLAP),
    )

    assert not result.success
    assert result.termination_reason == "overlap_query_failure"
    assert result.failure_stage == "policy_response_validation"
    assert "guidance trace is missing" in result.failure_message
