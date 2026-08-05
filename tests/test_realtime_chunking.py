from __future__ import annotations

import numpy as np
import pytest

from integrations.openpi.realtime_chunking import (
    _finite_difference_vjp,
    RealtimeChunkingError,
    build_committed_action_condition,
    build_overlap_plan,
    prefix_attention_weights,
    projected_flow_inpainting,
    reverse_rtc_guidance_gain,
    rtc_pseudoinverse_guidance_reverse,
)


def _indexed_chunk(offset: float = 0.0) -> np.ndarray:
    values = np.arange(20, dtype=np.float64).reshape(10, 2)
    return values + offset


def test_overlap_plan_matches_official_rtc_execution_window() -> None:
    previous = _indexed_chunk(0.0)
    response = _indexed_chunk(100.0)

    plan = build_overlap_plan(
        previous,
        response,
        inference_delay=4,
        execute_horizon=5,
    )

    np.testing.assert_array_equal(plan.inference_actions, previous[:4])
    np.testing.assert_array_equal(plan.response_actions, response[4:5])
    np.testing.assert_array_equal(
        plan.actions_to_execute,
        np.concatenate((previous[:4], response[4:5]), axis=0),
    )
    np.testing.assert_array_equal(plan.next_reference_actions[:5], response[5:])
    np.testing.assert_array_equal(plan.next_reference_actions[5:], 0.0)
    assert plan.actions_to_execute.shape == (5, 2)


@pytest.mark.parametrize("delay", [0, 5])
def test_overlap_boundary_keeps_fixed_execute_horizon(delay: int) -> None:
    previous = _indexed_chunk(0.0)
    response = _indexed_chunk(100.0)

    plan = build_overlap_plan(
        previous,
        response,
        inference_delay=delay,
        execute_horizon=5,
    )

    expected = np.concatenate((previous[:delay], response[delay:5]), axis=0)
    np.testing.assert_array_equal(plan.actions_to_execute, expected)
    assert plan.actions_to_execute.shape[0] == 5


@pytest.mark.parametrize(
    ("delay", "execute", "match"),
    [
        (6, 5, "must not exceed execute_horizon"),
        (0, 11, "must not exceed action horizon"),
        (-1, 5, "nonnegative"),
    ],
)
def test_overlap_rejects_invalid_horizons(
    delay: int, execute: int, match: str
) -> None:
    with pytest.raises(RealtimeChunkingError, match=match):
        build_overlap_plan(
            _indexed_chunk(),
            _indexed_chunk(100.0),
            inference_delay=delay,
            execute_horizon=execute,
        )


def test_overlap_rejects_shape_and_nonfinite_chunks() -> None:
    with pytest.raises(RealtimeChunkingError, match="equal shape"):
        build_overlap_plan(
            np.zeros((10, 7)),
            np.zeros((9, 7)),
            inference_delay=2,
            execute_horizon=5,
        )
    bad = np.zeros((10, 7))
    bad[3, 2] = np.nan
    with pytest.raises(RealtimeChunkingError, match="finite"):
        build_overlap_plan(
            bad,
            np.zeros((10, 7)),
            inference_delay=2,
            execute_horizon=5,
        )


def test_prefix_attention_weights_match_rtc_reference_example() -> None:
    expected = np.array([1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        prefix_attention_weights(2, 6, 10, "linear"), expected
    )
    np.testing.assert_array_equal(
        prefix_attention_weights(2, 6, 10, "zeros"),
        np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(
        prefix_attention_weights(2, 6, 10, "ones"),
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    )


def test_committed_condition_separates_hard_prefix_from_soft_guidance() -> None:
    reference = np.zeros((10, 7), dtype=np.float64)
    condition = build_committed_action_condition(
        reference,
        inference_delay=4,
        execute_horizon=5,
        schedule="exp",
    )

    np.testing.assert_array_equal(
        condition.hard_prefix_mask,
        np.array([True, True, True, True, False, False, False, False, False, False]),
    )
    assert np.all(condition.guidance_weights[:4] == 1.0)
    assert 0.0 < condition.guidance_weights[4] < 1.0
    assert np.all(condition.guidance_weights[5:] == 0.0)
    assert not condition.reference_actions.flags.writeable
    assert not condition.hard_prefix_mask.flags.writeable


def test_projected_inpainting_is_identical_when_mask_is_empty() -> None:
    noise = np.arange(12, dtype=np.float64).reshape(3, 4) / 10.0
    condition = np.full_like(noise, -3.0)

    def velocity(state: np.ndarray, time: float) -> np.ndarray:
        return 0.25 * state + time

    expected = noise[None, ...].copy()
    time = 1.0
    dt = -0.2
    for _ in range(5):
        expected = expected + dt * velocity(expected.copy(), time)
        time = max(0.0, time + dt)

    actual = projected_flow_inpainting(
        noise,
        condition,
        np.zeros(3, dtype=np.bool_),
        velocity,
        num_steps=5,
    )
    np.testing.assert_allclose(actual, expected[0], rtol=0.0, atol=1e-12)


def test_projected_inpainting_fixes_different_prefixes_in_batch() -> None:
    noise = np.arange(40, dtype=np.float64).reshape(2, 5, 4) / 10.0
    condition = -noise - 1.0
    mask = np.array(
        [
            [True, True, False, False, False],
            [True, True, True, True, False],
        ],
        dtype=np.bool_,
    )

    output = projected_flow_inpainting(
        noise,
        condition,
        mask,
        lambda state, _time: np.full_like(state, 0.75),
        num_steps=10,
    )

    np.testing.assert_array_equal(output[0, :2], condition[0, :2])
    np.testing.assert_array_equal(output[1, :4], condition[1, :4])
    assert np.max(np.abs(output[mask] - condition[mask])) == 0.0


def test_projected_inpainting_rejects_nonprefix_mask_and_bad_velocity() -> None:
    noise = np.zeros((5, 3), dtype=np.float64)
    condition = np.ones_like(noise)
    with pytest.raises(RealtimeChunkingError, match="contiguous prefix"):
        projected_flow_inpainting(
            noise,
            condition,
            np.array([True, False, True, False, False]),
            lambda state, _time: state,
        )
    with pytest.raises(RealtimeChunkingError, match="shape"):
        projected_flow_inpainting(
            noise,
            condition,
            np.array([True, False, False, False, False]),
            lambda _state, _time: np.zeros((1, 1)),
        )
    with pytest.raises(RealtimeChunkingError, match="finite"):
        projected_flow_inpainting(
            noise,
            condition,
            np.array([True, False, False, False, False]),
            lambda state, _time: np.full_like(state, np.nan),
        )


def test_reverse_rtc_guidance_gain_is_stable_at_flow_endpoints() -> None:
    assert reverse_rtc_guidance_gain(0.0) == 5.0
    assert reverse_rtc_guidance_gain(1.0) == 5.0
    assert reverse_rtc_guidance_gain(1e-15) == 5.0
    assert reverse_rtc_guidance_gain(1.0 - 1e-15) == 5.0
    assert reverse_rtc_guidance_gain(0.5) == 2.0
    assert reverse_rtc_guidance_gain(0.5, max_guidance_weight=0.0) == 0.0


def test_finite_difference_vjp_matches_linear_adjoint() -> None:
    matrix = np.array(
        [
            [0.8, -0.2, 0.0, 0.1],
            [0.3, 1.1, -0.4, 0.0],
            [0.0, 0.2, 0.7, -0.1],
            [-0.5, 0.0, 0.2, 0.9],
        ],
        dtype=np.float64,
    )
    value = np.array([[0.2, -0.4], [0.6, 0.8]], dtype=np.float64)
    cotangent = np.array([[0.5, -0.1], [0.3, -0.7]], dtype=np.float64)

    def linear(chunk: np.ndarray) -> np.ndarray:
        return (matrix @ chunk.reshape(-1)).reshape(chunk.shape)

    actual = _finite_difference_vjp(
        linear,
        value,
        cotangent,
        epsilon=1e-6,
    )
    expected = (matrix.T @ cotangent.reshape(-1)).reshape(value.shape)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-10)


def test_reverse_rtc_matches_official_forward_clock_on_linear_field() -> None:
    matrix = np.array(
        [
            [0.10, -0.03, 0.00, 0.02],
            [0.04, -0.08, 0.01, 0.00],
            [0.00, 0.02, 0.06, -0.01],
            [-0.02, 0.00, 0.03, 0.05],
        ],
        dtype=np.float64,
    )
    bias = np.array([[0.02, -0.04], [0.03, 0.01]], dtype=np.float64)
    noise = np.array([[0.4, -0.2], [0.1, 0.3]], dtype=np.float64)
    reference = np.array([[-0.3, 0.5], [0.2, -0.1]], dtype=np.float64)
    weights = np.array([1.0, 0.35], dtype=np.float64)
    steps = 8

    def official_velocity(chunk: np.ndarray, time: float) -> np.ndarray:
        flat = matrix @ chunk.reshape(-1)
        return flat.reshape(chunk.shape) + bias * (1.0 + time)

    forward_state = noise.copy()
    forward_dt = 1.0 / steps
    identity = np.eye(noise.size, dtype=np.float64)
    for step_index in range(steps):
        forward_time = step_index * forward_dt
        raw_velocity = official_velocity(forward_state, forward_time)
        denoised = forward_state + (1.0 - forward_time) * raw_velocity
        error = (reference - denoised) * weights[:, None]
        jacobian = identity + (1.0 - forward_time) * matrix
        correction = (
            jacobian.T @ error.reshape(-1)
        ).reshape(forward_state.shape)
        gain = reverse_rtc_guidance_gain(1.0 - forward_time)
        forward_state = forward_state + forward_dt * (
            raw_velocity + gain * correction
        )

    def reverse_velocity(batch: np.ndarray, time: float) -> np.ndarray:
        return -np.stack(
            [official_velocity(chunk, 1.0 - time) for chunk in batch], axis=0
        )

    def reverse_vjp(
        _chunk: np.ndarray, time: float, cotangent: np.ndarray
    ) -> np.ndarray:
        jacobian = identity + time * matrix
        return (jacobian.T @ cotangent.reshape(-1)).reshape(noise.shape)

    reverse_state = rtc_pseudoinverse_guidance_reverse(
        noise,
        reference,
        weights,
        reverse_velocity,
        num_steps=steps,
        denoiser_vjp=reverse_vjp,
    )

    np.testing.assert_allclose(reverse_state, forward_state, rtol=0.0, atol=1e-12)


def test_reverse_rtc_zero_weights_are_exact_legacy_sampler_parity() -> None:
    noise = np.arange(12, dtype=np.float64).reshape(3, 4) / 9.0

    def velocity(batch: np.ndarray, time: float) -> np.ndarray:
        return 0.15 * batch + time * 0.2

    expected = noise[None, ...].copy()
    time = 1.0
    dt = -0.1
    for step_index in range(10):
        expected = expected + dt * velocity(expected.copy(), time)
        time = 0.0 if step_index == 9 else max(0.0, time + dt)

    def forbidden_vjp(
        _chunk: np.ndarray, _time: float, _cotangent: np.ndarray
    ) -> np.ndarray:
        raise AssertionError("zero guidance must bypass the VJP")

    actual = rtc_pseudoinverse_guidance_reverse(
        noise,
        np.full_like(noise, -2.0),
        np.zeros(noise.shape[0], dtype=np.float64),
        velocity,
        denoiser_vjp=forbidden_vjp,
    )

    np.testing.assert_array_equal(actual, expected[0])


def test_reverse_rtc_guidance_reduces_weighted_endpoint_error() -> None:
    noise = np.zeros((2, 2), dtype=np.float64)
    reference = np.ones_like(noise)
    weights = np.array([1.0, 0.25], dtype=np.float64)

    def zero_velocity(batch: np.ndarray, _time: float) -> np.ndarray:
        return np.zeros_like(batch)

    baseline = rtc_pseudoinverse_guidance_reverse(
        noise,
        reference,
        np.zeros_like(weights),
        zero_velocity,
    )
    guided = rtc_pseudoinverse_guidance_reverse(
        noise,
        reference,
        weights,
        zero_velocity,
    )
    baseline_error = np.linalg.norm((reference - baseline) * weights[:, None])
    guided_error = np.linalg.norm((reference - guided) * weights[:, None])

    assert guided_error < baseline_error


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        (np.zeros(3), "shape"),
        (np.array([1.0, 0.2, 0.4, 0.0, 0.0]), "nonincreasing"),
        (np.array([1.0, -0.1, 0.0, 0.0, 0.0]), "within"),
        (np.array([1.0, np.nan, 0.0, 0.0, 0.0]), "finite"),
    ],
)
def test_reverse_rtc_rejects_malformed_weights(
    weights: np.ndarray, match: str
) -> None:
    with pytest.raises(RealtimeChunkingError, match=match):
        rtc_pseudoinverse_guidance_reverse(
            np.zeros((5, 2)),
            np.ones((5, 2)),
            weights,
            lambda batch, _time: np.zeros_like(batch),
        )
