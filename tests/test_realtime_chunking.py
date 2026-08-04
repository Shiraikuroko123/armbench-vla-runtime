from __future__ import annotations

import numpy as np
import pytest

from integrations.openpi.realtime_chunking import (
    RealtimeChunkingError,
    build_committed_action_condition,
    build_overlap_plan,
    prefix_attention_weights,
    projected_flow_inpainting,
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
