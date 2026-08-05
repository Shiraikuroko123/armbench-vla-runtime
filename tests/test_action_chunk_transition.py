from __future__ import annotations

import numpy as np
import pytest

from integrations.openpi.action_chunk_transition import (
    ARCHIVE_SCHEMA_VERSION,
    SOURCE_NEW_SUFFIX,
    SOURCE_OLD_PREFIX,
    ActionChunkTransitionError,
    build_action_chunk_transition,
    canonical_action_sha256,
    load_transition_archive,
    validate_transition_arrays,
    write_transition_archive,
)


def _next_reference(response: np.ndarray) -> np.ndarray:
    return np.concatenate((response[5:], np.zeros((5, 7), dtype=np.float32)))


def _fixture():
    first_response = np.arange(70, dtype=np.float32).reshape(10, 7)
    first_next = _next_reference(first_response)
    first = build_action_chunk_transition(
        None,
        first_response,
        first_next,
        inference_delay=4,
        execute_horizon=5,
        executed_old=0,
        executed_new=5,
    )
    second_response = first_response + 100.0
    second_next = _next_reference(second_response)
    second = build_action_chunk_transition(
        first_next,
        second_response,
        second_next,
        inference_delay=4,
        execute_horizon=5,
        executed_old=4,
        executed_new=1,
    )
    descriptor = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "scheduler": {
            "action_horizon": 10,
            "action_dim": 7,
            "execute_horizon": 5,
            "inference_delay": 4,
        },
        "rows": [
            {"episode_id": "e", "pair_id": "p", "method": "projected_overlap", "query_index": 0},
            {"episode_id": "e", "pair_id": "p", "method": "projected_overlap", "query_index": 1},
        ],
    }
    queries = [
        {
            **descriptor["rows"][0],
            "bootstrap": True,
            "executed_steps": 5,
            "old_prefix_steps": 0,
            "new_suffix_steps": 5,
            "response_action_sha256": canonical_action_sha256(first_response),
            "next_reference_sha256": canonical_action_sha256(first_next),
        },
        {
            **descriptor["rows"][1],
            "bootstrap": False,
            "executed_steps": 5,
            "old_prefix_steps": 4,
            "new_suffix_steps": 1,
            "response_action_sha256": canonical_action_sha256(second_response),
            "next_reference_sha256": canonical_action_sha256(second_next),
        },
    ]
    return (first, second), descriptor, queries


def test_transition_archive_recomputes_overlap_and_reference_chain(tmp_path) -> None:
    transitions, descriptor, queries = _fixture()
    path = tmp_path / "transitions.npz"
    write_transition_archive(path, transitions)

    arrays = load_transition_archive(path)
    report = validate_transition_arrays(descriptor, arrays, queries)

    assert report["valid"] is True
    assert report["transition_count"] == 2
    np.testing.assert_array_equal(
        arrays["action_source"][1],
        [SOURCE_OLD_PREFIX] * 4 + [SOURCE_NEW_SUFFIX],
    )


def test_transition_validator_rejects_resigned_action_tampering(tmp_path) -> None:
    transitions, descriptor, queries = _fixture()
    path = tmp_path / "transitions.npz"
    write_transition_archive(path, transitions)
    arrays = load_transition_archive(path)
    arrays["executed_window"][1, 4, 0] += 1.0

    with pytest.raises(ActionChunkTransitionError, match="executed window"):
        validate_transition_arrays(descriptor, arrays, queries)


def test_partial_old_prefix_execution_is_explicitly_padded() -> None:
    response = np.ones((10, 7), dtype=np.float32)
    previous = np.full((10, 7), 2.0, dtype=np.float32)
    transition = build_action_chunk_transition(
        previous,
        response,
        _next_reference(response),
        inference_delay=4,
        execute_horizon=5,
        executed_old=2,
        executed_new=0,
    )

    assert transition.executed_length == 2
    np.testing.assert_array_equal(transition.executed_window[:2], previous[:2])
    np.testing.assert_array_equal(transition.executed_window[2:], 0.0)
    np.testing.assert_array_equal(
        transition.action_source, [SOURCE_OLD_PREFIX, SOURCE_OLD_PREFIX, 0, 0, 0]
    )


def test_transition_rejects_new_suffix_before_old_prefix_finishes() -> None:
    response = np.ones((10, 7), dtype=np.float32)
    with pytest.raises(ActionChunkTransitionError, match="before the old prefix"):
        build_action_chunk_transition(
            np.zeros_like(response),
            response,
            _next_reference(response),
            inference_delay=4,
            execute_horizon=5,
            executed_old=2,
            executed_new=1,
        )


def test_transition_validator_accepts_reference_only_bootstrap_seed(tmp_path) -> None:
    bootstrap = np.arange(70, dtype=np.float32).reshape(10, 7)
    response = bootstrap + 100.0
    transition = build_action_chunk_transition(
        bootstrap,
        response,
        _next_reference(response),
        inference_delay=4,
        execute_horizon=5,
        executed_old=4,
        executed_new=1,
    )
    descriptor = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "scheduler": {
            "action_horizon": 10,
            "action_dim": 7,
            "execute_horizon": 5,
            "inference_delay": 4,
        },
        "rows": [
            {
                "episode_id": "e",
                "pair_id": "p",
                "method": "rtc_guided_overlap",
                "query_index": 1,
            }
        ],
    }
    query = {
        **descriptor["rows"][0],
        "bootstrap": False,
        "executed_steps": 5,
        "old_prefix_steps": 4,
        "new_suffix_steps": 1,
        "response_action_sha256": canonical_action_sha256(response),
        "next_reference_sha256": canonical_action_sha256(_next_reference(response)),
    }
    path = tmp_path / "transitions.npz"
    write_transition_archive(path, [transition])
    arrays = load_transition_archive(path)

    report = validate_transition_arrays(
        descriptor,
        arrays,
        [query],
        bootstrap_response_sha256_by_episode={
            "e": canonical_action_sha256(bootstrap)
        },
    )

    assert report["valid"] is True
    with pytest.raises(ActionChunkTransitionError, match="reference-only bootstrap"):
        validate_transition_arrays(descriptor, arrays, [query])
