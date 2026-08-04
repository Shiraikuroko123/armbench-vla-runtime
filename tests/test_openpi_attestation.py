from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

import integrations.openpi.serve_policy_attested as attested

from integrations.openpi.serve_policy_attested import (
    _command_output,
    _submodules_are_clean,
    checkpoint_content_manifest,
    public_attestation,
)


def test_checkpoint_manifest_hashes_every_file_deterministically(tmp_path) -> None:
    (tmp_path / "params").mkdir()
    (tmp_path / "params" / "a.bin").write_bytes(b"alpha")
    (tmp_path / "metadata.json").write_bytes(b"{}\n")

    first = checkpoint_content_manifest(tmp_path)
    second = checkpoint_content_manifest(tmp_path)

    assert first == second
    assert first["checkpoint_file_count"] == 2
    assert first["checkpoint_total_bytes"] == 8
    by_path = {item["path"]: item for item in first["checkpoint_files"]}
    assert by_path["params/a.bin"]["sha256"] == hashlib.sha256(b"alpha").hexdigest()
    canonical = json.dumps(
        first["checkpoint_files"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    assert first["checkpoint_content_sha256"] == hashlib.sha256(canonical).hexdigest()


def test_public_attestation_omits_local_paths_and_file_inventory() -> None:
    public = public_attestation(
        {
            "schema_version": "test",
            "checkpoint_local_path": "/private/cache",
            "checkpoint_files": [{"path": "secret"}],
            "checkpoint_content_sha256": "a" * 64,
        }
    )

    assert public == {
        "schema_version": "test",
        "checkpoint_content_sha256": "a" * 64,
    }


def test_submodule_status_rejects_uninitialized_or_modified_entries() -> None:
    assert _submodules_are_clean("")
    assert _submodules_are_clean(" abc123 submodule (heads/main)")
    assert not _submodules_are_clean("-abc123 submodule")
    assert not _submodules_are_clean("+abc123 submodule (heads/main-1-gabc123)")


def test_command_output_preserves_git_submodule_status_prefix(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        attested.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=" abc123 first\n def456 second\n"
        ),
    )

    status = _command_output(("git", "submodule", "status"), tmp_path)

    assert status == " abc123 first\n def456 second"
    assert _submodules_are_clean(status)


class _RecordingPolicy:
    def __init__(self) -> None:
        self.metadata = {"model": "pi05_libero"}
        self.calls = []

    def infer(self, observation, **kwargs):
        self.calls.append((observation, kwargs))
        return {"actions": np.zeros((10, 7), dtype=np.float32)}


def test_keyed_sampling_wrapper_passes_uncontrolled_requests_through_unchanged() -> None:
    policy = _RecordingPolicy()
    wrapper = attested.KeyedPolicySamplingWrapper(policy)
    observation = {"state": np.arange(8, dtype=np.float32)}

    response = wrapper.infer(observation)

    assert policy.calls == [(observation, {})]
    assert response.keys() == {"actions"}
    assert wrapper.metadata is policy.metadata


def test_keyed_sampling_wrapper_supplies_explicit_noise_and_audit_hashes() -> None:
    policy = _RecordingPolicy()
    wrapper = attested.KeyedPolicySamplingWrapper(policy)
    control = attested.build_policy_sampling_control(
        attested.POLICY_SAMPLING_SCORED_NAMESPACE,
        17,
        ["libero_object", 3, 5, 10],
        2,
    )
    observation = {
        "state": np.arange(8, dtype=np.float32),
        attested.POLICY_SAMPLING_REQUEST_FIELD: control,
    }

    response = wrapper.infer(observation)

    clean_observation, kwargs = policy.calls[0]
    assert clean_observation == {"state": observation["state"]}
    assert attested.POLICY_SAMPLING_REQUEST_FIELD not in clean_observation
    noise = kwargs["noise"]
    assert noise.shape == (10, 32)
    assert noise.dtype == np.dtype("<f4")
    np.testing.assert_array_equal(
        noise,
        attested.policy_sampling_noise(control["key_sha256"]),
    )
    assert response[attested.POLICY_SAMPLING_RESPONSE_FIELD] == {
        "schema_version": attested.POLICY_SAMPLING_SCHEMA_VERSION,
        "namespace": attested.POLICY_SAMPLING_SCORED_NAMESPACE,
        "key_sha256": control["key_sha256"],
        "noise_sha256": attested.policy_sampling_noise_sha256(noise),
        "generator": attested.POLICY_SAMPLING_GENERATOR,
    }
    assert attested.POLICY_SAMPLING_REQUEST_FIELD in observation


@pytest.mark.parametrize(
    "mutation",
    [
        lambda control: control.update(key_sha256="0" * 64),
        lambda control: control.update(namespace="invalid"),
        lambda control: control.update(pairing_key=None),
        lambda control: control.update(unexpected=True),
    ],
)
def test_keyed_sampling_wrapper_rejects_malformed_controls(mutation) -> None:
    policy = _RecordingPolicy()
    wrapper = attested.KeyedPolicySamplingWrapper(policy)
    control = attested.build_policy_sampling_control(
        attested.POLICY_SAMPLING_SCORED_NAMESPACE,
        17,
        ["libero_object", 3, 5, 10],
        2,
    )
    mutation(control)

    with pytest.raises(ValueError, match="policy sampling"):
        wrapper.infer({attested.POLICY_SAMPLING_REQUEST_FIELD: control})

    assert policy.calls == []


def test_sampling_keys_are_mode_independent_with_separate_warmup_namespace() -> None:
    async_key = attested.policy_sampling_key_sha256(
        attested.POLICY_SAMPLING_SCORED_NAMESPACE,
        23,
        ["libero_10", 1, 4, 5],
        8,
    )
    aligned_key = attested.policy_sampling_key_sha256(
        attested.POLICY_SAMPLING_SCORED_NAMESPACE,
        23,
        ["libero_10", 1, 4, 5],
        8,
    )
    warmup_key = attested.policy_sampling_key_sha256(
        attested.POLICY_SAMPLING_WARMUP_NAMESPACE,
        23,
        ["libero_10", 1, 4],
        8,
    )

    assert async_key == aligned_key
    assert warmup_key != async_key
    contract = attested.policy_sampling_contract()
    assert contract["mode_in_key"] is False
    assert contract["noise_shape"] == [10, 32]
