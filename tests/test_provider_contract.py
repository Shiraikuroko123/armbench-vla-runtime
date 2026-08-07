from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pytest

from armbench.mujoco_sim import MuJoCoPanda
from armbench.vla.async_worker import LatestPolicyWorker
from armbench.vla.cartesian_adapter import PandaCartesianActionAdapter
from armbench.vla.provider_contract import (
    AdaptedActionChunkPolicy,
    FrozenResponseProvider,
    FrozenResponseRecord,
    ProviderContractError,
    SemanticCompatibilityError,
    _fixture_identity,
    _fixture_observation,
    canonical_action_sha256,
    canonical_observation_sha256,
    libero_cartesian_semantics,
    require_semantic_compatibility,
    run_provider_contract_audit,
    validate_frozen_provider_bundle,
    validate_provider_contract_audit,
    write_frozen_provider_bundle,
)


def test_canonical_action_hash_is_layout_dtype_and_endian_stable() -> None:
    base = np.arange(21, dtype=np.float32).reshape(3, 7) / 10.0
    big_endian = base.astype(">f8")
    fortran = np.asfortranarray(base.astype(np.float64))

    assert canonical_action_sha256(base) == canonical_action_sha256(big_endian)
    assert canonical_action_sha256(base) == canonical_action_sha256(fortran)

    changed = base.copy()
    changed[1, 2] += 0.25
    assert canonical_action_sha256(base) != canonical_action_sha256(changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("coordinate_frame", "tool"),
        ("control_period_s", 0.1),
        ("rotation_representation", "euler_xyz_delta"),
        ("rotation_delta_scale_rad", 0.25),
        ("gripper_convention", "minus_one_closed_plus_one_open"),
        ("controller_semantics_id", "other-controller"),
    ],
)
def test_semantic_gate_rejects_same_width_but_different_meaning(
    field: str, value: object
) -> None:
    expected = libero_cartesian_semantics()
    candidate = replace(expected, **{field: value})

    with pytest.raises(SemanticCompatibilityError, match=field.split("_s")[0]):
        require_semantic_compatibility(candidate, expected)


def _write_bundle(path: Path) -> tuple[Path, object]:
    observation = _fixture_observation()
    actions = np.zeros((3, 7), dtype=float)
    actions[:, 0] = 0.05
    actions[:, 6] = -1.0
    bundle = write_frozen_provider_bundle(
        path,
        identity=_fixture_identity(),
        semantics=libero_cartesian_semantics(),
        responses=(
            FrozenResponseRecord(
                observation_sequence_id=0,
                actions=actions,
                inference_latency_ms=25.0,
                observation_sha256=canonical_observation_sha256(observation),
            ),
        ),
    )
    return bundle, observation


def test_frozen_provider_is_bound_to_observation_and_exhausts(
    tmp_path: Path,
) -> None:
    bundle, observation = _write_bundle(tmp_path / "provider")
    provider = FrozenResponseProvider.from_directory(bundle)

    chunk = provider.infer_raw(observation)

    assert chunk.actions.shape == (3, 7)
    assert not chunk.actions.flags.writeable
    assert chunk.received_at_s == pytest.approx(100.025)
    with pytest.raises(RuntimeError, match="exhausted"):
        provider.infer_raw(observation)
    provider.reset()
    wrong = replace(observation, prompt="different prompt")
    with pytest.raises(ProviderContractError, match="observation hash"):
        provider.infer_raw(wrong)


def test_manifest_detects_frozen_response_tamper(tmp_path: Path) -> None:
    bundle, _ = _write_bundle(tmp_path / "provider")
    descriptor_path = bundle / "provider.json"
    descriptor = json.loads(descriptor_path.read_text("utf-8"))
    descriptor["responses"][0]["inference_latency_ms"] = 999.0
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")

    with pytest.raises(ProviderContractError, match="manifest"):
        validate_frozen_provider_bundle(bundle)


def test_adapted_second_provider_runs_on_async_policy_worker(
    tmp_path: Path,
) -> None:
    bundle, observation = _write_bundle(tmp_path / "provider")
    policy = AdaptedActionChunkPolicy(
        FrozenResponseProvider.from_directory(bundle),
        PandaCartesianActionAdapter(MuJoCoPanda.create(obstacles=())),
    )
    worker = LatestPolicyWorker(policy)
    try:
        worker.submit(observation)
        deadline = time.monotonic() + 2.0
        outcomes = ()
        while not outcomes and time.monotonic() < deadline:
            outcomes = worker.drain()
            if not outcomes:
                time.sleep(0.005)
    finally:
        assert worker.close()

    assert len(outcomes) == 1
    assert outcomes[0].succeeded
    assert outcomes[0].chunk is not None
    assert outcomes[0].chunk.actions.shape == (3, 8)
    assert "openvla_oft_libero_contract_fixture" in outcomes[0].chunk.source
    assert policy.provenance["checkpoint_executed_this_run"] is False


def test_provider_audit_is_self_validating_and_claim_bounded(
    tmp_path: Path,
) -> None:
    output = run_provider_contract_audit(tmp_path / "audit")

    result = validate_provider_contract_audit(output)
    summary = json.loads((output / "summary.json").read_text("utf-8"))

    assert result["valid"]
    assert result["model_family"] == "OpenVLA-OFT"
    assert result["semantic_mismatch_rejections"] == 5
    assert result["adapted_shape"] == [6, 8]
    assert not any(summary["claims"].values())
