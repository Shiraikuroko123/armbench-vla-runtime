from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from armbench.vla.openpi_provider import (
    OpenPILiberoRawProvider,
    OpenPILiberoPandaPolicy,
    provider_identity_from_openpi_metadata,
)
from armbench.vla.provider_contract import ProviderContractError
from armbench.mujoco_sim.model import MuJoCoPanda
from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.vla.types import VLAObservation


OPENPI_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
CHECKPOINT_SHA256 = "9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5"


def _metadata() -> dict[str, object]:
    return {
        "armbench_server_attestation": {
            "policy_loaded": True,
            "policy_config": "pi05_libero",
            "action_horizon": 10,
            "openpi_commit": OPENPI_COMMIT,
            "checkpoint_uri": "gs://openpi-assets/checkpoints/pi05_libero",
            "checkpoint_content_sha256": CHECKPOINT_SHA256,
        }
    }


def _observation(sequence_id: int = 7) -> VLAObservation:
    return VLAObservation(
        exterior_image=np.zeros((224, 224, 3), dtype=np.uint8),
        wrist_image=np.ones((224, 224, 3), dtype=np.uint8),
        joint_position=np.linspace(-0.3, 0.3, 7),
        gripper_position=np.array([0.6]),
        prompt="pick up the block",
        sequence_id=sequence_id,
        captured_at_s=100.0,
    )


class _Backend:
    def __init__(self, actions: object) -> None:
        self.actions = actions
        self.request: dict[str, object] | None = None
        self.closed = False

    def get_server_metadata(self) -> dict[str, object]:
        return _metadata()

    def infer(self, observation: dict[str, object]) -> dict[str, object]:
        self.request = observation
        return {"actions": self.actions}

    def close(self) -> None:
        self.closed = True


def test_attested_live_provider_builds_libero_request_and_chunk() -> None:
    actions = np.linspace(-1.0, 1.0, 70).reshape(10, 7)
    backend = _Backend(actions)
    ticks = iter((100.01, 100.09))
    provider = OpenPILiberoRawProvider(backend, clock=lambda: next(ticks))

    chunk = provider.infer_raw(_observation())

    assert provider.identity.checkpoint_executed_this_run
    assert provider.identity.checkpoint_sha256 == CHECKPOINT_SHA256
    assert chunk.actions.shape == (10, 7)
    assert chunk.observation_sequence_id == 7
    assert chunk.inference_latency_ms == pytest.approx(80.0)
    assert backend.request is not None
    assert set(backend.request) == {
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
    }
    assert np.asarray(backend.request["observation/state"]).shape == (8,)
    provider.close()
    assert backend.closed


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("policy_loaded", False, "not loaded"),
        ("policy_config", "pi0_droid", "expected pi05_libero"),
        ("action_horizon", 15, "horizon"),
        ("openpi_commit", "not-a-commit", "identity"),
        ("checkpoint_content_sha256", "not-a-sha", "SHA-256"),
    ],
)
def test_live_identity_rejects_invalid_attestation(
    field: str, value: object, match: str
) -> None:
    metadata = _metadata()
    metadata["armbench_server_attestation"][field] = value
    with pytest.raises(ProviderContractError, match=match):
        provider_identity_from_openpi_metadata(metadata)


def test_live_identity_requires_attestation() -> None:
    with pytest.raises(ProviderContractError, match="attestation"):
        provider_identity_from_openpi_metadata({})


def test_live_provider_rejects_invalid_response_shape() -> None:
    provider = OpenPILiberoRawProvider(
        _Backend(np.zeros((9, 7))), clock=iter((1.0, 1.1)).__next__
    )
    with pytest.raises(ProviderContractError, match="shape"):
        provider.infer_raw(_observation())


def test_non_live_identity_cannot_claim_current_run_execution() -> None:
    identity = provider_identity_from_openpi_metadata(_metadata())
    with pytest.raises(ProviderContractError, match="non-live"):
        replace(identity, response_origin="attested_archive")


def test_live_libero_provider_adapts_to_panda_hx8_with_provenance() -> None:
    actions = np.zeros((10, 7), dtype=float)
    actions[:, 0] = 0.2
    provider = OpenPILiberoRawProvider(_Backend(actions))
    robot = MuJoCoPanda.create(obstacles=())
    policy = OpenPILiberoPandaPolicy(provider, robot)

    observation = _observation(sequence_id=3)
    observation = VLAObservation(
        exterior_image=observation.exterior_image,
        wrist_image=observation.wrist_image,
        joint_position=mujoco_scenarios()["free_space"].start,
        gripper_position=observation.gripper_position,
        prompt=observation.prompt,
        sequence_id=observation.sequence_id,
        captured_at_s=observation.captured_at_s,
    )
    chunk = policy.infer(observation)

    assert chunk.actions.shape == (10, 8)
    assert chunk.observation_sequence_id == 3
    assert chunk.source.startswith("live:openpi_pi05_libero_live")
    assert chunk.server_timing["total_latency_ms"] == pytest.approx(
        chunk.inference_latency_ms
    )
    metrics = policy.metrics()
    assert metrics["response_sha256"]
    assert chunk.source.endswith(str(metrics["response_sha256"]))
    assert metrics["adapter"]["action_space_id"] == "libero.ee_delta_pose_gripper.v1"
    policy.close()
