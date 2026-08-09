"""Attested live OpenPI providers for provider-neutral Panda execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import time

import numpy as np

from armbench.vla.cartesian_adapter import (
    LIBERO_ACTION_DIM,
    CartesianAdapterConfig,
    PandaCartesianActionAdapter,
    PandaCartesianKinematics,
)
from armbench.vla.policy import PolicyBackend
from armbench.vla.provider_contract import (
    ActionSemantics,
    ProviderContractError,
    ProviderIdentity,
    RawActionChunkProvider,
    RawActionChunk,
    canonical_action_sha256,
    libero_cartesian_semantics,
    require_semantic_compatibility,
)
from armbench.vla.types import ActionChunk, VLAObservation


PI05_LIBERO_POLICY_CONFIG = "pi05_libero"
PI05_LIBERO_ACTION_HORIZON = 10
OPENPI_REPOSITORY = "https://github.com/Physical-Intelligence/openpi"
ATTESTATION_METADATA_KEY = "armbench_server_attestation"


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProviderContractError(f"{label} must be a nonempty string")
    return value


def _required_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ProviderContractError(f"{label} must be a boolean")
    return value


def provider_identity_from_openpi_metadata(
    metadata: Mapping[str, object],
    *,
    provider_id: str = "openpi_pi05_libero_live",
    model_family: str = "Physical Intelligence pi0.5",
) -> ProviderIdentity:
    """Create a live identity only from the strict ArmBench attestation."""

    attestation = metadata.get(ATTESTATION_METADATA_KEY)
    if not isinstance(attestation, Mapping):
        raise ProviderContractError(
            "OpenPI metadata does not contain ArmBench server attestation"
        )
    if not _required_bool(attestation.get("policy_loaded"), "policy_loaded"):
        raise ProviderContractError("attested OpenPI policy is not loaded")
    policy_config = _required_string(
        attestation.get("policy_config"), "policy_config"
    )
    if policy_config != PI05_LIBERO_POLICY_CONFIG:
        raise ProviderContractError(
            f"expected {PI05_LIBERO_POLICY_CONFIG}, got {policy_config}"
        )
    horizon = attestation.get("action_horizon")
    if type(horizon) is not int or horizon != PI05_LIBERO_ACTION_HORIZON:
        raise ProviderContractError(
            "attested pi0.5 LIBERO action horizon must equal 10"
        )
    revision = _required_string(attestation.get("openpi_commit"), "openpi_commit")
    checkpoint_reference = _required_string(
        attestation.get("checkpoint_uri"), "checkpoint_uri"
    )
    checkpoint_sha256 = _required_string(
        attestation.get("checkpoint_content_sha256"),
        "checkpoint_content_sha256",
    )
    return ProviderIdentity(
        provider_id=provider_id,
        model_family=model_family,
        implementation_repository=OPENPI_REPOSITORY,
        implementation_revision=revision,
        checkpoint_reference=checkpoint_reference,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_identity_status="content_attested",
        response_origin="live_checkpoint_inference",
        checkpoint_executed_during_capture=False,
        checkpoint_executed_this_run=True,
    )


class OpenPILiberoRawProvider:
    """Run an attested live ``pi05_libero`` backend and emit native Hx7 chunks."""

    def __init__(
        self,
        backend: PolicyBackend,
        *,
        expected_horizon: int = PI05_LIBERO_ACTION_HORIZON,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(expected_horizon) is not int or expected_horizon <= 0:
            raise ValueError("expected_horizon must be a positive integer")
        self._backend = backend
        self._identity = provider_identity_from_openpi_metadata(
            backend.get_server_metadata()
        )
        self._semantics = libero_cartesian_semantics()
        self._expected_horizon = expected_horizon
        self._clock = clock

    @property
    def identity(self) -> ProviderIdentity:
        return self._identity

    @property
    def semantics(self) -> ActionSemantics:
        return self._semantics

    def infer_raw(self, observation: VLAObservation) -> RawActionChunk:
        started_at_s = float(self._clock())
        if not np.isfinite(started_at_s):
            raise ProviderContractError("provider clock returned a non-finite value")
        response = self._backend.infer(observation.to_openpi_libero())
        received_at_s = float(self._clock())
        if not np.isfinite(received_at_s) or received_at_s < started_at_s:
            raise ProviderContractError("provider clock moved backwards")
        if "actions" not in response:
            raise ProviderContractError("OpenPI response does not contain actions")
        try:
            actions = np.asarray(response["actions"], dtype=float)
        except (TypeError, ValueError) as error:
            raise ProviderContractError("OpenPI actions are not numeric") from error
        expected_shape = (self._expected_horizon, LIBERO_ACTION_DIM)
        if actions.shape != expected_shape or not np.all(np.isfinite(actions)):
            raise ProviderContractError(
                f"OpenPI actions must be finite with shape {expected_shape}"
            )
        response_sha256 = canonical_action_sha256(actions)
        return RawActionChunk(
            actions=actions,
            semantics=self.semantics,
            source=f"live:{self.identity.provider_id}:{response_sha256[:12]}",
            observation_sequence_id=observation.sequence_id,
            inference_latency_ms=(received_at_s - started_at_s) * 1000.0,
            received_at_s=received_at_s,
            response_sha256=response_sha256,
        )

    def close(self) -> None:
        close = getattr(self._backend, "close", None)
        if callable(close):
            close()


class OpenPILiberoPandaPolicy:
    """Adapt an attested live ``pi05_libero`` provider to Panda Hx8 actions.

    OpenPI's LIBERO policy emits seven normalized Cartesian/axis-angle values
    (Hx7).  The Panda runtime consumes seven joint velocities plus a gripper
    position (Hx8).  This wrapper keeps that conversion at an explicit,
    auditable boundary and reports the full provider-plus-adapter latency to
    the deadline-aware worker.
    """

    def __init__(
        self,
        provider: RawActionChunkProvider,
        robot: PandaCartesianKinematics,
        *,
        adapter_config: CartesianAdapterConfig = CartesianAdapterConfig(),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(getattr(provider, "infer_raw", None)):
            raise TypeError("provider must provide infer_raw(observation)")
        if not hasattr(provider, "identity") or not hasattr(provider, "semantics"):
            raise TypeError("provider must expose identity and semantics")
        identity = provider.identity
        if not isinstance(identity, ProviderIdentity):
            raise TypeError("provider.identity must be a ProviderIdentity")
        if identity.response_origin != "live_checkpoint_inference":
            raise ProviderContractError(
                "OpenPILiberoPandaPolicy requires a live checkpoint provider"
            )
        self.provider = provider
        self.adapter = PandaCartesianActionAdapter(robot, adapter_config)
        self.expected_semantics = libero_cartesian_semantics(adapter_config)
        require_semantic_compatibility(provider.semantics, self.expected_semantics)
        self._clock = clock
        self._last_metrics: dict[str, object] = {}

    @property
    def identity(self) -> ProviderIdentity:
        return self.provider.identity

    @property
    def policy_source(self) -> str:
        return (
            f"live:{self.identity.provider_id}"
            "|libero_hx7_to_panda_hx8_cartesian_adapter"
        )

    @property
    def provenance(self) -> dict[str, object]:
        """Expose immutable provider/semantic identity to episode artifacts."""

        return {
            "identity": self.identity.to_dict(),
            "source_semantics": self.provider.semantics.to_dict(),
            "source_semantics_sha256": self.provider.semantics.semantic_sha256,
            "adapter": "PandaCartesianActionAdapter",
            "adapter_contract": "libero.ee_delta_pose_gripper.v1->panda.droid_hx8.v1",
            "checkpoint_executed_this_run": (
                self.identity.checkpoint_executed_this_run
            ),
        }

    def infer(self, observation: VLAObservation) -> ActionChunk:
        started_at_s = float(self._clock())
        if not np.isfinite(started_at_s):
            raise ProviderContractError("policy clock returned a non-finite value")
        raw = self.provider.infer_raw(observation)
        require_semantic_compatibility(raw.semantics, self.expected_semantics)
        if raw.observation_sequence_id != observation.sequence_id:
            raise ProviderContractError(
                "provider response sequence does not match observation"
            )
        adapted = self.adapter.adapt(
            raw.actions,
            observation.joint_position,
            source=raw.source,
            observation_sequence_id=raw.observation_sequence_id,
            inference_latency_ms=raw.inference_latency_ms,
            received_at_s=raw.received_at_s,
        )
        finished_at_s = float(self._clock())
        if not np.isfinite(finished_at_s) or finished_at_s < started_at_s:
            raise ProviderContractError("policy clock moved backwards")
        total_latency_ms = (finished_at_s - started_at_s) * 1000.0
        self._last_metrics = {
            "provider_id": self.identity.provider_id,
            "implementation_revision": self.identity.implementation_revision,
            "checkpoint_reference": self.identity.checkpoint_reference,
            "checkpoint_sha256": self.identity.checkpoint_sha256,
            "response_sha256": raw.response_sha256,
            "provider_inference_latency_ms": raw.inference_latency_ms,
            "adapter_latency_ms": adapted.adapter_latency_ms,
            "total_latency_ms": total_latency_ms,
            "adapter": adapted.metrics(),
        }
        timing = {
            "provider_inference_latency_ms": float(raw.inference_latency_ms),
            "adapter_latency_ms": float(adapted.adapter_latency_ms),
            "total_latency_ms": float(total_latency_ms),
        }
        return ActionChunk(
            actions=adapted.chunk.actions,
            source=f"{self.policy_source}|response_sha256:{raw.response_sha256}",
            observation_sequence_id=observation.sequence_id,
            inference_latency_ms=total_latency_ms,
            received_at_s=finished_at_s,
            server_timing=timing,
        )

    def metrics(self) -> dict[str, object]:
        """Return the most recent response provenance without mutable arrays."""

        return dict(self._last_metrics)

    def close(self) -> None:
        self.provider.close()
