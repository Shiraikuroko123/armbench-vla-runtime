"""Policy clients that produce validated OpenPI-compatible action chunks."""

from __future__ import annotations

from collections.abc import Sequence
from time import monotonic
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike

from armbench.vla.types import (
    ActionChunk,
    PI05_DROID_ACTION_HORIZON,
    VLAObservation,
)


class PolicyBackend(Protocol):
    def infer(self, observation: dict[str, object]) -> dict[str, object]: ...

    def get_server_metadata(self) -> dict[str, object]: ...


class ActionChunkPolicy(Protocol):
    def infer(self, observation: VLAObservation) -> ActionChunk: ...


class OpenPIPolicyClient:
    """Thin wrapper around Physical Intelligence's official WebSocket client."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        *,
        api_key: str | None = None,
        expected_horizon: int = PI05_DROID_ACTION_HORIZON,
        backend: PolicyBackend | None = None,
    ) -> None:
        if expected_horizon <= 0:
            raise ValueError("expected_horizon must be positive")
        if backend is None:
            try:
                from openpi_client import websocket_client_policy
            except ImportError as error:
                raise RuntimeError(
                    "OpenPI client is not installed; install ArmBench with the "
                    "'vla' optional dependency"
                ) from error
            backend = websocket_client_policy.WebsocketClientPolicy(
                host=host, port=port, api_key=api_key
            )
        self._backend = backend
        self.expected_horizon = int(expected_horizon)

    @property
    def server_metadata(self) -> dict[str, object]:
        return dict(self._backend.get_server_metadata())

    def infer(self, observation: VLAObservation) -> ActionChunk:
        started = monotonic()
        result = self._backend.infer(observation.to_openpi_droid())
        received = monotonic()
        if "actions" not in result:
            raise ValueError("OpenPI response does not contain 'actions'")
        actions = np.asarray(result["actions"], dtype=float)
        if actions.shape != (self.expected_horizon, 8):
            raise ValueError(
                "OpenPI action shape mismatch: expected "
                f"({self.expected_horizon}, 8), got {actions.shape}"
            )
        return ActionChunk(
            actions=actions,
            source="openpi_remote",
            observation_sequence_id=observation.sequence_id,
            inference_latency_ms=(received - started) * 1000.0,
            received_at_s=received,
            server_timing=dict(result.get("server_timing", {})),
        )


class ScriptedActionChunkPolicy:
    """Deterministic non-learned source for local guard tests and fault injection."""

    def __init__(
        self,
        chunks: Sequence[ArrayLike],
        *,
        latencies_ms: Sequence[float] | None = None,
        repeat_last: bool = False,
    ) -> None:
        if not chunks:
            raise ValueError("at least one scripted chunk is required")
        self._chunks = [np.asarray(chunk, dtype=float).copy() for chunk in chunks]
        if latencies_ms is None:
            latencies_ms = [0.0] * len(chunks)
        if len(latencies_ms) != len(chunks):
            raise ValueError("latencies must match scripted chunks")
        self._latencies = [float(value) for value in latencies_ms]
        self._repeat_last = bool(repeat_last)
        self._index = 0

    def infer(self, observation: VLAObservation) -> ActionChunk:
        if self._index >= len(self._chunks) and not self._repeat_last:
            raise RuntimeError(
                "scripted action chunks exhausted; reset the policy or opt in "
                "to repeat_last"
            )
        index = min(self._index, len(self._chunks) - 1)
        self._index += 1
        return ActionChunk(
            actions=self._chunks[index],
            source="scripted_non_learned",
            observation_sequence_id=observation.sequence_id,
            inference_latency_ms=self._latencies[index],
            received_at_s=observation.captured_at_s
            + self._latencies[index] / 1000.0,
        )

    def reset(self) -> None:
        self._index = 0
