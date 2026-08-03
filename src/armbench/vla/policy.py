"""Policy clients that produce validated OpenPI-compatible action chunks."""

from __future__ import annotations

from collections.abc import Sequence
import threading
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


class BoundedOpenPIBackend:
    """OpenPI wire-compatible WebSocket transport with bounded blocking calls."""

    def __init__(
        self,
        host: str,
        port: int | None,
        *,
        api_key: str | None,
        connect_timeout_s: float,
        inference_timeout_s: float,
    ) -> None:
        if not host.strip():
            raise ValueError("OpenPI host must be nonempty")
        if port is not None and not 0 < port <= 65535:
            raise ValueError("OpenPI port must be within [1, 65535]")
        if (
            not np.isfinite(connect_timeout_s)
            or not np.isfinite(inference_timeout_s)
            or connect_timeout_s <= 0.0
            or inference_timeout_s <= 0.0
        ):
            raise ValueError("OpenPI transport timeouts must be finite and positive")
        try:
            from openpi_client import msgpack_numpy
            from websockets.sync.client import connect
        except ImportError as error:
            raise RuntimeError(
                "OpenPI client is not installed; install ArmBench with the "
                "'vla' optional dependency"
            ) from error
        uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}"
        if port is not None:
            uri = f"{uri}:{port}"
        headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
        self._inference_timeout_s = float(inference_timeout_s)
        self._packer = msgpack_numpy.Packer()
        self._unpack = msgpack_numpy.unpackb
        self._lock = threading.Lock()
        self._connection: Any | None = None
        try:
            self._connection = connect(
                uri,
                compression=None,
                max_size=None,
                additional_headers=headers,
                open_timeout=float(connect_timeout_s),
                close_timeout=min(float(connect_timeout_s), 0.25),
            )
            packed_metadata = self._connection.recv(
                timeout=float(connect_timeout_s)
            )
            metadata = self._unpack(packed_metadata)
            if not isinstance(metadata, dict):
                raise ValueError("OpenPI server metadata must be a mapping")
            self._server_metadata = dict(metadata)
        except Exception:
            self.close()
            raise

    def get_server_metadata(self) -> dict[str, object]:
        return dict(self._server_metadata)

    def infer(self, observation: dict[str, object]) -> dict[str, object]:
        with self._lock:
            connection = self._connection
            if connection is None:
                raise ConnectionError("OpenPI WebSocket connection is closed")
            try:
                connection.send(self._packer.pack(observation))
                packed_response = connection.recv(
                    timeout=self._inference_timeout_s
                )
                if isinstance(packed_response, str):
                    raise RuntimeError(
                        f"OpenPI inference server error: {packed_response}"
                    )
                response = self._unpack(packed_response)
                if not isinstance(response, dict):
                    raise ValueError("OpenPI response must be a mapping")
                return dict(response)
            except Exception:
                self.close()
                raise

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


class OpenPIPolicyClient:
    """Validated DROID policy wrapper with bounded OpenPI transport by default."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        *,
        api_key: str | None = None,
        expected_horizon: int = PI05_DROID_ACTION_HORIZON,
        connect_timeout_s: float = 3.0,
        inference_timeout_s: float = 1.0,
        backend: PolicyBackend | None = None,
    ) -> None:
        if expected_horizon <= 0:
            raise ValueError("expected_horizon must be positive")
        if backend is None:
            backend = BoundedOpenPIBackend(
                host,
                port,
                api_key=api_key,
                connect_timeout_s=connect_timeout_s,
                inference_timeout_s=inference_timeout_s,
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

    def close(self) -> None:
        close = getattr(self._backend, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "OpenPIPolicyClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


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
        if any(
            chunk.ndim != 2
            or chunk.shape[1] != 8
            or len(chunk) == 0
            or not np.all(np.isfinite(chunk))
            for chunk in self._chunks
        ):
            raise ValueError("scripted chunks must be finite nonempty Nx8 arrays")
        if latencies_ms is None:
            latencies_ms = [0.0] * len(chunks)
        if len(latencies_ms) != len(chunks):
            raise ValueError("latencies must match scripted chunks")
        self._latencies = [float(value) for value in latencies_ms]
        if any(
            not np.isfinite(value) or value < 0.0 for value in self._latencies
        ):
            raise ValueError("scripted latencies must be finite and nonnegative")
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
