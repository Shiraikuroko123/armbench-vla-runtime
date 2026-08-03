"""Local OpenPI-compatible protocol diagnostic with a non-learned policy."""

from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Lock, Thread
from time import perf_counter
from typing import Any

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from armbench.mujoco_sim.model import MuJoCoPanda
from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.vla.benchmark import (
    _guard_config,
    _integrate_actions,
    _safe_stream,
    _write_json,
    load_vla_config,
)
from armbench.vla.online_benchmark import execute_openpi_online_run

LOOPBACK_POLICY_PROVENANCE = "scripted_non_learned_loopback"
_DROID_KEYS = {
    "observation/exterior_image_1_left",
    "observation/wrist_image_left",
    "observation/joint_position",
    "observation/gripper_position",
    "prompt",
}


class OpenPIProtocolLoopbackServer:
    """Serve deterministic reference actions through the real wire protocol."""

    def __init__(
        self,
        reference_positions: np.ndarray,
        *,
        action_dt_s: float,
        action_horizon: int = 15,
        velocity_limit_rad_s: float = 1.0,
    ) -> None:
        references = np.asarray(reference_positions, dtype=float)
        if (
            references.ndim != 2
            or references.shape[1] != 7
            or len(references) < 2
            or not np.all(np.isfinite(references))
        ):
            raise ValueError("loopback references must be a finite Nx7 path")
        timing = np.asarray(
            [action_dt_s, velocity_limit_rad_s], dtype=float
        )
        if (
            not np.all(np.isfinite(timing))
            or np.any(timing <= 0.0)
            or action_horizon <= 0
        ):
            raise ValueError("loopback action timing and limits must be positive")
        self.reference_positions = references.copy()
        self.action_dt_s = float(action_dt_s)
        self.action_horizon = int(action_horizon)
        self.velocity_limit_rad_s = float(velocity_limit_rad_s)
        self.metadata = {
            "armbench_loopback": True,
            "checkpoint_identity_verified": False,
            "model_config": "armbench_scripted_droid_loopback",
            "policy_source": LOOPBACK_POLICY_PROVENANCE,
        }
        self._requests: list[dict[str, object]] = []
        self._request_lock = Lock()
        self._server: Any | None = None
        self._thread: Thread | None = None
        self.port: int | None = None

    @property
    def request_audit(self) -> list[dict[str, object]]:
        with self._request_lock:
            return [dict(item) for item in self._requests]

    def _actions(self, q_observed: np.ndarray, gripper: float) -> np.ndarray:
        distances = np.max(
            np.abs(self.reference_positions - q_observed[None, :]), axis=1
        )
        cursor = int(np.argmin(distances))
        predicted_q = q_observed.copy()
        actions = np.zeros((self.action_horizon, 8), dtype=float)
        for index in range(self.action_horizon):
            target_index = min(
                cursor + index + 1, len(self.reference_positions) - 1
            )
            target = self.reference_positions[target_index]
            velocity = np.clip(
                (target - predicted_q) / self.action_dt_s,
                -self.velocity_limit_rad_s,
                self.velocity_limit_rad_s,
            )
            actions[index, :7] = velocity
            actions[index, 7] = gripper
            predicted_q = predicted_q + self.action_dt_s * velocity
        return actions

    @staticmethod
    def _validated_request(request: object) -> dict[str, object]:
        if not isinstance(request, dict) or set(request) != _DROID_KEYS:
            raise ValueError("loopback request does not match DROID keys")
        q = np.asarray(request["observation/joint_position"], dtype=float)
        gripper = np.asarray(
            request["observation/gripper_position"], dtype=float
        )
        exterior = np.asarray(request["observation/exterior_image_1_left"])
        wrist = np.asarray(request["observation/wrist_image_left"])
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("loopback joint observation must be finite length 7")
        if (
            gripper.shape != (1,)
            or not np.all(np.isfinite(gripper))
            or not 0.0 <= float(gripper[0]) <= 1.0
        ):
            raise ValueError("loopback gripper observation is invalid")
        for label, image in (("exterior", exterior), ("wrist", wrist)):
            if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                raise ValueError(f"loopback {label} image contract is invalid")
        prompt = request["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("loopback prompt must be nonempty")
        return {
            "q": q,
            "gripper": float(gripper[0]),
            "exterior": exterior,
            "wrist": wrist,
            "prompt": prompt,
        }

    def _handler(self, websocket: Any) -> None:
        from openpi_client import msgpack_numpy

        websocket.send(msgpack_numpy.packb(self.metadata))
        while True:
            try:
                packed_request = websocket.recv()
            except ConnectionClosed:
                return
            started = perf_counter()
            try:
                request = self._validated_request(
                    msgpack_numpy.unpackb(packed_request)
                )
                actions = self._actions(
                    np.asarray(request["q"], dtype=float),
                    float(request["gripper"]),
                )
                audit = {
                    "request_index": len(self.request_audit),
                    "prompt": request["prompt"],
                    "joint_position": np.asarray(request["q"]).tolist(),
                    "exterior_image_sha256": hashlib.sha256(
                        np.asarray(request["exterior"]).tobytes(order="C")
                    ).hexdigest(),
                    "wrist_image_sha256": hashlib.sha256(
                        np.asarray(request["wrist"]).tobytes(order="C")
                    ).hexdigest(),
                }
                with self._request_lock:
                    self._requests.append(audit)
                elapsed_ms = (perf_counter() - started) * 1000.0
                websocket.send(
                    msgpack_numpy.packb(
                        {
                            "actions": actions,
                            "server_timing": {
                                "loopback_policy_ms": elapsed_ms
                            },
                        }
                    )
                )
            except Exception as error:
                websocket.send(f"ArmBench loopback error: {error}")
                return

    def __enter__(self) -> "OpenPIProtocolLoopbackServer":
        self._server = serve(
            self._handler,
            "127.0.0.1",
            0,
            compression=None,
            max_size=None,
        )
        self.port = int(self._server.socket.getsockname()[1])
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.shutdown()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                raise RuntimeError("loopback server thread did not stop")


def execute_openpi_loopback_run(
    config_path: Path,
    output_directory: Path,
    *,
    scenario_name: str = "single_block",
    execution_horizon: int = 5,
    payload_mass: float = 0.0,
    max_policy_queries: int = 3,
    prompt: str | None = None,
    make_video: bool = False,
) -> Path:
    """Exercise the complete remote-policy loop without a learned checkpoint."""

    config = load_vla_config(config_path)
    scenario = mujoco_scenarios().get(scenario_name)
    if scenario is None:
        raise ValueError(f"unknown loopback scenario: {scenario_name}")
    guard_config = _guard_config(config)
    reference_robot = MuJoCoPanda.create(obstacles=scenario.obstacles)
    references = _integrate_actions(
        reference_robot,
        scenario.start,
        _safe_stream(scenario_name, config, guard_config),
        guard_config,
    )
    expected_horizon = int(dict(config["openpi_contract"])["action_horizon"])
    with OpenPIProtocolLoopbackServer(
        references,
        action_dt_s=guard_config.control_dt_s,
        action_horizon=expected_horizon,
        velocity_limit_rad_s=guard_config.joint_velocity_clip_rad_s,
    ) as server:
        if server.port is None:
            raise RuntimeError("loopback server did not bind a port")
        result = execute_openpi_online_run(
            config_path,
            output_directory,
            host="127.0.0.1",
            port=server.port,
            scenario_name=scenario_name,
            execution_horizon=execution_horizon,
            payload_mass=payload_mass,
            max_policy_queries=max_policy_queries,
            prompt=prompt,
            connect_timeout_s=1.0,
            inference_timeout_s=1.0,
            make_video=make_video,
            policy_provenance=LOOPBACK_POLICY_PROVENANCE,
        )
        request_audit = server.request_audit
    _write_json(
        result / "loopback_server.json",
        {
            "checkpoint_identity_verified": False,
            "policy_provenance": LOOPBACK_POLICY_PROVENANCE,
            "request_count": len(request_audit),
            "requests": request_audit,
        },
    )
    return result
