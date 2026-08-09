#!/usr/bin/env python3
"""Run one attested pi0.5-LIBERO -> Panda runtime smoke.

This is an integration smoke, not a LIBERO task benchmark.  It deliberately
keeps the policy checkpoint on the remote OpenPI server and records the
provider identity, response digest, adapter cost, worker timing, and MuJoCo
trace needed to audit the bridge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from armbench.mujoco_sim.model import MuJoCoPanda
from armbench.mujoco_sim.scenarios import mujoco_scenarios
from armbench.vla.async_panda import AsyncPandaConfig, run_async_panda_episode
from armbench.vla.openpi_provider import (
    OpenPILiberoPandaPolicy,
    OpenPILiberoRawProvider,
)
from armbench.vla.policy import BoundedOpenPIBackend


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(root: Path) -> None:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    _write_json(
        root / "manifest.json",
        {
            "schema_version": "armbench.live_pi05_panda_smoke.v1",
            "files": files,
            "inventory_sha256": hashlib.sha256(
                json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key")
    parser.add_argument("--scenario", choices=tuple(mujoco_scenarios()), default="free_space")
    parser.add_argument("--mode", choices=("unguarded", "legacy_greedy", "braking_invariant"), default="braking_invariant")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--extra-steps", type=int, default=6)
    parser.add_argument("--deadline-ms", type=float, default=200.0)
    parser.add_argument("--inference-timeout-s", type=float, default=1.0)
    parser.add_argument("--connect-timeout-s", type=float, default=5.0)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.steps < 2 or args.extra_steps < 0:
        raise SystemExit("--steps must be >= 2 and --extra-steps must be nonnegative")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit("output directory must be absent or empty")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    if platform.system() == "Linux":
        # MuJoCo's EGL path keeps the smoke headless on a cloud host.
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    scenario = mujoco_scenarios()[args.scenario]
    reference = np.linspace(scenario.start, scenario.goal, args.steps + 1)
    adapter_robot = MuJoCoPanda.create(obstacles=())
    backend = None
    provider = None
    started = time.perf_counter()
    try:
        backend = BoundedOpenPIBackend(
            args.host,
            args.port,
            api_key=args.api_key,
            connect_timeout_s=args.connect_timeout_s,
            inference_timeout_s=args.inference_timeout_s,
        )
        server_metadata = backend.get_server_metadata()
        provider = OpenPILiberoRawProvider(backend)
        policy = OpenPILiberoPandaPolicy(provider, adapter_robot)
        config = AsyncPandaConfig(
            response_deadline_s=args.deadline_ms / 1000.0,
            max_action_steps=args.steps + args.extra_steps,
            action_horizon=10,
        )
        video_path = (
            args.output_directory / "panda_trace.mp4" if args.video else None
        )
        result = run_async_panda_episode(
            args.scenario,
            args.mode,
            reference,
            policy=policy,
            config=config,
            prompt=f"move the gripper to the {args.scenario} goal",
            video_path=video_path,
        )
        _write_json(
            args.output_directory / "summary.json",
            {
                "schema_version": "armbench.live_pi05_panda_smoke.v1",
                "scope": "attested_live_pi05_libero_to_panda_mujoco_smoke",
                "elapsed_wall_s": time.perf_counter() - started,
                "server": {"host": args.host, "port": args.port},
                "server_metadata": server_metadata,
                "policy_provenance": policy.provenance,
                "last_policy_response": policy.metrics(),
                "episode": result.metrics(),
                "limitations": [
                    "One integration smoke; not an official LIBERO task-success benchmark.",
                    "Panda is simulated in MuJoCo; no physical robot claim.",
                    "Python wall-clock scheduling is not an OS hard-real-time guarantee.",
                    "LIBERO Cartesian actions are adapted by local differential IK, not robosuite OSC torque equivalence.",
                ],
            },
        )
        _write_json(args.output_directory / "events.json", list(result.events))
        np.savez_compressed(
            args.output_directory / "trace.npz",
            scheduled_wall_times_s=result.scheduled_wall_times_s,
            actual_wall_times_s=result.actual_wall_times_s,
            simulated_times_s=result.simulated_times_s,
            desired_positions=result.desired_positions,
            actual_positions=result.actual_positions,
            command_velocities=result.command_velocities,
            command_statuses=result.command_statuses,
            observation_ages_ms=result.observation_ages_ms,
            request_ids=result.request_ids,
            action_indices=result.action_indices,
        )
        _write_manifest(args.output_directory)
        print(json.dumps(_jsonable(result.metrics()), ensure_ascii=False, indent=2))
        print(f"results: {args.output_directory.resolve()}")
        return 0
    finally:
        if provider is not None:
            provider.close()
        elif backend is not None:
            backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
