"""Run the pre-rollout gate for pi0.5 projected flow conditioning."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import statistics
import subprocess
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


SCHEMA_VERSION = "armbench.pi05_projected_conditioning_g0.v1"
METHOD = "projected_flow_inpainting"
DEFAULT_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
DEFAULT_CONFIG = "pi05_libero"
MAX_WARM_MODEL_P95_RATIO = 1.25
MAX_PEAK_DEVICE_BYTES = 23 * 1024**3


def _command_output(command: Sequence[str], cwd: pathlib.Path) -> str:
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.rstrip("\r\n")


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_source_sha256(path: pathlib.Path) -> str:
    canonical = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def _checkpoint_manifest(checkpoint_directory: pathlib.Path) -> Dict[str, Any]:
    root = checkpoint_directory.resolve()
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    if not files:
        raise ValueError("checkpoint directory contains no files")
    canonical = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return {
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "content_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("latency values cannot be empty")
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _timing_summary(values: Sequence[float]) -> Dict[str, Any]:
    return {
        "count": len(values),
        "mean_ms": float(statistics.fmean(values)),
        "p50_ms": _percentile(values, 50.0),
        "p95_ms": _percentile(values, 95.0),
        "max_ms": float(max(values)),
    }


def _memory_stats() -> Dict[str, Optional[int]]:
    import jax

    raw = jax.devices()[0].memory_stats() or {}
    return {
        key: int(raw[key]) if key in raw and raw[key] is not None else None
        for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
    }


def _fixed_observation() -> Dict[str, Any]:
    height = width = 224
    row = np.arange(height, dtype=np.uint8)[:, None]
    column = np.arange(width, dtype=np.uint8)[None, :]
    base = np.stack(
        (
            np.broadcast_to(row, (height, width)),
            np.broadcast_to(column, (height, width)),
            np.bitwise_xor(row, column),
        ),
        axis=-1,
    )
    wrist = np.ascontiguousarray(base[:, ::-1])
    return {
        "observation/state": np.linspace(-0.25, 0.25, 8, dtype=np.float32),
        "observation/image": np.ascontiguousarray(base),
        "observation/wrist_image": wrist,
        "prompt": "pick up the black bowl and place it on the plate",
    }


def _inference(policy: Any, observation: Mapping[str, Any], **kwargs: Any) -> tuple[Mapping[str, Any], float]:
    started = time.perf_counter()
    response = policy.infer(dict(observation), **kwargs)
    wall_ms = (time.perf_counter() - started) * 1000.0
    if not isinstance(response, Mapping):
        raise TypeError("policy response must be an object")
    return response, wall_ms


def run_gate(
    *,
    openpi_root: pathlib.Path,
    checkpoint: str,
    policy_config: str,
    iterations: int,
    prefix_steps: int,
    allow_dirty_openpi: bool,
) -> Dict[str, Any]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if prefix_steps <= 0 or prefix_steps > 10:
        raise ValueError("prefix_steps must be within [1, 10]")
    openpi_root = openpi_root.resolve()
    commit = _command_output(("git", "rev-parse", "HEAD"), openpi_root)
    status = _command_output(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        openpi_root,
    )
    if status and not allow_dirty_openpi:
        raise RuntimeError("OpenPI worktree must be clean for a final G0 artifact")

    from openpi.policies import policy_config as openpi_policy_config
    from openpi.shared import download
    from openpi.training import config as training_config

    config = training_config.get_config(policy_config)
    if int(config.model.action_horizon) != 10 or int(config.model.action_dim) != 32:
        raise ValueError("G0 requires the pi0.5 LIBERO 10x32 model contract")
    local_checkpoint = pathlib.Path(str(download.maybe_download(checkpoint)))
    checkpoint_manifest = _checkpoint_manifest(local_checkpoint)
    policy = openpi_policy_config.create_trained_policy(config, local_checkpoint)
    observation = _fixed_observation()
    random = np.random.RandomState(20260805)
    noise = np.asarray(random.standard_normal((10, 32)), dtype="<f4", order="C")

    compile_baseline, compile_baseline_ms = _inference(
        policy,
        observation,
        noise=noise,
    )
    baseline_actions = np.array(
        compile_baseline["actions"], dtype=np.float32, order="C", copy=True
    )
    if baseline_actions.shape != (10, 7) or not np.all(np.isfinite(baseline_actions)):
        raise RuntimeError("baseline policy returned invalid LIBERO actions")

    empty_mask = np.zeros(10, dtype=np.bool_)
    empty_response, compile_conditioned_ms = _inference(
        policy,
        observation,
        noise=noise,
        condition_actions=baseline_actions,
        condition_mask=empty_mask,
    )
    empty_actions = np.asarray(empty_response["actions"], dtype=np.float32)
    empty_mask_exact_parity = bool(np.array_equal(empty_actions, baseline_actions))
    if not empty_mask_exact_parity:
        raise RuntimeError("empty-mask conditioning changed the official sampler output")

    active_mask = np.arange(10) < prefix_steps
    active_response, first_active_ms = _inference(
        policy,
        observation,
        noise=noise,
        condition_actions=baseline_actions,
        condition_mask=active_mask,
    )
    audit = active_response.get("policy_conditioning")
    if not isinstance(audit, Mapping):
        raise RuntimeError("conditioned policy response omitted its audit")
    model_residual = float(audit.get("max_model_residual", float("inf")))
    if not np.isfinite(model_residual) or model_residual >= 1e-6:
        raise RuntimeError("conditioned model-space residual failed the 1e-6 gate")
    active_actions = np.asarray(active_response["actions"], dtype=np.float32)
    raw_prefix_residual = float(
        np.max(np.abs(active_actions[:prefix_steps] - baseline_actions[:prefix_steps]))
    )

    baseline_wall_ms: list[float] = []
    baseline_model_ms: list[float] = []
    conditioned_wall_ms: list[float] = []
    conditioned_model_ms: list[float] = []
    for _ in range(iterations):
        baseline_response, wall_ms = _inference(policy, observation, noise=noise)
        repeated_baseline = np.asarray(baseline_response["actions"], dtype=np.float32)
        if not np.array_equal(repeated_baseline, baseline_actions):
            max_difference = float(np.max(np.abs(repeated_baseline - baseline_actions)))
            raise RuntimeError(
                "explicit-noise baseline was not exactly repeatable; "
                f"max_abs_difference={max_difference:.9g}"
            )
        baseline_wall_ms.append(wall_ms)
        baseline_model_ms.append(float(baseline_response["policy_timing"]["infer_ms"]))

        conditioned_response, wall_ms = _inference(
            policy,
            observation,
            noise=noise,
            condition_actions=baseline_actions,
            condition_mask=active_mask,
        )
        conditioned_audit = conditioned_response.get("policy_conditioning")
        if not isinstance(conditioned_audit, Mapping):
            raise RuntimeError("warm conditioned response omitted its audit")
        residual = float(conditioned_audit.get("max_model_residual", float("inf")))
        if not np.isfinite(residual) or residual >= 1e-6:
            raise RuntimeError("warm conditioned residual failed the 1e-6 gate")
        conditioned_wall_ms.append(wall_ms)
        conditioned_model_ms.append(float(conditioned_response["policy_timing"]["infer_ms"]))

    source_files = {
        relative: _sha256_file(openpi_root / relative)
        for relative in (
            "src/openpi/models/pi0.py",
            "src/openpi/policies/policy.py",
        )
    }
    baseline_wall_summary = _timing_summary(baseline_wall_ms)
    baseline_model_summary = _timing_summary(baseline_model_ms)
    conditioned_wall_summary = _timing_summary(conditioned_wall_ms)
    conditioned_model_summary = _timing_summary(conditioned_model_ms)
    warm_model_p95_ratio = (
        conditioned_model_summary["p95_ms"] / baseline_model_summary["p95_ms"]
    )
    memory = _memory_stats()
    peak_bytes = memory["peak_bytes_in_use"]
    if warm_model_p95_ratio > MAX_WARM_MODEL_P95_RATIO:
        raise RuntimeError("conditioned warm model P95 exceeded the prespecified ratio gate")
    if peak_bytes is None or peak_bytes > MAX_PEAK_DEVICE_BYTES:
        raise RuntimeError("peak device memory exceeded the prespecified G0 gate")

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "passed": True,
        "method": METHOD,
        "identity": {
            "policy_config": policy_config,
            "checkpoint": checkpoint,
            "checkpoint_content_sha256": checkpoint_manifest["content_sha256"],
            "checkpoint_file_count": checkpoint_manifest["file_count"],
            "checkpoint_total_bytes": checkpoint_manifest["total_bytes"],
            "openpi_commit": commit,
            "openpi_status": status,
            "source_sha256": source_files,
            "gate_source_canonical_sha256": _canonical_source_sha256(
                pathlib.Path(__file__).resolve()
            ),
        },
        "contract": {
            "raw_action_shape": [10, 7],
            "model_action_shape": [10, 32],
            "prefix_steps": prefix_steps,
            "explicit_noise_sha256": hashlib.sha256(noise.tobytes(order="C")).hexdigest(),
        },
        "gates": {
            "baseline_exact_repeatability": True,
            "empty_mask_exact_parity": empty_mask_exact_parity,
            "max_model_residual": model_residual,
            "max_raw_prefix_residual": raw_prefix_residual,
            "warm_model_p95_ratio": warm_model_p95_ratio,
            "max_warm_model_p95_ratio": MAX_WARM_MODEL_P95_RATIO,
            "max_peak_device_bytes": MAX_PEAK_DEVICE_BYTES,
            "latency_gate": True,
            "memory_gate": True,
        },
        "compile_wall_ms": {
            "baseline": compile_baseline_ms,
            "conditioned_array_variant": compile_conditioned_ms,
            "first_active_after_empty": first_active_ms,
        },
        "warm_latency": {
            "baseline_wall": baseline_wall_summary,
            "baseline_model": baseline_model_summary,
            "conditioned_wall": conditioned_wall_summary,
            "conditioned_model": conditioned_model_summary,
        },
        "device_memory": memory,
    }


def _write_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--openpi-root", default="/app")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--policy-config", default=DEFAULT_CONFIG)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--prefix-steps", type=int, default=4)
    parser.add_argument("--allow-dirty-openpi", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    report = run_gate(
        openpi_root=pathlib.Path(args.openpi_root),
        checkpoint=args.checkpoint,
        policy_config=args.policy_config,
        iterations=args.iterations,
        prefix_steps=args.prefix_steps,
        allow_dirty_openpi=bool(args.allow_dirty_openpi),
    )
    output = pathlib.Path(args.output)
    _write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
