"""Run the official-checkpoint G0 gate for pi0.5 RTC VJP guidance."""

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


SCHEMA_VERSION = "armbench.pi05_rtc_guidance_g0.v1"
METHOD = "rtc_pseudoinverse_guidance"
OPENPI_EXTENSION_COMMIT = "54592c7148ba69bf52757385502782f80f2285e0"
RTC_REFERENCE_COMMIT = "9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b"
DEFAULT_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
DEFAULT_CONFIG = "pi05_libero"
DEFAULT_SCHEDULE = "exp"
DEFAULT_MAX_GUIDANCE_WEIGHT = 5.0
MAX_WARM_WALL_P95_MS = 200.0
MAX_WARM_WALL_P95_RATIO = 3.0
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


def _timing_summary(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        raise ValueError("latency values cannot be empty")
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean_ms": float(statistics.fmean(values)),
        "p50_ms": float(np.percentile(array, 50.0)),
        "p95_ms": float(np.percentile(array, 95.0)),
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
    return {
        "observation/state": np.linspace(-0.25, 0.25, 8, dtype=np.float32),
        "observation/image": np.ascontiguousarray(base),
        "observation/wrist_image": np.ascontiguousarray(base[:, ::-1]),
        "prompt": "pick up the black bowl and place it on the plate",
    }


def _prefix_weights(
    inference_delay: int,
    execute_horizon: int,
    action_horizon: int = 10,
    schedule: str = DEFAULT_SCHEDULE,
) -> np.ndarray:
    if type(inference_delay) is not int or inference_delay < 0:
        raise ValueError("inference_delay must be a nonnegative integer")
    if type(execute_horizon) is not int or execute_horizon <= 0:
        raise ValueError("execute_horizon must be a positive integer")
    if inference_delay > execute_horizon or execute_horizon > action_horizon:
        raise ValueError("guidance horizons are inconsistent")
    if schedule not in ("linear", "exp", "ones", "zeros"):
        raise ValueError("unsupported guidance schedule")
    end = action_horizon - execute_horizon
    start = min(inference_delay, end)
    index = np.arange(action_horizon, dtype=np.float64)
    if schedule == "ones":
        weights = np.ones(action_horizon, dtype=np.float64)
    elif schedule == "zeros":
        weights = (index < start).astype(np.float64)
    else:
        weights = np.clip(
            (start - 1 - index) / (end - start + 1) + 1.0,
            0.0,
            1.0,
        )
        if schedule == "exp":
            weights = weights * np.expm1(weights) / np.expm1(1.0)
    return np.asarray(np.where(index >= end, 0.0, weights), dtype="<f4")


def _shift_reference(actions: np.ndarray, execute_horizon: int) -> np.ndarray:
    canonical = np.asarray(actions, dtype="<f4", order="C")
    if canonical.ndim != 2 or execute_horizon <= 0 or execute_horizon > canonical.shape[0]:
        raise ValueError("reference shift is inconsistent with the action chunk")
    zeros = np.zeros((execute_horizon, canonical.shape[1]), dtype="<f4")
    return np.ascontiguousarray(
        np.concatenate((canonical[execute_horizon:], zeros), axis=0),
        dtype="<f4",
    )


def _inference(
    policy: Any,
    observation: Mapping[str, Any],
    **kwargs: Any,
) -> tuple[Mapping[str, Any], float]:
    started = time.perf_counter()
    response = policy.infer(dict(observation), **kwargs)
    wall_ms = (time.perf_counter() - started) * 1000.0
    if not isinstance(response, Mapping):
        raise TypeError("policy response must be an object")
    return response, wall_ms


def _guidance_audit(response: Mapping[str, Any]) -> Mapping[str, Any]:
    audit = response.get("policy_guidance")
    if not isinstance(audit, Mapping):
        raise RuntimeError("guided policy response omitted its audit")
    if audit.get("method") != METHOD:
        raise RuntimeError("guided policy response method mismatch")
    for field in ("max_weighted_model_residual", "weighted_model_rmse"):
        value = audit.get(field)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise RuntimeError("guided policy response residual is invalid")
    return audit


def run_gate(
    *,
    openpi_root: pathlib.Path,
    checkpoint: str,
    policy_config: str,
    iterations: int,
    inference_delay: int,
    execute_horizon: int,
    allow_dirty_openpi: bool,
) -> Dict[str, Any]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    weights = _prefix_weights(inference_delay, execute_horizon)
    openpi_root = openpi_root.resolve()
    commit = _command_output(("git", "rev-parse", "HEAD"), openpi_root)
    if commit != OPENPI_EXTENSION_COMMIT:
        raise RuntimeError("OpenPI RTC extension commit mismatch")
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

    baseline_response, compile_baseline_ms = _inference(
        policy,
        observation,
        noise=noise,
    )
    baseline_actions = np.asarray(
        baseline_response["actions"], dtype="<f4", order="C"
    )
    if baseline_actions.shape != (10, 7) or not np.all(np.isfinite(baseline_actions)):
        raise RuntimeError("baseline policy returned invalid LIBERO actions")
    reference = _shift_reference(baseline_actions, execute_horizon)

    zero_response, zero_weight_ms = _inference(
        policy,
        observation,
        noise=noise,
        guidance_actions=reference,
        guidance_weights=np.zeros(10, dtype="<f4"),
        guidance_schedule="zeros",
    )
    zero_actions = np.asarray(zero_response["actions"], dtype="<f4")
    zero_weight_exact_parity = bool(np.array_equal(zero_actions, baseline_actions))
    if not zero_weight_exact_parity:
        raise RuntimeError("zero RTC weights changed the official sampler output")
    zero_audit = _guidance_audit(zero_response)
    if zero_audit.get("guidance_active") is not False:
        raise RuntimeError("zero RTC weights did not use the legacy sampler bypass")

    reference_baseline, reference_baseline_ms = _inference(
        policy,
        observation,
        noise=noise,
        guidance_actions=reference,
        guidance_weights=weights,
        guidance_schedule=DEFAULT_SCHEDULE,
        max_guidance_weight=0.0,
    )
    if not np.array_equal(
        np.asarray(reference_baseline["actions"], dtype="<f4"), baseline_actions
    ):
        raise RuntimeError("zero maximum guidance changed the baseline output")
    reference_baseline_audit = _guidance_audit(reference_baseline)
    baseline_model_residual = float(
        reference_baseline_audit["weighted_model_rmse"]
    )
    if baseline_model_residual <= 0.0:
        raise RuntimeError("shifted RTC reference did not create a measurable residual")

    guided_response, compile_guided_ms = _inference(
        policy,
        observation,
        noise=noise,
        guidance_actions=reference,
        guidance_weights=weights,
        guidance_schedule=DEFAULT_SCHEDULE,
        max_guidance_weight=DEFAULT_MAX_GUIDANCE_WEIGHT,
    )
    guided_actions = np.asarray(guided_response["actions"], dtype="<f4")
    if guided_actions.shape != (10, 7) or not np.all(np.isfinite(guided_actions)):
        raise RuntimeError("RTC-guided policy returned invalid LIBERO actions")
    if np.array_equal(guided_actions, baseline_actions):
        raise RuntimeError("active RTC guidance did not change the sampled actions")
    guided_audit = _guidance_audit(guided_response)
    guided_model_residual = float(guided_audit["weighted_model_rmse"])
    residual_ratio = guided_model_residual / baseline_model_residual
    if not guided_model_residual < baseline_model_residual:
        raise RuntimeError("RTC guidance did not reduce model-space weighted residual")

    baseline_wall_ms: list[float] = []
    guided_wall_ms: list[float] = []
    baseline_policy_ms: list[float] = []
    guided_policy_ms: list[float] = []
    for _ in range(iterations):
        repeated_baseline, wall_ms = _inference(
            policy, observation, noise=noise
        )
        if not np.array_equal(
            np.asarray(repeated_baseline["actions"], dtype="<f4"),
            baseline_actions,
        ):
            raise RuntimeError("explicit-noise baseline was not exactly repeatable")
        baseline_wall_ms.append(wall_ms)
        baseline_policy_ms.append(
            float(repeated_baseline["policy_timing"]["infer_ms"])
        )

        repeated_guided, wall_ms = _inference(
            policy,
            observation,
            noise=noise,
            guidance_actions=reference,
            guidance_weights=weights,
            guidance_schedule=DEFAULT_SCHEDULE,
            max_guidance_weight=DEFAULT_MAX_GUIDANCE_WEIGHT,
        )
        if not np.array_equal(
            np.asarray(repeated_guided["actions"], dtype="<f4"), guided_actions
        ):
            raise RuntimeError("RTC-guided sampling was not exactly repeatable")
        repeated_audit = _guidance_audit(repeated_guided)
        if repeated_audit["weighted_model_rmse"] != guided_model_residual:
            raise RuntimeError("RTC-guided audit was not exactly repeatable")
        guided_wall_ms.append(wall_ms)
        guided_policy_ms.append(
            float(repeated_guided["policy_timing"]["infer_ms"])
        )

    baseline_wall = _timing_summary(baseline_wall_ms)
    guided_wall = _timing_summary(guided_wall_ms)
    wall_p95_ratio = guided_wall["p95_ms"] / baseline_wall["p95_ms"]
    memory = _memory_stats()
    peak_bytes = memory["peak_bytes_in_use"]
    if guided_wall["p95_ms"] > MAX_WARM_WALL_P95_MS:
        raise RuntimeError("RTC-guided warm wall P95 exceeded the deadline gate")
    if wall_p95_ratio > MAX_WARM_WALL_P95_RATIO:
        raise RuntimeError("RTC-guided warm wall P95 exceeded the ratio gate")
    if peak_bytes is None or peak_bytes > MAX_PEAK_DEVICE_BYTES:
        raise RuntimeError("peak device memory exceeded the prespecified G0 gate")

    source_files = {
        relative: _sha256_file(openpi_root / relative)
        for relative in (
            "src/openpi/models/pi0.py",
            "src/openpi/policies/policy.py",
        )
    }
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
            "rtc_reference_commit": RTC_REFERENCE_COMMIT,
            "source_sha256": source_files,
            "gate_source_canonical_sha256": _canonical_source_sha256(
                pathlib.Path(__file__).resolve()
            ),
        },
        "contract": {
            "raw_action_shape": [10, 7],
            "model_action_shape": [10, 32],
            "inference_delay": inference_delay,
            "execute_horizon": execute_horizon,
            "schedule": DEFAULT_SCHEDULE,
            "max_guidance_weight": DEFAULT_MAX_GUIDANCE_WEIGHT,
            "weights": weights.tolist(),
            "weights_sha256": hashlib.sha256(
                weights.tobytes(order="C")
            ).hexdigest(),
            "explicit_noise_sha256": hashlib.sha256(
                noise.tobytes(order="C")
            ).hexdigest(),
            "reference_actions_sha256": hashlib.sha256(
                reference.tobytes(order="C")
            ).hexdigest(),
        },
        "gates": {
            "baseline_exact_repeatability": True,
            "guided_exact_repeatability": True,
            "zero_weight_exact_parity": zero_weight_exact_parity,
            "finite_guided_actions": True,
            "baseline_weighted_model_rmse": baseline_model_residual,
            "guided_weighted_model_rmse": guided_model_residual,
            "weighted_model_residual_ratio": residual_ratio,
            "weighted_model_residual_reduction": True,
            "warm_wall_p95_ms": guided_wall["p95_ms"],
            "max_warm_wall_p95_ms": MAX_WARM_WALL_P95_MS,
            "warm_wall_p95_ratio": wall_p95_ratio,
            "max_warm_wall_p95_ratio": MAX_WARM_WALL_P95_RATIO,
            "max_peak_device_bytes": MAX_PEAK_DEVICE_BYTES,
            "latency_gate": True,
            "memory_gate": True,
        },
        "compile_wall_ms": {
            "baseline": compile_baseline_ms,
            "zero_weight_policy_variant": zero_weight_ms,
            "zero_maximum_reference_variant": reference_baseline_ms,
            "active_guidance": compile_guided_ms,
        },
        "warm_latency": {
            "baseline_wall": baseline_wall,
            "guided_wall": guided_wall,
            "baseline_policy_dispatch": _timing_summary(baseline_policy_ms),
            "guided_policy_dispatch": _timing_summary(guided_policy_ms),
        },
        "device_memory": memory,
        "action_sha256": {
            "baseline": hashlib.sha256(
                baseline_actions.tobytes(order="C")
            ).hexdigest(),
            "guided": hashlib.sha256(
                guided_actions.tobytes(order="C")
            ).hexdigest(),
        },
        "guidance_audit": dict(guided_audit),
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
    parser.add_argument("--inference-delay", type=int, default=4)
    parser.add_argument("--execute-horizon", type=int, default=5)
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
        inference_delay=args.inference_delay,
        execute_horizon=args.execute_horizon,
        allow_dirty_openpi=bool(args.allow_dirty_openpi),
    )
    output = pathlib.Path(args.output)
    _write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
