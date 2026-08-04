"""Serve an OpenPI checkpoint with auditable model and source provenance."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import pathlib
import platform
import re
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


ATTESTATION_SCHEMA_VERSION = "armbench.openpi_server_attestation.v1"
OPENPI_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
OPENPI_PROJECTED_CONDITIONING_COMMIT = (
    "2c8e61d5fbfde4b670ae428ef4b8440d35c1c7fd"
)
DEFAULT_POLICY_CONFIG = "pi05_libero"
DEFAULT_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
POLICY_SAMPLING_SCHEMA_VERSION = "armbench.policy_sampling.v1"
POLICY_SAMPLING_TRACE_SCHEMA_VERSION = "armbench.policy_sampling_trace.v1"
POLICY_SAMPLING_REQUEST_FIELD = "armbench_policy_sampling"
POLICY_SAMPLING_RESPONSE_FIELD = "armbench_policy_sampling"
POLICY_SAMPLING_METADATA_FIELD = "armbench_policy_sampling_contract"
POLICY_SAMPLING_GENERATOR = (
    "numpy_randomstate_mt19937_standard_normal_float32_v1"
)
POLICY_SAMPLING_SCORED_NAMESPACE = "scored"
POLICY_SAMPLING_WARMUP_NAMESPACE = "warmup"
POLICY_CONDITIONING_SCHEMA_VERSION = "armbench.policy_conditioning.v1"
POLICY_CONDITIONING_TRACE_SCHEMA_VERSION = "armbench.policy_conditioning_trace.v1"
POLICY_CONDITIONING_REQUEST_FIELD = "armbench_policy_conditioning"
POLICY_CONDITIONING_RESPONSE_FIELD = "armbench_policy_conditioning"
POLICY_CONDITIONING_METADATA_FIELD = "armbench_policy_conditioning_contract"
POLICY_CONDITIONING_METHOD = "projected_flow_inpainting"
OPENPI_INTERNAL_CONDITIONING_FIELD = "policy_conditioning"
PI05_ACTION_HORIZON = 10
PI05_MODEL_ACTION_DIM = 32
PI05_RAW_ACTION_DIM = 7

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def policy_sampling_contract(
    action_horizon: int = PI05_ACTION_HORIZON,
    model_action_dim: int = PI05_MODEL_ACTION_DIM,
) -> Dict[str, Any]:
    """Return the exact request, noise, and audit contract advertised by the server."""

    return {
        "schema_version": POLICY_SAMPLING_SCHEMA_VERSION,
        "trace_schema_version": POLICY_SAMPLING_TRACE_SCHEMA_VERSION,
        "request_field": POLICY_SAMPLING_REQUEST_FIELD,
        "response_field": POLICY_SAMPLING_RESPONSE_FIELD,
        "generator": POLICY_SAMPLING_GENERATOR,
        "noise_shape": [int(action_horizon), int(model_action_dim)],
        "noise_dtype": "little-endian float32",
        "mode_in_key": False,
        "payload_fields": [
            "schema_version",
            "namespace",
            "seed",
            "pairing_key",
            "query_index",
        ],
        "json_encoding": "utf-8 canonical sorted compact ASCII",
        "namespaces": {
            POLICY_SAMPLING_SCORED_NAMESPACE: {
                "pairing_key_fields": [
                    "task_suite",
                    "task_id",
                    "episode_index",
                    "replan_steps",
                ]
            },
            POLICY_SAMPLING_WARMUP_NAMESPACE: {
                "pairing_key_fields": [
                    "task_suite",
                    "task_id",
                    "episode_index",
                ]
            },
        },
    }


def policy_conditioning_contract(
    action_horizon: int = PI05_ACTION_HORIZON,
    raw_action_dim: int = PI05_RAW_ACTION_DIM,
    model_action_dim: int = PI05_MODEL_ACTION_DIM,
) -> Dict[str, Any]:
    return {
        "schema_version": POLICY_CONDITIONING_SCHEMA_VERSION,
        "trace_schema_version": POLICY_CONDITIONING_TRACE_SCHEMA_VERSION,
        "request_field": POLICY_CONDITIONING_REQUEST_FIELD,
        "response_field": POLICY_CONDITIONING_RESPONSE_FIELD,
        "method": POLICY_CONDITIONING_METHOD,
        "raw_action_shape": [int(action_horizon), int(raw_action_dim)],
        "model_action_shape": [int(action_horizon), int(model_action_dim)],
        "actions_dtype": "little-endian float32",
        "mask_shape": [int(action_horizon)],
        "mask_dtype": "bool",
        "mask_semantics": "contiguous committed prefix",
    }


def _canonical_sampling_payload(
    namespace: str,
    seed: int,
    pairing_key: Sequence[Any],
    query_index: int,
) -> Dict[str, Any]:
    if namespace not in (
        POLICY_SAMPLING_SCORED_NAMESPACE,
        POLICY_SAMPLING_WARMUP_NAMESPACE,
    ):
        raise ValueError("unsupported policy sampling namespace")
    if type(seed) is not int or seed < 0 or seed > 2**63 - 1:
        raise ValueError("policy sampling seed must be an unsigned 63-bit integer")
    if type(query_index) is not int or query_index < 0 or query_index > 2**63 - 1:
        raise ValueError("policy sampling query_index must be an unsigned 63-bit integer")
    if not isinstance(pairing_key, Sequence) or isinstance(
        pairing_key, (str, bytes, bytearray)
    ):
        raise ValueError("policy sampling pairing_key must be an array")
    key = list(pairing_key)
    if namespace == POLICY_SAMPLING_SCORED_NAMESPACE:
        valid = (
            len(key) == 4
            and type(key[0]) is str
            and bool(key[0])
            and all(type(value) is int and value >= 0 for value in key[1:])
        )
    else:
        valid = (
            len(key) == 3
            and type(key[0]) is str
            and bool(key[0])
            and all(type(value) is int and value >= 0 for value in key[1:])
        )
    if not valid:
        raise ValueError("policy sampling pairing_key does not match its namespace")
    return {
        "schema_version": POLICY_SAMPLING_SCHEMA_VERSION,
        "namespace": namespace,
        "seed": seed,
        "pairing_key": key,
        "query_index": query_index,
    }


def policy_sampling_key_sha256(
    namespace: str,
    seed: int,
    pairing_key: Sequence[Any],
    query_index: int,
) -> str:
    payload = _canonical_sampling_payload(namespace, seed, pairing_key, query_index)
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def policy_sampling_noise(
    key_sha256: str,
    action_horizon: int = PI05_ACTION_HORIZON,
    model_action_dim: int = PI05_MODEL_ACTION_DIM,
) -> np.ndarray:
    """Generate versioned standard-normal noise from all 256 key bits."""

    if not isinstance(key_sha256, str) or not _SHA256.fullmatch(key_sha256):
        raise ValueError("policy sampling key_sha256 is invalid")
    if type(action_horizon) is not int or action_horizon <= 0:
        raise ValueError("action_horizon must be a positive integer")
    if type(model_action_dim) is not int or model_action_dim <= 0:
        raise ValueError("model_action_dim must be a positive integer")
    digest = bytes.fromhex(key_sha256)
    seed_words = [int.from_bytes(digest[index : index + 4], "big") for index in range(0, 32, 4)]
    random = np.random.RandomState(seed_words)
    values = random.standard_normal((action_horizon, model_action_dim))
    return np.asarray(values, dtype="<f4", order="C")


def policy_sampling_noise_sha256(noise: np.ndarray) -> str:
    canonical = np.asarray(noise, dtype="<f4", order="C")
    if canonical.ndim != 2:
        raise ValueError("policy sampling noise must be a two-dimensional array")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def build_policy_sampling_control(
    namespace: str,
    seed: int,
    pairing_key: Sequence[Any],
    query_index: int,
) -> Dict[str, Any]:
    payload = _canonical_sampling_payload(namespace, seed, pairing_key, query_index)
    payload["key_sha256"] = policy_sampling_key_sha256(
        namespace, seed, pairing_key, query_index
    )
    return payload


def policy_conditioning_actions_sha256(actions: np.ndarray) -> str:
    canonical = np.asarray(actions, dtype="<f4", order="C")
    if canonical.ndim != 2:
        raise ValueError("conditioning actions must be a two-dimensional array")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def policy_conditioning_mask_sha256(mask: np.ndarray) -> str:
    canonical = np.asarray(mask)
    if canonical.ndim != 1 or canonical.dtype != np.bool_:
        raise ValueError("conditioning mask must be a one-dimensional boolean array")
    return hashlib.sha256(np.asarray(canonical, dtype="u1").tobytes(order="C")).hexdigest()


def _canonical_conditioning_arrays(
    actions: Any,
    mask: Any,
    *,
    action_horizon: int,
    raw_action_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        canonical_actions = np.asarray(actions, dtype="<f4", order="C")
    except (TypeError, ValueError) as exc:
        raise ValueError("conditioning actions must be numeric") from exc
    canonical_mask = np.asarray(mask)
    if canonical_actions.shape != (action_horizon, raw_action_dim):
        raise ValueError("conditioning actions do not match the raw action shape")
    if not np.all(np.isfinite(canonical_actions)):
        raise ValueError("conditioning actions must contain only finite values")
    if canonical_mask.shape != (action_horizon,) or canonical_mask.dtype != np.bool_:
        raise ValueError("conditioning mask must be boolean with shape (action_horizon,)")
    false_seen = np.maximum.accumulate(~canonical_mask)
    if np.any(canonical_mask & false_seen):
        raise ValueError("conditioning mask must select a contiguous prefix")
    return (
        np.array(canonical_actions, dtype="<f4", order="C", copy=True),
        np.array(canonical_mask, dtype=np.bool_, order="C", copy=True),
    )


def build_policy_conditioning_control(
    actions: Any,
    *,
    inference_delay: int,
    execute_horizon: int,
    action_horizon: int = PI05_ACTION_HORIZON,
    raw_action_dim: int = PI05_RAW_ACTION_DIM,
) -> Dict[str, Any]:
    if type(inference_delay) is not int or inference_delay < 0:
        raise ValueError("conditioning inference_delay must be a nonnegative integer")
    if type(execute_horizon) is not int or execute_horizon <= 0:
        raise ValueError("conditioning execute_horizon must be a positive integer")
    if inference_delay > execute_horizon or execute_horizon > action_horizon:
        raise ValueError("conditioning horizons are inconsistent")
    mask = np.arange(action_horizon) < inference_delay
    canonical_actions, canonical_mask = _canonical_conditioning_arrays(
        actions,
        mask,
        action_horizon=action_horizon,
        raw_action_dim=raw_action_dim,
    )
    return {
        "schema_version": POLICY_CONDITIONING_SCHEMA_VERSION,
        "method": POLICY_CONDITIONING_METHOD,
        "inference_delay": inference_delay,
        "execute_horizon": execute_horizon,
        "condition_actions": canonical_actions,
        "condition_mask": canonical_mask,
        "raw_actions_sha256": policy_conditioning_actions_sha256(canonical_actions),
        "mask_sha256": policy_conditioning_mask_sha256(canonical_mask),
    }


def _validate_policy_conditioning_control(
    value: Any,
    *,
    action_horizon: int,
    raw_action_dim: int,
) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("policy conditioning control must be an object")
    expected_fields = {
        "schema_version",
        "method",
        "inference_delay",
        "execute_horizon",
        "condition_actions",
        "condition_mask",
        "raw_actions_sha256",
        "mask_sha256",
    }
    if set(value) != expected_fields:
        raise ValueError("policy conditioning control fields do not match the contract")
    if value.get("schema_version") != POLICY_CONDITIONING_SCHEMA_VERSION:
        raise ValueError("policy conditioning schema_version mismatch")
    if value.get("method") != POLICY_CONDITIONING_METHOD:
        raise ValueError("unsupported policy conditioning method")
    delay = value.get("inference_delay")
    execute = value.get("execute_horizon")
    if type(delay) is not int or delay < 0:
        raise ValueError("policy conditioning inference_delay is invalid")
    if type(execute) is not int or execute <= 0:
        raise ValueError("policy conditioning execute_horizon is invalid")
    if delay > execute or execute > action_horizon:
        raise ValueError("policy conditioning horizons are inconsistent")
    actions, mask = _canonical_conditioning_arrays(
        value.get("condition_actions"),
        value.get("condition_mask"),
        action_horizon=action_horizon,
        raw_action_dim=raw_action_dim,
    )
    if not np.array_equal(mask, np.arange(action_horizon) < delay):
        raise ValueError("policy conditioning mask does not match inference_delay")
    actions_sha256 = policy_conditioning_actions_sha256(actions)
    mask_sha256 = policy_conditioning_mask_sha256(mask)
    if value.get("raw_actions_sha256") != actions_sha256:
        raise ValueError("policy conditioning raw_actions_sha256 mismatch")
    if value.get("mask_sha256") != mask_sha256:
        raise ValueError("policy conditioning mask_sha256 mismatch")
    return {
        "schema_version": POLICY_CONDITIONING_SCHEMA_VERSION,
        "method": POLICY_CONDITIONING_METHOD,
        "inference_delay": delay,
        "execute_horizon": execute,
        "condition_actions": actions,
        "condition_mask": mask,
        "raw_actions_sha256": actions_sha256,
        "mask_sha256": mask_sha256,
    }


def _validate_policy_sampling_control(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("policy sampling control must be an object")
    expected_fields = {
        "schema_version",
        "namespace",
        "seed",
        "pairing_key",
        "query_index",
        "key_sha256",
    }
    if set(value) != expected_fields:
        raise ValueError("policy sampling control fields do not match the contract")
    namespace = value.get("namespace")
    if type(namespace) is not str:
        raise ValueError("policy sampling namespace must be a string")
    payload = _canonical_sampling_payload(
        namespace,
        value.get("seed"),
        value.get("pairing_key"),
        value.get("query_index"),
    )
    if value.get("schema_version") != POLICY_SAMPLING_SCHEMA_VERSION:
        raise ValueError("policy sampling schema_version mismatch")
    expected_key = policy_sampling_key_sha256(
        payload["namespace"],
        payload["seed"],
        payload["pairing_key"],
        payload["query_index"],
    )
    if value.get("key_sha256") != expected_key:
        raise ValueError("policy sampling key_sha256 mismatch")
    payload["key_sha256"] = expected_key
    return payload


class KeyedPolicySamplingWrapper:
    """Inject keyed noise and optional committed actions with auditable traces."""

    def __init__(
        self,
        policy: Any,
        *,
        action_horizon: int = PI05_ACTION_HORIZON,
        model_action_dim: int = PI05_MODEL_ACTION_DIM,
        raw_action_dim: int = PI05_RAW_ACTION_DIM,
    ) -> None:
        self._policy = policy
        self._action_horizon = int(action_horizon)
        self._model_action_dim = int(model_action_dim)
        self._raw_action_dim = int(raw_action_dim)

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._policy.metadata

    def infer(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        has_sampling = POLICY_SAMPLING_REQUEST_FIELD in observation
        has_conditioning = POLICY_CONDITIONING_REQUEST_FIELD in observation
        if not has_sampling and not has_conditioning:
            return self._policy.infer(observation)
        clean_observation = dict(observation)
        kwargs: Dict[str, Any] = {}
        sampling_control = None
        key_sha256 = None
        noise_sha256 = None
        if has_sampling:
            sampling_control = _validate_policy_sampling_control(
                clean_observation.pop(POLICY_SAMPLING_REQUEST_FIELD)
            )
            key_sha256 = str(sampling_control["key_sha256"])
            noise = policy_sampling_noise(
                key_sha256, self._action_horizon, self._model_action_dim
            )
            noise_sha256 = policy_sampling_noise_sha256(noise)
            kwargs["noise"] = noise
        conditioning_control = None
        if has_conditioning:
            conditioning_control = _validate_policy_conditioning_control(
                clean_observation.pop(POLICY_CONDITIONING_REQUEST_FIELD),
                action_horizon=self._action_horizon,
                raw_action_dim=self._raw_action_dim,
            )
            kwargs["condition_actions"] = conditioning_control["condition_actions"]
            kwargs["condition_mask"] = conditioning_control["condition_mask"]

        response = self._policy.infer(clean_observation, **kwargs)
        if not isinstance(response, Mapping):
            raise TypeError("controlled policy response must be an object")
        output = dict(response)
        if sampling_control is not None:
            if POLICY_SAMPLING_RESPONSE_FIELD in output:
                raise ValueError("policy response already contains reserved sampling audit field")
            output[POLICY_SAMPLING_RESPONSE_FIELD] = {
                "schema_version": POLICY_SAMPLING_SCHEMA_VERSION,
                "namespace": sampling_control["namespace"],
                "key_sha256": key_sha256,
                "noise_sha256": noise_sha256,
                "generator": POLICY_SAMPLING_GENERATOR,
            }
        if conditioning_control is not None:
            if POLICY_CONDITIONING_RESPONSE_FIELD in output:
                raise ValueError("policy response already contains reserved conditioning audit field")
            internal = output.pop(OPENPI_INTERNAL_CONDITIONING_FIELD, None)
            if not isinstance(internal, Mapping):
                raise ValueError("policy response omitted its internal conditioning audit")
            if internal.get("schema_version") != "armbench.openpi_conditioning.v1":
                raise ValueError("policy conditioning audit schema mismatch")
            if internal.get("method") != POLICY_CONDITIONING_METHOD:
                raise ValueError("policy conditioning audit method mismatch")
            if internal.get("raw_actions_sha256") != conditioning_control["raw_actions_sha256"]:
                raise ValueError("policy conditioning audit raw action hash mismatch")
            if internal.get("mask_sha256") != conditioning_control["mask_sha256"]:
                raise ValueError("policy conditioning audit mask hash mismatch")
            if internal.get("conditioned_prefix_steps") != conditioning_control["inference_delay"]:
                raise ValueError("policy conditioning audit prefix length mismatch")
            model_actions_sha256 = internal.get("model_actions_sha256")
            if not isinstance(model_actions_sha256, str) or not _SHA256.fullmatch(model_actions_sha256):
                raise ValueError("policy conditioning audit model action hash is invalid")
            residual = internal.get("max_model_residual")
            if not isinstance(residual, (int, float)) or not np.isfinite(residual) or residual < 0:
                raise ValueError("policy conditioning audit residual is invalid")
            output[POLICY_CONDITIONING_RESPONSE_FIELD] = {
                "schema_version": POLICY_CONDITIONING_TRACE_SCHEMA_VERSION,
                "method": POLICY_CONDITIONING_METHOD,
                "inference_delay": conditioning_control["inference_delay"],
                "execute_horizon": conditioning_control["execute_horizon"],
                "raw_actions_sha256": conditioning_control["raw_actions_sha256"],
                "model_actions_sha256": model_actions_sha256,
                "mask_sha256": conditioning_control["mask_sha256"],
                "max_model_residual": float(residual),
            }
        return output


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


def _submodules_are_clean(status: str) -> bool:
    return all(line.startswith(" ") for line in status.splitlines() if line)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_content_manifest(checkpoint_directory: pathlib.Path) -> Dict[str, Any]:
    root = checkpoint_directory.resolve()
    if not root.is_dir():
        raise FileNotFoundError("checkpoint directory does not exist: %s" % root)
    files: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        files.append(
            {
                "path": relative,
                "bytes": size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError("checkpoint directory contains no files: %s" % root)
    canonical = json.dumps(
        files, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return {
        "checkpoint_local_path": str(root),
        "checkpoint_file_count": len(files),
        "checkpoint_total_bytes": sum(int(item["bytes"]) for item in files),
        "checkpoint_content_sha256": hashlib.sha256(canonical).hexdigest(),
        "checkpoint_files": files,
    }


def build_attestation(
    *,
    openpi_root: pathlib.Path,
    policy_config: str,
    checkpoint_uri: str,
    checkpoint_manifest: Mapping[str, Any],
    server_source: pathlib.Path,
    action_horizon: int,
    model_action_dim: int,
    upstream_base_commit: str = OPENPI_COMMIT,
) -> Dict[str, Any]:
    commit = _command_output(("git", "rev-parse", "HEAD"), openpi_root)
    tracked_status = _command_output(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        openpi_root,
    )
    submodule_status = _command_output(
        ("git", "submodule", "status", "--recursive"), openpi_root
    )
    return {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "policy_loaded": True,
        "policy_config": policy_config,
        "checkpoint_uri": checkpoint_uri,
        "checkpoint_local_path": checkpoint_manifest["checkpoint_local_path"],
        "checkpoint_file_count": checkpoint_manifest["checkpoint_file_count"],
        "checkpoint_total_bytes": checkpoint_manifest["checkpoint_total_bytes"],
        "checkpoint_content_sha256": checkpoint_manifest[
            "checkpoint_content_sha256"
        ],
        "checkpoint_files": checkpoint_manifest["checkpoint_files"],
        "openpi_commit": commit,
        "openpi_upstream_base_commit": upstream_base_commit,
        "openpi_extension_files": {
            relative: sha256_file(openpi_root / relative)
            for relative in (
                "src/openpi/models/pi0.py",
                "src/openpi/policies/policy.py",
            )
        },
        "openpi_tracked_status": tracked_status,
        "openpi_tracked_clean": tracked_status == "",
        "openpi_submodule_status": submodule_status,
        "openpi_submodules_clean": _submodules_are_clean(submodule_status),
        "action_horizon": action_horizon,
        "model_action_dim": model_action_dim,
        "server_source_sha256": sha256_file(server_source),
        "python": sys.version,
        "platform": platform.platform(),
    }


def public_attestation(attestation: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in attestation.items()
        if key not in {"checkpoint_files", "checkpoint_local_path"}
    }


def _write_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--policy-config", default=DEFAULT_POLICY_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--openpi-root", default="/app")
    parser.add_argument("--expected-openpi-commit", default=OPENPI_COMMIT)
    parser.add_argument("--expected-openpi-base-commit", default=OPENPI_COMMIT)
    parser.add_argument("--attestation-output", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    openpi_root = pathlib.Path(args.openpi_root).resolve()
    commit = _command_output(("git", "rev-parse", "HEAD"), openpi_root)
    if commit != args.expected_openpi_commit:
        raise RuntimeError(
            "OpenPI commit mismatch: expected %s, got %s"
            % (args.expected_openpi_commit, commit)
        )
    merge_base = _command_output(
        ("git", "merge-base", args.expected_openpi_base_commit, commit),
        openpi_root,
    )
    if merge_base != args.expected_openpi_base_commit:
        raise RuntimeError(
            "OpenPI extension is not based on the expected upstream commit: "
            f"expected {args.expected_openpi_base_commit}, merge-base {merge_base}"
        )
    tracked_status = _command_output(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        openpi_root,
    )
    if tracked_status:
        raise RuntimeError("OpenPI worktree must be clean")
    submodule_status = _command_output(
        ("git", "submodule", "status", "--recursive"), openpi_root
    )
    if not _submodules_are_clean(submodule_status):
        raise RuntimeError("OpenPI submodules must be initialized at recorded commits")

    from openpi.policies import policy_config as openpi_policy_config
    from openpi.serving import websocket_policy_server
    from openpi.shared import download
    from openpi.training import config as training_config

    logging.info("Resolving checkpoint %s", args.checkpoint)
    local_checkpoint = pathlib.Path(str(download.maybe_download(args.checkpoint)))
    logging.info("Hashing checkpoint content at %s", local_checkpoint)
    content_manifest = checkpoint_content_manifest(local_checkpoint)
    config = training_config.get_config(args.policy_config)
    action_horizon = int(config.model.action_horizon)
    model_action_dim = int(config.model.action_dim)
    if action_horizon != PI05_ACTION_HORIZON:
        raise ValueError("attested pi05_libero config must have action_horizon=10")
    if model_action_dim != PI05_MODEL_ACTION_DIM:
        raise ValueError("attested pi05_libero config must have action_dim=32")
    logging.info("Loading policy config=%s", args.policy_config)
    policy = openpi_policy_config.create_trained_policy(config, local_checkpoint)
    attestation = build_attestation(
        openpi_root=openpi_root,
        policy_config=args.policy_config,
        checkpoint_uri=args.checkpoint,
        checkpoint_manifest=content_manifest,
        server_source=pathlib.Path(__file__).resolve(),
        action_horizon=action_horizon,
        model_action_dim=model_action_dim,
        upstream_base_commit=args.expected_openpi_base_commit,
    )
    _write_json(pathlib.Path(args.attestation_output), attestation)
    metadata = dict(policy.metadata)
    metadata["armbench_server_attestation"] = public_attestation(attestation)
    metadata[POLICY_SAMPLING_METADATA_FIELD] = policy_sampling_contract(
        action_horizon, model_action_dim
    )
    metadata[POLICY_CONDITIONING_METADATA_FIELD] = policy_conditioning_contract(
        action_horizon, PI05_RAW_ACTION_DIM, model_action_dim
    )
    served_policy = KeyedPolicySamplingWrapper(
        policy,
        action_horizon=action_horizon,
        model_action_dim=model_action_dim,
    )
    logging.info(
        "Serving attested policy checkpoint_sha256=%s",
        attestation["checkpoint_content_sha256"],
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=served_policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    raise SystemExit(main())
