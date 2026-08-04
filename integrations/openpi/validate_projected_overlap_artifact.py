"""Independently validate projected-overlap transition evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from integrations.openpi.action_chunk_transition import (
    ARCHIVE_SCHEMA_VERSION,
    ActionChunkTransitionError,
    canonical_action_sha256,
    load_transition_archive,
    validate_transition_arrays,
)


VALIDATION_SCHEMA_VERSION = "armbench.projected_overlap_transition_validation.v1"
REQUIRED_FILES = {
    "environment.json",
    "episodes.json",
    "progress.json",
    "queries.json",
    "resolved_protocol.json",
    "summary.json",
    "summary.md",
    "transition_descriptor.json",
    "transitions.npz",
}


class ProjectedOverlapArtifactError(ValueError):
    pass


def _load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ProjectedOverlapArtifactError(f"nonfinite JSON token: {token}")
            ),
        )
    except ProjectedOverlapArtifactError:
        raise
    except Exception as exc:
        raise ProjectedOverlapArtifactError(f"cannot read JSON: {path.name}") from exc


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_manifest(root: pathlib.Path) -> Mapping[str, Any]:
    manifest = _load_json(root / "manifest.json")
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != "armbench.root_manifest.v1":
        raise ProjectedOverlapArtifactError("root manifest schema mismatch")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ProjectedOverlapArtifactError("root manifest files must be an array")
    recorded = []
    for item in files:
        if not isinstance(item, Mapping):
            raise ProjectedOverlapArtifactError("root manifest record must be an object")
        relative = item.get("path")
        if not isinstance(relative, str):
            raise ProjectedOverlapArtifactError("root manifest path must be a string")
        pure = pathlib.PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
            raise ProjectedOverlapArtifactError("unsafe root manifest path")
        path = root.joinpath(*pure.parts)
        if path.is_symlink() or not path.is_file():
            raise ProjectedOverlapArtifactError(f"manifest path is not a regular file: {relative}")
        if type(item.get("bytes")) is not int or path.stat().st_size != item["bytes"]:
            raise ProjectedOverlapArtifactError(f"manifest byte count mismatch: {relative}")
        digest = item.get("sha256")
        if not isinstance(digest, str) or _sha256_file(path) != digest:
            raise ProjectedOverlapArtifactError(f"manifest SHA-256 mismatch: {relative}")
        recorded.append(relative)
    if len(recorded) != len(set(recorded)):
        raise ProjectedOverlapArtifactError("root manifest contains duplicate paths")
    actual = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    if recorded != actual:
        raise ProjectedOverlapArtifactError("root manifest inventory is incomplete")
    if not REQUIRED_FILES.issubset(recorded):
        raise ProjectedOverlapArtifactError("root manifest omits transition evidence")
    canonical = json.dumps(
        files, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    if hashlib.sha256(canonical).hexdigest() != manifest.get("files_sha256"):
        raise ProjectedOverlapArtifactError("root manifest aggregate hash mismatch")
    return manifest


def _sampling_key(protocol: Mapping[str, Any], query: Mapping[str, Any]) -> str:
    payload = {
        "schema_version": "armbench.policy_sampling.v1",
        "namespace": "scored",
        "seed": protocol["sampling_seed"],
        "pairing_key": [
            query["task_suite"],
            query["task_id"],
            query["episode_index"],
            query["execute_horizon"],
        ],
        "query_index": query["query_index"],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sampling_noise_hash(key_sha256: str) -> str:
    digest = bytes.fromhex(key_sha256)
    seed_words = [
        int.from_bytes(digest[index : index + 4], "big")
        for index in range(0, 32, 4)
    ]
    random = np.random.RandomState(seed_words)
    noise = np.asarray(random.standard_normal((10, 32)), dtype="<f4", order="C")
    return hashlib.sha256(noise.tobytes(order="C")).hexdigest()


def _validate_audit_fields(
    protocol: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    queries: Sequence[Mapping[str, Any]],
) -> None:
    scheduler = descriptor["scheduler"]
    delay = int(scheduler["inference_delay"])
    expected_mask_hash = hashlib.sha256(
        np.asarray(np.arange(int(scheduler["action_horizon"])) < delay, dtype="u1").tobytes()
    ).hexdigest()
    hex64 = set("0123456789abcdef")
    for index, query in enumerate(queries):
        key = query.get("sampling_key_sha256")
        noise = query.get("sampling_noise_sha256")
        if key != _sampling_key(protocol, query):
            raise ProjectedOverlapArtifactError("sampling key is not reproducible")
        if noise != _sampling_noise_hash(key):
            raise ProjectedOverlapArtifactError("sampling noise hash is not reproducible")
        conditioned = query.get("method") == "projected_overlap" and not query.get("bootstrap")
        fields = (
            "condition_raw_actions_sha256",
            "condition_model_actions_sha256",
            "condition_mask_sha256",
            "max_model_residual",
        )
        if conditioned:
            if query.get("condition_raw_actions_sha256") != canonical_action_sha256(
                arrays["previous_reference"][index]
            ):
                raise ProjectedOverlapArtifactError("conditioning action hash mismatch")
            model_hash = query.get("condition_model_actions_sha256")
            if not isinstance(model_hash, str) or len(model_hash) != 64 or set(model_hash) - hex64:
                raise ProjectedOverlapArtifactError("model conditioning hash is invalid")
            if query.get("condition_mask_sha256") != expected_mask_hash:
                raise ProjectedOverlapArtifactError("conditioning mask hash mismatch")
            residual = float(query.get("max_model_residual"))
            if not np.isfinite(residual) or residual >= 1e-6:
                raise ProjectedOverlapArtifactError("conditioning residual failed")
        elif any(query.get(field) is not None for field in fields):
            raise ProjectedOverlapArtifactError("unconditioned query contains conditioning evidence")


def validate_artifact(root: pathlib.Path) -> Dict[str, Any]:
    root = root.resolve()
    _validate_manifest(root)
    protocol = _load_json(root / "resolved_protocol.json")
    environment = _load_json(root / "environment.json")
    queries = _load_json(root / "queries.json")
    descriptor = _load_json(root / "transition_descriptor.json")
    if not all(isinstance(value, Mapping) for value in (protocol, environment, descriptor)):
        raise ProjectedOverlapArtifactError("artifact metadata roots must be objects")
    if not isinstance(queries, list) or not all(isinstance(row, Mapping) for row in queries):
        raise ProjectedOverlapArtifactError("queries must be an array of objects")
    if descriptor.get("schema_version") != ARCHIVE_SCHEMA_VERSION:
        raise ProjectedOverlapArtifactError("transition descriptor schema mismatch")
    archive = descriptor.get("archive")
    if not isinstance(archive, Mapping) or archive.get("path") != "transitions.npz":
        raise ProjectedOverlapArtifactError("transition archive descriptor is invalid")
    archive_path = root / "transitions.npz"
    if archive.get("sha256") != _sha256_file(archive_path) or archive.get("bytes") != archive_path.stat().st_size:
        raise ProjectedOverlapArtifactError("transition archive identity mismatch")
    policy = descriptor.get("policy")
    scheduler = descriptor.get("scheduler")
    adapter = descriptor.get("action_adapter")
    if not all(isinstance(value, Mapping) for value in (policy, scheduler, adapter)):
        raise ProjectedOverlapArtifactError("transition identity sections are missing")
    expected_policy = {
        "family": "pi0.5",
        "config": protocol.get("policy_config"),
        "checkpoint": protocol.get("checkpoint"),
        "checkpoint_content_sha256": protocol.get("checkpoint_content_sha256"),
    }
    if dict(policy) != expected_policy:
        raise ProjectedOverlapArtifactError("transition policy identity mismatch")
    expected_scheduler = {
        "action_horizon": 10,
        "action_dim": 7,
        "execute_horizon": 5,
        "inference_delay": 4,
    }
    if dict(scheduler) != expected_scheduler or (
        scheduler.get("action_horizon") != protocol.get("action_horizon")
        or scheduler.get("execute_horizon") != protocol.get("execute_horizon")
        or scheduler.get("inference_delay") != protocol.get("inference_delay_steps")
    ):
        raise ProjectedOverlapArtifactError("transition scheduler identity mismatch")
    expected_adapter = {
        "name": "openpi.pi05_libero.raw_action_v1",
        "version": 1,
        "action_space_id": "libero.ee_delta_pose_gripper.v1",
        "source": "integrations/openpi/libero_runtime.py",
        "source_sha256": adapter.get("source_sha256"),
    }
    if dict(adapter) != expected_adapter:
        raise ProjectedOverlapArtifactError("action adapter identity mismatch")
    source_hashes = environment.get("source_sha256")
    if not isinstance(source_hashes, Mapping) or adapter.get("source_sha256") != source_hashes.get(
        adapter.get("source")
    ):
        raise ProjectedOverlapArtifactError("action adapter source identity mismatch")
    try:
        arrays = load_transition_archive(archive_path)
        report = validate_transition_arrays(descriptor, arrays, queries)
    except ActionChunkTransitionError as exc:
        raise ProjectedOverlapArtifactError(str(exc)) from exc
    _validate_audit_fields(protocol, descriptor, arrays, queries)
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "valid": True,
        **{key: value for key, value in report.items() if key not in {"schema_version", "valid"}},
        "manifest_sha256": _sha256_file(root / "manifest.json"),
        "archive_sha256": archive["sha256"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("artifact")
    args = parser.parse_args(argv)
    report = validate_artifact(pathlib.Path(args.artifact))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
