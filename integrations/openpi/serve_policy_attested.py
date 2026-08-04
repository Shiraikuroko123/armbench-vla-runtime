"""Serve an OpenPI checkpoint with auditable model and source provenance."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import pathlib
import platform
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence


ATTESTATION_SCHEMA_VERSION = "armbench.openpi_server_attestation.v1"
OPENPI_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
DEFAULT_POLICY_CONFIG = "pi05_libero"
DEFAULT_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"


def _command_output(command: Sequence[str], cwd: pathlib.Path) -> str:
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


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
    if int(config.model.action_horizon) != 10:
        raise ValueError("attested pi05_libero config must have action_horizon=10")
    logging.info("Loading policy config=%s", args.policy_config)
    policy = openpi_policy_config.create_trained_policy(config, local_checkpoint)
    attestation = build_attestation(
        openpi_root=openpi_root,
        policy_config=args.policy_config,
        checkpoint_uri=args.checkpoint,
        checkpoint_manifest=content_manifest,
        server_source=pathlib.Path(__file__).resolve(),
        action_horizon=int(config.model.action_horizon),
        model_action_dim=int(config.model.action_dim),
    )
    _write_json(pathlib.Path(args.attestation_output), attestation)
    metadata = dict(policy.metadata)
    metadata["armbench_server_attestation"] = public_attestation(attestation)
    logging.info(
        "Serving attested policy checkpoint_sha256=%s",
        attestation["checkpoint_content_sha256"],
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    raise SystemExit(main())
