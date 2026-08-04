from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from integrations.openpi.action_chunk_transition import (
    ARCHIVE_SCHEMA_VERSION,
    build_action_chunk_transition,
    canonical_action_sha256,
    load_transition_archive,
    write_transition_archive,
)
from integrations.openpi.validate_projected_overlap_artifact import (
    ProjectedOverlapArtifactError,
    _sampling_key,
    _sampling_noise_hash,
    validate_artifact,
)


def _write_json(path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_manifest(root) -> None:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("ascii")
    _write_json(
        root / "manifest.json",
        {
            "schema_version": "armbench.root_manifest.v1",
            "files": files,
            "files_sha256": hashlib.sha256(encoded).hexdigest(),
        },
    )


def _artifact(root):
    root.mkdir()
    response0 = np.arange(70, dtype=np.float32).reshape(10, 7)
    next0 = np.concatenate((response0[5:], np.zeros((5, 7), dtype=np.float32)))
    response1 = response0 + 1.0
    next1 = np.concatenate((response1[5:], np.zeros((5, 7), dtype=np.float32)))
    transitions = [
        build_action_chunk_transition(
            None,
            response0,
            next0,
            inference_delay=4,
            execute_horizon=5,
            executed_old=0,
            executed_new=5,
        ),
        build_action_chunk_transition(
            next0,
            response1,
            next1,
            inference_delay=4,
            execute_horizon=5,
            executed_old=4,
            executed_new=1,
        ),
    ]
    archive_path = root / "transitions.npz"
    write_transition_archive(archive_path, transitions)
    source = "integrations/openpi/libero_runtime.py"
    source_hash = "c" * 64
    protocol = {
        "policy_config": "pi05_libero",
        "checkpoint": "gs://openpi-assets/checkpoints/pi05_libero",
        "checkpoint_content_sha256": "d" * 64,
        "action_horizon": 10,
        "execute_horizon": 5,
        "inference_delay_steps": 4,
        "sampling_seed": 9,
    }
    rows = [
        {"episode_id": "e", "pair_id": "p", "method": "projected_overlap", "query_index": 0},
        {"episode_id": "e", "pair_id": "p", "method": "projected_overlap", "query_index": 1},
    ]
    descriptor = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "policy": {
            "family": "pi0.5",
            "config": protocol["policy_config"],
            "checkpoint": protocol["checkpoint"],
            "checkpoint_content_sha256": protocol["checkpoint_content_sha256"],
        },
        "action_adapter": {
            "name": "openpi.pi05_libero.raw_action_v1",
            "version": 1,
            "action_space_id": "libero.ee_delta_pose_gripper.v1",
            "source": source,
            "source_sha256": source_hash,
        },
        "scheduler": {
            "action_horizon": 10,
            "action_dim": 7,
            "execute_horizon": 5,
            "inference_delay": 4,
        },
        "archive": {
            "path": "transitions.npz",
            "bytes": archive_path.stat().st_size,
            "sha256": _sha256(archive_path),
        },
        "rows": rows,
    }
    mask_hash = hashlib.sha256(
        np.asarray(np.arange(10) < 4, dtype="u1").tobytes()
    ).hexdigest()
    queries = []
    for index, (row, response, next_reference) in enumerate(
        zip(rows, (response0, response1), (next0, next1))
    ):
        query = {
            **row,
            "task_suite": "libero_10",
            "task_id": 0,
            "episode_index": 0,
            "execute_horizon": 5,
            "bootstrap": index == 0,
            "executed_steps": 5,
            "old_prefix_steps": 0 if index == 0 else 4,
            "new_suffix_steps": 5 if index == 0 else 1,
            "response_action_sha256": canonical_action_sha256(response),
            "next_reference_sha256": canonical_action_sha256(next_reference),
            "sampling_noise_sha256": None,
            "condition_raw_actions_sha256": None,
            "condition_model_actions_sha256": None,
            "condition_mask_sha256": None,
            "max_model_residual": None,
        }
        query["sampling_key_sha256"] = _sampling_key(protocol, query)
        query["sampling_noise_sha256"] = _sampling_noise_hash(
            query["sampling_key_sha256"]
        )
        if index == 1:
            query.update(
                {
                    "condition_raw_actions_sha256": canonical_action_sha256(next0),
                    "condition_model_actions_sha256": "b" * 64,
                    "condition_mask_sha256": mask_hash,
                    "max_model_residual": 0.0,
                }
            )
        queries.append(query)
    _write_json(root / "resolved_protocol.json", protocol)
    _write_json(root / "environment.json", {"source_sha256": {source: source_hash}})
    _write_json(root / "queries.json", queries)
    _write_json(root / "transition_descriptor.json", descriptor)
    _write_json(root / "episodes.json", [])
    _write_json(root / "progress.json", {})
    _write_json(root / "summary.json", {})
    (root / "summary.md").write_text("fixture\n", encoding="utf-8")
    _refresh_manifest(root)
    return root


def test_independent_transition_artifact_validator(tmp_path) -> None:
    report = validate_artifact(_artifact(tmp_path / "artifact"))

    assert report["valid"] is True
    assert report["transition_count"] == 2
    assert report["episode_count"] == 1


def test_validator_rejects_resigned_transition_tampering(tmp_path) -> None:
    root = _artifact(tmp_path / "artifact")
    archive_path = root / "transitions.npz"
    arrays = load_transition_archive(archive_path)
    arrays["executed_window"][1, 4, 0] += 1.0
    with archive_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    descriptor_path = root / "transition_descriptor.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["archive"].update(
        {"bytes": archive_path.stat().st_size, "sha256": _sha256(archive_path)}
    )
    _write_json(descriptor_path, descriptor)
    _refresh_manifest(root)

    with pytest.raises(ProjectedOverlapArtifactError, match="executed window"):
        validate_artifact(root)
