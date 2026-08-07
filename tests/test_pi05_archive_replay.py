from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from armbench.mujoco_sim.model import default_panda_scene_path
from armbench.vla.cartesian_adapter import LIBERO_ACTION_SPACE_ID
from armbench.vla.pi05_archive_replay import (
    PI05_ACTION_ADAPTER,
    PI05_ACTION_ADAPTER_SOURCE,
    PI05_CHECKPOINT,
    PI05_CHECKPOINT_SHA256,
    PI05_POLICY_CONFIG,
    PI05_POLICY_FAMILY,
    ArchiveReplayConfig,
    Pi05ArchiveReplayError,
    execute_pi05_archive_replay,
    select_stratified_chunks,
    validate_pi05_replay_artifact,
    validate_pi05_source_archive,
)
from armbench.vla.pi05_braking_repair import (
    Pi05BrakingComparisonConfig,
    execute_pi05_braking_comparison,
    validate_pi05_braking_comparison,
)


METHODS = (
    "overlap_unconditioned",
    "projected_overlap",
    "rtc_guided_overlap",
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _action_hash(actions: np.ndarray) -> str:
    values = np.asarray(actions, dtype="<f4", order="C")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _rewrite_manifest(root: Path) -> None:
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _file_hash(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    canonical = json.dumps(
        files, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    _write_json(
        root / "manifest.json",
        {
            "schema_version": "armbench.root_manifest.v1",
            "files": files,
            "files_sha256": hashlib.sha256(canonical).hexdigest(),
        },
    )


def _query(
    *,
    episode_id: str,
    pair_id: str,
    method: str,
    task_id: int,
    bootstrap: bool,
    response: np.ndarray,
    next_reference: np.ndarray,
    latency_ms: float,
) -> dict[str, object]:
    return {
        "bootstrap": bootstrap,
        "episode_id": episode_id,
        "episode_index": 0,
        "pair_id": pair_id,
        "method": method,
        "task_suite": "synthetic_suite",
        "task_id": task_id,
        "query_index": 0 if bootstrap else 1,
        "inference_latency_ms": latency_ms,
        "policy_inference_latency_ms": latency_ms - 5.0,
        "server_inference_latency_ms": latency_ms - 2.0,
        "response_action_sha256": _action_hash(response),
        "next_reference_sha256": _action_hash(next_reference),
        "executed_steps": 0 if bootstrap else 5,
        "old_prefix_steps": 0 if bootstrap else 4,
        "new_suffix_steps": 0 if bootstrap else 1,
    }


def _source_artifact(tmp_path: Path) -> Path:
    root = tmp_path / "source" / "evaluation"
    root.mkdir(parents=True)
    arrays: dict[str, list[np.ndarray | np.generic | int | bool]] = {
        "has_previous_reference": [],
        "previous_reference": [],
        "response_actions": [],
        "executed_window": [],
        "action_source": [],
        "next_reference": [],
        "executed_length": [],
        "old_prefix_steps": [],
        "new_suffix_steps": [],
    }
    queries: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for task_id in range(2):
        for method_index, method in enumerate(METHODS):
            episode_id = f"task_{task_id}__episode_00__{method}"
            pair_id = f"task_{task_id}__episode_00"
            bootstrap = np.zeros((10, 7), dtype="<f4")
            bootstrap[:, 6] = -1.0
            bootstrap[:, 0] = 0.01 * (method_index + 1)
            bootstrap_next = np.concatenate(
                (bootstrap[5:], np.zeros((5, 7), dtype="<f4")), axis=0
            )
            latency_ms = 120.0 if task_id == 0 else 240.0
            queries.append(
                _query(
                    episode_id=episode_id,
                    pair_id=pair_id,
                    method=method,
                    task_id=task_id,
                    bootstrap=True,
                    response=bootstrap,
                    next_reference=bootstrap_next,
                    latency_ms=latency_ms,
                )
            )

            response = np.zeros((10, 7), dtype="<f4")
            response[:, 0] = 0.08 + method_index * 0.01
            response[:, 6] = 1.0
            if task_id == 1:
                response[0, 0] = 2.0
            next_reference = np.concatenate(
                (response[5:], np.zeros((5, 7), dtype="<f4")), axis=0
            )
            executed = np.concatenate(
                (bootstrap[:4], response[4:5]), axis=0
            ).astype("<f4")
            queries.append(
                _query(
                    episode_id=episode_id,
                    pair_id=pair_id,
                    method=method,
                    task_id=task_id,
                    bootstrap=False,
                    response=response,
                    next_reference=next_reference,
                    latency_ms=latency_ms,
                )
            )
            rows.append(
                {
                    "episode_id": episode_id,
                    "pair_id": pair_id,
                    "method": method,
                    "query_index": 1,
                }
            )
            arrays["has_previous_reference"].append(True)
            arrays["previous_reference"].append(bootstrap)
            arrays["response_actions"].append(response)
            arrays["executed_window"].append(executed)
            arrays["action_source"].append(
                np.array([1, 1, 1, 1, 2], dtype=np.uint8)
            )
            arrays["next_reference"].append(next_reference)
            arrays["executed_length"].append(5)
            arrays["old_prefix_steps"].append(4)
            arrays["new_suffix_steps"].append(1)

    archive_arrays = {
        "has_previous_reference": np.asarray(
            arrays["has_previous_reference"], dtype=np.bool_
        ),
        "previous_reference": np.stack(arrays["previous_reference"]),
        "response_actions": np.stack(arrays["response_actions"]),
        "executed_window": np.stack(arrays["executed_window"]),
        "action_source": np.stack(arrays["action_source"]),
        "next_reference": np.stack(arrays["next_reference"]),
        "executed_length": np.asarray(arrays["executed_length"], dtype="<i4"),
        "old_prefix_steps": np.asarray(arrays["old_prefix_steps"], dtype="<i4"),
        "new_suffix_steps": np.asarray(arrays["new_suffix_steps"], dtype="<i4"),
    }
    archive_path = root / "transitions.npz"
    np.savez_compressed(archive_path, **archive_arrays)
    source_hash = "a" * 64
    descriptor = {
        "schema_version": "armbench.action_chunk_transition_archive.v1",
        "policy": {
            "family": PI05_POLICY_FAMILY,
            "config": PI05_POLICY_CONFIG,
            "checkpoint": PI05_CHECKPOINT,
            "checkpoint_content_sha256": PI05_CHECKPOINT_SHA256,
        },
        "action_adapter": {
            "action_space_id": LIBERO_ACTION_SPACE_ID,
            "name": PI05_ACTION_ADAPTER,
            "source": PI05_ACTION_ADAPTER_SOURCE,
            "source_sha256": source_hash,
            "version": 1,
        },
        "scheduler": {
            "action_dim": 7,
            "action_horizon": 10,
            "execute_horizon": 5,
            "inference_delay": 4,
        },
        "archive": {
            "path": "transitions.npz",
            "bytes": archive_path.stat().st_size,
            "sha256": _file_hash(archive_path),
        },
        "rows": rows,
    }
    environment = {
        "server_attestation": {
            "policy_loaded": True,
            "policy_config": PI05_POLICY_CONFIG,
            "checkpoint_uri": PI05_CHECKPOINT,
            "checkpoint_content_sha256": PI05_CHECKPOINT_SHA256,
            "openpi_tracked_clean": True,
            "openpi_submodules_clean": True,
        },
        "source_sha256": {PI05_ACTION_ADAPTER_SOURCE: source_hash},
    }
    _write_json(root / "queries.json", queries)
    _write_json(root / "transition_descriptor.json", descriptor)
    _write_json(root / "environment.json", environment)
    _rewrite_manifest(root)
    return root


def test_source_validator_recomputes_every_transition_hash(tmp_path: Path) -> None:
    source = _source_artifact(tmp_path)

    result = validate_pi05_source_archive(source)

    assert result.transition_count == 6
    assert len(result.response_hashes) == 6
    assert result.descriptor["policy"]["config"] == "pi05_libero"


def test_source_validator_rejects_file_tampering(tmp_path: Path) -> None:
    source = _source_artifact(tmp_path)
    with (source / "queries.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")

    with pytest.raises(Pi05ArchiveReplayError, match="file size mismatch"):
        validate_pi05_source_archive(source)


def test_source_validator_rejects_a_rehashed_false_response_hash(
    tmp_path: Path,
) -> None:
    source = _source_artifact(tmp_path)
    queries = json.loads((source / "queries.json").read_text(encoding="utf-8"))
    next(row for row in queries if not row["bootstrap"])[
        "response_action_sha256"
    ] = "b" * 64
    _write_json(source / "queries.json", queries)
    _rewrite_manifest(source)

    with pytest.raises(Pi05ArchiveReplayError, match="response action hash mismatch"):
        validate_pi05_source_archive(source)


def test_stratified_selection_is_balanced_and_deterministic(tmp_path: Path) -> None:
    archive = validate_pi05_source_archive(_source_artifact(tmp_path))

    first = select_stratified_chunks(archive, 6, 17)
    second = select_stratified_chunks(archive, 6, 17)
    strata = {
        (
            int(archive.transition_queries[index]["task_id"]),
            str(archive.transition_queries[index]["method"]),
        )
        for index in first
    }

    assert first == second
    assert len(first) == 6
    assert len(strata) == 6
    with pytest.raises(Pi05ArchiveReplayError, match="divisible"):
        select_stratified_chunks(archive, 5, 17)


def test_replay_writes_self_validating_claim_bounded_artifact(
    tmp_path: Path,
) -> None:
    try:
        default_panda_scene_path()
    except FileNotFoundError as error:
        pytest.skip(str(error))
    source = _source_artifact(tmp_path)
    output = tmp_path / "replay"

    execute_pi05_archive_replay(
        source,
        output,
        ArchiveReplayConfig(
            chunk_count=6,
            selection_seed=19,
            scenarios=("free_space",),
            deadline_ms=200.0,
        ),
    )
    validation = validate_pi05_replay_artifact(output, source)
    provenance = json.loads(
        (output / "provenance.json").read_text(encoding="utf-8")
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))

    assert validation["valid"] is True
    assert validation["chunks"] == 6
    assert validation["cases"] == 6
    assert "source_archive_reverified" in validation["checks"]
    assert provenance["source_policy_checkpoint_attested"] is True
    assert provenance["policy_checkpoint_executed_in_replay"] is False
    assert provenance["task_success_evaluated"] is False
    assert provenance["panda_closed_loop_executed"] is False
    assert summary["overall"]["deadline_exceeded_cases"] == 3
    assert summary["overall"]["guard_safe_cases"] == 6


def test_replay_refuses_to_overwrite_an_existing_directory(tmp_path: Path) -> None:
    source = _source_artifact(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="must not already exist"):
        execute_pi05_archive_replay(source, output)


def test_paired_braking_comparison_binds_trajectories_to_source(
    tmp_path: Path,
) -> None:
    try:
        default_panda_scene_path()
    except FileNotFoundError as error:
        pytest.skip(str(error))
    source = _source_artifact(tmp_path)
    output = tmp_path / "braking-comparison"

    execute_pi05_braking_comparison(
        source,
        output,
        Pi05BrakingComparisonConfig(
            chunk_count=6,
            selection_seed=23,
            scenarios=("free_space",),
            repair_selection_deadline_ms=1000.0,
        ),
    )
    validation = validate_pi05_braking_comparison(output, source)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    with np.load(output / "trajectories.npz", allow_pickle=False) as trace:
        assert trace["raw_positions"].shape == (6, 11, 7)
        assert trace["legacy_positions"].shape == (6, 11, 7)
        assert trace["repair_positions"].shape == (6, 11, 7)

    assert validation["valid"] is True
    assert validation["checks"][-1] == "source_archive_reverified"
    assert summary["overall"]["cases"] == 6
    assert summary["overall"]["repair_regressions"] == 0
