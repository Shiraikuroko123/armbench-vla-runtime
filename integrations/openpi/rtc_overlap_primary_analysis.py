"""Combine two frozen held-out RTC overlap artifacts into one strict analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import pathlib
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from integrations.openpi import rtc_overlap_analysis as pilot_analysis
from integrations.openpi import rtc_overlap_pilot


ANALYSIS_SCHEMA_VERSION = "armbench.pi05_rtc_overlap_primary_analysis.v2"
SOURCE_SCHEMA_VERSION = rtc_overlap_pilot.SCHEMA_VERSION
METHODS = tuple(rtc_overlap_pilot.V2_OVERLAP_METHODS)
BASELINE = rtc_overlap_pilot.OVERLAP_UNCONDITIONED
CANDIDATES = (
    rtc_overlap_pilot.PROJECTED_OVERLAP,
    rtc_overlap_pilot.RTC_GUIDED_OVERLAP,
)
TASK_SUITE = "libero_10"
TASK_IDS = tuple(range(10))
EPISODE_INDICES = tuple(range(2, 7))
SAMPLING_SEEDS = (20260806, 20260807)
RUN_ID_BY_SEED = {
    seed: "pi05_rtc_overlap_primary_v3_seed_%d_001" % seed
    for seed in SAMPLING_SEEDS
}
ACTION_HORIZON = 10
EXECUTE_HORIZON = 5
INFERENCE_DELAY_STEPS = 4
ROLL_OUTS_PER_ARTIFACT = 150
TRIPLETS_PER_ARTIFACT = 50
TOTAL_ROLLOUTS = 300
TOTAL_TRIPLETS = 100
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260805
FROZEN_ARMBENCH_COMMIT = "44c358731c5493284b74bb29eefa7d538d0f38dd"
FROZEN_OPENPI_UPSTREAM_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
FROZEN_OPENPI_EXTENSION_COMMIT = "54592c7148ba69bf52757385502782f80f2285e0"
FROZEN_EXTERNAL_PROTOCOL_COMMIT = "509f6f4cbcc9e8b02804edf640e565673d4a3855"
FROZEN_POLICY_CONFIG = "pi05_libero"
FROZEN_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
FROZEN_CHECKPOINT_CONTENT_SHA256 = (
    "9cd1b00d402cc0447454dad6054dcc6f019b53e498469f209d2b749d4487e1d5"
)
ANALYZER_SOURCE = "integrations/openpi/rtc_overlap_primary_analysis.py"
HELPER_SOURCE = "integrations/openpi/rtc_overlap_analysis.py"
VALIDATOR_SOURCE = "integrations/openpi/rtc_overlap_pilot.py"
SEAM_METRICS = ("seam_motion_l2", "seam_gripper_abs")

AnalysisError = pilot_analysis.AnalysisError

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_SNAPSHOT_PATHS = (
    "manifest.json",
    "resolved_protocol.json",
    "environment.json",
    "episodes.json",
    "queries.json",
    "progress.json",
    "summary.json",
    "transition_descriptor.json",
)
_IDENTITY_FIELDS = (
    "schema_version",
    "episode_id",
    "pair_id",
    "task_suite",
    "task_id",
    "episode_index",
    "method",
    "condition_order",
    "execute_horizon",
    "inference_delay_steps",
)
_PROTOCOL_IDENTITY_FIELDS = tuple(
    field for field in _IDENTITY_FIELDS if field != "schema_version"
)
_CONDITION_GUIDANCE_FIELDS = (
    "condition_raw_actions_sha256",
    "condition_model_actions_sha256",
    "condition_mask_sha256",
    "max_model_residual",
    "guidance_raw_actions_sha256",
    "guidance_model_actions_sha256",
    "guidance_weights_sha256",
    "max_weighted_model_residual",
    "weighted_model_rmse",
)
_OUTPUT_FILES = frozenset(("analysis.json", "per_triplet.csv", "summary.md"))
_PER_TRIPLET_FIELDS = (
    (
        "triplet_id",
        "source_artifact",
        "pair_id",
        "task_suite",
        "task_id",
        "episode_index",
        "sampling_seed",
        "initial_state_sha256",
    )
    + tuple(
        field
        for method in METHODS
        for field in (
            "%s_condition_order" % method,
            "%s_success" % method,
            "%s_policy_queries" % method,
            "%s_scored_transition_queries" % method,
            "%s_seam_motion_l2" % method,
            "%s_seam_gripper_abs" % method,
        )
    )
    + tuple(
        field
        for candidate in CANDIDATES
        for field in (
            "%s_success_difference" % candidate,
            "%s_seam_motion_l2_difference" % candidate,
            "%s_seam_gripper_abs_difference" % candidate,
        )
    )
)


@dataclass(frozen=True)
class SourceEvidence:
    root: pathlib.Path
    label: str
    sampling_seed: int
    snapshots: Mapping[str, bytes]
    validator_summary: Mapping[str, Any]
    episodes: Mapping[Tuple[int, int, str], Mapping[str, Any]]
    seam_by_episode: Mapping[str, Mapping[str, float]]
    sampling_keys: Mapping[Tuple[int, int, int], str]
    identity: Mapping[str, Any]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise AnalysisError("%s is not a canonical SHA-256" % label)
    return value


def _implementation_identity() -> Dict[str, str]:
    source_root = pathlib.Path(__file__).resolve().parents[2]
    return {
        "analyzer_source": ANALYZER_SOURCE,
        "analyzer_sha256": _sha256_file(
            source_root / pathlib.PurePosixPath(ANALYZER_SOURCE)
        ),
        "pilot_helper_source": HELPER_SOURCE,
        "pilot_helper_sha256": _sha256_file(
            source_root / pathlib.PurePosixPath(HELPER_SOURCE)
        ),
        "validator_source": VALIDATOR_SOURCE,
        "validator_sha256": _sha256_file(
            source_root / pathlib.PurePosixPath(VALIDATOR_SOURCE)
        ),
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy_version": np.__version__,
    }


def _portable_source_label(root: pathlib.Path, sampling_seed: int) -> str:
    repository_root = pathlib.Path(__file__).resolve().parents[2]
    try:
        base = root.relative_to(repository_root).as_posix()
    except ValueError:
        parts = root.parts[-2:]
        base = pathlib.PurePosixPath(*parts).as_posix()
    return "%s#sampling_seed=%d" % (base, sampling_seed)


def _strict_json(snapshot: bytes, label: str, expected: type) -> Any:
    return pilot_analysis._strict_json_bytes(snapshot, label, expected)


def _require_fields(row: Mapping[str, Any], fields: Sequence[str], label: str) -> None:
    pilot_analysis._require_fields(row, fields, label)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    return pilot_analysis._integer(value, label, minimum=minimum)


def _finite_nonnegative(value: Any, label: str) -> float:
    return pilot_analysis._finite_nonnegative(value, label)


def _validate_all_first(
    artifacts: Sequence[pathlib.Path],
) -> Tuple[List[pathlib.Path], List[Mapping[str, Any]]]:
    if len(artifacts) != 2:
        raise AnalysisError(
            "exactly two corrected v3 evaluation artifacts are required"
        )
    raw_roots = [pathlib.Path(value) for value in artifacts]
    if any(root.is_symlink() for root in raw_roots):
        raise AnalysisError("source artifacts must not be symbolic links")
    roots = [root.resolve() for root in raw_roots]
    if len(set(roots)) != 2:
        raise AnalysisError("the two source artifacts must be distinct directories")
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            raise AnalysisError(
                "source artifact must be a regular directory: %s" % root
            )

    validated: List[Mapping[str, Any]] = []
    for root in roots:
        try:
            result = rtc_overlap_pilot.validate_artifact(root)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise AnalysisError(
                "source artifact failed independent validation: %s" % exc
            ) from exc
        if not isinstance(result, Mapping):
            raise AnalysisError("source validator returned a non-object result")
        validated.append(result)
    return roots, validated


def _snapshot_sources(
    roots: Sequence[pathlib.Path],
    first_validation: Sequence[Mapping[str, Any]],
) -> List[Tuple[Mapping[str, bytes], Mapping[str, Any]]]:
    output: List[Tuple[Mapping[str, bytes], Mapping[str, Any]]] = []
    for root, first in zip(roots, first_validation):
        snapshots: Dict[str, bytes] = {}
        try:
            for relative in _SNAPSHOT_PATHS:
                path = root / relative
                if path.is_symlink() or not path.is_file():
                    raise AnalysisError(
                        "source artifact has a missing or non-regular %s" % relative
                    )
                snapshots[relative] = path.read_bytes()
        except OSError as exc:
            raise AnalysisError("cannot snapshot source artifact: %s" % exc) from exc
        try:
            second = rtc_overlap_pilot.validate_artifact(root)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise AnalysisError(
                "source artifact changed or became invalid: %s" % exc
            ) from exc
        if not isinstance(second, Mapping) or first != second:
            raise AnalysisError("source validator result changed while reading")
        for relative, before in snapshots.items():
            try:
                after = (root / relative).read_bytes()
            except OSError as exc:
                raise AnalysisError(
                    "source artifact changed while reading: %s" % relative
                ) from exc
            if after != before:
                raise AnalysisError(
                    "source artifact changed while reading: %s" % relative
                )
        output.append((snapshots, first))
    return output


def _expected_cells() -> Dict[Tuple[int, int, str], int]:
    output: Dict[Tuple[int, int, str], int] = {}
    pair_index = 0
    for task_id in TASK_IDS:
        for episode_index in EPISODE_INDICES:
            offset = pair_index % len(METHODS)
            order = METHODS[offset:] + METHODS[:offset]
            for condition_order, method in enumerate(order):
                output[(task_id, episode_index, method)] = condition_order
            pair_index += 1
    return output


def _pair_id(task_id: int, episode_index: int) -> str:
    return "%s__task_%02d__episode_%02d" % (
        TASK_SUITE,
        task_id,
        episode_index,
    )


def _validate_protocol(protocol: Mapping[str, Any]) -> int:
    expected_scalars = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "pilot_only": True,
        "policy_config": FROZEN_POLICY_CONFIG,
        "checkpoint": FROZEN_CHECKPOINT,
        "checkpoint_content_sha256": FROZEN_CHECKPOINT_CONTENT_SHA256,
        "openpi_upstream_commit": FROZEN_OPENPI_UPSTREAM_COMMIT,
        "openpi_extension_commit": FROZEN_OPENPI_EXTENSION_COMMIT,
        "task_suite": TASK_SUITE,
        "execute_horizon": EXECUTE_HORIZON,
        "inference_delay_steps": INFERENCE_DELAY_STEPS,
        "action_horizon": ACTION_HORIZON,
        "control_period_ms": 50.0,
        "fixed_delay_ms": 200.0,
        "bootstrap_rule": (
            "query 0 generates an unexecuted reference; query 1 samples from "
            "the same observation"
        ),
        "scheduler": "old[:d] + new[d:E], then new[E:H] + zeros(E)",
        "planned_rollouts": ROLL_OUTS_PER_ARTIFACT,
        "complete_triplets": TRIPLETS_PER_ARTIFACT,
    }
    for field, expected in expected_scalars.items():
        if protocol.get(field) != expected:
            raise AnalysisError("frozen protocol identity mismatch: %s" % field)
    expected_lists = {
        "task_ids": list(TASK_IDS),
        "episode_indices": list(EPISODE_INDICES),
        "methods": list(METHODS),
        "pairing_key_fields": [
            "task_suite",
            "task_id",
            "episode_index",
            "execute_horizon",
            "query_index",
        ],
    }
    for field, expected in expected_lists.items():
        if protocol.get(field) != expected:
            raise AnalysisError("frozen protocol matrix mismatch: %s" % field)
    if protocol.get("video_mode") not in ("failures", "all"):
        raise AnalysisError("held-out artifacts must preserve required videos")
    sampling_seed = _integer(protocol.get("sampling_seed"), "protocol.sampling_seed")
    if sampling_seed not in SAMPLING_SEEDS:
        raise AnalysisError("sampling_seed is outside the frozen held-out pair")

    matrix = protocol.get("matrix")
    if not isinstance(matrix, list) or len(matrix) != ROLL_OUTS_PER_ARTIFACT:
        raise AnalysisError("protocol matrix must contain exactly 150 cells")
    expected_cells = _expected_cells()
    seen = set()
    for index, raw in enumerate(matrix):
        label = "protocol.matrix[%d]" % index
        if not isinstance(raw, Mapping):
            raise AnalysisError("%s must be an object" % label)
        _require_fields(raw, _PROTOCOL_IDENTITY_FIELDS, label)
        task_id = _integer(raw["task_id"], "%s.task_id" % label)
        episode_index = _integer(raw["episode_index"], "%s.episode_index" % label)
        method = raw["method"]
        key = (task_id, episode_index, method)
        if key in seen:
            raise AnalysisError("protocol matrix contains duplicate cell %r" % (key,))
        seen.add(key)
        if key not in expected_cells:
            raise AnalysisError("protocol matrix contains unexpected cell %r" % (key,))
        pair_id = _pair_id(task_id, episode_index)
        if raw["task_suite"] != TASK_SUITE or raw["pair_id"] != pair_id:
            raise AnalysisError("%s pair identity mismatch" % label)
        if raw["episode_id"] != "%s__%s" % (pair_id, method):
            raise AnalysisError("%s episode identity mismatch" % label)
        if raw["condition_order"] != expected_cells[key]:
            raise AnalysisError("%s violates the frozen Latin rotation" % label)
        if raw["execute_horizon"] != EXECUTE_HORIZON:
            raise AnalysisError("%s execute horizon mismatch" % label)
        if raw["inference_delay_steps"] != INFERENCE_DELAY_STEPS:
            raise AnalysisError("%s inference delay mismatch" % label)
    if seen != set(expected_cells):
        raise AnalysisError("protocol matrix is incomplete")
    return sampling_seed


def _expected_environment_command_options(sampling_seed: int) -> Dict[str, str]:
    return {
        "--output-dir": "/armbench_results/%s/evaluation"
        % RUN_ID_BY_SEED[sampling_seed],
        "--host": "127.0.0.1",
        "--port": "8001",
        "--openpi-root": "/app",
        "--armbench-root": "/armbench",
        "--task-suite": TASK_SUITE,
        "--task-ids": "all",
        "--episode-indices": "2,3,4,5,6",
        "--sampling-seed": str(sampling_seed),
        "--environment-seed": "7",
        "--video-mode": "failures",
        "--server-startup-timeout-s": "120",
        "--inference-timeout-s": "600",
    }


def _validate_environment_command(value: Any, sampling_seed: int) -> None:
    label = "environment.command"
    if not isinstance(value, list) or not value or any(
        not isinstance(token, str) or not token for token in value
    ):
        raise AnalysisError("%s must be a nonempty string list" % label)
    if value[0] != "/armbench/integrations/openpi/rtc_overlap_pilot.py":
        raise AnalysisError("%s program identity mismatch" % label)
    if len(value) < 2 or value[1] != "run":
        raise AnalysisError("%s must invoke the run command" % label)

    expected = _expected_environment_command_options(sampling_seed)
    observed: Dict[str, str] = {}
    index = 2
    while index < len(value):
        option = value[index]
        if not option.startswith("--"):
            raise AnalysisError("%s contains an unexpected positional token" % label)
        if option == "--max-task-steps":
            raise AnalysisError("%s must not contain --max-task-steps" % label)
        if option not in expected:
            raise AnalysisError("%s contains unknown option %s" % (label, option))
        if option in observed:
            raise AnalysisError("%s contains duplicate option %s" % (label, option))
        if index + 1 >= len(value) or value[index + 1].startswith("--"):
            raise AnalysisError("%s option %s has no value" % (label, option))
        observed[option] = value[index + 1]
        index += 2

    missing = sorted(set(expected) - set(observed))
    if missing:
        raise AnalysisError(
            "%s is missing frozen options: %s" % (label, ", ".join(missing))
        )
    for option, expected_value in expected.items():
        if observed[option] != expected_value:
            raise AnalysisError(
                "%s frozen value mismatch for %s" % (label, option)
            )


def _validate_environment(
    environment: Mapping[str, Any], sampling_seed: int
) -> Dict[str, Any]:
    expected = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "armbench_commit": FROZEN_ARMBENCH_COMMIT,
        "armbench_status": "",
        "openpi_commit": FROZEN_OPENPI_EXTENSION_COMMIT,
        "openpi_status": "",
    }
    for field, value in expected.items():
        if environment.get(field) != value:
            raise AnalysisError("frozen environment identity mismatch: %s" % field)
    _validate_environment_command(environment.get("command"), sampling_seed)

    source_hashes = environment.get("source_sha256")
    if not isinstance(source_hashes, Mapping):
        raise AnalysisError("environment source_sha256 must be an object")
    if set(source_hashes) != set(rtc_overlap_pilot.SOURCE_FILES):
        raise AnalysisError("environment source inventory mismatch")
    canonical_source_hashes = {
        str(path): _canonical_sha256(digest, "source_sha256.%s" % path)
        for path, digest in sorted(source_hashes.items())
    }

    attestation = environment.get("server_attestation")
    if not isinstance(attestation, Mapping):
        raise AnalysisError("environment server attestation is missing")
    attested_expected = {
        "schema_version": "armbench.openpi_server_attestation.v1",
        "policy_config": FROZEN_POLICY_CONFIG,
        "checkpoint_uri": FROZEN_CHECKPOINT,
        "checkpoint_content_sha256": FROZEN_CHECKPOINT_CONTENT_SHA256,
        "openpi_commit": FROZEN_OPENPI_EXTENSION_COMMIT,
        "openpi_upstream_base_commit": FROZEN_OPENPI_UPSTREAM_COMMIT,
        "openpi_tracked_clean": True,
        "openpi_submodules_clean": True,
        "action_horizon": ACTION_HORIZON,
    }
    for field, value in attested_expected.items():
        if attestation.get(field) != value:
            raise AnalysisError("frozen server identity mismatch: %s" % field)
    extension_files = attestation.get("openpi_extension_files")
    if not isinstance(extension_files, Mapping) or not extension_files:
        raise AnalysisError("server extension source identity is missing")
    canonical_extension_files = {
        str(path): _canonical_sha256(digest, "openpi_extension_files.%s" % path)
        for path, digest in sorted(extension_files.items())
    }
    server_source = _canonical_sha256(
        attestation.get("server_source_sha256"), "server_source_sha256"
    )
    expected_server_source = canonical_source_hashes[
        "integrations/openpi/serve_policy_attested.py"
    ]
    if server_source != expected_server_source:
        raise AnalysisError("server and evaluator source identity mismatch")
    return {
        "source_sha256": canonical_source_hashes,
        "openpi_extension_files": canonical_extension_files,
        "server_source_sha256": server_source,
        "video_mode": None,
    }


def _validate_episodes(
    episodes: Sequence[Any],
) -> Dict[Tuple[int, int, str], Mapping[str, Any]]:
    if len(episodes) != ROLL_OUTS_PER_ARTIFACT:
        raise AnalysisError("episodes.json must contain exactly 150 rollouts")
    expected = _expected_cells()
    output: Dict[Tuple[int, int, str], Mapping[str, Any]] = {}
    episode_ids = set()
    for index, raw in enumerate(episodes):
        label = "episodes[%d]" % index
        if not isinstance(raw, Mapping):
            raise AnalysisError("%s must be an object" % label)
        _require_fields(
            raw,
            _IDENTITY_FIELDS
            + ("success", "policy_queries", "initial_state_sha256", "task_description"),
            label,
        )
        task_id = _integer(raw["task_id"], "%s.task_id" % label)
        episode_index = _integer(raw["episode_index"], "%s.episode_index" % label)
        method = raw["method"]
        key = (task_id, episode_index, method)
        if key in output:
            raise AnalysisError("episodes.json contains duplicate cell %r" % (key,))
        if key not in expected:
            raise AnalysisError("episodes.json contains unexpected cell %r" % (key,))
        pair_id = _pair_id(task_id, episode_index)
        episode_id = "%s__%s" % (pair_id, method)
        if raw["schema_version"] != SOURCE_SCHEMA_VERSION:
            raise AnalysisError("%s schema mismatch" % label)
        if raw["task_suite"] != TASK_SUITE or raw["pair_id"] != pair_id:
            raise AnalysisError("%s pair identity mismatch" % label)
        if raw["episode_id"] != episode_id or episode_id in episode_ids:
            raise AnalysisError("%s episode identity is duplicate or malformed" % label)
        if raw["condition_order"] != expected[key]:
            raise AnalysisError("%s violates the frozen Latin rotation" % label)
        if raw["execute_horizon"] != EXECUTE_HORIZON:
            raise AnalysisError("%s execute horizon mismatch" % label)
        if raw["inference_delay_steps"] != INFERENCE_DELAY_STEPS:
            raise AnalysisError("%s inference delay mismatch" % label)
        if not isinstance(raw["success"], bool):
            raise AnalysisError("%s.success must be boolean" % label)
        _integer(raw["policy_queries"], "%s.policy_queries" % label, minimum=2)
        _canonical_sha256(
            raw["initial_state_sha256"], "%s.initial_state_sha256" % label
        )
        if not isinstance(raw["task_description"], str) or not raw["task_description"]:
            raise AnalysisError("%s task description is invalid" % label)
        output[key] = raw
        episode_ids.add(episode_id)
    if set(output) != set(expected):
        raise AnalysisError("episodes.json is missing one or more frozen cells")
    for task_id in TASK_IDS:
        for episode_index in EPISODE_INDICES:
            selected = [output[(task_id, episode_index, method)] for method in METHODS]
            if len({row["initial_state_sha256"] for row in selected}) != 1:
                raise AnalysisError(
                    "task %d episode %d does not share one initial state"
                    % (task_id, episode_index)
                )
            if len({row["task_description"] for row in selected}) != 1:
                raise AnalysisError(
                    "task %d episode %d has inconsistent task identity"
                    % (task_id, episode_index)
                )
        task_descriptions = {
            output[(task_id, episode_index, method)]["task_description"]
            for episode_index in EPISODE_INDICES
            for method in METHODS
        }
        if len(task_descriptions) != 1:
            raise AnalysisError("task %d has inconsistent descriptions" % task_id)
    return output


def _validate_queries(
    queries: Sequence[Any],
    episodes: Mapping[Tuple[int, int, str], Mapping[str, Any]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[Tuple[int, int, int], str]]:
    by_episode: Dict[str, List[Mapping[str, Any]]] = {
        str(row["episode_id"]): [] for row in episodes.values()
    }
    episode_by_id = {str(row["episode_id"]): row for row in episodes.values()}
    seen = set()
    sampling_keys: Dict[Tuple[int, int, int], str] = {}
    sampling_noise: Dict[Tuple[int, int, int], str] = {}
    bootstrap_by_triplet: Dict[Tuple[int, int], Dict[str, Mapping[str, Any]]] = {}
    for index, raw in enumerate(queries):
        label = "queries[%d]" % index
        if not isinstance(raw, Mapping):
            raise AnalysisError("%s must be an object" % label)
        _require_fields(
            raw,
            _IDENTITY_FIELDS
            + (
                "query_index",
                "bootstrap",
                "sampling_key_sha256",
                "sampling_noise_sha256",
                "policy_input_sha256",
                "response_action_sha256",
            )
            + _CONDITION_GUIDANCE_FIELDS
            + SEAM_METRICS,
            label,
        )
        episode_id = raw["episode_id"]
        if episode_id not in by_episode:
            raise AnalysisError("%s references an unknown episode" % label)
        query_index = _integer(raw["query_index"], "%s.query_index" % label)
        query_key = (episode_id, query_index)
        if query_key in seen:
            raise AnalysisError(
                "queries.json contains duplicate query %r" % (query_key,)
            )
        seen.add(query_key)
        episode = episode_by_id[str(episode_id)]
        for field in _IDENTITY_FIELDS:
            if raw[field] != episode[field]:
                raise AnalysisError("%s mismatches its episode on %s" % (label, field))
        if not isinstance(raw["bootstrap"], bool):
            raise AnalysisError("%s.bootstrap must be boolean" % label)
        sampling_key = _canonical_sha256(
            raw["sampling_key_sha256"], "%s.sampling_key_sha256" % label
        )
        noise_hash = _canonical_sha256(
            raw["sampling_noise_sha256"], "%s.sampling_noise_sha256" % label
        )
        _canonical_sha256(raw["policy_input_sha256"], "%s.policy_input_sha256" % label)
        _canonical_sha256(
            raw["response_action_sha256"], "%s.response_action_sha256" % label
        )
        pair_query = (
            int(raw["task_id"]),
            int(raw["episode_index"]),
            query_index,
        )
        previous = sampling_keys.setdefault(pair_query, sampling_key)
        if previous != sampling_key:
            raise AnalysisError(
                "%s violates method-independent sampling identity" % label
            )
        previous_noise = sampling_noise.setdefault(pair_query, noise_hash)
        if previous_noise != noise_hash:
            raise AnalysisError(
                "%s violates method-independent sampling-noise identity" % label
            )
        if raw["bootstrap"]:
            if query_index != 0 or any(
                raw[metric] is not None for metric in SEAM_METRICS
            ):
                raise AnalysisError("%s has an invalid bootstrap seam record" % label)
            triplet = (int(raw["task_id"]), int(raw["episode_index"]))
            methods = bootstrap_by_triplet.setdefault(triplet, {})
            method = str(raw["method"])
            if method in methods:
                raise AnalysisError(
                    "bootstrap pairing contains duplicate method %s for %r"
                    % (method, triplet)
                )
            methods[method] = raw
        else:
            if query_index == 0:
                raise AnalysisError("%s scored query cannot use index zero" % label)
            for metric in SEAM_METRICS:
                _finite_nonnegative(raw[metric], "%s.%s" % (label, metric))
        by_episode[str(episode_id)].append(raw)

    aggregate: Dict[str, Dict[str, float]] = {}
    for episode in episodes.values():
        episode_id = str(episode["episode_id"])
        selected = sorted(by_episode[episode_id], key=lambda row: row["query_index"])
        expected_indices = list(range(int(episode["policy_queries"])))
        if [row["query_index"] for row in selected] != expected_indices:
            raise AnalysisError(
                "episode %s has missing or noncontiguous queries" % episode_id
            )
        if len(selected) < 2 or selected[0]["bootstrap"] is not True:
            raise AnalysisError("episode %s lacks scored seam queries" % episode_id)
        if any(row["bootstrap"] for row in selected[1:]):
            raise AnalysisError(
                "episode %s contains multiple bootstrap queries" % episode_id
            )
        aggregate[episode_id] = {
            "scored_queries": len(selected) - 1,
            **{
                metric: float(np.mean([float(row[metric]) for row in selected[1:]]))
                for metric in SEAM_METRICS
            },
        }
    expected_triplets = {
        (task_id, episode_index)
        for task_id in TASK_IDS
        for episode_index in EPISODE_INDICES
    }
    if set(bootstrap_by_triplet) != expected_triplets:
        raise AnalysisError("bootstrap pairing does not cover all 50 source triplets")
    for triplet in sorted(expected_triplets):
        methods = bootstrap_by_triplet[triplet]
        if set(methods) != set(METHODS):
            raise AnalysisError(
                "bootstrap pairing lacks all three methods for task %d episode %d"
                % triplet
            )
        records = [methods[method] for method in METHODS]
        if any(record["bootstrap"] is not True for record in records):
            raise AnalysisError(
                "query0 must be bootstrap for task %d episode %d" % triplet
            )
        if any(
            record[field] is not None
            for record in records
            for field in _CONDITION_GUIDANCE_FIELDS
        ):
            raise AnalysisError(
                "query0 conditioning/guidance audit must be null for task %d episode %d"
                % triplet
            )
        for field, description in (
            ("policy_input_sha256", "policy input"),
            ("sampling_key_sha256", "sampling key"),
            ("sampling_noise_sha256", "sampling noise"),
            ("response_action_sha256", "response action"),
        ):
            if len({record[field] for record in records}) != 1:
                raise AnalysisError(
                    "query0 %s mismatch across methods for task %d episode %d"
                    % (description, *triplet)
                )
    return aggregate, sampling_keys


def _source_evidence(
    root: pathlib.Path,
    snapshots: Mapping[str, bytes],
    validator_summary: Mapping[str, Any],
) -> SourceEvidence:
    manifest = _strict_json(snapshots["manifest.json"], "manifest.json", dict)
    protocol = _strict_json(
        snapshots["resolved_protocol.json"], "resolved_protocol.json", dict
    )
    environment = _strict_json(snapshots["environment.json"], "environment.json", dict)
    episodes_json = _strict_json(snapshots["episodes.json"], "episodes.json", list)
    queries_json = _strict_json(snapshots["queries.json"], "queries.json", list)
    progress = _strict_json(snapshots["progress.json"], "progress.json", dict)
    _strict_json(snapshots["summary.json"], "summary.json", dict)
    _strict_json(
        snapshots["transition_descriptor.json"], "transition_descriptor.json", dict
    )
    if manifest.get("schema_version") != "armbench.root_manifest.v1":
        raise AnalysisError("source root manifest schema mismatch")
    if progress != {
        "planned": ROLL_OUTS_PER_ARTIFACT,
        "completed": ROLL_OUTS_PER_ARTIFACT,
        "complete": True,
    }:
        raise AnalysisError("analysis requires a complete 150-rollout artifact")
    sampling_seed = _validate_protocol(protocol)
    identity = _validate_environment(environment, sampling_seed)
    identity["video_mode"] = protocol["video_mode"]
    episodes = _validate_episodes(episodes_json)
    seam_by_episode, sampling_keys = _validate_queries(queries_json, episodes)
    return SourceEvidence(
        root=root,
        label=_portable_source_label(root, sampling_seed),
        sampling_seed=sampling_seed,
        snapshots=snapshots,
        validator_summary=validator_summary,
        episodes=episodes,
        seam_by_episode=seam_by_episode,
        sampling_keys=sampling_keys,
        identity=identity,
    )


def _load_sources(artifacts: Sequence[pathlib.Path]) -> List[SourceEvidence]:
    roots, first_validation = _validate_all_first(artifacts)
    snapshots = _snapshot_sources(roots, first_validation)
    sources = [
        _source_evidence(root, source_snapshots, validator_summary)
        for root, (source_snapshots, validator_summary) in zip(roots, snapshots)
    ]
    sources.sort(key=lambda source: source.sampling_seed)
    if tuple(source.sampling_seed for source in sources) != SAMPLING_SEEDS:
        raise AnalysisError(
            "sources must contain each frozen sampling seed exactly once"
        )
    if sources[0].identity != sources[1].identity:
        raise AnalysisError("source implementation or recording identity mismatch")

    for task_id in TASK_IDS:
        for episode_index in EPISODE_INDICES:
            across_sources = [
                source.episodes[(task_id, episode_index, BASELINE)]
                for source in sources
            ]
            if len({row["initial_state_sha256"] for row in across_sources}) != 1:
                raise AnalysisError(
                    "initial-state mismatch across sampling artifacts for task %d episode %d"
                    % (task_id, episode_index)
                )
            if len({row["task_description"] for row in across_sources}) != 1:
                raise AnalysisError(
                    "task identity mismatch across sampling artifacts for task %d"
                    % task_id
                )
    shared_sampling_cells = set(sources[0].sampling_keys) & set(
        sources[1].sampling_keys
    )
    for key in shared_sampling_cells:
        if sources[0].sampling_keys[key] == sources[1].sampling_keys[key]:
            raise AnalysisError(
                "cross-artifact duplicate sampling key for task/state/query %r" % (key,)
            )
    return sources


def _source_record(source: SourceEvidence) -> Dict[str, Any]:
    validator_payload = json.dumps(
        source.validator_summary,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "artifact": source.label,
        "sampling_seed": source.sampling_seed,
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "raw_protocol_pilot_only": True,
        "bootstrap_triplets_bitwise_verified": TRIPLETS_PER_ARTIFACT,
        "bootstrap_pairing_fields": [
            "policy_input_sha256",
            "response_action_sha256",
            "sampling_key_sha256",
            "sampling_noise_sha256",
        ],
        "frozen_environment_command_verified": True,
        "raw_protocol_role": (
            "immutable corrected-v3 evaluator field; not the held-out analysis designation"
        ),
        "held_out_role_basis": (
            "external protocol commit %s with disjoint states 2-6 and sampling seeds"
            % FROZEN_EXTERNAL_PROTOCOL_COMMIT
        ),
        "manifest_sha256": _sha256_bytes(source.snapshots["manifest.json"]),
        "resolved_protocol_sha256": _sha256_bytes(
            source.snapshots["resolved_protocol.json"]
        ),
        "environment_sha256": _sha256_bytes(source.snapshots["environment.json"]),
        "episodes_sha256": _sha256_bytes(source.snapshots["episodes.json"]),
        "queries_sha256": _sha256_bytes(source.snapshots["queries.json"]),
        "validator_summary_sha256": _sha256_bytes(validator_payload),
    }


def _paired_rows(sources: Sequence[SourceEvidence]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    source_by_seed = {source.sampling_seed: source for source in sources}
    seen_triplets = set()
    for task_id in TASK_IDS:
        for episode_index in EPISODE_INDICES:
            for sampling_seed in SAMPLING_SEEDS:
                source = source_by_seed[sampling_seed]
                triplet_key = (task_id, episode_index, sampling_seed)
                if triplet_key in seen_triplets:
                    raise AnalysisError(
                        "combined analysis contains a duplicate triplet"
                    )
                seen_triplets.add(triplet_key)
                selected = {
                    method: source.episodes[(task_id, episode_index, method)]
                    for method in METHODS
                }
                baseline = selected[BASELINE]
                row: Dict[str, Any] = {
                    "triplet_id": (
                        "%s__sampling_seed_%d" % (baseline["pair_id"], sampling_seed)
                    ),
                    "source_artifact": source.label,
                    "pair_id": baseline["pair_id"],
                    "task_suite": TASK_SUITE,
                    "task_id": task_id,
                    "episode_index": episode_index,
                    "sampling_seed": sampling_seed,
                    "initial_state_sha256": baseline["initial_state_sha256"],
                }
                for method, episode in selected.items():
                    row["%s_condition_order" % method] = int(episode["condition_order"])
                    row["%s_success" % method] = int(bool(episode["success"]))
                    row["%s_policy_queries" % method] = int(episode["policy_queries"])
                    seam = source.seam_by_episode[str(episode["episode_id"])]
                    row["%s_scored_transition_queries" % method] = int(
                        seam["scored_queries"]
                    )
                    for metric in SEAM_METRICS:
                        row["%s_%s" % (method, metric)] = float(seam[metric])
                for candidate in CANDIDATES:
                    row["%s_success_difference" % candidate] = (
                        row["%s_success" % candidate] - row["%s_success" % BASELINE]
                    )
                    for metric in SEAM_METRICS:
                        row["%s_%s_difference" % (candidate, metric)] = (
                            row["%s_%s" % (candidate, metric)]
                            - row["%s_%s" % (BASELINE, metric)]
                        )
                output.append(row)
    expected = {
        (task_id, episode_index, sampling_seed)
        for task_id in TASK_IDS
        for episode_index in EPISODE_INDICES
        for sampling_seed in SAMPLING_SEEDS
    }
    if seen_triplets != expected or len(output) != TOTAL_TRIPLETS:
        raise AnalysisError("combined task/state/seed triplet matrix is incomplete")
    return output


def _describe(values: Sequence[float]) -> Dict[str, Any]:
    return pilot_analysis._describe(values)


def _task_blocks(rows: Sequence[Mapping[str, Any]], field: str) -> np.ndarray:
    blocks = []
    expected_nested = {
        (episode_index, sampling_seed)
        for episode_index in EPISODE_INDICES
        for sampling_seed in SAMPLING_SEEDS
    }
    for task_id in TASK_IDS:
        selected = [row for row in rows if int(row["task_id"]) == task_id]
        observed_nested = {
            (int(row["episode_index"]), int(row["sampling_seed"])) for row in selected
        }
        if observed_nested != expected_nested or len(selected) != 10:
            raise AnalysisError(
                "task-block bootstrap requires five states by two seeds per task"
            )
        selected.sort(
            key=lambda row: (int(row["episode_index"]), int(row["sampling_seed"]))
        )
        blocks.append([float(row[field]) for row in selected])
    result = np.asarray(blocks, dtype=np.float64)
    if result.shape != (10, 10) or not np.all(np.isfinite(result)):
        raise AnalysisError("task-block bootstrap requires ten complete task blocks")
    return result


def _task_block_interval(
    rows: Sequence[Mapping[str, Any]], field: str
) -> Dict[str, float]:
    blocks = _task_blocks(rows, field)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        len(TASK_IDS),
        size=(BOOTSTRAP_RESAMPLES, len(TASK_IDS)),
    )
    estimates = np.mean(blocks[indices], axis=(1, 2))
    return {
        "low": float(np.percentile(estimates, 2.5)),
        "high": float(np.percentile(estimates, 97.5)),
    }


def _task_effects(rows: Sequence[Mapping[str, Any]], field: str) -> List[float]:
    blocks = _task_blocks(rows, field)
    return [float(value) for value in np.mean(blocks, axis=1)]


def _exact_task_sign_flip(
    rows: Sequence[Mapping[str, Any]], field: str
) -> Dict[str, Any]:
    effects = _task_effects(rows, field)
    observed = abs(float(np.mean(effects)))
    tolerance = 1e-15
    extreme = 0
    assignments = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(effects)):
        statistic = abs(float(np.mean(np.asarray(signs) * np.asarray(effects))))
        extreme += statistic + tolerance >= observed
        assignments += 1
    return {
        "unit": "task-level paired risk difference",
        "alternative": "two-sided",
        "observed_mean_task_difference": float(np.mean(effects)),
        "enumerated_assignments": assignments,
        "extreme_assignments": extreme,
        "exact_p": extreme / float(assignments),
    }


def _success_analysis(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    methods: Dict[str, Any] = {}
    for method in METHODS:
        values = [int(row["%s_success" % method]) for row in rows]
        successes = sum(values)
        interval = pilot_analysis._wilson_interval(successes, len(values))
        methods[method] = {
            "successes": successes,
            "rollouts": len(values),
            "rate": successes / float(len(values)),
            "wilson95_low": interval[0],
            "wilson95_high": interval[1],
        }
    raw_p: Dict[str, float] = {}
    contrasts: Dict[str, Any] = {}
    for candidate in CANDIDATES:
        field = "%s_success_difference" % candidate
        differences = [int(row[field]) for row in rows]
        wins = sum(value > 0 for value in differences)
        losses = sum(value < 0 for value in differences)
        raw_p[candidate] = pilot_analysis._mcnemar_exact_p(wins, losses)
        interval = _task_block_interval(rows, field)
        contrasts[candidate] = {
            "pairs": len(rows),
            "risk_difference": float(np.mean(differences)),
            "candidate_wins": wins,
            "candidate_losses": losses,
            "ties": len(rows) - wins - losses,
            "mcnemar_exact_two_sided_p": raw_p[candidate],
            "task_block_bootstrap95_low": interval["low"],
            "task_block_bootstrap95_high": interval["high"],
            "exact_task_sign_flip": _exact_task_sign_flip(rows, field),
        }
    adjusted = pilot_analysis._holm_adjust(raw_p)
    for candidate in CANDIDATES:
        result = contrasts[candidate]
        result["holm_adjusted_p"] = adjusted[candidate]
        result["success_improvement_supported"] = (
            result["risk_difference"] > 0.0 and adjusted[candidate] < 0.05
        )
    return {"methods": methods, "contrasts_vs_unconditioned": contrasts}


def _seam_analysis(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for metric in SEAM_METRICS:
        methods: Dict[str, Any] = {}
        for method in METHODS:
            values = [float(row["%s_%s" % (method, metric)]) for row in rows]
            described = _describe(values)
            methods[method] = {
                "valid_rollouts": len(values),
                "scored_transitions": sum(
                    int(row["%s_scored_transition_queries" % method]) for row in rows
                ),
                **described,
            }
        contrasts: Dict[str, Any] = {}
        for candidate in CANDIDATES:
            field = "%s_%s_difference" % (candidate, metric)
            differences = [float(row[field]) for row in rows]
            interval = _task_block_interval(rows, field)
            contrasts[candidate] = {
                "pairs": len(differences),
                "paired_episode_mean_difference": float(np.mean(differences)),
                "paired_episode_median_difference": float(np.median(differences)),
                "task_block_bootstrap95_low": interval["low"],
                "task_block_bootstrap95_high": interval["high"],
            }
        output[metric] = {
            "unit_of_analysis": "rollout mean across non-bootstrap scored transitions",
            "methods": methods,
            "contrasts_vs_unconditioned": contrasts,
        }
    return output


def _subset_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise AnalysisError("stratified summary requires at least one triplet")
    methods: Dict[str, Any] = {}
    for method in METHODS:
        success = [int(row["%s_success" % method]) for row in rows]
        methods[method] = {
            "rollouts": len(rows),
            "scored_transitions": sum(
                int(row["%s_scored_transition_queries" % method]) for row in rows
            ),
            "successes": sum(success),
            "success_rate": float(np.mean(success)),
            **{
                metric: _describe(
                    [float(row["%s_%s" % (method, metric)]) for row in rows]
                )
                for metric in SEAM_METRICS
            },
        }
    contrasts: Dict[str, Any] = {}
    for candidate in CANDIDATES:
        result: Dict[str, Any] = {
            "success_risk_difference": float(
                np.mean([row["%s_success_difference" % candidate] for row in rows])
            )
        }
        for metric in SEAM_METRICS:
            differences = [
                float(row["%s_%s_difference" % (candidate, metric)]) for row in rows
            ]
            result[metric] = {
                "paired_mean_difference": float(np.mean(differences)),
                "paired_median_difference": float(np.median(differences)),
            }
        contrasts[candidate] = result
    return {"methods": methods, "contrasts_vs_unconditioned": contrasts}


def _per_task(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for task_id in TASK_IDS:
        selected = [row for row in rows if int(row["task_id"]) == task_id]
        if len(selected) != 10:
            raise AnalysisError("per-task summary requires ten triplets per task")
        output.append(
            {
                "task_id": task_id,
                "initial_states": len(EPISODE_INDICES),
                "sampling_seeds": len(SAMPLING_SEEDS),
                "triplets": len(selected),
                **_subset_summary(selected),
            }
        )
    return output


def _leave_one_task_out(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    omissions = []
    for task_id in TASK_IDS:
        selected = [row for row in rows if int(row["task_id"]) != task_id]
        omissions.append(
            {
                "omitted_task_id": task_id,
                "remaining_tasks": 9,
                "remaining_triplets": len(selected),
                **_subset_summary(selected),
            }
        )
    ranges: Dict[str, Any] = {}
    for candidate in CANDIDATES:
        candidate_rows = [
            row["contrasts_vs_unconditioned"][candidate] for row in omissions
        ]
        ranges[candidate] = {
            "success_risk_difference_min": min(
                row["success_risk_difference"] for row in candidate_rows
            ),
            "success_risk_difference_max": max(
                row["success_risk_difference"] for row in candidate_rows
            ),
            **{
                "%s_paired_mean_difference_min" % metric: min(
                    row[metric]["paired_mean_difference"] for row in candidate_rows
                )
                for metric in SEAM_METRICS
            },
            **{
                "%s_paired_mean_difference_max" % metric: max(
                    row[metric]["paired_mean_difference"] for row in candidate_rows
                )
                for metric in SEAM_METRICS
            },
        }
    return {"omissions": omissions, "ranges": ranges}


def _condition_order(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    strata = []
    for order in range(len(METHODS)):
        methods: Dict[str, Any] = {}
        for method in METHODS:
            selected = [
                row for row in rows if int(row["%s_condition_order" % method]) == order
            ]
            methods[method] = _subset_summary(selected)["methods"][method]
        strata.append({"condition_order": order, "methods": methods})
    rotations = []
    for baseline_order in range(len(METHODS)):
        selected = [
            row
            for row in rows
            if int(row["%s_condition_order" % BASELINE]) == baseline_order
        ]
        sequences = {
            tuple(
                sorted(
                    METHODS,
                    key=lambda method: int(row["%s_condition_order" % method]),
                )
            )
            for row in selected
        }
        if len(sequences) != 1:
            raise AnalysisError("condition-order rotation identity is inconsistent")
        rotations.append(
            {
                "baseline_condition_order": baseline_order,
                "method_sequence": list(next(iter(sequences))),
                "triplets": len(selected),
                **_subset_summary(selected),
            }
        )
    return {
        "design": "three-way Latin rotation assigned within every task/state/seed triplet",
        "method_position_strata": strata,
        "strata": strata,
        "triplet_rotation_strata": rotations,
    }


def analyze_artifacts(
    artifacts: Sequence[pathlib.Path],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    sources = _load_sources(artifacts)
    rows = _paired_rows(sources)
    source_records = [_source_record(source) for source in sources]
    analysis = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_type": "corrected_v3_frozen_held_out_rtc_overlap_primary_300",
        "implementation": _implementation_identity(),
        "sources": source_records,
        "frozen_identity": {
            "armbench_commit": FROZEN_ARMBENCH_COMMIT,
            "external_held_out_protocol_commit": FROZEN_EXTERNAL_PROTOCOL_COMMIT,
            "openpi_upstream_commit": FROZEN_OPENPI_UPSTREAM_COMMIT,
            "openpi_extension_commit": FROZEN_OPENPI_EXTENSION_COMMIT,
            "policy_config": FROZEN_POLICY_CONFIG,
            "checkpoint": FROZEN_CHECKPOINT,
            "checkpoint_content_sha256": FROZEN_CHECKPOINT_CONTENT_SHA256,
        },
        "cohort": {
            "task_suite": TASK_SUITE,
            "task_ids": list(TASK_IDS),
            "initial_state_indices": list(EPISODE_INDICES),
            "sampling_seeds": list(SAMPLING_SEEDS),
            "methods": list(METHODS),
            "tasks": len(TASK_IDS),
            "initial_states_per_task": len(EPISODE_INDICES),
            "sampling_seeds_per_state": len(SAMPLING_SEEDS),
            "matched_triplets": len(rows),
            "rollouts": len(rows) * len(METHODS),
            "triplet_key_fields": [
                "task_suite",
                "task_id",
                "episode_index",
                "sampling_seed",
            ],
            "action_horizon": ACTION_HORIZON,
            "execute_horizon": EXECUTE_HORIZON,
            "inference_delay_steps": INFERENCE_DELAY_STEPS,
            "scored_transitions_by_method": {
                method: sum(
                    int(row["%s_scored_transition_queries" % method]) for row in rows
                )
                for method in METHODS
            },
        },
        "success": _success_analysis(rows),
        "seam": _seam_analysis(rows),
        "per_task": _per_task(rows),
        "leave_one_task_out": _leave_one_task_out(rows),
        "condition_order": _condition_order(rows),
        "statistics": {
            "confidence_level": 0.95,
            "success_rate_interval": "Wilson score interval",
            "paired_success_test": "exact two-sided McNemar binomial test",
            "multiplicity_adjustment": (
                "Holm step-down across the two prespecified contrasts versus "
                "overlap_unconditioned"
            ),
            "bootstrap_unit": (
                "whole LIBERO task retaining 5 states x 2 sampling seeds x 3 methods"
            ),
            "bootstrap_interval": "percentile interval of the paired mean difference",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "task_sign_flip": "two-sided exhaustive 2^10 task-level sign assignments",
            "task_sign_flip_role": (
                "dependence-sensitivity diagnostic; does not replace Holm-adjusted McNemar"
            ),
            "seam_unit": (
                "rollout mean after excluding bootstrap queries; rollouts are equally weighted"
            ),
        },
        "protocol_provenance": {
            "raw_v3_protocol_field": "pilot_only=true",
            "raw_artifacts_rewritten": False,
            "held_out_primary_design_source": FROZEN_EXTERNAL_PROTOCOL_COMMIT,
            "corrected_runtime_schema": SOURCE_SCHEMA_VERSION,
            "rejected_v2_attempts": {
                "included_in_estimates": False,
                "status": "preserved but excluded after the query-0 pairing audit",
            },
            "disjoint_from_development_pilot": {
                "initial_state_indices": list(EPISODE_INDICES),
                "sampling_seeds": list(SAMPLING_SEEDS),
            },
        },
        "claim_boundary": (
            "Same-checkpoint pi0.5 evidence on ten fixed LIBERO-10 simulation tasks. "
            "Each immutable corrected-v3 evaluator protocol retains pilot_only=true; "
            "the held-out primary role comes from external frozen protocol commit "
            "509f6f4cbcc9e8b02804edf640e565673d4a3855 and disjoint state/seed "
            "cohorts. The preserved v2 attempts are excluded from every estimate. "
            "It does not establish independent control/inference timing, deadline or "
            "collision safety, cross-policy generalization, or real-hardware efficacy."
        ),
    }
    if analysis["cohort"]["matched_triplets"] != TOTAL_TRIPLETS:
        raise AnalysisError("combined analysis did not produce 100 triplets")
    if analysis["cohort"]["rollouts"] != TOTAL_ROLLOUTS:
        raise AnalysisError("combined analysis did not produce 300 rollouts")
    return analysis, rows


def _summary_markdown(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# pi0.5 RTC overlap corrected-v3 held-out primary analysis",
        "",
        "- Matrix: 10 tasks x 5 initial states x 2 sampling seeds x 3 methods (300 rollouts)",
        "- Matched task/state/seed triplets: `100`",
        "- Whole-task bootstrap seed/resamples: `%d/%d`"
        % (BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES),
        "- Confirmatory family: two exact McNemar tests with Holm correction",
        "",
        "| Method | Success (Wilson 95%) | Motion seam mean / median | Gripper seam mean / median | Scored transitions |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        success = analysis["success"]["methods"][method]
        motion = analysis["seam"]["seam_motion_l2"]["methods"][method]
        gripper = analysis["seam"]["seam_gripper_abs"]["methods"][method]
        lines.append(
            "| %s | %d/%d (%.3f [%.3f, %.3f]) | %.6f / %.6f | %.6f / %.6f | %d |"
            % (
                method,
                success["successes"],
                success["rollouts"],
                success["rate"],
                success["wilson95_low"],
                success["wilson95_high"],
                motion["mean"],
                motion["median"],
                gripper["mean"],
                gripper["median"],
                motion["scored_transitions"],
            )
        )
    lines.extend(
        [
            "",
            "| Contrast vs unconditioned | Success difference (task-block 95%) | Wins/losses/ties | McNemar raw/Holm | Task sign-flip p | Motion seam difference (task-block 95%) | Gripper seam difference (task-block 95%) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for candidate in CANDIDATES:
        success = analysis["success"]["contrasts_vs_unconditioned"][candidate]
        motion = analysis["seam"]["seam_motion_l2"]["contrasts_vs_unconditioned"][
            candidate
        ]
        gripper = analysis["seam"]["seam_gripper_abs"]["contrasts_vs_unconditioned"][
            candidate
        ]
        lines.append(
            "| %s | %+.3f [%+.3f, %+.3f] | %d/%d/%d | %.6g/%.6g | %.6g | %+.6f [%+.6f, %+.6f] | %+.6f [%+.6f, %+.6f] |"
            % (
                candidate,
                success["risk_difference"],
                success["task_block_bootstrap95_low"],
                success["task_block_bootstrap95_high"],
                success["candidate_wins"],
                success["candidate_losses"],
                success["ties"],
                success["mcnemar_exact_two_sided_p"],
                success["holm_adjusted_p"],
                success["exact_task_sign_flip"]["exact_p"],
                motion["paired_episode_mean_difference"],
                motion["task_block_bootstrap95_low"],
                motion["task_block_bootstrap95_high"],
                gripper["paired_episode_mean_difference"],
                gripper["task_block_bootstrap95_low"],
                gripper["task_block_bootstrap95_high"],
            )
        )
    lines.extend(["", str(analysis["claim_boundary"]), ""])
    return "\n".join(lines)


def _write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=_PER_TRIPLET_FIELDS,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _manifest_source_record(source: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        field: source[field]
        for field in (
            "artifact",
            "sampling_seed",
            "manifest_sha256",
            "resolved_protocol_sha256",
            "environment_sha256",
            "episodes_sha256",
            "queries_sha256",
        )
    }


def validate_analysis_manifest(root: pathlib.Path) -> Dict[str, Any]:
    errors: List[str] = []
    raw_root = pathlib.Path(root)
    if raw_root.is_symlink():
        errors.append("analysis directory must not be a symbolic link")
    output = raw_root.resolve()
    if not output.is_dir():
        return {"valid": False, "errors": errors + ["analysis directory is missing"]}
    manifest_path = output / "manifest.json"
    try:
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise AnalysisError("analysis manifest is missing or non-regular")
        manifest = _strict_json(manifest_path.read_bytes(), "manifest.json", dict)
    except (OSError, ValueError) as exc:
        return {"valid": False, "errors": errors + [str(exc)]}
    if manifest.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        errors.append("analysis manifest schema mismatch")
    records = manifest.get("files")
    if not isinstance(records, Mapping) or set(records) != _OUTPUT_FILES:
        errors.append("analysis manifest output inventory mismatch")
        records = {} if not isinstance(records, Mapping) else records
    actual: Dict[str, pathlib.Path] = {}
    for path in sorted(output.iterdir()):
        if path.is_symlink():
            errors.append("analysis output contains a symbolic link: %s" % path.name)
        elif path.is_file() and path.name != "manifest.json":
            actual[path.name] = path
        elif path.name != "manifest.json":
            errors.append(
                "analysis output contains an unexpected entry: %s" % path.name
            )
    if set(actual) != _OUTPUT_FILES:
        errors.append("analysis output set mismatch")
    for name in sorted(set(actual) & set(records)):
        record = records[name]
        if not isinstance(record, Mapping):
            errors.append("invalid manifest record: %s" % name)
            continue
        if record.get("bytes") != actual[name].stat().st_size or record.get(
            "sha256"
        ) != _sha256_file(actual[name]):
            errors.append("analysis manifest digest mismatch: %s" % name)
    analysis: Any = None
    if "analysis.json" in actual:
        try:
            analysis = _strict_json(
                actual["analysis.json"].read_bytes(), "analysis.json", dict
            )
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    analysis_sources = (
        analysis.get("sources") if isinstance(analysis, Mapping) else None
    )
    manifest_sources = manifest.get("sources")
    if not isinstance(analysis_sources, list) or len(analysis_sources) != 2:
        errors.append("analysis.json must bind exactly two sources")
    else:
        try:
            expected_sources = [
                _manifest_source_record(source) for source in analysis_sources
            ]
        except (KeyError, TypeError):
            errors.append("analysis.json contains an invalid source record")
        else:
            if manifest_sources != expected_sources:
                errors.append(
                    "analysis manifest source hashes do not match analysis.json"
                )
    return {
        "valid": not errors,
        "errors": errors,
        "files_checked": len(actual),
    }


def generate_report(
    artifacts: Sequence[pathlib.Path], output_directory: pathlib.Path
) -> Dict[str, Any]:
    output = pathlib.Path(output_directory).resolve()
    if output.exists():
        raise AnalysisError("output directory already exists: %s" % output)
    source_roots = [pathlib.Path(artifact).resolve() for artifact in artifacts]
    if any(output == root or root in output.parents for root in source_roots):
        raise AnalysisError("output directory must be outside both source artifacts")
    analysis, rows = analyze_artifacts(artifacts)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(
        tempfile.mkdtemp(prefix=".%s.tmp-" % output.name, dir=str(output.parent))
    )
    try:
        _write_json(staging / "analysis.json", analysis)
        _write_csv(staging / "per_triplet.csv", rows)
        (staging / "summary.md").write_text(
            _summary_markdown(analysis), encoding="utf-8", newline="\n"
        )
        files = {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        _write_json(
            staging / "manifest.json",
            {
                "schema_version": ANALYSIS_SCHEMA_VERSION,
                "sources": [
                    _manifest_source_record(source) for source in analysis["sources"]
                ],
                "files": files,
            },
        )
        validation = validate_analysis_manifest(staging)
        if validation["valid"] is not True:
            raise AnalysisError(
                "generated analysis failed self-validation: %s"
                % "; ".join(validation["errors"])
            )
        if output.exists():
            raise AnalysisError("output directory already exists: %s" % output)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("artifacts", type=pathlib.Path, nargs=2)
    parser.add_argument("--output-directory", type=pathlib.Path, required=True)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        analysis = generate_report(args.artifacts, args.output_directory)
    except (AnalysisError, OSError, ValueError) as exc:
        if args.json:
            print(
                json.dumps(
                    {"valid": False, "error": str(exc)}, indent=2, sort_keys=True
                )
            )
        else:
            print("INVALID", exc)
        return 2
    if args.json:
        print(json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False))
    else:
        print("VALID", pathlib.Path(args.output_directory).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
