"""Command-line runner for paired pi0.5-LIBERO robustness experiments."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import pathlib
import platform
import re
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from integrations.openpi.libero_runtime import (
    ASYNC_UNGUARDED,
    STATE_GUARD,
    EpisodeResult,
    RuntimeConfig,
    VALID_MODES,
    run_episode,
)


SCHEMA_VERSION = "armbench.pi05_libero_async.v1"
SERVER_ATTESTATION_SCHEMA_VERSION = "armbench.openpi_server_attestation.v1"
OPENPI_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
DEFAULT_POLICY_CONFIG = "pi05_libero"
DEFAULT_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
LIBERO_ENV_RESOLUTION = 256
LIBERO_CONTROL_FREQUENCY_HZ = 20
LIBERO_CONTROL_PERIOD_MS = 1000.0 / LIBERO_CONTROL_FREQUENCY_HZ
PI05_LIBERO_ACTION_HORIZON = 10
PER_PROTOCOL_EXCLUDED_CATEGORIES = frozenset(
    ("environment_runtime", "observation_contract", "policy_transport_or_server")
)
FORMAL_RUN_ABORT_CATEGORIES = PER_PROTOCOL_EXCLUDED_CATEGORIES | frozenset(
    ("policy_timeout",)
)
SUITE_TASK_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}
SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
RUNTIME_SOURCE_FILES = (
    "integrations/openpi/libero_runtime.py",
    "integrations/openpi/libero_runtime_eval.py",
    "integrations/openpi/preflight.py",
    "integrations/openpi/compose.libero-runtime.yml",
    "integrations/openpi/serve_policy_attested.py",
)

EPISODE_FIELDS = (
    "schema_version",
    "episode_id",
    "pair_id",
    "condition_order",
    "task_suite",
    "task_id",
    "episode_index",
    "task_description",
    "mode",
    "replan_steps",
    "latency_steps",
    "injected_latency_ms",
    "seed",
    "success",
    "termination_reason",
    "initial_state_sha256",
    "environment_steps",
    "task_action_steps",
    "latency_action_steps",
    "policy_queries",
    "accepted_chunks",
    "rejected_chunks",
    "stale_chunks_executed",
    "stale_action_steps",
    "interventions",
    "inference_latency_p50_ms",
    "inference_latency_p95_ms",
    "policy_inference_latency_p50_ms",
    "policy_inference_latency_p95_ms",
    "server_inference_latency_p50_ms",
    "server_inference_latency_p95_ms",
    "wall_time_s",
    "failure_category",
    "failure_type",
    "failure_message",
    "video_required",
    "video_path",
    "video_error_type",
    "video_error_message",
)

QUERY_FIELDS = (
    "schema_version",
    "episode_id",
    "pair_id",
    "task_suite",
    "task_id",
    "episode_index",
    "mode",
    "replan_steps",
    "latency_steps",
    "query_index",
    "observation_step",
    "response_step",
    "inference_latency_ms",
    "policy_inference_latency_ms",
    "server_inference_latency_ms",
    "injected_latency_steps_requested",
    "injected_latency_steps_executed",
    "action_chunk_steps",
    "accepted",
    "decision",
    "rejection_reasons",
    "position_mismatch_m",
    "orientation_mismatch_rad",
    "gripper_mismatch_linf",
    "error_stage",
    "error_type",
    "error_message",
)

CONTRAST_FIELDS = (
    "schema_version",
    "scope",
    "contrast_type",
    "task_suite",
    "selected_task_count",
    "suite_task_count",
    "suite_coverage_complete",
    "mode",
    "fixed_replan_steps",
    "fixed_latency_steps",
    "reference_value",
    "comparison_value",
    "candidate_pairs",
    "matched_pairs",
    "per_protocol_pairs",
    "excluded_runtime_pairs",
    "excluded_pairing_mismatch_pairs",
    "reference_success_rate",
    "comparison_success_rate",
    "success_rate_difference",
    "success_difference_bootstrap95_low",
    "success_difference_bootstrap95_high",
    "comparison_wins",
    "reference_wins",
    "ties",
    "mcnemar_exact_p",
    "mcnemar_holm_p",
    "mean_policy_query_difference",
    "mean_stale_action_step_difference",
)


@dataclasses.dataclass(frozen=True)
class ExperimentCell:
    task_suite: str
    task_id: int
    episode_index: int
    mode: str
    replan_steps: int
    latency_steps: int
    pair_id: str
    condition_order: int

    @property
    def episode_id(self) -> str:
        return "%s__%s" % (self.pair_id.replace("/", "__"), self.mode)


class _OpenPICodec:
    def __init__(self) -> None:
        from openpi_client import msgpack_numpy

        self._packer = msgpack_numpy.Packer()
        self._unpack = msgpack_numpy.unpackb

    def pack(self, value: Mapping[str, Any]) -> bytes:
        return self._packer.pack(value)

    def unpack(self, value: bytes) -> Any:
        return self._unpack(value)


class BoundedOpenPIClient:
    """OpenPI-compatible websocket client with bounded startup and inference."""

    def __init__(
        self,
        host: str,
        port: int,
        startup_timeout_s: float,
        inference_timeout_s: float,
        connect_fn: Optional[Any] = None,
        codec: Optional[Any] = None,
        monotonic: Any = time.monotonic,
        sleeper: Any = time.sleep,
    ) -> None:
        if not math.isfinite(startup_timeout_s) or startup_timeout_s <= 0.0:
            raise ValueError("startup_timeout_s must be finite and positive")
        if not math.isfinite(inference_timeout_s) or inference_timeout_s <= 0.0:
            raise ValueError("inference_timeout_s must be finite and positive")
        if connect_fn is None:
            from websockets.sync.client import connect

            connect_fn = connect
        uri = host if host.startswith("ws") else "ws://%s" % host
        if port is not None:
            uri += ":%d" % port
        self._codec = codec or _OpenPICodec()
        self._inference_timeout_s = inference_timeout_s
        self._connection: Optional[Any] = None
        self._broken = False
        deadline = monotonic() + startup_timeout_s
        last_error: Optional[Exception] = None
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0.0:
                raise TimeoutError(
                    "OpenPI server was not ready within %.1f seconds" % startup_timeout_s
                ) from last_error
            connection: Optional[Any] = None
            try:
                connection = connect_fn(
                    uri,
                    compression=None,
                    max_size=None,
                    open_timeout=min(5.0, remaining),
                )
                metadata_message = connection.recv(timeout=remaining)
                metadata = self._codec.unpack(metadata_message)
                if not isinstance(metadata, Mapping):
                    raise TypeError("OpenPI server metadata must be a mapping")
                self._connection = connection
                self._server_metadata = dict(metadata)
                break
            except (OSError, TimeoutError) as exc:
                try:
                    if connection is not None:
                        connection.close()
                except (AttributeError, OSError):
                    pass
                last_error = exc
                sleeper(min(1.0, max(0.0, deadline - monotonic())))
            except Exception:
                try:
                    if connection is not None:
                        connection.close()
                except (AttributeError, OSError):
                    pass
                raise

    def get_server_metadata(self) -> Dict[str, Any]:
        return dict(self._server_metadata)

    def infer(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._broken or self._connection is None:
            raise ConnectionError("OpenPI websocket is closed after a prior protocol failure")
        try:
            self._connection.send(self._codec.pack(observation))
            response = self._connection.recv(timeout=self._inference_timeout_s)
            if isinstance(response, str):
                raise RuntimeError(
                    "OpenPI inference server returned an error:\n%s" % response
                )
            unpacked = self._codec.unpack(response)
            if not isinstance(unpacked, Mapping):
                raise TypeError("OpenPI policy response must be a mapping")
            return unpacked
        except Exception:
            self._broken = True
            try:
                self._connection.close()
            except (AttributeError, OSError):
                pass
            raise

    def reset(self) -> None:
        # The pinned official server has no remote reset/RNG-reset message.
        return None

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()

    @property
    def broken(self) -> bool:
        return self._broken


def _parse_int_selection(value: str, upper_bound: int, name: str) -> List[int]:
    value = value.strip().lower()
    if value == "all":
        return list(range(upper_bound))
    selected = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ValueError("%s contains an empty item" % name)
        if ":" in part:
            bounds = part.split(":")
            if len(bounds) != 2:
                raise ValueError("%s range must use start:stop" % name)
            start, stop = int(bounds[0]), int(bounds[1])
            selected.extend(range(start, stop))
        else:
            selected.append(int(part))
    if not selected:
        raise ValueError("%s must select at least one value" % name)
    if len(selected) != len(set(selected)):
        raise ValueError("%s contains duplicate values" % name)
    if min(selected) < 0 or max(selected) >= upper_bound:
        raise ValueError("%s values must be in [0, %d)" % (name, upper_bound))
    return selected


def _parse_positive_csv(value: str, name: str) -> List[int]:
    values = [int(item.strip()) for item in value.split(",")]
    if not values or any(item <= 0 for item in values):
        raise ValueError("%s must contain positive integers" % name)
    if len(values) != len(set(values)):
        raise ValueError("%s contains duplicate values" % name)
    return values


def _parse_nonnegative_csv(value: str, name: str) -> List[int]:
    values = [int(item.strip()) for item in value.split(",")]
    if not values or any(item < 0 for item in values):
        raise ValueError("%s must contain nonnegative integers" % name)
    if len(values) != len(set(values)):
        raise ValueError("%s contains duplicate values" % name)
    return values


def _parse_modes(value: str) -> List[str]:
    modes = [item.strip() for item in value.split(",")]
    if not modes or any(mode not in VALID_MODES for mode in modes):
        raise ValueError("modes must be selected from: %s" % ", ".join(VALID_MODES))
    if len(modes) != len(set(modes)):
        raise ValueError("modes contains duplicate values")
    return modes


def _validate_server_launch_args(
    server_launch_args: Optional[str], declared_checkpoint: str
) -> None:
    if not server_launch_args or not server_launch_args.strip():
        raise ValueError(
            "server launch provenance is required; pass --server-launch-args "
            "or set ARMBENCH_SERVER_ARGS"
        )
    normalized = " ".join(server_launch_args.lower().split())
    default_libero = bool(re.search(r"--env(?:=|\s+)libero(?:\s|$)", normalized))
    if default_libero:
        if declared_checkpoint != DEFAULT_CHECKPOINT:
            raise ValueError(
                "--env LIBERO loads the official default checkpoint, but "
                "--checkpoint declares a different path"
            )
        return
    explicit_checkpoint = (
        DEFAULT_POLICY_CONFIG.lower() in normalized
        and declared_checkpoint.lower() in normalized
    )
    if not explicit_checkpoint:
        raise ValueError(
            "server launch args must declare either '--env LIBERO' or both "
            "pi05_libero and the declared checkpoint path"
        )


def _validate_server_attestation(
    metadata: Mapping[str, Any],
    args: argparse.Namespace,
    expected_server_source_sha256: str,
) -> Optional[Mapping[str, Any]]:
    attestation = metadata.get("armbench_server_attestation")
    if args.allow_unattested_server:
        return attestation if isinstance(attestation, Mapping) else None
    if not isinstance(attestation, Mapping):
        raise ValueError(
            "OpenPI server did not provide ArmBench checkpoint attestation; "
            "use the attested server entrypoint"
        )
    required_equal = {
        "schema_version": SERVER_ATTESTATION_SCHEMA_VERSION,
        "policy_loaded": True,
        "policy_config": DEFAULT_POLICY_CONFIG,
        "checkpoint_uri": args.checkpoint,
        "openpi_commit": args.expected_openpi_commit,
        "openpi_tracked_clean": True,
        "openpi_tracked_status": "",
        "openpi_submodules_clean": True,
        "action_horizon": PI05_LIBERO_ACTION_HORIZON,
        "server_source_sha256": expected_server_source_sha256,
    }
    mismatches = [
        "%s=%r (expected %r)" % (key, attestation.get(key), expected)
        for key, expected in required_equal.items()
        if attestation.get(key) != expected
    ]
    for key in ("checkpoint_content_sha256", "server_source_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(attestation.get(key, ""))):
            mismatches.append("%s is not a SHA-256 digest" % key)
    for key in ("checkpoint_file_count", "checkpoint_total_bytes"):
        value = attestation.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            mismatches.append("%s must be a positive integer" % key)
    if mismatches:
        raise ValueError("server attestation mismatch: %s" % "; ".join(mismatches))
    return attestation


def build_matrix(
    task_suite: str,
    task_ids: Sequence[int],
    episode_indices: Sequence[int],
    modes: Sequence[str],
    replan_steps: Sequence[int],
    latency_steps: Sequence[int],
) -> List[ExperimentCell]:
    """Build adjacent paired conditions with alternating mode order."""

    if task_suite not in SUITE_TASK_COUNTS:
        raise ValueError("unknown task suite: %s" % task_suite)
    if not task_ids or not episode_indices or not modes:
        raise ValueError("matrix dimensions must be nonempty")
    if any(mode not in VALID_MODES for mode in modes):
        raise ValueError("invalid runtime mode")
    cells = []
    pair_index = 0
    condition_order = 0
    for task_id in task_ids:
        for episode_index in episode_indices:
            for horizon in replan_steps:
                for delay in latency_steps:
                    pair_id = "%s/task_%03d/episode_%03d/h_%02d/l_%03d" % (
                        task_suite,
                        task_id,
                        episode_index,
                        horizon,
                        delay,
                    )
                    ordered_modes = list(modes)
                    if len(ordered_modes) > 1 and pair_index % 2 == 1:
                        ordered_modes.reverse()
                    for mode in ordered_modes:
                        cells.append(
                            ExperimentCell(
                                task_suite=task_suite,
                                task_id=int(task_id),
                                episode_index=int(episode_index),
                                mode=mode,
                                replan_steps=int(horizon),
                                latency_steps=int(delay),
                                pair_id=pair_id,
                                condition_order=condition_order,
                            )
                        )
                        condition_order += 1
                    pair_index += 1
    return cells


def matrix_plan(cells: Sequence[ExperimentCell]) -> Dict[str, Any]:
    pairs = {cell.pair_id for cell in cells}
    return {
        "schema_version": SCHEMA_VERSION,
        "rollouts": len(cells),
        "paired_conditions": len(pairs),
        "task_suites": sorted({cell.task_suite for cell in cells}),
        "task_ids": sorted({cell.task_id for cell in cells}),
        "episode_indices": sorted({cell.episode_index for cell in cells}),
        "modes": sorted({cell.mode for cell in cells}),
        "replan_steps": sorted({cell.replan_steps for cell in cells}),
        "latency_steps": sorted({cell.latency_steps for cell in cells}),
        "warning": (
            "This count is environment rollouts, not policy queries. Query cost grows "
            "as replan_steps decreases and state-guard rejections increase."
        ),
    }


def _percentiles(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    return float(np.percentile(values, 50)), float(np.percentile(values, 95))


def episode_rows(
    cell: ExperimentCell,
    result: EpisodeResult,
    task_description: str,
    seed: int,
    wall_time_s: float,
    video_path: Optional[str],
    video_required: bool = False,
    video_error_type: Optional[str] = None,
    video_error_message: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    latencies = [record.inference_latency_ms for record in result.query_records]
    p50, p95 = _percentiles(latencies)
    policy_p50, policy_p95 = _percentiles(
        [
            record.policy_inference_latency_ms
            for record in result.query_records
            if record.policy_inference_latency_ms is not None
        ]
    )
    server_p50, server_p95 = _percentiles(
        [
            record.server_inference_latency_ms
            for record in result.query_records
            if record.server_inference_latency_ms is not None
        ]
    )
    episode = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": cell.episode_id,
        "pair_id": cell.pair_id,
        "condition_order": cell.condition_order,
        "task_suite": cell.task_suite,
        "task_id": cell.task_id,
        "episode_index": cell.episode_index,
        "task_description": task_description,
        "mode": cell.mode,
        "replan_steps": cell.replan_steps,
        "latency_steps": cell.latency_steps,
        "injected_latency_ms": cell.latency_steps * LIBERO_CONTROL_PERIOD_MS,
        "seed": seed,
        "success": result.success,
        "termination_reason": result.termination_reason,
        "initial_state_sha256": result.initial_state_sha256,
        "environment_steps": result.environment_steps,
        "task_action_steps": result.task_action_steps,
        "latency_action_steps": result.latency_action_steps,
        "policy_queries": result.policy_queries,
        "accepted_chunks": result.accepted_chunks,
        "rejected_chunks": result.rejected_chunks,
        "stale_chunks_executed": result.stale_chunks_executed,
        "stale_action_steps": result.stale_action_steps,
        "interventions": result.interventions,
        "inference_latency_p50_ms": p50,
        "inference_latency_p95_ms": p95,
        "policy_inference_latency_p50_ms": policy_p50,
        "policy_inference_latency_p95_ms": policy_p95,
        "server_inference_latency_p50_ms": server_p50,
        "server_inference_latency_p95_ms": server_p95,
        "wall_time_s": wall_time_s,
        "failure_category": result.failure_category,
        "failure_type": result.failure_type,
        "failure_message": result.failure_message,
        "video_required": video_required,
        "video_path": video_path,
        "video_error_type": video_error_type,
        "video_error_message": video_error_message,
    }
    queries = []
    for record in result.query_records:
        mismatch = record.mismatch
        queries.append(
            {
                "schema_version": SCHEMA_VERSION,
                "episode_id": cell.episode_id,
                "pair_id": cell.pair_id,
                "task_suite": cell.task_suite,
                "task_id": cell.task_id,
                "episode_index": cell.episode_index,
                "mode": cell.mode,
                "replan_steps": cell.replan_steps,
                "latency_steps": cell.latency_steps,
                "query_index": record.query_index,
                "observation_step": record.observation_step,
                "response_step": record.response_step,
                "inference_latency_ms": record.inference_latency_ms,
                "policy_inference_latency_ms": record.policy_inference_latency_ms,
                "server_inference_latency_ms": record.server_inference_latency_ms,
                "injected_latency_steps_requested": record.injected_latency_steps_requested,
                "injected_latency_steps_executed": record.injected_latency_steps_executed,
                "action_chunk_steps": record.action_chunk_steps,
                "accepted": record.accepted,
                "decision": record.decision,
                "rejection_reasons": "|".join(record.rejection_reasons),
                "position_mismatch_m": None if mismatch is None else mismatch.position_m,
                "orientation_mismatch_rad": (
                    None if mismatch is None else mismatch.orientation_rad
                ),
                "gripper_mismatch_linf": (
                    None if mismatch is None else mismatch.gripper_linf
                ),
                "error_stage": record.error_stage,
                "error_type": record.error_type,
                "error_message": record.error_message,
            }
        )
    return episode, queries


def _wilson_interval(
    successes: int, total: int
) -> Tuple[Optional[float], Optional[float]]:
    if total <= 0:
        return None, None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - margin, center + margin


def _mean(
    rows: Sequence[Mapping[str, Any]], field: str
) -> Optional[float]:
    return float(np.mean([float(row[field]) for row in rows])) if rows else None


def _per_protocol_eligible(row: Mapping[str, Any]) -> bool:
    return row.get("failure_category") not in PER_PROTOCOL_EXCLUDED_CATEGORIES


def aggregate_episodes(
    episodes: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    selected_tasks_by_suite = {
        str(suite): {
            int(row["task_id"])
            for row in episodes
            if str(row["task_suite"]) == str(suite)
        }
        for suite in {row["task_suite"] for row in episodes}
    }
    query_by_episode: Dict[str, List[Mapping[str, Any]]] = {}
    for query in queries:
        query_by_episode.setdefault(str(query["episode_id"]), []).append(query)
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for episode in episodes:
        base = (
            episode["task_suite"],
            episode["mode"],
            int(episode["replan_steps"]),
            int(episode["latency_steps"]),
        )
        groups.setdefault(("selected_tasks",) + base, []).append(episode)
        groups.setdefault(("task", int(episode["task_id"])) + base, []).append(
            episode
        )

    aggregate = []
    for key, rows in sorted(groups.items(), key=lambda item: repr(item[0])):
        if key[0] == "selected_tasks":
            _, suite, mode, horizon, delay = key
            scope = "selected_tasks"
            task_id = None
        else:
            _, task_id, suite, mode, horizon, delay = key
            scope = "task"
        eligible_rows = [row for row in rows if _per_protocol_eligible(row)]
        successes = sum(bool(row["success"]) for row in rows)
        lower, upper = _wilson_interval(successes, len(rows))
        eligible_successes = sum(bool(row["success"]) for row in eligible_rows)
        eligible_lower, eligible_upper = _wilson_interval(
            eligible_successes, len(eligible_rows)
        )
        group_queries = []
        for row in rows:
            group_queries.extend(query_by_episode.get(str(row["episode_id"]), []))
        inference_latencies = [
            float(query["inference_latency_ms"])
            for query in group_queries
            if query["inference_latency_ms"] is not None
        ]
        p50, p95 = _percentiles(inference_latencies)
        policy_p50, policy_p95 = _percentiles(
            [
                float(query["policy_inference_latency_ms"])
                for query in group_queries
                if query["policy_inference_latency_ms"] is not None
            ]
        )
        server_p50, server_p95 = _percentiles(
            [
                float(query["server_inference_latency_ms"])
                for query in group_queries
                if query["server_inference_latency_ms"] is not None
            ]
        )
        aggregate.append(
            {
                "schema_version": SCHEMA_VERSION,
                "scope": scope,
                "task_suite": suite,
                "task_id": task_id,
                "selected_task_count": len(selected_tasks_by_suite[str(suite)]),
                "suite_task_count": SUITE_TASK_COUNTS[str(suite)],
                "suite_coverage_complete": len(
                    selected_tasks_by_suite[str(suite)]
                )
                == SUITE_TASK_COUNTS[str(suite)],
                "mode": mode,
                "replan_steps": horizon,
                "latency_steps": delay,
                "injected_latency_ms": delay * LIBERO_CONTROL_PERIOD_MS,
                "rollouts": len(rows),
                "eligible_rollouts": len(eligible_rows),
                "successes": successes,
                "success_rate": successes / len(rows),
                "success_wilson95_low": lower,
                "success_wilson95_high": upper,
                "per_protocol_successes": eligible_successes,
                "per_protocol_success_rate": (
                    eligible_successes / len(eligible_rows)
                    if eligible_rows
                    else None
                ),
                "per_protocol_wilson95_low": eligible_lower,
                "per_protocol_wilson95_high": eligible_upper,
                "mean_policy_queries": _mean(rows, "policy_queries"),
                "mean_rejections": _mean(rows, "rejected_chunks"),
                "mean_stale_action_steps": _mean(
                    rows, "stale_action_steps"
                ),
                "mean_environment_steps": _mean(
                    rows, "environment_steps"
                ),
                "inference_latency_p50_ms": p50,
                "inference_latency_p95_ms": p95,
                "policy_inference_latency_p50_ms": policy_p50,
                "policy_inference_latency_p95_ms": policy_p95,
                "server_inference_latency_p50_ms": server_p50,
                "server_inference_latency_p95_ms": server_p95,
                "excluded_per_protocol_failures": sum(
                    not _per_protocol_eligible(row) for row in rows
                ),
                "policy_contract_failures": sum(
                    row.get("failure_category") == "policy_contract"
                    for row in rows
                ),
                "policy_timeout_failures": sum(
                    row.get("failure_category") == "policy_timeout"
                    for row in rows
                ),
                "all_recorded_failures": sum(
                    bool(row["failure_type"]) for row in rows
                ),
                "excluded_failure_categories": sorted(
                    {
                        str(row["failure_category"])
                        for row in rows
                        if row["failure_category"]
                    }
                ),
            }
        )
    return aggregate


def _bootstrap_mean_interval(
    values: Sequence[float], resamples: int, seed_key: str
) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), float(values[0])
    seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:16], 16)
    random = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    sample_indices = random.integers(0, len(array), size=(resamples, len(array)))
    means = np.mean(array[sample_indices], axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _mcnemar_exact_p(guard_wins: int, unguarded_wins: int) -> float:
    """Two-sided exact McNemar p-value over discordant paired outcomes."""

    discordant = guard_wins + unguarded_wins
    if discordant == 0:
        return 1.0
    smaller = min(guard_wins, unguarded_wins)
    lower_tail = sum(
        math.comb(discordant, index) for index in range(smaller + 1)
    ) / (2.0 ** discordant)
    return min(1.0, 2.0 * lower_tail)


def _apply_primary_holm_correction(comparisons: List[Dict[str, Any]]) -> None:
    primary_indices = [
        index
        for index, row in enumerate(comparisons)
        if row["scope"] == "selected_tasks"
        and row["mcnemar_exact_p"] is not None
    ]
    ordered = sorted(
        primary_indices,
        key=lambda index: float(comparisons[index]["mcnemar_exact_p"]),
    )
    running_maximum = 0.0
    hypotheses = len(ordered)
    for rank, index in enumerate(ordered):
        raw = float(comparisons[index]["mcnemar_exact_p"])
        adjusted = min(1.0, raw * (hypotheses - rank))
        running_maximum = max(running_maximum, adjusted)
        comparisons[index]["mcnemar_holm_p"] = running_maximum


def paired_comparisons(
    episodes: Sequence[Mapping[str, Any]], bootstrap_resamples: int
) -> List[Dict[str, Any]]:
    selected_tasks_by_suite = {
        str(suite): {
            int(row["task_id"])
            for row in episodes
            if str(row["task_suite"]) == str(suite)
        }
        for suite in {row["task_suite"] for row in episodes}
    }
    by_pair: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for episode in episodes:
        by_pair.setdefault(str(episode["pair_id"]), {})[
            str(episode["mode"])
        ] = episode

    candidate_pairs = [
        modes
        for modes in by_pair.values()
        if ASYNC_UNGUARDED in modes and STATE_GUARD in modes
    ]
    groups: Dict[Tuple[Any, ...], List[Dict[str, Mapping[str, Any]]]] = {}
    for modes in candidate_pairs:
        baseline = modes[ASYNC_UNGUARDED]
        base = (
            baseline["task_suite"],
            int(baseline["replan_steps"]),
            int(baseline["latency_steps"]),
        )
        groups.setdefault(("selected_tasks",) + base, []).append(modes)
        groups.setdefault(("task", int(baseline["task_id"])) + base, []).append(
            modes
        )

    comparisons = []
    for key, rows in sorted(groups.items(), key=lambda item: repr(item[0])):
        if key[0] == "selected_tasks":
            _, suite, horizon, delay = key
            scope = "selected_tasks"
            task_id = None
        else:
            _, task_id, suite, horizon, delay = key
            scope = "task"
        structurally_valid_rows = []
        infrastructure_exclusions = 0
        pairing_mismatch_exclusions = 0
        for pair in rows:
            baseline = pair[ASYNC_UNGUARDED]
            guard = pair[STATE_GUARD]
            pairing_fields = (
                "task_suite",
                "task_id",
                "episode_index",
                "replan_steps",
                "latency_steps",
                "seed",
                "initial_state_sha256",
            )
            if any(baseline[field] != guard[field] for field in pairing_fields):
                pairing_mismatch_exclusions += 1
                continue
            structurally_valid_rows.append(pair)

        eligible_rows = [
            pair
            for pair in structurally_valid_rows
            if _per_protocol_eligible(pair[ASYNC_UNGUARDED])
            and _per_protocol_eligible(pair[STATE_GUARD])
        ]
        infrastructure_exclusions = len(structurally_valid_rows) - len(
            eligible_rows
        )

        differences = [
            float(pair[STATE_GUARD]["success"])
            - float(pair[ASYNC_UNGUARDED]["success"])
            for pair in structurally_valid_rows
        ]
        lower, upper = _bootstrap_mean_interval(
            differences, bootstrap_resamples, repr(key)
        )
        baseline_queries = np.asarray(
            [
                float(pair[ASYNC_UNGUARDED]["policy_queries"])
                for pair in structurally_valid_rows
            ]
        )
        guard_queries = np.asarray(
            [
                float(pair[STATE_GUARD]["policy_queries"])
                for pair in structurally_valid_rows
            ]
        )
        baseline_query_mean = (
            float(np.mean(baseline_queries))
            if structurally_valid_rows
            else None
        )
        query_overhead = (
            float(
                (np.mean(guard_queries) - baseline_query_mean)
                / baseline_query_mean
            )
            if baseline_query_mean is not None and baseline_query_mean > 0.0
            else None
        )
        guard_wins = sum(value > 0.0 for value in differences)
        unguarded_wins = sum(value < 0.0 for value in differences)
        comparisons.append(
            {
                "schema_version": SCHEMA_VERSION,
                "scope": scope,
                "task_suite": suite,
                "task_id": task_id,
                "selected_task_count": len(selected_tasks_by_suite[str(suite)]),
                "suite_task_count": SUITE_TASK_COUNTS[str(suite)],
                "suite_coverage_complete": len(
                    selected_tasks_by_suite[str(suite)]
                )
                == SUITE_TASK_COUNTS[str(suite)],
                "replan_steps": horizon,
                "latency_steps": delay,
                "injected_latency_ms": delay * LIBERO_CONTROL_PERIOD_MS,
                "candidate_pairs": len(rows),
                "paired_episodes": len(structurally_valid_rows),
                "per_protocol_pairs": len(eligible_rows),
                "excluded_infrastructure_pairs": infrastructure_exclusions,
                "excluded_pairing_mismatch_pairs": pairing_mismatch_exclusions,
                "unguarded_success_rate": (
                    float(
                        np.mean(
                            [
                                float(pair[ASYNC_UNGUARDED]["success"])
                                for pair in structurally_valid_rows
                            ]
                        )
                    )
                    if structurally_valid_rows
                    else None
                ),
                "state_guard_success_rate": (
                    float(
                        np.mean(
                            [
                                float(pair[STATE_GUARD]["success"])
                                for pair in structurally_valid_rows
                            ]
                        )
                    )
                    if structurally_valid_rows
                    else None
                ),
                "success_rate_difference": (
                    float(np.mean(differences)) if differences else None
                ),
                "per_protocol_success_rate_difference": (
                    float(
                        np.mean(
                            [
                                float(pair[STATE_GUARD]["success"])
                                - float(pair[ASYNC_UNGUARDED]["success"])
                                for pair in eligible_rows
                            ]
                        )
                    )
                    if eligible_rows
                    else None
                ),
                "success_difference_bootstrap95_low": lower,
                "success_difference_bootstrap95_high": upper,
                "guard_wins": guard_wins,
                "unguarded_wins": unguarded_wins,
                "ties": sum(value == 0.0 for value in differences),
                "mcnemar_exact_p": (
                    _mcnemar_exact_p(guard_wins, unguarded_wins)
                    if structurally_valid_rows
                    else None
                ),
                "mcnemar_holm_p": None,
                "mean_unguarded_policy_queries": baseline_query_mean,
                "mean_state_guard_policy_queries": (
                    float(np.mean(guard_queries))
                    if structurally_valid_rows
                    else None
                ),
                "relative_policy_query_overhead": query_overhead,
                "mean_state_guard_rejections": (
                    float(
                        np.mean(
                            [
                                float(pair[STATE_GUARD]["rejected_chunks"])
                                for pair in structurally_valid_rows
                            ]
                        )
                    )
                    if structurally_valid_rows
                    else None
                ),
            }
        )
    _apply_primary_holm_correction(comparisons)
    return comparisons


def matched_condition_contrasts(
    episodes: Sequence[Mapping[str, Any]], bootstrap_resamples: int
) -> List[Dict[str, Any]]:
    """Compare injected delay and horizon against their registered references."""

    selected_tasks_by_suite = {
        str(suite): {
            int(row["task_id"])
            for row in episodes
            if str(row["task_suite"]) == str(suite)
        }
        for suite in {row["task_suite"] for row in episodes}
    }
    contrasts = []
    specifications = []
    suites = sorted({str(row["task_suite"]) for row in episodes})
    modes = sorted({str(row["mode"]) for row in episodes})
    horizons = sorted({int(row["replan_steps"]) for row in episodes})
    delays = sorted({int(row["latency_steps"]) for row in episodes})
    for suite in suites:
        for mode in modes:
            for horizon in horizons:
                for delay in (value for value in delays if value != 0):
                    specifications.append(
                        ("delay_vs_zero", suite, mode, horizon, None, 0, delay)
                    )
            for delay in delays:
                for horizon in (
                    value for value in horizons if value != 5
                ):
                    specifications.append(
                        ("horizon_vs_five", suite, mode, None, delay, 5, horizon)
                    )

    for (
        contrast_type,
        suite,
        mode,
        fixed_horizon,
        fixed_delay,
        reference_value,
        comparison_value,
    ) in specifications:
        candidates = [
            row
            for row in episodes
            if str(row["task_suite"]) == suite
            and str(row["mode"]) == mode
            and (
                fixed_horizon is None
                or int(row["replan_steps"]) == fixed_horizon
            )
            and (
                fixed_delay is None
                or int(row["latency_steps"]) == fixed_delay
            )
        ]
        by_unit: Dict[Tuple[int, int], Dict[int, Mapping[str, Any]]] = {}
        for row in candidates:
            varying_value = (
                int(row["latency_steps"])
                if contrast_type == "delay_vs_zero"
                else int(row["replan_steps"])
            )
            if varying_value not in (reference_value, comparison_value):
                continue
            unit = (int(row["task_id"]), int(row["episode_index"]))
            by_unit.setdefault(unit, {})[varying_value] = row
        candidate_pairs = [
            pair
            for pair in by_unit.values()
            if reference_value in pair and comparison_value in pair
        ]
        valid_pairs = []
        mismatch_exclusions = 0
        for pair in candidate_pairs:
            reference = pair[reference_value]
            comparison = pair[comparison_value]
            if reference["seed"] != comparison["seed"] or reference[
                "initial_state_sha256"
            ] != comparison["initial_state_sha256"]:
                mismatch_exclusions += 1
                continue
            valid_pairs.append(pair)
        per_protocol_pairs = [
            pair
            for pair in valid_pairs
            if _per_protocol_eligible(pair[reference_value])
            and _per_protocol_eligible(pair[comparison_value])
        ]
        if not candidate_pairs:
            continue
        differences = [
            float(pair[comparison_value]["success"])
            - float(pair[reference_value]["success"])
            for pair in valid_pairs
        ]
        lower, upper = _bootstrap_mean_interval(
            differences,
            bootstrap_resamples,
            "%s/%s/%s/%s/%s"
            % (contrast_type, suite, mode, fixed_horizon, fixed_delay),
        )
        comparison_wins = sum(value > 0.0 for value in differences)
        reference_wins = sum(value < 0.0 for value in differences)
        reference_success_rate = (
            float(
                np.mean(
                    [float(pair[reference_value]["success"]) for pair in valid_pairs]
                )
            )
            if valid_pairs
            else None
        )
        comparison_success_rate = (
            float(
                np.mean(
                    [
                        float(pair[comparison_value]["success"])
                        for pair in valid_pairs
                    ]
                )
            )
            if valid_pairs
            else None
        )
        contrasts.append(
            {
                "schema_version": SCHEMA_VERSION,
                "scope": "selected_tasks",
                "contrast_type": contrast_type,
                "task_suite": suite,
                "selected_task_count": len(selected_tasks_by_suite[suite]),
                "suite_task_count": SUITE_TASK_COUNTS[suite],
                "suite_coverage_complete": len(selected_tasks_by_suite[suite])
                == SUITE_TASK_COUNTS[suite],
                "mode": mode,
                "fixed_replan_steps": fixed_horizon,
                "fixed_latency_steps": fixed_delay,
                "reference_value": reference_value,
                "comparison_value": comparison_value,
                "candidate_pairs": len(candidate_pairs),
                "matched_pairs": len(valid_pairs),
                "per_protocol_pairs": len(per_protocol_pairs),
                "excluded_runtime_pairs": len(valid_pairs)
                - len(per_protocol_pairs),
                "excluded_pairing_mismatch_pairs": mismatch_exclusions,
                "reference_success_rate": reference_success_rate,
                "comparison_success_rate": comparison_success_rate,
                "success_rate_difference": (
                    float(np.mean(differences)) if differences else None
                ),
                "success_difference_bootstrap95_low": lower,
                "success_difference_bootstrap95_high": upper,
                "comparison_wins": comparison_wins,
                "reference_wins": reference_wins,
                "ties": sum(value == 0.0 for value in differences),
                "mcnemar_exact_p": (
                    _mcnemar_exact_p(comparison_wins, reference_wins)
                    if valid_pairs
                    else None
                ),
                "mcnemar_holm_p": None,
                "mean_policy_query_difference": (
                    float(
                        np.mean(
                            [
                                float(pair[comparison_value]["policy_queries"])
                                - float(pair[reference_value]["policy_queries"])
                                for pair in valid_pairs
                            ]
                        )
                    )
                    if valid_pairs
                    else None
                ),
                "mean_stale_action_step_difference": (
                    float(
                        np.mean(
                            [
                                float(pair[comparison_value]["stale_action_steps"])
                                - float(pair[reference_value]["stale_action_steps"])
                                for pair in valid_pairs
                            ]
                        )
                    )
                    if valid_pairs
                    else None
                ),
            }
        )
    _apply_primary_holm_correction(contrasts)
    return contrasts


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


def _write_json(path: pathlib.Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(
    path: pathlib.Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    if fieldnames is None:
        fieldnames = tuple(rows[0].keys()) if rows else ()
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _append_csv(
    path: pathlib.Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    if not rows:
        return
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _write_progress(
    output_directory: pathlib.Path,
    planned_rollouts: int,
    completed_rollouts: int,
    complete: bool,
) -> None:
    _write_json(
        output_directory / "progress.json",
        {
            "schema_version": SCHEMA_VERSION,
            "planned_rollouts": planned_rollouts,
            "completed_rollouts": completed_rollouts,
            "complete": complete,
            "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )


def initialize_incremental_artifacts(
    output_directory: pathlib.Path,
    resolved_protocol: Mapping[str, Any],
    environment: Mapping[str, Any],
    planned_rollouts: int,
) -> None:
    _write_csv(output_directory / "per_episode.csv", [], EPISODE_FIELDS)
    _write_csv(output_directory / "per_query.csv", [], QUERY_FIELDS)
    _write_json(output_directory / "resolved_protocol.json", resolved_protocol)
    _write_json(output_directory / "environment.json", environment)
    _write_progress(output_directory, planned_rollouts, 0, False)


def _summary_markdown(
    episodes: Sequence[Mapping[str, Any]],
    aggregate: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    planned_rollouts: int,
) -> str:
    completed = len(episodes)
    successes = sum(bool(row["success"]) for row in episodes)
    failures = sum(bool(row["failure_type"]) for row in episodes)

    def decimal(value: Any, digits: int = 3) -> str:
        return "n/a" if value is None else ("%.*f" % (digits, float(value)))

    lines = [
        "# pi0.5-LIBERO asynchronous runtime evaluation",
        "",
        "This artifact reports official LIBERO task completion (`done`). It does not "
        "interpret object contact as a safety violation.",
        "",
        "- Planned rollouts: %d" % planned_rollouts,
        "- Completed rollouts: %d" % completed,
        "- Successful rollouts: %d" % successes,
        "- Non-efficacy runtime/contract failures retained: %d" % failures,
        "- Artifact schema: `%s`" % SCHEMA_VERSION,
        "",
        "## Aggregate conditions",
        "",
        "| Scope | Task | Mode | Horizon | Delay steps | PP/ITT N | ITT success | ITT 95% Wilson CI | PP success | Queries | Rejections |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        if row["scope"] != "selected_tasks":
            continue
        lines.append(
            "| %s | all | %s | %d | %d | %d/%d | %s | [%s, %s] | %s | %s | %s |"
            % (
                row["task_suite"],
                row["mode"],
                row["replan_steps"],
                row["latency_steps"],
                row["eligible_rollouts"],
                row["rollouts"],
                decimal(row["success_rate"]),
                decimal(row["success_wilson95_low"]),
                decimal(row["success_wilson95_high"]),
                decimal(row["per_protocol_success_rate"]),
                decimal(row["mean_policy_queries"], digits=2),
                decimal(row["mean_rejections"], digits=2),
            )
        )
    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "| Scope | Horizon | Delay steps | Pairs | Guard - unguarded | Bootstrap 95% CI | Holm p | Query overhead |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for row in comparisons:
        if row["scope"] != "selected_tasks":
            continue
        overhead = row["relative_policy_query_overhead"]
        overhead_text = "n/a" if overhead is None else "%.1f%%" % (100.0 * overhead)
        holm_text = decimal(row["mcnemar_holm_p"], digits=4)
        lines.append(
            "| %s | %d | %d | %d | %s | [%s, %s] | %s | %s |"
            % (
                row["task_suite"],
                row["replan_steps"],
                row["latency_steps"],
                row["paired_episodes"],
                decimal(row["success_rate_difference"]),
                decimal(row["success_difference_bootstrap95_low"]),
                decimal(row["success_difference_bootstrap95_high"]),
                holm_text,
                overhead_text,
            )
        )
    lines.extend(
        [
            "",
            "Results with small N are smoke-test evidence only. A competitive claim requires "
            "the preregistered paired sample count, all negative results, and the pinned "
            "checkpoint recorded in `resolved_protocol.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def artifact_integrity_errors(
    output_directory: pathlib.Path,
    episodes: Sequence[Mapping[str, Any]],
    expected_cells: Optional[Sequence[ExperimentCell]] = None,
) -> List[str]:
    errors = []
    episode_ids = [str(row["episode_id"]) for row in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        errors.append("episode_id values are not unique")
    by_episode_id = {str(row["episode_id"]): row for row in episodes}

    if expected_cells is not None:
        expected_ids = {cell.episode_id for cell in expected_cells}
        actual_ids = set(episode_ids)
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        if missing:
            errors.append("missing planned episodes: %s" % ", ".join(missing))
        if unexpected:
            errors.append("unexpected episodes: %s" % ", ".join(unexpected))
        expected_by_pair: Dict[str, List[ExperimentCell]] = {}
        for cell in expected_cells:
            expected_by_pair.setdefault(cell.pair_id, []).append(cell)
        for pair_id, cells in expected_by_pair.items():
            expected_modes = {cell.mode for cell in cells}
            if expected_modes != {ASYNC_UNGUARDED, STATE_GUARD}:
                continue
            rows = [
                by_episode_id[cell.episode_id]
                for cell in cells
                if cell.episode_id in by_episode_id
            ]
            if len(rows) != 2:
                errors.append("incomplete paired condition: %s" % pair_id)
                continue
            pairing_fields = (
                "task_suite",
                "task_id",
                "episode_index",
                "replan_steps",
                "latency_steps",
                "seed",
                "initial_state_sha256",
            )
            if any(rows[0][field] != rows[1][field] for field in pairing_fields):
                errors.append("paired condition mismatch: %s" % pair_id)

    for row in episodes:
        if not bool(row.get("video_required")):
            continue
        video_path = row.get("video_path")
        if row.get("video_error_type"):
            errors.append(
                "video encoding failed for %s: %s"
                % (row["episode_id"], row["video_error_type"])
            )
        elif not video_path:
            errors.append("required video missing for %s" % row["episode_id"])
        else:
            resolved_video = output_directory / str(video_path)
            if not resolved_video.is_file() or resolved_video.stat().st_size <= 0:
                errors.append("required video file missing for %s" % row["episode_id"])
    return errors


def write_run_artifacts(
    output_directory: pathlib.Path,
    episodes: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    resolved_protocol: Mapping[str, Any],
    environment: Mapping[str, Any],
    planned_rollouts: int,
    bootstrap_resamples: int,
    final: bool,
    expected_cells: Optional[Sequence[ExperimentCell]] = None,
) -> Dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    if not final:
        _write_csv(output_directory / "per_episode.csv", episodes, EPISODE_FIELDS)
        _write_csv(output_directory / "per_query.csv", queries, QUERY_FIELDS)
        _write_json(output_directory / "resolved_protocol.json", resolved_protocol)
        _write_json(output_directory / "environment.json", environment)
        _write_progress(output_directory, planned_rollouts, len(episodes), False)
        return {"valid": False, "errors": ["run is not finalized"]}
    aggregate = aggregate_episodes(episodes, queries)
    comparisons = paired_comparisons(episodes, bootstrap_resamples)
    contrasts = matched_condition_contrasts(episodes, bootstrap_resamples)
    _write_csv(output_directory / "per_episode.csv", episodes, EPISODE_FIELDS)
    _write_csv(output_directory / "per_query.csv", queries, QUERY_FIELDS)
    _write_json(output_directory / "aggregate.json", aggregate)
    _write_csv(output_directory / "aggregate.csv", aggregate)
    _write_json(output_directory / "paired_comparisons.json", comparisons)
    _write_csv(output_directory / "paired_comparisons.csv", comparisons)
    _write_json(output_directory / "condition_contrasts.json", contrasts)
    _write_csv(
        output_directory / "condition_contrasts.csv",
        contrasts,
        CONTRAST_FIELDS,
    )
    _write_json(output_directory / "resolved_protocol.json", resolved_protocol)
    _write_json(output_directory / "environment.json", environment)
    integrity_errors = artifact_integrity_errors(
        output_directory, episodes, expected_cells
    )
    integrity = {
        "schema_version": SCHEMA_VERSION,
        "valid": not integrity_errors,
        "errors": integrity_errors,
        "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    _write_json(output_directory / "integrity.json", integrity)
    _write_progress(
        output_directory,
        planned_rollouts,
        len(episodes),
        len(episodes) == planned_rollouts and not integrity_errors,
    )
    summary = _summary_markdown(episodes, aggregate, comparisons, planned_rollouts)
    summary += "\n## Artifact integrity\n\n"
    summary += "- Valid: %s\n" % ("yes" if not integrity_errors else "no")
    for error in integrity_errors:
        summary += "- ERROR: %s\n" % error
    (output_directory / "summary.md").write_text(summary, encoding="utf-8")
    _write_manifest(output_directory)
    return integrity


def _write_manifest(output_directory: pathlib.Path) -> None:
    files = {}
    for path in sorted(output_directory.rglob("*")):
        if not path.is_file() or path.name == "manifest.json" or path.suffix == ".tmp":
            continue
        relative = path.relative_to(output_directory).as_posix()
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    _write_json(
        output_directory / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "files": files,
        },
    )


def _command_output(command: Sequence[str], cwd: Optional[pathlib.Path] = None) -> Optional[str]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=None if cwd is None else str(cwd),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or completed.stderr.strip() or None


def capture_environment(
    openpi_root: pathlib.Path,
    armbench_root: pathlib.Path,
    server_metadata: Mapping[str, Any],
    arguments: argparse.Namespace,
) -> Dict[str, Any]:
    packages = {}
    for package in ("numpy", "imageio", "openpi-client", "libero"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    armbench_diff = _command_output(
        ("git", "diff", "--binary", "HEAD"), cwd=armbench_root
    )
    source_hashes = {}
    for relative in RUNTIME_SOURCE_FILES:
        path = armbench_root / pathlib.PurePosixPath(relative)
        if path.is_file():
            source_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "command": sys.argv,
        "openpi_root": str(openpi_root.resolve()),
        "openpi_git_commit": _command_output(
            ("git", "rev-parse", "HEAD"), cwd=openpi_root
        ),
        "armbench_root": str(armbench_root.resolve()),
        "armbench_git_commit": _command_output(
            ("git", "rev-parse", "HEAD"), cwd=armbench_root
        ),
        "armbench_git_status": _command_output(
            ("git", "status", "--porcelain"), cwd=armbench_root
        )
        or "",
        "armbench_git_diff_sha256": hashlib.sha256(
            (armbench_diff or "").encode("utf-8")
        ).hexdigest(),
        "runtime_source_sha256": source_hashes,
        "nvidia_smi": _command_output(
            (
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            )
        ),
        "packages": packages,
        "server_metadata": _json_safe(server_metadata),
        "selected_environment_variables": {
            name: os.environ.get(name)
            for name in (
                "MUJOCO_GL",
                "MUJOCO_EGL_DEVICE_ID",
                "NVIDIA_VISIBLE_DEVICES",
                "OPENPI_DATA_HOME",
                "SERVER_ARGS",
                "ARMBENCH_SERVER_ARGS",
            )
        },
        "arguments": vars(arguments),
    }


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _write_video(path: pathlib.Path, frames: Sequence[np.ndarray], fps: int = 10) -> None:
    if not frames:
        raise ValueError("cannot write a video without frames")
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, [np.asarray(frame) for frame in frames], fps=fps)


def _make_libero_environment(task: Any, seed: int) -> Any:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    environment = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    environment.seed(seed)
    return environment


def _resolved_protocol(
    args: argparse.Namespace, cells: Sequence[ExperimentCell]
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "research_question": (
            "How do asynchronous inference delay and action execution horizon affect "
            "pi0.5-LIBERO task success and query cost, with and without state-mismatch rejection?"
        ),
        "openpi_commit": args.expected_openpi_commit,
        "policy_config": DEFAULT_POLICY_CONFIG,
        "declared_checkpoint": args.checkpoint,
        "server_launch_args": args.server_launch_args,
        "checkpoint_provenance": (
            "launcher_declaration_only"
            if args.allow_unattested_server
            else "server_attestation_with_checkpoint_content_sha256"
        ),
        "official_protocol": {
            "environment_render_resolution": [
                LIBERO_ENV_RESOLUTION,
                LIBERO_ENV_RESOLUTION,
            ],
            "resize": [args.resize_size, args.resize_size],
            "camera_rotation_degrees": 180,
            "state_dimension": 8,
            "action_dimension": 7,
            "task_success_source": "LIBERO environment done",
            "control_frequency_hz": LIBERO_CONTROL_FREQUENCY_HZ,
            "control_period_ms": LIBERO_CONTROL_PERIOD_MS,
            "video_playback_fps": 10,
        },
        "experimental_mechanism": {
            "async_delay": (
                "Advance the environment for latency_steps with the last commanded action "
                "after the query observation is captured and before its chunk is consumed."
            ),
            "state_guard": (
                "Compare current and query-time end-effector position, quaternion angular "
                "distance, and gripper position; reject and requery from a Cartesian hold."
            ),
            "mode_order": "adjacent pairs with alternating order",
            "inference_latency": "measured websocket infer wall time; not converted into injected steps",
            "step_budget": (
                "Injected delay steps consume the same environment-step budget as task actions."
            ),
        },
        "episode_budget": {
            "stabilization_steps": args.num_steps_wait,
            "task_steps_override": args.max_task_steps,
            "official_suite_task_steps": SUITE_MAX_STEPS,
        },
        "thresholds": {
            "position_m": args.position_threshold_m,
            "orientation_rad": args.orientation_threshold_rad,
            "gripper_linf": args.gripper_threshold,
            "max_requeries": args.max_requeries,
        },
        "matrix": matrix_plan(cells),
        "seed": args.seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "timeouts": {
            "server_startup_s": args.server_startup_timeout_s,
            "policy_inference_s": args.inference_timeout_s,
        },
        "runtime_failure_policy": (
            "continue_diagnostic"
            if args.continue_after_runtime_failure
            else "abort_formal_run"
        ),
        "limitations": [
            "This runtime check is not a formal safety certificate.",
            "Injected latency steps model asynchronous execution independently of measured wall latency.",
            "Identical LIBERO initial states do not reset hidden policy-server sampling state.",
            "Checkpoint attestation proves the loaded cache content and launcher identity, not the upstream publisher's intent.",
            "Object contacts are not labeled as safety violations.",
        ],
    }


def _prepare_output_directory(path: pathlib.Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("output directory must be absent or empty: %s" % path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "videos").mkdir(exist_ok=True)


def snapshot_runtime_sources(
    armbench_root: pathlib.Path, output_directory: pathlib.Path
) -> None:
    snapshot_root = output_directory / "provenance" / "armbench_source"
    required = set(RUNTIME_SOURCE_FILES[:2])
    copied = set()
    for relative in RUNTIME_SOURCE_FILES:
        source = armbench_root / pathlib.PurePosixPath(relative)
        if not source.is_file():
            continue
        destination = snapshot_root / pathlib.PurePosixPath(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        copied.add(relative)
    missing = sorted(required - copied)
    if missing:
        raise FileNotFoundError(
            "required ArmBench runtime sources are missing: %s" % ", ".join(missing)
        )


def _execute_connected_benchmark(
    args: argparse.Namespace,
    cells: Sequence[ExperimentCell],
    output_directory: pathlib.Path,
    openpi_root: pathlib.Path,
    armbench_root: pathlib.Path,
    client: BoundedOpenPIClient,
) -> int:
    from libero.libero import benchmark

    server_metadata = client.get_server_metadata()
    attested_server_source = (
        armbench_root / "integrations" / "openpi" / "serve_policy_attested.py"
    )
    if not attested_server_source.is_file() and not args.allow_unattested_server:
        raise FileNotFoundError(
            "attested server source is missing: %s" % attested_server_source
        )
    expected_server_source_sha256 = (
        hashlib.sha256(attested_server_source.read_bytes()).hexdigest()
        if attested_server_source.is_file()
        else ""
    )
    _validate_server_attestation(
        server_metadata,
        args,
        expected_server_source_sha256,
    )
    environment_record = capture_environment(
        openpi_root, armbench_root, server_metadata, args
    )
    protocol = _resolved_protocol(args, cells)
    episodes: List[Dict[str, Any]] = []
    queries: List[Dict[str, Any]] = []
    consecutive_infrastructure_failures = 0
    aborted = False
    run_errors = []
    initialize_incremental_artifacts(
        output_directory, protocol, environment_record, len(cells)
    )

    cells_by_task: Dict[int, List[ExperimentCell]] = {}
    for cell in cells:
        cells_by_task.setdefault(cell.task_id, []).append(cell)
    try:
        benchmark_class = benchmark.get_benchmark_dict()[args.task_suite]
        task_suite = benchmark_class()
    except Exception as exc:
        logging.exception("LIBERO benchmark initialization failed")
        run_errors.append(
            {
                "stage": "benchmark_initialization",
                "type": type(exc).__name__,
                "message": str(exc),
            }
        )
        aborted = True
        task_suite = None

    if task_suite is not None:
        for task_id, task_cells in cells_by_task.items():
            environment = None
            try:
                task = task_suite.get_task(task_id)
                task_description = str(task.language)
                initial_states = task_suite.get_task_init_states(task_id)
                environment = _make_libero_environment(task, args.seed)
                for cell in task_cells:
                    if cell.episode_index >= len(initial_states):
                        raise IndexError(
                            "episode index %d exceeds %d available initial states"
                            % (cell.episode_index, len(initial_states))
                        )
                    environment.seed(args.seed)
                    runtime_config = RuntimeConfig(
                        mode=cell.mode,
                        replan_steps=cell.replan_steps,
                        latency_steps=cell.latency_steps,
                        max_task_steps=(
                            args.max_task_steps
                            if args.max_task_steps is not None
                            else SUITE_MAX_STEPS[args.task_suite]
                        ),
                        num_steps_wait=args.num_steps_wait,
                        resize_size=args.resize_size,
                        position_threshold_m=args.position_threshold_m,
                        orientation_threshold_rad=args.orientation_threshold_rad,
                        gripper_threshold=args.gripper_threshold,
                        max_requeries=args.max_requeries,
                        record_video=args.video_mode != "none",
                    )
                    logging.info("Starting %s", cell.episode_id)
                    started = time.perf_counter()
                    result = run_episode(
                        environment,
                        client,
                        initial_states[cell.episode_index],
                        task_description,
                        runtime_config,
                    )
                    wall_time_s = time.perf_counter() - started
                    video_relative = None
                    video_error_type = None
                    video_error_message = None
                    should_write_video = args.video_mode == "all" or (
                        args.video_mode == "failures" and not result.success
                    )
                    if should_write_video and not result.replay_frames:
                        video_error_type = "MissingReplayFrames"
                        video_error_message = "runtime produced no replay frames"
                    elif should_write_video:
                        outcome = "success" if result.success else "failure"
                        video_name = "%s__%s.mp4" % (
                            _safe_filename(cell.episode_id),
                            outcome,
                        )
                        video_relative = "videos/%s" % video_name
                        try:
                            _write_video(
                                output_directory / video_relative,
                                result.replay_frames,
                            )
                        except Exception as exc:
                            logging.exception(
                                "Video write failed for %s", cell.episode_id
                            )
                            video_relative = None
                            video_error_type = type(exc).__name__
                            video_error_message = str(exc)
                    episode, episode_queries = episode_rows(
                        cell,
                        result,
                        task_description,
                        args.seed,
                        wall_time_s,
                        video_relative,
                        video_required=should_write_video,
                        video_error_type=video_error_type,
                        video_error_message=video_error_message,
                    )
                    episodes.append(episode)
                    queries.extend(episode_queries)
                    _append_csv(
                        output_directory / "per_episode.csv",
                        [episode],
                        EPISODE_FIELDS,
                    )
                    _append_csv(
                        output_directory / "per_query.csv",
                        episode_queries,
                        QUERY_FIELDS,
                    )
                    _write_progress(
                        output_directory,
                        len(cells),
                        len(episodes),
                        False,
                    )
                    logging.info(
                        "Finished %s success=%s reason=%s queries=%d rejections=%d",
                        cell.episode_id,
                        result.success,
                        result.termination_reason,
                        result.policy_queries,
                        result.rejected_chunks,
                    )
                    if result.failure_type:
                        consecutive_infrastructure_failures += 1
                    else:
                        consecutive_infrastructure_failures = 0
                    if (
                        result.failure_category in FORMAL_RUN_ABORT_CATEGORIES
                        and not args.continue_after_runtime_failure
                    ):
                        logging.error(
                            "Aborting formal run after failure category=%s",
                            result.failure_category,
                        )
                        aborted = True
                        break
                    if (
                        consecutive_infrastructure_failures
                        >= args.max_consecutive_infrastructure_failures
                    ):
                        logging.error(
                            "Aborting after %d consecutive runtime failures",
                            consecutive_infrastructure_failures,
                        )
                        aborted = True
                        break
            except Exception as exc:
                logging.exception("LIBERO task %d failed outside the episode runtime", task_id)
                run_errors.append(
                    {
                        "stage": "task_or_environment_runtime",
                        "task_id": task_id,
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                aborted = True
            finally:
                close = getattr(environment, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as exc:
                        logging.exception("LIBERO environment close failed")
                        run_errors.append(
                            {
                                "stage": "environment_close",
                                "task_id": task_id,
                                "type": type(exc).__name__,
                                "message": str(exc),
                            }
                        )
                        aborted = True
            if aborted:
                break

    if run_errors:
        _write_json(
            output_directory / "run_error.json",
            {
                "schema_version": SCHEMA_VERSION,
                "errors": run_errors,
                "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
    integrity = write_run_artifacts(
        output_directory,
        episodes,
        queries,
        protocol,
        environment_record,
        len(cells),
        args.bootstrap_resamples,
        final=True,
        expected_cells=cells,
    )
    return (
        2
        if aborted or len(episodes) != len(cells) or not integrity["valid"]
        else 0
    )


def execute_benchmark(args: argparse.Namespace, cells: Sequence[ExperimentCell]) -> int:
    output_directory = pathlib.Path(args.output_dir).resolve()
    openpi_root = pathlib.Path(args.openpi_root).resolve()
    armbench_root = pathlib.Path(args.armbench_root).resolve()
    actual_commit = _command_output(("git", "rev-parse", "HEAD"), cwd=openpi_root)
    if actual_commit != args.expected_openpi_commit and not args.allow_commit_mismatch:
        raise RuntimeError(
            "OpenPI commit mismatch: expected %s, got %s"
            % (args.expected_openpi_commit, actual_commit)
        )

    _prepare_output_directory(output_directory)
    snapshot_runtime_sources(armbench_root, output_directory)
    log_path = output_directory / "run.log"
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)
    np.random.seed(args.seed)

    try:
        client = BoundedOpenPIClient(
            args.host,
            args.port,
            startup_timeout_s=args.server_startup_timeout_s,
            inference_timeout_s=args.inference_timeout_s,
        )
    except Exception as exc:
        logging.exception("OpenPI client startup failed")
        _write_json(
            output_directory / "run_error.json",
            {
                "schema_version": SCHEMA_VERSION,
                "errors": [
                    {
                        "stage": "policy_client_startup",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
                "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
        _write_manifest(output_directory)
        return 2

    try:
        return _execute_connected_benchmark(
            args,
            cells,
            output_directory,
            openpi_root,
            armbench_root,
            client,
        )
    finally:
        try:
            client.close()
        except Exception:
            logging.exception("OpenPI client close failed")


def _add_matrix_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task-suite", choices=tuple(SUITE_TASK_COUNTS), default="libero_spatial")
    parser.add_argument("--task-ids", default="0", help="Comma list, start:stop range, or all")
    parser.add_argument("--episode-indices", default="0", help="Comma list, start:stop range, or all")
    parser.add_argument(
        "--modes", default=ASYNC_UNGUARDED, help="Comma-separated runtime modes"
    )
    parser.add_argument("--replan-steps", default="5", help="Comma-separated positive horizons")
    parser.add_argument("--latency-steps", default="0", help="Comma-separated injected delays")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan", help="Resolve matrix size without loading LIBERO")
    _add_matrix_arguments(plan_parser)

    run_parser = subparsers.add_parser("run", help="Execute the resolved matrix")
    _add_matrix_arguments(run_parser)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--host", default="0.0.0.0")
    run_parser.add_argument("--port", type=int, default=8000)
    run_parser.add_argument("--server-startup-timeout-s", type=float, default=1200.0)
    run_parser.add_argument("--inference-timeout-s", type=float, default=600.0)
    run_parser.add_argument("--openpi-root", default="/app")
    run_parser.add_argument("--armbench-root", default="/armbench")
    run_parser.add_argument("--expected-openpi-commit", default=OPENPI_COMMIT)
    run_parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    run_parser.add_argument(
        "--server-launch-args",
        default=os.environ.get("ARMBENCH_SERVER_ARGS")
        or os.environ.get("SERVER_ARGS"),
        help="Exact OpenPI server launch arguments retained as checkpoint provenance",
    )
    run_parser.add_argument("--allow-commit-mismatch", action="store_true")
    run_parser.add_argument(
        "--allow-unattested-server",
        action="store_true",
        help="Diagnostic only: accept official server metadata without checkpoint attestation",
    )
    run_parser.add_argument("--resize-size", type=int, default=224)
    run_parser.add_argument("--num-steps-wait", type=int, default=10)
    run_parser.add_argument("--max-task-steps", type=int)
    run_parser.add_argument("--position-threshold-m", type=float, default=0.01)
    run_parser.add_argument("--orientation-threshold-rad", type=float, default=0.10)
    run_parser.add_argument("--gripper-threshold", type=float, default=0.05)
    run_parser.add_argument("--max-requeries", type=int, default=2)
    run_parser.add_argument("--seed", type=int, default=7)
    run_parser.add_argument("--video-mode", choices=("none", "failures", "all"), default="failures")
    run_parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    run_parser.add_argument("--max-consecutive-infrastructure-failures", type=int, default=3)
    run_parser.add_argument(
        "--continue-after-runtime-failure",
        action="store_true",
        help="Diagnostic only: continue after environment/transport failures",
    )
    return parser


def _resolve_matrix(args: argparse.Namespace) -> List[ExperimentCell]:
    task_ids = _parse_int_selection(
        args.task_ids, SUITE_TASK_COUNTS[args.task_suite], "task_ids"
    )
    episode_indices = _parse_int_selection(
        args.episode_indices, 50, "episode_indices"
    )
    return build_matrix(
        args.task_suite,
        task_ids,
        episode_indices,
        _parse_modes(args.modes),
        _parse_positive_csv(args.replan_steps, "replan_steps"),
        _parse_nonnegative_csv(args.latency_steps, "latency_steps"),
    )


def _validate_run_arguments(
    args: argparse.Namespace, cells: Sequence[ExperimentCell]
) -> None:
    if args.resize_size != 224:
        raise ValueError("official pi05_libero evaluation requires --resize-size 224")
    if args.num_steps_wait != 10:
        raise ValueError("official LIBERO evaluation requires --num-steps-wait 10")
    if args.port <= 0 or args.port > 65535:
        raise ValueError("port must be in [1, 65535]")
    if args.max_task_steps is not None and args.max_task_steps <= 0:
        raise ValueError("max_task_steps must be positive when provided")
    if any(cell.replan_steps > PI05_LIBERO_ACTION_HORIZON for cell in cells):
        raise ValueError(
            "replan_steps cannot exceed pi05_libero action horizon %d"
            % PI05_LIBERO_ACTION_HORIZON
        )
    if args.bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    if args.max_consecutive_infrastructure_failures <= 0:
        raise ValueError("max_consecutive_infrastructure_failures must be positive")
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_openpi_commit):
        raise ValueError("expected_openpi_commit must be a lowercase 40-character SHA")
    _validate_server_launch_args(args.server_launch_args, args.checkpoint)
    max_steps = (
        args.max_task_steps
        if args.max_task_steps is not None
        else SUITE_MAX_STEPS[args.task_suite]
    )
    for cell in cells:
        RuntimeConfig(
            mode=cell.mode,
            replan_steps=cell.replan_steps,
            latency_steps=cell.latency_steps,
            max_task_steps=max_steps,
            num_steps_wait=args.num_steps_wait,
            resize_size=args.resize_size,
            position_threshold_m=args.position_threshold_m,
            orientation_threshold_rad=args.orientation_threshold_rad,
            gripper_threshold=args.gripper_threshold,
            max_requeries=args.max_requeries,
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        cells = _resolve_matrix(args)
        if args.command == "plan":
            print(json.dumps(matrix_plan(cells), indent=2, sort_keys=True))
            return 0
        _validate_run_arguments(args, cells)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        return execute_benchmark(args, cells)
    except Exception as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
