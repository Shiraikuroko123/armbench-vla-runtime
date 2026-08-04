"""Measured-age pi0.5-LIBERO evaluator with a versioned artifact contract.

The legacy deterministic-delay evaluator remains unchanged.  This module fixes
``latency_source=measured_wall`` and ``latency_steps=0`` and emits a distinct v2
schema whose query rows retain the monotonic clock samples used by the runtime.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import math
import os
import pathlib
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from integrations.openpi.deadline_alignment import (
    CEIL,
    FLOOR,
    VALID_ROUNDING,
    completed_control_steps,
    estimate_stale_steps,
    keyed_discrete_jitter_ms,
)
from integrations.openpi.libero_runtime import (
    ASYNC_UNGUARDED,
    LATENCY_ALIGNED,
    LIBERO_DUMMY_ACTION,
    MEASURED_WALL_LATENCY,
    EpisodeResult,
    QueryRecord,
    RuntimeConfig,
    build_libero_request,
    run_episode,
    validate_action_chunk,
)
from integrations.openpi.libero_runtime_eval import (
    BoundedOpenPIClient,
    DEFAULT_CHECKPOINT,
    DEFAULT_POLICY_CONFIG,
    LIBERO_CONTROL_FREQUENCY_HZ,
    LIBERO_CONTROL_PERIOD_MS,
    LIBERO_ENV_RESOLUTION,
    OPENPI_COMMIT,
    PI05_LIBERO_ACTION_HORIZON,
    SUITE_MAX_STEPS,
    SUITE_TASK_COUNTS,
    ExperimentCell,
    _command_output,
    _make_libero_environment,
    _safe_filename,
    _validate_server_attestation,
    _validate_server_launch_args,
    _write_video,
    build_matrix,
    capture_environment as capture_v1_environment,
)


SCHEMA_VERSION = "armbench.pi05_libero_measured_age.v2"
VALID_MODES = (ASYNC_UNGUARDED, LATENCY_ALIGNED)
DEFAULT_JITTER_CANDIDATES_MS = (0.0, 40.0, 80.0, 160.0)
DEFAULT_WARMUP_QUERIES = 3
FORMAL_ABORT_CATEGORIES = frozenset(
    (
        "environment_runtime",
        "observation_contract",
        "policy_transport_or_server",
        "policy_timeout",
    )
)

RUNTIME_SOURCE_FILES = (
    "integrations/openpi/libero_runtime.py",
    "integrations/openpi/libero_runtime_eval.py",
    "integrations/openpi/measured_age_libero_eval.py",
    "integrations/openpi/validate_measured_age_artifact.py",
    "integrations/openpi/measured_age_compose_run.py",
    "integrations/openpi/deadline_alignment.py",
    "integrations/openpi/preflight.py",
    "integrations/openpi/compose.libero-runtime.yml",
    "integrations/openpi/compose.libero-measured-age.yml",
    "integrations/openpi/serve_policy_attested.py",
)

WARMUP_FIELDS = (
    "schema_version",
    "warmup_index",
    "scored",
    "client_session_id",
    "checkpoint_content_sha256",
    "observation_captured_monotonic_ns",
    "policy_call_started_monotonic_ns",
    "policy_call_finished_monotonic_ns",
    "response_ready_monotonic_ns",
    "observation_age_ms",
    "inference_latency_ms",
    "action_chunk_steps",
    "action_dimension",
    "accepted",
    "error_type",
    "error_message",
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
    "latency_source",
    "replan_steps",
    "latency_steps",
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
    "deadline_misses",
    "horizon_overruns",
    "age_refreshes",
    "fallback_hold_steps",
    "simulated_catchup_steps",
    "observation_age_p50_ms",
    "observation_age_p95_ms",
    "observation_age_max_ms",
    "inference_latency_p50_ms",
    "inference_latency_p95_ms",
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
    "condition_order",
    "task_suite",
    "task_id",
    "episode_index",
    "mode",
    "latency_source",
    "replan_steps",
    "latency_steps",
    "query_index",
    "observation_step",
    "response_step",
    "observation_captured_monotonic_ns",
    "policy_call_started_monotonic_ns",
    "policy_call_finished_monotonic_ns",
    "response_ready_monotonic_ns",
    "clock_trace_complete",
    "observation_age_ms",
    "inference_latency_ms",
    "response_delivery_elapsed_ms",
    "policy_inference_latency_ms",
    "server_inference_latency_ms",
    "response_jitter_requested_ms",
    "jitter_key_sha256",
    "completed_controller_steps",
    "simulated_catchup_steps",
    "action_chunk_steps",
    "measured_stale_steps",
    "action_offset_steps",
    "selected_stop_step",
    "available_suffix_steps",
    "deadline_exceeded",
    "horizon_overrun",
    "age_refresh_index",
    "fallback_hold_steps",
    "alignment_disposition",
    "alignment_reason",
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


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("artifact values must be finite")
    return value


def _write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(
    path: pathlib.Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _jitter_payload(cell: ExperimentCell, seed: int, query_index: int) -> bytes:
    return json.dumps(
        {
            "pairing_key": [
                cell.task_suite,
                int(cell.task_id),
                int(cell.episode_index),
                int(cell.replan_steps),
            ],
            "query_index": int(query_index),
            "seed": int(seed),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def jitter_key_sha256(cell: ExperimentCell, seed: int, query_index: int) -> str:
    """Hash the exact mode-independent jitter key retained in each query row."""

    return _sha256_bytes(_jitter_payload(cell, seed, query_index))


def jitter_value_ms(
    cell: ExperimentCell,
    seed: int,
    query_index: int,
    candidates_ms: Sequence[float],
) -> float:
    return keyed_discrete_jitter_ms(
        seed=int(seed),
        pairing_key=(
            cell.task_suite,
            int(cell.task_id),
            int(cell.episode_index),
            int(cell.replan_steps),
        ),
        query_index=int(query_index),
        values_ms=tuple(candidates_ms),
    )


def make_jitter_provider(
    cell: ExperimentCell, seed: int, candidates_ms: Sequence[float]
) -> Callable[[int], float]:
    candidates = tuple(float(value) for value in candidates_ms)

    def provide(query_index: int) -> float:
        return jitter_value_ms(cell, seed, query_index, candidates)

    return provide


def _parse_modes(value: str) -> List[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    if not modes or len(modes) != len(set(modes)):
        raise ValueError("modes must be nonempty and unique")
    invalid = sorted(set(modes) - set(VALID_MODES))
    if invalid:
        raise ValueError(
            "measured-age v2 only allows modes: %s" % ", ".join(VALID_MODES)
        )
    return modes


def _parse_int_selection(value: str, upper: int, name: str) -> List[int]:
    if value == "all":
        return list(range(upper))
    values: List[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise ValueError("%s contains an empty selection" % name)
        if ":" in item:
            parts = item.split(":")
            if len(parts) != 2:
                raise ValueError("%s range must be start:stop" % name)
            start, stop = (int(part) for part in parts)
            values.extend(range(start, stop))
        else:
            values.append(int(item))
    if not values or len(values) != len(set(values)):
        raise ValueError("%s must be nonempty and unique" % name)
    if any(item < 0 or item >= upper for item in values):
        raise ValueError("%s values must be in [0, %d)" % (name, upper))
    return values


def _parse_positive_csv(value: str, name: str) -> List[int]:
    try:
        values = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError("%s must contain integers" % name) from exc
    if not values or any(item <= 0 for item in values) or len(values) != len(set(values)):
        raise ValueError("%s must contain unique positive integers" % name)
    return values


def parse_jitter_candidates(value: str) -> Tuple[float, ...]:
    try:
        candidates = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("jitter candidates must be numeric") from exc
    if (
        not candidates
        or any(not math.isfinite(item) or item < 0.0 for item in candidates)
        or len(candidates) != len(set(candidates))
    ):
        raise ValueError("jitter candidates must be unique, finite, and nonnegative")
    return candidates


def resolve_matrix(args: argparse.Namespace) -> List[ExperimentCell]:
    modes = _parse_modes(args.modes)
    horizons = _parse_positive_csv(args.replan_steps, "replan_steps")
    if any(horizon > PI05_LIBERO_ACTION_HORIZON for horizon in horizons):
        raise ValueError(
            "replan_steps cannot exceed pi05_libero action horizon %d"
            % PI05_LIBERO_ACTION_HORIZON
        )
    return build_matrix(
        args.task_suite,
        _parse_int_selection(
            args.task_ids, SUITE_TASK_COUNTS[args.task_suite], "task_ids"
        ),
        _parse_int_selection(args.episode_indices, 50, "episode_indices"),
        modes,
        horizons,
        (0,),
    )


def matrix_plan(cells: Sequence[ExperimentCell]) -> Dict[str, Any]:
    pairs = {cell.pair_id for cell in cells}
    return {
        "schema_version": SCHEMA_VERSION,
        "rollouts": len(cells),
        "paired_conditions": len(pairs),
        "matched_condition_groups": len(pairs),
        "task_suites": sorted({cell.task_suite for cell in cells}),
        "task_ids": sorted({cell.task_id for cell in cells}),
        "episode_indices": sorted({cell.episode_index for cell in cells}),
        "modes": sorted({cell.mode for cell in cells}),
        "replan_steps": sorted({cell.replan_steps for cell in cells}),
        "latency_sources": [MEASURED_WALL_LATENCY],
        "latency_steps": [0],
    }


def resolved_protocol(
    args: argparse.Namespace,
    cells: Sequence[ExperimentCell],
    jitter_candidates_ms: Sequence[float],
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "research_question": (
            "Can a frozen pi0.5 policy remain useful under measured, variable response "
            "age when the runtime must select a suffix or fail closed?"
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
            "action_dimension": 7,
            "action_horizon": PI05_LIBERO_ACTION_HORIZON,
            "control_frequency_hz": LIBERO_CONTROL_FREQUENCY_HZ,
            "environment_render_resolution": [
                LIBERO_ENV_RESOLUTION,
                LIBERO_ENV_RESOLUTION,
            ],
            "resize": [args.resize_size, args.resize_size],
            "state_dimension": 8,
            "stabilization_steps": args.num_steps_wait,
            "task_success_source": "LIBERO environment done",
        },
        "temporal_alignment": {
            "latency_source": MEASURED_WALL_LATENCY,
            "latency_steps": 0,
            "clock": "time.perf_counter_ns",
            "age_origin": "before request construction",
            "age_endpoint": "after response delivery jitter",
            "control_period_ms": args.control_period_ms,
            "completed_step_rounding": FLOOR,
            "action_offset_rounding": args.age_rounding,
            "boundary_tolerance_ms": 1e-9,
            "deadline_ms": args.deadline_ms,
            "max_age_refreshes": args.max_age_refreshes,
            "controller_model": "post_response_catchup_simulation",
            "hold_policy": "zero Cartesian delta and preserve gripper",
        },
        "jitter": {
            "generator": "sha256_first_u64_mod_v1",
            "seed": args.seed,
            "candidates_ms": list(jitter_candidates_ms),
            "pairing_key_fields": [
                "task_suite",
                "task_id",
                "episode_index",
                "replan_steps",
            ],
            "query_index_field": "query_index",
            "mode_in_key": False,
            "payload_fields": ["seed", "pairing_key", "query_index"],
            "json_encoding": "utf-8 canonical sorted compact ASCII",
        },
        "warmup": {
            "queries": args.warmup_queries,
            "scored": False,
            "required_before_scoring": True,
            "same_checkpoint_and_action_contract": True,
            "task_suite": args.task_suite,
            "task_id": args.warmup_task_id,
            "episode_index": args.warmup_episode_index,
        },
        "matrix": matrix_plan(cells),
        "registered_cells": [
            {
                "condition_order": cell.condition_order,
                "episode_id": cell.episode_id,
                "pair_id": cell.pair_id,
                "task_suite": cell.task_suite,
                "task_id": cell.task_id,
                "episode_index": cell.episode_index,
                "mode": cell.mode,
                "replan_steps": cell.replan_steps,
                "latency_source": MEASURED_WALL_LATENCY,
                "latency_steps": 0,
            }
            for cell in cells
        ],
        "seed": args.seed,
        "episode_budget": {
            "task_steps_override": args.max_task_steps,
            "official_suite_task_steps": SUITE_MAX_STEPS,
        },
        "timeouts": {
            "server_startup_s": args.server_startup_timeout_s,
            "policy_inference_s": args.inference_timeout_s,
        },
        "limitations": [
            "Inference remains blocking; catch-up is simulated after response arrival.",
            "This is not an OS hard-real-time loop or a formal safety certificate.",
            "Temporal suffix selection is not dynamics-aware action repair.",
        ],
    }


def _percentiles(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    return float(np.percentile(values, 50)), float(np.percentile(values, 95))


def _alignment_disposition(mode: str, record: QueryRecord) -> str:
    if record.error_stage is not None:
        return "error"
    if record.decision in (
        "success_during_inference_delay",
        "step_budget_exhausted_during_delay",
    ):
        return "terminal_before_dispatch"
    if mode == ASYNC_UNGUARDED:
        return "not_applied"
    if record.alignment_disposition is not None:
        return str(record.alignment_disposition)
    if record.accepted:
        return "execute"
    if record.decision.endswith("_hold_refresh"):
        return "hold_refresh"
    if record.decision.endswith("_fail_closed"):
        return "fail_closed"
    return "terminal_before_dispatch"


def _alignment_reason(mode: str, record: QueryRecord) -> str:
    if record.error_stage is not None:
        return str(record.error_stage)
    if record.decision in (
        "success_during_inference_delay",
        "step_budget_exhausted_during_delay",
    ):
        return str(record.decision)
    if mode == ASYNC_UNGUARDED:
        return "async_unguarded"
    if record.alignment_reason is not None:
        return str(record.alignment_reason)
    if record.accepted:
        return "fresh_suffix_available"
    if record.deadline_exceeded and record.horizon_overrun:
        return "deadline_and_horizon_overrun"
    if record.deadline_exceeded:
        return "deadline_exceeded"
    if record.horizon_overrun:
        return "horizon_overrun"
    return str(record.decision)


def result_rows(
    cell: ExperimentCell,
    result: EpisodeResult,
    task_description: str,
    seed: int,
    wall_time_s: float,
    jitter_candidates_ms: Sequence[float],
    control_period_ms: float,
    video_path: Optional[str] = None,
    video_required: bool = False,
    video_error_type: Optional[str] = None,
    video_error_message: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    queries: List[Dict[str, Any]] = []
    for record in result.query_records:
        mismatch = record.mismatch
        stale_steps = record.measured_stale_steps
        selected_stop = (
            record.selected_stop_step
            if record.selected_stop_step is not None
            else (
                None
                if stale_steps is None
                else int(stale_steps) + int(cell.replan_steps)
            )
        )
        observation_age = record.observation_age_ms
        completed_steps = (
            None
            if observation_age is None
            else completed_control_steps(observation_age, control_period_ms)
        )
        raw_clock = (
            record.observation_captured_monotonic_ns,
            record.policy_call_started_monotonic_ns,
            record.policy_call_finished_monotonic_ns,
            record.response_ready_monotonic_ns,
        )
        query = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": cell.episode_id,
            "pair_id": cell.pair_id,
            "condition_order": cell.condition_order,
            "task_suite": cell.task_suite,
            "task_id": cell.task_id,
            "episode_index": cell.episode_index,
            "mode": cell.mode,
            "latency_source": record.latency_source,
            "replan_steps": cell.replan_steps,
            "latency_steps": 0,
            "query_index": record.query_index,
            "observation_step": record.observation_step,
            "response_step": record.response_step,
            "observation_captured_monotonic_ns": raw_clock[0],
            "policy_call_started_monotonic_ns": raw_clock[1],
            "policy_call_finished_monotonic_ns": raw_clock[2],
            "response_ready_monotonic_ns": raw_clock[3],
            "clock_trace_complete": all(value is not None for value in raw_clock),
            "observation_age_ms": observation_age,
            "inference_latency_ms": record.inference_latency_ms,
            "response_delivery_elapsed_ms": record.response_delivery_elapsed_ms,
            "policy_inference_latency_ms": record.policy_inference_latency_ms,
            "server_inference_latency_ms": record.server_inference_latency_ms,
            "response_jitter_requested_ms": record.response_jitter_ms,
            "jitter_key_sha256": jitter_key_sha256(
                cell, seed, record.query_index
            ),
            "completed_controller_steps": completed_steps,
            "simulated_catchup_steps": record.simulated_catchup_steps,
            "action_chunk_steps": record.action_chunk_steps,
            "measured_stale_steps": stale_steps,
            "action_offset_steps": record.action_offset_steps,
            "selected_stop_step": selected_stop,
            "available_suffix_steps": record.available_suffix_steps,
            "deadline_exceeded": record.deadline_exceeded,
            "horizon_overrun": record.horizon_overrun,
            "age_refresh_index": record.age_refresh_index,
            "fallback_hold_steps": record.fallback_hold_steps,
            "alignment_disposition": _alignment_disposition(cell.mode, record),
            "alignment_reason": _alignment_reason(cell.mode, record),
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
        queries.append(query)

    ages = [
        float(row["observation_age_ms"])
        for row in queries
        if row["observation_age_ms"] is not None
    ]
    inference = [float(row["inference_latency_ms"]) for row in queries]
    age_p50, age_p95 = _percentiles(ages)
    inference_p50, inference_p95 = _percentiles(inference)
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
        "latency_source": MEASURED_WALL_LATENCY,
        "replan_steps": cell.replan_steps,
        "latency_steps": 0,
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
        "deadline_misses": result.deadline_misses,
        "horizon_overruns": result.horizon_overruns,
        "age_refreshes": result.age_refreshes,
        "fallback_hold_steps": result.fallback_hold_steps,
        "simulated_catchup_steps": sum(
            int(row["simulated_catchup_steps"] or 0) for row in queries
        ),
        "observation_age_p50_ms": age_p50,
        "observation_age_p95_ms": age_p95,
        "observation_age_max_ms": max(ages) if ages else None,
        "inference_latency_p50_ms": inference_p50,
        "inference_latency_p95_ms": inference_p95,
        "wall_time_s": wall_time_s,
        "failure_category": result.failure_category,
        "failure_type": result.failure_type,
        "failure_message": result.failure_message,
        "video_required": video_required,
        "video_path": video_path,
        "video_error_type": video_error_type,
        "video_error_message": video_error_message,
    }
    return episode, queries


def _summary(
    episodes: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    warmups: Sequence[Mapping[str, Any]],
    planned_rollouts: int,
    complete: bool,
    warmup_queries_planned: int,
) -> Dict[str, Any]:
    aggregate = []
    for mode in VALID_MODES:
        mode_episodes = [row for row in episodes if row["mode"] == mode]
        mode_queries = [row for row in queries if row["mode"] == mode]
        ages = [
            float(row["observation_age_ms"])
            for row in mode_queries
            if row["observation_age_ms"] is not None
        ]
        p50, p95 = _percentiles(ages)
        aggregate.append(
            {
                "mode": mode,
                "rollouts": len(mode_episodes),
                "successes": sum(bool(row["success"]) for row in mode_episodes),
                "success_rate": (
                    None
                    if not mode_episodes
                    else sum(bool(row["success"]) for row in mode_episodes)
                    / float(len(mode_episodes))
                ),
                "policy_queries": len(mode_queries),
                "mean_policy_queries": (
                    None
                    if not mode_episodes
                    else float(
                        np.mean([int(row["policy_queries"]) for row in mode_episodes])
                    )
                ),
                "observation_age_p50_ms": p50,
                "observation_age_p95_ms": p95,
                "observation_age_max_ms": max(ages) if ages else None,
                "deadline_misses": sum(
                    int(row["deadline_misses"]) for row in mode_episodes
                ),
                "horizon_overruns": sum(
                    int(row["horizon_overruns"]) for row in mode_episodes
                ),
                "age_refreshes": sum(
                    int(row["age_refreshes"]) for row in mode_episodes
                ),
                "fallback_hold_steps": sum(
                    int(row["fallback_hold_steps"]) for row in mode_episodes
                ),
            }
        )

    by_pair: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for row in episodes:
        by_pair.setdefault(str(row["pair_id"]), {})[str(row["mode"])] = row
    pairs = [
        modes
        for modes in by_pair.values()
        if set(modes) == set(VALID_MODES)
    ]
    async_successes = sum(
        bool(pair[ASYNC_UNGUARDED]["success"]) for pair in pairs
    )
    aligned_successes = sum(
        bool(pair[LATENCY_ALIGNED]["success"]) for pair in pairs
    )
    paired = {
        "pairs": len(pairs),
        "async_successes": async_successes,
        "aligned_successes": aligned_successes,
        "candidate_wins": sum(
            bool(pair[LATENCY_ALIGNED]["success"])
            and not bool(pair[ASYNC_UNGUARDED]["success"])
            for pair in pairs
        ),
        "reference_wins": sum(
            bool(pair[ASYNC_UNGUARDED]["success"])
            and not bool(pair[LATENCY_ALIGNED]["success"])
            for pair in pairs
        ),
        "ties": sum(
            bool(pair[ASYNC_UNGUARDED]["success"])
            == bool(pair[LATENCY_ALIGNED]["success"])
            for pair in pairs
        ),
        "success_rate_difference": (
            None
            if not pairs
            else (aligned_successes - async_successes) / float(len(pairs))
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "planned_rollouts": planned_rollouts,
        "completed_rollouts": len(episodes),
        "valid": complete,
        "complete": complete,
        "warmup_queries_planned": warmup_queries_planned,
        "warmup_queries_completed": len(warmups),
        "warmup_queries_valid": sum(bool(row["accepted"]) for row in warmups),
        "aggregate": aggregate,
        "paired": paired,
    }


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# pi0.5-LIBERO measured-age evaluation",
        "",
        "- Schema: `%s`" % summary["schema_version"],
        "- Planned/completed rollouts: %d/%d"
        % (summary["planned_rollouts"], summary["completed_rollouts"]),
        "- Non-scoring warm-up queries: %d"
        % summary["warmup_queries_completed"],
        "- Complete: %s" % ("yes" if summary["complete"] else "no"),
        "",
        "| Mode | Success | Queries | Age P95 ms | Deadline misses | Horizon overruns |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["aggregate"]:
        age_p95 = (
            "n/a"
            if row["observation_age_p95_ms"] is None
            else "%.3f" % row["observation_age_p95_ms"]
        )
        lines.append(
            "| %s | %d/%d | %d | %s | %d | %d |"
            % (
                row["mode"],
                row["successes"],
                row["rollouts"],
                row["policy_queries"],
                age_p95,
                row["deadline_misses"],
                row["horizon_overruns"],
            )
        )
    lines.extend(
        [
            "",
            "This artifact uses blocking inference plus post-response simulator catch-up. "
            "It is not an OS hard-real-time or real-robot result.",
            "",
        ]
    )
    return "\n".join(lines)


def artifact_errors(
    protocol: Mapping[str, Any],
    warmups: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    planned_rollouts: int,
    complete: bool,
) -> List[str]:
    """Recompute the core v2 timing and row invariants before finalization."""

    errors: List[str] = []
    if protocol.get("schema_version") != SCHEMA_VERSION:
        errors.append("protocol schema_version mismatch")
        return errors
    temporal = protocol.get("temporal_alignment")
    jitter = protocol.get("jitter")
    if not isinstance(temporal, Mapping) or not isinstance(jitter, Mapping):
        errors.append("protocol timing or jitter section is missing")
        return errors
    period = float(temporal["control_period_ms"])
    tolerance = float(temporal["boundary_tolerance_ms"])
    rounding = str(temporal["action_offset_rounding"])
    deadline = temporal.get("deadline_ms")
    candidates = tuple(float(value) for value in jitter["candidates_ms"])
    seed = int(jitter["seed"])

    expected_warmups = int(protocol["warmup"]["queries"])
    if len(warmups) != expected_warmups:
        errors.append("warmup query count mismatch")
    if any(
        row.get("schema_version") != SCHEMA_VERSION
        or bool(row.get("scored"))
        or not bool(row.get("accepted"))
        for row in warmups
    ):
        errors.append("warmup rows are not valid non-scoring successes")
    if complete and len(episodes) != planned_rollouts:
        errors.append("completed rollout count does not match plan")

    registered = {
        str(row["episode_id"]): row for row in protocol.get("registered_cells", [])
    }
    episode_by_id = {str(row["episode_id"]): row for row in episodes}
    if len(episode_by_id) != len(episodes):
        errors.append("episode_id values are not unique")
    query_by_episode: Dict[str, List[Mapping[str, Any]]] = {}
    first_scored_ns: Optional[int] = None
    for row in queries:
        episode_id = str(row.get("episode_id"))
        query_by_episode.setdefault(episode_id, []).append(row)
        if row.get("schema_version") != SCHEMA_VERSION:
            errors.append("query schema_version mismatch")
            continue
        if episode_id not in episode_by_id or episode_id not in registered:
            errors.append("query references unknown episode %s" % episode_id)
            continue
        if row.get("latency_source") != MEASURED_WALL_LATENCY or int(
            row.get("latency_steps", -1)
        ) != 0:
            errors.append("query does not use measured_wall with zero fixed latency")
        raw = (
            row.get("observation_captured_monotonic_ns"),
            row.get("policy_call_started_monotonic_ns"),
            row.get("policy_call_finished_monotonic_ns"),
            row.get("response_ready_monotonic_ns"),
        )
        if not bool(row.get("clock_trace_complete")) or any(
            value is None for value in raw
        ):
            errors.append("query %s/%s lacks a complete clock trace" % (episode_id, row["query_index"]))
            continue
        samples = tuple(int(value) for value in raw)
        if list(samples) != sorted(samples):
            errors.append("query clock samples are not monotonic")
            continue
        first_scored_ns = (
            samples[0]
            if first_scored_ns is None
            else min(first_scored_ns, samples[0])
        )
        age = (samples[3] - samples[0]) / 1_000_000.0
        delivery = (samples[3] - samples[2]) / 1_000_000.0
        if not math.isclose(age, float(row["observation_age_ms"]), abs_tol=1e-3):
            errors.append("observation age does not match raw clock samples")
        if not math.isclose(
            delivery, float(row["response_delivery_elapsed_ms"]), abs_tol=1e-3
        ):
            errors.append("response delivery time does not match raw clock samples")
        if row.get("error_stage") is not None:
            errors.append(
                "query %s/%s records runtime error %s"
                % (episode_id, row["query_index"], row["error_stage"])
            )
            continue
        completed = completed_control_steps(age, period, tolerance)
        stale = estimate_stale_steps(age, period, rounding, tolerance)
        if int(row["completed_controller_steps"]) != completed:
            errors.append("completed controller step mismatch")
        if int(row["measured_stale_steps"]) != stale:
            errors.append("measured stale step mismatch")
        simulated_catchup = row.get("simulated_catchup_steps")
        if simulated_catchup is None or int(simulated_catchup) != int(
            row["response_step"]
        ) - int(row["observation_step"]):
            errors.append("simulated catch-up step mismatch")
        expected_key = jitter_key_sha256(
            ExperimentCell(
                task_suite=str(row["task_suite"]),
                task_id=int(row["task_id"]),
                episode_index=int(row["episode_index"]),
                mode=str(row["mode"]),
                replan_steps=int(row["replan_steps"]),
                latency_steps=0,
                pair_id=str(row["pair_id"]),
                condition_order=int(row["condition_order"]),
            ),
            seed,
            int(row["query_index"]),
        )
        if row["jitter_key_sha256"] != expected_key:
            errors.append("jitter key hash mismatch")
        expected_jitter = jitter_value_ms(
            ExperimentCell(
                task_suite=str(row["task_suite"]),
                task_id=int(row["task_id"]),
                episode_index=int(row["episode_index"]),
                mode=str(row["mode"]),
                replan_steps=int(row["replan_steps"]),
                latency_steps=0,
                pair_id=str(row["pair_id"]),
                condition_order=int(row["condition_order"]),
            ),
            seed,
            int(row["query_index"]),
            candidates,
        )
        if not math.isclose(
            float(row["response_jitter_requested_ms"]),
            expected_jitter,
            abs_tol=1e-9,
        ):
            errors.append("requested response jitter mismatch")
        expected_offset = stale if row["mode"] == LATENCY_ALIGNED else 0
        if int(row["action_offset_steps"]) != expected_offset:
            errors.append("action offset mismatch")
        if int(row["selected_stop_step"]) != stale + int(row["replan_steps"]):
            errors.append("selected stop step mismatch")
        if int(row["available_suffix_steps"]) != max(
            0, int(row["action_chunk_steps"]) - stale
        ):
            errors.append("available suffix mismatch")
        expected_deadline = deadline is not None and age > float(deadline) + tolerance
        expected_overrun = stale + int(row["replan_steps"]) > int(
            row["action_chunk_steps"]
        )
        if bool(row["deadline_exceeded"]) != expected_deadline:
            errors.append("deadline flag mismatch")
        if bool(row["horizon_overrun"]) != expected_overrun:
            errors.append("horizon-overrun flag mismatch")
        if (
            row["mode"] == LATENCY_ALIGNED
            and bool(row["accepted"])
            and (expected_deadline or expected_overrun)
        ):
            errors.append("aligned query accepted an unavailable suffix")

    if warmups and first_scored_ns is not None:
        warmup_ready = max(int(row["response_ready_monotonic_ns"]) for row in warmups)
        if warmup_ready > first_scored_ns:
            errors.append("warmup did not complete before scored queries")

    for episode_id, episode in episode_by_id.items():
        if episode.get("schema_version") != SCHEMA_VERSION:
            errors.append("episode schema_version mismatch")
            continue
        rows = query_by_episode.get(episode_id, [])
        if [int(row["query_index"]) for row in rows] != list(range(len(rows))):
            errors.append("query indices are not contiguous for %s" % episode_id)
        expected_values = {
            "policy_queries": len(rows),
            "deadline_misses": sum(bool(row["deadline_exceeded"]) for row in rows),
            "horizon_overruns": sum(bool(row["horizon_overrun"]) for row in rows),
            "age_refreshes": sum(
                row["alignment_disposition"] == "hold_refresh" for row in rows
            ),
            "fallback_hold_steps": sum(
                int(row["fallback_hold_steps"]) for row in rows
            ),
            "simulated_catchup_steps": sum(
                int(row["simulated_catchup_steps"] or 0) for row in rows
            ),
        }
        for field, expected in expected_values.items():
            if int(episode[field]) != expected:
                errors.append("%s %s mismatch" % (episode_id, field))
    return errors


def _write_manifest(output_directory: pathlib.Path) -> None:
    files = {}
    for path in sorted(output_directory.rglob("*")):
        if not path.is_file() or path.name == "manifest.json" or path.suffix == ".tmp":
            continue
        relative = path.relative_to(output_directory).as_posix()
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256_bytes(path.read_bytes()),
        }
    _write_json(
        output_directory / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "files": files,
        },
    )


def write_artifacts(
    output_directory: pathlib.Path,
    protocol: Mapping[str, Any],
    environment: Mapping[str, Any],
    warmups: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    planned_rollouts: int,
    complete: bool,
) -> Dict[str, Any]:
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "videos").mkdir(exist_ok=True)
    _write_json(output_directory / "resolved_protocol.json", protocol)
    _write_json(output_directory / "environment.json", environment)
    _write_csv(output_directory / "warmup_queries.csv", warmups, WARMUP_FIELDS)
    _write_csv(output_directory / "per_episode.csv", episodes, EPISODE_FIELDS)
    _write_csv(output_directory / "per_query.csv", queries, QUERY_FIELDS)
    errors = artifact_errors(
        protocol, warmups, episodes, queries, planned_rollouts, complete
    )
    valid = complete and not errors
    progress = {
        "schema_version": SCHEMA_VERSION,
        "planned_rollouts": planned_rollouts,
        "completed_rollouts": len(episodes),
        "warmup_queries_required": int(protocol["warmup"]["queries"]),
        "warmup_queries_completed": len(warmups),
        "complete": valid,
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    _write_json(output_directory / "progress.json", progress)
    summary = _summary(
        episodes,
        queries,
        warmups,
        planned_rollouts,
        valid,
        int(protocol["warmup"]["queries"]),
    )
    _write_json(output_directory / "summary.json", summary)
    (output_directory / "summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    integrity = {
        "schema_version": SCHEMA_VERSION,
        "valid": valid,
        "errors": errors,
        "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    _write_json(output_directory / "integrity.json", integrity)
    _write_manifest(output_directory)
    return integrity


def snapshot_runtime_sources(
    armbench_root: pathlib.Path, output_directory: pathlib.Path
) -> Dict[str, str]:
    snapshot_root = output_directory / "provenance" / "armbench_source"
    hashes: Dict[str, str] = {}
    missing = []
    for relative in RUNTIME_SOURCE_FILES:
        source = armbench_root / pathlib.PurePosixPath(relative)
        if not source.is_file():
            missing.append(relative)
            continue
        destination = snapshot_root / pathlib.PurePosixPath(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = source.read_bytes()
        destination.write_bytes(payload)
        hashes[relative] = _sha256_bytes(payload)
    if missing:
        raise FileNotFoundError(
            "required measured-age runtime sources are missing: %s"
            % ", ".join(missing)
        )
    return hashes


def capture_environment(
    openpi_root: pathlib.Path,
    armbench_root: pathlib.Path,
    server_metadata: Mapping[str, Any],
    args: argparse.Namespace,
    client_session_id: str,
    source_hashes: Mapping[str, str],
) -> Dict[str, Any]:
    environment = capture_v1_environment(
        openpi_root, armbench_root, server_metadata, args
    )
    clock_info = time.get_clock_info("perf_counter")
    environment.update(
        {
            "schema_version": SCHEMA_VERSION,
            "client_session_id": client_session_id,
            "runtime_source_sha256": dict(source_hashes),
            "clock": {
                "name": "perf_counter",
                "implementation": clock_info.implementation,
                "monotonic": clock_info.monotonic,
                "adjustable": clock_info.adjustable,
                "resolution_s": clock_info.resolution,
                "recorded_unit": "nanoseconds",
            },
        }
    )
    return environment


def _stabilized_observation(
    environment: Any, initial_state: np.ndarray, num_steps_wait: int
) -> Mapping[str, Any]:
    environment.reset()
    observation = environment.set_init_state(np.array(initial_state, copy=True))
    if not isinstance(observation, Mapping):
        raise TypeError("environment.set_init_state must return an observation mapping")
    for _ in range(num_steps_wait):
        step_result = environment.step(np.asarray(LIBERO_DUMMY_ACTION).copy())
        if not isinstance(step_result, tuple) or len(step_result) != 4:
            raise TypeError("LIBERO environment.step must return a four-item tuple")
        observation = step_result[0]
        if not isinstance(observation, Mapping):
            raise TypeError("LIBERO environment returned a non-mapping observation")
    return observation


def run_warmups(
    args: argparse.Namespace,
    task_suite: Any,
    client: BoundedOpenPIClient,
    client_session_id: str,
    checkpoint_content_sha256: str,
) -> List[Dict[str, Any]]:
    task = task_suite.get_task(args.warmup_task_id)
    initial_states = task_suite.get_task_init_states(args.warmup_task_id)
    if args.warmup_episode_index >= len(initial_states):
        raise IndexError(
            "warmup episode index %d exceeds %d available initial states"
            % (args.warmup_episode_index, len(initial_states))
        )
    environment = _make_libero_environment(task, args.seed)
    rows: List[Dict[str, Any]] = []
    try:
        observation = _stabilized_observation(
            environment,
            initial_states[args.warmup_episode_index],
            args.num_steps_wait,
        )
        task_description = str(task.language)
        for warmup_index in range(args.warmup_queries):
            observation_captured_ns = time.perf_counter_ns()
            inference_started_ns: Optional[int] = None
            inference_finished_ns: Optional[int] = None
            response_ready_ns: Optional[int] = None
            action_chunk_steps = 0
            action_dimension = 0
            accepted = False
            error_type = None
            error_message = None
            try:
                request = build_libero_request(
                    observation, task_description, args.resize_size
                )
                inference_started_ns = time.perf_counter_ns()
                response = client.infer(request)
                inference_finished_ns = time.perf_counter_ns()
                response_ready_ns = inference_finished_ns
                actions = validate_action_chunk(
                    response, PI05_LIBERO_ACTION_HORIZON
                )
                action_chunk_steps = int(actions.shape[0])
                action_dimension = int(actions.shape[1])
                accepted = True
            except Exception as exc:
                if inference_finished_ns is None and inference_started_ns is not None:
                    inference_finished_ns = time.perf_counter_ns()
                if response_ready_ns is None and inference_finished_ns is not None:
                    response_ready_ns = inference_finished_ns
                error_type = type(exc).__name__
                error_message = str(exc)
            age_ms = (
                None
                if response_ready_ns is None
                else (response_ready_ns - observation_captured_ns) / 1_000_000.0
            )
            inference_ms = (
                None
                if inference_started_ns is None or inference_finished_ns is None
                else (inference_finished_ns - inference_started_ns) / 1_000_000.0
            )
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "warmup_index": warmup_index,
                    "scored": False,
                    "client_session_id": client_session_id,
                    "checkpoint_content_sha256": checkpoint_content_sha256,
                    "observation_captured_monotonic_ns": observation_captured_ns,
                    "policy_call_started_monotonic_ns": inference_started_ns,
                    "policy_call_finished_monotonic_ns": inference_finished_ns,
                    "response_ready_monotonic_ns": response_ready_ns,
                    "observation_age_ms": age_ms,
                    "inference_latency_ms": inference_ms,
                    "action_chunk_steps": action_chunk_steps,
                    "action_dimension": action_dimension,
                    "accepted": accepted,
                    "error_type": error_type,
                    "error_message": error_message,
                }
            )
            if not accepted:
                break
    finally:
        close = getattr(environment, "close", None)
        if callable(close):
            close()
    return rows


def _prepare_output_directory(path: pathlib.Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("output directory must be absent or empty: %s" % path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "videos").mkdir(exist_ok=True)


def make_runtime_config(
    args: argparse.Namespace, cell: ExperimentCell
) -> RuntimeConfig:
    """Build the only scored runtime configuration allowed by the v2 protocol."""

    return RuntimeConfig(
        mode=cell.mode,
        replan_steps=cell.replan_steps,
        latency_steps=0,
        max_task_steps=(
            args.max_task_steps
            if args.max_task_steps is not None
            else SUITE_MAX_STEPS[args.task_suite]
        ),
        num_steps_wait=args.num_steps_wait,
        resize_size=args.resize_size,
        record_video=args.video_mode != "none",
        latency_source=MEASURED_WALL_LATENCY,
        control_period_ms=args.control_period_ms,
        age_rounding=args.age_rounding,
        deadline_ms=args.deadline_ms,
        max_age_refreshes=args.max_age_refreshes,
    )


def _connected_benchmark(
    args: argparse.Namespace,
    cells: Sequence[ExperimentCell],
    output_directory: pathlib.Path,
    openpi_root: pathlib.Path,
    armbench_root: pathlib.Path,
    client: BoundedOpenPIClient,
) -> int:
    from libero.libero import benchmark

    server_metadata = client.get_server_metadata()
    server_source = armbench_root / "integrations" / "openpi" / "serve_policy_attested.py"
    if not server_source.is_file() and not args.allow_unattested_server:
        raise FileNotFoundError("attested server source is missing: %s" % server_source)
    server_source_sha256 = (
        _sha256_bytes(server_source.read_bytes()) if server_source.is_file() else ""
    )
    attestation = _validate_server_attestation(
        server_metadata, args, server_source_sha256
    )
    client_session_id = uuid.uuid4().hex
    source_hashes = snapshot_runtime_sources(armbench_root, output_directory)
    environment_record = capture_environment(
        openpi_root,
        armbench_root,
        server_metadata,
        args,
        client_session_id,
        source_hashes,
    )
    jitter_candidates = parse_jitter_candidates(args.jitter_values_ms)
    protocol = resolved_protocol(args, cells, jitter_candidates)
    benchmark_class = benchmark.get_benchmark_dict()[args.task_suite]
    task_suite = benchmark_class()
    checkpoint_sha = (
        str(attestation.get("checkpoint_content_sha256", ""))
        if isinstance(attestation, Mapping)
        else ""
    )
    warmups = run_warmups(
        args, task_suite, client, client_session_id, checkpoint_sha
    )
    episodes: List[Dict[str, Any]] = []
    queries: List[Dict[str, Any]] = []
    if len(warmups) != args.warmup_queries or not all(
        bool(row["accepted"]) for row in warmups
    ):
        write_artifacts(
            output_directory,
            protocol,
            environment_record,
            warmups,
            episodes,
            queries,
            len(cells),
            complete=False,
        )
        return 2

    cells_by_task: Dict[int, List[ExperimentCell]] = {}
    for cell in cells:
        cells_by_task.setdefault(cell.task_id, []).append(cell)
    aborted = False
    for task_id, task_cells in cells_by_task.items():
        task = task_suite.get_task(task_id)
        task_description = str(task.language)
        initial_states = task_suite.get_task_init_states(task_id)
        environment = _make_libero_environment(task, args.seed)
        try:
            for cell in task_cells:
                if cell.episode_index >= len(initial_states):
                    raise IndexError(
                        "episode index %d exceeds %d available initial states"
                        % (cell.episode_index, len(initial_states))
                    )
                environment.seed(args.seed)
                config = make_runtime_config(args, cell)
                started = time.perf_counter()
                result = run_episode(
                    environment,
                    client,
                    initial_states[cell.episode_index],
                    task_description,
                    config,
                    response_jitter_ms=make_jitter_provider(
                        cell, args.seed, jitter_candidates
                    ),
                )
                wall_time_s = time.perf_counter() - started
                should_write_video = args.video_mode == "all" or (
                    args.video_mode == "failures" and not result.success
                )
                video_relative = None
                video_error_type = None
                video_error_message = None
                if should_write_video:
                    try:
                        outcome = "success" if result.success else "failure"
                        video_relative = "videos/%s__%s.mp4" % (
                            _safe_filename(cell.episode_id),
                            outcome,
                        )
                        _write_video(
                            output_directory / video_relative, result.replay_frames
                        )
                    except Exception as exc:
                        video_relative = None
                        video_error_type = type(exc).__name__
                        video_error_message = str(exc)
                episode, episode_queries = result_rows(
                    cell,
                    result,
                    task_description,
                    args.seed,
                    wall_time_s,
                    jitter_candidates,
                    args.control_period_ms,
                    video_path=video_relative,
                    video_required=should_write_video,
                    video_error_type=video_error_type,
                    video_error_message=video_error_message,
                )
                episodes.append(episode)
                queries.extend(episode_queries)
                write_artifacts(
                    output_directory,
                    protocol,
                    environment_record,
                    warmups,
                    episodes,
                    queries,
                    len(cells),
                    complete=False,
                )
                if result.failure_category in FORMAL_ABORT_CATEGORIES:
                    _append_run_error(
                        output_directory,
                        "formal_runtime_abort",
                        RuntimeError(
                            "%s: %s"
                            % (result.failure_category, result.termination_reason)
                        ),
                    )
                    aborted = True
                    break
        finally:
            close = getattr(environment, "close", None)
            if callable(close):
                close()
        if aborted:
            break

    integrity = write_artifacts(
        output_directory,
        protocol,
        environment_record,
        warmups,
        episodes,
        queries,
        len(cells),
        complete=not aborted and len(episodes) == len(cells),
    )
    return 0 if integrity["valid"] else 2


def _append_run_error(
    output_directory: pathlib.Path, stage: str, error: BaseException
) -> None:
    path = output_directory / "run_error.json"
    existing: List[Mapping[str, Any]] = []
    if path.is_file():
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(parsed, Mapping) and isinstance(parsed.get("errors"), list):
                existing = [
                    item for item in parsed["errors"] if isinstance(item, Mapping)
                ]
        except (OSError, ValueError):
            existing = []
    _write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "errors": existing
            + [
                {
                    "stage": stage,
                    "type": type(error).__name__,
                    "message": str(error),
                }
            ],
            "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )


def execute_benchmark(
    args: argparse.Namespace, cells: Sequence[ExperimentCell]
) -> int:
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
    log_handler: Optional[logging.FileHandler] = None
    client: Optional[BoundedOpenPIClient] = None
    stage = "runtime_setup"
    exit_code = 2
    try:
        log_handler = logging.FileHandler(
            str(output_directory / "run.log"), encoding="utf-8"
        )
        log_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logging.getLogger().addHandler(log_handler)
        np.random.seed(args.seed)
        stage = "policy_client_startup"
        client = BoundedOpenPIClient(
            args.host,
            args.port,
            startup_timeout_s=args.server_startup_timeout_s,
            inference_timeout_s=args.inference_timeout_s,
        )
        stage = "connected_benchmark"
        exit_code = _connected_benchmark(
            args,
            cells,
            output_directory,
            openpi_root,
            armbench_root,
            client,
        )
    except Exception as exc:
        logging.exception("measured-age evaluation failed during %s", stage)
        _append_run_error(output_directory, stage, exc)
        exit_code = 2
    finally:
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                _append_run_error(output_directory, "policy_client_close", exc)
                exit_code = 2
        if log_handler is not None:
            logging.getLogger().removeHandler(log_handler)
            log_handler.close()
        _write_manifest(output_directory)
    return exit_code


def _add_matrix_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--task-suite", choices=tuple(SUITE_TASK_COUNTS), default="libero_spatial"
    )
    parser.add_argument("--task-ids", default="0")
    parser.add_argument("--episode-indices", default="0")
    parser.add_argument(
        "--modes",
        default="%s,%s" % (ASYNC_UNGUARDED, LATENCY_ALIGNED),
        help="Only async_unguarded and latency_aligned are accepted",
    )
    parser.add_argument("--replan-steps", default="5")


def _add_timing_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--control-period-ms", type=float, default=LIBERO_CONTROL_PERIOD_MS
    )
    parser.add_argument("--age-rounding", choices=VALID_ROUNDING, default=CEIL)
    parser.add_argument("--deadline-ms", type=float, default=250.0)
    parser.add_argument("--max-age-refreshes", type=int, default=2)
    parser.add_argument(
        "--jitter-values-ms",
        default=",".join("%g" % value for value in DEFAULT_JITTER_CANDIDATES_MS),
    )
    parser.add_argument(
        "--warmup-queries", type=int, default=DEFAULT_WARMUP_QUERIES
    )
    parser.add_argument("--warmup-task-id", type=int, default=0)
    parser.add_argument("--warmup-episode-index", type=int, default=49)
    parser.add_argument("--seed", type=int, default=7)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan", allow_abbrev=False)
    _add_matrix_arguments(plan_parser)
    _add_timing_arguments(plan_parser)

    run_parser = subparsers.add_parser("run", allow_abbrev=False)
    _add_matrix_arguments(run_parser)
    _add_timing_arguments(run_parser)
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
    )
    run_parser.add_argument("--allow-commit-mismatch", action="store_true")
    run_parser.add_argument("--allow-unattested-server", action="store_true")
    run_parser.add_argument("--resize-size", type=int, default=224)
    run_parser.add_argument("--num-steps-wait", type=int, default=10)
    run_parser.add_argument("--max-task-steps", type=int)
    run_parser.add_argument(
        "--video-mode", choices=("none", "failures", "all"), default="all"
    )
    return parser


def _validate_common_arguments(
    args: argparse.Namespace, cells: Sequence[ExperimentCell]
) -> Tuple[float, ...]:
    if not math.isfinite(args.control_period_ms) or args.control_period_ms <= 0.0:
        raise ValueError("control_period_ms must be finite and positive")
    if not math.isfinite(args.deadline_ms) or args.deadline_ms < 0.0:
        raise ValueError("deadline_ms must be finite and nonnegative")
    if args.max_age_refreshes < 0:
        raise ValueError("max_age_refreshes must be nonnegative")
    if args.warmup_queries < 1:
        raise ValueError("warmup_queries must be at least 1")
    if args.warmup_task_id < 0 or args.warmup_task_id >= SUITE_TASK_COUNTS[
        args.task_suite
    ]:
        raise ValueError("warmup_task_id is outside the selected suite")
    if args.warmup_episode_index < 0 or args.warmup_episode_index >= 50:
        raise ValueError("warmup_episode_index must be in [0, 50)")
    jitter = parse_jitter_candidates(args.jitter_values_ms)
    for cell in cells:
        if cell.mode not in VALID_MODES or cell.latency_steps != 0:
            raise ValueError("v2 cells must use allowed modes and latency_steps=0")
        RuntimeConfig(
            mode=cell.mode,
            replan_steps=cell.replan_steps,
            latency_steps=0,
            max_task_steps=SUITE_MAX_STEPS[args.task_suite],
            latency_source=MEASURED_WALL_LATENCY,
            control_period_ms=args.control_period_ms,
            age_rounding=args.age_rounding,
            deadline_ms=args.deadline_ms,
            max_age_refreshes=args.max_age_refreshes,
        )
    return jitter


def _validate_run_arguments(args: argparse.Namespace) -> None:
    if args.resize_size != 224:
        raise ValueError("official pi05_libero evaluation requires --resize-size 224")
    if args.num_steps_wait != 10:
        raise ValueError("official LIBERO evaluation requires --num-steps-wait 10")
    if args.port <= 0 or args.port > 65535:
        raise ValueError("port must be in [1, 65535]")
    if args.max_task_steps is not None and args.max_task_steps <= 0:
        raise ValueError("max_task_steps must be positive")
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_openpi_commit):
        raise ValueError("expected_openpi_commit must be a lowercase Git SHA")
    _validate_server_launch_args(args.server_launch_args, args.checkpoint)


def plan_payload(
    args: argparse.Namespace,
    cells: Sequence[ExperimentCell],
    jitter_candidates: Sequence[float],
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "matrix": matrix_plan(cells),
        "temporal_alignment": {
            "latency_source": MEASURED_WALL_LATENCY,
            "latency_steps": 0,
            "control_period_ms": args.control_period_ms,
            "completed_step_rounding": FLOOR,
            "action_offset_rounding": args.age_rounding,
            "boundary_tolerance_ms": 1e-9,
            "deadline_ms": args.deadline_ms,
            "max_age_refreshes": args.max_age_refreshes,
            "controller_model": "post_response_catchup_simulation",
        },
        "jitter": {
            "generator": "sha256_first_u64_mod_v1",
            "seed": args.seed,
            "candidates_ms": list(jitter_candidates),
            "pairing_key_fields": [
                "task_suite",
                "task_id",
                "episode_index",
                "replan_steps",
            ],
            "query_index_field": "query_index",
            "mode_in_key": False,
            "payload_fields": ["seed", "pairing_key", "query_index"],
        },
        "warmup": {
            "queries": args.warmup_queries,
            "task_suite": args.task_suite,
            "task_id": args.warmup_task_id,
            "episode_index": args.warmup_episode_index,
            "scored": False,
            "required_before_scoring": True,
        },
        "registered_cells": [
            {
                "condition_order": cell.condition_order,
                "episode_id": cell.episode_id,
                "pair_id": cell.pair_id,
                "task_suite": cell.task_suite,
                "task_id": cell.task_id,
                "episode_index": cell.episode_index,
                "mode": cell.mode,
                "replan_steps": cell.replan_steps,
                "latency_source": MEASURED_WALL_LATENCY,
                "latency_steps": 0,
            }
            for cell in cells
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        cells = resolve_matrix(args)
        jitter = _validate_common_arguments(args, cells)
        if args.command == "plan":
            print(json.dumps(plan_payload(args, cells, jitter), indent=2, sort_keys=True))
            return 0
        _validate_run_arguments(args)
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
        )
        return execute_benchmark(args, cells)
    except Exception as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
