"""Strict offline analysis for the frozen three-arm RTC overlap pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from integrations.openpi import rtc_overlap_pilot


ANALYSIS_SCHEMA_VERSION = "armbench.pi05_rtc_overlap_analysis.v1"
SOURCE_SCHEMA_VERSION = rtc_overlap_pilot.SCHEMA_VERSION
METHODS = tuple(rtc_overlap_pilot.V2_OVERLAP_METHODS)
BASELINE = rtc_overlap_pilot.OVERLAP_UNCONDITIONED
CANDIDATES = (
    rtc_overlap_pilot.PROJECTED_OVERLAP,
    rtc_overlap_pilot.RTC_GUIDED_OVERLAP,
)
TASK_SUITE = "libero_10"
TASK_IDS = tuple(range(10))
EPISODE_INDICES = (0, 1)
EXECUTE_HORIZON = 5
INFERENCE_DELAY_STEPS = 4
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260805
ANALYZER_SOURCE = "integrations/openpi/rtc_overlap_analysis.py"
VALIDATOR_SOURCE = "integrations/openpi/rtc_overlap_pilot.py"
SEAM_METRICS = ("seam_motion_l2", "seam_gripper_abs")

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
_PER_EPISODE_FIELDS = (
    "pair_id",
    "task_suite",
    "task_id",
    "episode_index",
    "initial_state_sha256",
) + tuple(
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
) + tuple(
    field
    for candidate in CANDIDATES
    for field in (
        "%s_success_difference" % candidate,
        "%s_seam_motion_l2_difference" % candidate,
        "%s_seam_gripper_abs_difference" % candidate,
    )
)


class AnalysisError(ValueError):
    """Raised when source evidence cannot support the frozen analysis."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_identity() -> Dict[str, str]:
    source_root = pathlib.Path(__file__).resolve().parents[2]
    analyzer = source_root / pathlib.PurePosixPath(ANALYZER_SOURCE)
    validator = source_root / pathlib.PurePosixPath(VALIDATOR_SOURCE)
    return {
        "analyzer_source": ANALYZER_SOURCE,
        "analyzer_sha256": _sha256_file(analyzer),
        "validator_source": VALIDATOR_SOURCE,
        "validator_sha256": _sha256_file(validator),
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy_version": np.__version__,
    }


def _portable_source_label(root: pathlib.Path) -> str:
    repository_root = pathlib.Path(__file__).resolve().parents[2]
    try:
        return root.relative_to(repository_root).as_posix()
    except ValueError:
        return root.name


def _check_finite_tree(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise AnalysisError("%s contains a non-finite number" % label)
    if isinstance(value, Mapping):
        for key, item in value.items():
            _check_finite_tree(item, "%s.%s" % (label, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite_tree(item, "%s[%d]" % (label, index))


def _strict_json_bytes(value: bytes, label: str, expected: type) -> Any:
    def reject(token: str) -> Any:
        raise ValueError("non-finite JSON constant: %s" % token)

    def unique(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        for key, item in pairs:
            if key in output:
                raise ValueError("duplicate JSON key: %s" % key)
            output[key] = item
        return output

    try:
        parsed = json.loads(
            value.decode("utf-8"),
            parse_constant=reject,
            object_pairs_hook=unique,
        )
    except (UnicodeError, ValueError) as exc:
        raise AnalysisError("%s is not strict UTF-8 JSON: %s" % (label, exc)) from exc
    if not isinstance(parsed, expected):
        raise AnalysisError("%s must contain a %s" % (label, expected.__name__))
    _check_finite_tree(parsed, label)
    return parsed


def _require_fields(row: Mapping[str, Any], fields: Sequence[str], label: str) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise AnalysisError("%s is missing fields: %s" % (label, ", ".join(missing)))


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AnalysisError("%s must be an integer >= %d" % (label, minimum))
    return value


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError("%s must be numeric" % label)
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise AnalysisError("%s must be finite and nonnegative" % label)
    return result


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


def _stable_source(root: pathlib.Path) -> Tuple[Mapping[str, bytes], Mapping[str, Any]]:
    # The independent validator is deliberately the first artifact operation.
    try:
        first = rtc_overlap_pilot.validate_artifact(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise AnalysisError("source artifact failed independent validation: %s" % exc) from exc
    relative_paths = (
        "manifest.json",
        "resolved_protocol.json",
        "environment.json",
        "episodes.json",
        "queries.json",
        "progress.json",
        "summary.json",
        "transition_descriptor.json",
    )
    snapshots: Dict[str, bytes] = {}
    try:
        for relative in relative_paths:
            snapshots[relative] = (root / relative).read_bytes()
    except OSError as exc:
        raise AnalysisError("cannot snapshot source artifact: %s" % exc) from exc
    try:
        second = rtc_overlap_pilot.validate_artifact(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise AnalysisError("source artifact changed or became invalid: %s" % exc) from exc
    if first != second:
        raise AnalysisError("source validator result changed while reading")
    for relative, before in snapshots.items():
        try:
            after = (root / relative).read_bytes()
        except OSError as exc:
            raise AnalysisError("source artifact changed while reading: %s" % relative) from exc
        if after != before:
            raise AnalysisError("source artifact changed while reading: %s" % relative)
    if not isinstance(first, Mapping):
        raise AnalysisError("source validator returned a non-object result")
    return snapshots, first


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise AnalysisError("analysis requires the RTC overlap v2 source schema")
    if protocol.get("planned_rollouts") != 60:
        raise AnalysisError("analysis requires exactly 60 planned rollouts")
    matrix = protocol.get("matrix")
    if not isinstance(matrix, list) or len(matrix) != 60:
        raise AnalysisError("protocol matrix must contain exactly 60 cells")
    seen = set()
    expected = _expected_cells()
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
        if key not in expected:
            raise AnalysisError("protocol matrix contains unexpected cell %r" % (key,))
        if raw["task_suite"] != TASK_SUITE:
            raise AnalysisError("%s task suite mismatch" % label)
        if raw["condition_order"] != expected[key]:
            raise AnalysisError("%s condition order violates the frozen Latin rotation" % label)
        if raw["execute_horizon"] != EXECUTE_HORIZON:
            raise AnalysisError("%s execute horizon mismatch" % label)
        if raw["inference_delay_steps"] != INFERENCE_DELAY_STEPS:
            raise AnalysisError("%s inference delay mismatch" % label)
    if seen != set(expected):
        raise AnalysisError("protocol matrix is incomplete")


def _validate_episodes(
    episodes: Sequence[Any],
) -> Dict[Tuple[int, int, str], Mapping[str, Any]]:
    if len(episodes) != 60:
        raise AnalysisError("episodes.json must contain exactly 60 rollouts")
    expected = _expected_cells()
    output: Dict[Tuple[int, int, str], Mapping[str, Any]] = {}
    episode_ids = set()
    for index, raw in enumerate(episodes):
        label = "episodes[%d]" % index
        if not isinstance(raw, Mapping):
            raise AnalysisError("%s must be an object" % label)
        _require_fields(
            raw,
            _IDENTITY_FIELDS + ("success", "policy_queries", "initial_state_sha256"),
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
        pair_id = "%s__task_%02d__episode_%02d" % (
            TASK_SUITE,
            task_id,
            episode_index,
        )
        episode_id = "%s__%s" % (pair_id, method)
        if raw["schema_version"] != SOURCE_SCHEMA_VERSION:
            raise AnalysisError("%s schema mismatch" % label)
        if raw["task_suite"] != TASK_SUITE or raw["pair_id"] != pair_id:
            raise AnalysisError("%s pair identity mismatch" % label)
        if raw["episode_id"] != episode_id or raw["episode_id"] in episode_ids:
            raise AnalysisError("%s episode identity is duplicate or malformed" % label)
        if raw["condition_order"] != expected[key]:
            raise AnalysisError("%s condition order violates the frozen Latin rotation" % label)
        if raw["execute_horizon"] != EXECUTE_HORIZON:
            raise AnalysisError("%s execute horizon mismatch" % label)
        if raw["inference_delay_steps"] != INFERENCE_DELAY_STEPS:
            raise AnalysisError("%s inference delay mismatch" % label)
        if not isinstance(raw["success"], bool):
            raise AnalysisError("%s.success must be boolean" % label)
        _integer(raw["policy_queries"], "%s.policy_queries" % label, minimum=1)
        state_hash = raw["initial_state_sha256"]
        if (
            not isinstance(state_hash, str)
            or len(state_hash) != 64
            or any(character not in "0123456789abcdef" for character in state_hash)
        ):
            raise AnalysisError("%s initial state hash is not canonical SHA-256" % label)
        output[key] = raw
        episode_ids.add(raw["episode_id"])
    if set(output) != set(expected):
        raise AnalysisError("episodes.json is missing one or more frozen cells")
    for task_id in TASK_IDS:
        for episode_index in EPISODE_INDICES:
            states = {
                output[(task_id, episode_index, method)]["initial_state_sha256"]
                for method in METHODS
            }
            if len(states) != 1:
                raise AnalysisError(
                    "task %d episode %d does not share one initial state"
                    % (task_id, episode_index)
                )
    return output


def _validate_queries(
    queries: Sequence[Any],
    episodes: Mapping[Tuple[int, int, str], Mapping[str, Any]],
) -> Dict[str, Dict[str, float]]:
    by_episode: Dict[str, List[Mapping[str, Any]]] = {
        str(row["episode_id"]): [] for row in episodes.values()
    }
    episode_by_id = {
        str(row["episode_id"]): row for row in episodes.values()
    }
    seen = set()
    for index, raw in enumerate(queries):
        label = "queries[%d]" % index
        if not isinstance(raw, Mapping):
            raise AnalysisError("%s must be an object" % label)
        _require_fields(
            raw,
            _IDENTITY_FIELDS + ("query_index", "bootstrap") + SEAM_METRICS,
            label,
        )
        episode_id = raw["episode_id"]
        if episode_id not in by_episode:
            raise AnalysisError("%s references an unknown episode" % label)
        query_index = _integer(raw["query_index"], "%s.query_index" % label)
        query_key = (episode_id, query_index)
        if query_key in seen:
            raise AnalysisError("queries.json contains duplicate query %r" % (query_key,))
        seen.add(query_key)
        episode = episode_by_id[str(episode_id)]
        for field in _IDENTITY_FIELDS:
            if raw[field] != episode[field]:
                raise AnalysisError("%s mismatches its episode on %s" % (label, field))
        if not isinstance(raw["bootstrap"], bool):
            raise AnalysisError("%s.bootstrap must be boolean" % label)
        if raw["bootstrap"]:
            if query_index != 0 or any(raw[metric] is not None for metric in SEAM_METRICS):
                raise AnalysisError("%s has an invalid bootstrap seam record" % label)
        else:
            if query_index == 0:
                raise AnalysisError("%s scored query cannot use index zero" % label)
            for metric in SEAM_METRICS:
                _finite_nonnegative(raw[metric], "%s.%s" % (label, metric))
        by_episode[episode_id].append(raw)

    aggregate: Dict[str, Dict[str, float]] = {}
    for episode in episodes.values():
        episode_id = str(episode["episode_id"])
        selected = sorted(by_episode[episode_id], key=lambda row: row["query_index"])
        expected_indices = list(range(int(episode["policy_queries"])))
        if [row["query_index"] for row in selected] != expected_indices:
            raise AnalysisError("episode %s has missing or noncontiguous queries" % episode_id)
        if len(selected) < 2 or selected[0]["bootstrap"] is not True:
            raise AnalysisError("episode %s lacks scored seam queries" % episode_id)
        if any(row["bootstrap"] for row in selected[1:]):
            raise AnalysisError("episode %s contains multiple bootstrap queries" % episode_id)
        aggregate[episode_id] = {
            "scored_queries": len(selected) - 1,
            **{
                metric: float(
                    np.mean([float(row[metric]) for row in selected[1:]])
                )
                for metric in SEAM_METRICS
            },
        }
    return aggregate


def _paired_rows(
    episodes: Mapping[Tuple[int, int, str], Mapping[str, Any]],
    seam_by_episode: Mapping[str, Mapping[str, float]],
) -> List[Dict[str, Any]]:
    output = []
    for task_id in TASK_IDS:
        for episode_index in EPISODE_INDICES:
            selected = {
                method: episodes[(task_id, episode_index, method)] for method in METHODS
            }
            baseline = selected[BASELINE]
            row: Dict[str, Any] = {
                "pair_id": baseline["pair_id"],
                "task_suite": TASK_SUITE,
                "task_id": task_id,
                "episode_index": episode_index,
                "initial_state_sha256": baseline["initial_state_sha256"],
            }
            for method, episode in selected.items():
                row["%s_condition_order" % method] = int(episode["condition_order"])
                row["%s_success" % method] = int(bool(episode["success"]))
                row["%s_policy_queries" % method] = int(episode["policy_queries"])
                row["%s_scored_transition_queries" % method] = int(
                    seam_by_episode[str(episode["episode_id"])]["scored_queries"]
                )
                for metric in SEAM_METRICS:
                    row["%s_%s" % (method, metric)] = seam_by_episode[
                        str(episode["episode_id"])
                    ][metric]
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
    return output


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _describe(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise AnalysisError("descriptive statistics require finite observations")
    q1 = _percentile(values, 25.0)
    q3 = _percentile(values, 75.0)
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if array.size > 1 else None,
        "median": float(np.median(array)),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }


def _wilson_interval(successes: int, total: int) -> Tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / float(total)
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def _mcnemar_exact_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1))
    return min(1.0, 2.0 * probability / float(2**discordant))


def _holm_adjust(raw: Mapping[str, float]) -> Dict[str, float]:
    ordered = sorted(raw.items(), key=lambda item: (item[1], CANDIDATES.index(item[0])))
    adjusted: Dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted


def _task_block_interval(
    rows: Sequence[Mapping[str, Any]], field: str
) -> Dict[str, float]:
    blocks = np.asarray(
        [
            [float(row[field]) for row in rows if int(row["task_id"]) == task_id]
            for task_id in TASK_IDS
        ],
        dtype=np.float64,
    )
    if blocks.shape != (10, 2) or not np.all(np.isfinite(blocks)):
        raise AnalysisError("task-block bootstrap requires ten complete two-episode blocks")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(TASK_IDS), size=(BOOTSTRAP_RESAMPLES, len(TASK_IDS)))
    sampled = blocks[indices].reshape(BOOTSTRAP_RESAMPLES, -1)
    means = np.mean(sampled, axis=1)
    medians = np.median(sampled, axis=1)
    return {
        "mean_low": float(np.percentile(means, 2.5)),
        "mean_high": float(np.percentile(means, 97.5)),
        "median_low": float(np.percentile(medians, 2.5)),
        "median_high": float(np.percentile(medians, 97.5)),
    }


def _success_analysis(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    methods: Dict[str, Any] = {}
    for method in METHODS:
        values = [int(row["%s_success" % method]) for row in rows]
        successes = sum(values)
        interval = _wilson_interval(successes, len(values))
        methods[method] = {
            "successes": successes,
            "rollouts": len(values),
            "rate": successes / float(len(values)),
            "wilson95_low": interval[0],
            "wilson95_high": interval[1],
        }
    raw_p = {}
    contrasts: Dict[str, Any] = {}
    for candidate in CANDIDATES:
        differences = [int(row["%s_success_difference" % candidate]) for row in rows]
        wins = sum(value > 0 for value in differences)
        losses = sum(value < 0 for value in differences)
        raw_p[candidate] = _mcnemar_exact_p(wins, losses)
        interval = _task_block_interval(rows, "%s_success_difference" % candidate)
        contrasts[candidate] = {
            "pairs": len(rows),
            "rate_difference": float(np.mean(differences)),
            "candidate_wins": wins,
            "candidate_losses": losses,
            "ties": len(rows) - wins - losses,
            "mcnemar_exact_two_sided_p": raw_p[candidate],
            "task_block_bootstrap95_mean_low": interval["mean_low"],
            "task_block_bootstrap95_mean_high": interval["mean_high"],
        }
    adjusted = _holm_adjust(raw_p)
    for candidate in CANDIDATES:
        contrasts[candidate]["holm_adjusted_p"] = adjusted[candidate]
    return {"methods": methods, "contrasts_vs_unconditioned": contrasts}


def _seam_analysis(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for metric in SEAM_METRICS:
        methods = {}
        for method in METHODS:
            methods[method] = {
                "episodes": len(rows),
                "scored_transitions": sum(
                    int(row["%s_scored_transition_queries" % method]) for row in rows
                ),
                **_describe(
                    [float(row["%s_%s" % (method, metric)]) for row in rows]
                ),
            }
        contrasts = {}
        for candidate in CANDIDATES:
            field = "%s_%s_difference" % (candidate, metric)
            differences = [float(row[field]) for row in rows]
            interval = _task_block_interval(rows, field)
            contrasts[candidate] = {
                "pairs": len(differences),
                "paired_mean_difference": float(np.mean(differences)),
                "paired_median_difference": float(np.median(differences)),
                "task_block_bootstrap95_mean_low": interval["mean_low"],
                "task_block_bootstrap95_mean_high": interval["mean_high"],
                "task_block_bootstrap95_median_low": interval["median_low"],
                "task_block_bootstrap95_median_high": interval["median_high"],
            }
        output[metric] = {
            "unit_of_analysis": "episode-level mean across scored queries",
            "methods": methods,
            "contrasts_vs_unconditioned": contrasts,
        }
    return output


def _subset_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
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
        candidate_result: Dict[str, Any] = {
            "success_rate_difference": float(
                np.mean([row["%s_success_difference" % candidate] for row in rows])
            )
        }
        for metric in SEAM_METRICS:
            differences = [
                float(row["%s_%s_difference" % (candidate, metric)]) for row in rows
            ]
            candidate_result[metric] = {
                "paired_mean_difference": float(np.mean(differences)),
                "paired_median_difference": float(np.median(differences)),
            }
        contrasts[candidate] = candidate_result
    return {"methods": methods, "contrasts_vs_unconditioned": contrasts}


def _per_task(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "task_id": task_id,
            "episodes": 2,
            **_subset_summary([row for row in rows if row["task_id"] == task_id]),
        }
        for task_id in TASK_IDS
    ]


def _leave_one_task_out(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "omitted_task_id": task_id,
            "tasks": 9,
            "episodes": 18,
            **_subset_summary([row for row in rows if row["task_id"] != task_id]),
        }
        for task_id in TASK_IDS
    ]


def _condition_order(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    strata = []
    for order in range(3):
        methods = {}
        for method in METHODS:
            selected = [
                row for row in rows if row["%s_condition_order" % method] == order
            ]
            successes = [int(row["%s_success" % method]) for row in selected]
            methods[method] = {
                "rollouts": len(selected),
                "scored_transitions": sum(
                    int(row["%s_scored_transition_queries" % method])
                    for row in selected
                ),
                "successes": sum(successes),
                "success_rate": float(np.mean(successes)),
                **{
                    metric: _describe(
                        [float(row["%s_%s" % (method, metric)]) for row in selected]
                    )
                    for metric in SEAM_METRICS
                },
            }
        strata.append({"condition_order": order, "methods": methods})
    return {
        "design": "three-way Latin rotation assigned at the episode triplet level",
        "strata": strata,
    }


def analyze_artifact(
    artifact: pathlib.Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    root = pathlib.Path(artifact).resolve()
    snapshots, validator_summary = _stable_source(root)
    protocol = _strict_json_bytes(
        snapshots["resolved_protocol.json"], "resolved_protocol.json", dict
    )
    _strict_json_bytes(snapshots["manifest.json"], "manifest.json", dict)
    progress = _strict_json_bytes(snapshots["progress.json"], "progress.json", dict)
    episodes_json = _strict_json_bytes(snapshots["episodes.json"], "episodes.json", list)
    queries_json = _strict_json_bytes(snapshots["queries.json"], "queries.json", list)
    _strict_json_bytes(snapshots["environment.json"], "environment.json", dict)
    _strict_json_bytes(snapshots["summary.json"], "summary.json", dict)
    _strict_json_bytes(
        snapshots["transition_descriptor.json"], "transition_descriptor.json", dict
    )
    if progress != {"planned": 60, "completed": 60, "complete": True}:
        raise AnalysisError("analysis requires a complete 60-rollout artifact")
    _validate_protocol(protocol)
    episodes = _validate_episodes(episodes_json)
    seam_by_episode = _validate_queries(queries_json, episodes)
    rows = _paired_rows(episodes, seam_by_episode)
    analysis = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "implementation": _implementation_identity(),
        "source": {
            "artifact": _portable_source_label(root),
            "source_schema_version": SOURCE_SCHEMA_VERSION,
            "manifest_sha256": _sha256_bytes(snapshots["manifest.json"]),
            "episodes_sha256": _sha256_bytes(snapshots["episodes.json"]),
            "queries_sha256": _sha256_bytes(snapshots["queries.json"]),
            "validator_summary_sha256": _sha256_bytes(
                json.dumps(
                    validator_summary,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ),
        },
        "cohort": {
            "task_suite": TASK_SUITE,
            "tasks": len(TASK_IDS),
            "episodes_per_task": len(EPISODE_INDICES),
            "paired_triplets": len(rows),
            "rollouts": len(episodes_json),
            "queries": len(queries_json),
            "methods": list(METHODS),
            "execute_horizon": EXECUTE_HORIZON,
            "inference_delay_steps": INFERENCE_DELAY_STEPS,
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
            "multiplicity_adjustment": "Holm across two prespecified contrasts versus overlap_unconditioned",
            "bootstrap_unit": "whole LIBERO task retaining both episode indices",
            "bootstrap_interval": "descriptive percentile interval",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "seam_unit": "episode mean; paired only after per-query aggregation",
        },
        "claim_boundary": (
            "Simulation-only 10-task pilot with two initial states per task. "
            "Task-block bootstrap intervals are descriptive; the matrix is not a "
            "submission-scale estimate of general VLA task performance."
        ),
    }
    return analysis, rows


def _summary_markdown(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# RTC overlap pilot analysis",
        "",
        "- Source manifest SHA-256: `%s`" % analysis["source"]["manifest_sha256"],
        "- Matrix: 10 tasks x 2 initial states x 3 methods (60 rollouts)",
        "- Task-block bootstrap seed/resamples: `%d/%d`"
        % (BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES),
        "",
        "| Method | Success (Wilson 95%) | Motion seam mean / median | Gripper seam mean / median |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        success = analysis["success"]["methods"][method]
        motion = analysis["seam"]["seam_motion_l2"]["methods"][method]
        gripper = analysis["seam"]["seam_gripper_abs"]["methods"][method]
        lines.append(
            "| %s | %d/%d [%.3f, %.3f] | %.6f / %.6f | %.6f / %.6f |"
            % (
                method,
                success["successes"],
                success["rollouts"],
                success["wilson95_low"],
                success["wilson95_high"],
                motion["mean"],
                motion["median"],
                gripper["mean"],
                gripper["median"],
            )
        )
    lines.extend(
        [
            "",
            "| Contrast vs unconditioned | Success diff | McNemar p | Holm p | Motion seam mean diff (task-block 95%) | Gripper seam mean diff (task-block 95%) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for candidate in CANDIDATES:
        success = analysis["success"]["contrasts_vs_unconditioned"][candidate]
        motion = analysis["seam"]["seam_motion_l2"]["contrasts_vs_unconditioned"][candidate]
        gripper = analysis["seam"]["seam_gripper_abs"]["contrasts_vs_unconditioned"][candidate]
        lines.append(
            "| %s | %+.3f | %.6g | %.6g | %+.6f [%+.6f, %+.6f] | %+.6f [%+.6f, %+.6f] |"
            % (
                candidate,
                success["rate_difference"],
                success["mcnemar_exact_two_sided_p"],
                success["holm_adjusted_p"],
                motion["paired_mean_difference"],
                motion["task_block_bootstrap95_mean_low"],
                motion["task_block_bootstrap95_mean_high"],
                gripper["paired_mean_difference"],
                gripper["task_block_bootstrap95_mean_low"],
                gripper["task_block_bootstrap95_mean_high"],
            )
        )
    lines.extend(["", str(analysis["claim_boundary"]), ""])
    return "\n".join(lines)


def _write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_manifest(root: pathlib.Path) -> None:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            relative = path.relative_to(root).as_posix()
            files[relative] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    _write_json(
        root / "manifest.json",
        {"schema_version": ANALYSIS_SCHEMA_VERSION, "files": files},
    )


def generate_report(
    artifact: pathlib.Path,
    output_directory: pathlib.Path,
) -> Dict[str, Any]:
    output = pathlib.Path(output_directory).resolve()
    if output.exists():
        raise AnalysisError("output directory already exists: %s" % output)
    analysis, rows = analyze_artifact(artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(
        tempfile.mkdtemp(prefix=".%s." % output.name, dir=str(output.parent))
    )
    try:
        _write_json(staging / "analysis.json", analysis)
        with (staging / "per_episode.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=_PER_EPISODE_FIELDS, extrasaction="raise"
            )
            writer.writeheader()
            writer.writerows(rows)
        (staging / "summary.md").write_text(
            _summary_markdown(analysis), encoding="utf-8"
        )
        _write_manifest(staging)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("artifact", type=pathlib.Path)
    parser.add_argument("--output-directory", type=pathlib.Path, required=True)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        analysis = generate_report(args.artifact, args.output_directory)
    except (AnalysisError, OSError, ValueError) as exc:
        if args.json:
            print(json.dumps({"valid": False, "error": str(exc)}, indent=2, sort_keys=True))
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
