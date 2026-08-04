"""Strict offline paired analysis for a validated measured-age v2 artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import pathlib
import shutil
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from integrations.openpi.validate_measured_age_artifact import (
    EPISODE_FIELDS,
    QUERY_FIELDS,
    SCHEMA_VERSION as SOURCE_SCHEMA_VERSION,
    validate_artifact,
)


ANALYSIS_SCHEMA_VERSION = "armbench.pi05_libero_measured_age_analysis.v1"
ASYNC_UNGUARDED = "async_unguarded"
LATENCY_ALIGNED = "latency_aligned"
MODES = (ASYNC_UNGUARDED, LATENCY_ALIGNED)
DEFAULT_BOOTSTRAP_SEED = 20260805
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
ANALYZER_SOURCE = "integrations/openpi/measured_age_analysis.py"
VALIDATOR_SOURCE = "integrations/openpi/validate_measured_age_artifact.py"
_PAIR_FIELDS = (
    "pair_id", "task_suite", "task_id", "episode_index", "replan_steps",
    "seed", "initial_state_sha256",
)
_BURDEN_FIELDS = (
    "policy_queries", "accepted_chunks", "rejected_chunks", "interventions",
    "deadline_misses", "horizon_overruns", "age_refreshes",
    "fallback_hold_steps", "simulated_catchup_steps",
)
_PER_PAIR_FIELDS = (
    "pair_id", "task_suite", "task_id", "episode_index", "replan_steps",
    "seed", "initial_state_sha256", "async_success", "aligned_success",
    "success_difference",
) + tuple(
    item
    for field in _BURDEN_FIELDS
    for item in ("async_%s" % field, "aligned_%s" % field, "%s_difference" % field)
)


class AnalysisError(ValueError):
    """Raised when source evidence is incomplete, invalid, or internally mutable."""


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


def _strict_json_bytes(value: bytes, label: str) -> Mapping[str, Any]:
    def reject(value: str) -> Any:
        raise ValueError("non-finite JSON constant: %s" % value)

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
    if not isinstance(parsed, Mapping):
        raise AnalysisError("%s must contain an object" % label)
    return parsed


def _csv_rows(value: bytes, fields: Sequence[str], label: str) -> List[Dict[str, str]]:
    try:
        text = value.decode("utf-8")
    except UnicodeError as exc:
        raise AnalysisError("%s is not UTF-8" % label) from exc
    try:
        with io.StringIO(text, newline="") as handle:
            rows = list(csv.reader(handle, strict=True))
    except csv.Error as exc:
        raise AnalysisError("%s is malformed CSV: %s" % (label, exc)) from exc
    if not rows or rows[0] != list(fields):
        raise AnalysisError("%s exact header mismatch" % label)
    output = []
    for number, row in enumerate(rows[1:], start=2):
        if len(row) != len(fields):
            raise AnalysisError("%s row %d field-count mismatch" % (label, number))
        output.append(dict(zip(fields, row)))
    return output


def _boolean(value: str, label: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise AnalysisError("%s must be exactly True or False" % label)


def _integer(value: str, label: str) -> int:
    if not value.isascii() or not value.isdigit() or (len(value) > 1 and value[0] == "0"):
        raise AnalysisError("%s must be a canonical nonnegative integer" % label)
    return int(value)


def _finite(value: str, label: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise AnalysisError("%s must be numeric" % label) from exc
    if not math.isfinite(result) or result < 0.0:
        raise AnalysisError("%s must be finite and nonnegative" % label)
    return result


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _wilson_interval(successes: int, total: int) -> Tuple[Optional[float], Optional[float]]:
    if total <= 0:
        return None, None
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


def _mcnemar_exact_p(candidate_wins: int, reference_wins: int) -> float:
    discordant = candidate_wins + reference_wins
    if discordant == 0:
        return 1.0
    tail = min(candidate_wins, reference_wins)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1))
    return min(1.0, 2.0 * probability / float(2 ** discordant))


def _paired_bootstrap_mean_interval(
    differences: Sequence[float], resamples: int, seed: int
) -> Tuple[Optional[float], Optional[float]]:
    if not differences:
        return None, None
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = np.mean(values[indices], axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _mode_timing(rows: Sequence[Mapping[str, str]], mode: str) -> Dict[str, Any]:
    subset = [row for row in rows if row["mode"] == mode]

    def values(field: str) -> List[float]:
        return [_finite(row[field], "%s.%s" % (mode, field)) for row in subset]

    ages = values("observation_age_ms")
    inference = values("inference_latency_ms")
    delivery = values("response_delivery_elapsed_ms")
    jitter = values("response_jitter_requested_ms")
    queries = len(subset)
    deadline = sum(_boolean(row["deadline_exceeded"], "deadline_exceeded") for row in subset)
    horizon = sum(_boolean(row["horizon_overrun"], "horizon_overrun") for row in subset)
    refresh = sum(row["alignment_disposition"] == "hold_refresh" for row in subset)
    fail_closed = sum(row["alignment_disposition"] == "fail_closed" for row in subset)
    terminal = sum(row["alignment_disposition"] == "terminal_before_dispatch" for row in subset)
    return {
        "mode": mode,
        "queries": queries,
        "observation_age_ms": {
            "p50": _percentile(ages, 50), "p95": _percentile(ages, 95),
            "max": max(ages) if ages else None,
        },
        "inference_latency_ms": {
            "p50": _percentile(inference, 50), "p95": _percentile(inference, 95),
            "max": max(inference) if inference else None,
        },
        "response_delivery_ms": {
            "p50": _percentile(delivery, 50), "p95": _percentile(delivery, 95),
            "max": max(delivery) if delivery else None,
        },
        "requested_jitter_ms": {
            "p50": _percentile(jitter, 50), "p95": _percentile(jitter, 95),
            "max": max(jitter) if jitter else None,
        },
        "deadline_misses": deadline,
        "deadline_miss_rate_per_query": deadline / float(queries) if queries else None,
        "horizon_overruns": horizon,
        "horizon_overrun_rate_per_query": horizon / float(queries) if queries else None,
        "hold_refresh_queries": refresh,
        "fail_closed_queries": fail_closed,
        "terminal_before_dispatch_queries": terminal,
    }


def _episode(row: Mapping[str, str]) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        field: row[field] for field in ("episode_id", "pair_id", "task_suite", "initial_state_sha256")
    }
    for field in ("condition_order", "task_id", "episode_index", "replan_steps", "seed") + _BURDEN_FIELDS:
        output[field] = _integer(row[field], field)
    output["mode"] = row["mode"]
    output["success"] = _boolean(row["success"], "success")
    return output


def _paired_rows(episodes: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for raw in episodes:
        row = _episode(raw)
        grouped.setdefault(str(row["pair_id"]), {})[str(row["mode"])] = row
    output = []
    for pair_id, modes in sorted(grouped.items()):
        if set(modes) != set(MODES) or len(modes) != 2:
            raise AnalysisError("pair %s is incomplete" % pair_id)
        baseline = modes[ASYNC_UNGUARDED]
        aligned = modes[LATENCY_ALIGNED]
        for field in _PAIR_FIELDS:
            if baseline[field] != aligned[field]:
                raise AnalysisError("pair %s mismatches on %s" % (pair_id, field))
        row: Dict[str, Any] = {field: baseline[field] for field in _PAIR_FIELDS}
        row.update(
            {
                "async_success": baseline["success"],
                "aligned_success": aligned["success"],
                "success_difference": int(aligned["success"]) - int(baseline["success"]),
            }
        )
        for field in _BURDEN_FIELDS:
            row["async_%s" % field] = baseline[field]
            row["aligned_%s" % field] = aligned[field]
            row["%s_difference" % field] = aligned[field] - baseline[field]
        output.append(row)
    return output


def _burden_analysis(
    pairs: Sequence[Mapping[str, Any]], resamples: int, seed: int
) -> List[Dict[str, Any]]:
    output = []
    for offset, field in enumerate(_BURDEN_FIELDS):
        baseline = [float(row["async_%s" % field]) for row in pairs]
        aligned = [float(row["aligned_%s" % field]) for row in pairs]
        differences = [float(row["%s_difference" % field]) for row in pairs]
        interval = _paired_bootstrap_mean_interval(differences, resamples, seed + offset + 1)
        output.append(
            {
                "metric": field,
                "pairs": len(pairs),
                "async_total": int(sum(baseline)),
                "aligned_total": int(sum(aligned)),
                "async_mean_per_rollout": float(np.mean(baseline)) if baseline else None,
                "aligned_mean_per_rollout": float(np.mean(aligned)) if aligned else None,
                "paired_mean_difference": float(np.mean(differences)) if differences else None,
                "paired_bootstrap95_low": interval[0],
                "paired_bootstrap95_high": interval[1],
            }
        )
    return output


def _stable_source(root: pathlib.Path) -> Tuple[Mapping[str, bytes], Mapping[str, Any]]:
    first = validate_artifact(root)
    if not first.valid:
        raise AnalysisError("source artifact failed independent validation: %s" % "; ".join(first.errors))
    relative_paths = (
        "manifest.json", "resolved_protocol.json", "environment.json",
        "per_episode.csv", "per_query.csv", "progress.json", "integrity.json",
        "summary.json",
    )
    snapshots: Dict[str, bytes] = {}
    try:
        for relative in relative_paths:
            snapshots[relative] = (root / relative).read_bytes()
    except OSError as exc:
        raise AnalysisError("cannot snapshot source artifact: %s" % exc) from exc
    second = validate_artifact(root)
    if not second.valid:
        raise AnalysisError("source artifact changed or became invalid while reading")
    for relative, before in snapshots.items():
        try:
            after = (root / relative).read_bytes()
        except OSError as exc:
            raise AnalysisError("source artifact changed while reading: %s" % relative) from exc
        if after != before:
            raise AnalysisError("source artifact changed while reading: %s" % relative)
    return snapshots, first.to_dict()


def analyze_artifact(
    artifact: pathlib.Path,
    *,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or bootstrap_resamples < 100
        or bootstrap_resamples > 100_000
    ):
        raise AnalysisError("bootstrap_resamples must be an integer in [100, 100000]")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise AnalysisError("bootstrap_seed must be a nonnegative integer")
    root = pathlib.Path(artifact).resolve()
    snapshots, validator_report = _stable_source(root)
    protocol = _strict_json_bytes(snapshots["resolved_protocol.json"], "resolved_protocol.json")
    progress = _strict_json_bytes(snapshots["progress.json"], "progress.json")
    integrity = _strict_json_bytes(snapshots["integrity.json"], "integrity.json")
    if (
        protocol.get("schema_version") != SOURCE_SCHEMA_VERSION
        or progress.get("complete") is not True
        or integrity.get("valid") is not True
        or integrity.get("errors") != []
    ):
        raise AnalysisError("analysis requires a complete valid measured-age v2 artifact")
    episodes = _csv_rows(snapshots["per_episode.csv"], EPISODE_FIELDS, "per_episode.csv")
    queries = _csv_rows(snapshots["per_query.csv"], QUERY_FIELDS, "per_query.csv")
    pairs = _paired_rows(episodes)
    if not pairs:
        raise AnalysisError("analysis requires at least one complete pair")

    async_successes = sum(bool(row["async_success"]) for row in pairs)
    aligned_successes = sum(bool(row["aligned_success"]) for row in pairs)
    differences = [float(row["success_difference"]) for row in pairs]
    candidate_wins = sum(value > 0.0 for value in differences)
    reference_wins = sum(value < 0.0 for value in differences)
    ties = sum(value == 0.0 for value in differences)
    async_wilson = _wilson_interval(async_successes, len(pairs))
    aligned_wilson = _wilson_interval(aligned_successes, len(pairs))
    difference_ci = _paired_bootstrap_mean_interval(
        differences, bootstrap_resamples, bootstrap_seed
    )
    matrix = protocol.get("matrix")
    analysis = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "implementation": _implementation_identity(),
        "source": {
            "artifact": str(root),
            "source_schema_version": SOURCE_SCHEMA_VERSION,
            "source_manifest_sha256": _sha256_bytes(snapshots["manifest.json"]),
            "per_episode_sha256": _sha256_bytes(snapshots["per_episode.csv"]),
            "per_query_sha256": _sha256_bytes(snapshots["per_query.csv"]),
            "validator_schema_version": validator_report["schema_version"],
            "validator_checks": validator_report["checks"],
        },
        "cohort": {
            "rollouts": len(episodes),
            "pairs": len(pairs),
            "queries": len(queries),
            "task_suites": matrix.get("task_suites") if isinstance(matrix, Mapping) else None,
            "replan_steps": matrix.get("replan_steps") if isinstance(matrix, Mapping) else None,
            "age_rounding": protocol.get("temporal_alignment", {}).get("action_offset_rounding") if isinstance(protocol.get("temporal_alignment"), Mapping) else None,
            "deadline_ms": protocol.get("temporal_alignment", {}).get("deadline_ms") if isinstance(protocol.get("temporal_alignment"), Mapping) else None,
        },
        "success": {
            "async_unguarded": {
                "successes": async_successes, "rollouts": len(pairs),
                "rate": async_successes / float(len(pairs)),
                "wilson95_low": async_wilson[0], "wilson95_high": async_wilson[1],
            },
            "latency_aligned": {
                "successes": aligned_successes, "rollouts": len(pairs),
                "rate": aligned_successes / float(len(pairs)),
                "wilson95_low": aligned_wilson[0], "wilson95_high": aligned_wilson[1],
            },
            "paired": {
                "pairs": len(pairs),
                "rate_difference": float(np.mean(differences)),
                "paired_bootstrap95_low": difference_ci[0],
                "paired_bootstrap95_high": difference_ci[1],
                "candidate_wins": candidate_wins,
                "reference_wins": reference_wins,
                "ties": ties,
                "mcnemar_exact_two_sided_p": _mcnemar_exact_p(candidate_wins, reference_wins),
            },
        },
        "timing": [_mode_timing(queries, mode) for mode in MODES],
        "runtime_burden": _burden_analysis(pairs, bootstrap_resamples, bootstrap_seed),
        "statistics": {
            "confidence_level": 0.95,
            "success_rate_interval": "Wilson score interval",
            "paired_difference_interval": "paired percentile bootstrap of the mean",
            "paired_test": "exact two-sided McNemar binomial test",
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_resamples": bootstrap_resamples,
            "multiplicity_adjustment": "none; one prespecified success comparison",
        },
        "claim_boundary": (
            "Simulation-only measured-age analysis. Inference remains blocking and "
            "controller catch-up is simulated after response arrival."
        ),
    }
    return analysis, pairs


def _summary_markdown(analysis: Mapping[str, Any]) -> str:
    success = analysis["success"]
    paired = success["paired"]
    lines = [
        "# Measured-age paired analysis",
        "",
        "- Source manifest SHA-256: `%s`" % analysis["source"]["source_manifest_sha256"],
        "- Rollouts/pairs/queries: %d/%d/%d"
        % (analysis["cohort"]["rollouts"], analysis["cohort"]["pairs"], analysis["cohort"]["queries"]),
        "- Bootstrap seed/resamples: `%d/%d`"
        % (analysis["statistics"]["bootstrap_seed"], analysis["statistics"]["bootstrap_resamples"]),
        "",
        "| Mode | Success (Wilson 95%) | Age P95 / max ms | Deadline | Horizon | Refresh |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    timing_by_mode = {row["mode"]: row for row in analysis["timing"]}
    for mode in MODES:
        result = success[mode]
        timing = timing_by_mode[mode]
        lines.append(
            "| %s | %d/%d [%.3f, %.3f] | %.3f / %.3f | %d | %d | %d |"
            % (
                mode, result["successes"], result["rollouts"], result["wilson95_low"],
                result["wilson95_high"], timing["observation_age_ms"]["p95"],
                timing["observation_age_ms"]["max"], timing["deadline_misses"],
                timing["horizon_overruns"], timing["hold_refresh_queries"],
            )
        )
    lines.extend(
        [
            "",
            "Paired aligned-minus-async success difference: **%+.3f** "
            "(bootstrap 95%% [%+.3f, %+.3f]); wins/losses/ties %d/%d/%d; "
            "exact McNemar p=%.6g."
            % (
                paired["rate_difference"], paired["paired_bootstrap95_low"],
                paired["paired_bootstrap95_high"], paired["candidate_wins"],
                paired["reference_wins"], paired["ties"],
                paired["mcnemar_exact_two_sided_p"],
            ),
            "",
            str(analysis["claim_boundary"]),
            "",
        ]
    )
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
            files[relative] = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
    _write_json(root / "manifest.json", {"schema_version": ANALYSIS_SCHEMA_VERSION, "files": files})


def generate_report(
    artifact: pathlib.Path,
    output_directory: pathlib.Path,
    *,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    output = pathlib.Path(output_directory).resolve()
    if output.exists():
        raise AnalysisError("output directory already exists: %s" % output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(tempfile.mkdtemp(prefix=".%s." % output.name, dir=str(output.parent)))
    try:
        analysis, pairs = analyze_artifact(
            artifact,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        )
        _write_json(staging / "analysis.json", analysis)
        with (staging / "per_pair.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_PER_PAIR_FIELDS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(pairs)
        (staging / "summary.md").write_text(_summary_markdown(analysis), encoding="utf-8")
        _write_manifest(staging)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=pathlib.Path)
    parser.add_argument("--output-directory", type=pathlib.Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        analysis = generate_report(
            args.artifact,
            args.output_directory,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
    except (AnalysisError, OSError, ValueError) as exc:
        if args.json:
            print(json.dumps({"valid": False, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print("INVALID", exc)
        return 2
    if args.json:
        print(json.dumps(analysis, indent=2, sort_keys=True))
    else:
        print("VALID", pathlib.Path(args.output_directory).resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
