"""Task-cluster robustness analysis for a validated measured-age artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import os
import pathlib
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from integrations.openpi import measured_age_analysis as base_analysis
from integrations.openpi.validate_measured_age_artifact import EPISODE_FIELDS


ANALYSIS_SCHEMA_VERSION = "armbench.pi05_measured_age_confirmatory_analysis.v1"
BASE_ANALYSIS_SCHEMA_VERSION = base_analysis.ANALYSIS_SCHEMA_VERSION
CLUSTER_BOOTSTRAP_RESAMPLES = 10_000
CLUSTER_BOOTSTRAP_SEED = 20260805
EXPECTED_TASK_IDS = tuple(range(10))
EXPECTED_EPISODE_INDICES = tuple(range(5, 17))
ASYNC_UNGUARDED = base_analysis.ASYNC_UNGUARDED
LATENCY_ALIGNED = base_analysis.LATENCY_ALIGNED
MODES = (ASYNC_UNGUARDED, LATENCY_ALIGNED)
EXPECTED_OUTPUT_FILES = (
    "analysis.json",
    "condition_first.csv",
    "leave_one_task_out.csv",
    "per_task.csv",
    "summary.md",
)
AnalysisError = base_analysis.AnalysisError

PER_TASK_FIELDS = (
    "task_suite",
    "task_id",
    "pairs",
    "async_successes",
    "async_success_rate",
    "aligned_successes",
    "aligned_success_rate",
    "candidate_wins",
    "reference_wins",
    "ties",
    "risk_difference",
)
LEAVE_ONE_TASK_OUT_FIELDS = (
    "omitted_task_suite",
    "omitted_task_id",
    "remaining_tasks",
    "remaining_pairs",
    "candidate_wins",
    "reference_wins",
    "ties",
    "risk_difference",
)
CONDITION_FIRST_FIELDS = (
    "condition_first",
    "pairs",
    "async_successes",
    "async_success_rate",
    "aligned_successes",
    "aligned_success_rate",
    "candidate_wins",
    "reference_wins",
    "ties",
    "risk_difference",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_identity() -> Dict[str, str]:
    this_path = pathlib.Path(__file__).resolve()
    base_path = pathlib.Path(base_analysis.__file__).resolve()
    validator_path = this_path.with_name("validate_measured_age_artifact.py")
    return {
        "analyzer_source": "integrations/openpi/measured_age_confirmatory_analysis.py",
        "analyzer_sha256": _sha256_file(this_path),
        "base_analyzer_source": "integrations/openpi/measured_age_analysis.py",
        "base_analyzer_sha256": _sha256_file(base_path),
        "validator_source": "integrations/openpi/validate_measured_age_artifact.py",
        "validator_sha256": _sha256_file(validator_path),
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy_version": np.__version__,
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _validate_pair(raw: Mapping[str, Any]) -> Dict[str, Any]:
    required = (
        "pair_id",
        "task_suite",
        "task_id",
        "episode_index",
        "async_success",
        "aligned_success",
        "success_difference",
    )
    missing = [field for field in required if field not in raw]
    if missing:
        raise AnalysisError("pair is missing fields: %s" % ", ".join(missing))
    pair_id = raw["pair_id"]
    suite = raw["task_suite"]
    if not isinstance(pair_id, str) or not pair_id:
        raise AnalysisError("pair_id must be a nonempty string")
    if not isinstance(suite, str) or not suite:
        raise AnalysisError("task_suite must be a nonempty string")
    values: Dict[str, int] = {}
    for field in ("task_id", "episode_index"):
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise AnalysisError("%s must be an integer" % field)
        values[field] = int(value)
    async_success = raw["async_success"]
    aligned_success = raw["aligned_success"]
    if type(async_success) is not bool or type(aligned_success) is not bool:
        raise AnalysisError("paired success values must be booleans")
    expected_difference = int(aligned_success) - int(async_success)
    if raw["success_difference"] != expected_difference:
        raise AnalysisError("pair %s success_difference mismatch" % pair_id)
    return {
        "pair_id": pair_id,
        "task_suite": suite,
        "task_id": values["task_id"],
        "episode_index": values["episode_index"],
        "async_success": async_success,
        "aligned_success": aligned_success,
        "success_difference": expected_difference,
    }


def _outcome_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise AnalysisError("outcome summary requires at least one pair")
    pairs = len(rows)
    async_successes = sum(bool(row["async_success"]) for row in rows)
    aligned_successes = sum(bool(row["aligned_success"]) for row in rows)
    differences = [int(row["success_difference"]) for row in rows]
    return {
        "pairs": pairs,
        "async_successes": async_successes,
        "async_success_rate": async_successes / float(pairs),
        "aligned_successes": aligned_successes,
        "aligned_success_rate": aligned_successes / float(pairs),
        "candidate_wins": sum(value > 0 for value in differences),
        "reference_wins": sum(value < 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "risk_difference": sum(differences) / float(pairs),
    }


def _task_rows(
    grouped: Mapping[Tuple[str, int], Sequence[Mapping[str, Any]]]
) -> List[Dict[str, Any]]:
    output = []
    for (suite, task_id), rows in sorted(grouped.items()):
        output.append(
            {"task_suite": suite, "task_id": task_id, **_outcome_summary(rows)}
        )
    return output


def _cluster_bootstrap(
    task_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    task_sums = np.asarray(
        [
            int(row["candidate_wins"]) - int(row["reference_wins"])
            for row in task_rows
        ],
        dtype=np.int64,
    )
    task_counts = np.asarray([int(row["pairs"]) for row in task_rows], dtype=np.int64)
    rng = np.random.default_rng(CLUSTER_BOOTSTRAP_SEED)
    sampled = rng.integers(
        0,
        len(task_rows),
        size=(CLUSTER_BOOTSTRAP_RESAMPLES, len(task_rows)),
    )
    estimates = np.sum(task_sums[sampled], axis=1) / np.sum(
        task_counts[sampled], axis=1
    )
    return {
        "unit": "task block",
        "tasks_per_resample": len(task_rows),
        "resamples": CLUSTER_BOOTSTRAP_RESAMPLES,
        "seed": CLUSTER_BOOTSTRAP_SEED,
        "statistic": "pooled paired risk difference after resampling whole tasks",
        "percentile95_low": float(np.percentile(estimates, 2.5)),
        "percentile95_high": float(np.percentile(estimates, 97.5)),
    }


def _exact_task_sign_flip(task_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(task_rows) != 10:
        raise AnalysisError("exact task sign-flip requires exactly 10 tasks")
    effects = [float(row["risk_difference"]) for row in task_rows]
    observed = abs(sum(effects) / len(effects))
    tolerance = 1e-15
    extreme = 0
    permutations = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(effects)):
        statistic = abs(
            sum(sign * effect for sign, effect in zip(signs, effects))
            / len(effects)
        )
        extreme += statistic + tolerance >= observed
        permutations += 1
    return {
        "unit": "task-level risk difference",
        "alternative": "two-sided",
        "statistic": "absolute mean task risk difference",
        "observed_mean_task_risk_difference": sum(effects) / len(effects),
        "enumerated_assignments": permutations,
        "extreme_assignments": extreme,
        "exact_p": extreme / float(permutations),
    }


def _leave_one_task_out(
    grouped: Mapping[Tuple[str, int], Sequence[Mapping[str, Any]]]
) -> List[Dict[str, Any]]:
    output = []
    for omitted in sorted(grouped):
        remaining = [
            row
            for task, task_pairs in grouped.items()
            if task != omitted
            for row in task_pairs
        ]
        summary = _outcome_summary(remaining)
        output.append(
            {
                "omitted_task_suite": omitted[0],
                "omitted_task_id": omitted[1],
                "remaining_tasks": len(grouped) - 1,
                "remaining_pairs": summary["pairs"],
                "candidate_wins": summary["candidate_wins"],
                "reference_wins": summary["reference_wins"],
                "ties": summary["ties"],
                "risk_difference": summary["risk_difference"],
            }
        )
    return output


def analyze_pairs(
    raw_pairs: Sequence[Mapping[str, Any]],
    condition_first_by_pair: Mapping[str, str],
) -> Dict[str, Any]:
    pairs = [_validate_pair(row) for row in raw_pairs]
    pair_ids = [str(row["pair_id"]) for row in pairs]
    if len(set(pair_ids)) != len(pair_ids):
        raise AnalysisError("paired input contains duplicate pair_id values")
    if set(condition_first_by_pair) != set(pair_ids):
        raise AnalysisError("condition-first mapping does not exactly cover pairs")
    invalid_first = sorted(set(condition_first_by_pair.values()) - set(MODES))
    if invalid_first:
        raise AnalysisError("invalid condition-first modes: %s" % invalid_first)

    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in pairs:
        grouped.setdefault((str(row["task_suite"]), int(row["task_id"])), []).append(row)
    observed_suites = {key[0] for key in grouped}
    observed_task_ids = tuple(sorted(key[1] for key in grouped))
    if observed_suites != {"libero_spatial"} or observed_task_ids != EXPECTED_TASK_IDS:
        raise AnalysisError(
            "confirmatory task matrix must be libero_spatial task IDs 0 through 9"
        )
    episode_sets = {
        key: tuple(sorted(int(row["episode_index"]) for row in rows))
        for key, rows in grouped.items()
    }
    if any(len(values) != len(set(values)) for values in episode_sets.values()):
        raise AnalysisError("a task contains duplicate episode indices")
    if len(set(episode_sets.values())) != 1:
        raise AnalysisError("all tasks must use the same episode-index set")

    task_results = _task_rows(grouped)
    pooled = _outcome_summary(pairs)
    primary = {
        **pooled,
        "test": "two-sided exact McNemar",
        "alpha": 0.05,
        "exact_p": base_analysis._mcnemar_exact_p(
            int(pooled["candidate_wins"]), int(pooled["reference_wins"])
        ),
        "role": "single primary confirmatory test",
    }
    strata: List[Dict[str, Any]] = []
    for mode in MODES:
        rows = [row for row in pairs if condition_first_by_pair[str(row["pair_id"])] == mode]
        if not rows:
            raise AnalysisError("both condition-first strata must be nonempty")
        strata.append({"condition_first": mode, **_outcome_summary(rows)})

    leave_one_out = _leave_one_task_out(grouped)
    loto_effects = [float(row["risk_difference"]) for row in leave_one_out]
    return {
        "cohort": {
            "task_suites": ["libero_spatial"],
            "task_ids": list(EXPECTED_TASK_IDS),
            "tasks": len(grouped),
            "pairs": len(pairs),
            "rollouts": 2 * len(pairs),
            "episode_indices": list(next(iter(episode_sets.values()))),
            "pairs_per_task": len(next(iter(grouped.values()))),
            "condition_first_counts": {
                row["condition_first"]: row["pairs"] for row in strata
            },
        },
        "primary": primary,
        "secondary": {
            "per_task": task_results,
            "task_cluster_bootstrap": _cluster_bootstrap(task_results),
            "exact_task_sign_flip": _exact_task_sign_flip(task_results),
            "leave_one_task_out": leave_one_out,
            "leave_one_task_out_range": {
                "minimum_risk_difference": min(loto_effects),
                "maximum_risk_difference": max(loto_effects),
            },
            "condition_first": strata,
        },
    }


def _condition_first_from_episode_bytes(value: bytes) -> Dict[str, str]:
    rows = base_analysis._csv_rows(value, EPISODE_FIELDS, "per_episode.csv")
    grouped: Dict[str, List[Tuple[int, str]]] = {}
    for row in rows:
        order = base_analysis._integer(row["condition_order"], "condition_order")
        grouped.setdefault(row["pair_id"], []).append((order, row["mode"]))
    output: Dict[str, str] = {}
    for pair_id, entries in grouped.items():
        if len(entries) != 2 or {mode for _, mode in entries} != set(MODES):
            raise AnalysisError("pair %s does not contain exactly two modes" % pair_id)
        if entries[0][0] == entries[1][0]:
            raise AnalysisError("pair %s has duplicate condition_order" % pair_id)
        output[pair_id] = min(entries)[1]
    return output


def _require_frozen_confirmatory_cohort(
    result: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    condition_first_by_pair: Mapping[str, str],
) -> None:
    cohort = result.get("cohort")
    if not isinstance(cohort, Mapping):
        raise AnalysisError("confirmatory cohort is missing")
    if cohort.get("pairs") != 120 or cohort.get("rollouts") != 240:
        raise AnalysisError("confirmatory artifact must contain 120 pairs and 240 rollouts")
    if cohort.get("episode_indices") != list(EXPECTED_EPISODE_INDICES):
        raise AnalysisError("confirmatory artifact episode split must be 5:17")
    if cohort.get("pairs_per_task") != 12:
        raise AnalysisError("confirmatory artifact must contain 12 pairs per task")
    if cohort.get("condition_first_counts") != {
        ASYNC_UNGUARDED: 60,
        LATENCY_ALIGNED: 60,
    }:
        raise AnalysisError("confirmatory condition-first totals must be balanced")
    for task_id in EXPECTED_TASK_IDS:
        task_pairs = [row for row in pairs if int(row["task_id"]) == task_id]
        first_modes = [
            condition_first_by_pair[str(row["pair_id"])] for row in task_pairs
        ]
        if first_modes.count(ASYNC_UNGUARDED) != 6:
            raise AnalysisError("task %d async-first count must be 6" % task_id)
        if first_modes.count(LATENCY_ALIGNED) != 6:
            raise AnalysisError("task %d aligned-first count must be 6" % task_id)


def analyze_artifact(artifact: pathlib.Path) -> Dict[str, Any]:
    root = pathlib.Path(artifact).resolve()
    inherited, pairs = base_analysis.analyze_artifact(
        root,
        bootstrap_resamples=CLUSTER_BOOTSTRAP_RESAMPLES,
        bootstrap_seed=CLUSTER_BOOTSTRAP_SEED,
    )
    if inherited.get("schema_version") != BASE_ANALYSIS_SCHEMA_VERSION:
        raise AnalysisError("base measured-age analysis schema mismatch")
    snapshots, validator_report = base_analysis._stable_source(root)
    expected_hashes = {
        "manifest.json": inherited["source"]["source_manifest_sha256"],
        "per_episode.csv": inherited["source"]["per_episode_sha256"],
        "per_query.csv": inherited["source"]["per_query_sha256"],
    }
    for relative, expected in expected_hashes.items():
        if _sha256_bytes(snapshots[relative]) != expected:
            raise AnalysisError("source changed between base and confirmatory analysis")
    condition_first = _condition_first_from_episode_bytes(snapshots["per_episode.csv"])
    result = analyze_pairs(pairs, condition_first)
    _require_frozen_confirmatory_cohort(result, pairs, condition_first)
    if result["primary"]["exact_p"] != inherited["success"]["paired"][
        "mcnemar_exact_two_sided_p"
    ]:
        raise AnalysisError("pooled McNemar result disagrees with base analysis")
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "valid": True,
        "implementation": _implementation_identity(),
        "source": {
            **inherited["source"],
            "base_analysis_schema_version": BASE_ANALYSIS_SCHEMA_VERSION,
            "base_analysis_canonical_sha256": _sha256_bytes(
                _canonical_json_bytes(inherited)
            ),
            "second_validator_schema_version": validator_report["schema_version"],
        },
        **result,
        "statistics": {
            "primary_test": "pooled two-sided exact McNemar",
            "primary_alpha": 0.05,
            "secondary_status": "prespecified robustness analyses",
            "cluster_bootstrap_unit": "whole task",
            "cluster_bootstrap_resamples": CLUSTER_BOOTSTRAP_RESAMPLES,
            "cluster_bootstrap_seed": CLUSTER_BOOTSTRAP_SEED,
            "task_sign_flip": "two-sided exhaustive 2^10 sign assignments",
            "task_level_rows": "descriptive; no task-level significance claims",
            "multiplicity": "no secondary result replaces the single primary test",
        },
        "claim_boundary": (
            "Simulation-only task-cluster robustness analysis of a validated "
            "measured-age pi0.5-LIBERO artifact."
        ),
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    with io.StringIO(newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        return handle.getvalue().encode("utf-8")


def _summary_markdown(analysis: Mapping[str, Any]) -> str:
    primary = analysis["primary"]
    secondary = analysis["secondary"]
    bootstrap = secondary["task_cluster_bootstrap"]
    sign_flip = secondary["exact_task_sign_flip"]
    lines = [
        "# Measured-age confirmatory task-cluster analysis",
        "",
        "- Source manifest SHA-256: `%s`" % analysis["source"]["source_manifest_sha256"],
        "- Tasks/pairs/rollouts: %d/%d/%d"
        % (
            analysis["cohort"]["tasks"],
            analysis["cohort"]["pairs"],
            analysis["cohort"]["rollouts"],
        ),
        "- Primary pooled exact McNemar: wins/losses/ties %d/%d/%d, "
        "risk difference %+.4f, two-sided p=%.8g"
        % (
            primary["candidate_wins"],
            primary["reference_wins"],
            primary["ties"],
            primary["risk_difference"],
            primary["exact_p"],
        ),
        "- Whole-task bootstrap: %+.4f to %+.4f (%d resamples, seed %d)"
        % (
            bootstrap["percentile95_low"],
            bootstrap["percentile95_high"],
            bootstrap["resamples"],
            bootstrap["seed"],
        ),
        "- Exact task sign-flip: %d/%d extreme assignments, p=%.8g"
        % (
            sign_flip["extreme_assignments"],
            sign_flip["enumerated_assignments"],
            sign_flip["exact_p"],
        ),
        "- Leave-one-task-out risk-difference range: %+.4f to %+.4f"
        % (
            secondary["leave_one_task_out_range"]["minimum_risk_difference"],
            secondary["leave_one_task_out_range"]["maximum_risk_difference"],
        ),
        "",
        "| Task | Async | Aligned | Wins / losses / ties | Risk difference |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in secondary["per_task"]:
        lines.append(
            "| %d | %d/%d | %d/%d | %d / %d / %d | %+.4f |"
            % (
                row["task_id"],
                row["async_successes"],
                row["pairs"],
                row["aligned_successes"],
                row["pairs"],
                row["candidate_wins"],
                row["reference_wins"],
                row["ties"],
                row["risk_difference"],
            )
        )
    lines.extend(
        [
            "",
            "| Condition first | Pairs | Async | Aligned | Wins / losses / ties | Risk difference |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in secondary["condition_first"]:
        lines.append(
            "| %s | %d | %d/%d | %d/%d | %d / %d / %d | %+.4f |"
            % (
                row["condition_first"],
                row["pairs"],
                row["async_successes"],
                row["pairs"],
                row["aligned_successes"],
                row["pairs"],
                row["candidate_wins"],
                row["reference_wins"],
                row["ties"],
                row["risk_difference"],
            )
        )
    lines.extend(["", str(analysis["claim_boundary"]), ""])
    return "\n".join(lines)


def _write_manifest(root: pathlib.Path) -> None:
    files: Dict[str, Dict[str, Any]] = {}
    for path in sorted(root.iterdir()):
        if path.is_symlink():
            raise AnalysisError("report must not contain symlinks")
        if path.is_file() and path.name != "manifest.json":
            files[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    (root / "manifest.json").write_text(
        json.dumps(
            {"schema_version": ANALYSIS_SCHEMA_VERSION, "files": files},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def validate_report(report_directory: pathlib.Path) -> Dict[str, Any]:
    root = pathlib.Path(report_directory).resolve()
    errors: List[str] = []
    if not root.is_dir():
        return {"valid": False, "errors": ["report directory does not exist"]}
    try:
        paths = list(root.iterdir())
        if any(path.is_symlink() or not path.is_file() for path in paths):
            errors.append("report contains a symlink or non-file entry")
        actual = {path.name for path in paths}
        expected = set(EXPECTED_OUTPUT_FILES) | {"manifest.json"}
        if actual != expected:
            errors.append("report file set mismatch")
        manifest = base_analysis._strict_json_bytes(
            (root / "manifest.json").read_bytes(), "manifest.json"
        )
        if manifest.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
            errors.append("manifest schema mismatch")
        records = manifest.get("files")
        if not isinstance(records, Mapping) or set(records) != set(EXPECTED_OUTPUT_FILES):
            errors.append("manifest coverage mismatch")
            records = {}
        for name in EXPECTED_OUTPUT_FILES:
            path = root / name
            record = records.get(name) if isinstance(records, Mapping) else None
            if not path.is_file() or not isinstance(record, Mapping):
                errors.append("missing manifest-protected file: %s" % name)
                continue
            if record.get("bytes") != path.stat().st_size:
                errors.append("byte count mismatch: %s" % name)
            if record.get("sha256") != _sha256_file(path):
                errors.append("SHA-256 mismatch: %s" % name)
        analysis = base_analysis._strict_json_bytes(
            (root / "analysis.json").read_bytes(), "analysis.json"
        )
        if analysis.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
            errors.append("analysis schema mismatch")
        if analysis.get("valid") is not True:
            errors.append("analysis is not explicitly valid")
        secondary = analysis.get("secondary")
        if not isinstance(secondary, Mapping):
            errors.append("analysis secondary section is missing")
        else:
            expected_payloads = {
                "per_task.csv": _csv_bytes(secondary.get("per_task", []), PER_TASK_FIELDS),
                "leave_one_task_out.csv": _csv_bytes(
                    secondary.get("leave_one_task_out", []), LEAVE_ONE_TASK_OUT_FIELDS
                ),
                "condition_first.csv": _csv_bytes(
                    secondary.get("condition_first", []), CONDITION_FIRST_FIELDS
                ),
                "summary.md": _summary_markdown(analysis).encode("utf-8"),
            }
            for name, expected_bytes in expected_payloads.items():
                if (root / name).read_bytes() != expected_bytes:
                    errors.append("derived output mismatch: %s" % name)
    except (AnalysisError, OSError, TypeError, ValueError, KeyError) as exc:
        errors.append(str(exc))
    return {"valid": not errors, "errors": errors}


def generate_report(
    artifact: pathlib.Path, output_directory: pathlib.Path
) -> Dict[str, Any]:
    output = pathlib.Path(output_directory).resolve()
    if output.exists():
        raise AnalysisError("output directory already exists: %s" % output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(
        tempfile.mkdtemp(prefix=".%s." % output.name, dir=str(output.parent))
    )
    try:
        analysis = analyze_artifact(artifact)
        (staging / "analysis.json").write_text(
            json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        secondary = analysis["secondary"]
        (staging / "per_task.csv").write_bytes(
            _csv_bytes(secondary["per_task"], PER_TASK_FIELDS)
        )
        (staging / "leave_one_task_out.csv").write_bytes(
            _csv_bytes(secondary["leave_one_task_out"], LEAVE_ONE_TASK_OUT_FIELDS)
        )
        (staging / "condition_first.csv").write_bytes(
            _csv_bytes(secondary["condition_first"], CONDITION_FIRST_FIELDS)
        )
        (staging / "summary.md").write_bytes(
            _summary_markdown(analysis).encode("utf-8")
        )
        _write_manifest(staging)
        validation = validate_report(staging)
        if not validation["valid"]:
            raise AnalysisError(
                "generated report failed validation: %s"
                % "; ".join(validation["errors"])
            )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", allow_abbrev=False)
    analyze.add_argument("artifact", type=pathlib.Path)
    analyze.add_argument("--output-directory", type=pathlib.Path, required=True)
    analyze.add_argument("--json", action="store_true")
    validate = subparsers.add_parser("validate", allow_abbrev=False)
    validate.add_argument("report_directory", type=pathlib.Path)
    validate.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "analyze":
            result = generate_report(args.artifact, args.output_directory)
            valid = True
        else:
            result = validate_report(args.report_directory)
            valid = bool(result["valid"])
    except (AnalysisError, OSError, TypeError, ValueError, KeyError) as exc:
        result = {"valid": False, "errors": [str(exc)]}
        valid = False
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        print("VALID" if valid else "INVALID")
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
