"""Validate formal pi0.5 evidence and open the interview acceptance dashboard."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import webbrowser
from typing import Any, Dict, List, Mapping, Optional, Sequence

from integrations.openpi.acceptance_dashboard import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUT,
    DEFAULT_RUN_ROOT,
    build_dashboard,
)


REPORT_SCHEMA_VERSION = "armbench.interview_acceptance.v1"


def _read_json(path: pathlib.Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read JSON %s: %s" % (path, exc)) from exc
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object: %s" % path)
    return value


def _read_csv(path: pathlib.Path) -> List[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise ValueError("cannot read CSV %s: %s" % (path, exc)) from exc


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("%s must be an object" % label)
    return value


def _strict_bool(value: str, label: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError("%s must be exactly True or False" % label)


def _strict_int(value: str, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be an integer" % label) from exc
    if str(parsed) != str(value):
        raise ValueError("%s must use canonical integer syntax" % label)
    return parsed


def _verified_video(evaluation_root: pathlib.Path, raw_path: str) -> pathlib.Path:
    if not raw_path:
        raise ValueError("representative episode has no video path")
    root = evaluation_root.resolve()
    video = (evaluation_root / raw_path).resolve()
    try:
        video.relative_to(root)
    except ValueError as exc:
        raise ValueError("representative video escapes evaluation root") from exc
    if not video.is_file() or video.stat().st_size <= 0:
        raise ValueError("representative video is absent or empty: %s" % video)
    return video


def _representative_pair(
    run_root: pathlib.Path, analysis_root: pathlib.Path
) -> Dict[str, Any]:
    pairs = _read_csv(analysis_root / "per_pair.csv")
    candidates = []
    for row in pairs:
        if (
            row.get("latency_steps") == "4"
            and not _strict_bool(
                row.get("async_unguarded_success", ""),
                "async_unguarded_success",
            )
            and _strict_bool(
                row.get("latency_aligned_success", ""),
                "latency_aligned_success",
            )
            and not _strict_bool(
                row.get("async_unguarded_runtime_failure", ""),
                "async_unguarded_runtime_failure",
            )
            and not _strict_bool(
                row.get("latency_aligned_runtime_failure", ""),
                "latency_aligned_runtime_failure",
            )
        ):
            candidates.append(row)
    if not candidates:
        raise ValueError("no validated 200 ms baseline-failure/aligned-success pair")
    selected = min(
        candidates,
        key=lambda row: (
            _strict_int(row.get("task_id", ""), "task_id"),
            _strict_int(row.get("episode_index", ""), "episode_index"),
        ),
    )
    episodes = _read_csv(run_root / "evaluation" / "per_episode.csv")
    by_mode = {
        row.get("mode", ""): row
        for row in episodes
        if row.get("pair_id") == selected.get("pair_id")
    }
    expected_modes = ("async_unguarded", "latency_aligned")
    if any(mode not in by_mode for mode in expected_modes):
        raise ValueError("representative pair is missing a runtime mode")
    evaluation_root = run_root / "evaluation"
    async_row = by_mode["async_unguarded"]
    aligned_row = by_mode["latency_aligned"]
    return {
        "pair_id": selected["pair_id"],
        "task_id": _strict_int(selected["task_id"], "task_id"),
        "episode_index": _strict_int(selected["episode_index"], "episode_index"),
        "latency_ms": float(async_row["injected_latency_ms"]),
        "task_description": async_row["task_description"],
        "baseline": {
            "mode": "async_unguarded",
            "success": _strict_bool(async_row["success"], "baseline.success"),
            "policy_queries": _strict_int(
                async_row["policy_queries"], "baseline.policy_queries"
            ),
            "video": str(_verified_video(evaluation_root, async_row["video_path"])),
        },
        "method": {
            "mode": "latency_aligned",
            "success": _strict_bool(aligned_row["success"], "method.success"),
            "policy_queries": _strict_int(
                aligned_row["policy_queries"], "method.policy_queries"
            ),
            "video": str(
                _verified_video(evaluation_root, aligned_row["video_path"])
            ),
        },
    }


def build_interview_acceptance(
    run_root: pathlib.Path = DEFAULT_RUN_ROOT,
    analysis_root: pathlib.Path = DEFAULT_ANALYSIS_ROOT,
    output: pathlib.Path = DEFAULT_OUTPUT,
) -> Dict[str, Any]:
    """Fail closed unless the evidence, identity, result, and videos agree."""

    run_root = run_root.resolve()
    analysis_root = analysis_root.resolve()
    output = output.resolve()

    # This performs root and nested validation, rechecks analysis bindings, and
    # verifies every required video before creating the page.
    dashboard = build_dashboard(run_root, analysis_root, output)
    analysis = _read_json(analysis_root / "analysis.json")
    attestation = _read_json(run_root / "checkpoint_attestation.json")
    identity = _required_mapping(analysis.get("frozen_identity"), "frozen_identity")

    if attestation.get("policy_loaded") is not True:
        raise ValueError("checkpoint attestation does not record policy_loaded=true")
    identity_bindings = {
        "policy_config": "policy_config",
        "checkpoint": "checkpoint_uri",
        "checkpoint_content_sha256": "checkpoint_content_sha256",
        "openpi_commit": "openpi_commit",
    }
    for analysis_field, attestation_field in identity_bindings.items():
        if identity.get(analysis_field) != attestation.get(attestation_field):
            raise ValueError(
                "analysis/attestation identity mismatch: %s" % analysis_field
            )
    if attestation.get("openpi_tracked_clean") is not True:
        raise ValueError("attested OpenPI checkout was not clean")
    if attestation.get("action_horizon") != 10:
        raise ValueError("attested pi0.5 action horizon is not 10")

    itt = _required_mapping(analysis.get("itt"), "itt")
    strata = analysis.get("latency_strata")
    if not isinstance(strata, list):
        raise ValueError("latency_strata must be a list")
    primary_rows = [
        row
        for row in strata
        if isinstance(row, Mapping) and row.get("analysis_role") == "primary"
    ]
    if len(primary_rows) != 1:
        raise ValueError("analysis must contain exactly one primary latency stratum")
    primary = primary_rows[0]
    representative = _representative_pair(run_root, analysis_root)

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "valid": True,
        "problem": (
            "Asynchronous inference can return an action chunk whose leading "
            "actions are stale before execution starts."
        ),
        "method": (
            "Without retraining pi0.5, discard the delay-matched stale prefix "
            "and execute the following five-action suffix."
        ),
        "formal_result": {
            "condition": "%g ms deterministic injected LIBERO delay"
            % float(primary["injected_latency_ms"]),
            "matched_pairs": int(primary["paired_n"]),
            "baseline_successes": int(primary["async_unguarded_successes"]),
            "method_successes": int(primary["latency_aligned_successes"]),
            "success_rate_difference": float(
                primary["aligned_minus_async_success_rate_difference"]
            ),
            "paired_bootstrap95": [
                float(primary["paired_bootstrap95_low"]),
                float(primary["paired_bootstrap95_high"]),
            ],
            "mcnemar_holm_p": float(primary["mcnemar_holm_p"]),
        },
        "pi05_identity": {
            "policy_loaded": True,
            "policy_config": attestation["policy_config"],
            "checkpoint_uri": attestation["checkpoint_uri"],
            "checkpoint_content_sha256": attestation[
                "checkpoint_content_sha256"
            ],
            "checkpoint_file_count": attestation["checkpoint_file_count"],
            "openpi_commit": attestation["openpi_commit"],
            "armbench_run_commit": identity["armbench_run_commit"],
        },
        "evidence": {
            "root_and_nested_validation": "valid / complete",
            "protected_files_checked": int(dashboard["files_checked"]),
            "rollouts": int(itt["rollouts"]),
            "matched_pairs": int(itt["pairs"]),
            "videos_verified": int(dashboard["videos_verified"]),
            "runtime_failures_retained": int(itt["runtime_failures_retained"]),
        },
        "demo": {
            "dashboard": str(output),
            "dashboard_uri": output.as_uri(),
            "representative_pair": representative,
        },
        "claim_boundary": analysis.get("claim_boundary"),
        "five_minute_order": [
            "Read problem and method in this report.",
            "Confirm valid/complete evidence and the checkpoint content SHA-256.",
            "Open the dashboard at the primary 200 ms condition.",
            "Play the representative baseline and aligned videos together.",
            "State the simulation-only, deterministic-delay claim boundary.",
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-root", type=pathlib.Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--analysis-root", type=pathlib.Path, default=DEFAULT_ANALYSIS_ROOT
    )
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-open", action="store_true", help="validate and build without opening a browser"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = build_interview_acceptance(
            args.run_root, args.analysis_root, args.output
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": REPORT_SCHEMA_VERSION,
                    "valid": False,
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        )
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not args.no_open:
        webbrowser.open(report["demo"]["dashboard_uri"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
