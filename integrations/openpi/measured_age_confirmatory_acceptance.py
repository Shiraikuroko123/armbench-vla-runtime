"""Validate and present the held-out measured-age confirmatory evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import webbrowser
from typing import Any, Dict, Mapping, Optional, Sequence

from integrations.openpi import measured_age_analysis as base_analysis
from integrations.openpi.measured_age_compose_run import validate_run_manifest
from integrations.openpi.measured_age_confirmatory_analysis import (
    analyze_artifact,
    validate_report,
)
from integrations.openpi.measured_age_dashboard import build_dashboard


REPORT_SCHEMA_VERSION = "armbench.measured_age_confirmatory_acceptance.v1"
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE_ROOT = (
    PROJECT_ROOT / "evidence" / "pi05_libero_measured_age_confirmatory_001"
)
DEFAULT_RUN_ROOT = DEFAULT_EVIDENCE_ROOT / "run"
DEFAULT_BASE_ANALYSIS_ROOT = DEFAULT_EVIDENCE_ROOT / "analysis"
DEFAULT_CONFIRMATORY_ANALYSIS_ROOT = DEFAULT_EVIDENCE_ROOT / "confirmatory_analysis"
DEFAULT_DASHBOARD = (
    PROJECT_ROOT
    / "reports"
    / "pi05_libero_measured_age_confirmatory_001"
    / "index.html"
)
_SOURCE_HASH_FIELDS = (
    "source_manifest_sha256",
    "per_episode_sha256",
    "per_query_sha256",
)
_SCIENTIFIC_FIELDS = (
    "schema_version",
    "cohort",
    "primary",
    "secondary",
    "statistics",
    "claim_boundary",
)
_BASE_SCIENTIFIC_FIELDS = (
    "schema_version",
    "cohort",
    "success",
    "timing",
    "runtime_burden",
    "statistics",
    "claim_boundary",
)
_CHECKPOINT_URI = "gs://openpi-assets/checkpoints/pi05_libero"
_POLICY_CONFIG = "pi05_libero"
_CLAIM_BOUNDARY = (
    "Simulation-only measured-age pi0.5-LIBERO evidence. Inference remains "
    "blocking and controller catch-up is simulated after response arrival; "
    "this is not true concurrent control, a hard real-time guarantee, physical "
    "safety, cross-model evidence, RTC-style policy-internal continuation, or "
    "a real-robot result."
)


def _strict_json(path: pathlib.Path) -> Dict[str, Any]:
    def reject(value: str) -> Any:
        raise ValueError("non-finite JSON constant: %s" % value)

    def unique(pairs: Sequence[tuple[str, Any]]) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError("duplicate JSON key: %s" % key)
            output[key] = value
        return output

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject,
            object_pairs_hook=unique,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("cannot read strict JSON %s: %s" % (path, exc)) from exc
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object: %s" % path)
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("%s must be an object" % label)
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_recomputed_analysis(
    source_root: pathlib.Path,
    base_analysis_root: pathlib.Path,
    confirmatory_analysis_root: pathlib.Path,
) -> Dict[str, Any]:
    validation = validate_report(confirmatory_analysis_root)
    if validation.get("valid") is not True:
        raise ValueError(
            "confirmatory report validation failed: %s"
            % "; ".join(str(value) for value in validation.get("errors", []))
        )
    recorded = _strict_json(confirmatory_analysis_root / "analysis.json")
    recomputed = analyze_artifact(source_root)
    for field in _SCIENTIFIC_FIELDS:
        if recorded.get(field) != recomputed.get(field):
            raise ValueError("confirmatory analysis disagrees with source: %s" % field)

    recorded_source = _mapping(recorded.get("source"), "confirmatory source")
    recomputed_source = _mapping(recomputed.get("source"), "recomputed source")
    base = _strict_json(base_analysis_root / "analysis.json")
    fresh_base, _fresh_pairs = base_analysis.analyze_artifact(
        source_root,
        bootstrap_resamples=base_analysis.DEFAULT_BOOTSTRAP_RESAMPLES,
        bootstrap_seed=base_analysis.DEFAULT_BOOTSTRAP_SEED,
    )
    for field in _BASE_SCIENTIFIC_FIELDS:
        if base.get(field) != fresh_base.get(field):
            raise ValueError("base analysis disagrees with source: %s" % field)
    base_source = _mapping(base.get("source"), "base-analysis source")
    for field in _SOURCE_HASH_FIELDS:
        expected = recorded_source.get(field)
        if (
            expected != recomputed_source.get(field)
            or expected != base_source.get(field)
            or expected != fresh_base["source"].get(field)
        ):
            raise ValueError("analysis source binding mismatch: %s" % field)
    saved_base_canonical = _canonical_sha256(base)
    fresh_base_canonical = _canonical_sha256(fresh_base)
    if recorded_source.get("base_analysis_canonical_sha256") != saved_base_canonical:
        raise ValueError("confirmatory analysis base canonical hash mismatch")
    if recomputed_source.get("base_analysis_canonical_sha256") != fresh_base_canonical:
        raise ValueError("recomputed confirmatory base canonical hash mismatch")
    return recorded


def build_acceptance(
    run_root: pathlib.Path = DEFAULT_RUN_ROOT,
    base_analysis_root: pathlib.Path = DEFAULT_BASE_ANALYSIS_ROOT,
    confirmatory_analysis_root: pathlib.Path = DEFAULT_CONFIRMATORY_ANALYSIS_ROOT,
    dashboard_output: pathlib.Path = DEFAULT_DASHBOARD,
) -> Dict[str, Any]:
    run_root = pathlib.Path(run_root).resolve()
    source_root = run_root / "evaluation"
    base_analysis_root = pathlib.Path(base_analysis_root).resolve()
    confirmatory_analysis_root = pathlib.Path(confirmatory_analysis_root).resolve()
    dashboard_output = pathlib.Path(dashboard_output).resolve()

    run_validation = validate_run_manifest(run_root)
    if run_validation.get("valid") is not True or run_validation.get("complete") is not True:
        raise ValueError(
            "root run validation failed: %s"
            % "; ".join(str(value) for value in run_validation.get("errors", []))
        )
    analysis = _require_recomputed_analysis(
        source_root, base_analysis_root, confirmatory_analysis_root
    )
    cohort = _mapping(analysis.get("cohort"), "confirmatory cohort")
    if cohort.get("tasks") != 10 or cohort.get("pairs") != 120 or cohort.get("rollouts") != 240:
        raise ValueError("confirmatory cohort is not the frozen 10/120/240 matrix")

    primary = _mapping(analysis.get("primary"), "primary analysis")
    secondary = _mapping(analysis.get("secondary"), "secondary analysis")
    bootstrap = _mapping(
        secondary.get("task_cluster_bootstrap"), "task-cluster bootstrap"
    )
    sign_flip = _mapping(
        secondary.get("exact_task_sign_flip"), "task sign-flip"
    )
    leave_one_out = _mapping(
        secondary.get("leave_one_task_out_range"), "leave-one-task-out range"
    )
    environment = _strict_json(source_root / "environment.json")
    metadata = _mapping(environment.get("server_metadata"), "server metadata")
    attestation = _mapping(
        metadata.get("armbench_server_attestation"), "server attestation"
    )
    sampling = _mapping(
        metadata.get("armbench_policy_sampling_contract"),
        "policy-sampling contract",
    )
    if (
        attestation.get("policy_loaded") is not True
        or attestation.get("policy_config") != _POLICY_CONFIG
        or attestation.get("checkpoint_uri") != _CHECKPOINT_URI
        or attestation.get("action_horizon") != 10
        or attestation.get("model_action_dim") != 32
        or attestation.get("openpi_tracked_clean") is not True
        or attestation.get("openpi_submodules_clean") is not True
        or attestation.get("openpi_commit") != environment.get("openpi_git_commit")
    ):
        raise ValueError("server attestation is not the frozen clean pi05_libero policy")
    if (
        sampling.get("schema_version") != "armbench.policy_sampling.v1"
        or sampling.get("noise_shape") != [10, 32]
        or sampling.get("mode_in_key") is not False
    ):
        raise ValueError("policy-sampling contract is not the frozen paired-noise contract")

    # Rendering is the final operation: no new dashboard is written until all
    # source, analysis, cohort, identity, and sampling checks have passed.
    dashboard = build_dashboard(source_root, base_analysis_root, dashboard_output)
    if dashboard.get("pairs") != 120 or dashboard.get("rollouts") != 240:
        raise ValueError("dashboard cohort disagrees with confirmatory analysis")

    exact_p = float(primary["exact_p"])
    candidate_wins = int(primary["candidate_wins"])
    reference_wins = int(primary["reference_wins"])
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "valid": True,
        "primary_positive": exact_p <= 0.05 and candidate_wins > reference_wins,
        "primary": {
            "async_successes": int(primary["async_successes"]),
            "aligned_successes": int(primary["aligned_successes"]),
            "pairs": int(primary["pairs"]),
            "candidate_wins": candidate_wins,
            "reference_wins": reference_wins,
            "ties": int(primary["ties"]),
            "risk_difference": float(primary["risk_difference"]),
            "exact_mcnemar_p": exact_p,
        },
        "task_sensitivity": {
            "cluster_bootstrap95": [
                float(bootstrap["percentile95_low"]),
                float(bootstrap["percentile95_high"]),
            ],
            "exact_task_sign_flip_p": float(sign_flip["exact_p"]),
            "leave_one_task_out_range": [
                float(leave_one_out["minimum_risk_difference"]),
                float(leave_one_out["maximum_risk_difference"]),
            ],
        },
        "identity": {
            "armbench_run_commit": environment.get("armbench_git_commit"),
            "openpi_commit": environment.get("openpi_git_commit"),
            "openpi_tracked_clean": attestation.get("openpi_tracked_clean"),
            "openpi_submodules_clean": attestation.get("openpi_submodules_clean"),
            "policy_config": attestation.get("policy_config"),
            "checkpoint_uri": attestation.get("checkpoint_uri"),
            "checkpoint_content_sha256": attestation.get(
                "checkpoint_content_sha256"
            ),
            "action_horizon": attestation.get("action_horizon"),
            "source_manifest_sha256": analysis["source"][
                "source_manifest_sha256"
            ],
            "base_analysis_canonical_sha256": analysis["source"][
                "base_analysis_canonical_sha256"
            ],
            "base_analysis_manifest_sha256": _sha256_file(
                base_analysis_root / "manifest.json"
            ),
            "confirmatory_analysis_manifest_sha256": _sha256_file(
                confirmatory_analysis_root / "manifest.json"
            ),
            "confirmatory_analyzer_sha256": analysis["implementation"][
                "analyzer_sha256"
            ],
            "base_analyzer_sha256": analysis["implementation"][
                "base_analyzer_sha256"
            ],
            "validator_sha256": analysis["implementation"]["validator_sha256"],
            "paired_policy_noise": True,
        },
        "evidence": {
            "rollouts": int(cohort["rollouts"]),
            "pairs": int(cohort["pairs"]),
            "videos_verified": int(dashboard["videos_verified"]),
            "root_files_checked": int(run_validation["files_checked"]),
        },
        "dashboard": str(dashboard_output),
        "dashboard_uri": dashboard_output.as_uri(),
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-root", type=pathlib.Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--base-analysis-root", type=pathlib.Path, default=DEFAULT_BASE_ANALYSIS_ROOT
    )
    parser.add_argument(
        "--confirmatory-analysis-root",
        type=pathlib.Path,
        default=DEFAULT_CONFIRMATORY_ANALYSIS_ROOT,
    )
    parser.add_argument("--dashboard", type=pathlib.Path, default=DEFAULT_DASHBOARD)
    parser.add_argument("--no-open", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = build_acceptance(
            args.run_root,
            args.base_analysis_root,
            args.confirmatory_analysis_root,
            args.dashboard,
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(
            json.dumps(
                {"schema_version": REPORT_SCHEMA_VERSION, "valid": False, "error": str(exc)},
                ensure_ascii=False,
            )
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    if not args.no_open:
        webbrowser.open(report["dashboard_uri"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
