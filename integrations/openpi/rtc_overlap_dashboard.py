"""Build the offline three-video acceptance dashboard for the RTC overlap pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import tempfile
import webbrowser
from typing import Any, Dict, Mapping, Optional, Sequence
from urllib.parse import quote

from integrations.openpi import rtc_overlap_analysis, rtc_overlap_pilot


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_ROOT = (
    PROJECT_ROOT / "evidence" / "pi05_rtc_overlap_pilot_001" / "evaluation"
)
DEFAULT_ANALYSIS_ROOT = (
    PROJECT_ROOT / "evidence" / "pi05_rtc_overlap_pilot_001" / "analysis"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "pi05_rtc_overlap_pilot_001" / "index.html"
DASHBOARD_SCHEMA_VERSION = "armbench.pi05_rtc_overlap_dashboard.v1"
METHODS = tuple(rtc_overlap_analysis.METHODS)
BASELINE = rtc_overlap_analysis.BASELINE
CANDIDATES = tuple(rtc_overlap_analysis.CANDIDATES)
METHOD_LABELS = {
    rtc_overlap_pilot.OVERLAP_UNCONDITIONED: "Unconditioned overlap",
    rtc_overlap_pilot.PROJECTED_OVERLAP: "Hard projection",
    rtc_overlap_pilot.RTC_GUIDED_OVERLAP: "RTC guidance",
}
EXPECTED_ANALYSIS_FILES = {"analysis.json", "per_episode.csv", "summary.md"}
BOOTSTRAP_IDENTITY_FIELDS = (
    "response_action_sha256",
    "sampling_key_sha256",
    "sampling_noise_sha256",
)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_finite(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("%s contains a non-finite number" % label)
    if isinstance(value, Mapping):
        for key, item in value.items():
            _check_finite(item, "%s.%s" % (label, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite(item, "%s[%d]" % (label, index))


def _read_strict_json(path: pathlib.Path, expected: type) -> Any:
    def reject(token: str) -> Any:
        raise ValueError("non-finite JSON constant: %s" % token)

    def unique(pairs: Sequence[tuple[str, Any]]) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError("duplicate JSON key: %s" % key)
            output[key] = value
        return output

    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject,
            object_pairs_hook=unique,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("cannot read strict JSON %s: %s" % (path, exc)) from exc
    if not isinstance(value, expected):
        raise ValueError("%s must contain a %s" % (path, expected.__name__))
    _check_finite(value, path.name)
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("%s must be a canonical SHA-256" % label)
    return value


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("%s must be an integer >= %d" % (label, minimum))
    return value


def _strict_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be numeric" % label)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("%s must be finite" % label)
    return result


def _validate_analysis_manifest(analysis_root: pathlib.Path) -> Dict[str, Any]:
    manifest_path = analysis_root / "manifest.json"
    manifest = _read_strict_json(manifest_path, dict)
    if manifest.get("schema_version") != rtc_overlap_analysis.ANALYSIS_SCHEMA_VERSION:
        raise ValueError("saved analysis manifest schema mismatch")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != EXPECTED_ANALYSIS_FILES:
        raise ValueError(
            "saved analysis manifest inventory is not the frozen three-file report"
        )
    actual = {
        path.relative_to(analysis_root).as_posix()
        for path in analysis_root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual != EXPECTED_ANALYSIS_FILES:
        raise ValueError(
            "saved analysis directory inventory disagrees with its manifest"
        )
    verified: Dict[str, Dict[str, Any]] = {}
    for relative in sorted(EXPECTED_ANALYSIS_FILES):
        if "\\" in relative or pathlib.PurePosixPath(relative).is_absolute():
            raise ValueError("analysis manifest contains a non-canonical path")
        record = files[relative]
        if not isinstance(record, Mapping):
            raise ValueError("analysis manifest record is not an object: %s" % relative)
        expected_bytes = _strict_int(record.get("bytes"), relative + ".bytes")
        expected_sha = _require_sha256(record.get("sha256"), relative + ".sha256")
        path = analysis_root / pathlib.PurePosixPath(relative)
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise ValueError("saved analysis file size mismatch: %s" % relative)
        actual_sha = _sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError("saved analysis file SHA-256 mismatch: %s" % relative)
        verified[relative] = {"bytes": expected_bytes, "sha256": actual_sha}
    return {
        "manifestSha256": _sha256_file(manifest_path),
        "files": verified,
    }


def _relative_video_url(
    evaluation_root: pathlib.Path, output_parent: pathlib.Path, raw_path: Any
) -> str:
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
        raise ValueError("required video path is empty or non-canonical")
    pure = pathlib.PurePosixPath(raw_path)
    if pure.is_absolute() or pure.as_posix() != raw_path or ".." in pure.parts:
        raise ValueError(
            "required video path is not a canonical relative path: %s" % raw_path
        )
    if pure.parts[:1] != ("videos",) or pure.suffix.lower() != ".mp4":
        raise ValueError(
            "required video must be a videos/*.mp4 artifact: %s" % raw_path
        )
    resolved_root = evaluation_root.resolve()
    resolved_video = (evaluation_root / pure).resolve()
    try:
        resolved_video.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("video escapes evaluation directory: %s" % raw_path) from exc
    if not resolved_video.is_file() or resolved_video.stat().st_size <= 0:
        raise ValueError("required video is missing or empty: %s" % resolved_video)
    try:
        relative = pathlib.Path(
            os.path.relpath(resolved_video, output_parent.resolve())
        )
    except ValueError as exc:
        raise ValueError("video and dashboard must be on the same filesystem") from exc
    url = quote(relative.as_posix(), safe="/")
    if pathlib.PurePosixPath(relative.as_posix()).is_absolute() or "://" in url:
        raise ValueError("dashboard video URL must remain relative")
    return url


def _method_summary(analysis: Mapping[str, Any], method: str) -> Dict[str, Any]:
    try:
        success = analysis["success"]["methods"][method]
        motion = analysis["seam"]["seam_motion_l2"]["methods"][method]
        gripper = analysis["seam"]["seam_gripper_abs"]["methods"][method]
    except (KeyError, TypeError) as exc:
        raise ValueError("recomputed analysis is missing method statistics") from exc
    return {
        "id": method,
        "label": METHOD_LABELS[method],
        "successes": _strict_int(success.get("successes"), method + ".successes"),
        "rollouts": _strict_int(
            success.get("rollouts"), method + ".rollouts", minimum=1
        ),
        "rate": _strict_number(success.get("rate"), method + ".rate"),
        "wilsonLow": _strict_number(
            success.get("wilson95_low"), method + ".wilson95_low"
        ),
        "wilsonHigh": _strict_number(
            success.get("wilson95_high"), method + ".wilson95_high"
        ),
        "meanMotionSeam": _strict_number(motion.get("mean"), method + ".motion.mean"),
        "medianMotionSeam": _strict_number(
            motion.get("median"), method + ".motion.median"
        ),
        "meanGripperSeam": _strict_number(
            gripper.get("mean"), method + ".gripper.mean"
        ),
        "medianGripperSeam": _strict_number(
            gripper.get("median"), method + ".gripper.median"
        ),
        "scoredTransitions": _strict_int(
            motion.get("scored_transitions"), method + ".scored_transitions", minimum=1
        ),
    }


def _contrast_summary(analysis: Mapping[str, Any], candidate: str) -> Dict[str, Any]:
    try:
        success = analysis["success"]["contrasts_vs_unconditioned"][candidate]
        motion = analysis["seam"]["seam_motion_l2"]["contrasts_vs_unconditioned"][
            candidate
        ]
        gripper = analysis["seam"]["seam_gripper_abs"]["contrasts_vs_unconditioned"][
            candidate
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError("recomputed analysis is missing contrast statistics") from exc
    return {
        "id": candidate,
        "label": METHOD_LABELS[candidate] + " vs unconditioned",
        "pairs": _strict_int(success.get("pairs"), candidate + ".pairs", minimum=1),
        "rateDifference": _strict_number(
            success.get("rate_difference"), candidate + ".rate_difference"
        ),
        "successCiLow": _strict_number(
            success.get("task_block_bootstrap95_mean_low"),
            candidate + ".success_ci_low",
        ),
        "successCiHigh": _strict_number(
            success.get("task_block_bootstrap95_mean_high"),
            candidate + ".success_ci_high",
        ),
        "wins": _strict_int(success.get("candidate_wins"), candidate + ".wins"),
        "losses": _strict_int(success.get("candidate_losses"), candidate + ".losses"),
        "ties": _strict_int(success.get("ties"), candidate + ".ties"),
        "rawP": _strict_number(
            success.get("mcnemar_exact_two_sided_p"), candidate + ".raw_p"
        ),
        "holmP": _strict_number(success.get("holm_adjusted_p"), candidate + ".holm_p"),
        "motionDifference": _strict_number(
            motion.get("paired_mean_difference"), candidate + ".motion_difference"
        ),
        "motionCiLow": _strict_number(
            motion.get("task_block_bootstrap95_mean_low"), candidate + ".motion_ci_low"
        ),
        "motionCiHigh": _strict_number(
            motion.get("task_block_bootstrap95_mean_high"),
            candidate + ".motion_ci_high",
        ),
        "gripperDifference": _strict_number(
            gripper.get("paired_mean_difference"), candidate + ".gripper_difference"
        ),
        "gripperCiLow": _strict_number(
            gripper.get("task_block_bootstrap95_mean_low"),
            candidate + ".gripper_ci_low",
        ),
        "gripperCiHigh": _strict_number(
            gripper.get("task_block_bootstrap95_mean_high"),
            candidate + ".gripper_ci_high",
        ),
    }


def _validate_bootstrap_triplet_identity(
    evaluation_root: pathlib.Path, expected_pair_ids: Sequence[str]
) -> None:
    queries = _read_strict_json(evaluation_root / "queries.json", list)
    expected_keys = {
        (pair_id, method) for pair_id in expected_pair_ids for method in METHODS
    }
    bootstraps: Dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, raw in enumerate(queries):
        if not isinstance(raw, Mapping):
            raise ValueError("queries[%d] must be an object" % index)
        query_index = raw.get("query_index")
        if isinstance(query_index, bool) or not isinstance(query_index, int):
            raise ValueError("queries[%d].query_index must be an integer" % index)
        if query_index != 0:
            continue
        pair_id = raw.get("pair_id")
        method = raw.get("method")
        key = (pair_id, method)
        if (
            not isinstance(pair_id, str)
            or method not in METHODS
            or key not in expected_keys
        ):
            raise ValueError("queries[%d] has an unexpected bootstrap identity" % index)
        if key in bootstraps:
            raise ValueError("duplicate bootstrap query for %s/%s" % key)
        if raw.get("bootstrap") is not True:
            raise ValueError("query zero must be marked bootstrap for %s/%s" % key)
        for field in BOOTSTRAP_IDENTITY_FIELDS:
            _require_sha256(raw.get(field), "%s/%s.%s" % (pair_id, method, field))
        bootstraps[key] = raw
    if set(bootstraps) != expected_keys:
        missing = sorted(expected_keys.difference(bootstraps))
        raise ValueError(
            "bootstrap query-zero matrix is incomplete: missing %s"
            % ", ".join("%s/%s" % key for key in missing)
        )
    for pair_id in expected_pair_ids:
        for field in BOOTSTRAP_IDENTITY_FIELDS:
            values = {bootstraps[(pair_id, method)][field] for method in METHODS}
            if len(values) != 1:
                raise ValueError(
                    "triplet %s bootstrap %s differs across methods" % (pair_id, field)
                )


def _build_triplets(
    evaluation_root: pathlib.Path,
    output: pathlib.Path,
    rows: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    if len(rows) != 20:
        raise ValueError("recomputed analysis must contain exactly 20 triplets")
    expected_pair_ids = []
    for row in rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or pair_id in expected_pair_ids:
            raise ValueError("recomputed rows contain a duplicate or invalid pair_id")
        expected_pair_ids.append(pair_id)
    _validate_bootstrap_triplet_identity(evaluation_root, expected_pair_ids)

    episodes_json = _read_strict_json(evaluation_root / "episodes.json", list)
    if len(episodes_json) != 60:
        raise ValueError("dashboard requires exactly 60 source episodes")
    episodes: Dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, episode in enumerate(episodes_json):
        if not isinstance(episode, Mapping):
            raise ValueError("episodes[%d] must be an object" % index)
        pair_id = episode.get("pair_id")
        method = episode.get("method")
        if not isinstance(pair_id, str) or method not in METHODS:
            raise ValueError("episodes[%d] has invalid triplet identity" % index)
        key = (pair_id, str(method))
        if key in episodes:
            raise ValueError("duplicate source episode: %s/%s" % key)
        episodes[key] = episode

    triplets: list[Dict[str, Any]] = []
    video_paths = set()
    pair_ids = set()
    for row_index, row in enumerate(rows):
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or pair_id in pair_ids:
            raise ValueError("recomputed rows contain a duplicate or invalid pair_id")
        pair_ids.add(pair_id)
        task_id = _strict_int(row.get("task_id"), pair_id + ".task_id")
        episode_index = _strict_int(
            row.get("episode_index"), pair_id + ".episode_index"
        )
        if task_id != row_index // 2 or episode_index != row_index % 2:
            raise ValueError("recomputed triplets are not in frozen task/episode order")
        method_records = []
        descriptions = set()
        state_hashes = set()
        for method in METHODS:
            key = (pair_id, method)
            if key not in episodes:
                raise ValueError("triplet %s is missing method %s" % key)
            episode = episodes[key]
            for field, expected in (
                ("task_suite", rtc_overlap_analysis.TASK_SUITE),
                ("task_id", task_id),
                ("episode_index", episode_index),
            ):
                if episode.get(field) != expected:
                    raise ValueError("source episode %s disagrees on %s" % (key, field))
            success = episode.get("success")
            if not isinstance(success, bool):
                raise ValueError("source episode %s success must be boolean" % (key,))
            if int(success) != row.get("%s_success" % method):
                raise ValueError(
                    "source episode %s success disagrees with analysis" % (key,)
                )
            queries = _strict_int(
                episode.get("policy_queries"), "%s.%s.policy_queries" % key, minimum=1
            )
            if queries != row.get("%s_policy_queries" % method):
                raise ValueError(
                    "source episode %s query count disagrees with analysis" % (key,)
                )
            order = _strict_int(episode.get("condition_order"), "%s.%s.order" % key)
            if order != row.get("%s_condition_order" % method):
                raise ValueError(
                    "source episode %s order disagrees with analysis" % (key,)
                )
            description = episode.get("task_description")
            if not isinstance(description, str) or not description:
                raise ValueError("source episode %s task description is empty" % (key,))
            descriptions.add(description)
            state_hashes.add(
                _require_sha256(
                    episode.get("initial_state_sha256"), "%s.state" % pair_id
                )
            )
            raw_video = episode.get("video_path")
            if not isinstance(raw_video, str):
                raise ValueError(
                    "source episode %s video path must be a string" % (key,)
                )
            if raw_video in video_paths:
                raise ValueError("source episodes reuse a video path: %s" % raw_video)
            video_paths.add(raw_video)
            method_records.append(
                {
                    "id": method,
                    "label": METHOD_LABELS[method],
                    "success": success,
                    "policyQueries": queries,
                    "scoredQueries": _strict_int(
                        row.get("%s_scored_transition_queries" % method),
                        "%s.%s.scored_queries" % key,
                        minimum=1,
                    ),
                    "conditionOrder": order,
                    "termination": str(episode.get("termination_reason", "")),
                    "motionSeam": _strict_number(
                        row.get("%s_seam_motion_l2" % method), "%s.%s.motion_seam" % key
                    ),
                    "gripperSeam": _strict_number(
                        row.get("%s_seam_gripper_abs" % method),
                        "%s.%s.gripper_seam" % key,
                    ),
                    "video": _relative_video_url(
                        evaluation_root, output.parent, raw_video
                    ),
                }
            )
        if len(descriptions) != 1 or len(state_hashes) != 1:
            raise ValueError(
                "triplet %s does not share task description and initial state" % pair_id
            )
        triplets.append(
            {
                "pairId": pair_id,
                "taskId": task_id,
                "episodeIndex": episode_index,
                "taskDescription": descriptions.pop(),
                "initialStateSha256": state_hashes.pop(),
                "methods": method_records,
            }
        )
    if len(episodes) != 60 or len(video_paths) != 60:
        raise ValueError("dashboard must bind exactly 60 unique episode videos")
    return triplets


def build_dashboard(
    evaluation_root: pathlib.Path,
    analysis_root: pathlib.Path,
    output: pathlib.Path,
) -> Dict[str, Any]:
    evaluation_root = pathlib.Path(evaluation_root).resolve()
    analysis_root = pathlib.Path(analysis_root).resolve()
    output = pathlib.Path(output).resolve()

    # Independent raw validation is deliberately the first evidence read.
    try:
        raw_summary = rtc_overlap_pilot.validate_artifact(evaluation_root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            "raw RTC pilot failed independent validation: %s" % exc
        ) from exc
    if not isinstance(raw_summary, Mapping):
        raise ValueError("raw RTC validator returned a non-object result")
    try:
        recomputed_analysis, recomputed_rows = rtc_overlap_analysis.analyze_artifact(
            evaluation_root
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            "RTC pilot analysis could not be recomputed: %s" % exc
        ) from exc

    validator_hash = _canonical_sha256(raw_summary)
    source = recomputed_analysis.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("validator_summary_sha256") != validator_hash
    ):
        raise ValueError(
            "raw validator result disagrees with recomputed analysis provenance"
        )
    manifest_record = _validate_analysis_manifest(analysis_root)
    saved_analysis = _read_strict_json(analysis_root / "analysis.json", dict)
    if saved_analysis != recomputed_analysis:
        raise ValueError(
            "saved analysis.json disagrees with a fresh analyze_artifact result"
        )
    if (
        saved_analysis.get("schema_version")
        != rtc_overlap_analysis.ANALYSIS_SCHEMA_VERSION
    ):
        raise ValueError("saved analysis schema mismatch")
    cohort = saved_analysis.get("cohort")
    if not isinstance(cohort, Mapping) or (
        cohort.get("rollouts"),
        cohort.get("paired_triplets"),
        cohort.get("tasks"),
    ) != (60, 20, 10):
        raise ValueError("saved analysis does not describe the frozen 60/20/10 cohort")

    triplets = _build_triplets(evaluation_root, output, recomputed_rows)
    environment = _read_strict_json(evaluation_root / "environment.json", dict)
    attestation = environment.get("server_attestation")
    if not isinstance(attestation, Mapping):
        raise ValueError("source environment has no server attestation")
    implementation = saved_analysis.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ValueError("saved analysis has no implementation provenance")
    payload = {
        "schemaVersion": DASHBOARD_SCHEMA_VERSION,
        "title": "pi0.5 RTC overlap pilot",
        "cohort": {
            "rollouts": 60,
            "triplets": 20,
            "tasks": 10,
            "episodesPerTask": 2,
            "queries": _strict_int(cohort.get("queries"), "cohort.queries", minimum=1),
            "executeHorizon": _strict_int(
                cohort.get("execute_horizon"), "cohort.execute_horizon", minimum=1
            ),
            "delaySteps": _strict_int(
                cohort.get("inference_delay_steps"), "cohort.delay_steps", minimum=1
            ),
        },
        "methods": [_method_summary(saved_analysis, method) for method in METHODS],
        "contrasts": [
            _contrast_summary(saved_analysis, candidate) for candidate in CANDIDATES
        ],
        "triplets": triplets,
        "claimBoundary": str(saved_analysis.get("claim_boundary", "")),
        "statistics": {
            "successInterval": str(
                saved_analysis.get("statistics", {}).get("success_rate_interval", "")
            ),
            "successTest": str(
                saved_analysis.get("statistics", {}).get("paired_success_test", "")
            ),
            "bootstrapUnit": str(
                saved_analysis.get("statistics", {}).get("bootstrap_unit", "")
            ),
            "bootstrapResamples": _strict_int(
                saved_analysis.get("statistics", {}).get("bootstrap_resamples"),
                "statistics.bootstrap_resamples",
                minimum=1,
            ),
            "multiplicity": str(
                saved_analysis.get("statistics", {}).get("multiplicity_adjustment", "")
            ),
        },
        "provenance": {
            "artifact": str(source.get("artifact", "")),
            "sourceSchema": str(source.get("source_schema_version", "")),
            "sourceManifestSha256": _require_sha256(
                source.get("manifest_sha256"), "source.manifest_sha256"
            ),
            "episodesSha256": _require_sha256(
                source.get("episodes_sha256"), "source.episodes_sha256"
            ),
            "queriesSha256": _require_sha256(
                source.get("queries_sha256"), "source.queries_sha256"
            ),
            "validatorSummarySha256": validator_hash,
            "analysisManifestSha256": manifest_record["manifestSha256"],
            "analysisJsonSha256": manifest_record["files"]["analysis.json"]["sha256"],
            "policyConfig": str(attestation.get("policy_config", "")),
            "checkpointUri": str(attestation.get("checkpoint_uri", "")),
            "checkpointSha256": _require_sha256(
                attestation.get("checkpoint_content_sha256"),
                "server_attestation.checkpoint_content_sha256",
            ),
            "armbenchCommit": str(environment.get("armbench_commit", "")),
            "openpiCommit": str(attestation.get("openpi_commit", "")),
            "analyzerSource": str(implementation.get("analyzer_source", "")),
            "analyzerSha256": _require_sha256(
                implementation.get("analyzer_sha256"), "implementation.analyzer_sha256"
            ),
            "validatorSource": str(implementation.get("validator_source", "")),
            "validatorSha256": _require_sha256(
                implementation.get("validator_sha256"),
                "implementation.validator_sha256",
            ),
        },
    }
    if not payload["claimBoundary"]:
        raise ValueError("saved analysis claim boundary is empty")

    serialized = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    )
    serialized = (
        serialized.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    html = HTML_TEMPLATE.replace("__ARMBENCH_DATA__", serialized)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[pathlib.Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=output.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(html)
            temporary = pathlib.Path(handle.name)
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return {
        "output": str(output),
        "rollouts": 60,
        "triplets": 20,
        "videos_verified": 60,
        "tasks": 10,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>ArmBench RTC Pilot Acceptance</title>
  <style>
    :root {
      color-scheme: light dark;
      --page: #f3f6f7; --surface: #ffffff; --surface-2: #e7edef;
      --ink: #18242a; --muted: #55656d; --line: #c5d0d4;
      --header: #142127; --header-ink: #f7fafb; --baseline: #b7591b;
      --projected: #08747d; --rtc: #247a49; --failure: #b42318;
      --selected: #1769aa; --focus: #6547d7; --warning: #8a5a08;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --page: #101619; --surface: #192227; --surface-2: #253137;
        --ink: #edf3f5; --muted: #b4c0c6; --line: #425158;
        --header: #080d0f; --baseline: #efa260; --projected: #55c0c7;
        --rtc: #69cf92; --failure: #ff8a80; --selected: #66aef0;
        --focus: #b7a5ff; --warning: #f2c66d;
      }
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body { margin: 0; min-width: 320px; background: var(--page); color: var(--ink); font-family: Inter, "Segoe UI", Arial, sans-serif; font-size: 15px; line-height: 1.45; }
    button, select { min-height: 40px; font: inherit; }
    button:focus-visible, select:focus-visible, video:focus-visible, summary:focus-visible { outline: 3px solid var(--focus); outline-offset: 2px; }
    button:hover { border-color: var(--selected); }
    button:active { background: var(--surface-2); }
    .top { background: var(--header); color: var(--header-ink); }
    .top-inner, main { width: min(1500px, calc(100% - 32px)); margin: 0 auto; }
    .top-inner { padding: 24px 0 22px; }
    h1 { margin: 0; font-size: 28px; font-weight: 620; letter-spacing: 0; text-wrap: balance; }
    .subtitle { max-width: 75ch; margin: 7px 0 0; color: #d4e0e4; text-wrap: pretty; }
    main { padding: 18px 0 40px; }
    section { padding: 22px 0; border-bottom: 1px solid var(--line); }
    h2 { margin: 0 0 12px; font-size: 19px; font-weight: 650; letter-spacing: 0; text-wrap: balance; }
    h3 { margin: 0; font-size: 16px; font-weight: 650; letter-spacing: 0; }
    .summary-strip { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); margin: 0; border-block: 1px solid var(--line); background: var(--surface); }
    .summary-strip div { min-width: 0; padding: 13px 14px; }
    .summary-strip div + div { border-left: 1px solid var(--line); }
    dt { color: var(--muted); font-size: 12px; }
    dd { margin: 3px 0 0; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
    .summary-strip dd { font-size: 18px; font-weight: 630; }
    .table-wrap { overflow-x: auto; border-block: 1px solid var(--line); background: var(--surface); }
    table { width: 100%; min-width: 960px; border-collapse: collapse; }
    th, td { padding: 10px 11px; border-bottom: 1px solid var(--line); text-align: left; font-variant-numeric: tabular-nums; }
    th { color: var(--muted); font-size: 12px; font-weight: 650; }
    tbody tr:last-child td { border-bottom: 0; }
    .method-name { font-weight: 650; }
    .method-name.baseline { color: var(--baseline); }
    .method-name.projected { color: var(--projected); }
    .method-name.rtc { color: var(--rtc); }
    .note { max-width: 78ch; margin: 8px 0 0; color: var(--muted); font-size: 13px; text-wrap: pretty; }
    .controls { display: flex; flex-wrap: wrap; align-items: end; gap: 12px; padding: 12px 0; border-block: 1px solid var(--line); }
    .field { display: grid; gap: 4px; }
    .field label { color: var(--muted); font-size: 12px; }
    select { min-width: 165px; padding: 7px 32px 7px 9px; color: var(--ink); background: var(--surface); border: 1px solid var(--line); border-radius: 4px; }
    .triplet-count { margin-left: auto; color: var(--muted); font-variant-numeric: tabular-nums; }
    .triplet-grid { display: grid; grid-template-columns: repeat(10, minmax(78px, 1fr)); gap: 6px; margin-top: 12px; }
    .triplet-button { border: 1px solid var(--line); border-radius: 4px; color: var(--ink); background: var(--surface); cursor: pointer; font-size: 13px; }
    .triplet-button[data-failures="1"] { color: var(--failure); }
    .triplet-button[aria-pressed="true"] { border-color: var(--selected); box-shadow: inset 0 0 0 2px var(--selected); }
    .selection { margin-top: 18px; border-block: 1px solid var(--line); background: var(--surface); }
    .selection-head { display: flex; justify-content: space-between; align-items: start; gap: 16px; padding: 14px 0; margin: 0 14px; border-bottom: 1px solid var(--line); }
    .description { max-width: 75ch; margin: 5px 0 0; color: var(--muted); overflow-wrap: anywhere; }
    .state-hash { max-width: 34ch; margin: 0; color: var(--muted); font: 12px/1.4 Consolas, monospace; overflow-wrap: anywhere; text-align: right; }
    .videos { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .video-pane { min-width: 0; padding: 14px; }
    .video-pane + .video-pane { border-left: 1px solid var(--line); }
    .video-pane.baseline h3 { color: var(--baseline); }
    .video-pane.projected h3 { color: var(--projected); }
    .video-pane.rtc h3 { color: var(--rtc); }
    video { display: block; width: 100%; aspect-ratio: 1 / 1; margin-top: 9px; background: #050607; object-fit: contain; }
    .rollout-meta { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 7px 12px; margin: 10px 0 0; }
    .rollout-meta dd { font-size: 13px; }
    .success { color: var(--rtc); font-weight: 650; }
    .failure { color: var(--failure); font-weight: 650; }
    .playback { display: flex; flex-wrap: wrap; gap: 7px; padding: 0 14px 14px; }
    .command { padding: 7px 12px; border: 1px solid var(--line); border-radius: 4px; color: var(--ink); background: var(--surface); cursor: pointer; }
    .claim { padding: 15px 0; color: var(--warning); font-weight: 560; text-wrap: pretty; }
    details { padding: 4px 0 20px; }
    summary { cursor: pointer; font-weight: 650; }
    .provenance { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 9px 22px; margin-top: 13px; }
    .provenance dd { font: 12px/1.45 Consolas, monospace; }
    @media (max-width: 1050px) { .triplet-grid { grid-template-columns: repeat(5, minmax(78px, 1fr)); } .videos { grid-template-columns: 1fr; } .video-pane + .video-pane { border-left: 0; border-top: 1px solid var(--line); } video { max-height: 70vh; } }
    @media (max-width: 700px) { .summary-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); } .summary-strip div:nth-child(3) { border-left: 0; border-top: 1px solid var(--line); } .summary-strip div:nth-child(4) { border-top: 1px solid var(--line); } .triplet-grid { grid-template-columns: repeat(4, minmax(62px, 1fr)); } .selection-head { display: grid; } .state-hash { max-width: none; text-align: left; } .provenance { grid-template-columns: 1fr; } }
    @media (max-width: 480px) { .top-inner, main { width: min(100% - 20px, 1500px); } h1 { font-size: 23px; } .summary-strip { grid-template-columns: 1fr; } .summary-strip div + div { border-left: 0; border-top: 1px solid var(--line); } .controls, .field, select { width: 100%; } .triplet-count { margin-left: 0; } .triplet-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } .rollout-meta { grid-template-columns: 1fr; } }
    @media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } *, *::before, *::after { animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; transition-duration: 0.01ms !important; } }
  </style>
</head>
<body>
  <header class="top"><div class="top-inner">
    <h1>pi0.5 RTC Overlap Pilot</h1>
    <p class="subtitle">Offline acceptance view for the frozen LIBERO-10 three-arm comparison: unconditioned overlap, hard projection, and reverse-time RTC guidance.</p>
  </div></header>
  <main>
    <section aria-labelledby="overview-title"><h2 id="overview-title">Evidence scope</h2><dl class="summary-strip" id="summary-strip"></dl></section>
    <section aria-labelledby="methods-title"><h2 id="methods-title">Method-level statistics</h2><div class="table-wrap"><table><thead><tr><th>Method</th><th>Success (Wilson 95%)</th><th>Motion seam mean / median</th><th>Gripper seam mean / median</th><th>Scored transitions</th></tr></thead><tbody id="methods-body"></tbody></table></div><p class="note">Seam statistics use the episode mean after per-query aggregation.</p></section>
    <section aria-labelledby="contrast-title"><h2 id="contrast-title">Prespecified contrasts</h2><div class="table-wrap"><table><thead><tr><th>Contrast</th><th>Success difference (task-block 95%)</th><th>Wins / losses / ties</th><th>McNemar p / Holm p</th><th>Motion seam difference (95%)</th><th>Gripper seam difference (95%)</th></tr></thead><tbody id="contrast-body"></tbody></table></div><p class="note" id="statistics-note"></p></section>
    <section aria-labelledby="review-title"><h2 id="review-title">Triplet review</h2><div class="controls"><div class="field"><label for="task-select">Task</label><select id="task-select" aria-label="Select LIBERO task"></select></div><div class="field"><label for="episode-select">Initial state</label><select id="episode-select" aria-label="Select initial-state episode"></select></div><span class="triplet-count">20 / 20 triplets, 60 videos</span></div><div class="triplet-grid" id="triplet-grid" aria-label="All 20 task and initial-state triplets"></div><div class="selection" id="selection"><div class="selection-head"><div><h3 id="selection-id"></h3><p class="description" id="task-description"></p></div><p class="state-hash" id="state-hash"></p></div><div class="videos" id="videos"></div><div class="playback"><button class="command" id="play-all" type="button" aria-label="Play all three videos">Play all</button><button class="command" id="pause-all" type="button" aria-label="Pause all three videos">Pause all</button><button class="command" id="restart-all" type="button" aria-label="Restart all three videos">Restart all</button></div></div></section>
    <section aria-labelledby="boundary-title"><h2 id="boundary-title">Claim boundary</h2><p class="claim" id="claim-boundary"></p></section>
    <details><summary>Evidence provenance and integrity</summary><dl class="provenance" id="provenance"></dl></details>
  </main>
  <script id="armbench-data" type="application/json">__ARMBENCH_DATA__</script>
  <script>
    (function () {
      "use strict";
      const data = JSON.parse(document.getElementById("armbench-data").textContent);
      const byId = (id) => document.getElementById(id);
      const pct = (value) => (value * 100).toFixed(1) + "%";
      const signed = (value, digits) => (value >= 0 ? "+" : "") + value.toFixed(digits);
      const interval = (low, high, digits) => "[" + signed(low, digits) + ", " + signed(high, digits) + "]";
      const pvalue = (value) => value < 0.001 ? value.toExponential(2) : value.toFixed(4);
      const classes = {overlap_unconditioned: "baseline", projected_overlap: "projected", rtc_guided_overlap: "rtc"};
      const state = {taskId: 0, episodeIndex: 0};

      function addDefinition(root, label, value) { const wrapper = document.createElement("div"); const dt = document.createElement("dt"); const dd = document.createElement("dd"); dt.textContent = label; dd.textContent = value; wrapper.append(dt, dd); root.appendChild(wrapper); }
      const cohort = data.cohort;
      [["Frozen matrix", cohort.rollouts + " rollouts"], ["Matched design", cohort.triplets + " triplets"], ["Task coverage", cohort.tasks + " LIBERO-10 tasks"], ["Temporal protocol", "H=10, E=" + cohort.executeHorizon + ", d=" + cohort.delaySteps]].forEach((item) => addDefinition(byId("summary-strip"), item[0], item[1]));

      data.methods.forEach((method) => { const row = document.createElement("tr"); const values = [method.label, method.successes + "/" + method.rollouts + " (" + pct(method.rate) + ") [" + pct(method.wilsonLow) + ", " + pct(method.wilsonHigh) + "]", method.meanMotionSeam.toFixed(5) + " / " + method.medianMotionSeam.toFixed(5), method.meanGripperSeam.toFixed(5) + " / " + method.medianGripperSeam.toFixed(5), String(method.scoredTransitions)]; values.forEach((value, index) => { const cell = document.createElement("td"); cell.textContent = value; if (index === 0) cell.className = "method-name " + classes[method.id]; row.appendChild(cell); }); byId("methods-body").appendChild(row); });
      data.contrasts.forEach((contrast) => { const row = document.createElement("tr"); const values = [contrast.label, signed(contrast.rateDifference * 100, 1) + " points " + interval(contrast.successCiLow * 100, contrast.successCiHigh * 100, 1), contrast.wins + " / " + contrast.losses + " / " + contrast.ties, pvalue(contrast.rawP) + " / " + pvalue(contrast.holmP), signed(contrast.motionDifference, 5) + " " + interval(contrast.motionCiLow, contrast.motionCiHigh, 5), signed(contrast.gripperDifference, 5) + " " + interval(contrast.gripperCiLow, contrast.gripperCiHigh, 5)]; values.forEach((value, index) => { const cell = document.createElement("td"); cell.textContent = value; if (index === 0) cell.className = "method-name " + classes[contrast.id]; row.appendChild(cell); }); byId("contrast-body").appendChild(row); });
      byId("statistics-note").textContent = data.statistics.successTest + "; " + data.statistics.multiplicity + ". Bootstrap unit: " + data.statistics.bootstrapUnit + " (" + data.statistics.bootstrapResamples + " resamples).";

      const taskSelect = byId("task-select"); for (let task = 0; task < 10; task += 1) { const option = document.createElement("option"); option.value = String(task); option.textContent = "Task " + task; taskSelect.appendChild(option); }
      const episodeSelect = byId("episode-select"); for (let episode = 0; episode < 2; episode += 1) { const option = document.createElement("option"); option.value = String(episode); option.textContent = "Episode " + episode; episodeSelect.appendChild(option); }
      function selectedTriplet() { return data.triplets.find((triplet) => triplet.taskId === state.taskId && triplet.episodeIndex === state.episodeIndex); }
      function setSelection(taskId, episodeIndex) { state.taskId = taskId; state.episodeIndex = episodeIndex; taskSelect.value = String(taskId); episodeSelect.value = String(episodeIndex); renderSelection(); }
      data.triplets.forEach((triplet) => { const button = document.createElement("button"); button.type = "button"; button.className = "triplet-button"; button.dataset.pairId = triplet.pairId; button.dataset.failures = triplet.methods.some((method) => !method.success) ? "1" : "0"; button.textContent = "T" + triplet.taskId + " / E" + triplet.episodeIndex; button.setAttribute("aria-label", "Task " + triplet.taskId + ", episode " + triplet.episodeIndex + ", " + triplet.methods.filter((method) => method.success).length + " of 3 methods succeeded"); button.setAttribute("aria-pressed", "false"); button.addEventListener("click", () => setSelection(triplet.taskId, triplet.episodeIndex)); byId("triplet-grid").appendChild(button); });
      taskSelect.addEventListener("change", () => setSelection(Number(taskSelect.value), state.episodeIndex));
      episodeSelect.addEventListener("change", () => setSelection(state.taskId, Number(episodeSelect.value)));

      function renderSelection() { const triplet = selectedTriplet(); if (!triplet) return; byId("selection-id").textContent = "LIBERO-10 / Task " + triplet.taskId + " / Episode " + triplet.episodeIndex; byId("task-description").textContent = triplet.taskDescription; byId("state-hash").textContent = "Initial state SHA-256: " + triplet.initialStateSha256; const videosRoot = byId("videos"); videosRoot.replaceChildren(); triplet.methods.forEach((method) => { const pane = document.createElement("article"); pane.className = "video-pane " + classes[method.id]; const heading = document.createElement("h3"); heading.textContent = method.label; const video = document.createElement("video"); video.controls = true; video.preload = "metadata"; video.src = method.video; video.setAttribute("aria-label", method.label + " rollout for task " + triplet.taskId + ", episode " + triplet.episodeIndex); const meta = document.createElement("dl"); meta.className = "rollout-meta"; addDefinition(meta, "Outcome", method.success ? "Success" : "Failure"); meta.lastElementChild.lastElementChild.className = method.success ? "success" : "failure"; addDefinition(meta, "Queries", method.policyQueries + " policy / " + method.scoredQueries + " scored"); addDefinition(meta, "Episode seam", "motion " + method.motionSeam.toFixed(5) + " / gripper " + method.gripperSeam.toFixed(5)); addDefinition(meta, "Order", (method.conditionOrder + 1) + " of 3 / " + method.termination); pane.append(heading, video, meta); videosRoot.appendChild(pane); }); byId("triplet-grid").querySelectorAll("button").forEach((button) => button.setAttribute("aria-pressed", button.dataset.pairId === triplet.pairId ? "true" : "false")); }
      const currentVideos = () => Array.from(byId("videos").querySelectorAll("video"));
      byId("play-all").addEventListener("click", () => { const videos = currentVideos(); const start = Math.min.apply(null, videos.map((video) => Number.isFinite(video.currentTime) ? video.currentTime : 0)); videos.forEach((video) => { video.currentTime = start; }); Promise.allSettled(videos.map((video) => video.play())); });
      byId("pause-all").addEventListener("click", () => currentVideos().forEach((video) => video.pause()));
      byId("restart-all").addEventListener("click", () => currentVideos().forEach((video) => { video.pause(); video.currentTime = 0; }));
      byId("claim-boundary").textContent = data.claimBoundary;
      Object.entries(data.provenance).forEach((entry) => addDefinition(byId("provenance"), entry[0], String(entry[1] || "-")));
      renderSelection();
    }());
  </script>
</body>
</html>
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--evaluation-root", type=pathlib.Path, default=DEFAULT_EVALUATION_ROOT
    )
    parser.add_argument(
        "--analysis-root", type=pathlib.Path, default=DEFAULT_ANALYSIS_ROOT
    )
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--open", action="store_true", help="open the generated offline page"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_dashboard(args.evaluation_root, args.analysis_root, args.output)
    except (OSError, ValueError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps({"valid": True, **result}, ensure_ascii=True, indent=2))
    if args.open:
        webbrowser.open(pathlib.Path(result["output"]).as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
