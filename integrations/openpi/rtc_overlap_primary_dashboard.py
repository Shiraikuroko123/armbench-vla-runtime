"""Build an offline acceptance dashboard for corrected RTC-overlap evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import pathlib
import tempfile
import webbrowser
from typing import Any, Dict, Mapping, Optional, Sequence
from urllib.parse import quote

from integrations.openpi import rtc_overlap_pilot
from integrations.openpi import rtc_overlap_primary_analysis as primary_analysis


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_SEED06_ROOT = (
    PROJECT_ROOT
    / "evidence"
    / "pi05_rtc_overlap_primary_v3_seed_20260806_001"
    / "evaluation"
)
DEFAULT_SEED07_ROOT = (
    PROJECT_ROOT
    / "evidence"
    / "pi05_rtc_overlap_primary_v3_seed_20260807_001"
    / "evaluation"
)
DEFAULT_ANALYSIS_ROOT = (
    PROJECT_ROOT / "evidence" / "pi05_rtc_overlap_primary_v3_300_001" / "analysis"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "reports" / "pi05_rtc_overlap_primary_v3_300_001" / "index.html"
)

DASHBOARD_SCHEMA_VERSION = "armbench.pi05_rtc_overlap_primary_dashboard.v1"
METHODS = tuple(primary_analysis.METHODS)
BASELINE = primary_analysis.BASELINE
CANDIDATES = tuple(primary_analysis.CANDIDATES)
METHOD_LABELS = {
    rtc_overlap_pilot.OVERLAP_UNCONDITIONED: "Unconditioned overlap",
    rtc_overlap_pilot.PROJECTED_OVERLAP: "Hard projection",
    rtc_overlap_pilot.RTC_GUIDED_OVERLAP: "RTC guidance",
}
METHOD_CLASSES = {
    rtc_overlap_pilot.OVERLAP_UNCONDITIONED: "baseline",
    rtc_overlap_pilot.PROJECTED_OVERLAP: "projected",
    rtc_overlap_pilot.RTC_GUIDED_OVERLAP: "rtc",
}
ANALYSIS_FILES = frozenset(
    ("analysis.json", "per_triplet.csv", "summary.md", "manifest.json")
)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: pathlib.Path, expected: type) -> Any:
    def reject(value: str) -> Any:
        raise ValueError("non-finite JSON constant: %s" % value)

    def unique(pairs: Sequence[tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: %s" % key)
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_bytes().decode("utf-8"),
            parse_constant=reject,
            object_pairs_hook=unique,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("cannot read strict JSON %s: %s" % (path, exc)) from exc
    if not isinstance(value, expected):
        raise ValueError("%s must contain a %s" % (path, expected.__name__))
    _check_finite(value, path.name)
    return value


def _check_finite(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("%s contains a non-finite value" % label)
    if isinstance(value, Mapping):
        for key, item in value.items():
            _check_finite(item, "%s.%s" % (label, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite(item, "%s[%d]" % (label, index))


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _canonical_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=primary_analysis._PER_TRIPLET_FIELDS,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("%s must be a canonical SHA-256" % label)
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("%s must be an integer >= %d" % (label, minimum))
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be numeric" % label)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("%s must be finite" % label)
    return result


def _relative_video_url(
    evaluation_root: pathlib.Path, output_parent: pathlib.Path, raw_path: str
) -> str:
    pure = pathlib.PurePosixPath(raw_path)
    if (
        not raw_path
        or "\\" in raw_path
        or pure.is_absolute()
        or pure.as_posix() != raw_path
        or ".." in pure.parts
        or pure.parts[:1] != ("videos",)
        or pure.suffix.lower() != ".mp4"
    ):
        raise ValueError("invalid video path: %s" % raw_path)
    source_root = evaluation_root.resolve()
    video = (evaluation_root / pure).resolve()
    try:
        video.relative_to(source_root)
    except ValueError as exc:
        raise ValueError("video path escapes evaluation root: %s" % raw_path) from exc
    if not video.is_file() or video.stat().st_size <= 0:
        raise ValueError("video is missing or empty: %s" % video)
    try:
        relative = pathlib.Path(os.path.relpath(video, output_parent.resolve()))
    except ValueError as exc:
        raise ValueError("dashboard and video must be on the same filesystem") from exc
    url = quote(relative.as_posix(), safe="/")
    if pathlib.PurePosixPath(relative.as_posix()).is_absolute() or "://" in url:
        raise ValueError("video URL must remain relative")
    return url


def _expected_analysis_files(analysis_root: pathlib.Path) -> None:
    actual = {path.name for path in analysis_root.iterdir() if path.is_file()}
    if actual != ANALYSIS_FILES:
        raise ValueError("primary analysis file inventory is unexpected")


def _validate_and_rebuild(
    seed06_root: pathlib.Path,
    seed07_root: pathlib.Path,
    analysis_root: pathlib.Path,
) -> tuple[Mapping[str, Any], Sequence[Mapping[str, Any]], Mapping[str, Any]]:
    seed06_root = seed06_root.resolve()
    seed07_root = seed07_root.resolve()
    analysis_root = analysis_root.resolve()
    if any(path.is_symlink() or not path.is_dir() for path in (seed06_root, seed07_root)):
        raise ValueError("raw evaluation roots must be regular directories")
    if analysis_root.is_symlink() or not analysis_root.is_dir():
        raise ValueError("analysis root must be a regular directory")

    # These checks include the corrected-v3 query-zero input/response/key/noise gate.
    for root in (seed06_root, seed07_root):
        try:
            result = rtc_overlap_pilot.validate_artifact(root)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError("raw evidence failed v3 validation: %s" % exc) from exc
        if not isinstance(result, Mapping) or result.get("complete_triplets") != 50:
            raise ValueError("raw evidence does not contain 50 validated triplets")

    manifest_report = primary_analysis.validate_analysis_manifest(analysis_root)
    if manifest_report.get("valid") is not True:
        raise ValueError(
            "saved primary analysis manifest is invalid: %s"
            % "; ".join(str(item) for item in manifest_report.get("errors", []))
        )
    _expected_analysis_files(analysis_root)

    try:
        recomputed_analysis, recomputed_rows = primary_analysis.analyze_artifacts(
            (seed06_root, seed07_root)
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("primary analysis could not be rebuilt: %s" % exc) from exc

    saved_analysis_path = analysis_root / "analysis.json"
    saved_analysis = _strict_json(saved_analysis_path, dict)
    if saved_analysis != recomputed_analysis:
        raise ValueError("saved analysis.json disagrees with a fresh primary rebuild")
    if saved_analysis_path.read_bytes() != _canonical_json(recomputed_analysis):
        raise ValueError("saved analysis.json does not use the canonical encoding")
    if (analysis_root / "per_triplet.csv").read_bytes() != _canonical_csv(
        recomputed_rows
    ):
        raise ValueError("saved per_triplet.csv disagrees with a fresh primary rebuild")
    expected_summary = primary_analysis._summary_markdown(recomputed_analysis).encode(
        "utf-8"
    )
    if (analysis_root / "summary.md").read_bytes() != expected_summary:
        raise ValueError("saved summary.md disagrees with a fresh primary rebuild")

    cohort = recomputed_analysis.get("cohort")
    if not isinstance(cohort, Mapping) or (
        cohort.get("rollouts"),
        cohort.get("matched_triplets"),
        cohort.get("tasks"),
    ) != (300, 100, 10):
        raise ValueError("recomputed analysis is not the frozen 300/100/10 cohort")
    return recomputed_analysis, recomputed_rows, manifest_report


def _method_summary(analysis: Mapping[str, Any], method: str) -> Dict[str, Any]:
    try:
        success = analysis["success"]["methods"][method]
        motion = analysis["seam"]["seam_motion_l2"]["methods"][method]
        gripper = analysis["seam"]["seam_gripper_abs"]["methods"][method]
    except (KeyError, TypeError) as exc:
        raise ValueError("analysis is missing method statistics") from exc
    return {
        "id": method,
        "label": METHOD_LABELS[method],
        "successes": _integer(success.get("successes"), method + ".successes"),
        "rollouts": _integer(success.get("rollouts"), method + ".rollouts", minimum=1),
        "rate": _number(success.get("rate"), method + ".rate"),
        "wilsonLow": _number(success.get("wilson95_low"), method + ".wilson_low"),
        "wilsonHigh": _number(
            success.get("wilson95_high"), method + ".wilson_high"
        ),
        "motionMean": _number(motion.get("mean"), method + ".motion_mean"),
        "motionMedian": _number(motion.get("median"), method + ".motion_median"),
        "gripperMean": _number(
            gripper.get("mean"), method + ".gripper_mean"
        ),
        "gripperMedian": _number(
            gripper.get("median"), method + ".gripper_median"
        ),
        "scoredTransitions": _integer(
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
        raise ValueError("analysis is missing contrast statistics") from exc
    return {
        "id": candidate,
        "label": METHOD_LABELS[candidate] + " vs unconditioned",
        "riskDifference": _number(
            success.get("risk_difference"), candidate + ".risk_difference"
        ),
        "successCiLow": _number(
            success.get("task_block_bootstrap95_low"), candidate + ".success_ci_low"
        ),
        "successCiHigh": _number(
            success.get("task_block_bootstrap95_high"), candidate + ".success_ci_high"
        ),
        "wins": _integer(success.get("candidate_wins"), candidate + ".wins"),
        "losses": _integer(success.get("candidate_losses"), candidate + ".losses"),
        "ties": _integer(success.get("ties"), candidate + ".ties"),
        "rawP": _number(
            success.get("mcnemar_exact_two_sided_p"), candidate + ".raw_p"
        ),
        "holmP": _number(success.get("holm_adjusted_p"), candidate + ".holm_p"),
        "motionDifference": _number(
            motion.get("paired_episode_mean_difference"), candidate + ".motion_difference"
        ),
        "motionCiLow": _number(
            motion.get("task_block_bootstrap95_low"), candidate + ".motion_ci_low"
        ),
        "motionCiHigh": _number(
            motion.get("task_block_bootstrap95_high"), candidate + ".motion_ci_high"
        ),
        "gripperDifference": _number(
            gripper.get("paired_episode_mean_difference"), candidate + ".gripper_difference"
        ),
        "gripperCiLow": _number(
            gripper.get("task_block_bootstrap95_low"), candidate + ".gripper_ci_low"
        ),
        "gripperCiHigh": _number(
            gripper.get("task_block_bootstrap95_high"), candidate + ".gripper_ci_high"
        ),
        "successImprovementSupported": success.get("success_improvement_supported") is True,
    }


def _episode_lookup(
    seed06_root: pathlib.Path, seed07_root: pathlib.Path
) -> Dict[tuple[int, int, int, str], Mapping[str, Any]]:
    output: Dict[tuple[int, int, int, str], Mapping[str, Any]] = {}
    for root, seed in ((seed06_root, 20260806), (seed07_root, 20260807)):
        episodes = _strict_json(root / "episodes.json", list)
        if len(episodes) != 150:
            raise ValueError("source episodes.json does not contain 150 rows")
        for index, episode in enumerate(episodes):
            if not isinstance(episode, Mapping):
                raise ValueError("source episode %d is not an object" % index)
            task_id = _integer(episode.get("task_id"), "episode.task_id", minimum=0)
            episode_index = _integer(
                episode.get("episode_index"), "episode.episode_index", minimum=0
            )
            method = episode.get("method")
            if method not in METHODS:
                raise ValueError("source episode has an unknown method")
            key = (task_id, episode_index, seed, method)
            if key in output:
                raise ValueError("source episodes contain a duplicate triplet method")
            output[key] = episode
    if len(output) != 300:
        raise ValueError("source episodes do not cover all 300 method rollouts")
    return output


def _build_triplets(
    rows: Sequence[Mapping[str, Any]],
    episodes: Mapping[tuple[int, int, int, str], Mapping[str, Any]],
    seed06_root: pathlib.Path,
    seed07_root: pathlib.Path,
    output: pathlib.Path,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    roots_by_seed = {20260806: seed06_root, 20260807: seed07_root}
    triplets: list[Dict[str, Any]] = []
    clips: list[Dict[str, Any]] = []
    seen_ids = set()
    for row in rows:
        task_id = _integer(row.get("task_id"), "row.task_id", minimum=0)
        episode_index = _integer(row.get("episode_index"), "row.episode_index", minimum=0)
        sampling_seed = _integer(row.get("sampling_seed"), "row.sampling_seed", minimum=1)
        if sampling_seed not in roots_by_seed:
            raise ValueError("analysis row has an unexpected sampling seed")
        triplet_id = row.get("triplet_id")
        pair_id = row.get("pair_id")
        if not isinstance(triplet_id, str) or not isinstance(pair_id, str):
            raise ValueError("analysis row has an invalid triplet identity")
        if triplet_id in seen_ids:
            raise ValueError("analysis rows contain a duplicate triplet")
        seen_ids.add(triplet_id)

        descriptions = set()
        state_hashes = set()
        methods = []
        for method in METHODS:
            episode = episodes[(task_id, episode_index, sampling_seed, method)]
            if episode.get("pair_id") != pair_id:
                raise ValueError("episode pair identity disagrees with analysis")
            if episode.get("success") not in (True, False):
                raise ValueError("episode success must be boolean")
            expected_success = _integer(
                row.get("%s_success" % method), method + ".success", minimum=0
            )
            if int(episode["success"]) != expected_success:
                raise ValueError("episode success disagrees with analysis")
            description = episode.get("task_description")
            if not isinstance(description, str) or not description:
                raise ValueError("episode task description is missing")
            descriptions.add(description)
            state_hashes.add(
                _require_sha256(
                    episode.get("initial_state_sha256"), method + ".initial_state_sha256"
                )
            )
            video_path = episode.get("video_path")
            video_url: Optional[str] = None
            if video_path is not None:
                if episode["success"] is True:
                    raise ValueError("failure-only capture contains a success video")
                if not isinstance(video_path, str):
                    raise ValueError("episode video path is not a string")
                video_url = _relative_video_url(
                    roots_by_seed[sampling_seed], output.parent, video_path
                )
            methods.append(
                {
                    "id": method,
                    "label": METHOD_LABELS[method],
                    "className": METHOD_CLASSES[method],
                    "success": bool(episode["success"]),
                    "policyQueries": _integer(
                        episode.get("policy_queries"), method + ".policy_queries", minimum=1
                    ),
                    "conditionOrder": _integer(
                        episode.get("condition_order"), method + ".condition_order", minimum=0
                    ),
                    "termination": str(episode.get("termination_reason", "")),
                    "motionSeam": _number(
                        row.get("%s_seam_motion_l2" % method), method + ".motion_seam"
                    ),
                    "gripperSeam": _number(
                        row.get("%s_seam_gripper_abs" % method), method + ".gripper_seam"
                    ),
                    "video": video_url,
                }
            )
        if len(descriptions) != 1 or len(state_hashes) != 1:
            raise ValueError("triplet does not share task description and initial state")
        triplet = {
            "id": triplet_id,
            "pairId": pair_id,
            "taskId": task_id,
            "episodeIndex": episode_index,
            "samplingSeed": sampling_seed,
            "taskDescription": descriptions.pop(),
            "initialStateSha256": state_hashes.pop(),
            "methods": methods,
        }
        triplets.append(triplet)
        for method in methods:
            if method["video"] is not None:
                clips.append(
                    {
                        "tripletId": triplet_id,
                        "taskId": task_id,
                        "episodeIndex": episode_index,
                        "samplingSeed": sampling_seed,
                        "method": method["id"],
                        "label": method["label"],
                        "video": method["video"],
                    }
                )
    if len(triplets) != 100:
        raise ValueError("dashboard must bind exactly 100 matched triplets")
    return triplets, clips


def build_dashboard(
    seed06_root: pathlib.Path = DEFAULT_SEED06_ROOT,
    seed07_root: pathlib.Path = DEFAULT_SEED07_ROOT,
    analysis_root: pathlib.Path = DEFAULT_ANALYSIS_ROOT,
    output: pathlib.Path = DEFAULT_OUTPUT,
) -> Dict[str, Any]:
    """Fail closed before generating a view of the corrected primary evidence."""

    seed06_root = pathlib.Path(seed06_root).resolve()
    seed07_root = pathlib.Path(seed07_root).resolve()
    analysis_root = pathlib.Path(analysis_root).resolve()
    output = pathlib.Path(output).resolve()
    if output.is_symlink():
        raise ValueError("dashboard output may not be a symbolic link")

    analysis, rows, manifest_report = _validate_and_rebuild(
        seed06_root, seed07_root, analysis_root
    )
    episodes = _episode_lookup(seed06_root, seed07_root)
    triplets, clips = _build_triplets(
        rows, episodes, seed06_root, seed07_root, output
    )
    if not clips:
        raise ValueError("failure-only evidence contains no video clips")

    cohort = analysis["cohort"]
    frozen_identity = analysis["frozen_identity"]
    sources = analysis.get("sources")
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError("analysis does not bind exactly two raw sources")
    default_triplet = next(
        triplet
        for triplet in triplets
        if any(method["video"] is not None for method in triplet["methods"])
    )
    payload = {
        "schemaVersion": DASHBOARD_SCHEMA_VERSION,
        "cohort": {
            "rollouts": _integer(cohort.get("rollouts"), "cohort.rollouts"),
            "triplets": _integer(
                cohort.get("matched_triplets"), "cohort.matched_triplets"
            ),
            "tasks": _integer(cohort.get("tasks"), "cohort.tasks"),
            "states": _integer(
                cohort.get("initial_states_per_task"), "cohort.initial_states"
            ),
            "seeds": _integer(
                cohort.get("sampling_seeds_per_state"), "cohort.sampling_seeds"
            ),
            "executeHorizon": _integer(
                cohort.get("execute_horizon"), "cohort.execute_horizon", minimum=1
            ),
            "delaySteps": _integer(
                cohort.get("inference_delay_steps"), "cohort.delay_steps", minimum=1
            ),
            "failureClips": len(clips),
        },
        "methods": [_method_summary(analysis, method) for method in METHODS],
        "contrasts": [_contrast_summary(analysis, method) for method in CANDIDATES],
        "triplets": triplets,
        "defaultTripletId": default_triplet["id"],
        "claimBoundary": str(analysis.get("claim_boundary", "")),
        "statistics": {
            "successTest": str(analysis["statistics"].get("paired_success_test", "")),
            "multiplicity": str(
                analysis["statistics"].get("multiplicity_adjustment", "")
            ),
            "bootstrapUnit": str(analysis["statistics"].get("bootstrap_unit", "")),
            "bootstrapResamples": _integer(
                analysis["statistics"].get("bootstrap_resamples"),
                "statistics.bootstrap_resamples",
                minimum=1,
            ),
        },
        "provenance": {
            "runtimeSchema": primary_analysis.SOURCE_SCHEMA_VERSION,
            "frozenEvaluatorCommit": frozen_identity["armbench_commit"],
            "frozenProtocolCommit": frozen_identity[
                "external_held_out_protocol_commit"
            ],
            "policyConfig": frozen_identity["policy_config"],
            "checkpointSha256": _require_sha256(
                frozen_identity["checkpoint_content_sha256"], "checkpoint SHA-256"
            ),
            "analysisManifestSha256": _sha256_file(analysis_root / "manifest.json"),
            "analysisFilesChecked": _integer(
                manifest_report.get("files_checked"), "analysis files checked", minimum=1
            ),
            "sourceManifestSha256": [
                _require_sha256(source.get("manifest_sha256"), "source manifest SHA-256")
                for source in sources
            ],
            "queryZeroFields": "policy input, response action, sampling key, sampling noise",
        },
    }
    if not payload["claimBoundary"]:
        raise ValueError("analysis claim boundary is empty")

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
        "rollouts": 300,
        "triplets": 100,
        "failure_videos_verified": len(clips),
        "tasks": 10,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>ArmBench RTC Primary Evidence</title>
  <style>
    :root { color-scheme: light dark; --page:#f4f5f2; --surface:#ffffff; --surface-alt:#edf1ed; --ink:#18231d; --muted:#52635a; --line:#c7d0c8; --header:#1b3028; --header-ink:#f5faf6; --baseline:#9a521b; --projected:#087b78; --rtc:#31834d; --failure:#b42820; --select:#366fb0; --focus:#7548ad; --warning:#725314; }
    @media (prefers-color-scheme: dark) { :root { --page:#151a16; --surface:#1d2721; --surface-alt:#28342c; --ink:#edf4ef; --muted:#bdcabf; --line:#46564b; --header:#0f2018; --header-ink:#f0faf2; --baseline:#edab72; --projected:#5dc8c0; --rtc:#79d395; --failure:#ff998e; --select:#78aff2; --focus:#c5a7ed; --warning:#f2cf79; } }
    * { box-sizing:border-box; }
    body { margin:0; min-width:320px; background:var(--page); color:var(--ink); font:15px/1.45 Inter,"Segoe UI",Arial,sans-serif; }
    button, select { min-height:40px; font:inherit; }
    button:focus-visible, select:focus-visible, video:focus-visible, summary:focus-visible { outline:3px solid var(--focus); outline-offset:2px; }
    .top { background:var(--header); color:var(--header-ink); }
    .top-inner, main { width:min(1460px, calc(100% - 32px)); margin:0 auto; }
    .top-inner { padding:25px 0 22px; }
    h1, h2, h3 { margin:0; font-weight:600; letter-spacing:0; }
    h1 { font-size:28px; }
    h2 { font-size:19px; margin-bottom:12px; }
    h3 { font-size:16px; }
    .subtitle { max-width:82ch; margin:7px 0 0; color:#d8e5dc; }
    main { padding:18px 0 42px; }
    section { padding:22px 0; border-bottom:1px solid var(--line); }
    .metrics { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); border-block:1px solid var(--line); }
    .metric { min-width:0; padding:12px 14px; background:var(--surface); }
    .metric + .metric { border-left:1px solid var(--line); }
    dt { color:var(--muted); font-size:12px; }
    dd { margin:3px 0 0; overflow-wrap:anywhere; font-variant-numeric:tabular-nums; }
    .metric dd { font-size:19px; font-weight:600; }
    .table-wrap { overflow-x:auto; border-block:1px solid var(--line); background:var(--surface); }
    table { width:100%; min-width:940px; border-collapse:collapse; }
    th, td { padding:10px 11px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; font-variant-numeric:tabular-nums; }
    th { color:var(--muted); font-size:12px; font-weight:600; }
    tbody tr:last-child td { border-bottom:0; }
    .method { font-weight:600; }
    .method.baseline { color:var(--baseline); }
    .method.projected { color:var(--projected); }
    .method.rtc { color:var(--rtc); }
    .note { max-width:88ch; margin:9px 0 0; color:var(--muted); font-size:13px; }
    .controls { display:flex; flex-wrap:wrap; gap:12px; align-items:end; padding:12px 0; border-block:1px solid var(--line); }
    .field { display:grid; gap:4px; }
    .field label { color:var(--muted); font-size:12px; }
    select { min-width:155px; padding:7px 32px 7px 9px; color:var(--ink); background:var(--surface); border:1px solid var(--line); border-radius:4px; }
    .triplet-count { margin-left:auto; color:var(--muted); font-variant-numeric:tabular-nums; }
    .triplet-grid { display:grid; grid-template-columns:repeat(10,minmax(80px,1fr)); gap:6px; margin-top:12px; }
    .triplet-button { min-height:48px; border:1px solid var(--line); border-radius:4px; color:var(--ink); background:var(--surface); cursor:pointer; text-align:left; padding:6px 8px; font-variant-numeric:tabular-nums; }
    .triplet-button[data-failure="1"] { border-color:var(--failure); }
    .triplet-button[aria-pressed="true"] { border-color:var(--select); box-shadow:inset 0 0 0 2px var(--select); }
    .triplet-button .small { display:block; color:var(--muted); font-size:11px; }
    .selection { margin-top:18px; border-block:1px solid var(--line); background:var(--surface); }
    .selection-head { display:flex; justify-content:space-between; align-items:start; gap:14px; margin:0 14px; padding:14px 0; border-bottom:1px solid var(--line); }
    .description { max-width:80ch; margin:5px 0 0; color:var(--muted); }
    .state-hash { max-width:34ch; margin:0; color:var(--muted); font:12px/1.45 Consolas,monospace; overflow-wrap:anywhere; text-align:right; }
    .rollouts { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); }
    .rollout { min-width:0; padding:14px; }
    .rollout + .rollout { border-left:1px solid var(--line); }
    .rollout-meta { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px 12px; margin:11px 0 0; }
    .rollout-meta dd { font-size:13px; }
    .outcome.success { color:var(--rtc); font-weight:600; }
    .outcome.failure { color:var(--failure); font-weight:600; }
    .clip-empty { display:grid; place-items:center; min-height:212px; margin-top:10px; color:var(--muted); background:var(--surface-alt); text-align:center; padding:12px; }
    video { display:block; width:100%; aspect-ratio:1 / 1; margin-top:10px; background:#060807; object-fit:contain; }
    .playback { display:flex; flex-wrap:wrap; gap:7px; padding:0 14px 14px; }
    .command { padding:7px 12px; border:1px solid var(--line); border-radius:4px; color:var(--ink); background:var(--surface); cursor:pointer; }
    .claim { max-width:94ch; margin:0; color:var(--warning); font-weight:500; }
    details { padding:18px 0 0; }
    summary { cursor:pointer; }
    .provenance { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:9px 24px; margin-top:14px; }
    .provenance dd { font:12px/1.45 Consolas,monospace; }
    @media (max-width:1050px) { .metrics { grid-template-columns:repeat(3,minmax(0,1fr)); } .metric:nth-child(4) { border-left:0; border-top:1px solid var(--line); } .metric:nth-child(5) { border-top:1px solid var(--line); } .triplet-grid { grid-template-columns:repeat(5,minmax(82px,1fr)); } .rollouts { grid-template-columns:1fr; } .rollout + .rollout { border-left:0; border-top:1px solid var(--line); } video { max-height:70vh; } }
    @media (max-width:700px) { .top-inner, main { width:min(100% - 20px, 1460px); } .metrics { grid-template-columns:repeat(2,minmax(0,1fr)); } .metric:nth-child(n+3) { border-left:0; border-top:1px solid var(--line); } .triplet-grid { grid-template-columns:repeat(4,minmax(64px,1fr)); } .selection-head { display:grid; } .state-hash { max-width:none; text-align:left; } .provenance { grid-template-columns:1fr; } }
    @media (max-width:480px) { h1 { font-size:24px; } .metrics { grid-template-columns:1fr; } .metric + .metric { border-left:0; border-top:1px solid var(--line); } .controls, .field, select { width:100%; } .triplet-count { margin-left:0; } .triplet-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } .rollout-meta { grid-template-columns:1fr; } }
    @media (prefers-reduced-motion: reduce) { * { scroll-behavior:auto !important; } }
  </style>
</head>
<body>
  <header class="top"><div class="top-inner"><h1>pi0.5 RTC overlap primary evidence</h1><p class="subtitle">Corrected held-out LIBERO-10 runtime study: unconditioned overlap, hard projection, and reverse-time RTC guidance.</p></div></header>
  <main>
    <section aria-labelledby="scope-title"><h2 id="scope-title">Evidence scope</h2><dl class="metrics" id="metrics"></dl></section>
    <section aria-labelledby="methods-title"><h2 id="methods-title">Method statistics</h2><div class="table-wrap"><table><thead><tr><th>Method</th><th>Success, Wilson 95%</th><th>Motion seam, mean / median</th><th>Gripper seam, mean / median</th><th>Scored transitions</th></tr></thead><tbody id="methods-body"></tbody></table></div></section>
    <section aria-labelledby="contrast-title"><h2 id="contrast-title">Prespecified contrasts</h2><div class="table-wrap"><table><thead><tr><th>Contrast</th><th>Success difference, task-block 95%</th><th>Wins / losses / ties</th><th>McNemar / Holm</th><th>Motion seam difference, 95%</th><th>Gripper seam difference, 95%</th></tr></thead><tbody id="contrast-body"></tbody></table></div><p class="note" id="statistics-note"></p></section>
    <section aria-labelledby="rollout-title"><h2 id="rollout-title">Matched rollout inspection</h2><div class="controls"><div class="field"><label for="task-select">Task</label><select id="task-select"></select></div><div class="field"><label for="state-select">Initial state</label><select id="state-select"></select></div><div class="field"><label for="seed-select">Sampling seed</label><select id="seed-select"></select></div><span class="triplet-count" id="triplet-count"></span></div><div class="triplet-grid" id="triplet-grid" aria-label="All matched rollout triplets"></div><div class="selection"><div class="selection-head"><div><h3 id="selection-id"></h3><p class="description" id="task-description"></p></div><p class="state-hash" id="state-hash"></p></div><div class="rollouts" id="rollouts"></div><div class="playback"><button class="command" id="play-available" type="button">Play available clips</button><button class="command" id="pause-available" type="button">Pause available clips</button><button class="command" id="restart-available" type="button">Restart available clips</button></div></div></section>
    <section aria-labelledby="boundary-title"><h2 id="boundary-title">Claim boundary</h2><p class="claim" id="claim-boundary"></p></section>
    <details><summary>Evidence provenance and integrity</summary><dl class="provenance" id="provenance"></dl></details>
  </main>
  <script id="armbench-data" type="application/json">__ARMBENCH_DATA__</script>
  <script>
    (function () {
      "use strict";
      const data = JSON.parse(document.getElementById("armbench-data").textContent);
      const byId = (id) => document.getElementById(id);
      const pct = (value) => (100 * value).toFixed(1) + "%";
      const signed = (value, digits) => (value >= 0 ? "+" : "") + value.toFixed(digits);
      const interval = (low, high, digits) => "[" + signed(low, digits) + ", " + signed(high, digits) + "]";
      const pvalue = (value) => value < 0.001 ? value.toExponential(2) : value.toFixed(4);
      const addDefinition = (root, label, value) => { const holder = document.createElement("div"); const dt = document.createElement("dt"); const dd = document.createElement("dd"); dt.textContent = label; dd.textContent = value; holder.append(dt, dd); root.append(holder); };
      const metrics = data.cohort;
      [["Frozen matrix", metrics.rollouts + " rollouts"], ["Matched design", metrics.triplets + " triplets"], ["Task coverage", metrics.tasks + " LIBERO-10 tasks"], ["Timing", "H=10, E=" + metrics.executeHorizon + ", d=" + metrics.delaySteps], ["Failure clips", String(metrics.failureClips)]].forEach((item) => addDefinition(byId("metrics"), item[0], item[1]));
      data.methods.forEach((method) => { const row = document.createElement("tr"); const values = [method.label, method.successes + "/" + method.rollouts + " (" + pct(method.rate) + ") [" + pct(method.wilsonLow) + ", " + pct(method.wilsonHigh) + "]", method.motionMean.toFixed(6) + " / " + method.motionMedian.toFixed(6), method.gripperMean.toFixed(6) + " / " + method.gripperMedian.toFixed(6), String(method.scoredTransitions)]; values.forEach((value, index) => { const cell = document.createElement("td"); cell.textContent = value; if (index === 0) cell.className = "method " + method.id.replace("overlap_unconditioned", "baseline").replace("projected_overlap", "projected").replace("rtc_guided_overlap", "rtc"); row.append(cell); }); byId("methods-body").append(row); });
      data.contrasts.forEach((contrast) => { const row = document.createElement("tr"); const values = [contrast.label, signed(100 * contrast.riskDifference, 1) + " points " + interval(100 * contrast.successCiLow, 100 * contrast.successCiHigh, 1), contrast.wins + " / " + contrast.losses + " / " + contrast.ties, pvalue(contrast.rawP) + " / " + pvalue(contrast.holmP), signed(contrast.motionDifference, 6) + " " + interval(contrast.motionCiLow, contrast.motionCiHigh, 6), signed(contrast.gripperDifference, 6) + " " + interval(contrast.gripperCiLow, contrast.gripperCiHigh, 6)]; values.forEach((value, index) => { const cell = document.createElement("td"); cell.textContent = value; if (index === 0) cell.className = "method " + contrast.id.replace("projected_overlap", "projected").replace("rtc_guided_overlap", "rtc"); row.append(cell); }); byId("contrast-body").append(row); });
      byId("statistics-note").textContent = data.statistics.successTest + "; " + data.statistics.multiplicity + ". Bootstrap unit: " + data.statistics.bootstrapUnit + " (" + data.statistics.bootstrapResamples + " resamples).";
      const state = { id: data.defaultTripletId };
      const taskSelect = byId("task-select"), stateSelect = byId("state-select"), seedSelect = byId("seed-select");
      [...new Set(data.triplets.map((item) => item.taskId))].forEach((task) => { const option = document.createElement("option"); option.value = String(task); option.textContent = "Task " + task; taskSelect.append(option); });
      [...new Set(data.triplets.map((item) => item.episodeIndex))].forEach((initialState) => { const option = document.createElement("option"); option.value = String(initialState); option.textContent = "State " + initialState; stateSelect.append(option); });
      [...new Set(data.triplets.map((item) => item.samplingSeed))].forEach((seed) => { const option = document.createElement("option"); option.value = String(seed); option.textContent = String(seed); seedSelect.append(option); });
      function selected() { return data.triplets.find((triplet) => triplet.id === state.id); }
      function choose(id) { state.id = id; renderSelection(); }
      function renderSelection() { const triplet = selected(); if (!triplet) return; taskSelect.value = String(triplet.taskId); stateSelect.value = String(triplet.episodeIndex); seedSelect.value = String(triplet.samplingSeed); byId("selection-id").textContent = "LIBERO-10 / Task " + triplet.taskId + " / State " + triplet.episodeIndex + " / Seed " + triplet.samplingSeed; byId("task-description").textContent = triplet.taskDescription; byId("state-hash").textContent = "Initial state SHA-256: " + triplet.initialStateSha256; const root = byId("rollouts"); root.replaceChildren(); triplet.methods.forEach((method) => { const pane = document.createElement("article"); pane.className = "rollout"; const heading = document.createElement("h3"); heading.className = "method " + method.className; heading.textContent = method.label; pane.append(heading); if (method.video) { const video = document.createElement("video"); video.controls = true; video.preload = "metadata"; video.src = method.video; video.setAttribute("aria-label", method.label + " failure rollout"); pane.append(video); } else { const empty = document.createElement("div"); empty.className = "clip-empty"; empty.textContent = method.success ? "Success: no failure clip captured" : "No clip recorded"; pane.append(empty); } const meta = document.createElement("dl"); meta.className = "rollout-meta"; addDefinition(meta, "Outcome", method.success ? "Success" : "Failure"); meta.lastElementChild.lastElementChild.className = "outcome " + (method.success ? "success" : "failure"); addDefinition(meta, "Policy queries", String(method.policyQueries)); addDefinition(meta, "Motion seam", method.motionSeam.toFixed(6)); addDefinition(meta, "Gripper seam", method.gripperSeam.toFixed(6)); addDefinition(meta, "Latin order", String(method.conditionOrder + 1) + " of 3"); addDefinition(meta, "Termination", method.termination); pane.append(meta); root.append(pane); }); document.querySelectorAll(".triplet-button").forEach((button) => button.setAttribute("aria-pressed", button.dataset.id === triplet.id ? "true" : "false")); }
      const getTriplet = () => data.triplets.find((item) => item.taskId === Number(taskSelect.value) && item.episodeIndex === Number(stateSelect.value) && item.samplingSeed === Number(seedSelect.value));
      [taskSelect, stateSelect, seedSelect].forEach((select) => select.addEventListener("change", () => { const triplet = getTriplet(); if (triplet) choose(triplet.id); }));
      data.triplets.forEach((triplet) => { const button = document.createElement("button"); button.type = "button"; button.className = "triplet-button"; button.dataset.id = triplet.id; const failures = triplet.methods.filter((method) => !method.success).length; button.dataset.failure = failures ? "1" : "0"; button.setAttribute("aria-label", "Task " + triplet.taskId + ", state " + triplet.episodeIndex + ", seed " + triplet.samplingSeed + ", " + failures + " failures"); button.innerHTML = "T" + triplet.taskId + " / S" + triplet.episodeIndex + "<span class=\"small\">" + String(triplet.samplingSeed).slice(-2) + " | " + (failures ? failures + " fail" : "3 success") + "</span>"; button.addEventListener("click", () => choose(triplet.id)); byId("triplet-grid").append(button); });
      byId("triplet-count").textContent = data.cohort.triplets + " triplets / " + data.cohort.failureClips + " failure clips";
      const videos = () => Array.from(byId("rollouts").querySelectorAll("video"));
      byId("play-available").addEventListener("click", () => Promise.allSettled(videos().map((video) => { video.currentTime = 0; return video.play(); })));
      byId("pause-available").addEventListener("click", () => videos().forEach((video) => video.pause()));
      byId("restart-available").addEventListener("click", () => videos().forEach((video) => { video.pause(); video.currentTime = 0; }));
      byId("claim-boundary").textContent = data.claimBoundary;
      Object.entries(data.provenance).forEach(([label, value]) => addDefinition(byId("provenance"), label, Array.isArray(value) ? value.join(" | ") : String(value)));
      renderSelection();
    }());
  </script>
</body>
</html>
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--seed06-root", type=pathlib.Path, default=DEFAULT_SEED06_ROOT)
    parser.add_argument("--seed07-root", type=pathlib.Path, default=DEFAULT_SEED07_ROOT)
    parser.add_argument("--analysis-root", type=pathlib.Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--open", action="store_true", help="open the offline page")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_dashboard(
            args.seed06_root, args.seed07_root, args.analysis_root, args.output
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps({"valid": True, **result}, ensure_ascii=True, indent=2))
    if args.open:
        webbrowser.open(pathlib.Path(result["output"]).as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
