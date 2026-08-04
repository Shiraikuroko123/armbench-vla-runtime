"""Build an offline dashboard for the frozen pi0.5 cross-suite validation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import tempfile
import webbrowser
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence
from urllib.parse import quote

from integrations.openpi.cross_suite_external_analysis import (
    ANALYSIS_SCHEMA_VERSION,
    FROZEN_RUN_IDS,
    FROZEN_SUITES,
    validate_analysis_manifest,
)
from integrations.openpi.libero_compose_run import sha256_file, validate_run_manifest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_EXTERNAL_ROOT = PROJECT_ROOT.parent / "evidence" / "cloud" / "external"
DEFAULT_RUN_ROOTS = tuple(DEFAULT_EXTERNAL_ROOT / run_id for run_id in FROZEN_RUN_IDS)
DEFAULT_ANALYSIS_ROOT = DEFAULT_EXTERNAL_ROOT / "pi05_cross_suite_external_analysis_001"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "reports" / "pi05_cross_suite_external_001" / "index.html"
)


def _read_json(path: pathlib.Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read JSON %s: %s" % (path, exc)) from exc
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object: %s" % path)
    return value


def _read_csv(path: pathlib.Path) -> list[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            if len(fields) != len(set(fields)):
                raise ValueError("CSV has duplicate headers: %s" % path)
            return list(reader)
    except OSError as exc:
        raise ValueError("cannot read CSV %s: %s" % (path, exc)) from exc


def _strict_bool(value: Any, label: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError("%s must be exactly True or False" % label)


def _strict_int(value: Any, label: str) -> int:
    text = str(value)
    try:
        parsed = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be an integer" % label) from exc
    if str(parsed) != text:
        raise ValueError("%s must use canonical integer syntax" % label)
    return parsed


def _strict_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be numeric" % label) from exc
    if not (float("-inf") < parsed < float("inf")):
        raise ValueError("%s must be finite" % label)
    return parsed


def _index_unique(
    rows: Iterable[Mapping[str, str]], fields: Sequence[str], label: str
) -> Dict[tuple[str, ...], Mapping[str, str]]:
    indexed: Dict[tuple[str, ...], Mapping[str, str]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in fields)
        if not all(key):
            raise ValueError("empty %s key: %s" % (label, key))
        if key in indexed:
            raise ValueError("duplicate %s key: %s" % (label, key))
        indexed[key] = row
    return indexed


def _relative_video_url(
    evaluation_root: pathlib.Path, output_parent: pathlib.Path, raw_path: str
) -> str:
    if not raw_path or "\\" in raw_path:
        raise ValueError("a required video path is empty or non-canonical")
    resolved_root = evaluation_root.resolve()
    resolved_video = (evaluation_root / pathlib.PurePosixPath(raw_path)).resolve()
    try:
        resolved_video.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("video escapes evaluation directory: %s" % raw_path) from exc
    if not resolved_video.is_file() or resolved_video.stat().st_size <= 0:
        raise ValueError("required video is missing or empty: %s" % resolved_video)
    relative = pathlib.Path(os.path.relpath(resolved_video, output_parent.resolve()))
    return quote(relative.as_posix(), safe="/:")


def _outcome(async_success: bool, aligned_success: bool) -> str:
    if aligned_success and not async_success:
        return "aligned_win"
    if async_success and not aligned_success:
        return "async_win"
    if async_success:
        return "both_success"
    return "both_failure"


def _validate_source(
    run_root: pathlib.Path,
    source: Mapping[str, Any],
    suite: str,
    run_id: str,
) -> tuple[Dict[tuple[str, str], Mapping[str, str]], Dict[str, Any]]:
    run_root = run_root.resolve()
    if run_root.name != run_id or source.get("run_id") != run_id:
        raise ValueError("run/source order mismatch for %s" % run_id)
    if source.get("task_suite") != suite:
        raise ValueError("source suite mismatch for %s" % run_id)
    validation = validate_run_manifest(run_root)
    if validation.get("valid") is not True or validation.get("complete") is not True:
        raise ValueError(
            "source run %s failed independent validation: %s"
            % (
                run_id,
                "; ".join(str(item) for item in validation.get("errors", [])),
            )
        )
    expected_count = source.get("independently_validated_files")
    if validation.get("files_checked") != expected_count:
        raise ValueError("source validator file count mismatch for %s" % run_id)

    evaluation = run_root / "evaluation"
    bindings = (
        ("root_manifest_sha256", run_root / "manifest.json", "root manifest"),
        (
            "evaluation_manifest_sha256",
            evaluation / "manifest.json",
            "evaluation manifest",
        ),
        ("per_episode_csv_sha256", evaluation / "per_episode.csv", "per_episode.csv"),
        (
            "resolved_protocol_sha256",
            evaluation / "resolved_protocol.json",
            "resolved_protocol.json",
        ),
        ("environment_sha256", evaluation / "environment.json", "environment.json"),
    )
    verified_hashes: Dict[str, str] = {}
    for field, path, label in bindings:
        expected = source.get(field)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("source %s is not a SHA-256 for %s" % (field, run_id))
        if not path.is_file():
            raise ValueError("source %s is missing for %s" % (label, run_id))
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError("source %s SHA-256 mismatch for %s" % (label, run_id))
        verified_hashes[field] = actual

    rows = _read_csv(evaluation / "per_episode.csv")
    if len(rows) != 100:
        raise ValueError("%s must contain exactly 100 episode rows" % run_id)
    indexed = _index_unique(rows, ("pair_id", "mode"), "%s episode/mode" % suite)
    videos = set()
    for number, row in enumerate(rows, 2):
        label = "%s row %d" % (suite, number)
        if row.get("task_suite") != suite:
            raise ValueError("%s suite mismatch" % label)
        if not _strict_bool(row.get("video_required"), label + ".video_required"):
            raise ValueError("%s does not require video" % label)
        if row.get("video_error_type") or row.get("video_error_message"):
            raise ValueError("%s records a video encoding error" % label)
        path = str(row.get("video_path", ""))
        _relative_video_url(evaluation, evaluation, path)
        if path in videos:
            raise ValueError("%s contains a duplicate video path" % suite)
        videos.add(path)
    if len(videos) != 100 or source.get("videos_verified") != 100:
        raise ValueError("%s does not bind exactly 100 videos" % run_id)
    return indexed, {
        "suite": suite,
        "runId": run_id,
        "filesChecked": int(validation["files_checked"]),
        "videosVerified": len(videos),
        **verified_hashes,
    }


def _normalized_suite_rows(analysis: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows = analysis.get("suite_results")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("analysis must contain exactly three suite results")
    normalized = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or raw.get("task_suite") != FROZEN_SUITES[index]:
            raise ValueError("suite result order mismatch")
        normalized.append(
            {
                "suite": str(raw["task_suite"]),
                "n": int(raw["paired_n"]),
                "asyncSuccesses": int(raw["async_unguarded_successes"]),
                "alignedSuccesses": int(raw["latency_aligned_successes"]),
                "asyncRate": float(raw["async_unguarded_success_rate"]),
                "alignedRate": float(raw["latency_aligned_success_rate"]),
                "difference": float(raw["aligned_minus_async_success_rate_difference"]),
                "ciLow": float(raw["paired_bootstrap95_low"]),
                "ciHigh": float(raw["paired_bootstrap95_high"]),
                "wins": int(raw["aligned_wins"]),
                "losses": int(raw["async_unguarded_wins"]),
                "ties": int(raw["ties"]),
                "rawP": float(raw["mcnemar_exact_p"]),
                "holmP": float(raw["mcnemar_holm_p"]),
                "holmReject": bool(raw["holm_reject_alpha_0_05"]),
                "asyncQueries": float(raw["async_unguarded_mean_policy_queries"]),
                "alignedQueries": float(raw["latency_aligned_mean_policy_queries"]),
            }
        )
    if any(row["n"] != 50 for row in normalized):
        raise ValueError("every suite result must contain 50 matched pairs")
    return normalized


def build_dashboard(
    run_roots: Sequence[pathlib.Path],
    analysis_root: pathlib.Path,
    output: pathlib.Path,
) -> Dict[str, Any]:
    if len(run_roots) != 3:
        raise ValueError("exactly three run roots are required in frozen order")
    analysis_root = pathlib.Path(analysis_root).resolve()
    output = pathlib.Path(output).resolve()
    analysis_validation = validate_analysis_manifest(analysis_root)
    if analysis_validation.get("valid") is not True:
        raise ValueError(
            "cross-suite analysis failed validation: %s"
            % "; ".join(str(item) for item in analysis_validation.get("errors", []))
        )
    analysis = _read_json(analysis_root / "analysis.json")
    if analysis.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise ValueError("analysis schema mismatch")
    sources = analysis.get("sources")
    if not isinstance(sources, list) or len(sources) != 3:
        raise ValueError("analysis must bind exactly three sources")

    episodes: Dict[tuple[str, str], Mapping[str, str]] = {}
    source_records = []
    roots_by_suite: Dict[str, pathlib.Path] = {}
    for run_root, source, suite, run_id in zip(
        run_roots, sources, FROZEN_SUITES, FROZEN_RUN_IDS
    ):
        if not isinstance(source, Mapping):
            raise ValueError("analysis source must be an object")
        root = pathlib.Path(run_root).resolve()
        indexed, record = _validate_source(root, source, suite, run_id)
        overlap = set(episodes).intersection(indexed)
        if overlap:
            raise ValueError("episode keys overlap across suites")
        episodes.update(indexed)
        source_records.append(record)
        roots_by_suite[suite] = root

    pair_rows = _read_csv(analysis_root / "per_pair.csv")
    if len(pair_rows) != 150:
        raise ValueError("per_pair.csv must contain exactly 150 matched pairs")
    pair_index = _index_unique(pair_rows, ("pair_id",), "pair")
    if len(pair_index) != 150:
        raise ValueError("per_pair.csv must contain 150 unique pairs")
    pairs: list[Dict[str, Any]] = []
    suite_pair_counts = {suite: 0 for suite in FROZEN_SUITES}
    for row_number, row in enumerate(pair_rows, 2):
        suite = str(row.get("task_suite", ""))
        if suite not in roots_by_suite:
            raise ValueError("per_pair.csv row %d has an unknown suite" % row_number)
        pair_id = str(row.get("pair_id", ""))
        mode_rows = {}
        for mode in ("async_unguarded", "latency_aligned"):
            key = (pair_id, mode)
            if key not in episodes:
                raise ValueError("pair %s is missing mode %s" % (pair_id, mode))
            mode_rows[mode] = episodes[key]
        async_row = mode_rows["async_unguarded"]
        aligned_row = mode_rows["latency_aligned"]
        task_id = _strict_int(row.get("task_id"), "task_id")
        episode_index = _strict_int(row.get("episode_index"), "episode_index")
        for mode, episode in mode_rows.items():
            if episode.get("task_suite") != suite:
                raise ValueError("pair %s has a suite mismatch" % pair_id)
            if _strict_int(episode.get("task_id"), "episode.task_id") != task_id:
                raise ValueError("pair %s has a task mismatch" % pair_id)
            if _strict_int(episode.get("episode_index"), "episode.episode_index") != episode_index:
                raise ValueError("pair %s has an episode-index mismatch" % pair_id)
            if episode.get("task_description") != async_row.get("task_description"):
                raise ValueError("pair %s has mismatched task descriptions" % pair_id)
            if episode.get("video_error_type") or episode.get("video_error_message"):
                raise ValueError("pair %s mode %s has a video error" % (pair_id, mode))
        async_success = _strict_bool(
            row.get("async_unguarded_success"), pair_id + ".async_success"
        )
        aligned_success = _strict_bool(
            row.get("latency_aligned_success"), pair_id + ".aligned_success"
        )
        if async_success != _strict_bool(async_row.get("success"), pair_id + ".async_episode"):
            raise ValueError("pair %s async success disagrees with source" % pair_id)
        if aligned_success != _strict_bool(aligned_row.get("success"), pair_id + ".aligned_episode"):
            raise ValueError("pair %s aligned success disagrees with source" % pair_id)
        async_queries = _strict_int(row.get("async_unguarded_policy_queries"), "async queries")
        aligned_queries = _strict_int(row.get("latency_aligned_policy_queries"), "aligned queries")
        if async_queries != _strict_int(async_row.get("policy_queries"), "source async queries"):
            raise ValueError("pair %s async query count disagrees with source" % pair_id)
        if aligned_queries != _strict_int(aligned_row.get("policy_queries"), "source aligned queries"):
            raise ValueError("pair %s aligned query count disagrees with source" % pair_id)
        evaluation = roots_by_suite[suite] / "evaluation"
        pairs.append(
            {
                "pairId": pair_id,
                "suite": suite,
                "taskId": task_id,
                "episodeIndex": episode_index,
                "taskDescription": str(async_row.get("task_description", "")),
                "outcome": _outcome(async_success, aligned_success),
                "async": {
                    "success": async_success,
                    "queries": async_queries,
                    "termination": str(async_row.get("termination_reason", "")),
                    "video": _relative_video_url(
                        evaluation, output.parent, str(async_row.get("video_path", ""))
                    ),
                },
                "aligned": {
                    "success": aligned_success,
                    "queries": aligned_queries,
                    "termination": str(aligned_row.get("termination_reason", "")),
                    "video": _relative_video_url(
                        evaluation, output.parent, str(aligned_row.get("video_path", ""))
                    ),
                },
            }
        )
        suite_pair_counts[suite] += 1
    if any(count != 50 for count in suite_pair_counts.values()):
        raise ValueError("every suite must contain exactly 50 matched pairs")
    pairs.sort(key=lambda item: (FROZEN_SUITES.index(item["suite"]), item["taskId"], item["episodeIndex"]))

    task_rows = _read_csv(analysis_root / "task_descriptives.csv")
    if len(task_rows) != 30:
        raise ValueError("task_descriptives.csv must contain exactly 30 rows")
    task_index = _index_unique(task_rows, ("task_suite", "task_id"), "task descriptive")
    tasks = []
    for suite in FROZEN_SUITES:
        for task_id in range(10):
            key = (suite, str(task_id))
            if key not in task_index:
                raise ValueError("missing task descriptive %s/%d" % (suite, task_id))
            row = task_index[key]
            tasks.append(
                {
                    "suite": suite,
                    "taskId": task_id,
                    "asyncSuccesses": _strict_int(row.get("async_unguarded_successes"), "task async successes"),
                    "alignedSuccesses": _strict_int(row.get("latency_aligned_successes"), "task aligned successes"),
                    "difference": _strict_float(row.get("aligned_minus_async_success_rate_difference"), "task difference"),
                    "asyncQueries": _strict_float(row.get("async_unguarded_mean_policy_queries"), "task async queries"),
                    "alignedQueries": _strict_float(row.get("latency_aligned_mean_policy_queries"), "task aligned queries"),
                }
            )

    suites = _normalized_suite_rows(analysis)
    pooled = analysis.get("pooled_descriptive")
    if not isinstance(pooled, Mapping) or pooled.get("analysis_scope") != "pooled_descriptive_only_no_p_value":
        raise ValueError("pooled result must remain descriptive only")
    acceptance = analysis.get("acceptance")
    if not isinstance(acceptance, Mapping):
        raise ValueError("analysis has no acceptance object")
    identity = analysis.get("frozen_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("analysis has no frozen identity")
    payload = {
        "title": "pi0.5 cross-suite external validation",
        "claimBoundary": str(analysis.get("claim_boundary", "")),
        "suites": suites,
        "tasks": tasks,
        "pairs": pairs,
        "pooled": {
            "scope": "descriptive_only_no_p_value",
            "asyncSuccesses": int(pooled["async_unguarded_successes"]),
            "alignedSuccesses": int(pooled["latency_aligned_successes"]),
            "difference": float(pooled["aligned_minus_async_success_rate_difference"]),
            "ciLow": float(pooled["paired_bootstrap95_low"]),
            "ciHigh": float(pooled["paired_bootstrap95_high"]),
            "asyncQueries": float(pooled["async_unguarded_mean_policy_queries"]),
            "alignedQueries": float(pooled["latency_aligned_mean_policy_queries"]),
        },
        "acceptance": {
            "passed": acceptance.get("passed") is True,
            "rollouts": int(analysis.get("itt", {}).get("rollouts", 0)),
            "pairs": int(analysis.get("itt", {}).get("pairs", 0)),
            "videosVerified": int(acceptance.get("videos_verified", 0)),
            "runtimeFailures": int(acceptance.get("runtime_failures", 0)),
        },
        "identity": {
            "policy": str(identity.get("policy_config", "")),
            "checkpointSha256": str(identity.get("checkpoint_content_sha256", "")),
            "armbenchCommit": str(identity.get("armbench_run_commit", "")),
            "alignmentCommit": str(identity.get("temporal_alignment_implementation_commit", "")),
            "openpiCommit": str(identity.get("openpi_commit", "")),
        },
        "integrity": {
            "analysisManifestSha256": sha256_file(analysis_root / "manifest.json"),
            "analysisFilesChecked": int(analysis_validation.get("files_checked", 0)),
            "sources": source_records,
        },
    }
    if payload["acceptance"]["rollouts"] != 300 or payload["acceptance"]["pairs"] != 150:
        raise ValueError("analysis ITT counts are not the frozen 300/150")
    if payload["acceptance"]["videosVerified"] != 300:
        raise ValueError("analysis acceptance does not verify 300 videos")

    serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    serialized = serialized.replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__ARMBENCH_DATA__", serialized)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[pathlib.Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=output.parent,
            prefix=output.name + ".", suffix=".tmp", delete=False,
        ) as handle:
            handle.write(html)
            temporary = pathlib.Path(handle.name)
        temporary.replace(output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return {
        "output": str(output),
        "rollouts": 300,
        "matched_pairs": 150,
        "videos_verified": 300,
        "suites": 3,
        "tasks": 30,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>ArmBench Cross-Suite Acceptance</title>
  <style>
    :root {
      color-scheme: light dark;
      --page: #f3f6f7; --surface: #ffffff; --surface-2: #e8eef0;
      --ink: #18242a; --muted: #5c6970; --line: #c9d3d7;
      --header: #142127; --header-ink: #f7fafb; --baseline: #bd5b18;
      --aligned: #087b83; --success: #217a48; --failure: #b42318;
      --selected: #1769aa; --focus: #6d4aff;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --page: #101619; --surface: #192227; --surface-2: #253137;
        --ink: #edf3f5; --muted: #aab8bf; --line: #3c4b52;
        --header: #080d0f; --baseline: #efa260; --aligned: #55c0c7;
        --success: #69cf92; --failure: #ff8a80; --selected: #66aef0;
        --focus: #b7a5ff;
      }
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-width: 320px; background: var(--page); color: var(--ink); font-family: Inter, "Segoe UI", Arial, sans-serif; font-size: 15px; line-height: 1.45; }
    button, select { font: inherit; min-height: 38px; }
    button:focus-visible, select:focus-visible, video:focus-visible, summary:focus-visible { outline: 3px solid var(--focus); outline-offset: 2px; }
    .top { background: var(--header); color: var(--header-ink); }
    .top-inner, main { width: min(1440px, calc(100% - 32px)); margin: 0 auto; }
    .top-inner { padding: 22px 0 20px; }
    .eyebrow { margin: 0 0 5px; color: #b9c8cf; font-size: 12px; text-transform: uppercase; }
    h1 { margin: 0; font-size: 27px; font-weight: 550; letter-spacing: 0; }
    .subtitle { margin: 7px 0 0; color: #d5e0e4; }
    main { padding: 20px 0 36px; }
    section { margin-top: 25px; }
    h2 { margin: 0 0 12px; font-size: 19px; font-weight: 600; letter-spacing: 0; }
    h3 { margin: 0; font-size: 16px; font-weight: 600; letter-spacing: 0; }
    .metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 9px; }
    .metric { min-height: 94px; padding: 14px; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); }
    .metric-label { color: var(--muted); font-size: 12px; }
    .metric-value { margin-top: 3px; font-size: 24px; font-variant-numeric: tabular-nums; }
    .metric-detail { margin-top: 2px; color: var(--muted); font-size: 12px; }
    .suite-table, .task-table { width: 100%; border-collapse: collapse; background: var(--surface); }
    .suite-table { min-width: 900px; }
    .task-table { min-width: 680px; }
    th, td { padding: 10px 11px; border-bottom: 1px solid var(--line); text-align: left; font-variant-numeric: tabular-nums; }
    th { color: var(--muted); font-size: 12px; font-weight: 600; }
    tbody tr:last-child td { border-bottom: 0; }
    .table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 6px; }
    .bar-cell { min-width: 210px; }
    .bar-line { display: grid; grid-template-columns: minmax(100px, 1fr) 48px; gap: 7px; align-items: center; margin: 3px 0; }
    .track { height: 14px; background: var(--surface-2); }
    .fill { height: 100%; min-width: 2px; }
    .fill.async { background: var(--baseline); }
    .fill.aligned { background: var(--aligned); }
    .pass { color: var(--success); font-weight: 650; }
    .fail { color: var(--failure); font-weight: 650; }
    .note { color: var(--muted); font-size: 13px; margin: 8px 0 0; }
    .controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: end; padding: 13px 0; border-block: 1px solid var(--line); }
    .field { display: grid; gap: 4px; }
    .field label { color: var(--muted); font-size: 12px; }
    select { min-width: 170px; padding: 7px 32px 7px 9px; color: var(--ink); background: var(--surface); border: 1px solid var(--line); border-radius: 4px; }
    .pair-head { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; margin-top: 14px; }
    .pair-count { color: var(--muted); }
    .pair-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(88px, 1fr)); gap: 7px; margin-top: 10px; }
    .pair { min-height: 44px; border: 1px solid transparent; border-radius: 4px; color: var(--ink); cursor: pointer; }
    .pair.aligned_win { background: color-mix(in srgb, var(--success) 22%, var(--surface)); }
    .pair.async_win { background: color-mix(in srgb, var(--failure) 22%, var(--surface)); }
    .pair.both_success { background: color-mix(in srgb, var(--aligned) 16%, var(--surface)); }
    .pair.both_failure { background: var(--surface-2); }
    .pair[aria-pressed="true"] { border-color: var(--selected); box-shadow: inset 0 0 0 2px var(--selected); }
    .legend { display: flex; flex-wrap: wrap; gap: 13px; color: var(--muted); font-size: 12px; margin-top: 9px; }
    .dot { display: inline-block; width: 9px; height: 9px; margin-right: 5px; border-radius: 2px; }
    .dot.aligned_win { background: var(--success); } .dot.async_win { background: var(--failure); }
    .dot.both_success { background: var(--aligned); } .dot.both_failure { background: var(--muted); }
    .selection { margin-top: 17px; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); }
    .selection-head { display: flex; justify-content: space-between; gap: 14px; align-items: start; padding: 14px; border-bottom: 1px solid var(--line); }
    .description { margin: 5px 0 0; color: var(--muted); overflow-wrap: anywhere; }
    .badge { flex: 0 0 auto; padding: 3px 8px; border-radius: 4px; font-size: 12px; background: var(--surface-2); }
    .videos { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .video-pane { padding: 13px; min-width: 0; }
    .video-pane + .video-pane { border-left: 1px solid var(--line); }
    .video-title { display: flex; justify-content: space-between; gap: 8px; align-items: baseline; margin-bottom: 8px; }
    .video-meta { color: var(--muted); font-size: 12px; text-align: right; }
    video { display: block; width: 100%; aspect-ratio: 1 / 1; background: #050607; object-fit: contain; }
    .playback { display: flex; flex-wrap: wrap; gap: 7px; padding: 0 13px 13px; }
    .command { padding: 7px 11px; border: 1px solid var(--line); border-radius: 4px; color: var(--ink); background: var(--surface); cursor: pointer; }
    .hidden { display: none; }
    details { border-top: 1px solid var(--line); margin-top: 24px; padding-top: 14px; }
    summary { cursor: pointer; font-weight: 600; }
    .integrity { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 20px; margin-top: 12px; }
    .integrity div { min-width: 0; }
    dt { color: var(--muted); font-size: 12px; }
    dd { margin: 2px 0 0; font-family: Consolas, monospace; font-size: 12px; overflow-wrap: anywhere; }
    @media (max-width: 860px) { .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); } .videos { grid-template-columns: 1fr; } .video-pane + .video-pane { border-left: 0; border-top: 1px solid var(--line); } }
    @media (max-width: 560px) { .top-inner, main { width: min(100% - 20px, 1440px); } h1 { font-size: 22px; } .metrics { grid-template-columns: 1fr; } .metric { min-height: 82px; } .controls, .field, select { width: 100%; } .selection-head { display: grid; } .integrity { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header class="top"><div class="top-inner">
    <p class="eyebrow">Official pi0.5-LIBERO checkpoint / frozen paired study</p>
    <h1>Cross-Suite Temporal Alignment</h1>
    <p class="subtitle">Three held-out LIBERO suites, 300 rollouts, 150 matched pairs, deterministic 200 ms delay</p>
  </div></header>
  <main>
    <section aria-labelledby="overview-title"><h2 id="overview-title">Acceptance overview</h2><div class="metrics" id="metrics"></div></section>
    <section aria-labelledby="suite-title"><h2 id="suite-title">Confirmatory suite results</h2><div class="table-wrap"><table class="suite-table"><thead><tr><th>Suite</th><th>Success</th><th>Aligned - async</th><th>Wins / losses / ties</th><th>Holm decision</th><th>Mean queries</th></tr></thead><tbody id="suite-body"></tbody></table></div><p class="note">Three suite-level exact McNemar tests form the confirmatory family. Bootstrap intervals are descriptive.</p></section>
    <section aria-labelledby="pairs-title"><h2 id="pairs-title">Matched-pair evidence</h2><div class="controls"><div class="field"><label for="suite-filter">Suite</label><select id="suite-filter"></select></div><div class="field"><label for="task-filter">Task</label><select id="task-filter"></select></div><div class="field"><label for="outcome-filter">Outcome</label><select id="outcome-filter"><option value="all">All outcomes</option><option value="aligned_win">Aligned win</option><option value="async_win">Async win</option><option value="both_success">Both succeed</option><option value="both_failure">Both fail</option></select></div></div><div class="pair-head"><h3 id="pair-scope">Pairs</h3><span class="pair-count" id="pair-count"></span></div><div class="pair-grid" id="pair-grid"></div><div class="legend"><span><i class="dot aligned_win"></i>Aligned win</span><span><i class="dot async_win"></i>Async win</span><span><i class="dot both_success"></i>Both succeed</span><span><i class="dot both_failure"></i>Both fail</span></div><div class="selection hidden" id="selection"><div class="selection-head"><div><h3 id="selection-id"></h3><p class="description" id="task-description"></p></div><span class="badge" id="pair-outcome"></span></div><div class="videos"><div class="video-pane"><div class="video-title"><h3>Async baseline</h3><span class="video-meta" id="async-meta"></span></div><video id="async-video" controls preload="metadata" aria-label="Asynchronous baseline rollout"></video></div><div class="video-pane"><div class="video-title"><h3>Latency aligned</h3><span class="video-meta" id="aligned-meta"></span></div><video id="aligned-video" controls preload="metadata" aria-label="Latency-aligned rollout"></video></div></div><div class="playback"><button class="command" id="play-both" type="button">Play both</button><button class="command" id="pause-both" type="button">Pause both</button><button class="command" id="restart-both" type="button">Restart both</button></div></div></section>
    <section aria-labelledby="task-title"><h2 id="task-title">Task-level descriptives</h2><div class="table-wrap"><table class="task-table"><thead><tr><th>Suite</th><th>Task</th><th>Async</th><th>Aligned</th><th>Difference</th><th>Mean queries A / L</th></tr></thead><tbody id="task-body"></tbody></table></div><p class="note">Task rows are descriptive only. No task-level or pooled significance claim is made.</p></section>
    <details><summary>Evidence identity and integrity</summary><dl class="integrity" id="integrity"></dl></details>
    <p class="note" id="claim-boundary"></p>
  </main>
  <script id="armbench-data" type="application/json">__ARMBENCH_DATA__</script>
  <script>
    (function () {
      "use strict";
      const data = JSON.parse(document.getElementById("armbench-data").textContent);
      const byId = (id) => document.getElementById(id);
      const pct = (value) => (value * 100).toFixed(1) + "%";
      const pvalue = (value) => value < 0.001 ? value.toExponential(2) : value.toFixed(4);
      const names = {libero_object: "LIBERO Object", libero_goal: "LIBERO Goal", libero_10: "LIBERO-10"};
      const outcomes = {aligned_win: "Aligned win", async_win: "Async win", both_success: "Both succeed", both_failure: "Both fail"};
      const state = {suite: data.suites[0].suite, task: "all", outcome: "all", selected: null};

      function metric(label, value, detail) { const node = document.createElement("div"); node.className = "metric"; node.innerHTML = "<div class=\"metric-label\"></div><div class=\"metric-value\"></div><div class=\"metric-detail\"></div>"; node.children[0].textContent = label; node.children[1].textContent = value; node.children[2].textContent = detail; return node; }
      const pooled = data.pooled;
      [
        ["Formal acceptance", data.acceptance.passed ? "PASSED" : "FAILED", data.acceptance.rollouts + "/300 rollouts, " + data.acceptance.runtimeFailures + " runtime failures"],
        ["Task success", pooled.asyncSuccesses + " -> " + pooled.alignedSuccesses, "of 150 per mode; pooled descriptive"],
        ["Success difference", "+" + (pooled.difference * 100).toFixed(1) + " points", "bootstrap 95% [" + (pooled.ciLow * 100).toFixed(1) + ", " + (pooled.ciHigh * 100).toFixed(1) + "]"],
        ["Mean policy queries", pooled.asyncQueries.toFixed(1) + " -> " + pooled.alignedQueries.toFixed(1), (pooled.alignedQueries - pooled.asyncQueries).toFixed(1) + " queries per rollout"]
      ].forEach((item) => byId("metrics").appendChild(metric(item[0], item[1], item[2])));

      data.suites.forEach((suite) => { const row = document.createElement("tr"); const bars = document.createElement("td"); bars.className = "bar-cell"; [["async", suite.asyncRate], ["aligned", suite.alignedRate]].forEach((bar) => { const line = document.createElement("div"); line.className = "bar-line"; line.innerHTML = "<div class=\"track\"><div class=\"fill " + bar[0] + "\" style=\"width:" + pct(bar[1]) + "\"></div></div><span>" + pct(bar[1]) + "</span>"; bars.appendChild(line); }); const cells = [names[suite.suite], bars, "+" + (suite.difference * 100).toFixed(0) + " pts [" + (suite.ciLow * 100).toFixed(0) + ", " + (suite.ciHigh * 100).toFixed(0) + "]", suite.wins + " / " + suite.losses + " / " + suite.ties, (suite.holmReject ? "Reject H0" : "Do not reject") + " / p=" + pvalue(suite.holmP), suite.asyncQueries.toFixed(1) + " / " + suite.alignedQueries.toFixed(1)]; cells.forEach((cell, index) => { if (index === 1) { row.appendChild(cell); return; } const td = document.createElement("td"); td.textContent = cell; if (index === 4) td.className = suite.holmReject ? "pass" : "fail"; row.appendChild(td); }); byId("suite-body").appendChild(row); });

      const suiteFilter = byId("suite-filter"); data.suites.forEach((suite) => { const option = document.createElement("option"); option.value = suite.suite; option.textContent = names[suite.suite]; suiteFilter.appendChild(option); });
      function rebuildTasks() { const select = byId("task-filter"); select.replaceChildren(); const all = document.createElement("option"); all.value = "all"; all.textContent = "All tasks"; select.appendChild(all); data.tasks.filter((task) => task.suite === state.suite).forEach((task) => { const option = document.createElement("option"); option.value = String(task.taskId); option.textContent = "Task " + task.taskId; select.appendChild(option); }); state.task = "all"; }
      suiteFilter.addEventListener("change", () => { state.suite = suiteFilter.value; state.selected = null; rebuildTasks(); renderPairs(); renderTasks(); });
      byId("task-filter").addEventListener("change", (event) => { state.task = event.target.value; state.selected = null; renderPairs(); });
      byId("outcome-filter").addEventListener("change", (event) => { state.outcome = event.target.value; state.selected = null; renderPairs(); });

      function filteredPairs() { return data.pairs.filter((pair) => pair.suite === state.suite && (state.task === "all" || String(pair.taskId) === state.task) && (state.outcome === "all" || pair.outcome === state.outcome)); }
      function status(node, result) { node.textContent = (result.success ? "Success" : "Failure") + " / " + result.queries + " queries / " + result.termination; node.style.color = result.success ? "var(--success)" : "var(--failure)"; }
      function selectPair(pair) { state.selected = pair.pairId; byId("selection").classList.remove("hidden"); byId("selection-id").textContent = names[pair.suite] + " / Task " + pair.taskId + " / Episode " + pair.episodeIndex; byId("task-description").textContent = pair.taskDescription; byId("pair-outcome").textContent = outcomes[pair.outcome]; status(byId("async-meta"), pair.async); status(byId("aligned-meta"), pair.aligned); [["async-video", pair.async.video], ["aligned-video", pair.aligned.video]].forEach((entry) => { const video = byId(entry[0]); video.src = entry[1]; video.load(); }); byId("pair-grid").querySelectorAll("button").forEach((button) => button.setAttribute("aria-pressed", button.dataset.pairId === pair.pairId ? "true" : "false")); }
      function renderPairs() { const pairs = filteredPairs(); byId("pair-scope").textContent = names[state.suite] + " pairs"; byId("pair-count").textContent = pairs.length + " / 50"; const grid = byId("pair-grid"); grid.replaceChildren(); pairs.forEach((pair) => { const button = document.createElement("button"); button.type = "button"; button.className = "pair " + pair.outcome; button.dataset.pairId = pair.pairId; button.textContent = "T" + pair.taskId + " / E" + pair.episodeIndex; button.setAttribute("aria-label", names[pair.suite] + ", task " + pair.taskId + ", episode " + pair.episodeIndex + ", " + outcomes[pair.outcome]); button.setAttribute("aria-pressed", "false"); button.addEventListener("click", () => selectPair(pair)); grid.appendChild(button); }); if (!pairs.length) { byId("selection").classList.add("hidden"); return; } selectPair(pairs.find((pair) => pair.pairId === state.selected) || pairs.find((pair) => pair.outcome === "aligned_win") || pairs[0]); }
      function renderTasks() { const body = byId("task-body"); body.replaceChildren(); data.tasks.filter((task) => task.suite === state.suite).forEach((task) => { const row = document.createElement("tr"); [names[task.suite], String(task.taskId), task.asyncSuccesses + "/5", task.alignedSuccesses + "/5", (task.difference >= 0 ? "+" : "") + (task.difference * 100).toFixed(0) + " pts", task.asyncQueries.toFixed(1) + " / " + task.alignedQueries.toFixed(1)].forEach((value) => { const td = document.createElement("td"); td.textContent = value; row.appendChild(td); }); body.appendChild(row); }); }
      const videos = [byId("async-video"), byId("aligned-video")]; byId("play-both").addEventListener("click", () => { const start = Math.min.apply(null, videos.map((video) => Number.isFinite(video.currentTime) ? video.currentTime : 0)); videos.forEach((video) => video.currentTime = start); Promise.allSettled(videos.map((video) => video.play())); }); byId("pause-both").addEventListener("click", () => videos.forEach((video) => video.pause())); byId("restart-both").addEventListener("click", () => videos.forEach((video) => { video.pause(); video.currentTime = 0; }));
      function addIntegrity(label, value) { const wrapper = document.createElement("div"); const dt = document.createElement("dt"); const dd = document.createElement("dd"); dt.textContent = label; dd.textContent = value || "-"; wrapper.append(dt, dd); byId("integrity").appendChild(wrapper); }
      addIntegrity("Policy", data.identity.policy); addIntegrity("Checkpoint SHA-256", data.identity.checkpointSha256); addIntegrity("ArmBench run commit", data.identity.armbenchCommit); addIntegrity("Alignment implementation commit", data.identity.alignmentCommit); addIntegrity("OpenPI commit", data.identity.openpiCommit); addIntegrity("Analysis manifest SHA-256", data.integrity.analysisManifestSha256); data.integrity.sources.forEach((source) => { addIntegrity(names[source.suite] + " root manifest SHA-256", source.root_manifest_sha256); addIntegrity(names[source.suite] + " per_episode.csv SHA-256", source.per_episode_csv_sha256); });
      byId("claim-boundary").textContent = data.claimBoundary;
      rebuildTasks(); renderPairs(); renderTasks();
    }());
  </script>
</body>
</html>
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-roots", nargs=3, type=pathlib.Path, default=DEFAULT_RUN_ROOTS)
    parser.add_argument("--analysis-root", type=pathlib.Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--open", action="store_true", help="open the generated page")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = build_dashboard(args.run_roots, args.analysis_root, args.output)
    except (OSError, ValueError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps({"valid": True, **result}, ensure_ascii=True, indent=2))
    if args.open:
        webbrowser.open(pathlib.Path(result["output"]).as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
