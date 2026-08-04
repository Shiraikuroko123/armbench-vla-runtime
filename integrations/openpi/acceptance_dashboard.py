"""Build an offline acceptance dashboard for a validated LIBERO run."""

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

from integrations.openpi.latency_aligned_analysis import ANALYSIS_SCHEMA_VERSION
from integrations.openpi.libero_compose_run import (
    sha256_file,
    validate_directory_manifest,
    validate_run_manifest,
)


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = (
    PROJECT_ROOT / "evidence" / "pi05_libero_alignment_core_001" / "run"
)
DEFAULT_ANALYSIS_ROOT = (
    PROJECT_ROOT / "evidence" / "pi05_libero_alignment_core_001" / "analysis"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "pi05_libero_alignment_core_001"
    / "index.html"
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
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise ValueError("cannot read CSV %s: %s" % (path, exc)) from exc


def _strict_bool(value: str, label: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError("%s must be True or False" % label)


def _strict_int(value: str, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be an integer" % label) from exc
    if str(parsed) != str(value):
        raise ValueError("%s must use canonical integer syntax" % label)
    return parsed


def _relative_video_url(
    evaluation_root: pathlib.Path, output_parent: pathlib.Path, raw_path: str
) -> str:
    if not raw_path:
        raise ValueError("a required video path is empty")
    resolved_root = evaluation_root.resolve()
    resolved_video = (evaluation_root / raw_path).resolve()
    try:
        resolved_video.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("video escapes evaluation directory: %s" % raw_path) from exc
    if not resolved_video.is_file() or resolved_video.stat().st_size <= 0:
        raise ValueError("required video is missing or empty: %s" % resolved_video)
    relative = pathlib.Path(os.path.relpath(resolved_video, output_parent.resolve()))
    return quote(relative.as_posix(), safe="/:")


def _index_unique(
    rows: Iterable[Mapping[str, str]], fields: Sequence[str], label: str
) -> Dict[tuple[str, ...], Mapping[str, str]]:
    indexed: Dict[tuple[str, ...], Mapping[str, str]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in fields)
        if key in indexed:
            raise ValueError("duplicate %s key: %s" % (label, key))
        indexed[key] = row
    return indexed


def _outcome(async_success: bool, aligned_success: bool) -> str:
    if aligned_success and not async_success:
        return "aligned_win"
    if async_success and not aligned_success:
        return "async_win"
    if async_success:
        return "both_success"
    return "both_failure"


def build_dashboard(
    run_root: pathlib.Path,
    analysis_root: pathlib.Path,
    output: pathlib.Path,
) -> Dict[str, Any]:
    run_root = run_root.resolve()
    analysis_root = analysis_root.resolve()
    output = output.resolve()
    validation = validate_run_manifest(run_root)
    if not validation.get("valid") or not validation.get("complete"):
        raise ValueError(
            "source run failed independent validation: %s"
            % "; ".join(str(item) for item in validation.get("errors", []))
        )

    analysis_validation = validate_directory_manifest(
        analysis_root, expected_schema=ANALYSIS_SCHEMA_VERSION
    )
    if not analysis_validation.get("valid"):
        raise ValueError(
            "derived analysis failed independent validation: %s"
            % "; ".join(
                str(item) for item in analysis_validation.get("errors", [])
            )
        )

    analysis = _read_json(analysis_root / "analysis.json")
    if analysis.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise ValueError(
            "analysis.json schema_version=%r, expected %r"
            % (analysis.get("schema_version"), ANALYSIS_SCHEMA_VERSION)
        )
    source = analysis.get("source")
    if not isinstance(source, dict):
        raise ValueError("analysis.json source must be an object")
    source_bindings = (
        (
            "root_manifest_sha256",
            run_root / "manifest.json",
            "root manifest",
        ),
        (
            "per_episode_csv_sha256",
            run_root / "evaluation" / "per_episode.csv",
            "per_episode.csv",
        ),
    )
    for field, path, label in source_bindings:
        expected_sha256 = source.get(field)
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError("analysis source %s is not a SHA-256" % field)
        if not path.is_file():
            raise ValueError("analysis source %s is missing: %s" % (label, path))
        actual_sha256 = sha256_file(path)
        if expected_sha256 != actual_sha256:
            raise ValueError(
                "analysis source %s SHA-256 mismatch: expected %s, got %s"
                % (label, expected_sha256, actual_sha256)
            )

    pair_rows = _read_csv(analysis_root / "per_pair.csv")
    episode_rows = _read_csv(run_root / "evaluation" / "per_episode.csv")
    if not pair_rows:
        raise ValueError("per_pair.csv contains no rows")

    episodes = _index_unique(
        episode_rows, ("pair_id", "mode"), "episode/mode"
    )
    evaluation_root = run_root / "evaluation"
    pairs: list[Dict[str, Any]] = []
    expected_modes = ("async_unguarded", "latency_aligned")
    for row_index, pair_row in enumerate(pair_rows, start=2):
        pair_id = pair_row.get("pair_id", "")
        if not pair_id:
            raise ValueError("per_pair.csv row %d has no pair_id" % row_index)
        mode_rows: Dict[str, Mapping[str, str]] = {}
        for mode in expected_modes:
            key = (pair_id, mode)
            if key not in episodes:
                raise ValueError("pair %s is missing mode %s" % (pair_id, mode))
            mode_rows[mode] = episodes[key]

        async_row = mode_rows["async_unguarded"]
        aligned_row = mode_rows["latency_aligned"]
        if async_row.get("task_description") != aligned_row.get("task_description"):
            raise ValueError("pair %s has mismatched task descriptions" % pair_id)
        for mode, episode in mode_rows.items():
            if not _strict_bool(
                episode.get("video_required", ""), "%s.%s.video_required" % (pair_id, mode)
            ):
                raise ValueError("pair %s does not require every video" % pair_id)

        async_success = _strict_bool(
            pair_row.get("async_unguarded_success", ""),
            "%s.async_unguarded_success" % pair_id,
        )
        aligned_success = _strict_bool(
            pair_row.get("latency_aligned_success", ""),
            "%s.latency_aligned_success" % pair_id,
        )
        pairs.append(
            {
                "pairId": pair_id,
                "taskId": _strict_int(pair_row.get("task_id", ""), "task_id"),
                "episodeIndex": _strict_int(
                    pair_row.get("episode_index", ""), "episode_index"
                ),
                "latencySteps": _strict_int(
                    pair_row.get("latency_steps", ""), "latency_steps"
                ),
                "latencyMs": float(async_row.get("injected_latency_ms", "")),
                "taskDescription": async_row.get("task_description", ""),
                "outcome": _outcome(async_success, aligned_success),
                "async": {
                    "success": async_success,
                    "queries": _strict_int(
                        pair_row.get("async_unguarded_policy_queries", ""),
                        "async_unguarded_policy_queries",
                    ),
                    "termination": async_row.get("termination_reason", ""),
                    "video": _relative_video_url(
                        evaluation_root,
                        output.parent,
                        async_row.get("video_path", ""),
                    ),
                },
                "aligned": {
                    "success": aligned_success,
                    "queries": _strict_int(
                        pair_row.get("latency_aligned_policy_queries", ""),
                        "latency_aligned_policy_queries",
                    ),
                    "termination": aligned_row.get("termination_reason", ""),
                    "video": _relative_video_url(
                        evaluation_root,
                        output.parent,
                        aligned_row.get("video_path", ""),
                    ),
                },
            }
        )

    pairs.sort(key=lambda item: (item["latencyMs"], item["taskId"], item["episodeIndex"]))
    strata = analysis.get("latency_strata")
    if not isinstance(strata, list) or not strata:
        raise ValueError("analysis.json contains no latency strata")
    summary = []
    for raw in strata:
        if not isinstance(raw, dict):
            raise ValueError("latency_strata entries must be objects")
        summary.append(
            {
                "latencyMs": float(raw["injected_latency_ms"]),
                "role": str(raw["analysis_role"]),
                "n": int(raw["paired_n"]),
                "asyncSuccesses": int(raw["async_unguarded_successes"]),
                "alignedSuccesses": int(raw["latency_aligned_successes"]),
                "difference": float(raw["aligned_minus_async_success_rate_difference"]),
                "ciLow": float(raw["paired_bootstrap95_low"]),
                "ciHigh": float(raw["paired_bootstrap95_high"]),
                "holmP": float(raw["mcnemar_holm_p"]),
                "asyncQueries": float(raw["async_unguarded_mean_policy_queries"]),
                "alignedQueries": float(raw["latency_aligned_mean_policy_queries"]),
            }
        )
    summary.sort(key=lambda item: item["latencyMs"])

    identity = analysis.get("frozen_identity", {})
    payload = {
        "runId": run_root.parent.name,
        "claimBoundary": analysis.get("claim_boundary", ""),
        "pairs": pairs,
        "summary": summary,
        "validation": {
            "valid": True,
            "complete": True,
            "filesChecked": int(validation.get("files_checked", 0)),
            "runtimeFailures": len(analysis.get("runtime_failures", [])),
            "rollouts": int(analysis.get("itt", {}).get("rollouts", len(pairs) * 2)),
            "matchedPairs": int(analysis.get("itt", {}).get("pairs", len(pairs))),
            "checkpoint": identity.get("policy_config", ""),
            "checkpointSha256": identity.get("checkpoint_content_sha256", ""),
            "runCommit": identity.get("armbench_run_commit", ""),
            "manifestSha256": source.get("root_manifest_sha256", ""),
            "perEpisodeSha256": source.get("per_episode_csv_sha256", ""),
            "analysisManifestSha256": sha256_file(
                analysis_root / "manifest.json"
            ),
            "analysisFilesChecked": int(
                analysis_validation.get("files_checked", 0)
            ),
        },
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    serialized = serialized.replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__ARMBENCH_DATA__", serialized)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[pathlib.Path] = None
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
            temporary_path = pathlib.Path(handle.name)
        temporary_path.replace(output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return {
        "output": str(output),
        "rollouts": payload["validation"]["rollouts"],
        "matched_pairs": len(pairs),
        "videos_verified": len(pairs) * 2,
        "files_checked": payload["validation"]["filesChecked"],
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>ArmBench VLA Runtime Acceptance</title>
  <style>
    :root {
      color-scheme: light dark;
      --page: #f2f5f7;
      --surface: #ffffff;
      --surface-alt: #e9eef1;
      --ink: #17232b;
      --muted: #5d6a72;
      --line: #ccd5da;
      --header: #18242b;
      --header-ink: #f7fafb;
      --baseline: #b54708;
      --method: #086f83;
      --success: #1c7c45;
      --failure: #b42318;
      --selected: #1769aa;
      --focus: #6d4aff;
    }

    @media (prefers-color-scheme: dark) {
      :root {
        --page: #111619;
        --surface: #1a2227;
        --surface-alt: #232d33;
        --ink: #edf3f5;
        --muted: #aab8bf;
        --line: #3a484f;
        --header: #090d0f;
        --header-ink: #f7fafb;
        --baseline: #f3a45f;
        --method: #55bfd0;
        --success: #69cf92;
        --failure: #ff8a80;
        --selected: #66aef0;
        --focus: #b7a5ff;
      }
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-width: 320px;
      background: var(--page);
      color: var(--ink);
      font-family: Inter, "Segoe UI", "Microsoft YaHei", sans-serif;
      font-size: 15px;
      line-height: 1.5;
    }
    button, select, input { font: inherit; }
    button, select { min-height: 38px; }
    button:focus-visible, select:focus-visible, input:focus-visible, video:focus-visible {
      outline: 3px solid var(--focus);
      outline-offset: 2px;
    }
    .top-band { background: var(--header); color: var(--header-ink); }
    .top-inner, main { width: min(1400px, calc(100% - 32px)); margin: 0 auto; }
    .top-inner { padding: 24px 0 22px; }
    .eyebrow { color: #b9c8cf; font-size: 13px; margin: 0 0 6px; }
    h1 { font-size: 28px; font-weight: 500; margin: 0; letter-spacing: 0; }
    .run-line { color: #d7e1e5; margin: 7px 0 0; overflow-wrap: anywhere; }
    main { padding: 22px 0 36px; }
    h2 { font-size: 19px; font-weight: 500; margin: 0 0 14px; letter-spacing: 0; }
    h3 { font-size: 16px; font-weight: 500; margin: 0; letter-spacing: 0; }
    section { margin-top: 26px; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 10px;
    }
    .metric {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
      min-height: 98px;
      padding: 15px;
    }
    .metric-label { color: var(--muted); font-size: 13px; }
    .metric-value { font-size: 25px; font-weight: 500; margin-top: 3px; }
    .metric-detail { color: var(--muted); font-size: 13px; margin-top: 2px; }
    .chart { display: grid; gap: 14px; }
    .chart-row {
      display: grid;
      grid-template-columns: 70px minmax(0, 1fr);
      gap: 12px;
      align-items: start;
    }
    .chart-label { font-variant-numeric: tabular-nums; padding-top: 4px; }
    .bar-stack { display: grid; gap: 6px; }
    .bar-line { display: grid; grid-template-columns: minmax(0, 1fr) 62px; gap: 8px; align-items: center; }
    .bar-track { height: 24px; background: var(--surface-alt); position: relative; }
    .bar-fill { height: 100%; min-width: 2px; transition: width 180ms ease-out; }
    .bar-fill.baseline { background: var(--baseline); }
    .bar-fill.method { background: var(--method); }
    .bar-value { text-align: right; font-variant-numeric: tabular-nums; }
    .legend { display: flex; gap: 18px; flex-wrap: wrap; color: var(--muted); margin-top: 10px; }
    .legend-item { display: inline-flex; gap: 7px; align-items: center; }
    .swatch { width: 12px; height: 12px; display: inline-block; }
    .swatch.baseline { background: var(--baseline); }
    .swatch.method { background: var(--method); }
    .control-band {
      display: flex;
      flex-wrap: wrap;
      align-items: end;
      gap: 14px;
      border-block: 1px solid var(--line);
      padding: 14px 0;
    }
    .field { display: grid; gap: 5px; }
    .field label, .group-label { color: var(--muted); font-size: 13px; }
    select {
      border: 1px solid var(--line);
      border-radius: 4px;
      background: var(--surface);
      color: var(--ink);
      padding: 7px 34px 7px 10px;
    }
    .segments { display: flex; gap: 0; }
    .segments button, .command {
      border: 1px solid var(--line);
      background: var(--surface);
      color: var(--ink);
      padding: 7px 12px;
      cursor: pointer;
    }
    .segments button + button { border-left: 0; }
    .segments button:first-child { border-radius: 4px 0 0 4px; }
    .segments button:last-child { border-radius: 0 4px 4px 0; }
    .segments button[aria-pressed="true"] {
      background: var(--selected);
      color: #ffffff;
      border-color: var(--selected);
    }
    .pair-header { display: flex; justify-content: space-between; gap: 16px; align-items: baseline; }
    .pair-count { color: var(--muted); font-variant-numeric: tabular-nums; }
    .pair-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(76px, 1fr));
      gap: 7px;
      margin-top: 12px;
    }
    .pair-cell {
      border: 1px solid transparent;
      border-radius: 4px;
      color: var(--ink);
      min-height: 44px;
      cursor: pointer;
      font-variant-numeric: tabular-nums;
    }
    .pair-cell.aligned_win { background: color-mix(in srgb, var(--success) 22%, var(--surface)); }
    .pair-cell.async_win { background: color-mix(in srgb, var(--failure) 22%, var(--surface)); }
    .pair-cell.both_success { background: color-mix(in srgb, var(--method) 16%, var(--surface)); }
    .pair-cell.both_failure { background: var(--surface-alt); }
    .pair-cell[aria-pressed="true"] { border-color: var(--selected); box-shadow: inset 0 0 0 2px var(--selected); }
    .outcome-legend { display: flex; flex-wrap: wrap; gap: 14px; color: var(--muted); margin-top: 10px; font-size: 13px; }
    .dot { width: 10px; height: 10px; border-radius: 2px; display: inline-block; margin-right: 5px; }
    .dot.aligned_win { background: var(--success); }
    .dot.async_win { background: var(--failure); }
    .dot.both_success { background: var(--method); }
    .dot.both_failure { background: var(--muted); }
    .selection {
      margin-top: 18px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 16px;
    }
    .selection-top { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px 20px; }
    .task-text { margin: 6px 0 0; color: var(--muted); max-width: 900px; }
    .badge { display: inline-flex; align-items: center; border-radius: 4px; padding: 3px 8px; font-size: 13px; }
    .badge.success { color: var(--success); border: 1px solid currentColor; }
    .badge.failure { color: var(--failure); border: 1px solid currentColor; }
    .video-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 15px; }
    .video-head { display: flex; justify-content: space-between; gap: 10px; align-items: center; margin-bottom: 7px; }
    .mode-meta { color: var(--muted); font-size: 13px; }
    video { display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: contain; background: #000000; }
    .playback { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
    .command { border-radius: 4px; }
    .command.primary { background: var(--selected); color: #ffffff; border-color: var(--selected); }
    .integrity {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 10px 20px;
      border-top: 1px solid var(--line);
      padding-top: 15px;
    }
    .integrity dt { color: var(--muted); font-size: 13px; }
    .integrity dd { margin: 2px 0 0; overflow-wrap: anywhere; font-variant-numeric: tabular-nums; }
    .scope { color: var(--muted); margin: 24px 0 0; padding-top: 14px; border-top: 1px solid var(--line); }
    .hidden { display: none !important; }
    @media (max-width: 760px) {
      .top-inner, main { width: min(100% - 20px, 1400px); }
      h1 { font-size: 23px; }
      .chart-row { grid-template-columns: 55px minmax(0, 1fr); gap: 8px; }
      .video-grid { grid-template-columns: 1fr; }
      .control-band { align-items: stretch; }
      .field, .segments { width: 100%; }
      .segments button { flex: 1 1 0; }
      select { width: 100%; }
    }
  </style>
</head>
<body>
  <header class="top-band">
    <div class="top-inner">
      <p class="eyebrow">OFFICIAL pi0.5-LIBERO CHECKPOINT / PAIRED RUNTIME STUDY</p>
      <h1>ArmBench VLA 运行时验收台</h1>
      <p class="run-line" id="run-line"></p>
    </div>
  </header>
  <main>
    <section class="metrics" aria-label="核心验收指标">
      <article class="metric">
        <div class="metric-label">200 ms 基线成功率</div>
        <div class="metric-value" id="primary-baseline">-</div>
        <div class="metric-detail">async_unguarded</div>
      </article>
      <article class="metric">
        <div class="metric-label">200 ms 对齐成功率</div>
        <div class="metric-value" id="primary-method">-</div>
        <div class="metric-detail">latency_aligned</div>
      </article>
      <article class="metric">
        <div class="metric-label">配对提升</div>
        <div class="metric-value" id="primary-difference">-</div>
        <div class="metric-detail" id="primary-ci">-</div>
      </article>
      <article class="metric">
        <div class="metric-label">完整性</div>
        <div class="metric-value" id="integrity-status">-</div>
        <div class="metric-detail" id="integrity-detail">-</div>
      </article>
    </section>

    <section aria-labelledby="success-chart-title">
      <h2 id="success-chart-title">延迟条件成功率</h2>
      <div class="chart" id="success-chart" role="img" aria-label="三档延迟下基线和时间对齐方法的成功率比较"></div>
      <div class="legend" aria-label="图例">
        <span class="legend-item"><span class="swatch baseline"></span>异步基线</span>
        <span class="legend-item"><span class="swatch method"></span>时间对齐</span>
      </div>
    </section>

    <section aria-labelledby="pair-title">
      <h2 id="pair-title">逐对视频验收</h2>
      <div class="control-band">
        <div class="field">
          <span class="group-label">注入延迟</span>
          <div class="segments" id="delay-controls" role="group" aria-label="选择注入延迟"></div>
        </div>
        <div class="field">
          <label for="task-filter">任务</label>
          <select id="task-filter"></select>
        </div>
        <div class="field">
          <label for="outcome-filter">配对结果</label>
          <select id="outcome-filter">
            <option value="all">全部</option>
            <option value="aligned_win">基线失败 / 对齐成功</option>
            <option value="both_success">双方成功</option>
            <option value="both_failure">双方失败</option>
            <option value="async_win">基线成功 / 对齐失败</option>
          </select>
        </div>
      </div>
      <div class="pair-header">
        <h3>匹配初始状态</h3>
        <span class="pair-count" id="pair-count"></span>
      </div>
      <div class="pair-grid" id="pair-grid"></div>
      <div class="outcome-legend" aria-label="配对结果图例">
        <span><span class="dot aligned_win"></span>对齐获胜</span>
        <span><span class="dot both_success"></span>双方成功</span>
        <span><span class="dot both_failure"></span>双方失败</span>
        <span><span class="dot async_win"></span>基线获胜</span>
      </div>

      <div class="selection" id="selection" aria-live="polite">
        <div class="selection-top">
          <div>
            <h3 id="selection-id">-</h3>
            <p class="task-text" id="task-description">-</p>
          </div>
          <span class="badge" id="pair-outcome">-</span>
        </div>
        <div class="video-grid">
          <div class="video-column">
            <div class="video-head">
              <h3>异步基线</h3>
              <span class="mode-meta" id="async-meta">-</span>
            </div>
            <video id="async-video" controls muted playsinline preload="metadata"></video>
          </div>
          <div class="video-column">
            <div class="video-head">
              <h3>时间对齐</h3>
              <span class="mode-meta" id="aligned-meta">-</span>
            </div>
            <video id="aligned-video" controls muted playsinline preload="metadata"></video>
          </div>
        </div>
        <div class="playback" aria-label="对照播放控制">
          <button type="button" class="command primary" id="play-both">同步播放</button>
          <button type="button" class="command" id="pause-both">暂停</button>
          <button type="button" class="command" id="restart-both">从头播放</button>
        </div>
      </div>
    </section>

    <section aria-labelledby="integrity-title">
      <h2 id="integrity-title">证据身份与完整性</h2>
      <dl class="integrity" id="integrity-list"></dl>
    </section>
    <p class="scope" id="claim-boundary"></p>
  </main>
  <script id="armbench-data" type="application/json">__ARMBENCH_DATA__</script>
  <script>
    (function () {
      "use strict";
      const data = JSON.parse(document.getElementById("armbench-data").textContent);
      const state = { delay: 200, task: "all", outcome: "all", selected: null };
      const byId = (id) => document.getElementById(id);
      const percent = (value) => Math.round(value * 100) + "%";
      const signedPoints = (value) => (value >= 0 ? "+" : "") + Math.round(value * 100) + " points";
      const outcomeText = {
        aligned_win: "基线失败 / 对齐成功",
        async_win: "基线成功 / 对齐失败",
        both_success: "双方成功",
        both_failure: "双方失败"
      };

      function initializeSummary() {
        const primary = data.summary.find((row) => row.role === "primary") || data.summary[data.summary.length - 1];
        byId("primary-baseline").textContent = primary.asyncSuccesses + "/" + primary.n;
        byId("primary-method").textContent = primary.alignedSuccesses + "/" + primary.n;
        byId("primary-difference").textContent = signedPoints(primary.difference);
        byId("primary-ci").textContent = "95% CI [" + signedPoints(primary.ciLow) + ", " + signedPoints(primary.ciHigh) + "]";
        byId("integrity-status").textContent = data.validation.valid && data.validation.complete ? "VALID" : "INVALID";
        byId("integrity-detail").textContent = data.validation.filesChecked + " run + " + data.validation.analysisFilesChecked + " analysis files";
        byId("run-line").textContent = data.runId + " / " + data.validation.rollouts + " rollouts / " + data.validation.matchedPairs + " matched pairs";
        byId("claim-boundary").textContent = "Claim boundary: " + data.claimBoundary;
      }

      function renderChart() {
        const chart = byId("success-chart");
        chart.replaceChildren();
        data.summary.forEach((row) => {
          const group = document.createElement("div");
          group.className = "chart-row";
          const label = document.createElement("div");
          label.className = "chart-label";
          label.textContent = row.latencyMs + " ms";
          const stack = document.createElement("div");
          stack.className = "bar-stack";
          [
            { value: row.asyncSuccesses / row.n, className: "baseline", name: "异步基线" },
            { value: row.alignedSuccesses / row.n, className: "method", name: "时间对齐" }
          ].forEach((bar) => {
            const line = document.createElement("div");
            line.className = "bar-line";
            const track = document.createElement("div");
            track.className = "bar-track";
            const fill = document.createElement("div");
            fill.className = "bar-fill " + bar.className;
            fill.style.width = percent(bar.value);
            fill.setAttribute("aria-label", bar.name + " " + percent(bar.value));
            track.appendChild(fill);
            const value = document.createElement("span");
            value.className = "bar-value";
            value.textContent = percent(bar.value);
            line.append(track, value);
            stack.appendChild(line);
          });
          group.append(label, stack);
          chart.appendChild(group);
        });
      }

      function initializeControls() {
        const delays = Array.from(new Set(data.pairs.map((pair) => pair.latencyMs))).sort((a, b) => a - b);
        if (!delays.includes(state.delay)) state.delay = delays[delays.length - 1];
        const delayControls = byId("delay-controls");
        delays.forEach((delay) => {
          const button = document.createElement("button");
          button.type = "button";
          button.textContent = delay + " ms";
          button.dataset.delay = String(delay);
          button.setAttribute("aria-pressed", delay === state.delay ? "true" : "false");
          button.addEventListener("click", () => {
            state.delay = delay;
            delayControls.querySelectorAll("button").forEach((item) => {
              item.setAttribute("aria-pressed", item === button ? "true" : "false");
            });
            state.selected = null;
            renderPairs();
          });
          delayControls.appendChild(button);
        });

        const taskFilter = byId("task-filter");
        const all = document.createElement("option");
        all.value = "all";
        all.textContent = "全部任务";
        taskFilter.appendChild(all);
        const tasks = Array.from(new Set(data.pairs.map((pair) => pair.taskId))).sort((a, b) => a - b);
        tasks.forEach((task) => {
          const option = document.createElement("option");
          option.value = String(task);
          option.textContent = "Task " + task;
          taskFilter.appendChild(option);
        });
        taskFilter.addEventListener("change", () => {
          state.task = taskFilter.value;
          state.selected = null;
          renderPairs();
        });
        byId("outcome-filter").addEventListener("change", (event) => {
          state.outcome = event.target.value;
          state.selected = null;
          renderPairs();
        });
      }

      function filteredPairs() {
        return data.pairs.filter((pair) => {
          return pair.latencyMs === state.delay &&
            (state.task === "all" || String(pair.taskId) === state.task) &&
            (state.outcome === "all" || pair.outcome === state.outcome);
        });
      }

      function renderPairs() {
        const pairs = filteredPairs();
        const grid = byId("pair-grid");
        grid.replaceChildren();
        byId("pair-count").textContent = pairs.length + " pairs";
        pairs.forEach((pair) => {
          const button = document.createElement("button");
          button.type = "button";
          button.className = "pair-cell " + pair.outcome;
          button.textContent = "T" + pair.taskId + " / E" + pair.episodeIndex;
          button.setAttribute("aria-label", "Task " + pair.taskId + ", episode " + pair.episodeIndex + ", " + outcomeText[pair.outcome]);
          button.setAttribute("aria-pressed", pair.pairId === state.selected ? "true" : "false");
          button.addEventListener("click", () => selectPair(pair));
          grid.appendChild(button);
        });
        if (!pairs.length) {
          byId("selection").classList.add("hidden");
          return;
        }
        const selected = pairs.find((pair) => pair.pairId === state.selected) ||
          pairs.find((pair) => pair.outcome === "aligned_win") || pairs[0];
        selectPair(selected, false);
      }

      function setStatus(element, result) {
        element.textContent = (result.success ? "成功" : "失败") + " / " + result.queries + " queries";
        element.style.color = result.success ? "var(--success)" : "var(--failure)";
      }

      function selectPair(pair, scroll) {
        state.selected = pair.pairId;
        byId("selection").classList.remove("hidden");
        byId("selection-id").textContent = "Task " + pair.taskId + " / Episode " + pair.episodeIndex + " / " + pair.latencyMs + " ms";
        byId("task-description").textContent = pair.taskDescription;
        const outcome = byId("pair-outcome");
        outcome.textContent = outcomeText[pair.outcome];
        outcome.className = "badge " + (pair.outcome === "async_win" || pair.outcome === "both_failure" ? "failure" : "success");
        setStatus(byId("async-meta"), pair.async);
        setStatus(byId("aligned-meta"), pair.aligned);
        const asyncVideo = byId("async-video");
        const alignedVideo = byId("aligned-video");
        asyncVideo.src = pair.async.video;
        alignedVideo.src = pair.aligned.video;
        asyncVideo.load();
        alignedVideo.load();
        byId("pair-grid").querySelectorAll("button").forEach((button) => {
          button.setAttribute("aria-pressed", button.getAttribute("aria-label") === "Task " + pair.taskId + ", episode " + pair.episodeIndex + ", " + outcomeText[pair.outcome] ? "true" : "false");
        });
        if (scroll) byId("selection").scrollIntoView({ behavior: "smooth", block: "nearest" });
      }

      function initializePlayback() {
        const videos = [byId("async-video"), byId("aligned-video")];
        byId("play-both").addEventListener("click", () => {
          const start = Math.min.apply(null, videos.map((video) => Number.isFinite(video.currentTime) ? video.currentTime : 0));
          videos.forEach((video) => { video.currentTime = start; });
          Promise.allSettled(videos.map((video) => video.play()));
        });
        byId("pause-both").addEventListener("click", () => videos.forEach((video) => video.pause()));
        byId("restart-both").addEventListener("click", () => {
          videos.forEach((video) => { video.pause(); video.currentTime = 0; });
        });
      }

      function renderIntegrity() {
        const fields = [
          ["Run", data.runId],
          ["Validation", data.validation.valid && data.validation.complete ? "valid / complete" : "invalid"],
          ["Protected files", String(data.validation.filesChecked)],
          ["Runtime failures", String(data.validation.runtimeFailures)],
          ["Checkpoint", data.validation.checkpoint],
          ["Checkpoint SHA-256", data.validation.checkpointSha256],
          ["Run commit", data.validation.runCommit],
          ["Root manifest SHA-256", data.validation.manifestSha256],
          ["per_episode.csv SHA-256", data.validation.perEpisodeSha256],
          ["Analysis manifest SHA-256", data.validation.analysisManifestSha256],
          ["Analysis protected files", String(data.validation.analysisFilesChecked)]
        ];
        const list = byId("integrity-list");
        fields.forEach((field) => {
          const wrapper = document.createElement("div");
          const term = document.createElement("dt");
          const detail = document.createElement("dd");
          term.textContent = field[0];
          detail.textContent = field[1] || "-";
          wrapper.append(term, detail);
          list.appendChild(wrapper);
        });
      }

      initializeSummary();
      renderChart();
      initializeControls();
      initializePlayback();
      renderIntegrity();
      renderPairs();
    }());
  </script>
</body>
</html>
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-root", type=pathlib.Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--analysis-root", type=pathlib.Path, default=DEFAULT_ANALYSIS_ROOT
    )
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--open", action="store_true", help="open the generated page")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = build_dashboard(args.run_root, args.analysis_root, args.output)
    except (OSError, ValueError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"valid": True, **result}, ensure_ascii=False, indent=2))
    if args.open:
        webbrowser.open(pathlib.Path(result["output"]).as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
