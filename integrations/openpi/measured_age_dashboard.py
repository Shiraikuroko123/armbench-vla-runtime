"""Build a fail-closed offline dashboard for measured-age v2 evidence."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import pathlib
import re
import tempfile
import webbrowser
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import quote

from integrations.openpi.measured_age_analysis import (
    ANALYSIS_SCHEMA_VERSION,
    ANALYZER_SOURCE,
    VALIDATOR_SOURCE,
    analyze_artifact,
)
from integrations.openpi.validate_measured_age_artifact import (
    EPISODE_FIELDS,
    SCHEMA_VERSION as SOURCE_SCHEMA_VERSION,
    validate_artifact,
)


DASHBOARD_SCHEMA_VERSION = "armbench.pi05_libero_measured_age_dashboard.v1"
MODES = ("async_unguarded", "latency_aligned")
PAIR_FIELDS = (
    "pair_id",
    "task_suite",
    "task_id",
    "episode_index",
    "replan_steps",
    "seed",
    "initial_state_sha256",
)
BURDEN_FIELDS = (
    "policy_queries",
    "accepted_chunks",
    "rejected_chunks",
    "interventions",
    "deadline_misses",
    "horizon_overruns",
    "age_refreshes",
    "fallback_hold_steps",
    "simulated_catchup_steps",
)
PER_PAIR_FIELDS = (
    "pair_id",
    "task_suite",
    "task_id",
    "episode_index",
    "replan_steps",
    "seed",
    "initial_state_sha256",
    "async_success",
    "aligned_success",
    "success_difference",
) + tuple(
    item
    for field in BURDEN_FIELDS
    for item in (
        "async_%s" % field,
        "aligned_%s" % field,
        "%s_difference" % field,
    )
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ANALYSIS_CORE_FILES = frozenset(
    ("analysis.json", "per_pair.csv", "summary.md")
)
_CONFIRMATORY_TASK_IDS = tuple(range(10))
_CONFIRMATORY_EPISODE_INDICES = tuple(range(5, 17))


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: pathlib.Path) -> Mapping[str, Any]:
    def reject(value: str) -> Any:
        raise ValueError("non-finite JSON constant: %s" % value)

    def unique(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
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
    if not isinstance(value, Mapping):
        raise ValueError("JSON root must be an object: %s" % path)
    return value


def _safe_relative(value: Any) -> Optional[pathlib.PurePosixPath]:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        return None
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        return None
    if any(part in ("", ".", "..") for part in path.parts):
        return None
    return path


def _analysis_manifest(root: pathlib.Path) -> Dict[str, Any]:
    manifest = _strict_json(root / "manifest.json")
    if manifest.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise ValueError("analysis manifest schema mismatch")
    records = manifest.get("files")
    if not isinstance(records, Mapping):
        raise ValueError("analysis manifest files must be an object")

    safe_records: Dict[str, Mapping[str, Any]] = {}
    normalized = set()
    root_resolved = root.resolve()
    for raw_path, record in records.items():
        relative = _safe_relative(raw_path)
        if relative is None:
            raise ValueError("unsafe analysis manifest path: %r" % raw_path)
        key = relative.as_posix().casefold()
        if key in normalized:
            raise ValueError("duplicate normalized analysis path: %s" % raw_path)
        normalized.add(key)
        target = root.joinpath(*relative.parts)
        try:
            target.resolve().relative_to(root_resolved)
        except (OSError, ValueError) as exc:
            raise ValueError("analysis path escapes root: %s" % raw_path) from exc
        if target.is_symlink():
            raise ValueError("analysis manifest cannot protect a symlink: %s" % raw_path)
        if not isinstance(record, Mapping):
            raise ValueError("analysis manifest record must be an object: %s" % raw_path)
        safe_records[relative.as_posix()] = record

    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json" and path.suffix != ".tmp"
    }
    if set(actual) != set(safe_records):
        omitted = sorted(set(actual) - set(safe_records))
        absent = sorted(set(safe_records) - set(actual))
        raise ValueError(
            "analysis manifest coverage mismatch; omitted=%r absent=%r"
            % (omitted, absent)
        )
    missing = sorted(_ANALYSIS_CORE_FILES - set(safe_records))
    if missing:
        raise ValueError("analysis manifest is missing core files: %s" % ", ".join(missing))
    for relative, path in actual.items():
        record = safe_records[relative]
        size = record.get("bytes")
        digest = record.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("invalid analysis byte count: %s" % relative)
        if path.stat().st_size != size:
            raise ValueError("analysis byte count mismatch: %s" % relative)
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("invalid analysis SHA-256: %s" % relative)
        if _sha256(path) != digest:
            raise ValueError("analysis SHA-256 mismatch: %s" % relative)
    return {
        "files_checked": len(actual),
        "manifest_sha256": _sha256(root / "manifest.json"),
    }


def _read_csv_exact(
    path: pathlib.Path, fields: Sequence[str], label: str
) -> List[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle, strict=True))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError("cannot read %s: %s" % (label, exc)) from exc
    if not rows or rows[0] != list(fields):
        raise ValueError("%s exact header mismatch" % label)
    output = []
    for number, row in enumerate(rows[1:], start=2):
        if len(row) != len(fields):
            raise ValueError("%s row %d field-count mismatch" % (label, number))
        output.append(dict(zip(fields, row)))
    return output


def _csv_value(value: Any) -> str:
    return "" if value is None else str(value)


def _portable_analysis_equal(
    recorded: Mapping[str, Any], recomputed: Mapping[str, Any]
) -> bool:
    expected = copy.deepcopy(dict(recomputed))
    observed = copy.deepcopy(dict(recorded))
    # Absolute source paths and interpreter/package versions are provenance,
    # not portable scientific results. Source code hashes are checked below.
    expected_source = expected.get("source")
    observed_source = observed.get("source")
    if isinstance(expected_source, dict) and isinstance(observed_source, Mapping):
        expected_source["artifact"] = observed_source.get("artifact")
        # The current validator may add checks while remaining compatible with
        # an artifact-bound historical validator snapshot.
        expected_source["validator_checks"] = observed_source.get(
            "validator_checks"
        )
    expected_impl = expected.get("implementation")
    observed_impl = observed.get("implementation")
    if isinstance(expected_impl, dict) and isinstance(observed_impl, Mapping):
        for field in ("python_version", "numpy_version"):
            expected_impl[field] = observed_impl.get(field)
        expected_impl["validator_source"] = observed_impl.get("validator_source")
        expected_impl["validator_sha256"] = observed_impl.get("validator_sha256")
    return expected == observed


def _validate_source_bindings(
    source_root: pathlib.Path,
    analysis: Mapping[str, Any],
    validator_report: Mapping[str, Any],
) -> Dict[str, str]:
    source = analysis.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("analysis source must be an object")
    if source.get("source_schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError("analysis source schema mismatch")
    bindings = (
        ("source_manifest_sha256", source_root / "manifest.json"),
        ("per_episode_sha256", source_root / "per_episode.csv"),
        ("per_query_sha256", source_root / "per_query.csv"),
    )
    verified: Dict[str, str] = {}
    for field, path in bindings:
        expected = source.get(field)
        if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
            raise ValueError("analysis source %s is not a SHA-256" % field)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError("analysis source hash mismatch: %s" % field)
        verified[field] = expected
    if source.get("validator_schema_version") != validator_report.get("schema_version"):
        raise ValueError("analysis validator schema binding mismatch")

    implementation = analysis.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ValueError("analysis implementation must be an object")
    project_root = pathlib.Path(__file__).resolve().parents[2]
    if implementation.get("analyzer_source") != ANALYZER_SOURCE:
        raise ValueError("analysis implementation source mismatch: analyzer_source")
    analyzer_digest = implementation.get("analyzer_sha256")
    analyzer_path = project_root / pathlib.PurePosixPath(ANALYZER_SOURCE)
    if not isinstance(analyzer_digest, str) or _sha256(analyzer_path) != analyzer_digest:
        raise ValueError("analysis implementation hash mismatch: analyzer_sha256")

    if implementation.get("validator_source") != VALIDATOR_SOURCE:
        raise ValueError("analysis implementation source mismatch: validator_source")
    validator_digest = implementation.get("validator_sha256")
    current_validator = project_root / pathlib.PurePosixPath(VALIDATOR_SOURCE)
    if not isinstance(validator_digest, str) or not _SHA256.fullmatch(
        validator_digest
    ):
        raise ValueError("analysis validator_sha256 is invalid")
    current_digest = _sha256(current_validator)
    if validator_digest == current_digest:
        if source.get("validator_checks") != validator_report.get("checks"):
            raise ValueError("analysis validator check binding mismatch")
    else:
        frozen_validator = (
            source_root
            / "provenance"
            / "armbench_source"
            / pathlib.PurePosixPath(VALIDATOR_SOURCE)
        )
        if not frozen_validator.is_file() or _sha256(frozen_validator) != validator_digest:
            raise ValueError("analysis implementation hash mismatch: validator_sha256")
        recorded_checks = source.get("validator_checks")
        if not isinstance(recorded_checks, list) or not all(
            isinstance(value, str) and value for value in recorded_checks
        ):
            raise ValueError("historical validator checks are invalid")
    verified["analyzer_sha256"] = analyzer_digest
    verified["validator_sha256"] = validator_digest
    return verified


def _strict_bool(value: str, label: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError("%s must be exactly True or False" % label)


def _strict_int(value: str, label: str) -> int:
    if not value.isascii() or not value.isdigit() or (len(value) > 1 and value[0] == "0"):
        raise ValueError("%s must be a canonical nonnegative integer" % label)
    return int(value)


def _relative_video(
    source_root: pathlib.Path, output_parent: pathlib.Path, raw_path: str
) -> str:
    relative = _safe_relative(raw_path)
    if relative is None:
        raise ValueError("video path is empty or non-canonical")
    root = source_root.resolve()
    video = source_root.joinpath(*relative.parts).resolve()
    try:
        video.relative_to(root)
    except ValueError as exc:
        raise ValueError("video escapes source artifact: %s" % raw_path) from exc
    if not video.is_file() or video.stat().st_size <= 0:
        raise ValueError("required video is missing or empty: %s" % raw_path)
    try:
        relative = os.path.relpath(video, output_parent.resolve())
    except ValueError:
        # Windows cannot form a relative path across drive letters.
        return video.as_uri()
    return quote(pathlib.Path(relative).as_posix(), safe="/:")


def _outcome(async_success: bool, aligned_success: bool) -> str:
    if aligned_success and not async_success:
        return "aligned_win"
    if async_success and not aligned_success:
        return "async_win"
    return "both_success" if async_success else "both_failure"


def _evidence_presentation(
    pairs: Sequence[Mapping[str, Any]],
) -> Dict[str, str]:
    identities = {
        (
            pair.get("task_suite"),
            pair.get("task_id"),
            pair.get("episode_index"),
            pair.get("replan_steps"),
            pair.get("seed"),
        )
        for pair in pairs
    }
    expected = {
        ("libero_spatial", task_id, episode_index, 5, 7)
        for task_id in _CONFIRMATORY_TASK_IDS
        for episode_index in _CONFIRMATORY_EPISODE_INDICES
    }
    if len(pairs) == len(expected) and identities == expected:
        return {
            "stage": "confirmatory",
            "title": "Measured-Age VLA Confirmatory Study",
            "scope_label": "CONFIRMATORY / SIMULATION / BLOCKING INFERENCE",
        }
    return {
        "stage": "pilot",
        "title": "Measured-Age VLA Pilot",
        "scope_label": "PILOT / SIMULATION / BLOCKING INFERENCE",
    }


def _pair_payload(
    source_root: pathlib.Path,
    output: pathlib.Path,
    expected_pairs: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    episodes = _read_csv_exact(
        source_root / "per_episode.csv", EPISODE_FIELDS, "source per_episode.csv"
    )
    indexed: Dict[Tuple[str, str], Mapping[str, str]] = {}
    for row in episodes:
        key = (row["pair_id"], row["mode"])
        if key in indexed:
            raise ValueError("duplicate source episode/mode: %r" % (key,))
        indexed[key] = row

    payload = []
    for pair in expected_pairs:
        pair_id = str(pair["pair_id"])
        mode_rows = {}
        for mode in MODES:
            key = (pair_id, mode)
            if key not in indexed:
                raise ValueError("pair %s is missing mode %s" % (pair_id, mode))
            row = indexed[key]
            if row["video_required"] != "True" or row["video_error_type"] or row["video_error_message"]:
                raise ValueError("pair %s mode %s has invalid video evidence" % (pair_id, mode))
            mode_rows[mode] = row
        async_row = mode_rows["async_unguarded"]
        aligned_row = mode_rows["latency_aligned"]
        async_success = _strict_bool(async_row["success"], "async success")
        aligned_success = _strict_bool(aligned_row["success"], "aligned success")
        if async_success != bool(pair["async_success"]) or aligned_success != bool(pair["aligned_success"]):
            raise ValueError("pair %s success disagrees with recomputed analysis" % pair_id)
        for field in PAIR_FIELDS:
            expected = pair[field]
            actual = async_row[field]
            if _csv_value(expected) != actual:
                raise ValueError("pair %s identity mismatch: %s" % (pair_id, field))
            if aligned_row[field] != actual:
                raise ValueError("pair %s modes mismatch: %s" % (pair_id, field))
        payload.append(
            {
                "pairId": pair_id,
                "taskId": int(pair["task_id"]),
                "episodeIndex": int(pair["episode_index"]),
                "taskDescription": async_row["task_description"],
                "outcome": _outcome(async_success, aligned_success),
                "async": {
                    "success": async_success,
                    "queries": int(pair["async_policy_queries"]),
                    "deadline": int(pair["async_deadline_misses"]),
                    "horizon": int(pair["async_horizon_overruns"]),
                    "refresh": int(pair["async_age_refreshes"]),
                    "video": _relative_video(source_root, output.parent, async_row["video_path"]),
                },
                "aligned": {
                    "success": aligned_success,
                    "queries": int(pair["aligned_policy_queries"]),
                    "deadline": int(pair["aligned_deadline_misses"]),
                    "horizon": int(pair["aligned_horizon_overruns"]),
                    "refresh": int(pair["aligned_age_refreshes"]),
                    "video": _relative_video(source_root, output.parent, aligned_row["video_path"]),
                },
            }
        )
    if len(indexed) != len(expected_pairs) * 2:
        raise ValueError("source episode rows exceed the recomputed paired cohort")
    return payload


def build_dashboard(
    source_artifact: pathlib.Path,
    analysis_root: pathlib.Path,
    output: pathlib.Path,
) -> Dict[str, Any]:
    source_root = source_artifact.resolve()
    analysis_root = analysis_root.resolve()
    output = output.resolve()

    report = validate_artifact(source_root)
    if not report.valid:
        raise ValueError(
            "source artifact failed independent validation: %s"
            % "; ".join(report.errors)
        )
    validator_report = report.to_dict()
    analysis_manifest = _analysis_manifest(analysis_root)
    recorded = _strict_json(analysis_root / "analysis.json")
    if recorded.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise ValueError("analysis.json schema mismatch")
    verified_hashes = _validate_source_bindings(
        source_root, recorded, validator_report
    )

    statistics = recorded.get("statistics")
    if not isinstance(statistics, Mapping):
        raise ValueError("analysis statistics must be an object")
    resamples = statistics.get("bootstrap_resamples")
    seed = statistics.get("bootstrap_seed")
    if (
        isinstance(resamples, bool)
        or not isinstance(resamples, int)
        or resamples < 100
        or resamples > 100_000
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ValueError("analysis bootstrap registration is invalid")
    recomputed, expected_pairs = analyze_artifact(
        source_root, bootstrap_resamples=resamples, bootstrap_seed=seed
    )
    if not _portable_analysis_equal(recorded, recomputed):
        raise ValueError("analysis.json disagrees with fresh source recomputation")

    recorded_pairs = _read_csv_exact(
        analysis_root / "per_pair.csv", PER_PAIR_FIELDS, "analysis per_pair.csv"
    )
    expected_csv = [
        {field: _csv_value(pair.get(field)) for field in PER_PAIR_FIELDS}
        for pair in expected_pairs
    ]
    if recorded_pairs != expected_csv:
        raise ValueError("per_pair.csv disagrees with fresh source recomputation")
    pairs = _pair_payload(source_root, output, expected_pairs)
    presentation = _evidence_presentation(expected_pairs)

    payload = {
        "schemaVersion": DASHBOARD_SCHEMA_VERSION,
        "evidenceStage": presentation["stage"],
        "pageTitle": presentation["title"],
        "cohort": recomputed["cohort"],
        "success": recomputed["success"],
        "timing": recomputed["timing"],
        "runtimeBurden": recomputed["runtime_burden"],
        "pairs": pairs,
        "integrity": {
            "sourceValid": True,
            "sourceFilesChecked": len(
                _strict_json(source_root / "manifest.json").get("files", {})
            ),
            "analysisFilesChecked": analysis_manifest["files_checked"],
            "analysisManifestSha256": analysis_manifest["manifest_sha256"],
            "sourceManifestSha256": verified_hashes["source_manifest_sha256"],
            "perEpisodeSha256": verified_hashes["per_episode_sha256"],
            "perQuerySha256": verified_hashes["per_query_sha256"],
            "validatorChecks": validator_report["checks"],
            "analyzerSha256": recomputed["implementation"]["analyzer_sha256"],
            "validatorSha256": recomputed["implementation"]["validator_sha256"],
        },
        "claimBoundary": recomputed["claim_boundary"],
        "scopeLabel": presentation["scope_label"],
    }
    serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    serialized = serialized.replace("</", "<\\/")
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
        temporary.replace(output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return {
        "output": str(output),
        "rollouts": int(recomputed["cohort"]["rollouts"]),
        "pairs": len(pairs),
        "queries": int(recomputed["cohort"]["queries"]),
        "videos_verified": len(pairs) * 2,
        "source_files_checked": payload["integrity"]["sourceFilesChecked"],
        "analysis_files_checked": analysis_manifest["files_checked"],
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>ArmBench Measured-Age Evidence</title>
  <style>
    :root { color-scheme: light; --page:#eef2f3; --surface:#fff; --ink:#172126; --muted:#5b676d; --line:#ccd5d8; --header:#20282c; --green:#176b43; --amber:#a64811; --blue:#126b8a; --red:#a52d27; --focus:#6c4bd2; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--page); color:var(--ink); font:14px/1.45 Inter,Segoe UI,Arial,sans-serif; }
    header { background:var(--header); color:#fff; padding:22px max(20px,calc((100vw - 1220px)/2)); }
    header h1 { margin:5px 0 3px; font-size:26px; letter-spacing:0; }
    header p { margin:0; color:#d9e1e4; }
    .scope { display:inline-block; color:#fff; background:#8b3e15; border:1px solid #c5784d; padding:3px 7px; font-size:11px; font-weight:700; }
    main { max-width:1220px; margin:0 auto; padding:20px; }
    section { margin-bottom:22px; }
    h2 { margin:0 0 11px; font-size:18px; }
    .metrics { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }
    .metric { background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:13px; min-height:92px; }
    .metric span { display:block; color:var(--muted); font-size:12px; }
    .metric strong { display:block; margin-top:7px; font-size:24px; }
    .metric small { color:var(--muted); }
    .table-wrap { overflow:auto; background:var(--surface); border:1px solid var(--line); border-radius:6px; }
    table { width:100%; border-collapse:collapse; }
    th,td { padding:10px 12px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
    th:first-child,td:first-child { text-align:left; }
    tbody tr:last-child td { border-bottom:0; }
    .controls { display:flex; gap:12px; flex-wrap:wrap; align-items:end; margin-bottom:10px; }
    label { display:block; color:var(--muted); font-size:12px; margin-bottom:3px; }
    select,button { min-height:36px; border:1px solid #aebbc0; background:#fff; color:var(--ink); border-radius:5px; padding:7px 10px; }
    button { cursor:pointer; }
    button:focus-visible,select:focus-visible,video:focus-visible { outline:3px solid var(--focus); outline-offset:2px; }
    .pair-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(92px,1fr)); gap:6px; }
    .pair-cell { text-align:left; border-left:5px solid var(--blue); min-height:48px; }
    .pair-cell.aligned_win { border-left-color:var(--green); }
    .pair-cell.async_win,.pair-cell.both_failure { border-left-color:var(--red); }
    .pair-cell[aria-pressed="true"] { outline:3px solid var(--focus); }
    .selection { background:var(--surface); border:1px solid var(--line); border-radius:6px; margin-top:12px; }
    .selection-head { padding:13px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:12px; }
    .selection-head h3,.selection-head p { margin:0; }
    .selection-head p { color:var(--muted); margin-top:3px; }
    .videos { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); }
    .video-pane { padding:13px; min-width:0; }
    .video-pane + .video-pane { border-left:1px solid var(--line); }
    .video-title { display:flex; justify-content:space-between; align-items:baseline; gap:8px; }
    .video-title h3 { font-size:15px; margin:0 0 7px; }
    .video-meta { color:var(--muted); font-size:12px; text-align:right; }
    video { display:block; width:100%; aspect-ratio:1/1; object-fit:contain; background:#050708; }
    .playback { display:flex; gap:7px; padding:0 13px 13px; }
    .hidden { display:none; }
    .integrity { background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:12px; display:grid; grid-template-columns:170px minmax(0,1fr); gap:6px 12px; }
    .integrity dt { color:var(--muted); }
    .integrity dd { margin:0; font-family:Consolas,monospace; overflow-wrap:anywhere; }
    .boundary { border-left:5px solid var(--amber); background:#fff7ed; padding:12px 14px; }
    @media (max-width:820px) { .metrics { grid-template-columns:repeat(2,minmax(0,1fr)); } .videos { grid-template-columns:1fr; } .video-pane + .video-pane { border-left:0; border-top:1px solid var(--line); } }
    @media (max-width:600px) {
      .table-wrap { overflow:visible; border:0; background:transparent; }
      .timing-table, .timing-table tbody, .timing-table tr, .timing-table td { display:block; width:100%; }
      .timing-table thead { position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; }
      .timing-table tbody { display:grid; gap:8px; }
      .timing-table tr { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); border:1px solid var(--line); border-radius:6px; background:var(--surface); overflow:hidden; }
      .timing-table td { min-width:0; padding:8px 10px; border:0; border-top:1px solid var(--line); overflow-wrap:anywhere; }
      .timing-table td:nth-child(even) { border-left:1px solid var(--line); }
      .timing-table td:first-child { grid-column:1/-1; border-top:0; border-left:0; font-weight:700; }
      .timing-table td::before { content:attr(data-label); display:block; margin-bottom:2px; color:var(--muted); font-size:10px; font-weight:600; }
    }
    @media (max-width:480px) { .metrics { grid-template-columns:1fr; } main { padding:14px; } }
  </style>
</head>
<body>
  <header><span class="scope" id="scope"></span><h1 id="evidence-title"></h1><p>Training-free action-chunk alignment under observed response age</p></header>
  <main>
    <section><h2>Paired outcome</h2><div class="metrics"><div class="metric"><span>Async success</span><strong id="async-success"></strong><small id="async-rate"></small></div><div class="metric"><span>Aligned success</span><strong id="aligned-success"></strong><small id="aligned-rate"></small></div><div class="metric"><span>Paired effect</span><strong id="effect"></strong><small id="effect-ci"></small></div><div class="metric"><span>Evidence</span><strong id="cohort"></strong><small id="query-count"></small></div></div></section>
    <section><h2>Measured timing and runtime burden</h2><div class="table-wrap"><table class="timing-table"><thead><tr><th>Mode</th><th>Age P95</th><th>Age max</th><th>Deadline</th><th>Horizon</th><th>Hold-refresh</th><th>Fail closed</th></tr></thead><tbody id="timing-body"></tbody></table></div></section>
    <section><h2>Matched-pair rollouts</h2><div class="controls"><div><label for="task-filter">Task</label><select id="task-filter"><option value="all">All tasks</option></select></div><div><label for="outcome-filter">Outcome</label><select id="outcome-filter"><option value="all">All outcomes</option><option value="aligned_win">Aligned win</option><option value="async_win">Async win</option><option value="both_success">Both succeed</option><option value="both_failure">Both fail</option></select></div></div><div class="pair-grid" id="pair-grid"></div><div class="selection hidden" id="selection"><div class="selection-head"><div><h3 id="selection-id"></h3><p id="task-description"></p></div><span id="pair-outcome"></span></div><div class="videos"><div class="video-pane"><div class="video-title"><h3>Async baseline</h3><span class="video-meta" id="async-meta"></span></div><video id="async-video" controls preload="metadata"></video></div><div class="video-pane"><div class="video-title"><h3>Latency aligned</h3><span class="video-meta" id="aligned-meta"></span></div><video id="aligned-video" controls preload="metadata"></video></div></div><div class="playback"><button id="play-both" type="button">Play both</button><button id="pause-both" type="button">Pause</button><button id="restart-both" type="button">Restart</button></div></div></section>
    <section><h2>Evidence integrity</h2><dl class="integrity" id="integrity"></dl></section>
    <p class="boundary" id="boundary"></p>
  </main>
  <script id="armbench-data" type="application/json">__ARMBENCH_DATA__</script>
  <script>
    (() => {
      "use strict";
      const data=JSON.parse(document.getElementById("armbench-data").textContent), byId=(id)=>document.getElementById(id), pct=(v)=>((v*100).toFixed(1)+"%"), signed=(v)=>((v>=0?"+":"")+(v*100).toFixed(1)+" pp");
      document.title=data.pageTitle; byId("evidence-title").textContent=data.pageTitle;
      byId("scope").textContent=data.scopeLabel;
      const a=data.success.async_unguarded,l=data.success.latency_aligned,p=data.success.paired;
      byId("async-success").textContent=a.successes+"/"+a.rollouts; byId("async-rate").textContent=pct(a.rate);
      byId("aligned-success").textContent=l.successes+"/"+l.rollouts; byId("aligned-rate").textContent=pct(l.rate);
      byId("effect").textContent=signed(p.rate_difference); byId("effect-ci").textContent="paired 95% ["+signed(p.paired_bootstrap95_low)+", "+signed(p.paired_bootstrap95_high)+"]";
      byId("cohort").textContent=data.cohort.rollouts+" rollouts"; byId("query-count").textContent=data.cohort.pairs+" pairs / "+data.cohort.queries+" queries";
      data.timing.forEach((row)=>{ const tr=document.createElement("tr"), labels=["Mode","Age P95","Age max","Deadline","Horizon","Hold-refresh","Fail closed"], values=[row.mode,row.observation_age_ms.p95.toFixed(1)+" ms",row.observation_age_ms.max.toFixed(1)+" ms",row.deadline_misses+" ("+pct(row.deadline_miss_rate_per_query)+")",row.horizon_overruns+" ("+pct(row.horizon_overrun_rate_per_query)+")",String(row.hold_refresh_queries),String(row.fail_closed_queries)]; values.forEach((value,index)=>{const td=document.createElement("td");td.dataset.label=labels[index];td.textContent=value;tr.appendChild(td);});byId("timing-body").appendChild(tr);});
      const taskFilter=byId("task-filter"), outcomeFilter=byId("outcome-filter"); Array.from(new Set(data.pairs.map((row)=>row.taskId))).sort((x,y)=>x-y).forEach((task)=>{const option=document.createElement("option");option.value=String(task);option.textContent="Task "+task;taskFilter.appendChild(option);});
      const labels={aligned_win:"Aligned win",async_win:"Async win",both_success:"Both succeed",both_failure:"Both fail"}; let selected=null;
      function status(result){return (result.success?"success":"failure")+" / "+result.queries+" queries / D"+result.deadline+" H"+result.horizon+" R"+result.refresh;}
      function select(pair){selected=pair.pairId;byId("selection").classList.remove("hidden");byId("selection-id").textContent="Task "+pair.taskId+" / Episode "+pair.episodeIndex;byId("task-description").textContent=pair.taskDescription;byId("pair-outcome").textContent=labels[pair.outcome];byId("async-meta").textContent=status(pair.async);byId("aligned-meta").textContent=status(pair.aligned);[["async-video",pair.async.video],["aligned-video",pair.aligned.video]].forEach((entry)=>{const video=byId(entry[0]);video.src=entry[1];video.load();});byId("pair-grid").querySelectorAll("button").forEach((button)=>button.setAttribute("aria-pressed",button.dataset.pairId===selected?"true":"false"));}
      function render(){const rows=data.pairs.filter((pair)=>(taskFilter.value==="all"||String(pair.taskId)===taskFilter.value)&&(outcomeFilter.value==="all"||pair.outcome===outcomeFilter.value));const grid=byId("pair-grid");grid.replaceChildren();rows.forEach((pair)=>{const button=document.createElement("button");button.type="button";button.className="pair-cell "+pair.outcome;button.dataset.pairId=pair.pairId;button.textContent="T"+pair.taskId+" / E"+pair.episodeIndex;button.title=labels[pair.outcome];button.addEventListener("click",()=>select(pair));grid.appendChild(button);});if(!rows.length){byId("selection").classList.add("hidden");return;}select(rows.find((pair)=>pair.pairId===selected)||rows.find((pair)=>pair.outcome==="aligned_win")||rows[0]);}
      taskFilter.addEventListener("change",render);outcomeFilter.addEventListener("change",render);render();
      const videos=[byId("async-video"),byId("aligned-video")];byId("play-both").addEventListener("click",()=>{const start=Math.min(...videos.map((video)=>Number.isFinite(video.currentTime)?video.currentTime:0));videos.forEach((video)=>video.currentTime=start);Promise.allSettled(videos.map((video)=>video.play()));});byId("pause-both").addEventListener("click",()=>videos.forEach((video)=>video.pause()));byId("restart-both").addEventListener("click",()=>videos.forEach((video)=>{video.pause();video.currentTime=0;}));
      const integrity=[["Source artifact","VALID"],["Source files",data.integrity.sourceFilesChecked],["Analysis files",data.integrity.analysisFilesChecked],["Source manifest SHA-256",data.integrity.sourceManifestSha256],["per_episode.csv SHA-256",data.integrity.perEpisodeSha256],["per_query.csv SHA-256",data.integrity.perQuerySha256],["Analysis manifest SHA-256",data.integrity.analysisManifestSha256],["Analyzer SHA-256",data.integrity.analyzerSha256],["Validator SHA-256",data.integrity.validatorSha256]];integrity.forEach((row)=>{const dt=document.createElement("dt"),dd=document.createElement("dd");dt.textContent=row[0];dd.textContent=String(row[1]);byId("integrity").append(dt,dd);});
      byId("boundary").textContent=data.claimBoundary;
    })();
  </script>
</body>
</html>
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("source_artifact", type=pathlib.Path)
    parser.add_argument("analysis_root", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--open", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_dashboard(args.source_artifact, args.analysis_root, args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"valid": True, **result}, indent=2, ensure_ascii=False))
    if args.open:
        webbrowser.open(pathlib.Path(result["output"]).as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
