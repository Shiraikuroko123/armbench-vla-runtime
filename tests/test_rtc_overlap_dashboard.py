from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

import pytest

from integrations.openpi import rtc_overlap_analysis, rtc_overlap_dashboard


METHODS = tuple(rtc_overlap_analysis.METHODS)
BASELINE = rtc_overlap_analysis.BASELINE
PROJECTED, RTC = tuple(rtc_overlap_analysis.CANDIDATES)


def _write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_analysis_manifest(root: pathlib.Path) -> None:
    files = {}
    for relative in ("analysis.json", "per_episode.csv", "summary.md"):
        path = root / relative
        files[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    _write_json(
        root / "manifest.json",
        {
            "schema_version": rtc_overlap_analysis.ANALYSIS_SCHEMA_VERSION,
            "files": files,
        },
    )


def _analysis(raw_summary: dict[str, Any], source_label: str) -> dict[str, Any]:
    methods = {}
    motion_methods = {}
    gripper_methods = {}
    for index, method in enumerate(METHODS):
        successes = 20 if method == BASELINE else 19
        methods[method] = {
            "successes": successes,
            "rollouts": 20,
            "rate": successes / 20.0,
            "wilson95_low": 0.75,
            "wilson95_high": 1.0,
        }
        motion_methods[method] = {
            "mean": 0.10 - index * 0.01,
            "median": 0.09 - index * 0.01,
            "scored_transitions": 1000 + index,
        }
        gripper_methods[method] = {
            "mean": 0.06 - index * 0.01,
            "median": 0.05 - index * 0.01,
            "scored_transitions": 1000 + index,
        }
    success_contrasts = {}
    motion_contrasts = {}
    gripper_contrasts = {}
    for candidate in (PROJECTED, RTC):
        success_contrasts[candidate] = {
            "pairs": 20,
            "rate_difference": -0.05,
            "task_block_bootstrap95_mean_low": -0.15,
            "task_block_bootstrap95_mean_high": 0.0,
            "candidate_wins": 0,
            "candidate_losses": 1,
            "ties": 19,
            "mcnemar_exact_two_sided_p": 1.0,
            "holm_adjusted_p": 1.0,
        }
        motion_contrasts[candidate] = {
            "paired_mean_difference": -0.02,
            "task_block_bootstrap95_mean_low": -0.03,
            "task_block_bootstrap95_mean_high": -0.01,
        }
        gripper_contrasts[candidate] = {
            "paired_mean_difference": -0.01,
            "task_block_bootstrap95_mean_low": -0.02,
            "task_block_bootstrap95_mean_high": 0.0,
        }
    return {
        "schema_version": rtc_overlap_analysis.ANALYSIS_SCHEMA_VERSION,
        "implementation": {
            "analyzer_source": "integrations/openpi/rtc_overlap_analysis.py",
            "analyzer_sha256": "a" * 64,
            "validator_source": "integrations/openpi/rtc_overlap_pilot.py",
            "validator_sha256": "b" * 64,
        },
        "source": {
            "artifact": source_label,
            "source_schema_version": "armbench.pi05_rtc_overlap_pilot.v2",
            "manifest_sha256": "c" * 64,
            "episodes_sha256": "d" * 64,
            "queries_sha256": "e" * 64,
            "validator_summary_sha256": _canonical_sha256(raw_summary),
        },
        "cohort": {
            "rollouts": 60,
            "paired_triplets": 20,
            "tasks": 10,
            "queries": 3146,
            "execute_horizon": 5,
            "inference_delay_steps": 4,
        },
        "success": {
            "methods": methods,
            "contrasts_vs_unconditioned": success_contrasts,
        },
        "seam": {
            "seam_motion_l2": {
                "methods": motion_methods,
                "contrasts_vs_unconditioned": motion_contrasts,
            },
            "seam_gripper_abs": {
                "methods": gripper_methods,
                "contrasts_vs_unconditioned": gripper_contrasts,
            },
        },
        "statistics": {
            "success_rate_interval": "Wilson score interval",
            "paired_success_test": "exact two-sided McNemar binomial test",
            "bootstrap_unit": "whole LIBERO task retaining both episode indices",
            "bootstrap_resamples": 10000,
            "multiplicity_adjustment": "Holm across two prespecified contrasts",
        },
        "claim_boundary": (
            "Simulation-only 10-task pilot; descriptive task-block intervals do not "
            "estimate general VLA performance."
        ),
    }


def _fixture(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    task_description: str = "put the objects in the basket",
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, dict[str, Any], list[str]]:
    evaluation = tmp_path / "evaluation"
    analysis_root = tmp_path / "analysis"
    evaluation.mkdir(parents=True)
    analysis_root.mkdir()
    videos = evaluation / "videos"
    videos.mkdir()
    raw_summary = {"complete_triplets": 20, "pilot_only": True}
    analysis = _analysis(raw_summary, evaluation.name)
    episodes = []
    bootstrap_queries = []
    rows = []
    for task_id in range(10):
        for episode_index in range(2):
            pair_id = "libero_10__task_%02d__episode_%02d" % (
                task_id,
                episode_index,
            )
            row: dict[str, Any] = {
                "pair_id": pair_id,
                "task_id": task_id,
                "episode_index": episode_index,
            }
            state_hash = hashlib.sha256(pair_id.encode("ascii")).hexdigest()
            for method_index, method in enumerate(METHODS):
                success = not (
                    (method == PROJECTED and (task_id, episode_index) == (8, 1))
                    or (method == RTC and (task_id, episode_index) == (4, 0))
                )
                query_count = 50 + method_index
                video_name = "%s__%s__%s.mp4" % (
                    pair_id,
                    method,
                    "success" if success else "failure",
                )
                (videos / video_name).write_bytes((pair_id + method).encode("ascii"))
                episodes.append(
                    {
                        "pair_id": pair_id,
                        "task_suite": "libero_10",
                        "task_id": task_id,
                        "episode_index": episode_index,
                        "method": method,
                        "success": success,
                        "policy_queries": query_count,
                        "condition_order": method_index,
                        "termination_reason": "task_success"
                        if success
                        else "step_limit",
                        "task_description": task_description,
                        "initial_state_sha256": state_hash,
                        "video_path": "videos/" + video_name,
                    }
                )
                bootstrap_queries.append(
                    {
                        "pair_id": pair_id,
                        "method": method,
                        "query_index": 0,
                        "bootstrap": True,
                        "policy_input_sha256": hashlib.sha256(
                            (pair_id + "/policy-input").encode("ascii")
                        ).hexdigest(),
                        "response_action_sha256": hashlib.sha256(
                            (pair_id + "/response").encode("ascii")
                        ).hexdigest(),
                        "sampling_key_sha256": hashlib.sha256(
                            (pair_id + "/key").encode("ascii")
                        ).hexdigest(),
                        "sampling_noise_sha256": hashlib.sha256(
                            (pair_id + "/noise").encode("ascii")
                        ).hexdigest(),
                    }
                )
                row["%s_success" % method] = int(success)
                row["%s_policy_queries" % method] = query_count
                row["%s_scored_transition_queries" % method] = query_count - 1
                row["%s_condition_order" % method] = method_index
                row["%s_seam_motion_l2" % method] = (
                    0.1 + task_id * 0.001 - method_index * 0.01
                )
                row["%s_seam_gripper_abs" % method] = (
                    0.05 + task_id * 0.001 - method_index * 0.005
                )
            rows.append(row)
    _write_json(evaluation / "episodes.json", episodes)
    _write_json(evaluation / "queries.json", bootstrap_queries)
    _write_json(
        evaluation / "environment.json",
        {
            "armbench_commit": "f" * 40,
            "server_attestation": {
                "policy_config": "pi05_libero",
                "checkpoint_uri": "gs://openpi-assets/checkpoints/pi05_libero",
                "checkpoint_content_sha256": "1" * 64,
                "openpi_commit": "2" * 40,
            },
        },
    )
    _write_json(analysis_root / "analysis.json", analysis)
    (analysis_root / "per_episode.csv").write_text("fixture\n", encoding="utf-8")
    (analysis_root / "summary.md").write_text("# fixture\n", encoding="utf-8")
    _write_analysis_manifest(analysis_root)
    calls: list[str] = []

    def validate(path: pathlib.Path) -> dict[str, Any]:
        calls.append("raw")
        assert pathlib.Path(path).resolve() == evaluation.resolve()
        return raw_summary

    def analyze(path: pathlib.Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        calls.append("analyze")
        assert pathlib.Path(path).resolve() == evaluation.resolve()
        return analysis, rows

    monkeypatch.setattr(
        rtc_overlap_dashboard.rtc_overlap_pilot, "validate_artifact", validate
    )
    monkeypatch.setattr(
        rtc_overlap_dashboard.rtc_overlap_analysis, "analyze_artifact", analyze
    )
    return (
        evaluation,
        analysis_root,
        tmp_path / "reports" / "index.html",
        analysis,
        calls,
    )


def _payload(html: str) -> dict[str, Any]:
    start = html.index('<script id="armbench-data" type="application/json">')
    start = html.index(">", start) + 1
    end = html.index("</script>", start)
    return json.loads(html[start:end])


def test_builds_20_triplet_60_video_dashboard_with_relative_links(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation, analysis_root, output, _analysis_value, calls = _fixture(
        tmp_path, monkeypatch
    )

    result = rtc_overlap_dashboard.build_dashboard(evaluation, analysis_root, output)

    assert calls == ["raw", "analyze"]
    assert result == {
        "output": str(output.resolve()),
        "rollouts": 60,
        "triplets": 20,
        "videos_verified": 60,
        "tasks": 10,
    }
    html = output.read_text(encoding="utf-8")
    payload = _payload(html)
    assert payload["schemaVersion"] == "armbench.pi05_rtc_overlap_dashboard.v3"
    assert len(payload["triplets"]) == 20
    assert sum(len(item["methods"]) for item in payload["triplets"]) == 60
    links = [
        method["video"]
        for triplet in payload["triplets"]
        for method in triplet["methods"]
    ]
    assert len(set(links)) == 60
    assert all(not link.startswith(("/", "file:", "http:")) for link in links)
    assert all(":\\" not in link for link in links)
    assert 'aria-label="Play all three videos"' in html
    assert "autoplay" not in html.lower()
    assert html.count('"pairId":') == 20
    assert html.count('"video":') == 60


def test_embedded_json_escapes_script_terminators(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    malicious = '</script><script>alert("rtc")</script>&'
    evaluation, analysis_root, output, _analysis_value, _calls = _fixture(
        tmp_path,
        monkeypatch,
        task_description=malicious,
    )

    rtc_overlap_dashboard.build_dashboard(evaluation, analysis_root, output)

    html = output.read_text(encoding="utf-8")
    assert malicious not in html
    assert "\\u003c/script\\u003e" in html
    assert _payload(html)["triplets"][0]["taskDescription"] == malicious


def test_tampered_saved_analysis_fails_closed_without_replacing_output(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation, analysis_root, output, _analysis_value, calls = _fixture(
        tmp_path, monkeypatch
    )
    output.parent.mkdir(parents=True)
    output.write_text("previous verified report", encoding="utf-8")
    with (analysis_root / "analysis.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")

    with pytest.raises(ValueError, match="analysis file size mismatch"):
        rtc_overlap_dashboard.build_dashboard(evaluation, analysis_root, output)

    assert calls == ["raw", "analyze"]
    assert output.read_text(encoding="utf-8") == "previous verified report"


def test_manifest_valid_but_stale_analysis_disagrees_with_recompute(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation, analysis_root, output, _analysis_value, _calls = _fixture(
        tmp_path, monkeypatch
    )
    saved = json.loads((analysis_root / "analysis.json").read_text(encoding="utf-8"))
    saved["claim_boundary"] = "tampered but rehashed"
    _write_json(analysis_root / "analysis.json", saved)
    _write_analysis_manifest(analysis_root)

    with pytest.raises(ValueError, match="disagrees with a fresh analyze_artifact"):
        rtc_overlap_dashboard.build_dashboard(evaluation, analysis_root, output)

    assert not output.exists()


def test_raw_validator_is_first_and_missing_video_fails_closed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation, analysis_root, output, _analysis_value, calls = _fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        rtc_overlap_dashboard.rtc_overlap_pilot,
        "validate_artifact",
        lambda _path: (_ for _ in ()).throw(ValueError("raw hash mismatch")),
    )

    with pytest.raises(ValueError, match="failed independent validation"):
        rtc_overlap_dashboard.build_dashboard(evaluation, analysis_root, output)
    assert calls == []
    assert not output.exists()

    evaluation, analysis_root, output, _analysis_value, calls = _fixture(
        tmp_path / "missing", monkeypatch
    )
    next((evaluation / "videos").iterdir()).unlink()
    with pytest.raises(ValueError, match="required video is missing or empty"):
        rtc_overlap_dashboard.build_dashboard(evaluation, analysis_root, output)
    assert calls == ["raw", "analyze"]
    assert not output.exists()


@pytest.mark.parametrize(
    "field",
    [
        "policy_input_sha256",
        "response_action_sha256",
        "sampling_key_sha256",
        "sampling_noise_sha256",
    ],
)
def test_mismatched_bootstrap_identity_fails_closed_and_preserves_old_output(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    evaluation, analysis_root, output, _analysis_value, calls = _fixture(
        tmp_path, monkeypatch
    )
    output.parent.mkdir(parents=True)
    output.write_text("previous verified report", encoding="utf-8")
    queries_path = evaluation / "queries.json"
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    target = next(
        row
        for row in queries
        if row["pair_id"] == "libero_10__task_00__episode_00" and row["method"] == RTC
    )
    target[field] = "9" * 64
    _write_json(queries_path, queries)

    with pytest.raises(ValueError, match=field + " differs across methods"):
        rtc_overlap_dashboard.build_dashboard(evaluation, analysis_root, output)

    assert calls == ["raw", "analyze"]
    assert output.read_text(encoding="utf-8") == "previous verified report"


def test_missing_bootstrap_method_fails_closed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation, analysis_root, output, _analysis_value, _calls = _fixture(
        tmp_path, monkeypatch
    )
    queries_path = evaluation / "queries.json"
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    queries.pop()
    _write_json(queries_path, queries)

    with pytest.raises(ValueError, match="bootstrap query-zero matrix is incomplete"):
        rtc_overlap_dashboard.build_dashboard(evaluation, analysis_root, output)

    assert not output.exists()


def test_windows_entrypoint_is_cwd_independent_and_supports_no_open() -> None:
    script = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts"
        / "rtc_pilot_acceptance.cmd"
    ).read_text(encoding="utf-8")

    assert "%~dp0.." in script
    assert "..\\.venv\\Scripts\\python.exe" in script
    assert "-NoOpen" in script
    assert 'pushd "%PROJECT_DIR%"' in script
    assert "-m integrations.openpi.rtc_overlap_dashboard" in script
