from __future__ import annotations

import copy
import json
import pathlib

import pytest

from integrations.openpi import measured_age_confirmatory_acceptance as acceptance


def _write_json(path: pathlib.Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def _analysis() -> dict:
    source = {
        "source_manifest_sha256": "a" * 64,
        "per_episode_sha256": "b" * 64,
        "per_query_sha256": "c" * 64,
    }
    return {
        "schema_version": "armbench.measured_age_confirmatory_analysis.v1",
        "source": source,
        "cohort": {"tasks": 10, "pairs": 120, "rollouts": 240},
        "primary": {
            "async_successes": 80,
            "aligned_successes": 100,
            "pairs": 120,
            "candidate_wins": 22,
            "reference_wins": 2,
            "ties": 96,
            "risk_difference": 1.0 / 6.0,
            "exact_p": 0.000277,
        },
        "secondary": {
            "task_cluster_bootstrap": {
                "percentile95_low": 0.08,
                "percentile95_high": 0.25,
            },
            "exact_task_sign_flip": {"exact_p": 0.015625},
            "leave_one_task_out_range": {
                "minimum_risk_difference": 0.14,
                "maximum_risk_difference": 0.19,
            },
        },
        "statistics": {"primary_test": "pooled two-sided exact McNemar"},
        "claim_boundary": "simulation only",
    }


@pytest.fixture
def acceptance_fixture(tmp_path: pathlib.Path, monkeypatch):
    run_root = tmp_path / "run"
    source = run_root / "evaluation"
    source.mkdir(parents=True)
    base_root = tmp_path / "base"
    confirmatory_root = tmp_path / "confirmatory"
    analysis = _analysis()
    _write_json(base_root / "analysis.json", {"source": analysis["source"]})
    _write_json(confirmatory_root / "analysis.json", analysis)
    _write_json(
        source / "environment.json",
        {
            "armbench_git_commit": "1" * 40,
            "openpi_git_commit": "2" * 40,
            "server_metadata": {
                "armbench_server_attestation": {
                    "policy_loaded": True,
                    "checkpoint_content_sha256": "3" * 64,
                },
                "armbench_policy_sampling_contract": {
                    "schema_version": "armbench.policy_sampling.v1",
                    "noise_shape": [10, 32],
                    "mode_in_key": False,
                },
            },
        },
    )
    monkeypatch.setattr(
        acceptance,
        "validate_run_manifest",
        lambda _root: {
            "valid": True,
            "complete": True,
            "errors": [],
            "files_checked": 273,
        },
    )
    monkeypatch.setattr(
        acceptance,
        "build_dashboard",
        lambda *_args: {"pairs": 120, "rollouts": 240, "videos_verified": 240},
    )
    monkeypatch.setattr(
        acceptance, "validate_report", lambda _root: {"valid": True, "errors": []}
    )
    monkeypatch.setattr(
        acceptance, "analyze_artifact", lambda _root: copy.deepcopy(analysis)
    )
    return run_root, base_root, confirmatory_root, tmp_path / "index.html", analysis


def test_acceptance_recomputes_and_reports_frozen_evidence(acceptance_fixture) -> None:
    run_root, base_root, confirmatory_root, dashboard, _analysis_value = acceptance_fixture

    report = acceptance.build_acceptance(
        run_root, base_root, confirmatory_root, dashboard
    )

    assert report["valid"] is True
    assert report["primary_positive"] is True
    assert report["evidence"] == {
        "rollouts": 240,
        "pairs": 120,
        "videos_verified": 240,
        "root_files_checked": 273,
    }
    assert report["identity"]["paired_policy_noise"] is True
    assert report["task_sensitivity"]["exact_task_sign_flip_p"] == 0.015625


def test_changed_scientific_result_fails_recomputation(
    acceptance_fixture, monkeypatch
) -> None:
    run_root, base_root, confirmatory_root, dashboard, analysis = acceptance_fixture
    changed = copy.deepcopy(analysis)
    changed["primary"]["aligned_successes"] = 99
    monkeypatch.setattr(acceptance, "analyze_artifact", lambda _root: changed)

    with pytest.raises(ValueError, match="disagrees with source: primary"):
        acceptance.build_acceptance(run_root, base_root, confirmatory_root, dashboard)


def test_invalid_report_and_sampling_contract_fail_closed(
    acceptance_fixture, monkeypatch
) -> None:
    run_root, base_root, confirmatory_root, dashboard, _analysis_value = acceptance_fixture
    monkeypatch.setattr(
        acceptance,
        "validate_report",
        lambda _root: {"valid": False, "errors": ["tampered"]},
    )
    with pytest.raises(ValueError, match="tampered"):
        acceptance.build_acceptance(run_root, base_root, confirmatory_root, dashboard)

    monkeypatch.setattr(
        acceptance, "validate_report", lambda _root: {"valid": True, "errors": []}
    )
    environment_path = run_root / "evaluation" / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["server_metadata"]["armbench_policy_sampling_contract"][
        "mode_in_key"
    ] = True
    _write_json(environment_path, environment)
    with pytest.raises(ValueError, match="paired-noise contract"):
        acceptance.build_acceptance(run_root, base_root, confirmatory_root, dashboard)


def test_cli_opens_only_after_success(
    acceptance_fixture, monkeypatch, capsys
) -> None:
    run_root, base_root, confirmatory_root, dashboard, _analysis_value = acceptance_fixture
    opened = []
    monkeypatch.setattr(acceptance.webbrowser, "open", opened.append)

    code = acceptance.main(
        [
            "--run-root", str(run_root),
            "--base-analysis-root", str(base_root),
            "--confirmatory-analysis-root", str(confirmatory_root),
            "--dashboard", str(dashboard),
        ]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
    assert opened == [dashboard.resolve().as_uri()]
