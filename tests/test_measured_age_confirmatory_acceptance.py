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
    base = {
        "schema_version": "armbench.pi05_libero_measured_age_analysis.v1",
        "source": source,
        "cohort": {"pairs": 120, "rollouts": 240},
        "success": {"paired": {"rate_difference": 1.0 / 6.0}},
        "timing": [],
        "runtime_burden": [],
        "statistics": {"paired_test": "exact two-sided McNemar binomial test"},
        "claim_boundary": "simulation only",
    }
    confirmatory = {
        "schema_version": "armbench.measured_age_confirmatory_analysis.v1",
        "implementation": {
            "analyzer_sha256": "4" * 64,
            "base_analyzer_sha256": "5" * 64,
            "validator_sha256": "6" * 64,
        },
        "source": {
            **source,
            "base_analysis_canonical_sha256": acceptance._canonical_sha256(base),
        },
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
    return {"base": base, "confirmatory": confirmatory}


@pytest.fixture
def acceptance_fixture(tmp_path: pathlib.Path, monkeypatch):
    run_root = tmp_path / "run"
    source = run_root / "evaluation"
    source.mkdir(parents=True)
    base_root = tmp_path / "base"
    confirmatory_root = tmp_path / "confirmatory"
    analyses = _analysis()
    base = analyses["base"]
    analysis = analyses["confirmatory"]
    _write_json(base_root / "analysis.json", base)
    _write_json(base_root / "manifest.json", {})
    _write_json(confirmatory_root / "analysis.json", analysis)
    _write_json(confirmatory_root / "manifest.json", {})
    _write_json(
        source / "environment.json",
        {
            "armbench_git_commit": "1" * 40,
            "openpi_git_commit": "2" * 40,
            "server_metadata": {
                "armbench_server_attestation": {
                    "policy_loaded": True,
                    "policy_config": "pi05_libero",
                    "checkpoint_uri": "gs://openpi-assets/checkpoints/pi05_libero",
                    "checkpoint_content_sha256": "3" * 64,
                    "action_horizon": 10,
                    "model_action_dim": 32,
                    "openpi_tracked_clean": True,
                    "openpi_submodules_clean": True,
                    "openpi_commit": "2" * 40,
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
    monkeypatch.setattr(
        acceptance.base_analysis,
        "analyze_artifact",
        lambda *_args, **_kwargs: (copy.deepcopy(base), []),
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


def test_base_canonical_mismatch_fails_before_dashboard(
    acceptance_fixture, monkeypatch
) -> None:
    run_root, base_root, confirmatory_root, dashboard, _analysis_value = acceptance_fixture
    recorded = json.loads(
        (confirmatory_root / "analysis.json").read_text(encoding="utf-8")
    )
    recorded["source"]["base_analysis_canonical_sha256"] = "f" * 64
    _write_json(confirmatory_root / "analysis.json", recorded)
    monkeypatch.setattr(acceptance, "analyze_artifact", lambda _root: recorded)
    calls = []
    monkeypatch.setattr(acceptance, "build_dashboard", lambda *_args: calls.append(True))

    with pytest.raises(ValueError, match="base canonical hash mismatch"):
        acceptance.build_acceptance(run_root, base_root, confirmatory_root, dashboard)

    assert calls == []
    assert not dashboard.exists()


def test_invalid_cohort_fails_before_dashboard(
    acceptance_fixture, monkeypatch
) -> None:
    run_root, base_root, confirmatory_root, dashboard, analysis = acceptance_fixture
    changed = copy.deepcopy(analysis)
    changed["cohort"]["pairs"] = 119
    changed["cohort"]["rollouts"] = 238
    _write_json(confirmatory_root / "analysis.json", changed)
    monkeypatch.setattr(acceptance, "analyze_artifact", lambda _root: changed)
    calls = []
    monkeypatch.setattr(acceptance, "build_dashboard", lambda *_args: calls.append(True))

    with pytest.raises(ValueError, match="frozen 10/120/240 matrix"):
        acceptance.build_acceptance(run_root, base_root, confirmatory_root, dashboard)

    assert calls == []
    assert not dashboard.exists()


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


def test_cli_failure_and_no_open_never_open_browser(
    acceptance_fixture, monkeypatch, capsys
) -> None:
    run_root, base_root, confirmatory_root, dashboard, _analysis_value = acceptance_fixture
    opened = []
    monkeypatch.setattr(acceptance.webbrowser, "open", opened.append)
    monkeypatch.setattr(
        acceptance,
        "build_acceptance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("failed")),
    )

    code = acceptance.main(
        [
            "--run-root", str(run_root),
            "--base-analysis-root", str(base_root),
            "--confirmatory-analysis-root", str(confirmatory_root),
            "--dashboard", str(dashboard),
        ]
    )
    assert code == 2
    assert json.loads(capsys.readouterr().out)["valid"] is False
    assert opened == []

    monkeypatch.setattr(
        acceptance,
        "build_acceptance",
        lambda *_args, **_kwargs: {
            "valid": True,
            "dashboard_uri": dashboard.resolve().as_uri(),
        },
    )
    assert acceptance.main(["--no-open"]) == 0
    assert opened == []
