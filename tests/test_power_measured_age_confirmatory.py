from __future__ import annotations

import json
import pathlib

import pytest

from scripts.power_measured_age_confirmatory import (
    DEFAULT_PAIR_COUNTS,
    SCENARIOS,
    Scenario,
    exact_mcnemar_p,
    exact_rejection_probability,
    main,
    run_power_analysis,
    serialize_report,
)


def test_exact_mcnemar_boundaries() -> None:
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(5, 0) == 0.0625
    assert exact_mcnemar_p(6, 0) == 0.03125
    assert exact_mcnemar_p(0, 6) == 0.03125


def test_exact_analysis_is_byte_reproducible() -> None:
    first = run_power_analysis()
    second = run_power_analysis()
    assert serialize_report(first) == serialize_report(second)


def test_frozen_power_artifact_matches_implementation() -> None:
    project_root = pathlib.Path(__file__).resolve().parents[1]
    frozen = json.loads(
        (project_root / "docs/research/pi05_measured_age_confirmatory_exact_power.json")
        .read_text(encoding="utf-8")
    )
    assert frozen == run_power_analysis()


def test_requested_matrix_contract() -> None:
    report = run_power_analysis()
    expected = {
        (scenario.scenario_id, pair_count)
        for scenario in SCENARIOS
        for pair_count in DEFAULT_PAIR_COUNTS
    }
    observed = {
        (row["scenario_id"], row["pair_count"]) for row in report["results"]
    }
    assert observed == expected
    assert len(report["results"]) == len(SCENARIOS) * len(DEFAULT_PAIR_COUNTS)
    assert report["calculation"]["monte_carlo"] is False
    for row in report["results"]:
        assert 0.0 <= row["exact_rejection_probability"] <= 1.0


def test_frozen_120_pair_sensitivity_values() -> None:
    report = run_power_analysis(pair_counts=[120])
    values = {
        row["scenario_id"]: row["exact_rejection_probability"]
        for row in report["results"]
    }
    assert values["pilot_informed_primary_alternative"] == pytest.approx(
        0.9035135562763672, abs=1e-12
    )
    assert values["pilot_informed_sensitivity_low"] == pytest.approx(
        0.8239372664261995, abs=1e-12
    )
    assert values["pilot_informed_sensitivity_high"] > 0.99
    assert values["symmetric_null_calibration"] < 0.05


def test_power_rises_with_sample_size_and_null_stays_conservative() -> None:
    report = run_power_analysis(pair_counts=[50, 120, 150])
    by_scenario = {}
    for row in report["results"]:
        by_scenario.setdefault(row["scenario_id"], {})[row["pair_count"]] = row
    for scenario in SCENARIOS:
        rates = {
            count: row["exact_rejection_probability"]
            for count, row in by_scenario[scenario.scenario_id].items()
        }
        if scenario.role == "power":
            assert rates[150] > rates[120] > rates[50]
        else:
            assert all(rate < 0.05 for rate in rates.values())


def test_zero_discordance_has_zero_power() -> None:
    scenario = Scenario("none", "power", 0.0, 0.0, "boundary")
    assert exact_rejection_probability(scenario, 120, 0.05) == 0.0


def test_cli_writes_valid_json(tmp_path) -> None:
    destination = tmp_path / "nested" / "power.json"
    assert main(
        [
            "--pair-counts",
            "50",
            "120",
            "--output",
            str(destination),
        ]
    ) == 0
    parsed = json.loads(destination.read_text(encoding="utf-8"))
    assert parsed["design"]["pair_counts"] == [50, 120]
    assert parsed["calculation"]["monte_carlo"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": 1.0}, "alpha"),
        ({"pair_counts": []}, "pair count"),
        ({"pair_counts": [0]}, "pair_count"),
        ({"pair_counts": [50, 50]}, "unique"),
    ],
)
def test_invalid_analysis_inputs_fail(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        run_power_analysis(**kwargs)


@pytest.mark.parametrize(
    "scenario",
    [
        Scenario("negative", "power", -0.1, 0.1, "invalid"),
        Scenario("too_large", "power", 0.8, 0.3, "invalid"),
        Scenario("bad_role", "other", 0.2, 0.05, "invalid"),
    ],
)
def test_invalid_scenarios_fail(scenario: Scenario) -> None:
    with pytest.raises(ValueError):
        run_power_analysis(pair_counts=[50], scenarios=[scenario])
