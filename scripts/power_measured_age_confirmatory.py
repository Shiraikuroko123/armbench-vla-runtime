"""Exact McNemar power planning for the measured-age confirmatory study.

The planning unit is a matched rollout pair.  Candidate-only, reference-only,
and concordant outcomes follow a multinomial distribution.  Enumerating the
discordant count and its conditional split gives the exact rejection
probability of the same two-sided McNemar test used by the study.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence


SCHEMA_VERSION = "armbench.pi05_measured_age_exact_mcnemar_power.v1"
DEFAULT_ALPHA = 0.05
DEFAULT_PAIR_COUNTS = (50, 75, 100, 120, 125, 150)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    role: str
    candidate_only_probability: float
    reference_only_probability: float
    assumption_basis: str

    @property
    def concordant_probability(self) -> float:
        return 1.0 - (
            self.candidate_only_probability + self.reference_only_probability
        )


SCENARIOS = (
    Scenario(
        scenario_id="pilot_informed_primary_alternative",
        role="power",
        candidate_only_probability=0.20,
        reference_only_probability=0.05,
        assumption_basis=(
            "pilot-informed paired alternative with a +0.15 smallest effect "
            "of interest and 0.25 total discordance"
        ),
    ),
    Scenario(
        scenario_id="pilot_informed_sensitivity_low",
        role="power",
        candidate_only_probability=0.18,
        reference_only_probability=0.05,
        assumption_basis="lower pilot-informed sensitivity alternative",
    ),
    Scenario(
        scenario_id="pilot_informed_sensitivity_high",
        role="power",
        candidate_only_probability=0.22,
        reference_only_probability=0.03,
        assumption_basis="upper pilot-informed sensitivity alternative",
    ),
    Scenario(
        scenario_id="symmetric_null_calibration",
        role="type_i_error",
        candidate_only_probability=0.10,
        reference_only_probability=0.10,
        assumption_basis="symmetric null used to expose test discreteness",
    ),
)


def _require_integer(value: int, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("%s must be an integer" % label)
    if value < minimum:
        raise ValueError("%s must be at least %d" % (label, minimum))
    return value


def _validate_probability(value: float, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be numeric" % label) from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("%s must be finite and in [0, 1]" % label)
    return result


def validate_scenario(scenario: Scenario) -> None:
    candidate = _validate_probability(
        scenario.candidate_only_probability, "candidate_only_probability"
    )
    reference = _validate_probability(
        scenario.reference_only_probability, "reference_only_probability"
    )
    if candidate + reference > 1.0:
        raise ValueError("discordant probabilities must sum to at most 1")
    if scenario.role not in ("power", "type_i_error"):
        raise ValueError("scenario role must be power or type_i_error")
    if not scenario.scenario_id:
        raise ValueError("scenario_id must not be empty")


def exact_mcnemar_p(candidate_only: int, reference_only: int) -> float:
    """Return the doubled smaller binomial tail for discordant paired counts."""

    candidate = _require_integer(candidate_only, "candidate_only")
    reference = _require_integer(reference_only, "reference_only")
    discordant = candidate + reference
    if discordant == 0:
        return 1.0
    tail = min(candidate, reference)
    cumulative_mass = sum(math.comb(discordant, index) for index in range(tail + 1))
    return min(1.0, 2.0 * cumulative_mass / float(2**discordant))


def _binomial_mass(successes: int, total: int, probability: float) -> float:
    return (
        math.comb(total, successes)
        * probability**successes
        * (1.0 - probability) ** (total - successes)
    )


def exact_rejection_probability(
    scenario: Scenario, pair_count: int, alpha: float
) -> float:
    """Enumerate the exact rejection probability under one paired alternative."""

    validate_scenario(scenario)
    pairs = _require_integer(pair_count, "pair_count", minimum=1)
    significance = _validate_probability(alpha, "alpha")
    if significance in (0.0, 1.0):
        raise ValueError("alpha must be in (0, 1)")

    discordance_probability = (
        scenario.candidate_only_probability
        + scenario.reference_only_probability
    )
    if discordance_probability == 0.0:
        return 0.0
    conditional_candidate = (
        scenario.candidate_only_probability / discordance_probability
    )
    rejection_probability = 0.0
    for discordant in range(pairs + 1):
        discordant_mass = _binomial_mass(
            discordant, pairs, discordance_probability
        )
        if discordant_mass == 0.0:
            continue
        conditional_rejection = 0.0
        for candidate_only in range(discordant + 1):
            reference_only = discordant - candidate_only
            if exact_mcnemar_p(candidate_only, reference_only) <= significance:
                conditional_rejection += _binomial_mass(
                    candidate_only, discordant, conditional_candidate
                )
        rejection_probability += discordant_mass * conditional_rejection
    return min(1.0, max(0.0, rejection_probability))


def run_power_analysis(
    *,
    alpha: float = DEFAULT_ALPHA,
    pair_counts: Iterable[int] = DEFAULT_PAIR_COUNTS,
    scenarios: Sequence[Scenario] = SCENARIOS,
) -> Dict[str, Any]:
    try:
        significance = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be numeric") from exc
    if not math.isfinite(significance) or not 0.0 < significance < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    pairs = tuple(_require_integer(value, "pair_count", minimum=1) for value in pair_counts)
    if not pairs:
        raise ValueError("at least one pair count is required")
    if len(set(pairs)) != len(pairs):
        raise ValueError("pair counts must be unique")
    selected_scenarios = tuple(scenarios)
    if not selected_scenarios:
        raise ValueError("at least one scenario is required")
    for scenario in selected_scenarios:
        validate_scenario(scenario)
    if len({scenario.scenario_id for scenario in selected_scenarios}) != len(
        selected_scenarios
    ):
        raise ValueError("scenario IDs must be unique")

    results = [
        {
            "scenario_id": scenario.scenario_id,
            "metric": scenario.role,
            "pair_count": pair_count,
            "exact_rejection_probability": exact_rejection_probability(
                scenario, pair_count, significance
            ),
        }
        for scenario in selected_scenarios
        for pair_count in pairs
    ]
    scenario_records: List[Dict[str, Any]] = []
    for scenario in selected_scenarios:
        record = asdict(scenario)
        record["concordant_probability"] = scenario.concordant_probability
        record["total_discordance_probability"] = (
            scenario.candidate_only_probability
            + scenario.reference_only_probability
        )
        record["paired_success_difference"] = (
            scenario.candidate_only_probability
            - scenario.reference_only_probability
        )
        scenario_records.append(record)
    return {
        "schema_version": SCHEMA_VERSION,
        "design": {
            "analysis_unit": "matched rollout pair",
            "planned_test": "two-sided exact McNemar",
            "alpha": significance,
            "pair_counts": list(pairs),
            "primary_outcome": "paired LIBERO task success",
        },
        "calculation": {
            "method": (
                "exact enumeration of D~Binomial(n,p10+p01) and "
                "C|D~Binomial(D,p10/(p10+p01))"
            ),
            "monte_carlo": False,
        },
        "interpretation": (
            "Prospective sensitivity analysis under paired-outcome assumptions; "
            "it does not account for within-task clustering or estimate observed power."
        ),
        "scenarios": scenario_records,
        "results": results,
    }


def serialize_report(report: Dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enumerate exact McNemar power for the measured-age study."
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--pair-counts", type=int, nargs="+", default=list(DEFAULT_PAIR_COUNTS)
    )
    parser.add_argument("--output", type=pathlib.Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    report = run_power_analysis(alpha=args.alpha, pair_counts=args.pair_counts)
    payload = serialize_report(report)
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
