import csv
import json
from pathlib import Path

import pytest

from armbench.benchmark import (
    aggregate_planning,
    execute_benchmark,
    parse_seed_spec,
)


def test_parse_seed_spec_supports_ranges_and_lists() -> None:
    assert parse_seed_spec("0:4") == [0, 1, 2, 3]
    assert parse_seed_spec("1:7:2") == [1, 3, 5]
    assert parse_seed_spec("0,7,19") == [0, 7, 19]


@pytest.mark.parametrize("specification", ["", "3:3", "0,0", "-1"])
def test_parse_seed_spec_rejects_empty_duplicate_or_negative_values(
    specification: str,
) -> None:
    with pytest.raises(ValueError):
        parse_seed_spec(specification)


def test_planning_aggregate_includes_failures_in_latency() -> None:
    rows = [
        {
            "scenario": "test",
            "planner": "planner",
            "status": "success",
            "elapsed_ms": 10.0,
            "path_length": 2.0,
            "smoothed_length": 1.5,
            "collision_queries": 10,
            "edge_queries": 2,
        },
        {
            "scenario": "test",
            "planner": "planner",
            "status": "timeout",
            "elapsed_ms": 100.0,
            "path_length": None,
            "smoothed_length": None,
            "collision_queries": 100,
            "edge_queries": 20,
        },
    ]

    result = aggregate_planning(rows)[0]

    assert result["success_rate"] == 0.5
    assert result["latency_p50_ms"] == 55.0
    assert result["path_length_mean"] == 2.0


def test_minimal_experiment_writes_complete_artifact(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "configs" / "benchmark.json").open(
        encoding="utf-8"
    ) as handle:
        config = json.load(handle)
    config["scenarios"] = ["free_space"]
    config["seeds"] = [0]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    run_directory = execute_benchmark(
        config_path,
        tmp_path / "results",
        run_id="test_run",
        skip_control=True,
        make_figures=False,
    )

    expected = {
        "config.json",
        "environment.json",
        "per_trial.csv",
        "aggregate.json",
        "summary.md",
        "run.log",
    }
    assert expected.issubset(path.name for path in run_directory.iterdir())
    with (run_directory / "per_trial.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(row["status"] == "success" for row in rows)
    failures = list((run_directory / "failures").glob("diagnostic__*.json"))
    assert len(failures) == 3

