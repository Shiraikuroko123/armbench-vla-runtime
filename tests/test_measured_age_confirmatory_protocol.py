from __future__ import annotations

import json

from integrations.openpi import measured_age_libero_eval as evaluator


FROZEN_PLAN_ARGS = [
    "plan",
    "--task-suite",
    "libero_spatial",
    "--task-ids",
    "all",
    "--episode-indices",
    "5:17",
    "--modes",
    "async_unguarded,latency_aligned",
    "--replan-steps",
    "5",
    "--control-period-ms",
    "50",
    "--age-rounding",
    "ceil",
    "--deadline-ms",
    "250",
    "--max-age-refreshes",
    "2",
    "--jitter-values-ms",
    "0,40,80,160",
    "--warmup-queries",
    "3",
    "--warmup-task-id",
    "0",
    "--warmup-episode-index",
    "49",
    "--seed",
    "7",
]


def test_frozen_confirmatory_plan_is_disjoint_balanced_and_complete(capsys) -> None:
    assert evaluator.main(FROZEN_PLAN_ARGS) == 0
    plan = json.loads(capsys.readouterr().out)

    assert plan["schema_version"] == "armbench.pi05_libero_measured_age.v2"
    matrix = plan["matrix"]
    assert matrix["rollouts"] == 240
    assert matrix["matched_condition_groups"] == 120
    assert matrix["episode_indices"] == list(range(5, 17))

    cells = plan["registered_cells"]
    assert [cell["condition_order"] for cell in cells] == list(range(240))
    for task_id in range(10):
        task_cells = [cell for cell in cells if cell["task_id"] == task_id]
        assert len(task_cells) == 24
        pairs = [task_cells[index : index + 2] for index in range(0, 24, 2)]
        assert all(pair[0]["pair_id"] == pair[1]["pair_id"] for pair in pairs)
        assert all(
            {cell["mode"] for cell in pair}
            == {"async_unguarded", "latency_aligned"}
            for pair in pairs
        )
        first_modes = [pair[0]["mode"] for pair in pairs]
        assert first_modes.count("async_unguarded") == 6
        assert first_modes.count("latency_aligned") == 6


def test_frozen_confirmatory_noise_and_jitter_keys_exclude_mode(capsys) -> None:
    assert evaluator.main(FROZEN_PLAN_ARGS) == 0
    plan = json.loads(capsys.readouterr().out)

    assert plan["jitter"]["mode_in_key"] is False
    sampling = plan.get("policy_sampling")
    assert sampling is not None
    assert sampling["mode_in_key"] is False
    assert sampling["namespaces"]["warmup"] != sampling["namespaces"]["scored"]
