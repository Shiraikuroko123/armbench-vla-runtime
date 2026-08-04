from __future__ import annotations

from integrations.openpi.projected_overlap_pilot import (
    OVERLAP_UNCONDITIONED,
    PROJECTED_OVERLAP,
    _mcnemar_exact_p,
    build_cells,
    summarize,
)


def test_cells_are_paired_and_condition_order_alternates() -> None:
    cells = build_cells(
        "libero_10",
        [0, 1],
        [0, 1],
        execute_horizon=5,
        inference_delay_steps=4,
    )

    assert len(cells) == 8
    assert [cell.method for cell in cells[:4]] == [
        OVERLAP_UNCONDITIONED,
        PROJECTED_OVERLAP,
        PROJECTED_OVERLAP,
        OVERLAP_UNCONDITIONED,
    ]
    assert cells[0].pair_id == cells[1].pair_id
    assert cells[2].pair_id == cells[3].pair_id


def test_summary_uses_paired_success_and_nonbootstrap_seams() -> None:
    episodes = [
        {"pair_id": "a", "method": OVERLAP_UNCONDITIONED, "success": False, "policy_queries": 2},
        {"pair_id": "a", "method": PROJECTED_OVERLAP, "success": True, "policy_queries": 2},
        {"pair_id": "b", "method": OVERLAP_UNCONDITIONED, "success": True, "policy_queries": 3},
        {"pair_id": "b", "method": PROJECTED_OVERLAP, "success": True, "policy_queries": 3},
    ]
    queries = [
        {
            "method": method,
            "bootstrap": bootstrap,
            "seam_motion_l2": None if bootstrap else seam,
            "seam_gripper_abs": None if bootstrap else seam / 2,
            "max_model_residual": (0.0 if method == PROJECTED_OVERLAP and not bootstrap else None),
        }
        for method in (OVERLAP_UNCONDITIONED, PROJECTED_OVERLAP)
        for bootstrap, seam in ((True, 0.0), (False, 0.2))
    ]

    summary = summarize(episodes, queries)

    assert summary["paired_rollouts"] == 2
    assert summary["projected_wins"] == 1
    assert summary["unconditioned_wins"] == 0
    assert summary["ties"] == 1
    assert summary["projected_minus_unconditioned_success_rate"] == 0.5
    assert summary["methods"][PROJECTED_OVERLAP]["max_model_residual"] == 0.0


def test_exact_mcnemar_handles_no_discordance_and_one_sided_wins() -> None:
    assert _mcnemar_exact_p(0, 0) == 1.0
    assert _mcnemar_exact_p(5, 0) == 0.0625
