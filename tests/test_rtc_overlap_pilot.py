from __future__ import annotations

from integrations.openpi.rtc_overlap_pilot import OVERLAP_UNCONDITIONED
from integrations.openpi.rtc_overlap_pilot import PROJECTED_OVERLAP
from integrations.openpi.rtc_overlap_pilot import RTC_GUIDED_OVERLAP
from integrations.openpi.rtc_overlap_pilot import build_cells
from integrations.openpi.rtc_overlap_pilot import summarize


def test_three_arm_cells_use_latin_rotation() -> None:
    cells = build_cells(
        "libero_10",
        [0, 1, 2],
        [0],
        execute_horizon=5,
        inference_delay_steps=4,
    )

    assert len(cells) == 9
    assert [cell.method for cell in cells] == [
        OVERLAP_UNCONDITIONED,
        PROJECTED_OVERLAP,
        RTC_GUIDED_OVERLAP,
        PROJECTED_OVERLAP,
        RTC_GUIDED_OVERLAP,
        OVERLAP_UNCONDITIONED,
        RTC_GUIDED_OVERLAP,
        OVERLAP_UNCONDITIONED,
        PROJECTED_OVERLAP,
    ]
    for start in range(0, len(cells), 3):
        assert len({cell.pair_id for cell in cells[start : start + 3]}) == 1


def test_three_arm_summary_keeps_hard_and_soft_residuals_separate() -> None:
    episodes = []
    for pair_id, success in (("a", False), ("b", True)):
        episodes.extend(
            [
                {
                    "pair_id": pair_id,
                    "method": OVERLAP_UNCONDITIONED,
                    "success": success,
                    "policy_queries": 3,
                },
                {
                    "pair_id": pair_id,
                    "method": PROJECTED_OVERLAP,
                    "success": True,
                    "policy_queries": 3,
                },
                {
                    "pair_id": pair_id,
                    "method": RTC_GUIDED_OVERLAP,
                    "success": pair_id == "a",
                    "policy_queries": 3,
                },
            ]
        )
    queries = []
    for method in (
        OVERLAP_UNCONDITIONED,
        PROJECTED_OVERLAP,
        RTC_GUIDED_OVERLAP,
    ):
        queries.extend(
            [
                {
                    "method": method,
                    "bootstrap": True,
                    "seam_motion_l2": None,
                    "seam_gripper_abs": None,
                    "max_model_residual": None,
                    "weighted_model_rmse": None,
                },
                {
                    "method": method,
                    "bootstrap": False,
                    "seam_motion_l2": 0.2,
                    "seam_gripper_abs": 0.1,
                    "max_model_residual": 0.0 if method == PROJECTED_OVERLAP else None,
                    "weighted_model_rmse": 0.03 if method == RTC_GUIDED_OVERLAP else None,
                },
            ]
        )

    summary = summarize(episodes, queries)

    assert summary["complete_triplets"] == 2
    assert summary["methods"][PROJECTED_OVERLAP]["max_model_residual"] == 0.0
    assert summary["methods"][RTC_GUIDED_OVERLAP]["mean_weighted_model_rmse"] == 0.03
    assert summary["contrasts_vs_unconditioned"][PROJECTED_OVERLAP]["wins"] == 1
    assert summary["contrasts_vs_unconditioned"][RTC_GUIDED_OVERLAP]["wins"] == 1
