from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.source_archetype_v2_separated_learnability_report import (
    _aggregate,
    _selection_summary,
)


def test_selection_summary_reports_side_concentration() -> None:
    metrics = pd.DataFrame(
        {
            "u_policy_net": [0.02, 0.01, -0.01, 0.03],
            "mae_norm": [0.2, 0.4, 1.2, 0.3],
            "barrier": [0.01, 0.015, 0.03, 0.012],
            "is_timeout": [False, False, True, False],
            "side": [1, -1, -1, -1],
        }
    )
    target = pd.DataFrame(
        {
            "target_soft": [0.1, 0.2, 0.9, 0.2],
            "target_hard": [0.0, 0.0, 1.0, 0.0],
        }
    )

    summary = _selection_summary(
        metrics=metrics,
        target=target,
        score=pd.Series([0.1, 0.2, 0.9, 0.3]),
        idx=np.array([0, 1, 3]),
    )

    assert summary["top_side"] == "short"
    assert summary["top_side_share"] == 2 / 3
    assert summary["short_share"] == 2 / 3


def test_aggregate_carries_max_side_concentration() -> None:
    monthly = pd.DataFrame(
        [
            {
                "period": "2026-04",
                "head": "timeout_holding_archetype_score",
                "head_kind": "risk",
                "direction": "risk",
                "feature_set": "base_plus_v2",
                "selector": "low_score_keep",
                "top_frac": 0.5,
                "score_ic_target": 0.04,
                "score_ic_u": 0.01,
                "score_ic_timeout": 0.04,
                "score_ic_bad_mae": 0.01,
                "score_ic_wide_barrier": 0.01,
                "target_auc": 0.55,
                "target_pr_auc_lift": 1.2,
                "target_rate_lift_vs_valid": 0.8,
                "timeout_reduction_frac_vs_valid": 0.12,
                "mean_u": 0.01,
                "delta_mean_u_vs_valid": 0.001,
                "bad_mae_1r_rate": 0.2,
                "wide_barrier_25bps_rate": 0.1,
                "timeout_rate": 0.08,
                "selected_rows": 30,
                "valid_target_hard_rate": 0.2,
                "top_side_share": 0.7,
            },
            {
                "period": "2026-05",
                "head": "timeout_holding_archetype_score",
                "head_kind": "risk",
                "direction": "risk",
                "feature_set": "base_plus_v2",
                "selector": "low_score_keep",
                "top_frac": 0.5,
                "score_ic_target": 0.05,
                "score_ic_u": 0.02,
                "score_ic_timeout": 0.05,
                "score_ic_bad_mae": 0.01,
                "score_ic_wide_barrier": 0.01,
                "target_auc": 0.56,
                "target_pr_auc_lift": 1.3,
                "target_rate_lift_vs_valid": 0.7,
                "timeout_reduction_frac_vs_valid": 0.15,
                "mean_u": 0.02,
                "delta_mean_u_vs_valid": 0.002,
                "bad_mae_1r_rate": 0.1,
                "wide_barrier_25bps_rate": 0.08,
                "timeout_rate": 0.06,
                "selected_rows": 40,
                "valid_target_hard_rate": 0.2,
                "top_side_share": 0.9,
            },
        ]
    )

    aggregate = _aggregate(monthly, expected_months=2)

    assert "max_side_top_share" in aggregate.columns
    assert float(aggregate["max_side_top_share"].iloc[0]) == 0.9
