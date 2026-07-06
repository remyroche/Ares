from __future__ import annotations

import pandas as pd
import pytest

from scripts.report_conditional_gmm_archetypes import summarize_beneficial_pairs


def test_summarize_beneficial_pairs_uses_target_direction() -> None:
    pairs = pd.DataFrame(
        {
            "feature": ["trend_6h", "volatility_24h", "range_12h_pct"],
            "target": ["favorable_excursion", "bad_MAE", "timeout"],
            "family": ["momentum_trend", "volatility", "volatility"],
            "primary_category": ["global", "risk_tail", "risk_tail"],
            "pair_score": [0.8, 0.7, 0.6],
            "global_spearman_ic": [0.12, -0.08, 0.05],
            "mean_bucket_spearman_ic_shrunk": [0.10, -0.05, 0.02],
            "long_spearman_ic": [0.11, -0.07, 0.05],
            "short_spearman_ic": [0.10, -0.08, 0.04],
            "long_short_ic_difference": [0.01, 0.01, 0.01],
            "sign_flip_rate": [0.0, 0.0, 0.0],
            "bucket_ic_std": [0.01, 0.01, 0.01],
        }
    )

    beneficial, rollup, counts = summarize_beneficial_pairs(pairs)

    assert set(beneficial["feature"]) == {"trend_6h", "volatility_24h"}
    risk_row = beneficial[beneficial["feature"].eq("volatility_24h")].iloc[0]
    assert risk_row["beneficial_direction"] == "lower_is_better"
    assert float(risk_row["good_direction_ic"]) == pytest.approx(0.08)
    assert "range_12h_pct" not in set(rollup["feature"])
    assert set(counts["dimension"]) == {"target", "primary_category", "family"}
