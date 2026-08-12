import numpy as np
import pandas as pd

from extreme_price_movements.base_relationship_drift import (
    adjacent_month_fixed_bin_decomposition,
    monthly_relationship_metrics,
)


def _panel() -> pd.DataFrame:
    rows = []
    for month, shift in (("2024-01", 0.0), ("2024-02", 1.0)):
        for side in ("long", "short"):
            for hour in range(2):
                for candidate in range(10):
                    score = float(candidate)
                    outcome = score + shift
                    rows.append({
                        "candidate_id": f"{month}-{side}-{hour}-{candidate}",
                        "decision_ts": f"{month}-{hour + 1:02d}T00:00:00Z",
                        "side_name": side,
                        "score": score,
                        "outcome": outcome,
                        "winner": int(candidate >= 7),
                    })
    return pd.DataFrame(rows)


def test_monthly_metrics_are_query_local_and_order_invariant():
    panel = _panel()
    metrics, deciles = monthly_relationship_metrics(
        panel, score_col="score", target_col="outcome", winner_col="winner"
    )
    shuffled_metrics, shuffled_deciles = monthly_relationship_metrics(
        panel.sample(frac=1.0, random_state=7), score_col="score", target_col="outcome", winner_col="winner"
    )
    pooled = metrics.loc[metrics["scope"].eq("pooled")]
    assert np.allclose(pooled["within_query_rank_ic"], 1.0)
    assert np.allclose(pooled["top30_winner_recall"], 1.0)
    assert np.allclose(pooled["top40_winner_recall"], 1.0)
    assert (pooled["top5_uplift"] > 0.0).all()
    assert (pooled["decile_monotonicity"] > 0.99).all()
    pd.testing.assert_frame_equal(metrics.sort_index(axis=1).sort_values(list(metrics.columns)).reset_index(drop=True), shuffled_metrics.sort_index(axis=1).sort_values(list(shuffled_metrics.columns)).reset_index(drop=True))
    pd.testing.assert_frame_equal(deciles.sort_index(axis=1).sort_values(list(deciles.columns)).reset_index(drop=True), shuffled_deciles.sort_index(axis=1).sort_values(list(shuffled_deciles.columns)).reset_index(drop=True))


def test_adjacent_month_decomposition_identifies_pure_relationship_shift():
    panel = _panel()
    result = adjacent_month_fixed_bin_decomposition(panel, score_col="score", outcome_col="outcome")
    assert len(result) == 1
    row = result.iloc[0]
    assert abs(float(row["composition_effect"])) < 1e-12
    assert abs(float(row["relationship_effect"]) - 1.0) < 1e-12
    assert abs(float(row["decomposition_residual"])) < 1e-12


def test_adjacent_month_decomposition_identifies_pure_composition_shift():
    panel = _panel()
    panel.loc[panel["decision_ts"].str.startswith("2024-02"), "outcome"] = panel.loc[panel["decision_ts"].str.startswith("2024-02"), "score"]
    # Add extra high-score February candidates: relationship is identical, only
    # score-bin composition changes while every cell remains in common support.
    high = panel.loc[panel["decision_ts"].str.startswith("2024-02") & panel["score"].ge(5.0)].copy()
    high["candidate_id"] = "extra-" + high["candidate_id"]
    panel = pd.concat([panel, high], ignore_index=True)
    result = adjacent_month_fixed_bin_decomposition(panel, score_col="score", outcome_col="outcome")
    row = result.iloc[0]
    assert abs(float(row["relationship_effect"])) < 1e-12
    assert float(row["composition_effect"]) > 0.0
