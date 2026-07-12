from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.hierarchical_state_validation import (
    annotate_base_decision_zones,
    state_validation_metrics,
)


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    timestamps = pd.date_range("2025-06-01", periods=12, freq="D", tz="UTC")
    specifications = (
        ("long", "long_mixed", 0.99, 0.015, 1.0, 0.0),
        ("short", "short_default", 0.98, -0.010, 0.0, 1.0),
        ("long", "long_mixed", 0.97, 0.010, 1.0, 0.0),
        ("short", "short_default", 0.96, -0.002, 0.0, 0.0),
        ("long", "long_mixed", 0.95, 0.001, 1.0, 0.0),
        ("short", "short_default", 0.94, -0.001, 0.0, 0.0),
        ("long", "long_mixed", 0.93, 0.001, 1.0, 0.0),
        ("short", "short_default", 0.92, -0.001, 0.0, 0.0),
        ("long", "long_mixed", 0.91, 0.001, 1.0, 0.0),
        ("short", "short_default", 0.90, -0.001, 0.0, 0.0),
    )
    for timestamp in timestamps:
        for rank, (side, archetype, score, ev, clean, bad) in enumerate(specifications):
            rows.append(
                {
                    "__ts__": timestamp,
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "score_base": score,
                    "ev_after_1pct": ev,
                    "clean_exec": clean,
                    "dirty_positive": 1.0 - clean,
                    "first_touch_bad_mae_1r": bad,
                    "timeout": 0.0,
                    "local_econ_aegmm_market_state_gmm_cluster_id": 0
                    if rank in (0, 2)
                    else 1,
                    "local_econ_aegmm_market_state_expected_top10_ev": 0.012
                    if rank in (0, 2)
                    else -0.008,
                    "local_econ_aegmm_market_state_expected_ev": 0.012
                    if rank in (0, 2)
                    else -0.008,
                    "local_econ_aegmm_market_state_expected_top10_bad_mae": 0.02
                    if rank in (0, 2)
                    else 0.80,
                    "local_econ_aegmm_market_state_expected_bad_mae": 0.02
                    if rank in (0, 2)
                    else 0.80,
                    "local_econ_aegmm_market_state_support_log1p": np.log1p(500.0),
                }
            )
    return pd.DataFrame(rows)


def test_validation_uses_global_base_book_zones() -> None:
    ranked = annotate_base_decision_zones(_frame())
    by_score = ranked.groupby("score_base", observed=True)[
        "state_validation_zone"
    ].first()
    assert str(by_score.loc[0.99]) == "incumbent_top10"
    assert str(by_score.loc[0.98]) == "incumbent_top10"
    assert str(by_score.loc[0.97]) == "near_miss_top10_20"
    assert str(by_score.loc[0.90]) == "outside"


def test_state_validation_reports_local_top10_and_near_miss_metrics() -> None:
    summary, states, daily, autocorr = state_validation_metrics(
        _frame(), fold="2025-06", state_block="market_state"
    )
    assert set(summary["zone"]) == {"incumbent_top10", "near_miss_top10_20"}
    incumbent = summary.loc[
        summary["zone"].eq("incumbent_top10") & summary["scope"].eq("overall")
    ].iloc[0]
    assert incumbent["mean_ev_after_1pct"] > 0.0
    assert incumbent["expected_ev_top_minus_bottom"] > 0.0
    assert not states.empty
    assert set(daily["zone"]) == {"incumbent_top10", "near_miss_top10_20"}
    assert set(autocorr["side_name"]) == {"long", "short"}
