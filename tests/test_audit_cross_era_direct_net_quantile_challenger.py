import pandas as pd
import numpy as np

from scripts.audit_cross_era_direct_net_quantile_challenger import (
    predictive_metrics,
    promotion_gates,
)


def _economics(historical: bool, current_positive: bool = False) -> pd.DataFrame:
    if historical:
        return pd.DataFrame(
            [
                {"level": "aggregate", "period": "all", "scope": "global", "net_ev_bps": 10.0},
                {"level": "month", "period": "2026-06", "scope": "global", "net_ev_bps": 5.0},
                {"level": "month", "period": "2026-07", "scope": "global", "net_ev_bps": 2.0},
            ]
        )
    value = 4.0 if current_positive else -4.0
    return pd.DataFrame(
        [
            {"level": "aggregate", "period": "all", "scope": "global", "net_ev_bps": value},
            {"level": "aggregate", "period": "all", "scope": "side_local_long", "net_ev_bps": value},
            {"level": "aggregate", "period": "all", "scope": "side_local_short", "net_ev_bps": value},
        ]
    )


def test_promotion_gates_fail_before_portfolio_when_current_is_negative():
    gates = promotion_gates(_economics(True), _economics(False, current_positive=False))
    assert gates["historical_global_top10_positive"] is True
    assert gates["current_global_top10_positive"] is False
    assert gates["pre_portfolio_gate_passed"] is False


def test_promotion_gates_require_all_economic_checks():
    gates = promotion_gates(_economics(True), _economics(False, current_positive=True))
    assert gates["pre_portfolio_gate_passed"] is True


def test_predictive_metrics_are_side_and_month_specific():
    rows = []
    for side in ("long", "short"):
        for index in range(5):
            net = float(index - 2)
            rows.append(
                {
                    "__ts__": pd.Timestamp("2026-07-01T00:00:00Z") + pd.Timedelta(days=index),
                    "side_name": side,
                    "execution_net_ev_12h": net / 1e4,
                    "q10_net_bps": net - 2,
                    "q25_net_bps": net - 1,
                    "q50_net_bps": net,
                    "q75_net_bps": net + 1,
                }
            )
    result = predictive_metrics(pd.DataFrame(rows), "test")
    assert set(result["side_name"]) == {"long", "short"}
    assert set(result["level"]) == {"aggregate", "month"}
    assert np.allclose(result["rank_ic"], 1.0)
