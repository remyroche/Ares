import pandas as pd

from scripts.run_portfolio_marginal_utility_ablation import (
    PORTFOLIO_HEADS,
    _apply_pressure_reallocation,
    _suggest_reallocation_params,
)


class _FakeTrial:
    def suggest_float(self, name, low, high):
        return (float(low) + float(high)) / 2.0

    def suggest_categorical(self, name, choices):
        return choices[0]


def test_reallocation_optuna_suggests_recent_hr_ev_per_head():
    params = _suggest_reallocation_params(_FakeTrial(), constrained=True)

    for head in PORTFOLIO_HEADS:
        assert f"recent_hr_head_weight__{head}" in params
        assert f"recent_hr_level_weight__{head}" in params
        assert f"recent_hr_ev_boost__{head}" in params
        assert f"recent_hr_ev_penalty__{head}" in params
        assert f"recent_hr_count_scale__{head}" in params
    assert "recent_hr_ev_penalty" not in params


def test_reallocation_applies_recent_hr_ev_independently_per_head():
    now = pd.Timestamp("2026-06-21T00:00:00Z")
    base = {
        "timestamp": now,
        "symbol": "BTC/USD:USD",
        "calibrated_score": 0.80,
        "normalized_rank_score": 0.80,
        "rank_pct": 0.80,
        "base_strategy_threshold": 0.70,
        "entry_price": 100.0,
        "exit_timestamp": now + pd.Timedelta(hours=1),
        "exit_price": 100.0,
        "net_return": 0.0,
        "gross_return": 0.0,
        "holding_bars": 1,
        "simple_policy_exit_reason": "timeout",
        "strategy_recent_hr_surprise": -0.10,
        "head_recent_hr_surprise": -0.10,
        "strategy_recent_hit_rate": 0.40,
        "head_recent_hit_rate": 0.40,
        "strategy_recent_trade_count": 20.0,
        "head_recent_trade_count": 20.0,
    }
    rows = []
    long_row = dict(base)
    long_row.update({"strategy_id": "long_bars_momentum", "strategy_head": "long_bars", "side": "long"})
    rows.append(long_row)
    short_row = dict(base)
    short_row.update(
        {
            "symbol": "ETH/USD:USD",
            "strategy_id": "short_asset_reversal",
            "strategy_head": "short_asset",
            "side": "short",
        }
    )
    rows.append(short_row)

    params = {
        "pressure_start": 0.0,
        "pressure_scale": 1.0,
        "top_cut": 0.90,
        "weak_cut": 0.10,
        "rank_gap_scale": 0.10,
        "priority_boost": 0.0,
        "priority_penalty": 0.0,
        "top_size_boost": 0.0,
        "weak_size_cut": 0.0,
        "weak_threshold_uplift": 0.0,
        "min_size_multiplier": 0.25,
        "max_size_multiplier": 1.0,
        "recent_hr_head_weight__long_bars": 1.0,
        "recent_hr_head_weight__short_asset": 1.0,
        "recent_hr_level_weight__long_bars": 0.0,
        "recent_hr_level_weight__short_asset": 0.0,
        "recent_hr_ev_boost__long_bars": 0.0,
        "recent_hr_ev_boost__short_asset": 0.0,
        "recent_hr_ev_penalty__long_bars": 0.0,
        "recent_hr_ev_penalty__short_asset": 2.0,
        "recent_hr_count_scale__long_bars": 4.0,
        "recent_hr_count_scale__short_asset": 4.0,
    }
    out = _apply_pressure_reallocation(
        pd.DataFrame(rows),
        pd.DataFrame(),
        params_dict=params,
        max_entries_per_bar=1,
    )

    long_delta = out.loc[
        out["strategy_id"].eq("long_bars_momentum"),
        "portfolio_recent_hr_ev_priority_delta",
    ].iloc[0]
    short_delta = out.loc[
        out["strategy_id"].eq("short_asset_reversal"),
        "portfolio_recent_hr_ev_priority_delta",
    ].iloc[0]

    assert long_delta == 0.0
    assert short_delta < -0.35
    assert set(out["portfolio_recent_hr_head"]) == {"long_bars", "short_asset"}
