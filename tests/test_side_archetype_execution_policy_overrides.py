import numpy as np
import pandas as pd

from scripts import ablate_simple_policy_exit_geometry as ablate
from scripts.validate_simple_policy_exit_geometry_walkforward import (
    _bundle_group_key,
    _regime_labels,
    _resolve_state_column,
)


def test_categorical_archetype_labels_are_preserved() -> None:
    rows = pd.DataFrame(
        {
            "policy_archetype": [
                "long_mixed_gmm_2",
                "liquidity-stress / avoid",
                None,
            ]
        }
    )

    labels = _regime_labels(
        rows,
        regime_column="policy_archetype",
        regime_edges=[],
        label_prefix="archetype",
    )

    assert labels.tolist() == [
        "archetype_long_mixed_gmm_2",
        "archetype_liquidity_stress_avoid",
        "archetype_missing",
    ]


def test_side_archetype_group_key_does_not_include_strategy_id() -> None:
    bundle = ablate.StrategyBundle(
        strategy_id="short_mean_reversion",
        rows=pd.DataFrame({"side": [-1.0]}),
        paths=(np.empty((1, 1)), np.empty((1, 1)), np.empty((1, 1)), np.empty((1, 1))),
        base_params={},
        base_threshold=0.70,
        best_size_power=1.0,
    )

    assert (
        _bundle_group_key(
            bundle,
            group_by="side_archetype",
            regime_label="archetype_noisy_breakout",
        )
        == "short|archetype_noisy_breakout"
    )


def test_side_archetype_column_resolver_prefers_available_archetype() -> None:
    rows = pd.DataFrame({"archetype": ["trend_pullback"], "oof_regime": [0.1]})

    assert (
        _resolve_state_column(
            rows,
            group_by="side_archetype",
            regime_column="oof_regime",
            archetype_column="policy_archetype",
        )
        == "archetype"
    )


def test_local_policy_overrides_materialize_replay_columns(monkeypatch) -> None:
    rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-04-01T00:00:00Z"], utc=True),
            "symbol": ["BTC-PERP"],
            "side": [-1.0],
            "rank_pct": [0.91],
            "calibrated_score": [0.4],
            "barrier_pct": [0.02],
        }
    )
    bundle = ablate.StrategyBundle(
        strategy_id="short_mean_reversion",
        rows=rows,
        paths=(np.empty((1, 1)), np.empty((1, 1)), np.empty((1, 1)), np.empty((1, 1))),
        base_params={"sl_mult": 1.0},
        base_threshold=0.70,
        best_size_power=1.0,
    )

    def fake_build_simple_policy_candidate_rows(**kwargs):
        assert kwargs["best_size_power"] == 1.75
        assert kwargs["base_strategy_threshold"] == 0.84
        assert "portfolio_size_multiplier" not in kwargs["best_params"]
        return pd.DataFrame(
            {
                "timestamp": rows["timestamp"],
                "symbol": rows["symbol"],
                "side": ["short"],
                "strategy_id": ["short_mean_reversion"],
                "normalized_rank_score": [0.91],
                "base_strategy_threshold": [kwargs["base_strategy_threshold"]],
                "calibrated_score": [0.4],
                "entry_price": [100.0],
                "exit_timestamp": pd.to_datetime(["2026-04-01T01:00:00Z"], utc=True),
                "exit_price": [99.0],
                "net_return": [0.01],
                "gross_return": [0.011],
                "holding_bars": [4],
                "simple_policy_exit_reason": ["trailing"],
            }
        )

    monkeypatch.setattr(
        ablate,
        "_build_simple_policy_candidate_rows",
        fake_build_simple_policy_candidate_rows,
    )

    out = ablate._candidate_table_for_overrides(
        [bundle],
        overrides={
            "best_size_power": 1.75,
            "base_strategy_threshold": 0.84,
            "portfolio_size_multiplier": 0.50,
            "portfolio_priority_adjustment": 0.125,
            "portfolio_max_concurrent_per_strategy": 2,
        },
        cost_pct=0.001,
        market_mode="perps",
        arm="unit",
    )

    assert float(out["base_strategy_threshold"].iloc[0]) == 0.84
    assert float(out["portfolio_size_multiplier"].iloc[0]) == 0.50
    assert float(out["portfolio_priority_adjustment"].iloc[0]) == 0.125
    assert int(out["portfolio_max_concurrent_per_strategy"].iloc[0]) == 2
