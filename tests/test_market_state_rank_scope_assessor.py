from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.build_market_state_rank_scope_assessor import (
    build_router_schedule,
    build_training_table,
    dedupe_state_panel,
    fit_final_model,
    select_market_state_features,
)
from scripts.run_market_state_short_boll_rank_scope_switch import (
    load_rank_reference_router_schedule,
)


def _state_panel(n: int = 40) -> pd.DataFrame:
    ts = pd.date_range("2026-05-01", periods=n, freq="h", tz="UTC")
    x = np.linspace(-1.0, 1.0, n)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "fold": np.repeat(np.arange(1, 5), n // 4),
            "state_shock": x,
            "state_realized_vol": np.abs(x),
            "forecast_h6_shock_up": (x > 0).astype(float),
            "latent_entropy": 1.0 - np.abs(x),
            "state_input_coverage": 0.9,
            "state_feature_count": 25,
            "state_rank_like_bad": x,
            "not_state_feature": x,
        }
    )


def _utility(n: int = 40) -> pd.DataFrame:
    ts = pd.date_range("2026-05-01", periods=n, freq="h", tz="UTC")
    x = np.linspace(-1.0, 1.0, n)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "fold": np.repeat(np.arange(1, 5), n // 4),
            "timestamp_minus_global_net_pnl": np.where(x > 0, 2.0, -1.0),
            "timestamp_minus_global_short_boll_net_pnl": np.where(x > 0, 1.5, -0.5),
        }
    )


def test_market_state_feature_selection_excludes_nuisance_columns() -> None:
    features = select_market_state_features(_state_panel())

    assert "state_shock" in features
    assert "forecast_h6_shock_up" in features
    assert "latent_entropy" in features
    assert "state_input_coverage" not in features
    assert "state_feature_count" not in features
    assert "state_rank_like_bad" not in features
    assert "not_state_feature" not in features


def test_training_table_builds_bounded_rank_scope_target() -> None:
    train, scale = build_training_table(
        _state_panel(),
        _utility(),
        target_col="timestamp_minus_global_net_pnl",
    )

    assert len(train) == 40
    assert scale > 0.0
    assert train["rank_scope_target"].between(-1.0, 1.0).all()
    assert train["rank_scope_sample_weight"].min() >= 1.0


def test_rank_scope_assessor_outputs_formal_router_schedule(tmp_path) -> None:
    state = _state_panel()
    train, _scale = build_training_table(
        state,
        _utility(),
        target_col="timestamp_minus_global_net_pnl",
    )
    features = select_market_state_features(train)
    model, medians = fit_final_model(train, features, backend="lgbm", seed=7)
    schedule = build_router_schedule(
        state,
        model,
        features,
        medians,
        prediction_temperature=0.25,
    )
    path = tmp_path / "rank_reference_router_schedule.parquet"
    schedule.to_parquet(path, index=False)

    loaded = load_rank_reference_router_schedule(path)

    assert len(loaded) == len(dedupe_state_panel(state))
    assert loaded["short_boll_timestamp_weight"].between(0.0, 1.0).all()
    assert set(loaded["short_boll_rank_scope"]).issubset({"timestamp_rank", "global_rank"})
    assert loaded["router_valid"].all()
