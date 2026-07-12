from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.market_residual_archetypes import (
    MarketResidualConfig,
    MarketResidualStateRecognizer,
    market_residual_feature_names,
)
from extreme_price_movements.meta_residual_archetypes import strip_outcomes_for_oos


def _frame(timestamps: int = 1000, assets: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(77)
    ts = pd.date_range("2025-01-01", periods=timestamps, freq="h", tz="UTC")
    phase = np.sin(np.arange(timestamps) / 24.0 * np.pi / 2.0).astype(np.float32)
    shock = np.repeat(phase, assets)
    rows = timestamps * assets
    score = np.clip(0.65 + 0.12 * shock + rng.normal(0.0, 0.03, rows), 0.01, 0.99)
    clean_probability = np.clip(score - 0.45 * np.maximum(shock, 0.0), 0.02, 0.98)
    clean = (rng.random(rows) < clean_probability).astype(np.float32)
    ev = (0.012 * clean - 0.016 * (1.0 - clean)).astype(np.float32)
    return pd.DataFrame(
        {
            "__ts__": ts.repeat(assets),
            "__symbol__": [f"S{i % assets}" for i in range(rows)],
            "side_name": np.where(np.arange(rows) % 2, "short", "long"),
            "archetype_policy_key": np.where(
                np.arange(rows) % 4 < 2, "continuation", "compression"
            ),
            "score_regime_calibrated": score.astype(np.float32),
            "clean_exec": clean,
            "dirty_positive": ((ev > 0.0) & (clean < 0.5)).astype(np.float32),
            "full_path_bad_mae_1r": (clean < 0.5).astype(np.float32),
            "timeout": np.zeros(rows, dtype=np.float32),
            "ev_after_1pct": ev,
            "mkt_shock": shock,
            "market_breadth_chg_1h": (-shock).astype(np.float32),
            "cross_asset_downside_corr_1h": np.abs(shock).astype(np.float32),
            "asset_minus_mkt_oi_chg_1h_rz": rng.normal(size=rows).astype(np.float32),
        }
    )


def test_market_residual_recognizer_is_market_only_and_oos_safe() -> None:
    frame = _frame()
    train = frame.iloc[:7200].copy()
    valid = frame.iloc[7200:].copy()
    recognizer = MarketResidualStateRecognizer(
        config=MarketResidualConfig(
            score_col="score_regime_calibrated",
            min_rows=100,
            max_fit_rows=1000,
            max_features=6,
            random_state=11,
        ),
        candidate_features=[
            "mkt_shock",
            "market_breadth_chg_1h",
            "cross_asset_downside_corr_1h",
            "asset_minus_mkt_oi_chg_1h_rz",
        ],
    ).fit(train)
    assert "asset_minus_mkt_oi_chg_1h_rz" not in recognizer.feature_columns
    output = recognizer.transform_oos(strip_outcomes_for_oos(valid))
    assert set(market_residual_feature_names()).issubset(output.columns)
    assert np.isfinite(output.to_numpy(dtype=np.float32)).all()
    np.testing.assert_allclose(output.filter(like="prob__").sum(axis=1), 1.0, atol=1e-5)
    assert recognizer.manifest()["leakage_contract"]["recent_hit_rate_inputs"] is False
