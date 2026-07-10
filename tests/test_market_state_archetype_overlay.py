import numpy as np
import pandas as pd

from extreme_price_movements.market_state_archetype_overlay import (
    MarketStateOverlayConfig,
    fit_market_state_archetype_overlay,
    select_market_state_columns,
)
from extreme_price_movements.regime_ev_calibration import apply_regime_ev_calibration


def _panel() -> pd.DataFrame:
    n = 240
    ts = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    stress = np.linspace(0.0, 1.0, n, dtype=np.float32)
    ret = (0.018 - 0.040 * stress).astype(np.float32)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": "AAA/USD:USD",
            "side_name": "long",
            "policy_archetype": "long__breakout",
            "rank_pct": np.full(n, 0.90, dtype=np.float32),
            "ctx_market_stress": stress,
            "gmm_cluster_posterior_0": (1.0 - stress).astype(np.float32),
            "ret_net_notional": ret,
            "full_sl": (ret < -0.01).astype(np.int8),
        }
    )


def test_market_state_overlay_learns_unfavorable_bucket_and_adjusts_score() -> None:
    frame = _panel()
    result = fit_market_state_archetype_overlay(
        frame,
        feature_columns=["ctx_market_stress"],
        outcome_col="ret_net_notional",
        config=MarketStateOverlayConfig(
            min_group_rows=50,
            min_bucket_rows=20,
            max_features_per_group=2,
            min_abs_effect=0.0001,
            effect_scale=2.5,
        ),
    )
    assert result.artifact["effects"]
    effect = result.artifact["effects"][0]
    effects = {int(k): float(v) for k, v in effect["params"]["effects"].items()}
    assert effects[max(effects)] > 0.0

    scored = apply_regime_ev_calibration(
        frame,
        result.artifact,
        source_score_col="rank_pct",
        side_col="side_name",
        archetype_col="policy_archetype",
    )
    high = scored.loc[scored["ctx_market_stress"] > 0.90, "score_market_state_calibrated"].mean()
    low = scored.loc[scored["ctx_market_stress"] < 0.10, "score_market_state_calibrated"].mean()
    assert float(high) < float(low)


def test_market_state_feature_selector_excludes_realized_outcome_columns() -> None:
    frame = _panel()
    frame["ctx_target_like"] = np.arange(len(frame), dtype=np.float32)
    frame["ctx_clean_signal"] = np.arange(len(frame), dtype=np.float32)
    frame["ret_net_context"] = np.arange(len(frame), dtype=np.float32)
    selected = select_market_state_columns(
        frame,
        include_prefixes=("ctx_", "gmm_"),
        required_columns=["timestamp", "side_name", "policy_archetype", "rank_pct"],
    )
    assert "ctx_clean_signal" in selected
    assert "gmm_cluster_posterior_0" in selected
    assert "ctx_target_like" not in selected
    assert "ret_net_context" not in selected

