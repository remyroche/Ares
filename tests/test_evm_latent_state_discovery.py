import numpy as np
import pandas as pd

from extreme_price_movements.evm_latent_state_discovery import (
    EvmLatentStateConfig,
    discover_evm_latent_states,
    select_evm_state_feature_columns,
)


def _panel() -> pd.DataFrame:
    n = 360
    ts = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    half = n // 2
    trend = np.concatenate(
        [
            np.linspace(0.0, 1.0, half, dtype=np.float32),
            np.linspace(0.0, 1.0, n - half, dtype=np.float32),
        ]
    ).astype(np.float32)
    # Same feature/state relationship in train and OOS: high trend is favorable
    # for this side x archetype, low trend is unfavorable.
    ev = (0.006 * (trend > 0.66) - 0.006 * (trend < 0.33) + 0.001).astype(np.float32)
    return pd.DataFrame(
        {
            "__ts__": ts,
            "month": np.where(np.arange(n) < half, "2026-01", "2026-02"),
            "week_start": "2026-01-01",
            "__symbol__": "AAA/USD:USD",
            "side_name": "long",
            "archetype_policy_key": "long__trend_pullback",
            "score_meta_base_soft_label": np.linspace(0.4, 0.9, n, dtype=np.float32),
            "trend_strength_3h_12h": trend,
            "market_breadth_24h": (1.0 - trend).astype(np.float32),
            "gmm_cluster_posterior_0": trend,
            "ev_after_1pct": ev,
            "clean_exec": (ev > 0.0).astype(np.int8),
            "dirty_positive": np.zeros(n, dtype=np.int8),
            "full_path_bad_mae_1r": (ev < 0.0).astype(np.int8),
            "timeout": np.zeros(n, dtype=np.int8),
            "stop_or_adverse": (ev < 0.0).astype(np.int8),
        }
    )


def test_evm_state_feature_selector_excludes_aegmm_and_outcomes_by_default() -> None:
    frame = _panel()
    frame["target_like_market_feature"] = np.arange(len(frame), dtype=np.float32)
    frame["exec_margin"] = np.arange(len(frame), dtype=np.float32)
    selected = select_evm_state_feature_columns(
        frame,
        required_columns=["__ts__", "__symbol__", "side_name", "archetype_policy_key"],
    )
    assert "trend_strength_3h_12h" in selected
    assert "market_breadth_24h" in selected
    assert "gmm_cluster_posterior_0" not in selected
    assert "ev_after_1pct" not in selected
    assert "target_like_market_feature" not in selected
    assert "exec_margin" not in selected


def test_evm_state_feature_selector_keeps_only_market_context_shock_entropy() -> None:
    frame = _panel()
    frame["vol_range_shock"] = np.linspace(0.0, 1.0, len(frame), dtype=np.float32)
    frame["direction_entropy_20"] = np.linspace(1.0, 0.0, len(frame), dtype=np.float32)
    frame["asset_minus_mkt_vol_range_shock"] = np.linspace(0.0, 1.0, len(frame), dtype=np.float32)
    frame["mkt_resid_direction_entropy_20"] = np.linspace(1.0, 0.0, len(frame), dtype=np.float32)
    frame["market_shock_index"] = np.linspace(0.0, 1.0, len(frame), dtype=np.float32)
    frame["xs_entropy_dispersion"] = np.linspace(1.0, 0.0, len(frame), dtype=np.float32)
    frame["state_spectral_eig_entropy"] = np.linspace(0.2, 0.8, len(frame), dtype=np.float32)
    selected = select_evm_state_feature_columns(
        frame,
        include_aegmm=True,
        required_columns=["__ts__", "__symbol__", "side_name", "archetype_policy_key"],
    )
    assert "vol_range_shock" not in selected
    assert "direction_entropy_20" not in selected
    assert "asset_minus_mkt_vol_range_shock" not in selected
    assert "mkt_resid_direction_entropy_20" not in selected
    assert "market_shock_index" in selected
    assert "xs_entropy_dispersion" in selected
    assert "state_spectral_eig_entropy" in selected


def test_evm_latent_state_discovery_finds_oos_favorable_high_state() -> None:
    frame = _panel()
    train = frame.loc[frame["month"].eq("2026-01")].copy()
    oos = frame.loc[frame["month"].eq("2026-02")].copy()
    result = discover_evm_latent_states(
        train,
        oos,
        ["trend_strength_3h_12h", "market_breadth_24h"],
        config=EvmLatentStateConfig(
            min_group_rows=80,
            min_state_rows=15,
            max_features_per_group=4,
            top_features_for_pairs=2,
            min_oos_objective_delta=0.0001,
        ),
        eval_label="2026-02",
    )
    assert not result.feature_state_metrics.empty
    high = result.catalog.loc[
        result.catalog["state_name"].eq("trend_strength_3h_12h=high")
        & result.catalog["direction"].eq("favorable")
    ]
    assert not high.empty
    assert float(high["oos_mean_ev_after_1pct"].max()) > 0.0
    low = result.feature_state_metrics.loc[
        result.feature_state_metrics["state_name"].eq("trend_strength_3h_12h=low")
    ]
    assert not low.empty
    assert float(low["oos_objective_delta"].min()) < 0.0
