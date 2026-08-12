import numpy as np
import pandas as pd

from extreme_price_movements.market_transition_sidecar import (
    TransitionSidecarConfig,
    build_market_transition_sidecar,
    resolve_spine_sources,
)
from extreme_price_movements.config import CFG


def test_spine_resolution_prefers_causal_hourly_proxies() -> None:
    columns = [
        "mv__breakout_efficiency_4h__robust_z_1h",
        "mv__breadth_dispersion__realized_vol_1h",
        "mv__breadth_dispersion__vol_of_vol_1h",
        "mv__downside_breadth_intensity__robust_z_1h",
        "mv__btc_decoupling_dispersion__robust_z_1h",
        "mv__correlation_heterogeneity_dispersion__robust_z_1h",
        "mv__liquidity_xs__log_quote_volume__mean__robust_z_1h",
        "mv__liquidity_xs__amihud_illiq__mean__robust_z_1h",
        "mv__funding_deleveraging_divergence__robust_z_1h",
        "mv__broad_washout_recovery__robust_z_1h",
        "mv__funding_confirmed_long_flush__robust_z_1h",
        "mv__short_breakout_exhaustion__robust_z_1h",
    ]
    assert len(resolve_spine_sources(columns)) >= 8


def test_sidecar_is_causal_under_future_perturbation() -> None:
    rows = 96
    timestamp = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    names = [f"x_{index}" for index in range(8)]
    data = pd.DataFrame({"source_utc": timestamp, **{name: np.sin(np.arange(rows) / (index + 2)) for index, name in enumerate(names)}})
    config = TransitionSidecarConfig(robust_window_hours=24, min_reference_hours=12, bocpd_inputs=2, bocpd_max_run_hours=48, covariance_update_hours=3, distribution_reference_hours=24)
    first, features = build_market_transition_sidecar(data, source_columns={name: name for name in names}, config=config)
    altered = data.copy(); altered.loc[70:, names] += 1000.
    second, _ = build_market_transition_sidecar(altered, source_columns={name: name for name in names}, config=config)
    check = [name for name in features if first[name].notna().any()]
    assert np.allclose(first.loc[:60, check], second.loc[:60, check], equal_nan=True)


def test_current_features_only_is_the_canonical_sidecar_placement() -> None:
    placement = CFG["MARKET_TRANSITION_SIDECAR_CANONICAL_PLACEMENT"]
    assert placement["status"] == "CURRENT_FEATURES_ONLY_CANONICAL_SELECTOR_SAMPLE_CONTROL"
    assert placement["correctness_head_feature_keys"] == ()
    assert placement["regular_meta_head_feature_keys"] == ()
    assert "full-universe" in placement["scope_limitation"]
    assert CFG["MARKET_TRANSITION_SIDECAR_PROMOTION_GATE"]["status"] == (
        "rejected_current_features_only_selector_sample_canonical"
    )
