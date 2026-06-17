import numpy as np
import pandas as pd

from extreme_price_movements.config import (
    CFG,
    REGIME_ADAPTOR_FEATURE_ORDER,
    ROLLING_ALPHA_FEATURE_KEYS,
)
from extreme_price_movements.feature_family_registry import FeatureFamily, get_feature_family
from extreme_price_movements.rolling_alpha_targets import (
    build_gross_residual_alpha_target,
)
from extreme_price_movements.training_utils import get_base_feature_keys, get_meta_feature_keys


def _sample_alpha_frame(n_ts: int = 40, symbols: tuple[str, ...] = ("A", "B", "C", "D")):
    ts_index = pd.date_range("2025-01-01", periods=n_ts, freq="h", tz="UTC")
    rows = []
    for t_i, ts in enumerate(ts_index):
        market = 0.002 * np.sin(t_i / 3.0)
        cluster_a = 0.0015 * np.cos(t_i / 5.0)
        cluster_b = -0.0010 * np.sin(t_i / 4.0)
        for s_i, symbol in enumerate(symbols):
            cluster = "x" if s_i < 2 else "y"
            cluster_factor = cluster_a if cluster == "x" else cluster_b
            idio = 0.00035 * np.sin((t_i + 1) * (s_i + 2) / 7.0)
            rows.append(
                {
                    "__ts__": ts,
                    "__symbol__": symbol,
                    "__y_ret__": market * (1.0 + 0.12 * s_i)
                    + cluster_factor
                    + idio,
                    "asset_cluster": cluster,
                }
            )
    return pd.DataFrame(rows)


def _target_cfg(**overrides):
    cfg = {
        "rolling_alpha_target_horizon_hours": 5,
        "rolling_alpha_target_transform": "asinh_scaled",
        "rolling_alpha_target_market_beta_window": 8,
        "rolling_alpha_target_cluster_beta_window": 8,
        "rolling_alpha_target_beta_min_periods": 3,
        "rolling_alpha_target_scale_window": 8,
        "rolling_alpha_target_scale_min_periods": 3,
        "rolling_alpha_target_scale_floor": 1e-6,
        "rolling_alpha_target_beta_var_floor": 1e-12,
        "rolling_alpha_target_cluster_columns": ["asset_cluster"],
        "rolling_alpha_target_kalman_enabled": False,
        "rolling_alpha_target_clip_abs": 20.0,
    }
    cfg.update(overrides)
    return cfg


def test_gross_residual_alpha_target_builds_asinh_scaled_audit_columns():
    df = _sample_alpha_frame()
    bundle = build_gross_residual_alpha_target(
        df,
        side="long",
        y_ret=None,
        cfg=_target_cfg(),
    )

    target = np.asarray(bundle["target"], dtype=np.float32)
    audit = bundle["target_audit"]
    raw = np.asarray(audit["raw_gross_residual_alpha_5h"], dtype=np.float32)
    scale = np.asarray(audit["gross_residual_alpha_scale_5h"], dtype=np.float32)

    assert bundle["target_name"] == "asinh_scaled_gross_residual_alpha_5h"
    assert bundle["residualization_status"] == "gross_residual_alpha_market_cluster"
    assert bundle["nuisance_columns"] == ["asset_cluster"]
    assert target.shape == (len(df),)
    assert np.isfinite(target).all()
    assert np.all(scale > 0.0)
    assert np.allclose(target, np.arcsinh(raw / scale), atol=1e-6)
    assert np.nanstd(audit["market_factor_component_5h"]) > 0.0
    assert np.nanstd(audit["cluster_factor_component_5h"]) > 0.0
    assert bundle["target_diagnostics"]["cluster_source"] == "explicit_column"
    assert bundle["target_audit_summary"][
        "raw_gross_residual_alpha_5h"
    ]["finite_fraction"] == 1.0


def test_gross_residual_alpha_target_rolling_state_is_future_safe():
    df = _sample_alpha_frame(n_ts=48)
    cfg = _target_cfg()
    baseline = build_gross_residual_alpha_target(df, side="long", y_ret=None, cfg=cfg)

    cutoff = df["__ts__"].sort_values().unique()[24]
    mutated = df.copy()
    future_mask = mutated["__ts__"] > cutoff
    mutated.loc[future_mask, "__y_ret__"] = (
        mutated.loc[future_mask, "__y_ret__"].to_numpy(dtype=np.float64) * 25.0
        + 0.03
    )
    changed = build_gross_residual_alpha_target(mutated, side="long", y_ret=None, cfg=cfg)

    past_mask = df["__ts__"].to_numpy() <= cutoff
    for key in (
        "raw_gross_residual_alpha_5h",
        "market_beta_5h",
        "cluster_beta_5h",
        "gross_residual_alpha_scale_5h",
    ):
        left = np.asarray(baseline["target_audit"][key], dtype=np.float32)[past_mask]
        right = np.asarray(changed["target_audit"][key], dtype=np.float32)[past_mask]
        assert np.allclose(left, right, atol=1e-7)


def test_kalman_target_is_optional_and_named():
    df = _sample_alpha_frame()
    raw_bundle = build_gross_residual_alpha_target(
        df,
        side="long",
        y_ret=None,
        cfg=_target_cfg(),
    )
    kalman_bundle = build_gross_residual_alpha_target(
        df,
        side="long",
        y_ret=None,
        cfg=_target_cfg(
            rolling_alpha_target_kalman_enabled=True,
            rolling_alpha_target_kalman_blend=1.0,
            rolling_alpha_target_kalman_process_var=1e-7,
            rolling_alpha_target_kalman_obs_var=1e-4,
        ),
    )

    assert kalman_bundle["target_name"].endswith("_partial_kalman")
    assert not np.allclose(raw_bundle["target"], kalman_bundle["target"])


def test_cluster_target_uses_feature_bucket_fallback_when_no_cluster_column():
    df = _sample_alpha_frame()
    df = df.drop(columns=["asset_cluster"])
    df["asset_vol_level_pct"] = df["__symbol__"].map(
        {"A": 0.2, "B": 0.2, "C": 0.8, "D": 0.8}
    )
    bundle = build_gross_residual_alpha_target(
        df,
        side="long",
        y_ret=None,
        cfg=_target_cfg(
            rolling_alpha_target_cluster_columns=["missing_cluster"],
            rolling_alpha_target_cluster_feature_columns=["asset_vol_level_pct"],
            rolling_alpha_target_cluster_feature_max_columns=1,
        ),
    )

    assert bundle["target_diagnostics"]["cluster_source"] == "feature_buckets"
    assert bundle["target_diagnostics"]["cluster_columns"] == ["asset_vol_level_pct"]
    assert bundle["target_diagnostics"]["cluster_count"] == 2
    assert np.nanstd(bundle["target_audit"]["cluster_factor_component_5h"]) > 0.0


def test_rolling_alpha_features_feed_regime_adaptor_not_base_or_meta():
    base_cfg = {
        "base_shared_feature_keys": ["ROLLING_ALPHA_FEATURE_KEYS"],
        "base_long_feature_keys": ["ra_ret1h_robust_z"],
        "base_short_feature_keys": [],
        "meta_shared_feature_keys": ["ROLLING_ALPHA_FEATURE_KEYS"],
        "meta_product_feature_keys": ["ROLLING_ALPHA_FEATURE_KEYS"],
        "meta_reg_feature_keys": ["ra_market_beta_24h"],
        "ROLLING_ALPHA_FEATURE_KEYS": list(ROLLING_ALPHA_FEATURE_KEYS),
        "rolling_alpha_features_enabled": False,
    }

    enabled_cfg = dict(base_cfg)
    enabled_cfg["rolling_alpha_features_enabled"] = True
    assert not set(ROLLING_ALPHA_FEATURE_KEYS).intersection(
        get_base_feature_keys("long", enabled_cfg)
    )
    assert not set(ROLLING_ALPHA_FEATURE_KEYS).intersection(
        get_meta_feature_keys("reg", enabled_cfg)
    )
    assert set(ROLLING_ALPHA_FEATURE_KEYS).issubset(set(REGIME_ADAPTOR_FEATURE_ORDER))
    assert set(ROLLING_ALPHA_FEATURE_KEYS).issubset(
        set(CFG["REGIME_ADAPTOR_FEATURE_ORDER"])
    )


def test_rolling_alpha_feature_families_avoid_double_normalizing_scaled_features():
    for key in ROLLING_ALPHA_FEATURE_KEYS:
        family = get_feature_family(key)
        if key.endswith("_robust_z") or key == "ra_market_beta_24h":
            assert family == FeatureFamily.ALREADY_STANDARDIZED
        elif key.endswith("_cs_rank") or key.endswith("_pct"):
            assert family == FeatureFamily.BOUNDED_GEOMETRY
        elif key == "ra_market_resid_ret_6h":
            assert family == FeatureFamily.RISK_NORMALIZED_CONTINUOUS
