from typing import List, Dict, Any


_BASE_NON_LOCATION_48H_PLUS_EXACT = {
    "ret48h",
    "ret72h",
    "ret120h",
    "rv_48h",
    "rv_120h",
    "spectral_entropy_ret_48",
    "ffd_donch_dist_04_48",
    "ffd_donch_dist_06_48",
    "regime_transition_entropy_48h",
    "hurst_proxy_x_regime_trend_48h",
    "trapped_longs_96",
    "ema20_gt_ema50",
    "ema50_gt_ema200",
    "ema50_ema200_spread_atr",
    "price_lt_ema200",
    "ema50_slope",
    "volume_zscore_48h",
    "return_autocorr_48",
    "variance_ratio_10_48",
    "volume_trend_48",
    "volume_autocorr_48",
    "volatility_of_volatility_48",
    "volatility_autocorr_48",
    "trend_slope_48h",
    "trend_slope_120h",
    "trend_accel_120h",
    "rv_ratio_24_120",
    "higher_highs_count_48h",
    "volume_trend_alignment",
    "trend_regime_stability",
}

_BASE_META_ONLY_EXACT = {
    "regime_stability_24h",
    "complexity_regime_24h",
    "entropy_jump_24h",
    "coherence_24",
    "pers_24",
    "ts_24",
    "prog_eff_24",
}

_BASE_LOCATION_PREFIXES = (
    "loc_",
    "dist_from_",
    "donch_dist_",
    "pullback_",
    "dist_vwap_",
    "dist_ema",
    "zscore_price_",
    "dist_prior_",
    "dist_rolling_",
    "dist_local_",
    "dist_range_",
    "distance_to_",
)

_META_SHARED_BASELIKE_EXACT = {
    # 15m/24h bar-structure features that should stay out of the shared meta basket.
    "clv_t",
    "body_ratio_15m",
    "rejection_proxy",
    "range_norm_12",
    "sv_imb_12",
    "press_12",
    "impact_12",
    "hh_count_12",
    "ll_count_12",
    "skew_12",
    "climax_range_12",
    "climax_vol_12",
    "z_vwap_12",
    "z_r_12",
    "bb_pos_12",
    "range_norm_24",
    "sv_imb_24",
    "press_24",
    "impact_24",
    "hh_count_24",
    "ll_count_24",
    "skew_24",
    "climax_range_24",
    "climax_vol_24",
    "z_vwap_24",
    "z_r_24",
    "bb_pos_24",
    # Structural-Z normalization features should remain on the base side.
    "z_hl_range",
    "z_intrabar_range_atr",
    "z_compression_expansion",
    "z_volume",
    "z_breakout_up_24",
    "z_breakout_dn_24",
    "z_dist_ema_24",
    "z_dist_vwap_24",
    "z_atr_norm_ret_24",
    "z_sm_momentum_24",
    "z_slope_change_24",
    "z_path_efficiency_24",
}

def dedupe_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _is_location_feature(name: str) -> bool:
    if name.startswith(_BASE_LOCATION_PREFIXES):
        return True
    if name in {
        "dist_weekly_vwap",
        "bars_since_ema20_ema50_cross_log_norm",
        "bars_in_high_vol_state_log_norm",
        "bars_outside_ema20_atr_band_log_norm",
        "loc_pullback_depth_48",
    }:
        return True
    return False


def _filter_base_feature_keys(keys: List[str]) -> List[str]:
    filtered: List[str] = []
    for key in keys:
        if key in _BASE_META_ONLY_EXACT:
            continue
        if key in _BASE_NON_LOCATION_48H_PLUS_EXACT and not _is_location_feature(key):
            continue
        filtered.append(key)
    return filtered


def _filter_meta_shared_feature_keys(keys: List[str]) -> List[str]:
    filtered: List[str] = []
    for key in keys:
        if key.startswith(("loc_", "dist_", "pullback_", "donch_dist_")):
            continue
        if key in _META_SHARED_BASELIKE_EXACT:
            continue
        filtered.append(key)
    return filtered

def expand_feature_group_refs(keys: List[str], cfg: Dict[str, Any], visited=None) -> List[str]:
    if visited is None:
        visited = set()

    expanded = []
    for k in keys:
        if k in visited:
            continue
        visited.add(k)

        # Support dict-like or object-like config access
        if isinstance(cfg, dict):
            val = cfg.get(k)
            if val is None:
                # Check module level imports just in case
                import extreme_price_movements.config as cfg_mod
                val = getattr(cfg_mod, k, None)
        else:
            val = getattr(cfg, k, None)

        if isinstance(val, list):
            expanded.extend(expand_feature_group_refs(val, cfg, visited.copy()))
        else:
            expanded.append(k)
    return expanded

def get_base_feature_keys(side: str, cfg: Dict[str, Any]) -> List[str]:
    if side not in {"long", "short"}:
        raise ValueError(f"Invalid side {side}")

    shared = expand_feature_group_refs(cfg.get("base_shared_feature_keys", []), cfg)
    if side == "long":
        specific = expand_feature_group_refs(cfg.get("base_long_feature_keys", []), cfg)
    else:
        specific = expand_feature_group_refs(cfg.get("base_short_feature_keys", []), cfg)

    return dedupe_keep_order(_filter_base_feature_keys(shared + specific))

def get_meta_feature_keys(head: str, cfg: Dict[str, Any]) -> List[str]:
    if head not in {"reg", "clf", "mfe", "mae", "asym"}:
        raise ValueError(f"Invalid head {head}")

    shared = _filter_meta_shared_feature_keys(
        expand_feature_group_refs(cfg.get("meta_shared_feature_keys", []), cfg)
    )

    if head == "reg":
        specific = expand_feature_group_refs(cfg.get("meta_reg_feature_keys", []), cfg)
        return dedupe_keep_order(shared + specific)
    elif head == "clf":
        specific = expand_feature_group_refs(cfg.get("meta_clf_feature_keys", []), cfg)
        return dedupe_keep_order(shared + specific)
    elif head == "mfe":
        specific = expand_feature_group_refs(cfg.get("meta_mfe_feature_keys", []), cfg)
        return dedupe_keep_order(shared + specific)
    elif head == "mae":
        specific = expand_feature_group_refs(cfg.get("meta_mae_feature_keys", []), cfg)
        return dedupe_keep_order(shared + specific)
    elif head == "asym":
        mfe_specific = expand_feature_group_refs(cfg.get("meta_mfe_feature_keys", []), cfg)
        mae_specific = expand_feature_group_refs(cfg.get("meta_mae_feature_keys", []), cfg)
        asym_specific = expand_feature_group_refs(cfg.get("meta_asym_feature_keys", []), cfg)
        return dedupe_keep_order(shared + mfe_specific + mae_specific + asym_specific)

def validate_feature_keys_exist(df, keys: List[str], context: str) -> None:
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature keys for {context}: {missing}")

def compute_unused_features(all_feature_columns: List[str], configured_feature_keys: List[str]) -> List[str]:
    configured = set(configured_feature_keys)
    unused = [c for c in all_feature_columns if c not in configured]
    return sorted(unused)

def audit_feature_coverage(all_feature_columns: List[str], cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    exclude_prefixes = [
        "__y", "__w", "id", "timestamp", "symbol",
        "fold_", "oof_", "p_", "pred_", "is_up"
    ]

    all_cols = []
    for c in all_feature_columns:
        if c in {"id", "timestamp", "symbol"}:
            continue
        if any(c.startswith(p) for p in exclude_prefixes) and not c.startswith("p_vol_high") and not c.startswith("p_cusum_high") and not c.startswith("p_liq_low"):
            continue
        all_cols.append(c)

    base_long = set(get_base_feature_keys("long", cfg))
    base_short = set(get_base_feature_keys("short", cfg))
    base_all = base_long.union(base_short)

    meta_reg = set(get_meta_feature_keys("reg", cfg))
    meta_clf = set(get_meta_feature_keys("clf", cfg))
    meta_mfe = set(get_meta_feature_keys("mfe", cfg))
    meta_mae = set(get_meta_feature_keys("mae", cfg))
    meta_asym = set(get_meta_feature_keys("asym", cfg))
    meta_all = meta_reg.union(meta_clf).union(meta_mfe).union(meta_mae).union(meta_asym)

    global_all = base_all.union(meta_all)

    base_unused = sorted(list(set(all_cols) - base_all))
    meta_unused = sorted(list(set(all_cols) - meta_all))
    global_unused = sorted(list(set(all_cols) - global_all))

    stale_orphans = []
    for f in global_unused:
        if (
            f.startswith("tf_")
            or f.startswith("mr_")
            or "legacy" in f.lower()
            or f in {"tf_feature_keys", "mr_feature_keys", "meta_feature_keys"}
        ):
            stale_orphans.append(f)

    computed_but_unused = sorted(list(set(all_cols) - global_all))
    configured_but_missing = sorted(list(global_all - set(all_cols)))

    return {
        "computed_but_unused": computed_but_unused,
        "configured_but_missing": configured_but_missing,
        "base_unused": base_unused,
        "meta_unused": meta_unused,
        "stale_orphans": stale_orphans,
        "base_all": sorted(list(base_all)),
        "meta_all": sorted(list(meta_all)),
        "global_all": sorted(list(global_all)),
        "base_long": sorted(list(base_long)),
        "base_short": sorted(list(base_short)),
        "meta_reg": sorted(list(meta_reg)),
        "meta_clf": sorted(list(meta_clf)),
        "meta_mfe": sorted(list(meta_mfe)),
        "meta_mae": sorted(list(meta_mae)),
        "meta_asym": sorted(list(meta_asym))
    }
