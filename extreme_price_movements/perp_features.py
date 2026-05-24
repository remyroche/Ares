#!/usr/bin/env python3
"""
Derivatives feature engineering (fully vectorized) on hourly bars with multi-timeframe
features at 2h / 4h / 8h horizons + cross-timeframe interactions.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


HORIZONS = (2, 4, 8)


def _safe_div(numer: pd.Series, denom: pd.Series, eps: float = 1e-12) -> pd.Series:
    return numer / (denom.replace(0, np.nan) + eps)


def _rolling_zscore(x: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    if min_periods is None:
        min_periods = min(window, max(5, 24 * 30 if window >= 24 * 30 else window // 5))
    m = x.rolling(window, min_periods=min_periods).mean()
    s = x.rolling(window, min_periods=min_periods).std(ddof=0)
    return _safe_div(x - m, s)


def _rolling_robust_zscore(
    x: pd.Series,
    window: int,
    min_periods: Optional[int] = None,
) -> pd.Series:
    if min_periods is None:
        min_periods = min(window, max(24 * 7, window // 4))
    median = x.rolling(window, min_periods=min_periods).median()
    q75 = x.rolling(window, min_periods=min_periods).quantile(0.75)
    q25 = x.rolling(window, min_periods=min_periods).quantile(0.25)
    iqr = (q75 - q25).replace(0.0, np.nan)
    return _safe_div(x - median, iqr)


def _rolling_rank_pct(x: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    if min_periods is None:
        min_periods = min(window, max(20, 24 * 30 if window >= 24 * 30 else window // 2))

    def _rank_last(a: np.ndarray) -> float:
        last = a[-1]
        return float(np.mean(a <= last))

    return x.rolling(window, min_periods=min_periods).apply(_rank_last, raw=True)


def get_perp_feature_names(horizons: Iterable[int] = HORIZONS) -> list[str]:
    out = [
        "basis",
        "basis_frac",
        "basis_pct",
        "basis_frac_z_14d",
        "basis_frac_rank_30d",
        "ret1h",
        "mom_slow",
        "funding_per_hour",
        "funding_z",
        "funding_rank_30d",
        "oi_z",
        "oi_value_log_1d_robust_z",
        "oi_value_log_7d_robust_z",
        "basis_pct_z",
        "oi_rank",
        "funding_abs_z",
        "basis_stretch",
        "funding_persistence",
        "basis_vol",
    ]

    horizons = tuple(sorted(set(int(h) for h in horizons)))
    for h in horizons:
        out.extend(
            [
                f"funding_mom_{h}h",
                f"oi_chg_{h}h",
                f"oi_chg_z_{h}h",
                f"oi_chg_{h}h_robust_z",
                f"oi_vel_{h}h",
                f"oi_rel_vol_{h}h",
                f"basis_mom_{h}h",
                f"basis_funding_div_{h}h",
            ]
        )

    out.extend(
        [
            "oi_up_agree",
            "funding_up_agree",
            "basis_up_agree",
            "funding_mom_w",
            "oi_chg_w",
            "basis_mom_w",
            "leverage_build",
            "leverage_build_score",
            "unwind",
            "unwind_score",
            "mom_slow_z",
            "squeeze_prob",
            "basis_funding_div",
        ]
    )
    for h in (5, 10):
        out.extend(
            [
                f"fund_pre_drift_{h}h",
                f"fund_post_reversal_{h}h",
                f"fund_ret_cond_sign_{h}h",
                f"fund_payment_pressure_{h}h",
                f"mark_gap_vol_{h}h",
                f"premium_expansion_speed_{h}h",
                f"mark_trigger_risk_{h}h",
                f"carry_adj_ret_{h}h",
                f"carry_adj_short_ret_{h}h",
                f"basis_adjusted_trend_{h}h",
                f"funding_crowded_mom_exhaustion_{h}h",
                f"fund_high_neg_mom_{h}h",
                f"persistent_pos_funding_failed_breakout_{h}h",
                f"persistent_neg_funding_failed_breakdown_{h}h",
                f"fund_flip_x_vol_expansion_{h}h",
            ]
        )
    out.extend(
        [
            "fund_hours_to_next",
            "fund_hours_since_last",
            "fund_next_event_proximity_5h",
            "fund_next_event_proximity_10h",
            "premium_mean_reversion_halflife_24h",
            "liq_buffer_long_mark_frac",
            "liq_buffer_short_mark_frac",
            "liq_buffer_atr",
            "liq_stop_safety_long_atr",
            "liq_stop_safety_short_atr",
        ]
    )
    return out


def compute_features(
    df: pd.DataFrame,
    horizons: Iterable[int] = HORIZONS,
    z_window_hours: int = 14 * 24,
    basis_vol_window_hours: int = 7 * 24,
    persistence_window_hours: int = 7 * 24,
    price_mom_slow_short: int = 2,
    price_mom_slow_long: int = 8,
) -> pd.DataFrame:
    required = {"funding_rate", "open_interest", "perp_price", "spot_price", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    idx = df.index
    out = pd.DataFrame(index=idx)

    funding = (
        df["funding_rate"]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
    )
    perp = df["perp_price"].astype(float)
    spot = df["spot_price"].astype(float)
    mark = (
        df["mark_price"].astype(float)
        if "mark_price" in df.columns
        else pd.Series(np.nan, index=idx)
    )
    contract_size = (
        df["contract_size"].astype(float)
        if "contract_size" in df.columns
        else pd.Series(1.0, index=idx)
    )
    oi_native = (
        df["open_interest"]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .where(lambda s: s > 0.0)
        .ffill()
    )
    oi = (
        df["open_interest_quote"]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .where(lambda s: s > 0.0)
        .ffill()
        if "open_interest_quote" in df.columns
        else (oi_native * contract_size * mark)
    ).replace([np.inf, -np.inf], np.nan).where(lambda s: s > 0.0)
    vol = df["volume"].astype(float)
    close = df["close"].astype(float) if "close" in df.columns else perp
    quote_volume = (
        df["quote_volume"].astype(float)
        if "quote_volume" in df.columns
        else (vol * perp)
    ).replace([np.inf, -np.inf], np.nan).where(lambda s: s > 0.0)

    basis_pct = (perp / spot.replace(0.0, np.nan)) - 1.0

    out["basis"] = basis_pct
    out["basis_frac"] = basis_pct
    out["basis_pct"] = basis_pct
    out["basis_frac_z_14d"] = _rolling_zscore(basis_pct, z_window_hours)
    out["basis_frac_rank_30d"] = _rolling_rank_pct(basis_pct, 24 * 30)

    logp = np.log(close.replace(0, np.nan))
    ret1h = logp.diff(1)
    out["ret1h"] = ret1h

    mom_short = logp.diff(price_mom_slow_short)
    mom_long = logp.diff(price_mom_slow_long)
    mom_slow = mom_short - mom_long
    out["mom_slow"] = mom_slow

    out["funding_per_hour"] = funding
    out["funding_z"] = _rolling_zscore(funding, z_window_hours)
    out["funding_rank_30d"] = _rolling_rank_pct(funding, 24 * 30)
    log_oi = np.log(oi.replace(0, np.nan))
    out["oi_z"] = _rolling_zscore(log_oi, z_window_hours)
    out["oi_value_log_1d_robust_z"] = _rolling_robust_zscore(
        log_oi.rolling(24, min_periods=1).mean(),
        24 * 30,
        min_periods=24 * 7,
    ).clip(-10.0, 10.0)
    out["oi_value_log_7d_robust_z"] = _rolling_robust_zscore(
        log_oi.rolling(24 * 7, min_periods=1).mean(),
        24 * 30,
        min_periods=24 * 7,
    ).clip(-10.0, 10.0)
    out["basis_pct_z"] = _rolling_zscore(basis_pct, z_window_hours)

    out["oi_rank"] = _rolling_rank_pct(log_oi, z_window_hours)
    out["funding_abs_z"] = out["funding_z"].abs()
    out["basis_stretch"] = out["basis_pct_z"].abs()

    out["funding_persistence"] = (out["funding_z"] > 0).astype(float).rolling(
        persistence_window_hours, min_periods=max(24, persistence_window_hours // 7)
    ).mean()

    out["basis_vol"] = basis_pct.rolling(
        basis_vol_window_hours, min_periods=max(24, basis_vol_window_hours // 7)
    ).std(ddof=0)

    horizons = tuple(sorted(set(int(h) for h in horizons)))
    funding_delta_cols: list[str] = []
    oi_delta_cols: list[str] = []
    basis_delta_cols: list[str] = []
    for h in horizons:
        funding_delta = funding.diff(h)
        oi_log_delta = log_oi.diff(h)
        basis_delta = basis_pct.diff(h)
        funding_delta_cols.append(f"_funding_delta_{h}h")
        oi_delta_cols.append(f"_oi_log_delta_{h}h")
        basis_delta_cols.append(f"_basis_delta_{h}h")
        out[f"_funding_delta_{h}h"] = funding_delta
        out[f"_oi_log_delta_{h}h"] = oi_log_delta
        out[f"_basis_delta_{h}h"] = basis_delta

        out[f"funding_mom_{h}h"] = _rolling_zscore(funding_delta, z_window_hours)
        out[f"oi_chg_{h}h"] = _rolling_zscore(oi_log_delta, z_window_hours)
        out[f"oi_chg_z_{h}h"] = out[f"oi_chg_{h}h"]
        out[f"oi_chg_{h}h_robust_z"] = _rolling_robust_zscore(
            oi_log_delta,
            24 * 30,
            min_periods=24 * 7,
        ).clip(-10.0, 10.0)
        out[f"oi_vel_{h}h"] = _rolling_zscore(oi_log_delta / float(h), z_window_hours)

        quote_volume_sum_h = quote_volume.fillna(0.0).rolling(h, min_periods=1).sum()
        oi_notional_delta = oi.diff(h)
        out[f"oi_rel_vol_{h}h"] = _safe_div(oi_notional_delta, quote_volume_sum_h).clip(
            -25.0, 25.0
        )

        out[f"basis_mom_{h}h"] = _rolling_zscore(basis_delta, z_window_hours)
        out[f"basis_funding_div_{h}h"] = out["basis_pct_z"] - out["funding_z"]

    oi_up = pd.concat([(out[f"_oi_log_delta_{h}h"] > 0).astype(float) for h in horizons], axis=1).mean(axis=1)
    funding_up = pd.concat([(out[f"_funding_delta_{h}h"] > 0).astype(float) for h in horizons], axis=1).mean(axis=1)
    basis_up = pd.concat([(out[f"_basis_delta_{h}h"] > 0).astype(float) for h in horizons], axis=1).mean(axis=1)

    out["oi_up_agree"] = oi_up
    out["funding_up_agree"] = funding_up
    out["basis_up_agree"] = basis_up

    w = np.array([1.0 / h for h in horizons], dtype=float)
    w = w / w.sum()

    def _wavg(cols: tuple[str, ...]) -> pd.Series:
        mat = pd.concat([out[c] for c in cols], axis=1)
        return (mat.values * w).sum(axis=1)

    funding_delta_w = _wavg(tuple(f"_funding_delta_{h}h" for h in horizons))
    oi_delta_w = _wavg(tuple(f"_oi_log_delta_{h}h" for h in horizons))
    basis_delta_w = _wavg(tuple(f"_basis_delta_{h}h" for h in horizons))
    out["funding_mom_w"] = _rolling_zscore(pd.Series(funding_delta_w, index=idx), z_window_hours)
    out["oi_chg_w"] = _rolling_zscore(pd.Series(oi_delta_w, index=idx), z_window_hours)
    out["basis_mom_w"] = _rolling_zscore(pd.Series(basis_delta_w, index=idx), z_window_hours)

    mom_slow_z = _rolling_zscore(out["mom_slow"], z_window_hours)
    out["mom_slow_z"] = mom_slow_z

    def sigmoid(x: pd.Series) -> pd.Series:
        return 1.0 / (1.0 + np.exp(-x))

    oi_soft = sigmoid(out["oi_chg_w"])
    funding_soft = sigmoid(out["funding_mom_w"])
    basis_soft = sigmoid(out["basis_mom_w"])
    leverage_components = pd.concat(
        [oi_soft, funding_soft, basis_soft],
        axis=1,
    )
    out["leverage_build"] = (
        leverage_components.mean(axis=1, skipna=True)
        .where(out["oi_chg_w"].notna())
        .clip(0, 1)
    )
    out["leverage_build_score"] = out["leverage_build"]
    out["unwind"] = (
        sigmoid(-out["oi_chg_w"]) * sigmoid(out["funding_abs_z"]) * sigmoid(-out["basis_mom_w"])
    ).clip(0, 1)
    out["unwind_score"] = out["unwind"]

    oi_high = out["oi_rank"].clip(0, 1)
    funding_ext = out["funding_rank_30d"].sub(0.5).abs().mul(2.0).clip(0, 1)
    basis_ext = out["basis_frac_rank_30d"].sub(0.5).abs().mul(2.0).clip(0, 1)
    mom_slowing = sigmoid(-mom_slow_z)

    squeeze_score = (oi_high * funding_ext * basis_ext * mom_slowing).pow(0.5)
    out["squeeze_prob"] = squeeze_score

    out["basis_funding_div"] = out["basis_pct_z"] - out["funding_z"]
    out.drop(
        columns=[c for c in out.columns if c.startswith(("_funding_delta_", "_oi_log_delta_", "_basis_delta_"))],
        inplace=True,
        errors="ignore",
    )

    return out
