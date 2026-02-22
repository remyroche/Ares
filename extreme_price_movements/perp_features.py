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
        min_periods = max(5, window // 5)
    m = x.rolling(window, min_periods=min_periods).mean()
    s = x.rolling(window, min_periods=min_periods).std(ddof=0)
    return _safe_div(x - m, s)


def _rolling_rank_pct(x: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    if min_periods is None:
        min_periods = max(20, window // 2)

    def _rank_last(a: np.ndarray) -> float:
        last = a[-1]
        return float(np.mean(a <= last))

    return x.rolling(window, min_periods=min_periods).apply(_rank_last, raw=True)


def get_perp_feature_names(horizons: Iterable[int] = HORIZONS) -> list[str]:
    out = [
        "basis",
        "basis_pct",
        "ret1h",
        "mom_slow",
        "funding_z",
        "oi_z",
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
            "unwind",
            "mom_slow_z",
            "squeeze_prob",
            "basis_funding_div",
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

    funding = df["funding_rate"].astype(float)
    oi = df["open_interest"].astype(float)
    perp = df["perp_price"].astype(float)
    spot = df["spot_price"].astype(float)
    vol = df["volume"].astype(float)
    close = df["close"].astype(float) if "close" in df.columns else perp

    basis = perp - spot
    basis_pct = _safe_div(basis, spot)

    out["basis"] = basis
    out["basis_pct"] = basis_pct

    logp = np.log(close.replace(0, np.nan))
    ret1h = logp.diff(1)
    out["ret1h"] = ret1h

    mom_short = logp.diff(price_mom_slow_short)
    mom_long = logp.diff(price_mom_slow_long)
    mom_slow = mom_short - mom_long
    out["mom_slow"] = mom_slow

    out["funding_z"] = _rolling_zscore(funding, z_window_hours)
    out["oi_z"] = _rolling_zscore(oi, z_window_hours)
    out["basis_pct_z"] = _rolling_zscore(basis_pct, z_window_hours)

    out["oi_rank"] = _rolling_rank_pct(oi, z_window_hours)
    out["funding_abs_z"] = out["funding_z"].abs()
    out["basis_stretch"] = out["basis_pct_z"].abs()

    out["funding_persistence"] = (funding > 0).astype(float).rolling(
        persistence_window_hours, min_periods=max(24, persistence_window_hours // 7)
    ).mean()

    out["basis_vol"] = basis_pct.rolling(
        basis_vol_window_hours, min_periods=max(24, basis_vol_window_hours // 7)
    ).std(ddof=0)

    horizons = tuple(sorted(set(int(h) for h in horizons)))
    for h in horizons:
        out[f"funding_mom_{h}h"] = funding.diff(h)
        out[f"oi_chg_{h}h"] = oi.diff(h)
        out[f"oi_vel_{h}h"] = out[f"oi_chg_{h}h"] / float(h)

        vol_sum_h = vol.rolling(h, min_periods=max(1, h)).sum()
        out[f"oi_rel_vol_{h}h"] = _safe_div(out[f"oi_chg_{h}h"], vol_sum_h)

        out[f"basis_mom_{h}h"] = basis_pct.diff(h)
        out[f"basis_funding_div_{h}h"] = out["basis_pct_z"] - out["funding_z"]

    oi_up = pd.concat([(out[f"oi_chg_{h}h"] > 0).astype(float) for h in horizons], axis=1).mean(axis=1)
    funding_up = pd.concat([(out[f"funding_mom_{h}h"] > 0).astype(float) for h in horizons], axis=1).mean(axis=1)
    basis_up = pd.concat([(out[f"basis_mom_{h}h"] > 0).astype(float) for h in horizons], axis=1).mean(axis=1)

    out["oi_up_agree"] = oi_up
    out["funding_up_agree"] = funding_up
    out["basis_up_agree"] = basis_up

    w = np.array([1.0 / h for h in horizons], dtype=float)
    w = w / w.sum()

    def _wavg(cols: tuple[str, ...]) -> pd.Series:
        mat = pd.concat([out[c] for c in cols], axis=1)
        return (mat.values * w).sum(axis=1)

    out["funding_mom_w"] = _wavg(tuple(f"funding_mom_{h}h" for h in horizons))
    out["oi_chg_w"] = _wavg(tuple(f"oi_chg_{h}h" for h in horizons))
    out["basis_mom_w"] = _wavg(tuple(f"basis_mom_{h}h" for h in horizons))

    out["leverage_build"] = (
        (out["oi_up_agree"] >= 2 / 3)
        & (out["funding_up_agree"] >= 2 / 3)
        & (out["basis_up_agree"] >= 2 / 3)
        & (out["oi_chg_w"] > 0)
        & (out["funding_mom_w"] > 0)
        & (out["basis_mom_w"] > 0)
    ).astype(int)

    funding_extreme_z = 1.5
    basis_stretch_z = 1.5

    out["unwind"] = (
        (out["oi_chg_w"] < 0)
        & (pd.concat([(out[f"oi_chg_{h}h"] < 0).astype(float) for h in horizons], axis=1).mean(axis=1) >= 2 / 3)
        & (out["funding_abs_z"] > funding_extreme_z)
        & (out["basis_mom_w"] < 0)
        & (out["basis_stretch"] > basis_stretch_z)
    ).astype(int)

    mom_slow_z = _rolling_zscore(out["mom_slow"], z_window_hours)
    out["mom_slow_z"] = mom_slow_z

    def sigmoid(x: pd.Series) -> pd.Series:
        return 1.0 / (1.0 + np.exp(-x))

    k = 1.5
    oi_high = out["oi_rank"].clip(0, 1)
    funding_ext = sigmoid(out["funding_abs_z"] - k)
    basis_ext = sigmoid(out["basis_stretch"] - k)
    mom_slowing = sigmoid(-mom_slow_z)

    squeeze_score = (oi_high * funding_ext * basis_ext * mom_slowing).pow(0.5)
    out["squeeze_prob"] = squeeze_score

    out["basis_funding_div"] = out["basis_pct_z"] - out["funding_z"]

    return out

