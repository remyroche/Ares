from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except Exception:  # pragma: no cover - optional runtime dependency fallback.
    optuna = None  # type: ignore[assignment]
    MedianPruner = None  # type: ignore[assignment]
    TPESampler = None  # type: ignore[assignment]

EPS = 1e-9
REGIME_FEATURE_ORDER = [
    "rv_24h",
    "rv1_rv24",
    "rv4_rv24",
    "signed_adx",
    "dist_ema_fast",
    "dist_ema_slow",
    "dist_vwap",
    "prior_day_low",
    "prior_day_high",
    "rvol_z",
    "asset_volume_30d",
    "is_weekend",
    "asset_atr_30d",
    "ebm_unc_logodds_var",
    "ebm_unc_pi_width",
    "ebm_unc_entropy_mean",
    "ebm_unc_entropy_std",
    "ebm_unc_conflict_norm",
    "ebm_unc_proximity_min",
    "ebm_unc_support_mean",
    "ebm_unc_support_min",
    "ebm_unc_concentration",
    "ebm_unc_sign_ratio",
    "ebm_unc_interaction_share",
    "ebm_unc_gap50rel",
    "ebm_unc_support_adjusted_uncertainty",
    "ebm_unc_uncertainty_weight",
    "ebm_unc_friction_weight",
]

FEATURE_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "rv_24h": (
        "rv_24h",
        "realized_volatility_24h",
        "ffd_rv_24h_04",
        "ffd_rv_24_04",
        "range_24h_pct",
        "range_norm_24",
        "vol_regime_z",
        "vol_z24",
        "vol_z",
        "volatility_zscore",
        "atr_12_15m",
        "atr_pct",
        "atr_pct_base",
    ),
    "rv1": (
        "rv_1h",
        "rv_2h",
        "realized_volatility_1h",
        "realized_vol_15m_realized_vol_2h",
        "rv_1h_proxy",
        "ret1h_abs",
        "range_norm_12",
        "range_12h_pct",
        "z_r_12",
    ),
    "rv4": (
        "rv_4h",
        "rv_6h",
        "ffd_rv_6h_04",
        "realized_volatility_4h",
        "rv_4h_proxy",
        "range_norm_24",
        "z_r_24",
    ),
    "signed_adx": (
        "signed_adx",
        "adx_zscore",
        "adx_14",
        "adx_10",
        "adx_7",
        "trend_slope_48h",
        "trend_slope_120h",
        "regime_trend_score",
        "trend_regime",
    ),
    "trend_sign": ("trend_regime", "trend_24h", "ret24h", "slope"),
    "dist_ema_fast": (
        "dist_ema_fast",
        "dist_ema_fast_base",
        "dist_ema_fast_z",
        "ffd_dist_ema_fast_04",
        "dist_ema20_atr",
        "z_dist_ema_24",
    ),
    "dist_ema_slow": (
        "dist_ema_slow",
        "dist_ema_slow_base",
        "ffd_dist_ema_slow_04",
        "dist_ema50_atr",
        "dist_ema200_atr",
    ),
    "dist_vwap": (
        "loc_vwap_dev_z_24",
        "dist_vwap_norm",
        "dist_vwap_norm_z",
        "loc_vwap_dev_z_48",
        "z_vwap_24",
        "z_vwap_12",
        "z_dist_vwap_24",
        "dist_vwap_24_atr",
        "dist_vwap_12_atr",
        "dist_weekly_vwap",
    ),
    "prior_day_low": (
        "dist_prior_day_low",
        "loc_prev_day_low",
        "loc_prev_day_range_pos_24",
    ),
    "prior_day_high": (
        "dist_prior_day_high",
        "loc_prev_day_high",
        "loc_prev_day_range_pos_24",
    ),
    "rvol_z": ("rvol_z", "volume_zscore_48h", "volume_z_24", "regime_liquidity_score"),
    "entropy_24h": (
        "spectral_entropy_ret_24",
        "perm_entropy_ret_24",
        "shannon_entropy_ret_16",
        "direction_entropy_20",
        "regime_transition_entropy_48h",
    ),
    "asset_volume_30d": (
        "asset_vol_level",
        "volume",
        "quote_volume",
        "dollar_volume",
        "volume_24h",
        "volume_percentile",
    ),
    "asset_atr_30d": (
        "asset_atr_level",
        "atr_pct",
        "atr_pct_base",
        "atr_12_15m",
        "rv_24h",
        "realized_volatility_24h",
    ),
    "ebm_unc_logodds_var": ("oof_ebm_unc_logodds_var", "ebm_unc_logodds_var"),
    "ebm_unc_pi_width": ("oof_ebm_unc_pi_width", "ebm_unc_pi_width"),
    "ebm_unc_entropy_mean": ("oof_ebm_unc_entropy_mean", "ebm_unc_entropy_mean"),
    "ebm_unc_entropy_std": ("oof_ebm_unc_entropy_std", "ebm_unc_entropy_std"),
    "ebm_unc_conflict_norm": (
        "oof_ebm_unc_conflict_norm",
        "ebm_unc_conflict_norm",
    ),
    "ebm_unc_proximity_min": (
        "oof_ebm_unc_proximity_min",
        "ebm_unc_proximity_min",
    ),
    "ebm_unc_support_mean": ("oof_ebm_unc_support_mean", "ebm_unc_support_mean"),
    "ebm_unc_support_min": ("oof_ebm_unc_support_min", "ebm_unc_support_min"),
    "ebm_unc_concentration": (
        "oof_ebm_unc_concentration",
        "ebm_unc_concentration",
    ),
    "ebm_unc_sign_ratio": ("oof_ebm_unc_sign_ratio", "ebm_unc_sign_ratio"),
    "ebm_unc_interaction_share": (
        "oof_ebm_unc_interaction_share",
        "ebm_unc_interaction_share",
    ),
    "ebm_unc_gap50rel": ("oof_ebm_unc_gap50rel", "ebm_unc_gap50rel"),
    "ebm_unc_support_adjusted_uncertainty": (
        "oof_ebm_unc_support_adjusted_uncertainty",
        "ebm_unc_support_adjusted_uncertainty",
    ),
    "ebm_unc_uncertainty_weight": (
        "oof_ebm_unc_uncertainty_weight",
        "ebm_unc_uncertainty_weight",
    ),
    "ebm_unc_friction_weight": (
        "oof_ebm_unc_friction_weight",
        "ebm_unc_friction_weight",
    ),
}


@dataclass
class RegimeAdaptorFit:
    artifact: Dict[str, Any]
    fixed_diagnostics: pd.DataFrame
    adaptive_diagnostics: pd.DataFrame
    asset_diagnostics: pd.DataFrame
    metrics: pd.DataFrame
    regime_weight_oof: np.ndarray
    eligible_oof: np.ndarray
    deployment_score_oof: np.ndarray
    deployment_score_rank_oof: np.ndarray


def safe_strategy_slug(strategy_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(strategy_id or "")).strip("_")
    return slug[:180] or "strategy"


def _as_float_array(values: Any, n: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(arr) < n:
        out = np.full(n, np.nan, dtype=np.float64)
        out[: len(arr)] = arr
        return out
    return arr[:n].astype(np.float64, copy=False)


def _first_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def _col(df: pd.DataFrame, names: Sequence[str], n: int) -> Optional[np.ndarray]:
    name = _first_col(df, names)
    if name is None:
        return None
    return _as_float_array(df[name].values, n)


def _fill_numeric(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64).copy()
    finite = np.isfinite(x)
    if finite.any():
        med = float(np.nanmedian(x[finite]))
    else:
        med = float(fill)
    x[~finite] = med
    return x.astype(np.float32)


def build_regime_feature_frame(
    feature_frame: pd.DataFrame,
    timestamps: Optional[Sequence[Any]] = None,
    symbols: Optional[Sequence[Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Map available training/live features to the regime adaptor contract."""
    n = len(feature_frame)
    out: Dict[str, np.ndarray] = {}
    mapping: Dict[str, Any] = {}

    rv24 = _col(feature_frame, FEATURE_CANDIDATES["rv_24h"], n)
    if rv24 is not None:
        out["rv_24h"] = _fill_numeric(rv24)
        mapping["rv_24h"] = _first_col(feature_frame, FEATURE_CANDIDATES["rv_24h"])

    rv1 = _col(feature_frame, FEATURE_CANDIDATES["rv1"], n)
    if rv1 is None and "ret1h" in feature_frame.columns:
        rv1 = np.abs(_as_float_array(feature_frame["ret1h"].values, n))
    if rv1 is not None and rv24 is not None:
        out["rv1_rv24"] = _fill_numeric(rv1 / (np.abs(rv24) + EPS))
        mapping["rv1_rv24"] = {
            "numerator": _first_col(feature_frame, FEATURE_CANDIDATES["rv1"])
            or "abs(ret1h)",
            "denominator": mapping.get("rv_24h"),
        }

    rv4 = _col(feature_frame, FEATURE_CANDIDATES["rv4"], n)
    if rv4 is not None and rv24 is not None:
        out["rv4_rv24"] = _fill_numeric(rv4 / (np.abs(rv24) + EPS))
        mapping["rv4_rv24"] = {
            "numerator": _first_col(feature_frame, FEATURE_CANDIDATES["rv4"]),
            "denominator": mapping.get("rv_24h"),
        }

    adx = _col(feature_frame, FEATURE_CANDIDATES["signed_adx"], n)
    if adx is not None:
        sign_src = _col(feature_frame, FEATURE_CANDIDATES["trend_sign"], n)
        if sign_src is not None and (
            _first_col(feature_frame, FEATURE_CANDIDATES["signed_adx"]) or ""
        ).startswith("adx"):
            adx = adx * np.sign(sign_src)
        out["signed_adx"] = _fill_numeric(adx)
        mapping["signed_adx"] = _first_col(
            feature_frame, FEATURE_CANDIDATES["signed_adx"]
        )

    for key in (
        "dist_ema_fast",
        "dist_ema_slow",
        "dist_vwap",
        "prior_day_low",
        "prior_day_high",
        "rvol_z",
        "entropy_24h",
    ):
        arr = _col(feature_frame, FEATURE_CANDIDATES[key], n)
        if arr is not None:
            out[key] = _fill_numeric(arr)
            mapping[key] = _first_col(feature_frame, FEATURE_CANDIDATES[key])

    for key in (
        "ebm_unc_logodds_var",
        "ebm_unc_pi_width",
        "ebm_unc_entropy_mean",
        "ebm_unc_entropy_std",
        "ebm_unc_conflict_norm",
        "ebm_unc_proximity_min",
        "ebm_unc_support_mean",
        "ebm_unc_support_min",
        "ebm_unc_concentration",
        "ebm_unc_sign_ratio",
        "ebm_unc_interaction_share",
        "ebm_unc_gap50rel",
        "ebm_unc_support_adjusted_uncertainty",
        "ebm_unc_uncertainty_weight",
        "ebm_unc_friction_weight",
    ):
        arr = _col(feature_frame, FEATURE_CANDIDATES[key], n)
        if arr is not None:
            out[key] = _fill_numeric(arr)
            mapping[key] = _first_col(feature_frame, FEATURE_CANDIDATES[key])

    if timestamps is not None and len(timestamps) >= n:
        ts = pd.to_datetime(np.asarray(timestamps)[:n], utc=True, errors="coerce")
        out["is_weekend"] = (pd.DatetimeIndex(ts).dayofweek >= 5).astype(np.float32)
        mapping["is_weekend"] = "timestamp.dayofweek>=5"

    sym_arr = (
        np.asarray(symbols).astype(str)[:n]
        if symbols is not None and len(symbols) >= n
        else np.repeat("all", n).astype(str)
    )
    for key in ("asset_volume_30d", "asset_atr_30d"):
        arr = _col(feature_frame, FEATURE_CANDIDATES[key], n)
        if arr is None:
            continue
        series = pd.Series(_fill_numeric(arr), index=np.arange(n))
        group = pd.Series(sym_arr, index=np.arange(n))
        roll = (
            series.groupby(group, sort=False)
            .transform(lambda s: s.shift(1).rolling(30 * 24, min_periods=24).mean())
            .to_numpy(dtype=np.float64)
        )
        fallback = (
            series.groupby(group, sort=False)
            .transform("median")
            .to_numpy(dtype=np.float64)
        )
        out[key] = _fill_numeric(np.where(np.isfinite(roll), roll, fallback))
        mapping[key] = (
            f"rolling30d({_first_col(feature_frame, FEATURE_CANDIDATES[key])})"
        )

    ordered = {key: out[key] for key in REGIME_FEATURE_ORDER if key in out}
    return pd.DataFrame(ordered), mapping


def _rank_pct(scores: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(scores, dtype=np.float64))
    return s.rank(method="average", pct=True).to_numpy(dtype=np.float64)


def _top_mask(scores: np.ndarray, frac: float) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(s)
    mask = np.zeros(len(s), dtype=bool)
    if not finite.any():
        return mask
    n_top = max(1, int(math.ceil(float(np.sum(finite)) * frac)))
    finite_idx = np.where(finite)[0]
    order = finite_idx[np.argsort(s[finite_idx])[-n_top:]]
    mask[order] = True
    return mask


def _drawdown(rets: np.ndarray) -> float:
    r = np.asarray(rets, dtype=np.float64)
    if len(r) == 0:
        return 0.0
    eq = np.cumsum(np.nan_to_num(r, nan=0.0))
    peak = np.maximum.accumulate(eq)
    return float(np.nanmax(peak - eq)) if len(eq) else 0.0


def _period_std(rets: np.ndarray, timestamps: Optional[np.ndarray], freq: str) -> float:
    if timestamps is None or len(timestamps) != len(rets):
        return float(np.nanstd(rets)) if len(rets) > 1 else 0.0
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    df = pd.DataFrame({"ret": rets, "ts": ts}).dropna()
    if df.empty:
        return 0.0
    ts_naive = df["ts"].dt.tz_convert(None)
    grouped = df.groupby(ts_naive.dt.to_period(freq))["ret"].sum()
    return float(grouped.std(ddof=0)) if len(grouped) > 1 else 0.0


def _period_count(rets: np.ndarray, timestamps: Optional[np.ndarray], freq: str) -> int:
    if timestamps is None or len(timestamps) != len(rets):
        return 0
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    df = pd.DataFrame({"ret": rets, "ts": ts}).dropna()
    if df.empty:
        return 0
    ts_naive = df["ts"].dt.tz_convert(None)
    return int(df.groupby(ts_naive.dt.to_period(freq))["ret"].sum().shape[0])


def score_metrics(
    scores: np.ndarray,
    returns: np.ndarray,
    timestamps: Optional[Sequence[Any]] = None,
    *,
    top_fracs: Sequence[float] = (0.01, 0.05, 0.10, 0.20),
    cost_pct: float = 0.003,
) -> pd.DataFrame:
    s = np.asarray(scores, dtype=np.float64)
    r = np.asarray(returns, dtype=np.float64)
    n = min(len(s), len(r))
    s, r = s[:n], r[:n]
    ts = (
        np.asarray(timestamps)[:n]
        if timestamps is not None and len(timestamps) >= n
        else None
    )
    finite = np.isfinite(s) & np.isfinite(r)
    overall_hit = float(np.mean(r[finite] > 0.0)) if finite.any() else 0.0
    rows: List[Dict[str, Any]] = []
    for frac in top_fracs:
        mask = _top_mask(np.where(finite, s, np.nan), frac)
        sel = mask & finite
        sr = r[sel]
        sts = ts[sel] if ts is not None else None
        net = sr - cost_pct
        hit = float(np.mean(sr > 0.0)) if len(sr) else 0.0
        n_sel = int(len(sr))
        gross_std = float(np.std(sr)) if n_sel > 1 else 0.0
        net_std = float(np.std(net)) if n_sel > 1 else 0.0
        hit_se = float(math.sqrt(max(hit * (1.0 - hit), 0.0) / n_sel)) if n_sel else 0.0
        gross_se = float(gross_std / math.sqrt(n_sel)) if n_sel > 1 else 0.0
        net_se = float(net_std / math.sqrt(n_sel)) if n_sel > 1 else 0.0
        ds = net[net < 0.0]
        downside_std = float(np.std(ds)) if len(ds) > 1 else 0.0
        sortino = float(np.mean(net) / (downside_std + EPS)) if len(net) else 0.0
        equity = np.cumsum(net)
        stability = 0.0
        if len(equity) > 5:
            try:
                _, _, r_val, _, _ = linregress(np.arange(len(equity)), equity)
                stability = float(r_val**2) if np.isfinite(r_val) else 0.0
            except Exception:
                stability = 0.0
        rows.append(
            {
                "top_frac": float(frac),
                "lift": float(hit / (overall_hit + EPS)),
                "hit_rate": hit,
                "hit_rate_se": hit_se,
                "lift_se_approx": float(hit_se / (overall_hit + EPS)),
                "mean_gross_return": float(np.mean(sr)) if len(sr) else 0.0,
                "mean_gross_return_se": gross_se,
                "mean_net_return": float(np.mean(net)) if len(net) else 0.0,
                "mean_net_return_se": net_se,
                "net_ret": float(np.sum(net)) if len(net) else 0.0,
                "return_std": net_std,
                "std_weekly": _period_std(net, sts, "W"),
                "std_monthly": _period_std(net, sts, "M"),
                "weekly_periods": _period_count(net, sts, "W"),
                "monthly_periods": _period_count(net, sts, "M"),
                "worst_week_loss": _worst_period_loss(net, sts, "W"),
                "worst_month_loss": _worst_period_loss(net, sts, "M"),
                "sortino": sortino,
                "max_drawdown": _drawdown(net),
                "trades": n_sel,
                "lift_gt_1": bool((hit / (overall_hit + EPS)) > 1.0),
            }
        )
    return pd.DataFrame(rows)


def _worst_period_loss(
    rets: np.ndarray, timestamps: Optional[np.ndarray], freq: str
) -> float:
    if timestamps is None or len(timestamps) != len(rets):
        return float(abs(min(float(np.nansum(rets)), 0.0)))
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    df = pd.DataFrame({"ret": rets, "ts": ts}).dropna()
    if df.empty:
        return 0.0
    ts_naive = df["ts"].dt.tz_convert(None)
    grouped = df.groupby(ts_naive.dt.to_period(freq))["ret"].sum()
    return float(abs(min(float(grouped.min()), 0.0))) if len(grouped) else 0.0


def _safe_ratio(num: float, den: float, neutral: float = 1.0) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < EPS:
        return float(neutral)
    return float(num / den)


def _edge_score(bucket: Dict[str, float], strategy: Dict[str, float]) -> float:
    lift_ratio = _safe_ratio(bucket.get("lift", 0.0), strategy.get("lift", 0.0))
    gross_ratio = _safe_ratio(
        bucket.get("mean_gross_return", 0.0), strategy.get("mean_gross_return", 0.0)
    )
    hit_ratio = _safe_ratio(bucket.get("hit_rate", 0.0), strategy.get("hit_rate", 0.0))
    std_ratio = _safe_ratio(
        bucket.get("return_std", 0.0), strategy.get("return_std", 0.0)
    )
    dd_ratio = _safe_ratio(
        bucket.get("max_drawdown", 0.0), strategy.get("max_drawdown", 0.0)
    )
    return float(
        0.20 * math.log(max(lift_ratio, EPS))
        + 0.25 * math.log(max(gross_ratio, EPS))
        + 0.15 * math.log(max(hit_ratio, EPS))
        - 0.20 * math.log(max(std_ratio, EPS))
        - 0.20 * math.log(max(dd_ratio, EPS))
    )


def _fit_percentile(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.array([0.0], dtype=np.float64)
    if len(x) > 5000:
        qs = np.linspace(0.0, 1.0, 1001)
        return np.quantile(x, qs).astype(np.float64)
    return np.sort(x).astype(np.float64)


def _apply_percentile(values: np.ndarray, ref: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    if len(r) == 0:
        return np.full(len(x), 0.5, dtype=np.float64)
    pct = np.searchsorted(r, x, side="right") / max(len(r), 1)
    pct = np.where(np.isfinite(x), pct, 0.5)
    return np.clip(pct, 0.01, 0.99).astype(np.float64)


def _walk_forward_splits(
    timestamps: Sequence[Any], n: int, n_splits: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n < 20:
        return [(np.arange(0, max(1, n // 2)), np.arange(max(1, n // 2), n))]
    ts = pd.to_datetime(np.asarray(timestamps)[:n], utc=True, errors="coerce")
    order = np.argsort(np.where(pd.isna(ts), np.arange(n), ts.view("int64")))
    folds = np.array_split(order, n_splits)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, len(folds)):
        tr = np.concatenate(folds[:i])
        te = folds[i]
        if len(tr) and len(te):
            out.append((np.sort(tr), np.sort(te)))
    if not out:
        split = max(1, n // 2)
        out.append((np.arange(split), np.arange(split, n)))
    return out


def _rank_weight(scores: np.ndarray) -> np.ndarray:
    ranked = _rank_pct(scores)
    rank_in_top10 = np.clip((ranked - 0.90) / 0.10, 0.0, 1.0)
    return (1.0 + 0.5 * rank_in_top10).astype(np.float32)


def _feature_effect_from_stats(pct: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    centers = np.asarray(stats.get("spline_x", []), dtype=np.float64)
    values = np.asarray(stats.get("spline_y", []), dtype=np.float64)
    if len(centers) < 2 or len(values) < 2:
        return np.zeros(len(pct), dtype=np.float32)
    order = np.argsort(centers)
    y = np.interp(np.asarray(pct, dtype=np.float64), centers[order], values[order])
    clip = stats.get("log_effect_clip", [-0.10, 0.10])
    lo, hi = float(clip[0]), float(clip[1])
    return np.clip(y, lo, hi).astype(np.float32)


def _fit_feature_stats(
    pct: np.ndarray,
    scores: np.ndarray,
    returns: np.ndarray,
    timestamps: Optional[np.ndarray],
    *,
    min_bucket_n: int = 300,
    shrink_k: float = 1500.0,
    tree_min_leaf_frac: float = 0.05,
    max_leaf_nodes: int = 4,
    max_depth: int = 3,
    ccp_alpha: float = 0.001,
    max_bin_share: float = 0.72,
    rank_weight: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    top = _top_mask(scores, 0.05)
    n_top = int(np.sum(top & np.isfinite(pct) & np.isfinite(returns)))
    min_leaf = max(int(min_bucket_n), int(float(tree_min_leaf_frac) * n_top))
    if n_top < max(2 * min_leaf, 2 * min_bucket_n):
        return {
            "enabled": False,
            "reason": "insufficient_top5",
            "spline_x": [],
            "spline_y": [],
        }
    target = returns - float(np.nanmean(returns[top]))
    w = rank_weight if rank_weight is not None else np.ones(len(pct), dtype=np.float32)
    top_idx = np.where(top & np.isfinite(pct) & np.isfinite(target))[0]
    tree_target = target[top_idx] * 100.0
    thresholds: List[float] = []
    tree_alpha = float(ccp_alpha)
    alpha_candidates = [
        float(ccp_alpha),
        min(float(ccp_alpha), 0.0003),
        min(float(ccp_alpha), 0.0001),
        min(float(ccp_alpha), 0.00003),
        0.0,
    ]
    alpha_candidates = list(dict.fromkeys(alpha_candidates))
    balance_counts: List[int] = []
    for alpha in alpha_candidates:
        tree = DecisionTreeRegressor(
            max_leaf_nodes=int(max_leaf_nodes),
            max_depth=int(max_depth),
            min_samples_leaf=min_leaf,
            ccp_alpha=alpha,
            random_state=42,
        )
        tree.fit(
            pct[top_idx].reshape(-1, 1),
            tree_target,
            sample_weight=w[top_idx],
        )
        thresholds = sorted(
            float(t) for t in tree.tree_.threshold if np.isfinite(t) and 0.01 < t < 0.99
        )
        tree_alpha = float(alpha)
        if thresholds:
            candidate_edges = np.array([0.0] + thresholds + [1.0], dtype=np.float64)
            bin_ids = np.searchsorted(candidate_edges[1:-1], pct[top_idx], side="right")
            counts = np.bincount(bin_ids, minlength=len(candidate_edges) - 1)
            non_zero = counts[counts > 0]
            balance_counts = [int(x) for x in counts.tolist()]
            if (
                len(non_zero) >= 2
                and int(np.min(non_zero)) >= min_leaf
                and float(np.max(non_zero) / max(1, np.sum(non_zero)))
                <= float(max_bin_share)
            ):
                break
            thresholds = []
            balance_counts = []
        if thresholds:
            break
    edges = np.array([0.0] + thresholds + [1.0], dtype=np.float64)
    if len(edges) < 3:
        return {
            "enabled": False,
            "reason": "too_few_balanced_bins",
            "spline_x": [],
            "spline_y": [],
        }
    strategy_top = (
        score_metrics(scores, returns, timestamps, top_fracs=(0.05,)).iloc[0].to_dict()
    )
    rows: List[Dict[str, Any]] = []
    centers: List[float] = []
    ys: List[float] = []
    for b in range(len(edges) - 1):
        lo, hi = float(edges[b]), float(edges[b + 1])
        mask = (pct >= lo) & (pct < hi if b < len(edges) - 2 else pct <= hi)
        weighted_n = float(np.sum(w[mask & top]))
        if weighted_n < min_bucket_n:
            continue
        local_scores = np.where(mask, scores, np.nan)
        bm = (
            score_metrics(local_scores, returns, timestamps, top_fracs=(0.05,))
            .iloc[0]
            .to_dict()
        )
        edge = _edge_score(bm, strategy_top)
        reliability = weighted_n / (weighted_n + shrink_k)
        shrunk = float(np.clip(edge * reliability, -0.12, 0.12))
        center = float(np.clip((lo + hi) * 0.5, 0.01, 0.99))
        centers.append(center)
        ys.append(shrunk)
        rows.append(
            {
                "lo": lo,
                "hi": hi,
                "center": center,
                "weighted_n": weighted_n,
                "edge_score": edge,
                "shrunk_edge_score": shrunk,
                **bm,
            }
        )
    if len(centers) < 2:
        return {
            "enabled": False,
            "reason": "too_few_valid_bins",
            "spline_x": [],
            "spline_y": [],
            "bins": rows,
        }
    return {
        "enabled": True,
        "edges": edges.tolist(),
        "tree_ccp_alpha": tree_alpha,
        "tree_min_leaf_frac": float(tree_min_leaf_frac),
        "tree_min_samples_leaf": int(min_leaf),
        "tree_max_leaf_nodes": int(max_leaf_nodes),
        "tree_max_depth": int(max_depth),
        "tree_max_bin_share": float(max_bin_share),
        "tree_top_bin_counts": balance_counts,
        "min_bucket_n": int(min_bucket_n),
        "shrink_k": float(shrink_k),
        "spline_x": centers,
        "spline_y": ys,
        "bins": rows,
    }


def _fit_feature_stats_for_params(
    regime_df: pd.DataFrame,
    features: Sequence[str],
    percentile_refs: Dict[str, np.ndarray],
    scores: np.ndarray,
    returns: np.ndarray,
    timestamps: Optional[np.ndarray],
    rank_weight: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], np.ndarray]:
    stats: Dict[str, Any] = {}
    for feat in features:
        pct = _apply_percentile(regime_df[feat].values, percentile_refs[feat])
        feat_stats = _fit_feature_stats(
            pct,
            scores,
            returns,
            timestamps,
            min_bucket_n=int(params.get("min_bucket_n", 300)),
            shrink_k=float(params.get("shrink_k", 1500.0)),
            tree_min_leaf_frac=float(params.get("tree_min_leaf_frac", 0.05)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", 4)),
            max_depth=int(params.get("max_depth", 3)),
            ccp_alpha=float(params.get("ccp_alpha", 0.001)),
            max_bin_share=float(params.get("max_bin_share", 0.72)),
            rank_weight=rank_weight,
        )
        feat_stats["log_effect_clip"] = list(
            params.get("log_effect_clip", [-0.10, 0.10])
        )
        stats[feat] = feat_stats
    effects = _effects_from_artifact(
        regime_df,
        {
            "features": list(features),
            "percentile_refs": {k: v.tolist() for k, v in percentile_refs.items()},
            "feature_splines": stats,
        },
    )
    return stats, effects


def _select_spline_hyperparams(
    regime_df: pd.DataFrame,
    features: Sequence[str],
    percentile_refs: Dict[str, np.ndarray],
    scores: np.ndarray,
    returns: np.ndarray,
    timestamps: Optional[np.ndarray],
    rank_weight: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray]:
    base_params: Dict[str, Any] = {
        "min_bucket_n": 300,
        "shrink_k": 1500.0,
        "tree_min_leaf_frac": 0.05,
        "max_leaf_nodes": 4,
        "max_depth": 3,
        "ccp_alpha": 0.001,
        "max_bin_share": 0.72,
        "log_effect_clip": [-0.10, 0.10],
    }

    def score_params(
        params: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any], np.ndarray]:
        stats, effects = _fit_feature_stats_for_params(
            regime_df,
            features,
            percentile_refs,
            scores,
            returns,
            timestamps,
            rank_weight,
            params,
        )
        enabled = sum(1 for s in stats.values() if bool(s.get("enabled", False)))
        if enabled <= 0:
            return float("-inf"), stats, effects
        log_effect = np.clip(
            np.sum(effects.astype(np.float64), axis=1),
            -0.35,
            0.22,
        )
        adjusted = scores * np.clip(np.exp(log_effect), 0.70, 1.25)
        value = _objective(scores, adjusted, returns, timestamps)
        value += 0.002 * min(enabled, 8)
        return float(value), stats, effects

    best_score, best_stats, best_effects = score_params(base_params)
    best_params = dict(base_params)
    trials: List[Dict[str, Any]] = [
        {
            "trial": -1,
            "value": float(best_score),
            "params": dict(base_params),
            "enabled_features": int(
                sum(1 for s in best_stats.values() if bool(s.get("enabled", False)))
            ),
        }
    ]
    if optuna is not None:
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42) if TPESampler is not None else None,
            pruner=(
                MedianPruner(n_startup_trials=4, n_min_trials=2)
                if MedianPruner is not None
                else None
            ),
        )

        def objective(trial: Any) -> float:
            clip_hi = float(trial.suggest_float("log_effect_clip_hi", 0.06, 0.14))
            params = {
                "min_bucket_n": int(
                    trial.suggest_categorical("min_bucket_n", [200, 300, 450])
                ),
                "shrink_k": float(
                    trial.suggest_categorical("shrink_k", [800.0, 1500.0, 2500.0])
                ),
                "tree_min_leaf_frac": float(
                    trial.suggest_float("tree_min_leaf_frac", 0.035, 0.09)
                ),
                "max_leaf_nodes": int(
                    trial.suggest_categorical("max_leaf_nodes", [3, 4])
                ),
                "max_depth": int(trial.suggest_categorical("max_depth", [2, 3])),
                "ccp_alpha": float(
                    trial.suggest_categorical(
                        "ccp_alpha", [0.003, 0.001, 0.0003, 0.0001]
                    )
                ),
                "max_bin_share": float(
                    trial.suggest_float("max_bin_share", 0.60, 0.78)
                ),
                "log_effect_clip": [-clip_hi, clip_hi],
            }
            value, stats, _effects = score_params(params)
            enabled = int(
                sum(1 for s in stats.values() if bool(s.get("enabled", False)))
            )
            trial.set_user_attr("enabled_features", enabled)
            trial.report(float(value), step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()
            return float(value)

        study.optimize(
            objective, n_trials=12, gc_after_trial=True, show_progress_bar=False
        )
        for tr in study.trials:
            if tr.value is None:
                continue
            params = {
                "min_bucket_n": int(tr.params.get("min_bucket_n", 300)),
                "shrink_k": float(tr.params.get("shrink_k", 1500.0)),
                "tree_min_leaf_frac": float(tr.params.get("tree_min_leaf_frac", 0.05)),
                "max_leaf_nodes": int(tr.params.get("max_leaf_nodes", 4)),
                "max_depth": int(tr.params.get("max_depth", 3)),
                "ccp_alpha": float(tr.params.get("ccp_alpha", 0.001)),
                "max_bin_share": float(tr.params.get("max_bin_share", 0.72)),
                "log_effect_clip": [
                    -float(tr.params.get("log_effect_clip_hi", 0.10)),
                    float(tr.params.get("log_effect_clip_hi", 0.10)),
                ],
            }
            trials.append(
                {
                    "trial": int(tr.number),
                    "value": float(tr.value),
                    "params": params,
                    "enabled_features": int(tr.user_attrs.get("enabled_features", 0)),
                }
            )
        if study.best_trial is not None and study.best_value > best_score:
            bp = study.best_trial.params
            clip_hi = float(bp.get("log_effect_clip_hi", 0.10))
            best_params = {
                "min_bucket_n": int(bp.get("min_bucket_n", 300)),
                "shrink_k": float(bp.get("shrink_k", 1500.0)),
                "tree_min_leaf_frac": float(bp.get("tree_min_leaf_frac", 0.05)),
                "max_leaf_nodes": int(bp.get("max_leaf_nodes", 4)),
                "max_depth": int(bp.get("max_depth", 3)),
                "ccp_alpha": float(bp.get("ccp_alpha", 0.001)),
                "max_bin_share": float(bp.get("max_bin_share", 0.72)),
                "log_effect_clip": [-clip_hi, clip_hi],
            }
            best_score, best_stats, best_effects = score_params(best_params)

    best_params["objective"] = float(best_score)
    best_params["trials"] = _jsonify(trials)
    return best_params, best_stats, best_effects


def _effects_from_artifact(
    regime_df: pd.DataFrame, artifact: Dict[str, Any]
) -> np.ndarray:
    features = list(artifact.get("features", []))
    n = len(regime_df)
    effects = np.zeros((n, len(features)), dtype=np.float32)
    refs = artifact.get("percentile_refs", {})
    stats = artifact.get("feature_splines", {})
    for j, feat in enumerate(features):
        if feat not in regime_df.columns:
            continue
        pct = _apply_percentile(
            regime_df[feat].to_numpy(dtype=np.float64),
            np.asarray(refs.get(feat, [0.0]), dtype=np.float64),
        )
        effects[:, j] = _feature_effect_from_stats(pct, stats.get(feat, {}))
    return effects


def apply_regime_adaptor(
    feature_frame: pd.DataFrame,
    pred_calibrated: Sequence[float],
    artifact: Dict[str, Any],
    timestamps: Optional[Sequence[Any]] = None,
    symbols: Optional[Sequence[Any]] = None,
) -> Dict[str, np.ndarray]:
    n = len(pred_calibrated)
    score = _as_float_array(pred_calibrated, n)
    regime_df, _ = build_regime_feature_frame(
        feature_frame.iloc[:n], timestamps, symbols
    )
    effects = _effects_from_artifact(regime_df, artifact)
    scaler = artifact.get("elastic_net", {}).get("scaler", {})
    center = np.asarray(
        scaler.get("center", np.zeros(effects.shape[1])), dtype=np.float64
    )
    scale = np.asarray(scaler.get("scale", np.ones(effects.shape[1])), dtype=np.float64)
    if len(center) != effects.shape[1]:
        center = np.zeros(effects.shape[1], dtype=np.float64)
    if len(scale) != effects.shape[1]:
        scale = np.ones(effects.shape[1], dtype=np.float64)
    x_scaled = (effects.astype(np.float64) - center) / np.where(
        np.abs(scale) > EPS, scale, 1.0
    )
    coefs = np.asarray(
        artifact.get("elastic_net", {}).get("coef", np.zeros(effects.shape[1])),
        dtype=np.float64,
    )
    if len(coefs) != effects.shape[1]:
        coefs = np.zeros(effects.shape[1], dtype=np.float64)
    intercept = float(artifact.get("elastic_net", {}).get("intercept", 0.0))
    train_mean = float(
        artifact.get("elastic_net", {}).get("train_prediction_mean", 0.0)
    )
    log_weight = x_scaled @ coefs + intercept - train_mean
    clips = artifact.get("clips", {})
    log_lo, log_hi = clips.get("total_log_weight_clip", [-0.35, 0.22])
    wt_lo, wt_hi = clips.get("regime_weight_clip", [0.70, 1.25])
    log_weight = np.clip(log_weight, float(log_lo), float(log_hi))
    weight = np.clip(np.exp(log_weight), float(wt_lo), float(wt_hi)).astype(np.float64)
    eligible = np.ones(n, dtype=bool)
    if bool(artifact.get("enable_regime_adaptor", False)):
        eligible &= ~_apply_bucket_gates(regime_df, artifact)
        eligible &= ~_apply_asset_gates(
            (
                np.asarray(symbols).astype(str)[:n]
                if symbols is not None and len(symbols) >= n
                else np.repeat("all", n)
            ),
            artifact,
        )
    else:
        weight[:] = 1.0
    deployment = score * weight
    deployment[~eligible] = -np.inf
    rank = _rank_pct(np.where(np.isfinite(deployment), deployment, np.nan))
    rank[~np.isfinite(deployment)] = 0.0
    return {
        "regime_weight": weight.astype(np.float64),
        "eligible": eligible,
        "deployment_score": deployment.astype(np.float64),
        "deployment_score_rank": rank.astype(np.float64),
        "spline_effects": effects.astype(np.float32),
    }


def _apply_bucket_gates(
    regime_df: pd.DataFrame, artifact: Dict[str, Any]
) -> np.ndarray:
    n = len(regime_df)
    gated = np.zeros(n, dtype=bool)
    for gate in artifact.get("bucket_gates", []):
        feat = gate.get("feature")
        if feat not in regime_df.columns:
            continue
        ref = np.asarray(
            artifact.get("percentile_refs", {}).get(feat, [0.0]), dtype=np.float64
        )
        pct = _apply_percentile(regime_df[feat].to_numpy(dtype=np.float64), ref)
        lo, hi = float(gate.get("lo", 0.0)), float(gate.get("hi", 1.0))
        gated |= (pct >= lo) & (pct < hi if hi < 1.0 else pct <= hi)
    return gated


def _apply_asset_gates(symbols: np.ndarray, artifact: Dict[str, Any]) -> np.ndarray:
    gated_assets = {str(x) for x in artifact.get("asset_gates", [])}
    if not gated_assets:
        return np.zeros(len(symbols), dtype=bool)
    return np.asarray([str(s) in gated_assets for s in symbols], dtype=bool)


def _fit_elastic_net(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    scores: np.ndarray,
    returns: np.ndarray,
    timestamps: Optional[np.ndarray],
) -> Tuple[ElasticNet, RobustScaler, float, Dict[str, float]]:
    scaler = RobustScaler()
    x_scaled = scaler.fit_transform(x)
    candidates: List[Tuple[float, float]] = [
        (0.00003, 0.05),
        (0.0001, 0.05),
        (0.0003, 0.10),
        (0.001, 0.10),
        (0.003, 0.20),
        (0.01, 0.20),
        (0.1, 0.5),
        (0.3, 0.5),
        (1.0, 0.5),
        (0.3, 0.2),
        (0.3, 0.8),
    ]
    if optuna is not None:
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42) if TPESampler is not None else None,
            pruner=(
                MedianPruner(n_startup_trials=8, n_min_trials=4)
                if MedianPruner is not None
                else None
            ),
        )
        best_seen = {"trial": -1, "value": -np.inf}

        def objective(trial: Any) -> float:
            alpha = float(trial.suggest_float("alpha", 1e-5, 10.0, log=True))
            l1_ratio = float(trial.suggest_float("l1_ratio", 0.1, 0.9))
            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=True,
                max_iter=5000,
                random_state=42,
            )
            model.fit(x_scaled, y, sample_weight=weights)
            pred = np.asarray(model.predict(x_scaled), dtype=np.float64)
            pred -= float(np.nanmean(pred))
            adjusted = scores * np.clip(np.exp(np.clip(pred, -0.35, 0.22)), 0.70, 1.25)
            val = _objective(
                raw_scores=scores,
                adjusted_scores=adjusted,
                returns=returns,
                timestamps=timestamps,
            )
            if val > best_seen["value"]:
                best_seen.update({"trial": int(trial.number), "value": float(val)})
            elif int(trial.number) - int(best_seen["trial"]) >= 25:
                study.stop()
            return float(val)

        study.optimize(
            objective, n_trials=50, gc_after_trial=True, show_progress_bar=False
        )
        if study.best_trial is not None:
            candidates.insert(
                0,
                (
                    float(study.best_trial.params["alpha"]),
                    float(study.best_trial.params["l1_ratio"]),
                ),
            )
    best_score = -np.inf
    best_pair = candidates[0]
    for alpha, l1_ratio in candidates:
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=True,
            max_iter=5000,
            random_state=42,
        )
        model.fit(x_scaled, y, sample_weight=weights)
        pred = np.asarray(model.predict(x_scaled), dtype=np.float64)
        pred -= float(np.nanmean(pred))
        adjusted = scores * np.clip(np.exp(np.clip(pred, -0.35, 0.22)), 0.70, 1.25)
        val = _objective(
            raw_scores=scores,
            adjusted_scores=adjusted,
            returns=returns,
            timestamps=timestamps,
        )
        if val > best_score:
            best_score = val
            best_pair = (alpha, l1_ratio)
    final = ElasticNet(
        alpha=best_pair[0],
        l1_ratio=best_pair[1],
        fit_intercept=True,
        max_iter=5000,
        random_state=42,
    )
    final.fit(x_scaled, y, sample_weight=weights)
    train_mean = float(np.nanmean(final.predict(x_scaled)))
    return (
        final,
        scaler,
        train_mean,
        {
            "alpha": float(best_pair[0]),
            "l1_ratio": float(best_pair[1]),
            "objective": float(best_score),
        },
    )


def _objective(
    raw_scores: np.ndarray,
    adjusted_scores: np.ndarray,
    returns: np.ndarray,
    timestamps: Optional[np.ndarray],
) -> float:
    raw_df = score_metrics(raw_scores, returns, timestamps, top_fracs=(0.01, 0.05))
    adj_df = score_metrics(adjusted_scores, returns, timestamps, top_fracs=(0.01, 0.05))
    weights = {0.01: 0.35, 0.05: 0.65}
    objective = 0.0
    weight_sum = 0.0
    for frac, weight in weights.items():
        raw_rows = raw_df[np.isclose(raw_df["top_frac"].astype(float), frac)]
        adj_rows = adj_df[np.isclose(adj_df["top_frac"].astype(float), frac)]
        if raw_rows.empty or adj_rows.empty:
            continue
        raw = raw_rows.iloc[0].to_dict()
        adj = adj_rows.iloc[0].to_dict()
        std_weekly_ratio = _safe_ratio(adj["std_weekly"], raw["std_weekly"])
        std_monthly_ratio = _safe_ratio(adj["std_monthly"], raw["std_monthly"])
        worst_week_loss_ratio = _safe_ratio(
            adj["worst_week_loss"], raw["worst_week_loss"]
        )
        worst_month_loss_ratio = _safe_ratio(
            adj["worst_month_loss"], raw["worst_month_loss"]
        )
        net_ret_ratio = _safe_ratio(adj["net_ret"], raw["net_ret"])
        lift_ratio = _safe_ratio(adj["lift"], raw["lift"])
        frac_score = float(
            -0.15 * math.log(max(std_weekly_ratio, EPS))
            - 0.15 * math.log(max(std_monthly_ratio, EPS))
            - 0.15 * math.log(max(worst_week_loss_ratio, EPS))
            - 0.15 * math.log(max(worst_month_loss_ratio, EPS))
            + 0.20 * math.log(max(net_ret_ratio, EPS))
            + 0.10 * math.log(max(lift_ratio, EPS))
            - 0.30 * max(0.0, 0.98 - net_ret_ratio)
            - 0.20 * max(0.0, 0.98 - lift_ratio)
        )
        objective += weight * frac_score
        weight_sum += weight
    if weight_sum <= 0.0:
        return float("-inf")
    return float(objective / weight_sum)


def _regime_enable_decision(summary: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    improve_eps = 0.001
    worse_tol = 0.015
    top_priority = {0.05: 0, 0.01: 1}
    best_any_score = -np.inf
    best_pass_key: Tuple[int, float] = (999, np.inf)
    best_any_decision: Dict[str, Any] = {
        "enabled": False,
        "reason": "no_candidate_passed_loose_gate",
        "improve_eps": improve_eps,
        "worse_tol": worse_tol,
        "top_priority": top_priority,
    }
    best_pass_decision: Dict[str, Any] = {}
    for _, r in summary.iterrows():
        top_frac = float(r.get("top_frac", np.nan))
        if round(top_frac, 2) not in top_priority:
            continue
        lift_ratio = float(r.get("lift_ratio", 1.0))
        net_ret_ratio = float(r.get("net_ret_ratio", 1.0))
        gross_ret_ratio = float(r.get("gross_ret_ratio", 1.0))
        std_ratio = float(r.get("std_ratio", 1.0))
        dd_ratio = float(r.get("dd_ratio", 1.0))
        improvements = {
            "lift": lift_ratio > 1.0 + improve_eps,
            "net_ret": net_ret_ratio > 1.0 + improve_eps,
            "gross_ret": gross_ret_ratio > 1.0 + improve_eps,
            "return_std": std_ratio < 1.0 - improve_eps,
            "max_drawdown": dd_ratio < 1.0 - improve_eps,
        }
        no_material_worse = (
            lift_ratio >= 1.0 - worse_tol
            and net_ret_ratio >= 1.0 - worse_tol
            and gross_ret_ratio >= 1.0 - worse_tol
            and std_ratio <= 1.0 + worse_tol
            and dd_ratio <= 1.0 + worse_tol
        )
        score = float(
            0.35 * math.log(max(lift_ratio, EPS))
            + 0.25 * math.log(max(net_ret_ratio, EPS))
            + 0.15 * math.log(max(gross_ret_ratio, EPS))
            - 0.15 * math.log(max(std_ratio, EPS))
            - 0.10 * math.log(max(dd_ratio, EPS))
        )
        enabled = bool(no_material_worse and any(improvements.values()) and score > 0.0)
        decision = {
            "enabled": enabled,
            "reason": (
                "loose_gate_passed" if enabled else "best_candidate_failed_loose_gate"
            ),
            "top_frac": top_frac,
            "selection_score": score,
            "improvements": improvements,
            "no_material_worse": bool(no_material_worse),
            "improve_eps": improve_eps,
            "worse_tol": worse_tol,
            "top_priority": top_priority,
            "ratios": {
                "lift_ratio": lift_ratio,
                "net_ret_ratio": net_ret_ratio,
                "gross_ret_ratio": gross_ret_ratio,
                "std_ratio": std_ratio,
                "dd_ratio": dd_ratio,
            },
        }
        if score > best_any_score:
            best_any_score = score
            best_any_decision = decision
        pass_key = (
            int(top_priority.get(round(top_frac, 2), 999)),
            -score,
        )
        if enabled and pass_key < best_pass_key:
            best_pass_key = pass_key
            best_pass_decision = {
                **decision,
                "enabled": enabled,
            }
    if best_pass_decision:
        return True, best_pass_decision
    return False, best_any_decision


def fit_regime_adaptor(
    feature_frame: pd.DataFrame,
    pred_calibrated: Sequence[float],
    returns: Sequence[float],
    timestamps: Optional[Sequence[Any]],
    symbols: Optional[Sequence[Any]],
    *,
    strategy_id: str,
    model_name: str,
    cost_pct: float = 0.003,
) -> RegimeAdaptorFit:
    n = min(len(feature_frame), len(pred_calibrated), len(returns))
    scores = _as_float_array(pred_calibrated, n)
    rets = _as_float_array(returns, n)
    ts = (
        np.asarray(timestamps)[:n]
        if timestamps is not None and len(timestamps) >= n
        else None
    )
    sy = (
        np.asarray(symbols).astype(str)[:n]
        if symbols is not None and len(symbols) >= n
        else np.repeat("all", n)
    )
    regime_df, mapping = build_regime_feature_frame(feature_frame.iloc[:n], ts, sy)
    features = [f for f in REGIME_FEATURE_ORDER if f in regime_df.columns]
    if not features or n < 50:
        artifact = _empty_artifact(strategy_id, model_name, features, mapping)
        applied = apply_regime_adaptor(feature_frame.iloc[:n], scores, artifact, ts, sy)
        empty = pd.DataFrame()
        return RegimeAdaptorFit(
            artifact,
            empty,
            empty,
            empty,
            score_metrics(scores, rets, ts),
            applied["regime_weight"],
            applied["eligible"],
            applied["deployment_score"],
            applied["deployment_score_rank"],
        )

    rank_weight = _rank_weight(scores)
    percentile_refs = {
        feat: _fit_percentile(regime_df[feat].values) for feat in features
    }
    spline_hpo, full_stats, effects = _select_spline_hyperparams(
        regime_df,
        features,
        percentile_refs,
        scores,
        rets,
        ts,
        rank_weight,
    )
    adaptive_rows: List[Dict[str, Any]] = []
    for feat in features:
        stats = full_stats.get(feat, {})
        for row in stats.get("bins", []):
            adaptive_rows.append(
                {
                    "strategy_id": strategy_id,
                    "model": model_name,
                    "feature": feat,
                    **row,
                }
            )

    fixed = fixed_bucket_diagnostics(
        regime_df, scores, rets, ts, sy, strategy_id, model_name, percentile_refs
    )
    asset_diag = asset_diagnostics(scores, rets, ts, sy, strategy_id, model_name)
    bucket_gates = _bucket_gates(fixed)
    asset_gates = _asset_gates(asset_diag)

    top = _top_mask(scores, 0.10)
    target_center = (
        float(np.nanmean(rets[top])) if top.any() else float(np.nanmean(rets))
    )
    target = (rets - target_center).astype(np.float64)
    model, scaler, train_mean, params = _fit_elastic_net(
        effects, target, rank_weight, scores, rets, ts
    )
    artifact = {
        "schema_version": "v1",
        "strategy_id": str(strategy_id),
        "model_name": str(model_name),
        "features": features,
        "feature_mapping": mapping,
        "percentile_refs": {
            k: v.astype(float).tolist() for k, v in percentile_refs.items()
        },
        "feature_splines": _jsonify(full_stats),
        "spline_hpo": _jsonify(spline_hpo),
        "elastic_net": {
            "coef": np.asarray(model.coef_, dtype=float).tolist(),
            "intercept": float(model.intercept_),
            "train_prediction_mean": train_mean,
            "params": params,
            "scaler": {
                "center": np.asarray(
                    getattr(scaler, "center_", np.zeros(effects.shape[1])), dtype=float
                ).tolist(),
                "scale": np.asarray(
                    getattr(scaler, "scale_", np.ones(effects.shape[1])), dtype=float
                ).tolist(),
            },
        },
        "clips": {
            "log_effect_clip": [-0.10, 0.10],
            "group_clip": [-0.12, 0.12],
            "total_log_weight_clip": [-0.35, 0.22],
            "regime_weight_clip": [0.70, 1.25],
        },
        "bucket_gates": bucket_gates,
        "asset_gates": asset_gates,
        "rank_normalization": {
            "method": "pandas_rank_pct_average",
            "score": "deployment_score",
        },
        "enable_regime_adaptor": False,
    }
    candidate_applied = apply_regime_adaptor(
        feature_frame.iloc[:n],
        scores,
        artifact | {"enable_regime_adaptor": True},
        ts,
        sy,
    )
    raw_m = score_metrics(
        scores, rets, ts, top_fracs=(0.01, 0.05, 0.10, 0.20), cost_pct=cost_pct
    )
    candidate_m = score_metrics(
        candidate_applied["deployment_score_rank"],
        rets,
        ts,
        top_fracs=(0.01, 0.05, 0.10, 0.20),
        cost_pct=cost_pct,
    )
    summary = _compare_metrics(raw_m, candidate_m)
    enabled, enable_decision = _regime_enable_decision(summary)
    artifact["enable_regime_adaptor"] = bool(enabled)
    artifact["selection_score"] = float(enable_decision.get("selection_score", 0.0))
    artifact["enable_gate"] = _jsonify(enable_decision)
    final_applied = apply_regime_adaptor(
        feature_frame.iloc[:n], scores, artifact, ts, sy
    )
    deployed_m = score_metrics(
        final_applied["deployment_score_rank"],
        rets,
        ts,
        top_fracs=(0.01, 0.05, 0.10, 0.20),
        cost_pct=cost_pct,
    )
    deployed_summary = _compare_metrics(raw_m, deployed_m)
    metrics = pd.concat(
        [
            raw_m.assign(stage="raw"),
            candidate_m.assign(stage="regime_adjusted_candidate"),
            deployed_m.assign(stage="regime_adjusted_deployed"),
        ],
        ignore_index=True,
    )
    metrics["strategy_id"] = strategy_id
    metrics["model"] = model_name
    metrics["regime_adaptor_enabled"] = bool(artifact["enable_regime_adaptor"])
    artifact["candidate_before_after"] = _jsonify(summary.to_dict(orient="records"))
    artifact["deployed_before_after"] = _jsonify(
        deployed_summary.to_dict(orient="records")
    )
    artifact["before_after_top10"] = artifact["candidate_before_after"]
    artifact["metrics"] = _jsonify(metrics.to_dict(orient="records"))
    return RegimeAdaptorFit(
        artifact=artifact,
        fixed_diagnostics=fixed,
        adaptive_diagnostics=pd.DataFrame(adaptive_rows),
        asset_diagnostics=asset_diag,
        metrics=metrics,
        regime_weight_oof=final_applied["regime_weight"],
        eligible_oof=final_applied["eligible"],
        deployment_score_oof=final_applied["deployment_score"],
        deployment_score_rank_oof=final_applied["deployment_score_rank"],
    )


def _compare_metrics(raw: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for frac in sorted(set(raw["top_frac"]).intersection(set(adj["top_frac"]))):
        r = raw[raw["top_frac"] == frac].iloc[0]
        a = adj[adj["top_frac"] == frac].iloc[0]
        rows.append(
            {
                "top_frac": float(frac),
                "lift_ratio": _safe_ratio(float(a["lift"]), float(r["lift"])),
                "net_ret_ratio": _safe_ratio(float(a["net_ret"]), float(r["net_ret"])),
                "gross_ret_ratio": _safe_ratio(
                    float(a["mean_gross_return"]), float(r["mean_gross_return"])
                ),
                "std_ratio": _safe_ratio(
                    float(a["return_std"]), float(r["return_std"])
                ),
                "dd_ratio": _safe_ratio(
                    float(a["max_drawdown"]), float(r["max_drawdown"])
                ),
            }
        )
    return pd.DataFrame(rows)


def _empty_artifact(
    strategy_id: str, model_name: str, features: Sequence[str], mapping: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "schema_version": "v1",
        "strategy_id": str(strategy_id),
        "model_name": str(model_name),
        "features": list(features),
        "feature_mapping": mapping,
        "percentile_refs": {},
        "feature_splines": {},
        "elastic_net": {
            "coef": [],
            "intercept": 0.0,
            "train_prediction_mean": 0.0,
            "params": {},
            "scaler": {"center": [], "scale": []},
        },
        "clips": {
            "total_log_weight_clip": [-0.35, 0.22],
            "regime_weight_clip": [0.70, 1.25],
        },
        "bucket_gates": [],
        "asset_gates": [],
        "rank_normalization": {
            "method": "pandas_rank_pct_average",
            "score": "deployment_score",
        },
        "enable_regime_adaptor": False,
    }


def fixed_bucket_diagnostics(
    regime_df: pd.DataFrame,
    scores: np.ndarray,
    returns: np.ndarray,
    timestamps: Optional[np.ndarray],
    symbols: np.ndarray,
    strategy_id: str,
    model_name: str,
    percentile_refs: Dict[str, np.ndarray],
) -> pd.DataFrame:
    strategy_tops = score_metrics(scores, returns, timestamps, top_fracs=(0.01, 0.05))
    rows: List[Dict[str, Any]] = []
    buckets = [(0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0)]
    for feat in regime_df.columns:
        pct = _apply_percentile(
            regime_df[feat].values,
            percentile_refs.get(feat, _fit_percentile(regime_df[feat].values)),
        )
        for lo, hi in buckets:
            mask = (pct >= lo) & (pct < hi if hi < 1.0 else pct <= hi)
            local_scores = np.where(mask, scores, np.nan)
            bucket_metrics = score_metrics(
                local_scores, returns, timestamps, top_fracs=(0.01, 0.05)
            )
            for _, bm_row in bucket_metrics.iterrows():
                bm = bm_row.to_dict()
                frac = float(bm.get("top_frac", 0.0))
                st_rows = strategy_tops[
                    np.isclose(strategy_tops["top_frac"].astype(float), frac)
                ]
                if st_rows.empty:
                    continue
                strategy_top = st_rows.iloc[0].to_dict()
                gross_ratio = _safe_ratio(
                    bm["mean_gross_return"], strategy_top["mean_gross_return"]
                )
                std_ratio = _safe_ratio(bm["return_std"], strategy_top["return_std"])
                dd_ratio = _safe_ratio(bm["max_drawdown"], strategy_top["max_drawdown"])
                rows.append(
                    {
                        "strategy_id": strategy_id,
                        "model": model_name,
                        "feature": feat,
                        "bucket_type": "fixed",
                        "lo": float(lo),
                        "hi": float(hi),
                        "n": int(np.sum(mask)),
                        "lift_ratio_vs_strategy": _safe_ratio(
                            bm["lift"], strategy_top["lift"]
                        ),
                        "gross_ret_ratio": gross_ratio,
                        "hit_rate_ratio": _safe_ratio(
                            bm["hit_rate"], strategy_top["hit_rate"]
                        ),
                        "return_std_ratio": std_ratio,
                        "drawdown_ratio": dd_ratio,
                        "regime_gated": bool(
                            gross_ratio < 0.7 and std_ratio > 1.3 and dd_ratio > 1.3
                        ),
                        **bm,
                    }
                )
    return pd.DataFrame(rows)


def asset_diagnostics(
    scores: np.ndarray,
    returns: np.ndarray,
    timestamps: Optional[np.ndarray],
    symbols: np.ndarray,
    strategy_id: str,
    model_name: str,
) -> pd.DataFrame:
    strategy_tops = score_metrics(scores, returns, timestamps, top_fracs=(0.01, 0.05))
    rows: List[Dict[str, Any]] = []
    for sym in sorted(set(str(s) for s in symbols)):
        mask = np.asarray([str(s) == sym for s in symbols], dtype=bool)
        if int(np.sum(mask)) < 10:
            continue
        local_scores = np.where(mask, scores, np.nan)
        asset_metrics = score_metrics(
            local_scores, returns, timestamps, top_fracs=(0.01, 0.05)
        )
        for _, bm_row in asset_metrics.iterrows():
            bm = bm_row.to_dict()
            frac = float(bm.get("top_frac", 0.0))
            st_rows = strategy_tops[
                np.isclose(strategy_tops["top_frac"].astype(float), frac)
            ]
            if st_rows.empty:
                continue
            strategy_top = st_rows.iloc[0].to_dict()
            gross_ratio = _safe_ratio(
                bm["mean_gross_return"], strategy_top["mean_gross_return"]
            )
            std_ratio = _safe_ratio(bm["return_std"], strategy_top["return_std"])
            dd_ratio = _safe_ratio(bm["max_drawdown"], strategy_top["max_drawdown"])
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "model": model_name,
                    "symbol": sym,
                    "n": int(np.sum(mask)),
                    "gross_ret_ratio": gross_ratio,
                    "return_std_ratio": std_ratio,
                    "drawdown_ratio": dd_ratio,
                    "asset_gated": bool(
                        gross_ratio < 0.6 and std_ratio > 1.4 and dd_ratio > 1.4
                    ),
                    **bm,
                }
            )
    return pd.DataFrame(rows)


def _bucket_gates(fixed: pd.DataFrame) -> List[Dict[str, Any]]:
    if fixed.empty or "regime_gated" not in fixed.columns:
        return []
    rows = fixed[fixed["regime_gated"]]
    return [
        {"feature": str(r["feature"]), "lo": float(r["lo"]), "hi": float(r["hi"])}
        for _, r in rows.iterrows()
    ]


def _asset_gates(asset_diag: pd.DataFrame) -> List[str]:
    if asset_diag.empty or "asset_gated" not in asset_diag.columns:
        return []
    return sorted(
        str(s) for s in asset_diag.loc[asset_diag["asset_gated"], "symbol"].tolist()
    )


def save_regime_adaptor_outputs(
    data_root: str,
    run_id: str,
    strategy_id: str,
    fit: RegimeAdaptorFit,
) -> Path:
    out_dir = (
        Path(data_root)
        / "artifacts"
        / run_id
        / "ridge_sizer"
        / "regime_adaptors"
        / safe_strategy_slug(strategy_id)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "regime_adaptor.json"
    artifact_path.write_text(
        json.dumps(_jsonify(fit.artifact), indent=2, sort_keys=True)
    )
    for name, frame in (
        ("regime_diagnostics_fixed", fit.fixed_diagnostics),
        ("regime_diagnostics_adaptive", fit.adaptive_diagnostics),
        ("regime_asset_diagnostics", fit.asset_diagnostics),
        ("regime_before_after_metrics", fit.metrics),
    ):
        if frame is None or frame.empty:
            continue
        frame.to_parquet(out_dir / f"{name}.parquet", index=False)
        (out_dir / f"{name}.json").write_text(frame.to_json(orient="records", indent=2))
    return artifact_path


def load_regime_adaptor(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonify(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
