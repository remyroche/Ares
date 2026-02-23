"""Prediction metrics + proxy TP/SL analysis for meta-training outputs.

Usage example:
    python -m extreme_price_movements.offline_optimisers.preds_metrics_computations \
      --input data/artifacts/<run_id>/meta_oof/meta_oof_long_tf_H4.parquet \
      --outdir extreme_price_movements/offline_optimisers/reports/preds_metrics \
      --ret-all

Expected minimum columns (or accepted aliases):
- ts (aliases: timestamp, __ts__)
- score (aliases: oof_pred, pred, oof_probs, base_score)
- fwd_ret_H<N> (auto-detected; legacy alias: return -> fwd_ret_H4)
Optional:
- asset (aliases: symbol, __symbol__, asset_id)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

FEE_RT: np.float32 = np.float32(0.005)  # legacy default (0.5% round-trip)
FEE_LEVELS: Tuple[float, ...] = (0.002, 0.005)  # 0.2% and 0.5% round-trip
DTYPE_F32 = np.float32
DTYPE_I32 = np.int32

# Top regime features to bucket on (ordered by economic relevance)
TOP_REGIME_FEATURES: List[str] = [
    "__meta_raw__vol_z",
    "__meta_raw__trend_pct",
    "__meta_raw__vol_z_30_calm",
    "__meta_raw__trend_t",
    "__meta_raw__trend_z_t",
    "__meta_raw__rv_ratio_6_24",
    "__meta_raw__atr_pct",
    "__meta_raw__spike_score",
    "__meta_raw__grind_score",
    "__meta_raw__chop_score",
    "__meta_raw__accept",
    "__meta_raw__ambig",
    "__meta_raw__stage_tf",
    "__meta_raw__stage_mr",
    "__meta_raw__cusum_strength",
    # discrete regime gates (fallback if continuous not present)
    "__regime_vol_12h__",
    "__regime_trend_12h__",
    "__regime_vol_48h__",
    "__regime_trend_48h__",
    "G_VOL",
    "G_TREND",
]

# Optimise-step baseline params (mirrors Policy.baseline_params())
_OPTIMISE_BASELINE: Dict = {"tp_mult": 3.0, "sl_mult": 1.0}


# =============================================================================
# 0) Utilities: casting + safe ops
# =============================================================================

def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].astype(DTYPE_F32)
        elif pd.api.types.is_integer_dtype(out[c]):
            out[c] = out[c].astype(DTYPE_I32)
    if "asset" in out.columns and out["asset"].dtype != "category":
        out["asset"] = out["asset"].astype("category")
    return out


RET_PATTERNS = [
    re.compile(r"fwd_ret[_]?h(?P<h>\d+)", re.IGNORECASE),
    re.compile(r"fwd_ret_(?P<h>\d+)h", re.IGNORECASE),
    re.compile(r"return[_]?h(?P<h>\d+)", re.IGNORECASE),
    re.compile(r"ret[_]?h(?P<h>\d+)", re.IGNORECASE),
    re.compile(r"y_ret[_]?h(?P<h>\d+)", re.IGNORECASE),
]


def _infer_horizon_from_name(col: str) -> Optional[int]:
    lower = col.lower()
    for pattern in RET_PATTERNS:
        m = pattern.search(lower)
        if m:
            try:
                return int(m.group("h"))
            except (KeyError, ValueError):
                continue
    return None


def _detect_forward_columns(df: pd.DataFrame) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for col in df.columns:
        horizon = _infer_horizon_from_name(col)
        if horizon is not None and col not in mapping.values():
            mapping.setdefault(horizon, col)
    return mapping


def _prepare_target_columns(
    df: pd.DataFrame,
    preferred_col: Optional[str] = None,
    preferred_horizon: Optional[int] = None,
    all_horizons: bool = False,
) -> Tuple[Tuple[Optional[int], str], ...]:
    if preferred_col:
        if preferred_col not in df.columns:
            raise ValueError(f"Requested ret column '{preferred_col}' not found in input")
        inferred = _infer_horizon_from_name(preferred_col)
        return ((inferred, preferred_col),)

    mapping = _detect_forward_columns(df)
    if not mapping:
        raise ValueError(
            "Could not detect any forward return columns. "
            "Ensure your input (or returns table) includes columns like 'fwd_ret_H4'."
        )

    if all_horizons:
        return tuple(sorted(mapping.items(), key=lambda kv: (kv[0] if kv[0] is not None else 10**9)))

    if preferred_horizon is not None and preferred_horizon in mapping:
        return ((preferred_horizon, mapping[preferred_horizon]),)

    if 4 in mapping:
        return ((4, mapping[4]),)

    first_h = sorted(mapping.keys())[0]
    return ((first_h, mapping[first_h]),)


def _load_returns_table(path: Path, ts_col: str = "ts", asset_col: str = "asset") -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    rename_map = {}
    if ts_col not in df.columns:
        raise ValueError(f"Returns table missing ts column '{ts_col}'")
    if asset_col not in df.columns:
        raise ValueError(f"Returns table missing asset column '{asset_col}'")
    if ts_col != "ts":
        rename_map[ts_col] = "ts"
    if asset_col != "asset":
        rename_map[asset_col] = "asset"
    df = df.rename(columns=rename_map)

    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["asset"] = df["asset"].astype(str)
    return df


def _merge_returns_columns(base_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    if returns_df is None:
        return base_df
    if not {"ts", "asset"}.issubset(base_df.columns):
        raise ValueError("Input data must include 'ts' and 'asset' columns before merging returns")

    ret_cols = [c for c in returns_df.columns if c not in {"ts", "asset", "score"}]
    if not ret_cols:
        return base_df

    payload = returns_df[["ts", "asset", *ret_cols]]
    merged = base_df.merge(payload, on=["ts", "asset"], how="left", suffixes=("", "_ret"))

    for col in ret_cols:
        if col in {"ts", "asset"}:
            continue
        ret_col = f"{col}_ret"
        if ret_col in merged.columns and col in merged.columns:
            merged[col] = merged[col].where(~merged[col].isna(), merged[ret_col])
            merged = merged.drop(columns=[ret_col])
        elif ret_col in merged.columns:
            merged.rename(columns={ret_col: col}, inplace=True)

    return merged


def _assert_inputs(df: pd.DataFrame, ret_col: str) -> None:
    required = {"ts", "score", ret_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["score"].isna().all():
        raise ValueError("All NaN in score column")
    if df[ret_col].isna().all():
        raise ValueError(f"All NaN in forward return column '{ret_col}'")
    # Check if ts is datetime-like (handles datetime64 with or without timezone)
    if not (pd.api.types.is_datetime64_any_dtype(df["ts"]) or np.issubdtype(df["ts"].dtype, np.integer)):
        raise ValueError("ts must be datetime64 or integer")


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    if a.size < 3:
        return np.nan
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# =============================================================================
# 1) Core prediction diagnostics
# =============================================================================

def compute_ic(df: pd.DataFrame, by: Optional[str] = "ts", ret_col: str = "fwd_ret") -> Dict[str, float]:
    score = df["score"].to_numpy()
    ret = df[ret_col].to_numpy()
    out: Dict[str, float] = {"ic_global": _safe_corr(score, ret)}

    # --- Event count stats ---
    n_total = int(len(df))
    out["n_total_events"] = n_total
    if by is not None and by in df.columns and pd.api.types.is_datetime64_any_dtype(df[by]):
        date_span_days = float((df[by].max() - df[by].min()).total_seconds() / 86400.0)
        out["date_span_days"] = round(date_span_days, 1)
        out["events_per_day"] = round(n_total / max(date_span_days, 1.0), 2)
    else:
        out["date_span_days"] = np.nan
        out["events_per_day"] = np.nan

    if by is None or by not in df.columns:
        out.update({"ic_mean": np.nan, "ic_std": np.nan, "ic_ir": np.nan, "mean_group_size": np.nan, "ic_mode": "global_only"})
        return out

    # Auto-detect whether cross-sectional IC is meaningful
    g = df.groupby(by, sort=False)
    sizes = g.size().to_numpy()
    mean_size = float(np.mean(sizes)) if sizes.size else 0.0
    out["mean_group_size"] = mean_size
    if mean_size < 3.0:
        out["ic_mean"] = np.nan
        out["ic_std"] = np.nan
        out["ic_ir"] = np.nan
        out["ic_mode"] = "global_only"
        return out

    ic_series = g.apply(lambda x: _safe_corr(x["score"].to_numpy(), x[ret_col].to_numpy()))
    ic_mean = float(np.nanmean(ic_series.values))
    ic_std = float(np.nanstd(ic_series.values))
    out["ic_mean"] = ic_mean
    out["ic_std"] = ic_std
    out["ic_ir"] = float(ic_mean / ic_std) if ic_std > 0 else np.nan
    out["ic_mode"] = "per_ts_cross_sectional"
    return out


def compute_deciles(df: pd.DataFrame, q: int = 10, by: str = "ts", ret_col: str = "fwd_ret") -> pd.DataFrame:
    # Use float64 for ranking stability; cast back later
    score64 = df["score"].astype(np.float64)
    pct = score64.groupby(df[by], sort=False).rank(pct=True, method="first")
    qbin = np.minimum((pct.to_numpy() * q).astype(np.int32), q - 1)
    tmp = pd.DataFrame({"qbin": qbin, "ret": df[ret_col].to_numpy()})
    out = tmp.groupby("qbin", sort=True)["ret"].agg(["count", "mean", "median"]).reset_index()
    out.rename(columns={"mean": "mean_ret", "median": "median_ret"}, inplace=True)
    return out


# =============================================================================
# 2) Trade inference via score selection + proxy TP/SL
# =============================================================================

@dataclass(frozen=True)
class SelectionSpec:
    top_frac: float


@dataclass(frozen=True)
class TPSLSpec:
    tp: float
    sl_ratio: float


def infer_positions_top_frac(df: pd.DataFrame, top_frac: float, by: str = "ts") -> np.ndarray:
    pct = df.groupby(by, sort=False)["score"].rank(pct=True, method="first").to_numpy()
    sel = pct > (1.0 - top_frac)
    return sel.astype(DTYPE_F32)


def apply_tpsl_proxy(
    fwd_ret: np.ndarray,
    pos_w: np.ndarray,
    tp: float,
    sl: float,
    fee_rt: float = float(FEE_RT),
    mode: str = "optimistic",
) -> np.ndarray:
    """Apply TP/SL proxy and return per-row net returns.

    Modes:
      optimistic  – barrier trades exit at exact barrier level; non-barrier at
                    actual horizon close return.
      pessimistic – same barrier logic but non-barrier trades capped at 25% of TP.
      mid         – realistic: barrier trades exit at barrier; non-barrier trades
                    exit at actual close (no cap/floor). Best expected-value estimate.
      optimise    – alias for mid; intended for use with optimise-step resolved params.
    """
    fwd_ret = fwd_ret.astype(np.float32, copy=False)
    pos_w = pos_w.astype(np.float32, copy=False)
    tp = np.float32(tp)
    sl = np.float32(sl)
    fee_rt = np.float32(fee_rt)

    hit_tp = fwd_ret >= tp
    hit_sl = fwd_ret <= -sl

    if mode == "optimistic":
        realised = np.where(hit_tp, tp, np.where(hit_sl, -sl, fwd_ret)).astype(np.float32)
    elif mode == "pessimistic":
        _mid = np.where(fwd_ret < 0, -sl, np.minimum(fwd_ret, tp * np.float32(0.25))).astype(np.float32)
        realised = np.where(hit_tp, tp, np.where(hit_sl, -sl, _mid)).astype(np.float32)
    elif mode in ("mid", "optimise"):
        # Realistic: barrier trades exit at barrier; non-barrier exit at actual close.
        realised = np.where(hit_tp, tp, np.where(hit_sl, -sl, fwd_ret)).astype(np.float32)
    else:
        raise ValueError("mode must be 'optimistic', 'pessimistic', 'mid', or 'optimise'")

    net = (pos_w * realised) - (pos_w * fee_rt)
    return net.astype(np.float32)


def _load_optimise_params(params_path: Optional[str]) -> Dict:
    """Load best TP/SL params from the optimise-step JSON. Falls back to baseline."""
    defaults = dict(_OPTIMISE_BASELINE)
    if not params_path:
        return defaults
    p = Path(params_path)
    if not p.exists():
        warnings.warn(f"[preds_metrics] optimise params not found: {p}; using baseline")
        return defaults
    try:
        payload = json.loads(p.read_text())
        # bucket_params.json: {"buckets": {"LONG_TF": {"tp_sl": {...}, ...}}}
        # or flat: {"tp_mult": ..., "sl_mult": ...}
        if "buckets" in payload:
            tps, sls = [], []
            for bkt_data in payload["buckets"].values():
                tp_sl = bkt_data.get("tp_sl", bkt_data)
                tps.append(float(tp_sl.get("tp_mult", defaults["tp_mult"])))
                sls.append(float(tp_sl.get("sl_mult", defaults["sl_mult"])))
            return {
                "tp_mult": float(np.median(tps)) if tps else defaults["tp_mult"],
                "sl_mult": float(np.median(sls)) if sls else defaults["sl_mult"],
            }
        tp_sl = payload.get("tp_sl", payload)
        return {
            "tp_mult": float(tp_sl.get("tp_mult", defaults["tp_mult"])),
            "sl_mult": float(tp_sl.get("sl_mult", defaults["sl_mult"])),
        }
    except Exception as exc:
        warnings.warn(f"[preds_metrics] Could not parse optimise params ({exc}); using baseline")
        return defaults




def lead_lag_sanity(
    df: pd.DataFrame,
    by_ts: str = "ts",
    lags: Tuple[int, ...] = (-2, -1, 0, 1, 2),
    ret_col: str = "fwd_ret",
) -> pd.DataFrame:
    """Quick lookahead/alignment check in 4h increments."""
    d = df.sort_values(by_ts)
    score = d["score"].to_numpy()
    ret = d[ret_col].to_numpy()
    rows = []
    for k in lags:
        if k < 0:
            a = score[-k:]
            b = ret[:k]
        elif k > 0:
            a = score[:-k]
            b = ret[k:]
        else:
            a = score
            b = ret
        rows.append({"lag_steps": k, "corr": _safe_corr(a, b)})
    return pd.DataFrame(rows)


def proxy_hit_rates(fwd_ret: np.ndarray, pos_w: np.ndarray, tp: float, sl: float) -> Dict[str, float]:
    """Report how often TP/SL binds among active trades (proxy world)."""
    active = pos_w > 0
    n = int(np.sum(active))
    if n == 0:
        return {"tp_hit_rate": np.nan, "sl_hit_rate": np.nan, "hold_rate": np.nan}
    r = fwd_ret[active]
    return {
        "tp_hit_rate": float(np.mean(r >= tp)),
        "sl_hit_rate": float(np.mean(r <= -sl)),
        "hold_rate": float(np.mean((r < tp) & (r > -sl))),
    }

def compute_strategy_kpis(net_ret: np.ndarray, fee_rt: float = float(FEE_RT)) -> Dict[str, float]:
    """Core KPIs for a vector of per-trade net returns at a given fee level."""
    x = net_ret.astype(np.float64, copy=False)
    if x.size == 0:
        return {k: np.nan for k in ["mean", "median", "win_rate", "p10", "p90", "sharpe", "sortino", "expectancy_bps"]}
    neg = x[x < 0]
    sortino = float(np.mean(x) / (np.std(neg) + 1e-12)) if neg.size else np.nan
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "win_rate": float(np.mean(x > 0)),
        "p10": float(np.quantile(x, 0.10)),
        "p90": float(np.quantile(x, 0.90)),
        "sharpe": float(np.mean(x) / (np.std(x) + 1e-12)),
        "sortino": sortino,
        "expectancy_bps": float(np.mean(x) * 10_000),
        "fee_rt_used": fee_rt,
    }


def compute_pnl_with_fees(
    fwd_ret: np.ndarray,
    pos_w: np.ndarray,
    tp: float,
    sl: float,
    mode: str = "mid",
) -> Dict[str, Dict[str, float]]:
    """Compute expected PnL at both standard fee levels (0.2% and 0.5% round-trip).

    For non-barrier trades (neither TP nor SL hit), uses actual horizon close return.
    Returns a dict keyed by fee label, each containing compute_strategy_kpis output.
    """
    results: Dict[str, Dict[str, float]] = {}
    for fee in FEE_LEVELS:
        net = apply_tpsl_proxy(fwd_ret, pos_w, tp=tp, sl=sl, fee_rt=fee, mode=mode)
        kpis = compute_strategy_kpis(net, fee_rt=fee)
        label = f"fee_{int(fee * 1000)}bps"
        results[label] = kpis
    return results


def run_proxy_grid(
    df: pd.DataFrame,
    selections: Tuple[SelectionSpec, ...] = (SelectionSpec(0.2), SelectionSpec(0.3), SelectionSpec(0.4)),
    tpsl: Tuple[TPSLSpec, ...] = (TPSLSpec(0.02, 0.5), TPSLSpec(0.03, 0.5), TPSLSpec(0.04, 0.5)),
    by: str = "ts",
    mode: str = "optimistic",
    ret_col: str = "fwd_ret",
    fee_levels: Tuple[float, ...] = FEE_LEVELS,
) -> pd.DataFrame:
    """Run proxy TP/SL grid. Reports KPIs at every fee level in fee_levels."""
    fwd = df[ret_col].to_numpy(dtype=np.float32)
    rows = []
    primary_fee_label = f"fee_{int(fee_levels[0] * 1000)}bps"
    for sel in selections:
        pos_w = infer_positions_top_frac(df, sel.top_frac, by=by)
        for spec in tpsl:
            sl = spec.tp * spec.sl_ratio
            hr = proxy_hit_rates(fwd, pos_w, tp=np.float32(spec.tp), sl=np.float32(sl))
            row: Dict = {
                "mode": mode,
                "top_frac": sel.top_frac,
                "tp": spec.tp,
                "sl": sl,
                "n_active": int(np.sum(pos_w > 0)),
                "n_total": int(len(pos_w)),
                "active_rate": float(np.mean(pos_w > 0)),
                **hr,
            }
            for fee in fee_levels:
                net = apply_tpsl_proxy(fwd, pos_w, tp=spec.tp, sl=sl, fee_rt=fee, mode=mode)
                k = compute_strategy_kpis(net, fee_rt=fee)
                fee_label = f"fee_{int(fee * 1000)}bps"
                for kname, kval in k.items():
                    if kname != "fee_rt_used":
                        row[f"{fee_label}__{kname}"] = kval
            rows.append(row)
    sort_col = f"{primary_fee_label}__mean"
    df_out = pd.DataFrame(rows)
    if sort_col in df_out.columns:
        df_out = df_out.sort_values(sort_col, ascending=False)
    return df_out.reset_index(drop=True)


# =============================================================================
# 3) Regime bucket analysis
# =============================================================================

def _pick_regime_features(df: pd.DataFrame, max_features: int = 15) -> List[str]:
    """Return available regime features from TOP_REGIME_FEATURES, up to max_features."""
    return [f for f in TOP_REGIME_FEATURES if f in df.columns][:max_features]


def _bucket_continuous(series: pd.Series, n_buckets: int = 3) -> pd.Series:
    """Tertile-bucket a continuous series; returns integer labels 0/1/2."""
    try:
        return pd.qcut(series, q=n_buckets, labels=False, duplicates="drop").astype("Int64")
    except Exception:
        return pd.Series(np.nan, index=series.index, dtype="Int64")


def compute_regime_bucket_analysis(
    df: pd.DataFrame,
    ret_col: str,
    regime_features: Optional[List[str]] = None,
    n_buckets: int = 3,
) -> pd.DataFrame:
    """For each regime feature, bucket events and compute IC + PnL stats per bucket.

    Returns a long-form DataFrame with columns:
      feature, bucket, n, ic, mean_ret_bps, win_rate, sortino, anova_p
    """
    if regime_features is None:
        regime_features = _pick_regime_features(df)
    if not regime_features:
        return pd.DataFrame()

    score = df["score"].to_numpy(dtype=np.float64)
    ret = df[ret_col].to_numpy(dtype=np.float64)
    rows = []

    for feat in regime_features:
        col = df[feat]
        # Discrete (integer/bool/low-cardinality) -> use values directly
        if col.dtype.kind in ("i", "u", "b") or col.nunique() <= 5:
            bucket_col = col.astype("Int64")
        else:
            bucket_col = _bucket_continuous(col.astype(np.float64), n_buckets=n_buckets)

        bucket_rets: Dict[int, np.ndarray] = {}
        for bv in sorted(bucket_col.dropna().unique()):
            mask = (bucket_col == bv).to_numpy()
            if mask.sum() < 5:
                continue
            s_b = score[mask]
            r_b = ret[mask]
            ic_b = _safe_corr(s_b, r_b)
            neg = r_b[r_b < 0]
            sortino_b = float(np.mean(r_b) / (np.std(neg) + 1e-12)) if neg.size else np.nan
            rows.append({
                "feature": feat,
                "bucket": int(bv),
                "n": int(mask.sum()),
                "ic": round(ic_b, 4) if np.isfinite(ic_b) else np.nan,
                "mean_ret_bps": float(np.mean(r_b) * 10_000),
                "win_rate": float(np.mean(r_b > 0)),
                "sortino": sortino_b,
                "_rets": r_b,  # temp, removed below
            })
            bucket_rets[int(bv)] = r_b

        # ANOVA across buckets
        groups = list(bucket_rets.values())
        if len(groups) >= 2 and all(g.size >= 3 for g in groups):
            try:
                _, p_val = scipy_stats.f_oneway(*groups)
                anova_p = float(p_val)
            except Exception:
                anova_p = np.nan
        else:
            anova_p = np.nan

        for row in rows:
            if row["feature"] == feat and "anova_p" not in row:
                row["anova_p"] = anova_p

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.drop(columns=["_rets"], errors="ignore")
        result = result.sort_values(["feature", "bucket"]).reset_index(drop=True)
    return result


# =============================================================================
# 4) Step 1 — Extreme-move gate + high-low range payoff metrics
# =============================================================================

def compute_extreme_gate_stats(
    df: pd.DataFrame,
    ret_col: str,
    move_thresh: float = 0.07,
    lookback_hours: int = 7,
    hl_windows: Tuple[int, ...] = (12, 16, 24),
) -> Dict[str, object]:
    """Gate analysis: only consider trades where |price move| > move_thresh in lookback_hours.

    Requires columns:
      - '__meta_raw__atr_pct'  (proxy for recent move magnitude, causal)
      - OR 'extreme_gate' (pre-computed boolean)
      - hl_range_<W>h  for each W in hl_windows (high-low range over W hours prior)

    Returns a dict with:
      gate_coverage   : fraction of rows passing the gate
      gated_ic        : IC on gated subset
      ungated_ic      : IC on complement
      gated_kpis      : compute_strategy_kpis on gated subset (no TP/SL, raw ret)
      ungated_kpis    : same for complement
      hl_payoff       : dict keyed by window -> mean/median/p90 of hl_range column
    """
    out: Dict[str, object] = {}

    # --- Build gate mask ---
    # Prefer explicit extreme_gate column; fall back to atr_pct threshold
    if "extreme_gate" in df.columns:
        gate_mask = df["extreme_gate"].astype(bool).to_numpy()
        out["gate_source"] = "extreme_gate_col"
    elif "__meta_raw__atr_pct" in df.columns:
        # atr_pct is a causal rolling ATR as % of price — use as move-magnitude proxy
        gate_mask = (df["__meta_raw__atr_pct"].to_numpy(dtype=np.float64) >= move_thresh)
        out["gate_source"] = f"atr_pct>={move_thresh}"
    else:
        # No gate data available — report all as gated
        gate_mask = np.ones(len(df), dtype=bool)
        out["gate_source"] = "no_gate_data_all_pass"

    n_total = len(df)
    n_gated = int(gate_mask.sum())
    out["gate_coverage"] = round(n_gated / max(n_total, 1), 4)
    out["n_gated"] = n_gated
    out["n_total"] = n_total
    out["move_thresh"] = move_thresh
    out["lookback_hours"] = lookback_hours

    score = df["score"].to_numpy(dtype=np.float64)
    ret   = df[ret_col].to_numpy(dtype=np.float64)

    # IC on gated vs ungated
    out["gated_ic"]   = _safe_corr(score[gate_mask],  ret[gate_mask])  if gate_mask.sum() >= 3 else np.nan
    out["ungated_ic"] = _safe_corr(score[~gate_mask], ret[~gate_mask]) if (~gate_mask).sum() >= 3 else np.nan

    # KPIs on raw returns (no TP/SL) for gated/ungated
    out["gated_kpis"]   = compute_strategy_kpis(ret[gate_mask].astype(np.float64))  if gate_mask.sum() >= 3 else {}
    out["ungated_kpis"] = compute_strategy_kpis(ret[~gate_mask].astype(np.float64)) if (~gate_mask).sum() >= 3 else {}

    # --- High-low range payoff metrics ---
    hl_payoff: Dict[str, Dict[str, float]] = {}
    for w in hl_windows:
        col = f"hl_range_{w}h"
        if col in df.columns:
            vals = df.loc[gate_mask, col].to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size >= 3:
                hl_payoff[f"{w}h"] = {
                    "mean_bps":   float(np.mean(vals) * 10_000),
                    "median_bps": float(np.median(vals) * 10_000),
                    "p90_bps":    float(np.quantile(vals, 0.90) * 10_000),
                    "n":          int(vals.size),
                }
    out["hl_payoff"] = hl_payoff

    return out


# =============================================================================
# 5) Step 2 — ATR-relative TP/SL proxy grid
# =============================================================================

def run_atr_proxy_grid(
    df: pd.DataFrame,
    ret_col: str,
    atr_col: str = "__meta_raw__atr_pct",
    tp_atr_mults: Tuple[float, ...] = (2.0, 3.0, 4.0, 6.0),
    sl_atr_mults: Tuple[float, ...] = (1.0, 1.5, 2.0),
    selections: Tuple[SelectionSpec, ...] = (SelectionSpec(0.1), SelectionSpec(0.2), SelectionSpec(0.3)),
    by: str = "ts",
    fee_levels: Tuple[float, ...] = FEE_LEVELS,
) -> pd.DataFrame:
    """Proxy TP/SL grid where barriers are ATR-multiples (causal, per-row ATR).

    For each row, TP = tp_mult * atr_pct, SL = sl_mult * atr_pct.
    This avoids the fixed-% problem where barriers are unreachable for low-vol events.

    If atr_col is not present, falls back to fixed 1% ATR proxy.
    """
    fwd = df[ret_col].to_numpy(dtype=np.float32)

    if atr_col in df.columns:
        atr = df[atr_col].to_numpy(dtype=np.float32)
        atr = np.where(np.isfinite(atr) & (atr > 0), atr, np.float32(0.01))
        atr_source = atr_col
    else:
        atr = np.full(len(fwd), np.float32(0.01))
        atr_source = "fixed_1pct_fallback"

    rows = []
    primary_fee_label = f"fee_{int(fee_levels[0] * 1000)}bps"

    for sel in selections:
        pos_w = infer_positions_top_frac(df, sel.top_frac, by=by)
        active = pos_w > 0

        for tp_m in tp_atr_mults:
            for sl_m in sl_atr_mults:
                # Per-row barriers
                tp_arr = (np.float32(tp_m) * atr).astype(np.float32)
                sl_arr = (np.float32(sl_m) * atr).astype(np.float32)

                # Hit rates (per-row comparison)
                n_active = int(active.sum())
                if n_active == 0:
                    continue
                r_active = fwd[active]
                tp_active = tp_arr[active]
                sl_active = sl_arr[active]
                tp_hit = float(np.mean(r_active >= tp_active))
                sl_hit = float(np.mean(r_active <= -sl_active))
                hold   = float(np.mean((r_active < tp_active) & (r_active > -sl_active)))

                row: Dict = {
                    "atr_source": atr_source,
                    "top_frac": sel.top_frac,
                    "tp_atr_mult": tp_m,
                    "sl_atr_mult": sl_m,
                    "tp_pct_median": float(np.median(tp_active)),
                    "sl_pct_median": float(np.median(sl_active)),
                    "n_active": n_active,
                    "tp_hit_rate": tp_hit,
                    "sl_hit_rate": sl_hit,
                    "hold_rate": hold,
                }

                # Per-row net return: realised = clip(fwd, -sl, tp) per row
                realised = np.where(
                    fwd >= tp_arr, tp_arr,
                    np.where(fwd <= -sl_arr, -sl_arr, fwd)
                ).astype(np.float32)

                for fee in fee_levels:
                    net = (pos_w * realised) - (pos_w * np.float32(fee))
                    k = compute_strategy_kpis(net[active].astype(np.float64), fee_rt=fee)
                    fee_label = f"fee_{int(fee * 1000)}bps"
                    for kname, kval in k.items():
                        if kname != "fee_rt_used":
                            row[f"{fee_label}__{kname}"] = kval

                rows.append(row)

    df_out = pd.DataFrame(rows)
    sort_col = f"{primary_fee_label}__mean"
    if sort_col in df_out.columns:
        df_out = df_out.sort_values(sort_col, ascending=False)
    return df_out.reset_index(drop=True)


# =============================================================================
# 6) Step 3 — Stratified isotonic calibration
# =============================================================================

def _isotonic_calibrate(score: np.ndarray, ret: np.ndarray) -> np.ndarray:
    """Fit isotonic regression of ret on score; return calibrated scores (same length)."""
    from sklearn.isotonic import IsotonicRegression
    order = np.argsort(score)
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(score[order], ret[order])
    return ir.predict(score)


def compute_stratified_calibration(
    df: pd.DataFrame,
    ret_col: str,
    chop_col: str = "__meta_raw__chop_score",
    vol_col: str = "__meta_raw__vol_z",
    chop_high_quantile: float = 0.80,
    vol_high_quantile: float = 0.67,
) -> Dict[str, object]:
    """Stratified isotonic calibration across regime bins.

    Bins:
      chop_high  : chop_score >= chop_high_quantile
      chop_low   : chop_score <  chop_high_quantile
      vol_high   : vol_z >= vol_high_quantile
      vol_low    : vol_z <  vol_high_quantile
      chop_high × vol_high  (2×2 cross)

    For each bin, fits isotonic regression of ret on score and reports:
      - IC before vs after calibration
      - Spearman rank corr before vs after
      - n, fraction of total
    """
    score = df["score"].to_numpy(dtype=np.float64)
    ret   = df[ret_col].to_numpy(dtype=np.float64)
    out: Dict[str, object] = {}

    def _bin_stats(mask: np.ndarray, label: str) -> None:
        n = int(mask.sum())
        if n < 10:
            out[label] = {"n": n, "skipped": "too_few_samples"}
            return
        s_b = score[mask]
        r_b = ret[mask]
        ic_before = _safe_corr(s_b, r_b)
        try:
            s_cal = _isotonic_calibrate(s_b, r_b)
            ic_after = _safe_corr(s_cal, r_b)
            spear_before = float(scipy_stats.spearmanr(s_b, r_b).statistic)
            spear_after  = float(scipy_stats.spearmanr(s_cal, r_b).statistic)
        except Exception:
            ic_after = np.nan
            spear_before = spear_after = np.nan
        out[label] = {
            "n": n,
            "frac": round(n / max(len(df), 1), 4),
            "ic_before": round(ic_before, 4) if np.isfinite(ic_before) else np.nan,
            "ic_after":  round(ic_after, 4)  if np.isfinite(ic_after)  else np.nan,
            "ic_lift":   round(ic_after - ic_before, 4) if (np.isfinite(ic_before) and np.isfinite(ic_after)) else np.nan,
            "spear_before": round(spear_before, 4) if np.isfinite(spear_before) else np.nan,
            "spear_after":  round(spear_after, 4)  if np.isfinite(spear_after)  else np.nan,
        }

    # Global baseline
    _bin_stats(np.ones(len(df), dtype=bool), "global")

    # Chop bins
    if chop_col in df.columns:
        chop = df[chop_col].to_numpy(dtype=np.float64)
        chop_thresh = float(np.nanquantile(chop, chop_high_quantile))
        chop_high = np.isfinite(chop) & (chop >= chop_thresh)
        chop_low  = np.isfinite(chop) & (chop <  chop_thresh)
        _bin_stats(chop_high, f"chop_high(q{int(chop_high_quantile*100)})")
        _bin_stats(chop_low,  f"chop_low")
        out["chop_thresh"] = chop_thresh
    else:
        chop_high = np.zeros(len(df), dtype=bool)
        chop_low  = np.ones(len(df), dtype=bool)

    # Vol bins
    if vol_col in df.columns:
        vol = df[vol_col].to_numpy(dtype=np.float64)
        vol_thresh = float(np.nanquantile(vol, vol_high_quantile))
        vol_high = np.isfinite(vol) & (vol >= vol_thresh)
        vol_low  = np.isfinite(vol) & (vol <  vol_thresh)
        _bin_stats(vol_high, f"vol_high(q{int(vol_high_quantile*100)})")
        _bin_stats(vol_low,  f"vol_low")
        out["vol_thresh"] = vol_thresh
    else:
        vol_high = np.zeros(len(df), dtype=bool)

    # 2×2 cross: chop_high × vol_high
    if chop_col in df.columns and vol_col in df.columns:
        _bin_stats(chop_high & vol_high,  "chop_high_x_vol_high")
        _bin_stats(chop_high & ~vol_high, "chop_high_x_vol_low")
        _bin_stats(~chop_high & vol_high, "chop_low_x_vol_high")
        _bin_stats(~chop_high & ~vol_high,"chop_low_x_vol_low")

    return out


# =============================================================================
# 7) Step 4 — Top-k decile metrics (10% and 30%)
# =============================================================================

def compute_topk_metrics(
    df: pd.DataFrame,
    ret_col: str,
    top_fracs: Tuple[float, ...] = (0.10, 0.30),
    by: str = "ts",
    fee_levels: Tuple[float, ...] = FEE_LEVELS,
) -> pd.DataFrame:
    """KPIs for top-k% selection (no TP/SL, raw horizon return minus fee).

    Returns a DataFrame with one row per (top_frac, fee_level).
    """
    fwd = df[ret_col].to_numpy(dtype=np.float64)
    rows = []
    for top_frac in top_fracs:
        pos_w = infer_positions_top_frac(df, top_frac, by=by).astype(np.float64)
        active = pos_w > 0
        r_active = fwd[active]
        for fee in fee_levels:
            net = r_active - fee
            k = compute_strategy_kpis(net, fee_rt=fee)
            rows.append({
                "top_frac": top_frac,
                "fee_rt": fee,
                "n_active": int(active.sum()),
                "n_total": len(fwd),
                **{kk: vv for kk, vv in k.items() if kk != "fee_rt_used"},
            })
    return pd.DataFrame(rows)


# =============================================================================
# 8) Meta-output loader + top-level analysis
# =============================================================================


def _normalize_meta_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    aliases = {
        "ts": ["ts", "timestamp", "__ts__"],
        "score": ["score", "oof_pred", "pred", "oof_probs", "base_score"],
        "asset": ["asset", "symbol", "__symbol__", "asset_id"],
    }
    cols = set(df.columns)
    for target, alts in aliases.items():
        for a in alts:
            if a in cols:
                col_map[a] = target
                break
    out = df.rename(columns=col_map)

    if "asset" not in out.columns:
        out["asset"] = "ALL"

    if "ts" in out.columns and not pd.api.types.is_datetime64_any_dtype(out["ts"]):
        try:
            out["ts"] = pd.to_datetime(out["ts"], utc=True)
        except Exception:
            pass

    # Legacy: bare 'return' column with no horizon tag -> treat as H4 proxy
    if "return" in out.columns and _infer_horizon_from_name("return") is None:
        if "fwd_ret_H4" not in out.columns:
            out = out.rename(columns={"return": "fwd_ret_H4"})

    return out


def analyse_predictions(
    df_raw: pd.DataFrame,
    ret_col: str,
    horizon: Optional[int] = None,
    optimise_params_path: Optional[str] = None,
    regime_parquet: Optional[str] = None,
    side: str = "long",
) -> Dict[str, object]:
    _assert_inputs(df_raw, ret_col)
    df = df_raw.copy()
    side_norm = str(side or "long").lower()
    side_mult = -1.0 if side_norm == "short" else 1.0
    ret_eff_col = "__ret_eff__"
    df[ret_eff_col] = df[ret_col].astype(np.float32) * np.float32(side_mult)

    # Merge regime features from external parquet if provided and not already present
    if regime_parquet:
        rp = Path(regime_parquet)
        if rp.exists():
            try:
                rdf = pd.read_parquet(rp)
                # Normalise label parquet key columns (__ts__ -> ts, __symbol__ -> asset)
                rdf_rename = {}
                for alias, target in [("__ts__", "ts"), ("__symbol__", "asset"), ("timestamp", "ts"), ("symbol", "asset")]:
                    if alias in rdf.columns and target not in rdf.columns:
                        rdf_rename[alias] = target
                if rdf_rename:
                    rdf = rdf.rename(columns=rdf_rename)
                # Normalise ts dtype
                if "ts" in rdf.columns and not pd.api.types.is_datetime64_any_dtype(rdf["ts"]):
                    rdf["ts"] = pd.to_datetime(rdf["ts"], utc=True, errors="coerce")
                regime_cols = [c for c in rdf.columns if c in TOP_REGIME_FEATURES]
                if regime_cols:
                    merge_keys = [k for k in ["ts", "asset"] if k in df.columns and k in rdf.columns]
                    if merge_keys:
                        rdf_sub = rdf[merge_keys + regime_cols].drop_duplicates(subset=merge_keys)
                        # Align ts timezones: normalize both to UTC before merging
                        if "ts" in merge_keys:
                            if pd.api.types.is_datetime64_any_dtype(df["ts"]):
                                df["ts"] = pd.to_datetime(df["ts"], utc=True)
                            if pd.api.types.is_datetime64_any_dtype(rdf_sub["ts"]):
                                rdf_sub = rdf_sub.copy()
                                rdf_sub["ts"] = pd.to_datetime(rdf_sub["ts"], utc=True)
                        df = df.merge(rdf_sub, on=merge_keys, how="left", suffixes=("", "_regime"))
                    else:
                        # Positional fallback: same length
                        if len(rdf) == len(df):
                            for c in regime_cols:
                                if c not in df.columns:
                                    df[c] = rdf[c].values
                    merged_n = sum(1 for c in regime_cols if c in df.columns)
                    print(f"  [preds_metrics] Merged {merged_n} regime features from {rp.name}", file=sys.stderr)
            except Exception as exc:
                warnings.warn(f"[preds_metrics] Could not merge regime parquet: {exc}")

    df = _downcast(df)

    ic = compute_ic(df, by="ts", ret_col=ret_eff_col)
    dec = compute_deciles(df, q=10, by="ts", ret_col=ret_eff_col)
    ll = lead_lag_sanity(df, by_ts="ts", ret_col=ret_eff_col)

    # --- Proxy grids: optimistic, pessimistic, mid (realistic) ---
    grid_opt = run_proxy_grid(df, mode="optimistic", ret_col=ret_eff_col)
    grid_pess = run_proxy_grid(df, mode="pessimistic", ret_col=ret_eff_col)
    grid_mid = run_proxy_grid(df, mode="mid", ret_col=ret_eff_col)

    # --- Optimise-step policy simulation ---
    opt_params = _load_optimise_params(optimise_params_path)
    opt_tp = opt_params["tp_mult"] * 0.01   # convert mult to fraction (assume 1% ATR proxy)
    opt_sl = opt_params["sl_mult"] * 0.01
    # Use a fixed ATR proxy of 1% — realistic for crypto 4h bars
    # The optimise-step uses ATR-scaled barriers; here we use the median ATR proxy
    opt_tpsl_spec = (TPSLSpec(opt_tp, opt_sl / opt_tp if opt_tp > 0 else 0.5),)
    grid_optimise_policy = run_proxy_grid(
        df,
        tpsl=opt_tpsl_spec,
        mode="optimise",
        ret_col=ret_eff_col,
    )

    # --- Regime bucket analysis ---
    regime_analysis = compute_regime_bucket_analysis(df, ret_col=ret_eff_col)

    # --- Step 1: Extreme-move gate + HL-range payoff ---
    gate_stats = compute_extreme_gate_stats(df, ret_col=ret_eff_col)

    # --- Step 2: ATR-relative TP/SL grid ---
    atr_grid = run_atr_proxy_grid(df, ret_col=ret_eff_col)

    # --- Step 3: Stratified isotonic calibration ---
    calibration = compute_stratified_calibration(df, ret_col=ret_eff_col)

    # --- Step 4: Top-k decile metrics (10% and 30%) ---
    topk_metrics = compute_topk_metrics(df, ret_col=ret_eff_col)

    horizon_label = f"H{horizon}" if horizon is not None else ret_col

    return {
        "ic": ic,
        "deciles": dec,
        "lead_lag": ll,
        "grid_optimistic": grid_opt,
        "grid_pessimistic": grid_pess,
        "grid_mid": grid_mid,
        "grid_optimise_policy": grid_optimise_policy,
        "regime_analysis": regime_analysis,
        "gate_stats": gate_stats,
        "atr_grid": atr_grid,
        "calibration": calibration,
        "topk_metrics": topk_metrics,
        "notes": {
            "fee_levels": list(FEE_LEVELS),
            "horizon": horizon_label,
            "ret_column": ret_col,
            "ret_column_effective": ret_eff_col,
            "side": side_norm,
            "tp_levels": [0.02, 0.03, 0.04],
            "sl_ratio": 0.5,
            "optimise_params": opt_params,
            "optimise_tp_pct": opt_tp,
            "optimise_sl_pct": opt_sl,
            "caveat": (
                "TP/SL proxy uses only horizon returns; intrahorizon barrier order unknown. "
                "optimistic/pessimistic are bounds; mid is the realistic expected-value estimate. "
                "optimise_policy uses ATR-scaled params from the optimise step (1% ATR proxy). "
                "atr_grid uses per-row causal ATR for barrier sizing. "
                "calibration shows isotonic IC lift per regime bin."
            ),
        },
    }


def _load_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df = _normalize_meta_columns(df)
    # Drop rows where score is NaN (purged-CV holdout rows have no OOF prediction)
    if "score" in df.columns:
        n_before = len(df)
        df = df.dropna(subset=["score"]).reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"  [preds_metrics] Dropped {n_dropped}/{n_before} rows with NaN score ({100*n_dropped/n_before:.1f}% purged-CV holdout)", file=sys.stderr)
    return df


def run_and_save(
    df: pd.DataFrame,
    out_dir: str,
    targets: Tuple[Tuple[Optional[int], str], ...],
    optimise_params_path: Optional[str] = None,
    regime_parquet: Optional[str] = None,
    side: str = "long",
) -> Dict[str, Dict[str, Path]]:
    """Persist artifacts for each requested horizon under out_dir."""

    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Dict[str, Path]] = {}

    for horizon, ret_col in targets:
        label = f"H{horizon}" if horizon is not None else ret_col
        res = analyse_predictions(
            df,
            ret_col=ret_col,
            horizon=horizon,
            optimise_params_path=optimise_params_path,
            regime_parquet=regime_parquet,
            side=side,
        )
        label_dir = outdir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        # Core artifacts
        ic_json = label_dir / "ic_metrics.json"
        ll_path = label_dir / "lead_lag.csv"
        dec_path = label_dir / "deciles.csv"
        notes_path = label_dir / "notes.json"

        # Grid artifacts
        opt_path = label_dir / "proxy_grid_optimistic.csv"
        pess_path = label_dir / "proxy_grid_pessimistic.csv"
        mid_path = label_dir / "proxy_grid_mid.csv"
        policy_path = label_dir / "proxy_grid_optimise_policy.csv"

        # Regime artifact
        regime_path   = label_dir / "regime_bucket_analysis.csv"
        gate_path     = label_dir / "gate_stats.json"
        atr_grid_path = label_dir / "atr_proxy_grid.csv"
        calib_path    = label_dir / "stratified_calibration.json"
        topk_path     = label_dir / "topk_metrics.csv"

        def _to_json_safe(obj):
            """Recursively convert numpy scalars/arrays to Python natives."""
            if isinstance(obj, dict):
                return {k: _to_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_json_safe(v) for v in obj]
            if isinstance(obj, (np.floating, float)):
                return float(obj) if np.isfinite(obj) else None
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Serialise IC
        ic_json.write_text(json.dumps(_to_json_safe(res["ic"]), indent=2, sort_keys=True))

        res["lead_lag"].to_csv(ll_path, index=False)
        res["deciles"].to_csv(dec_path, index=False)
        res["grid_optimistic"].to_csv(opt_path, index=False)
        res["grid_pessimistic"].to_csv(pess_path, index=False)
        res["grid_mid"].to_csv(mid_path, index=False)
        res["grid_optimise_policy"].to_csv(policy_path, index=False)

        if not res["regime_analysis"].empty:
            res["regime_analysis"].to_csv(regime_path, index=False)

        # Gate stats (JSON)
        gate_path.write_text(json.dumps(_to_json_safe(res["gate_stats"]), indent=2, sort_keys=True))

        # ATR proxy grid (CSV)
        if not res["atr_grid"].empty:
            res["atr_grid"].to_csv(atr_grid_path, index=False)

        # Stratified calibration (JSON)
        calib_path.write_text(json.dumps(_to_json_safe(res["calibration"]), indent=2, sort_keys=True))

        # Top-k metrics (CSV)
        if not res["topk_metrics"].empty:
            res["topk_metrics"].to_csv(topk_path, index=False)

        # Notes
        notes_path.write_text(json.dumps(_to_json_safe(res["notes"]), indent=2, sort_keys=True))

        outputs[label] = {
            "ic_metrics": ic_json,
            "lead_lag": ll_path,
            "deciles": dec_path,
            "grid_optimistic": opt_path,
            "grid_pessimistic": pess_path,
            "grid_mid": mid_path,
            "grid_optimise_policy": policy_path,
            "regime_analysis": regime_path,
            "gate_stats": gate_path,
            "atr_grid": atr_grid_path,
            "calibration": calib_path,
            "topk_metrics": topk_path,
            "notes": notes_path,
        }

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse meta-training / Ridge sizer OOF outputs and save CSV diagnostics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Meta-model OOF (single horizon):
  python -m extreme_price_movements.offline_optimisers.preds_metrics_computations \\
    --input data/artifacts/20260214_190000/meta_oof/meta_oof_long_tf_H4.parquet \\
    --outdir extreme_price_movements/offline_optimisers/reports/preds_metrics/long_tf

  # Ridge sizer OOF (cross-horizon combined signal) with regime features + optimise params:
  python -m extreme_price_movements.offline_optimisers.preds_metrics_computations \\
    --input data/artifacts/20260214_190000/ridge_sizer/ridge_sizer_oof.parquet \\
    --outdir extreme_price_movements/offline_optimisers/reports/preds_metrics/ridge \\
    --regime-parquet data/artifacts/20260214_190000/labels/train_long_tf_4.parquet \\
    --optimise-params extreme_price_movements/reports/20260214_190000/bucket_params.json
""",
    )
    parser.add_argument("--input", required=True, help="Path to OOF CSV/Parquet")
    parser.add_argument(
        "--outdir",
        default="extreme_price_movements/offline_optimisers/reports/preds_metrics",
        help="Output directory for CSV diagnostics",
    )
    parser.add_argument("--ret-column", help="Explicit forward return column to use")
    parser.add_argument(
        "--ret-horizon",
        type=int,
        help="Preferred horizon (in hours) to select when multiple forward columns exist",
    )
    parser.add_argument(
        "--ret-all",
        action="store_true",
        help="Run diagnostics for every detected forward-return horizon",
    )
    parser.add_argument(
        "--returns-table",
        help="Optional parquet/CSV with ts, asset, and forward return columns to merge before analysis",
    )
    parser.add_argument("--returns-ts-col", default="ts", help="Timestamp column name inside returns table")
    parser.add_argument("--returns-asset-col", default="asset", help="Asset column name inside returns table")
    parser.add_argument(
        "--optimise-params",
        default=None,
        help=(
            "Path to bucket_params.json from the optimise step. "
            "Used to simulate the exit policy with realistic ATR-scaled TP/SL. "
            "Falls back to baseline (tp_mult=3.0, sl_mult=1.0) if not provided."
        ),
    )
    parser.add_argument(
        "--regime-parquet",
        default=None,
        help=(
            "Path to a label parquet containing regime features "
            "(__meta_raw__vol_z, __regime_vol_12h__, G_VOL, etc.). "
            "Merged on ts+asset for regime bucket analysis."
        ),
    )
    parser.add_argument(
        "--side",
        choices=["auto", "long", "short"],
        default="auto",
        help="Payoff orientation. 'short' flips forward returns for IC/decile/proxy payoff metrics.",
    )
    args = parser.parse_args()

    _df = _load_input(Path(args.input))
    if args.returns_table:
        ret_df = _load_returns_table(
            Path(args.returns_table),
            ts_col=args.returns_ts_col,
            asset_col=args.returns_asset_col,
        )
        _df = _merge_returns_columns(_df, ret_df)

    targets = _prepare_target_columns(
        _df,
        preferred_col=args.ret_column,
        preferred_horizon=args.ret_horizon,
        all_horizons=args.ret_all,
    )

    side_arg = str(args.side).lower()
    if side_arg == "auto":
        inp = str(args.input).lower()
        side_arg = "short" if "short" in inp else "long"

    outputs = run_and_save(
        _df,
        args.outdir,
        targets,
        optimise_params_path=args.optimise_params,
        regime_parquet=args.regime_parquet,
        side=side_arg,
    )
    print(json.dumps(
        {k: {kk: str(vv) for kk, vv in paths.items()} for k, paths in outputs.items()},
        indent=2,
        sort_keys=True,
    ))
