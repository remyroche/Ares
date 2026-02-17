"""Prediction metrics + proxy TP/SL analysis for meta-training outputs.

Usage example:
    python -m extreme_price_movements.offline_optimisers.preds_metrics_computations \
      --input extreme_price_movements/reports/meta_model_long_tf_race.csv \
      --outdir extreme_price_movements/offline_optimisers/reports/preds_metrics

Expected minimum columns (or accepted aliases):
- ts (aliases: timestamp, __ts__)
- score (aliases: oof_pred, pred, oof_probs, base_score)
- fwd_ret_4h (aliases: __y_ret__, y_ret, return, fwd_return, fwd_ret)
Optional:
- asset (aliases: symbol, asset_id)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

FEE_RT: np.float32 = np.float32(0.005)  # 0.5% round-trip
DTYPE_F32 = np.float32
DTYPE_I32 = np.int32


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


def _assert_inputs(df: pd.DataFrame) -> None:
    required = {"ts", "score", "fwd_ret_4h"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["score"].isna().any():
        raise ValueError("NaNs in score")
    if df["fwd_ret_4h"].isna().any():
        raise ValueError("NaNs in fwd_ret_4h")
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

def compute_ic(df: pd.DataFrame, by: Optional[str] = "ts") -> Dict[str, float]:
    score = df["score"].to_numpy()
    ret = df["fwd_ret_4h"].to_numpy()
    out: Dict[str, float] = {"ic_global": _safe_corr(score, ret)}

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

    ic_series = g.apply(lambda x: _safe_corr(x["score"].to_numpy(), x["fwd_ret_4h"].to_numpy()))
    ic_mean = float(np.nanmean(ic_series.values))
    ic_std = float(np.nanstd(ic_series.values))
    out["ic_mean"] = ic_mean
    out["ic_std"] = ic_std
    out["ic_ir"] = float(ic_mean / ic_std) if ic_std > 0 else np.nan
    out["ic_mode"] = "per_ts_cross_sectional"
    return out


def compute_deciles(df: pd.DataFrame, q: int = 10, by: str = "ts") -> pd.DataFrame:
    # Use float64 for ranking stability; cast back later
    score64 = df["score"].astype(np.float64)
    pct = score64.groupby(df[by], sort=False).rank(pct=True, method="first")
    qbin = np.minimum((pct.to_numpy() * q).astype(np.int32), q - 1)
    tmp = pd.DataFrame({"qbin": qbin, "ret": df["fwd_ret_4h"].to_numpy()})
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
        mid = np.where(fwd_ret < 0, -sl, np.minimum(fwd_ret, tp * np.float32(0.25))).astype(np.float32)
        realised = np.where(hit_tp, tp, np.where(hit_sl, -sl, mid)).astype(np.float32)
    else:
        raise ValueError("mode must be 'optimistic' or 'pessimistic'")

    net = (pos_w * realised) - (pos_w * fee_rt)
    return net.astype(np.float32)




def lead_lag_sanity(df: pd.DataFrame, by_ts: str = "ts", lags: Tuple[int, ...] = (-2, -1, 0, 1, 2)) -> pd.DataFrame:
    """Quick lookahead/alignment check in 4h increments."""
    d = df.sort_values(by_ts)
    score = d["score"].to_numpy()
    ret = d["fwd_ret_4h"].to_numpy()
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

def compute_strategy_kpis(net_ret: np.ndarray) -> Dict[str, float]:
    x = net_ret.astype(np.float64, copy=False)
    if x.size == 0:
        return {k: np.nan for k in ["mean", "median", "win_rate", "p10", "p90"]}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "win_rate": float(np.mean(x > 0)),
        "p10": float(np.quantile(x, 0.10)),
        "p90": float(np.quantile(x, 0.90)),
    }


def run_proxy_grid(
    df: pd.DataFrame,
    selections: Tuple[SelectionSpec, ...] = (SelectionSpec(0.2), SelectionSpec(0.3), SelectionSpec(0.4)),
    tpsl: Tuple[TPSLSpec, ...] = (TPSLSpec(0.02, 0.5), TPSLSpec(0.03, 0.5), TPSLSpec(0.04, 0.5)),
    by: str = "ts",
    mode: str = "optimistic",
) -> pd.DataFrame:
    fwd = df["fwd_ret_4h"].to_numpy(dtype=np.float32)
    rows = []
    for sel in selections:
        pos_w = infer_positions_top_frac(df, sel.top_frac, by=by)
        for spec in tpsl:
            sl = spec.tp * spec.sl_ratio
            net = apply_tpsl_proxy(fwd, pos_w, tp=spec.tp, sl=sl, mode=mode)
            k = compute_strategy_kpis(net)
            hr = proxy_hit_rates(fwd, pos_w, tp=np.float32(spec.tp), sl=np.float32(sl))
            rows.append(
                {
                    "mode": mode,
                    "top_frac": sel.top_frac,
                    "tp": spec.tp,
                    "sl": sl,
                    "fee_rt": float(FEE_RT),
                    **k,
                    **hr,
                    "n_active": int(np.sum(pos_w > 0)),
                    "n_total": int(len(pos_w)),
                    "active_rate": float(np.mean(pos_w > 0)),
                    "avg_fee_per_row": float(FEE_RT) * float(np.mean(pos_w > 0)),
                }
            )
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


# =============================================================================
# 3) Meta-output loader + top-level analysis
# =============================================================================


def _normalize_meta_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    aliases = {
        "ts": ["ts", "timestamp", "__ts__"],
        "score": ["score", "oof_pred", "pred", "oof_probs", "base_score"],
        "fwd_ret_4h": ["fwd_ret_4h", "__y_ret__", "y_ret", "return", "fwd_return", "fwd_ret"],
        "asset": ["asset", "symbol", "asset_id"],
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

    if "ts" in out.columns and not np.issubdtype(out["ts"].dtype, np.datetime64):
        try:
            out["ts"] = pd.to_datetime(out["ts"], utc=True)
        except Exception:
            pass

    return out


def analyse_predictions(df_raw: pd.DataFrame) -> Dict[str, object]:
    _assert_inputs(df_raw)
    df = _downcast(df_raw)

    ic = compute_ic(df, by="ts")
    dec = compute_deciles(df, q=10, by="ts")
    ll = lead_lag_sanity(df, by_ts="ts")
    grid_opt = run_proxy_grid(df, mode="optimistic")
    grid_pess = run_proxy_grid(df, mode="pessimistic")

    return {
        "ic": ic,
        "deciles": dec,
        "lead_lag": ll,
        "grid_optimistic": grid_opt,
        "grid_pessimistic": grid_pess,
        "notes": {
            "fee_rt": float(FEE_RT),
            "horizon": "4h",
            "tp_levels": [0.02, 0.03, 0.04],
            "sl_ratio": 0.5,
            "caveat": (
                "TP/SL proxy uses only horizon returns; intrahorizon barrier order unknown. "
                "Use optimistic/pessimistic bounds as sanity checks."
            ),
        },
    }


def _load_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return _normalize_meta_columns(df)


def run_and_save(df: pd.DataFrame, out_dir: str) -> Dict[str, Path]:
    """Persist artifacts as CSV/JSON under out_dir."""
    res = analyse_predictions(df)
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    ic_json = outdir / "ic_metrics.json"
    ll_path = outdir / "lead_lag.csv"
    dec_path = outdir / "deciles.csv"
    opt_path = outdir / "proxy_grid_optimistic.csv"
    pess_path = outdir / "proxy_grid_pessimistic.csv"
    notes_path = outdir / "notes.json"

    ic_json.write_text(json.dumps(res["ic"], indent=2, sort_keys=True))
    res["lead_lag"].to_csv(ll_path, index=False)
    res["deciles"].to_csv(dec_path, index=False)
    res["grid_optimistic"].to_csv(opt_path, index=False)
    res["grid_pessimistic"].to_csv(pess_path, index=False)
    notes_path.write_text(json.dumps(res["notes"], indent=2, sort_keys=True))

    return {
        "ic_metrics": ic_json,
        "lead_lag": ll_path,
        "deciles": dec_path,
        "grid_optimistic": opt_path,
        "grid_pessimistic": pess_path,
        "notes": notes_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse meta-training outputs and save CSV diagnostics.")
    parser.add_argument("--input", required=True, help="Path to meta-training output CSV/Parquet")
    parser.add_argument(
        "--outdir",
        default="extreme_price_movements/offline_optimisers/reports/preds_metrics",
        help="Output directory for CSV diagnostics",
    )
    args = parser.parse_args()

    _df = _load_input(Path(args.input))
    outputs = run_and_save(_df, args.outdir)
    print(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2, sort_keys=True))
