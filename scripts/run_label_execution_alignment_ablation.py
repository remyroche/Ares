#!/usr/bin/env python3
"""Ablate label/execution alignment on monthly walk-forward OOS rows.

This is a development diagnostic, not a promotion script.  For each monthly
fold it trains a small label-specific LGBM on the policy-optimisation half and
evaluates on the untouched policy-validation half.  Execution metrics are
replayed through the existing simple_policy_optimiser exit/cost model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("EPM_EXCHANGE", "krakenfutures")
os.environ.setdefault("EPM_SIMPLE_POLICY_15M_DOWNLOAD", "0")
os.environ.setdefault("EPM_SIMPLE_POLICY_1M_DOWNLOAD", "0")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import scripts.report_single_head_monthly_vanilla_walkforward_oos as vanilla
import scripts.run_single_head_monthly_walkforward_oos as wf
from extreme_price_movements import simple_policy_optimiser as spo


DEFAULT_EXPERIMENT_ID = (
    "20260701_193000_single_head_monthly_walkforward_forwardburnin_"
    "no_window_hpo_no_regime_fe"
)
DEFAULT_SOURCE_RUN_ID = wf.DEFAULT_SOURCE_RUN_ID
FOLD_MONTHS = {
    "train_through_march_score_april": "2026-04",
    "train_through_april_score_may": "2026-05",
    "train_through_may_score_june": "2026-06",
}
TOP_FRACS = (0.30, 0.15, 0.10, 0.05, 0.03)
ROUND_TRIP_COST = 0.0030
MODEL_RANDOM_SEED = 7301


@dataclass(frozen=True)
class LabelArm:
    name: str
    description: str
    production_safe: bool


LABEL_ARMS = [
    LabelArm("S0_current_y_bin", "current hard TP/SL label", True),
    LabelArm("S2_cost_aware_return", "final return minus explicit round-trip cost", True),
    LabelArm("S3_path_quality", "MFE/MAE/timing path-quality soft label", True),
    LabelArm("S6_asymmetric_downside", "path-quality with hard downside/SL caps", True),
    LabelArm("S7_horizon_blended", "blend current, vol-normalized TP/SL, and fast-MFE labels", True),
    LabelArm("S8_timestamp_rank_path", "timestamp-balanced rank of path-quality target", True),
    LabelArm("S9_fast_mfe_3bars", "fast favorable excursion within three bars", True),
    LabelArm("S10_vanilla_independent_net", "independent replay of vanilla exit net after costs", True),
    LabelArm("S11_fixed_tp2_sl1_net", "independent fixed ATR TP2/SL1 net after costs", True),
    LabelArm(
        "S14_policy_net_path_blend",
        "50/50 blend of independent vanilla exit net and asymmetric path quality",
        True,
    ),
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _safe_mean(values: Any) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def _safe_quantile(values: Any, q: float) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(arr.quantile(q)) if len(arr) else float("nan")


def _safe_spearman(x: Any, y: Any) -> float:
    x_ser = pd.to_numeric(pd.Series(x), errors="coerce")
    y_ser = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = x_ser.notna() & y_ser.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    xr = x_ser[mask].rank(method="average")
    yr = y_ser[mask].rank(method="average")
    if xr.nunique(dropna=True) < 2 or yr.nunique(dropna=True) < 2:
        return float("nan")
    return float(xr.corr(yr))


def _sigmoid(x: Any) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)))


def _rank_top_indices(score: Any, frac: float) -> np.ndarray:
    score_ser = pd.to_numeric(pd.Series(score), errors="coerce")
    valid = score_ser.notna().to_numpy()
    if not bool(valid.any()):
        return np.array([], dtype=np.int64)
    valid_idx = np.flatnonzero(valid)
    k = max(1, int(math.ceil(float(frac) * len(valid_idx))))
    order = np.argsort(-score_ser.iloc[valid_idx].to_numpy(dtype=np.float64), kind="mergesort")
    return valid_idx[order[:k]].astype(np.int64, copy=False)


def _rank_pct_from_score(score: pd.Series) -> pd.Series:
    ranked = pd.to_numeric(score, errors="coerce").rank(method="max", pct=True)
    return ranked.fillna(0.0).clip(0.0, 1.0)


def _drawdown(values: Iterable[float]) -> float:
    arr = np.nan_to_num(np.asarray(list(values), dtype=np.float64), nan=0.0)
    if arr.size == 0:
        return 0.0
    curve = np.cumsum(arr)
    return float(np.min(curve - np.maximum.accumulate(curve)))


def _ndcg_at_frac(score: Any, gain: Any, frac: float) -> float:
    score_ser = pd.to_numeric(pd.Series(score), errors="coerce")
    gain_ser = pd.to_numeric(pd.Series(gain), errors="coerce").clip(lower=0.0, upper=1.0)
    valid = score_ser.notna() & gain_ser.notna()
    if int(valid.sum()) < 2:
        return float("nan")
    k = max(1, int(math.ceil(float(frac) * int(valid.sum()))))
    scores = score_ser[valid].to_numpy(dtype=np.float64)
    gains = gain_ser[valid].to_numpy(dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")[:k]
    ideal = np.argsort(-gains, kind="mergesort")[:k]
    discount = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    dcg = float(np.sum(((2.0 ** gains[order]) - 1.0) * discount))
    idcg = float(np.sum(((2.0 ** gains[ideal]) - 1.0) * discount))
    return dcg / idcg if idcg > 0.0 else float("nan")


def _path_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    barrier = pd.to_numeric(frame.get("barrier_pct"), errors="coerce").abs().clip(lower=1e-8)
    mfe = pd.to_numeric(frame.get("mfe_ret"), errors="coerce").clip(lower=0.0)
    mae_raw = pd.to_numeric(frame.get("mae_ret"), errors="coerce")
    finite_mae = mae_raw.dropna()
    if len(finite_mae) and float(finite_mae.median()) < 0.0:
        mae = (-mae_raw).clip(lower=0.0)
    else:
        mae = mae_raw.clip(lower=0.0)
    bars_to_mfe = pd.to_numeric(frame.get("bars_to_mfe"), errors="coerce")
    bars_policy = pd.to_numeric(frame.get("bars_policy"), errors="coerce")
    ret = pd.to_numeric(frame.get("return"), errors="coerce")
    y_bin = pd.to_numeric(frame.get("y_bin"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["barrier"] = barrier
    out["mfe"] = mfe.fillna(0.0)
    out["mae"] = mae.fillna(0.0)
    out["mfe_norm"] = out["mfe"] / barrier
    out["mae_norm"] = out["mae"] / barrier
    out["bars_to_mfe"] = bars_to_mfe.fillna(bars_policy).fillna(24.0).clip(lower=0.0)
    out["bars_policy"] = bars_policy.fillna(24.0).clip(lower=0.0)
    out["return"] = ret.fillna(0.0)
    out["y_bin"] = y_bin
    out["y_outcome"] = pd.to_numeric(frame.get("y_outcome"), errors="coerce")
    return out


def _independent_vanilla_net_returns(
    rows: pd.DataFrame,
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[pd.Series, pd.Series]:
    if rows.empty:
        empty = pd.Series(dtype=np.float64)
        return empty, empty
    metrics = spo.simulate_and_score(
        rows.reset_index(drop=True),
        *paths,
        cost_pct=spo.DEFAULT_POLICY_PER_SIDE_COST_PCT,
        size_power=1.0,
        market_mode="perps",
        max_concurrent_trades=max(10_000, len(rows) + 1),
        max_concurrent_per_asset=max(10_000, len(rows) + 1),
    )
    selected = np.asarray(metrics.get("selected_mask", []), dtype=bool)
    out = np.full(len(rows), np.nan, dtype=np.float64)
    reasons = np.full(len(rows), "", dtype=object)
    gains = np.asarray(metrics.get("raw_gains", []), dtype=np.float64)
    sizes = np.asarray(metrics.get("sizes", []), dtype=np.float64)
    idx = np.flatnonzero(selected) if len(selected) == len(rows) else np.arange(min(len(gains), len(rows)))
    if len(idx) and len(gains):
        unit = np.divide(gains[: len(idx)], np.maximum(sizes[: len(idx)], 1e-12))
        out[idx[: len(unit)]] = unit
    exit_reason = metrics.get("exit_reason")
    if exit_reason is not None and len(idx):
        reason_arr = np.asarray(exit_reason, dtype=object)
        reasons[idx[: len(reason_arr)]] = reason_arr[: len(idx)].astype(str)
    return pd.Series(out, index=rows.index), pd.Series(reasons, index=rows.index)


def _fixed_tp2_sl1_net_returns(
    rows: pd.DataFrame,
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[pd.Series, pd.Series]:
    if rows.empty:
        empty = pd.Series(dtype=np.float64)
        return empty, empty
    sim = spo._simulate_simple_tp_sl_rows(
        rows.reset_index(drop=True),
        paths,
        cost_pct=spo.DEFAULT_POLICY_PER_SIDE_COST_PCT,
        size_power=1.0,
        sl_mult=1.0,
        tp_mult=2.0,
        market_mode="perps",
    )
    out = np.full(len(rows), np.nan, dtype=np.float64)
    reasons = np.full(len(rows), "", dtype=object)
    if not sim.empty and "net_gain" in sim.columns:
        gains = pd.to_numeric(sim["net_gain"], errors="coerce").to_numpy(dtype=np.float64)
        sizes = pd.to_numeric(sim.get("size", pd.Series(1.0, index=sim.index)), errors="coerce").to_numpy(
            dtype=np.float64
        )
        n = min(len(out), len(gains))
        out[:n] = np.divide(gains[:n], np.maximum(sizes[:n], 1e-12))
        reason_col = "simple_policy_exit_reason" if "simple_policy_exit_reason" in sim.columns else "exit_reason"
        if reason_col in sim.columns:
            reasons[:n] = sim[reason_col].astype(str).to_numpy()[:n]
    return pd.Series(out, index=rows.index), pd.Series(reasons, index=rows.index)


def _make_targets(
    frame: pd.DataFrame,
    *,
    vanilla_net: pd.Series,
    fixed_net: pd.Series,
) -> dict[str, pd.DataFrame]:
    m = _path_metrics(frame)
    ret_net = m["return"] - ROUND_TRIP_COST
    vn_tp2_sl1 = ((m["mfe_norm"] >= 2.0) & (m["mae_norm"] < 1.0)).astype(float)
    fast_mfe = ((m["mfe_norm"] >= 1.0) & (m["bars_to_mfe"] <= 3.0)).astype(float)
    current = m["y_bin"].astype(float)

    cost_aware = pd.Series(_sigmoid(ret_net / 0.006), index=frame.index)

    path_raw = (
        1.05 * m["mfe_norm"]
        - 1.30 * m["mae_norm"]
        - 0.10 * np.log1p(m["bars_to_mfe"])
        + 0.35 * (m["return"] > 0.0).astype(float)
    )
    path_quality = pd.Series(_sigmoid((path_raw - 0.25) / 1.20), index=frame.index)

    downside_raw = (
        0.90 * m["mfe_norm"]
        - 1.85 * m["mae_norm"]
        + (ret_net / m["barrier"].clip(lower=1e-8))
        - 0.15 * np.log1p(m["bars_to_mfe"])
    )
    asymmetric = pd.Series(_sigmoid((downside_raw - 0.10) / 1.25), index=frame.index)
    bad_path = (m["mae_norm"] >= 1.0) | ((m["y_outcome"] == 0.0).fillna(False))
    asymmetric = asymmetric.where(~bad_path, np.minimum(asymmetric, 0.25))

    blended = (0.40 * current) + (0.30 * vn_tp2_sl1) + (0.30 * fast_mfe)

    timestamp_rank = path_quality.groupby(frame["timestamp"], dropna=False).rank(method="average", pct=True)
    timestamp_rank = timestamp_rank.fillna(path_quality.rank(method="average", pct=True)).clip(0.0, 1.0)
    rank_path = (0.50 * path_quality) + (0.50 * timestamp_rank)

    vanilla_soft = pd.Series(_sigmoid(vanilla_net.fillna(-0.02) / 0.004), index=frame.index)
    fixed_soft = pd.Series(_sigmoid(fixed_net.fillna(-0.02) / 0.004), index=frame.index)
    policy_path_blend = ((0.50 * vanilla_soft) + (0.50 * asymmetric)).clip(0.0, 1.0)

    raw_targets = {
        "S0_current_y_bin": current,
        "S2_cost_aware_return": cost_aware,
        "S3_path_quality": path_quality,
        "S6_asymmetric_downside": asymmetric,
        "S7_horizon_blended": blended,
        "S8_timestamp_rank_path": rank_path,
        "S9_fast_mfe_3bars": fast_mfe,
        "S10_vanilla_independent_net": vanilla_soft,
        "S11_fixed_tp2_sl1_net": fixed_soft,
        "S14_policy_net_path_blend": policy_path_blend,
    }
    hard_targets = {
        "S0_current_y_bin": current >= 0.5,
        "S2_cost_aware_return": ret_net > 0.0,
        "S3_path_quality": path_quality >= 0.55,
        "S6_asymmetric_downside": asymmetric >= 0.55,
        "S7_horizon_blended": blended >= 0.50,
        "S8_timestamp_rank_path": rank_path >= 0.70,
        "S9_fast_mfe_3bars": fast_mfe >= 0.5,
        "S10_vanilla_independent_net": vanilla_net > 0.0,
        "S11_fixed_tp2_sl1_net": fixed_net > 0.0,
        "S14_policy_net_path_blend": policy_path_blend >= 0.55,
    }
    out: dict[str, pd.DataFrame] = {}
    for arm in LABEL_ARMS:
        soft = pd.to_numeric(raw_targets[arm.name], errors="coerce").clip(0.0, 1.0)
        hard = pd.Series(hard_targets[arm.name], index=frame.index).fillna(False).astype(float)
        out[arm.name] = pd.DataFrame({"target_soft": soft, "target_hard": hard}, index=frame.index)
    return out


def _feature_columns(run_root: Path, frame: pd.DataFrame) -> list[str]:
    path = run_root / "quality_reports" / "base_model_feature_importance.csv"
    if not path.exists():
        return []
    features = pd.read_csv(path)
    if "used_by_model" in features.columns:
        features = features[features["used_by_model"].fillna(False).astype(bool)].copy()
    if "selected_feature_position" in features.columns:
        features = features.sort_values("selected_feature_position")
    cols = [str(v) for v in features["feature"].dropna().drop_duplicates().tolist()]
    return [col for col in cols if col in frame.columns]


def _fit_predict_lgbm(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    y_train: pd.Series,
) -> tuple[pd.Series, dict[str, Any]]:
    if len(train) < 50 or len(valid) < 10 or not features:
        return pd.Series(np.nan, index=valid.index), {"trained": False, "reason": "insufficient_rows_or_features"}
    y = pd.to_numeric(y_train.loc[train.index], errors="coerce")
    mask = y.notna()
    if int(mask.sum()) < 50 or y[mask].nunique(dropna=True) < 2:
        return pd.Series(np.nan, index=valid.index), {"trained": False, "reason": "constant_or_sparse_target"}
    x_train = train.loc[mask, features].replace([np.inf, -np.inf], np.nan)
    x_valid = valid.loc[:, features].replace([np.inf, -np.inf], np.nan)
    med = x_train.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_train = x_train.fillna(med)
    x_valid = x_valid.fillna(med)
    model = LGBMRegressor(
        objective="regression",
        n_estimators=120,
        learning_rate=0.045,
        num_leaves=31,
        min_child_samples=25,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=MODEL_RANDOM_SEED,
        n_jobs=2,
        verbosity=-1,
    )
    model.fit(x_train, y.loc[mask].to_numpy(dtype=np.float32))
    pred = np.clip(model.predict(x_valid), 0.0, 1.0)
    diag = {
        "trained": True,
        "train_rows": int(mask.sum()),
        "valid_rows": int(len(valid)),
        "feature_count": int(len(features)),
        "target_mean_train": float(y.loc[mask].mean()),
        "target_std_train": float(y.loc[mask].std(ddof=0)),
        "pred_mean": float(np.mean(pred)) if len(pred) else float("nan"),
        "pred_std": float(np.std(pred)) if len(pred) else float("nan"),
    }
    return pd.Series(pred, index=valid.index), diag


def _score_ranking(
    *,
    eval_month: str,
    run_id: str,
    arm: str,
    selector: str,
    score: pd.Series,
    target: pd.DataFrame,
    train_rows: int,
    valid_rows: int,
    model_diag: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hard = pd.to_numeric(target["target_hard"], errors="coerce")
    soft = pd.to_numeric(target["target_soft"], errors="coerce")
    base_rate = _safe_mean(hard)
    for frac in TOP_FRACS:
        idx = _rank_top_indices(score, frac)
        top_hard = hard.iloc[idx] if len(idx) else pd.Series(dtype=np.float64)
        top_soft = soft.iloc[idx] if len(idx) else pd.Series(dtype=np.float64)
        rows.append(
            {
                "eval_month": eval_month,
                "run_id": run_id,
                "arm": arm,
                "selector": selector,
                "top_frac": float(frac),
                "train_rows": int(train_rows),
                "valid_rows": int(valid_rows),
                "selected_rows": int(len(idx)),
                "target_base_rate": base_rate,
                "target_top_hard_rate": _safe_mean(top_hard),
                "target_lift": _safe_mean(top_hard) / base_rate if base_rate and math.isfinite(base_rate) else float("nan"),
                "target_top_soft_mean": _safe_mean(top_soft),
                "ic_soft": _safe_spearman(score, soft),
                "ic_hard": _safe_spearman(score, hard),
                "ndcg": _ndcg_at_frac(score, soft, frac),
                **{f"model_{k}": v for k, v in dict(model_diag or {}).items()},
            }
        )
    return rows


def _execution_metrics(
    rows: pd.DataFrame,
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    if rows.empty:
        return {
            "candidate_rows": 0,
            "n_trades": 0,
            "net_pnl": 0.0,
            "gross_pnl": 0.0,
            "mean_net_trade": 0.0,
            "mean_gross_trade": 0.0,
            "hit_rate": 0.0,
            "gross_hit_rate": 0.0,
            "max_drawdown": 0.0,
            "avg_holding_bars": 0.0,
            "full_sl_exit_rate": 0.0,
            "trailing_exit_rate": 0.0,
            "timeout_exit_rate": 0.0,
        }
    metrics = spo.simulate_and_score(
        rows.reset_index(drop=True),
        *paths,
        cost_pct=spo.DEFAULT_POLICY_PER_SIDE_COST_PCT,
        size_power=1.0,
        market_mode="perps",
        max_concurrent_trades=spo.MAX_CONCURRENT_TRADES,
        max_concurrent_per_asset=spo.DEPLOYMENT_MAX_CONCURRENT_PER_ASSET,
    )
    raw = np.asarray(metrics.get("raw_gains", []), dtype=np.float64)
    gross = np.asarray(metrics.get("gross_gains", []), dtype=np.float64)
    sizes = np.asarray(metrics.get("sizes", []), dtype=np.float64)
    net_unit = np.divide(raw, np.maximum(sizes, 1e-12)) if len(raw) else np.array([], dtype=np.float64)
    gross_unit = np.divide(gross, np.maximum(sizes, 1e-12)) if len(gross) else np.array([], dtype=np.float64)
    n_trades = int(metrics.get("total_trades", 0) or 0)
    return {
        "candidate_rows": int(len(rows)),
        "n_trades": n_trades,
        "net_pnl": float(np.nansum(raw)) if len(raw) else 0.0,
        "gross_pnl": float(np.nansum(gross)) if len(gross) else 0.0,
        "mean_net_trade": float(np.nanmean(net_unit)) if len(net_unit) else 0.0,
        "mean_gross_trade": float(np.nanmean(gross_unit)) if len(gross_unit) else 0.0,
        "hit_rate": float(np.nanmean(net_unit > 0.0)) if len(net_unit) else 0.0,
        "gross_hit_rate": float(np.nanmean(gross_unit > 0.0)) if len(gross_unit) else 0.0,
        "max_drawdown": _drawdown(raw),
        "avg_holding_bars": float(metrics.get("avg_holding_bars", 0.0) or 0.0),
        "full_sl_exit_rate": float(metrics.get("full_sl_exit_count", 0) / max(n_trades, 1)),
        "trailing_exit_rate": float(metrics.get("trailing_exit_count", 0) / max(n_trades, 1)),
        "timeout_exit_rate": float(metrics.get("timeout_exit_rate", 0.0) or 0.0),
    }


def _score_execution(
    *,
    eval_month: str,
    run_id: str,
    arm: str,
    selector: str,
    score: pd.Series,
    validation: pd.DataFrame,
    validation_paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    target: pd.DataFrame,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for frac in TOP_FRACS:
        idx = _rank_top_indices(score, frac)
        selected = validation.iloc[idx].copy().reset_index(drop=True)
        selected_paths = spo._path_take(validation_paths, idx)
        if len(selected):
            selected["rank_pct"] = _rank_pct_from_score(pd.Series(score.iloc[idx].to_numpy(), index=selected.index))
        metrics = _execution_metrics(selected, selected_paths)
        target_sel = target.iloc[idx] if len(idx) else pd.DataFrame()
        ret = pd.to_numeric(selected.get("return"), errors="coerce") if len(selected) else pd.Series(dtype=np.float64)
        out.append(
            {
                "eval_month": eval_month,
                "run_id": run_id,
                "arm": arm,
                "selector": selector,
                "top_frac": float(frac),
                "selected_current_label_mean_return": _safe_mean(ret),
                "selected_target_soft_mean": _safe_mean(target_sel.get("target_soft")),
                "selected_target_hard_rate": _safe_mean(target_sel.get("target_hard")),
                **metrics,
            }
        )
    return out


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    rows: list[dict[str, Any]] = []
    group_cols = ["arm", "selector", "top_frac"]
    for key, group in frame.groupby(group_cols, dropna=False, observed=True):
        arm, selector, top_frac = key
        rows.append(
            {
                "arm": arm,
                "selector": selector,
                "top_frac": top_frac,
                "months": int(group["eval_month"].nunique()),
                "candidate_rows": int(pd.to_numeric(group["candidate_rows"], errors="coerce").sum()),
                "n_trades": int(pd.to_numeric(group["n_trades"], errors="coerce").sum()),
                "net_pnl": float(pd.to_numeric(group["net_pnl"], errors="coerce").sum()),
                "gross_pnl": float(pd.to_numeric(group["gross_pnl"], errors="coerce").sum()),
                "positive_months": int((pd.to_numeric(group["net_pnl"], errors="coerce") > 0.0).sum()),
                "worst_month_net_pnl": float(pd.to_numeric(group["net_pnl"], errors="coerce").min()),
                "best_month_net_pnl": float(pd.to_numeric(group["net_pnl"], errors="coerce").max()),
                "mean_net_trade": _safe_mean(group["mean_net_trade"]),
                "hit_rate": _safe_mean(group["hit_rate"]),
                "full_sl_exit_rate": _safe_mean(group["full_sl_exit_rate"]),
                "trailing_exit_rate": _safe_mean(group["trailing_exit_rate"]),
                "selected_target_hard_rate": _safe_mean(group["selected_target_hard_rate"]),
                "selected_target_soft_mean": _safe_mean(group["selected_target_soft_mean"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["selector", "top_frac", "net_pnl", "worst_month_net_pnl"],
        ascending=[True, True, False, False],
    )


def _write_markdown(
    *,
    output_dir: Path,
    execution_aggregate: pd.DataFrame,
    execution_monthly: pd.DataFrame,
    ranking_metrics: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    md = output_dir / "label_execution_alignment_ablation.md"
    top = execution_aggregate[
        execution_aggregate["selector"].eq("ablation_lgbm")
        & execution_aggregate["top_frac"].eq(0.15)
    ].copy()
    top = top.sort_values(["net_pnl", "worst_month_net_pnl"], ascending=[False, False]).head(12)
    high_conf = execution_aggregate[
        execution_aggregate["selector"].eq("ablation_lgbm")
        & execution_aggregate["top_frac"].isin([0.10, 0.05, 0.03])
    ].copy()
    high_conf = high_conf.sort_values(
        ["top_frac", "net_pnl", "worst_month_net_pnl"],
        ascending=[False, False, False],
    )
    oracle = execution_aggregate[
        execution_aggregate["selector"].eq("oracle_label")
        & execution_aggregate["top_frac"].eq(0.15)
    ].copy()
    oracle = oracle.sort_values(["net_pnl", "worst_month_net_pnl"], ascending=[False, False]).head(8)
    baseline = execution_aggregate[
        execution_aggregate["selector"].isin(["current_meta", "current_base"])
        & execution_aggregate["top_frac"].eq(0.15)
    ].copy()
    baseline = baseline.sort_values(["selector", "net_pnl"], ascending=[True, False])

    rank_view = ranking_metrics[
        ranking_metrics["selector"].eq("ablation_lgbm")
        & ranking_metrics["top_frac"].isin([0.30, 0.10])
    ].copy()
    rank_view = (
        rank_view.groupby(["arm", "top_frac"], observed=True)
        .agg(
            mean_ic_soft=("ic_soft", "mean"),
            mean_target_lift=("target_lift", "mean"),
            mean_ndcg=("ndcg", "mean"),
            months=("eval_month", "nunique"),
        )
        .reset_index()
        .sort_values(["top_frac", "mean_target_lift"], ascending=[True, False])
    )

    monthly_view = execution_monthly[
        execution_monthly["selector"].eq("ablation_lgbm")
        & execution_monthly["top_frac"].eq(0.15)
    ].copy()
    monthly_view = monthly_view.sort_values(["arm", "eval_month"])

    def table(frame: pd.DataFrame, cols: list[str]) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[c for c in cols if c in frame.columns]].copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    lines = [
        "# Label/Execution Alignment Ablation",
        "",
        "Scope: Apr/May/Jun 2026 single-head monthly walk-forward policy-validation halves only.",
        "",
        "This is development evidence.  `ablation_lgbm` trains on each month policy-optimisation half and scores the later validation half.  `oracle_label` is an economic ceiling, not deployable.",
        "",
        "## Baseline Selectors",
        "",
        table(
            baseline,
            [
                "arm",
                "selector",
                "top_frac",
                "months",
                "n_trades",
                "net_pnl",
                "positive_months",
                "worst_month_net_pnl",
                "mean_net_trade",
                "hit_rate",
                "full_sl_exit_rate",
                "trailing_exit_rate",
            ],
        ),
        "",
        "## Ablation LGBM Top 15%",
        "",
        table(
            top,
            [
                "arm",
                "selector",
                "top_frac",
                "months",
                "n_trades",
                "net_pnl",
                "positive_months",
                "worst_month_net_pnl",
                "mean_net_trade",
                "hit_rate",
                "full_sl_exit_rate",
                "trailing_exit_rate",
            ],
        ),
        "",
        "## Ablation LGBM High Confidence",
        "",
        table(
            high_conf,
            [
                "arm",
                "selector",
                "top_frac",
                "months",
                "n_trades",
                "net_pnl",
                "positive_months",
                "worst_month_net_pnl",
                "mean_net_trade",
                "hit_rate",
                "full_sl_exit_rate",
                "trailing_exit_rate",
            ],
        ),
        "",
        "## Oracle Label Top 15% Ceiling",
        "",
        table(
            oracle,
            [
                "arm",
                "selector",
                "top_frac",
                "months",
                "n_trades",
                "net_pnl",
                "positive_months",
                "worst_month_net_pnl",
                "mean_net_trade",
                "hit_rate",
                "full_sl_exit_rate",
                "trailing_exit_rate",
            ],
        ),
        "",
        "## Learnability",
        "",
        table(
            rank_view,
            ["arm", "top_frac", "months", "mean_ic_soft", "mean_target_lift", "mean_ndcg"],
        ),
        "",
        "## Monthly Ablation LGBM Top 15%",
        "",
        table(
            monthly_view,
            [
                "eval_month",
                "arm",
                "n_trades",
                "net_pnl",
                "mean_net_trade",
                "hit_rate",
                "full_sl_exit_rate",
                "trailing_exit_rate",
                "selected_target_hard_rate",
            ],
        ),
        "",
        "## Outputs",
        "",
        f"- Ranking metrics: `{manifest['outputs']['ranking_metrics']}`",
        f"- Monthly execution: `{manifest['outputs']['execution_monthly']}`",
        f"- Aggregate execution: `{manifest['outputs']['execution_aggregate']}`",
        f"- Materialized policy-net labels: `{manifest['outputs'].get('materialized_policy_net_labels', '')}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md


def run_ablation(
    *,
    experiment_id: str,
    source_run_id: str,
    strategy_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = spo._make_policy_replay_store(str(wf.DATA_ROOT), "perps")
    ranking_rows: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []
    materialized_label_frames: list[pd.DataFrame] = []
    fold_manifest: list[dict[str, Any]] = []

    for fold in wf._folds(experiment_id):
        eval_month = FOLD_MONTHS.get(fold.name, fold.name)
        run_root = wf.DATA_ROOT / "artifacts" / fold.run_id
        frame, split_info = vanilla._prepare_policy_frame(fold.run_id, strategy_id)
        stage_view = split_info["stage_view"]
        opt_mask, validation_mask, split = spo._policy_optimisation_validation_masks(frame, stage_view)
        all_paths = spo._fetch_policy_paths(frame, ds)
        frame, all_paths = spo._apply_delayed_entry_execution_model(
            frame,
            all_paths,
            data_root=str(wf.DATA_ROOT),
            market_mode="perps",
        )
        opt_idx = np.flatnonzero(opt_mask.to_numpy(dtype=bool))
        val_idx = np.flatnonzero(validation_mask.to_numpy(dtype=bool))
        train = frame.iloc[opt_idx].copy()
        valid = frame.iloc[val_idx].copy()
        train_paths = spo._path_take(all_paths, opt_idx)
        valid_paths = spo._path_take(all_paths, val_idx)

        combined = pd.concat([train, valid], axis=0)
        combined_paths = tuple(np.concatenate([tp, vp], axis=0) for tp, vp in zip(train_paths, valid_paths))
        vanilla_net, vanilla_reason = _independent_vanilla_net_returns(combined, combined_paths)
        fixed_net, fixed_reason = _fixed_tp2_sl1_net_returns(combined, combined_paths)
        combined_targets = _make_targets(combined, vanilla_net=vanilla_net, fixed_net=fixed_net)
        role = pd.Series("unknown", index=combined.index, dtype=object)
        role.loc[train.index] = "policy_optimisation"
        role.loc[valid.index] = "policy_validation"
        id_cols = [
            col
            for col in (
                "timestamp",
                "symbol",
                "side",
                "direction",
                "strategy_id",
                "barrier_pct",
                "y_bin",
                "y_outcome",
                "return",
                "calibrated_score",
                "oof_base_clf",
            )
            if col in combined.columns
        ]
        label_export = combined.loc[:, id_cols].copy()
        label_export.insert(0, "frame_index", combined.index.to_numpy())
        label_export.insert(0, "split_role", role.to_numpy())
        label_export.insert(0, "eval_month", eval_month)
        label_export.insert(0, "run_id", fold.run_id)
        label_export.insert(0, "fold", fold.name)
        label_export["u_policy_net"] = pd.to_numeric(vanilla_net, errors="coerce").to_numpy(dtype=np.float64)
        label_export["u_policy_net_soft_s10"] = pd.to_numeric(
            combined_targets["S10_vanilla_independent_net"]["target_soft"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        label_export["u_policy_net_hard_s10"] = pd.to_numeric(
            combined_targets["S10_vanilla_independent_net"]["target_hard"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        label_export["u_policy_exit_reason"] = vanilla_reason.astype(str).to_numpy()
        label_export["u_fixed_tp2_sl1_net"] = pd.to_numeric(fixed_net, errors="coerce").to_numpy(dtype=np.float64)
        label_export["u_fixed_tp2_sl1_exit_reason"] = fixed_reason.astype(str).to_numpy()
        materialized_label_frames.append(label_export.reset_index(drop=True))
        features = _feature_columns(run_root, frame)
        fold_manifest.append(
            {
                "fold": fold.name,
                "run_id": fold.run_id,
                "eval_month": eval_month,
                "optimisation_rows": int(len(train)),
                "validation_rows": int(len(valid)),
                "feature_count": int(len(features)),
                "split": split,
                "vanilla_independent_reason_counts": vanilla_reason.value_counts(dropna=False).to_dict(),
                "fixed_tp2_sl1_reason_counts": fixed_reason.value_counts(dropna=False).to_dict(),
            }
        )

        baseline_scores = {
            "current_meta": pd.to_numeric(valid.get("calibrated_score", valid.get("clf")), errors="coerce"),
            "current_base": pd.to_numeric(valid.get("oof_base_clf"), errors="coerce"),
        }
        for arm in LABEL_ARMS:
            target = combined_targets[arm.name]
            train_target = target.loc[train.index]
            valid_target = target.loc[valid.index]
            pred, model_diag = _fit_predict_lgbm(train, valid, features, train_target["target_soft"])
            selectors = {
                **baseline_scores,
                "ablation_lgbm": pred,
                "oracle_label": valid_target["target_soft"],
            }
            for selector, score in selectors.items():
                ranking_rows.extend(
                    _score_ranking(
                        eval_month=eval_month,
                        run_id=fold.run_id,
                        arm=arm.name,
                        selector=selector,
                        score=score,
                        target=valid_target,
                        train_rows=len(train),
                        valid_rows=len(valid),
                        model_diag=model_diag if selector == "ablation_lgbm" else None,
                    )
                )
                execution_rows.extend(
                    _score_execution(
                        eval_month=eval_month,
                        run_id=fold.run_id,
                        arm=arm.name,
                        selector=selector,
                        score=score,
                        validation=valid,
                        validation_paths=valid_paths,
                        target=valid_target,
                    )
                )

    ranking = pd.DataFrame(ranking_rows)
    execution_monthly = pd.DataFrame(execution_rows)
    execution_aggregate = _aggregate(execution_monthly)
    materialized_labels = (
        pd.concat(materialized_label_frames, ignore_index=True)
        if materialized_label_frames
        else pd.DataFrame()
    )

    paths = {
        "ranking_metrics": output_dir / "label_ranking_metrics.csv",
        "execution_monthly": output_dir / "label_execution_monthly.csv",
        "execution_aggregate": output_dir / "label_execution_aggregate.csv",
        "materialized_policy_net_labels": output_dir / "materialized_policy_net_labels.parquet",
        "manifest": output_dir / "manifest.json",
    }
    ranking.to_csv(paths["ranking_metrics"], index=False)
    execution_monthly.to_csv(paths["execution_monthly"], index=False)
    execution_aggregate.to_csv(paths["execution_aggregate"], index=False)
    materialized_labels.to_parquet(paths["materialized_policy_net_labels"], index=False)
    manifest = {
        "experiment_id": experiment_id,
        "source_run_id": source_run_id,
        "strategy_id": strategy_id,
        "output_dir": str(output_dir),
        "label_arms": [arm.__dict__ for arm in LABEL_ARMS],
        "top_fracs": list(TOP_FRACS),
        "round_trip_cost_assumption": ROUND_TRIP_COST,
        "model": {
            "type": "LGBMRegressor",
            "train_scope": "policy_optimisation_half",
            "validation_scope": "policy_validation_half",
            "random_seed": MODEL_RANDOM_SEED,
        },
        "folds": fold_manifest,
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    md = _write_markdown(
        output_dir=output_dir,
        execution_aggregate=execution_aggregate,
        execution_monthly=execution_monthly,
        ranking_metrics=ranking,
        manifest=manifest,
    )
    manifest["outputs"]["markdown"] = str(md)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default=os.environ.get("EPM_MONTHLY_WF_ID", DEFAULT_EXPERIMENT_ID))
    parser.add_argument("--source-run-id", default=os.environ.get("EPM_SOURCE_RUN_ID", DEFAULT_SOURCE_RUN_ID))
    parser.add_argument("--strategy-id", default="")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_id = str(args.experiment_id).strip()
    source_run_id = str(args.source_run_id).strip()
    strategy_id = str(args.strategy_id or wf._select_june_best_strategy(source_run_id)["strategy_id"]).strip()
    output_dir = (
        args.output_dir
        or wf.DATA_ROOT
        / "reports"
        / experiment_id
        / "label_execution_alignment_ablation"
    )
    manifest = run_ablation(
        experiment_id=experiment_id,
        source_run_id=source_run_id,
        strategy_id=strategy_id,
        output_dir=output_dir,
    )
    print(json.dumps(_json_safe(manifest["outputs"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
