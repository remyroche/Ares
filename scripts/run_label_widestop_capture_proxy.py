#!/usr/bin/env python3
"""Wide-stop fixed-capture label proxy before base/meta training.

This diagnostic tests whether the positive-but-high-MAE pockets seen in prior
label screens can become a separate bounded execution style. It does not run
production LightGBM, Optuna, or policy geometry. It fits cheap month-forward
tree proxies on prior months only, then evaluates fixed TP / wider SL capture
outcomes after costs.

The execution accounting is intentionally conservative for the available label
artifact: if both TP and SL are touched at any point in the horizon, the row is
counted as an SL because event ordering is unavailable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_feature_store_model_smoke import _fit_predict, _month_model_frame  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    ROUND_TRIP_COST,
    _effective_n,
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _rank_top_indices,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _sigmoid,
    _spearman,
)
from scripts.run_label_weighted_proxy_ablation import _effective_sample_size  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_widestop_capture_proxy_v1")
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_TOP_FRACS = (0.0025, 0.005, 0.010)
DEFAULT_SEEDS = (42, 7301, 999)


@dataclass(frozen=True)
class CaptureArm:
    name: str
    tp_r: float
    sl_r: float
    max_bars_to_mfe: float
    max_barrier: float
    trail_r: float = 0.50


CAPTURE_ARMS = (
    CaptureArm("C0_tp075_sl15_fast6_bar30", 0.75, 1.50, 6.0, 0.030),
    CaptureArm("C1_tp100_sl15_fast6_bar30", 1.00, 1.50, 6.0, 0.030),
    CaptureArm("C2_tp100_sl20_fast6_bar30", 1.00, 2.00, 6.0, 0.030),
    CaptureArm("C3_tp125_sl20_fast12_bar30", 1.25, 2.00, 12.0, 0.030),
    CaptureArm("C4_tp150_sl25_fast12_bar30", 1.50, 2.50, 12.0, 0.030),
    CaptureArm("C5_tp100_sl20_fast12_bar25", 1.00, 2.00, 12.0, 0.025),
)


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_int_csv(value: str | None, default: tuple[int, ...]) -> list[int]:
    if value is None or not str(value).strip():
        return list(default)
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _parse_float_csv(value: str | None, default: tuple[float, ...]) -> list[float]:
    if value is None or not str(value).strip():
        return list(default)
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _rank_pct(values: pd.Series) -> pd.Series:
    return _safe_numeric(values).rank(method="average", pct=True).fillna(0.0).clip(0.0, 1.0)


def _capture_outcome(metrics: pd.DataFrame, arm: CaptureArm) -> pd.DataFrame:
    barrier = _safe_numeric(metrics["barrier"]).abs().clip(lower=1e-8)
    mfe_norm = _safe_numeric(metrics["mfe_norm"]).fillna(0.0)
    mae_norm = _safe_numeric(metrics["mae_norm"]).fillna(999.0)
    bars_to_mfe = _safe_numeric(metrics["bars_to_mfe"]).fillna(999.0)
    ret_net = _safe_numeric(metrics["ret_net"]).fillna(-ROUND_TRIP_COST)

    barrier_ok = barrier <= float(arm.max_barrier)
    tp_reached_fast = (mfe_norm >= float(arm.tp_r)) & (bars_to_mfe <= float(arm.max_bars_to_mfe))
    sl_touched = mae_norm >= float(arm.sl_r)
    hard_hit = barrier_ok & tp_reached_fast & (~sl_touched)
    hard_stop = barrier_ok & sl_touched
    eligible = barrier_ok

    tp_abs = float(arm.tp_r) * barrier
    sl_abs = float(arm.sl_r) * barrier
    fallback_net = np.minimum(ret_net.to_numpy(dtype=np.float64), 0.0)
    capture_net = pd.Series(fallback_net, index=metrics.index, dtype=np.float64)
    capture_net.loc[hard_hit] = tp_abs.loc[hard_hit] - ROUND_TRIP_COST
    capture_net.loc[hard_stop] = -sl_abs.loc[hard_stop] - ROUND_TRIP_COST
    capture_net.loc[~eligible] = -ROUND_TRIP_COST

    soft = (
        pd.Series(_sigmoid((mfe_norm - float(arm.tp_r)) / 0.30), index=metrics.index)
        * pd.Series(_sigmoid((float(arm.sl_r) - mae_norm) / 0.35), index=metrics.index)
        * pd.Series(_sigmoid((float(arm.max_bars_to_mfe) - bars_to_mfe) / 3.0), index=metrics.index)
        * pd.Series(_sigmoid((float(arm.max_barrier) - barrier) / 0.006), index=metrics.index)
    ).clip(0.0, 1.0)
    return pd.DataFrame(
        {
            "target_soft": soft,
            "target_hard": hard_hit.astype(float),
            "capture_net": capture_net,
            "capture_hit": hard_hit.astype(float),
            "capture_stop": hard_stop.astype(float),
            "capture_eligible": eligible.astype(float),
            "tp_r": float(arm.tp_r),
            "sl_r": float(arm.sl_r),
            "effective_tp_abs": tp_abs,
            "effective_sl_abs": sl_abs,
            "mae_to_sl": (mae_norm / float(arm.sl_r)).replace([np.inf, -np.inf], np.nan),
            "mfe_to_tp": (mfe_norm / float(arm.tp_r)).replace([np.inf, -np.inf], np.nan),
        },
        index=metrics.index,
    )


def _weights_for_target(target: pd.DataFrame, *, max_weight: float, min_weight: float) -> pd.Series:
    hard = _safe_numeric(target["target_hard"]).fillna(0.0) > 0.5
    pos = int(hard.sum())
    neg = int((~hard).sum())
    balance = min(float(max_weight), float(neg / max(pos, 1)))
    weights = pd.Series(1.0, index=target.index, dtype=np.float64)
    weights.loc[hard] = balance
    weights = weights.clip(lower=float(min_weight), upper=float(max_weight))
    mean = float(weights.mean())
    if math.isfinite(mean) and mean > 0.0:
        weights = weights / mean
    return weights.astype(np.float32)


def _timestamp_top_indices(frame: pd.DataFrame, score: pd.Series, top_frac: float) -> np.ndarray:
    score_series = _safe_numeric(score).reset_index(drop=True)
    timestamps = pd.to_datetime(frame["__ts__"], errors="coerce").reset_index(drop=True)
    chosen: list[np.ndarray] = []
    for _, ids in pd.Series(np.arange(len(score_series)), index=score_series.index).groupby(timestamps, dropna=False):
        pos = ids.to_numpy(dtype=np.int64)
        valid_pos = pos[np.isfinite(score_series.iloc[pos].to_numpy(dtype=np.float64))]
        if len(valid_pos) == 0:
            continue
        k = max(1, int(math.ceil(float(top_frac) * len(valid_pos))))
        values = score_series.iloc[valid_pos].to_numpy(dtype=np.float64)
        order = np.argsort(-values, kind="mergesort")[:k]
        chosen.append(valid_pos[order].astype(np.int64, copy=False))
    if not chosen:
        return np.array([], dtype=np.int64)
    return np.concatenate(chosen).astype(np.int64, copy=False)


def _selection_indices(frame: pd.DataFrame, score: pd.Series, top_frac: float, selection_mode: str) -> np.ndarray:
    if selection_mode == "timestamp":
        return _timestamp_top_indices(frame, score, top_frac)
    return _rank_top_indices(score, top_frac)


def _selection_metrics(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    arm: str,
    period: str,
    top_frac: float,
    selection_mode: str,
) -> dict[str, Any]:
    idx = _selection_indices(frame, score, top_frac, selection_mode)
    selected_frame = frame.iloc[idx] if len(idx) else frame.iloc[:0]
    selected_metrics = metrics.iloc[idx] if len(idx) else metrics.iloc[:0]
    selected_target = target.iloc[idx] if len(idx) else target.iloc[:0]
    symbols = selected_frame.get("__symbol__", pd.Series(dtype=object))
    timestamps = selected_frame.get("__ts__", pd.Series(dtype="datetime64[ns]"))
    capture_net = _safe_numeric(selected_target.get("capture_net"))
    # Gross payoff is used only for base-layer precision weighting. It avoids
    # making candidate-source geometry search fail solely because deployment
    # costs belong later in meta/execution evaluation.
    if "round_trip_cost" in selected_target.columns:
        cost = _safe_numeric(selected_target.get("round_trip_cost")).fillna(float(ROUND_TRIP_COST))
    else:
        cost = pd.Series(float(ROUND_TRIP_COST), index=selected_target.index)
    gross_payoff = capture_net + cost
    gross_abs_weight = gross_payoff.abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    hit = _safe_numeric(selected_target.get("capture_hit")).fillna(0.0).clip(0.0, 1.0)
    stop = _safe_numeric(selected_target.get("capture_stop")).fillna(0.0).clip(0.0, 1.0)
    timeout = _safe_numeric(selected_target.get("capture_timeout")).fillna(0.0).clip(0.0, 1.0)
    gross_denom = float(gross_abs_weight.sum())
    gross_hit_value = float((hit * gross_payoff.clip(lower=0.0)).sum()) if len(hit) else float("nan")
    gross_stop_value = float((stop * gross_abs_weight).sum()) if len(stop) else float("nan")
    gross_timeout_value = float((timeout * gross_abs_weight).sum()) if len(timeout) else float("nan")
    ev_weighted_precision = gross_hit_value / gross_denom if gross_denom > 1e-12 else float("nan")
    selected_path_mae = _safe_numeric(selected_metrics.get("mae_norm"))
    selected_path_mfe = _safe_numeric(selected_metrics.get("mfe_norm"))
    first_touch_mae = _safe_numeric(selected_target.get("first_touch_mae_norm"))
    first_touch_mfe = _safe_numeric(selected_target.get("first_touch_mfe_norm"))
    first_touch_mae_to_sl = _safe_numeric(selected_target.get("mae_to_sl"))
    target_full_path_mae = _safe_numeric(selected_target.get("full_path_mae_norm"))
    target_full_path_mae_to_sl = _safe_numeric(selected_target.get("full_path_mae_to_sl"))
    mfe_before_mae_1r = _safe_numeric(selected_target.get("mfe_1r_before_mae_1r"))
    mae_before_mfe_1r = _safe_numeric(selected_target.get("mae_1r_before_mfe_1r"))
    max_adverse_before_mfe_1r = _safe_numeric(selected_target.get("max_adverse_before_mfe_1r"))
    underwater_bars_before_mfe_1r = _safe_numeric(selected_target.get("underwater_bars_before_mfe_1r"))
    underwater_fraction_before_mfe_1r = _safe_numeric(selected_target.get("underwater_fraction_before_mfe_1r"))
    area_underwater_before_mfe_1r = _safe_numeric(selected_target.get("area_underwater_before_mfe_1r"))
    return {
        "arm": arm,
        "selector": f"widestop_capture_proxy_{selection_mode}",
        "period": str(period),
        "top_frac": float(top_frac),
        "rows": int(len(frame)),
        "selected_rows": int(len(idx)),
        "capture_net_mean": _safe_mean(selected_target.get("capture_net")),
        "capture_gross_mean": _safe_mean(gross_payoff),
        "ev_weighted_first_touch_precision": ev_weighted_precision,
        "ev_weighted_clean_precision": ev_weighted_precision,
        "gross_hit_value_mean": gross_hit_value / max(1, len(selected_target)),
        "gross_stop_value_mean": gross_stop_value / max(1, len(selected_target)),
        "gross_timeout_value_mean": gross_timeout_value / max(1, len(selected_target)),
        "capture_net_q10": _safe_quantile(selected_target.get("capture_net"), 0.10),
        "capture_hit_rate": _safe_mean(selected_target.get("capture_hit")),
        "capture_stop_rate": _safe_mean(selected_target.get("capture_stop")),
        "capture_eligible_rate": _safe_mean(selected_target.get("capture_eligible")),
        "target_top_soft_mean": _safe_mean(selected_target.get("target_soft")),
        "target_top_hard_rate": _safe_mean(selected_target.get("target_hard")),
        "mean_u_policy_net": _safe_mean(selected_metrics.get("u_policy_net")),
        "mean_ret_net": _safe_mean(selected_metrics.get("ret_net")),
        "bad_mae_1r_rate": _safe_mean(selected_path_mae >= 1.0),
        "mean_mae_norm": _safe_mean(selected_path_mae),
        "p90_mae_norm": _safe_quantile(selected_path_mae, 0.90),
        "mean_mfe_norm": _safe_mean(selected_path_mfe),
        "p90_mfe_norm": _safe_quantile(selected_path_mfe, 0.90),
        "selected_path_bad_mae_1r_rate": _safe_mean(selected_path_mae >= 1.0),
        "selected_path_mean_mae_norm": _safe_mean(selected_path_mae),
        "selected_path_p90_mae_norm": _safe_quantile(selected_path_mae, 0.90),
        "selected_path_mean_mfe_norm": _safe_mean(selected_path_mfe),
        "selected_path_p90_mfe_norm": _safe_quantile(selected_path_mfe, 0.90),
        "first_touch_bad_mae_1r_rate": _safe_mean(first_touch_mae >= 1.0),
        "first_touch_mean_mae_norm": _safe_mean(first_touch_mae),
        "first_touch_p90_mae_norm": _safe_quantile(first_touch_mae, 0.90),
        "first_touch_mean_mfe_norm": _safe_mean(first_touch_mfe),
        "first_touch_p90_mfe_norm": _safe_quantile(first_touch_mfe, 0.90),
        "first_touch_mae_to_sl_mean": _safe_mean(first_touch_mae_to_sl),
        "first_touch_mae_to_sl_p90": _safe_quantile(first_touch_mae_to_sl, 0.90),
        "mfe_1r_before_mae_1r_rate": _safe_mean(mfe_before_mae_1r > 0.5),
        "mae_1r_before_mfe_1r_rate": _safe_mean(mae_before_mfe_1r > 0.5),
        "mean_max_adverse_before_mfe_1r": _safe_mean(max_adverse_before_mfe_1r),
        "p90_max_adverse_before_mfe_1r": _safe_quantile(max_adverse_before_mfe_1r, 0.90),
        "mean_underwater_bars_before_mfe_1r": _safe_mean(underwater_bars_before_mfe_1r),
        "p90_underwater_bars_before_mfe_1r": _safe_quantile(underwater_bars_before_mfe_1r, 0.90),
        "mean_underwater_fraction_before_mfe_1r": _safe_mean(underwater_fraction_before_mfe_1r),
        "mean_area_underwater_before_mfe_1r": _safe_mean(area_underwater_before_mfe_1r),
        "target_full_path_bad_mae_1r_rate": _safe_mean(target_full_path_mae >= 1.0),
        "target_full_path_mae_to_sl_p90": _safe_quantile(target_full_path_mae_to_sl, 0.90),
        "mean_bars_to_mfe": _safe_mean(selected_metrics.get("bars_to_mfe")),
        "p90_bars_to_mfe": _safe_quantile(selected_metrics.get("bars_to_mfe"), 0.90),
        "mean_barrier": _safe_mean(selected_metrics.get("barrier")),
        "p90_barrier": _safe_quantile(selected_metrics.get("barrier"), 0.90),
        "wide_barrier_25bps_rate": _safe_mean(selected_metrics.get("barrier") > 0.025),
        "timeout_rate": _safe_mean(selected_metrics.get("is_timeout").astype(float)) if len(selected_metrics) else float("nan"),
        "effective_tp_abs_mean": _safe_mean(selected_target.get("effective_tp_abs")),
        "effective_sl_abs_mean": _safe_mean(selected_target.get("effective_sl_abs")),
        "effective_sl_abs_p90": _safe_quantile(selected_target.get("effective_sl_abs"), 0.90),
        "mae_to_sl_mean": _safe_mean(selected_target.get("mae_to_sl")),
        "mae_to_sl_p90": _safe_quantile(selected_target.get("mae_to_sl"), 0.90),
        "mfe_to_tp_mean": _safe_mean(selected_target.get("mfe_to_tp")),
        "symbol_effective_n": _effective_n(symbols),
        "top_symbol_share": float(symbols.value_counts(normalize=True, dropna=False).iloc[0]) if len(symbols) else 0.0,
        "timestamp_effective_n": _effective_n(timestamps.astype(str)),
        "top_timestamp_share": float(timestamps.astype(str).value_counts(normalize=True, dropna=False).iloc[0])
        if len(timestamps)
        else 0.0,
    }


def _weekly_rows(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    arm: str,
    period: str,
    top_frac: float,
    selection_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    weeks = frame["__ts__"].dt.to_period("W-SUN").astype(str)
    for week, ids in pd.Series(np.arange(len(frame)), index=frame.index).groupby(weeks, dropna=False):
        pos = ids.to_numpy(dtype=np.int64)
        if len(pos) < 20:
            continue
        row = _selection_metrics(
            frame=frame.iloc[pos].reset_index(drop=True),
            metrics=metrics.iloc[pos].reset_index(drop=True),
            target=target.iloc[pos].reset_index(drop=True),
            score=score.iloc[pos].reset_index(drop=True),
            arm=arm,
            period=period,
            top_frac=top_frac,
            selection_mode=selection_mode,
        )
        row["week"] = str(week)
        row["week_selected_rows"] = int(row["selected_rows"])
        row["week_selected_share"] = float(row["selected_rows"] / len(pos)) if len(pos) else float("nan")
        rows.append(row)
    return rows


def _seed_average_predict(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    x_valid: pd.DataFrame,
    seeds: list[int],
) -> tuple[pd.Series, float, float]:
    preds = [
        _fit_predict(x_train=x_train, y_train=y_train, w_train=w_train, x_valid=x_valid, seed=seed)
        for seed in seeds
    ]
    matrix = np.vstack(preds)
    pred = np.mean(matrix, axis=0).astype(np.float32)
    std = np.std(matrix, axis=0).astype(np.float32) if len(preds) > 1 else np.zeros_like(pred)
    return pd.Series(pred), float(np.mean(std)), float(np.percentile(std, 90))


def _run_month(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    features: list[str],
    month: str,
    arms: list[CaptureArm],
    top_fracs: list[float],
    selection_modes: list[str],
    seeds: list[int],
    train_lookback_months: int | None,
    max_weight: float,
    min_weight: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    train_mask = month_period < month
    if train_lookback_months is not None and int(train_lookback_months) > 0:
        prior_months = sorted(month_period[train_mask].dropna().unique())
        train_mask = train_mask & month_period.isin(set(prior_months[-int(train_lookback_months) :]))
    valid_mask = month_period == month
    if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
        return [], [], [{"period": month, "skipped": True, "train_rows": int(train_mask.sum()), "valid_rows": int(valid_mask.sum())}]

    x_train, x_valid = _month_model_frame(frame, train_mask=train_mask, valid_mask=valid_mask, features=features)
    valid = frame.loc[valid_mask].copy().reset_index(drop=True)
    valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for arm in arms:
        train_target = _capture_outcome(metrics.loc[train_mask].copy(), arm)
        valid_target = _capture_outcome(valid_metrics, arm)
        weights = _weights_for_target(train_target, max_weight=max_weight, min_weight=min_weight)
        pred, seed_std_mean, seed_std_p90 = _seed_average_predict(
            x_train=x_train,
            y_train=train_target["target_soft"],
            w_train=weights,
            x_valid=x_valid,
            seeds=seeds,
        )
        pred = pred.reset_index(drop=True)
        score = _rank_pct(pred)
        diagnostics.append(
            {
                "period": str(month),
                "arm": arm.name,
                "tp_r": arm.tp_r,
                "sl_r": arm.sl_r,
                "max_bars_to_mfe": arm.max_bars_to_mfe,
                "max_barrier": arm.max_barrier,
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
                "train_hard_rate": _safe_mean(train_target["target_hard"]),
                "valid_hard_rate": _safe_mean(valid_target["target_hard"]),
                "train_capture_net_mean": _safe_mean(train_target["capture_net"]),
                "valid_capture_net_mean": _safe_mean(valid_target["capture_net"]),
                "weight_mean": _safe_mean(weights),
                "weight_p90": _safe_quantile(weights, 0.90),
                "weight_effective_n": _effective_sample_size(weights),
                "weight_effective_frac": _effective_sample_size(weights) / float(len(weights)) if len(weights) else float("nan"),
                "score_ic_capture_net": _spearman(score, valid_target["capture_net"]),
                "score_ic_hard": _spearman(score, valid_target["target_hard"]),
                "score_ic_u_policy_net": _spearman(score, valid_metrics["u_policy_net"]),
                "seed_std_mean": seed_std_mean,
                "seed_std_p90": seed_std_p90,
            }
        )
        for selection_mode in selection_modes:
            for top_frac in top_fracs:
                row = _selection_metrics(
                    frame=valid,
                    metrics=valid_metrics,
                    target=valid_target,
                    score=score,
                    arm=arm.name,
                    period=str(month),
                    top_frac=float(top_frac),
                    selection_mode=selection_mode,
                )
                row.update(
                    {
                        "selection_mode": selection_mode,
                        "tp_r": arm.tp_r,
                        "sl_r": arm.sl_r,
                        "max_bars_to_mfe": arm.max_bars_to_mfe,
                        "max_barrier": arm.max_barrier,
                        "score_ic_capture_net": _spearman(score, valid_target["capture_net"]),
                        "score_ic_hard": _spearman(score, valid_target["target_hard"]),
                    }
                )
                monthly_rows.append(row)
                for week_row in _weekly_rows(
                    frame=valid,
                    metrics=valid_metrics,
                    target=valid_target,
                    score=score,
                    arm=arm.name,
                    period=str(month),
                    top_frac=float(top_frac),
                    selection_mode=selection_mode,
                ):
                    week_row.update(
                        {
                            "selection_mode": selection_mode,
                            "tp_r": arm.tp_r,
                            "sl_r": arm.sl_r,
                            "max_bars_to_mfe": arm.max_bars_to_mfe,
                            "max_barrier": arm.max_barrier,
                            "score_ic_capture_net": _spearman(score, valid_target["capture_net"]),
                            "score_ic_hard": _spearman(score, valid_target["target_hard"]),
                        }
                    )
                    weekly_rows.append(week_row)
    return monthly_rows, weekly_rows, diagnostics


def _weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str) -> float:
    if value_col not in frame.columns or weight_col not in frame.columns:
        return float("nan")
    values = _safe_numeric(frame[value_col])
    weights = _safe_numeric(frame[weight_col]).fillna(0.0)
    mask = values.notna() & weights.gt(0.0)
    if not bool(mask.any()):
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def _summarize_month(prefix: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            f"{prefix}_months": 0,
            f"{prefix}_positive_months": 0,
            f"{prefix}_mean_capture_net": float("nan"),
            f"{prefix}_worst_capture_net": float("nan"),
            f"{prefix}_selected_rows": 0,
        }
    cap = _safe_numeric(frame["capture_net_mean"])
    return {
        f"{prefix}_months": int(frame["period"].astype(str).nunique()),
        f"{prefix}_positive_months": int(cap.gt(0.0).sum()),
        f"{prefix}_mean_capture_net": _safe_mean(cap),
        f"{prefix}_worst_capture_net": float(cap.min()) if len(cap.dropna()) else float("nan"),
        f"{prefix}_selected_rows": int(_safe_numeric(frame["selected_rows"]).sum()),
        f"{prefix}_hit_rate": _weighted_mean(frame, "capture_hit_rate", "selected_rows"),
        f"{prefix}_ev_weighted_first_touch_precision": _weighted_mean(
            frame,
            "ev_weighted_first_touch_precision"
            if "ev_weighted_first_touch_precision" in frame.columns
            else "ev_weighted_clean_precision",
            "selected_rows",
        ),
        f"{prefix}_ev_weighted_clean_precision": _weighted_mean(frame, "ev_weighted_clean_precision", "selected_rows"),
        f"{prefix}_capture_gross_mean": _weighted_mean(frame, "capture_gross_mean", "selected_rows"),
        f"{prefix}_gross_hit_value_mean": _weighted_mean(frame, "gross_hit_value_mean", "selected_rows"),
        f"{prefix}_gross_stop_value_mean": _weighted_mean(frame, "gross_stop_value_mean", "selected_rows"),
        f"{prefix}_stop_rate": _weighted_mean(frame, "capture_stop_rate", "selected_rows"),
        f"{prefix}_target_hard_rate": _weighted_mean(frame, "target_top_hard_rate", "selected_rows"),
        f"{prefix}_bad_mae_1r_rate": _weighted_mean(frame, "bad_mae_1r_rate", "selected_rows"),
        f"{prefix}_selected_path_bad_mae_1r_rate": _weighted_mean(
            frame, "selected_path_bad_mae_1r_rate", "selected_rows"
        ),
        f"{prefix}_selected_path_p90_mae_norm": _weighted_mean(frame, "selected_path_p90_mae_norm", "selected_rows"),
        f"{prefix}_first_touch_bad_mae_1r_rate": _weighted_mean(
            frame, "first_touch_bad_mae_1r_rate", "selected_rows"
        ),
        f"{prefix}_first_touch_p90_mae_norm": _weighted_mean(frame, "first_touch_p90_mae_norm", "selected_rows"),
        f"{prefix}_first_touch_mae_to_sl_p90": _weighted_mean(frame, "first_touch_mae_to_sl_p90", "selected_rows"),
        f"{prefix}_mfe_1r_before_mae_1r_rate": _weighted_mean(frame, "mfe_1r_before_mae_1r_rate", "selected_rows"),
        f"{prefix}_mae_1r_before_mfe_1r_rate": _weighted_mean(frame, "mae_1r_before_mfe_1r_rate", "selected_rows"),
        f"{prefix}_mean_max_adverse_before_mfe_1r": _weighted_mean(
            frame, "mean_max_adverse_before_mfe_1r", "selected_rows"
        ),
        f"{prefix}_p90_max_adverse_before_mfe_1r": _weighted_mean(
            frame, "p90_max_adverse_before_mfe_1r", "selected_rows"
        ),
        f"{prefix}_mean_underwater_bars_before_mfe_1r": _weighted_mean(
            frame, "mean_underwater_bars_before_mfe_1r", "selected_rows"
        ),
        f"{prefix}_p90_underwater_bars_before_mfe_1r": _weighted_mean(
            frame, "p90_underwater_bars_before_mfe_1r", "selected_rows"
        ),
        f"{prefix}_mean_underwater_fraction_before_mfe_1r": _weighted_mean(
            frame, "mean_underwater_fraction_before_mfe_1r", "selected_rows"
        ),
        f"{prefix}_mean_area_underwater_before_mfe_1r": _weighted_mean(
            frame, "mean_area_underwater_before_mfe_1r", "selected_rows"
        ),
        f"{prefix}_target_full_path_bad_mae_1r_rate": _weighted_mean(
            frame, "target_full_path_bad_mae_1r_rate", "selected_rows"
        ),
        f"{prefix}_target_full_path_mae_to_sl_p90": _weighted_mean(
            frame, "target_full_path_mae_to_sl_p90", "selected_rows"
        ),
        f"{prefix}_effective_sl_abs_p90": _weighted_mean(frame, "effective_sl_abs_p90", "selected_rows"),
        f"{prefix}_mae_to_sl_p90": _weighted_mean(frame, "mae_to_sl_p90", "selected_rows"),
        f"{prefix}_timeout_rate": _weighted_mean(frame, "timeout_rate", "selected_rows"),
        f"{prefix}_wide_25bps_rate": _weighted_mean(frame, "wide_barrier_25bps_rate", "selected_rows"),
        f"{prefix}_top_symbol_share": float(_safe_numeric(frame["top_symbol_share"]).max()),
    }


def _summarize_week(prefix: str, frame: pd.DataFrame, *, min_week_rows: int) -> dict[str, Any]:
    if frame.empty:
        return {
            f"{prefix}_weeks": 0,
            f"{prefix}_material_weeks": 0,
            f"{prefix}_material_positive_week_rate": float("nan"),
            f"{prefix}_q25_week_capture_net": float("nan"),
            f"{prefix}_worst_week_capture_net": float("nan"),
        }
    cap = _safe_numeric(frame["capture_net_mean"])
    week_rows = _safe_numeric(frame["week_selected_rows"]).fillna(0.0)
    material = week_rows >= int(min_week_rows)
    positive = cap > 0.0
    return {
        f"{prefix}_weeks": int(len(frame)),
        f"{prefix}_material_weeks": int(material.sum()),
        f"{prefix}_material_positive_week_rate": float((positive & material).sum() / material.sum())
        if int(material.sum())
        else float("nan"),
        f"{prefix}_q25_week_capture_net": _safe_quantile(cap, 0.25),
        f"{prefix}_worst_week_capture_net": float(cap.min()) if len(cap.dropna()) else float("nan"),
        f"{prefix}_weekly_stop_rate": _weighted_mean(frame, "capture_stop_rate", "week_selected_rows"),
        f"{prefix}_weekly_hit_rate": _weighted_mean(frame, "capture_hit_rate", "week_selected_rows"),
    }


def _fit_holdout_summary(
    *,
    monthly: pd.DataFrame,
    weekly: pd.DataFrame,
    fit_months: list[str],
    holdout_month: str,
    min_week_rows: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["arm", "selection_mode", "top_frac"]
    if "regime_family" in monthly.columns:
        group_cols.append("regime_family")
    for key, group in monthly.groupby(group_cols, observed=True, dropna=False):
        key_dict = dict(zip(group_cols, key))
        week_group = weekly[
            weekly["arm"].astype(str).eq(str(key_dict["arm"]))
            & weekly["selection_mode"].astype(str).eq(str(key_dict["selection_mode"]))
            & _safe_numeric(weekly["top_frac"]).eq(float(key_dict["top_frac"]))
        ].copy()
        fit_month = group[group["period"].astype(str).isin(fit_months)].copy()
        holdout_monthly = group[group["period"].astype(str).eq(str(holdout_month))].copy()
        fit_week = week_group[week_group["period"].astype(str).isin(fit_months)].copy()
        holdout_week = week_group[week_group["period"].astype(str).eq(str(holdout_month))].copy()
        if fit_month.empty or holdout_monthly.empty or fit_week.empty or holdout_week.empty:
            continue
        row: dict[str, Any] = dict(key_dict)
        for col in ("tp_r", "sl_r", "trail_r", "max_bars_to_mfe", "max_barrier"):
            row[col] = float(group[col].dropna().iloc[0]) if col in group and group[col].dropna().size else float("nan")
        row.update(_summarize_month("fit", fit_month))
        row.update(_summarize_month("holdout", holdout_monthly))
        row.update(_summarize_week("fit", fit_week, min_week_rows=min_week_rows))
        row.update(_summarize_week("holdout", holdout_week, min_week_rows=min_week_rows))
        fit_pos_week = row.get("fit_material_positive_week_rate", float("nan"))
        holdout_pos_week = row.get("holdout_material_positive_week_rate", float("nan"))
        fit_sign = (
            row["fit_months"] == len(fit_months)
            and row["fit_positive_months"] == len(fit_months)
            and row["fit_worst_capture_net"] > 0.0
            and row["fit_material_weeks"] >= 4
            and math.isfinite(fit_pos_week)
            and fit_pos_week >= 0.55
        )
        holdout_sign = (
            row["holdout_mean_capture_net"] > 0.0
            and row["holdout_material_weeks"] >= 2
            and math.isfinite(holdout_pos_week)
            and holdout_pos_week >= 0.50
        )
        fit_bounded = (
            fit_sign
            and row["fit_stop_rate"] <= 0.45
            and row["fit_effective_sl_abs_p90"] <= 0.060
            and row["fit_timeout_rate"] <= 0.25
            and row["fit_top_symbol_share"] <= 0.35
        )
        holdout_bounded = (
            holdout_sign
            and row["holdout_stop_rate"] <= 0.45
            and row["holdout_effective_sl_abs_p90"] <= 0.060
            and row["holdout_timeout_rate"] <= 0.25
            and row["holdout_top_symbol_share"] <= 0.35
        )
        fit_strict = (
            fit_sign
            and row["fit_stop_rate"] <= 0.25
            and row["fit_effective_sl_abs_p90"] <= 0.045
            and row["fit_timeout_rate"] <= 0.20
            and row["fit_top_symbol_share"] <= 0.25
        )
        holdout_strict = (
            holdout_sign
            and row["holdout_stop_rate"] <= 0.25
            and row["holdout_effective_sl_abs_p90"] <= 0.045
            and row["holdout_timeout_rate"] <= 0.20
            and row["holdout_top_symbol_share"] <= 0.25
        )
        row["fit_sign_pass"] = bool(fit_sign)
        row["holdout_sign_pass"] = bool(holdout_sign)
        row["fit_bounded_pass"] = bool(fit_bounded)
        row["holdout_bounded_standalone_pass"] = bool(holdout_bounded)
        row["holdout_bounded_pass"] = bool(fit_bounded and holdout_bounded)
        row["fit_strict_pass"] = bool(fit_strict)
        row["holdout_strict_standalone_pass"] = bool(holdout_strict)
        row["holdout_strict_pass"] = bool(fit_strict and holdout_strict)
        row["positive_dirty_holdout"] = bool(fit_sign and holdout_sign and not holdout_bounded)
        row["capture_proxy_score"] = float(
            (row["holdout_mean_capture_net"] if pd.notna(row["holdout_mean_capture_net"]) else 0.0)
            + 0.60 * (row["holdout_q25_week_capture_net"] if pd.notna(row["holdout_q25_week_capture_net"]) else 0.0)
            + 0.30 * (row["fit_worst_capture_net"] if pd.notna(row["fit_worst_capture_net"]) else 0.0)
            - 0.030 * (row["holdout_stop_rate"] if pd.notna(row["holdout_stop_rate"]) else 0.0)
            - 0.020 * (row["holdout_timeout_rate"] if pd.notna(row["holdout_timeout_rate"]) else 0.0)
            - 0.010 * (row["holdout_top_symbol_share"] if pd.notna(row["holdout_top_symbol_share"]) else 0.0)
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["holdout_strict_pass", "holdout_bounded_pass", "positive_dirty_holdout", "capture_proxy_score"],
        ascending=[False, False, False, False],
    )


def _format_table(frame: pd.DataFrame, cols: list[str], limit: int = 40) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[col for col in cols if col in frame.columns]].head(limit).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
    return view.to_markdown(index=False)


def _write_markdown(output_dir: Path, fit_holdout: pd.DataFrame, diagnostics: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "label_widestop_capture_proxy.md"
    cols = [
        "arm",
        "selection_mode",
        "top_frac",
        "capture_proxy_score",
        "fit_sign_pass",
        "fit_bounded_pass",
        "fit_strict_pass",
        "fit_mean_capture_net",
        "fit_worst_capture_net",
        "fit_material_positive_week_rate",
        "fit_hit_rate",
        "fit_stop_rate",
        "fit_effective_sl_abs_p90",
        "holdout_bounded_pass",
        "holdout_strict_pass",
        "holdout_mean_capture_net",
        "holdout_material_positive_week_rate",
        "holdout_q25_week_capture_net",
        "holdout_hit_rate",
        "holdout_stop_rate",
        "holdout_effective_sl_abs_p90",
        "holdout_timeout_rate",
    ]
    diag_cols = [
        "period",
        "arm",
        "train_hard_rate",
        "valid_hard_rate",
        "train_capture_net_mean",
        "valid_capture_net_mean",
        "score_ic_capture_net",
        "score_ic_hard",
        "score_ic_u_policy_net",
        "weight_effective_frac",
    ]
    strict = fit_holdout[fit_holdout["holdout_strict_pass"].eq(True)] if not fit_holdout.empty else fit_holdout
    bounded = fit_holdout[fit_holdout["holdout_bounded_pass"].eq(True)] if not fit_holdout.empty else fit_holdout
    positive_dirty = fit_holdout[fit_holdout["positive_dirty_holdout"].eq(True)] if not fit_holdout.empty else fit_holdout
    best = fit_holdout.sort_values("capture_proxy_score", ascending=False) if not fit_holdout.empty else fit_holdout
    lines = [
        "# Wide-Stop Capture Label Proxy",
        "",
        "Scope: proxy diagnostic only. No production base/meta training, Optuna, or policy geometry optimisation is run.",
        "",
        "Execution accounting is conservative for aggregate MFE/MAE labels: if TP and SL are both touched within the horizon, the row is counted as SL because event ordering is unavailable.",
        "",
        f"Labels: `{manifest['labels_path']}`",
        f"Feature dir: `{manifest['feature_dir']}`",
        f"Feature count: `{manifest['feature_count']}`",
        f"Months: `{','.join(manifest['months'])}`",
        f"Train lookback months: `{manifest['train_lookback_months']}`",
        f"Selection modes: `{','.join(manifest['selection_modes'])}`",
        "",
        "## Counts",
        "",
        f"- Rows: `{manifest['rows']}`",
        f"- Monthly rows: `{manifest['rows_monthly']}`",
        f"- Weekly rows: `{manifest['rows_weekly']}`",
        f"- Fit bounded pass: `{manifest['fit_bounded_pass_rows']}`",
        f"- Holdout bounded pass after fit: `{manifest['holdout_bounded_pass_rows']}`",
        f"- Fit strict pass: `{manifest['fit_strict_pass_rows']}`",
        f"- Holdout strict pass after fit: `{manifest['holdout_strict_pass_rows']}`",
        f"- Positive but bounded-failing holdout: `{manifest['positive_dirty_holdout_rows']}`",
        "",
        "## Strict Passes",
        "",
        _format_table(strict, cols),
        "",
        "## Bounded Passes",
        "",
        _format_table(bounded, cols),
        "",
        "## Positive But Bounded-Failing",
        "",
        _format_table(positive_dirty, cols),
        "",
        "## Best Rejected Rows",
        "",
        _format_table(best, cols),
        "",
        "## Diagnostics",
        "",
        _format_table(diagnostics.sort_values(["period", "arm"]), diag_cols, limit=80),
        "",
        "## Outputs",
        "",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Weekly: `{manifest['outputs']['weekly']}`",
        f"- Diagnostics: `{manifest['outputs']['diagnostics']}`",
        f"- Fit/Holdout: `{manifest['outputs']['fit_holdout']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_proxy(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    months: list[str],
    arm_names: list[str],
    top_fracs: list[float],
    selection_modes: list[str],
    seeds: list[int],
    train_lookback_months: int | None,
    max_weight: float,
    min_weight: float,
    min_week_rows: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_labels(labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_store_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    for col in feature_matrix.columns:
        frame[col] = feature_matrix[col].to_numpy(dtype=np.float32, copy=False)
    metrics = _path_metrics(frame)
    features = _feature_columns(frame)
    arms_by_name = {arm.name: arm for arm in CAPTURE_ARMS}
    arms = [arms_by_name[name] for name in arm_names] if arm_names else list(CAPTURE_ARMS)
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for month in months:
        rows, weeks, diagnostics = _run_month(
            frame=frame,
            metrics=metrics,
            features=features,
            month=str(month),
            arms=arms,
            top_fracs=top_fracs,
            selection_modes=selection_modes,
            seeds=seeds,
            train_lookback_months=train_lookback_months,
            max_weight=max_weight,
            min_weight=min_weight,
        )
        monthly_rows.extend(rows)
        weekly_rows.extend(weeks)
        diagnostic_rows.extend(diagnostics)
    monthly = pd.DataFrame(monthly_rows)
    weekly = pd.DataFrame(weekly_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    fit_holdout = _fit_holdout_summary(
        monthly=monthly,
        weekly=weekly,
        fit_months=["2026-04", "2026-05"],
        holdout_month="2026-06",
        min_week_rows=min_week_rows,
    )
    paths = {
        "monthly": output_dir / "label_widestop_capture_proxy_monthly.csv",
        "weekly": output_dir / "label_widestop_capture_proxy_weekly.csv",
        "diagnostics": output_dir / "label_widestop_capture_proxy_diagnostics.csv",
        "fit_holdout": output_dir / "label_widestop_capture_proxy_fit_holdout.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    fit_holdout.to_csv(paths["fit_holdout"], index=False)
    manifest = {
        "scope": "wide_stop_capture_proxy_not_full_policy_training",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "feature_store": feature_store_report,
        "feature_count": int(len(features)),
        "features": features,
        "months": [str(v) for v in months],
        "arms": [arm.__dict__ for arm in arms],
        "top_fracs": [float(v) for v in top_fracs],
        "selection_modes": [str(v) for v in selection_modes],
        "seeds": [int(v) for v in seeds],
        "train_lookback_months": int(train_lookback_months) if train_lookback_months is not None else None,
        "max_weight": float(max_weight),
        "min_weight": float(min_weight),
        "min_week_rows": int(min_week_rows),
        "rows_monthly": int(len(monthly)),
        "rows_weekly": int(len(weekly)),
        "fit_bounded_pass_rows": int(fit_holdout["fit_bounded_pass"].sum()) if not fit_holdout.empty else 0,
        "holdout_bounded_pass_rows": int(fit_holdout["holdout_bounded_pass"].sum()) if not fit_holdout.empty else 0,
        "fit_strict_pass_rows": int(fit_holdout["fit_strict_pass"].sum()) if not fit_holdout.empty else 0,
        "holdout_strict_pass_rows": int(fit_holdout["holdout_strict_pass"].sum()) if not fit_holdout.empty else 0,
        "positive_dirty_holdout_rows": int(fit_holdout["positive_dirty_holdout"].sum()) if not fit_holdout.empty else 0,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(output_dir, fit_holdout, diagnostics, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--arms", default=",".join(arm.name for arm in CAPTURE_ARMS))
    parser.add_argument("--top-fracs", default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--selection-modes", default="global,timestamp")
    parser.add_argument("--seeds", default=",".join(str(v) for v in DEFAULT_SEEDS))
    parser.add_argument("--train-lookback-months", type=int, default=None)
    parser.add_argument("--max-weight", type=float, default=12.0)
    parser.add_argument("--min-weight", type=float, default=0.10)
    parser.add_argument("--min-week-rows", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_proxy(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        arm_names=_parse_csv(args.arms, tuple(arm.name for arm in CAPTURE_ARMS)),
        top_fracs=_parse_float_csv(args.top_fracs, DEFAULT_TOP_FRACS),
        selection_modes=_parse_csv(args.selection_modes, ("global", "timestamp")),
        seeds=_parse_int_csv(args.seeds, DEFAULT_SEEDS),
        train_lookback_months=args.train_lookback_months,
        max_weight=float(args.max_weight),
        min_weight=float(args.min_weight),
        min_week_rows=int(args.min_week_rows),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
