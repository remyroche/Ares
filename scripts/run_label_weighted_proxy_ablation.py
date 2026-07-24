#!/usr/bin/env python3
"""No-training soft-label sample-weight proxy ablations.

This script approximates the soft-label/sample-weight plan without fitting a
model. It changes only the feature association estimator used to build an
out-of-time rank proxy: weighted rank correlation on prior months, scored on
the next month.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_economic_proxy_ablation import LABEL_ARMS, _label_targets
from scripts.run_label_quality_proxy_diagnostics import (
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    PROXY_TOP_K_FEATURES,
    TOP_FRACS,
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _rank_top_indices,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _selection_metrics,
    _sigmoid,
    _spearman,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_weighted_proxy_ablation_v1")
PROXY_METHODS = (
    "weighted_ic",
    "tail_recovery",
    "stable_tail_recovery",
)
WEIGHT_ARMS = (
    "W0_base",
    "W1_confidence_g2",
    "W2_boundary_top30",
    "W3_downside_mae",
    "W4_opportunity_miss",
    "W6_decisive_path",
    "W7_timestamp_balanced",
    "W8_combined_conservative",
    "W9_tail_utility",
    "W10_payoff_clean",
    "W11_tail_clean_utility",
    "W12_tail_timestamp_balanced",
    "W13_lowbarrier_timestamp",
    "W14_clean_dirty_contrast",
    "W15_symbol_timestamp_balanced",
    "W16_severe_adverse_contrast",
    "W_execres_clean_dirty",
    "W_execres_hpo_topk_v1",
    "W_side_target_strength_v1",
)
FIXED_ARTIFACT_LABEL_ARMS = (
    "S10_policy_net_replay",
    "FT_C0_fast6_policy_soft",
    "FT_C0_fast6_proxy_soft",
    "STAGE15_quiet_mid_clean_utility",
)


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _normalize_weights(
    weights: pd.Series,
    *,
    min_weight: float = 0.10,
    max_weight: float = 5.0,
) -> pd.Series:
    w = _safe_numeric(weights).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    w = w.clip(lower=min_weight, upper=max_weight)
    mean = float(w.mean()) if len(w) else 1.0
    if not math.isfinite(mean) or mean <= 0.0:
        return pd.Series(1.0, index=w.index)
    return (w / mean).clip(lower=min_weight, upper=max_weight)


def _effective_sample_size(weights: pd.Series) -> float:
    w = _safe_numeric(weights).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    denom = float(np.square(w).sum())
    numer = float(w.sum()) ** 2
    return numer / denom if denom > 0.0 else 0.0


def _timestamp_balance(frame: pd.DataFrame) -> pd.Series:
    counts = frame["__ts__"].map(frame["__ts__"].value_counts(dropna=False)).astype(float)
    return _normalize_weights(1.0 / counts.clip(lower=1.0), min_weight=0.10, max_weight=5.0)


def _period_difficulty(frame: pd.DataFrame, utility: pd.Series) -> pd.Series:
    weeks = frame["__ts__"].dt.to_period("W-SUN").astype(str)
    week_mean = utility.groupby(weeks).transform("mean")
    cutoff = float(week_mean.quantile(0.35)) if len(week_mean.dropna()) else float("nan")
    if not math.isfinite(cutoff):
        return pd.Series(1.0, index=frame.index)
    difficulty = 1.0 + 2.0 * (week_mean <= cutoff).astype(float)
    return _normalize_weights(difficulty, min_weight=0.25, max_weight=3.0)


def _weight_series(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    arm: str,
) -> pd.Series:
    index = frame.index
    target_soft = _safe_numeric(target["target_soft"]).reindex(index)
    u = _safe_numeric(metrics["u_policy_net"]).reindex(index).fillna(-0.02)
    mae_norm = _safe_numeric(metrics["mae_norm"]).reindex(index).fillna(0.0)
    mfe_norm = _safe_numeric(metrics["mfe_norm"]).reindex(index).fillna(0.0)
    bars_to_mfe = _safe_numeric(metrics["bars_to_mfe"]).reindex(index).fillna(24.0)
    barrier = _safe_numeric(metrics["barrier"]).reindex(index).fillna(0.0)
    timeout_float = _safe_numeric(metrics["is_timeout"].astype(float)).reindex(index).fillna(0.0)
    base = _safe_numeric(frame.get("__w__", pd.Series(1.0, index=index))).reindex(index).fillna(1.0)
    mfe_mae = (mfe_norm / mae_norm.clip(lower=0.25)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(upper=10.0)

    confidence = (2.0 * (target_soft - 0.5).abs()).clip(0.0, 1.0)
    boundary_rank = target_soft.rank(method="average", pct=True)
    boundary = np.exp(-np.square((boundary_rank - 0.70) / 0.12))
    downside = 1.0 + 2.5 * (mae_norm - 0.75).clip(lower=0.0, upper=2.0) / 2.0
    downside = downside + 0.75 * (u < 0.0).astype(float)
    opportunity_miss = 1.0 + 2.0 * ((mfe_norm >= 2.0) & (target_soft <= 0.55)).astype(float)
    decisive = 1.0 + 1.5 * (
        ((mfe_norm >= 1.0) & (bars_to_mfe <= 3.0)) | (mae_norm >= 1.0)
    ).astype(float)
    utility_rank = u.rank(method="average", pct=True).fillna(0.0).clip(0.0, 1.0)
    tail_utility = 0.50 + 4.00 * np.power(utility_rank, 4.0)
    path_clean_score = (
        pd.Series(_sigmoid((mfe_norm - 2.0) / 0.75), index=index)
        * pd.Series(_sigmoid((2.50 - mae_norm) / 0.75), index=index)
        * pd.Series(_sigmoid((10.0 - bars_to_mfe) / 5.0), index=index)
    ).clip(0.0, 1.0)
    payoff_clean = 1.0 + 3.0 * (
        (u > 0.010)
        & (mfe_norm >= 2.0)
        & (mae_norm <= 2.0)
        & (bars_to_mfe <= 6.0)
    ).astype(float)
    ts_balanced = _timestamp_balance(frame)
    symbol_counts = frame["__symbol__"].map(frame["__symbol__"].value_counts(dropna=False)).astype(float)
    symbol_balanced = _normalize_weights(
        1.0 / symbol_counts.clip(lower=1.0),
        min_weight=0.20,
        max_weight=4.0,
    )
    lowbarrier = pd.Series(_sigmoid((0.030 - barrier) / 0.008), index=index).clip(0.0, 1.0)
    clean_path = (
        pd.Series(_sigmoid((3.00 - mae_norm) / 1.00), index=index)
        * pd.Series(_sigmoid((12.0 - bars_to_mfe) / 6.0), index=index)
    ).clip(0.0, 1.0)
    clean_positive = (
        (u > 0.002)
        & (target_soft >= target_soft.quantile(0.70))
        & (mae_norm <= 0.95)
        & (barrier <= 0.030)
        & (bars_to_mfe <= 16.0)
        & (timeout_float <= 0.5)
    ).astype(float)
    dirty_negative = (
        (target_soft <= target_soft.quantile(0.55))
        & (
            (mae_norm >= 1.0)
            | (barrier > 0.035)
            | (timeout_float > 0.5)
        )
    ).astype(float)
    severe_clean_positive = (
        (u > 0.0005)
        & (target_soft >= target_soft.quantile(0.70))
        & (mae_norm <= 0.85)
        & (barrier <= 0.024)
        & (mfe_mae >= 1.35)
        & (timeout_float <= 0.5)
    ).astype(float)
    severe_dirty_negative = (
        (target_soft <= target_soft.quantile(0.60))
        & (
            (mae_norm >= 1.0)
            | (barrier > 0.026)
            | (timeout_float > 0.5)
            | ((mae_norm > 0.75) & (mfe_mae < 1.25))
        )
    ).astype(float)
    positive_utility = u > 0.0
    execres_clean = _safe_numeric(
        frame.get(
            "__econ_sideaware_execres_clean__",
            frame.get("__econ_side_resolution_clean__", pd.Series(np.nan, index=index)),
        )
    ).reindex(index)
    execres_dirty_positive = _safe_numeric(
        frame.get(
            "__econ_sideaware_execres_dirty_positive__",
            frame.get("__econ_side_resolution_dirty_positive__", pd.Series(np.nan, index=index)),
        )
    ).reindex(index)
    if execres_clean.notna().sum() < 10:
        execres_clean = ((positive_utility) & (mae_norm < 1.0) & (timeout_float <= 0.5)).astype(float)
    else:
        execres_clean = execres_clean.fillna(0.0).clip(0.0, 1.0)
    if execres_dirty_positive.notna().sum() < 10:
        execres_dirty_positive = (
            positive_utility
            & ((mae_norm >= 1.0) | (timeout_float > 0.5))
        ).astype(float)
    else:
        execres_dirty_positive = execres_dirty_positive.fillna(0.0).clip(0.0, 1.0)
    bad_mae_positive = (positive_utility & (mae_norm >= 1.0)).astype(float)
    timeout_positive = (positive_utility & (timeout_float > 0.5)).astype(float)
    side = _safe_numeric(metrics.get("side", pd.Series(1.0, index=index))).reindex(index).fillna(1.0)
    month_side_key = (
        frame["__ts__"].dt.to_period("M").astype(str)
        + "_"
        + np.where(side.to_numpy(dtype=np.float64) < 0.0, "short", "long")
    )
    month_side_counts = pd.Series(month_side_key, index=index).map(
        pd.Series(month_side_key, index=index).value_counts(dropna=False)
    ).astype(float)
    month_side_balance = _normalize_weights(
        1.0 / month_side_counts.clip(lower=1.0),
        min_weight=0.25,
        max_weight=3.0,
    )
    month_side_balance_hpo_topk_v1 = _normalize_weights(
        np.power(1.0 / month_side_counts.clip(lower=1.0), 0.1403950683025824),
        min_weight=0.25,
        max_weight=3.0,
    )
    side_key = np.where(side.to_numpy(dtype=np.float64) < 0.0, "short", "long")
    side_counts = pd.Series(side_key, index=index).map(
        pd.Series(side_key, index=index).value_counts(dropna=False)
    ).astype(float)
    side_balance_hpo_topk_v1 = _normalize_weights(
        np.power(1.0 / side_counts.clip(lower=1.0), 0.03485016730091967),
        min_weight=0.25,
        max_weight=3.0,
    )
    spread_proxy = _safe_numeric(
        frame.get(
            "median_spread_bps",
            frame.get("p75_spread_bps", pd.Series(np.nan, index=index)),
        )
    ).reindex(index)
    spread_cutoff = float(spread_proxy.quantile(0.75)) if int(spread_proxy.notna().sum()) else float("nan")
    high_spread_dirty = (
        execres_dirty_positive.astype(bool)
        & spread_proxy.ge(spread_cutoff).fillna(False)
        if math.isfinite(spread_cutoff)
        else pd.Series(False, index=index)
    ).astype(float)

    if arm == "W0_base":
        weights = base
    elif arm == "W1_confidence_g2":
        weights = 0.25 + np.square(confidence)
    elif arm == "W2_boundary_top30":
        weights = 0.50 + 2.50 * boundary
    elif arm == "W3_downside_mae":
        weights = downside
    elif arm == "W4_opportunity_miss":
        weights = opportunity_miss
    elif arm == "W6_decisive_path":
        weights = decisive
    elif arm == "W7_timestamp_balanced":
        weights = ts_balanced
    elif arm == "W8_combined_conservative":
        hard_period = _period_difficulty(frame, u)
        weights = ts_balanced * (0.50 + confidence) * downside.clip(upper=2.75) * hard_period.clip(upper=2.0)
    elif arm == "W9_tail_utility":
        weights = tail_utility
    elif arm == "W10_payoff_clean":
        weights = payoff_clean
    elif arm == "W11_tail_clean_utility":
        weights = tail_utility * (0.50 + 1.75 * path_clean_score)
    elif arm == "W12_tail_timestamp_balanced":
        weights = tail_utility * ts_balanced
    elif arm == "W13_lowbarrier_timestamp":
        weights = ts_balanced * (0.50 + 1.50 * lowbarrier) * (0.75 + 0.50 * clean_path)
    elif arm == "W14_clean_dirty_contrast":
        weights = ts_balanced * (0.75 + 2.50 * clean_positive + 1.50 * dirty_negative)
    elif arm == "W15_symbol_timestamp_balanced":
        weights = ts_balanced * symbol_balanced * (0.75 + 1.25 * clean_path) * (0.75 + 0.50 * lowbarrier)
    elif arm == "W16_severe_adverse_contrast":
        weights = ts_balanced * (0.60 + 3.00 * severe_clean_positive + 2.00 * severe_dirty_negative)
    elif arm == "W_execres_clean_dirty":
        weights = (
            1.00
            + 2.00 * execres_clean
            + 1.50 * execres_dirty_positive
            + 1.50 * bad_mae_positive
            + 0.75 * timeout_positive
            + 0.50 * high_spread_dirty
        ) * month_side_balance
    elif arm == "W_execres_hpo_topk_v1":
        hpo_tail = 1.00 + 0.7213380140918505 * np.power(
            utility_rank.clip(0.0, 1.0),
            4.531616510212273,
        )
        weights = (
            1.00
            + 1.6236203565420875 * execres_clean
            + 3.327500072434707 * execres_dirty_positive
            + 3.061978796339918 * bad_mae_positive
            + 0.8979877262955549 * timeout_positive
            + 0.23402796066365478 * high_spread_dirty
        ) * hpo_tail * month_side_balance_hpo_topk_v1 * side_balance_hpo_topk_v1
    elif arm == "W_side_target_strength_v1":
        from extreme_price_movements.base_side_target_contract import (
            build_promoted_side_weights,
        )

        return build_promoted_side_weights(frame, target)
    else:
        raise ValueError(f"Unknown weight arm: {arm}")
    max_weight = 5.124217733388137 if arm == "W_execres_hpo_topk_v1" else 5.0
    return _normalize_weights(pd.Series(weights, index=index), min_weight=0.10, max_weight=max_weight)


def _weighted_corr(x: pd.Series, y: pd.Series, weights: pd.Series) -> float:
    xs = _safe_numeric(x)
    ys = _safe_numeric(y)
    ws = _safe_numeric(weights)
    mask = xs.notna() & ys.notna() & ws.notna() & (ws > 0.0)
    if int(mask.sum()) < 10:
        return float("nan")
    xv = xs[mask].rank(method="average", pct=True).to_numpy(dtype=np.float64)
    yv = ys[mask].rank(method="average", pct=True).to_numpy(dtype=np.float64)
    wv = ws[mask].to_numpy(dtype=np.float64)
    wsum = float(wv.sum())
    if wsum <= 0.0:
        return float("nan")
    wv = wv / wsum
    x_mean = float(np.sum(wv * xv))
    y_mean = float(np.sum(wv * yv))
    x_dev = xv - x_mean
    y_dev = yv - y_mean
    cov = float(np.sum(wv * x_dev * y_dev))
    x_var = float(np.sum(wv * x_dev * x_dev))
    y_var = float(np.sum(wv * y_dev * y_dev))
    denom = math.sqrt(x_var * y_var)
    return cov / denom if denom > 0.0 else float("nan")


def _weighted_feature_ic(
    train: pd.DataFrame,
    features: list[str],
    target: pd.Series,
    weights: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        ic = _weighted_corr(train[feature], target, weights)
        if not math.isfinite(ic):
            continue
        rows.append({"feature": feature, "ic": ic, "abs_ic": abs(ic)})
    return pd.DataFrame(rows).sort_values("abs_ic", ascending=False) if rows else pd.DataFrame()


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    xs = _safe_numeric(values)
    ws = _safe_numeric(weights)
    mask = xs.notna() & ws.notna() & (ws > 0.0)
    if int(mask.sum()) < 3:
        return float("nan")
    xv = xs[mask].to_numpy(dtype=np.float64)
    wv = ws[mask].to_numpy(dtype=np.float64)
    denom = float(wv.sum())
    return float(np.sum(xv * wv) / denom) if denom > 0.0 else float("nan")


def _tail_mask(target: pd.Series, frac: float) -> pd.Series:
    mask = pd.Series(False, index=target.index)
    idx = _rank_top_indices(target, frac)
    if len(idx):
        mask.iloc[idx] = True
    return mask


def _weighted_tail_feature_scores(
    train: pd.DataFrame,
    features: list[str],
    target: pd.Series,
    weights: pd.Series,
    *,
    tail_frac: float,
) -> pd.DataFrame:
    target_ser = _safe_numeric(target)
    weights = _safe_numeric(weights).reindex(train.index).fillna(0.0)
    tail = _tail_mask(target_ser, tail_frac)
    rest = ~tail
    rows: list[dict[str, Any]] = []
    if int(tail.sum()) < 10 or int(rest.sum()) < 50:
        return pd.DataFrame()
    for feature in features:
        ranks = _safe_numeric(train[feature]).rank(method="average", pct=True)
        tail_mean = _weighted_mean(ranks[tail], weights[tail])
        rest_mean = _weighted_mean(ranks[rest], weights[rest])
        if not math.isfinite(tail_mean) or not math.isfinite(rest_mean):
            continue
        tail_gain = tail_mean - rest_mean
        label_ic = _weighted_corr(train[feature], target_ser, weights)
        if not math.isfinite(label_ic):
            label_ic = 0.0
        selection_score = 0.75 * abs(tail_gain) + 0.25 * abs(label_ic)
        rows.append(
            {
                "feature": feature,
                "ic": tail_gain,
                "abs_ic": abs(tail_gain),
                "tail_gain": tail_gain,
                "abs_tail_gain": abs(tail_gain),
                "weighted_label_ic": label_ic,
                "selection_score": selection_score,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("selection_score", ascending=False)


def _stable_tail_feature_scores(
    train: pd.DataFrame,
    features: list[str],
    target: pd.Series,
    weights: pd.Series,
    *,
    tail_frac: float,
    min_features: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    full = _weighted_tail_feature_scores(
        train,
        features,
        target,
        weights,
        tail_frac=tail_frac,
    )
    if full.empty:
        return full, {"stability_fallback": "empty_full_scores"}

    ts = pd.to_datetime(train["__ts__"], errors="coerce")
    finite_ts = ts.dropna()
    if finite_ts.empty:
        return full, {"stability_fallback": "missing_timestamps"}
    split_ts = finite_ts.quantile(0.50)
    early_mask = ts <= split_ts
    late_mask = ts > split_ts
    if int(early_mask.sum()) < 200 or int(late_mask.sum()) < 200:
        return full, {"stability_fallback": "insufficient_split_rows"}

    early = _weighted_tail_feature_scores(
        train.loc[early_mask].copy(),
        features,
        target.loc[early_mask],
        weights.loc[early_mask],
        tail_frac=tail_frac,
    )
    late = _weighted_tail_feature_scores(
        train.loc[late_mask].copy(),
        features,
        target.loc[late_mask],
        weights.loc[late_mask],
        tail_frac=tail_frac,
    )
    if early.empty or late.empty:
        return full, {"stability_fallback": "empty_split_scores"}

    merged = full.merge(
        early[["feature", "tail_gain", "weighted_label_ic"]].rename(
            columns={
                "tail_gain": "early_tail_gain",
                "weighted_label_ic": "early_weighted_label_ic",
            }
        ),
        on="feature",
        how="left",
    ).merge(
        late[["feature", "tail_gain", "weighted_label_ic"]].rename(
            columns={
                "tail_gain": "late_tail_gain",
                "weighted_label_ic": "late_weighted_label_ic",
            }
        ),
        on="feature",
        how="left",
    )
    tail_sign = np.sign(merged["tail_gain"].to_numpy(dtype=np.float64))
    early_sign = np.sign(merged["early_tail_gain"].fillna(0.0).to_numpy(dtype=np.float64))
    late_sign = np.sign(merged["late_tail_gain"].fillna(0.0).to_numpy(dtype=np.float64))
    label_sign = np.sign(merged["weighted_label_ic"].to_numpy(dtype=np.float64))
    early_label_sign = np.sign(merged["early_weighted_label_ic"].fillna(0.0).to_numpy(dtype=np.float64))
    late_label_sign = np.sign(merged["late_weighted_label_ic"].fillna(0.0).to_numpy(dtype=np.float64))
    merged["tail_sign_stable"] = (tail_sign != 0.0) & (tail_sign == early_sign) & (tail_sign == late_sign)
    merged["label_ic_sign_stable"] = (
        (label_sign == 0.0)
        | ((label_sign == early_label_sign) & (label_sign == late_label_sign))
    )
    merged["split_min_abs_tail_gain"] = np.minimum(
        merged["early_tail_gain"].abs(),
        merged["late_tail_gain"].abs(),
    )
    stable = merged[merged["tail_sign_stable"] & merged["label_ic_sign_stable"]].copy()
    if len(stable) < min_features:
        return full, {
            "stability_fallback": "too_few_stable_features",
            "stable_feature_count": int(len(stable)),
        }
    stable["selection_score"] = (
        0.70 * stable["split_min_abs_tail_gain"].fillna(0.0)
        + 0.20 * stable["abs_tail_gain"].fillna(0.0)
        + 0.10 * stable["weighted_label_ic"].abs().fillna(0.0)
    )
    return stable.sort_values("selection_score", ascending=False), {
        "stability_fallback": "",
        "stable_feature_count": int(len(stable)),
    }


def _signed_rank_score(valid: pd.DataFrame, chosen: pd.DataFrame) -> pd.Series:
    parts: list[pd.Series] = []
    for _, row in chosen.iterrows():
        feature = str(row["feature"])
        sign_source = "ic" if "ic" in row else "tail_gain"
        sign = 1.0 if float(row[sign_source]) >= 0.0 else -1.0
        ranks = _safe_numeric(valid[feature]).rank(method="average", pct=True)
        if sign < 0.0:
            ranks = 1.0 - ranks
        parts.append(ranks.fillna(0.5))
    return pd.concat(parts, axis=1).mean(axis=1) if parts else pd.Series(np.nan, index=valid.index)


def _weighted_proxy_score(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    target: pd.Series,
    weights: pd.Series,
    *,
    method: str = "weighted_ic",
    tail_frac: float = 0.01,
) -> tuple[pd.Series, dict[str, Any]]:
    if method == "weighted_ic":
        scores = _weighted_feature_ic(train, features, target, weights)
        diag_extra: dict[str, Any] = {}
    elif method == "tail_recovery":
        scores = _weighted_tail_feature_scores(
            train,
            features,
            target,
            weights,
            tail_frac=tail_frac,
        )
        diag_extra = {"tail_recovery_frac": float(tail_frac)}
    elif method == "stable_tail_recovery":
        scores, diag_extra = _stable_tail_feature_scores(
            train,
            features,
            target,
            weights,
            tail_frac=tail_frac,
        )
        diag_extra["tail_recovery_frac"] = float(tail_frac)
    else:
        raise ValueError(f"Unknown proxy method: {method}")

    if scores.empty:
        return pd.Series(np.nan, index=valid.index), {"proxy_features": []}
    sort_col = "selection_score" if "selection_score" in scores.columns else "abs_ic"
    chosen = scores.sort_values(sort_col, ascending=False).head(PROXY_TOP_K_FEATURES).copy()
    score = _signed_rank_score(valid, chosen)
    diag = {
        "proxy_method": method,
        "proxy_features": chosen["feature"].astype(str).tolist(),
        "proxy_top_abs_ic": float(chosen["abs_ic"].iloc[0]) if len(chosen) else float("nan"),
        "proxy_mean_top_abs_ic": float(chosen["abs_ic"].mean()) if len(chosen) else float("nan"),
    }
    if "selection_score" in chosen.columns:
        diag["proxy_mean_selection_score"] = float(chosen["selection_score"].mean()) if len(chosen) else float("nan")
    if "tail_gain" in chosen.columns:
        diag["proxy_mean_tail_gain_abs"] = float(chosen["tail_gain"].abs().mean()) if len(chosen) else float("nan")
    diag.update(diag_extra)
    return score.reindex(valid.index), diag


def _tail_recovery_metrics(score: pd.Series, target: pd.Series, frac: float) -> dict[str, Any]:
    score_ser = _safe_numeric(score)
    target_ser = _safe_numeric(target)
    oracle_idx = _rank_top_indices(target_ser, frac)
    proxy_idx = _rank_top_indices(score_ser, frac)
    oracle = set(int(i) for i in oracle_idx)
    proxy = set(int(i) for i in proxy_idx)
    recovered = oracle & proxy
    false_positive = proxy - oracle
    missed = oracle - proxy
    recovery_rate = float(len(recovered) / len(oracle)) if oracle else float("nan")
    precision = float(len(recovered) / len(proxy)) if proxy else float("nan")
    return {
        "target_oracle_rows": int(len(oracle)),
        "target_oracle_recovered": int(len(recovered)),
        "target_oracle_recovery_rate": recovery_rate,
        "target_oracle_proxy_precision": precision,
        "target_oracle_missed": int(len(missed)),
        "target_false_positive_rows": int(len(false_positive)),
        "target_oracle_mean_soft": _safe_mean(target_ser.iloc[list(oracle)]) if oracle else float("nan"),
        "target_false_positive_mean_soft": _safe_mean(target_ser.iloc[list(false_positive)])
        if false_positive
        else float("nan"),
    }


def _fixed_artifact_targets(frame: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    u = _safe_numeric(metrics["u_policy_net"]).fillna(-0.02)
    targets: dict[str, pd.DataFrame] = {
        "S10_policy_net_replay": pd.DataFrame(
            {
                "target_soft": pd.Series(_sigmoid(u / 0.004), index=frame.index).clip(0.0, 1.0),
                "target_hard": (u > 0.0).fillna(False).astype(float),
            },
            index=frame.index,
        )
    }
    if "__first_touch_policy_soft__" in frame.columns:
        soft = _safe_numeric(frame["__first_touch_policy_soft__"]).clip(0.0, 1.0)
        targets["FT_C0_fast6_policy_soft"] = pd.DataFrame(
            {
                "target_soft": soft,
                "target_hard": (u > 0.0).fillna(False).astype(float),
            },
            index=frame.index,
        )
    if "__first_touch_target_soft__" in frame.columns:
        soft = _safe_numeric(frame["__first_touch_target_soft__"]).clip(0.0, 1.0)
        hard_source = (
            _safe_numeric(frame["__first_touch_hit__"])
            if "__first_touch_hit__" in frame.columns
            else (u > 0.0).astype(float)
        )
        targets["FT_C0_fast6_proxy_soft"] = pd.DataFrame(
            {
                "target_soft": soft,
                "target_hard": hard_source.fillna(0.0).clip(0.0, 1.0),
            },
            index=frame.index,
        )
    if "__stage15_target_soft__" in frame.columns:
        soft = _safe_numeric(frame["__stage15_target_soft__"]).clip(0.0, 1.0)
        if "__stage15_target_hard__" in frame.columns:
            hard_source = _safe_numeric(frame["__stage15_target_hard__"])
        else:
            hard_source = (soft >= 0.50).astype(float)
        targets["STAGE15_quiet_mid_clean_utility"] = pd.DataFrame(
            {
                "target_soft": soft.fillna(0.0).astype(np.float32),
                "target_hard": hard_source.fillna(0.0).clip(0.0, 1.0).astype(np.float32),
            },
            index=frame.index,
        )
    return targets


def _baseline_row(valid_metrics: pd.DataFrame) -> dict[str, float]:
    return {
        "period_baseline_mean_u": _safe_mean(valid_metrics["u_policy_net"]),
        "period_baseline_hit_u": _safe_mean(valid_metrics["u_policy_net"] > 0.0),
        "period_baseline_q10_u": _safe_quantile(valid_metrics["u_policy_net"], 0.10),
    }


def _add_delta_fields(row: dict[str, Any], baseline: dict[str, float]) -> None:
    mean_u = float(row["mean_u"])
    hit_u = float(row["hit_u"])
    q10_u = float(row["q10_u"])
    row.update(baseline)
    row["delta_mean_u_vs_period"] = (
        mean_u - baseline["period_baseline_mean_u"]
        if math.isfinite(mean_u) and math.isfinite(baseline["period_baseline_mean_u"])
        else float("nan")
    )
    row["delta_hit_u_vs_period"] = (
        hit_u - baseline["period_baseline_hit_u"]
        if math.isfinite(hit_u) and math.isfinite(baseline["period_baseline_hit_u"])
        else float("nan")
    )
    row["delta_q10_u_vs_period"] = (
        q10_u - baseline["period_baseline_q10_u"]
        if math.isfinite(q10_u) and math.isfinite(baseline["period_baseline_q10_u"])
        else float("nan")
    )


def _run_month(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    features: list[str],
    month: str,
    label_arms: list[str],
    weight_arms: list[str],
    proxy_methods: list[str],
    tail_recovery_frac: float,
) -> list[dict[str, Any]]:
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    train_mask = month_period < month
    valid_mask = month_period == month
    if int(train_mask.sum()) < 100 or int(valid_mask.sum()) < 50:
        return []

    train = frame.loc[train_mask].copy()
    valid = frame.loc[valid_mask].copy().reset_index(drop=True)
    train_metrics = metrics.loc[train_mask].copy()
    valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
    baseline = _baseline_row(valid_metrics)
    rows: list[dict[str, Any]] = []

    for label_arm in label_arms:
        target_train = targets[label_arm].loc[train_mask].copy()
        target_valid = targets[label_arm].loc[valid_mask].copy().reset_index(drop=True)
        for weight_arm in weight_arms:
            weights = _weight_series(
                frame=train,
                metrics=train_metrics,
                target=target_train,
                arm=weight_arm,
            )
            for proxy_method in proxy_methods:
                score, diag = _weighted_proxy_score(
                    train,
                    frame.loc[valid_mask].copy(),
                    features,
                    target_train["target_soft"],
                    weights,
                    method=proxy_method,
                    tail_frac=tail_recovery_frac,
                )
                score = score.reset_index(drop=True)
                for top_frac in TOP_FRACS:
                    row = _selection_metrics(
                        frame=valid,
                        metrics=valid_metrics,
                        target=target_valid,
                        score=score,
                        arm=f"{label_arm}::{weight_arm}::{proxy_method}",
                        selector=f"{proxy_method}_proxy_oos",
                        period=month,
                        top_frac=top_frac,
                    )
                    _add_delta_fields(row, baseline)
                    row.update(
                        {
                            "label_arm": label_arm,
                            "weight_arm": weight_arm,
                            "proxy_method": proxy_method,
                            "score_ic_u": _spearman(score, valid_metrics["u_policy_net"]),
                            "score_ic_label": _spearman(score, target_valid["target_soft"]),
                            "weight_mean": _safe_mean(weights),
                            "weight_p90": _safe_quantile(weights, 0.90),
                            "weight_p99": _safe_quantile(weights, 0.99),
                            "weight_effective_n": _effective_sample_size(weights),
                            "weight_effective_frac": _effective_sample_size(weights) / float(len(weights))
                            if len(weights)
                            else float("nan"),
                            "proxy_features": ",".join(diag.get("proxy_features", [])),
                            "proxy_top_abs_ic": diag.get("proxy_top_abs_ic"),
                            "proxy_mean_top_abs_ic": diag.get("proxy_mean_top_abs_ic"),
                            "proxy_mean_selection_score": diag.get("proxy_mean_selection_score"),
                            "proxy_mean_tail_gain_abs": diag.get("proxy_mean_tail_gain_abs"),
                            "stability_fallback": diag.get("stability_fallback", ""),
                            "stable_feature_count": diag.get("stable_feature_count", float("nan")),
                            **_tail_recovery_metrics(score, target_valid["target_soft"], top_frac),
                        }
                    )
                    rows.append(row)
    return rows


def _aggregate(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    rows: list[dict[str, Any]] = []
    groups = monthly.groupby(
        ["selector", "arm", "label_arm", "weight_arm", "proxy_method", "top_frac"],
        dropna=False,
        observed=True,
    )
    for key, group in groups:
        selector, arm, label_arm, weight_arm, proxy_method, top_frac = key
        mean_u = pd.to_numeric(group["mean_u"], errors="coerce")
        selected_rows = pd.to_numeric(group["selected_rows"], errors="coerce")
        rows.append(
            {
                "selector": selector,
                "arm": arm,
                "label_arm": label_arm,
                "weight_arm": weight_arm,
                "proxy_method": proxy_method,
                "top_frac": float(top_frac),
                "months": int(group["period"].nunique()),
                "positive_months": int((mean_u > 0.0).sum()),
                "mean_u": _safe_mean(mean_u),
                "worst_month_mean_u": _safe_quantile(mean_u, 0.0),
                "hit_u": _safe_mean(group["hit_u"]),
                "q10_u": _safe_mean(group["q10_u"]),
                "delta_mean_u_vs_period": _safe_mean(group["delta_mean_u_vs_period"]),
                "delta_hit_u_vs_period": _safe_mean(group["delta_hit_u_vs_period"]),
                "delta_q10_u_vs_period": _safe_mean(group["delta_q10_u_vs_period"]),
                "score_ic_u": _safe_mean(group["score_ic_u"]),
                "score_ic_label": _safe_mean(group["score_ic_label"]),
                "bad_mae_1r_rate": _safe_mean(group["bad_mae_1r_rate"]),
                "wide_barrier_25bps_rate": _safe_mean(group["wide_barrier_25bps_rate"]),
                "top_symbol_share": _safe_mean(group["top_symbol_share"]),
                "mean_selected_rows": _safe_mean(selected_rows),
                "min_selected_rows": int(selected_rows.min()) if len(selected_rows.dropna()) else 0,
                "target_oracle_recovery_rate": _safe_mean(group["target_oracle_recovery_rate"]),
                "target_oracle_proxy_precision": _safe_mean(group["target_oracle_proxy_precision"]),
                "target_oracle_recovered": int(
                    pd.to_numeric(group["target_oracle_recovered"], errors="coerce").fillna(0).sum()
                ),
                "target_oracle_rows": int(
                    pd.to_numeric(group["target_oracle_rows"], errors="coerce").fillna(0).sum()
                ),
                "weight_effective_frac": _safe_mean(group["weight_effective_frac"]),
                "weight_p99": _safe_mean(group["weight_p99"]),
                "proxy_mean_selection_score": _safe_mean(group["proxy_mean_selection_score"]),
                "proxy_mean_tail_gain_abs": _safe_mean(group["proxy_mean_tail_gain_abs"]),
                "stable_feature_count": _safe_mean(group["stable_feature_count"]),
                "stability_fallback": str(group["stability_fallback"].dropna().iloc[0])
                if group["stability_fallback"].dropna().size
                else "",
                "proxy_features": str(group["proxy_features"].dropna().iloc[0])
                if group["proxy_features"].dropna().size
                else "",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["top_frac", "mean_u", "worst_month_mean_u"],
        ascending=[True, False, False],
    )


def _write_markdown(output_dir: Path, aggregate: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "label_weighted_proxy_ablation.md"

    def table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[c for c in cols if c in frame.columns]].copy()
        if limit is not None:
            view = view.head(limit)
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    cols = [
        "proxy_method",
        "arm",
        "label_arm",
        "weight_arm",
        "months",
        "positive_months",
        "mean_u",
        "worst_month_mean_u",
        "hit_u",
        "q10_u",
        "delta_mean_u_vs_period",
        "score_ic_u",
        "score_ic_label",
        "bad_mae_1r_rate",
        "wide_barrier_25bps_rate",
        "target_oracle_recovery_rate",
        "target_oracle_proxy_precision",
        "mean_selected_rows",
        "min_selected_rows",
        "weight_effective_frac",
        "top_symbol_share",
    ]
    lines = [
        "# Label Weighted Proxy Ablation",
        "",
        "Scope: no model training. Weight arms only affect prior-month weighted feature-IC feature selection.",
        "",
    ]
    for frac in TOP_FRACS:
        subset = aggregate[aggregate["top_frac"].eq(frac)].sort_values(
            ["mean_u", "worst_month_mean_u"],
            ascending=[False, False],
        )
        lines.extend([f"## Top {frac:.0%}", "", table(subset, cols, limit=25), ""])
    lines.extend(
        [
            "## Outputs",
            "",
            f"- Monthly: `{manifest['outputs']['monthly']}`",
            f"- Aggregate: `{manifest['outputs']['aggregate']}`",
            f"- Manifest: `{manifest['outputs']['manifest']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_ablation(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    label_arms: list[str],
    weight_arms: list[str],
    proxy_methods: list[str],
    tail_recovery_frac: float,
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
    targets = _label_targets(frame, metrics)
    targets.update(_fixed_artifact_targets(frame, metrics))
    if not label_arms:
        label_arms = list(LABEL_ARMS)
    missing_labels = sorted(set(label_arms) - set(targets))
    missing_weights = sorted(set(weight_arms) - set(WEIGHT_ARMS))
    missing_methods = sorted(set(proxy_methods) - set(PROXY_METHODS))
    if missing_labels:
        raise ValueError(f"Unknown label arms: {missing_labels}")
    if missing_weights:
        raise ValueError(f"Unknown weight arms: {missing_weights}")
    if missing_methods:
        raise ValueError(f"Unknown proxy methods: {missing_methods}")
    months = sorted(frame["__ts__"].dt.to_period("M").dropna().astype(str).unique())

    monthly_rows: list[dict[str, Any]] = []
    for month in months[1:]:
        monthly_rows.extend(
            _run_month(
                frame=frame,
                metrics=metrics,
                targets=targets,
                features=features,
                month=month,
                label_arms=label_arms,
                weight_arms=weight_arms,
                proxy_methods=proxy_methods,
                tail_recovery_frac=tail_recovery_frac,
            )
        )
    monthly = pd.DataFrame(monthly_rows)
    aggregate = _aggregate(monthly)

    paths = {
        "monthly": output_dir / "label_weighted_proxy_monthly.csv",
        "aggregate": output_dir / "label_weighted_proxy_aggregate.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)

    manifest = {
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_count": int(len(features)),
        "features": features,
        "label_arms": list(label_arms),
        "weight_arms": list(weight_arms),
        "proxy_methods": list(proxy_methods),
        "tail_recovery_frac": float(tail_recovery_frac),
        "proxy_top_k_features": int(PROXY_TOP_K_FEATURES),
        "top_fracs": list(TOP_FRACS),
        "feature_store": feature_store_report,
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_markdown(output_dir, aggregate, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    lowered = str(value).strip().lower()
    if lowered == "all":
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--label-arms", default="all", help="Comma-separated label arms, or 'all'.")
    parser.add_argument(
        "--weight-arms",
        default=",".join(WEIGHT_ARMS),
        help="Comma-separated weight arms.",
    )
    parser.add_argument(
        "--proxy-methods",
        default="weighted_ic",
        help=f"Comma-separated proxy methods from {PROXY_METHODS}.",
    )
    parser.add_argument("--tail-recovery-frac", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_ablation(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        label_arms=_parse_csv(args.label_arms, LABEL_ARMS),
        weight_arms=_parse_csv(args.weight_arms, WEIGHT_ARMS),
        proxy_methods=_parse_csv(args.proxy_methods, ("weighted_ic",)),
        tail_recovery_frac=float(args.tail_recovery_frac),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
