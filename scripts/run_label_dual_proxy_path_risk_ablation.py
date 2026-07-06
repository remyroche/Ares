#!/usr/bin/env python3
"""Dual-proxy label/path-risk ablation before base/meta training.

This diagnostic tests whether causal feature proxies can recover rows that are
both profitable and economically clean. It fits only simple prior-month feature
rank proxies, then evaluates monthly/weekly OOS selections.
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

from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    _effective_n,
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _selection_metrics,
    _sigmoid,
    _spearman,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_dual_proxy_path_risk_ablation_v1")
DEFAULT_TOP_FRACS = (0.005, 0.010, 0.030, 0.050)
DEFAULT_FIT_MONTHS = ("2026-04", "2026-05")
DEFAULT_HOLDOUT_MONTH = "2026-06"


@dataclass(frozen=True)
class ScoreSpec:
    name: str
    utility_weight: float
    clean_weight: float
    bad_mae_weight: float
    wide_weight: float
    timeout_weight: float
    slow_weight: float
    max_bad_mae_proxy: float | None = None
    max_wide_proxy: float | None = None
    max_timeout_proxy: float | None = None


@dataclass(frozen=True)
class RiskFirstSpec:
    name: str
    utility_weight: float
    clean_weight: float
    dirty_weight: float
    max_dirty_proxy: float


@dataclass(frozen=True)
class RiskQuantileSpec:
    name: str
    utility_weight: float
    clean_weight: float
    dirty_weight: float
    keep_frac: float


SCORE_SPECS = (
    ScoreSpec("P0_utility_proxy", 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ScoreSpec("P1_clean_direct_proxy", 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    ScoreSpec("P2_utility_minus_badmae025", 1.0, 0.0, 0.25, 0.0, 0.0, 0.0),
    ScoreSpec("P3_utility_minus_badmae050", 1.0, 0.0, 0.50, 0.0, 0.0, 0.0),
    ScoreSpec("P4_utility_minus_path", 1.0, 0.0, 0.50, 0.25, 0.20, 0.15),
    ScoreSpec("P5_clean_plus_utility_minus_path", 0.50, 0.50, 0.50, 0.25, 0.20, 0.15),
    ScoreSpec("P6_utility_low_badmae_gate50", 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, max_bad_mae_proxy=0.50),
    ScoreSpec("P7_clean_low_badmae_gate50", 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, max_bad_mae_proxy=0.50),
    ScoreSpec(
        "P8_utility_path_gate60",
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        max_bad_mae_proxy=0.60,
        max_wide_proxy=0.60,
        max_timeout_proxy=0.70,
    ),
    ScoreSpec(
        "P9_clean_path_gate60",
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        max_bad_mae_proxy=0.60,
        max_wide_proxy=0.60,
        max_timeout_proxy=0.70,
    ),
)

RISK_FIRST_SPECS = (
    RiskFirstSpec("R0_utility_dirty_keep20", 1.0, 0.0, 0.0, 0.20),
    RiskFirstSpec("R1_utility_dirty_keep35", 1.0, 0.0, 0.0, 0.35),
    RiskFirstSpec("R2_utility_dirty_keep50", 1.0, 0.0, 0.0, 0.50),
    RiskFirstSpec("R3_utility_minus_dirty025_keep35", 1.0, 0.0, 0.25, 0.35),
    RiskFirstSpec("R4_utility_minus_dirty050_keep50", 1.0, 0.0, 0.50, 0.50),
    RiskFirstSpec("R5_clean_dirty_keep35", 0.0, 1.0, 0.0, 0.35),
    RiskFirstSpec("R6_clean_dirty_keep50", 0.0, 1.0, 0.0, 0.50),
    RiskFirstSpec("R7_utility_clean_minus_dirty_keep50", 0.5, 0.5, 0.50, 0.50),
)

RISK_QUANTILE_SPECS = (
    RiskQuantileSpec("Q0_utility_veto_worst15", 1.0, 0.0, 0.0, 0.85),
    RiskQuantileSpec("Q1_utility_veto_worst30", 1.0, 0.0, 0.0, 0.70),
    RiskQuantileSpec("Q2_utility_keep50", 1.0, 0.0, 0.0, 0.50),
    RiskQuantileSpec("Q3_utility_keep35", 1.0, 0.0, 0.0, 0.35),
    RiskQuantileSpec("Q4_utility_minus_dirty_veto30", 1.0, 0.0, 0.25, 0.70),
    RiskQuantileSpec("Q5_utility_minus_dirty_keep50", 1.0, 0.0, 0.50, 0.50),
    RiskQuantileSpec("Q6_clean_keep50", 0.0, 1.0, 0.0, 0.50),
    RiskQuantileSpec("Q7_utility_clean_keep50", 0.5, 0.5, 0.25, 0.50),
)


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_float_csv(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _safe_min(values: Any) -> float:
    series = _safe_numeric(values).dropna()
    return float(series.min()) if len(series) else float("nan")


def _safe_max(values: Any) -> float:
    series = _safe_numeric(values).dropna()
    return float(series.max()) if len(series) else float("nan")


def _weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str) -> float:
    values = _safe_numeric(frame[value_col])
    weights = _safe_numeric(frame[weight_col]).fillna(0.0)
    mask = values.notna() & weights.gt(0.0)
    if not bool(mask.any()):
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def _rank_pct(values: pd.Series, *, high_good: bool = True) -> pd.Series:
    ranks = _safe_numeric(values).rank(method="average", pct=True)
    if not high_good:
        ranks = 1.0 - ranks
    return ranks.clip(0.0, 1.0)


def _proxy_score(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    target_train: pd.Series,
    top_k: int,
    method: str,
    tail_frac: float,
) -> tuple[pd.Series, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y = _safe_numeric(target_train)
    full_mean = _safe_mean(y)
    for feature in features:
        values = _safe_numeric(train[feature])
        if str(method) == "tail_lift":
            for high_good in (True, False):
                ranks = _rank_pct(values, high_good=high_good)
                mask = ranks >= 1.0 - float(tail_frac)
                if int(mask.sum()) < 20:
                    continue
                top_mean = _safe_mean(y[mask])
                lift = top_mean - full_mean if math.isfinite(top_mean) and math.isfinite(full_mean) else float("nan")
                if math.isfinite(lift):
                    rows.append(
                        {
                            "feature": feature,
                            "ic": 1.0 if high_good else -1.0,
                            "abs_ic": float(lift),
                            "tail_lift": float(lift),
                            "top_mean": float(top_mean),
                        }
                    )
        else:
            ic = _spearman(values, y)
            if math.isfinite(ic):
                rows.append({"feature": feature, "ic": float(ic), "abs_ic": abs(float(ic)), "tail_lift": float("nan")})
    if not rows:
        return pd.Series(np.nan, index=valid.index), {
            "features": [],
            "top_abs_ic": float("nan"),
            "mean_top_abs_ic": float("nan"),
            "method": str(method),
        }
    ic_frame = pd.DataFrame(rows).sort_values("abs_ic", ascending=False).head(int(top_k))
    parts: list[pd.Series] = []
    for _, row in ic_frame.iterrows():
        feature = str(row["feature"])
        high_good = float(row["ic"]) >= 0.0
        parts.append(_rank_pct(valid[feature], high_good=high_good).fillna(0.5))
    score = pd.concat(parts, axis=1).mean(axis=1) if parts else pd.Series(np.nan, index=valid.index)
    return score.reindex(valid.index), {
        "features": ic_frame["feature"].astype(str).tolist(),
        "top_abs_ic": float(ic_frame["abs_ic"].iloc[0]),
        "mean_top_abs_ic": float(ic_frame["abs_ic"].mean()),
        "mean_tail_lift": _safe_mean(ic_frame.get("tail_lift", pd.Series(dtype=float))),
        "method": str(method),
    }


def _target_components(metrics: pd.DataFrame) -> dict[str, pd.Series]:
    u = _safe_numeric(metrics["u_policy_net"]).fillna(-0.02)
    barrier = _safe_numeric(metrics["barrier"]).fillna(0.0)
    mae = _safe_numeric(metrics["mae_norm"]).fillna(10.0)
    bars = _safe_numeric(metrics["bars_to_mfe"]).fillna(24.0)
    timeout = metrics["is_timeout"].astype(float).fillna(0.0)
    utility = pd.Series(_sigmoid((u - 0.0015) / 0.008), index=metrics.index).clip(0.0, 1.0)
    clean_envelope = (
        pd.Series(_sigmoid((1.00 - mae) / 0.25), index=metrics.index)
        * pd.Series(_sigmoid((0.025 - barrier) / 0.006), index=metrics.index)
        * pd.Series(_sigmoid((14.0 - bars) / 5.0), index=metrics.index)
        * (1.0 - timeout)
    ).clip(0.0, 1.0)
    clean = (utility * clean_envelope).clip(0.0, 1.0)
    return {
        "utility": utility,
        "clean": clean,
        "bad_mae": pd.Series(_sigmoid((mae - 1.0) / 0.25), index=metrics.index).clip(0.0, 1.0),
        "wide": pd.Series(_sigmoid((barrier - 0.025) / 0.006), index=metrics.index).clip(0.0, 1.0),
        "timeout": timeout.clip(0.0, 1.0),
        "slow": pd.Series(_sigmoid((bars - 14.0) / 5.0), index=metrics.index).clip(0.0, 1.0),
    }


def _score_from_components(
    *,
    spec: ScoreSpec,
    utility_proxy: pd.Series,
    clean_proxy: pd.Series,
    bad_mae_proxy: pd.Series,
    wide_proxy: pd.Series,
    timeout_proxy: pd.Series,
    slow_proxy: pd.Series,
) -> pd.Series:
    score = (
        float(spec.utility_weight) * utility_proxy
        + float(spec.clean_weight) * clean_proxy
        - float(spec.bad_mae_weight) * bad_mae_proxy
        - float(spec.wide_weight) * wide_proxy
        - float(spec.timeout_weight) * timeout_proxy
        - float(spec.slow_weight) * slow_proxy
    )
    mask = pd.Series(True, index=score.index)
    if spec.max_bad_mae_proxy is not None:
        mask &= bad_mae_proxy <= float(spec.max_bad_mae_proxy)
    if spec.max_wide_proxy is not None:
        mask &= wide_proxy <= float(spec.max_wide_proxy)
    if spec.max_timeout_proxy is not None:
        mask &= timeout_proxy <= float(spec.max_timeout_proxy)
    return score.where(mask)


def _dirty_proxy_score(
    *,
    bad_mae_proxy: pd.Series,
    wide_proxy: pd.Series,
    timeout_proxy: pd.Series,
    slow_proxy: pd.Series,
) -> pd.Series:
    parts = [
        _safe_numeric(bad_mae_proxy),
        _safe_numeric(wide_proxy),
        _safe_numeric(timeout_proxy),
        _safe_numeric(slow_proxy),
    ]
    return pd.concat(parts, axis=1).max(axis=1).clip(0.0, 1.0)


def _risk_first_score_from_components(
    *,
    spec: RiskFirstSpec,
    utility_proxy: pd.Series,
    clean_proxy: pd.Series,
    dirty_proxy: pd.Series,
) -> pd.Series:
    score = (
        float(spec.utility_weight) * _safe_numeric(utility_proxy)
        + float(spec.clean_weight) * _safe_numeric(clean_proxy)
        - float(spec.dirty_weight) * _safe_numeric(dirty_proxy)
    )
    return score.where(_safe_numeric(dirty_proxy) <= float(spec.max_dirty_proxy))


def _dirty_keep_rank(
    *,
    frame: pd.DataFrame,
    dirty_proxy: pd.Series,
    selection_mode: str,
) -> pd.Series:
    dirty = _safe_numeric(dirty_proxy)
    if str(selection_mode) == "timestamp" and "__ts__" in frame.columns:
        timestamps = pd.to_datetime(frame["__ts__"], errors="coerce")
        ranked = dirty.groupby(timestamps, dropna=False).rank(method="first", pct=True, ascending=True)
        return ranked.reindex(dirty.index)
    return dirty.rank(method="first", pct=True, ascending=True).reindex(dirty.index)


def _risk_quantile_score_from_components(
    *,
    spec: RiskQuantileSpec,
    frame: pd.DataFrame,
    utility_proxy: pd.Series,
    clean_proxy: pd.Series,
    dirty_proxy: pd.Series,
    selection_mode: str,
) -> pd.Series:
    dirty = _safe_numeric(dirty_proxy)
    score = (
        float(spec.utility_weight) * _safe_numeric(utility_proxy)
        + float(spec.clean_weight) * _safe_numeric(clean_proxy)
        - float(spec.dirty_weight) * dirty
    )
    keep_rank = _dirty_keep_rank(
        frame=frame,
        dirty_proxy=dirty,
        selection_mode=str(selection_mode),
    )
    return score.where(keep_rank <= float(spec.keep_frac))


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


def _selection_metrics_from_indices(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    idx: np.ndarray,
    arm: str,
    selector: str,
    period: str,
    top_frac: float,
) -> dict[str, Any]:
    selected_metrics = metrics.iloc[idx] if len(idx) else metrics.iloc[:0]
    selected_frame = frame.iloc[idx] if len(idx) else frame.iloc[:0]
    selected_target = target.iloc[idx] if len(idx) else target.iloc[:0]
    utility = selected_metrics["u_policy_net"]
    symbols = selected_frame.get("__symbol__", pd.Series(dtype=object))
    timestamps = selected_frame.get("__ts__", pd.Series(dtype="datetime64[ns]"))
    return {
        "arm": arm,
        "selector": selector,
        "period": period,
        "top_frac": float(top_frac),
        "rows": int(len(frame)),
        "selected_rows": int(len(idx)),
        "target_top_soft_mean": _safe_mean(selected_target.get("target_soft")),
        "target_top_hard_rate": _safe_mean(selected_target.get("target_hard")),
        "mean_u": _safe_mean(utility),
        "median_u": _safe_quantile(utility, 0.50),
        "q10_u": _safe_quantile(utility, 0.10),
        "hit_u": _safe_mean(utility > 0.0),
        "mean_return_net": _safe_mean(selected_metrics["ret_net"]),
        "hit_return_net": _safe_mean(selected_metrics["ret_net"] > 0.0),
        "mean_barrier": _safe_mean(selected_metrics["barrier"]),
        "p90_barrier": _safe_quantile(selected_metrics["barrier"], 0.90),
        "wide_barrier_25bps_rate": _safe_mean(selected_metrics["barrier"] > 0.025),
        "wide_barrier_35bps_rate": _safe_mean(selected_metrics["barrier"] > 0.035),
        "mean_mae_norm": _safe_mean(selected_metrics["mae_norm"]),
        "p90_mae_norm": _safe_quantile(selected_metrics["mae_norm"], 0.90),
        "bad_mae_1r_rate": _safe_mean(selected_metrics["mae_norm"] >= 1.0),
        "mean_mfe_norm": _safe_mean(selected_metrics["mfe_norm"]),
        "mean_bars_to_mfe": _safe_mean(selected_metrics["bars_to_mfe"]),
        "p90_bars_to_mfe": _safe_quantile(selected_metrics["bars_to_mfe"], 0.90),
        "timeout_rate": _safe_mean(selected_metrics["is_timeout"].astype(float)),
        "symbol_effective_n": _effective_n(symbols),
        "top_symbol_share": float(symbols.value_counts(normalize=True, dropna=False).iloc[0]) if len(symbols) else 0.0,
        "timestamp_effective_n": _effective_n(timestamps.astype(str)),
        "top_timestamp_share": float(timestamps.astype(str).value_counts(normalize=True, dropna=False).iloc[0])
        if len(timestamps)
        else 0.0,
    }


def _selection_metrics_by_mode(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    arm: str,
    selector: str,
    period: str,
    top_frac: float,
    selection_mode: str,
) -> dict[str, Any]:
    if selection_mode == "timestamp":
        idx = _timestamp_top_indices(frame, score, top_frac)
        return _selection_metrics_from_indices(
            frame=frame,
            metrics=metrics,
            target=target,
            idx=idx,
            arm=arm,
            selector=selector,
            period=period,
            top_frac=top_frac,
        )
    return _selection_metrics(
        frame=frame,
        metrics=metrics,
        target=target,
        score=score,
        arm=arm,
        selector=selector,
        period=period,
        top_frac=top_frac,
    )


def _target_for_selection(components: dict[str, pd.Series], index: pd.Index) -> pd.DataFrame:
    soft = components["clean"].reindex(index).clip(0.0, 1.0)
    hard = (soft >= 0.50).astype(float)
    return pd.DataFrame({"target_soft": soft, "target_hard": hard}, index=index)


def _monthly_weekly_rows(
    *,
    valid_frame: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    valid_target: pd.DataFrame,
    score: pd.Series,
    score_arm: str,
    month: str,
    top_fracs: list[float],
    component_diag: dict[str, dict[str, Any]],
    selection_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    score_reset = score.reset_index(drop=True)
    frame_reset = valid_frame.reset_index(drop=True)
    metrics_reset = valid_metrics.reset_index(drop=True)
    target_reset = valid_target.reset_index(drop=True)
    selector = f"dual_proxy_oos_{selection_mode}"
    for frac in top_fracs:
        row = _selection_metrics_by_mode(
            frame=frame_reset,
            metrics=metrics_reset,
            target=target_reset,
            score=score_reset,
            arm=score_arm,
            selector=selector,
            period=str(month),
            top_frac=float(frac),
            selection_mode=selection_mode,
        )
        row.update(_diag_columns(component_diag))
        monthly_rows.append(row)

        weeks = frame_reset["__ts__"].dt.to_period("W-SUN").astype(str)
        for week, ids in pd.Series(np.arange(len(frame_reset)), index=frame_reset.index).groupby(weeks, dropna=False):
            pos = ids.to_numpy(dtype=np.int64)
            if len(pos) < 20:
                continue
            week_frame = frame_reset.iloc[pos].reset_index(drop=True)
            week_row = _selection_metrics_by_mode(
                frame=week_frame,
                metrics=metrics_reset.iloc[pos].reset_index(drop=True),
                target=target_reset.iloc[pos].reset_index(drop=True),
                score=score_reset.iloc[pos].reset_index(drop=True),
                arm=score_arm,
                selector=selector,
                period=str(month),
                top_frac=float(frac),
                selection_mode=selection_mode,
            )
            week_row["week"] = str(week)
            week_row["week_selected_rows"] = int(week_row["selected_rows"])
            week_row["week_selected_share"] = (
                float(week_row["selected_rows"] / len(pos)) if len(pos) else float("nan")
            )
            week_row.update(_diag_columns(component_diag))
            weekly_rows.append(week_row)
    return monthly_rows, weekly_rows


def _diag_columns(component_diag: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, diag in component_diag.items():
        out[f"{name}_proxy_top_abs_ic"] = diag.get("top_abs_ic")
        out[f"{name}_proxy_mean_top_abs_ic"] = diag.get("mean_top_abs_ic")
        out[f"{name}_proxy_features"] = ",".join(str(v) for v in diag.get("features", []))
    return out


def _summarize_month(prefix: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            f"{prefix}_months": 0,
            f"{prefix}_positive_months": 0,
            f"{prefix}_mean_month_u": float("nan"),
            f"{prefix}_worst_month_u": float("nan"),
            f"{prefix}_selected_rows": 0,
            f"{prefix}_bad_mae_1r_rate": float("nan"),
            f"{prefix}_p90_mae_norm": float("nan"),
            f"{prefix}_wide_25bps_rate": float("nan"),
            f"{prefix}_timeout_rate": float("nan"),
        }
    mean_u = _safe_numeric(frame["mean_u"])
    return {
        f"{prefix}_months": int(frame["period"].astype(str).nunique()),
        f"{prefix}_positive_months": int(mean_u.gt(0.0).sum()),
        f"{prefix}_mean_month_u": _safe_mean(mean_u),
        f"{prefix}_worst_month_u": _safe_min(mean_u),
        f"{prefix}_selected_rows": int(_safe_numeric(frame["selected_rows"]).sum()),
        f"{prefix}_bad_mae_1r_rate": _weighted_mean(frame, "bad_mae_1r_rate", "selected_rows"),
        f"{prefix}_p90_mae_norm": _weighted_mean(frame, "p90_mae_norm", "selected_rows"),
        f"{prefix}_mean_mfe_norm": _weighted_mean(frame, "mean_mfe_norm", "selected_rows"),
        f"{prefix}_mean_mae_norm": _weighted_mean(frame, "mean_mae_norm", "selected_rows"),
        f"{prefix}_wide_25bps_rate": _weighted_mean(frame, "wide_barrier_25bps_rate", "selected_rows"),
        f"{prefix}_timeout_rate": _weighted_mean(frame, "timeout_rate", "selected_rows"),
        f"{prefix}_max_top_symbol_share": _safe_max(frame["top_symbol_share"]),
    }


def _summarize_week(prefix: str, frame: pd.DataFrame, *, min_week_rows: int) -> dict[str, Any]:
    if frame.empty:
        return {
            f"{prefix}_weeks": 0,
            f"{prefix}_material_weeks": 0,
            f"{prefix}_material_positive_week_rate": float("nan"),
            f"{prefix}_q25_week_u": float("nan"),
            f"{prefix}_worst_week_u": float("nan"),
            f"{prefix}_weekly_bad_mae_1r_rate": float("nan"),
        }
    mean_u = _safe_numeric(frame["mean_u"])
    week_rows = _safe_numeric(frame["week_selected_rows"]).fillna(0.0)
    material = week_rows >= int(min_week_rows)
    positive = mean_u > 0.0
    return {
        f"{prefix}_weeks": int(len(frame)),
        f"{prefix}_material_weeks": int(material.sum()),
        f"{prefix}_material_positive_week_rate": float((positive & material).sum() / material.sum())
        if int(material.sum())
        else float("nan"),
        f"{prefix}_q25_week_u": _safe_quantile(mean_u, 0.25),
        f"{prefix}_worst_week_u": _safe_min(mean_u),
        f"{prefix}_weekly_bad_mae_1r_rate": _weighted_mean(frame, "bad_mae_1r_rate", "week_selected_rows"),
        f"{prefix}_weekly_timeout_rate": _weighted_mean(frame, "timeout_rate", "week_selected_rows"),
        f"{prefix}_weekly_wide_25bps_rate": _weighted_mean(frame, "wide_barrier_25bps_rate", "week_selected_rows"),
    }


def _fit_holdout_summary(
    *,
    monthly: pd.DataFrame,
    weekly: pd.DataFrame,
    fit_months: list[str],
    holdout_month: str,
    min_week_rows: int,
    min_fit_material_weeks: int,
    min_holdout_material_weeks: int,
    min_fit_positive_week_rate: float,
    min_holdout_positive_week_rate: float,
    max_clean_bad_mae_1r_rate: float,
    max_clean_p90_mae_norm: float,
    max_bounded_bad_mae_1r_rate: float,
    max_bounded_p90_mae_norm: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (arm, frac), group in monthly.groupby(["arm", "top_frac"], observed=True, dropna=False):
        week_group = weekly[
            weekly["arm"].astype(str).eq(str(arm))
            & _safe_numeric(weekly["top_frac"]).eq(float(frac))
        ].copy()
        fit_month = group[group["period"].astype(str).isin(fit_months)].copy()
        holdout_monthly = group[group["period"].astype(str).eq(str(holdout_month))].copy()
        fit_week = week_group[week_group["period"].astype(str).isin(fit_months)].copy()
        holdout_week = week_group[week_group["period"].astype(str).eq(str(holdout_month))].copy()
        if fit_month.empty or holdout_monthly.empty:
            continue
        row: dict[str, Any] = {"score_arm": str(arm), "top_frac": float(frac)}
        row.update(_summarize_month("fit", fit_month))
        row.update(_summarize_month("holdout", holdout_monthly))
        row.update(_summarize_week("fit", fit_week, min_week_rows=min_week_rows))
        row.update(_summarize_week("holdout", holdout_week, min_week_rows=min_week_rows))
        fit_sign = (
            row["fit_months"] == len(fit_months)
            and row["fit_positive_months"] == len(fit_months)
            and row["fit_worst_month_u"] > 0.0
            and row["fit_material_weeks"] >= min_fit_material_weeks
            and row["fit_material_positive_week_rate"] >= min_fit_positive_week_rate
        )
        holdout_sign = (
            row["holdout_mean_month_u"] > 0.0
            and row["holdout_material_weeks"] >= min_holdout_material_weeks
            and row["holdout_material_positive_week_rate"] >= min_holdout_positive_week_rate
        )
        fit_clean = (
            fit_sign
            and row["fit_bad_mae_1r_rate"] <= max_clean_bad_mae_1r_rate
            and row["fit_p90_mae_norm"] <= max_clean_p90_mae_norm
        )
        holdout_clean_standalone = (
            holdout_sign
            and row["holdout_bad_mae_1r_rate"] <= max_clean_bad_mae_1r_rate
            and row["holdout_p90_mae_norm"] <= max_clean_p90_mae_norm
        )
        fit_bounded = (
            fit_sign
            and row["fit_bad_mae_1r_rate"] <= max_bounded_bad_mae_1r_rate
            and row["fit_p90_mae_norm"] <= max_bounded_p90_mae_norm
        )
        holdout_bounded_standalone = (
            holdout_sign
            and row["holdout_bad_mae_1r_rate"] <= max_bounded_bad_mae_1r_rate
            and row["holdout_p90_mae_norm"] <= max_bounded_p90_mae_norm
        )
        row["fit_sign_pass"] = bool(fit_sign)
        row["holdout_sign_pass"] = bool(holdout_sign)
        row["fit_clean_pass"] = bool(fit_clean)
        row["holdout_clean_standalone_pass"] = bool(holdout_clean_standalone)
        row["holdout_clean_pass"] = bool(fit_clean and holdout_clean_standalone)
        row["fit_bounded_pass"] = bool(fit_bounded)
        row["holdout_bounded_standalone_pass"] = bool(holdout_bounded_standalone)
        row["holdout_bounded_pass"] = bool(fit_bounded and holdout_bounded_standalone)
        row["positive_dirty_holdout"] = bool(holdout_sign and not holdout_bounded_standalone)
        row["path_risk_score"] = float(
            (row["holdout_mean_month_u"] if pd.notna(row["holdout_mean_month_u"]) else 0.0)
            + 0.50 * (row["holdout_q25_week_u"] if pd.notna(row["holdout_q25_week_u"]) else 0.0)
            - 0.020 * (row["holdout_bad_mae_1r_rate"] if pd.notna(row["holdout_bad_mae_1r_rate"]) else 0.0)
            - 0.003 * (row["holdout_p90_mae_norm"] if pd.notna(row["holdout_p90_mae_norm"]) else 0.0)
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["holdout_clean_pass", "holdout_bounded_pass", "positive_dirty_holdout", "path_risk_score"],
        ascending=[False, False, False, False],
    )


def run_ablation(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    proxy_top_k: int,
    proxy_method: str,
    proxy_tail_frac: float,
    top_fracs: list[float],
    fit_months: list[str],
    holdout_month: str,
    min_week_rows: int,
    selection_mode: str,
    include_risk_first: bool,
    include_risk_quantile: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_labels(labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_store_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    if not feature_matrix.empty:
        new_cols = [col for col in feature_matrix.columns if col not in frame.columns]
        frame = pd.concat(
            [
                frame.reset_index(drop=True),
                feature_matrix.loc[:, new_cols].reset_index(drop=True),
            ],
            axis=1,
        )
    metrics = _path_metrics(frame)
    components = _target_components(metrics)
    features = _feature_columns(frame)

    month_series = frame["__ts__"].dt.to_period("M").astype(str)
    months = sorted(month_series.dropna().unique())
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []

    for month in months[1:]:
        train_mask = month_series < str(month)
        valid_mask = month_series == str(month)
        if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
            continue
        train = frame.loc[train_mask].copy()
        valid = frame.loc[valid_mask].copy()
        valid_metrics = metrics.loc[valid_mask].copy()
        valid_components = {name: series.loc[valid_mask] for name, series in components.items()}
        proxies: dict[str, pd.Series] = {}
        diag: dict[str, dict[str, Any]] = {}
        for name in ("utility", "clean", "bad_mae", "wide", "timeout", "slow"):
            score, score_diag = _proxy_score(
                train=train,
                valid=valid,
                features=features,
                target_train=components[name].loc[train_mask],
                top_k=proxy_top_k,
                method=str(proxy_method),
                tail_frac=float(proxy_tail_frac),
            )
            proxies[name] = score
            diag[name] = score_diag
            component_rows.append(
                {
                    "period": str(month),
                    "component": name,
                    "proxy_top_abs_ic": score_diag.get("top_abs_ic"),
                    "proxy_mean_top_abs_ic": score_diag.get("mean_top_abs_ic"),
                    "proxy_ic_actual_u": _spearman(score, valid_metrics["u_policy_net"]),
                    "proxy_ic_actual_bad_mae": _spearman(score, (valid_metrics["mae_norm"] >= 1.0).astype(float)),
                    "proxy_ic_actual_clean": _spearman(score, valid_components["clean"]),
                    "proxy_features": ",".join(score_diag.get("features", [])),
                }
            )

        valid_target = _target_for_selection(valid_components, valid.index)
        dirty_proxy = _dirty_proxy_score(
            bad_mae_proxy=proxies["bad_mae"],
            wide_proxy=proxies["wide"],
            timeout_proxy=proxies["timeout"],
            slow_proxy=proxies["slow"],
        )
        for spec in SCORE_SPECS:
            score = _score_from_components(
                spec=spec,
                utility_proxy=proxies["utility"],
                clean_proxy=proxies["clean"],
                bad_mae_proxy=proxies["bad_mae"],
                wide_proxy=proxies["wide"],
                timeout_proxy=proxies["timeout"],
                slow_proxy=proxies["slow"],
            )
            m_rows, w_rows = _monthly_weekly_rows(
                valid_frame=valid,
                valid_metrics=valid_metrics,
                valid_target=valid_target,
                score=score,
                score_arm=spec.name,
                month=str(month),
                top_fracs=top_fracs,
                component_diag=diag,
                selection_mode=str(selection_mode),
            )
            monthly_rows.extend(m_rows)
            weekly_rows.extend(w_rows)
        if include_risk_first:
            for spec in RISK_FIRST_SPECS:
                score = _risk_first_score_from_components(
                    spec=spec,
                    utility_proxy=proxies["utility"],
                    clean_proxy=proxies["clean"],
                    dirty_proxy=dirty_proxy,
                )
                m_rows, w_rows = _monthly_weekly_rows(
                    valid_frame=valid,
                    valid_metrics=valid_metrics,
                    valid_target=valid_target,
                    score=score,
                    score_arm=spec.name,
                    month=str(month),
                    top_fracs=top_fracs,
                    component_diag=diag,
                    selection_mode=str(selection_mode),
                )
                monthly_rows.extend(m_rows)
                weekly_rows.extend(w_rows)
        if include_risk_quantile:
            for spec in RISK_QUANTILE_SPECS:
                score = _risk_quantile_score_from_components(
                    spec=spec,
                    frame=valid,
                    utility_proxy=proxies["utility"],
                    clean_proxy=proxies["clean"],
                    dirty_proxy=dirty_proxy,
                    selection_mode=str(selection_mode),
                )
                m_rows, w_rows = _monthly_weekly_rows(
                    valid_frame=valid,
                    valid_metrics=valid_metrics,
                    valid_target=valid_target,
                    score=score,
                    score_arm=spec.name,
                    month=str(month),
                    top_fracs=top_fracs,
                    component_diag=diag,
                    selection_mode=str(selection_mode),
                )
                monthly_rows.extend(m_rows)
                weekly_rows.extend(w_rows)

    monthly = pd.DataFrame(monthly_rows)
    weekly = pd.DataFrame(weekly_rows)
    components_summary = pd.DataFrame(component_rows)
    fit_holdout = _fit_holdout_summary(
        monthly=monthly,
        weekly=weekly,
        fit_months=fit_months,
        holdout_month=holdout_month,
        min_week_rows=min_week_rows,
        min_fit_material_weeks=4,
        min_holdout_material_weeks=2,
        min_fit_positive_week_rate=0.55,
        min_holdout_positive_week_rate=0.50,
        max_clean_bad_mae_1r_rate=0.50,
        max_clean_p90_mae_norm=3.0,
        max_bounded_bad_mae_1r_rate=0.80,
        max_bounded_p90_mae_norm=4.0,
    )

    paths = {
        "monthly": output_dir / "dual_proxy_path_risk_monthly.csv",
        "weekly": output_dir / "dual_proxy_path_risk_weekly.csv",
        "component_proxy": output_dir / "dual_proxy_component_ic.csv",
        "fit_holdout": output_dir / "dual_proxy_path_risk_fit_holdout.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    components_summary.to_csv(paths["component_proxy"], index=False)
    fit_holdout.to_csv(paths["fit_holdout"], index=False)

    manifest = {
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
        "features": features,
        "feature_count": int(len(features)),
        "proxy_top_k": int(proxy_top_k),
        "proxy_method": str(proxy_method),
        "proxy_tail_frac": float(proxy_tail_frac),
        "selection_mode": str(selection_mode),
        "include_risk_first": bool(include_risk_first),
        "include_risk_quantile": bool(include_risk_quantile),
        "score_specs": [spec.__dict__ for spec in SCORE_SPECS],
        "risk_first_specs": [spec.__dict__ for spec in RISK_FIRST_SPECS] if include_risk_first else [],
        "risk_quantile_specs": [spec.__dict__ for spec in RISK_QUANTILE_SPECS] if include_risk_quantile else [],
        "top_fracs": [float(v) for v in top_fracs],
        "fit_months": [str(v) for v in fit_months],
        "holdout_month": str(holdout_month),
        "rows_monthly": int(len(monthly)),
        "rows_weekly": int(len(weekly)),
        "fit_clean_pass_rows": int(fit_holdout["fit_clean_pass"].sum()) if not fit_holdout.empty else 0,
        "holdout_clean_pass_rows": int(fit_holdout["holdout_clean_pass"].sum()) if not fit_holdout.empty else 0,
        "fit_bounded_pass_rows": int(fit_holdout["fit_bounded_pass"].sum()) if not fit_holdout.empty else 0,
        "holdout_bounded_pass_rows": int(fit_holdout["holdout_bounded_pass"].sum()) if not fit_holdout.empty else 0,
        "positive_dirty_holdout_rows": int(fit_holdout["positive_dirty_holdout"].sum()) if not fit_holdout.empty else 0,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(output_dir, fit_holdout, components_summary, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def _format_table(frame: pd.DataFrame, cols: list[str], limit: int = 30) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[col for col in cols if col in frame.columns]].head(limit).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
    return view.to_markdown(index=False)


def _write_markdown(
    output_dir: Path,
    fit_holdout: pd.DataFrame,
    components_summary: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "dual_proxy_path_risk_ablation.md"
    cols = [
        "score_arm",
        "top_frac",
        "path_risk_score",
        "fit_sign_pass",
        "fit_clean_pass",
        "fit_bounded_pass",
        "fit_mean_month_u",
        "fit_worst_month_u",
        "fit_material_positive_week_rate",
        "fit_bad_mae_1r_rate",
        "fit_p90_mae_norm",
        "holdout_clean_pass",
        "holdout_bounded_pass",
        "holdout_mean_month_u",
        "holdout_material_positive_week_rate",
        "holdout_bad_mae_1r_rate",
        "holdout_p90_mae_norm",
        "holdout_timeout_rate",
        "holdout_wide_25bps_rate",
    ]
    component_cols = [
        "period",
        "component",
        "proxy_top_abs_ic",
        "proxy_mean_top_abs_ic",
        "proxy_ic_actual_u",
        "proxy_ic_actual_bad_mae",
        "proxy_ic_actual_clean",
        "proxy_features",
    ]
    clean_pass = fit_holdout[fit_holdout["holdout_clean_pass"].eq(True)] if not fit_holdout.empty else fit_holdout
    bounded_pass = fit_holdout[fit_holdout["holdout_bounded_pass"].eq(True)] if not fit_holdout.empty else fit_holdout
    positive_dirty = (
        fit_holdout[fit_holdout["positive_dirty_holdout"].eq(True)].sort_values(
            "holdout_mean_month_u",
            ascending=False,
        )
        if not fit_holdout.empty
        else fit_holdout
    )
    best = fit_holdout.sort_values("path_risk_score", ascending=False) if not fit_holdout.empty else fit_holdout
    comp = (
        components_summary[components_summary["period"].astype(str).isin(manifest["fit_months"] + [manifest["holdout_month"]])]
        .sort_values(["period", "component"])
        if not components_summary.empty
        else components_summary
    )
    lines = [
        "# Dual Proxy Path-Risk Label Ablation",
        "",
        "Scope: proxy tests only. No LightGBM, Optuna, policy geometry, or base/meta training is run.",
        "",
        f"Labels: `{manifest['labels_path']}`",
        f"Feature dir: `{manifest['feature_dir']}`",
        f"Feature count: `{manifest['feature_count']}`",
        f"Fit months: `{','.join(manifest['fit_months'])}`",
        f"Holdout month: `{manifest['holdout_month']}`",
        f"Selection mode: `{manifest['selection_mode']}`",
        f"Risk-first arms: `{manifest['include_risk_first']}`",
        f"Risk-quantile arms: `{manifest['include_risk_quantile']}`",
        f"Top fractions: `{','.join(str(v) for v in manifest['top_fracs'])}`",
        "",
        "Each score arm is built from prior-month feature-rank proxies for utility, clean path, bad-MAE, wide barrier, timeout, and slow MFE.",
        "Timestamp selection mode chooses the top fraction inside each timestamp bucket before aggregating monthly/weekly economics.",
        "Risk-first arms first filter on a combined dirty-path proxy: max(bad-MAE, wide-barrier, timeout, slow-MFE proxy).",
        "Risk-quantile arms keep a fixed cleanest fraction by dirty-path proxy, timestamp-local when timestamp selection is used.",
        "",
        "## Counts",
        "",
        f"- Monthly rows: `{manifest['rows_monthly']}`",
        f"- Weekly rows: `{manifest['rows_weekly']}`",
        f"- Fit clean pass: `{manifest['fit_clean_pass_rows']}`",
        f"- Holdout clean pass after fit selection: `{manifest['holdout_clean_pass_rows']}`",
        f"- Fit bounded pass: `{manifest['fit_bounded_pass_rows']}`",
        f"- Holdout bounded pass after fit selection: `{manifest['holdout_bounded_pass_rows']}`",
        f"- Positive but economically dirty holdout: `{manifest['positive_dirty_holdout_rows']}`",
        "",
        "## Clean Holdout Passes",
        "",
        _format_table(clean_pass, cols, limit=40),
        "",
        "## Bounded Holdout Passes",
        "",
        _format_table(bounded_pass, cols, limit=40),
        "",
        "## Positive But Economically Dirty Holdout",
        "",
        _format_table(positive_dirty, cols, limit=40),
        "",
        "## Best Rejected Or Failed Rows",
        "",
        _format_table(best, cols, limit=40),
        "",
        "## Component Proxy ICs",
        "",
        _format_table(comp, component_cols, limit=80),
        "",
        "## Outputs",
        "",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Weekly: `{manifest['outputs']['weekly']}`",
        f"- Component IC: `{manifest['outputs']['component_proxy']}`",
        f"- Fit/Holdout: `{manifest['outputs']['fit_holdout']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--proxy-top-k", type=int, default=12)
    parser.add_argument("--proxy-method", choices=["ic", "tail_lift"], default="ic")
    parser.add_argument("--proxy-tail-frac", type=float, default=0.05)
    parser.add_argument("--top-fracs", default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--fit-months", default=",".join(DEFAULT_FIT_MONTHS))
    parser.add_argument("--holdout-month", default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--min-week-rows", type=int, default=3)
    parser.add_argument("--selection-mode", choices=["global", "timestamp"], default="global")
    parser.add_argument("--include-risk-first", action="store_true")
    parser.add_argument("--include-risk-quantile", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_ablation(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        proxy_top_k=int(args.proxy_top_k),
        proxy_method=str(args.proxy_method),
        proxy_tail_frac=float(args.proxy_tail_frac),
        top_fracs=_parse_float_csv(args.top_fracs),
        fit_months=_parse_csv(args.fit_months),
        holdout_month=str(args.holdout_month),
        min_week_rows=int(args.min_week_rows),
        selection_mode=str(args.selection_mode),
        include_risk_first=bool(args.include_risk_first),
        include_risk_quantile=bool(args.include_risk_quantile),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
