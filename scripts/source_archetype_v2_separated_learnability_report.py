#!/usr/bin/env python3
"""Separated learnability diagnostics for source archetype v2 heads.

This diagnostic tests whether distinct utility, path-risk, wide-barrier,
timeout/holding, and acceptable-path targets are learnable in month-forward
smoke models, and whether adding v2 archetype features changes that.

It does not modify production training, Optuna, feature selection, or policy
artifacts.
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
    _json_safe,
    _load_feature_store_columns,
    _path_metrics,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _spearman,
)
from scripts.run_source_quality_label_walkforward_ablation import (  # noqa: E402
    DEFAULT_MONTHS,
    DEFAULT_SEEDS,
    _load_joined_frame,
    _parse_csv,
    _parse_int_csv,
)


DEFAULT_ARCHETYPES = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1_sideaware_20260702/source_archetypes_v2/candidate_source_archetypes_v2.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1_sideaware_20260702/source_archetypes_v2_learnability"
)
DEFAULT_HEADS = (
    "utility_head_v1",
    "bad_mae_risk_head_v1",
    "wide_barrier_risk_head_v1",
    "timeout_risk_head_v1",
    "holding_risk_head_v1",
    "acceptable_path_head_v1",
)
DEFAULT_FEATURE_SETS = ("base", "base_plus_v2", "v2_only")
DEFAULT_TOP_FRACS = (0.10,)
V2_FEATURE_COLS = (
    "prior_symbol_event_density_score",
    "prior_symbol_event_density_rank",
    "source_evidence_archetype_score",
    "source_independence_archetype_score",
    "source_freshness_archetype_score",
    "path_geometry_archetype_score",
    "timeout_holding_archetype_score",
    "regime_archetype_score",
    "symbol_behavior_archetype_score",
    "tag_source_evidence_archetype",
    "tag_source_independence_archetype",
    "tag_source_freshness_archetype",
    "tag_path_geometry_archetype",
    "tag_timeout_holding_archetype",
    "tag_regime_archetype",
    "tag_symbol_behavior_archetype",
)


@dataclass(frozen=True)
class HeadSpec:
    name: str
    kind: str
    direction: str


HEAD_SPECS = (
    HeadSpec("utility_head_v1", "utility", "positive"),
    HeadSpec("bad_mae_risk_head_v1", "bad_mae", "risk"),
    HeadSpec("wide_barrier_risk_head_v1", "wide_barrier", "risk"),
    HeadSpec("timeout_risk_head_v1", "timeout", "risk"),
    HeadSpec("holding_risk_head_v1", "holding", "risk"),
    HeadSpec("acceptable_path_head_v1", "acceptable_path", "positive"),
)


def _head_specs_by_name(names: list[str]) -> list[HeadSpec]:
    available = {spec.name: spec for spec in HEAD_SPECS}
    missing = sorted(set(names) - set(available))
    if missing:
        raise ValueError(f"Unknown head(s): {missing}; available={sorted(available)}")
    return [available[name] for name in names]


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _safe_auc(y_true: Any, score: Any) -> float:
    y = _safe_numeric(y_true)
    s = _safe_numeric(score)
    mask = y.notna() & s.notna()
    if int(mask.sum()) < 20 or y[mask].nunique(dropna=True) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y[mask].astype(int), s[mask]))
    except Exception:
        return float("nan")


def _safe_ap(y_true: Any, score: Any) -> float:
    y = _safe_numeric(y_true)
    s = _safe_numeric(score)
    mask = y.notna() & s.notna()
    if int(mask.sum()) < 20 or y[mask].nunique(dropna=True) < 2:
        return float("nan")
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y[mask].astype(int), s[mask]))
    except Exception:
        return float("nan")


def _balanced_weights(target_hard: pd.Series, train_mask: pd.Series) -> pd.Series:
    hard = _safe_numeric(target_hard).fillna(0.0).clip(0.0, 1.0)
    train = hard.loc[train_mask].dropna()
    prevalence = float(train.mean()) if len(train) else 0.0
    if prevalence <= 0.0 or prevalence >= 1.0:
        return pd.Series(1.0, index=target_hard.index, dtype=np.float32)
    pos_w = min(5.0, 0.5 / max(prevalence, 1e-6))
    neg_w = min(5.0, 0.5 / max(1.0 - prevalence, 1e-6))
    return hard.map({1.0: pos_w, 0.0: neg_w}).fillna(1.0).astype(np.float32)


def _train_quantile(values: pd.Series, train_mask: pd.Series, q: float, fallback: float) -> float:
    finite = _safe_numeric(values).loc[train_mask].replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return float(fallback)
    out = float(finite.quantile(float(q)))
    return out if math.isfinite(out) else float(fallback)


def _build_month_target(metrics: pd.DataFrame, train_mask: pd.Series, spec: HeadSpec) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    utility = _safe_numeric(metrics["u_policy_net"]).fillna(0.0)
    mae_norm = _safe_numeric(metrics["mae_norm"]).fillna(0.0)
    barrier = _safe_numeric(metrics["barrier"]).fillna(0.0)
    timeout = metrics["is_timeout"].astype(float).fillna(0.0)
    bars = _safe_numeric(metrics["bars_policy"]).fillna(24.0).clip(lower=0.0)
    u_q05 = _train_quantile(utility, train_mask, 0.05, -0.03)
    u_q95 = _train_quantile(utility, train_mask, 0.95, 0.03)
    u_span = max(u_q95 - u_q05, 1e-6)
    utility_soft = ((utility - u_q05) / u_span).clip(0.0, 1.0)
    bars_q40 = _train_quantile(bars, train_mask, 0.40, 4.0)
    bars_q80 = _train_quantile(bars, train_mask, 0.80, 16.0)
    bars_q90 = _train_quantile(bars, train_mask, 0.90, max(bars_q80, 24.0))
    bars_span = max(bars_q90 - bars_q40, 1.0)

    if spec.kind == "utility":
        target_soft = utility_soft
        target_hard = utility.gt(0.0).astype(float)
    elif spec.kind == "bad_mae":
        target_soft = (mae_norm / 4.0).clip(0.0, 1.0)
        target_hard = mae_norm.ge(1.0).astype(float)
    elif spec.kind == "wide_barrier":
        target_soft = (barrier / 0.050).clip(0.0, 1.0)
        target_hard = barrier.gt(0.025).astype(float)
    elif spec.kind == "timeout":
        target_soft = timeout.clip(0.0, 1.0)
        target_hard = timeout.gt(0.5).astype(float)
    elif spec.kind == "holding":
        target_soft = ((bars - bars_q40) / bars_span).clip(0.0, 1.0).where(timeout.le(0.5), 1.0)
        target_hard = bars.ge(bars_q80).astype(float).where(timeout.le(0.5), 1.0)
    elif spec.kind == "acceptable_path":
        mae_clean = (1.0 - (mae_norm / 2.0)).clip(0.0, 1.0)
        barrier_clean = (1.0 - (barrier / 0.040)).clip(0.0, 1.0)
        timeout_clean = 1.0 - timeout.clip(0.0, 1.0)
        target_soft = (0.40 * utility_soft + 0.25 * mae_clean + 0.20 * barrier_clean + 0.15 * timeout_clean).clip(0.0, 1.0)
        target_hard = ((utility > 0.0) & (mae_norm < 1.0) & (barrier <= 0.025) & timeout.le(0.5)).astype(float)
    else:
        raise ValueError(f"Unsupported head kind: {spec.kind}")

    target = pd.DataFrame(
        {
            "target_soft": _safe_numeric(target_soft).clip(0.0, 1.0),
            "target_hard": _safe_numeric(target_hard).clip(0.0, 1.0),
        },
        index=metrics.index,
    )
    weights = _balanced_weights(target["target_hard"], train_mask)
    report = {
        "u_q05": u_q05,
        "u_q95": u_q95,
        "bars_q40": bars_q40,
        "bars_q80": bars_q80,
        "bars_q90": bars_q90,
        "train_target_hard_rate": _safe_mean(target.loc[train_mask, "target_hard"]),
        "train_target_soft_mean": _safe_mean(target.loc[train_mask, "target_soft"]),
    }
    return target, weights, report


def _rank_indices(score: pd.Series, frac: float, highest: bool) -> np.ndarray:
    score_s = _safe_numeric(score).reset_index(drop=True)
    valid = score_s.notna().to_numpy()
    if not bool(valid.any()):
        return np.array([], dtype=np.int64)
    valid_idx = np.flatnonzero(valid)
    k = min(max(1, int(math.ceil(float(frac) * len(valid_idx)))), len(valid_idx))
    values = score_s.iloc[valid_idx].to_numpy(dtype=np.float64)
    order = np.argsort(-values if highest else values, kind="mergesort")
    return valid_idx[order[:k]].astype(np.int64, copy=False)


def _side_summary(metrics: pd.DataFrame) -> dict[str, Any]:
    if "side" not in metrics.columns or metrics.empty:
        return {
            "top_side": "",
            "top_side_share": float("nan"),
            "long_share": float("nan"),
            "short_share": float("nan"),
        }
    side = _safe_numeric(metrics["side"]).dropna()
    if side.empty:
        return {
            "top_side": "",
            "top_side_share": float("nan"),
            "long_share": float("nan"),
            "short_share": float("nan"),
        }
    side_name = side.map(lambda value: "short" if value < 0.0 else "long")
    shares = side_name.value_counts(normalize=True)
    return {
        "top_side": str(shares.index[0]) if len(shares) else "",
        "top_side_share": float(shares.iloc[0]) if len(shares) else float("nan"),
        "long_share": float(side_name.eq("long").mean()),
        "short_share": float(side_name.eq("short").mean()),
    }


def _selection_summary(
    *,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    idx: np.ndarray,
) -> dict[str, Any]:
    selected_metrics = metrics.iloc[idx] if len(idx) else metrics.iloc[:0]
    selected_target = target.iloc[idx] if len(idx) else target.iloc[:0]
    valid_target_rate = _safe_mean(target["target_hard"])
    selected_rate = _safe_mean(selected_target.get("target_hard"))
    valid_timeout = _safe_mean(metrics["is_timeout"].astype(float))
    selected_timeout = _safe_mean(selected_metrics["is_timeout"].astype(float)) if len(selected_metrics) else float("nan")
    return {
        "selected_rows": int(len(idx)),
        "target_hard_rate": selected_rate,
        "target_soft_mean": _safe_mean(selected_target.get("target_soft")),
        "target_rate_lift_vs_valid": selected_rate / valid_target_rate
        if math.isfinite(selected_rate) and math.isfinite(valid_target_rate) and valid_target_rate > 0.0
        else float("nan"),
        "mean_u": _safe_mean(selected_metrics.get("u_policy_net")),
        "delta_mean_u_vs_valid": _safe_mean(selected_metrics.get("u_policy_net")) - _safe_mean(metrics["u_policy_net"])
        if len(selected_metrics)
        else float("nan"),
        "bad_mae_1r_rate": _safe_mean(selected_metrics.get("mae_norm") >= 1.0),
        "wide_barrier_25bps_rate": _safe_mean(selected_metrics.get("barrier") > 0.025),
        "timeout_rate": selected_timeout,
        "timeout_reduction_frac_vs_valid": (valid_timeout - selected_timeout) / valid_timeout
        if math.isfinite(valid_timeout) and valid_timeout > 0.0 and math.isfinite(selected_timeout)
        else float("nan"),
        **_side_summary(selected_metrics),
    }


def _feature_map(base_features: list[str], frame: pd.DataFrame, feature_sets: list[str]) -> dict[str, list[str]]:
    v2_features = [col for col in V2_FEATURE_COLS if col in frame.columns]
    out: dict[str, list[str]] = {}
    for feature_set in feature_sets:
        if feature_set == "base":
            out[feature_set] = list(base_features)
        elif feature_set == "base_plus_v2":
            out[feature_set] = list(dict.fromkeys(base_features + v2_features))
        elif feature_set == "v2_only":
            out[feature_set] = list(v2_features)
        else:
            raise ValueError(f"Unsupported feature set: {feature_set}")
    return out


def _month_masks(frame: pd.DataFrame, month: str, train_lookback_months: int | None) -> tuple[pd.Series, pd.Series]:
    month_period = frame["__ts__"].dt.tz_convert(None).dt.to_period("M").astype(str)
    train_mask = month_period < str(month)
    if train_lookback_months is not None and int(train_lookback_months) > 0:
        prior = sorted(month_period[train_mask].dropna().unique())
        keep = set(prior[-int(train_lookback_months) :])
        train_mask = train_mask & month_period.isin(keep)
    valid_mask = month_period.eq(str(month))
    return train_mask, valid_mask


def _aggregate(monthly: pd.DataFrame, *, expected_months: int) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["head", "head_kind", "direction", "feature_set", "selector", "top_frac"]
    for key, group in monthly.groupby(group_cols, dropna=False, observed=True):
        head, head_kind, direction, feature_set, selector, top_frac = key
        months = int(group["period"].nunique())
        score_ic_target = _safe_mean(group["score_ic_target"])
        score_ic_u = _safe_mean(group["score_ic_u"])
        auc = _safe_mean(group["target_auc"])
        pr_lift = _safe_mean(group["target_pr_auc_lift"])
        target_lift = _safe_mean(group["target_rate_lift_vs_valid"])
        timeout_reduction = _safe_mean(group["timeout_reduction_frac_vs_valid"])
        selected_u = _safe_mean(group["mean_u"])
        max_side_top_share = _safe_quantile(group.get("top_side_share", pd.Series(dtype=float)), 1.0)
        positive_model_months = int((_safe_numeric(group["mean_u"]) > 0.0).sum())
        ic_positive_months = int((_safe_numeric(group["score_ic_target"]) > 0.0).sum())
        risk_learnable = (
            direction == "risk"
            and months >= expected_months
            and ic_positive_months >= max(1, expected_months - 1)
            and (
                (math.isfinite(auc) and auc >= 0.53)
                or (math.isfinite(score_ic_target) and score_ic_target >= 0.03)
                or (math.isfinite(pr_lift) and pr_lift >= 1.10)
            )
        )
        positive_learnable = (
            direction == "positive"
            and months >= expected_months
            and ic_positive_months >= max(1, expected_months - 1)
            and math.isfinite(score_ic_target)
            and score_ic_target >= 0.03
        )
        if direction == "risk" and selector == "low_score_keep" and risk_learnable and timeout_reduction >= 0.10:
            decision = "risk_head_filter_candidate"
        elif direction == "risk" and selector == "high_score_top" and risk_learnable and target_lift >= 1.10:
            decision = "risk_head_learnable"
        elif direction == "positive" and selector == "high_score_top" and positive_learnable and selected_u > 0.0:
            decision = "positive_head_candidate"
        elif risk_learnable or positive_learnable:
            decision = "learnable_diagnostic_only"
        else:
            decision = "diagnostic_only"
        rows.append(
            {
                "decision": decision,
                "head": head,
                "head_kind": head_kind,
                "direction": direction,
                "feature_set": feature_set,
                "selector": selector,
                "top_frac": float(top_frac),
                "months": months,
                "ic_positive_months": ic_positive_months,
                "positive_model_months": positive_model_months,
                "valid_target_hard_rate": _safe_mean(group["valid_target_hard_rate"]),
                "score_ic_target": score_ic_target,
                "score_ic_u": score_ic_u,
                "score_ic_timeout": _safe_mean(group["score_ic_timeout"]),
                "score_ic_bad_mae": _safe_mean(group["score_ic_bad_mae"]),
                "score_ic_wide_barrier": _safe_mean(group["score_ic_wide_barrier"]),
                "target_auc": auc,
                "target_pr_auc_lift": pr_lift,
                "target_rate_lift_vs_valid": target_lift,
                "timeout_reduction_frac_vs_valid": timeout_reduction,
                "mean_u": selected_u,
                "delta_mean_u_vs_valid": _safe_mean(group["delta_mean_u_vs_valid"]),
                "bad_mae_1r_rate": _safe_mean(group["bad_mae_1r_rate"]),
                "wide_barrier_25bps_rate": _safe_mean(group["wide_barrier_25bps_rate"]),
                "timeout_rate": _safe_mean(group["timeout_rate"]),
                "max_side_top_share": max_side_top_share,
                "mean_selected_rows": _safe_mean(group["selected_rows"]),
                "min_selected_rows": _safe_quantile(group["selected_rows"], 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["decision", "head", "feature_set", "selector"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )


def _table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[col for col in cols if col in frame.columns]].copy()
    if limit is not None:
        view = view.head(limit)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
    return view.to_markdown(index=False)


def _write_report(output_dir: Path, aggregate: pd.DataFrame, monthly: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "source_archetype_v2_separated_learnability_report.md"
    cols = [
        "decision",
        "head",
        "feature_set",
        "selector",
        "top_frac",
        "months",
        "valid_target_hard_rate",
        "score_ic_target",
        "score_ic_u",
        "target_auc",
        "target_pr_auc_lift",
        "target_rate_lift_vs_valid",
        "timeout_reduction_frac_vs_valid",
        "mean_u",
        "delta_mean_u_vs_valid",
        "bad_mae_1r_rate",
        "wide_barrier_25bps_rate",
        "timeout_rate",
        "max_side_top_share",
        "mean_selected_rows",
    ]
    monthly_cols = [
        "period",
        "head",
        "feature_set",
        "selector",
        "valid_target_hard_rate",
        "score_ic_target",
        "score_ic_u",
        "target_auc",
        "target_pr_auc_lift",
        "target_rate_lift_vs_valid",
        "timeout_reduction_frac_vs_valid",
        "mean_u",
        "bad_mae_1r_rate",
        "wide_barrier_25bps_rate",
        "timeout_rate",
        "top_side",
        "top_side_share",
    ]
    lines = [
        "# Source Archetype V2 Separated Learnability",
        "",
        "Scope: diagnostic month-forward smoke models for separate utility, path-risk, wide-barrier, timeout/holding, and acceptable-path heads.",
        "",
        f"Rows: `{manifest['rows']}`",
        f"Months: `{', '.join(manifest['months'])}`",
        f"Feature sets: `{', '.join(manifest['feature_sets'])}`",
        f"Base feature count: `{manifest['base_feature_count']}`",
        f"V2 feature count: `{manifest['v2_feature_count']}`",
        f"Side counts: `{manifest.get('side_counts', {})}`",
        "",
        "## Decisions",
        "",
        _table(aggregate, cols, limit=120),
        "",
        "## Best By Head",
        "",
        _table(
            aggregate.sort_values(["head", "decision", "score_ic_target"], ascending=[True, True, False]),
            cols,
            limit=120,
        ),
        "",
        "## Monthly Detail",
        "",
        _table(monthly.sort_values(["period", "head", "feature_set", "selector"]), monthly_cols, limit=180),
        "",
        "## Outputs",
        "",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Aggregate: `{manifest['outputs']['aggregate']}`",
        f"- Diagnostics: `{manifest['outputs']['diagnostics']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    archetypes_path: Path,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    months: list[str],
    heads: list[str],
    feature_sets: list[str],
    top_fracs: list[float],
    seeds: list[int],
    train_lookback_months: int | None,
    min_train_rows: int,
    min_valid_rows: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    head_specs = _head_specs_by_name(heads)
    frame, join_report = _load_joined_frame(quality_labels_path=archetypes_path, labels_path=labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    for col in feature_matrix.columns:
        frame[col] = feature_matrix[col].to_numpy(dtype=np.float32, copy=False)
    for col in V2_FEATURE_COLS:
        if col in frame.columns:
            frame[col] = _safe_numeric(frame[col]).fillna(0.0).astype(np.float32)
    metrics = _path_metrics(frame)
    base_features = list(feature_matrix.columns)
    fmap = _feature_map(base_features, frame, feature_sets)

    monthly_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for month in months:
        train_mask, valid_mask = _month_masks(frame, month, train_lookback_months)
        if int(valid_mask.sum()) < min_valid_rows:
            diagnostics.append({"period": month, "skipped": True, "reason": "too_few_valid_rows"})
            continue
        for spec in head_specs:
            target, weights, target_report = _build_month_target(metrics, train_mask, spec)
            train_label_mask = train_mask & target["target_soft"].notna() & weights.gt(0.0)
            valid_label_mask = valid_mask & target["target_soft"].notna()
            if int(train_label_mask.sum()) < min_train_rows:
                diagnostics.append(
                    {
                        "period": month,
                        "head": spec.name,
                        "skipped": True,
                        "reason": "too_few_train_rows",
                        "train_rows": int(train_label_mask.sum()),
                        "valid_rows": int(valid_label_mask.sum()),
                        **target_report,
                    }
                )
                continue
            valid_frame = frame.loc[valid_label_mask].reset_index(drop=True)
            valid_metrics = metrics.loc[valid_label_mask].reset_index(drop=True)
            valid_target = target.loc[valid_label_mask].reset_index(drop=True)
            valid_prevalence = _safe_mean(valid_target["target_hard"])
            for feature_set, features in fmap.items():
                if not features:
                    diagnostics.append(
                        {
                            "period": month,
                            "head": spec.name,
                            "feature_set": feature_set,
                            "skipped": True,
                            "reason": "empty_feature_set",
                        }
                    )
                    continue
                x_train, x_valid = _month_model_frame(
                    frame,
                    train_mask=train_label_mask,
                    valid_mask=valid_label_mask,
                    features=features,
                )
                pred_matrix = np.vstack(
                    [
                        _fit_predict(
                            x_train=x_train,
                            y_train=target.loc[train_label_mask, "target_soft"],
                            w_train=weights.loc[train_label_mask],
                            x_valid=x_valid,
                            seed=seed,
                        )
                        for seed in seeds
                    ]
                )
                score = pd.Series(np.mean(pred_matrix, axis=0).astype(np.float32), index=valid_frame.index)
                ap = _safe_ap(valid_target["target_hard"], score)
                auc = _safe_auc(valid_target["target_hard"], score)
                diag_base = {
                    "period": month,
                    "head": spec.name,
                    "head_kind": spec.kind,
                    "direction": spec.direction,
                    "feature_set": feature_set,
                    "skipped": False,
                    "train_rows": int(train_label_mask.sum()),
                    "valid_rows": int(valid_label_mask.sum()),
                    "model_feature_count": int(len(features)),
                    "valid_target_hard_rate": valid_prevalence,
                    "valid_target_soft_mean": _safe_mean(valid_target["target_soft"]),
                    "score_ic_target": _spearman(score, valid_target["target_soft"]),
                    "score_ic_u": _spearman(score, valid_metrics["u_policy_net"]),
                    "score_ic_timeout": _spearman(score, valid_metrics["is_timeout"].astype(float)),
                    "score_ic_bad_mae": _spearman(score, valid_metrics["mae_norm"] >= 1.0),
                    "score_ic_wide_barrier": _spearman(score, valid_metrics["barrier"] > 0.025),
                    "target_auc": auc,
                    "target_average_precision": ap,
                    "target_pr_auc_lift": ap / valid_prevalence
                    if math.isfinite(ap) and math.isfinite(valid_prevalence) and valid_prevalence > 0.0
                    else float("nan"),
                    "prediction_seed_std_mean": float(np.std(pred_matrix, axis=0).mean()) if pred_matrix.size else float("nan"),
                    **target_report,
                }
                diagnostics.append(diag_base)
                for top_frac in top_fracs:
                    selectors = [("high_score_top", True)]
                    if spec.direction == "risk":
                        selectors.append(("low_score_keep", False))
                    for selector, highest in selectors:
                        idx = _rank_indices(score, top_frac, highest=highest)
                        row = {
                            **diag_base,
                            "selector": selector,
                            "top_frac": float(top_frac),
                            **_selection_summary(metrics=valid_metrics, target=valid_target, score=score, idx=idx),
                        }
                        monthly_rows.append(row)

    monthly = pd.DataFrame(monthly_rows)
    diagnostics_frame = pd.DataFrame(diagnostics)
    aggregate = _aggregate(monthly, expected_months=len(months))

    paths = {
        "monthly": output_dir / "source_archetype_v2_separated_learnability_monthly.csv",
        "aggregate": output_dir / "source_archetype_v2_separated_learnability_aggregate.csv",
        "diagnostics": output_dir / "source_archetype_v2_separated_learnability_diagnostics.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    diagnostics_frame.to_csv(paths["diagnostics"], index=False)
    manifest = {
        "scope": "source_archetype_v2_separated_learnability",
        "archetypes_path": str(archetypes_path),
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "months": list(months),
        "heads": [spec.name for spec in head_specs],
        "feature_sets": feature_sets,
        "top_fracs": [float(v) for v in top_fracs],
        "seeds": [int(seed) for seed in seeds],
        "join_report": join_report,
        "feature_store": feature_report,
        "side_counts": {
            "long": int((_safe_numeric(frame["side"]) > 0.0).sum()),
            "short": int((_safe_numeric(frame["side"]) < 0.0).sum()),
        }
        if "side" in frame.columns
        else {},
        "base_feature_count": int(len(base_features)),
        "v2_feature_count": int(len([col for col in V2_FEATURE_COLS if col in frame.columns])),
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_report(output_dir, aggregate, monthly, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def _parse_float_csv(value: str | None, default: tuple[float, ...]) -> list[float]:
    if value is None or not str(value).strip():
        return list(default)
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archetypes-path", type=Path, default=DEFAULT_ARCHETYPES)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=96)
    parser.add_argument("--months", type=str, default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--heads", type=str, default=",".join(DEFAULT_HEADS))
    parser.add_argument("--feature-sets", type=str, default=",".join(DEFAULT_FEATURE_SETS))
    parser.add_argument("--top-fracs", type=str, default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--train-lookback-months", type=int, default=None)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--min-valid-rows", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        archetypes_path=args.archetypes_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        heads=_parse_csv(args.heads, DEFAULT_HEADS),
        feature_sets=_parse_csv(args.feature_sets, DEFAULT_FEATURE_SETS),
        top_fracs=_parse_float_csv(args.top_fracs, DEFAULT_TOP_FRACS),
        seeds=_parse_int_csv(args.seeds, DEFAULT_SEEDS),
        train_lookback_months=args.train_lookback_months,
        min_train_rows=int(args.min_train_rows),
        min_valid_rows=int(args.min_valid_rows),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
