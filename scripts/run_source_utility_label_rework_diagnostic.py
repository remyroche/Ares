#!/usr/bin/env python3
"""Walk-forward diagnostics for utility-first source-aware label reworks.

The current v17 source-quality labels are economically positive under an oracle
ranking, but the month-forward smoke model often learns label structure that
does not translate to realized utility. This script tests stricter utility-first
label targets without modifying the materialized label artifact or production
training code.

For each validation month, all label thresholds are calibrated only from prior
months. Future outcomes are used only as supervised diagnostic targets and OOS
evaluation metrics.
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

from scripts.run_label_feature_store_model_smoke import (  # noqa: E402
    _fit_predict,
    _month_model_frame,
    _timestamp_ranking_metrics,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    _decile_diagnostics,
    _json_safe,
    _load_feature_store_columns,
    _path_metrics,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _selection_metrics,
    _spearman,
)
from scripts.run_source_quality_label_walkforward_ablation import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_MONTHS,
    DEFAULT_QUALITY_LABELS,
    DEFAULT_SEEDS,
    _load_joined_frame,
    _parse_csv,
    _parse_float_csv,
    _parse_int_csv,
    _source_feature_columns,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/source_quality_label_walkforward_ablation_v1_sideaware_20260702/utility_label_rework")
DEFAULT_TOP_FRACS = (0.01, 0.03, 0.05, 0.10)


@dataclass(frozen=True)
class UtilityLabelSpec:
    name: str
    mode: str
    source_conditioned: bool
    good_q: float = 0.70
    bad_q: float = 0.40
    low_q: float = 0.05
    high_q: float = 0.95
    min_group_rows: int = 200
    min_good_utility: float = 0.0
    path_guard: bool = False
    extreme_weight: bool = False


LABEL_SPECS = (
    UtilityLabelSpec(
        name="utility_linear_global_q70_v1",
        mode="linear",
        source_conditioned=False,
        good_q=0.70,
        bad_q=0.40,
        extreme_weight=False,
    ),
    UtilityLabelSpec(
        name="utility_linear_source_q70_v1",
        mode="linear",
        source_conditioned=True,
        good_q=0.70,
        bad_q=0.40,
        extreme_weight=False,
    ),
    UtilityLabelSpec(
        name="utility_linear_source_q80_v1",
        mode="linear",
        source_conditioned=True,
        good_q=0.80,
        bad_q=0.50,
        extreme_weight=True,
    ),
    UtilityLabelSpec(
        name="utility_sign_source_margin_v1",
        mode="sigmoid",
        source_conditioned=True,
        good_q=0.60,
        bad_q=0.40,
        extreme_weight=False,
    ),
    UtilityLabelSpec(
        name="utility_path_guard_source_q70_v1",
        mode="linear",
        source_conditioned=True,
        good_q=0.70,
        bad_q=0.40,
        path_guard=True,
        extreme_weight=True,
    ),
)


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _sigmoid(values: pd.Series) -> pd.Series:
    clipped = values.clip(lower=-20.0, upper=20.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _bool_metric(metrics: pd.DataFrame, col: str) -> pd.Series:
    if col not in metrics.columns:
        return pd.Series(False, index=metrics.index)
    values = metrics[col]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return _safe_numeric(values).fillna(0.0).gt(0.5)


def _group_keys(frame: pd.DataFrame, source_conditioned: bool) -> pd.Series:
    if source_conditioned and "primary_source_tag" in frame.columns:
        return frame["primary_source_tag"].fillna("unknown").astype(str)
    return pd.Series("all", index=frame.index, dtype=object)


def _thresholds_for_spec(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    train_mask: pd.Series,
    spec: UtilityLabelSpec,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    utility = _safe_numeric(metrics["u_policy_net"])
    groups = _group_keys(frame, spec.source_conditioned)
    qs = sorted(set([spec.low_q, spec.bad_q, 0.50, spec.good_q, spec.high_q]))

    def quantiles(values: pd.Series) -> dict[str, float]:
        finite = values.replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty:
            return {f"q{int(round(q * 100)):02d}": float("nan") for q in qs}
        return {f"q{int(round(q * 100)):02d}": float(finite.quantile(q)) for q in qs}

    global_thresholds = quantiles(utility.loc[train_mask])
    thresholds: dict[str, dict[str, float]] = {}
    train_groups = groups.loc[train_mask]
    for group_value, idx in train_groups.groupby(train_groups, sort=False).groups.items():
        if len(idx) < int(spec.min_group_rows):
            continue
        thresholds[str(group_value)] = quantiles(utility.loc[idx])
    return thresholds, global_thresholds


def _threshold_series(
    *,
    frame: pd.DataFrame,
    spec: UtilityLabelSpec,
    thresholds: dict[str, dict[str, float]],
    global_thresholds: dict[str, float],
    key: str,
) -> pd.Series:
    groups = _group_keys(frame, spec.source_conditioned)
    fallback = float(global_thresholds.get(key, float("nan")))
    values = np.full(len(frame), fallback, dtype=np.float32)
    for group_value, threshold_map in thresholds.items():
        threshold = float(threshold_map.get(key, fallback))
        if not math.isfinite(threshold):
            threshold = fallback
        positions = groups.eq(str(group_value)).to_numpy()
        values[positions] = threshold
    return pd.Series(values, index=frame.index)


def _build_target(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    train_mask: pd.Series,
    valid_mask: pd.Series,
    spec: UtilityLabelSpec,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    thresholds, global_thresholds = _thresholds_for_spec(
        frame=frame,
        metrics=metrics,
        train_mask=train_mask,
        spec=spec,
    )
    eval_mask = train_mask | valid_mask
    utility = _safe_numeric(metrics["u_policy_net"])
    low_key = f"q{int(round(spec.low_q * 100)):02d}"
    high_key = f"q{int(round(spec.high_q * 100)):02d}"
    good_key = f"q{int(round(spec.good_q * 100)):02d}"
    bad_key = f"q{int(round(spec.bad_q * 100)):02d}"
    median_key = "q50"

    low = _threshold_series(frame=frame, spec=spec, thresholds=thresholds, global_thresholds=global_thresholds, key=low_key)
    high = _threshold_series(frame=frame, spec=spec, thresholds=thresholds, global_thresholds=global_thresholds, key=high_key)
    good = _threshold_series(frame=frame, spec=spec, thresholds=thresholds, global_thresholds=global_thresholds, key=good_key)
    bad = _threshold_series(frame=frame, spec=spec, thresholds=thresholds, global_thresholds=global_thresholds, key=bad_key)
    median = _threshold_series(frame=frame, spec=spec, thresholds=thresholds, global_thresholds=global_thresholds, key=median_key)

    spread = (high - low).replace(0.0, np.nan)
    if spec.mode == "sigmoid":
        scale = (high - low).abs().replace(0.0, np.nan) / 4.0
        target_soft = _sigmoid((utility - median) / scale)
    else:
        target_soft = ((utility - low) / spread).clip(0.0, 1.0)
    target_soft = target_soft.where(eval_mask & utility.notna())

    good_cutoff = pd.concat([good, pd.Series(float(spec.min_good_utility), index=frame.index)], axis=1).max(axis=1)
    target_hard = (utility >= good_cutoff).astype(float)
    bad_flag = (utility <= bad) | (utility < 0.0)

    mae_norm = _safe_numeric(metrics["mae_norm"]) if "mae_norm" in metrics.columns else pd.Series(np.nan, index=frame.index)
    barrier = _safe_numeric(metrics["barrier"]) if "barrier" in metrics.columns else pd.Series(np.nan, index=frame.index)
    timeout = _bool_metric(metrics, "is_timeout")
    path_failure = mae_norm.ge(1.0) | (timeout & utility.le(0.0))
    if spec.path_guard:
        good_path = mae_norm.lt(0.75) & (~timeout) & barrier.le(0.025)
        target_hard = (target_hard.gt(0.5) & good_path).astype(float)
        target_soft = target_soft.where(~path_failure, 0.0)
        bad_flag = bad_flag | path_failure | barrier.gt(0.025)

    target_hard = target_hard.where(eval_mask & utility.notna())
    target = pd.DataFrame({"target_soft": target_soft, "target_hard": target_hard}, index=frame.index)

    weights = pd.Series(1.0, index=frame.index, dtype=np.float32)
    weights[~eval_mask | utility.isna()] = 0.0
    if spec.extreme_weight:
        distance = (target_soft - 0.5).abs().fillna(0.0)
        weights *= (0.50 + 2.0 * distance).clip(0.25, 1.50).astype(np.float32)
    weights[bad_flag & eval_mask & utility.notna()] = np.maximum(weights[bad_flag & eval_mask & utility.notna()], 1.0)

    report = {
        "source_conditioned": bool(spec.source_conditioned),
        "mode": spec.mode,
        "path_guard": bool(spec.path_guard),
        "groups_with_thresholds": int(len(thresholds)),
        "global_thresholds": global_thresholds,
    }
    return target, weights.astype(np.float32), report


def _bucket_masks(frame: pd.DataFrame, valid_mask: pd.Series, min_bucket_rows: int) -> list[tuple[str, pd.Series]]:
    masks: list[tuple[str, pd.Series]] = [("all_rows", valid_mask.copy())]
    if "primary_source_tag" not in frame.columns:
        return masks
    groups = frame.loc[valid_mask, "primary_source_tag"].fillna("unknown").astype(str)
    for tag, idx in groups.groupby(groups, sort=True).groups.items():
        mask = pd.Series(False, index=frame.index)
        mask.loc[idx] = True
        if int(mask.sum()) >= int(min_bucket_rows):
            masks.append((str(tag), valid_mask & mask))
    return masks


def _score_rows(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    month: str,
    label_name: str,
    feature_set: str,
    ranker: str,
    valid_mask: pd.Series,
    top_fracs: list[float],
    min_bucket_rows: int,
    train_rows: int,
    model_feature_count: int,
    label_report: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_bucket, bucket_mask in _bucket_masks(frame, valid_mask, min_bucket_rows):
        valid_frame = frame.loc[bucket_mask].reset_index(drop=True)
        valid_metrics = metrics.loc[bucket_mask].reset_index(drop=True)
        valid_target = target.loc[bucket_mask].reset_index(drop=True)
        valid_score = score.loc[bucket_mask].reset_index(drop=True)
        if len(valid_frame) < int(min_bucket_rows):
            continue
        decile = _decile_diagnostics(valid_score, valid_metrics["u_policy_net"])
        ts_rank = _timestamp_ranking_metrics(
            frame=valid_frame,
            metrics=valid_metrics,
            target=valid_target,
            score=valid_score,
        )
        for top_frac in top_fracs:
            row = _selection_metrics(
                frame=valid_frame,
                metrics=valid_metrics,
                target=valid_target,
                score=valid_score,
                arm=label_name,
                selector=ranker,
                period=month,
                top_frac=top_frac,
            )
            row.update(
                {
                    "label": label_name,
                    "feature_set": feature_set,
                    "ranker": ranker,
                    "source_bucket": source_bucket,
                    "train_rows": int(train_rows),
                    "valid_bucket_rows": int(len(valid_frame)),
                    "model_feature_count": int(model_feature_count),
                    "score_ic_label": _spearman(valid_score, valid_target["target_soft"]),
                    "score_ic_u": _spearman(valid_score, valid_metrics["u_policy_net"]),
                    "target_ic_u": _spearman(valid_target["target_soft"], valid_metrics["u_policy_net"]),
                    "label_source_conditioned": bool(label_report.get("source_conditioned", False)),
                    "label_mode": label_report.get("mode", ""),
                    "label_path_guard": bool(label_report.get("path_guard", False)),
                    "label_threshold_groups": int(label_report.get("groups_with_thresholds", 0)),
                    **decile,
                    **ts_rank,
                }
            )
            rows.append(row)
    return rows


def _diagnosis(group: pd.DataFrame) -> str:
    model = group[group["ranker"].eq("model_score")]
    oracle = group[group["ranker"].eq("target_oracle")]
    if model.empty or oracle.empty:
        return "missing_ranker"
    oracle_u = _safe_mean(oracle["mean_u"])
    model_u = _safe_mean(model["mean_u"])
    model_ic_u = _safe_mean(model["score_ic_u"])
    model_ic_label = _safe_mean(model["score_ic_label"])
    target_ic_u = _safe_mean(model["target_ic_u"])
    if math.isfinite(target_ic_u) and target_ic_u <= 0.0 and math.isfinite(oracle_u) and oracle_u <= 0.0:
        return "label_not_utility_aligned"
    if math.isfinite(model_ic_label) and model_ic_label > 0.0 and math.isfinite(model_ic_u) and model_ic_u < 0.0:
        return "model_learns_label_but_not_utility"
    if math.isfinite(model_ic_label) and model_ic_label < 0.0:
        return "model_anti_learns_label"
    if math.isfinite(oracle_u) and oracle_u > 0.0 and math.isfinite(model_u) and model_u <= 0.0:
        return "model_selection_failure"
    if math.isfinite(model_u) and model_u > 0.0 and math.isfinite(model_ic_u) and model_ic_u > 0.0:
        return "promising"
    return "weak_or_mixed"


def _summarize(monthly: pd.DataFrame, *, expected_months: int) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["label", "feature_set", "source_bucket", "top_frac"]
    for key, group in monthly.groupby(group_cols, dropna=False, observed=True):
        label, feature_set, source_bucket, top_frac = key
        model = group[group["ranker"].eq("model_score")]
        oracle = group[group["ranker"].eq("target_oracle")]
        month_count = int(group["period"].nunique())
        model_mean_u = _safe_mean(model["mean_u"])
        model_ic_u = _safe_mean(model["score_ic_u"])
        mean_bad_mae = _safe_mean(model["bad_mae_1r_rate"])
        mean_timeout = _safe_mean(model["timeout_rate"])
        mean_wide_barrier = _safe_mean(model["wide_barrier_25bps_rate"])
        mean_selected_rows = _safe_mean(model["selected_rows"])
        min_selected_rows = _safe_quantile(model["selected_rows"], 0.0)
        positive_months = int((_safe_numeric(model["mean_u"]) > 0.0).sum())
        ic_positive_months = int((_safe_numeric(model["score_ic_u"]) > 0.0).sum())
        oracle_positive_months = int((_safe_numeric(oracle["mean_u"]) > 0.0).sum())
        learnable = (
            month_count >= int(expected_months)
            and positive_months >= int(expected_months)
            and ic_positive_months >= int(expected_months)
            and oracle_positive_months >= int(expected_months)
            and math.isfinite(model_mean_u)
            and model_mean_u > 0.0
            and math.isfinite(model_ic_u)
            and model_ic_u > 0.0
        )
        size_ok = float(top_frac) >= 0.03 and math.isfinite(min_selected_rows) and min_selected_rows >= 25.0
        risk_ok = (
            math.isfinite(mean_bad_mae)
            and mean_bad_mae <= 0.65
            and math.isfinite(mean_timeout)
            and mean_timeout <= 0.20
            and math.isfinite(mean_wide_barrier)
            and mean_wide_barrier <= 0.25
        )
        if learnable and size_ok and risk_ok:
            decision = "candidate_for_materialized_label_ablation"
        elif learnable and risk_ok and not size_ok:
            decision = "narrow_gate_candidate_only"
        elif learnable and not risk_ok:
            decision = "learnable_but_economic_limits_fail"
        else:
            decision = "diagnostic_only"
        rows.append(
            {
                "label": label,
                "feature_set": feature_set,
                "source_bucket": source_bucket,
                "top_frac": float(top_frac),
                "months": month_count,
                "model_positive_months": positive_months,
                "model_ic_u_positive_months": ic_positive_months,
                "oracle_positive_months": oracle_positive_months,
                "mean_oracle_u": _safe_mean(oracle["mean_u"]),
                "mean_model_u": model_mean_u,
                "worst_model_month_u": _safe_quantile(model["mean_u"], 0.0),
                "mean_model_ic_label": _safe_mean(model["score_ic_label"]),
                "mean_model_ic_u": model_ic_u,
                "mean_target_ic_u": _safe_mean(model["target_ic_u"]),
                "mean_model_bad_mae_1r_rate": mean_bad_mae,
                "mean_model_timeout_rate": mean_timeout,
                "mean_model_wide_barrier_25bps_rate": mean_wide_barrier,
                "mean_selected_rows": mean_selected_rows,
                "min_selected_rows": min_selected_rows,
                "mean_ts_rank_hr30_u": _safe_mean(model["ts_rank_hr30_u"]),
                "mean_ts_rank_ndcg30_u": _safe_mean(model["ts_rank_ndcg30_u"]),
                "diagnosis": _diagnosis(group),
                "decision": decision,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["top_frac", "decision", "mean_model_u", "mean_model_ic_u"],
        ascending=[True, True, False, False],
        na_position="last",
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
    path = output_dir / "source_utility_label_rework_diagnostic.md"
    top10 = aggregate[aggregate["top_frac"].eq(0.10)].copy()
    all_rows = top10[top10["source_bucket"].eq("all_rows")]
    candidates = aggregate[aggregate["decision"].eq("candidate_for_materialized_label_ablation")]
    cols = [
        "decision",
        "label",
        "feature_set",
        "source_bucket",
        "top_frac",
        "months",
        "model_positive_months",
        "model_ic_u_positive_months",
        "oracle_positive_months",
        "mean_oracle_u",
        "mean_model_u",
        "worst_model_month_u",
        "mean_model_ic_label",
        "mean_model_ic_u",
        "mean_target_ic_u",
        "mean_model_bad_mae_1r_rate",
        "mean_model_timeout_rate",
        "mean_model_wide_barrier_25bps_rate",
        "mean_selected_rows",
        "min_selected_rows",
        "diagnosis",
    ]
    lines = [
        "# Source Utility Label Rework Diagnostic",
        "",
        "Scope: diagnostic utility-first source-aware label targets. Thresholds are calibrated from prior months only.",
        "",
        f"Rows joined to label ledger: `{manifest['rows']}`",
        f"Utility source: `{manifest.get('utility_source', '')}`",
        f"Months: `{', '.join(manifest['months'])}`",
        f"Feature sets: `{', '.join(manifest['feature_sets'])}`",
        "",
        "## Candidates",
        "",
        _table(candidates, cols, limit=80),
        "",
        "## Top 10% All-Rows View",
        "",
        _table(all_rows, cols, limit=120),
        "",
        "## Best Top 10% Source Buckets",
        "",
        _table(
            top10[~top10["source_bucket"].eq("all_rows")].sort_values("mean_model_u", ascending=False),
            cols,
            limit=120,
        ),
        "",
        "## Monthly Top 10% Model Detail",
        "",
        _table(
            monthly[(monthly["top_frac"].eq(0.10)) & (monthly["ranker"].eq("model_score"))].sort_values(
                ["period", "label", "feature_set", "source_bucket"]
            ),
            [
                "period",
                "label",
                "feature_set",
                "source_bucket",
                "selected_rows",
                "mean_u",
                "hit_u",
                "score_ic_label",
                "score_ic_u",
                "target_ic_u",
                "bad_mae_1r_rate",
                "timeout_rate",
                "wide_barrier_25bps_rate",
            ],
            limit=160,
        ),
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


def run_diagnostic(
    *,
    quality_labels_path: Path,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    months: list[str],
    top_fracs: list[float],
    seeds: list[int],
    feature_sets: list[str],
    train_lookback_months: int | None,
    min_train_rows: int,
    min_valid_rows: int,
    min_bucket_rows: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, join_report = _load_joined_frame(quality_labels_path=quality_labels_path, labels_path=labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    for col in feature_matrix.columns:
        frame[col] = feature_matrix[col].to_numpy(dtype=np.float32, copy=False)
    metrics = _path_metrics(frame)
    base_features = list(feature_matrix.columns)
    source_features = _source_feature_columns(frame)
    feature_map = {"base": base_features, "base_plus_source": list(dict.fromkeys(base_features + source_features))}

    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    for month in months:
        valid_mask = month_period.eq(month)
        train_mask = month_period < month
        if train_lookback_months is not None and int(train_lookback_months) > 0:
            prior_months = sorted(month_period[train_mask].dropna().unique())
            keep = set(prior_months[-int(train_lookback_months) :])
            train_mask = train_mask & month_period.isin(keep)
        if int(valid_mask.sum()) < int(min_valid_rows):
            continue
        for spec in LABEL_SPECS:
            target, weights, label_report = _build_target(
                frame=frame,
                metrics=metrics,
                train_mask=train_mask,
                valid_mask=valid_mask,
                spec=spec,
            )
            train_target_mask = train_mask & target["target_soft"].notna() & weights.gt(0.0)
            if int(train_target_mask.sum()) < int(min_train_rows):
                diag_rows.append(
                    {
                        "label": spec.name,
                        "period": month,
                        "skipped": True,
                        "reason": "too_few_train_rows",
                        "train_rows": int(train_target_mask.sum()),
                        "valid_rows": int(valid_mask.sum()),
                        **label_report,
                    }
                )
                continue
            for feature_set in feature_sets:
                features = feature_map.get(feature_set)
                if not features:
                    continue
                x_train, x_valid = _month_model_frame(
                    frame,
                    train_mask=train_target_mask,
                    valid_mask=valid_mask,
                    features=features,
                )
                pred_matrix = np.vstack(
                    [
                        _fit_predict(
                            x_train=x_train,
                            y_train=target.loc[train_target_mask, "target_soft"],
                            w_train=weights.loc[train_target_mask],
                            x_valid=x_valid,
                            seed=seed,
                        )
                        for seed in seeds
                    ]
                )
                model_score_valid = pd.Series(
                    np.mean(pred_matrix, axis=0).astype(np.float32),
                    index=frame.loc[valid_mask].index,
                )
                model_score = pd.Series(np.nan, index=frame.index, dtype=np.float32)
                model_score.loc[valid_mask] = model_score_valid
                target_score = target["target_soft"].copy()
                for ranker, score in {"model_score": model_score, "target_oracle": target_score}.items():
                    rows.extend(
                        _score_rows(
                            frame=frame,
                            metrics=metrics,
                            target=target,
                            score=score,
                            month=month,
                            label_name=spec.name,
                            feature_set=feature_set,
                            ranker=ranker,
                            valid_mask=valid_mask,
                            top_fracs=top_fracs,
                            min_bucket_rows=min_bucket_rows,
                            train_rows=int(train_target_mask.sum()),
                            model_feature_count=int(len(features)),
                            label_report=label_report,
                        )
                    )
                diag_rows.append(
                    {
                        "label": spec.name,
                        "period": month,
                        "feature_set": feature_set,
                        "skipped": False,
                        "train_rows": int(train_target_mask.sum()),
                        "valid_rows": int(valid_mask.sum()),
                        "model_feature_count": int(len(features)),
                        "target_train_mean": _safe_mean(target.loc[train_target_mask, "target_soft"]),
                        "target_valid_mean": _safe_mean(target.loc[valid_mask, "target_soft"]),
                        "target_valid_ic_u": _spearman(
                            target.loc[valid_mask, "target_soft"],
                            metrics.loc[valid_mask, "u_policy_net"],
                        ),
                        "model_valid_ic_label": _spearman(
                            model_score.loc[valid_mask],
                            target.loc[valid_mask, "target_soft"],
                        ),
                        "model_valid_ic_u": _spearman(
                            model_score.loc[valid_mask],
                            metrics.loc[valid_mask, "u_policy_net"],
                        ),
                        **label_report,
                    }
                )

    monthly = pd.DataFrame(rows)
    aggregate = _summarize(monthly, expected_months=len(months))
    diagnostics = pd.DataFrame(diag_rows)
    paths = {
        "monthly": output_dir / "source_utility_label_rework_monthly.csv",
        "aggregate": output_dir / "source_utility_label_rework_aggregate.csv",
        "diagnostics": output_dir / "source_utility_label_rework_diagnostics.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    manifest = {
        "scope": "source_utility_label_rework_diagnostic",
        "quality_labels_path": str(quality_labels_path),
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "utility_source": metrics.attrs.get("utility_source"),
        "months": list(months),
        "top_fracs": [float(v) for v in top_fracs],
        "seeds": [int(seed) for seed in seeds],
        "feature_sets": list(feature_sets),
        "label_specs": [spec.__dict__ for spec in LABEL_SPECS],
        "join_report": join_report,
        "feature_store": feature_report,
        "base_feature_count": int(len(base_features)),
        "source_feature_count": int(len(source_features)),
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_report(output_dir, aggregate, monthly, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-labels-path", type=Path, default=DEFAULT_QUALITY_LABELS)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--label-ablation-manifest", type=Path, default=DEFAULT_MANIFEST, help="Reserved for provenance.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=96)
    parser.add_argument("--months", type=str, default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--top-fracs", type=str, default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--feature-sets", type=str, default="base,base_plus_source")
    parser.add_argument("--train-lookback-months", type=int, default=None)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--min-valid-rows", type=int, default=100)
    parser.add_argument("--min-bucket-rows", type=int, default=80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_diagnostic(
        quality_labels_path=args.quality_labels_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        top_fracs=_parse_float_csv(args.top_fracs, DEFAULT_TOP_FRACS),
        seeds=_parse_int_csv(args.seeds, DEFAULT_SEEDS),
        feature_sets=_parse_csv(args.feature_sets, ("base", "base_plus_source")),
        train_lookback_months=args.train_lookback_months,
        min_train_rows=int(args.min_train_rows),
        min_valid_rows=int(args.min_valid_rows),
        min_bucket_rows=int(args.min_bucket_rows),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
