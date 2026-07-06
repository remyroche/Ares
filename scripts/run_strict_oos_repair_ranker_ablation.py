#!/usr/bin/env python3
"""Strict OOS repair-ranker diagnostic.

This script tests whether prior-month oracle/proxy mistakes contain a causal
feature signal that improves next-month row selection. It is diagnostic-only:
it does not tune production models, alter training labels, or promote a gate.

For each source bucket, proxy score, and top-k fraction:

1. train on prior-month strict OOS event rows where
   - 1 = oracle top-k row missed by the proxy,
   - 0 = proxy top-k row outside oracle top-k;
2. score the full eligible strict OOS universe in the next month;
3. compare repair top-k to the original proxy top-k and realized oracle top-k.

Only frozen feature-store columns are used by default. Optional proxy rank is
allowed as a prediction-time context feature, ranked only inside the current
month/source universe.
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
from sklearn.ensemble import ExtraTreesRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_strict_oos_source_label_proxy_diagnostic import (  # noqa: E402
    DEFAULT_MONTHS,
    DEFAULT_PREDICTIONS,
    DEFAULT_PROXY_COLUMNS,
    DEFAULT_QUALITY_LABELS,
    DEFAULT_TOP_FRACS,
    LABEL_SPECS,
    SOURCE_BUCKETS,
    _join_strict_oos,
    _rank_top_indices,
    _source_bucket_mask,
    _utility_stats,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    _json_safe,
    _load_feature_store_columns,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _spearman,
)
from scripts.run_source_quality_label_walkforward_ablation import (  # noqa: E402
    _parse_csv,
    _parse_float_csv,
)


DEFAULT_EVENT_ROWS = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "strict_oos_proxy_missed_opportunities/strict_oos_proxy_oracle_gap_event_rows.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "strict_oos_repair_ranker_ablation"
)
MISSED_EVENT_TYPE = "missed_oracle_topk"
FALSE_POSITIVE_EVENT_TYPE = "proxy_false_positive_vs_oracle"
DEFAULT_FEATURE_MODES = ("frozen_features", "frozen_features_plus_proxy_rank")
DEFAULT_SELECTION_METHODS = ("repair_score", "repair_proxy_blend_70_30")


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _pct_rank_high(score: Any) -> pd.Series:
    values = _safe_numeric(score)
    ranks = values.rank(method="average", pct=True)
    return ranks.where(values.notna(), np.nan)


def _label_col(name: str) -> str:
    for spec in LABEL_SPECS:
        if spec.name == name:
            return spec.column
    raise ValueError(f"Unknown label name: {name}")


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


def _load_events(
    path: Path,
    *,
    months: list[str],
    source_buckets: list[str],
    proxy_cols: list[str],
    top_fracs: list[float],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    events = pd.read_csv(path)
    required = {
        "period",
        "source_bucket",
        "proxy_col",
        "top_frac",
        "event_type",
        "__strict_oos_row_id",
        "__ts__",
        "__symbol__",
    }
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    top_frac_set = {round(float(v), 6) for v in top_fracs}
    events["top_frac"] = _safe_numeric(events["top_frac"]).round(6)
    events["period"] = events["period"].astype(str)
    events["__ts__"] = pd.to_datetime(events["__ts__"], utc=True, errors="coerce")
    mask = events["period"].isin(set(months))
    mask &= events["source_bucket"].isin(set(source_buckets))
    mask &= events["proxy_col"].isin(set(proxy_cols))
    mask &= events["top_frac"].isin(top_frac_set)
    mask &= events["event_type"].isin({MISSED_EVENT_TYPE, FALSE_POSITIVE_EVENT_TYPE})
    events = events.loc[mask].copy()
    dedupe_cols = ["period", "source_bucket", "proxy_col", "top_frac", "event_type", "__strict_oos_row_id"]
    events = events.drop_duplicates(dedupe_cols, keep="first").reset_index(drop=True)
    events["repair_target"] = events["event_type"].eq(MISSED_EVENT_TYPE).astype(np.float32)
    return events


def _validate_event_alignment(strict: pd.DataFrame, events: pd.DataFrame) -> dict[str, Any]:
    keys = strict[["__strict_oos_row_id", "__ts__", "__symbol__"]].copy()
    merged = events[["__strict_oos_row_id", "__ts__", "__symbol__"]].drop_duplicates().merge(
        keys,
        on="__strict_oos_row_id",
        how="left",
        suffixes=("_event", "_strict"),
        validate="many_to_one",
    )
    missing = int(merged["__ts___strict"].isna().sum()) if "__ts___strict" in merged.columns else 0
    mismatched_ts = int(
        (
            pd.to_datetime(merged["__ts___event"], utc=True, errors="coerce")
            != pd.to_datetime(merged["__ts___strict"], utc=True, errors="coerce")
        ).sum()
    )
    mismatched_symbol = int((merged["__symbol___event"].astype(str) != merged["__symbol___strict"].astype(str)).sum())
    report = {
        "unique_event_row_ids": int(events["__strict_oos_row_id"].nunique(dropna=True)),
        "missing_row_ids": missing,
        "mismatched_timestamps": mismatched_ts,
        "mismatched_symbols": mismatched_symbol,
    }
    if missing or mismatched_ts or mismatched_symbol:
        raise ValueError(f"Strict OOS row-id alignment failed: {report}")
    return report


def _prepare_matrix(
    feature_matrix: pd.DataFrame,
    train_ids: pd.Series,
    valid_ids: pd.Series,
    *,
    feature_cols: list[str],
    train_proxy_rank: pd.Series | None = None,
    valid_proxy_rank: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = feature_matrix.loc[train_ids.to_numpy(dtype=np.int64), feature_cols].copy()
    x_valid = feature_matrix.loc[valid_ids.to_numpy(dtype=np.int64), feature_cols].copy()
    if train_proxy_rank is not None and valid_proxy_rank is not None:
        x_train["repair_proxy_pct_rank"] = train_proxy_rank.reset_index(drop=True).to_numpy(dtype=np.float32)
        x_valid["repair_proxy_pct_rank"] = valid_proxy_rank.reset_index(drop=True).to_numpy(dtype=np.float32)
    med = x_train.replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    x_train = x_train.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    x_valid = x_valid.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    return x_train.astype(np.float32, copy=False), x_valid.astype(np.float32, copy=False)


def _fit_predict_repair(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    model = ExtraTreesRegressor(
        n_estimators=80,
        max_depth=7,
        min_samples_leaf=8,
        max_features="sqrt",
        random_state=int(seed),
        n_jobs=2,
    )
    model.fit(x_train, _safe_numeric(y_train).fillna(0.0).to_numpy(dtype=np.float32))
    return model.predict(x_valid).astype(np.float32)


def _subset_stats(frame: pd.DataFrame, *, ids: set[int]) -> dict[str, Any]:
    selected = frame[frame["__strict_oos_row_id"].isin(ids)].copy()
    stats = _utility_stats(selected)
    econ_col = _label_col("economic_capture_v4")
    rec_col = _label_col("recoverable_opportunity_v2")
    stats.update(
        {
            "economic_capture_rate": _safe_mean(_safe_numeric(selected.get(econ_col)).eq(1.0))
            if not selected.empty and econ_col in selected.columns
            else float("nan"),
            "recoverable_opportunity_rate": _safe_mean(_safe_numeric(selected.get(rec_col)).eq(1.0))
            if not selected.empty and rec_col in selected.columns
            else float("nan"),
        }
    )
    return stats


def _selection_ids(frame: pd.DataFrame, score: Any, k: int) -> set[int]:
    idx = _rank_top_indices(score, k)
    if len(idx) == 0:
        return set()
    return set(frame.iloc[idx]["__strict_oos_row_id"].astype(int).tolist())


def _selection_rows(
    *,
    frame: pd.DataFrame,
    ids: set[int],
    context: dict[str, Any],
    selection_type: str,
    repair_score: pd.Series | None = None,
    repair_blend_score: pd.Series | None = None,
) -> list[dict[str, Any]]:
    if not ids:
        return []
    keep_cols = [
        "__strict_oos_row_id",
        "__ts__",
        "__symbol__",
        "primary_source_tag",
        "u_policy_net",
        "mae_norm",
        "mfe_norm",
        "barrier",
        "bad_mae_1r_flag",
        "timeout_or_slow_holding_flag",
        "wide_barrier_25bps_flag",
    ]
    selected = frame[frame["__strict_oos_row_id"].isin(ids)].copy()
    if repair_score is not None:
        selected["repair_score"] = repair_score.loc[selected.index].to_numpy(dtype=np.float32)
    if repair_blend_score is not None:
        selected["repair_blend_score"] = repair_blend_score.loc[selected.index].to_numpy(dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for row in selected[[col for col in keep_cols + ["repair_score", "repair_blend_score"] if col in selected.columns]].to_dict("records"):
        rows.append({**context, "selection_type": selection_type, **row})
    return rows


def _result_row(
    *,
    context: dict[str, Any],
    valid: pd.DataFrame,
    train_events: pd.DataFrame,
    selected_ids: set[int],
    proxy_ids: set[int],
    oracle_ids: set[int],
    selection_method: str,
) -> dict[str, Any]:
    selected_stats = _subset_stats(valid, ids=selected_ids)
    proxy_stats = _subset_stats(valid, ids=proxy_ids)
    oracle_stats = _subset_stats(valid, ids=oracle_ids)
    selected_rows = len(selected_ids)
    oracle_capture = float(len(selected_ids & oracle_ids) / len(oracle_ids)) if oracle_ids else float("nan")
    proxy_oracle_capture = float(len(proxy_ids & oracle_ids) / len(oracle_ids)) if oracle_ids else float("nan")
    proxy_overlap = float(len(selected_ids & proxy_ids) / selected_rows) if selected_rows else float("nan")
    return {
        **context,
        "selection_method": selection_method,
        "train_events": int(len(train_events)),
        "train_positive_events": int(train_events["repair_target"].eq(1.0).sum()),
        "train_negative_events": int(train_events["repair_target"].eq(0.0).sum()),
        "scope_rows": int(len(valid)),
        "selected_rows": int(selected_rows),
        "repair_mean_u": selected_stats["mean_u"],
        "proxy_mean_u": proxy_stats["mean_u"],
        "oracle_mean_u": oracle_stats["mean_u"],
        "scope_mean_u": _safe_mean(valid["u_policy_net"]),
        "repair_delta_mean_u_vs_proxy": (
            selected_stats["mean_u"] - proxy_stats["mean_u"]
            if math.isfinite(selected_stats["mean_u"]) and math.isfinite(proxy_stats["mean_u"])
            else float("nan")
        ),
        "repair_delta_mean_u_vs_scope": (
            selected_stats["mean_u"] - _safe_mean(valid["u_policy_net"])
            if math.isfinite(selected_stats["mean_u"]) and math.isfinite(_safe_mean(valid["u_policy_net"]))
            else float("nan")
        ),
        "repair_hit_u": selected_stats["hit_u"],
        "proxy_hit_u": proxy_stats["hit_u"],
        "repair_bad_mae_1r_rate": selected_stats["bad_mae_1r_rate"],
        "proxy_bad_mae_1r_rate": proxy_stats["bad_mae_1r_rate"],
        "repair_timeout_or_slow_holding_rate": selected_stats["timeout_or_slow_holding_rate"],
        "proxy_timeout_or_slow_holding_rate": proxy_stats["timeout_or_slow_holding_rate"],
        "repair_economic_capture_rate": selected_stats["economic_capture_rate"],
        "proxy_economic_capture_rate": proxy_stats["economic_capture_rate"],
        "repair_recoverable_rate": selected_stats["recoverable_opportunity_rate"],
        "proxy_recoverable_rate": proxy_stats["recoverable_opportunity_rate"],
        "repair_oracle_capture_at_k": oracle_capture,
        "proxy_oracle_capture_at_k": proxy_oracle_capture,
        "repair_delta_oracle_capture_at_k": (
            oracle_capture - proxy_oracle_capture
            if math.isfinite(oracle_capture) and math.isfinite(proxy_oracle_capture)
            else float("nan")
        ),
        "repair_proxy_overlap_at_k": proxy_overlap,
    }


def _aggregate(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    group_cols = ["source_bucket", "proxy_col", "top_frac", "feature_mode", "selection_method"]
    rows: list[dict[str, Any]] = []
    for key, group in monthly.groupby(group_cols, dropna=False, observed=True):
        row = {col: value for col, value in zip(group_cols, key)}
        repair_mean = _safe_numeric(group["repair_mean_u"])
        delta = _safe_numeric(group["repair_delta_mean_u_vs_proxy"])
        row.update(
            {
                "months": int(group["period"].nunique(dropna=True)),
                "repair_positive_months": int(repair_mean.gt(0.0).sum()),
                "delta_positive_months": int(delta.gt(0.0).sum()),
                "repair_mean_u": _safe_mean(group["repair_mean_u"]),
                "repair_worst_month_u": _safe_quantile(group["repair_mean_u"], 0.0),
                "proxy_mean_u": _safe_mean(group["proxy_mean_u"]),
                "oracle_mean_u": _safe_mean(group["oracle_mean_u"]),
                "scope_mean_u": _safe_mean(group["scope_mean_u"]),
                "repair_delta_mean_u_vs_proxy": _safe_mean(group["repair_delta_mean_u_vs_proxy"]),
                "repair_delta_mean_u_vs_scope": _safe_mean(group["repair_delta_mean_u_vs_scope"]),
                "repair_bad_mae_1r_rate": _safe_mean(group["repair_bad_mae_1r_rate"]),
                "proxy_bad_mae_1r_rate": _safe_mean(group["proxy_bad_mae_1r_rate"]),
                "repair_timeout_or_slow_holding_rate": _safe_mean(
                    group["repair_timeout_or_slow_holding_rate"]
                ),
                "proxy_timeout_or_slow_holding_rate": _safe_mean(
                    group["proxy_timeout_or_slow_holding_rate"]
                ),
                "repair_economic_capture_rate": _safe_mean(group["repair_economic_capture_rate"]),
                "proxy_economic_capture_rate": _safe_mean(group["proxy_economic_capture_rate"]),
                "repair_oracle_capture_at_k": _safe_mean(group["repair_oracle_capture_at_k"]),
                "proxy_oracle_capture_at_k": _safe_mean(group["proxy_oracle_capture_at_k"]),
                "repair_delta_oracle_capture_at_k": _safe_mean(group["repair_delta_oracle_capture_at_k"]),
                "repair_proxy_overlap_at_k": _safe_mean(group["repair_proxy_overlap_at_k"]),
                "mean_selected_rows": _safe_mean(group["selected_rows"]),
            }
        )
        if row["months"] < 2:
            decision = "insufficient_forward_months"
        elif row["repair_delta_mean_u_vs_proxy"] > 0.0 and row["repair_positive_months"] == row["months"]:
            decision = "positive_and_beats_proxy"
        elif row["repair_delta_mean_u_vs_proxy"] > 0.0 and row["delta_positive_months"] == row["months"]:
            decision = "beats_proxy_both_months"
        elif row["repair_delta_mean_u_vs_proxy"] > 0.0:
            decision = "mixed_but_beats_proxy_mean"
        else:
            decision = "fails_vs_proxy"
        row["decision"] = decision
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["repair_delta_mean_u_vs_proxy", "repair_mean_u", "repair_delta_oracle_capture_at_k"],
        ascending=[False, False, False],
        kind="mergesort",
    )


def _write_report(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    aggregate: pd.DataFrame,
    monthly: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> Path:
    path = output_dir / "strict_oos_repair_ranker_ablation_report.md"
    reviewable = aggregate.copy()
    if not reviewable.empty and "months" in reviewable.columns:
        reviewable = reviewable[reviewable["months"].ge(2)].copy()
    one_month = aggregate.copy()
    if not one_month.empty and "months" in one_month.columns:
        one_month = one_month[one_month["months"].lt(2)].copy()
    cols = [
        "decision",
        "source_bucket",
        "proxy_col",
        "top_frac",
        "feature_mode",
        "selection_method",
        "months",
        "repair_mean_u",
        "proxy_mean_u",
        "repair_delta_mean_u_vs_proxy",
        "repair_worst_month_u",
        "repair_oracle_capture_at_k",
        "proxy_oracle_capture_at_k",
        "repair_bad_mae_1r_rate",
        "proxy_bad_mae_1r_rate",
        "mean_selected_rows",
    ]
    monthly_cols = [
        "period",
        "source_bucket",
        "proxy_col",
        "top_frac",
        "feature_mode",
        "selection_method",
        "scope_rows",
        "selected_rows",
        "repair_mean_u",
        "proxy_mean_u",
        "oracle_mean_u",
        "repair_delta_mean_u_vs_proxy",
        "repair_oracle_capture_at_k",
        "proxy_oracle_capture_at_k",
    ]
    lines = [
        "# Strict OOS Repair Ranker Ablation",
        "",
        "Diagnostic-only month-forward repair model. Training labels are prior-month oracle/proxy errors; validation scores are applied to the next-month strict OOS universe.",
        "",
        "## Scope",
        "",
        f"- Strict rows: {manifest.get('strict_rows', 0):,}",
        f"- Event rows: {manifest.get('event_rows', 0):,}",
        f"- Loaded features: {manifest.get('feature_report', {}).get('retained_features', 0):,}",
        f"- Validation months: {', '.join(manifest.get('validation_months', []))}",
        f"- Source buckets: {', '.join(manifest.get('source_buckets', []))}",
        f"- Proxy columns: {', '.join(manifest.get('proxy_columns', []))}",
        "",
        "## Decision Counts",
        "",
        _table(aggregate["decision"].value_counts().rename_axis("decision").reset_index(name="rows"), ["decision", "rows"])
        if not aggregate.empty and "decision" in aggregate.columns
        else "No aggregate rows.",
        "",
        "## Best Two-Month Aggregate Deltas",
        "",
        _table(reviewable, cols, limit=40),
        "",
        "## One-Month Leads",
        "",
        _table(one_month, cols, limit=20),
        "",
        "## Best Monthly Deltas",
        "",
        _table(monthly.sort_values("repair_delta_mean_u_vs_proxy", ascending=False, kind="mergesort"), monthly_cols, limit=60)
        if not monthly.empty
        else "No monthly rows.",
        "",
        "## Skips And Guards",
        "",
        _table(diagnostics["reason"].value_counts().rename_axis("reason").reset_index(name="rows"), ["reason", "rows"])
        if not diagnostics.empty and "reason" in diagnostics.columns
        else "No skips.",
        "",
        "## Interpretation",
        "",
        "- This is not a deployable result; scenarios are numerous and still require a fixed pre-registered validation pass.",
        "- Positive repair deltas mean prior-month feature patterns selected better next-month rows than the original proxy for the same source bucket and budget.",
        "- A useful next step requires positive delta stability, better oracle capture, and no worsening of bad-MAE/timeout risk.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    quality_labels_path: Path,
    labels_path: Path,
    predictions_path: Path,
    event_rows_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    output_dir: Path,
    months: list[str],
    source_buckets: list[str],
    proxy_cols: list[str],
    top_fracs: list[float],
    feature_modes: list[str],
    selection_methods: list[str],
    max_features: int,
    min_train_class_rows: int,
    min_valid_scope_rows: int,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    strict, _metrics, join_report = _join_strict_oos(
        quality_labels_path=quality_labels_path,
        labels_path=labels_path,
        predictions_path=predictions_path,
        months=months,
    )
    strict["__strict_oos_row_id"] = np.arange(len(strict), dtype=np.int64)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_features)
    feature_matrix, feature_report = _load_feature_store_columns(
        strict,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    feature_cols = [col for col in selected_features if col in feature_matrix.columns]
    events = _load_events(
        event_rows_path,
        months=months,
        source_buckets=source_buckets,
        proxy_cols=proxy_cols,
        top_fracs=top_fracs,
    )
    alignment_report = _validate_event_alignment(strict, events)

    periods = sorted(strict["period"].dropna().astype(str).unique().tolist())
    validation_months = [period for period in periods if any(prev < period for prev in periods)]
    monthly_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for period in validation_months:
        period_frame = strict[strict["period"].astype(str).eq(period)].copy()
        prior_periods = [prior for prior in periods if prior < period]
        for source_bucket in source_buckets:
            source_mask = _source_bucket_mask(period_frame, source_bucket)
            source_frame = period_frame.loc[source_mask].copy()
            if len(source_frame) < min_valid_scope_rows:
                diagnostic_rows.append(
                    {
                        "period": period,
                        "source_bucket": source_bucket,
                        "reason": "valid_scope_too_small",
                        "valid_scope_rows": int(len(source_frame)),
                    }
                )
                continue
            for proxy_col in proxy_cols:
                if proxy_col not in source_frame.columns:
                    continue
                source_proxy = _safe_numeric(source_frame[proxy_col])
                if int(source_proxy.notna().sum()) < min_valid_scope_rows:
                    diagnostic_rows.append(
                        {
                            "period": period,
                            "source_bucket": source_bucket,
                            "proxy_col": proxy_col,
                            "reason": "valid_proxy_too_sparse",
                            "valid_proxy_rows": int(source_proxy.notna().sum()),
                        }
                    )
                    continue
                source_frame = source_frame.copy()
                source_frame["repair_proxy_pct_rank"] = _pct_rank_high(source_proxy)
                for top_frac in top_fracs:
                    top_frac_key = round(float(top_frac), 6)
                    train_events = events[
                        events["period"].isin(prior_periods)
                        & events["source_bucket"].eq(source_bucket)
                        & events["proxy_col"].eq(proxy_col)
                        & events["top_frac"].eq(top_frac_key)
                    ].copy()
                    positives = int(train_events["repair_target"].eq(1.0).sum())
                    negatives = int(train_events["repair_target"].eq(0.0).sum())
                    context_base = {
                        "period": period,
                        "train_periods": ",".join(prior_periods),
                        "source_bucket": source_bucket,
                        "proxy_col": proxy_col,
                        "top_frac": float(top_frac),
                    }
                    if positives < min_train_class_rows or negatives < min_train_class_rows:
                        diagnostic_rows.append(
                            {
                                **context_base,
                                "reason": "train_class_too_sparse",
                                "train_events": int(len(train_events)),
                                "train_positive_events": positives,
                                "train_negative_events": negatives,
                            }
                        )
                        continue

                    k = max(1, int(math.ceil(float(top_frac) * len(source_frame))))
                    proxy_ids = _selection_ids(source_frame, source_frame[proxy_col], k)
                    oracle_ids = _selection_ids(source_frame, source_frame["u_policy_net"], k)
                    valid_ids = source_frame["__strict_oos_row_id"].astype(int).reset_index(drop=True)
                    train_ids = train_events["__strict_oos_row_id"].astype(int).reset_index(drop=True)

                    for feature_mode in feature_modes:
                        if feature_mode not in DEFAULT_FEATURE_MODES:
                            raise ValueError(f"Unknown feature mode: {feature_mode}")
                        use_proxy_rank = feature_mode == "frozen_features_plus_proxy_rank"
                        train_proxy_rank = (
                            _safe_numeric(train_events["proxy_pct_rank"]).reset_index(drop=True)
                            if use_proxy_rank and "proxy_pct_rank" in train_events.columns
                            else None
                        )
                        valid_proxy_rank = (
                            source_frame["repair_proxy_pct_rank"].reset_index(drop=True)
                            if use_proxy_rank
                            else None
                        )
                        x_train, x_valid = _prepare_matrix(
                            feature_matrix,
                            train_ids,
                            valid_ids,
                            feature_cols=feature_cols,
                            train_proxy_rank=train_proxy_rank,
                            valid_proxy_rank=valid_proxy_rank,
                        )
                        repair_pred = pd.Series(
                            _fit_predict_repair(
                                x_train=x_train,
                                y_train=train_events["repair_target"].reset_index(drop=True),
                                x_valid=x_valid,
                                seed=seed,
                            ),
                            index=source_frame.index,
                        )
                        repair_rank = _pct_rank_high(repair_pred)
                        proxy_rank = source_frame["repair_proxy_pct_rank"]
                        repair_blend = (0.70 * repair_rank.fillna(0.0)) + (0.30 * proxy_rank.fillna(0.0))
                        score_by_method = {
                            "repair_score": repair_pred,
                            "repair_proxy_blend_70_30": repair_blend,
                        }
                        for selection_method in selection_methods:
                            if selection_method not in score_by_method:
                                raise ValueError(f"Unknown selection method: {selection_method}")
                            selected_ids = _selection_ids(source_frame, score_by_method[selection_method], k)
                            context = {
                                **context_base,
                                "feature_mode": feature_mode,
                            }
                            monthly_rows.append(
                                _result_row(
                                    context=context,
                                    valid=source_frame,
                                    train_events=train_events,
                                    selected_ids=selected_ids,
                                    proxy_ids=proxy_ids,
                                    oracle_ids=oracle_ids,
                                    selection_method=selection_method,
                                )
                            )
                            selected_context = {
                                **context,
                                "selection_method": selection_method,
                            }
                            selected_rows.extend(
                                _selection_rows(
                                    frame=source_frame,
                                    ids=selected_ids,
                                    context=selected_context,
                                    selection_type="repair",
                                    repair_score=repair_pred,
                                    repair_blend_score=repair_blend,
                                )
                            )
                            selected_rows.extend(
                                _selection_rows(
                                    frame=source_frame,
                                    ids=proxy_ids,
                                    context=selected_context,
                                    selection_type="proxy",
                                    repair_score=repair_pred,
                                    repair_blend_score=repair_blend,
                                )
                            )
                            selected_rows.extend(
                                _selection_rows(
                                    frame=source_frame,
                                    ids=oracle_ids,
                                    context=selected_context,
                                    selection_type="oracle",
                                    repair_score=repair_pred,
                                    repair_blend_score=repair_blend,
                                )
                            )

    monthly = pd.DataFrame(monthly_rows)
    aggregate = _aggregate(monthly)
    diagnostics = pd.DataFrame(diagnostic_rows)
    selected = pd.DataFrame(selected_rows)

    paths = {
        "monthly": output_dir / "strict_oos_repair_ranker_monthly.csv",
        "aggregate": output_dir / "strict_oos_repair_ranker_aggregate.csv",
        "diagnostics": output_dir / "strict_oos_repair_ranker_diagnostics.csv",
        "selected_rows": output_dir / "strict_oos_repair_ranker_selected_rows.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    selected.to_csv(paths["selected_rows"], index=False)

    manifest = {
        "scope": "strict_oos_repair_ranker_ablation",
        "quality_labels_path": str(quality_labels_path),
        "labels_path": str(labels_path),
        "predictions_path": str(predictions_path),
        "event_rows_path": str(event_rows_path),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "output_dir": str(output_dir),
        "strict_rows": int(len(strict)),
        "event_rows": int(len(events)),
        "months": periods,
        "validation_months": sorted(monthly["period"].dropna().unique().tolist()) if not monthly.empty else [],
        "source_buckets": source_buckets,
        "proxy_columns": proxy_cols,
        "top_fracs": [float(v) for v in top_fracs],
        "feature_modes": feature_modes,
        "selection_methods": selection_methods,
        "max_features": int(max_features),
        "min_train_class_rows": int(min_train_class_rows),
        "min_valid_scope_rows": int(min_valid_scope_rows),
        "join_report": join_report,
        "feature_report": feature_report,
        "alignment_report": alignment_report,
        "monthly_rows": int(len(monthly)),
        "aggregate_rows": int(len(aggregate)),
        "diagnostic_rows": int(len(diagnostics)),
        "decision_counts": aggregate["decision"].value_counts().to_dict()
        if not aggregate.empty and "decision" in aggregate.columns
        else {},
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    report = _write_report(
        output_dir=output_dir,
        manifest=manifest,
        aggregate=aggregate,
        monthly=monthly,
        diagnostics=diagnostics,
    )
    manifest["outputs"]["markdown"] = str(report)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-labels-path", type=Path, default=DEFAULT_QUALITY_LABELS)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--event-rows-path", type=Path, default=DEFAULT_EVENT_ROWS)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--months", type=str, default=",".join(DEFAULT_MONTHS))
    parser.add_argument(
        "--source-buckets",
        type=str,
        default=",".join(spec.name for spec in SOURCE_BUCKETS),
    )
    parser.add_argument("--proxy-cols", type=str, default=",".join(DEFAULT_PROXY_COLUMNS))
    parser.add_argument("--top-fracs", type=str, default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--feature-modes", type=str, default=",".join(DEFAULT_FEATURE_MODES))
    parser.add_argument("--selection-methods", type=str, default=",".join(DEFAULT_SELECTION_METHODS))
    parser.add_argument("--max-features", type=int, default=96)
    parser.add_argument("--min-train-class-rows", type=int, default=10)
    parser.add_argument("--min-valid-scope-rows", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        quality_labels_path=args.quality_labels_path,
        labels_path=args.labels_path,
        predictions_path=args.predictions_path,
        event_rows_path=args.event_rows_path,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        output_dir=args.output_dir,
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        source_buckets=_parse_csv(args.source_buckets, tuple(spec.name for spec in SOURCE_BUCKETS)),
        proxy_cols=_parse_csv(args.proxy_cols, DEFAULT_PROXY_COLUMNS),
        top_fracs=_parse_float_csv(args.top_fracs, DEFAULT_TOP_FRACS),
        feature_modes=_parse_csv(args.feature_modes, DEFAULT_FEATURE_MODES),
        selection_methods=_parse_csv(args.selection_methods, DEFAULT_SELECTION_METHODS),
        max_features=args.max_features,
        min_train_class_rows=args.min_train_class_rows,
        min_valid_scope_rows=args.min_valid_scope_rows,
        seed=args.seed,
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
