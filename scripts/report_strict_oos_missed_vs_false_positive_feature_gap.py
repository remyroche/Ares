#!/usr/bin/env python3
"""Strict OOS feature-gap report for missed winners vs proxy false positives.

This is a diagnostic-only report. It reads the strict OOS oracle/proxy event
ledger, joins those rows back to the frozen causal feature store, and contrasts
features between:

1. rows the oracle top-k would have selected but the proxy missed, and
2. rows the proxy selected that were not oracle top-k rows.

It does not train a model, tune a threshold, or integrate source tags into
training.
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

from scripts.report_strict_oos_source_label_proxy_diagnostic import (  # noqa: E402
    DEFAULT_PROXY_COLUMNS,
    DEFAULT_TOP_FRACS,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    _json_safe,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _schema_names,
    _symbol_to_feature_path,
)


DEFAULT_EVENT_ROWS = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "strict_oos_proxy_missed_opportunities/strict_oos_proxy_oracle_gap_event_rows.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "strict_oos_missed_vs_false_positive_feature_gap"
)
DEFAULT_SOURCE_BUCKETS = (
    "all_rows",
    "dirty_excluded",
    "risk_adjusted_capture_candidate",
    "compression_capture_candidate",
    "risk_adjusted_capture_dirty_excluded",
    "compression_capture_dirty_excluded",
)
MISSED_EVENT_TYPE = "missed_oracle_topk"
FALSE_POSITIVE_EVENT_TYPE = "proxy_false_positive_vs_oracle"
EVENT_TYPES = (MISSED_EVENT_TYPE, FALSE_POSITIVE_EVENT_TYPE)
SCENARIO_COLS = ("source_bucket", "proxy_col", "top_frac")
KEY_COLS = ("__strict_oos_row_id", "__ts__", "__symbol__")


def _parse_csv(raw: str, default: tuple[str, ...]) -> list[str]:
    if raw is None or not str(raw).strip():
        return list(default)
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _parse_float_csv(raw: str, default: tuple[float, ...]) -> list[float]:
    if raw is None or not str(raw).strip():
        return [float(v) for v in default]
    return [float(part.strip()) for part in str(raw).split(",") if part.strip()]


def _safe_numeric(values: Any) -> pd.Series:
    if values is None:
        return pd.Series(dtype=np.float64)
    return pd.to_numeric(values, errors="coerce")


def _bool_rate(values: Any) -> float:
    arr = _safe_numeric(values).dropna()
    return float(arr.gt(0.5).mean()) if len(arr) else float("nan")


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


def _top_frac_key(values: Any) -> pd.Series:
    return _safe_numeric(values).round(6)


def _load_events(
    event_rows_path: Path,
    *,
    source_buckets: list[str],
    proxy_cols: list[str],
    top_fracs: list[float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not event_rows_path.exists():
        raise FileNotFoundError(event_rows_path)
    events = pd.read_csv(event_rows_path)
    required = set(KEY_COLS) | set(SCENARIO_COLS) | {"event_type", "period"}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"{event_rows_path} is missing required columns: {missing}")

    events["__ts__"] = pd.to_datetime(events["__ts__"], utc=True, errors="coerce")
    events["period"] = events["period"].astype(str)
    events["top_frac"] = _top_frac_key(events["top_frac"])
    top_frac_set = {round(float(v), 6) for v in top_fracs}

    before_filter = len(events)
    mask = events["event_type"].isin(EVENT_TYPES)
    mask &= events["source_bucket"].isin(source_buckets)
    mask &= events["proxy_col"].isin(proxy_cols)
    mask &= events["top_frac"].isin(top_frac_set)
    events = events.loc[mask].copy()

    before_dedupe = len(events)
    dedupe_cols = list(SCENARIO_COLS) + ["period", "event_type", "__strict_oos_row_id"]
    events = events.drop_duplicates(dedupe_cols, keep="first").reset_index(drop=True)

    manifest = {
        "event_rows_path": str(event_rows_path),
        "raw_event_rows": int(before_filter),
        "filtered_event_rows_before_dedupe": int(before_dedupe),
        "filtered_event_rows": int(len(events)),
        "event_types": sorted(events["event_type"].dropna().unique().tolist()),
        "source_buckets": sorted(events["source_bucket"].dropna().unique().tolist()),
        "proxy_columns": sorted(events["proxy_col"].dropna().unique().tolist()),
        "top_fracs": sorted(float(v) for v in events["top_frac"].dropna().unique().tolist()),
        "months": sorted(events["period"].dropna().unique().tolist()),
        "symbols": int(events["__symbol__"].nunique(dropna=True)),
        "unique_strict_oos_rows": int(events["__strict_oos_row_id"].nunique(dropna=True)),
    }
    return events, manifest


def _load_feature_values(
    events: pd.DataFrame,
    *,
    feature_dir: Path,
    selected_features: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    unique = (
        events.loc[:, list(KEY_COLS)]
        .drop_duplicates(list(KEY_COLS), keep="first")
        .reset_index(drop=True)
    )
    unique["__ts_utc__"] = pd.to_datetime(unique["__ts__"], utc=True, errors="coerce")
    matrix = pd.DataFrame(index=unique.index, columns=selected_features, dtype=np.float32)

    loaded_symbols = 0
    missing_symbols: list[str] = []
    available_feature_counts: list[int] = []
    symbol_row_counts: dict[str, int] = {}

    for symbol, idx in unique.groupby("__symbol__", sort=False).indices.items():
        symbol_str = str(symbol)
        rows = np.asarray(idx, dtype=np.int64)
        symbol_row_counts[symbol_str] = int(len(rows))
        path = _symbol_to_feature_path(feature_dir, symbol_str)
        if not path.exists():
            missing_symbols.append(symbol_str)
            continue
        names = _schema_names(path)
        available = [feature for feature in selected_features if feature in names]
        available_feature_counts.append(len(available))
        if not available:
            continue
        try:
            features = pd.read_parquet(path, columns=available)
        except Exception:
            features = pd.read_parquet(path)
            available = [feature for feature in selected_features if feature in features.columns]
            features = features.loc[:, available].copy()
        if not available:
            continue
        features.index = pd.to_datetime(features.index, utc=True, errors="coerce")
        aligned = features.reindex(unique.loc[rows, "__ts_utc__"])
        matrix.loc[rows, available] = aligned.to_numpy(dtype=np.float32, copy=False)
        loaded_symbols += 1

    feature_cols = [
        feature
        for feature in selected_features
        if feature in matrix.columns and bool(matrix[feature].notna().any())
    ]
    matrix = matrix.loc[:, feature_cols].copy()
    feature_rows_matched = matrix.notna().any(axis=1) if len(matrix.columns) else pd.Series(False, index=matrix.index)
    joined_unique = pd.concat([unique.drop(columns=["__ts_utc__"]), matrix], axis=1)
    out = events.merge(joined_unique, on=list(KEY_COLS), how="left", validate="many_to_one")

    manifest = {
        "feature_dir": str(feature_dir),
        "requested_features": int(len(selected_features)),
        "loaded_features": int(len(feature_cols)),
        "unique_event_rows": int(len(unique)),
        "feature_rows_matched": int(feature_rows_matched.sum()),
        "feature_rows_missing": int((~feature_rows_matched).sum()),
        "loaded_symbols": int(loaded_symbols),
        "missing_symbols": sorted(missing_symbols),
        "mean_available_features_per_symbol": (
            float(np.mean(available_feature_counts)) if available_feature_counts else 0.0
        ),
        "top_event_symbols": dict(
            sorted(symbol_row_counts.items(), key=lambda item: item[1], reverse=True)[:20]
        ),
    }
    return out, manifest


def _pooled_scale(left: pd.Series, right: pd.Series) -> float:
    pooled = pd.concat([left, right], ignore_index=True).dropna()
    if len(pooled) < 3:
        return float("nan")
    iqr = float(pooled.quantile(0.75) - pooled.quantile(0.25))
    if math.isfinite(iqr) and iqr > 1e-12:
        return iqr
    std = float(pooled.std(ddof=0))
    return std if math.isfinite(std) and std > 1e-12 else float("nan")


def _feature_contrasts_for_group(
    group: pd.DataFrame,
    *,
    feature_cols: list[str],
    prefix: dict[str, Any],
) -> pd.DataFrame:
    missed = group.loc[group["event_type"] == MISSED_EVENT_TYPE]
    false_pos = group.loc[group["event_type"] == FALSE_POSITIVE_EVENT_TYPE]
    if missed.empty or false_pos.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for feature in feature_cols:
        left = _safe_numeric(missed[feature]).dropna()
        right = _safe_numeric(false_pos[feature]).dropna()
        if left.empty or right.empty:
            continue
        missed_median = float(left.median())
        false_median = float(right.median())
        median_diff = missed_median - false_median
        scale = _pooled_scale(left, right)
        robust_z = median_diff / scale if math.isfinite(scale) and scale > 0.0 else float("nan")
        rows.append(
            {
                **prefix,
                "feature": feature,
                "missed_n": int(len(left)),
                "false_positive_n": int(len(right)),
                "missed_mean": float(left.mean()),
                "false_positive_mean": float(right.mean()),
                "mean_diff": float(left.mean() - right.mean()),
                "missed_median": missed_median,
                "false_positive_median": false_median,
                "median_diff": float(median_diff),
                "pooled_scale": float(scale),
                "robust_z_delta": float(robust_z),
                "abs_robust_z_delta": float(abs(robust_z)) if math.isfinite(robust_z) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _event_metrics(group: pd.DataFrame, event_type: str) -> dict[str, float]:
    view = group.loc[group["event_type"] == event_type]
    return {
        "rows": float(len(view)),
        "mean_u": _safe_mean(view.get("u_policy_net")),
        "median_u": _safe_quantile(view.get("u_policy_net"), 0.50),
        "bad_mae_1r_rate": _bool_rate(view.get("bad_mae_1r_flag")),
        "timeout_or_slow_rate": _bool_rate(view.get("timeout_or_slow_holding_flag")),
        "economic_capture_rate": _bool_rate(view.get("quality_label_economic_capture_v4")),
        "recoverable_opportunity_rate": _bool_rate(view.get("quality_label_recoverable_opportunity_v2")),
        "mean_proxy_pct_rank": _safe_mean(view.get("proxy_pct_rank")),
    }


def _summarize_scenario(
    group: pd.DataFrame,
    contrasts: pd.DataFrame,
    *,
    min_group_rows: int,
    min_months: int,
) -> dict[str, Any]:
    missed = _event_metrics(group, MISSED_EVENT_TYPE)
    false_pos = _event_metrics(group, FALSE_POSITIVE_EVENT_TYPE)
    if contrasts.empty or "robust_z_delta" not in contrasts.columns:
        valid = pd.DataFrame(columns=["feature", "robust_z_delta", "abs_robust_z_delta"])
    else:
        valid = contrasts.dropna(subset=["robust_z_delta"]).copy()
    valid = valid.sort_values("abs_robust_z_delta", ascending=False, kind="mergesort")
    positive = valid.sort_values("robust_z_delta", ascending=False, kind="mergesort")
    negative = valid.sort_values("robust_z_delta", ascending=True, kind="mergesort")
    strong_05 = int(valid["abs_robust_z_delta"].ge(0.50).sum()) if not valid.empty else 0
    strong_10 = int(valid["abs_robust_z_delta"].ge(1.00).sum()) if not valid.empty else 0
    months = int(group["period"].nunique(dropna=True))

    if missed["rows"] < min_group_rows or false_pos["rows"] < min_group_rows:
        status = "sparse"
        diagnosis = "too_sparse"
    elif months < min_months:
        status = "limited_months"
        diagnosis = "needs_more_months"
    elif strong_05 >= 3:
        status = "stable_enough_for_review"
        diagnosis = "causal_feature_separation_present"
    else:
        status = "weak"
        diagnosis = "weak_feature_separation"

    def _feature_at(frame: pd.DataFrame, idx: int, col: str) -> Any:
        if frame.empty or len(frame) <= idx:
            return np.nan
        return frame.iloc[idx].get(col)

    top_abs = [
        f"{row.feature}:{float(row.robust_z_delta):+.2f}"
        for row in valid.head(8).itertuples(index=False)
        if pd.notna(row.robust_z_delta)
    ]
    return {
        "missed_rows": int(missed["rows"]),
        "false_positive_rows": int(false_pos["rows"]),
        "months": months,
        "missed_mean_u": missed["mean_u"],
        "false_positive_mean_u": false_pos["mean_u"],
        "utility_gap_missed_minus_false_positive": (
            missed["mean_u"] - false_pos["mean_u"]
            if math.isfinite(missed["mean_u"]) and math.isfinite(false_pos["mean_u"])
            else float("nan")
        ),
        "missed_bad_mae_1r_rate": missed["bad_mae_1r_rate"],
        "false_positive_bad_mae_1r_rate": false_pos["bad_mae_1r_rate"],
        "missed_timeout_or_slow_rate": missed["timeout_or_slow_rate"],
        "false_positive_timeout_or_slow_rate": false_pos["timeout_or_slow_rate"],
        "missed_economic_capture_rate": missed["economic_capture_rate"],
        "false_positive_economic_capture_rate": false_pos["economic_capture_rate"],
        "missed_recoverable_opportunity_rate": missed["recoverable_opportunity_rate"],
        "false_positive_recoverable_opportunity_rate": false_pos["recoverable_opportunity_rate"],
        "missed_mean_proxy_pct_rank": missed["mean_proxy_pct_rank"],
        "false_positive_mean_proxy_pct_rank": false_pos["mean_proxy_pct_rank"],
        "strong_feature_count_abs_z_ge_0_5": strong_05,
        "strong_feature_count_abs_z_ge_1_0": strong_10,
        "top_positive_feature": _feature_at(positive, 0, "feature"),
        "top_positive_robust_z": _feature_at(positive, 0, "robust_z_delta"),
        "top_negative_feature": _feature_at(negative, 0, "feature"),
        "top_negative_robust_z": _feature_at(negative, 0, "robust_z_delta"),
        "top_abs_features": "; ".join(top_abs),
        "stability_status": status,
        "diagnosis": diagnosis,
    }


def _build_contrasts(
    events: pd.DataFrame,
    *,
    feature_cols: list[str],
    min_group_rows: int,
    min_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contrast_frames: list[pd.DataFrame] = []
    monthly_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for key, group in events.groupby(list(SCENARIO_COLS), dropna=False, observed=True):
        prefix = {col: value for col, value in zip(SCENARIO_COLS, key)}
        contrasts = _feature_contrasts_for_group(group, feature_cols=feature_cols, prefix=prefix)
        if not contrasts.empty:
            contrast_frames.append(contrasts)
        summary_rows.append(
            {
                **prefix,
                **_summarize_scenario(
                    group,
                    contrasts,
                    min_group_rows=min_group_rows,
                    min_months=min_months,
                ),
            }
        )

    for key, group in events.groupby(["period", *SCENARIO_COLS], dropna=False, observed=True):
        prefix = {col: value for col, value in zip(("period", *SCENARIO_COLS), key)}
        monthly = _feature_contrasts_for_group(group, feature_cols=feature_cols, prefix=prefix)
        if not monthly.empty:
            monthly_frames.append(monthly)

    by_scenario = pd.concat(contrast_frames, ignore_index=True) if contrast_frames else pd.DataFrame()
    by_month = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)

    if not by_scenario.empty and not by_month.empty:
        stability_rows = []
        for key, group in by_month.groupby([*SCENARIO_COLS, "feature"], dropna=False, observed=True):
            vals = _safe_numeric(group["robust_z_delta"]).dropna()
            positive = int(vals.gt(0).sum())
            negative = int(vals.lt(0).sum())
            months = int(len(vals))
            stability_rows.append(
                {
                    **{col: value for col, value in zip((*SCENARIO_COLS, "feature"), key)},
                    "feature_month_count": months,
                    "feature_positive_months": positive,
                    "feature_negative_months": negative,
                    "feature_sign_stability": (
                        float(max(positive, negative) / months) if months else float("nan")
                    ),
                    "feature_median_monthly_robust_z": (
                        float(vals.median()) if months else float("nan")
                    ),
                    "feature_max_month_abs_robust_z": (
                        float(vals.abs().max()) if months else float("nan")
                    ),
                }
            )
        stability = pd.DataFrame(stability_rows)
        by_scenario = by_scenario.merge(
            stability,
            on=[*SCENARIO_COLS, "feature"],
            how="left",
            validate="one_to_one",
        )

    if not by_scenario.empty:
        by_scenario = by_scenario.sort_values(
            [*SCENARIO_COLS, "abs_robust_z_delta"],
            ascending=[True, True, True, False],
            kind="mergesort",
        )
    if not by_month.empty:
        by_month = by_month.sort_values(
            ["period", *SCENARIO_COLS, "abs_robust_z_delta"],
            ascending=[True, True, True, True, False],
            kind="mergesort",
        )
    if not summary.empty:
        summary = summary.sort_values(
            ["utility_gap_missed_minus_false_positive", "strong_feature_count_abs_z_ge_0_5"],
            ascending=[False, False],
            kind="mergesort",
        )
    return summary, by_scenario, by_month


def _write_report(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    summary: pd.DataFrame,
    by_scenario: pd.DataFrame,
    by_month: pd.DataFrame,
) -> Path:
    path = output_dir / "strict_oos_feature_gap_report.md"
    summary_cols = [
        "diagnosis",
        "source_bucket",
        "proxy_col",
        "top_frac",
        "missed_rows",
        "false_positive_rows",
        "missed_mean_u",
        "false_positive_mean_u",
        "utility_gap_missed_minus_false_positive",
        "missed_mean_proxy_pct_rank",
        "false_positive_mean_proxy_pct_rank",
        "strong_feature_count_abs_z_ge_0_5",
        "top_abs_features",
    ]
    feature_cols = [
        "source_bucket",
        "proxy_col",
        "top_frac",
        "feature",
        "robust_z_delta",
        "feature_month_count",
        "feature_sign_stability",
        "feature_median_monthly_robust_z",
        "missed_median",
        "false_positive_median",
    ]
    month_cols = [
        "period",
        "source_bucket",
        "proxy_col",
        "top_frac",
        "feature",
        "robust_z_delta",
        "missed_n",
        "false_positive_n",
    ]

    min_months = int(manifest.get("min_months", 3) or 3)
    stable_features = by_scenario.copy()
    if not stable_features.empty:
        stable_features = stable_features[
            stable_features["abs_robust_z_delta"].ge(0.50)
            & stable_features.get("feature_sign_stability", pd.Series(0.0, index=stable_features.index)).ge(0.67)
            & stable_features.get("feature_month_count", pd.Series(0, index=stable_features.index)).ge(min_months)
        ].sort_values("abs_robust_z_delta", ascending=False, kind="mergesort")

    monthly_examples = by_month.copy()
    if not monthly_examples.empty:
        monthly_examples = monthly_examples[
            monthly_examples["missed_n"].ge(5)
            & monthly_examples["false_positive_n"].ge(5)
            & monthly_examples["abs_robust_z_delta"].le(5.0)
        ].sort_values("abs_robust_z_delta", ascending=False, kind="mergesort")

    lines = [
        "# Strict OOS Missed vs False-Positive Feature Gap",
        "",
        "## Scope",
        "",
        "Diagnostic-only oracle/proxy contrast on strict OOS rows. Missed rows are oracle top-k rows "
        "not selected by the proxy; false positives are proxy-selected rows outside oracle top-k. "
        "The report joins those event rows to frozen causal features and does not fit any model.",
        "",
        "## Data",
        "",
        f"- Event rows: {manifest.get('event_rows', {}).get('filtered_event_rows', 0):,}",
        f"- Unique strict OOS rows with events: {manifest.get('event_rows', {}).get('unique_strict_oos_rows', 0):,}",
        f"- Feature rows matched: {manifest.get('features', {}).get('feature_rows_matched', 0):,}",
        f"- Loaded features: {manifest.get('features', {}).get('loaded_features', 0):,}",
        f"- Months: {', '.join(manifest.get('event_rows', {}).get('months', []))}",
        f"- Symbols: {manifest.get('event_rows', {}).get('symbols', 0):,}",
        "",
        "## Scenario Summary",
        "",
        _table(summary, summary_cols, limit=25),
        "",
        "## Strongest Stable Feature Contrasts",
        "",
        _table(stable_features, feature_cols, limit=40),
        "",
        "## Monthly Feature Contrast Examples",
        "",
        _table(monthly_examples, month_cols, limit=40)
        if not monthly_examples.empty
        else "No monthly feature contrasts.",
        "",
        "## Interpretation",
        "",
        "- Positive robust_z_delta means missed oracle winners had higher feature values than proxy false positives.",
        "- Negative robust_z_delta means missed oracle winners had lower feature values than proxy false positives.",
        "- Sparse scenarios are useful as leads only; they are not sufficient for training decisions.",
        "- If a feature contrast is directionally stable across months, it is a candidate for a source-conditioned repair target or gate diagnostic.",
        "",
        "## Recommended Next Step",
        "",
        "Review stable feature families by source bucket, then either revise source archetype scores to expose the missing causal structure "
        "or build a small strict walk-forward repair-ranker ablation. Keep it separate from production training until it beats the fixed baseline OOS.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_report(
    *,
    event_rows_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    output_dir: Path,
    source_buckets: list[str],
    proxy_cols: list[str],
    top_fracs: list[float],
    max_features: int,
    min_group_rows: int,
    min_months: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    events, event_manifest = _load_events(
        event_rows_path,
        source_buckets=source_buckets,
        proxy_cols=proxy_cols,
        top_fracs=top_fracs,
    )
    selected_features = _read_feature_list(feature_list_csv, max_features=max_features)
    events_with_features, feature_manifest = _load_feature_values(
        events,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    feature_cols = [
        feature
        for feature in selected_features
        if feature in events_with_features.columns and events_with_features[feature].notna().any()
    ]
    summary, by_scenario, by_month = _build_contrasts(
        events_with_features,
        feature_cols=feature_cols,
        min_group_rows=min_group_rows,
        min_months=min_months,
    )

    paths = {
        "summary": output_dir / "strict_oos_feature_gap_summary.csv",
        "by_scenario": output_dir / "strict_oos_feature_contrast_by_scenario.csv",
        "by_month": output_dir / "strict_oos_feature_contrast_by_month.csv",
        "manifest": output_dir / "manifest.json",
    }
    summary.to_csv(paths["summary"], index=False)
    by_scenario.to_csv(paths["by_scenario"], index=False)
    by_month.to_csv(paths["by_month"], index=False)

    manifest = {
        "scope": "strict_oos_missed_vs_false_positive_feature_gap",
        "event_rows": event_manifest,
        "features": feature_manifest,
        "feature_list_csv": str(feature_list_csv),
        "output_dir": str(output_dir),
        "min_group_rows": int(min_group_rows),
        "min_months": int(min_months),
        "outputs": {key: str(value) for key, value in paths.items()},
        "summary_rows": int(len(summary)),
        "feature_contrast_rows": int(len(by_scenario)),
        "monthly_feature_contrast_rows": int(len(by_month)),
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    report = _write_report(
        output_dir=output_dir,
        manifest=manifest,
        summary=summary,
        by_scenario=by_scenario,
        by_month=by_month,
    )
    manifest["outputs"]["markdown"] = str(report)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-rows-path", type=Path, default=DEFAULT_EVENT_ROWS)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-buckets", type=str, default=",".join(DEFAULT_SOURCE_BUCKETS))
    parser.add_argument("--proxy-cols", type=str, default=",".join(DEFAULT_PROXY_COLUMNS))
    parser.add_argument("--top-fracs", type=str, default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--max-features", type=int, default=96)
    parser.add_argument("--min-group-rows", type=int, default=20)
    parser.add_argument("--min-months", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        event_rows_path=args.event_rows_path,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        output_dir=args.output_dir,
        source_buckets=_parse_csv(args.source_buckets, DEFAULT_SOURCE_BUCKETS),
        proxy_cols=_parse_csv(args.proxy_cols, DEFAULT_PROXY_COLUMNS),
        top_fracs=_parse_float_csv(args.top_fracs, DEFAULT_TOP_FRACS),
        max_features=args.max_features,
        min_group_rows=args.min_group_rows,
        min_months=args.min_months,
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
