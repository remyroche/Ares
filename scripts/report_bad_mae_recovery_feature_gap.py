#!/usr/bin/env python3
"""Feature-gap diagnostic for negative vs recovered bad-MAE rows.

This diagnostic is read-only. It explains whether current causal features and
source scores separate economically bad MAE>1R rows from MAE>1R rows that
recover into positive realized utility. It does not train a model, choose a
threshold, or integrate any source labels into training.
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

from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    _json_safe,
    _load_feature_store_columns,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _safe_numeric,
    _spearman,
)


DEFAULT_SELECTED_ROWS = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "utility_path_timeout_joint_risk_badmae_loss_july_source_refresh_basegateoff_v1/"
    "source_utility_path_timeout_risk_selected_rows.parquet"
)
DEFAULT_QUALITY_LABELS = Path(
    "data_perp/reports/source_tags_s10_policy_net_v17_proxy_alignment_diagnostic_july_refresh_basegateoff/"
    "quality_label_candidates.parquet"
)
DEFAULT_FEATURE_DIR = Path(
    "data_perp/reports/source_tags_s10_policy_net_v17_proxy_alignment_diagnostic_july_refresh_basegateoff/"
    "source_feature_input_2026_03_01_to_2026_07_02_by_symbol"
)
DEFAULT_FEATURE_LIST_CSV = Path(
    "data_perp/artifacts/20260702_004500_single_head_monthly_walkforward_s10_policy_net_gateoff_train_april_score_may/"
    "quality_reports/base_model_feature_importance.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "bad_mae_recovery_feature_gap_july_source_refresh_basegateoff_v1"
)
DEFAULT_SELECTIONS = (
    "utility_only",
    "utility_minus_adverse_path_timeout_0p50",
    "utility_minus_bad_mae_loss_timeout_0p50",
    "stage1_bad_mae_loss_q50_then_timeout_0p50",
)
DEFAULT_SOURCE_BUCKETS = ("all_rows", "risk_adjusted_capture_candidate")
DEFAULT_CAUSAL_GATES = ("no_gate", "low_barrier_pressure_q50")
DEFAULT_FEATURE_SETS = ("base_plus_source", "base_plus_source_v2")
GROUP_COLS = ("selection", "feature_set", "source_bucket", "causal_gate")


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


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


def _pooled_scale(left: pd.Series, right: pd.Series) -> float:
    pooled = pd.concat([left, right], ignore_index=True).dropna()
    if len(pooled) < 3:
        return float("nan")
    iqr = float(pooled.quantile(0.75) - pooled.quantile(0.25))
    if math.isfinite(iqr) and iqr > 1e-12:
        return iqr
    std = float(pooled.std(ddof=0))
    return std if math.isfinite(std) and std > 1e-12 else float("nan")


def _rank_auc(labels: pd.Series, values: pd.Series) -> float:
    y = _safe_numeric(labels)
    x = _safe_numeric(values)
    mask = y.notna() & x.notna()
    y = y.loc[mask]
    x = x.loc[mask]
    pos = int(y.eq(1.0).sum())
    neg = int(y.eq(0.0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = x.rank(method="average")
    rank_sum_pos = float(ranks.loc[y.eq(1.0)].sum())
    auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / float(pos * neg)
    return float(auc)


def _feature_family(feature: str) -> str:
    if feature.startswith("tag_"):
        return "source_tag"
    if feature.endswith("_score") or "source" in feature:
        return "source_score"
    tokens = {
        "spread": "liquidity_execution",
        "slippage": "liquidity_execution",
        "liquidity": "liquidity_execution",
        "barrier": "barrier_geometry",
        "distance": "location_geometry",
        "overextension": "location_geometry",
        "shock": "impulse",
        "impulse": "impulse",
        "breakout": "impulse",
        "range": "range_volatility",
        "vol": "range_volatility",
        "atr": "range_volatility",
        "trend": "trend_path",
        "momentum": "trend_path",
        "oi": "open_interest",
        "funding": "open_interest",
        "volume": "volume_confirmation",
        "turnover": "volume_confirmation",
        "pullback": "pullback_retest",
        "retest": "pullback_retest",
        "compression": "compression",
        "squeeze": "compression",
    }
    lower = feature.lower()
    for token, family in tokens.items():
        if token in lower:
            return family
    return "other"


def _add_bad_mae_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True, errors="coerce")
    utility = _safe_numeric(out["u_policy_net"])
    mae = _safe_numeric(out["mae_norm"])
    bars = _safe_numeric(out["bars_policy"]).fillna(24.0)
    timeout = out.get("is_timeout", False)
    if isinstance(timeout, pd.Series):
        timeout = timeout.astype(bool)
    else:
        timeout = pd.Series(bool(timeout), index=out.index)
    out["row_bad_mae_1r"] = mae.ge(1.0)
    out["row_bad_mae_negative"] = out["row_bad_mae_1r"] & utility.le(0.0)
    out["row_bad_mae_recovered"] = out["row_bad_mae_1r"] & utility.gt(0.0)
    out["row_fast_bad_mae"] = out["row_bad_mae_1r"] & bars.le(4.0)
    out["row_late_bad_mae"] = out["row_bad_mae_1r"] & (bars.ge(16.0) | timeout)
    out["row_timeout"] = timeout
    out["row_wide25"] = _safe_numeric(out["barrier"]).gt(0.025)
    return out


def _source_feature_columns(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        if col in {"__ts__", "__symbol__"}:
            continue
        if col.startswith("tag_") or col.endswith("_score"):
            cols.append(col)
    return cols


def _load_source_features(selected: pd.DataFrame, quality_labels_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not quality_labels_path.exists():
        return pd.DataFrame(index=selected.index), {"enabled": False, "reason": "missing_quality_labels"}
    q = _load_table(quality_labels_path)
    if "__ts__" not in q.columns or "__symbol__" not in q.columns:
        return pd.DataFrame(index=selected.index), {"enabled": False, "reason": "missing_join_keys"}
    source_cols = _source_feature_columns(q)
    keep_cols = ["__ts__", "__symbol__"] + source_cols
    q = q.loc[:, keep_cols].copy()
    q["__ts__"] = pd.to_datetime(q["__ts__"], utc=True, errors="coerce")
    dupes = int(q.duplicated(["__ts__", "__symbol__"]).sum())
    if dupes:
        q = q.drop_duplicates(["__ts__", "__symbol__"], keep="first")
    joined = selected[["__ts__", "__symbol__"]].merge(q, on=["__ts__", "__symbol__"], how="left", validate="many_to_one")
    out = pd.DataFrame(index=selected.index)
    loaded_cols: list[str] = []
    for col in source_cols:
        values = joined[col]
        if values.dtype == bool:
            out[col] = values.astype(float)
        elif col.startswith("tag_"):
            out[col] = values.astype(str).str.lower().isin({"1", "true", "t", "yes", "y"}).astype(float)
        else:
            out[col] = _safe_numeric(values).astype(np.float32)
        if bool(out[col].notna().any()):
            loaded_cols.append(col)
    out = out.loc[:, loaded_cols].copy()
    return out, {
        "enabled": True,
        "quality_labels_path": str(quality_labels_path),
        "requested_source_features": int(len(source_cols)),
        "loaded_source_features": int(len(loaded_cols)),
        "duplicate_timestamp_symbol_rows": dupes,
    }


def _join_features(
    selected: pd.DataFrame,
    *,
    quality_labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    feature_list = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_report = _load_feature_store_columns(
        selected,
        feature_dir=feature_dir,
        selected_features=feature_list,
    )
    source_matrix, source_report = _load_source_features(selected, quality_labels_path)
    out = selected.reset_index(drop=True).copy()
    matrices: list[pd.DataFrame] = []
    feature_cols: list[str] = []
    if not feature_matrix.empty:
        fm = feature_matrix.reset_index(drop=True)
        matrices.append(fm)
        feature_cols.extend(list(fm.columns))
    if not source_matrix.empty:
        sm = source_matrix.reset_index(drop=True)
        overlap = set(feature_cols).intersection(sm.columns)
        if overlap:
            sm = sm.rename(columns={col: f"source_{col}" for col in overlap})
        matrices.append(sm)
        feature_cols.extend(list(sm.columns))
    if matrices:
        out = pd.concat([out] + matrices, axis=1)
    return out, feature_cols, {"feature_store": feature_report, "source_features": source_report}


def _contrast_group(
    group: pd.DataFrame,
    *,
    feature_cols: list[str],
    negative_mask: pd.Series,
    recovered_mask: pd.Series,
    prefix: dict[str, Any],
    min_rows: int,
) -> pd.DataFrame:
    negative = group.loc[negative_mask.reindex(group.index, fill_value=False)]
    recovered = group.loc[recovered_mask.reindex(group.index, fill_value=False)]
    if len(negative) < min_rows or len(recovered) < min_rows:
        return pd.DataFrame()
    target = pd.Series(np.nan, index=group.index, dtype=np.float64)
    target.loc[negative.index] = 1.0
    target.loc[recovered.index] = 0.0

    rows: list[dict[str, Any]] = []
    for feature in feature_cols:
        left = _safe_numeric(negative[feature]).dropna()
        right = _safe_numeric(recovered[feature]).dropna()
        if len(left) < min_rows or len(right) < min_rows:
            continue
        left_median = float(left.median())
        right_median = float(right.median())
        median_diff = left_median - right_median
        scale = _pooled_scale(left, right)
        robust_z = median_diff / scale if math.isfinite(scale) and scale > 0.0 else float("nan")
        auc = _rank_auc(target, group[feature])
        best_auc = max(auc, 1.0 - auc) if math.isfinite(auc) else float("nan")
        direction = "higher_in_negative" if math.isfinite(auc) and auc >= 0.5 else "higher_in_recovered"
        rows.append(
            {
                **prefix,
                "feature": feature,
                "feature_family": _feature_family(feature),
                "negative_n": int(len(left)),
                "recovered_n": int(len(right)),
                "negative_mean": float(left.mean()),
                "recovered_mean": float(right.mean()),
                "mean_diff_negative_minus_recovered": float(left.mean() - right.mean()),
                "negative_median": left_median,
                "recovered_median": right_median,
                "median_diff_negative_minus_recovered": float(median_diff),
                "pooled_scale": float(scale),
                "robust_z_delta": float(robust_z),
                "abs_robust_z_delta": float(abs(robust_z)) if math.isfinite(robust_z) else float("nan"),
                "auc_negative_high": float(auc),
                "best_auc": float(best_auc),
                "best_direction": direction,
                "spearman_negative_target": _spearman(target, group[feature]),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["best_auc", "abs_robust_z_delta"], ascending=False, kind="mergesort"
    )


def _bucket_metrics(group: pd.DataFrame, prefix: dict[str, Any]) -> dict[str, Any]:
    bad = group.loc[group["row_bad_mae_1r"]]
    negative = group.loc[group["row_bad_mae_negative"]]
    recovered = group.loc[group["row_bad_mae_recovered"]]
    fast_negative = group.loc[group["row_bad_mae_negative"] & group["row_fast_bad_mae"]]
    fast_recovered = group.loc[group["row_bad_mae_recovered"] & group["row_fast_bad_mae"]]
    return {
        **prefix,
        "rows": int(len(group)),
        "bad_mae_rows": int(len(bad)),
        "bad_mae_rate": float(len(bad) / len(group)) if len(group) else float("nan"),
        "bad_mae_negative_rows": int(len(negative)),
        "bad_mae_recovered_rows": int(len(recovered)),
        "fast_bad_mae_negative_rows": int(len(fast_negative)),
        "fast_bad_mae_recovered_rows": int(len(fast_recovered)),
        "mean_u": _safe_mean(group["u_policy_net"]),
        "negative_mean_u": _safe_mean(negative["u_policy_net"]),
        "recovered_mean_u": _safe_mean(recovered["u_policy_net"]),
        "negative_p90_mae": _safe_quantile(negative["mae_norm"], 0.90),
        "recovered_p90_mae": _safe_quantile(recovered["mae_norm"], 0.90),
        "timeout_rate": _safe_mean(group["row_timeout"].astype(float)),
        "wide25_rate": _safe_mean(group["row_wide25"].astype(float)),
        "top_symbol_share": (
            float(group["__symbol__"].value_counts(normalize=True).iloc[0]) if len(group) else float("nan")
        ),
    }


def _summarize_contrasts(contrasts: pd.DataFrame, bucket_metrics: pd.DataFrame) -> pd.DataFrame:
    if contrasts.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["scope", "contrast"] + list(GROUP_COLS)
    for key, group in contrasts.groupby(keys, dropna=False, observed=True):
        context = dict(zip(keys, key, strict=False))
        valid = group.dropna(subset=["best_auc"]).sort_values(
            ["best_auc", "abs_robust_z_delta"], ascending=False, kind="mergesort"
        )
        if valid.empty:
            continue
        strong_z = int(valid["abs_robust_z_delta"].ge(0.50).sum())
        strong_auc = int(valid["best_auc"].ge(0.60).sum())
        top = valid.iloc[0]
        metric_mask = pd.Series(True, index=bucket_metrics.index)
        for col, value in context.items():
            if col in bucket_metrics.columns:
                metric_mask &= bucket_metrics[col].astype(str).eq(str(value))
        metrics_row = bucket_metrics.loc[metric_mask].head(1)
        if strong_z >= 3 and float(top["best_auc"]) >= 0.60:
            diagnosis = "feature_separation_present"
        elif float(top["best_auc"]) >= 0.62:
            diagnosis = "single_feature_signal"
        else:
            diagnosis = "weak_feature_separation"
        row = {
            **context,
            "features_tested": int(len(valid)),
            "strong_z_features": strong_z,
            "strong_auc_features": strong_auc,
            "top_feature": top["feature"],
            "top_feature_family": top["feature_family"],
            "top_best_auc": float(top["best_auc"]),
            "top_direction": top["best_direction"],
            "top_robust_z_delta": float(top["robust_z_delta"]),
            "diagnosis": diagnosis,
        }
        if not metrics_row.empty:
            for col in [
                "rows",
                "bad_mae_rows",
                "bad_mae_negative_rows",
                "bad_mae_recovered_rows",
                "fast_bad_mae_negative_rows",
                "fast_bad_mae_recovered_rows",
                "mean_u",
                "negative_mean_u",
                "recovered_mean_u",
            ]:
                row[col] = metrics_row.iloc[0].get(col)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["top_best_auc", "strong_z_features"], ascending=False)


def _write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    bucket_metrics: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "bad_mae_recovery_feature_gap_report.md"
    lines: list[str] = []
    lines.append("# Bad-MAE Recovery Feature Gap")
    lines.append("")
    lines.append("Diagnostic-only contrast of MAE>1R rows with negative utility vs MAE>1R rows that recovered.")
    lines.append("")
    lines.append(f"Selected rows path: `{manifest.get('selected_rows_path')}`")
    lines.append(f"Rows after filters: `{manifest.get('filtered_rows')}`")
    lines.append(f"Feature columns tested: `{manifest.get('feature_cols')}`")
    lines.append("")
    lines.append("## Summary")
    summary_cols = [
        "scope",
        "contrast",
        "selection",
        "feature_set",
        "source_bucket",
        "causal_gate",
        "bad_mae_negative_rows",
        "bad_mae_recovered_rows",
        "fast_bad_mae_negative_rows",
        "fast_bad_mae_recovered_rows",
        "top_feature",
        "top_feature_family",
        "top_best_auc",
        "top_direction",
        "top_robust_z_delta",
        "strong_z_features",
        "diagnosis",
    ]
    lines.append(_table(summary, summary_cols, limit=40))
    lines.append("")
    lines.append("## Bucket Metrics")
    metric_cols = [
        "scope",
        "selection",
        "feature_set",
        "source_bucket",
        "causal_gate",
        "rows",
        "bad_mae_rate",
        "bad_mae_negative_rows",
        "bad_mae_recovered_rows",
        "fast_bad_mae_negative_rows",
        "fast_bad_mae_recovered_rows",
        "negative_mean_u",
        "recovered_mean_u",
        "top_symbol_share",
    ]
    lines.append(_table(bucket_metrics.sort_values("bad_mae_negative_rows", ascending=False), metric_cols, limit=40))
    lines.append("")
    lines.append("## Top Feature Contrasts")
    contrast_cols = [
        "scope",
        "contrast",
        "selection",
        "feature_set",
        "source_bucket",
        "causal_gate",
        "feature",
        "feature_family",
        "negative_n",
        "recovered_n",
        "best_auc",
        "best_direction",
        "robust_z_delta",
        "negative_median",
        "recovered_median",
    ]
    lines.append(_table(contrasts.sort_values(["best_auc", "abs_robust_z_delta"], ascending=False), contrast_cols, limit=60))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `feature_separation_present` means multiple current causal features distinguish negative from recovered bad-MAE rows.")
    lines.append("- `weak_feature_separation` means simple selector tuning is unlikely to clean bad-MAE without new features or a different row universe.")
    lines.append("- A feature direction of `higher_in_negative` means larger values are associated with economically bad MAE rows.")
    path.write_text("\n".join(lines) + "\n")
    return path


def run_report(
    *,
    selected_rows_path: Path,
    quality_labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    output_dir: Path,
    selections: list[str],
    feature_sets: list[str],
    source_buckets: list[str],
    causal_gates: list[str],
    max_feature_store_features: int,
    min_rows: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = _load_table(selected_rows_path)
    selected = _add_bad_mae_flags(selected)
    raw_rows = int(len(selected))
    mask = selected["selection"].isin(selections)
    mask &= selected["feature_set"].isin(feature_sets)
    mask &= selected["source_bucket"].isin(source_buckets)
    mask &= selected["causal_gate"].isin(causal_gates)
    selected = selected.loc[mask].copy().reset_index(drop=True)

    selected, feature_cols, join_report = _join_features(
        selected,
        quality_labels_path=quality_labels_path,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        max_feature_store_features=max_feature_store_features,
    )
    feature_cols = [col for col in feature_cols if col in selected.columns and bool(selected[col].notna().any())]

    bucket_rows: list[dict[str, Any]] = []
    contrast_frames: list[pd.DataFrame] = []
    grouped = selected.groupby(list(GROUP_COLS), dropna=False, observed=True)
    for key, group in grouped:
        prefix = dict(zip(GROUP_COLS, key, strict=False))
        prefix["scope"] = "selector"
        bucket_rows.append(_bucket_metrics(group, prefix))
        contrast_frames.append(
            _contrast_group(
                group,
                feature_cols=feature_cols,
                negative_mask=group["row_bad_mae_negative"],
                recovered_mask=group["row_bad_mae_recovered"],
                prefix={**prefix, "contrast": "bad_mae_negative_vs_recovered"},
                min_rows=min_rows,
            )
        )
        contrast_frames.append(
            _contrast_group(
                group,
                feature_cols=feature_cols,
                negative_mask=group["row_bad_mae_negative"] & group["row_fast_bad_mae"],
                recovered_mask=group["row_bad_mae_recovered"] & group["row_fast_bad_mae"],
                prefix={**prefix, "contrast": "fast_bad_mae_negative_vs_recovered"},
                min_rows=min_rows,
            )
        )

    overall_prefix = {
        "scope": "all_filtered",
        "selection": "all",
        "feature_set": "all",
        "source_bucket": "all",
        "causal_gate": "all",
    }
    bucket_rows.append(_bucket_metrics(selected, overall_prefix))
    contrast_frames.append(
        _contrast_group(
            selected,
            feature_cols=feature_cols,
            negative_mask=selected["row_bad_mae_negative"],
            recovered_mask=selected["row_bad_mae_recovered"],
            prefix={**overall_prefix, "contrast": "bad_mae_negative_vs_recovered"},
            min_rows=min_rows,
        )
    )
    contrast_frames.append(
        _contrast_group(
            selected,
            feature_cols=feature_cols,
            negative_mask=selected["row_bad_mae_negative"] & selected["row_fast_bad_mae"],
            recovered_mask=selected["row_bad_mae_recovered"] & selected["row_fast_bad_mae"],
            prefix={**overall_prefix, "contrast": "fast_bad_mae_negative_vs_recovered"},
            min_rows=min_rows,
        )
    )

    contrasts = pd.concat([frame for frame in contrast_frames if not frame.empty], ignore_index=True) if contrast_frames else pd.DataFrame()
    bucket_metrics = pd.DataFrame(bucket_rows)
    summary = _summarize_contrasts(contrasts, bucket_metrics)

    contrast_out = output_dir / "bad_mae_recovery_feature_gap_contrasts.csv"
    summary_out = output_dir / "bad_mae_recovery_feature_gap_summary.csv"
    bucket_out = output_dir / "bad_mae_recovery_bucket_metrics.csv"
    contrasts.to_csv(contrast_out, index=False)
    summary.to_csv(summary_out, index=False)
    bucket_metrics.to_csv(bucket_out, index=False)

    manifest = {
        "scope": "bad_mae_recovery_feature_gap",
        "selected_rows_path": str(selected_rows_path),
        "quality_labels_path": str(quality_labels_path),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "output_dir": str(output_dir),
        "raw_rows": raw_rows,
        "filtered_rows": int(len(selected)),
        "months": sorted(selected["period"].dropna().astype(str).unique().tolist()) if "period" in selected.columns else [],
        "symbols": int(selected["__symbol__"].nunique(dropna=True)) if "__symbol__" in selected.columns else 0,
        "selections": selections,
        "feature_sets": feature_sets,
        "source_buckets": source_buckets,
        "causal_gates": causal_gates,
        "min_rows": int(min_rows),
        "feature_cols": int(len(feature_cols)),
        "join_report": join_report,
        "outputs": {
            "summary": str(summary_out),
            "contrasts": str(contrast_out),
            "bucket_metrics": str(bucket_out),
            "manifest": str(output_dir / "manifest.json"),
            "markdown": str(output_dir / "bad_mae_recovery_feature_gap_report.md"),
        },
    }
    report_path = _write_report(output_dir, summary, contrasts, bucket_metrics, manifest)
    manifest["outputs"]["markdown"] = str(report_path)
    (output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-rows-path", type=Path, default=DEFAULT_SELECTED_ROWS)
    parser.add_argument("--quality-labels-path", type=Path, default=DEFAULT_QUALITY_LABELS)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selections", type=str, default=",".join(DEFAULT_SELECTIONS))
    parser.add_argument("--feature-sets", type=str, default=",".join(DEFAULT_FEATURE_SETS))
    parser.add_argument("--source-buckets", type=str, default=",".join(DEFAULT_SOURCE_BUCKETS))
    parser.add_argument("--causal-gates", type=str, default=",".join(DEFAULT_CAUSAL_GATES))
    parser.add_argument("--max-feature-store-features", type=int, default=96)
    parser.add_argument("--min-rows", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        selected_rows_path=args.selected_rows_path,
        quality_labels_path=args.quality_labels_path,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        output_dir=args.output_dir,
        selections=_parse_csv(args.selections, DEFAULT_SELECTIONS),
        feature_sets=_parse_csv(args.feature_sets, DEFAULT_FEATURE_SETS),
        source_buckets=_parse_csv(args.source_buckets, DEFAULT_SOURCE_BUCKETS),
        causal_gates=_parse_csv(args.causal_gates, DEFAULT_CAUSAL_GATES),
        max_feature_store_features=int(args.max_feature_store_features),
        min_rows=int(args.min_rows),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
