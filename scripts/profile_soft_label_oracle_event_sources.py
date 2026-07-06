#!/usr/bin/env python3
"""Profile where soft-label oracle winners come from.

This is a label QA diagnostic, not a model-training step. It ranks a holdout
month by a chosen soft label, then checks whether the top rows are concentrated
by symbol/timestamp, recent symbol history, path metrics, or causal event
features that are available at decision time.
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
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _rank_top_indices,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    _all_targets,
    _event_confirmation_features,
    _parse_csv,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/soft_label_oracle_event_source_profile_v1")
DEFAULT_COMPARISON_GAP_DIR = Path(
    "data_perp/reports/soft_label_proxy_feature_gap_e9_low_mae_event_only_june_top005_v1"
)

LOW_GOOD_EVENT_FEATURES = {
    "time_since_event_extreme_12h",
    "spread_proxy_hl_range_bps_robust_z",
    "spread_proxy_abs_return_bps_robust_z",
    "median_spread_bps",
    "distance_to_support_daily_vwap_atr",
    "distance_to_resistance_daily_vwap_atr",
    "up_barrier_pressure_daily_vwap",
    "down_barrier_pressure_daily_vwap",
    "pullback_2",
    "pullback_4",
    "pullback_8",
    "pullback_48",
    "dist_from_low_event_12h",
    "dist_from_low_48h",
    "dist_from_high_48h",
    "oiw_pos_delta_entry_dist_1d_atr",
    "oiw_pos_delta_entry_dist_7d_atr",
    "oiw_pos_delta_entry_dist_14d_atr",
    "dist_oiw_abs_delta_12h_atr",
    "dist_oiw_signed_delta_12h_atr",
}


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _effective_n(values: pd.Series) -> float:
    counts = values.astype(str).value_counts(dropna=False)
    if counts.empty:
        return 0.0
    shares = counts.to_numpy(dtype=np.float64) / float(counts.sum())
    denom = float(np.sum(shares * shares))
    return 1.0 / denom if denom > 0.0 else 0.0


def _row_key(frame: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(frame["__ts__"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    return ts.fillna("") + "|" + frame["__symbol__"].astype(str)


def _week_label(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.to_period("W-SUN").astype(str)


def _group_metrics(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    mask: pd.Series,
    group: str,
) -> dict[str, Any]:
    mask = mask.fillna(False).astype(bool)
    selected = metrics.loc[mask]
    selected_frame = frame.loc[mask]
    selected_target = target.loc[mask]
    u = selected.get("u_policy_net", pd.Series(dtype=float))
    return {
        "group": group,
        "rows": int(mask.sum()),
        "mean_target_soft": _safe_mean(selected_target.get("target_soft")),
        "hard_rate": _safe_mean(selected_target.get("target_hard")),
        "mean_u": _safe_mean(u),
        "median_u": _safe_quantile(u, 0.50),
        "q10_u": _safe_quantile(u, 0.10),
        "hit_u": _safe_mean(u > 0.0),
        "mean_ret_net": _safe_mean(selected.get("ret_net")),
        "mean_barrier": _safe_mean(selected.get("barrier")),
        "wide_barrier_25bps_rate": _safe_mean(selected.get("barrier") > 0.025),
        "mean_mae_norm": _safe_mean(selected.get("mae_norm")),
        "p90_mae_norm": _safe_quantile(selected.get("mae_norm"), 0.90),
        "bad_mae_1r_rate": _safe_mean(selected.get("mae_norm") >= 1.0),
        "mean_mfe_norm": _safe_mean(selected.get("mfe_norm")),
        "mean_bars_to_mfe": _safe_mean(selected.get("bars_to_mfe")),
        "timeout_rate": _safe_mean(selected.get("is_timeout").astype(float)) if len(selected) else float("nan"),
        "unique_symbols": int(selected_frame["__symbol__"].nunique()) if len(selected_frame) else 0,
        "symbol_effective_n": _effective_n(selected_frame["__symbol__"]) if len(selected_frame) else 0.0,
        "top_symbol_share": (
            float(selected_frame["__symbol__"].value_counts(normalize=True, dropna=False).iloc[0])
            if len(selected_frame)
            else 0.0
        ),
        "unique_timestamps": int(selected_frame["__ts__"].nunique()) if len(selected_frame) else 0,
        "timestamp_effective_n": _effective_n(selected_frame["__ts__"].astype(str)) if len(selected_frame) else 0.0,
        "top_timestamp_share": (
            float(selected_frame["__ts__"].astype(str).value_counts(normalize=True, dropna=False).iloc[0])
            if len(selected_frame)
            else 0.0
        ),
    }


def _build_clusters(oracle: pd.DataFrame, metrics: pd.DataFrame, *, gap_hours: float) -> tuple[pd.DataFrame, pd.Series]:
    if oracle.empty:
        return pd.DataFrame(), pd.Series(dtype=object)
    work = oracle[["__ts__", "__symbol__"]].copy()
    work["__source_pos__"] = np.arange(len(work), dtype=np.int64)
    work["u_policy_net"] = metrics["u_policy_net"].to_numpy(dtype=np.float64, copy=False)
    work["target_soft"] = oracle["target_soft"].to_numpy(dtype=np.float64, copy=False)
    work["mae_norm"] = metrics["mae_norm"].to_numpy(dtype=np.float64, copy=False)
    work["mfe_norm"] = metrics["mfe_norm"].to_numpy(dtype=np.float64, copy=False)
    work = work.sort_values(["__symbol__", "__ts__"], kind="mergesort").reset_index(drop=True)

    gap = pd.Timedelta(hours=float(gap_hours))
    cluster_numbers: list[int] = []
    cluster = -1
    prev_symbol: str | None = None
    prev_ts: pd.Timestamp | None = None
    for _, row in work.iterrows():
        symbol = str(row["__symbol__"])
        ts = pd.Timestamp(row["__ts__"])
        if prev_symbol != symbol or prev_ts is None or ts - prev_ts > gap:
            cluster += 1
        cluster_numbers.append(cluster)
        prev_symbol = symbol
        prev_ts = ts
    work["cluster_n"] = cluster_numbers
    work["cluster_id"] = work["__symbol__"].astype(str) + "#" + work["cluster_n"].astype(str)
    cluster_ids_by_pos = work.set_index("__source_pos__")["cluster_id"].sort_index()

    summary = (
        work.groupby("cluster_id", dropna=False)
        .agg(
            symbol=("__symbol__", "first"),
            start_ts=("__ts__", "min"),
            end_ts=("__ts__", "max"),
            rows=("__ts__", "size"),
            mean_target_soft=("target_soft", "mean"),
            mean_u=("u_policy_net", "mean"),
            min_u=("u_policy_net", "min"),
            mean_mae_norm=("mae_norm", "mean"),
            max_mae_norm=("mae_norm", "max"),
            mean_mfe_norm=("mfe_norm", "mean"),
        )
        .reset_index()
    )
    summary["duration_hours"] = (
        pd.to_datetime(summary["end_ts"]) - pd.to_datetime(summary["start_ts"])
    ).dt.total_seconds() / 3600.0
    summary = summary.sort_values(["rows", "mean_u"], ascending=[False, False]).reset_index(drop=True)
    return summary, cluster_ids_by_pos


def _good_rank(frame: pd.DataFrame, feature: str) -> pd.Series:
    raw = _safe_numeric(frame[feature])
    rank = raw.groupby(frame["__ts__"], dropna=False).rank(method="average", pct=True)
    if feature in LOW_GOOD_EVENT_FEATURES or feature.startswith("event_xs_lo_"):
        rank = 1.0 - rank
    return rank.clip(0.0, 1.0)


def _feature_summary(
    *,
    valid: pd.DataFrame,
    oracle_mask: pd.Series,
    comparison_mask: pd.Series,
    event_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_mask = pd.Series(True, index=valid.index)
    for feature in event_cols:
        values = _safe_numeric(valid[feature])
        if int(values.notna().sum()) < 20:
            continue
        good_rank = _good_rank(valid, feature)
        oracle_values = values.loc[oracle_mask]
        comparison_values = values.loc[comparison_mask]
        row = {
            "feature": feature,
            "finite_frac": float(values.notna().mean()),
            "oracle_median": _safe_quantile(oracle_values, 0.50),
            "all_median": _safe_quantile(values.loc[all_mask], 0.50),
            "oracle_good_rank_mean": _safe_mean(good_rank.loc[oracle_mask]),
            "all_good_rank_mean": _safe_mean(good_rank.loc[all_mask]),
            "oracle_minus_all_good_rank": _safe_mean(good_rank.loc[oracle_mask]) - _safe_mean(good_rank.loc[all_mask]),
        }
        if bool(comparison_mask.any()):
            row.update(
                {
                    "comparison_median": _safe_quantile(comparison_values, 0.50),
                    "comparison_good_rank_mean": _safe_mean(good_rank.loc[comparison_mask]),
                    "oracle_minus_comparison_good_rank": (
                        _safe_mean(good_rank.loc[oracle_mask]) - _safe_mean(good_rank.loc[comparison_mask])
                    ),
                }
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_col = "oracle_minus_comparison_good_rank" if "oracle_minus_comparison_good_rank" in out.columns else "oracle_minus_all_good_rank"
    out["abs_sort_gap"] = _safe_numeric(out[sort_col]).abs()
    return out.sort_values("abs_sort_gap", ascending=False).drop(columns=["abs_sort_gap"]).reset_index(drop=True)


def _write_markdown(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    group_summary: pd.DataFrame,
    weekly_summary: pd.DataFrame,
    symbol_summary: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
) -> Path:
    path = output_dir / "oracle_event_source_profile.md"

    def table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[col for col in cols if col in frame.columns]].copy()
        if limit is not None:
            view = view.head(limit)
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
        return view.to_markdown(index=False)

    lines = [
        "# Soft Label Oracle Event Source Profile",
        "",
        "Scope: no model training. Profiles oracle top soft-label rows inside the execution envelope.",
        "",
        f"Month: `{manifest['month']}`",
        f"Label arm: `{manifest['label_arm']}`",
        f"Top fraction: `{manifest['top_frac']}`",
        f"Oracle rows: `{manifest['oracle_rows']}`",
        f"Comparison gap dir: `{manifest.get('comparison_gap_dir') or ''}`",
        "",
        "## Concentration",
        "",
        f"- Unique symbols: `{manifest['concentration']['unique_symbols']}`",
        f"- Top symbol share: `{manifest['concentration']['top_symbol_share']:.4f}`",
        f"- Unique timestamps: `{manifest['concentration']['unique_timestamps']}`",
        f"- Top timestamp share: `{manifest['concentration']['top_timestamp_share']:.4f}`",
        f"- Multi-row clusters: `{manifest['concentration']['multi_row_clusters']}`",
        f"- Rows in multi-row clusters: `{manifest['concentration']['rows_in_multi_row_clusters']}`",
        "",
        "## Symbol Freshness",
        "",
        f"- Label-history age median days: `{manifest['symbol_age']['label_age_median_days']:.2f}`",
        f"- Feature-history age median days: `{manifest['symbol_age']['feature_age_median_days']:.2f}`",
        f"- Rows with <=7d label history: `{manifest['symbol_age']['rows_label_age_le_7d']}`",
        f"- Rows with <=14d label history: `{manifest['symbol_age']['rows_label_age_le_14d']}`",
        "",
        "## Economic Groups",
        "",
        table(
            group_summary,
            [
                "group",
                "rows",
                "mean_target_soft",
                "mean_u",
                "hit_u",
                "q10_u",
                "bad_mae_1r_rate",
                "wide_barrier_25bps_rate",
                "timeout_rate",
                "top_symbol_share",
                "top_timestamp_share",
            ],
        ),
        "",
        "## Oracle Weeks",
        "",
        table(
            weekly_summary,
            [
                "week",
                "rows",
                "unique_symbols",
                "mean_target_soft",
                "mean_u",
                "q10_u",
                "bad_mae_1r_rate",
                "top_symbol_share",
            ],
        ),
        "",
        "## Top Symbols",
        "",
        table(
            symbol_summary,
            [
                "symbol",
                "rows",
                "first_ts",
                "last_ts",
                "mean_target_soft",
                "mean_u",
                "min_u",
                "mean_label_age_days",
                "mean_feature_age_days",
            ],
            limit=20,
        ),
        "",
        "## Clusters",
        "",
        table(
            cluster_summary,
            [
                "cluster_id",
                "symbol",
                "start_ts",
                "end_ts",
                "rows",
                "duration_hours",
                "mean_target_soft",
                "mean_u",
                "min_u",
                "max_mae_norm",
            ],
            limit=30,
        ),
        "",
        "## Event Feature Contrast",
        "",
        table(
            feature_summary,
            [
                "feature",
                "oracle_good_rank_mean",
                "comparison_good_rank_mean",
                "oracle_minus_comparison_good_rank",
                "all_good_rank_mean",
                "oracle_median",
                "comparison_median",
                "all_median",
            ],
            limit=40,
        ),
        "",
        "## Outputs",
        "",
        f"- Oracle rows: `{manifest['outputs']['oracle_rows']}`",
        f"- Symbol summary: `{manifest['outputs']['symbol_summary']}`",
        f"- Timestamp summary: `{manifest['outputs']['timestamp_summary']}`",
        f"- Cluster summary: `{manifest['outputs']['cluster_summary']}`",
        f"- Feature summary: `{manifest['outputs']['feature_summary']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_profile(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    month: str,
    label_arm: str,
    top_frac: float,
    event_feature_store_features: list[str],
    comparison_gap_dir: Path | None,
    cluster_gap_hours: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_labels(labels_path)
    frame["__row_id__"] = np.arange(len(frame), dtype=np.int64)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    selected_features = list(dict.fromkeys(list(selected_features) + list(event_feature_store_features)))
    feature_matrix, feature_store_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    if not feature_matrix.empty:
        frame = pd.concat(
            [
                frame.drop(columns=[col for col in feature_matrix.columns if col in frame.columns]),
                feature_matrix.astype(np.float32, copy=False),
            ],
            axis=1,
        ).copy()

    event_diag, event_report = _event_confirmation_features(frame, event_features=event_feature_store_features)
    if not event_diag.empty:
        frame = pd.concat([frame, event_diag.astype(np.float32, copy=False)], axis=1).copy()

    metrics = _path_metrics(frame)
    targets, descriptions = _all_targets(frame, metrics)
    if label_arm not in targets:
        raise ValueError(f"Unknown label arm {label_arm!r}; available arms: {sorted(targets)}")
    target = targets[label_arm]

    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    valid_mask_full = month_period == str(month)
    if int(valid_mask_full.sum()) < 100:
        raise ValueError(f"Insufficient rows for month={month}: {int(valid_mask_full.sum())}")

    valid = frame.loc[valid_mask_full].reset_index(drop=True).copy()
    valid_metrics = metrics.loc[valid_mask_full].reset_index(drop=True).copy()
    valid_target = target.loc[valid_mask_full].reset_index(drop=True).copy()
    valid["target_soft"] = valid_target["target_soft"].to_numpy(dtype=np.float64, copy=False)
    valid["target_hard"] = valid_target["target_hard"].to_numpy(dtype=np.float64, copy=False)

    oracle_idx = _rank_top_indices(valid_target["target_soft"], top_frac)
    oracle_mask = pd.Series(False, index=valid.index)
    oracle_mask.iloc[oracle_idx] = True

    comparison_mask = pd.Series(False, index=valid.index)
    comparison_gap_dir_str: str | None = None
    if comparison_gap_dir is not None and comparison_gap_dir.exists():
        false_positive_path = comparison_gap_dir / "false_positives.csv"
        if false_positive_path.exists():
            comparison_gap_dir_str = str(comparison_gap_dir)
            false_positives = pd.read_csv(false_positive_path)
            if {"__ts__", "__symbol__"}.issubset(false_positives.columns):
                false_positives["__ts__"] = pd.to_datetime(false_positives["__ts__"], errors="coerce")
                fp_keys = set(_row_key(false_positives).tolist())
                comparison_mask = _row_key(valid).isin(fp_keys)

    label_first_ts = frame.groupby("__symbol__", sort=False)["__ts__"].min()
    event_cols_raw = [feature for feature in event_feature_store_features if feature in frame.columns]
    event_cols_diag = [feature for feature in event_diag.columns if feature in frame.columns]
    event_cols = list(dict.fromkeys(event_cols_raw + event_cols_diag))
    if event_cols_raw:
        feature_seen = frame[event_cols_raw].notna().any(axis=1)
        feature_first_ts = frame.loc[feature_seen].groupby("__symbol__", sort=False)["__ts__"].min()
    else:
        feature_first_ts = pd.Series(dtype="datetime64[ns]")

    oracle = valid.loc[oracle_mask].reset_index(drop=True).copy()
    oracle_metrics = valid_metrics.loc[oracle_mask].reset_index(drop=True).copy()
    oracle_target = valid_target.loc[oracle_mask].reset_index(drop=True).copy()
    oracle["target_soft"] = oracle_target["target_soft"].to_numpy(dtype=np.float64, copy=False)
    oracle["target_hard"] = oracle_target["target_hard"].to_numpy(dtype=np.float64, copy=False)

    label_age_days = []
    feature_age_days = []
    for _, row in oracle.iterrows():
        symbol = row["__symbol__"]
        ts = pd.Timestamp(row["__ts__"])
        label_first = label_first_ts.get(symbol, pd.NaT)
        feature_first = feature_first_ts.get(symbol, pd.NaT)
        label_age_days.append((ts - pd.Timestamp(label_first)).total_seconds() / 86400.0 if pd.notna(label_first) else np.nan)
        feature_age_days.append((ts - pd.Timestamp(feature_first)).total_seconds() / 86400.0 if pd.notna(feature_first) else np.nan)
    oracle["label_age_days"] = label_age_days
    oracle["feature_age_days"] = feature_age_days
    oracle["event_feature_coverage_frac"] = (
        oracle[event_cols_raw].notna().mean(axis=1).to_numpy(dtype=np.float64, copy=False) if event_cols_raw else np.nan
    )

    cluster_summary, cluster_ids = _build_clusters(oracle, oracle_metrics, gap_hours=cluster_gap_hours)
    if len(cluster_ids):
        oracle["cluster_id"] = cluster_ids.reindex(np.arange(len(oracle))).to_numpy(dtype=object, copy=False)
    else:
        oracle["cluster_id"] = ""

    path_cols = [
        "u_policy_net",
        "ret_net",
        "barrier",
        "mfe_norm",
        "mae_norm",
        "bars_to_mfe",
        "bars_policy",
        "is_timeout",
    ]
    for col in path_cols:
        oracle[col] = oracle_metrics[col].to_numpy(copy=False)

    oracle["week"] = _week_label(oracle["__ts__"])
    oracle_export_cols = [
        "__ts__",
        "__symbol__",
        "week",
        "cluster_id",
        "label_age_days",
        "feature_age_days",
        "event_feature_coverage_frac",
        "target_soft",
        "target_hard",
    ] + path_cols + event_cols
    oracle_export = oracle[[col for col in oracle_export_cols if col in oracle.columns]].copy()

    group_rows = [
        _group_metrics(
            frame=valid,
            metrics=valid_metrics,
            target=valid_target,
            mask=pd.Series(True, index=valid.index),
            group="all_valid_month",
        ),
        _group_metrics(
            frame=valid,
            metrics=valid_metrics,
            target=valid_target,
            mask=oracle_mask,
            group="oracle_top",
        ),
    ]
    if bool(comparison_mask.any()):
        group_rows.append(
            _group_metrics(
                frame=valid,
                metrics=valid_metrics,
                target=valid_target,
                mask=comparison_mask,
                group="comparison_false_positives",
            )
        )
    group_summary = pd.DataFrame(group_rows)

    weekly_summary = (
        oracle.assign(week=_week_label(oracle["__ts__"]))
        .groupby("week", dropna=False)
        .agg(
            rows=("__ts__", "size"),
            unique_symbols=("__symbol__", "nunique"),
            mean_target_soft=("target_soft", "mean"),
            mean_u=("u_policy_net", "mean"),
            q10_u=("u_policy_net", lambda values: _safe_quantile(values, 0.10)),
            bad_mae_1r_rate=("mae_norm", lambda values: _safe_mean(_safe_numeric(values) >= 1.0)),
            top_symbol_share=("__symbol__", lambda values: float(values.value_counts(normalize=True).iloc[0]) if len(values) else 0.0),
        )
        .reset_index()
        if len(oracle)
        else pd.DataFrame()
    )

    symbol_summary = (
        oracle.groupby("__symbol__", dropna=False)
        .agg(
            rows=("__ts__", "size"),
            first_ts=("__ts__", "min"),
            last_ts=("__ts__", "max"),
            mean_target_soft=("target_soft", "mean"),
            mean_u=("u_policy_net", "mean"),
            min_u=("u_policy_net", "min"),
            mean_label_age_days=("label_age_days", "mean"),
            mean_feature_age_days=("feature_age_days", "mean"),
            mean_event_feature_coverage=("event_feature_coverage_frac", "mean"),
        )
        .reset_index()
        .rename(columns={"__symbol__": "symbol"})
        .sort_values(["rows", "mean_u"], ascending=[False, False])
        .reset_index(drop=True)
        if len(oracle)
        else pd.DataFrame()
    )

    timestamp_summary = (
        oracle.groupby("__ts__", dropna=False)
        .agg(
            rows=("__symbol__", "size"),
            unique_symbols=("__symbol__", "nunique"),
            symbols=("__symbol__", lambda values: ",".join(values.astype(str).head(20))),
            mean_target_soft=("target_soft", "mean"),
            mean_u=("u_policy_net", "mean"),
            min_u=("u_policy_net", "min"),
        )
        .reset_index()
        .sort_values(["rows", "__ts__"], ascending=[False, True])
        .reset_index(drop=True)
        if len(oracle)
        else pd.DataFrame()
    )

    feature_summary = _feature_summary(
        valid=valid,
        oracle_mask=oracle_mask,
        comparison_mask=comparison_mask,
        event_cols=event_cols,
    )

    paths = {
        "oracle_rows": output_dir / "oracle_rows_profile.csv",
        "group_summary": output_dir / "group_summary.csv",
        "weekly_summary": output_dir / "weekly_summary.csv",
        "symbol_summary": output_dir / "symbol_summary.csv",
        "timestamp_summary": output_dir / "timestamp_summary.csv",
        "cluster_summary": output_dir / "cluster_summary.csv",
        "feature_summary": output_dir / "event_feature_summary.csv",
        "manifest": output_dir / "manifest.json",
    }
    oracle_export.to_csv(paths["oracle_rows"], index=False)
    group_summary.to_csv(paths["group_summary"], index=False)
    weekly_summary.to_csv(paths["weekly_summary"], index=False)
    symbol_summary.to_csv(paths["symbol_summary"], index=False)
    timestamp_summary.to_csv(paths["timestamp_summary"], index=False)
    cluster_summary.to_csv(paths["cluster_summary"], index=False)
    feature_summary.to_csv(paths["feature_summary"], index=False)

    multi_row_clusters = int((cluster_summary.get("rows", pd.Series(dtype=int)) > 1).sum()) if len(cluster_summary) else 0
    rows_in_multi_clusters = (
        int(cluster_summary.loc[cluster_summary["rows"] > 1, "rows"].sum()) if len(cluster_summary) and "rows" in cluster_summary else 0
    )
    concentration = {
        "unique_symbols": int(oracle["__symbol__"].nunique()) if len(oracle) else 0,
        "top_symbol_share": float(oracle["__symbol__"].value_counts(normalize=True).iloc[0]) if len(oracle) else 0.0,
        "unique_timestamps": int(oracle["__ts__"].nunique()) if len(oracle) else 0,
        "top_timestamp_share": float(oracle["__ts__"].astype(str).value_counts(normalize=True).iloc[0]) if len(oracle) else 0.0,
        "multi_row_clusters": multi_row_clusters,
        "rows_in_multi_row_clusters": rows_in_multi_clusters,
    }
    symbol_age = {
        "label_age_median_days": _safe_quantile(oracle.get("label_age_days"), 0.50),
        "label_age_p10_days": _safe_quantile(oracle.get("label_age_days"), 0.10),
        "label_age_p90_days": _safe_quantile(oracle.get("label_age_days"), 0.90),
        "feature_age_median_days": _safe_quantile(oracle.get("feature_age_days"), 0.50),
        "feature_age_p10_days": _safe_quantile(oracle.get("feature_age_days"), 0.10),
        "feature_age_p90_days": _safe_quantile(oracle.get("feature_age_days"), 0.90),
        "rows_label_age_le_7d": int((_safe_numeric(oracle.get("label_age_days")) <= 7.0).sum()),
        "rows_label_age_le_14d": int((_safe_numeric(oracle.get("label_age_days")) <= 14.0).sum()),
        "rows_feature_age_le_7d": int((_safe_numeric(oracle.get("feature_age_days")) <= 7.0).sum()),
        "rows_feature_age_le_14d": int((_safe_numeric(oracle.get("feature_age_days")) <= 14.0).sum()),
    }

    manifest = {
        "labels_path": str(labels_path),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "output_dir": str(output_dir),
        "month": str(month),
        "label_arm": str(label_arm),
        "label_description": descriptions.get(label_arm, ""),
        "top_frac": float(top_frac),
        "valid_rows": int(len(valid)),
        "oracle_rows": int(len(oracle)),
        "comparison_gap_dir": comparison_gap_dir_str,
        "comparison_rows": int(comparison_mask.sum()),
        "cluster_gap_hours": float(cluster_gap_hours),
        "feature_store": feature_store_report,
        "event_confirmation_features": event_report,
        "raw_event_features": event_cols_raw,
        "diagnostic_event_features": event_cols_diag,
        "concentration": concentration,
        "symbol_age": symbol_age,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        manifest=manifest,
        group_summary=group_summary,
        weekly_summary=weekly_summary,
        symbol_summary=symbol_summary,
        cluster_summary=cluster_summary,
        feature_summary=feature_summary,
    )
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
    parser.add_argument("--month", default="2026-06")
    parser.add_argument("--label-arm", default="E9_low_mae_mfe_ratio")
    parser.add_argument("--top-frac", type=float, default=0.005)
    parser.add_argument(
        "--event-feature-store-features",
        type=_parse_csv,
        default=",".join(DEFAULT_EVENT_FEATURE_STORE_FEATURES),
    )
    parser.add_argument("--comparison-gap-dir", type=Path, default=DEFAULT_COMPARISON_GAP_DIR)
    parser.add_argument("--no-comparison-gap", action="store_true")
    parser.add_argument("--cluster-gap-hours", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_profile(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        month=str(args.month),
        label_arm=str(args.label_arm),
        top_frac=float(args.top_frac),
        event_feature_store_features=list(args.event_feature_store_features),
        comparison_gap_dir=None if bool(args.no_comparison_gap) else args.comparison_gap_dir,
        cluster_gap_hours=float(args.cluster_gap_hours),
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "output_dir": manifest["output_dir"],
                    "month": manifest["month"],
                    "label_arm": manifest["label_arm"],
                    "top_frac": manifest["top_frac"],
                    "valid_rows": manifest["valid_rows"],
                    "oracle_rows": manifest["oracle_rows"],
                    "comparison_rows": manifest["comparison_rows"],
                    "concentration": manifest["concentration"],
                    "symbol_age": manifest["symbol_age"],
                    "outputs": manifest["outputs"],
                }
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
