#!/usr/bin/env python3
"""Strict OOS proxy missed-opportunity attribution.

This diagnostic starts from the strict OOS join used by
report_strict_oos_source_label_proxy_diagnostic.py, then compares each
prediction-time proxy top-k selection against an oracle top-k by realized
policy utility inside the same month/source bucket.

Oracle comparisons are target-side diagnostics only. They are used to answer
whether profitable OOS opportunities existed and whether the proxy ranked them,
not to define a deployable gate.
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
    DEFAULT_LABELS_DIR,
    _json_safe,
    _safe_mean,
    _safe_quantile,
    _spearman,
)
from scripts.run_source_quality_label_walkforward_ablation import (  # noqa: E402
    _parse_csv,
    _parse_float_csv,
)


DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "strict_oos_proxy_missed_opportunities"
)


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _bool_mean(values: Any) -> float:
    if isinstance(values, pd.Series):
        return float(values.fillna(False).astype(bool).mean()) if len(values) else float("nan")
    series = pd.Series(values)
    return float(series.fillna(False).astype(bool).mean()) if len(series) else float("nan")


def _label_col(name: str) -> str:
    for spec in LABEL_SPECS:
        if spec.name == name:
            return spec.column
    raise ValueError(f"Unknown label name: {name}")


def _pct_rank_high(score: pd.Series) -> pd.Series:
    values = _safe_numeric(score)
    ranks = values.rank(method="average", pct=True)
    return ranks.where(values.notna(), np.nan)


def _row_subset_stats(frame: pd.DataFrame) -> dict[str, Any]:
    stats = _utility_stats(frame)
    stats.update(
        {
            "economic_capture_rate": _bool_mean(_safe_numeric(frame.get(_label_col("economic_capture_v4"))).eq(1.0))
            if not frame.empty and _label_col("economic_capture_v4") in frame.columns
            else float("nan"),
            "recoverable_opportunity_rate": _bool_mean(
                _safe_numeric(frame.get(_label_col("recoverable_opportunity_v2"))).eq(1.0)
            )
            if not frame.empty and _label_col("recoverable_opportunity_v2") in frame.columns
            else float("nan"),
            "mean_proxy_pct_rank": _safe_mean(frame.get("proxy_pct_rank", pd.Series(dtype=float))),
        }
    )
    return stats


def _event_rows(frame: pd.DataFrame, context: dict[str, Any], event_type: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keep_cols = [
        "__strict_oos_row_id",
        "__ts__",
        "__symbol__",
        "period",
        "week_start",
        "primary_source_tag",
        "source_tag_reason_codes",
        "u_policy_net",
        "mae_norm",
        "mfe_norm",
        "barrier",
        "bars_policy",
        "is_timeout",
        "bad_mae_1r_flag",
        "bad_mae_wide25_flag",
        "timeout_or_slow_holding_flag",
        "wide_barrier_25bps_flag",
        "proxy_score",
        "proxy_pct_rank",
        "oracle_u_rank_pct",
        _label_col("economic_capture_v4"),
        _label_col("recoverable_opportunity_v2"),
    ]
    for row in frame[[col for col in keep_cols if col in frame.columns]].to_dict("records"):
        rows.append({**context, "event_type": event_type, **row})
    return rows


def _symbol_share(frame: pd.DataFrame) -> float:
    if frame.empty or "__symbol__" not in frame.columns:
        return float("nan")
    counts = frame["__symbol__"].astype(str).value_counts(dropna=False)
    return float(counts.iloc[0] / len(frame)) if len(counts) else float("nan")


def _overlap_count(left: pd.Series, right: pd.Series) -> int:
    return int(len(set(left.tolist()) & set(right.tolist())))


def _combo_rows(
    strict: pd.DataFrame,
    *,
    source_buckets: list[str],
    proxy_cols: list[str],
    top_fracs: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strict = strict.copy()
    strict["__strict_oos_row_id"] = np.arange(len(strict), dtype=np.int64)
    strict["week_start"] = pd.to_datetime(strict["__ts__"], utc=True).dt.to_period("W").dt.start_time.dt.date.astype(str)
    econ_col = _label_col("economic_capture_v4")
    rec_col = _label_col("recoverable_opportunity_v2")
    strict["economic_capture_good"] = _safe_numeric(strict[econ_col]).eq(1.0)
    strict["recoverable_opportunity_good"] = _safe_numeric(strict[rec_col]).eq(1.0)

    aggregate_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    periods = sorted(strict["period"].dropna().unique().tolist())
    for period in periods:
        period_frame = strict[strict["period"].eq(period)].copy()
        for source_bucket in source_buckets:
            bucket_frame = period_frame.loc[_source_bucket_mask(period_frame, source_bucket)].copy()
            if bucket_frame.empty:
                continue
            scope_stats = _row_subset_stats(bucket_frame)
            scope_econ = int(bucket_frame["economic_capture_good"].sum())
            scope_recoverable = int(bucket_frame["recoverable_opportunity_good"].sum())
            bucket_frame["oracle_u_rank_pct"] = _pct_rank_high(bucket_frame["u_policy_net"])
            for proxy_col in proxy_cols:
                if proxy_col not in bucket_frame.columns:
                    continue
                work = bucket_frame.copy()
                work["proxy_score"] = _safe_numeric(work[proxy_col])
                work["proxy_pct_rank"] = _pct_rank_high(work["proxy_score"])
                proxy_ic_u = _spearman(work["proxy_score"], work["u_policy_net"])
                proxy_ic_econ = _spearman(work["proxy_score"], work["economic_capture_good"].astype(float))
                proxy_ic_recoverable = _spearman(work["proxy_score"], work["recoverable_opportunity_good"].astype(float))
                for top_frac in top_fracs:
                    k = max(1, int(math.ceil(float(top_frac) * len(work))))
                    proxy_idx = _rank_top_indices(work["proxy_score"], k)
                    oracle_idx = _rank_top_indices(work["u_policy_net"], k)
                    proxy = work.iloc[proxy_idx].copy() if len(proxy_idx) else work.iloc[:0].copy()
                    oracle = work.iloc[oracle_idx].copy() if len(oracle_idx) else work.iloc[:0].copy()
                    proxy_ids = set(proxy["__strict_oos_row_id"].tolist())
                    oracle_ids = set(oracle["__strict_oos_row_id"].tolist())
                    overlap_ids = proxy_ids & oracle_ids
                    missed_oracle = oracle.loc[~oracle["__strict_oos_row_id"].isin(proxy_ids)].copy()
                    false_positive = proxy.loc[~proxy["__strict_oos_row_id"].isin(oracle_ids)].copy()
                    missed_econ = work.loc[
                        work["economic_capture_good"] & ~work["__strict_oos_row_id"].isin(proxy_ids)
                    ].copy()
                    missed_recoverable = work.loc[
                        work["recoverable_opportunity_good"] & ~work["__strict_oos_row_id"].isin(proxy_ids)
                    ].copy()

                    proxy_stats = _row_subset_stats(proxy)
                    oracle_stats = _row_subset_stats(oracle)
                    missed_stats = _row_subset_stats(missed_oracle)
                    false_stats = _row_subset_stats(false_positive)
                    context = {
                        "period": period,
                        "source_bucket": source_bucket,
                        "proxy_col": proxy_col,
                        "top_frac": float(top_frac),
                        "scope_rows": int(len(work)),
                        "selection_budget_rows": int(k),
                    }
                    aggregate_rows.append(
                        {
                            **context,
                            "proxy_ic_u": proxy_ic_u,
                            "proxy_ic_economic_capture": proxy_ic_econ,
                            "proxy_ic_recoverable_opportunity": proxy_ic_recoverable,
                            "scope_mean_u": scope_stats["mean_u"],
                            "scope_bad_mae_1r_rate": scope_stats["bad_mae_1r_rate"],
                            "scope_bad_mae_wide25_rate": scope_stats["bad_mae_wide25_rate"],
                            "scope_timeout_or_slow_holding_rate": scope_stats[
                                "timeout_or_slow_holding_rate"
                            ],
                            "scope_economic_capture_rows": scope_econ,
                            "scope_recoverable_rows": scope_recoverable,
                            "proxy_selected_rows": int(len(proxy)),
                            "proxy_mean_u": proxy_stats["mean_u"],
                            "proxy_bad_mae_1r_rate": proxy_stats["bad_mae_1r_rate"],
                            "proxy_bad_mae_wide25_rate": proxy_stats["bad_mae_wide25_rate"],
                            "proxy_timeout_or_slow_holding_rate": proxy_stats[
                                "timeout_or_slow_holding_rate"
                            ],
                            "proxy_economic_capture_rate": proxy_stats["economic_capture_rate"],
                            "proxy_recoverable_rate": proxy_stats["recoverable_opportunity_rate"],
                            "oracle_mean_u": oracle_stats["mean_u"],
                            "oracle_bad_mae_1r_rate": oracle_stats["bad_mae_1r_rate"],
                            "oracle_bad_mae_wide25_rate": oracle_stats["bad_mae_wide25_rate"],
                            "oracle_timeout_or_slow_holding_rate": oracle_stats[
                                "timeout_or_slow_holding_rate"
                            ],
                            "oracle_economic_capture_rate": oracle_stats["economic_capture_rate"],
                            "oracle_recoverable_rate": oracle_stats["recoverable_opportunity_rate"],
                            "oracle_overlap_rows": int(len(overlap_ids)),
                            "oracle_capture_at_k": float(len(overlap_ids) / len(oracle))
                            if len(oracle)
                            else float("nan"),
                            "utility_gap_oracle_minus_proxy": (
                                oracle_stats["mean_u"] - proxy_stats["mean_u"]
                                if math.isfinite(oracle_stats["mean_u"])
                                and math.isfinite(proxy_stats["mean_u"])
                                else float("nan")
                            ),
                            "proxy_oracle_ratio": (
                                proxy_stats["mean_u"] / oracle_stats["mean_u"]
                                if math.isfinite(proxy_stats["mean_u"])
                                and math.isfinite(oracle_stats["mean_u"])
                                and abs(oracle_stats["mean_u"]) > 1e-12
                                else float("nan")
                            ),
                            "economic_capture_at_k": float(proxy["economic_capture_good"].sum() / scope_econ)
                            if scope_econ
                            else float("nan"),
                            "recoverable_capture_at_k": float(
                                proxy["recoverable_opportunity_good"].sum() / scope_recoverable
                            )
                            if scope_recoverable
                            else float("nan"),
                            "missed_oracle_rows": int(len(missed_oracle)),
                            "missed_oracle_mean_u": missed_stats["mean_u"],
                            "missed_oracle_mean_proxy_pct_rank": missed_stats["mean_proxy_pct_rank"],
                            "missed_oracle_bad_mae_1r_rate": missed_stats["bad_mae_1r_rate"],
                            "missed_oracle_bad_mae_wide25_rate": missed_stats["bad_mae_wide25_rate"],
                            "missed_oracle_timeout_or_slow_holding_rate": missed_stats[
                                "timeout_or_slow_holding_rate"
                            ],
                            "false_positive_rows": int(len(false_positive)),
                            "false_positive_mean_u": false_stats["mean_u"],
                            "false_positive_bad_mae_1r_rate": false_stats["bad_mae_1r_rate"],
                            "false_positive_bad_mae_wide25_rate": false_stats["bad_mae_wide25_rate"],
                            "false_positive_timeout_or_slow_holding_rate": false_stats[
                                "timeout_or_slow_holding_rate"
                            ],
                            "missed_economic_capture_rows": int(len(missed_econ)),
                            "missed_economic_capture_mean_proxy_pct_rank": _safe_mean(
                                missed_econ["proxy_pct_rank"]
                            ),
                            "missed_recoverable_rows": int(len(missed_recoverable)),
                            "missed_recoverable_mean_proxy_pct_rank": _safe_mean(
                                missed_recoverable["proxy_pct_rank"]
                            ),
                            "proxy_top_symbol_share": _symbol_share(proxy),
                            "missed_oracle_top_symbol_share": _symbol_share(missed_oracle),
                        }
                    )
                    event_context = {
                        "period": period,
                        "source_bucket": source_bucket,
                        "proxy_col": proxy_col,
                        "top_frac": float(top_frac),
                    }
                    event_rows.extend(_event_rows(missed_oracle, event_context, "missed_oracle_topk"))
                    event_rows.extend(_event_rows(false_positive, event_context, "proxy_false_positive_vs_oracle"))

    monthly = pd.DataFrame(aggregate_rows)
    events = pd.DataFrame(event_rows)
    aggregate = _aggregate(monthly)
    return monthly, aggregate, events


def _aggregate(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    group_cols = ["source_bucket", "proxy_col", "top_frac"]
    rows: list[dict[str, Any]] = []
    for key, group in monthly.groupby(group_cols, dropna=False, observed=True):
        source_bucket, proxy_col, top_frac = key
        mean_u = _safe_numeric(group["proxy_mean_u"])
        worst_proxy_month = _safe_quantile(mean_u, 0.0)
        rows.append(
            {
                "source_bucket": source_bucket,
                "proxy_col": proxy_col,
                "top_frac": float(top_frac),
                "months": int(group["period"].nunique()),
                "proxy_positive_months": int((mean_u > 0.0).sum()),
                "scope_rows_mean": _safe_mean(group["scope_rows"]),
                "proxy_selected_rows_mean": _safe_mean(group["proxy_selected_rows"]),
                "scope_mean_u": _safe_mean(group["scope_mean_u"]),
                "proxy_mean_u": _safe_mean(group["proxy_mean_u"]),
                "proxy_worst_month_u": worst_proxy_month,
                "oracle_mean_u": _safe_mean(group["oracle_mean_u"]),
                "utility_gap_oracle_minus_proxy": _safe_mean(group["utility_gap_oracle_minus_proxy"]),
                "proxy_oracle_ratio": _safe_mean(group["proxy_oracle_ratio"]),
                "oracle_capture_at_k": _safe_mean(group["oracle_capture_at_k"]),
                "economic_capture_at_k": _safe_mean(group["economic_capture_at_k"]),
                "recoverable_capture_at_k": _safe_mean(group["recoverable_capture_at_k"]),
                "proxy_ic_u": _safe_mean(group["proxy_ic_u"]),
                "proxy_ic_economic_capture": _safe_mean(group["proxy_ic_economic_capture"]),
                "proxy_ic_recoverable_opportunity": _safe_mean(group["proxy_ic_recoverable_opportunity"]),
                "proxy_bad_mae_1r_rate": _safe_mean(group["proxy_bad_mae_1r_rate"]),
                "proxy_bad_mae_wide25_rate": _safe_mean(group["proxy_bad_mae_wide25_rate"]),
                "proxy_timeout_or_slow_holding_rate": _safe_mean(
                    group["proxy_timeout_or_slow_holding_rate"]
                ),
                "oracle_bad_mae_1r_rate": _safe_mean(group["oracle_bad_mae_1r_rate"]),
                "oracle_bad_mae_wide25_rate": _safe_mean(group["oracle_bad_mae_wide25_rate"]),
                "oracle_timeout_or_slow_holding_rate": _safe_mean(
                    group["oracle_timeout_or_slow_holding_rate"]
                ),
                "missed_oracle_mean_proxy_pct_rank": _safe_mean(
                    group["missed_oracle_mean_proxy_pct_rank"]
                ),
                "missed_economic_capture_mean_proxy_pct_rank": _safe_mean(
                    group["missed_economic_capture_mean_proxy_pct_rank"]
                ),
                "missed_recoverable_mean_proxy_pct_rank": _safe_mean(
                    group["missed_recoverable_mean_proxy_pct_rank"]
                ),
                "proxy_top_symbol_share": _safe_mean(group["proxy_top_symbol_share"]),
                "missed_oracle_top_symbol_share": _safe_mean(group["missed_oracle_top_symbol_share"]),
                "diagnosis": _diagnosis(group),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["utility_gap_oracle_minus_proxy", "oracle_mean_u", "proxy_mean_u"],
        ascending=[False, False, True],
        kind="mergesort",
    )


def _diagnosis(group: pd.DataFrame) -> str:
    proxy_mean = _safe_mean(group["proxy_mean_u"])
    worst = _safe_quantile(group["proxy_mean_u"], 0.0)
    oracle_mean = _safe_mean(group["oracle_mean_u"])
    capture = _safe_mean(group["oracle_capture_at_k"])
    gap = _safe_mean(group["utility_gap_oracle_minus_proxy"])
    bad = _safe_mean(group["proxy_bad_mae_1r_rate"])
    timeout = _safe_mean(group["proxy_timeout_or_slow_holding_rate"])
    if math.isfinite(proxy_mean) and proxy_mean > 0.0 and math.isfinite(worst) and worst > 0.0:
        return "proxy_selects_stable_positive"
    if math.isfinite(oracle_mean) and oracle_mean > 0.02 and math.isfinite(capture) and capture < 0.20:
        return "opportunities_exist_proxy_misses"
    if math.isfinite(gap) and gap > 0.03:
        return "large_oracle_proxy_gap"
    if math.isfinite(bad) and bad > 0.60:
        return "proxy_selects_bad_path_rows"
    if math.isfinite(timeout) and timeout > 0.20:
        return "proxy_selects_timeout_or_slow_rows"
    return "mixed_or_weak_signal"


def _group_events(events: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, group in events.groupby(by, dropna=False, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(by, key)}
        row.update(
            {
                "rows": int(len(group)),
                "mean_u": _safe_mean(group["u_policy_net"]),
                "bad_mae_1r_rate": _bool_mean(group["bad_mae_1r_flag"]),
                "bad_mae_wide25_rate": _bool_mean(group["bad_mae_wide25_flag"]),
                "timeout_or_slow_holding_rate": _bool_mean(group["timeout_or_slow_holding_flag"]),
                "economic_capture_rate": _bool_mean(
                    _safe_numeric(group.get(_label_col("economic_capture_v4"))).eq(1.0)
                ),
                "recoverable_opportunity_rate": _bool_mean(
                    _safe_numeric(group.get(_label_col("recoverable_opportunity_v2"))).eq(1.0)
                ),
                "mean_proxy_pct_rank": _safe_mean(group["proxy_pct_rank"]),
                "top_symbol_share": _symbol_share(group),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["rows", "mean_u"], ascending=[False, True], kind="mergesort")


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


def _write_report(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    aggregate: pd.DataFrame,
    monthly: pd.DataFrame,
    event_by_source: pd.DataFrame,
    event_by_symbol: pd.DataFrame,
) -> Path:
    path = output_dir / "strict_oos_proxy_missed_opportunities_report.md"
    cols = [
        "diagnosis",
        "source_bucket",
        "proxy_col",
        "top_frac",
        "months",
        "proxy_positive_months",
        "proxy_mean_u",
        "proxy_worst_month_u",
        "oracle_mean_u",
        "utility_gap_oracle_minus_proxy",
        "oracle_capture_at_k",
        "economic_capture_at_k",
        "recoverable_capture_at_k",
        "proxy_ic_u",
        "proxy_ic_economic_capture",
        "proxy_bad_mae_1r_rate",
        "proxy_timeout_or_slow_holding_rate",
        "missed_oracle_mean_proxy_pct_rank",
        "proxy_selected_rows_mean",
    ]
    event_cols = [
        "event_type",
        "primary_source_tag",
        "rows",
        "mean_u",
        "bad_mae_1r_rate",
        "bad_mae_wide25_rate",
        "timeout_or_slow_holding_rate",
        "economic_capture_rate",
        "recoverable_opportunity_rate",
        "mean_proxy_pct_rank",
        "top_symbol_share",
    ]
    symbol_cols = [
        "event_type",
        "__symbol__",
        "rows",
        "mean_u",
        "bad_mae_1r_rate",
        "timeout_or_slow_holding_rate",
        "economic_capture_rate",
        "mean_proxy_pct_rank",
    ]
    opportunities_missed = aggregate[aggregate["diagnosis"].eq("opportunities_exist_proxy_misses")]
    lines = [
        "# Strict OOS Proxy Missed Opportunities",
        "",
        "Diagnostic-only oracle comparison over strict policy-OOS matched rows.",
        "",
        f"Rows: `{manifest['rows']}`",
        f"Months: `{', '.join(manifest['months'])}`",
        f"Proxy columns: `{', '.join(manifest['proxy_columns'])}`",
        "",
        "## Main Diagnosis",
        "",
        _table(aggregate["diagnosis"].value_counts().rename_axis("diagnosis").reset_index(name="rows"), ["diagnosis", "rows"]),
        "",
        "## Largest Oracle Proxy Gaps",
        "",
        _table(aggregate.sort_values("utility_gap_oracle_minus_proxy", ascending=False), cols, limit=40),
        "",
        "## Opportunity Miss Buckets",
        "",
        _table(opportunities_missed.sort_values("utility_gap_oracle_minus_proxy", ascending=False), cols, limit=40),
        "",
        "## Best Proxy Utility Rows",
        "",
        _table(aggregate.sort_values(["proxy_mean_u", "proxy_worst_month_u"], ascending=[False, False]), cols, limit=40),
        "",
        "## Worst Monthly Oracle Gaps",
        "",
        _table(
            monthly.sort_values("utility_gap_oracle_minus_proxy", ascending=False),
            [
                "period",
                "source_bucket",
                "proxy_col",
                "top_frac",
                "scope_rows",
                "proxy_mean_u",
                "oracle_mean_u",
                "utility_gap_oracle_minus_proxy",
                "oracle_capture_at_k",
                "economic_capture_at_k",
                "proxy_bad_mae_1r_rate",
                "proxy_timeout_or_slow_holding_rate",
                "missed_oracle_mean_proxy_pct_rank",
            ],
            limit=60,
        ),
        "",
        "## Missed/False-Positive Events By Source",
        "",
        _table(event_by_source, event_cols, limit=80),
        "",
        "## Missed/False-Positive Events By Symbol",
        "",
        _table(event_by_symbol, symbol_cols, limit=80),
        "",
        "## Interpretation",
        "",
        "- `oracle_*` metrics use realized utility and are target-side diagnostics only.",
        "- Low `oracle_capture_at_k` means the proxy is not ranking realized top-utility rows into its top-k selection.",
        "- Low `missed_oracle_mean_proxy_pct_rank` means the missed best rows were ranked low by the proxy, not merely just outside the cutoff.",
        "- High false-positive risk rates indicate the proxy is selecting adverse-path rows instead of available opportunity rows.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in manifest["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    quality_labels_path: Path,
    labels_path: Path,
    predictions_path: Path,
    output_dir: Path,
    months: list[str],
    top_fracs: list[float],
    source_buckets: list[str] | None = None,
    proxy_cols: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    strict, _metrics, join_report = _join_strict_oos(
        quality_labels_path=quality_labels_path,
        labels_path=labels_path,
        predictions_path=predictions_path,
        months=months,
    )
    selected_source_buckets = source_buckets or [spec.name for spec in SOURCE_BUCKETS]
    selected_proxy_cols = [
        col
        for col in (proxy_cols or list(DEFAULT_PROXY_COLUMNS))
        if col in strict.columns and pd.api.types.is_numeric_dtype(strict[col])
    ]
    monthly, aggregate, events = _combo_rows(
        strict,
        source_buckets=selected_source_buckets,
        proxy_cols=selected_proxy_cols,
        top_fracs=top_fracs,
    )
    event_by_source = _group_events(events, ["event_type", "primary_source_tag"])
    event_by_symbol = _group_events(events, ["event_type", "__symbol__"])
    event_by_week = _group_events(events, ["period", "week_start", "event_type"])

    paths = {
        "monthly": output_dir / "strict_oos_proxy_oracle_gap_monthly.csv",
        "aggregate": output_dir / "strict_oos_proxy_oracle_gap_aggregate.csv",
        "event_rows": output_dir / "strict_oos_proxy_oracle_gap_event_rows.csv",
        "event_by_source": output_dir / "strict_oos_proxy_oracle_gap_by_source.csv",
        "event_by_symbol": output_dir / "strict_oos_proxy_oracle_gap_by_symbol.csv",
        "event_by_week": output_dir / "strict_oos_proxy_oracle_gap_by_week.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    events.to_csv(paths["event_rows"], index=False)
    event_by_source.to_csv(paths["event_by_source"], index=False)
    event_by_symbol.to_csv(paths["event_by_symbol"], index=False)
    event_by_week.to_csv(paths["event_by_week"], index=False)

    manifest = {
        "scope": "strict_oos_proxy_missed_opportunities",
        "quality_labels_path": str(quality_labels_path),
        "labels_path": str(labels_path),
        "predictions_path": str(predictions_path),
        "output_dir": str(output_dir),
        "rows": int(len(strict)),
        "timestamp_min": strict["__ts__"].min(),
        "timestamp_max": strict["__ts__"].max(),
        "months": sorted(strict["period"].dropna().unique().tolist()),
        "symbols": int(strict["__symbol__"].nunique(dropna=True)),
        "proxy_columns": selected_proxy_cols,
        "source_buckets": selected_source_buckets,
        "top_fracs": [float(v) for v in top_fracs],
        "join_report": join_report,
        "diagnosis_counts": aggregate["diagnosis"].value_counts().to_dict()
        if "diagnosis" in aggregate.columns
        else {},
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    report = _write_report(
        output_dir=output_dir,
        manifest=manifest,
        aggregate=aggregate,
        monthly=monthly,
        event_by_source=event_by_source,
        event_by_symbol=event_by_symbol,
    )
    manifest["outputs"]["markdown"] = str(report)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-labels-path", type=Path, default=DEFAULT_QUALITY_LABELS)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--months", type=str, default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--top-fracs", type=str, default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--source-buckets", type=str, default="")
    parser.add_argument("--proxy-cols", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_buckets = _parse_csv(args.source_buckets, tuple()) if args.source_buckets else None
    proxy_cols = _parse_csv(args.proxy_cols, tuple()) if args.proxy_cols else None
    manifest = run_report(
        quality_labels_path=args.quality_labels_path,
        labels_path=args.labels_path,
        predictions_path=args.predictions_path,
        output_dir=args.output_dir,
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        top_fracs=_parse_float_csv(args.top_fracs, DEFAULT_TOP_FRACS),
        source_buckets=source_buckets,
        proxy_cols=proxy_cols,
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
