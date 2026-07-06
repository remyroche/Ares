#!/usr/bin/env python3
"""Strict OOS source/label/proxy diagnostic.

This report is intentionally diagnostic-only. It restricts all proxy metrics to
rows that have all three pieces joined by timestamp/symbol:

1. materialized source/quality-label rows,
2. realized policy outcome labels,
3. policy-OOS proxy predictions.

It does not train models, tune thresholds, or integrate source tags into
production training.
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
    DEFAULT_LABELS_DIR,
    _json_safe,
    _path_metrics,
    _safe_mean,
    _safe_quantile,
    _spearman,
)
from scripts.run_source_quality_label_walkforward_ablation import (  # noqa: E402
    DEFAULT_MONTHS,
    DEFAULT_QUALITY_LABELS,
    _load_joined_frame,
    _parse_csv,
    _parse_float_csv,
)


DEFAULT_SOURCE_DIR = Path("data_perp/reports/source_tags_s10_policy_net_v17_proxy_alignment_diagnostic")
DEFAULT_PREDICTIONS = DEFAULT_SOURCE_DIR / "policy_oos_proxy_predictions_apr_may_jun.parquet"
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "strict_oos_source_label_proxy_diagnostic"
)
DEFAULT_TOP_FRACS = (0.01, 0.03, 0.05, 0.10)
DEFAULT_PROXY_COLUMNS = (
    "oof_pred",
    "oof_base_clf",
    "oof_meta_clf",
    "base_rank_pct",
    "base_model_score_pct",
    "pred_H10_pred_mean",
    "base_H10_pred_mean",
)


@dataclass(frozen=True)
class LabelSpec:
    name: str
    column: str
    description: str


LABEL_SPECS = (
    LabelSpec(
        "economic_capture_v4",
        "quality_label_economic_capture_v4",
        "Clean profitable economic captures with bounded path risk.",
    ),
    LabelSpec(
        "recoverable_opportunity_v2",
        "quality_label_recoverable_opportunity_v2",
        "Rows whose path offered recoverable opportunity even if current policy did not monetize it.",
    ),
)


@dataclass(frozen=True)
class SourceBucketSpec:
    name: str
    description: str


SOURCE_BUCKETS = (
    SourceBucketSpec("all_rows", "All matched policy-OOS rows."),
    SourceBucketSpec("dirty_excluded", "Rows excluding causal dirty_shock_avoid tag."),
    SourceBucketSpec("risk_adjusted_capture_candidate", "Causal risk-adjusted capture source tag."),
    SourceBucketSpec("compression_capture_candidate", "Causal compression capture source tag."),
    SourceBucketSpec(
        "risk_adjusted_capture_dirty_excluded",
        "Risk-adjusted capture source tag after excluding dirty_shock_avoid.",
    ),
    SourceBucketSpec(
        "compression_capture_dirty_excluded",
        "Compression capture source tag after excluding dirty_shock_avoid.",
    ),
)


@dataclass(frozen=True)
class AbstentionSpec:
    name: str
    description: str
    target_side_oracle: bool


ABSTENTIONS = (
    AbstentionSpec("none", "No abstention; select directly by proxy score.", False),
    AbstentionSpec(
        "exclude_bad_mae_wide25",
        "Target-side diagnostic removal of rows with MAE >= 1R and barrier width > 25 bps.",
        True,
    ),
    AbstentionSpec(
        "exclude_timeout_or_slow_holding",
        "Target-side diagnostic removal of timeout rows or rows exceeding prior-month holding q80.",
        True,
    ),
    AbstentionSpec(
        "exclude_bad_mae_wide25_timeout_or_slow_holding",
        "Target-side diagnostic removal of joint bad-MAE/wide rows and timeout/slow-holding rows.",
        True,
    ),
)


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _bool_series(frame: pd.DataFrame, column: str, *, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    lowered = values.astype(str).str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y"})


def _rank_top_indices(score: Any, k: int) -> np.ndarray:
    score_ser = _safe_numeric(score)
    valid = score_ser.notna().to_numpy()
    if not bool(valid.any()) or int(k) <= 0:
        return np.array([], dtype=np.int64)
    valid_idx = np.flatnonzero(valid)
    use_k = min(int(k), len(valid_idx))
    order = np.argsort(-score_ser.iloc[valid_idx].to_numpy(dtype=np.float64), kind="mergesort")
    return valid_idx[order[:use_k]].astype(np.int64, copy=False)


def _label_series(frame: pd.DataFrame, spec: LabelSpec) -> pd.Series:
    if spec.column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return _safe_numeric(frame[spec.column])


def _label_stats(frame: pd.DataFrame, spec: LabelSpec) -> dict[str, Any]:
    label = _label_series(frame, spec)
    good = label.eq(1.0)
    bad = label.eq(0.0)
    neutral = label.eq(-1.0) | label.isna()
    labeled = good | bad
    return {
        "label": spec.name,
        "label_column": spec.column,
        "rows": int(len(frame)),
        "labeled_rows": int(labeled.sum()),
        "positive_rows": int(good.sum()),
        "negative_rows": int(bad.sum()),
        "neutral_or_missing_rows": int(neutral.sum()),
        "positive_rate_labeled": float(good.sum() / labeled.sum()) if int(labeled.sum()) else float("nan"),
        "coverage_labeled": float(labeled.mean()) if len(label) else float("nan"),
    }


def _source_bucket_mask(frame: pd.DataFrame, name: str) -> pd.Series:
    dirty = _bool_series(frame, "tag_dirty_shock_avoid")
    if name == "all_rows":
        return pd.Series(True, index=frame.index, dtype=bool)
    if name == "dirty_excluded":
        return ~dirty
    if name == "risk_adjusted_capture_candidate":
        return _bool_series(frame, "tag_risk_adjusted_capture_candidate")
    if name == "compression_capture_candidate":
        return _bool_series(frame, "tag_compression_capture_candidate")
    if name == "risk_adjusted_capture_dirty_excluded":
        return _bool_series(frame, "tag_risk_adjusted_capture_candidate") & (~dirty)
    if name == "compression_capture_dirty_excluded":
        return _bool_series(frame, "tag_compression_capture_candidate") & (~dirty)
    raise ValueError(f"Unknown source bucket: {name}")


def _load_predictions(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_parquet(path).copy()
    if "timestamp" in frame.columns and "__ts__" not in frame.columns:
        frame = frame.rename(columns={"timestamp": "__ts__"})
    if "symbol" in frame.columns and "__symbol__" not in frame.columns:
        frame = frame.rename(columns={"symbol": "__symbol__"})
    if "__ts__" not in frame.columns or "__symbol__" not in frame.columns:
        raise ValueError(f"{path} must include timestamp/symbol or __ts__/__symbol__")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame["__symbol__"] = frame["__symbol__"].astype(str)
    key_cols = ["__ts__", "__symbol__"]
    duplicate_keys = int(frame.duplicated(key_cols).sum())
    if duplicate_keys:
        frame = frame.sort_values(key_cols, kind="mergesort").drop_duplicates(key_cols, keep="last")
    proxy_cols = [
        col
        for col in DEFAULT_PROXY_COLUMNS
        if col in frame.columns and pd.api.types.is_numeric_dtype(frame[col])
    ]
    keep_cols = key_cols + proxy_cols
    if "prediction_source_path" in frame.columns:
        keep_cols.append("prediction_source_path")
    return frame[keep_cols].copy(), {
        "prediction_rows": int(len(frame)),
        "prediction_duplicate_keys": duplicate_keys,
        "proxy_columns": proxy_cols,
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
    }


def _join_strict_oos(
    *,
    quality_labels_path: Path,
    labels_path: Path,
    predictions_path: Path,
    months: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    joined, join_report = _load_joined_frame(
        quality_labels_path=quality_labels_path,
        labels_path=labels_path,
    )
    predictions, prediction_report = _load_predictions(predictions_path)
    key_cols = ["__ts__", "__symbol__"]
    drop_proxy_cols = [col for col in DEFAULT_PROXY_COLUMNS if col in joined.columns]
    drop_proxy_cols += [col for col in ["mr_tf_policy_score_source", "prediction_source_path"] if col in joined.columns]
    joined = joined.drop(columns=drop_proxy_cols, errors="ignore")
    strict = joined.merge(predictions, on=key_cols, how="inner", validate="one_to_one")
    strict = strict.sort_values(key_cols, kind="mergesort").reset_index(drop=True)
    strict["period"] = strict["__ts__"].dt.to_period("M").astype(str)
    if months:
        strict = strict[strict["period"].isin(set(months))].reset_index(drop=True)

    metrics = _path_metrics(strict)
    strict["u_policy_net"] = metrics["u_policy_net"]
    strict["mae_norm"] = metrics["mae_norm"]
    strict["mfe_norm"] = metrics["mfe_norm"]
    strict["barrier"] = metrics["barrier"]
    strict["bars_policy"] = metrics["bars_policy"]
    strict["is_timeout"] = metrics["is_timeout"].astype(bool)
    strict["bad_mae_1r_flag"] = strict["mae_norm"].ge(1.0)
    strict["wide_barrier_25bps_flag"] = strict["barrier"].gt(0.025)
    strict["bad_mae_wide25_flag"] = strict["bad_mae_1r_flag"] & strict["wide_barrier_25bps_flag"]

    full_metrics = _path_metrics(joined)
    joined_period = joined["__ts__"].dt.to_period("M").astype(str)
    holding_threshold_by_month: dict[str, float] = {}
    for month in sorted(set(months or strict["period"].dropna().unique())):
        prior = full_metrics.loc[joined_period < str(month), "bars_policy"]
        holding_threshold_by_month[str(month)] = _safe_quantile(prior, 0.80)
    strict["holding_q80_prior"] = strict["period"].map(holding_threshold_by_month).astype(float)
    strict["slow_holding_prior_q80_flag"] = (
        strict["bars_policy"].gt(strict["holding_q80_prior"]) & strict["holding_q80_prior"].notna()
    )
    strict["timeout_or_slow_holding_flag"] = strict["is_timeout"] | strict["slow_holding_prior_q80_flag"]

    report = {
        "quality_label_join_report": join_report,
        "prediction_report": prediction_report,
        "strict_rows": int(len(strict)),
        "strict_match_rate_vs_predictions": (
            float(len(strict) / prediction_report["prediction_rows"])
            if prediction_report["prediction_rows"]
            else 0.0
        ),
        "strict_match_rate_vs_joined_outcomes": (
            float(len(strict) / join_report["joined_rows"]) if join_report["joined_rows"] else 0.0
        ),
        "timestamp_min": strict["__ts__"].min(),
        "timestamp_max": strict["__ts__"].max(),
        "symbols": int(strict["__symbol__"].nunique(dropna=True)),
        "months": sorted(strict["period"].dropna().unique().tolist()),
        "holding_q80_prior_by_month": holding_threshold_by_month,
    }
    return strict, metrics.loc[strict.index].copy(), report


def _abstention_mask(frame: pd.DataFrame, name: str) -> pd.Series:
    if name == "none":
        return pd.Series(True, index=frame.index, dtype=bool)
    if name == "exclude_bad_mae_wide25":
        return ~frame["bad_mae_wide25_flag"].fillna(False).astype(bool)
    if name == "exclude_timeout_or_slow_holding":
        return ~frame["timeout_or_slow_holding_flag"].fillna(False).astype(bool)
    if name == "exclude_bad_mae_wide25_timeout_or_slow_holding":
        return (
            ~frame["bad_mae_wide25_flag"].fillna(False).astype(bool)
            & ~frame["timeout_or_slow_holding_flag"].fillna(False).astype(bool)
        )
    raise ValueError(f"Unknown abstention: {name}")


def _utility_stats(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "mean_u": float("nan"),
            "median_u": float("nan"),
            "p25_u": float("nan"),
            "q10_u": float("nan"),
            "hit_u": float("nan"),
            "bad_mae_1r_rate": float("nan"),
            "bad_mae_wide25_rate": float("nan"),
            "wide_barrier_25bps_rate": float("nan"),
            "timeout_rate": float("nan"),
            "slow_holding_prior_q80_rate": float("nan"),
            "timeout_or_slow_holding_rate": float("nan"),
            "p90_mae_norm": float("nan"),
            "mean_bars_policy": float("nan"),
            "top_symbol_share": float("nan"),
            "unique_symbols": 0,
        }
    symbol_counts = frame["__symbol__"].astype(str).value_counts(dropna=False)
    return {
        "rows": int(len(frame)),
        "mean_u": _safe_mean(frame["u_policy_net"]),
        "median_u": _safe_quantile(frame["u_policy_net"], 0.50),
        "p25_u": _safe_quantile(frame["u_policy_net"], 0.25),
        "q10_u": _safe_quantile(frame["u_policy_net"], 0.10),
        "hit_u": _safe_mean(frame["u_policy_net"] > 0.0),
        "bad_mae_1r_rate": _safe_mean(frame["bad_mae_1r_flag"]),
        "bad_mae_wide25_rate": _safe_mean(frame["bad_mae_wide25_flag"]),
        "wide_barrier_25bps_rate": _safe_mean(frame["wide_barrier_25bps_flag"]),
        "timeout_rate": _safe_mean(frame["is_timeout"]),
        "slow_holding_prior_q80_rate": _safe_mean(frame["slow_holding_prior_q80_flag"]),
        "timeout_or_slow_holding_rate": _safe_mean(frame["timeout_or_slow_holding_flag"]),
        "p90_mae_norm": _safe_quantile(frame["mae_norm"], 0.90),
        "mean_bars_policy": _safe_mean(frame["bars_policy"]),
        "top_symbol_share": float(symbol_counts.iloc[0] / len(frame)) if len(symbol_counts) else float("nan"),
        "unique_symbols": int(frame["__symbol__"].nunique(dropna=True)),
    }


def _source_month_summary(frame: pd.DataFrame, source_buckets: list[SourceBucketSpec]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    periods = ["all"] + sorted(frame["period"].dropna().unique().tolist())
    for period in periods:
        period_frame = frame if period == "all" else frame[frame["period"].eq(period)]
        total = len(period_frame)
        for bucket in source_buckets:
            bucket_frame = period_frame[_source_bucket_mask(period_frame, bucket.name)]
            rows.append(
                {
                    "period": period,
                    "source_bucket": bucket.name,
                    "source_description": bucket.description,
                    "coverage": float(len(bucket_frame) / total) if total else float("nan"),
                    **_utility_stats(bucket_frame),
                }
            )
    return pd.DataFrame(rows)


def _primary_source_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if "primary_source_tag" not in frame.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for period in ["all"] + sorted(frame["period"].dropna().unique().tolist()):
        period_frame = frame if period == "all" else frame[frame["period"].eq(period)]
        total = len(period_frame)
        for tag, group in period_frame.groupby(period_frame["primary_source_tag"].astype(str), dropna=False):
            rows.append(
                {
                    "period": period,
                    "primary_source_tag": str(tag),
                    "coverage": float(len(group) / total) if total else float("nan"),
                    **_utility_stats(group),
                }
            )
    return pd.DataFrame(rows)


def _label_distribution(frame: pd.DataFrame, source_buckets: list[SourceBucketSpec], labels: list[LabelSpec]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period in ["all"] + sorted(frame["period"].dropna().unique().tolist()):
        period_frame = frame if period == "all" else frame[frame["period"].eq(period)]
        for bucket in source_buckets:
            bucket_frame = period_frame[_source_bucket_mask(period_frame, bucket.name)]
            for label in labels:
                rows.append(
                    {
                        "period": period,
                        "source_bucket": bucket.name,
                        **_label_stats(bucket_frame, label),
                    }
                )
    return pd.DataFrame(rows)


def _selected_label_metrics(scope: pd.DataFrame, selected: pd.DataFrame, label_spec: LabelSpec) -> dict[str, Any]:
    scope_label = _label_series(scope, label_spec)
    selected_label = _label_series(selected, label_spec)
    scope_good = scope_label.eq(1.0)
    selected_good = selected_label.eq(1.0)
    selected_labeled = selected_label.isin([0.0, 1.0])
    return {
        "selected_label_positive_rate_labeled": (
            float(selected_good.sum() / selected_labeled.sum()) if int(selected_labeled.sum()) else float("nan")
        ),
        "selected_label_positive_rows": int(selected_good.sum()),
        "scope_label_positive_rows": int(scope_good.sum()),
        "label_capture_at_k": (
            float(selected_good.sum() / scope_good.sum()) if int(scope_good.sum()) else float("nan")
        ),
    }


def _topk_rows(
    frame: pd.DataFrame,
    *,
    source_buckets: list[SourceBucketSpec],
    abstentions: list[AbstentionSpec],
    labels: list[LabelSpec],
    proxy_cols: list[str],
    top_fracs: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_rows: list[dict[str, Any]] = []
    periods = sorted(frame["period"].dropna().unique().tolist())
    for period in periods:
        period_frame = frame[frame["period"].eq(period)].reset_index(drop=True)
        for bucket in source_buckets:
            bucket_mask = _source_bucket_mask(period_frame, bucket.name)
            bucket_frame = period_frame.loc[bucket_mask].reset_index(drop=True)
            if bucket_frame.empty:
                continue
            bucket_baseline = _utility_stats(bucket_frame)
            for abstention in abstentions:
                eligible = bucket_frame.loc[_abstention_mask(bucket_frame, abstention.name)].reset_index(drop=True)
                for proxy_col in proxy_cols:
                    proxy_scope = _safe_numeric(bucket_frame[proxy_col])
                    proxy_eligible = _safe_numeric(eligible[proxy_col]) if not eligible.empty else pd.Series(dtype=float)
                    for label_spec in labels:
                        label_scope = _label_series(bucket_frame, label_spec)
                        label_eligible = _label_series(eligible, label_spec) if not eligible.empty else pd.Series(dtype=float)
                        for top_frac in top_fracs:
                            budget = max(1, int(math.ceil(float(top_frac) * len(bucket_frame))))
                            selected_idx = _rank_top_indices(proxy_eligible, budget)
                            selected = eligible.iloc[selected_idx].copy() if len(selected_idx) else eligible.iloc[:0].copy()
                            oracle_idx = _rank_top_indices(bucket_frame["u_policy_net"], budget)
                            oracle = bucket_frame.iloc[oracle_idx].copy() if len(oracle_idx) else bucket_frame.iloc[:0].copy()
                            stats = _utility_stats(selected)
                            oracle_mean = _safe_mean(oracle["u_policy_net"])
                            row = {
                                "period": period,
                                "source_bucket": bucket.name,
                                "abstention": abstention.name,
                                "target_side_oracle_abstention": bool(abstention.target_side_oracle),
                                "proxy_col": proxy_col,
                                "label": label_spec.name,
                                "label_column": label_spec.column,
                                "top_frac": float(top_frac),
                                "scope_rows": int(len(bucket_frame)),
                                "eligible_rows": int(len(eligible)),
                                "selection_budget_rows": int(budget),
                                "selected_rows": int(len(selected)),
                                "eligible_coverage": float(len(eligible) / len(bucket_frame)) if len(bucket_frame) else float("nan"),
                                "scope_mean_u": bucket_baseline["mean_u"],
                                "delta_mean_u_vs_scope": stats["mean_u"] - bucket_baseline["mean_u"]
                                if math.isfinite(stats["mean_u"]) and math.isfinite(bucket_baseline["mean_u"])
                                else float("nan"),
                                "proxy_ic_u_scope": _spearman(proxy_scope, bucket_frame["u_policy_net"]),
                                "proxy_ic_label_scope": _spearman(proxy_scope, label_scope),
                                "proxy_ic_u_eligible": _spearman(proxy_eligible, eligible["u_policy_net"])
                                if not eligible.empty
                                else float("nan"),
                                "proxy_ic_label_eligible": _spearman(proxy_eligible, label_eligible)
                                if not eligible.empty
                                else float("nan"),
                                "oracle_topk_mean_u": oracle_mean,
                                "proxy_oracle_mean_u_ratio": (
                                    stats["mean_u"] / oracle_mean
                                    if math.isfinite(stats["mean_u"]) and math.isfinite(oracle_mean) and abs(oracle_mean) > 1e-12
                                    else float("nan")
                                ),
                                **stats,
                                **_selected_label_metrics(bucket_frame, selected, label_spec),
                            }
                            monthly_rows.append(row)

    monthly = pd.DataFrame(monthly_rows)
    if monthly.empty:
        return monthly, monthly

    group_cols = ["source_bucket", "abstention", "proxy_col", "label", "top_frac"]
    aggregate_rows: list[dict[str, Any]] = []
    for key, group in monthly.groupby(group_cols, dropna=False, observed=True):
        source_bucket, abstention, proxy_col, label, top_frac = key
        mean_u = _safe_numeric(group["mean_u"])
        aggregate_rows.append(
            {
                "source_bucket": source_bucket,
                "abstention": abstention,
                "target_side_oracle_abstention": bool(group["target_side_oracle_abstention"].iloc[0]),
                "proxy_col": proxy_col,
                "label": label,
                "top_frac": float(top_frac),
                "months": int(group["period"].nunique()),
                "positive_months": int((mean_u > 0.0).sum()),
                "mean_u": _safe_mean(mean_u),
                "worst_month_u": _safe_quantile(mean_u, 0.0),
                "q10_u": _safe_mean(group["q10_u"]),
                "hit_u": _safe_mean(group["hit_u"]),
                "delta_mean_u_vs_scope": _safe_mean(group["delta_mean_u_vs_scope"]),
                "bad_mae_1r_rate": _safe_mean(group["bad_mae_1r_rate"]),
                "bad_mae_wide25_rate": _safe_mean(group["bad_mae_wide25_rate"]),
                "wide_barrier_25bps_rate": _safe_mean(group["wide_barrier_25bps_rate"]),
                "timeout_rate": _safe_mean(group["timeout_rate"]),
                "slow_holding_prior_q80_rate": _safe_mean(group["slow_holding_prior_q80_rate"]),
                "timeout_or_slow_holding_rate": _safe_mean(group["timeout_or_slow_holding_rate"]),
                "p90_mae_norm": _safe_mean(group["p90_mae_norm"]),
                "mean_bars_policy": _safe_mean(group["mean_bars_policy"]),
                "mean_scope_rows": _safe_mean(group["scope_rows"]),
                "mean_eligible_rows": _safe_mean(group["eligible_rows"]),
                "mean_selected_rows": _safe_mean(group["selected_rows"]),
                "min_selected_rows": int(pd.to_numeric(group["selected_rows"], errors="coerce").min()),
                "proxy_ic_u_scope": _safe_mean(group["proxy_ic_u_scope"]),
                "proxy_ic_label_scope": _safe_mean(group["proxy_ic_label_scope"]),
                "proxy_ic_u_eligible": _safe_mean(group["proxy_ic_u_eligible"]),
                "proxy_ic_label_eligible": _safe_mean(group["proxy_ic_label_eligible"]),
                "selected_label_positive_rate_labeled": _safe_mean(group["selected_label_positive_rate_labeled"]),
                "label_capture_at_k": _safe_mean(group["label_capture_at_k"]),
                "oracle_topk_mean_u": _safe_mean(group["oracle_topk_mean_u"]),
                "proxy_oracle_mean_u_ratio": _safe_mean(group["proxy_oracle_mean_u_ratio"]),
                "top_symbol_share": _safe_mean(group["top_symbol_share"]),
                "decision": _topk_decision(group),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        ["decision", "mean_u", "worst_month_u", "bad_mae_wide25_rate"],
        ascending=[True, False, False, True],
        kind="mergesort",
        na_position="last",
    )
    return monthly, aggregate


def _topk_decision(group: pd.DataFrame) -> str:
    months = int(group["period"].nunique())
    mean_u = _safe_mean(group["mean_u"])
    worst = _safe_quantile(group["mean_u"], 0.0)
    positive_months = int((_safe_numeric(group["mean_u"]) > 0.0).sum())
    bad = _safe_mean(group["bad_mae_1r_rate"])
    bad_wide = _safe_mean(group["bad_mae_wide25_rate"])
    timeout_or_slow = _safe_mean(group["timeout_or_slow_holding_rate"])
    min_rows = int(pd.to_numeric(group["selected_rows"], errors="coerce").min())
    if (
        months >= 3
        and positive_months >= 3
        and math.isfinite(mean_u)
        and mean_u > 0.0
        and math.isfinite(worst)
        and worst > 0.0
        and math.isfinite(bad)
        and bad <= 0.45
        and math.isfinite(bad_wide)
        and bad_wide <= 0.10
        and math.isfinite(timeout_or_slow)
        and timeout_or_slow <= 0.20
        and min_rows >= 5
    ):
        return "diagnostic_candidate"
    if math.isfinite(mean_u) and mean_u > 0.0 and positive_months >= 2:
        return "positive_but_incomplete"
    if math.isfinite(bad_wide) and bad_wide <= 0.10 and math.isfinite(timeout_or_slow) and timeout_or_slow <= 0.20:
        return "risk_clean_but_weak"
    return "diagnostic_only"


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
    source_summary: pd.DataFrame,
    label_distribution: pd.DataFrame,
    topk_aggregate: pd.DataFrame,
) -> Path:
    path = output_dir / "strict_oos_source_label_proxy_diagnostic_report.md"
    cols_top = [
        "decision",
        "source_bucket",
        "abstention",
        "proxy_col",
        "label",
        "top_frac",
        "months",
        "positive_months",
        "mean_u",
        "worst_month_u",
        "bad_mae_1r_rate",
        "bad_mae_wide25_rate",
        "timeout_or_slow_holding_rate",
        "proxy_ic_u_scope",
        "proxy_ic_label_scope",
        "selected_label_positive_rate_labeled",
        "label_capture_at_k",
        "proxy_oracle_mean_u_ratio",
        "mean_selected_rows",
    ]
    source_cols = [
        "period",
        "source_bucket",
        "rows",
        "coverage",
        "mean_u",
        "bad_mae_1r_rate",
        "bad_mae_wide25_rate",
        "timeout_or_slow_holding_rate",
        "top_symbol_share",
    ]
    label_cols = [
        "period",
        "source_bucket",
        "label",
        "rows",
        "labeled_rows",
        "positive_rows",
        "negative_rows",
        "positive_rate_labeled",
        "coverage_labeled",
    ]
    candidate = topk_aggregate[topk_aggregate["decision"].eq("diagnostic_candidate")]
    positive = topk_aggregate[topk_aggregate["decision"].eq("positive_but_incomplete")].sort_values(
        ["mean_u", "worst_month_u"], ascending=[False, False]
    )
    risk_clean = topk_aggregate[topk_aggregate["decision"].eq("risk_clean_but_weak")].sort_values(
        ["mean_u", "worst_month_u"], ascending=[False, False]
    )
    all_none = topk_aggregate[topk_aggregate["abstention"].eq("none")].sort_values(
        ["mean_u", "worst_month_u"], ascending=[False, False]
    )
    lines = [
        "# Strict OOS Source Label Proxy Diagnostic",
        "",
        "Diagnostic-only report. All proxy metrics are restricted to rows that match source labels, realized outcomes, and policy-OOS predictions.",
        "",
        f"Quality labels: `{manifest['quality_labels_path']}`",
        f"Labels: `{manifest['labels_path']}`",
        f"Predictions: `{manifest['predictions_path']}`",
        f"Rows: `{manifest['rows']}`",
        f"Months: `{', '.join(manifest['months'])}`",
        f"Symbols: `{manifest['symbols']}`",
        f"Proxy columns: `{', '.join(manifest['proxy_columns'])}`",
        "",
        "## Alignment",
        "",
        f"- Joined outcome rows before prediction filter: `{manifest['join_report']['quality_label_join_report']['joined_rows']}`",
        f"- Prediction rows: `{manifest['join_report']['prediction_report']['prediction_rows']}`",
        f"- Strict matched rows: `{manifest['join_report']['strict_rows']}`",
        f"- Match vs predictions: `{manifest['join_report']['strict_match_rate_vs_predictions']:.4f}`",
        f"- Match vs joined outcomes: `{manifest['join_report']['strict_match_rate_vs_joined_outcomes']:.4f}`",
        "",
        "## Diagnostic Candidates",
        "",
        table_or_empty(candidate, cols_top, 80),
        "",
        "## Positive But Incomplete",
        "",
        table_or_empty(positive, cols_top, 40),
        "",
        "## Risk Clean But Weak",
        "",
        table_or_empty(risk_clean, cols_top, 40),
        "",
        "## Best No-Abstention Proxy Rows",
        "",
        table_or_empty(all_none, cols_top, 40),
        "",
        "## Source Bucket Summary",
        "",
        table_or_empty(source_summary[source_summary["period"].eq("all")].sort_values("mean_u", ascending=False), source_cols, 80),
        "",
        "## Label Distribution",
        "",
        table_or_empty(label_distribution[label_distribution["period"].eq("all")], label_cols, 80),
        "",
        "## Notes",
        "",
        "- Abstentions with `target_side_oracle_abstention=true` use realized outcomes and are not causal deployment gates.",
        "- Treat those abstentions as an upper-bound diagnostic for whether a future risk head could help.",
        "- This report does not replace production LightGBM walk-forward training.",
        "",
        "## Outputs",
        "",
    ]
    for key, value in manifest["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def table_or_empty(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
    return _table(frame, cols, limit=limit) if not frame.empty else "No rows."


def run_report(
    *,
    quality_labels_path: Path,
    labels_path: Path,
    predictions_path: Path,
    output_dir: Path,
    months: list[str],
    top_fracs: list[float],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    strict, _metrics, join_report = _join_strict_oos(
        quality_labels_path=quality_labels_path,
        labels_path=labels_path,
        predictions_path=predictions_path,
        months=months,
    )
    proxy_cols = [
        col
        for col in DEFAULT_PROXY_COLUMNS
        if col in strict.columns and pd.api.types.is_numeric_dtype(strict[col])
    ]
    source_buckets = list(SOURCE_BUCKETS)
    labels = [spec for spec in LABEL_SPECS if spec.column in strict.columns]
    if not proxy_cols:
        raise ValueError("No numeric proxy columns found after strict OOS join.")
    if not labels:
        raise ValueError("No requested label columns found after strict OOS join.")

    universe = pd.DataFrame(
        [
            {
                "period": period,
                **_utility_stats(strict if period == "all" else strict[strict["period"].eq(period)]),
            }
            for period in ["all"] + sorted(strict["period"].dropna().unique().tolist())
        ]
    )
    source_summary = _source_month_summary(strict, source_buckets)
    primary_summary = _primary_source_summary(strict)
    label_distribution = _label_distribution(strict, source_buckets, labels)
    topk_monthly, topk_aggregate = _topk_rows(
        strict,
        source_buckets=source_buckets,
        abstentions=list(ABSTENTIONS),
        labels=labels,
        proxy_cols=proxy_cols,
        top_fracs=top_fracs,
    )

    paths = {
        "universe_summary": output_dir / "strict_oos_universe_summary.csv",
        "source_month_summary": output_dir / "strict_oos_source_month_summary.csv",
        "primary_source_summary": output_dir / "strict_oos_primary_source_summary.csv",
        "label_distribution": output_dir / "strict_oos_source_label_distribution.csv",
        "proxy_topk_monthly": output_dir / "strict_oos_proxy_topk_monthly.csv",
        "proxy_topk_aggregate": output_dir / "strict_oos_proxy_topk_aggregate.csv",
        "manifest": output_dir / "manifest.json",
    }
    universe.to_csv(paths["universe_summary"], index=False)
    source_summary.to_csv(paths["source_month_summary"], index=False)
    primary_summary.to_csv(paths["primary_source_summary"], index=False)
    label_distribution.to_csv(paths["label_distribution"], index=False)
    topk_monthly.to_csv(paths["proxy_topk_monthly"], index=False)
    topk_aggregate.to_csv(paths["proxy_topk_aggregate"], index=False)

    manifest = {
        "scope": "strict_oos_source_label_proxy_diagnostic",
        "quality_labels_path": str(quality_labels_path),
        "labels_path": str(labels_path),
        "predictions_path": str(predictions_path),
        "output_dir": str(output_dir),
        "rows": int(len(strict)),
        "timestamp_min": strict["__ts__"].min(),
        "timestamp_max": strict["__ts__"].max(),
        "months": sorted(strict["period"].dropna().unique().tolist()),
        "symbols": int(strict["__symbol__"].nunique(dropna=True)),
        "proxy_columns": proxy_cols,
        "labels": [spec.name for spec in labels],
        "source_buckets": [spec.name for spec in source_buckets],
        "abstentions": [
            {
                "name": spec.name,
                "description": spec.description,
                "target_side_oracle": bool(spec.target_side_oracle),
            }
            for spec in ABSTENTIONS
        ],
        "top_fracs": [float(v) for v in top_fracs],
        "join_report": join_report,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    report = _write_report(
        output_dir=output_dir,
        manifest=manifest,
        source_summary=source_summary,
        label_distribution=label_distribution,
        topk_aggregate=topk_aggregate,
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        quality_labels_path=args.quality_labels_path,
        labels_path=args.labels_path,
        predictions_path=args.predictions_path,
        output_dir=args.output_dir,
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        top_fracs=_parse_float_csv(args.top_fracs, DEFAULT_TOP_FRACS),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
