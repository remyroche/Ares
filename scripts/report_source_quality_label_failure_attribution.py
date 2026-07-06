#!/usr/bin/env python3
"""Attribute v17 source-quality label walk-forward failures.

Reads the diagnostic outputs from
``run_source_quality_label_walkforward_ablation.py`` and compares every
non-vanilla ablation against the matched vanilla row by period, top fraction,
and eval scope. The report answers whether positive deltas are mostly from
path-risk reduction or from rotating into different source buckets that are
still net negative.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("data_perp/reports/source_quality_label_walkforward_ablation_v1")
VANILLA_NAME = "vanilla_s10_policy_net_soft"


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _safe_mean(values: Any) -> float:
    series = _safe_numeric(values).dropna()
    return float(series.mean()) if len(series) else float("nan")


def _safe_sum(values: Any) -> float:
    series = _safe_numeric(values).dropna()
    return float(series.sum()) if len(series) else 0.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    return value


def _classify_month(row: pd.Series) -> str:
    delta = float(row.get("delta_mean_u_vs_vanilla", np.nan))
    bad_mae_delta = float(row.get("delta_bad_mae_1r_rate_vs_vanilla", np.nan))
    timeout_delta = float(row.get("delta_timeout_rate_vs_vanilla", np.nan))
    wide_delta = float(row.get("delta_wide_barrier_25bps_rate_vs_vanilla", np.nan))
    score_ic = float(row.get("score_ic_u", np.nan))
    mean_u = float(row.get("mean_u", np.nan))
    if not math.isfinite(delta):
        return "missing_delta"
    if delta <= 0.0:
        if math.isfinite(score_ic) and score_ic < 0.0:
            return "worse_selection_negative_ic"
        return "worse_or_flat"
    risk_improves = int(math.isfinite(bad_mae_delta) and bad_mae_delta < 0.0)
    risk_improves += int(math.isfinite(timeout_delta) and timeout_delta < 0.0)
    risk_improves += int(math.isfinite(wide_delta) and wide_delta < 0.0)
    risk_worsens = int(math.isfinite(bad_mae_delta) and bad_mae_delta > 0.0)
    risk_worsens += int(math.isfinite(timeout_delta) and timeout_delta > 0.0)
    risk_worsens += int(math.isfinite(wide_delta) and wide_delta > 0.0)
    if math.isfinite(mean_u) and mean_u <= 0.0 and risk_improves >= 2:
        return "less_bad_from_risk_reduction"
    if math.isfinite(mean_u) and mean_u <= 0.0 and risk_worsens >= 2:
        return "less_bad_despite_more_risk"
    if math.isfinite(mean_u) and mean_u <= 0.0:
        return "less_bad_bucket_rotation"
    if risk_improves >= 2:
        return "positive_from_risk_reduction"
    return "positive_bucket_rotation"


def _month_attribution(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = monthly[monthly["ablation"].ne(VANILLA_NAME)].copy()
    if rows.empty:
        return rows
    out = rows.copy()
    out["risk_delta_score"] = (
        -1.0 * _safe_numeric(out.get("delta_bad_mae_1r_rate_vs_vanilla"))
        -0.7 * _safe_numeric(out.get("delta_timeout_rate_vs_vanilla"))
        -0.7 * _safe_numeric(out.get("delta_wide_barrier_25bps_rate_vs_vanilla"))
    )
    out["driver"] = out.apply(_classify_month, axis=1)
    out["beats_vanilla"] = _safe_numeric(out["delta_mean_u_vs_vanilla"]) > 0.0
    out["positive_net"] = _safe_numeric(out["mean_u"]) > 0.0
    out["risk_bad_mae_improved"] = _safe_numeric(out["delta_bad_mae_1r_rate_vs_vanilla"]) < 0.0
    out["risk_timeout_improved"] = _safe_numeric(out["delta_timeout_rate_vs_vanilla"]) < 0.0
    out["risk_wide_barrier_improved"] = (
        _safe_numeric(out["delta_wide_barrier_25bps_rate_vs_vanilla"]) < 0.0
    )
    return out


def _ablation_summary(attribution: pd.DataFrame) -> pd.DataFrame:
    if attribution.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["ablation", "eval_scope", "top_frac"]
    for key, group in attribution.groupby(group_cols, dropna=False, observed=True):
        ablation, eval_scope, top_frac = key
        rows.append(
            {
                "ablation": ablation,
                "eval_scope": eval_scope,
                "top_frac": float(top_frac),
                "months": int(group["period"].nunique()),
                "beat_months": int(group["beats_vanilla"].sum()),
                "positive_net_months": int(group["positive_net"].sum()),
                "mean_delta_u": _safe_mean(group["delta_mean_u_vs_vanilla"]),
                "worst_delta_u": float(_safe_numeric(group["delta_mean_u_vs_vanilla"]).min()),
                "mean_u": _safe_mean(group["mean_u"]),
                "worst_month_u": float(_safe_numeric(group["mean_u"]).min()),
                "mean_risk_delta_score": _safe_mean(group["risk_delta_score"]),
                "bad_mae_improved_months": int(group["risk_bad_mae_improved"].sum()),
                "timeout_improved_months": int(group["risk_timeout_improved"].sum()),
                "wide_barrier_improved_months": int(group["risk_wide_barrier_improved"].sum()),
                "negative_ic_months": int((_safe_numeric(group["score_ic_u"]) < 0.0).sum()),
                "dominant_driver": str(group["driver"].mode().iloc[0]) if len(group["driver"].mode()) else "",
                "recommendation": _recommend_group(group),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["top_frac", "beat_months", "mean_delta_u"],
        ascending=[True, False, False],
        na_position="last",
    )


def _recommend_group(group: pd.DataFrame) -> str:
    months = int(group["period"].nunique())
    beat = int(group["beats_vanilla"].sum())
    pos = int(group["positive_net"].sum())
    neg_ic = int((_safe_numeric(group["score_ic_u"]) < 0.0).sum())
    mean_delta = _safe_mean(group["delta_mean_u_vs_vanilla"])
    worst_u = float(_safe_numeric(group["mean_u"]).min())
    if months < 3:
        return "incomplete_evidence"
    if beat == months and pos == months and mean_delta > 0.0 and worst_u > 0.0:
        return "promote_candidate"
    if beat >= 2 and mean_delta > 0.0 and pos == 0:
        return "rework_label_not_train_yet_less_bad_only"
    if neg_ic >= 2:
        return "feature_learnability_failure"
    if beat >= 2:
        return "diagnostic_follow_up_only"
    return "reject_or_rework"


def _bucket_shift(buckets: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    if buckets.empty:
        return pd.DataFrame()
    primary = buckets[buckets["bucket_col"].eq("primary_source_tag")].copy()
    if primary.empty:
        return pd.DataFrame()
    selected = monthly[
        [
            "ablation",
            "period",
            "top_frac",
            "eval_scope",
            "gate_column",
            "selected_rows",
            "mean_u",
            "delta_mean_u_vs_vanilla",
        ]
    ].copy()
    primary = primary.merge(
        selected,
        on=["ablation", "period", "top_frac", "eval_scope", "gate_column"],
        how="left",
        validate="many_to_one",
    )
    primary["bucket_share"] = _safe_numeric(primary["bucket_rows"]) / _safe_numeric(primary["selected_rows"]).clip(lower=1.0)
    primary["bucket_contribution_u"] = primary["bucket_share"] * _safe_numeric(primary["bucket_mean_u"])

    vanilla = primary[primary["ablation"].eq(VANILLA_NAME)].copy()
    compare = primary[primary["ablation"].ne(VANILLA_NAME)].copy()
    key_cols = ["period", "top_frac", "eval_scope", "gate_column", "bucket"]
    vanilla = vanilla[
        key_cols
        + [
            "bucket_rows",
            "bucket_share",
            "bucket_mean_u",
            "bucket_hit_u",
            "bucket_bad_mae_1r_rate",
            "bucket_timeout_rate",
            "bucket_wide_barrier_25bps_rate",
            "bucket_contribution_u",
        ]
    ].rename(
        columns={
            "bucket_rows": "vanilla_bucket_rows",
            "bucket_share": "vanilla_bucket_share",
            "bucket_mean_u": "vanilla_bucket_mean_u",
            "bucket_hit_u": "vanilla_bucket_hit_u",
            "bucket_bad_mae_1r_rate": "vanilla_bucket_bad_mae_1r_rate",
            "bucket_timeout_rate": "vanilla_bucket_timeout_rate",
            "bucket_wide_barrier_25bps_rate": "vanilla_bucket_wide_barrier_25bps_rate",
            "bucket_contribution_u": "vanilla_bucket_contribution_u",
        }
    )
    out = compare.merge(vanilla, on=key_cols, how="left", validate="many_to_one")
    for col in [
        "bucket_rows",
        "bucket_share",
        "bucket_mean_u",
        "bucket_hit_u",
        "bucket_bad_mae_1r_rate",
        "bucket_timeout_rate",
        "bucket_wide_barrier_25bps_rate",
        "bucket_contribution_u",
    ]:
        out[f"delta_{col}_vs_vanilla"] = (
            _safe_numeric(out[col]) - _safe_numeric(out[f"vanilla_{col}"])
        )
    out["abs_delta_bucket_contribution_u"] = _safe_numeric(out["delta_bucket_contribution_u_vs_vanilla"]).abs()
    return out.sort_values(
        ["period", "top_frac", "eval_scope", "ablation", "abs_delta_bucket_contribution_u"],
        ascending=[True, True, True, True, False],
        na_position="last",
    )


def _bucket_summary(bucket_shift: pd.DataFrame) -> pd.DataFrame:
    if bucket_shift.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, group in bucket_shift.groupby(["ablation", "eval_scope", "top_frac", "bucket"], dropna=False, observed=True):
        ablation, eval_scope, top_frac, bucket = key
        rows.append(
            {
                "ablation": ablation,
                "eval_scope": eval_scope,
                "top_frac": float(top_frac),
                "bucket": bucket,
                "months": int(group["period"].nunique()),
                "mean_bucket_share": _safe_mean(group["bucket_share"]),
                "mean_delta_bucket_share": _safe_mean(group["delta_bucket_share_vs_vanilla"]),
                "mean_bucket_u": _safe_mean(group["bucket_mean_u"]),
                "mean_delta_bucket_u": _safe_mean(group["delta_bucket_mean_u_vs_vanilla"]),
                "mean_delta_contribution_u": _safe_mean(group["delta_bucket_contribution_u_vs_vanilla"]),
                "mean_bad_mae": _safe_mean(group["bucket_bad_mae_1r_rate"]),
                "mean_delta_bad_mae": _safe_mean(group["delta_bucket_bad_mae_1r_rate_vs_vanilla"]),
                "mean_timeout": _safe_mean(group["bucket_timeout_rate"]),
                "mean_wide_barrier": _safe_mean(group["bucket_wide_barrier_25bps_rate"]),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["top_frac", "ablation", "mean_delta_contribution_u"],
        ascending=[True, True, False],
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


def _write_report(
    *,
    output_dir: Path,
    attribution: pd.DataFrame,
    summary: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "source_quality_label_failure_attribution.md"
    top10 = attribution[
        attribution["top_frac"].eq(0.10) & attribution["eval_scope"].eq("all_rows")
    ].sort_values(["period", "delta_mean_u_vs_vanilla"], ascending=[True, False])
    top10_buckets = bucket_summary[
        bucket_summary["top_frac"].eq(0.10) & bucket_summary["eval_scope"].eq("all_rows")
    ].sort_values(["ablation", "mean_delta_contribution_u"], ascending=[True, False])
    lines = [
        "# Source Quality Label Failure Attribution",
        "",
        "Scope: compares each v17 source-quality ablation against matched vanilla by month, top fraction, and evaluation scope.",
        "",
        "## Executive Finding",
        "",
        "- No ablation has enough evidence for production training promotion.",
        "- May/June improvements are mostly less-bad selection, not positive OOS utility.",
        "- April failure is primarily learnability failure: v17 labels have negative utility IC while vanilla remains positive in April.",
        "",
        "## Top 10% All-Rows Month Attribution",
        "",
        _table(
            top10,
            [
                "period",
                "ablation",
                "mean_u",
                "delta_mean_u_vs_vanilla",
                "hit_u",
                "score_ic_u",
                "driver",
                "delta_bad_mae_1r_rate_vs_vanilla",
                "delta_timeout_rate_vs_vanilla",
                "delta_wide_barrier_25bps_rate_vs_vanilla",
                "risk_delta_score",
            ],
            limit=80,
        ),
        "",
        "## Ablation Summary",
        "",
        _table(
            summary,
            [
                "recommendation",
                "ablation",
                "eval_scope",
                "top_frac",
                "months",
                "beat_months",
                "positive_net_months",
                "mean_delta_u",
                "worst_delta_u",
                "mean_u",
                "worst_month_u",
                "mean_risk_delta_score",
                "negative_ic_months",
                "dominant_driver",
            ],
            limit=120,
        ),
        "",
        "## Top 10% Bucket Shifts",
        "",
        _table(
            top10_buckets,
            [
                "ablation",
                "bucket",
                "months",
                "mean_bucket_share",
                "mean_delta_bucket_share",
                "mean_bucket_u",
                "mean_delta_bucket_u",
                "mean_delta_contribution_u",
                "mean_bad_mae",
                "mean_delta_bad_mae",
                "mean_timeout",
                "mean_wide_barrier",
            ],
            limit=120,
        ),
        "",
        "## Outputs",
        "",
        f"- Month attribution: `{manifest['outputs']['month_attribution']}`",
        f"- Ablation summary: `{manifest['outputs']['ablation_summary']}`",
        f"- Bucket shift: `{manifest['outputs']['bucket_shift']}`",
        f"- Bucket summary: `{manifest['outputs']['bucket_summary']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly_path = input_dir / "source_quality_label_walkforward_monthly.csv"
    bucket_path = input_dir / "source_quality_label_failure_buckets_by_source.csv"
    manifest_path = input_dir / "manifest.json"
    monthly = pd.read_csv(monthly_path)
    buckets = pd.read_csv(bucket_path)
    attribution = _month_attribution(monthly)
    summary = _ablation_summary(attribution)
    bucket_shift = _bucket_shift(buckets, monthly)
    bucket_summary = _bucket_summary(bucket_shift)

    paths = {
        "month_attribution": output_dir / "source_quality_label_month_attribution.csv",
        "ablation_summary": output_dir / "source_quality_label_ablation_attribution_summary.csv",
        "bucket_shift": output_dir / "source_quality_label_primary_source_bucket_shift.csv",
        "bucket_summary": output_dir / "source_quality_label_primary_source_bucket_summary.csv",
        "manifest": output_dir / "manifest.json",
    }
    attribution.to_csv(paths["month_attribution"], index=False)
    summary.to_csv(paths["ablation_summary"], index=False)
    bucket_shift.to_csv(paths["bucket_shift"], index=False)
    bucket_summary.to_csv(paths["bucket_summary"], index=False)
    upstream_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest = {
        "scope": "source_quality_label_failure_attribution",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "monthly_rows": int(len(monthly)),
        "bucket_rows": int(len(buckets)),
        "attribution_rows": int(len(attribution)),
        "summary_rows": int(len(summary)),
        "bucket_shift_rows": int(len(bucket_shift)),
        "upstream_outputs": upstream_manifest.get("outputs", {}),
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    report = _write_report(
        output_dir=output_dir,
        attribution=attribution,
        summary=summary,
        bucket_summary=bucket_summary,
        manifest=manifest,
    )
    manifest["outputs"]["markdown"] = str(report)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or (args.input_dir / "failure_attribution")
    manifest = run_report(args.input_dir, output_dir)
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
