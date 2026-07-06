#!/usr/bin/env python3
"""Materialize an ablation-safe joined source-quality subset.

The broad source-quality materialization can contain many rows that do not have
matching realized outcomes or prediction scores. This script exports the inner
joined row universe that does have a label-ledger match, preserving source tags,
quality labels, sample weights, optional V2 source archetypes, and realized path
metrics. It is diagnostic-only and does not integrate labels into training.
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

from scripts.run_label_quality_proxy_diagnostics import _path_metrics  # noqa: E402
from scripts.run_source_quality_label_walkforward_ablation import _load_joined_frame  # noqa: E402


DEFAULT_SOURCE_DIR = Path(
    "data_perp/reports/source_tags_s10_policy_net_v17_proxy_alignment_diagnostic_july_refresh_basegateoff"
)
DEFAULT_QUALITY_LABELS = DEFAULT_SOURCE_DIR / "quality_label_candidates.parquet"
DEFAULT_LABELS_PATH = Path(
    "data_perp/artifacts/20260702_180500_single_head_monthly_walkforward_july_feature_refresh_labels_"
    "labels_s10_policy_net_recent_cov95/labels"
)
DEFAULT_V2_ARCHETYPES = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "source_archetypes_v2_july_source_refresh_basegateoff_v1/candidate_source_archetypes_v2.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "source_quality_clean_joined_subset_july_refresh_basegateoff_v1"
)

LABEL_COLS = [
    "quality_label_v0",
    "quality_label_source_rank_v1",
    "quality_label_source_wf_v1",
    "quality_label_clean_path_v2",
    "quality_label_recoverable_opportunity_v2",
    "quality_label_opportunity_capture_v3",
    "quality_label_economic_capture_v4",
]

METRIC_COLS = [
    "u_policy_net",
    "mfe_norm",
    "mae_norm",
    "barrier",
    "bars_to_mfe",
    "bars_policy",
    "is_timeout",
    "bad_mae_1r",
    "bad_mae_negative",
    "bad_mae_recovered",
    "fast_bad_mae_negative",
    "late_bad_mae_negative",
    "wide_barrier_25bps",
    "wide_barrier_35bps",
]


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _safe_mean(values: Any) -> float:
    series = _safe_numeric(values).dropna()
    return float(series.mean()) if len(series) else float("nan")


def _safe_quantile(values: Any, q: float) -> float:
    series = _safe_numeric(values).dropna()
    return float(series.quantile(q)) if len(series) else float("nan")


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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _table(frame: pd.DataFrame, cols: list[str] | None = None, limit: int | None = None) -> str:
    if frame.empty:
        return "No rows."
    view = frame.copy()
    if cols is not None:
        view = view[[col for col in cols if col in view.columns]]
    if limit is not None:
        view = view.head(limit)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
    return view.to_markdown(index=False)


def _normalise_side(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "side" not in out.columns:
        if "__side__" in out.columns:
            out["side"] = out["__side__"]
        elif "side_name" in out.columns:
            out["side"] = out["side_name"]
        else:
            return out
    raw = out["side"]
    text = raw.astype(str).str.strip().str.lower()
    numeric = pd.to_numeric(raw, errors="coerce")
    side = pd.Series(np.nan, index=out.index, dtype=np.float32)
    side.loc[text.isin({"long", "buy", "+1", "1"})] = 1.0
    side.loc[text.isin({"short", "sell", "-1"})] = -1.0
    side = side.fillna(numeric)
    out["side"] = np.where(side.fillna(1.0) < 0.0, -1, 1).astype(np.int8)
    out["side_name"] = np.where(out["side"].to_numpy(dtype=np.int8) < 0, "short", "long")
    out["__side__"] = out["side"]
    return out


def _add_path_metric_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = frame.copy()
    metrics = _path_metrics(out)
    for col in ["u_policy_net", "mfe_norm", "mae_norm", "barrier", "bars_to_mfe", "bars_policy"]:
        out[col] = _safe_numeric(metrics[col]).astype(np.float32)
    out["is_timeout"] = metrics["is_timeout"].astype(bool)
    utility = _safe_numeric(metrics["u_policy_net"])
    mae_norm = _safe_numeric(metrics["mae_norm"])
    bars_policy = _safe_numeric(metrics["bars_policy"]).fillna(24.0)
    barrier = _safe_numeric(metrics["barrier"])
    bad_mae = mae_norm.ge(1.0)
    out["bad_mae_1r"] = bad_mae.astype(bool)
    out["bad_mae_negative"] = (bad_mae & utility.lt(0.0)).astype(bool)
    out["bad_mae_recovered"] = (bad_mae & utility.gt(0.0)).astype(bool)
    out["fast_bad_mae_negative"] = (out["bad_mae_negative"] & bars_policy.le(4.0)).astype(bool)
    out["late_bad_mae_negative"] = (
        out["bad_mae_negative"] & (bars_policy.ge(16.0) | out["is_timeout"].astype(bool))
    ).astype(bool)
    out["wide_barrier_25bps"] = barrier.gt(0.025).astype(bool)
    out["wide_barrier_35bps"] = barrier.gt(0.035).astype(bool)
    out["month"] = pd.to_datetime(out["__ts__"], utc=True, errors="coerce").dt.tz_convert(None).dt.to_period("M").astype(str)
    out["week_start"] = (
        pd.to_datetime(out["__ts__"], utc=True, errors="coerce").dt.tz_convert(None).dt.to_period("W-MON").astype(str)
    )
    return out, {
        "utility_source": metrics.attrs.get("utility_source"),
        "mae_encoding": metrics.attrs.get("mae_encoding"),
    }


def _v2_merge_columns(v2: pd.DataFrame, existing_cols: set[str], key_cols: list[str]) -> list[str]:
    cols: list[str] = []
    for col in v2.columns:
        if col in key_cols or col in existing_cols:
            continue
        if (
            col.endswith("_archetype_score")
            or col.startswith("tag_")
            or col
            in {
                "primary_source_archetype_v2",
                "archetype_v2_reason_codes",
                "prior_symbol_event_density_score",
                "prior_symbol_event_density_rank",
            }
        ):
            cols.append(col)
    return cols


def _merge_v2_archetypes(frame: pd.DataFrame, v2_path: Path | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if v2_path is None or not v2_path.exists():
        return frame, {"enabled": False, "reason": "missing_or_disabled", "path": str(v2_path) if v2_path else None}
    out = frame.copy()
    v2 = pd.read_parquet(v2_path)
    if "candidate_id" in out.columns and "candidate_id" in v2.columns:
        key_cols = ["candidate_id"]
        out["candidate_id"] = out["candidate_id"].astype(str)
        v2["candidate_id"] = v2["candidate_id"].astype(str)
    else:
        out = _normalise_side(out)
        v2 = _normalise_side(v2)
        key_cols = [col for col in ["__ts__", "__symbol__", "side"] if col in out.columns and col in v2.columns]
        if "__ts__" in key_cols:
            out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True, errors="coerce")
            v2["__ts__"] = pd.to_datetime(v2["__ts__"], utc=True, errors="coerce")
        if len(key_cols) < 2:
            return out, {"enabled": False, "reason": "no_usable_join_key", "path": str(v2_path)}
    duplicate_v2_keys = int(v2.duplicated(key_cols).sum())
    v2 = v2.sort_values(key_cols, kind="mergesort").drop_duplicates(key_cols, keep="last")
    add_cols = _v2_merge_columns(v2, set(out.columns), key_cols)
    if not add_cols:
        return out, {
            "enabled": True,
            "path": str(v2_path),
            "join_key": key_cols,
            "duplicate_v2_keys": duplicate_v2_keys,
            "added_columns": 0,
            "match_rate": 1.0,
        }
    merged = out.merge(v2[key_cols + add_cols], on=key_cols, how="left", validate="many_to_one")
    marker = "primary_source_archetype_v2" if "primary_source_archetype_v2" in add_cols else add_cols[0]
    match_rate = float(merged[marker].notna().mean()) if len(merged) else 0.0
    return merged, {
        "enabled": True,
        "path": str(v2_path),
        "join_key": key_cols,
        "duplicate_v2_keys": duplicate_v2_keys,
        "added_columns": int(len(add_cols)),
        "match_rate": match_rate,
    }


def _label_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = max(len(frame), 1)
    for label_col in LABEL_COLS:
        if label_col not in frame.columns:
            rows.append({"label_col": label_col, "label_value": "missing", "rows": 0, "pct": 0.0})
            continue
        for value, count in frame[label_col].value_counts(dropna=False).sort_index().items():
            rows.append(
                {
                    "label_col": label_col,
                    "label_value": str(value),
                    "rows": int(count),
                    "pct": int(count) / total,
                }
            )
    return pd.DataFrame(rows)


def _label_distribution_by_month(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label_col in LABEL_COLS:
        if label_col not in frame.columns:
            continue
        for (month, value), group in frame.groupby(["month", label_col], dropna=False, observed=True):
            rows.append(
                {
                    "month": str(month),
                    "label_col": label_col,
                    "label_value": str(value),
                    "rows": int(len(group)),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    totals = out.groupby(["month", "label_col"], dropna=False)["rows"].transform("sum").clip(lower=1)
    out["pct_within_month_label"] = out["rows"] / totals
    return out.sort_values(["month", "label_col", "label_value"], kind="mergesort")


def _group_quality(frame: pd.DataFrame, scope: str, group_col: str) -> pd.DataFrame:
    if group_col not in frame.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    total = max(len(frame), 1)
    if pd.api.types.is_bool_dtype(frame[group_col]):
        groups = {group_col.removeprefix("tag_"): frame[group_col].fillna(False).astype(bool)}
    elif group_col.startswith("tag_"):
        groups = {group_col.removeprefix("tag_"): frame[group_col].astype(str).str.lower().isin({"true", "1", "yes"})}
    else:
        groups = {
            str(value): frame[group_col].astype(str).eq(str(value))
            for value in frame[group_col].dropna().astype(str).unique()
        }
    for value, mask in groups.items():
        group = frame.loc[mask].copy()
        if group.empty:
            continue
        symbol_counts = group["__symbol__"].astype(str).value_counts(dropna=False) if "__symbol__" in group.columns else pd.Series(dtype=int)
        rows.append(
            {
                "scope": scope,
                "bucket": value,
                "rows": int(len(group)),
                "coverage": len(group) / total,
                "mean_u": _safe_mean(group["u_policy_net"]),
                "median_u": _safe_quantile(group["u_policy_net"], 0.50),
                "p25_u": _safe_quantile(group["u_policy_net"], 0.25),
                "hit_u_rate": _safe_mean(_safe_numeric(group["u_policy_net"]).gt(0.0)),
                "bad_mae_1r_rate": _safe_mean(group["bad_mae_1r"]),
                "bad_mae_negative_rate": _safe_mean(group["bad_mae_negative"]),
                "bad_mae_recovered_rate": _safe_mean(group["bad_mae_recovered"]),
                "fast_bad_mae_negative_rate": _safe_mean(group["fast_bad_mae_negative"]),
                "timeout_rate": _safe_mean(group["is_timeout"]),
                "wide_barrier_25bps_rate": _safe_mean(group["wide_barrier_25bps"]),
                "p90_mae_norm": _safe_quantile(group["mae_norm"], 0.90),
                "top_symbol": str(symbol_counts.index[0]) if len(symbol_counts) else "",
                "top_symbol_share": float(symbol_counts.iloc[0] / len(group)) if len(symbol_counts) else float("nan"),
                "unique_symbols": int(group["__symbol__"].nunique()) if "__symbol__" in group.columns else 0,
            }
        )
    return pd.DataFrame(rows)


def _source_quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for group_col in ["primary_source_tag", "primary_source_archetype_v2"]:
        part = _group_quality(frame, group_col, group_col)
        if not part.empty:
            parts.append(part)
    for tag_col in [col for col in frame.columns if col.startswith("tag_")]:
        part = _group_quality(frame, "multi_tag", tag_col)
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(
        ["scope", "mean_u", "rows"],
        ascending=[True, False, False],
        kind="mergesort",
    )


def _month_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month, group in frame.groupby("month", dropna=False, observed=True):
        rows.append(
            {
                "month": str(month),
                "rows": int(len(group)),
                "mean_u": _safe_mean(group["u_policy_net"]),
                "median_u": _safe_quantile(group["u_policy_net"], 0.50),
                "p25_u": _safe_quantile(group["u_policy_net"], 0.25),
                "hit_u_rate": _safe_mean(_safe_numeric(group["u_policy_net"]).gt(0.0)),
                "bad_mae_1r_rate": _safe_mean(group["bad_mae_1r"]),
                "bad_mae_negative_rate": _safe_mean(group["bad_mae_negative"]),
                "bad_mae_recovered_rate": _safe_mean(group["bad_mae_recovered"]),
                "timeout_rate": _safe_mean(group["is_timeout"]),
                "wide_barrier_25bps_rate": _safe_mean(group["wide_barrier_25bps"]),
                "p90_mae_norm": _safe_quantile(group["mae_norm"], 0.90),
                "unique_symbols": int(group["__symbol__"].nunique()) if "__symbol__" in group.columns else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("month", kind="mergesort")


def _status_from_failures(failures: list[str], warnings: list[str]) -> str:
    if failures:
        return "fail"
    if warnings:
        return "warning"
    return "pass"


def materialize_clean_subset(
    *,
    quality_labels_path: Path,
    labels_path: Path,
    output_dir: Path,
    v2_archetypes_path: Path | None,
    min_join_match_vs_labels: float,
    min_rows: int,
    max_duplicate_candidate_id_rows: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    joined, join_report = _load_joined_frame(quality_labels_path=quality_labels_path, labels_path=labels_path)
    joined, metric_report = _add_path_metric_columns(joined)
    joined, v2_report = _merge_v2_archetypes(joined, v2_archetypes_path)
    duplicate_candidate_rows = (
        int(joined["candidate_id"].astype(str).duplicated().sum()) if "candidate_id" in joined.columns else 0
    )
    failures: list[str] = []
    warnings: list[str] = []
    if int(len(joined)) < int(min_rows):
        failures.append(f"joined_rows {len(joined)} < min_rows {min_rows}")
    if float(join_report.get("join_match_rate_vs_labels", 0.0)) < float(min_join_match_vs_labels):
        failures.append(
            "join_match_rate_vs_labels "
            f"{float(join_report.get('join_match_rate_vs_labels', 0.0)):.4f} < {min_join_match_vs_labels:.4f}"
        )
    if duplicate_candidate_rows > int(max_duplicate_candidate_id_rows):
        failures.append(
            f"duplicate_candidate_id_rows {duplicate_candidate_rows} > {max_duplicate_candidate_id_rows}"
        )
    if float(join_report.get("join_match_rate_vs_quality", 0.0)) < 0.50:
        warnings.append(
            "broad_quality_coverage_low "
            f"{float(join_report.get('join_match_rate_vs_quality', 0.0)):.4f}; expected for sparse label-ledger subset"
        )
    if v2_report.get("enabled") and float(v2_report.get("match_rate", 0.0)) < 0.999:
        warnings.append(f"v2_match_rate {float(v2_report.get('match_rate', 0.0)):.4f} < 0.9990")
    elif not v2_report.get("enabled"):
        warnings.append(f"v2_archetype_merge_disabled: {v2_report.get('reason')}")

    label_distribution = _label_distribution(joined)
    label_distribution_month = _label_distribution_by_month(joined)
    source_quality = _source_quality_summary(joined)
    month_summary = _month_summary(joined)
    paths = {
        "subset_parquet": output_dir / "source_quality_clean_joined_subset.parquet",
        "subset_csv": output_dir / "source_quality_clean_joined_subset.csv",
        "label_distribution": output_dir / "source_quality_clean_subset_label_distribution.csv",
        "label_distribution_by_month": output_dir / "source_quality_clean_subset_label_distribution_by_month.csv",
        "source_quality": output_dir / "source_quality_clean_subset_source_quality.csv",
        "month_summary": output_dir / "source_quality_clean_subset_month_summary.csv",
        "report": output_dir / "source_quality_clean_subset_report.md",
        "manifest": output_dir / "manifest.json",
    }
    joined.to_parquet(paths["subset_parquet"], index=False)
    joined.to_csv(paths["subset_csv"], index=False)
    label_distribution.to_csv(paths["label_distribution"], index=False)
    label_distribution_month.to_csv(paths["label_distribution_by_month"], index=False)
    source_quality.to_csv(paths["source_quality"], index=False)
    month_summary.to_csv(paths["month_summary"], index=False)
    manifest = {
        "subset_status": _status_from_failures(failures, []),
        "overall_status": _status_from_failures(failures, warnings),
        "failures": failures,
        "warnings": warnings,
        "quality_labels_path": str(quality_labels_path),
        "labels_path": str(labels_path),
        "v2_archetypes_path": str(v2_archetypes_path) if v2_archetypes_path else None,
        "rows": int(len(joined)),
        "columns": int(joined.shape[1]),
        "date_min": pd.to_datetime(joined["__ts__"], utc=True, errors="coerce").min().isoformat() if len(joined) else None,
        "date_max": pd.to_datetime(joined["__ts__"], utc=True, errors="coerce").max().isoformat() if len(joined) else None,
        "symbols": int(joined["__symbol__"].nunique()) if "__symbol__" in joined.columns else 0,
        "months": sorted(joined["month"].dropna().astype(str).unique().tolist()) if "month" in joined.columns else [],
        "duplicate_candidate_id_rows": duplicate_candidate_rows,
        "join_report": join_report,
        "metric_report": metric_report,
        "v2_merge_report": v2_report,
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    _write_report(paths["report"], manifest, label_distribution, month_summary, source_quality)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def _write_report(
    path: Path,
    manifest: dict[str, Any],
    label_distribution: pd.DataFrame,
    month_summary: pd.DataFrame,
    source_quality: pd.DataFrame,
) -> None:
    lines = [
        "# Source Quality Clean Joined Subset",
        "",
        "Scope: inner joined source-quality rows with matched label-ledger outcomes. Diagnostic-only.",
        "",
        "## Status",
        "",
        f"- Subset status: `{manifest['subset_status']}`",
        f"- Overall status: `{manifest['overall_status']}`",
        f"- Rows: `{manifest['rows']}`",
        f"- Columns: `{manifest['columns']}`",
        f"- Date range: `{manifest['date_min']}` to `{manifest['date_max']}`",
        f"- Symbols: `{manifest['symbols']}`",
        f"- Months: `{', '.join(manifest['months'])}`",
        f"- Failures: `{'; '.join(manifest['failures'] or ['none'])}`",
        f"- Warnings: `{'; '.join(manifest['warnings'] or ['none'])}`",
        "",
        "## Join Report",
        "",
        f"- Quality rows: `{manifest['join_report'].get('quality_rows')}`",
        f"- Label rows: `{manifest['join_report'].get('label_rows')}`",
        f"- Joined rows: `{manifest['join_report'].get('joined_rows')}`",
        f"- Join key: `{manifest['join_report'].get('join_key')}`",
        f"- Join mode: `{manifest['join_report'].get('join_mode')}`",
        f"- Match vs quality: `{manifest['join_report'].get('join_match_rate_vs_quality')}`",
        f"- Match vs labels: `{manifest['join_report'].get('join_match_rate_vs_labels')}`",
        "",
        "## V2 Merge",
        "",
        f"- Enabled: `{manifest['v2_merge_report'].get('enabled')}`",
        f"- Join key: `{manifest['v2_merge_report'].get('join_key')}`",
        f"- Match rate: `{manifest['v2_merge_report'].get('match_rate')}`",
        f"- Added columns: `{manifest['v2_merge_report'].get('added_columns')}`",
        "",
        "## Label Distribution",
        "",
        _table(label_distribution, limit=80),
        "",
        "## Month Summary",
        "",
        _table(month_summary, limit=24),
        "",
        "## Source Quality Summary",
        "",
        _table(source_quality, limit=80),
        "",
        "## Outputs",
        "",
    ]
    for name, output_path in manifest["outputs"].items():
        lines.append(f"- {name}: `{output_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-labels-path", type=Path, default=DEFAULT_QUALITY_LABELS)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--v2-archetypes-path", type=Path, default=DEFAULT_V2_ARCHETYPES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-join-match-vs-labels", type=float, default=0.999)
    parser.add_argument("--min-rows", type=int, default=1000)
    parser.add_argument("--max-duplicate-candidate-id-rows", type=int, default=0)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless subset_status is pass.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = materialize_clean_subset(
        quality_labels_path=args.quality_labels_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        v2_archetypes_path=args.v2_archetypes_path,
        min_join_match_vs_labels=args.min_join_match_vs_labels,
        min_rows=args.min_rows,
        max_duplicate_candidate_id_rows=args.max_duplicate_candidate_id_rows,
    )
    print(f"wrote clean joined subset: {manifest['outputs']['subset_parquet']}")
    print(f"subset_status: {manifest['subset_status']}")
    print(f"overall_status: {manifest['overall_status']}")
    if args.strict and manifest["subset_status"] != "pass":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
