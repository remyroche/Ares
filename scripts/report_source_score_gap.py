#!/usr/bin/env python3
"""Explain ExtraTrees-only winners versus existing-score-only selections.

This consumes the discrepancy selected-ledger and joins it back to the clean
source-quality subset plus the same feature-store columns used by the
diagnostic ExtraTrees smoke model. It reports which causal features, source
scores, source tags, and outcome profiles separate the rows found only by the
smoke selector from the rows selected only by existing OOF/proxy scores.

Diagnostic-only: no training or production policy integration is performed.
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

from scripts.report_source_score_discrepancy import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_DISCREPANCY_DIR,
    EXTRATREES_SELECTOR,
)
from scripts.report_source_score_discrepancy import _period_strings  # noqa: E402
from scripts.report_source_gated_vanilla_diagnostic import DEFAULT_JOINED_SUBSET  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    _json_safe,
    _load_feature_store_columns,
    _path_metrics,
    _read_feature_list,
    _safe_mean,
    _safe_numeric,
    _safe_quantile,
)


DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "source_score_gap_july_refresh_basegateoff_v1"
)
DEFAULT_SELECTED_LEDGER = DEFAULT_DISCREPANCY_DIR / "source_score_discrepancy_selected_ledger.csv"
DEFAULT_FEATURE_GROUPS = ("feature_store", "source_score", "existing_score", "outcome_metric")


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_float_csv(value: str | None, default: tuple[float, ...]) -> list[float]:
    if value is None or not str(value).strip():
        return list(default)
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _load_joined(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path) if path.suffix.lower() != ".csv" else pd.read_csv(path)
    required = {"candidate_id", "__ts__", "__symbol__", "__u_policy_net__", "__barrier_pct__", "__mfe_ret__", "__mae_ret__"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    frame = frame.copy()
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    if frame["candidate_id"].duplicated().any():
        raise ValueError("joined subset contains duplicate candidate_id rows")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    if frame["__ts__"].isna().any():
        raise ValueError("joined subset contains unparsable __ts__ values")
    frame["month"] = _period_strings(frame["__ts__"], "M")
    return frame.reset_index(drop=True)


def _load_ledger(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    ledger = pd.read_csv(path)
    required = {"candidate_id", "selector", "score_col", "month", "gate", "top_frac"}
    missing = sorted(required - set(ledger.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    ledger = ledger.copy()
    ledger["candidate_id"] = ledger["candidate_id"].astype(str)
    ledger["month"] = ledger["month"].astype(str)
    ledger["gate"] = ledger["gate"].astype(str)
    ledger["score_col"] = ledger["score_col"].astype(str)
    ledger["top_frac"] = pd.to_numeric(ledger["top_frac"], errors="coerce")
    return ledger


def _attach_features(
    frame: pd.DataFrame,
    *,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected,
    )
    out = frame.copy()
    for col in feature_matrix.columns:
        out[col] = feature_matrix[col].to_numpy(dtype=np.float32, copy=False)
    report["selected_features"] = selected
    report["source"] = "feature_store"
    return out, report


def _add_metric_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    metrics = _path_metrics(frame)
    out = frame.copy()
    for col in ["u_policy_net", "mfe_norm", "mae_norm", "barrier", "bars_to_mfe", "is_timeout", "ret_net"]:
        if col in metrics.columns:
            out[col] = metrics[col].to_numpy()
    out["bad_mae_1r"] = _safe_numeric(out.get("mae_norm")).ge(1.0)
    out["wide_barrier_25bps"] = _safe_numeric(out.get("barrier")).gt(0.025)
    out["positive_utility"] = _safe_numeric(out.get("u_policy_net")).gt(0.0)
    return out, {
        "utility_source": metrics.attrs.get("utility_source"),
        "mae_encoding": metrics.attrs.get("mae_encoding"),
    }


def _feature_columns(frame: pd.DataFrame, feature_store_cols: list[str]) -> dict[str, list[str]]:
    source_scores = [
        col
        for col in frame.columns
        if col.endswith("_score") or col in {"prior_recent_source_strength", "base_positive_source_score"}
    ]
    existing_scores = [
        col
        for col in [
            "oof_pred",
            "oof_meta_clf",
            "oof_base_clf",
            "pred_H10_pred_mean",
            "base_H10_pred_mean",
            "base_rank_pct",
        ]
        if col in frame.columns
    ]
    outcome_metrics = [
        col
        for col in [
            "u_policy_net",
            "ret_net",
            "mfe_norm",
            "mae_norm",
            "barrier",
            "bars_to_mfe",
            "bad_mae_1r",
            "wide_barrier_25bps",
            "positive_utility",
        ]
        if col in frame.columns
    ]
    return {
        "feature_store": [col for col in feature_store_cols if col in frame.columns],
        "source_score": sorted(set(source_scores)),
        "existing_score": existing_scores,
        "outcome_metric": outcome_metrics,
    }


def _context_keys(ledger: pd.DataFrame, top_fracs: list[float] | None) -> pd.DataFrame:
    existing = ledger[ledger["selector"].eq("existing_score")].copy()
    if top_fracs:
        rounded = {round(float(v), 8) for v in top_fracs}
        existing = existing[existing["top_frac"].round(8).isin(rounded)].copy()
    return existing[["month", "gate", "top_frac", "score_col"]].drop_duplicates().sort_values(
        ["month", "gate", "top_frac", "score_col"]
    )


def _selected_ids(ledger: pd.DataFrame, *, selector: str, month: str, gate: str, top_frac: float, score_col: str | None = None) -> set[str]:
    mask = (
        ledger["selector"].eq(selector)
        & ledger["month"].eq(str(month))
        & ledger["gate"].eq(str(gate))
        & ledger["top_frac"].round(8).eq(round(float(top_frac), 8))
    )
    if score_col is not None:
        mask = mask & ledger["score_col"].eq(str(score_col))
    return set(ledger.loc[mask, "candidate_id"].astype(str).tolist())


def _profile(values: pd.Series) -> dict[str, float]:
    numeric = _safe_numeric(values)
    return {
        "mean": _safe_mean(numeric),
        "median": _safe_quantile(numeric, 0.50),
        "p25": _safe_quantile(numeric, 0.25),
        "p75": _safe_quantile(numeric, 0.75),
    }


def _std_delta(a: pd.Series, b: pd.Series) -> float:
    av = _safe_numeric(a).dropna()
    bv = _safe_numeric(b).dropna()
    if len(av) == 0 or len(bv) == 0:
        return float("nan")
    pooled = math.sqrt((float(av.var(ddof=0)) + float(bv.var(ddof=0))) / 2.0)
    if pooled <= 1e-12:
        return float("nan")
    return (float(av.mean()) - float(bv.mean())) / pooled


def _feature_delta_rows(
    *,
    frame: pd.DataFrame,
    et_ids: set[str],
    other_ids: set[str],
    context: dict[str, Any],
    feature_groups: dict[str, list[str]],
) -> list[dict[str, Any]]:
    et = frame[frame["candidate_id"].isin(et_ids)].copy()
    other = frame[frame["candidate_id"].isin(other_ids)].copy()
    rows: list[dict[str, Any]] = []
    for group_name, columns in feature_groups.items():
        for col in columns:
            if col not in frame.columns:
                continue
            et_values = _safe_numeric(et[col])
            other_values = _safe_numeric(other[col])
            if et_values.notna().sum() == 0 or other_values.notna().sum() == 0:
                continue
            et_profile = _profile(et_values)
            other_profile = _profile(other_values)
            rows.append(
                {
                    **context,
                    "feature_group": group_name,
                    "feature": col,
                    "extratrees_only_mean": et_profile["mean"],
                    "existing_only_mean": other_profile["mean"],
                    "delta_mean": et_profile["mean"] - other_profile["mean"],
                    "std_delta": _std_delta(et_values, other_values),
                    "extratrees_only_median": et_profile["median"],
                    "existing_only_median": other_profile["median"],
                    "extratrees_only_non_null": int(et_values.notna().sum()),
                    "existing_only_non_null": int(other_values.notna().sum()),
                }
            )
    return rows


def _bucket_rows(
    *,
    frame: pd.DataFrame,
    et_ids: set[str],
    other_ids: set[str],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    et = frame[frame["candidate_id"].isin(et_ids)].copy()
    other = frame[frame["candidate_id"].isin(other_ids)].copy()
    rows: list[dict[str, Any]] = []
    bucket_cols = ["primary_source_tag", "primary_source_archetype_v2"] + [col for col in frame.columns if col.startswith("tag_")]
    for col in bucket_cols:
        if col not in frame.columns:
            continue
        if col.startswith("tag_"):
            et_rate = _safe_numeric(et[col].astype(str).str.lower().isin({"1", "true", "yes", "y"}).astype(float)).mean()
            other_rate = _safe_numeric(other[col].astype(str).str.lower().isin({"1", "true", "yes", "y"}).astype(float)).mean()
            rows.append(
                {
                    **context,
                    "bucket_col": col,
                    "bucket": col.replace("tag_", ""),
                    "extratrees_only_rate": float(et_rate) if pd.notna(et_rate) else float("nan"),
                    "existing_only_rate": float(other_rate) if pd.notna(other_rate) else float("nan"),
                    "delta_rate": (float(et_rate) - float(other_rate)) if pd.notna(et_rate) and pd.notna(other_rate) else float("nan"),
                    "extratrees_only_rows": int(len(et)),
                    "existing_only_rows": int(len(other)),
                }
            )
            continue
        et_counts = et[col].astype(str).value_counts(normalize=True, dropna=False)
        other_counts = other[col].astype(str).value_counts(normalize=True, dropna=False)
        for bucket in sorted(set(et_counts.index).union(set(other_counts.index))):
            et_rate = float(et_counts.get(bucket, 0.0))
            other_rate = float(other_counts.get(bucket, 0.0))
            rows.append(
                {
                    **context,
                    "bucket_col": col,
                    "bucket": str(bucket),
                    "extratrees_only_rate": et_rate,
                    "existing_only_rate": other_rate,
                    "delta_rate": et_rate - other_rate,
                    "extratrees_only_rows": int(len(et)),
                    "existing_only_rows": int(len(other)),
                }
            )
    return rows


def _summary_row(frame: pd.DataFrame, ids: set[str]) -> dict[str, Any]:
    rows = frame[frame["candidate_id"].isin(ids)]
    return {
        "rows": int(len(rows)),
        "mean_u": _safe_mean(rows.get("u_policy_net")),
        "hit_u": _safe_mean(_safe_numeric(rows.get("u_policy_net")) > 0.0),
        "bad_mae_1r_rate": _safe_mean(_safe_numeric(rows.get("mae_norm")) >= 1.0),
        "timeout_rate": _safe_mean(_safe_numeric(rows.get("is_timeout")).fillna(0.0)),
        "wide_barrier_25bps_rate": _safe_mean(_safe_numeric(rows.get("barrier")) > 0.025),
        "top_symbol_share": (
            float(rows["__symbol__"].value_counts(normalize=True, dropna=False).iloc[0]) if len(rows) and "__symbol__" in rows else 0.0
        ),
    }


def _aggregate_feature_deltas(feature_deltas: pd.DataFrame) -> pd.DataFrame:
    if feature_deltas.empty:
        return feature_deltas
    rows: list[dict[str, Any]] = []
    for (group, feature), g in feature_deltas.groupby(["feature_group", "feature"], observed=True):
        rows.append(
            {
                "feature_group": group,
                "feature": feature,
                "contexts": int(len(g)),
                "mean_delta": _safe_mean(g["delta_mean"]),
                "mean_abs_delta": _safe_mean(_safe_numeric(g["delta_mean"]).abs()),
                "mean_std_delta": _safe_mean(g["std_delta"]),
                "mean_abs_std_delta": _safe_mean(_safe_numeric(g["std_delta"]).abs()),
                "positive_delta_rate": _safe_mean(_safe_numeric(g["delta_mean"]) > 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_abs_std_delta", "mean_abs_delta"], ascending=[False, False])


def _aggregate_bucket_deltas(bucket_deltas: pd.DataFrame) -> pd.DataFrame:
    if bucket_deltas.empty:
        return bucket_deltas
    rows: list[dict[str, Any]] = []
    for (bucket_col, bucket), g in bucket_deltas.groupby(["bucket_col", "bucket"], observed=True):
        rows.append(
            {
                "bucket_col": bucket_col,
                "bucket": bucket,
                "contexts": int(len(g)),
                "mean_delta_rate": _safe_mean(g["delta_rate"]),
                "mean_abs_delta_rate": _safe_mean(_safe_numeric(g["delta_rate"]).abs()),
                "extratrees_only_rate": _safe_mean(g["extratrees_only_rate"]),
                "existing_only_rate": _safe_mean(g["existing_only_rate"]),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_delta_rate", ascending=False)


def _table(frame: pd.DataFrame, cols: list[str], limit: int = 15) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[col for col in cols if col in frame.columns]].head(limit).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    return view.to_markdown(index=False)


def _write_report(
    output_dir: Path,
    *,
    context_summary: pd.DataFrame,
    feature_agg: pd.DataFrame,
    bucket_agg: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "source_score_gap_report.md"
    lines = [
        "# Source Score Gap Report",
        "",
        "Scope: diagnostic feature/source comparison of ExtraTrees-only winners vs existing-score-only selections.",
        "Outcome metrics are included for explanation only; causal feature/source columns are reported separately.",
        "",
        "## Inputs",
        "",
        f"- Joined subset: `{manifest['joined_subset_path']}`",
        f"- Selected ledger: `{manifest['selected_ledger_path']}`",
        f"- Rows: `{manifest['rows']}`",
        f"- Contexts compared: `{manifest['contexts_compared']}`",
        "",
        "## Context Summary",
        "",
        _table(
            context_summary.sort_values("delta_extratrees_only_vs_existing_only_mean_u", ascending=False),
            [
                "month",
                "gate",
                "top_frac",
                "existing_score_col",
                "extratrees_only_rows",
                "existing_only_rows",
                "overlap_rows",
                "jaccard",
                "extratrees_only_mean_u",
                "existing_only_mean_u",
                "delta_extratrees_only_vs_existing_only_mean_u",
                "extratrees_only_bad_mae_1r_rate",
                "existing_only_bad_mae_1r_rate",
            ],
            limit=25,
        ),
        "",
        "## Causal Feature / Source Score Gaps",
        "",
        _table(
            feature_agg[feature_agg["feature_group"].isin(["feature_store", "source_score", "existing_score"])],
            [
                "feature_group",
                "feature",
                "contexts",
                "mean_delta",
                "mean_std_delta",
                "mean_abs_std_delta",
                "positive_delta_rate",
            ],
            limit=30,
        ),
        "",
        "## Outcome Metric Gaps",
        "",
        _table(
            feature_agg[feature_agg["feature_group"].eq("outcome_metric")],
            [
                "feature",
                "contexts",
                "mean_delta",
                "mean_std_delta",
                "mean_abs_std_delta",
                "positive_delta_rate",
            ],
            limit=20,
        ),
        "",
        "## Source Bucket Gaps",
        "",
        _table(
            bucket_agg,
            [
                "bucket_col",
                "bucket",
                "contexts",
                "extratrees_only_rate",
                "existing_only_rate",
                "mean_delta_rate",
                "mean_abs_delta_rate",
            ],
            limit=30,
        ),
        "",
        "## Interpretation",
        "",
        "- Positive mean delta means ExtraTrees-only rows have a higher value than existing-score-only rows.",
        "- `feature_store`, `source_score`, and `existing_score` groups are prediction-time diagnostic columns.",
        "- `outcome_metric` rows explain realized behavior and must not be used as causal source tags.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_report(
    *,
    joined_subset_path: Path,
    selected_ledger_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    top_fracs: list[float],
    feature_groups: list[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_joined(joined_subset_path)
    frame, metric_report = _add_metric_columns(frame)
    frame, feature_report = _attach_features(
        frame,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        max_feature_store_features=max_feature_store_features,
    )
    ledger = _load_ledger(selected_ledger_path)
    contexts = _context_keys(ledger, top_fracs)
    feature_group_map = _feature_columns(frame, feature_report.get("selected_features", []))
    feature_group_map = {key: value for key, value in feature_group_map.items() if key in set(feature_groups)}

    by_candidate = frame.set_index("candidate_id", drop=False)
    context_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    for _, row in contexts.iterrows():
        month = str(row["month"])
        gate = str(row["gate"])
        top_frac = float(row["top_frac"])
        score_col = str(row["score_col"])
        et_ids = _selected_ids(ledger, selector=EXTRATREES_SELECTOR, month=month, gate=gate, top_frac=top_frac)
        other_ids = _selected_ids(
            ledger,
            selector="existing_score",
            month=month,
            gate=gate,
            top_frac=top_frac,
            score_col=score_col,
        )
        et_ids = set(by_candidate.index.intersection(list(et_ids)))
        other_ids = set(by_candidate.index.intersection(list(other_ids)))
        if not et_ids or not other_ids:
            continue
        overlap = et_ids & other_ids
        et_only = et_ids - other_ids
        other_only = other_ids - et_ids
        if not et_only or not other_only:
            continue
        context = {
            "month": month,
            "gate": gate,
            "top_frac": top_frac,
            "existing_score_col": score_col,
        }
        et_summary = _summary_row(frame, et_only)
        other_summary = _summary_row(frame, other_only)
        union_count = len(et_ids | other_ids)
        context_rows.append(
            {
                **context,
                "extratrees_rows": int(len(et_ids)),
                "existing_rows": int(len(other_ids)),
                "extratrees_only_rows": int(len(et_only)),
                "existing_only_rows": int(len(other_only)),
                "overlap_rows": int(len(overlap)),
                "jaccard": float(len(overlap) / union_count) if union_count else 0.0,
                "extratrees_only_mean_u": et_summary["mean_u"],
                "existing_only_mean_u": other_summary["mean_u"],
                "delta_extratrees_only_vs_existing_only_mean_u": et_summary["mean_u"] - other_summary["mean_u"],
                "extratrees_only_hit_u": et_summary["hit_u"],
                "existing_only_hit_u": other_summary["hit_u"],
                "extratrees_only_bad_mae_1r_rate": et_summary["bad_mae_1r_rate"],
                "existing_only_bad_mae_1r_rate": other_summary["bad_mae_1r_rate"],
                "extratrees_only_timeout_rate": et_summary["timeout_rate"],
                "existing_only_timeout_rate": other_summary["timeout_rate"],
                "extratrees_only_wide_barrier_25bps_rate": et_summary["wide_barrier_25bps_rate"],
                "existing_only_wide_barrier_25bps_rate": other_summary["wide_barrier_25bps_rate"],
                "extratrees_only_top_symbol_share": et_summary["top_symbol_share"],
                "existing_only_top_symbol_share": other_summary["top_symbol_share"],
            }
        )
        feature_rows.extend(
            _feature_delta_rows(
                frame=frame,
                et_ids=et_only,
                other_ids=other_only,
                context=context,
                feature_groups=feature_group_map,
            )
        )
        bucket_rows.extend(_bucket_rows(frame=frame, et_ids=et_only, other_ids=other_only, context=context))

    context_summary = pd.DataFrame(context_rows)
    feature_deltas = pd.DataFrame(feature_rows)
    bucket_deltas = pd.DataFrame(bucket_rows)
    feature_agg = _aggregate_feature_deltas(feature_deltas)
    bucket_agg = _aggregate_bucket_deltas(bucket_deltas)

    paths = {
        "context_summary": output_dir / "source_score_gap_context_summary.csv",
        "feature_deltas": output_dir / "source_score_gap_feature_deltas.csv",
        "feature_delta_aggregate": output_dir / "source_score_gap_feature_delta_aggregate.csv",
        "bucket_deltas": output_dir / "source_score_gap_bucket_deltas.csv",
        "bucket_delta_aggregate": output_dir / "source_score_gap_bucket_delta_aggregate.csv",
        "manifest": output_dir / "manifest.json",
    }
    context_summary.to_csv(paths["context_summary"], index=False)
    feature_deltas.to_csv(paths["feature_deltas"], index=False)
    feature_agg.to_csv(paths["feature_delta_aggregate"], index=False)
    bucket_deltas.to_csv(paths["bucket_deltas"], index=False)
    bucket_agg.to_csv(paths["bucket_delta_aggregate"], index=False)

    manifest: dict[str, Any] = {
        "scope": "diagnostic_extratrees_only_vs_existing_only_gap",
        "joined_subset_path": str(joined_subset_path),
        "selected_ledger_path": str(selected_ledger_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "contexts_available": int(len(contexts)),
        "contexts_compared": int(len(context_summary)),
        "top_fracs": [float(v) for v in top_fracs],
        "feature_groups": feature_groups,
        "feature_report": feature_report,
        "metric_report": metric_report,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    report_path = _write_report(
        output_dir,
        context_summary=context_summary,
        feature_agg=feature_agg,
        bucket_agg=bucket_agg,
        manifest=manifest,
    )
    manifest["outputs"]["report"] = str(report_path)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joined-subset-path", type=Path, default=DEFAULT_JOINED_SUBSET)
    parser.add_argument("--selected-ledger-path", type=Path, default=DEFAULT_SELECTED_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=24)
    parser.add_argument("--top-fracs", default="0.03,0.1")
    parser.add_argument("--feature-groups", default=",".join(DEFAULT_FEATURE_GROUPS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        joined_subset_path=args.joined_subset_path,
        selected_ledger_path=args.selected_ledger_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        top_fracs=_parse_float_csv(args.top_fracs, (0.03, 0.1)),
        feature_groups=_parse_csv(args.feature_groups, DEFAULT_FEATURE_GROUPS),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
