#!/usr/bin/env python3
"""Export a row-level ledger for one label/weight proxy candidate.

This is still pre-training: no model fitting is performed. The script rebuilds
the causal prior-month weighted feature proxy for each OOT month, exports the
selected rows, and summarizes monthly/weekly economic stability.
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

from scripts.run_label_economic_proxy_ablation import LABEL_ARMS, _label_targets
from scripts.run_label_quality_proxy_diagnostics import (
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _rank_top_indices,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _selection_metrics,
    _spearman,
)
from scripts.run_label_weighted_proxy_ablation import (
    WEIGHT_ARMS,
    _effective_sample_size,
    _weight_series,
    _weighted_proxy_score,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_proxy_candidate_ledger_s14_w12_top1_v1")


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _selected_ledger(
    *,
    valid: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    target_valid: pd.DataFrame,
    score: pd.Series,
    month: str,
    top_frac: float,
    proxy_features: list[str],
) -> pd.DataFrame:
    idx = _rank_top_indices(score, top_frac)
    selected = valid.iloc[idx].copy().reset_index(drop=True)
    selected_metrics = valid_metrics.iloc[idx].copy().reset_index(drop=True)
    selected_target = target_valid.iloc[idx].copy().reset_index(drop=True)
    selected_score = score.iloc[idx].reset_index(drop=True)
    out = selected[["__ts__", "__symbol__"]].copy()
    out["__valid_pos__"] = idx
    out["month"] = month
    out["top_frac"] = float(top_frac)
    out["week"] = pd.to_datetime(out["__ts__"]).dt.to_period("W-SUN").astype(str)
    out["score"] = selected_score.to_numpy(dtype=np.float64, copy=False)
    out["score_rank_pct_month"] = selected_score.rank(method="average", pct=True).to_numpy(dtype=np.float64)
    out["target_soft"] = selected_target["target_soft"].to_numpy(dtype=np.float64, copy=False)
    out["target_hard"] = selected_target["target_hard"].to_numpy(dtype=np.float64, copy=False)
    for col in [
        "u_policy_net",
        "ret_net",
        "return",
        "barrier",
        "mfe_norm",
        "mae_norm",
        "bars_to_mfe",
        "bars_policy",
    ]:
        out[col] = selected_metrics[col].to_numpy()
    out["is_timeout"] = selected_metrics["is_timeout"].astype(int).to_numpy()
    for feature in proxy_features:
        if feature in selected.columns:
            out[feature] = selected[feature].to_numpy()
    return out.sort_values(["__ts__", "score"], ascending=[True, False]).reset_index(drop=True)


def _period_summary(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    label_arm: str,
    weight_arm: str,
    period: str,
    top_frac: float,
) -> dict[str, Any]:
    row = _selection_metrics(
        frame=frame,
        metrics=metrics,
        target=target,
        score=score,
        arm=f"{label_arm}::{weight_arm}",
        selector="weighted_label_ic_proxy_oos",
        period=period,
        top_frac=top_frac,
    )
    baseline_mean = _safe_mean(metrics["u_policy_net"])
    baseline_hit = _safe_mean(metrics["u_policy_net"] > 0.0)
    baseline_q10 = _safe_quantile(metrics["u_policy_net"], 0.10)
    row["period_baseline_mean_u"] = baseline_mean
    row["period_baseline_hit_u"] = baseline_hit
    row["period_baseline_q10_u"] = baseline_q10
    row["delta_mean_u_vs_period"] = (
        row["mean_u"] - baseline_mean
        if math.isfinite(float(row["mean_u"])) and math.isfinite(baseline_mean)
        else float("nan")
    )
    row["delta_hit_u_vs_period"] = (
        row["hit_u"] - baseline_hit
        if math.isfinite(float(row["hit_u"])) and math.isfinite(baseline_hit)
        else float("nan")
    )
    row["delta_q10_u_vs_period"] = (
        row["q10_u"] - baseline_q10
        if math.isfinite(float(row["q10_u"])) and math.isfinite(baseline_q10)
        else float("nan")
    )
    row["label_arm"] = label_arm
    row["weight_arm"] = weight_arm
    return row


def _weekly_summary(
    *,
    valid: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    target_valid: pd.DataFrame,
    score: pd.Series,
    ledger: pd.DataFrame,
    label_arm: str,
    weight_arm: str,
    month: str,
    top_frac: float,
) -> pd.DataFrame:
    valid_weeks = valid["__ts__"].dt.to_period("W-SUN").astype(str)
    selected_by_week = {
        week: group.copy()
        for week, group in ledger.groupby("week", dropna=False, observed=True)
    }
    rows: list[dict[str, Any]] = []
    for week in sorted(valid_weeks.dropna().unique()):
        pos = np.flatnonzero(valid_weeks.eq(week).to_numpy())
        selected = selected_by_week.get(str(week), pd.DataFrame())
        if selected.empty:
            selected_metrics = valid_metrics.iloc[:0].copy()
            selected_target = target_valid.iloc[:0].copy()
            selected_score = score.iloc[:0].copy()
            selected_frame = valid.iloc[:0].copy()
        else:
            selected_idx = valid.index[valid["__ts__"].isin(pd.to_datetime(selected["__ts__"])) & valid["__symbol__"].isin(selected["__symbol__"])]
            # Duplicated timestamps/symbols are not expected, but use merge keys to be exact.
            keys = selected[["__ts__", "__symbol__"]].copy()
            keyed = valid.reset_index(names="__valid_pos__").merge(keys, on=["__ts__", "__symbol__"], how="inner")
            selected_pos = keyed["__valid_pos__"].to_numpy(dtype=np.int64)
            selected_frame = valid.iloc[selected_pos].reset_index(drop=True)
            selected_metrics = valid_metrics.iloc[selected_pos].reset_index(drop=True)
            selected_target = target_valid.iloc[selected_pos].reset_index(drop=True)
            selected_score = score.iloc[selected_pos].reset_index(drop=True)
        if len(selected_frame):
            row = _selection_metrics(
                frame=selected_frame,
                metrics=selected_metrics,
                target=selected_target,
                score=selected_score,
                arm=f"{label_arm}::{weight_arm}",
                selector="selected_rows_by_week",
                period=str(week),
                top_frac=1.0,
            )
        else:
            row = {
                "arm": f"{label_arm}::{weight_arm}",
                "selector": "selected_rows_by_week",
                "period": str(week),
                "top_frac": float(top_frac),
                "rows": int(len(pos)),
                "selected_rows": 0,
                "mean_u": float("nan"),
                "hit_u": float("nan"),
                "q10_u": float("nan"),
                "bad_mae_1r_rate": float("nan"),
                "wide_barrier_25bps_rate": float("nan"),
                "top_symbol_share": 0.0,
            }
        baseline_metrics = valid_metrics.iloc[pos]
        row["top_frac"] = float(top_frac)
        row["month"] = month
        row["week"] = str(week)
        row["label_arm"] = label_arm
        row["weight_arm"] = weight_arm
        row["period_baseline_rows"] = int(len(pos))
        row["period_baseline_mean_u"] = _safe_mean(baseline_metrics["u_policy_net"])
        row["period_baseline_hit_u"] = _safe_mean(baseline_metrics["u_policy_net"] > 0.0)
        row["period_baseline_q10_u"] = _safe_quantile(baseline_metrics["u_policy_net"], 0.10)
        row["delta_mean_u_vs_period"] = (
            row["mean_u"] - row["period_baseline_mean_u"]
            if math.isfinite(float(row.get("mean_u", float("nan"))))
            and math.isfinite(float(row["period_baseline_mean_u"]))
            else float("nan")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_summary(monthly: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    mean_u = _safe_numeric(monthly["mean_u"])
    week_mean = _safe_numeric(weekly["mean_u"])
    return pd.DataFrame(
        [
            {
                "months": int(monthly["period"].nunique()),
                "positive_months": int((mean_u > 0.0).sum()),
                "mean_u": _safe_mean(mean_u),
                "worst_month_mean_u": _safe_quantile(mean_u, 0.0),
                "weeks": int(weekly["week"].nunique()),
                "selected_weeks": int((_safe_numeric(weekly["selected_rows"]) > 0).sum()),
                "positive_selected_weeks": int((week_mean > 0.0).sum()),
                "q25_week_mean_u": _safe_quantile(week_mean, 0.25),
                "worst_week_mean_u": _safe_quantile(week_mean, 0.0),
                "hit_u": _safe_mean(monthly["hit_u"]),
                "q10_u": _safe_mean(monthly["q10_u"]),
                "bad_mae_1r_rate": _safe_mean(monthly["bad_mae_1r_rate"]),
                "wide_barrier_25bps_rate": _safe_mean(monthly["wide_barrier_25bps_rate"]),
                "mean_selected_rows_month": _safe_mean(monthly["selected_rows"]),
                "min_selected_rows_month": int(_safe_numeric(monthly["selected_rows"]).min()),
            }
        ]
    )


def _write_markdown(
    *,
    output_dir: Path,
    aggregate: pd.DataFrame,
    monthly: pd.DataFrame,
    weekly: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "label_proxy_candidate_ledger.md"

    def table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[c for c in cols if c in frame.columns]].copy()
        if limit is not None:
            view = view.head(limit)
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    lines = [
        "# Label Proxy Candidate Ledger",
        "",
        "Scope: row-level selected proxy ledger, no model training.",
        "",
        "## Aggregate",
        "",
        table(
            aggregate,
            [
                "months",
                "positive_months",
                "mean_u",
                "worst_month_mean_u",
                "weeks",
                "selected_weeks",
                "positive_selected_weeks",
                "q25_week_mean_u",
                "worst_week_mean_u",
                "hit_u",
                "q10_u",
                "bad_mae_1r_rate",
                "wide_barrier_25bps_rate",
            ],
        ),
        "",
        "## Monthly",
        "",
        table(
            monthly,
            [
                "period",
                "selected_rows",
                "mean_u",
                "period_baseline_mean_u",
                "delta_mean_u_vs_period",
                "hit_u",
                "q10_u",
                "bad_mae_1r_rate",
                "wide_barrier_25bps_rate",
                "top_symbol_share",
            ],
        ),
        "",
        "## Weekly",
        "",
        table(
            weekly.sort_values("week"),
            [
                "week",
                "month",
                "selected_rows",
                "mean_u",
                "period_baseline_mean_u",
                "delta_mean_u_vs_period",
                "hit_u",
                "q10_u",
                "bad_mae_1r_rate",
                "wide_barrier_25bps_rate",
                "top_symbol_share",
            ],
            limit=30,
        ),
        "",
        "## Outputs",
        "",
        f"- Ledger: `{manifest['outputs']['ledger']}`",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Weekly: `{manifest['outputs']['weekly']}`",
        f"- Aggregate: `{manifest['outputs']['aggregate']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_export(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    label_arm: str,
    weight_arm: str,
    top_frac: float,
) -> dict[str, Any]:
    if label_arm not in LABEL_ARMS:
        raise ValueError(f"label_arm must be one of {LABEL_ARMS}")
    if weight_arm not in WEIGHT_ARMS:
        raise ValueError(f"weight_arm must be one of {WEIGHT_ARMS}")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = _load_labels(labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_store_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    for col in feature_matrix.columns:
        frame[col] = feature_matrix[col].to_numpy(dtype=np.float32, copy=False)

    metrics = _path_metrics(frame)
    features = _feature_columns(frame)
    targets = _label_targets(frame, metrics)
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    months = sorted(month_period.dropna().unique())

    ledgers: list[pd.DataFrame] = []
    monthly_rows: list[dict[str, Any]] = []
    weekly_frames: list[pd.DataFrame] = []
    proxy_feature_records: dict[str, list[str]] = {}
    for month in months[1:]:
        train_mask = month_period < month
        valid_mask = month_period == month
        if int(train_mask.sum()) < 100 or int(valid_mask.sum()) < 50:
            continue
        train = frame.loc[train_mask].copy()
        valid = frame.loc[valid_mask].copy().reset_index(drop=True)
        train_metrics = metrics.loc[train_mask].copy()
        valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
        target_train = targets[label_arm].loc[train_mask].copy()
        target_valid = targets[label_arm].loc[valid_mask].copy().reset_index(drop=True)
        weights = _weight_series(
            frame=train,
            metrics=train_metrics,
            target=target_train,
            arm=weight_arm,
        )
        score, diag = _weighted_proxy_score(
            train,
            frame.loc[valid_mask].copy(),
            features,
            target_train["target_soft"],
            weights,
        )
        score = score.reset_index(drop=True)
        proxy_features = list(diag.get("proxy_features", []))
        proxy_feature_records[str(month)] = proxy_features
        ledger = _selected_ledger(
            valid=valid,
            valid_metrics=valid_metrics,
            target_valid=target_valid,
            score=score,
            month=str(month),
            top_frac=top_frac,
            proxy_features=proxy_features,
        )
        ledger["score_ic_u_month"] = _spearman(score, valid_metrics["u_policy_net"])
        ledger["score_ic_label_month"] = _spearman(score, target_valid["target_soft"])
        ledger["weight_effective_frac_train"] = _effective_sample_size(weights) / float(len(weights))
        ledgers.append(ledger)
        monthly_rows.append(
            _period_summary(
                frame=valid,
                metrics=valid_metrics,
                target=target_valid,
                score=score,
                label_arm=label_arm,
                weight_arm=weight_arm,
                period=str(month),
                top_frac=top_frac,
            )
        )
        weekly_frames.append(
            _weekly_summary(
                valid=valid,
                valid_metrics=valid_metrics,
                target_valid=target_valid,
                score=score,
                ledger=ledger,
                label_arm=label_arm,
                weight_arm=weight_arm,
                month=str(month),
                top_frac=top_frac,
            )
        )

    ledger_df = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    monthly = pd.DataFrame(monthly_rows)
    weekly = pd.concat(weekly_frames, ignore_index=True) if weekly_frames else pd.DataFrame()
    aggregate = _aggregate_summary(monthly, weekly) if not monthly.empty else pd.DataFrame()

    paths = {
        "ledger": output_dir / "selected_ledger.csv",
        "monthly": output_dir / "monthly_summary.csv",
        "weekly": output_dir / "weekly_summary.csv",
        "aggregate": output_dir / "aggregate_summary.csv",
        "manifest": output_dir / "manifest.json",
    }
    ledger_df.to_csv(paths["ledger"], index=False)
    monthly.to_csv(paths["monthly"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    manifest = {
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "label_arm": label_arm,
        "weight_arm": weight_arm,
        "top_frac": float(top_frac),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_count": int(len(features)),
        "proxy_features_by_month": proxy_feature_records,
        "feature_store": feature_store_report,
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_markdown(
        output_dir=output_dir,
        aggregate=aggregate,
        monthly=monthly,
        weekly=weekly,
        manifest=manifest,
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
    parser.add_argument("--label-arm", default="S14_policy_net_path_blend")
    parser.add_argument("--weight-arm", default="W12_tail_timestamp_balanced")
    parser.add_argument("--top-frac", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_export(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        label_arm=str(args.label_arm),
        weight_arm=str(args.weight_arm),
        top_frac=float(args.top_frac),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
