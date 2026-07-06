#!/usr/bin/env python3
"""Compare diagnostic ExtraTrees selections with existing proxy-score selections.

The source-gated label smoke suggested some source gates were promising, while
the existing-score source-gate diagnostic did not. This report compares the
selected rows directly, month by month, to identify whether the difference is
selector overlap, source mix, concentration, or realized path quality.

This remains diagnostic-only: no production training or policy code is touched.
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

from scripts.report_source_gated_vanilla_diagnostic import (  # noqa: E402
    DEFAULT_JOINED_SUBSET,
    DEFAULT_MONTHS,
    DEFAULT_SCORE_COLS,
    DEFAULT_SOURCE_GATES,
    DEFAULT_TOP_FRACS,
    GateSpec,
    _build_gate_specs,
    _gate_mask,
    _load_frame,
    _parse_csv,
    _parse_float_csv,
)
from scripts.run_label_feature_store_model_smoke import (  # noqa: E402
    _fit_predict,
    _month_model_frame,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    _effective_n,
    _json_safe,
    _load_feature_store_columns,
    _make_targets,
    _path_metrics,
    _rank_top_indices,
    _read_feature_list,
    _safe_mean,
    _safe_numeric,
    _safe_quantile,
    _spearman,
)


DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "source_score_discrepancy_july_refresh_basegateoff_v1"
)
EXTRATREES_SELECTOR = "extratrees_s10_policy_net_soft"


def _period_strings(ts: pd.Series, freq: str) -> pd.Series:
    values = pd.to_datetime(ts, utc=True, errors="coerce").dt.tz_convert(None)
    return values.dt.to_period(freq).astype(str)


def _parse_int_csv(value: str | None, default: tuple[int, ...]) -> list[int]:
    if value is None or not str(value).strip():
        return list(default)
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _available_score_columns(frame: pd.DataFrame, requested: list[str], min_non_null: int) -> list[str]:
    cols: list[str] = []
    for col in requested:
        if col not in frame.columns:
            continue
        score = _safe_numeric(frame[col])
        if int(score.notna().sum()) >= int(min_non_null) and score.nunique(dropna=True) >= 3:
            cols.append(col)
    return cols


def _load_or_use_feature_columns(
    frame: pd.DataFrame,
    *,
    feature_dir: Path | None,
    feature_list_csv: Path | None,
    feature_cols: list[str],
    max_feature_store_features: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if feature_cols:
        missing = [col for col in feature_cols if col not in frame.columns]
        if missing:
            raise ValueError(f"Requested feature_cols missing from joined subset: {missing}")
        matrix = frame[feature_cols].copy()
        return matrix, {
            "enabled": True,
            "source": "joined_subset_columns",
            "retained_features": int(len(matrix.columns)),
            "features": feature_cols,
        }
    if feature_dir is None or feature_list_csv is None:
        raise ValueError("Either feature_cols or both feature_dir and feature_list_csv are required")
    selected = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    matrix, report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected,
    )
    if matrix.empty:
        raise ValueError(f"No feature-store columns loaded from {feature_dir}")
    report["source"] = "feature_store"
    report["selected_features"] = selected
    return matrix, report


def _score_extratrees_by_month(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    features: list[str],
    months: list[str],
    seeds: list[int],
    train_lookback_months: int | None,
    min_train_rows: int,
) -> tuple[pd.Series, pd.DataFrame]:
    targets = _make_targets(frame, metrics)
    target = targets["S10_policy_net_soft"]
    month_period = _period_strings(frame["__ts__"], "M")
    all_score = pd.Series(np.nan, index=frame.index, dtype=np.float32)
    diag_rows: list[dict[str, Any]] = []
    for month in months:
        train_mask = month_period < str(month)
        if train_lookback_months is not None and int(train_lookback_months) > 0:
            prior = sorted(month_period[train_mask].dropna().unique())
            keep = set(prior[-int(train_lookback_months) :])
            train_mask = train_mask & month_period.isin(keep)
        valid_mask = month_period == str(month)
        if int(train_mask.sum()) < int(min_train_rows) or int(valid_mask.sum()) == 0:
            diag_rows.append(
                {
                    "month": month,
                    "skipped": True,
                    "reason": "too_few_rows",
                    "train_rows": int(train_mask.sum()),
                    "valid_rows": int(valid_mask.sum()),
                    "score_ic_u": float("nan"),
                    "score_ic_target": float("nan"),
                }
            )
            continue
        x_train, x_valid = _month_model_frame(
            frame,
            train_mask=train_mask,
            valid_mask=valid_mask,
            features=features,
        )
        if "__w__" in frame.columns:
            weights = _safe_numeric(frame.loc[train_mask, "__w__"]).fillna(1.0)
        else:
            weights = pd.Series(1.0, index=frame.loc[train_mask].index, dtype=np.float32)
        pred_matrix = np.vstack(
            [
                _fit_predict(
                    x_train=x_train,
                    y_train=target.loc[train_mask, "target_soft"],
                    w_train=weights,
                    x_valid=x_valid,
                    seed=seed,
                )
                for seed in seeds
            ]
        )
        score = pd.Series(np.mean(pred_matrix, axis=0).astype(np.float32), index=frame.loc[valid_mask].index)
        all_score.loc[score.index] = score
        valid_metrics = metrics.loc[valid_mask]
        valid_target = target.loc[valid_mask]
        diag_rows.append(
            {
                "month": month,
                "skipped": False,
                "reason": "",
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
                "feature_count": int(len(features)),
                "seed_count": int(len(seeds)),
                "prediction_seed_std_mean": float(np.std(pred_matrix, axis=0).mean()) if pred_matrix.size else float("nan"),
                "score_ic_u": _spearman(score, valid_metrics["u_policy_net"]),
                "score_ic_target": _spearman(score, valid_target["target_soft"]),
                "target_soft_mean_train": _safe_mean(target.loc[train_mask, "target_soft"]),
                "target_hard_rate_train": _safe_mean(target.loc[train_mask, "target_hard"] > 0.5),
            }
        )
    return all_score, pd.DataFrame(diag_rows)


def _candidate_key(frame: pd.DataFrame) -> pd.Series:
    if "candidate_id" in frame.columns:
        return frame["candidate_id"].astype(str)
    return pd.Series(frame.index.astype(str), index=frame.index)


def _mfe_mae_ratio(metrics: pd.DataFrame) -> pd.Series:
    if metrics.empty:
        return pd.Series(dtype=float)
    return (metrics["mfe_norm"] / metrics["mae_norm"].clip(lower=0.25)).replace([np.inf, -np.inf], np.nan).clip(upper=10.0)


def _selection_profile(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    score: pd.Series,
    selector: str,
    score_col: str,
    month: str,
    gate: GateSpec,
    top_frac: float,
    period_rows: int,
) -> dict[str, Any]:
    utility = metrics["u_policy_net"] if "u_policy_net" in metrics.columns else pd.Series(dtype=float)
    ratio = _mfe_mae_ratio(metrics)
    symbols = frame.get("__symbol__", pd.Series(dtype=object))
    weeks = _period_strings(frame["__ts__"], "W-SUN") if "__ts__" in frame.columns else pd.Series(dtype=object)
    return {
        "selector": selector,
        "score_col": score_col,
        "month": month,
        "gate": gate.name,
        "top_frac": float(top_frac),
        "period_rows": int(period_rows),
        "selected_rows": int(len(frame)),
        "coverage": float(len(frame) / period_rows) if period_rows else 0.0,
        "mean_score": _safe_mean(score),
        "score_ic_u_scope": _spearman(score, utility),
        "mean_u": _safe_mean(utility),
        "median_u": _safe_quantile(utility, 0.50),
        "q10_u": _safe_quantile(utility, 0.10),
        "hit_u": _safe_mean(utility > 0.0),
        "bad_mae_1r_rate": _safe_mean(_safe_numeric(metrics.get("mae_norm")) >= 1.0),
        "p90_mae_norm": _safe_quantile(metrics.get("mae_norm"), 0.90),
        "timeout_rate": _safe_mean(_safe_numeric(metrics.get("is_timeout")).fillna(0.0)),
        "wide_barrier_25bps_rate": _safe_mean(_safe_numeric(metrics.get("barrier")) > 0.025),
        "mean_barrier": _safe_mean(metrics.get("barrier")),
        "mean_mfe_norm": _safe_mean(metrics.get("mfe_norm")),
        "mean_mae_norm": _safe_mean(metrics.get("mae_norm")),
        "mean_mfe_mae_ratio": _safe_mean(ratio),
        "clean_row_rate": _safe_mean(
            (utility > 0.0)
            & (_safe_numeric(metrics.get("mae_norm")) <= 1.0)
            & (_safe_numeric(metrics.get("barrier")) <= 0.025)
            & (_safe_numeric(metrics.get("is_timeout")).fillna(0.0) <= 0.0)
        ),
        "symbol_effective_n": _effective_n(symbols),
        "top_symbol_share": float(symbols.value_counts(normalize=True, dropna=False).iloc[0]) if len(symbols) else 0.0,
        "top_week_share": float(weeks.value_counts(normalize=True, dropna=False).iloc[0]) if len(weeks) else 0.0,
    }


def _select_top(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    score: pd.Series,
    *,
    top_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    idx = _rank_top_indices(score, top_frac)
    if len(idx) == 0:
        return frame.iloc[:0].copy(), metrics.iloc[:0].copy(), score.iloc[:0].copy()
    return frame.iloc[idx].copy(), metrics.iloc[idx].copy(), score.iloc[idx].copy()


def _source_mix_rows(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bucket_cols = ["primary_source_tag", "primary_source_archetype_v2"] + [
        col for col in frame.columns if col.startswith("tag_")
    ]
    total = int(len(frame))
    if total == 0:
        return rows
    for col in bucket_cols:
        if col not in frame.columns:
            continue
        if col.startswith("tag_"):
            mask = frame[col].astype(str).str.lower().isin({"1", "true", "yes", "y"})
            buckets = pd.Series(np.where(mask, col.replace("tag_", ""), "__not_selected__"), index=frame.index)
        else:
            buckets = frame[col].astype(str)
        for bucket, idx in buckets.groupby(buckets, dropna=False).groups.items():
            if bucket == "__not_selected__":
                continue
            local_metrics = metrics.loc[idx]
            rows.append(
                {
                    **context,
                    "bucket_col": col,
                    "bucket": str(bucket),
                    "bucket_rows": int(len(local_metrics)),
                    "bucket_share": float(len(local_metrics) / total),
                    "bucket_mean_u": _safe_mean(local_metrics["u_policy_net"]),
                    "bucket_bad_mae_1r_rate": _safe_mean(_safe_numeric(local_metrics.get("mae_norm")) >= 1.0),
                    "bucket_timeout_rate": _safe_mean(_safe_numeric(local_metrics.get("is_timeout")).fillna(0.0)),
                    "bucket_wide_barrier_25bps_rate": _safe_mean(_safe_numeric(local_metrics.get("barrier")) > 0.025),
                }
            )
    return rows


def _overlap_profile(
    *,
    et_frame: pd.DataFrame,
    et_metrics: pd.DataFrame,
    other_frame: pd.DataFrame,
    other_metrics: pd.DataFrame,
    context: dict[str, Any],
) -> dict[str, Any]:
    et_keys = set(_candidate_key(et_frame).tolist())
    other_keys = set(_candidate_key(other_frame).tolist())
    both = et_keys & other_keys
    union = et_keys | other_keys
    et_only = et_keys - other_keys
    other_only = other_keys - et_keys

    def subset(frame: pd.DataFrame, metrics: pd.DataFrame, keys: set[str]) -> pd.DataFrame:
        if not keys:
            return metrics.iloc[:0].copy()
        key_ser = _candidate_key(frame)
        return metrics.loc[key_ser.isin(keys)].copy()

    both_metrics = subset(et_frame, et_metrics, both)
    et_only_metrics = subset(et_frame, et_metrics, et_only)
    other_only_metrics = subset(other_frame, other_metrics, other_only)
    return {
        **context,
        "extratrees_rows": int(len(et_keys)),
        "other_rows": int(len(other_keys)),
        "overlap_rows": int(len(both)),
        "union_rows": int(len(union)),
        "jaccard": float(len(both) / len(union)) if union else 0.0,
        "overlap_share_extratrees": float(len(both) / len(et_keys)) if et_keys else 0.0,
        "overlap_share_other": float(len(both) / len(other_keys)) if other_keys else 0.0,
        "extratrees_only_rows": int(len(et_only)),
        "other_only_rows": int(len(other_only)),
        "overlap_mean_u": _safe_mean(both_metrics["u_policy_net"]),
        "extratrees_only_mean_u": _safe_mean(et_only_metrics["u_policy_net"]),
        "other_only_mean_u": _safe_mean(other_only_metrics["u_policy_net"]),
        "extratrees_only_bad_mae_1r_rate": _safe_mean(_safe_numeric(et_only_metrics.get("mae_norm")) >= 1.0),
        "other_only_bad_mae_1r_rate": _safe_mean(_safe_numeric(other_only_metrics.get("mae_norm")) >= 1.0),
        "extratrees_only_timeout_rate": _safe_mean(_safe_numeric(et_only_metrics.get("is_timeout")).fillna(0.0)),
        "other_only_timeout_rate": _safe_mean(_safe_numeric(other_only_metrics.get("is_timeout")).fillna(0.0)),
    }


def _evaluate_discrepancy(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    extratrees_score: pd.Series,
    existing_score_cols: list[str],
    gates: list[GateSpec],
    months: list[str],
    top_fracs: list[float],
    min_valid_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    month_key = _period_strings(frame["__ts__"], "M")
    selection_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    source_mix_rows: list[dict[str, Any]] = []
    selected_ledgers: list[pd.DataFrame] = []

    for month in months:
        month_mask = month_key.eq(str(month))
        period_rows = int(month_mask.sum())
        if period_rows < int(min_valid_rows):
            continue
        for gate in gates:
            scope_mask = month_mask & _gate_mask(frame, metrics, gate)
            scope_frame = frame.loc[scope_mask].reset_index(drop=True)
            scope_metrics = metrics.loc[scope_mask].reset_index(drop=True)
            scope_et_score = extratrees_score.loc[scope_mask].reset_index(drop=True)
            if int(scope_et_score.notna().sum()) < int(min_valid_rows):
                continue
            for top_frac in top_fracs:
                et_sel_frame, et_sel_metrics, et_sel_score = _select_top(
                    scope_frame,
                    scope_metrics,
                    scope_et_score,
                    top_frac=top_frac,
                )
                context = {
                    "selector": EXTRATREES_SELECTOR,
                    "score_col": "extratrees_s10_policy_net_soft",
                    "month": month,
                    "gate": gate.name,
                    "top_frac": float(top_frac),
                }
                selection_rows.append(
                    _selection_profile(
                        frame=et_sel_frame,
                        metrics=et_sel_metrics,
                        score=et_sel_score,
                        selector=EXTRATREES_SELECTOR,
                        score_col="extratrees_s10_policy_net_soft",
                        month=month,
                        gate=gate,
                        top_frac=top_frac,
                        period_rows=period_rows,
                    )
                )
                source_mix_rows.extend(_source_mix_rows(frame=et_sel_frame, metrics=et_sel_metrics, context=context))
                selected_ledgers.append(_ledger_frame(et_sel_frame, et_sel_metrics, et_sel_score, context=context))

                for score_col in existing_score_cols:
                    if score_col not in scope_frame.columns:
                        continue
                    other_score = _safe_numeric(scope_frame[score_col])
                    valid = other_score.notna()
                    other_scope_frame = scope_frame.loc[valid].reset_index(drop=True)
                    other_scope_metrics = scope_metrics.loc[valid].reset_index(drop=True)
                    other_score_valid = other_score.loc[valid].reset_index(drop=True)
                    if int(len(other_score_valid)) < int(min_valid_rows):
                        continue
                    other_sel_frame, other_sel_metrics, other_sel_score = _select_top(
                        other_scope_frame,
                        other_scope_metrics,
                        other_score_valid,
                        top_frac=top_frac,
                    )
                    other_context = {
                        "selector": "existing_score",
                        "score_col": score_col,
                        "month": month,
                        "gate": gate.name,
                        "top_frac": float(top_frac),
                    }
                    selection_rows.append(
                        _selection_profile(
                            frame=other_sel_frame,
                            metrics=other_sel_metrics,
                            score=other_sel_score,
                            selector="existing_score",
                            score_col=score_col,
                            month=month,
                            gate=gate,
                            top_frac=top_frac,
                            period_rows=period_rows,
                        )
                    )
                    source_mix_rows.extend(
                        _source_mix_rows(frame=other_sel_frame, metrics=other_sel_metrics, context=other_context)
                    )
                    selected_ledgers.append(
                        _ledger_frame(other_sel_frame, other_sel_metrics, other_sel_score, context=other_context)
                    )
                    overlap_rows.append(
                        _overlap_profile(
                            et_frame=et_sel_frame,
                            et_metrics=et_sel_metrics,
                            other_frame=other_sel_frame,
                            other_metrics=other_sel_metrics,
                            context={
                                "month": month,
                                "gate": gate.name,
                                "top_frac": float(top_frac),
                                "other_score_col": score_col,
                            },
                        )
                    )

    return (
        pd.DataFrame(selection_rows),
        pd.DataFrame(overlap_rows),
        pd.DataFrame(source_mix_rows),
        pd.concat(selected_ledgers, ignore_index=True) if selected_ledgers else pd.DataFrame(),
    )


def _ledger_frame(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    score: pd.Series,
    *,
    context: dict[str, Any],
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    cols = [
        "candidate_id",
        "__ts__",
        "__symbol__",
        "side",
        "primary_source_tag",
        "primary_source_archetype_v2",
    ]
    out = pd.DataFrame({col: frame[col].to_numpy() for col in cols if col in frame.columns})
    if "candidate_id" not in out.columns:
        out["candidate_id"] = _candidate_key(frame).to_numpy()
    out["selector"] = context["selector"]
    out["score_col"] = context["score_col"]
    out["month"] = context["month"]
    out["gate"] = context["gate"]
    out["top_frac"] = context["top_frac"]
    out["score"] = score.to_numpy(dtype=np.float32, copy=False)
    for col in ["u_policy_net", "mae_norm", "mfe_norm", "barrier", "is_timeout"]:
        if col in metrics.columns:
            out[col] = metrics[col].to_numpy()
    return out


def _aggregate_selection(selection: pd.DataFrame) -> pd.DataFrame:
    if selection.empty:
        return selection
    rows: list[dict[str, Any]] = []
    group_cols = ["selector", "score_col", "gate", "top_frac"]
    for key, group in selection.groupby(group_cols, dropna=False, observed=True):
        selector, score_col, gate, top_frac = key
        rows.append(
            {
                "selector": selector,
                "score_col": score_col,
                "gate": gate,
                "top_frac": float(top_frac),
                "months": int(group["month"].nunique()),
                "positive_months": int((_safe_numeric(group["mean_u"]) > 0.0).sum()),
                "mean_u": _safe_mean(group["mean_u"]),
                "worst_month_mean_u": float(_safe_numeric(group["mean_u"]).min()),
                "hit_u": _safe_mean(group["hit_u"]),
                "bad_mae_1r_rate": _safe_mean(group["bad_mae_1r_rate"]),
                "timeout_rate": _safe_mean(group["timeout_rate"]),
                "wide_barrier_25bps_rate": _safe_mean(group["wide_barrier_25bps_rate"]),
                "clean_row_rate": _safe_mean(group["clean_row_rate"]),
                "mean_selected_rows": _safe_mean(group["selected_rows"]),
                "min_selected_rows": int(_safe_numeric(group["selected_rows"]).min()),
                "top_symbol_share": _safe_mean(group["top_symbol_share"]),
                "top_week_share": _safe_mean(group["top_week_share"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["top_frac", "mean_u"], ascending=[True, False])


def _aggregate_overlap(overlap: pd.DataFrame) -> pd.DataFrame:
    if overlap.empty:
        return overlap
    rows: list[dict[str, Any]] = []
    group_cols = ["other_score_col", "gate", "top_frac"]
    for key, group in overlap.groupby(group_cols, dropna=False, observed=True):
        other_score_col, gate, top_frac = key
        rows.append(
            {
                "other_score_col": other_score_col,
                "gate": gate,
                "top_frac": float(top_frac),
                "months": int(group["month"].nunique()),
                "mean_jaccard": _safe_mean(group["jaccard"]),
                "mean_overlap_share_extratrees": _safe_mean(group["overlap_share_extratrees"]),
                "mean_overlap_share_other": _safe_mean(group["overlap_share_other"]),
                "mean_extratrees_only_mean_u": _safe_mean(group["extratrees_only_mean_u"]),
                "mean_other_only_mean_u": _safe_mean(group["other_only_mean_u"]),
                "delta_extratrees_only_vs_other_only_u": (
                    _safe_mean(group["extratrees_only_mean_u"]) - _safe_mean(group["other_only_mean_u"])
                ),
                "mean_overlap_mean_u": _safe_mean(group["overlap_mean_u"]),
                "mean_extratrees_rows": _safe_mean(group["extratrees_rows"]),
                "mean_other_rows": _safe_mean(group["other_rows"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["top_frac", "delta_extratrees_only_vs_other_only_u", "mean_jaccard"],
        ascending=[True, False, False],
    )


def _table(frame: pd.DataFrame, cols: list[str], limit: int = 12) -> str:
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
    selection_agg: pd.DataFrame,
    overlap_agg: pd.DataFrame,
    extratrees_diag: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "source_score_discrepancy_report.md"
    lines = [
        "# Source Score Discrepancy Report",
        "",
        "Scope: diagnostic comparison of month-forward ExtraTrees `S10_policy_net_soft` selections versus existing OOF/proxy score selections.",
        "This does not integrate source tags into training or production policy.",
        "",
        "## Inputs",
        "",
        f"- Joined subset: `{manifest['joined_subset_path']}`",
        f"- Rows: `{manifest['rows']}`",
        f"- Months: `{', '.join(manifest['months'])}`",
        f"- Existing score columns: `{', '.join(manifest['existing_score_cols'])}`",
        f"- Feature count: `{manifest['feature_report'].get('retained_features')}`",
        "",
        "## ExtraTrees Diagnostics",
        "",
        _table(
            extratrees_diag,
            [
                "month",
                "skipped",
                "train_rows",
                "valid_rows",
                "score_ic_u",
                "score_ic_target",
                "target_soft_mean_train",
                "target_hard_rate_train",
            ],
            limit=20,
        ),
        "",
        "## Selection Aggregate",
        "",
        _table(
            selection_agg,
            [
                "selector",
                "score_col",
                "gate",
                "top_frac",
                "months",
                "positive_months",
                "mean_u",
                "worst_month_mean_u",
                "bad_mae_1r_rate",
                "timeout_rate",
                "wide_barrier_25bps_rate",
                "mean_selected_rows",
            ],
            limit=30,
        ),
        "",
        "## Overlap Aggregate",
        "",
        _table(
            overlap_agg,
            [
                "other_score_col",
                "gate",
                "top_frac",
                "months",
                "mean_jaccard",
                "mean_overlap_share_extratrees",
                "mean_overlap_share_other",
                "mean_extratrees_only_mean_u",
                "mean_other_only_mean_u",
                "delta_extratrees_only_vs_other_only_u",
            ],
            limit=30,
        ),
        "",
        "## Interpretation Guide",
        "",
        "- Low Jaccard means the smoke selector and existing scores are selecting different rows.",
        "- Positive `delta_extratrees_only_vs_other_only_u` means ExtraTrees-only rows beat existing-score-only rows inside the same month/gate/top-frac.",
        "- Treat all results as diagnostic because the ExtraTrees selector is a smoke model, not production training.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_report(
    *,
    joined_subset_path: Path,
    output_dir: Path,
    feature_dir: Path | None,
    feature_list_csv: Path | None,
    feature_cols: list[str],
    max_feature_store_features: int | None,
    existing_score_cols: list[str],
    gate_columns: list[str],
    dirty_excluded_column: str,
    barrier_guards: list[float],
    months: list[str],
    top_fracs: list[float],
    seeds: list[int],
    train_lookback_months: int | None,
    min_train_rows: int,
    min_valid_rows: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_frame(joined_subset_path)
    metrics = _path_metrics(frame)
    feature_matrix, feature_report = _load_or_use_feature_columns(
        frame,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        feature_cols=feature_cols,
        max_feature_store_features=max_feature_store_features,
    )
    for col in feature_matrix.columns:
        frame[col] = feature_matrix[col].to_numpy(dtype=np.float32, copy=False)
    features = list(feature_matrix.columns)
    existing_scores = _available_score_columns(frame, existing_score_cols, min_non_null=min_valid_rows)
    if not existing_scores:
        raise ValueError(f"No existing score columns have at least {min_valid_rows} non-null rows")
    gates = _build_gate_specs(
        frame,
        gate_columns=gate_columns,
        dirty_excluded_column=dirty_excluded_column,
        barrier_guards=barrier_guards,
    )
    extratrees_score, extratrees_diag = _score_extratrees_by_month(
        frame,
        metrics,
        features=features,
        months=months,
        seeds=seeds,
        train_lookback_months=train_lookback_months,
        min_train_rows=min_train_rows,
    )
    selection, overlap, source_mix, ledger = _evaluate_discrepancy(
        frame,
        metrics,
        extratrees_score=extratrees_score,
        existing_score_cols=existing_scores,
        gates=gates,
        months=months,
        top_fracs=top_fracs,
        min_valid_rows=min_valid_rows,
    )
    selection_agg = _aggregate_selection(selection)
    overlap_agg = _aggregate_overlap(overlap)

    paths = {
        "selection_summary": output_dir / "source_score_discrepancy_selection_summary.csv",
        "selection_aggregate": output_dir / "source_score_discrepancy_selection_aggregate.csv",
        "overlap": output_dir / "source_score_discrepancy_overlap.csv",
        "overlap_aggregate": output_dir / "source_score_discrepancy_overlap_aggregate.csv",
        "source_mix": output_dir / "source_score_discrepancy_source_mix.csv",
        "selected_ledger": output_dir / "source_score_discrepancy_selected_ledger.csv",
        "extratrees_diagnostics": output_dir / "source_score_discrepancy_extratrees_diagnostics.csv",
        "manifest": output_dir / "manifest.json",
    }
    selection.to_csv(paths["selection_summary"], index=False)
    selection_agg.to_csv(paths["selection_aggregate"], index=False)
    overlap.to_csv(paths["overlap"], index=False)
    overlap_agg.to_csv(paths["overlap_aggregate"], index=False)
    source_mix.to_csv(paths["source_mix"], index=False)
    ledger.to_csv(paths["selected_ledger"], index=False)
    extratrees_diag.to_csv(paths["extratrees_diagnostics"], index=False)

    manifest: dict[str, Any] = {
        "scope": "diagnostic_extratrees_vs_existing_source_score_discrepancy",
        "joined_subset_path": str(joined_subset_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min().isoformat(),
        "timestamp_max": frame["__ts__"].max().isoformat(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "months": months,
        "top_fracs": [float(v) for v in top_fracs],
        "existing_score_cols": existing_scores,
        "requested_existing_score_cols": existing_score_cols,
        "gate_columns": gate_columns,
        "dirty_excluded_column": dirty_excluded_column if dirty_excluded_column in frame.columns else "",
        "barrier_guards": [float(v) for v in barrier_guards],
        "seeds": [int(v) for v in seeds],
        "train_lookback_months": train_lookback_months,
        "min_train_rows": int(min_train_rows),
        "min_valid_rows": int(min_valid_rows),
        "feature_report": feature_report,
        "metric_report": {
            "utility_source": metrics.attrs.get("utility_source"),
            "mae_encoding": metrics.attrs.get("mae_encoding"),
            "extratrees_target": "S10_policy_net_soft",
            "selection": "descending score within each month/gate/top_frac",
        },
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    report_path = _write_report(
        output_dir,
        selection_agg=selection_agg,
        overlap_agg=overlap_agg,
        extratrees_diag=extratrees_diag,
        manifest=manifest,
    )
    manifest["outputs"]["report"] = str(report_path)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joined-subset-path", type=Path, default=DEFAULT_JOINED_SUBSET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--feature-cols", default="")
    parser.add_argument("--max-feature-store-features", type=int, default=24)
    parser.add_argument("--existing-score-cols", default=",".join(DEFAULT_SCORE_COLS))
    parser.add_argument("--source-gates", default=",".join(DEFAULT_SOURCE_GATES))
    parser.add_argument("--dirty-excluded-column", default="train_include_dirty_excluded_v0")
    parser.add_argument("--barrier-guards", default="0.025")
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--top-fracs", default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--train-lookback-months", type=int, default=None)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--min-valid-rows", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_cols = _parse_csv(args.feature_cols, ())
    feature_dir = None if feature_cols else args.feature_dir
    feature_list_csv = None if feature_cols else args.feature_list_csv
    manifest = run_report(
        joined_subset_path=args.joined_subset_path,
        output_dir=args.output_dir,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        feature_cols=feature_cols,
        max_feature_store_features=args.max_feature_store_features,
        existing_score_cols=_parse_csv(args.existing_score_cols, DEFAULT_SCORE_COLS),
        gate_columns=_parse_csv(args.source_gates, DEFAULT_SOURCE_GATES),
        dirty_excluded_column=str(args.dirty_excluded_column),
        barrier_guards=_parse_float_csv(args.barrier_guards, (0.025,)),
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        top_fracs=_parse_float_csv(args.top_fracs, DEFAULT_TOP_FRACS),
        seeds=_parse_int_csv(args.seeds, (42,)),
        train_lookback_months=args.train_lookback_months,
        min_train_rows=args.min_train_rows,
        min_valid_rows=args.min_valid_rows,
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
