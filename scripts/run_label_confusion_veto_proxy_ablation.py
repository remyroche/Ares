#!/usr/bin/env python3
"""Proxy-only label plus confusion-veto ablation.

This tests whether a hard-negative veto learned from prior-month proxy mistakes
can improve label-proxy selections before any model training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_label_matched_clean_dirty_feature_gap import (  # noqa: E402
    DEFAULT_LABELS_PATH,
    _build_frame,
    _parse_csv,
    _parse_float_csv,
    _top_mask,
)
from scripts.report_label_proxy_quality_with_economic_limits import (  # noqa: E402
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_PROXY_TOP_K,
    _add_delta,
    _baseline,
    _score_proxy,
    _slice_week_positions,
    _top_gate,
)
from scripts.run_label_adverse_path_proxy_gate_ablation import (  # noqa: E402
    DEFAULT_GATE_KEEP_FRACS,
    DEFAULT_RISK_PENALTIES,
    _aggregate,
    _path_targets,
    _table,
)
from scripts.run_label_economic_proxy_ablation import _label_targets  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    _decile_diagnostics,
    _feature_columns,
    _json_safe,
    _rank_top_indices,
    _safe_mean,
    _selection_metrics,
    _spearman,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
)
from scripts.run_soft_label_candidate_source_ablation import (  # noqa: E402
    _build_sources,
    _source_context,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_confusion_veto_proxy_ablation_v1")
DEFAULT_LABEL_ARMS = (
    "S61_tpnet_strict_adverse_veto_rank",
    "S62_tpnet_clean_dirty_contrast_rank",
)
DEFAULT_TOP_FRACS = (0.01, 0.03)
DEFAULT_CONFUSION_ARMS = (
    "C1_missed_clean_vs_dirty_fp",
    "C2_oracle_clean_vs_dirty_fp",
)
DEFAULT_CONFUSION_BLEND_WEIGHTS = (0.25, 0.50, 0.75)
DEFAULT_SOURCE_NAMES = ("all",)


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _target_counts(y: pd.Series) -> dict[str, int]:
    vals = _safe_numeric(y)
    return {
        "rows": int(vals.notna().sum()),
        "pos": int(vals.eq(1.0).sum()),
        "neg": int(vals.eq(0.0).sum()),
    }


def _score_sparse_proxy(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    y_train: pd.Series,
    proxy_top_k: int,
    min_pos_rows: int,
    min_neg_rows: int,
) -> tuple[pd.Series, dict[str, Any]]:
    counts = _target_counts(y_train)
    if counts["pos"] < int(min_pos_rows) or counts["neg"] < int(min_neg_rows):
        return pd.Series(0.5, index=valid.index, dtype=np.float32), {
            "proxy_features": [],
            "proxy_top_abs_ic": float("nan"),
            "proxy_mean_top_abs_ic": float("nan"),
            "reason": "insufficient_confusion_target_rows",
            **counts,
        }
    score, diag = _score_proxy(
        train=train,
        valid=valid,
        features=features,
        y_train=y_train,
        proxy_top_k=int(proxy_top_k),
    )
    diag = dict(diag)
    diag.update(counts)
    return score, diag


def _confusion_masks(
    *,
    target_valid: pd.DataFrame,
    label_score: pd.Series,
    strict_clean: pd.Series,
    dirty: pd.Series,
    top_frac: float,
    arm: str,
) -> tuple[pd.Series, pd.Series]:
    oracle = _top_mask(target_valid["target_soft"], float(top_frac))
    proxy = _top_mask(label_score, float(top_frac))
    strict_clean = strict_clean.reset_index(drop=True).astype(bool)
    dirty = dirty.reset_index(drop=True).astype(bool)
    if arm == "C1_missed_clean_vs_dirty_fp":
        pos = oracle & ~proxy & strict_clean
    elif arm == "C2_oracle_clean_vs_dirty_fp":
        pos = oracle & strict_clean
    else:
        raise ValueError(f"Unknown confusion arm: {arm}")
    neg = proxy & ~oracle & dirty
    return pos.reset_index(drop=True), neg.reset_index(drop=True)


def _build_inner_oos_confusion_target(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    features: list[str],
    path_targets: dict[str, pd.Series],
    month_period: pd.Series,
    source_mask: pd.Series,
    outer_month: str,
    top_frac: float,
    confusion_arm: str,
    proxy_top_k: int,
    min_inner_train_rows: int,
    min_inner_valid_rows: int,
    label_score_cache: dict[str, tuple[pd.Series, dict[str, Any]]] | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    y = pd.Series(np.nan, index=frame.index, dtype=np.float32)
    inner_reports: list[dict[str, Any]] = []
    source_mask = source_mask.fillna(False).astype(bool).reindex(frame.index, fill_value=False)
    if label_score_cache is None:
        label_score_cache = {}
    inner_months = sorted(m for m in month_period.dropna().unique().tolist() if str(m) < str(outer_month))
    for inner_month in inner_months:
        inner_train_mask = month_period.lt(str(inner_month)) & source_mask
        inner_valid_mask = month_period.eq(str(inner_month)) & source_mask
        train_rows = int(inner_train_mask.sum())
        valid_rows = int(inner_valid_mask.sum())
        if train_rows < int(min_inner_train_rows) or valid_rows < int(min_inner_valid_rows):
            inner_reports.append(
                {
                    "month": str(inner_month),
                    "train_rows": train_rows,
                    "valid_rows": valid_rows,
                    "skipped": True,
                    "reason": "insufficient_inner_rows",
                    "pos": 0,
                    "neg": 0,
                }
            )
            continue
        cache_key = str(inner_month)
        if cache_key in label_score_cache:
            label_score, label_diag = label_score_cache[cache_key]
        else:
            train = frame.loc[inner_train_mask].copy()
            valid_source = frame.loc[inner_valid_mask].copy()
            label_score, label_diag = _score_proxy(
                train=train,
                valid=valid_source,
                features=features,
                y_train=target.loc[inner_train_mask, "target_soft"],
                proxy_top_k=int(proxy_top_k),
            )
            label_score = label_score.reset_index(drop=True)
            label_score_cache[cache_key] = (label_score, dict(label_diag))
        label_score = label_score.reset_index(drop=True)
        target_valid = target.loc[inner_valid_mask].copy().reset_index(drop=True)
        strict_clean = path_targets["strict_clean"].loc[inner_valid_mask].reset_index(drop=True).gt(0.5)
        dirty = path_targets["dirty"].loc[inner_valid_mask].reset_index(drop=True).gt(0.5)
        pos, neg = _confusion_masks(
            target_valid=target_valid,
            label_score=label_score,
            strict_clean=strict_clean,
            dirty=dirty,
            top_frac=float(top_frac),
            arm=confusion_arm,
        )
        valid_indices = np.flatnonzero(inner_valid_mask.to_numpy(dtype=bool, copy=False))
        y.iloc[valid_indices[np.flatnonzero(pos.to_numpy(dtype=bool, copy=False))]] = 1.0
        y.iloc[valid_indices[np.flatnonzero(neg.to_numpy(dtype=bool, copy=False))]] = 0.0
        inner_reports.append(
            {
                "month": str(inner_month),
                "train_rows": train_rows,
                "valid_rows": valid_rows,
                "skipped": False,
                "pos": int(pos.sum()),
                "neg": int(neg.sum()),
                "label_proxy_features": ",".join(label_diag.get("proxy_features", [])),
            }
        )
    counts = _target_counts(y)
    return y, {
        "outer_month": str(outer_month),
        "top_frac": float(top_frac),
        "confusion_arm": str(confusion_arm),
        **counts,
        "inner_months": inner_reports,
    }


def _selector_scores(
    *,
    label_score: pd.Series,
    confusion_score: pd.Series,
    gate_keep_fracs: list[float],
    risk_penalties: list[float],
    blend_weights: list[float],
) -> list[tuple[str, pd.Series]]:
    label = _safe_numeric(label_score).reset_index(drop=True)
    confusion = _safe_numeric(confusion_score).reset_index(drop=True).fillna(0.5).clip(0.0, 1.0)
    risk = (1.0 - confusion).clip(0.0, 1.0)
    out: list[tuple[str, pd.Series]] = [
        ("label_proxy_oos", label),
        ("label_times_confusion", label * confusion),
    ]
    for weight in blend_weights:
        out.append(
            (
                f"label{1.0 - float(weight):.2f}_confusion{float(weight):.2f}_blend",
                (1.0 - float(weight)) * label + float(weight) * confusion,
            )
        )
    for penalty in risk_penalties:
        out.append((f"label_minus_confusion_risk_{penalty:.2f}", label - float(penalty) * risk))
    for keep_frac in gate_keep_fracs:
        gate = _top_gate(confusion, float(keep_frac))
        out.append((f"confusion_gate{keep_frac:.2f}_then_label", label.where(gate)))
    return out


def _score_period(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    oracle_score: pd.Series,
    period_type: str,
    period: str,
    month: str,
    source: str,
    selector: str,
    label_arm: str,
    confusion_arm: str,
    top_frac: float,
    label_score: pd.Series,
    confusion_score: pd.Series,
    clean_target: pd.Series,
    dirty_target: pd.Series,
    selector_features: str,
    confusion_target_rows: int,
    confusion_target_pos: int,
    confusion_target_neg: int,
) -> dict[str, Any]:
    score = _safe_numeric(score).reset_index(drop=True)
    frame = frame.reset_index(drop=True)
    metrics = metrics.reset_index(drop=True)
    target = target.reset_index(drop=True)
    oracle_score = _safe_numeric(oracle_score).reset_index(drop=True)
    label_score = _safe_numeric(label_score).reset_index(drop=True)
    confusion_score = _safe_numeric(confusion_score).reset_index(drop=True)
    clean_target = _safe_numeric(clean_target).reset_index(drop=True)
    dirty_target = _safe_numeric(dirty_target).reset_index(drop=True)
    row = _selection_metrics(
        frame=frame,
        metrics=metrics,
        target=target,
        score=score,
        arm=f"{selector}::{label_arm}::{confusion_arm}",
        selector=selector,
        period=period,
        top_frac=top_frac,
    )
    _add_delta(row, _baseline(metrics))
    selected = _top_mask(score, top_frac)
    oracle = _top_mask(oracle_score, top_frac)
    recovered = selected & oracle
    row.update(
        {
            "period_type": period_type,
            "month": month,
            "source": source,
            "label_arm": label_arm,
            "confusion_arm": confusion_arm,
            "selector_features": selector_features,
            "score_ic_u": _spearman(score, metrics["u_policy_net"]),
            "score_ic_label": _spearman(score, label_score),
            "score_ic_confusion": _spearman(score, confusion_score),
            "score_ic_clean": _spearman(score, clean_target),
            "score_ic_risk": _spearman(score, dirty_target),
            "oracle_top_rows": int(oracle.sum()),
            "oracle_recovered_rows": int(recovered.sum()),
            "oracle_recovery_rate": float(recovered.sum() / oracle.sum()) if int(oracle.sum()) else 0.0,
            "selected_oracle_overlap_rate": (
                float(recovered.sum() / selected.sum()) if int(selected.sum()) else 0.0
            ),
            "confusion_target_rows": int(confusion_target_rows),
            "confusion_target_pos": int(confusion_target_pos),
            "confusion_target_neg": int(confusion_target_neg),
        }
    )
    row.update(_decile_diagnostics(score, metrics["u_policy_net"]))
    return row


def _table_view(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[col for col in cols if col in frame.columns]].copy()
    if limit is not None:
        view = view.head(limit)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
    return view.to_markdown(index=False)


def _write_markdown(
    output_dir: Path,
    aggregate: pd.DataFrame,
    period_rows: pd.DataFrame,
    confusion_reports: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "label_confusion_veto_proxy_ablation.md"
    aggregate_cols = [
        "acceptance_gate",
        "period_type",
        "source",
        "selector",
        "label_arm",
        "confusion_arm",
        "top_frac",
        "periods",
        "positive_return_period_rate",
        "mean_return_net",
        "worst_period_return_net",
        "sum_return_net_plus10bps",
        "score_ic_u",
        "score_ic_clean",
        "score_ic_risk",
        "bad_mae_1r_rate",
        "p90_mae_norm",
        "strict_clean_row_rate",
        "timeout_rate",
        "overall_oracle_recovery_rate",
        "mean_selected_rows",
    ]
    period_cols = [
        "period",
        "source",
        "selector",
        "label_arm",
        "confusion_arm",
        "top_frac",
        "selected_rows",
        "mean_return_net",
        "score_ic_u",
        "score_ic_confusion",
        "bad_mae_1r_rate",
        "p90_mae_norm",
        "strict_clean_row_rate",
        "timeout_rate",
        "oracle_recovery_rate",
        "confusion_target_pos",
        "confusion_target_neg",
    ]
    confusion_cols = [
        "outer_month",
        "source",
        "label_arm",
        "top_frac",
        "confusion_arm",
        "rows",
        "pos",
        "neg",
        "inner_month_count",
        "inner_months_used",
    ]
    lines = [
        "# Label Confusion-Veto Proxy Ablation",
        "",
        "Scope: proxy-only development diagnostic. No LightGBM, Optuna, policy optimization, or tree smoke model is run.",
        "",
        "The confusion-veto target is built from prior-month OOS label-proxy mistakes only: clean oracle rows missed by the label proxy versus dirty proxy false positives. Validation months are never used to build their own veto target.",
        "",
        f"Labels: `{', '.join(manifest['label_arms'])}`",
        f"Sources: `{', '.join(manifest['sources'])}`",
        f"Confusion arms: `{', '.join(manifest['confusion_arms'])}`",
        f"Top fractions: `{manifest['top_fracs']}`",
        f"Gate keep fractions: `{manifest['gate_keep_fracs']}`",
        f"Risk penalties: `{manifest['risk_penalties']}`",
        f"Blend weights: `{manifest['confusion_blend_weights']}`",
        f"Causal outcome priors: `{manifest['include_causal_outcome_priors']}`",
        f"Causal state-path priors: `{manifest['include_causal_state_path_priors']}`",
        f"Event confirmation features: `{manifest['include_event_confirmation_features']}`",
        f"Adverse-path composites: `{manifest['include_adverse_path_composites']}`",
        "",
        "## Confusion Target Evidence",
        "",
        _table_view(confusion_reports, confusion_cols, limit=80),
        "",
    ]
    for period_type in ("month", "week"):
        subset = aggregate[aggregate["period_type"].eq(period_type)].copy()
        lines.extend(
            [
                f"## {period_type.title()} Aggregate",
                "",
                _table(
                    subset.sort_values(
                        ["acceptance_gate", "overall_oracle_recovery_rate", "sum_return_net_plus10bps"],
                        ascending=[False, False, False],
                    ),
                    aggregate_cols,
                    limit=60,
                ),
                "",
            ]
        )
    focus_selectors = {
        "label_proxy_oos",
        "label_times_confusion",
        "label0.50_confusion0.50_blend",
        "confusion_gate0.30_then_label",
        "confusion_gate0.50_then_label",
        "label_minus_confusion_risk_0.50",
    }
    focus = period_rows[
        period_rows["period_type"].eq("month") & period_rows["selector"].isin(focus_selectors)
    ].copy()
    lines.extend(
        [
            "## Month Detail Focus",
            "",
            _table_view(
                focus.sort_values(["period", "source", "label_arm", "confusion_arm", "selector", "top_frac"]),
                period_cols,
                limit=180,
            ),
            "",
            "## Outputs",
            "",
            f"- Period rows: `{manifest['outputs']['period_rows']}`",
            f"- Aggregate: `{manifest['outputs']['aggregate']}`",
            f"- Confusion target report: `{manifest['outputs']['confusion_targets']}`",
            f"- Manifest: `{manifest['outputs']['manifest']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    label_arms: list[str],
    confusion_arms: list[str],
    top_fracs: list[float],
    gate_keep_fracs: list[float],
    risk_penalties: list[float],
    confusion_blend_weights: list[float],
    proxy_top_k: int,
    include_causal_outcome_priors: bool,
    include_causal_state_path_priors: bool,
    include_event_confirmation_features: bool,
    include_adverse_path_composites: bool,
    prior_windows_days: list[float],
    prior_embargo_hours: float,
    state_path_prior_features: list[str],
    event_feature_store_features: list[str],
    min_inner_train_rows: int,
    min_inner_valid_rows: int,
    min_train_source_rows: int,
    min_valid_source_rows: int,
    min_confusion_pos_rows: int,
    min_confusion_neg_rows: int,
    min_material_selected_rows: int,
    sources: list[str] | None,
    run_gap_hours: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, metrics, reports = _build_frame(
        labels_path=labels_path,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        max_feature_store_features=max_feature_store_features,
        include_causal_outcome_priors=include_causal_outcome_priors,
        include_causal_state_path_priors=include_causal_state_path_priors,
        include_event_confirmation_features=include_event_confirmation_features,
        include_adverse_path_composites=include_adverse_path_composites,
        prior_windows_days=prior_windows_days,
        prior_embargo_hours=prior_embargo_hours,
        state_path_prior_features=state_path_prior_features,
        event_feature_store_features=event_feature_store_features,
    )
    source_context = _source_context(frame)
    overlap = [col for col in source_context.columns if col in frame.columns]
    if overlap:
        frame = frame.drop(columns=overlap)
    frame = pd.concat([frame, source_context.astype(np.float32, copy=False)], axis=1)
    source_masks_all = _build_sources(frame, source_context, run_gap_hours=float(run_gap_hours))
    requested_sources = list(sources) if sources else list(DEFAULT_SOURCE_NAMES)
    unknown_sources = sorted(set(requested_sources).difference(source_masks_all))
    if unknown_sources:
        raise ValueError(f"Unknown sources: {unknown_sources}. Available: {sorted(source_masks_all)}")
    selected_sources = {
        source: source_masks_all[source].fillna(False).astype(bool).reindex(frame.index, fill_value=False)
        for source in requested_sources
    }

    features = _feature_columns(frame)
    targets = _label_targets(frame, metrics)
    unknown = sorted(set(label_arms).difference(targets))
    if unknown:
        raise ValueError(f"Unknown label arms: {unknown}")
    path_targets = _path_targets(metrics)
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    months = sorted(m for m in month_period.dropna().unique().tolist() if m >= "2026-04")

    rows: list[dict[str, Any]] = []
    confusion_target_reports: list[dict[str, Any]] = []
    for source_name, source_mask in selected_sources.items():
        for month in months:
            train_mask = month_period.lt(str(month)) & source_mask
            valid_mask = month_period.eq(str(month)) & source_mask
            if int(train_mask.sum()) < int(min_train_source_rows) or int(valid_mask.sum()) < int(min_valid_source_rows):
                continue
            train = frame.loc[train_mask].copy()
            valid_source = frame.loc[valid_mask].copy()
            valid = valid_source.reset_index(drop=True)
            valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
            valid_indices = np.arange(len(valid), dtype=np.int64)
            valid_clean = path_targets["strict_clean"].loc[valid_mask].reset_index(drop=True)
            valid_dirty = path_targets["dirty"].loc[valid_mask].reset_index(drop=True)

            for label_arm in label_arms:
                target = targets[label_arm]
                target_valid = target.loc[valid_mask].copy().reset_index(drop=True)
                label_score, label_diag = _score_proxy(
                    train=train,
                    valid=valid_source,
                    features=features,
                    y_train=target.loc[train_mask, "target_soft"],
                    proxy_top_k=int(proxy_top_k),
                )
                label_score = label_score.reset_index(drop=True)
                inner_label_score_cache: dict[str, tuple[pd.Series, dict[str, Any]]] = {}
                for top_frac in top_fracs:
                    for confusion_arm in confusion_arms:
                        confusion_target, confusion_report = _build_inner_oos_confusion_target(
                            frame=frame,
                            metrics=metrics,
                            target=target,
                            features=features,
                            path_targets=path_targets,
                            month_period=month_period,
                            source_mask=source_mask,
                            outer_month=str(month),
                            top_frac=float(top_frac),
                            confusion_arm=str(confusion_arm),
                            proxy_top_k=int(proxy_top_k),
                            min_inner_train_rows=int(min_inner_train_rows),
                            min_inner_valid_rows=int(min_inner_valid_rows),
                            label_score_cache=inner_label_score_cache,
                        )
                        confusion_report.update({"source": str(source_name), "label_arm": str(label_arm)})
                        confusion_report["inner_month_count"] = len(confusion_report.get("inner_months", []))
                        confusion_report["inner_months_used"] = ",".join(
                            str(row["month"])
                            for row in confusion_report.get("inner_months", [])
                            if not bool(row.get("skipped"))
                        )
                        confusion_target_reports.append(confusion_report)
                        confusion_score, confusion_diag = _score_sparse_proxy(
                            train=train,
                            valid=valid_source,
                            features=features,
                            y_train=confusion_target.loc[train_mask],
                            proxy_top_k=int(proxy_top_k),
                            min_pos_rows=int(min_confusion_pos_rows),
                            min_neg_rows=int(min_confusion_neg_rows),
                        )
                        confusion_score = confusion_score.reset_index(drop=True)
                        selector_specs = _selector_scores(
                            label_score=label_score,
                            confusion_score=confusion_score,
                            gate_keep_fracs=gate_keep_fracs,
                            risk_penalties=risk_penalties,
                            blend_weights=confusion_blend_weights,
                        )
                        feature_summary = (
                            "label="
                            + ",".join(label_diag.get("proxy_features", []))
                            + "; confusion="
                            + ",".join(confusion_diag.get("proxy_features", []))
                            + f"; confusion_rows={int(confusion_diag.get('rows', 0) or 0)}"
                            + f"; confusion_pos={int(confusion_diag.get('pos', 0) or 0)}"
                            + f"; confusion_neg={int(confusion_diag.get('neg', 0) or 0)}"
                        )
                        period_slices = [("month", month, valid_indices)]
                        period_slices.extend(("week", week, pos) for week, pos in _slice_week_positions(valid))
                        for selector, score in selector_specs:
                            score = _safe_numeric(score).reset_index(drop=True)
                            for period_type, period, pos in period_slices:
                                local_frame = valid.iloc[pos].reset_index(drop=True)
                                local_metrics = valid_metrics.iloc[pos].reset_index(drop=True)
                                local_target = target_valid.iloc[pos].reset_index(drop=True)
                                local_score = score.iloc[pos].reset_index(drop=True)
                                local_label_score = label_score.iloc[pos].reset_index(drop=True)
                                local_confusion_score = confusion_score.iloc[pos].reset_index(drop=True)
                                local_clean = valid_clean.iloc[pos].reset_index(drop=True)
                                local_dirty = valid_dirty.iloc[pos].reset_index(drop=True)
                                local_oracle = target_valid["target_soft"].iloc[pos].reset_index(drop=True)
                                rows.append(
                                    _score_period(
                                        frame=local_frame,
                                        metrics=local_metrics,
                                        target=local_target,
                                        score=local_score,
                                        oracle_score=local_oracle,
                                        period_type=period_type,
                                        period=str(period),
                                        month=str(month),
                                        source=str(source_name),
                                        selector=str(selector),
                                        label_arm=str(label_arm),
                                        confusion_arm=str(confusion_arm),
                                        top_frac=float(top_frac),
                                        label_score=local_label_score,
                                        confusion_score=local_confusion_score,
                                        clean_target=local_clean,
                                        dirty_target=local_dirty,
                                        selector_features=feature_summary,
                                        confusion_target_rows=int(confusion_diag.get("rows", 0) or 0),
                                        confusion_target_pos=int(confusion_diag.get("pos", 0) or 0),
                                        confusion_target_neg=int(confusion_diag.get("neg", 0) or 0),
                                    )
                                )

    period_rows = pd.DataFrame(rows)
    aggregate = _aggregate(period_rows, min_material_selected_rows=min_material_selected_rows)
    confusion_targets = pd.DataFrame(confusion_target_reports)

    paths = {
        "period_rows": output_dir / "label_confusion_veto_period_rows.csv",
        "aggregate": output_dir / "label_confusion_veto_aggregate.csv",
        "confusion_targets": output_dir / "label_confusion_veto_target_rows.csv",
        "manifest": output_dir / "manifest.json",
    }
    period_rows.to_csv(paths["period_rows"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    confusion_targets.to_csv(paths["confusion_targets"], index=False)

    manifest = {
        "scope": "proxy_only_label_confusion_veto_ablation",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "feature_count": int(len(features)),
        "label_arms": list(label_arms),
        "sources": list(selected_sources),
        "run_gap_hours": float(run_gap_hours),
        "confusion_arms": list(confusion_arms),
        "top_fracs": [float(v) for v in top_fracs],
        "gate_keep_fracs": [float(v) for v in gate_keep_fracs],
        "risk_penalties": [float(v) for v in risk_penalties],
        "confusion_blend_weights": [float(v) for v in confusion_blend_weights],
        "proxy_top_k": int(proxy_top_k),
        "min_inner_train_rows": int(min_inner_train_rows),
        "min_inner_valid_rows": int(min_inner_valid_rows),
        "min_train_source_rows": int(min_train_source_rows),
        "min_valid_source_rows": int(min_valid_source_rows),
        "min_confusion_pos_rows": int(min_confusion_pos_rows),
        "min_confusion_neg_rows": int(min_confusion_neg_rows),
        "min_material_selected_rows": int(min_material_selected_rows),
        "include_causal_outcome_priors": bool(include_causal_outcome_priors),
        "include_causal_state_path_priors": bool(include_causal_state_path_priors),
        "include_event_confirmation_features": bool(include_event_confirmation_features),
        "include_adverse_path_composites": bool(include_adverse_path_composites),
        "prior_windows_days": [float(v) for v in prior_windows_days],
        "prior_embargo_hours": float(prior_embargo_hours),
        "months": months,
        "outputs": {key: str(value) for key, value in paths.items()},
        **reports,
    }
    markdown = _write_markdown(output_dir, aggregate, period_rows, confusion_targets, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--label-arms", type=lambda value: _parse_csv(value, DEFAULT_LABEL_ARMS), default=",".join(DEFAULT_LABEL_ARMS))
    parser.add_argument("--sources", type=lambda value: _parse_csv(value, DEFAULT_SOURCE_NAMES), default=",".join(DEFAULT_SOURCE_NAMES))
    parser.add_argument("--run-gap-hours", type=float, default=6.0)
    parser.add_argument("--confusion-arms", type=lambda value: _parse_csv(value, DEFAULT_CONFUSION_ARMS), default=",".join(DEFAULT_CONFUSION_ARMS))
    parser.add_argument("--top-fracs", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--gate-keep-fracs", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_GATE_KEEP_FRACS))
    parser.add_argument("--risk-penalties", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_RISK_PENALTIES))
    parser.add_argument(
        "--confusion-blend-weights",
        type=_parse_float_csv,
        default=",".join(str(v) for v in DEFAULT_CONFUSION_BLEND_WEIGHTS),
    )
    parser.add_argument("--proxy-top-k", type=int, default=DEFAULT_PROXY_TOP_K)
    parser.add_argument("--include-causal-outcome-priors", action="store_true")
    parser.add_argument("--include-causal-state-path-priors", action="store_true")
    parser.add_argument("--include-event-confirmation-features", action="store_true")
    parser.add_argument("--include-adverse-path-composites", action="store_true")
    parser.add_argument(
        "--prior-windows-days",
        type=_parse_float_csv,
        default=",".join(str(v) for v in DEFAULT_PRIOR_WINDOWS_DAYS),
    )
    parser.add_argument("--prior-embargo-hours", type=float, default=24.0)
    parser.add_argument(
        "--state-path-prior-features",
        type=lambda value: _parse_csv(value, DEFAULT_STATE_PATH_PRIOR_FEATURES),
        default=",".join(DEFAULT_STATE_PATH_PRIOR_FEATURES),
    )
    parser.add_argument(
        "--event-feature-store-features",
        type=lambda value: _parse_csv(value, DEFAULT_EVENT_FEATURE_STORE_FEATURES),
        default=",".join(DEFAULT_EVENT_FEATURE_STORE_FEATURES),
    )
    parser.add_argument("--min-inner-train-rows", type=int, default=500)
    parser.add_argument("--min-inner-valid-rows", type=int, default=100)
    parser.add_argument("--min-train-source-rows", type=int, default=500)
    parser.add_argument("--min-valid-source-rows", type=int, default=100)
    parser.add_argument("--min-confusion-pos-rows", type=int, default=5)
    parser.add_argument("--min-confusion-neg-rows", type=int, default=5)
    parser.add_argument("--min-material-selected-rows", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        label_arms=list(args.label_arms),
        confusion_arms=list(args.confusion_arms),
        top_fracs=[float(v) for v in args.top_fracs],
        gate_keep_fracs=[float(v) for v in args.gate_keep_fracs],
        risk_penalties=[float(v) for v in args.risk_penalties],
        confusion_blend_weights=[float(v) for v in args.confusion_blend_weights],
        proxy_top_k=int(args.proxy_top_k),
        include_causal_outcome_priors=bool(args.include_causal_outcome_priors),
        include_causal_state_path_priors=bool(args.include_causal_state_path_priors),
        include_event_confirmation_features=bool(args.include_event_confirmation_features),
        include_adverse_path_composites=bool(args.include_adverse_path_composites),
        prior_windows_days=[float(v) for v in args.prior_windows_days],
        prior_embargo_hours=float(args.prior_embargo_hours),
        state_path_prior_features=list(args.state_path_prior_features),
        event_feature_store_features=list(args.event_feature_store_features),
        min_inner_train_rows=int(args.min_inner_train_rows),
        min_inner_valid_rows=int(args.min_inner_valid_rows),
        min_train_source_rows=int(args.min_train_source_rows),
        min_valid_source_rows=int(args.min_valid_source_rows),
        min_confusion_pos_rows=int(args.min_confusion_pos_rows),
        min_confusion_neg_rows=int(args.min_confusion_neg_rows),
        min_material_selected_rows=int(args.min_material_selected_rows),
        sources=list(args.sources) if args.sources else None,
        run_gap_hours=float(args.run_gap_hours),
    )
    print(
        json.dumps(
            _json_safe(
                {
                    key: value
                    for key, value in manifest.items()
                    if key not in {
                        "feature_store",
                        "causal_outcome_priors",
                        "causal_state_path_priors",
                        "event_confirmation_features",
                        "adverse_path_composites",
                    }
                }
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
