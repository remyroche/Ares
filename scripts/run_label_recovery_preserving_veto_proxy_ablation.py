#!/usr/bin/env python3
"""Recovery-preserving label/confusion-veto proxy ablation.

This is a pre-training diagnostic. It tests whether a causal hard-negative
veto can clean up path risk while preserving a wider label-proxy candidate
pool, before fitting base/meta models or policy geometry.
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

from scripts.diagnose_label_matched_clean_dirty_feature_gap import (  # noqa: E402
    DEFAULT_LABELS_PATH,
    _build_frame,
    _parse_csv,
    _parse_float_csv,
)
from scripts.report_label_proxy_quality_with_economic_limits import (  # noqa: E402
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_PROXY_TOP_K,
    _score_proxy,
    _slice_week_positions,
    _top_gate,
)
from scripts.run_label_adverse_path_proxy_gate_ablation import (  # noqa: E402
    DEFAULT_RISK_PENALTIES,
    _aggregate,
    _path_targets,
    _table,
)
from scripts.run_label_confusion_veto_proxy_ablation import (  # noqa: E402
    DEFAULT_CONFUSION_ARMS,
    DEFAULT_LABEL_ARMS,
    DEFAULT_TOP_FRACS,
    _build_inner_oos_confusion_target,
    _score_period,
    _score_sparse_proxy,
    _safe_numeric,
)
from scripts.run_label_economic_proxy_ablation import _label_targets  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    _feature_columns,
    _json_safe,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_recovery_preserving_veto_proxy_ablation_v1")
DEFAULT_CANDIDATE_MULTS = (2.0, 3.0, 5.0, 8.0)
DEFAULT_POOL_KEEP_FRACS = (0.35, 0.50, 0.70)
DEFAULT_POOL_BLEND_WEIGHTS = (0.25, 0.50, 0.75)


def _score_floor(score: pd.Series) -> float:
    values = _safe_numeric(score).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return -1.0e6
    return float(values.min()) - 1.0e6


def _finite_pool_score(raw_score: pd.Series, pool: pd.Series, *, floor_from: pd.Series) -> pd.Series:
    raw = _safe_numeric(raw_score).reset_index(drop=True)
    pool = pool.reset_index(drop=True).astype(bool)
    out = pd.Series(_score_floor(floor_from), index=raw.index, dtype=np.float64)
    out.loc[pool] = raw.loc[pool].fillna(_score_floor(floor_from))
    return out


def _candidate_pool(label_score: pd.Series, *, top_frac: float, candidate_mult: float) -> pd.Series:
    label = _safe_numeric(label_score).reset_index(drop=True)
    frac = min(1.0, max(float(top_frac), float(top_frac) * float(candidate_mult)))
    return _top_gate(label, frac).reset_index(drop=True).astype(bool)


def _gate_within_pool(score: pd.Series, pool: pd.Series, keep_frac: float) -> pd.Series:
    masked = _safe_numeric(score).reset_index(drop=True).where(pool.reset_index(drop=True).astype(bool))
    return _top_gate(masked, float(keep_frac)).reset_index(drop=True).astype(bool)


def _selector_scores(
    *,
    label_score: pd.Series,
    confusion_score: pd.Series,
    top_frac: float,
    candidate_mults: list[float],
    pool_keep_fracs: list[float],
    pool_blend_weights: list[float],
    risk_penalties: list[float],
    include_strict_pool_gates: bool,
) -> list[dict[str, Any]]:
    label = _safe_numeric(label_score).reset_index(drop=True)
    confusion = _safe_numeric(confusion_score).reset_index(drop=True).fillna(0.5).clip(0.0, 1.0)
    risk = (1.0 - confusion).clip(0.0, 1.0)
    out: list[dict[str, Any]] = [
        {
            "name": "label_proxy_oos",
            "score": label,
            "mode": "baseline",
            "candidate_mult": float("nan"),
            "pool_keep_frac": float("nan"),
            "pool_rows": int(len(label)),
            "kept_rows": int(len(label)),
        }
    ]
    for candidate_mult in candidate_mults:
        pool = _candidate_pool(label, top_frac=float(top_frac), candidate_mult=float(candidate_mult))
        pool_rows = int(pool.sum())
        pool_label_score = _finite_pool_score(label, pool, floor_from=label)
        out.append(
            {
                "name": f"pool{candidate_mult:.1f}_label_only",
                "score": pool_label_score,
                "mode": "pool_label_only",
                "candidate_mult": float(candidate_mult),
                "pool_keep_frac": float("nan"),
                "pool_rows": pool_rows,
                "kept_rows": pool_rows,
            }
        )
        for weight in pool_blend_weights:
            raw = (1.0 - float(weight)) * label + float(weight) * confusion
            out.append(
                {
                    "name": f"pool{candidate_mult:.1f}_label{1.0 - float(weight):.2f}_conf{float(weight):.2f}_rerank",
                    "score": _finite_pool_score(raw, pool, floor_from=label),
                    "mode": "pool_confusion_rerank",
                    "candidate_mult": float(candidate_mult),
                    "pool_keep_frac": float("nan"),
                    "pool_rows": pool_rows,
                    "kept_rows": pool_rows,
                }
            )
        for penalty in risk_penalties:
            raw = label - float(penalty) * risk
            out.append(
                {
                    "name": f"pool{candidate_mult:.1f}_label_minus_risk{float(penalty):.2f}_rerank",
                    "score": _finite_pool_score(raw, pool, floor_from=label),
                    "mode": "pool_risk_penalty_rerank",
                    "candidate_mult": float(candidate_mult),
                    "pool_keep_frac": float("nan"),
                    "pool_rows": pool_rows,
                    "kept_rows": pool_rows,
                }
            )
        for keep_frac in pool_keep_fracs:
            keep = _gate_within_pool(confusion, pool, float(keep_frac))
            # Fill selector: kept rows receive a full-rank bonus, but if the
            # kept pool is smaller than the requested top bucket, high-label
            # vetoed candidates can fill the remaining slots.
            fill_raw = label + keep.astype(float)
            out.append(
                {
                    "name": f"pool{candidate_mult:.1f}_conf_keep{float(keep_frac):.2f}_bonus_fill",
                    "score": _finite_pool_score(fill_raw, pool, floor_from=label),
                    "mode": "pool_confusion_bonus_fill",
                    "candidate_mult": float(candidate_mult),
                    "pool_keep_frac": float(keep_frac),
                    "pool_rows": pool_rows,
                    "kept_rows": int(keep.sum()),
                }
            )
            if include_strict_pool_gates:
                strict = label.where(keep)
                out.append(
                    {
                        "name": f"pool{candidate_mult:.1f}_conf_keep{float(keep_frac):.2f}_strict",
                        "score": strict,
                        "mode": "pool_confusion_strict_gate",
                        "candidate_mult": float(candidate_mult),
                        "pool_keep_frac": float(keep_frac),
                        "pool_rows": pool_rows,
                        "kept_rows": int(keep.sum()),
                    }
                )
    return out


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
    path = output_dir / "label_recovery_preserving_veto_proxy_ablation.md"
    aggregate_cols = [
        "acceptance_gate",
        "period_type",
        "selector",
        "label_arm",
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
        "selector_mode",
        "candidate_mult",
        "pool_keep_frac",
        "pool_rows",
        "kept_rows",
    ]
    confusion_cols = [
        "outer_month",
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
        "# Recovery-Preserving Confusion-Veto Proxy Ablation",
        "",
        "Scope: proxy-only development diagnostic. No LightGBM, Optuna, policy optimization, or tree smoke model is run.",
        "",
        "This variant starts from a wider label-proxy candidate pool, then applies causal confusion-veto reranking or fill-style gating inside that pool. The intent is to reduce dirty path exposure without globally replacing the label score or throwing away recoverable clean-oracle rows.",
        "",
        f"Labels: `{', '.join(manifest['label_arms'])}`",
        f"Confusion arms: `{', '.join(manifest['confusion_arms'])}`",
        f"Top fractions: `{manifest['top_fracs']}`",
        f"Candidate multipliers: `{manifest['candidate_mults']}`",
        f"Pool keep fractions: `{manifest['pool_keep_fracs']}`",
        f"Pool blend weights: `{manifest['pool_blend_weights']}`",
        f"Risk penalties: `{manifest['risk_penalties']}`",
        f"Strict pool gates: `{manifest['include_strict_pool_gates']}`",
        "",
        "The confusion target is still built from prior-month OOS label-proxy mistakes only. The validation month is never used to build its own veto target.",
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
                    limit=80,
                ),
                "",
            ]
        )
    focus = period_rows[
        period_rows["period_type"].eq("month")
        & (
            period_rows["selector"].astype(str).str.contains("label_proxy_oos")
            | period_rows["selector"].astype(str).str.contains("bonus_fill")
            | period_rows["selector"].astype(str).str.contains("rerank")
        )
    ].copy()
    lines.extend(
        [
            "## Month Detail Focus",
            "",
            _table_view(
                focus.sort_values(["period", "label_arm", "confusion_arm", "top_frac", "selector"]),
                period_cols,
                limit=220,
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
    candidate_mults: list[float],
    pool_keep_fracs: list[float],
    pool_blend_weights: list[float],
    risk_penalties: list[float],
    include_strict_pool_gates: bool,
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
    min_confusion_pos_rows: int,
    min_confusion_neg_rows: int,
    min_material_selected_rows: int,
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
    for month in months:
        train_mask = month_period < month
        valid_mask = month_period == month
        if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
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
            for top_frac in top_fracs:
                for confusion_arm in confusion_arms:
                    confusion_target, confusion_report = _build_inner_oos_confusion_target(
                        frame=frame,
                        metrics=metrics,
                        target=target,
                        features=features,
                        path_targets=path_targets,
                        month_period=month_period,
                        outer_month=str(month),
                        top_frac=float(top_frac),
                        confusion_arm=str(confusion_arm),
                        proxy_top_k=int(proxy_top_k),
                        min_inner_train_rows=int(min_inner_train_rows),
                        min_inner_valid_rows=int(min_inner_valid_rows),
                    )
                    confusion_report.update({"label_arm": str(label_arm)})
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
                        top_frac=float(top_frac),
                        candidate_mults=candidate_mults,
                        pool_keep_fracs=pool_keep_fracs,
                        pool_blend_weights=pool_blend_weights,
                        risk_penalties=risk_penalties,
                        include_strict_pool_gates=include_strict_pool_gates,
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
                    for spec in selector_specs:
                        score = _safe_numeric(spec["score"]).reset_index(drop=True)
                        selector_name = f"{confusion_arm}::{spec['name']}"
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
                            row = _score_period(
                                frame=local_frame,
                                metrics=local_metrics,
                                target=local_target,
                                score=local_score,
                                oracle_score=local_oracle,
                                period_type=period_type,
                                period=str(period),
                                month=str(month),
                                selector=selector_name,
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
                            row.update(
                                {
                                    "selector_mode": spec["mode"],
                                    "candidate_mult": spec["candidate_mult"],
                                    "pool_keep_frac": spec["pool_keep_frac"],
                                    "pool_rows": int(spec["pool_rows"]),
                                    "kept_rows": int(spec["kept_rows"]),
                                }
                            )
                            rows.append(row)

    period_rows = pd.DataFrame(rows)
    aggregate = _aggregate(period_rows, min_material_selected_rows=min_material_selected_rows)
    confusion_targets = pd.DataFrame(confusion_target_reports)

    paths = {
        "period_rows": output_dir / "label_recovery_preserving_veto_period_rows.csv",
        "aggregate": output_dir / "label_recovery_preserving_veto_aggregate.csv",
        "confusion_targets": output_dir / "label_recovery_preserving_veto_target_rows.csv",
        "manifest": output_dir / "manifest.json",
    }
    period_rows.to_csv(paths["period_rows"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    confusion_targets.to_csv(paths["confusion_targets"], index=False)

    manifest = {
        "scope": "proxy_only_label_recovery_preserving_confusion_veto_ablation",
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
        "confusion_arms": list(confusion_arms),
        "top_fracs": [float(v) for v in top_fracs],
        "candidate_mults": [float(v) for v in candidate_mults],
        "pool_keep_fracs": [float(v) for v in pool_keep_fracs],
        "pool_blend_weights": [float(v) for v in pool_blend_weights],
        "risk_penalties": [float(v) for v in risk_penalties],
        "include_strict_pool_gates": bool(include_strict_pool_gates),
        "proxy_top_k": int(proxy_top_k),
        "min_inner_train_rows": int(min_inner_train_rows),
        "min_inner_valid_rows": int(min_inner_valid_rows),
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
    parser.add_argument("--confusion-arms", type=lambda value: _parse_csv(value, DEFAULT_CONFUSION_ARMS), default=",".join(DEFAULT_CONFUSION_ARMS))
    parser.add_argument("--top-fracs", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--candidate-mults", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_CANDIDATE_MULTS))
    parser.add_argument("--pool-keep-fracs", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_POOL_KEEP_FRACS))
    parser.add_argument("--pool-blend-weights", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_POOL_BLEND_WEIGHTS))
    parser.add_argument("--risk-penalties", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_RISK_PENALTIES))
    parser.add_argument("--include-strict-pool-gates", action="store_true")
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
        candidate_mults=[float(v) for v in args.candidate_mults],
        pool_keep_fracs=[float(v) for v in args.pool_keep_fracs],
        pool_blend_weights=[float(v) for v in args.pool_blend_weights],
        risk_penalties=[float(v) for v in args.risk_penalties],
        include_strict_pool_gates=bool(args.include_strict_pool_gates),
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
        min_confusion_pos_rows=int(args.min_confusion_pos_rows),
        min_confusion_neg_rows=int(args.min_confusion_neg_rows),
        min_material_selected_rows=int(args.min_material_selected_rows),
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
