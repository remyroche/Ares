#!/usr/bin/env python3
"""Recoverability diagnostic for Stage108 hard-binary path-grid labels.

No model training is run. The script reconstructs Stage108 path-grid targets,
reruns the prior-month economic-IC feature proxy, then compares clean oracle
rows with dirty proxy false positives inside matched day/regime buckets.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_label_matched_clean_dirty_feature_gap import (  # noqa: E402
    _bucket_key,
    _build_frame,
    _eligible_masks,
    _family_summary,
    _feature_contrast,
    _mfe_mae,
    _summary_row,
    _top_mask,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    _feature_columns,
    _json_safe,
    _rank_top_indices,
    _safe_mean,
    _safe_quantile,
    _selection_metrics,
)
from scripts.run_path_aware_label_target_grid import (  # noqa: E402
    DEFAULT_FEATURE_LIST_CSV,
    TargetSpec,
    _target_for_spec,
)
from scripts.run_soft_label_candidate_source_ablation import (  # noqa: E402
    _build_sources,
    _source_context,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
    _proxy_score,
)


DEFAULT_STAGE_DIR = Path("data_perp/reports/path_aware_label_target_grid_stage108_decisive_hard_econic_v1")
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/pathgrid_stage108_recoverability_diagnostic_v1")
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_TOP_FRACS = (0.005, 0.01)
DEFAULT_MATCH_MODES = ("day_side", "regime_side")
DEFAULT_PROFILE_SOURCES = (
    "quiet_mid",
    "loud_event",
    "confirmed_lowbarrier_quality",
    "confirmed_impulse_lowbarrier",
    "clean_breakout_quality",
    "rebound_mid",
    "run_entry_confirmed_lowbarrier_quality",
    "run_entry_dual_prior_confirmed_lowbarrier_quality",
)
PROFILE_CONTEXT_COLUMNS = (
    "source_loud_intensity",
    "source_quiet_score",
    "source_oi_location",
    "source_liquidity",
    "source_low_barrier",
    "source_low_zscore",
    "source_low_atr_compression",
    "source_low_range_location",
    "source_rebound_context",
    "source_event_quality",
    "source_time_edge_prior_quality",
    "source_adverse_prior_safety",
    "source_dual_prior_quality",
)
REGIME_COLUMNS = (
    "G_VOL",
    "__regime_vol_12h__",
    "__regime_vol_48h__",
    "__regime_volume_12h__",
    "__regime_volume_48h__",
    "__regime_trend_12h__",
    "__regime_trend_48h__",
)


def _parse_csv(value: str | list[str] | tuple[str, ...], default: tuple[str, ...] = ()) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    text = str(value).strip()
    if not text:
        return list(default)
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_float_csv(value: str | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(part) for part in value]
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _table(frame: pd.DataFrame, cols: list[str], *, limit: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[col for col in cols if col in frame.columns]].copy()
    if limit is not None:
        view = view.head(limit)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
    return view.to_markdown(index=False)


def _strict_clean(metrics: pd.DataFrame) -> pd.Series:
    mfe_mae = _mfe_mae(metrics)
    return (
        (_safe_numeric(metrics["u_policy_net"]) > 0.0)
        & (_safe_numeric(metrics["mae_norm"]) <= 0.85)
        & (_safe_numeric(metrics["barrier"]) <= 0.024)
        & (mfe_mae >= 1.35)
        & (~metrics["is_timeout"].astype(bool))
    )


def _dirty(metrics: pd.DataFrame) -> pd.Series:
    mfe_mae = _mfe_mae(metrics)
    return (
        (_safe_numeric(metrics["u_policy_net"]) <= 0.0)
        | (_safe_numeric(metrics["mae_norm"]) >= 1.0)
        | (_safe_numeric(metrics["barrier"]) > 0.025)
        | (mfe_mae < 1.25)
        | metrics["is_timeout"].astype(bool)
    )


def _spec_from_row(row: pd.Series) -> TargetSpec:
    return TargetSpec(
        name=str(row["candidate"]),
        u_floor=float(row["u_floor"]),
        u_temp=float(row["u_temp"]),
        mae_cap=float(row["mae_cap"]),
        mae_temp=float(row["mae_temp"]),
        mfe_mae_min=float(row["mfe_mae_min"]),
        mfe_mae_temp=float(row["mfe_mae_temp"]),
        barrier_cap=float(row["barrier_cap"]),
        barrier_temp=float(row["barrier_temp"]),
        timeout_mode=str(row["timeout_mode"]),
        bars_cap=float(row["bars_cap"]),
        bars_temp=float(row["bars_temp"]),
    )


def _choose_candidates(summary: pd.DataFrame, candidates: list[str], max_candidates: int) -> list[str]:
    if candidates:
        missing = sorted(set(candidates).difference(summary["candidate"].astype(str)))
        if missing:
            raise ValueError(f"Unknown candidates: {missing}")
        return list(dict.fromkeys(candidates))
    oracle = summary[summary["oracle_trainworthy"].gt(0)].copy()
    oracle = oracle.sort_values(
        ["oracle_trainworthy", "best_oracle_holdout_mean_return_net", "best_non_oracle_holdout_mean_return_net"],
        ascending=[False, False, False],
    )
    proxy = summary.sort_values(
        ["best_non_oracle_holdout_mean_return_net", "best_non_oracle_fit_mean_return_net"],
        ascending=[False, False],
    )
    out: list[str] = []
    for value in oracle["candidate"].astype(str).head(max_candidates).tolist():
        if value not in out:
            out.append(value)
    for value in proxy["candidate"].astype(str).head(max(2, max_candidates // 2)).tolist():
        if value not in out:
            out.append(value)
    return out[: int(max_candidates)]


def _selected_ledger_rows(
    *,
    valid: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    source_context: pd.DataFrame,
    source_masks: dict[str, pd.Series],
    candidate: str,
    month: str,
    top_frac: float,
    oracle_mask: pd.Series,
    proxy_mask: pd.Series,
    strict_clean: pd.Series,
    dirty: pd.Series,
) -> list[dict[str, Any]]:
    selected = oracle_mask | proxy_mask
    rows: list[dict[str, Any]] = []
    if not bool(selected.any()):
        return rows
    mfe_mae = _mfe_mae(metrics)
    for pos in np.flatnonzero(selected.to_numpy(dtype=bool, copy=False)):
        row: dict[str, Any] = {
            "candidate": candidate,
            "month": month,
            "top_frac": float(top_frac),
            "position": int(pos),
            "is_oracle_top": bool(oracle_mask.iloc[pos]),
            "is_proxy_top": bool(proxy_mask.iloc[pos]),
            "is_recovered": bool(oracle_mask.iloc[pos] and proxy_mask.iloc[pos]),
            "is_missed_oracle": bool(oracle_mask.iloc[pos] and not proxy_mask.iloc[pos]),
            "is_dirty_proxy_false_positive": bool(proxy_mask.iloc[pos] and not oracle_mask.iloc[pos] and dirty.iloc[pos]),
            "strict_clean": bool(strict_clean.iloc[pos]),
            "dirty": bool(dirty.iloc[pos]),
            "__ts__": valid["__ts__"].iloc[pos],
            "__symbol__": valid["__symbol__"].iloc[pos],
            "side": metrics["side"].iloc[pos],
            "target_soft": target["target_soft"].iloc[pos],
            "target_hard": target["target_hard"].iloc[pos],
            "proxy_score": score.iloc[pos],
            "u_policy_net": metrics["u_policy_net"].iloc[pos],
            "ret_net": metrics["ret_net"].iloc[pos],
            "mae_norm": metrics["mae_norm"].iloc[pos],
            "mfe_norm": metrics["mfe_norm"].iloc[pos],
            "mfe_mae": mfe_mae.iloc[pos],
            "barrier": metrics["barrier"].iloc[pos],
            "is_timeout": bool(metrics["is_timeout"].iloc[pos]),
            "bars_to_mfe": metrics["bars_to_mfe"].iloc[pos],
            "bars_policy": metrics["bars_policy"].iloc[pos],
        }
        for col in REGIME_COLUMNS:
            if col in valid.columns:
                row[col] = valid[col].iloc[pos]
        for col in PROFILE_CONTEXT_COLUMNS:
            if col in source_context.columns:
                row[col] = source_context[col].iloc[pos]
        for name, mask in source_masks.items():
            row[f"mask_{name}"] = bool(mask.iloc[pos])
        rows.append(row)
    return rows


def _profile_rows(
    *,
    valid: pd.DataFrame,
    metrics: pd.DataFrame,
    source_context: pd.DataFrame,
    source_masks: dict[str, pd.Series],
    candidate: str,
    month: str,
    top_frac: float,
    groups: dict[str, pd.Series],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    strict_clean = _strict_clean(metrics)
    dirty = _dirty(metrics)
    for group_name, mask in groups.items():
        mask = mask.fillna(False).astype(bool)
        selected_metrics = metrics.loc[mask]
        row: dict[str, Any] = {
            "candidate": candidate,
            "month": month,
            "top_frac": float(top_frac),
            "group": group_name,
            "rows": int(mask.sum()),
            "mean_return_net": _safe_mean(selected_metrics["ret_net"]) if int(mask.sum()) else float("nan"),
            "mean_u": _safe_mean(selected_metrics["u_policy_net"]) if int(mask.sum()) else float("nan"),
            "bad_mae_1r_rate": _safe_mean((selected_metrics["mae_norm"] >= 1.0).astype(float))
            if int(mask.sum())
            else float("nan"),
            "timeout_rate": _safe_mean(selected_metrics["is_timeout"].astype(float)) if int(mask.sum()) else float("nan"),
            "p90_mae_norm": _safe_quantile(selected_metrics["mae_norm"], 0.90) if int(mask.sum()) else float("nan"),
            "strict_clean_rate": _safe_mean(strict_clean.loc[mask].astype(float)) if int(mask.sum()) else float("nan"),
            "dirty_rate": _safe_mean(dirty.loc[mask].astype(float)) if int(mask.sum()) else float("nan"),
        }
        for col in PROFILE_CONTEXT_COLUMNS:
            if col in source_context.columns:
                row[f"mean_{col}"] = _safe_mean(source_context.loc[mask, col])
        for source_name, source_mask in source_masks.items():
            row[f"rate_{source_name}"] = _safe_mean(source_mask.loc[mask].astype(float)) if int(mask.sum()) else float("nan")
        rows.append(row)
    return rows


def _regime_rows(
    *,
    valid: pd.DataFrame,
    candidate: str,
    month: str,
    top_frac: float,
    groups: dict[str, pd.Series],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_name, mask in groups.items():
        mask = mask.fillna(False).astype(bool)
        total = int(mask.sum())
        if total <= 0:
            continue
        for col in REGIME_COLUMNS:
            if col not in valid.columns:
                continue
            values = valid.loc[mask, col].astype("string").fillna("NA")
            counts = values.value_counts(dropna=False)
            for value, count in counts.items():
                rows.append(
                    {
                        "candidate": candidate,
                        "month": month,
                        "top_frac": float(top_frac),
                        "group": group_name,
                        "regime_column": col,
                        "regime_value": str(value),
                        "rows": int(count),
                        "share": float(count / total),
                    }
                )
    return rows


def _write_report(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    contrast: pd.DataFrame,
    family: pd.DataFrame,
    profile: pd.DataFrame,
    regime: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "pathgrid_stage108_recoverability.md"
    summary_cols = [
        "month",
        "label_arm",
        "top_frac",
        "match_mode",
        "oracle_recovery_rate",
        "missed_clean_rows",
        "dirty_false_positive_rows",
        "matched_missed_clean_rows",
        "matched_dirty_false_positive_rows",
        "matched_bucket_count",
        "oracle_mean_return_net",
        "oracle_bad_mae_1r_rate",
        "oracle_timeout_rate",
        "proxy_mean_return_net",
        "proxy_bad_mae_1r_rate",
        "proxy_timeout_rate",
        "top_feature",
        "top_feature_family",
        "top_feature_best_auc",
        "top_feature_direction",
    ]
    contrast_cols = [
        "month",
        "label_arm",
        "top_frac",
        "match_mode",
        "feature",
        "feature_family",
        "is_proxy_feature",
        "best_auc",
        "best_direction",
        "bucket_equal_weight_gap_mean",
        "bucket_gap_positive_rate",
        "valid_utility_ic",
        "valid_label_ic",
    ]
    family_cols = [
        "feature_family",
        "rows",
        "months",
        "labels",
        "mean_best_auc",
        "max_best_auc",
        "mean_abs_bucket_gap",
        "proxy_feature_share",
        "top_features",
    ]
    profile_cols = [
        "candidate",
        "month",
        "top_frac",
        "group",
        "rows",
        "mean_return_net",
        "bad_mae_1r_rate",
        "timeout_rate",
        "p90_mae_norm",
        "mean_source_loud_intensity",
        "mean_source_event_quality",
        "mean_source_dual_prior_quality",
        "rate_loud_event",
        "rate_confirmed_lowbarrier_quality",
        "rate_rebound_mid",
    ]
    lines = [
        "# Stage108 Path-Grid Recoverability Diagnostic",
        "",
        "Scope: no model training. Reconstructs hard-binary decisive path-grid labels and economic-IC proxy scores, then compares clean oracle rows against dirty proxy false positives.",
        "",
        f"Stage dir: `{manifest['stage_dir']}`",
        f"Candidates: `{', '.join(manifest['candidates'])}`",
        f"Months: `{', '.join(manifest['months'])}`",
        f"Top fractions: `{manifest['top_fracs']}`",
        f"Match modes: `{manifest['match_modes']}`",
        f"Feature count: `{manifest['feature_count']}`",
        "",
        "## Confusion Summary",
        "",
        _table(summary.sort_values(["month", "label_arm", "top_frac", "match_mode"]), summary_cols, limit=120),
        "",
        "## Strongest Matched Separators",
        "",
        _table(
            contrast.sort_values(["best_auc", "abs_bucket_gap"], ascending=[False, False]) if not contrast.empty else contrast,
            contrast_cols,
            limit=80,
        ),
        "",
        "## Repeated Feature Families",
        "",
        _table(family, family_cols, limit=40),
        "",
        "## Source Profile",
        "",
        _table(profile.sort_values(["candidate", "month", "top_frac", "group"]), profile_cols, limit=160),
        "",
        "## Outputs",
        "",
        f"- Summary: `{manifest['outputs']['summary']}`",
        f"- Feature contrast: `{manifest['outputs']['feature_contrast']}`",
        f"- Family summary: `{manifest['outputs']['family_summary']}`",
        f"- Source profile: `{manifest['outputs']['source_profile']}`",
        f"- Regime profile: `{manifest['outputs']['regime_profile']}`",
        f"- Selected ledger: `{manifest['outputs']['selected_ledger']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_diagnostic(
    *,
    stage_dir: Path,
    output_dir: Path,
    candidates: list[str],
    max_candidates: int,
    months: list[str],
    top_fracs: list[float],
    match_modes: list[str],
    profile_sources: list[str],
    min_class_rows: int,
    strong_auc_threshold: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))
    candidate_summary = pd.read_csv(stage_dir / "path_aware_label_target_candidate_summary.csv")
    selected_candidates = _choose_candidates(candidate_summary, candidates, max_candidates=max_candidates)
    spec_rows = candidate_summary.set_index("candidate").loc[selected_candidates].reset_index()
    specs = [_spec_from_row(row) for _, row in spec_rows.iterrows()]

    reports = stage_manifest.get("reports", {})
    frame, metrics, load_reports = _build_frame(
        labels_path=Path(stage_manifest["labels_path"]),
        feature_dir=Path(stage_manifest.get("feature_dir", DEFAULT_FEATURE_DIR)),
        feature_list_csv=Path(stage_manifest.get("feature_list_csv", DEFAULT_FEATURE_LIST_CSV)),
        max_feature_store_features=stage_manifest.get("max_feature_store_features"),
        include_causal_outcome_priors=bool(reports.get("causal_outcome_priors", {}).get("enabled", False)),
        include_causal_state_path_priors=bool(reports.get("causal_state_path_priors", {}).get("enabled", False)),
        include_event_confirmation_features=bool(reports.get("event_confirmation_features", {}).get("enabled", False)),
        include_adverse_path_composites=bool(reports.get("adverse_path_composites", {}).get("enabled", False)),
        prior_windows_days=[float(v) for v in stage_manifest.get("prior_windows_days", DEFAULT_PRIOR_WINDOWS_DAYS)],
        prior_embargo_hours=float(stage_manifest.get("prior_embargo_hours", 24.0)),
        state_path_prior_features=list(stage_manifest.get("state_path_prior_features", DEFAULT_STATE_PATH_PRIOR_FEATURES)),
        event_feature_store_features=list(stage_manifest.get("event_feature_store_features", DEFAULT_EVENT_FEATURE_STORE_FEATURES)),
    )
    source_context = _source_context(frame)
    overlap = [col for col in source_context.columns if col in frame.columns]
    if overlap:
        frame = frame.drop(columns=overlap)
    frame = pd.concat([frame, source_context.astype(np.float32, copy=False)], axis=1).copy()
    source_masks_all = _build_sources(frame, source_context, run_gap_hours=6.0)
    available_profile_sources = [name for name in profile_sources if name in source_masks_all]
    source_masks = {
        name: source_masks_all[name].fillna(False).astype(bool).reindex(frame.index, fill_value=False)
        for name in available_profile_sources
    }

    features = _feature_columns(frame)
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    target_mode = stage_manifest.get("grid", {}).get("target_mode", "hard_binary")
    target_rank_weight = float(stage_manifest.get("grid", {}).get("target_rank_weight", 0.30))
    target_soft_power = float(stage_manifest.get("grid", {}).get("target_soft_power", 1.0))

    summary_rows: list[dict[str, Any]] = []
    contrast_frames: list[pd.DataFrame] = []
    profile_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []

    for spec in specs:
        target = _target_for_spec(
            metrics.reset_index(drop=True),
            spec,
            frame["__ts__"].reset_index(drop=True),
            target_mode=target_mode,
            target_rank_weight=target_rank_weight,
            target_soft_power=target_soft_power,
        )
        target.index = frame.index
        for month in months:
            train_mask = month_period.lt(str(month))
            valid_mask = month_period.eq(str(month))
            if int(train_mask.sum()) < int(stage_manifest.get("min_train_rows", 500)) or int(valid_mask.sum()) < int(
                stage_manifest.get("min_valid_rows", 100)
            ):
                continue
            train = frame.loc[train_mask].copy()
            valid_source = frame.loc[valid_mask].copy()
            valid = valid_source.reset_index(drop=True)
            valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
            valid_target = target.loc[valid_mask].copy().reset_index(drop=True)
            valid_source_context = source_context.loc[valid_mask].copy().reset_index(drop=True)
            valid_source_masks = {
                name: mask.loc[valid_mask].copy().reset_index(drop=True) for name, mask in source_masks.items()
            }
            score, diag = _proxy_score(
                train=train,
                valid=valid_source,
                features=features,
                target_train=target.loc[train_mask, "target_soft"],
                metrics_train=metrics.loc[train_mask].copy(),
                top_k=int(stage_manifest.get("proxy_top_k", 4)),
                proxy_objective=str(stage_manifest.get("proxy_objective", "economic_ic")),
                min_target_ic=float(stage_manifest.get("proxy_min_target_ic", 0.0)),
                min_utility_ic=float(stage_manifest.get("proxy_min_utility_ic", 0.0)),
                max_bad_mae_ic=float(stage_manifest.get("proxy_max_bad_mae_ic", 0.0)),
                max_wide_ic=float(stage_manifest.get("proxy_max_wide_ic", 0.0)),
                max_timeout_ic=float(stage_manifest.get("proxy_max_timeout_ic", 0.0)),
                utility_weight=float(stage_manifest.get("proxy_utility_weight", 1.0)),
                bad_mae_weight=float(stage_manifest.get("proxy_bad_mae_weight", 1.0)),
                wide_weight=float(stage_manifest.get("proxy_wide_weight", 0.5)),
                timeout_weight=float(stage_manifest.get("proxy_timeout_weight", 0.5)),
            )
            score = score.reset_index(drop=True)
            proxy_features = list(diag.get("proxy_features", []))
            strict_clean = _strict_clean(valid_metrics).reset_index(drop=True)
            dirty = _dirty(valid_metrics).reset_index(drop=True)

            for top_frac in top_fracs:
                oracle_mask = _top_mask(valid_target["target_soft"], float(top_frac)).reset_index(drop=True)
                proxy_mask = _top_mask(score, float(top_frac)).reset_index(drop=True)
                missed_clean_mask = oracle_mask & ~proxy_mask & strict_clean
                dirty_false_positive_mask = proxy_mask & ~oracle_mask & dirty
                recovered_mask = oracle_mask & proxy_mask
                group_masks = {
                    "oracle_top": oracle_mask,
                    "proxy_top": proxy_mask,
                    "recovered": recovered_mask,
                    "missed_clean_oracle": missed_clean_mask,
                    "dirty_proxy_false_positive": dirty_false_positive_mask,
                }
                ledger_rows.extend(
                    _selected_ledger_rows(
                        valid=valid,
                        metrics=valid_metrics,
                        target=valid_target,
                        score=score,
                        source_context=valid_source_context,
                        source_masks=valid_source_masks,
                        candidate=spec.name,
                        month=str(month),
                        top_frac=float(top_frac),
                        oracle_mask=oracle_mask,
                        proxy_mask=proxy_mask,
                        strict_clean=strict_clean,
                        dirty=dirty,
                    )
                )
                profile_rows.extend(
                    _profile_rows(
                        valid=valid,
                        metrics=valid_metrics,
                        source_context=valid_source_context,
                        source_masks=valid_source_masks,
                        candidate=spec.name,
                        month=str(month),
                        top_frac=float(top_frac),
                        groups=group_masks,
                    )
                )
                regime_rows.extend(
                    _regime_rows(
                        valid=valid,
                        candidate=spec.name,
                        month=str(month),
                        top_frac=float(top_frac),
                        groups=group_masks,
                    )
                )

                for match_mode in match_modes:
                    bucket = _bucket_key(valid, valid_metrics, match_mode).reset_index(drop=True)
                    clean_matched, dirty_matched, eligible_candidate, eligible_buckets = _eligible_masks(
                        bucket=bucket,
                        missed_clean_mask=missed_clean_mask,
                        dirty_false_positive_mask=dirty_false_positive_mask,
                    )
                    contrast = _feature_contrast(
                        train=train,
                        valid=valid,
                        valid_metrics=valid_metrics,
                        features=features,
                        target_train=target.loc[train_mask, "target_soft"],
                        target_valid=valid_target["target_soft"],
                        score=score,
                        bucket=bucket,
                        clean_mask=clean_matched,
                        dirty_mask=dirty_matched,
                        eligible_candidate_mask=eligible_candidate,
                        proxy_features=proxy_features,
                        min_class_rows=int(min_class_rows),
                        source="all",
                        month=str(month),
                        label_arm=spec.name,
                        top_frac=float(top_frac),
                        match_mode=match_mode,
                    )
                    if not contrast.empty:
                        contrast_frames.append(contrast)
                    summary_rows.append(
                        _summary_row(
                            valid=valid,
                            valid_metrics=valid_metrics,
                            target_valid=valid_target,
                            score=score,
                            oracle_mask=oracle_mask,
                            proxy_mask=proxy_mask,
                            missed_clean_mask=missed_clean_mask,
                            dirty_false_positive_mask=dirty_false_positive_mask,
                            clean_matched_mask=clean_matched,
                            dirty_matched_mask=dirty_matched,
                            eligible_buckets=eligible_buckets,
                            source="all",
                            month=str(month),
                            label_arm=spec.name,
                            top_frac=float(top_frac),
                            match_mode=match_mode,
                            proxy_features=proxy_features,
                            top_contrast=contrast,
                        )
                    )

    summary = pd.DataFrame(summary_rows)
    contrast = pd.concat(contrast_frames, ignore_index=True) if contrast_frames else pd.DataFrame()
    family = _family_summary(contrast, strong_auc_threshold)
    profile = pd.DataFrame(profile_rows)
    regime = pd.DataFrame(regime_rows)
    ledger = pd.DataFrame(ledger_rows)

    paths = {
        "summary": output_dir / "pathgrid_recoverability_summary.csv",
        "feature_contrast": output_dir / "pathgrid_recoverability_feature_contrast.csv",
        "family_summary": output_dir / "pathgrid_recoverability_family_summary.csv",
        "source_profile": output_dir / "pathgrid_recoverability_source_profile.csv",
        "regime_profile": output_dir / "pathgrid_recoverability_regime_profile.csv",
        "selected_ledger": output_dir / "pathgrid_recoverability_selected_ledger.csv",
        "manifest": output_dir / "manifest.json",
    }
    summary.to_csv(paths["summary"], index=False)
    contrast.to_csv(paths["feature_contrast"], index=False)
    family.to_csv(paths["family_summary"], index=False)
    profile.to_csv(paths["source_profile"], index=False)
    regime.to_csv(paths["regime_profile"], index=False)
    ledger.to_csv(paths["selected_ledger"], index=False)

    manifest = {
        "scope": "pathgrid_stage108_recoverability_diagnostic",
        "stage_dir": str(stage_dir),
        "output_dir": str(output_dir),
        "labels_path": stage_manifest["labels_path"],
        "rows": int(len(frame)),
        "feature_count": int(len(features)),
        "candidates": [spec.name for spec in specs],
        "candidate_specs": [asdict(spec) for spec in specs],
        "months": [str(v) for v in months],
        "top_fracs": [float(v) for v in top_fracs],
        "match_modes": [str(v) for v in match_modes],
        "profile_sources": list(source_masks),
        "min_class_rows": int(min_class_rows),
        "strong_auc_threshold": float(strong_auc_threshold),
        "stage_proxy_objective": stage_manifest.get("proxy_objective"),
        "stage_proxy_top_k": stage_manifest.get("proxy_top_k"),
        "stage_target_mode": target_mode,
        "outputs": {key: str(value) for key, value in paths.items()},
        "load_reports": load_reports,
    }
    markdown = _write_report(
        output_dir=output_dir,
        summary=summary,
        contrast=contrast,
        family=family,
        profile=profile,
        regime=regime,
        manifest={**manifest, "outputs": {**manifest["outputs"], "markdown": str(output_dir / "pathgrid_stage108_recoverability.md")}},
    )
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidates", type=lambda value: _parse_csv(value), default="")
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--months", type=lambda value: _parse_csv(value, DEFAULT_MONTHS), default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--top-fracs", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument(
        "--match-modes",
        type=lambda value: _parse_csv(value, DEFAULT_MATCH_MODES),
        default=",".join(DEFAULT_MATCH_MODES),
    )
    parser.add_argument(
        "--profile-sources",
        type=lambda value: _parse_csv(value, DEFAULT_PROFILE_SOURCES),
        default=",".join(DEFAULT_PROFILE_SOURCES),
    )
    parser.add_argument("--min-class-rows", type=int, default=4)
    parser.add_argument("--strong-auc-threshold", type=float, default=0.65)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_diagnostic(
        stage_dir=args.stage_dir,
        output_dir=args.output_dir,
        candidates=list(args.candidates),
        max_candidates=int(args.max_candidates),
        months=list(args.months),
        top_fracs=[float(v) for v in args.top_fracs],
        match_modes=list(args.match_modes),
        profile_sources=list(args.profile_sources),
        min_class_rows=int(args.min_class_rows),
        strong_auc_threshold=float(args.strong_auc_threshold),
    )
    print(json.dumps(_json_safe({k: v for k, v in manifest.items() if k != "load_reports"}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
