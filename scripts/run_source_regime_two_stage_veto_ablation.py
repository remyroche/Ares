#!/usr/bin/env python3
"""Source/regime-conditioned two-stage veto ablation.

Proxy-only diagnostic: keep the Stage 96 utility target and Stage 97 risk-veto
mechanics, but fit/evaluate the proxies inside deterministic decision-time
source and regime masks. No model training, Optuna, or policy optimisation.
"""

from __future__ import annotations

import argparse
import json
import re
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
)
from scripts.diagnose_source_conditioned_path_observability import (  # noqa: E402
    _fit_holdout_summary,
)
from scripts.report_label_proxy_quality_with_economic_limits import (  # noqa: E402
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_PROXY_TOP_K,
    _score_period,
    _score_proxy,
    _slice_week_positions,
    _table,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    _feature_columns,
    _json_safe,
    _safe_mean,
    _spearman,
)
from scripts.run_path_aware_label_target_grid import _target_for_spec  # noqa: E402
from scripts.run_path_aware_two_stage_veto_ablation import (  # noqa: E402
    _build_selector_scores,
    _risk_targets,
    _utility_spec,
)
from scripts.run_soft_label_candidate_source_ablation import (  # noqa: E402
    _build_sources,
    _causal_time_edge_prior_features,
    _source_context,
    _source_summary,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/source_regime_two_stage_veto_stage98_v1")
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_FIT_MONTHS = ("2026-04", "2026-05")
DEFAULT_HOLDOUT_MONTH = "2026-06"
DEFAULT_TOP_FRACS = (0.005, 0.01, 0.02, 0.03)
DEFAULT_RISK_KEEP_FRACS = (0.20, 0.30, 0.50)
DEFAULT_RISK_PENALTIES = (0.25, 0.50, 1.00)
DEFAULT_SOURCES = (
    "all",
    "quiet_mid",
    "quiet_quality",
    "loud_event",
    "any_event_quality",
    "confirmed_lowbarrier_family",
    "confirmed_event_quality_family",
    "time_edge_prior_confirmed_family",
    "dual_prior_confirmed_lowbarrier_quality",
    "run_entry_quiet_mid",
    "run_entry_confirmed_event_quality_family",
    "run_entry_time_edge_prior_confirmed_family",
)
DEFAULT_REGIME_COLS = (
    "__regime_trend_12h__",
    "__regime_volume_12h__",
    "__regime_trend_48h__",
    "__regime_volume_48h__",
)
DEFAULT_STATE_COLS = (
    "atr_compression_ratio",
    "xs_rank_range_24h_pct",
    "source_loud_intensity",
    "source_event_quality",
    "source_rebound_context",
    "source_oi_location",
    "loc_prev_week_range_pos_24",
    "distance_to_support_daily_vwap_atr",
    "distance_to_resistance_daily_vwap_atr",
)
DEFAULT_STATE_BINS = ("low20", "high80")
DEFAULT_INTERSECTION_SOURCES = (
    "quiet_mid",
    "confirmed_event_quality_family",
    "time_edge_prior_confirmed_family",
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


def _clean_mask_part(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9_.=+-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "value"


def _regime_masks(
    frame: pd.DataFrame,
    *,
    regime_cols: list[str],
    max_values_per_col: int,
    min_total_rows: int,
) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for col in regime_cols:
        if col not in frame.columns:
            continue
        values = frame[col].astype(str).fillna("nan")
        counts = values.value_counts(dropna=False).head(int(max_values_per_col))
        for value, count in counts.items():
            if int(count) < int(min_total_rows):
                continue
            clean_col = col.strip("_")
            clean_value = str(value).replace("/", "_").replace(" ", "_")
            out[f"regime::{clean_col}={clean_value}"] = values.eq(str(value)).reindex(frame.index, fill_value=False)
    return out


def _source_regime_intersections(
    frame: pd.DataFrame,
    *,
    sources: dict[str, pd.Series],
    source_names: list[str],
    regime_cols: list[str],
    max_values_per_col: int,
    min_total_rows: int,
) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for source_name in source_names:
        if source_name not in sources:
            continue
        source_mask = sources[source_name].fillna(False).astype(bool)
        for col in regime_cols:
            if col not in frame.columns:
                continue
            values = frame[col].astype(str).fillna("nan")
            counts = values[source_mask].value_counts(dropna=False).head(int(max_values_per_col))
            for value, count in counts.items():
                if int(count) < int(min_total_rows):
                    continue
                clean_col = col.strip("_")
                clean_value = str(value).replace("/", "_").replace(" ", "_")
                out[f"{source_name}::regime::{clean_col}={clean_value}"] = (
                    source_mask & values.eq(str(value))
                ).reindex(frame.index, fill_value=False)
    return out


def _state_thresholds(
    frame: pd.DataFrame,
    *,
    state_cols: list[str],
    reference_mask: pd.Series,
    quantiles: tuple[float, ...] = (0.20, 0.40, 0.60, 0.80),
) -> dict[str, dict[float, float]]:
    reference_mask = reference_mask.reindex(frame.index, fill_value=False).astype(bool)
    out: dict[str, dict[float, float]] = {}
    for col in state_cols:
        if col not in frame.columns:
            continue
        values = _safe_numeric(frame.loc[reference_mask, col]).replace([np.inf, -np.inf], np.nan).dropna()
        if int(values.nunique(dropna=True)) < 8:
            continue
        qs = values.quantile(list(quantiles)).to_dict()
        if any(not np.isfinite(float(value)) for value in qs.values()):
            continue
        out[col] = {float(q): float(value) for q, value in qs.items()}
    return out


def _mask_has_support(
    frame: pd.DataFrame,
    mask: pd.Series,
    *,
    min_total_rows: int,
    min_month_rows: int,
) -> bool:
    mask = mask.reindex(frame.index, fill_value=False).astype(bool)
    if int(mask.sum()) < int(min_total_rows):
        return False
    if int(min_month_rows) <= 0:
        return True
    month = frame["__ts__"].dt.to_period("M").astype(str)
    counts = month[mask].value_counts(dropna=False)
    months = month.dropna().unique().tolist()
    return all(int(counts.get(str(cur), 0)) >= int(min_month_rows) for cur in months)


def _state_mask_for_bin(values: pd.Series, thresholds: dict[float, float], state_bin: str) -> pd.Series | None:
    state_bin = str(state_bin).strip().lower()
    if state_bin == "low20":
        return values <= thresholds[0.20]
    if state_bin == "low40":
        return values <= thresholds[0.40]
    if state_bin == "mid40_60":
        return (values > thresholds[0.40]) & (values <= thresholds[0.60])
    if state_bin == "high60":
        return values >= thresholds[0.60]
    if state_bin == "high80":
        return values >= thresholds[0.80]
    raise ValueError(f"Unknown state bin: {state_bin}")


def _state_masks(
    frame: pd.DataFrame,
    *,
    state_cols: list[str],
    state_bins: list[str],
    reference_mask: pd.Series,
    min_total_rows: int,
    min_month_rows: int,
) -> dict[str, pd.Series]:
    thresholds_by_col = _state_thresholds(frame, state_cols=state_cols, reference_mask=reference_mask)
    out: dict[str, pd.Series] = {}
    for col, thresholds in thresholds_by_col.items():
        values = _safe_numeric(frame[col]).replace([np.inf, -np.inf], np.nan)
        clean_col = _clean_mask_part(col)
        for state_bin in state_bins:
            mask = _state_mask_for_bin(values, thresholds, state_bin)
            if mask is None:
                continue
            mask = mask.fillna(False).reindex(frame.index, fill_value=False).astype(bool)
            if not _mask_has_support(
                frame,
                mask,
                min_total_rows=min_total_rows,
                min_month_rows=min_month_rows,
            ):
                continue
            suffix = _clean_mask_part(state_bin)
            out[f"state::{clean_col}::{suffix}"] = mask
    return out


def _source_state_intersections(
    frame: pd.DataFrame,
    *,
    sources: dict[str, pd.Series],
    source_names: list[str],
    state_masks: dict[str, pd.Series],
    min_total_rows: int,
    min_month_rows: int,
) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for source_name in source_names:
        if source_name not in sources:
            continue
        source_mask = sources[source_name].fillna(False).astype(bool)
        for state_name, state_mask in state_masks.items():
            combined = (source_mask & state_mask.reindex(frame.index, fill_value=False).astype(bool)).reindex(
                frame.index, fill_value=False
            )
            if not _mask_has_support(
                frame,
                combined,
                min_total_rows=min_total_rows,
                min_month_rows=min_month_rows,
            ):
                continue
            out[f"{source_name}::{state_name}"] = combined
    return out


def _selected_sources(
    *,
    frame: pd.DataFrame,
    context: pd.DataFrame,
    requested_sources: list[str],
    regime_cols: list[str],
    include_regime_masks: bool,
    include_source_regime_intersections: bool,
    intersection_sources: list[str],
    state_cols: list[str],
    state_bins: list[str],
    state_reference_months: list[str],
    include_state_masks: bool,
    include_source_state_intersections: bool,
    state_intersection_sources: list[str],
    run_gap_hours: float,
    max_regime_values_per_col: int,
    min_mask_total_rows: int,
    min_state_month_rows: int,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    source_all = _build_sources(frame, context, run_gap_hours=run_gap_hours)
    missing = sorted(set(requested_sources).difference(source_all))
    if missing:
        raise ValueError(f"Unknown source(s): {missing}")
    masks = {name: source_all[name].reindex(frame.index, fill_value=False).astype(bool) for name in requested_sources}
    if include_regime_masks:
        masks.update(
            _regime_masks(
                frame,
                regime_cols=regime_cols,
                max_values_per_col=max_regime_values_per_col,
                min_total_rows=min_mask_total_rows,
            )
        )
    state_mask_map: dict[str, pd.Series] = {}
    if include_state_masks or include_source_state_intersections:
        month = frame["__ts__"].dt.to_period("M").astype(str)
        reference_mask = month.isin([str(v) for v in state_reference_months])
        state_mask_map = _state_masks(
            frame,
            state_cols=state_cols,
            state_bins=state_bins,
            reference_mask=reference_mask,
            min_total_rows=min_mask_total_rows,
            min_month_rows=min_state_month_rows,
        )
        if include_state_masks:
            masks.update(state_mask_map)
    if include_source_state_intersections:
        masks.update(
            _source_state_intersections(
                frame,
                sources=source_all,
                source_names=state_intersection_sources,
                state_masks=state_mask_map,
                min_total_rows=min_mask_total_rows,
                min_month_rows=min_state_month_rows,
            )
        )
    if include_source_regime_intersections:
        masks.update(
            _source_regime_intersections(
                frame,
                sources=source_all,
                source_names=intersection_sources,
                regime_cols=regime_cols,
                max_values_per_col=max_regime_values_per_col,
                min_total_rows=min_mask_total_rows,
            )
        )
    return masks, source_all


def _period_rows_for_month(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    utility_target: pd.DataFrame,
    risk_targets: dict[str, pd.Series],
    features: list[str],
    source_name: str,
    source_mask: pd.Series,
    month: str,
    top_fracs: list[float],
    proxy_top_k: int,
    risk_keep_fracs: list[float],
    risk_penalties: list[float],
    min_train_source_rows: int,
    min_valid_source_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    month_series = frame["__ts__"].dt.to_period("M").astype(str)
    source_mask = source_mask.reindex(frame.index, fill_value=False).astype(bool)
    train_mask = month_series.lt(str(month)) & source_mask
    valid_mask = month_series.eq(str(month)) & source_mask
    train_rows = int(train_mask.sum())
    valid_rows = int(valid_mask.sum())
    if train_rows < int(min_train_source_rows) or valid_rows < int(min_valid_source_rows):
        return [], []

    train = frame.loc[train_mask].copy()
    valid_raw = frame.loc[valid_mask].copy()
    valid = valid_raw.reset_index(drop=True)
    valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
    valid_target = utility_target.loc[valid_mask].copy().reset_index(drop=True)

    utility_score, utility_diag = _score_proxy(
        train=train,
        valid=valid_raw,
        features=features,
        y_train=utility_target.loc[train_mask, "target_soft"],
        proxy_top_k=int(proxy_top_k),
    )
    utility_score = utility_score.reset_index(drop=True)

    diagnostics: list[dict[str, Any]] = [
        {
            "source": source_name,
            "month": str(month),
            "component": "utility",
            "train_rows": train_rows,
            "valid_rows": valid_rows,
            "proxy_ic_u": _spearman(utility_score, valid_metrics["u_policy_net"]),
            "proxy_ic_bad_mae": _spearman(utility_score, (valid_metrics["mae_norm"] >= 1.0).astype(float)),
            "proxy_ic_tail_mae_4r": _spearman(utility_score, (valid_metrics["mae_norm"] >= 4.0).astype(float)),
            "proxy_ic_timeout": _spearman(utility_score, valid_metrics["is_timeout"].astype(float)),
            "proxy_ic_target": _spearman(utility_score, valid_target["target_soft"]),
            "proxy_features": ",".join(utility_diag.get("proxy_features", [])),
            "proxy_top_abs_ic": utility_diag.get("proxy_top_abs_ic"),
            "proxy_mean_top_abs_ic": utility_diag.get("proxy_mean_top_abs_ic"),
        }
    ]

    valid_risk_scores: dict[str, pd.Series] = {}
    risk_feature_names: dict[str, str] = {}
    for risk_arm, risk_target in risk_targets.items():
        score, diag = _score_proxy(
            train=train,
            valid=valid_raw,
            features=features,
            y_train=risk_target.loc[train_mask],
            proxy_top_k=int(proxy_top_k),
        )
        score = score.reset_index(drop=True)
        valid_risk_scores[risk_arm] = score
        risk_feature_names[risk_arm] = ",".join(diag.get("proxy_features", []))
        diagnostics.append(
            {
                "source": source_name,
                "month": str(month),
                "component": risk_arm,
                "train_rows": train_rows,
                "valid_rows": valid_rows,
                "proxy_ic_u": _spearman(score, valid_metrics["u_policy_net"]),
                "proxy_ic_bad_mae": _spearman(score, (valid_metrics["mae_norm"] >= 1.0).astype(float)),
                "proxy_ic_tail_mae_4r": _spearman(score, (valid_metrics["mae_norm"] >= 4.0).astype(float)),
                "proxy_ic_timeout": _spearman(score, valid_metrics["is_timeout"].astype(float)),
                "proxy_ic_target": _spearman(score, risk_target.loc[valid_mask].reset_index(drop=True)),
                "proxy_features": risk_feature_names[risk_arm],
                "proxy_top_abs_ic": diag.get("proxy_top_abs_ic"),
                "proxy_mean_top_abs_ic": diag.get("proxy_mean_top_abs_ic"),
            }
        )

    selector_specs = [
        {
            "selector": "oracle_utility_sort",
            "economic_arm": "oracle_utility",
            "score": valid_target["target_soft"],
            "risk_arm": "",
            "risk_keep_frac": float("nan"),
            "risk_penalty": float("nan"),
            "mode": "oracle",
        }
    ]
    selector_specs.extend(
        _build_selector_scores(
            valid=valid,
            utility_score=utility_score,
            risk_scores=valid_risk_scores,
            risk_keep_fracs=risk_keep_fracs,
            risk_penalties=risk_penalties,
        )
    )

    period_rows: list[dict[str, Any]] = []
    period_slices = [("month", str(month), np.arange(len(valid), dtype=np.int64))]
    period_slices.extend(("week", week, pos) for week, pos in _slice_week_positions(valid))
    for selector_spec in selector_specs:
        score = _safe_numeric(selector_spec["score"]).reset_index(drop=True)
        risk_arm = str(selector_spec["risk_arm"])
        econ_features = risk_feature_names.get(risk_arm, "")
        for period_type, period_name, pos in period_slices:
            for frac in top_fracs:
                row = _score_period(
                    frame=valid.iloc[pos].reset_index(drop=True),
                    metrics=valid_metrics.iloc[pos].reset_index(drop=True),
                    target=valid_target.iloc[pos].reset_index(drop=True),
                    score=score.iloc[pos].reset_index(drop=True),
                    period_type=period_type,
                    period=period_name,
                    month=str(month),
                    selector=str(selector_spec["selector"]),
                    label_arm="stage96_nearmiss_utility",
                    economic_arm=str(selector_spec["economic_arm"]),
                    top_frac=float(frac),
                    label_score=valid_target["target_soft"].iloc[pos].reset_index(drop=True),
                    economic_score=valid_risk_scores[risk_arm].iloc[pos].reset_index(drop=True)
                    if risk_arm in valid_risk_scores
                    else None,
                    economic_target=risk_targets[risk_arm].loc[valid_mask].reset_index(drop=True).iloc[pos].reset_index(drop=True)
                    if risk_arm in risk_targets
                    else None,
                    label_proxy_features=",".join(utility_diag.get("proxy_features", [])),
                    economic_proxy_features=econ_features,
                )
                row["source"] = source_name
                row["source_train_rows"] = train_rows
                row["source_valid_rows"] = valid_rows
                row["risk_arm"] = risk_arm
                row["risk_keep_frac"] = selector_spec["risk_keep_frac"]
                row["risk_penalty"] = selector_spec["risk_penalty"]
                row["selector_mode"] = selector_spec["mode"]
                period_rows.append(row)
    return period_rows, diagnostics


def _write_markdown(
    *,
    output_dir: Path,
    source_summary: pd.DataFrame,
    fit_holdout: pd.DataFrame,
    period_rows: pd.DataFrame,
    proxy_ic: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "source_regime_two_stage_veto_ablation.md"
    best = (
        fit_holdout.sort_values(
            [
                "trainworthy_pass",
                "holdout_economic_pass",
                "fit_economic_pass",
                "holdout_mean_return_net",
            ],
            ascending=[False, False, False, False],
        )
        if not fit_holdout.empty
        else fit_holdout
    )
    non_oracle = best[~best["selector"].astype(str).str.startswith("oracle_")].copy() if not best.empty else best
    june = (
        period_rows[
            period_rows["period_type"].astype(str).eq("month")
            & period_rows["month"].astype(str).eq(str(manifest["holdout_month"]))
        ].sort_values("mean_return_net", ascending=False)
        if not period_rows.empty
        else period_rows
    )
    lines = [
        "# Source/Regime Two-Stage Veto Ablation",
        "",
        "Scope: proxy-only label/execution diagnostic. Source and regime masks are deterministic decision-time masks; proxy scores are fitted on prior rows inside each mask.",
        "",
        f"Labels: `{manifest['labels_path']}`",
        f"Fit months: `{', '.join(manifest['fit_months'])}`",
        f"Holdout month: `{manifest['holdout_month']}`",
        f"Feature count: `{manifest['feature_count']}`",
        f"Mask count: `{manifest['mask_count']}`",
        f"Proxy top-k: `{manifest['proxy_top_k']}`",
        f"State masks: `{manifest.get('include_state_masks', False)}`",
        f"Source-state intersections: `{manifest.get('include_source_state_intersections', False)}`",
        f"State threshold months: `{', '.join(manifest.get('state_reference_months', []))}`",
        "",
        "## Counts",
        "",
        f"- Fit/holdout rows: `{len(fit_holdout)}`",
        f"- Non-oracle train-worthy rows: `{int(non_oracle['trainworthy_pass'].sum()) if not non_oracle.empty else 0}`",
        f"- Non-oracle fit economic pass: `{int(non_oracle['fit_economic_pass'].sum()) if not non_oracle.empty else 0}`",
        f"- Non-oracle holdout economic pass: `{int(non_oracle['holdout_economic_pass'].sum()) if not non_oracle.empty else 0}`",
        f"- Oracle train-worthy rows: `{int(best[best['selector'].astype(str).str.startswith('oracle_')]['trainworthy_pass'].sum()) if not best.empty else 0}`",
        "",
        "## Best Non-Oracle Rows",
        "",
        _table(
            non_oracle,
            [
                "trainworthy_pass",
                "fit_economic_pass",
                "holdout_economic_pass",
                "fit_sign_pass",
                "holdout_sign_pass",
                "source",
                "selector",
                "economic_arm",
                "top_frac",
                "fit_mean_return_net",
                "holdout_mean_return_net",
                "fit_bad_mae_1r_rate",
                "holdout_bad_mae_1r_rate",
                "fit_p90_mae_norm",
                "holdout_p90_mae_norm",
                "fit_timeout_rate",
                "holdout_timeout_rate",
                "fit_score_ic_u",
                "holdout_score_ic_u",
                "fit_selected_rows",
                "holdout_selected_rows",
            ],
            limit=100,
        ),
        "",
        "## June Month Selections",
        "",
        _table(
            june,
            [
                "source",
                "selector",
                "economic_arm",
                "top_frac",
                "selected_rows",
                "mean_return_net",
                "bad_mae_1r_rate",
                "p90_mae_norm",
                "timeout_rate",
                "wide_barrier_25bps_rate",
                "score_ic_u",
                "score_ic_economic",
            ],
            limit=100,
        ),
        "",
        "## June Proxy IC",
        "",
        _table(
            proxy_ic[proxy_ic["month"].astype(str).eq(str(manifest["holdout_month"]))].sort_values(
                ["source", "proxy_ic_u"],
                ascending=[True, False],
            )
            if not proxy_ic.empty
            else proxy_ic,
            [
                "source",
                "component",
                "valid_rows",
                "proxy_ic_target",
                "proxy_ic_u",
                "proxy_ic_bad_mae",
                "proxy_ic_tail_mae_4r",
                "proxy_ic_timeout",
                "proxy_features",
            ],
            limit=120,
        ),
        "",
        "## Mask Coverage",
        "",
        _table(
            source_summary,
            [
                "source",
                "rows",
                "rows_2026_04",
                "rows_2026_05",
                "rows_2026_06",
                "mean_u",
                "bad_mae_1r_rate",
                "timeout_rate",
                "mean_event_quality",
                "mean_time_edge_prior_quality",
                "mean_adverse_prior_safety",
            ],
            limit=120,
        ),
        "",
        "## Outputs",
        "",
        f"- Source summary: `{manifest['outputs']['source_summary']}`",
        f"- Fit/holdout: `{manifest['outputs']['fit_holdout']}`",
        f"- Period rows: `{manifest['outputs']['period_rows']}`",
        f"- Proxy IC: `{manifest['outputs']['proxy_ic']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    max_feature_columns: int | None,
    months: list[str],
    fit_months: list[str],
    holdout_month: str,
    top_fracs: list[float],
    proxy_top_k: int,
    min_train_source_rows: int,
    min_valid_source_rows: int,
    min_week_rows: int,
    max_timeout_rate: float | None,
    risk_keep_fracs: list[float],
    risk_penalties: list[float],
    sources: list[str],
    regime_cols: list[str],
    include_regime_masks: bool,
    include_source_regime_intersections: bool,
    intersection_sources: list[str],
    state_cols: list[str],
    state_bins: list[str],
    state_reference_months: list[str] | None,
    include_state_masks: bool,
    include_source_state_intersections: bool,
    state_intersection_sources: list[str],
    max_regime_values_per_col: int,
    min_mask_total_rows: int,
    min_state_month_rows: int,
    run_gap_hours: float,
    include_causal_outcome_priors: bool,
    include_causal_state_path_priors: bool,
    include_causal_time_edge_priors: bool,
    include_event_confirmation_features: bool,
    include_adverse_path_composites: bool,
    prior_windows_days: list[float],
    prior_embargo_hours: float,
    state_path_prior_features: list[str],
    event_feature_store_features: list[str],
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
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    reports["causal_time_edge_priors"] = {"enabled": False}
    if include_causal_time_edge_priors:
        time_edge, reports["causal_time_edge_priors"] = _causal_time_edge_prior_features(
            frame,
            metrics,
            windows_days=prior_windows_days,
            embargo_hours=prior_embargo_hours,
        )
        frame = pd.concat([frame, time_edge.astype(np.float32, copy=False)], axis=1).copy()

    context = _source_context(frame)
    frame = pd.concat([frame, context.astype(np.float32, copy=False)], axis=1).copy()
    mask_map, source_all = _selected_sources(
        frame=frame,
        context=context,
        requested_sources=sources,
        regime_cols=regime_cols,
        include_regime_masks=include_regime_masks,
        include_source_regime_intersections=include_source_regime_intersections,
        intersection_sources=intersection_sources,
        state_cols=state_cols,
        state_bins=state_bins,
        state_reference_months=state_reference_months if state_reference_months is not None else fit_months,
        include_state_masks=include_state_masks,
        include_source_state_intersections=include_source_state_intersections,
        state_intersection_sources=state_intersection_sources,
        run_gap_hours=run_gap_hours,
        max_regime_values_per_col=max_regime_values_per_col,
        min_mask_total_rows=min_mask_total_rows,
        min_state_month_rows=min_state_month_rows,
    )

    features = _feature_columns(frame)
    if max_feature_columns is not None and int(max_feature_columns) > 0:
        features = features[: int(max_feature_columns)]
    utility_target = _target_for_spec(metrics.reset_index(drop=True), _utility_spec(), frame["__ts__"].reset_index(drop=True))
    utility_target.index = frame.index
    risk_targets = _risk_targets(metrics.reset_index(drop=True))
    for target in risk_targets.values():
        target.index = frame.index

    source_summary = _source_summary(frame=frame, metrics=metrics, context=context, sources=mask_map)
    period_rows: list[dict[str, Any]] = []
    proxy_ic_rows: list[dict[str, Any]] = []
    for source_name, source_mask in mask_map.items():
        for month in months:
            cur_rows, cur_diag = _period_rows_for_month(
                frame=frame,
                metrics=metrics,
                utility_target=utility_target,
                risk_targets=risk_targets,
                features=features,
                source_name=source_name,
                source_mask=source_mask,
                month=str(month),
                top_fracs=top_fracs,
                proxy_top_k=int(proxy_top_k),
                risk_keep_fracs=risk_keep_fracs,
                risk_penalties=risk_penalties,
                min_train_source_rows=int(min_train_source_rows),
                min_valid_source_rows=int(min_valid_source_rows),
            )
            period_rows.extend(cur_rows)
            proxy_ic_rows.extend(cur_diag)
        print(json.dumps({"source": source_name, "period_rows_so_far": len(period_rows)}, sort_keys=True))

    period_frame = pd.DataFrame(period_rows)
    proxy_ic = pd.DataFrame(proxy_ic_rows)
    fit_holdout = _fit_holdout_summary(
        period_frame,
        fit_months=fit_months,
        holdout_month=holdout_month,
        min_week_rows=min_week_rows,
        max_timeout_rate=max_timeout_rate,
    )

    paths = {
        "source_summary": output_dir / "source_regime_two_stage_source_summary.csv",
        "fit_holdout": output_dir / "source_regime_two_stage_fit_holdout.csv",
        "period_rows": output_dir / "source_regime_two_stage_period_rows.csv",
        "proxy_ic": output_dir / "source_regime_two_stage_proxy_ic.csv",
        "manifest": output_dir / "manifest.json",
    }
    source_summary.to_csv(paths["source_summary"], index=False)
    fit_holdout.to_csv(paths["fit_holdout"], index=False)
    period_frame.to_csv(paths["period_rows"], index=False)
    proxy_ic.to_csv(paths["proxy_ic"], index=False)

    manifest = {
        "scope": "source_regime_two_stage_veto_ablation",
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
        "mask_count": int(len(mask_map)),
        "sources_requested": list(sources),
        "regime_cols": list(regime_cols),
        "include_regime_masks": bool(include_regime_masks),
        "include_source_regime_intersections": bool(include_source_regime_intersections),
        "intersection_sources": list(intersection_sources),
        "state_cols": list(state_cols),
        "state_bins": list(state_bins),
        "state_reference_months": [str(v) for v in (state_reference_months if state_reference_months is not None else fit_months)],
        "include_state_masks": bool(include_state_masks),
        "include_source_state_intersections": bool(include_source_state_intersections),
        "state_intersection_sources": list(state_intersection_sources),
        "min_state_month_rows": int(min_state_month_rows),
        "months": [str(v) for v in months],
        "fit_months": [str(v) for v in fit_months],
        "holdout_month": str(holdout_month),
        "top_fracs": [float(v) for v in top_fracs],
        "proxy_top_k": int(proxy_top_k),
        "min_train_source_rows": int(min_train_source_rows),
        "min_valid_source_rows": int(min_valid_source_rows),
        "min_week_rows": int(min_week_rows),
        "max_timeout_rate": max_timeout_rate,
        "risk_keep_fracs": [float(v) for v in risk_keep_fracs],
        "risk_penalties": [float(v) for v in risk_penalties],
        "risk_arms": list(risk_targets.keys()),
        "utility_spec": _utility_spec().to_dict(),
        "source_all_count": int(len(source_all)),
        "reports": reports,
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        source_summary=source_summary,
        fit_holdout=fit_holdout,
        period_rows=period_frame,
        proxy_ic=proxy_ic,
        manifest={**manifest, "outputs": {**manifest["outputs"], "markdown": str(output_dir / "source_regime_two_stage_veto_ablation.md")}},
    )
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=160)
    parser.add_argument("--max-feature-columns", type=int, default=0)
    parser.add_argument("--months", type=lambda value: _parse_csv(value, DEFAULT_MONTHS), default=list(DEFAULT_MONTHS))
    parser.add_argument("--fit-months", type=lambda value: _parse_csv(value, DEFAULT_FIT_MONTHS), default=list(DEFAULT_FIT_MONTHS))
    parser.add_argument("--holdout-month", type=str, default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--top-fracs", type=lambda value: _parse_float_csv(value), default=list(DEFAULT_TOP_FRACS))
    parser.add_argument("--proxy-top-k", type=int, default=DEFAULT_PROXY_TOP_K)
    parser.add_argument("--min-train-source-rows", type=int, default=250)
    parser.add_argument("--min-valid-source-rows", type=int, default=25)
    parser.add_argument("--min-week-rows", type=int, default=5)
    parser.add_argument("--max-timeout-rate", type=float, default=0.50)
    parser.add_argument("--risk-keep-fracs", type=lambda value: _parse_float_csv(value), default=list(DEFAULT_RISK_KEEP_FRACS))
    parser.add_argument("--risk-penalties", type=lambda value: _parse_float_csv(value), default=list(DEFAULT_RISK_PENALTIES))
    parser.add_argument("--sources", type=lambda value: _parse_csv(value, DEFAULT_SOURCES), default=list(DEFAULT_SOURCES))
    parser.add_argument("--regime-cols", type=lambda value: _parse_csv(value, DEFAULT_REGIME_COLS), default=list(DEFAULT_REGIME_COLS))
    parser.add_argument("--include-regime-masks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-source-regime-intersections", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--intersection-sources",
        type=lambda value: _parse_csv(value, DEFAULT_INTERSECTION_SOURCES),
        default=list(DEFAULT_INTERSECTION_SOURCES),
    )
    parser.add_argument("--state-cols", type=lambda value: _parse_csv(value, DEFAULT_STATE_COLS), default=list(DEFAULT_STATE_COLS))
    parser.add_argument("--state-bins", type=lambda value: _parse_csv(value, DEFAULT_STATE_BINS), default=list(DEFAULT_STATE_BINS))
    parser.add_argument("--state-reference-months", type=lambda value: _parse_csv(value, DEFAULT_FIT_MONTHS), default=None)
    parser.add_argument("--include-state-masks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-source-state-intersections", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--state-intersection-sources",
        type=lambda value: _parse_csv(value, DEFAULT_INTERSECTION_SOURCES),
        default=list(DEFAULT_INTERSECTION_SOURCES),
    )
    parser.add_argument("--max-regime-values-per-col", type=int, default=4)
    parser.add_argument("--min-mask-total-rows", type=int, default=300)
    parser.add_argument("--min-state-month-rows", type=int, default=50)
    parser.add_argument("--run-gap-hours", type=float, default=24.0)
    parser.add_argument("--include-causal-outcome-priors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-causal-state-path-priors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-causal-time-edge-priors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-event-confirmation-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-adverse-path-composites", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prior-windows-days", type=lambda value: _parse_float_csv(value), default=list(DEFAULT_PRIOR_WINDOWS_DAYS))
    parser.add_argument("--prior-embargo-hours", type=float, default=24.0)
    parser.add_argument(
        "--state-path-prior-features",
        type=lambda value: _parse_csv(value, tuple(DEFAULT_STATE_PATH_PRIOR_FEATURES)),
        default=list(DEFAULT_STATE_PATH_PRIOR_FEATURES),
    )
    parser.add_argument(
        "--event-feature-store-features",
        type=lambda value: _parse_csv(value, tuple(DEFAULT_EVENT_FEATURE_STORE_FEATURES)),
        default=list(DEFAULT_EVENT_FEATURE_STORE_FEATURES),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_report(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        max_feature_columns=args.max_feature_columns,
        months=list(args.months),
        fit_months=list(args.fit_months),
        holdout_month=str(args.holdout_month),
        top_fracs=list(args.top_fracs),
        proxy_top_k=int(args.proxy_top_k),
        min_train_source_rows=int(args.min_train_source_rows),
        min_valid_source_rows=int(args.min_valid_source_rows),
        min_week_rows=int(args.min_week_rows),
        max_timeout_rate=args.max_timeout_rate,
        risk_keep_fracs=list(args.risk_keep_fracs),
        risk_penalties=list(args.risk_penalties),
        sources=list(args.sources),
        regime_cols=list(args.regime_cols),
        include_regime_masks=bool(args.include_regime_masks),
        include_source_regime_intersections=bool(args.include_source_regime_intersections),
        intersection_sources=list(args.intersection_sources),
        state_cols=list(args.state_cols),
        state_bins=list(args.state_bins),
        state_reference_months=list(args.state_reference_months) if args.state_reference_months else None,
        include_state_masks=bool(args.include_state_masks),
        include_source_state_intersections=bool(args.include_source_state_intersections),
        state_intersection_sources=list(args.state_intersection_sources),
        max_regime_values_per_col=int(args.max_regime_values_per_col),
        min_mask_total_rows=int(args.min_mask_total_rows),
        min_state_month_rows=int(args.min_state_month_rows),
        run_gap_hours=float(args.run_gap_hours),
        include_causal_outcome_priors=bool(args.include_causal_outcome_priors),
        include_causal_state_path_priors=bool(args.include_causal_state_path_priors),
        include_causal_time_edge_priors=bool(args.include_causal_time_edge_priors),
        include_event_confirmation_features=bool(args.include_event_confirmation_features),
        include_adverse_path_composites=bool(args.include_adverse_path_composites),
        prior_windows_days=list(args.prior_windows_days),
        prior_embargo_hours=float(args.prior_embargo_hours),
        state_path_prior_features=list(args.state_path_prior_features),
        event_feature_store_features=list(args.event_feature_store_features),
    )
    print(json.dumps(_json_safe({"output_dir": manifest["output_dir"], "outputs": manifest["outputs"]}), indent=2))


if __name__ == "__main__":
    main()
