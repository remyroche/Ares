#!/usr/bin/env python3
"""Regime-conditioned clean-recoverability proxy before full model training.

This diagnostic asks whether the clean executable label tail becomes learnable
when the cheap rare-event proxy is fit only inside causal feature regimes. Each
validation month is scored out-of-time using thresholds computed from prior
months only.
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

from scripts.run_label_clean_recoverability_proxy import (  # noqa: E402
    DEFAULT_DIRTY_PENALTIES,
    DEFAULT_LABEL_ARMS as CLEAN_DEFAULT_LABEL_ARMS,
    DEFAULT_OUTPUT_DIR as CLEAN_DEFAULT_OUTPUT_DIR,
    DEFAULT_SEEDS,
    DEFAULT_TARGET_MODES,
    DEFAULT_TOP_FRACS,
    DEFAULT_WEIGHT_MODES,
    DERIVED_FEATURE_MODES,
    _add_derived_clean_recoverability_features,
    _baseline_row,
    _clean_weights,
    _dirty_target,
    _seed_average_predict,
    _target_for_mode,
)
from scripts.run_label_dual_target_execution_smoke import _rank_pct, _selection_weekly_rows  # noqa: E402
from scripts.run_label_economic_proxy_ablation import LABEL_ARMS, _label_targets  # noqa: E402
from scripts.run_label_feature_store_model_smoke import _add_delta_fields, _month_model_frame  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    _decile_diagnostics,
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _spearman,
    _selection_metrics,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
    _causal_outcome_prior_features,
    _causal_state_path_prior_features,
    _event_confirmation_features,
)


DEFAULT_OUTPUT_DIR = CLEAN_DEFAULT_OUTPUT_DIR.with_name("label_regime_conditioned_clean_proxy_v1")
DEFAULT_LABEL_ARMS = (
    "S49_clean_recoverable_tail_rank",
    "S51_clean_dirty_contrast_recoverable",
)
DEFAULT_TARGET_MODES_REGIME = ("soft_hard_blend",)
DEFAULT_WEIGHT_MODES_REGIME = ("hard_vs_dirty",)
DEFAULT_DIRTY_PENALTIES_REGIME = (0.0,)
DEFAULT_BAD_MAE_PENALTIES_REGIME = (0.0,)
DEFAULT_BAD_MAE_KEEP_FRACS_REGIME = (1.0,)
DEFAULT_TOP_FRACS_REGIME = (0.005, 0.010)
DEFAULT_REGIME_FEATURES = (
    "loc_prev_week_range_pos_24",
    "oi_rank",
    "zscore_price_200",
    "distance_to_resistance_daily_vwap_atr",
    "dn_vol",
    "atr_compression_ratio",
    "oiw_pos_delta_entry_dist_7d_atr",
    "adx_10",
)


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    lowered = str(value).strip().lower()
    if lowered == "all":
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_int_csv(value: str | None, default: tuple[int, ...]) -> list[int]:
    if value is None or not str(value).strip():
        return list(default)
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _parse_float_csv(value: str | None, default: tuple[float, ...]) -> list[float]:
    if value is None or not str(value).strip():
        return list(default)
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _regime_bins(
    *,
    train_values: pd.Series,
    valid_values: pd.Series,
    quantile: float,
    include_middle: bool,
) -> list[dict[str, Any]]:
    train_num = _safe_numeric(train_values).replace([np.inf, -np.inf], np.nan)
    valid_num = _safe_numeric(valid_values).replace([np.inf, -np.inf], np.nan)
    finite = train_num.dropna()
    if len(finite) < 500 or int(finite.nunique(dropna=True)) < 4:
        return []
    q = max(0.05, min(float(quantile), 0.45))
    low = float(finite.quantile(q))
    high = float(finite.quantile(1.0 - q))
    if not (math.isfinite(low) and math.isfinite(high)) or low >= high:
        return []

    bins = [
        {
            "regime_bin": f"low_q{int(round(q * 100)):02d}",
            "regime_lower": float("-inf"),
            "regime_upper": low,
            "train_mask": train_num <= low,
            "valid_mask": valid_num <= low,
        },
        {
            "regime_bin": f"high_q{int(round((1.0 - q) * 100)):02d}",
            "regime_lower": high,
            "regime_upper": float("inf"),
            "train_mask": train_num >= high,
            "valid_mask": valid_num >= high,
        },
    ]
    if include_middle:
        bins.append(
            {
                "regime_bin": f"mid_q{int(round(q * 100)):02d}_{int(round((1.0 - q) * 100)):02d}",
                "regime_lower": low,
                "regime_upper": high,
                "train_mask": train_num.gt(low) & train_num.lt(high),
                "valid_mask": valid_num.gt(low) & valid_num.lt(high),
            }
        )
    return bins


def _run_month(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    features: list[str],
    month: str,
    label_arms: list[str],
    target_modes: list[str],
    weight_modes: list[str],
    dirty_penalties: list[float],
    bad_mae_penalties: list[float],
    bad_mae_keep_fracs: list[float],
    top_fracs: list[float],
    seeds: list[int],
    train_lookback_months: int | None,
    max_weight: float,
    min_weight: float,
    regime_features: list[str],
    regime_quantile: float,
    include_middle_regime: bool,
    min_regime_train_rows: int,
    min_regime_valid_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    train_mask = month_period < month
    if train_lookback_months is not None and int(train_lookback_months) > 0:
        prior_months = sorted(month_period[train_mask].dropna().unique())
        keep_months = set(prior_months[-int(train_lookback_months) :])
        train_mask = train_mask & month_period.isin(keep_months)
    valid_mask = month_period == month
    if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
        return [], [], [
            {
                "period": month,
                "skipped": True,
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
            }
        ]

    train_metrics_all = metrics.loc[train_mask].copy()
    valid_metrics_all = metrics.loc[valid_mask].copy()
    dirty_all = _dirty_target(metrics)
    baseline = _baseline_row(valid_metrics_all.reset_index(drop=True))
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    train_indices = train_mask[train_mask].index
    valid_indices = valid_mask[valid_mask].index

    active_regime_features = [feature for feature in regime_features if feature in frame.columns]
    for regime_feature in active_regime_features:
        bins = _regime_bins(
            train_values=frame.loc[train_indices, regime_feature],
            valid_values=frame.loc[valid_indices, regime_feature],
            quantile=regime_quantile,
            include_middle=include_middle_regime,
        )
        for regime in bins:
            train_regime_mask = pd.Series(False, index=frame.index)
            valid_regime_mask = pd.Series(False, index=frame.index)
            train_regime_mask.loc[train_indices] = regime["train_mask"].to_numpy(dtype=bool)
            valid_regime_mask.loc[valid_indices] = regime["valid_mask"].to_numpy(dtype=bool)
            train_regime_mask = train_mask & train_regime_mask
            valid_regime_mask = valid_mask & valid_regime_mask
            train_rows = int(train_regime_mask.sum())
            valid_rows = int(valid_regime_mask.sum())
            if train_rows < int(min_regime_train_rows) or valid_rows < int(min_regime_valid_rows):
                diagnostic_rows.append(
                    {
                        "period": month,
                        "regime_feature": regime_feature,
                        "regime_bin": regime["regime_bin"],
                        "skipped": True,
                        "skip_reason": "insufficient_regime_rows",
                        "train_rows": train_rows,
                        "valid_rows": valid_rows,
                    }
                )
                continue

            x_train, x_valid = _month_model_frame(
                frame,
                train_mask=train_regime_mask,
                valid_mask=valid_regime_mask,
                features=features,
            )
            train = frame.loc[train_regime_mask].copy()
            valid = frame.loc[valid_regime_mask].copy().reset_index(drop=True)
            train_metrics = metrics.loc[train_regime_mask].copy()
            valid_metrics = metrics.loc[valid_regime_mask].copy().reset_index(drop=True)
            dirty_train = dirty_all.loc[train_regime_mask].copy()
            dirty_valid = dirty_all.loc[valid_regime_mask].copy().reset_index(drop=True)
            bad_mae_train = (train_metrics["mae_norm"] >= 1.0).astype(float)
            bad_mae_valid = (valid_metrics["mae_norm"] >= 1.0).astype(float).reset_index(drop=True)

            dirty_pred_cache: dict[str, pd.Series] = {}
            bad_mae_pred_cache: dict[str, pd.Series] = {}
            for label_arm in label_arms:
                target_train = targets[label_arm].loc[train_regime_mask].copy()
                target_valid = targets[label_arm].loc[valid_regime_mask].copy().reset_index(drop=True)
                clean_hard_train = pd.to_numeric(target_train["target_hard"], errors="coerce").fillna(0.0)
                clean_hard_valid = pd.to_numeric(target_valid["target_hard"], errors="coerce").fillna(0.0)
                for target_mode in target_modes:
                    y_train = _target_for_mode(target_train, target_mode)
                    for weight_mode in weight_modes:
                        weights = _clean_weights(
                            clean_hard=clean_hard_train,
                            dirty=dirty_train,
                            mode=weight_mode,
                            max_weight=max_weight,
                            min_weight=min_weight,
                        )
                        clean_pred, clean_seed_std_mean, clean_seed_std_p90 = _seed_average_predict(
                            x_train=x_train,
                            y_train=y_train,
                            w_train=weights,
                            x_valid=x_valid,
                            seeds=seeds,
                        )
                        clean_pred = clean_pred.reset_index(drop=True)
                        clean_rank = _rank_pct(clean_pred)
                        diagnostic_rows.append(
                            {
                                "period": month,
                                "label_arm": label_arm,
                                "target_mode": target_mode,
                                "weight_arm": weight_mode,
                                "regime_feature": regime_feature,
                                "regime_bin": regime["regime_bin"],
                                "regime_lower": regime["regime_lower"],
                                "regime_upper": regime["regime_upper"],
                                "train_rows": train_rows,
                                "valid_rows": valid_rows,
                                "model_feature_count": int(len(features)),
                                "train_clean_hard_rate": _safe_mean(clean_hard_train),
                                "valid_clean_hard_rate": _safe_mean(clean_hard_valid),
                                "train_dirty_rate": _safe_mean(dirty_train),
                                "valid_dirty_rate": _safe_mean(dirty_valid),
                                "train_bad_mae_rate": _safe_mean(bad_mae_train),
                                "valid_bad_mae_rate": _safe_mean(bad_mae_valid),
                                "clean_ic_u": _spearman(clean_pred, valid_metrics["u_policy_net"]),
                                "clean_ic_label_soft": _spearman(clean_pred, target_valid["target_soft"]),
                                "clean_ic_label_hard": _spearman(clean_pred, clean_hard_valid),
                                "clean_seed_std_mean": clean_seed_std_mean,
                                "clean_seed_std_p90": clean_seed_std_p90,
                            }
                        )
                        for dirty_penalty in dirty_penalties:
                            dirty_rank = pd.Series(0.0, index=clean_rank.index)
                            if float(dirty_penalty) > 0.0:
                                cache_key = f"{weight_mode}"
                                if cache_key not in dirty_pred_cache:
                                    dirty_weights = _clean_weights(
                                        clean_hard=dirty_train,
                                        dirty=dirty_train,
                                        mode="balanced",
                                        max_weight=max_weight,
                                        min_weight=min_weight,
                                    )
                                    dirty_pred, _dirty_seed_std_mean, _dirty_seed_std_p90 = _seed_average_predict(
                                        x_train=x_train,
                                        y_train=dirty_train,
                                        w_train=dirty_weights,
                                        x_valid=x_valid,
                                        seeds=seeds,
                                    )
                                    dirty_pred_cache[cache_key] = dirty_pred.reset_index(drop=True)
                                dirty_rank = _rank_pct(dirty_pred_cache[cache_key])
                            for bad_mae_penalty in bad_mae_penalties:
                                bad_mae_rank = pd.Series(0.0, index=clean_rank.index)
                                if float(bad_mae_penalty) > 0.0 or any(float(v) < 1.0 for v in bad_mae_keep_fracs):
                                    cache_key = f"{weight_mode}"
                                    if cache_key not in bad_mae_pred_cache:
                                        bad_mae_weights = _clean_weights(
                                            clean_hard=bad_mae_train,
                                            dirty=bad_mae_train,
                                            mode="balanced",
                                            max_weight=max_weight,
                                            min_weight=min_weight,
                                        )
                                        bad_mae_pred, _bad_mae_seed_std_mean, _bad_mae_seed_std_p90 = _seed_average_predict(
                                            x_train=x_train,
                                            y_train=bad_mae_train,
                                            w_train=bad_mae_weights,
                                            x_valid=x_valid,
                                            seeds=seeds,
                                        )
                                        bad_mae_pred_cache[cache_key] = bad_mae_pred.reset_index(drop=True)
                                    bad_mae_rank = _rank_pct(bad_mae_pred_cache[cache_key])
                                score = (
                                    clean_rank
                                    - float(dirty_penalty) * dirty_rank
                                    - float(bad_mae_penalty) * bad_mae_rank
                                )
                                decile = _decile_diagnostics(score, valid_metrics["u_policy_net"])
                                bad_mae_ic_actual = (
                                    _spearman(bad_mae_pred_cache.get(f"{weight_mode}", pd.Series(np.nan, index=score.index)), bad_mae_valid)
                                    if (float(bad_mae_penalty) > 0.0 or any(float(v) < 1.0 for v in bad_mae_keep_fracs))
                                    else float("nan")
                                )
                                for bad_mae_keep_frac in bad_mae_keep_fracs:
                                    keep_frac = float(bad_mae_keep_frac)
                                    use_gate = keep_frac < 1.0
                                    gate_mask = (bad_mae_rank <= keep_frac).fillna(False) if use_gate else pd.Series(True, index=score.index)
                                    gate_rows = int(gate_mask.sum()) if use_gate else int(len(score))
                                    gate_rate = gate_rows / float(len(score)) if len(score) else 0.0
                                    selection_mode = (
                                        "regime_conditioned_bad_mae_gate"
                                        if use_gate
                                        else "regime_conditioned_clean_proxy"
                                    )
                                    arm = (
                                        f"{label_arm}::{weight_mode}::{target_mode}"
                                        f"::dirty{float(dirty_penalty):.2f}"
                                        f"::badmae{float(bad_mae_penalty):.2f}"
                                        f"::keepmae{keep_frac:.2f}"
                                        f"::regime_{regime_feature}_{regime['regime_bin']}"
                                    )
                                    score_for_selection = score.where(gate_mask) if use_gate else score
                                    for top_frac in top_fracs:
                                        gate_rate_for_top = gate_rate if use_gate else 1.0
                                        selection_frac_for_top = (
                                            min(1.0, float(top_frac) / gate_rate_for_top)
                                            if use_gate and gate_rate_for_top > 0.0
                                            else float(top_frac)
                                        )
                                        row = _selection_metrics(
                                            frame=valid,
                                            metrics=valid_metrics,
                                            target=target_valid,
                                            score=score_for_selection,
                                            arm=arm,
                                            selector="regime_conditioned_clean_proxy_oos",
                                            period=month,
                                            top_frac=selection_frac_for_top,
                                        )
                                        row["top_frac"] = float(top_frac)
                                        _add_delta_fields(row, baseline)
                                        row.update(
                                            {
                                                "label_arm": label_arm,
                                                "weight_arm": weight_mode,
                                                "target_mode": target_mode,
                                                "selection_mode": selection_mode,
                                                "mae_penalty": float(dirty_penalty),
                                                "bad_mae_penalty": float(bad_mae_penalty),
                                                "wide_penalty": 0.0,
                                                "timeout_penalty": 0.0,
                                                "mae_keep_frac": keep_frac,
                                                "wide_keep_frac": 1.0,
                                                "timeout_keep_frac": 1.0,
                                                "selection_frac_within_gate": float(selection_frac_for_top),
                                                "gate_rows": gate_rows,
                                                "gate_candidate_rate": float(gate_rate_for_top),
                                                "regime_feature": regime_feature,
                                                "regime_bin": regime["regime_bin"],
                                                "regime_lower": regime["regime_lower"],
                                                "regime_upper": regime["regime_upper"],
                                                "regime_train_rows": train_rows,
                                                "regime_valid_rows": valid_rows,
                                                "global_rows": int(valid_mask.sum()),
                                                "global_selected_share": float(row["selected_rows"] / int(valid_mask.sum())),
                                                "model_feature_count": int(len(features)),
                                                "model_features": ",".join(features),
                                                "score_ic_u": _spearman(score_for_selection, valid_metrics["u_policy_net"]),
                                                "score_ic_label": _spearman(score_for_selection, target_valid["target_soft"]),
                                                "score_ic_label_hard": _spearman(score_for_selection, clean_hard_valid),
                                                "bad_mae_ic_actual": bad_mae_ic_actual,
                                                **decile,
                                            }
                                        )
                                        monthly_rows.append(row)
                                        for weekly_row in _selection_weekly_rows(
                                            frame=valid,
                                            metrics=valid_metrics,
                                            target=target_valid,
                                            score=score,
                                            arm=arm,
                                            selector="regime_conditioned_clean_proxy_oos",
                                            period=month,
                                            top_frac=float(top_frac),
                                            gate_mask=gate_mask if use_gate else None,
                                            selection_frac=selection_frac_for_top,
                                        ):
                                            weekly_row.update(
                                                {
                                                    "label_arm": label_arm,
                                                    "weight_arm": weight_mode,
                                                    "target_mode": target_mode,
                                                    "selection_mode": selection_mode,
                                                    "mae_penalty": float(dirty_penalty),
                                                    "bad_mae_penalty": float(bad_mae_penalty),
                                                    "wide_penalty": 0.0,
                                                    "timeout_penalty": 0.0,
                                                    "mae_keep_frac": keep_frac,
                                                    "wide_keep_frac": 1.0,
                                                    "timeout_keep_frac": 1.0,
                                                    "selection_frac_within_gate": float(selection_frac_for_top),
                                                    "gate_rows": gate_rows,
                                                    "gate_candidate_rate": float(gate_rate_for_top),
                                                    "regime_feature": regime_feature,
                                                    "regime_bin": regime["regime_bin"],
                                                    "regime_lower": regime["regime_lower"],
                                                    "regime_upper": regime["regime_upper"],
                                                    "regime_train_rows": train_rows,
                                                    "regime_valid_rows": valid_rows,
                                                    "global_rows": int(valid_mask.sum()),
                                                    "model_feature_count": int(len(features)),
                                                    "model_features": ",".join(features),
                                                    "score_ic_u": _spearman(score_for_selection, valid_metrics["u_policy_net"]),
                                                    "score_ic_label": _spearman(score_for_selection, target_valid["target_soft"]),
                                                    "score_ic_label_hard": _spearman(score_for_selection, clean_hard_valid),
                                                    "bad_mae_ic_actual": bad_mae_ic_actual,
                                                }
                                            )
                                            weekly_rows.append(weekly_row)
    return monthly_rows, weekly_rows, diagnostic_rows


def _aggregate(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    rows: list[dict[str, Any]] = []
    group_cols = [
        "arm",
        "label_arm",
        "weight_arm",
        "target_mode",
        "selection_mode",
        "mae_penalty",
        "bad_mae_penalty",
        "mae_keep_frac",
        "top_frac",
        "regime_feature",
        "regime_bin",
    ]
    for key, group in monthly.groupby(group_cols, observed=True, dropna=False):
        key_dict = dict(zip(group_cols, key))
        mean_u = pd.to_numeric(group["mean_u"], errors="coerce")
        selected_rows = pd.to_numeric(group["selected_rows"], errors="coerce")
        rows.append(
            {
                **key_dict,
                "months": int(group["period"].nunique()),
                "positive_months": int((mean_u > 0.0).sum()),
                "mean_u": _safe_mean(mean_u),
                "worst_month_mean_u": float(mean_u.min()) if len(mean_u.dropna()) else float("nan"),
                "hit_u": _safe_mean(group["hit_u"]),
                "q10_u": _safe_mean(group["q10_u"]),
                "target_top_hard_rate": _safe_mean(group["target_top_hard_rate"]),
                "score_ic_u": _safe_mean(group["score_ic_u"]),
                "score_ic_label": _safe_mean(group["score_ic_label"]),
                "score_ic_label_hard": _safe_mean(group["score_ic_label_hard"]),
                "bad_mae_ic_actual": _safe_mean(group.get("bad_mae_ic_actual")),
                "bad_mae_1r_rate": _safe_mean(group["bad_mae_1r_rate"]),
                "wide_barrier_25bps_rate": _safe_mean(group["wide_barrier_25bps_rate"]),
                "timeout_rate": _safe_mean(group["timeout_rate"]),
                "top_symbol_share": _safe_mean(group["top_symbol_share"]),
                "global_selected_share": _safe_mean(group.get("global_selected_share")),
                "gate_candidate_rate": _safe_mean(group.get("gate_candidate_rate")),
                "mean_selected_rows": _safe_mean(selected_rows),
                "min_selected_rows": int(selected_rows.min()) if len(selected_rows.dropna()) else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["top_frac", "mean_u", "worst_month_mean_u"],
        ascending=[True, False, False],
    )


def _write_markdown(output_dir: Path, aggregate: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "label_regime_conditioned_clean_proxy.md"

    def table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[col for col in cols if col in frame.columns]].copy()
        if limit is not None:
            view = view.head(limit)
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
        return view.to_markdown(index=False)

    cols = [
        "label_arm",
        "weight_arm",
        "target_mode",
        "regime_feature",
        "regime_bin",
        "mae_penalty",
        "bad_mae_penalty",
        "mae_keep_frac",
        "top_frac",
        "months",
        "positive_months",
        "mean_u",
        "worst_month_mean_u",
        "target_top_hard_rate",
        "score_ic_u",
        "score_ic_label_hard",
        "bad_mae_ic_actual",
        "bad_mae_1r_rate",
        "timeout_rate",
        "gate_candidate_rate",
        "mean_selected_rows",
        "global_selected_share",
    ]
    lines = [
        "# Label Regime-Conditioned Clean-Recoverability Proxy",
        "",
        "Scope: cheap month-forward clean proxy fit inside causal feature regimes. This is not production LightGBM training or a final OOS claim.",
        "",
        f"Rows: `{manifest['rows']}`",
        f"Symbols: `{manifest['symbols']}`",
        f"Feature count: `{manifest['feature_count']}`",
        f"Regime features: `{','.join(manifest['regime_features'])}`",
        f"Months: `{','.join(manifest['months'])}`",
        "",
    ]
    for frac in manifest["top_fracs"]:
        subset = aggregate[aggregate["top_frac"].eq(float(frac))].sort_values(
            ["mean_u", "worst_month_mean_u"],
            ascending=[False, False],
        )
        lines.extend([f"## Top {float(frac):.2%} Within Regime", "", table(subset, cols, limit=40), ""])
    lines.extend(
        [
            "## Outputs",
            "",
            f"- Monthly: `{manifest['outputs']['monthly']}`",
            f"- Weekly: `{manifest['outputs']['weekly']}`",
            f"- Aggregate: `{manifest['outputs']['aggregate']}`",
            f"- Diagnostics: `{manifest['outputs']['diagnostics']}`",
            f"- Manifest: `{manifest['outputs']['manifest']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_proxy(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    months: list[str],
    label_arms: list[str],
    target_modes: list[str],
    weight_modes: list[str],
    dirty_penalties: list[float],
    bad_mae_penalties: list[float],
    bad_mae_keep_fracs: list[float],
    top_fracs: list[float],
    seeds: list[int],
    train_lookback_months: int | None,
    max_weight: float,
    min_weight: float,
    derived_feature_mode: str,
    regime_features: list[str],
    regime_quantile: float,
    include_middle_regime: bool,
    min_regime_train_rows: int,
    min_regime_valid_rows: int,
    include_causal_outcome_priors: bool,
    include_causal_state_path_priors: bool,
    include_event_confirmation_features: bool,
    prior_windows_days: list[float],
    prior_embargo_hours: float,
    state_path_prior_features: list[str],
    event_feature_store_features: list[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_labels(labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    if include_event_confirmation_features:
        selected_features = list(dict.fromkeys(list(selected_features) + list(event_feature_store_features)))
    for feature in regime_features:
        if feature not in selected_features:
            selected_features.append(feature)
    feature_matrix, feature_store_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    if len(feature_matrix.columns):
        frame = pd.concat([frame, feature_matrix.astype(np.float32, copy=False)], axis=1).copy()
    derived_matrix = _add_derived_clean_recoverability_features(
        frame,
        mode=str(derived_feature_mode),
    )
    derived_features = list(derived_matrix.columns)
    if derived_features:
        frame = pd.concat([frame, derived_matrix], axis=1).copy()

    metrics = _path_metrics(frame)
    prior_reports: dict[str, Any] = {
        "causal_outcome_priors": {"enabled": False},
        "causal_state_path_priors": {"enabled": False},
        "event_confirmation_features": {"enabled": False},
    }
    if include_causal_outcome_priors:
        prior_features, prior_reports["causal_outcome_priors"] = _causal_outcome_prior_features(
            frame,
            metrics,
            windows_days=[float(v) for v in prior_windows_days],
            embargo_hours=float(prior_embargo_hours),
        )
        frame = pd.concat([frame, prior_features.astype(np.float32, copy=False)], axis=1).copy()
    if include_causal_state_path_priors:
        state_prior_features, prior_reports["causal_state_path_priors"] = _causal_state_path_prior_features(
            frame,
            metrics,
            state_features=list(state_path_prior_features),
            windows_days=[float(v) for v in prior_windows_days],
            embargo_hours=float(prior_embargo_hours),
        )
        frame = pd.concat([frame, state_prior_features.astype(np.float32, copy=False)], axis=1).copy()
    if include_event_confirmation_features:
        event_features, prior_reports["event_confirmation_features"] = _event_confirmation_features(
            frame,
            event_features=list(event_feature_store_features),
        )
        frame = pd.concat([frame, event_features.astype(np.float32, copy=False)], axis=1).copy()

    features = _feature_columns(frame)
    targets = _label_targets(frame, metrics)
    if not label_arms:
        label_arms = list(LABEL_ARMS)
    missing_labels = sorted(set(label_arms) - set(targets))
    if missing_labels:
        raise ValueError(f"Unknown label arms: {missing_labels}")
    active_regime_features = [feature for feature in regime_features if feature in frame.columns]
    if not active_regime_features:
        raise ValueError("None of the requested regime features are available in the feature store")

    available_months = sorted(frame["__ts__"].dt.to_period("M").dropna().astype(str).unique())
    eval_months = months or available_months[1:]
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for month in eval_months:
        rows, weeks, diagnostics = _run_month(
            frame=frame,
            metrics=metrics,
            targets=targets,
            features=features,
            month=str(month),
            label_arms=label_arms,
            target_modes=target_modes,
            weight_modes=weight_modes,
            dirty_penalties=dirty_penalties,
            bad_mae_penalties=bad_mae_penalties,
            bad_mae_keep_fracs=bad_mae_keep_fracs,
            top_fracs=top_fracs,
            seeds=seeds,
            train_lookback_months=train_lookback_months,
            max_weight=max_weight,
            min_weight=min_weight,
            regime_features=active_regime_features,
            regime_quantile=regime_quantile,
            include_middle_regime=include_middle_regime,
            min_regime_train_rows=min_regime_train_rows,
            min_regime_valid_rows=min_regime_valid_rows,
        )
        monthly_rows.extend(rows)
        weekly_rows.extend(weeks)
        diagnostic_rows.extend(diagnostics)

    monthly = pd.DataFrame(monthly_rows)
    weekly = pd.DataFrame(weekly_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    aggregate = _aggregate(monthly)
    paths = {
        "monthly": output_dir / "label_regime_conditioned_clean_proxy_monthly.csv",
        "weekly": output_dir / "label_regime_conditioned_clean_proxy_weekly.csv",
        "aggregate": output_dir / "label_regime_conditioned_clean_proxy_aggregate.csv",
        "diagnostics": output_dir / "label_regime_conditioned_clean_proxy_diagnostics.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    manifest = {
        "scope": "regime_conditioned_clean_recoverability_proxy_not_full_policy_training",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_count": int(len(features)),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "feature_store": feature_store_report,
        "derived_feature_mode": str(derived_feature_mode),
        "derived_feature_count": int(len(derived_features)),
        "derived_features": derived_features,
        "months": [str(v) for v in eval_months],
        "label_arms": label_arms,
        "target_modes": target_modes,
        "weight_modes": weight_modes,
        "dirty_penalties": [float(v) for v in dirty_penalties],
        "bad_mae_penalties": [float(v) for v in bad_mae_penalties],
        "bad_mae_keep_fracs": [float(v) for v in bad_mae_keep_fracs],
        "top_fracs": [float(v) for v in top_fracs],
        "regime_features": active_regime_features,
        "regime_quantile": float(regime_quantile),
        "include_middle_regime": bool(include_middle_regime),
        "min_regime_train_rows": int(min_regime_train_rows),
        "min_regime_valid_rows": int(min_regime_valid_rows),
        "include_causal_outcome_priors": bool(include_causal_outcome_priors),
        "include_causal_state_path_priors": bool(include_causal_state_path_priors),
        "include_event_confirmation_features": bool(include_event_confirmation_features),
        "prior_windows_days": [float(v) for v in prior_windows_days],
        "prior_embargo_hours": float(prior_embargo_hours),
        "state_path_prior_features": list(state_path_prior_features),
        "event_feature_store_features": list(event_feature_store_features),
        **prior_reports,
        "model": {
            "type": "ExtraTreesRegressor",
            "seeds": [int(v) for v in seeds],
            "seed_count": int(len(seeds)),
            "train_lookback_months": int(train_lookback_months)
            if train_lookback_months is not None
            else None,
            "max_weight": float(max_weight),
            "min_weight": float(min_weight),
        },
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_markdown(output_dir, aggregate, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=96)
    parser.add_argument("--months", default="2026-04,2026-05,2026-06")
    parser.add_argument("--label-arms", default=",".join(DEFAULT_LABEL_ARMS))
    parser.add_argument("--target-modes", default=",".join(DEFAULT_TARGET_MODES_REGIME))
    parser.add_argument("--weight-modes", default=",".join(DEFAULT_WEIGHT_MODES_REGIME))
    parser.add_argument("--dirty-penalties", default=",".join(str(v) for v in DEFAULT_DIRTY_PENALTIES_REGIME))
    parser.add_argument("--bad-mae-penalties", default=",".join(str(v) for v in DEFAULT_BAD_MAE_PENALTIES_REGIME))
    parser.add_argument("--bad-mae-keep-fracs", default=",".join(str(v) for v in DEFAULT_BAD_MAE_KEEP_FRACS_REGIME))
    parser.add_argument("--top-fracs", default=",".join(str(v) for v in DEFAULT_TOP_FRACS_REGIME))
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--train-lookback-months", type=int, default=None)
    parser.add_argument("--max-weight", type=float, default=12.0)
    parser.add_argument("--min-weight", type=float, default=0.10)
    parser.add_argument("--derived-feature-mode", choices=DERIVED_FEATURE_MODES, default="none")
    parser.add_argument("--regime-features", default=",".join(DEFAULT_REGIME_FEATURES))
    parser.add_argument("--regime-quantile", type=float, default=0.33)
    parser.add_argument("--include-middle-regime", action="store_true")
    parser.add_argument("--min-regime-train-rows", type=int, default=700)
    parser.add_argument("--min-regime-valid-rows", type=int, default=120)
    parser.add_argument("--include-causal-outcome-priors", action="store_true")
    parser.add_argument("--include-causal-state-path-priors", action="store_true")
    parser.add_argument("--include-event-confirmation-features", action="store_true")
    parser.add_argument(
        "--prior-windows-days",
        default=",".join(str(v) for v in DEFAULT_PRIOR_WINDOWS_DAYS),
    )
    parser.add_argument("--prior-embargo-hours", type=float, default=24.0)
    parser.add_argument(
        "--state-path-prior-features",
        default=",".join(DEFAULT_STATE_PATH_PRIOR_FEATURES),
    )
    parser.add_argument(
        "--event-feature-store-features",
        default=",".join(DEFAULT_EVENT_FEATURE_STORE_FEATURES),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_proxy(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        months=_parse_csv(args.months, ()),
        label_arms=_parse_csv(args.label_arms, CLEAN_DEFAULT_LABEL_ARMS),
        target_modes=_parse_csv(args.target_modes, DEFAULT_TARGET_MODES),
        weight_modes=_parse_csv(args.weight_modes, DEFAULT_WEIGHT_MODES),
        dirty_penalties=_parse_float_csv(args.dirty_penalties, DEFAULT_DIRTY_PENALTIES),
        bad_mae_penalties=_parse_float_csv(args.bad_mae_penalties, DEFAULT_BAD_MAE_PENALTIES_REGIME),
        bad_mae_keep_fracs=_parse_float_csv(args.bad_mae_keep_fracs, DEFAULT_BAD_MAE_KEEP_FRACS_REGIME),
        top_fracs=_parse_float_csv(args.top_fracs, DEFAULT_TOP_FRACS),
        seeds=_parse_int_csv(args.seeds, DEFAULT_SEEDS),
        train_lookback_months=args.train_lookback_months,
        max_weight=float(args.max_weight),
        min_weight=float(args.min_weight),
        derived_feature_mode=str(args.derived_feature_mode),
        regime_features=_parse_csv(args.regime_features, DEFAULT_REGIME_FEATURES),
        regime_quantile=float(args.regime_quantile),
        include_middle_regime=bool(args.include_middle_regime),
        min_regime_train_rows=int(args.min_regime_train_rows),
        min_regime_valid_rows=int(args.min_regime_valid_rows),
        include_causal_outcome_priors=bool(args.include_causal_outcome_priors),
        include_causal_state_path_priors=bool(args.include_causal_state_path_priors),
        include_event_confirmation_features=bool(args.include_event_confirmation_features),
        prior_windows_days=_parse_float_csv(args.prior_windows_days, DEFAULT_PRIOR_WINDOWS_DAYS),
        prior_embargo_hours=float(args.prior_embargo_hours),
        state_path_prior_features=_parse_csv(args.state_path_prior_features, DEFAULT_STATE_PATH_PRIOR_FEATURES),
        event_feature_store_features=_parse_csv(args.event_feature_store_features, DEFAULT_EVENT_FEATURE_STORE_FEATURES),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
