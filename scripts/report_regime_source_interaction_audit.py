#!/usr/bin/env python3
"""Report source x regime interaction quality for a base candidate stream.

This audit is intentionally downstream of a trained base model. It merges the
base scored ledger with materialized labels, then evaluates whether candidate
regime definitions create stable, learnable differences in executable path
quality beyond source tags alone.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_META_FEATURE_LEDGER_PATH = Path(
    "data_perp/reports/ae_gmm_archetype_ablation_existing_source_export_20260704_v1/"
    "g5_s41_local_timeout_dirty_sidefallback_source_spread170/"
    "label_feature_store_model_smoke_candidate_ledger.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/ae_gmm_archetype_validation_status_20260704/"
    "meta_prefeature_regime_source_interaction_audit_v1"
)
DEFAULT_PATH_ORDER_LABELS_PATH = Path(
    "data_perp/artifacts/20260704_s52_bidirectional_first_touch_tp075_sl075_fast16_bar50_cost100bps_ordercols_v2_labels/"
    "labels"
)

SOURCE_SCORE_COLUMNS = {
    "quiet_continuation": "__regime_source_quiet_continuation_score__",
    "compression_release": "__regime_source_compression_release_score__",
    "loud_breakout_impulse": "__regime_source_loud_breakout_impulse_score__",
    "run_entry": "__regime_source_run_entry_score__",
    "late_run_continuation": "__regime_source_late_run_continuation_score__",
    "retest_reversal": "__regime_source_retest_reversal_score__",
    "dirty_shock_avoid": "__regime_source_dirty_shock_avoid_score__",
}

SOURCE_SCORE_BASE_COLUMNS = {
    "quiet_continuation": "quiet_continuation_score",
    "compression_release": "compression_release_score",
    "loud_breakout_impulse": "loud_breakout_impulse_score",
    "run_entry": "run_entry_score",
    "late_run_continuation": "late_run_continuation_score",
    "retest_reversal": "retest_reversal_score",
    "dirty_shock_avoid": "dirty_shock_avoid_score",
}

AE_GMM_REQUIRED_PATTERNS = (
    "gmm_cluster_posterior",
    "gmm_prob_",
    "gmm_mahal",
    "gmm_dist_center",
    "ae_reconstruction",
    "reconstruction_error",
    "dae_b16_",
)

PATH_ORDER_LABEL_COLUMNS = (
    "__ts__",
    "__symbol__",
    "side_name",
    "__side__",
    "__first_touch_bar__",
    "__first_touch_same_bar_both__",
    "__trailing_profit_activation_bar__",
    "__bars_to_mfe_05r__",
    "__bars_to_mfe_075r__",
    "__bars_to_mfe_1r__",
    "__bars_to_mfe_125r__",
    "__bars_to_mfe_15r__",
    "__bars_to_mae_05r__",
    "__bars_to_mae_075r__",
    "__bars_to_mae_1r__",
    "__bars_to_mae_15r__",
    "__mfe_1r_before_mae_05r__",
    "__mfe_1r_before_mae_075r__",
    "__mfe_1r_before_mae_1r__",
    "__mae_05r_before_mfe_1r__",
    "__mae_075r_before_mfe_1r__",
    "__mae_1r_before_mfe_1r__",
    "__max_adverse_before_mfe_1r__",
    "__underwater_bars_before_mfe_1r__",
    "__underwater_fraction_before_mfe_1r__",
    "__area_underwater_before_mfe_1r__",
)

POLICY_MENU = (
    {
        "policy": "P0_abstain",
        "kind": "abstain",
        "tp_r": 0.0,
        "sl_r": 0.0,
        "max_holding_bars": 0,
    },
    {
        "policy": "P1_tight_scalp",
        "kind": "fixed_tp_sl",
        "tp_r": 0.75,
        "sl_r": 0.50,
        "max_holding_bars": 6,
    },
    {
        "policy": "P2_fast_clean_impulse",
        "kind": "fixed_tp_sl",
        "tp_r": 1.00,
        "sl_r": 0.50,
        "max_holding_bars": 8,
    },
    {
        "policy": "P3_standard",
        "kind": "fixed_tp_sl",
        "tp_r": 1.25,
        "sl_r": 0.75,
        "max_holding_bars": 12,
    },
    {
        "policy": "P4_trailing_runner",
        "kind": "trailing",
        "trail_start_r": 1.00,
        "trail_gap_r": 0.50,
        "sl_r": 0.75,
        "max_holding_bars": 16,
    },
    {
        "policy": "P5_wide_runner",
        "kind": "fixed_tp_sl",
        "tp_r": 2.00,
        "sl_r": 1.00,
        "max_holding_bars": 16,
    },
)

FROZEN_VALIDATION_REGIME_ALLOWLIST = {
    "observable_family",
    "base_score_decile",
    "liquidity_bin",
    "activity_liquidity_bin",
    "volatility_bin",
    "volatility_zscore_bin",
    "directional_vol_imbalance_bin",
    "market_dispersion_bin",
    "volatility_shape_bin",
    "aegmm_entropy_bin",
    "aegmm_distance_bin",
    "reconstruction_bin",
    "bad_mae_score_bin",
    "timeout_score_bin",
    "execres_score_bin",
    "exec_move_speed_bin",
    "exec_signal_to_spread_bin",
    "exec_slow_resolution_risk_bin",
    "exec_adverse_path_pressure_bin",
    "exec_opportunity_pressure_bin",
}

INCREMENTAL_VALUE_REGIME_ALLOWLIST = {
    "observable_family",
    "base_score_decile",
    "liquidity_bin",
    "activity_liquidity_bin",
    "volatility_bin",
    "volatility_zscore_bin",
    "directional_vol_imbalance_bin",
    "market_dispersion_bin",
    "volatility_shape_bin",
    "aegmm_entropy_bin",
    "aegmm_distance_bin",
    "reconstruction_bin",
    "bad_mae_score_bin",
    "timeout_score_bin",
    "execres_score_bin",
    "exec_move_speed_bin",
    "exec_signal_to_spread_bin",
    "exec_slow_resolution_risk_bin",
    "exec_adverse_path_pressure_bin",
    "exec_opportunity_pressure_bin",
}

EXECUTION_POLICY_REGIME_ALLOWLIST = INCREMENTAL_VALUE_REGIME_ALLOWLIST
REGIME_MATRIX_ALLOWLIST = FROZEN_VALIDATION_REGIME_ALLOWLIST


def _safe_num(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _float_array(values: Any) -> np.ndarray:
    if isinstance(values, pd.Series):
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    else:
        arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size:
        arr[~np.isfinite(arr)] = np.nan
    return arr


def _finite_array(values: Any) -> np.ndarray:
    arr = _float_array(values)
    return arr[np.isfinite(arr)]


def _nanmean_array(values: Any) -> float:
    arr = _finite_array(values)
    return float(arr.mean()) if arr.size else float("nan")


def _nanquantile_array(values: Any, q: float) -> float:
    arr = _finite_array(values)
    return float(np.quantile(arr, float(q))) if arr.size else float("nan")


def _rate_array(values: Any, *, observed: bool = False) -> float:
    arr = _float_array(values)
    if observed:
        arr = arr[np.isfinite(arr)]
        if not arr.size:
            return float("nan")
    else:
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if not arr.size:
        return float("nan") if observed else 0.0
    return float(np.clip(arr, 0.0, 1.0).mean())


def _mean_ci95_array(values: Any) -> tuple[float, float]:
    arr = _finite_array(values)
    n = int(arr.size)
    if n <= 1:
        mean = float(arr.mean()) if n else float("nan")
        return mean, mean
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(n))
    return mean - 1.96 * se, mean + 1.96 * se


def _rate_ci95_array(values: Any, *, observed: bool = False) -> tuple[float, float]:
    arr = _float_array(values)
    if observed:
        arr = arr[np.isfinite(arr)]
    else:
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if not arr.size:
        return float("nan"), float("nan")
    arr = np.clip(arr, 0.0, 1.0)
    p = float(arr.mean())
    se = math.sqrt(max(p * (1.0 - p), 0.0) / max(int(arr.size), 1))
    return max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


def _safe_mean(values: Any) -> float:
    return _nanmean_array(values)


def _safe_quantile(values: Any, q: float) -> float:
    return _nanquantile_array(values, q)


def _nanmean_or_nan(values: Any) -> float:
    return _nanmean_array(values)


def _rate(values: Any) -> float:
    return _rate_array(values, observed=False)


def _observed_rate(values: Any) -> float:
    return _rate_array(values, observed=True)


def _mean_ci95(values: Any) -> tuple[float, float]:
    return _mean_ci95_array(values)


def _rate_ci95(values: Any) -> tuple[float, float]:
    return _rate_ci95_array(values, observed=False)


def _observed_rate_ci95(values: Any) -> tuple[float, float]:
    return _rate_ci95_array(values, observed=True)


def _entropy_from_counts(counts: pd.Series) -> float:
    total = float(counts.sum())
    if total <= 0.0:
        return float("nan")
    p = counts.astype(float) / total
    p = p[p > 0.0]
    if len(p) <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / math.log(float(len(counts))))


def _hhi_from_counts(counts: pd.Series) -> float:
    total = float(counts.sum())
    if total <= 0.0:
        return float("nan")
    p = counts.astype(float) / total
    return float((p * p).sum())


def _bin_quantile(series: pd.Series, *, q: int, prefix: str) -> pd.Series:
    values = _safe_num(series)
    out = pd.Series(f"{prefix}_missing", index=series.index, dtype=object)
    finite = values.replace([np.inf, -np.inf], np.nan).notna()
    if int(finite.sum()) < max(10, q):
        return out
    try:
        bins = pd.qcut(values.loc[finite], q=q, labels=False, duplicates="drop")
    except ValueError:
        return out
    out.loc[finite] = [f"{prefix}_q{int(v)}" if pd.notna(v) else f"{prefix}_missing" for v in bins]
    return out


def _volatility_zscore_bin(series: pd.Series, *, prefix: str = "volatility_zscore") -> pd.Series:
    """Bucket sparse z-score state without arbitrary splitting of the zero mass."""
    values = _safe_num(series).replace([np.inf, -np.inf], np.nan)
    out = pd.Series(f"{prefix}_missing", index=series.index, dtype=object)
    finite = values.notna()
    if int(finite.sum()) < 10:
        return out
    eps = 1e-12
    out.loc[finite & values.lt(-eps)] = f"{prefix}_negative"
    out.loc[finite & values.abs().le(eps)] = f"{prefix}_flat"
    positive = finite & values.gt(eps)
    if int(positive.sum()) >= 20 and int(values.loc[positive].nunique(dropna=True)) >= 2:
        try:
            pos_bins = pd.qcut(values.loc[positive], q=min(3, int(values.loc[positive].nunique())), labels=False, duplicates="drop")
            out.loc[positive] = [f"{prefix}_positive_q{int(v)}" if pd.notna(v) else f"{prefix}_positive" for v in pos_bins]
        except ValueError:
            out.loc[positive] = f"{prefix}_positive"
    elif int(positive.sum()) > 0:
        out.loc[positive] = f"{prefix}_positive"
    return out


def _local_bin_quantile(
    frame: pd.DataFrame,
    value: pd.Series,
    *,
    group_cols: list[str],
    q: int,
    prefix: str,
    include_group_label: bool = True,
    min_group_rows: int | None = None,
) -> pd.Series:
    values = _safe_num(value)
    out = pd.Series(f"{prefix}_missing", index=frame.index, dtype=object)
    minimum_rows = int(min_group_rows if min_group_rows is not None else max(50, int(q) * 20))
    for key_values, idx in frame.groupby(group_cols, dropna=False).groups.items():
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        finite_count = int(values.loc[idx].replace([np.inf, -np.inf], np.nan).notna().sum())
        if finite_count < minimum_rows:
            out.loc[idx] = f"{prefix}_underpowered_source_side"
            continue
        local = _bin_quantile(values.loc[idx], q=q, prefix=prefix)
        if include_group_label:
            group_label = "__".join(str(v) for v in key_values)
            local = local.map(lambda bucket: f"{group_label}__{bucket}")
        out.loc[idx] = local.astype(str)
    return out


def _scope_regime_to_archetype_side(
    frame: pd.DataFrame,
    regime: pd.Series,
    *,
    group_cols: list[str],
    missing_label: str,
    min_group_rows: int | None = None,
    min_cell_rows: int | None = None,
) -> pd.Series:
    regime_values = regime.astype(str).where(regime.notna(), missing_label)
    out = pd.Series(f"{missing_label}_missing", index=frame.index, dtype=object)
    minimum_rows = int(min_group_rows) if min_group_rows is not None else 0
    minimum_cell_rows = int(min_cell_rows) if min_cell_rows is not None else 0
    for key_values, idx in frame.groupby(group_cols, dropna=False).groups.items():
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        if minimum_rows > 0 and len(idx) < minimum_rows:
            out.loc[idx] = f"{missing_label}_underpowered_source_side"
            continue
        group_label = "__".join(str(v) if pd.notna(v) else "missing" for v in key_values)
        local_regime = regime_values.loc[idx].astype(str)
        if minimum_cell_rows > 0:
            counts = local_regime.value_counts(dropna=False)
            rare = local_regime.map(counts).fillna(0).astype(int) < minimum_cell_rows
            if bool(rare.any()):
                out.loc[local_regime.index[rare]] = f"{missing_label}_rare_cluster_underpowered"
            common_idx = local_regime.index[~rare]
            out.loc[common_idx] = (group_label + "__" + local_regime.loc[common_idx]).astype(str)
        else:
            out.loc[idx] = (group_label + "__" + local_regime).astype(str)
    return out.astype(str)


def _candidate_summary_rows(frame: pd.DataFrame, mapping: dict[str, str], *, default_status: str = "available") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, col in mapping.items():
        counts = frame[col].astype(str).value_counts(dropna=False)
        rows.append(
            {
                "regime_model": name,
                "column": col,
                "rows": int(len(frame)),
                "regime_count": int(len(counts)),
                "min_regime_rows": int(counts.min()) if len(counts) else 0,
                "max_regime_rows": int(counts.max()) if len(counts) else 0,
                "entropy": _entropy_from_counts(counts),
                "hhi": _hhi_from_counts(counts),
                "month_coverage": int(frame.groupby(col)["month"].nunique().min()) if len(counts) else 0,
                "side_coverage": int(frame.groupby(col)["side_name"].nunique().min()) if len(counts) else 0,
                "status": default_status,
            }
        )
    return pd.DataFrame(rows)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input table format: {path}")


def _posterior_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        name = str(col)
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        if suffix.isdigit():
            cols.append(name)
    return sorted(cols, key=lambda name: int(str(name)[len(prefix) :]))


def _feature_family(col: str) -> str:
    name = str(col)
    lower = name.lower()
    if name in {"timestamp", "symbol", "side", "side_name", "ctx_side", "period", "month"}:
        return "key_or_split"
    if lower.startswith("ctx_exec_"):
        return "ctx_execution_proxy"
    if lower in {"source_tag", "source_family", "candidate_regime_family"} or lower.startswith("candidate_"):
        return "regime_candidate_feature"
    if lower in {
        "u_policy_net",
        "ret_net",
        "mae_norm",
        "mfe_norm",
        "is_timeout",
        "bad_mae_1r",
        "clean_positive",
        "dirty_positive",
        "oracle_top",
        "clean_oracle_top",
    } or lower.startswith("oracle_"):
        return "outcome_eval_only"
    if lower.startswith(
        (
            "__first_touch",
            "__mfe_",
            "__mae_",
            "__bars_to_",
            "__max_adverse",
            "__underwater",
            "__area_underwater",
            "__trailing_profit",
        )
    ):
        return "outcome_eval_only"
    if name.startswith("ctx_"):
        if any(token in lower for token in ("gmm", "cluster", "reconstruction", "mahal", "dae_b16")):
            return "ctx_ae_gmm_state"
        if "__meta_raw__" in lower:
            return "ctx_raw_meta_feature"
        if "_g_vol_" in lower:
            return "ctx_vol_regime_feature"
        return "ctx_meta_feature"
    if lower in {"primary_source_tag", "source_tag_reason_codes"} or lower.startswith("tag_"):
        return "semantic_source_tag"
    if lower.startswith("__regime_source_") or lower.endswith("_source_score") or lower.endswith("_candidate_score") or lower in {
        "quiet_continuation_score",
        "loud_breakout_impulse_score",
        "dirty_shock_avoid_score",
        "retest_reversal_score",
        "compression_release_score",
        "run_entry_score",
        "late_run_continuation_score",
    }:
            return "semantic_source_score"
    if lower.startswith("s22_bucket"):
        return "bucket_policy_diagnostic"
    if lower in {
        "trend_path_score",
        "shock_impulse_score",
        "execution_quality_score",
        "execution_risk_score",
        "oi_agreement_score",
        "location_quality_score",
        "pullback_retest_score",
        "compression_score",
        "volume_confirmation_score",
        "barrier_pressure_score",
    }:
        return "semantic_source_component"
    if any(token in lower for token in ("bad_mae_pred", "timeout_pred", "clean_path_pred", "dirty_positive", "_ts_pct")):
        return "prefit_path_risk_score"
    if lower in {"score", "selected_rank", "selected_count", "barrier", "prior_recent_source_strength"}:
        return "prefit_score_or_rank"
    if lower.startswith("selected_top") or "ranker_score" in lower or lower.endswith("_score") or lower.endswith("_rank_pct"):
        return "prefit_score_or_rank"
    if lower in {"label_arm", "weight_arm", "arm", "selector_variant", "model_feature_selector", "top_frac"}:
        return "source_or_run_tag"
    return "other_meta_input"


def _feature_schema_audit(frame: pd.DataFrame, *, outcome_cols: list[str]) -> pd.DataFrame:
    outcome_set = {str(c) for c in outcome_cols}
    regime_input_columns = {
        "ctx_gmm_entropy",
        "ctx_cluster_entropy",
        "ctx_state_spectral_top3_reconstruction_error",
        "lgbm_side_dirty_positive_bad_mae_pred",
        "lgbm_bad_mae_pred",
        "bad_mae_pred",
        "side_timeout_pred",
        "lgbm_timeout_pred",
        "timeout_pred",
        "lgbm_side_positive_clean_path_pred",
        "lgbm_clean_path_pred",
        "clean_path_pred",
        "selector_score",
        "base_model_score",
        "score",
    }
    regime_prefixes = (
        "ctx_exec_",
        "ctx_gmm_cluster_posterior_",
        "ctx_long_gmm_cluster_posterior_",
        "ctx_short_gmm_cluster_posterior_",
        "ctx_gmm_mahal_",
        "ctx_gmm_dist_center_",
        "ctx_long_gmm_mahal_",
        "ctx_short_gmm_mahal_",
    )
    rows: list[dict[str, Any]] = []
    n = max(int(len(frame)), 1)
    for col in frame.columns:
        series = frame[col]
        non_null = int(series.notna().sum())
        numeric = pd.api.types.is_numeric_dtype(series)
        used_as_regime_input = str(col) in regime_input_columns or str(col).startswith(regime_prefixes)
        rows.append(
            {
                "column": str(col),
                "feature_family": _feature_family(str(col)),
                "dtype": str(series.dtype),
                "non_null_rows": non_null,
                "missing_rate": float(1.0 - non_null / n),
                "numeric": bool(numeric),
                "used_as_regime_input": bool(used_as_regime_input),
                "used_as_outcome_eval": str(col) in outcome_set,
            }
        )
    return pd.DataFrame(rows)


def _posterior_argmax(frame: pd.DataFrame, cols: list[str], *, missing_label: str) -> pd.Series:
    out = pd.Series(missing_label, index=frame.index, dtype=object)
    if not cols:
        return out
    values = frame[cols].apply(pd.to_numeric, errors="coerce")
    finite = values.notna().any(axis=1)
    if not bool(finite.any()):
        return out
    out.loc[finite] = [f"cluster_{int(v)}" for v in np.nanargmax(values.loc[finite].to_numpy(float), axis=1)]
    return out


def _first_present(frame: pd.DataFrame, cols: tuple[str, ...], *, default: float = np.nan) -> pd.Series:
    for col in cols:
        if col in frame.columns:
            return _safe_num(frame[col])
    return pd.Series(default, index=frame.index, dtype=np.float32)


def _mean_present(frame: pd.DataFrame, cols: tuple[str, ...], *, default: float = np.nan) -> pd.Series:
    present = [col for col in cols if col in frame.columns]
    if not present:
        return pd.Series(default, index=frame.index, dtype=np.float32)
    values = frame[present].apply(pd.to_numeric, errors="coerce")
    return values.mean(axis=1)


def _rank01(values: pd.Series, *, invert: bool = False) -> pd.Series:
    """Cross-sectional rank proxy for combining heterogeneous pre-entry signals."""
    numeric = _safe_num(values).replace([np.inf, -np.inf], np.nan)
    out = pd.Series(np.nan, index=numeric.index, dtype=np.float32)
    finite = numeric.notna()
    if int(finite.sum()) == 0:
        return out
    ranked = numeric.loc[finite].rank(method="average", pct=True)
    if invert:
        ranked = 1.0 - ranked
    out.loc[finite] = ranked.astype(np.float32)
    return out


def _mean_rank01(signals: tuple[pd.Series, ...]) -> pd.Series:
    if not signals:
        return pd.Series(dtype=np.float32)
    ranked = [_rank01(signal) for signal in signals]
    values = pd.concat(ranked, axis=1)
    return values.mean(axis=1).astype(np.float32)


def _finite_or_zero(values: pd.Series) -> pd.Series:
    return _safe_num(values).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def _rolling_symbol_activity(frame: pd.DataFrame, *, lookback_hours: int = 168) -> pd.Series:
    """Prior candidate count per symbol over a rolling window.

    This is a deployable activity/liquidity proxy for ledgers without direct
    volume or order-book depth: each row only sees earlier rows for the same
    symbol.
    """
    out = pd.Series(np.nan, index=frame.index, dtype=np.float32)
    if "timestamp" not in frame.columns or "symbol" not in frame.columns:
        return out
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    window_ns = int(lookback_hours) * 60 * 60 * 1_000_000_000
    for _symbol, idx in frame.groupby("symbol", dropna=False).groups.items():
        idx_list = list(idx)
        valid_idx = [i for i in idx_list if pd.notna(timestamps.loc[i])]
        if not valid_idx:
            continue
        ordered = pd.Index(valid_idx)[np.argsort(timestamps.loc[valid_idx].astype("int64").to_numpy())]
        ns = timestamps.loc[ordered].astype("int64").to_numpy()
        left = np.searchsorted(ns, ns - window_ns, side="left")
        counts = np.arange(len(ns), dtype=np.float32) - left.astype(np.float32)
        out.loc[ordered] = counts
    return out


def _combine_bucket_labels(left: pd.Series, right: pd.Series, *, prefix: str) -> pd.Series:
    left_s = left.astype(str).where(left.notna(), "missing")
    right_s = right.astype(str).where(right.notna(), "missing")
    both_missing = left_s.str.endswith("_missing") & right_s.str.endswith("_missing")
    out = (prefix + "__" + left_s + "__" + right_s).astype(object)
    out.loc[both_missing] = f"{prefix}_missing"
    return out.astype(str)


def _install_if_signal(temp: pd.DataFrame, name: str, values: pd.Series) -> None:
    numeric = _safe_num(values).replace([np.inf, -np.inf], np.nan)
    if int(numeric.notna().sum()) < 5:
        return
    if int(numeric.nunique(dropna=True)) < 2:
        return
    temp[name] = numeric.astype(np.float32)


def _derive_semantic_source_scores(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Derive source tags from live-computable meta pre-FS features when absent."""
    if "primary_source_tag" in frame.columns:
        return frame, {"status": "existing_primary_source_tag"}
    if any(col in frame.columns for col in SOURCE_SCORE_COLUMNS.values()):
        return frame, {"status": "existing_source_scores"}
    try:
        from scripts.materialize_candidate_source_tags import (
            ARCHETYPE_COLS,
            COMPONENT_COLS,
            DEFAULT_CONFIG,
            TAG_COLS,
            load_config,
            materialize_source_tags,
        )
    except Exception as exc:
        return frame, {"status": "semantic_source_materializer_import_failed", "error": str(exc)}

    temp = pd.DataFrame(
        {
            "__ts__": frame["timestamp"],
            "__symbol__": frame["symbol"].astype(str),
            "__side__": np.where(frame["side_name"].astype(str).eq("short"), -1.0, 1.0),
            "score": _safe_num(frame["score"]),
        },
        index=frame.index,
    )
    side = _safe_num(temp["__side__"]).fillna(1.0)
    ret_1h = _mean_present(frame, ("ctx_ret1h_G_VOL_0", "ctx_ret1h_G_VOL_1"), default=np.nan)
    directional_ret_1h = side * ret_1h
    abs_ret_1h = ret_1h.abs()
    volatility = _first_present(frame, ("ctx___meta_raw__volatility_zscore", "ctx_volatility_zscore_G_VOL_0"), default=np.nan)
    oi_resid = _first_present(frame, ("ctx___meta_raw__asset_minus_mkt_oi_1d_peer_resid",), default=np.nan)
    autocorr = _first_present(frame, ("ctx___meta_raw__return_autocorr_48", "ctx_return_autocorr_48_G_VOL_0"), default=np.nan)
    dist_ema20 = _first_present(frame, ("ctx_dist_ema20_atr",), default=np.nan)
    price_z = _first_present(frame, ("ctx_zscore_price_200",), default=np.nan)
    breadth = _first_present(frame, ("ctx_pct_assets_above_vwap",), default=np.nan)
    spread = _first_present(frame, ("ctx_median_spread_bps",), default=np.nan)
    xs_dispersion = _first_present(frame, ("ctx_xs_dispersion__ffd_amihud_06",), default=np.nan)
    near_ema20 = -dist_ema20.abs()
    low_volatility = -volatility
    low_dispersion = -xs_dispersion
    overextension = price_z.abs()
    directional_location = side * dist_ema20

    feature_map = {
        "trend_strength_percentile": _mean_present(frame, ("ctx_adx_7", "ctx_adx_10", "ctx_adx_14"), default=np.nan),
        "regime_trend_score": breadth,
        "trend_alignment_1_3_6": autocorr,
        "trend_acceleration": directional_ret_1h,
        "resid_strength": directional_ret_1h,
        "impulse": directional_ret_1h,
        "speed": abs_ret_1h,
        "shock_12h": volatility,
        "shock_vol_ratio": volatility,
        "jump_intensity": abs_ret_1h,
        "breakout_24h": directional_location,
        "pct_breakout_t": directional_location,
        "range_24h_pct": volatility,
        "range_12h_pct": volatility,
        "spread_proxy_hl_range_bps_robust_z": spread,
        "spread_proxy_abs_return_bps_robust_z": spread,
        "median_spread_bps": spread,
        "xasset_ob_liquidity_peer_resid": _first_present(frame, ("ctx_xasset_ob_liquidity_peer_resid",), default=np.nan),
        "oi_rank": _first_present(frame, ("ctx_oi_rank",), default=np.nan),
        "oi_chg_2h": side * oi_resid,
        "oi_up_agree": side * oi_resid,
        "oi_expansion_compression_balance_24h": oi_resid,
        "dist_ema20_atr": dist_ema20,
        "trend_overextension_z": overextension,
        "pct_extreme": overextension,
        "pullback_depth": near_ema20,
        "retest_quality": near_ema20,
        "mr_potential": -directional_location.abs(),
        "mean_reversion_score": near_ema20,
        "atr_compression_ratio": low_volatility,
        "compression_score": low_volatility,
        "vol_compression": low_volatility,
        "vol_z_30_calm": low_volatility,
        "prior_range": low_dispersion,
        "rolling_range_20": low_dispersion,
        "realized_volatility_24h": volatility,
        "vol_z_base": volatility,
        "vol_price_spread": _first_present(frame, ("ctx_vol_price_spread",), default=np.nan),
        "up_vol_6": _first_present(frame, ("ctx_up_vol",), default=np.nan),
        "dn_vol_6": _first_present(frame, ("ctx_dn_vol_6",), default=np.nan),
        "up_barrier_pressure_daily_vwap": overextension + spread.fillna(0.0),
        "down_barrier_pressure_daily_vwap": overextension + spread.fillna(0.0),
        "rejection_proxy": overextension,
        "tail_fail": overextension,
        "trap_strength": overextension,
    }
    for name, values in feature_map.items():
        _install_if_signal(temp, name, values)

    try:
        cfg = load_config(DEFAULT_CONFIG)
    except Exception:
        cfg = {}
    cfg = dict(cfg)
    cfg["timestamp_col"] = "__ts__"
    cfg["symbol_col"] = "__symbol__"
    cfg["side_col"] = "__side__"
    cfg["proxy_score_columns"] = ["score"]
    source, report = materialize_source_tags(temp, cfg)
    out = frame.copy()
    copied_cols = [
        col
        for col in list(COMPONENT_COLS) + list(ARCHETYPE_COLS) + list(TAG_COLS) + ["primary_source_tag", "source_tag_reason_codes"]
        if col in source.columns
    ]
    for col in copied_cols:
        out[col] = source[col].to_numpy()
    for source_name, base_col in SOURCE_SCORE_BASE_COLUMNS.items():
        if base_col in source.columns:
            out[SOURCE_SCORE_COLUMNS[source_name]] = _safe_num(source[base_col]).fillna(0.0).astype(np.float32).to_numpy()
    return out, {
        "status": "semantic_source_tags_materialized_from_meta_prefeature_ctx",
        "input_feature_count": int(len([c for c in temp.columns if c not in {"__ts__", "__symbol__", "__side__", "score"}])),
        "source_columns_used": report.get("source_columns_used", []),
        "primary_source_tag_counts": out["primary_source_tag"].astype(str).value_counts(dropna=False).to_dict()
        if "primary_source_tag" in out.columns
        else {},
    }


def _standardize_meta_feature_ledger(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _read_table(path).copy()
    if frame.empty:
        raise ValueError(f"meta feature ledger is empty: {path}")
    if "timestamp" not in frame.columns:
        raise ValueError(f"meta feature ledger missing timestamp column: {path}")
    if "symbol" not in frame.columns:
        raise ValueError(f"meta feature ledger missing symbol column: {path}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["month"] = (
        frame["timestamp"].dt.tz_convert(None).dt.to_period("M").astype(str)
        if getattr(frame["timestamp"].dt, "tz", None) is not None
        else frame["timestamp"].dt.to_period("M").astype(str)
    )
    frame["symbol"] = frame["symbol"].astype(str)
    side_raw = _first_present(frame, ("side", "ctx_side", "__side__"), default=1.0).fillna(1.0)
    frame["side_name"] = np.where(side_raw < 0.0, "short", "long")
    if "selector_score" in frame.columns:
        frame["score"] = _safe_num(frame["selector_score"])
    elif "base_model_score" in frame.columns:
        frame["score"] = _safe_num(frame["base_model_score"])
    elif "score" in frame.columns:
        frame["score"] = _safe_num(frame["score"])
    else:
        raise ValueError("meta feature ledger needs selector_score, base_model_score, or score")
    if "selected_top10" not in frame.columns:
        frame["selected_top10"] = 1
    if "selected_top20" not in frame.columns:
        frame["selected_top20"] = 1
    if "selected_top30" not in frame.columns:
        frame["selected_top30"] = 1
    ctx_cols = [str(c) for c in frame.columns if str(c).startswith("ctx_")]
    ae_gmm_cols = [
        str(c)
        for c in ctx_cols
        if any(token in str(c).lower() for token in ("gmm", "cluster", "reconstruction", "mahal"))
    ]
    outcome_cols = [
        c
        for c in (
            "u_policy_net",
            "ret_net",
            "mae_norm",
            "mfe_norm",
            "is_timeout",
            "bad_mae_1r",
            "clean_positive",
            "dirty_positive",
            "oracle_top",
            "clean_oracle_top",
        )
        if c in frame.columns
    ]
    report = {
        "input_path": str(path),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "ctx_columns": int(len(ctx_cols)),
        "ae_gmm_ctx_columns": int(len(ae_gmm_cols)),
        "outcome_columns": outcome_cols,
        "feature_input_contract": "meta_model_pre_feature_selection_ledger",
    }
    return frame, report


def _read_label_frames(labels_path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(labels_path.glob("*.parquet")):
        df = pd.read_parquet(path)
        needed = [
            "__ts__",
            "__symbol__",
            "side_name",
            "__side__",
            "__regime_family__",
            "__archetype_label_family__",
            "__archetype_policy_key__",
            "__first_touch_capture_net__",
            "__first_touch_round_trip_cost__",
            "__first_touch_hit__",
            "__first_touch_stop__",
            "__first_touch_timeout__",
            "__first_touch_bar__",
            "__first_touch_mae_to_sl__",
            "__first_touch_mfe_to_tp__",
            "__first_touch_full_path_mae_to_sl__",
            "__first_touch_full_path_mfe_to_tp__",
            "__mfe_1r_before_mae_1r__",
            "__mae_1r_before_mfe_1r__",
            "__max_adverse_before_mfe_1r__",
            "__underwater_bars_before_mfe_1r__",
            "__underwater_fraction_before_mfe_1r__",
            "spread_proxy_abs_return_bps_robust_z",
            "spread_proxy_hl_range_bps_robust_z",
            "vol_price_spread",
            "xasset_ob_liquidity_peer_resid",
            "__meta_raw__volatility_zscore",
            "log_quote_volume",
        ]
        needed.extend(SOURCE_SCORE_COLUMNS.values())
        keep = [col for col in dict.fromkeys(needed) if col in df.columns]
        slim = df.loc[:, keep].copy()
        slim["label_file"] = path.name
        frames.append(slim)
    if not frames:
        raise RuntimeError(f"No parquet labels found in {labels_path}")
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["__ts__"], errors="coerce")
    out["symbol"] = out["__symbol__"].astype(str)
    if "side_name" not in out.columns or out["side_name"].isna().all():
        side = _safe_num(out.get("__side__", pd.Series(1.0, index=out.index))).fillna(1.0)
        out["side_name"] = np.where(side < 0.0, "short", "long")
    out["side_name"] = out["side_name"].astype(str)
    return out


def _read_path_order_label_frames(labels_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    if labels_path is None or not labels_path.exists():
        return pd.DataFrame(), {"status": "missing", "path": str(labels_path)}
    for path in sorted(labels_path.glob("*.parquet")):
        try:
            cols = pd.read_parquet(path, columns=None).columns
            keep = [col for col in PATH_ORDER_LABEL_COLUMNS if col in cols]
            if not {"__ts__", "__symbol__"}.issubset(set(keep)):
                continue
            df = pd.read_parquet(path, columns=keep).copy()
        except Exception as exc:
            frames.append(pd.DataFrame({"__read_error__": [str(exc)], "__path__": [str(path)]}))
            continue
        df["label_file"] = path.name
        frames.append(df)
    valid = [df for df in frames if "__read_error__" not in df.columns and not df.empty]
    errors = [df for df in frames if "__read_error__" in df.columns]
    if not valid:
        return pd.DataFrame(), {
            "status": "empty",
            "path": str(labels_path),
            "files_seen": int(len(list(labels_path.glob("*.parquet")))),
            "read_errors": pd.concat(errors, ignore_index=True).to_dict("records") if errors else [],
        }
    out = pd.concat(valid, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["__ts__"], utc=True, errors="coerce")
    out["symbol"] = out["__symbol__"].astype(str)
    if "side_name" not in out.columns or out["side_name"].isna().all():
        side = _safe_num(out.get("__side__", pd.Series(1.0, index=out.index))).fillna(1.0)
        out["side_name"] = np.where(side < 0.0, "short", "long")
    out["side_name"] = out["side_name"].astype(str)
    duplicate_keys = int(out.duplicated(["timestamp", "symbol", "side_name"]).sum())
    if duplicate_keys:
        out = out.drop_duplicates(["timestamp", "symbol", "side_name"], keep="last")
    report = {
        "status": "loaded",
        "path": str(labels_path),
        "files_seen": int(len(list(labels_path.glob("*.parquet")))),
        "rows": int(len(out)),
        "duplicate_keys_dropped": duplicate_keys,
        "columns_loaded": [str(c) for c in out.columns],
        "read_errors": pd.concat(errors, ignore_index=True).to_dict("records") if errors else [],
    }
    return out, report


def _merge_path_order_labels(frame: pd.DataFrame, labels_path: Path | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if labels_path is None:
        return frame, {"status": "disabled"}
    labels, report = _read_path_order_label_frames(labels_path)
    if labels.empty:
        return frame, report
    keep = [
        col
        for col in labels.columns
        if col in {"timestamp", "symbol", "side_name"}
        or col.startswith("__bars_to_")
        or col.startswith("__mfe_")
        or col.startswith("__mae_")
        or col.startswith("__max_adverse")
        or col.startswith("__underwater")
        or col in {
            "__first_touch_bar__",
            "__first_touch_same_bar_both__",
            "__trailing_profit_activation_bar__",
            "__area_underwater_before_mfe_1r__",
        }
    ]
    slim = labels.loc[:, keep].copy()
    before_cols = set(frame.columns)
    merged = frame.merge(slim, on=["timestamp", "symbol", "side_name"], how="left", suffixes=("", "__path_label"))
    matched = int(merged["__mfe_1r_before_mae_1r__"].notna().sum()) if "__mfe_1r_before_mae_1r__" in merged.columns else 0
    coalesced_columns: list[str] = []
    for col in list(merged.columns):
        if not str(col).endswith("__path_label"):
            continue
        base = str(col).removesuffix("__path_label")
        if base in merged.columns:
            merged[base] = merged[base].where(merged[base].notna(), merged[col])
            merged = merged.drop(columns=[col])
            coalesced_columns.append(base)
    report.update(
        {
            "merged_rows": int(len(merged)),
            "matched_rows": matched,
            "matched_rate": float(matched / max(len(merged), 1)),
            "new_columns_added": sorted(str(c) for c in set(merged.columns) - before_cols),
            "coalesced_columns": sorted(set(coalesced_columns)),
            "merge_role": "path_order_diagnostics_only_existing_economic_columns_not_overwritten",
        }
    )
    return merged, report


def _merge_ledger_labels(ledger_path: Path, labels_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    ledger = pd.read_parquet(ledger_path).copy()
    ledger["timestamp"] = pd.to_datetime(ledger["timestamp"], errors="coerce")
    ledger["symbol"] = ledger["symbol"].astype(str)
    ledger["side_name"] = ledger["side_name"].astype(str)
    labels = _read_label_frames(labels_path)
    dup_labels = int(labels.duplicated(["timestamp", "symbol", "side_name"]).sum())
    if dup_labels:
        labels = labels.drop_duplicates(["timestamp", "symbol", "side_name"], keep="last")
    merged = ledger.merge(labels, on=["timestamp", "symbol", "side_name"], how="left", suffixes=("", "_label"))
    report = {
        "ledger_rows": int(len(ledger)),
        "label_rows": int(len(labels)),
        "label_duplicate_keys_dropped": int(dup_labels),
        "merged_rows": int(len(merged)),
        "matched_label_rows": int(merged["__first_touch_capture_net__"].notna().sum())
        if "__first_touch_capture_net__" in merged.columns
        else 0,
    }
    return merged, report


def _derive_source_tags(frame: pd.DataFrame, *, min_score: float) -> tuple[pd.Series, str]:
    if "primary_source_tag" in frame.columns:
        primary = frame["primary_source_tag"].astype(str).fillna("unknown")
        if primary.ne("").any():
            return primary.where(primary.ne(""), "unknown"), "semantic_primary_source_tag"
    present = {name: col for name, col in SOURCE_SCORE_COLUMNS.items() if col in frame.columns}
    if not present:
        for col in ("source_tag", "archetype", "selector_variant", "label_arm", "model_feature_selector"):
            if col in frame.columns:
                return frame[col].astype(str).fillna("unknown"), f"fallback:{col}"
        return pd.Series("unknown", index=frame.index, dtype=object), "fallback:unknown"
    scores = pd.DataFrame({name: _safe_num(frame[col]).fillna(0.0) for name, col in present.items()}, index=frame.index)
    best = scores.idxmax(axis=1).astype(str)
    max_score = scores.max(axis=1)
    return best.where(max_score >= float(min_score), "ambiguous_none").astype(str), "observable_source_scores"


def _add_regime_candidates(frame: pd.DataFrame, *, source_min_score: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    out, semantic_source_report = _derive_semantic_source_scores(frame.copy())
    source_tags, source_tag_source = _derive_source_tags(out, min_score=source_min_score)
    out["source_tag"] = source_tags
    if "__archetype_label_family__" in out.columns:
        out["source_family"] = out["__archetype_label_family__"].astype(str)
    elif "archetype" in out.columns:
        out["source_family"] = out["archetype"].astype(str)
    else:
        out["source_family"] = out["source_tag"].astype(str)
    if "__regime_family__" in out.columns:
        out["candidate_regime_family"] = out["__regime_family__"].astype(str)
    elif "archetype" in out.columns:
        out["candidate_regime_family"] = out["archetype"].astype(str)
    else:
        out["candidate_regime_family"] = out["source_family"].astype(str)
    archetype_side_groups = ["source_family", "side_name"]
    out["candidate_base_score_decile"] = (
        out.groupby(["month", "side_name"], dropna=False)["score"]
        .transform(lambda s: _bin_quantile(s, q=10, prefix="base_score"))
        .astype(str)
    )
    out["candidate_archetype_side_base_score_decile"] = _local_bin_quantile(
        out,
        out["score"],
        group_cols=archetype_side_groups,
        q=10,
        prefix="local_base_score",
    )
    if "spread_proxy_abs_return_bps_robust_z" in out.columns:
        out["candidate_spread_bin"] = (
            out.groupby(["month"], dropna=False)["spread_proxy_abs_return_bps_robust_z"]
            .transform(lambda s: _bin_quantile(s, q=4, prefix="spread"))
            .astype(str)
        )
    else:
        out["candidate_spread_bin"] = "spread_missing"
    liquidity_signal = _first_present(
        out,
        (
            "xasset_ob_liquidity_peer_resid",
            "ctx_xasset_ob_liquidity_peer_resid",
            "log_quote_volume",
            "ctx_log_quote_volume",
            "ctx_quote_volume",
            "ctx_volume_usd",
        ),
        default=np.nan,
    )
    liquidity_source = "direct_liquidity_or_volume"
    if int(liquidity_signal.replace([np.inf, -np.inf], np.nan).notna().sum()) < 10:
        spread_proxy = _first_present(
            out,
            ("ctx_median_spread_bps", "median_spread_bps", "spread_proxy_abs_return_bps_robust_z"),
            default=np.nan,
        )
        liquidity_signal = -spread_proxy
        liquidity_source = "inverse_spread_proxy_fallback"
    if int(liquidity_signal.replace([np.inf, -np.inf], np.nan).notna().sum()) >= 10:
        out["candidate_liquidity_bin"] = (
            liquidity_signal.groupby(out["month"], dropna=False)
            .transform(lambda s: _bin_quantile(s, q=4, prefix="liquidity"))
            .astype(str)
        )
        out["candidate_archetype_side_liquidity_bin"] = _local_bin_quantile(
            out,
            liquidity_signal,
            group_cols=archetype_side_groups,
            q=4,
            prefix="local_liquidity",
        )
    else:
        out["candidate_liquidity_bin"] = "liquidity_missing"
        out["candidate_archetype_side_liquidity_bin"] = "local_liquidity_missing"
        liquidity_source = "missing"
    out["candidate_liquidity_signal_source"] = liquidity_source

    activity_liquidity_signal = _rolling_symbol_activity(out, lookback_hours=168)
    if int(activity_liquidity_signal.replace([np.inf, -np.inf], np.nan).notna().sum()) >= 10:
        out["candidate_activity_liquidity_bin"] = (
            activity_liquidity_signal.groupby(out["month"], dropna=False)
            .transform(lambda s: _bin_quantile(s, q=4, prefix="activity_liquidity"))
            .astype(str)
        )
        out["candidate_archetype_side_activity_liquidity_bin"] = _local_bin_quantile(
            out,
            activity_liquidity_signal,
            group_cols=archetype_side_groups,
            q=4,
            prefix="local_activity_liquidity",
        )
    else:
        out["candidate_activity_liquidity_bin"] = "activity_liquidity_missing"
        out["candidate_archetype_side_activity_liquidity_bin"] = "local_activity_liquidity_missing"

    volatility_signal = _mean_present(
        out,
        (
            "ctx_dn_vol_6",
            "ctx_up_vol",
            "ctx_dist_ema20_atr",
        ),
        default=np.nan,
    )
    volatility_source = "path_volatility_composite"
    if int(volatility_signal.replace([np.inf, -np.inf], np.nan).notna().sum()) < 10:
        volatility_signal = _mean_present(
            out,
            (
                "__meta_raw__volatility_zscore",
                "ctx___meta_raw__volatility_zscore",
                "ctx_volatility_zscore_G_VOL_0",
                "ctx_volatility_zscore_G_VOL_1",
                "realized_volatility_24h",
                "ctx_realized_volatility_24h",
            ),
            default=np.nan,
        )
        volatility_source = "direct_or_ctx_volatility_fallback"
    if int(volatility_signal.replace([np.inf, -np.inf], np.nan).notna().sum()) >= 10:
        out["candidate_volatility_bin"] = _bin_quantile(volatility_signal, q=4, prefix="volatility").astype(str)
        out["candidate_archetype_side_volatility_bin"] = _local_bin_quantile(
            out,
            volatility_signal,
            group_cols=archetype_side_groups,
            q=4,
            prefix="local_volatility",
        )
        out["candidate_volatility_signal_source"] = volatility_source
    else:
        out["candidate_volatility_bin"] = "volatility_missing"
        out["candidate_archetype_side_volatility_bin"] = "local_volatility_missing"
        out["candidate_volatility_signal_source"] = "missing"

    up_vol = _first_present(out, ("ctx_up_vol",), default=np.nan)
    down_vol = _first_present(out, ("ctx_dn_vol_6",), default=np.nan)
    side_multiplier = pd.Series(np.where(out["side_name"].astype(str).eq("short"), -1.0, 1.0), index=out.index)
    directional_vol_imbalance = side_multiplier * (up_vol - down_vol)
    if int(directional_vol_imbalance.replace([np.inf, -np.inf], np.nan).notna().sum()) >= 10:
        out["candidate_directional_vol_imbalance_bin"] = _bin_quantile(
            directional_vol_imbalance,
            q=4,
            prefix="dir_vol_imbalance",
        ).astype(str)
        out["candidate_archetype_side_directional_vol_imbalance_bin"] = _local_bin_quantile(
            out,
            directional_vol_imbalance,
            group_cols=archetype_side_groups,
            q=4,
            prefix="local_dir_vol_imbalance",
        )
    else:
        out["candidate_directional_vol_imbalance_bin"] = "dir_vol_imbalance_missing"
        out["candidate_archetype_side_directional_vol_imbalance_bin"] = "local_dir_vol_imbalance_missing"

    dispersion_signal = _first_present(out, ("ctx_xs_dispersion__ffd_amihud_06",), default=np.nan)
    if int(dispersion_signal.replace([np.inf, -np.inf], np.nan).notna().sum()) >= 10:
        out["candidate_market_dispersion_bin"] = _bin_quantile(
            dispersion_signal,
            q=4,
            prefix="market_dispersion",
        ).astype(str)
        out["candidate_archetype_side_market_dispersion_bin"] = _local_bin_quantile(
            out,
            dispersion_signal,
            group_cols=archetype_side_groups,
            q=4,
            prefix="local_market_dispersion",
        )
    else:
        out["candidate_market_dispersion_bin"] = "market_dispersion_missing"
        out["candidate_archetype_side_market_dispersion_bin"] = "local_market_dispersion_missing"

    volatility_zscore_signal = _mean_present(
        out,
        (
            "__meta_raw__volatility_zscore",
            "ctx___meta_raw__volatility_zscore",
            "ctx_volatility_zscore_G_VOL_0",
            "ctx_volatility_zscore_G_VOL_1",
            "realized_volatility_24h",
            "ctx_realized_volatility_24h",
        ),
        default=np.nan,
    )
    if int(volatility_zscore_signal.replace([np.inf, -np.inf], np.nan).notna().sum()) >= 10:
        out["candidate_volatility_zscore_bin"] = _volatility_zscore_bin(volatility_zscore_signal).astype(str)
    else:
        out["candidate_volatility_zscore_bin"] = "volatility_zscore_missing"

    out["candidate_volatility_shape_bin"] = _combine_bucket_labels(
        out["candidate_volatility_bin"],
        out["candidate_directional_vol_imbalance_bin"],
        prefix="vol_shape",
    )

    global_post = _posterior_columns(out, "ctx_gmm_cluster_posterior_")
    long_post = _posterior_columns(out, "ctx_long_gmm_cluster_posterior_")
    short_post = _posterior_columns(out, "ctx_short_gmm_cluster_posterior_")
    out["candidate_aegmm_global_argmax"] = _posterior_argmax(out, global_post, missing_label="global_missing")
    long_argmax = _posterior_argmax(out, long_post, missing_label="long_missing")
    short_argmax = _posterior_argmax(out, short_post, missing_label="short_missing")
    out["candidate_aegmm_side_argmax"] = np.where(out["side_name"].eq("short"), short_argmax, long_argmax)
    out["candidate_archetype_side_aegmm_global_argmax"] = _scope_regime_to_archetype_side(
        out,
        out["candidate_aegmm_global_argmax"],
        group_cols=archetype_side_groups,
        missing_label="global_missing",
        min_group_rows=100,
        min_cell_rows=30,
    )
    out["candidate_archetype_side_aegmm_side_argmax"] = _scope_regime_to_archetype_side(
        out,
        pd.Series(out["candidate_aegmm_side_argmax"], index=out.index),
        group_cols=archetype_side_groups,
        missing_label="side_missing",
        min_group_rows=100,
        min_cell_rows=30,
    )
    entropy = np.where(
        out["side_name"].eq("short"),
        _first_present(out, ("ctx_short_gmm_entropy", "ctx_short_cluster_entropy", "ctx_gmm_entropy"), default=np.nan),
        _first_present(out, ("ctx_long_gmm_entropy", "ctx_long_cluster_entropy", "ctx_gmm_entropy"), default=np.nan),
    )
    out["candidate_aegmm_entropy_bin"] = _bin_quantile(pd.Series(entropy, index=out.index), q=4, prefix="aegmm_entropy")
    out["candidate_archetype_side_aegmm_entropy_bin"] = _local_bin_quantile(
        out,
        pd.Series(entropy, index=out.index),
        group_cols=archetype_side_groups,
        q=4,
        prefix="local_aegmm_entropy",
    )
    global_dist_cols = [
        c for c in out.columns if str(c).startswith(("ctx_gmm_mahal_", "ctx_gmm_dist_center_"))
    ]
    long_dist_cols = [
        c for c in out.columns if str(c).startswith(("ctx_long_gmm_mahal_", "ctx_long_gmm_dist_center_"))
    ]
    short_dist_cols = [
        c for c in out.columns if str(c).startswith(("ctx_short_gmm_mahal_", "ctx_short_gmm_dist_center_"))
    ]
    side_dist = pd.Series(np.nan, index=out.index, dtype=np.float64)
    side_is_short = out["side_name"].astype(str).eq("short")
    if long_dist_cols:
        long_dist = out[long_dist_cols].apply(pd.to_numeric, errors="coerce").replace(0.0, np.nan).min(axis=1)
        side_dist.loc[~side_is_short] = long_dist.loc[~side_is_short]
    if short_dist_cols:
        short_dist = out[short_dist_cols].apply(pd.to_numeric, errors="coerce").replace(0.0, np.nan).min(axis=1)
        side_dist.loc[side_is_short] = short_dist.loc[side_is_short]
    if global_dist_cols:
        global_dist = out[global_dist_cols].apply(pd.to_numeric, errors="coerce").replace(0.0, np.nan).min(axis=1)
        side_dist = side_dist.fillna(global_dist)
    if int(side_dist.notna().sum()) >= 10:
        dist = side_dist
        out["candidate_aegmm_distance_bin"] = _bin_quantile(dist, q=4, prefix="aegmm_distance")
    else:
        out["candidate_aegmm_distance_bin"] = "aegmm_distance_missing"
    if "ctx_state_spectral_top3_reconstruction_error" in out.columns:
        out["candidate_reconstruction_bin"] = _bin_quantile(
            out["ctx_state_spectral_top3_reconstruction_error"],
            q=4,
            prefix="reconstruction",
        )
        out["candidate_archetype_side_reconstruction_bin"] = _local_bin_quantile(
            out,
            out["ctx_state_spectral_top3_reconstruction_error"],
            group_cols=archetype_side_groups,
            q=4,
            prefix="local_reconstruction",
        )
    else:
        out["candidate_reconstruction_bin"] = "reconstruction_missing"
        out["candidate_archetype_side_reconstruction_bin"] = "local_reconstruction_missing"
    bad_mae_score = _first_present(
        out,
        ("lgbm_side_dirty_positive_bad_mae_pred", "lgbm_bad_mae_pred", "bad_mae_pred"),
        default=np.nan,
    )
    timeout_score = _first_present(out, ("side_timeout_pred", "lgbm_timeout_pred", "timeout_pred"), default=np.nan)
    execres_score = _first_present(
        out,
        ("lgbm_side_positive_clean_path_pred", "lgbm_clean_path_pred", "clean_path_pred"),
        default=np.nan,
    )
    out["candidate_bad_mae_score_bin"] = _bin_quantile(
        bad_mae_score,
        q=4,
        prefix="bad_mae_score",
    )
    out["candidate_archetype_side_bad_mae_score_bin"] = _local_bin_quantile(
        out,
        bad_mae_score,
        group_cols=archetype_side_groups,
        q=4,
        prefix="local_bad_mae_score",
    )
    out["candidate_timeout_score_bin"] = _bin_quantile(
        timeout_score,
        q=4,
        prefix="timeout_score",
    )
    out["candidate_archetype_side_timeout_score_bin"] = _local_bin_quantile(
        out,
        timeout_score,
        group_cols=archetype_side_groups,
        q=4,
        prefix="local_timeout_score",
    )
    out["candidate_execres_score_bin"] = _bin_quantile(
        execres_score,
        q=4,
        prefix="execres_score",
    )
    out["candidate_archetype_side_execres_score_bin"] = _local_bin_quantile(
        out,
        execres_score,
        group_cols=archetype_side_groups,
        q=4,
        prefix="local_execres_score",
    )

    spread_bps = _first_present(
        out,
        (
            "ctx_median_spread_bps",
            "median_spread_bps",
            "ctx_spread_bps",
            "spread_bps",
            "spread_proxy_abs_return_bps_robust_z",
        ),
        default=np.nan,
    ).clip(lower=0.0)
    dist_ema_atr = _first_present(out, ("ctx_dist_ema20_atr", "dist_ema20_atr"), default=np.nan).abs()
    ret_1h = _mean_present(out, ("ctx_ret1h_G_VOL_0", "ctx_ret1h_G_VOL_1", "ret1h"), default=np.nan).abs()
    ret_autocorr = _first_present(
        out,
        ("ctx___meta_raw__return_autocorr_48", "__meta_raw__return_autocorr_48", "return_autocorr_48"),
        default=np.nan,
    ).abs()
    cluster_speed = _first_present(out, ("ctx_cluster_speed", "ctx_state_speed", "cluster_speed"), default=np.nan).abs()
    cluster_accel = _first_present(
        out,
        (
            "ctx_cluster_acceleration",
            "ctx_cluster_entropy_accel_1",
            "ctx_gmm_posterior_accel_1",
            "ctx_state_acceleration",
            "cluster_acceleration",
        ),
        default=np.nan,
    ).abs()
    reconstruction_error = _first_present(
        out,
        ("ctx_state_spectral_top3_reconstruction_error", "ctx_ae_reconstruction_error", "ae_reconstruction_error"),
        default=np.nan,
    )

    move_speed_proxy = _mean_rank01(
        (
            volatility_signal.abs(),
            volatility_zscore_signal.abs(),
            dist_ema_atr,
            ret_1h,
            ret_autocorr,
            cluster_speed,
            cluster_accel,
        )
    )
    spread_pressure = _rank01(spread_bps)
    liquidity_rank = _rank01(liquidity_signal)
    liquidity_pressure = _rank01(liquidity_signal, invert=True)
    aegmm_uncertainty = _mean_rank01(
        (
            pd.Series(entropy, index=out.index),
            side_dist,
            reconstruction_error,
        )
    )
    model_risk_pressure = _mean_rank01((bad_mae_score, timeout_score))
    adverse_path_pressure = _mean_rank01((bad_mae_score, dispersion_signal.abs(), aegmm_uncertainty))
    slow_resolution_risk = pd.concat(
        [
            (1.0 - _finite_or_zero(move_speed_proxy)).clip(0.0, 1.0),
            _finite_or_zero(timeout_score).clip(0.0, 1.0),
            _finite_or_zero(liquidity_pressure).clip(0.0, 1.0),
            _finite_or_zero(aegmm_uncertainty).clip(0.0, 1.0),
        ],
        axis=1,
    ).mean(axis=1)
    signal_to_spread = (
        _finite_or_zero(move_speed_proxy)
        / (1.0 + _finite_or_zero(spread_pressure) + (_finite_or_zero(spread_bps) / 100.0))
    ).astype(np.float32)
    opportunity_pressure = (
        0.45 * _finite_or_zero(move_speed_proxy)
        + 0.25 * _finite_or_zero(execres_score).clip(0.0, 1.0)
        + 0.20 * _finite_or_zero(signal_to_spread)
        - 0.25 * _finite_or_zero(adverse_path_pressure).clip(0.0, 1.0)
        - 0.20 * _finite_or_zero(slow_resolution_risk).clip(0.0, 1.0)
        - 0.10 * _finite_or_zero(spread_pressure).clip(0.0, 1.0)
    ).astype(np.float32)

    out["ctx_exec_spread_bps_proxy"] = spread_bps.astype(np.float32)
    out["ctx_exec_liquidity_rank_proxy"] = liquidity_rank.astype(np.float32)
    out["ctx_exec_spread_pressure_proxy"] = spread_pressure.astype(np.float32)
    out["ctx_exec_volatility_rank_proxy"] = _rank01(volatility_signal).astype(np.float32)
    out["ctx_exec_move_speed_proxy"] = move_speed_proxy.astype(np.float32)
    out["ctx_exec_signal_to_spread_proxy"] = signal_to_spread.astype(np.float32)
    out["ctx_exec_aegmm_uncertainty_proxy"] = aegmm_uncertainty.astype(np.float32)
    out["ctx_exec_model_risk_pressure_proxy"] = model_risk_pressure.astype(np.float32)
    out["ctx_exec_adverse_path_pressure_proxy"] = adverse_path_pressure.astype(np.float32)
    out["ctx_exec_slow_resolution_risk_proxy"] = slow_resolution_risk.astype(np.float32)
    out["ctx_exec_opportunity_pressure_proxy"] = opportunity_pressure.astype(np.float32)

    exec_proxy_bins = {
        "exec_move_speed_bin": ("candidate_exec_move_speed_bin", out["ctx_exec_move_speed_proxy"]),
        "exec_signal_to_spread_bin": ("candidate_exec_signal_to_spread_bin", out["ctx_exec_signal_to_spread_proxy"]),
        "exec_slow_resolution_risk_bin": (
            "candidate_exec_slow_resolution_risk_bin",
            out["ctx_exec_slow_resolution_risk_proxy"],
        ),
        "exec_adverse_path_pressure_bin": (
            "candidate_exec_adverse_path_pressure_bin",
            out["ctx_exec_adverse_path_pressure_proxy"],
        ),
        "exec_opportunity_pressure_bin": (
            "candidate_exec_opportunity_pressure_bin",
            out["ctx_exec_opportunity_pressure_proxy"],
        ),
    }
    for label, (column, values) in exec_proxy_bins.items():
        out[column] = _bin_quantile(values, q=4, prefix=label.replace("_bin", ""))
        out[f"candidate_archetype_side_{label}"] = _local_bin_quantile(
            out,
            values,
            group_cols=archetype_side_groups,
            q=4,
            prefix=f"local_{label.replace('_bin', '')}",
        )
    ae_status = "available" if (global_post or long_post or short_post) else "missing_required_columns"

    candidate_mapping = {
        "observable_family": "candidate_regime_family",
        "base_score_decile": "candidate_base_score_decile",
        "archetype_side_base_score_decile": "candidate_archetype_side_base_score_decile",
        "spread_bin": "candidate_spread_bin",
        "liquidity_bin": "candidate_liquidity_bin",
        "archetype_side_liquidity_bin": "candidate_archetype_side_liquidity_bin",
        "activity_liquidity_bin": "candidate_activity_liquidity_bin",
        "archetype_side_activity_liquidity_bin": "candidate_archetype_side_activity_liquidity_bin",
        "volatility_bin": "candidate_volatility_bin",
        "archetype_side_volatility_bin": "candidate_archetype_side_volatility_bin",
        "volatility_zscore_bin": "candidate_volatility_zscore_bin",
        "directional_vol_imbalance_bin": "candidate_directional_vol_imbalance_bin",
        "archetype_side_directional_vol_imbalance_bin": "candidate_archetype_side_directional_vol_imbalance_bin",
        "market_dispersion_bin": "candidate_market_dispersion_bin",
        "archetype_side_market_dispersion_bin": "candidate_archetype_side_market_dispersion_bin",
        "volatility_shape_bin": "candidate_volatility_shape_bin",
        "aegmm_global_argmax": "candidate_aegmm_global_argmax",
        "archetype_side_aegmm_global_argmax": "candidate_archetype_side_aegmm_global_argmax",
        "aegmm_side_argmax": "candidate_aegmm_side_argmax",
        "archetype_side_aegmm_side_argmax": "candidate_archetype_side_aegmm_side_argmax",
        "aegmm_entropy_bin": "candidate_aegmm_entropy_bin",
        "archetype_side_aegmm_entropy_bin": "candidate_archetype_side_aegmm_entropy_bin",
        "aegmm_distance_bin": "candidate_aegmm_distance_bin",
        "reconstruction_bin": "candidate_reconstruction_bin",
        "archetype_side_reconstruction_bin": "candidate_archetype_side_reconstruction_bin",
        "bad_mae_score_bin": "candidate_bad_mae_score_bin",
        "archetype_side_bad_mae_score_bin": "candidate_archetype_side_bad_mae_score_bin",
        "timeout_score_bin": "candidate_timeout_score_bin",
        "archetype_side_timeout_score_bin": "candidate_archetype_side_timeout_score_bin",
        "execres_score_bin": "candidate_execres_score_bin",
        "archetype_side_execres_score_bin": "candidate_archetype_side_execres_score_bin",
        "exec_move_speed_bin": "candidate_exec_move_speed_bin",
        "archetype_side_exec_move_speed_bin": "candidate_archetype_side_exec_move_speed_bin",
        "exec_signal_to_spread_bin": "candidate_exec_signal_to_spread_bin",
        "archetype_side_exec_signal_to_spread_bin": "candidate_archetype_side_exec_signal_to_spread_bin",
        "exec_slow_resolution_risk_bin": "candidate_exec_slow_resolution_risk_bin",
        "archetype_side_exec_slow_resolution_risk_bin": "candidate_archetype_side_exec_slow_resolution_risk_bin",
        "exec_adverse_path_pressure_bin": "candidate_exec_adverse_path_pressure_bin",
        "archetype_side_exec_adverse_path_pressure_bin": (
            "candidate_archetype_side_exec_adverse_path_pressure_bin"
        ),
        "exec_opportunity_pressure_bin": "candidate_exec_opportunity_pressure_bin",
        "archetype_side_exec_opportunity_pressure_bin": "candidate_archetype_side_exec_opportunity_pressure_bin",
    }
    candidate_summary = _candidate_summary_rows(out, candidate_mapping)
    aegmm_mask = candidate_summary["regime_model"].astype(str).str.startswith("aegmm") | candidate_summary[
        "regime_model"
    ].astype(str).eq("reconstruction_bin")
    candidate_summary.loc[aegmm_mask, "status"] = ae_status
    return out, candidate_summary, {
        "source_tag_source": source_tag_source,
        "semantic_source_report": semantic_source_report,
        "execution_proxy_report": {
            "columns": [
                "ctx_exec_spread_bps_proxy",
                "ctx_exec_liquidity_rank_proxy",
                "ctx_exec_spread_pressure_proxy",
                "ctx_exec_volatility_rank_proxy",
                "ctx_exec_move_speed_proxy",
                "ctx_exec_signal_to_spread_proxy",
                "ctx_exec_aegmm_uncertainty_proxy",
                "ctx_exec_model_risk_pressure_proxy",
                "ctx_exec_adverse_path_pressure_proxy",
                "ctx_exec_slow_resolution_risk_proxy",
                "ctx_exec_opportunity_pressure_proxy",
            ],
            "contract": "Derived only from pre-entry context, AE/GMM state, spread/liquidity/volatility proxies, and prefit path-risk scores.",
        },
    }


def _metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["u"] = _first_present(out, ("__first_touch_capture_net__", "first_touch_net", "u_policy_net", "ret_net"), default=0.0).fillna(0.0)
    out["cost"] = _first_present(out, ("__first_touch_round_trip_cost__", "round_trip_cost"), default=0.0).fillna(0.0)
    out["ev_after_cost"] = out["u"]
    out["gross_u"] = out["u"] + out["cost"]
    if "__first_touch_mae_to_sl__" in out.columns or "first_touch_mae_to_sl" in out.columns:
        out["bad_mae"] = _first_present(out, ("__first_touch_mae_to_sl__", "first_touch_mae_to_sl"), default=0.0).ge(1.0).astype(float)
    elif "bad_mae_1r" in out.columns:
        out["bad_mae"] = pd.Series(out["bad_mae_1r"]).astype(bool).astype(float)
    else:
        out["bad_mae"] = 0.0
    out["timeout"] = _first_present(out, ("__first_touch_timeout__", "first_touch_timeout", "is_timeout"), default=0.0).fillna(0.0).clip(0.0, 1.0)
    out["stop"] = _first_present(out, ("__first_touch_stop__", "first_touch_stop", "full_sl"), default=0.0).fillna(0.0).clip(0.0, 1.0)
    out["clean_exec"] = (
        (out["u"] > 0.0)
        & out["bad_mae"].eq(0.0)
        & out["timeout"].eq(0.0)
        & out["stop"].eq(0.0)
    ).astype(float)
    if "clean_positive" in out.columns:
        out["clean_exec"] = pd.Series(out["clean_positive"]).astype(bool).astype(float)
    out["dirty_positive"] = ((out["u"] > 0.0) & ((out["bad_mae"] > 0.0) | (out["timeout"] > 0.0))).astype(float)
    if "dirty_positive" in out.columns:
        out["dirty_positive"] = pd.Series(out["dirty_positive"]).astype(bool).astype(float)
    out["mfe_before_mae_1r"] = _safe_num(out.get("__mfe_1r_before_mae_1r__", pd.Series(np.nan, index=out.index)))
    out["mae_before_mfe_1r"] = _safe_num(out.get("__mae_1r_before_mfe_1r__", pd.Series(np.nan, index=out.index)))
    out["cusum_good_first"] = _first_present(
        out,
        ("__cusum_good_first__", "cusum_good_first", "__mfe_1r_before_mae_1r__"),
        default=np.nan,
    ).fillna(out["mfe_before_mae_1r"])
    out["cusum_bad_first"] = _first_present(
        out,
        ("__cusum_bad_first__", "cusum_bad_first", "__mae_1r_before_mfe_1r__"),
        default=np.nan,
    ).fillna(out["mae_before_mfe_1r"])
    out["max_adverse_before_mfe_1r"] = _safe_num(
        out.get("__max_adverse_before_mfe_1r__", pd.Series(np.nan, index=out.index))
    )
    out["underwater_bars_before_mfe_1r"] = _safe_num(
        out.get("__underwater_bars_before_mfe_1r__", pd.Series(np.nan, index=out.index))
    )
    out["bars_to_exit"] = _first_present(
        out,
        ("__first_touch_bar__", "first_touch_bar", "bars_to_exit", "__bars_to_exit__"),
        default=np.nan,
    )
    out["mae_r"] = _first_present(out, ("__first_touch_full_path_mae_to_sl__", "__first_touch_mae_to_sl__", "mae_norm"), default=np.nan)
    out["mfe_r"] = _first_present(out, ("__first_touch_full_path_mfe_to_tp__", "__first_touch_mfe_to_tp__", "mfe_norm"), default=np.nan)
    return out


def _leaf_embedding_feature_columns(frame: pd.DataFrame, *, max_features: int = 96) -> list[str]:
    excluded = set(DERIVED_OUTCOME_COLUMNS) | {
        "timestamp",
        "month",
        "symbol",
        "side_name",
        "row_pos",
        "source_tag",
        "source_family",
        "candidate_regime_family",
        "feature_gap_risk",
        "s22_bucket_quality_rank_pct",
        "s22_bucket_relaxed_pass_count",
        "s22_bucket_strict_pass_count",
    }
    families = {
        "ctx_ae_gmm_state",
        "ctx_meta_feature",
        "ctx_raw_meta_feature",
        "ctx_vol_regime_feature",
        "semantic_source_score",
        "semantic_source_component",
        "prefit_path_risk_score",
        "prefit_score_or_rank",
        "other_meta_input",
    }
    candidates: list[tuple[str, float, int]] = []
    for col in frame.columns:
        name = str(col)
        if name in excluded or name.startswith("candidate_") or _is_direct_outcome_column(name, set()):
            continue
        if name.startswith("s22_bucket"):
            continue
        if _feature_family(name) not in families and name not in {"score", "barrier", "prior_recent_source_strength"}:
            continue
        series = _safe_num(frame[col]).replace([np.inf, -np.inf], np.nan)
        non_null = int(series.notna().sum())
        if non_null < 50 or int(series.nunique(dropna=True)) < 3:
            continue
        candidates.append((name, float(series.var(skipna=True) or 0.0), non_null))
    candidates.sort(key=lambda item: (item[2], item[1]), reverse=True)
    return [name for name, _var, _n in candidates[: int(max_features)]]


def _assign_oof_leaf_cluster_regime(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    output_col: str,
    prefix: str,
    n_clusters: int = 6,
) -> tuple[pd.Series, dict[str, Any]]:
    out = pd.Series(f"{prefix}_no_prior_train", index=frame.index, dtype=object)
    report: dict[str, Any] = {
        "target_col": target_col,
        "output_col": output_col,
        "feature_cols": list(feature_cols),
        "folds": [],
        "status": "not_run",
    }
    if not feature_cols:
        report["status"] = "missing_features"
        return out, report
    try:
        import lightgbm as lgb
        from scipy import sparse
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.preprocessing import OneHotEncoder
    except Exception as exc:
        report["status"] = "dependency_missing"
        report["error"] = str(exc)
        return out, report

    months = sorted(str(m) for m in frame["month"].dropna().unique())
    if len(months) < 2:
        report["status"] = "insufficient_months"
        return out, report
    for test_month in months[1:]:
        train_idx = frame.index[frame["month"].astype(str) < str(test_month)]
        test_idx = frame.index[frame["month"].astype(str).eq(str(test_month))]
        if len(train_idx) < 500 or len(test_idx) == 0:
            report["folds"].append(
                {
                    "test_month": str(test_month),
                    "status": "skipped_insufficient_rows",
                    "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                }
            )
            continue
        y = _safe_num(frame.loc[train_idx, target_col]).fillna(0.0).clip(0.0, 1.0).astype(int)
        if int(y.nunique(dropna=True)) < 2:
            report["folds"].append(
                {
                    "test_month": str(test_month),
                    "status": "skipped_single_class",
                    "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                }
            )
            continue
        x_train = frame.loc[train_idx, feature_cols].apply(pd.to_numeric, errors="coerce")
        x_test = frame.loc[test_idx, feature_cols].apply(pd.to_numeric, errors="coerce")
        med = x_train.replace([np.inf, -np.inf], np.nan).median(axis=0).fillna(0.0)
        x_train = x_train.replace([np.inf, -np.inf], np.nan).fillna(med).astype(np.float32)
        x_test = x_test.replace([np.inf, -np.inf], np.nan).fillna(med).astype(np.float32)
        try:
            model = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=72,
                learning_rate=0.045,
                num_leaves=10,
                max_depth=4,
                min_child_samples=80,
                subsample=0.85,
                colsample_bytree=0.75,
                reg_lambda=2.0,
                n_jobs=1,
                random_state=1701,
                verbose=-1,
            )
            model.fit(x_train, y)
            train_leaf = model.predict(x_train, pred_leaf=True)
            test_leaf = model.predict(x_test, pred_leaf=True)
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
            train_encoded = encoder.fit_transform(train_leaf)
            test_encoded = encoder.transform(test_leaf)
            clusters = max(2, min(int(n_clusters), int(len(train_idx) // 250), int(train_encoded.shape[0])))
            if clusters < 2:
                raise ValueError("not enough rows for leaf clustering")
            clusterer = MiniBatchKMeans(
                n_clusters=clusters,
                random_state=1701,
                batch_size=min(2048, max(256, int(len(train_idx)))),
                n_init=5,
            )
            clusterer.fit(train_encoded if sparse.issparse(train_encoded) else np.asarray(train_encoded))
            labels = clusterer.predict(test_encoded if sparse.issparse(test_encoded) else np.asarray(test_encoded))
            out.loc[test_idx] = [f"{prefix}_{int(label)}" for label in labels]
            report["folds"].append(
                {
                    "test_month": str(test_month),
                    "status": "assigned",
                    "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                    "positive_rate_train": float(y.mean()),
                    "features": int(len(feature_cols)),
                    "trees": int(train_leaf.shape[1]) if getattr(train_leaf, "ndim", 1) > 1 else 1,
                    "clusters": int(clusters),
                    "assigned_rows": int(len(test_idx)),
                }
            )
        except Exception as exc:
            report["folds"].append(
                {
                    "test_month": str(test_month),
                    "status": "failed",
                    "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                    "error": str(exc),
                }
            )
    assigned = int(out.ne(f"{prefix}_no_prior_train").sum())
    report["assigned_rows"] = assigned
    report["assigned_rate"] = float(assigned / max(len(out), 1))
    report["status"] = "available" if assigned > 0 else "no_assignments"
    return out.astype(str), report


def _add_oof_leaf_regime_candidates(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, str]]:
    out = frame.copy()
    feature_cols = _leaf_embedding_feature_columns(out)
    reports: dict[str, Any] = {"feature_count": int(len(feature_cols)), "feature_cols": feature_cols}
    clean_cluster, clean_report = _assign_oof_leaf_cluster_regime(
        out,
        feature_cols=feature_cols,
        target_col="clean_exec",
        output_col="candidate_lgbm_leaf_clean_cluster_oof",
        prefix="leaf_clean_cluster",
    )
    dirty_cluster, dirty_report = _assign_oof_leaf_cluster_regime(
        out,
        feature_cols=feature_cols,
        target_col="dirty_positive",
        output_col="candidate_lgbm_leaf_dirty_cluster_oof",
        prefix="leaf_dirty_cluster",
    )
    out["candidate_lgbm_leaf_clean_cluster_oof"] = clean_cluster
    out["candidate_lgbm_leaf_dirty_cluster_oof"] = dirty_cluster
    out["candidate_archetype_side_lgbm_leaf_clean_cluster_oof"] = _scope_regime_to_archetype_side(
        out,
        out["candidate_lgbm_leaf_clean_cluster_oof"],
        group_cols=["source_family", "side_name"],
        missing_label="leaf_clean_cluster",
        min_group_rows=100,
        min_cell_rows=30,
    )
    out["candidate_archetype_side_lgbm_leaf_dirty_cluster_oof"] = _scope_regime_to_archetype_side(
        out,
        out["candidate_lgbm_leaf_dirty_cluster_oof"],
        group_cols=["source_family", "side_name"],
        missing_label="leaf_dirty_cluster",
        min_group_rows=100,
        min_cell_rows=30,
    )
    mapping = {
        "lgbm_leaf_clean_cluster_oof": "candidate_lgbm_leaf_clean_cluster_oof",
        "archetype_side_lgbm_leaf_clean_cluster_oof": "candidate_archetype_side_lgbm_leaf_clean_cluster_oof",
        "lgbm_leaf_dirty_cluster_oof": "candidate_lgbm_leaf_dirty_cluster_oof",
        "archetype_side_lgbm_leaf_dirty_cluster_oof": "candidate_archetype_side_lgbm_leaf_dirty_cluster_oof",
    }
    summary = _candidate_summary_rows(out, mapping, default_status="available_oof_prior_month_leaf_embedding")
    reports["clean_leaf"] = clean_report
    reports["dirty_leaf"] = dirty_report
    reports["leakage_contract"] = "LightGBM leaf models and KMeans clusterers are fitted on prior months only; held-out month regimes use frozen transforms."
    return out, summary, reports, mapping


def _source_concentration(frame: pd.DataFrame, *, regime_model: str, regime_col: str) -> pd.DataFrame:
    global_source = frame["source_tag"].astype(str).value_counts(normalize=True, dropna=False)
    rows: list[dict[str, Any]] = []
    total = float(len(frame))
    for regime, group in frame.groupby(regime_col, dropna=False):
        counts = group["source_tag"].astype(str).value_counts(dropna=False)
        entropy = _entropy_from_counts(counts)
        hhi = _hhi_from_counts(counts)
        for source, n in counts.items():
            p_source_regime = float(n / max(len(group), 1))
            p_regime_source = float(n / max((frame["source_tag"].astype(str) == str(source)).sum(), 1))
            base = float(global_source.get(source, np.nan))
            rows.append(
                {
                    "regime_model": regime_model,
                    "regime": str(regime),
                    "source_tag": str(source),
                    "rows": int(n),
                    "regime_rows": int(len(group)),
                    "coverage": float(len(group) / max(total, 1.0)),
                    "p_source_given_regime": p_source_regime,
                    "p_regime_given_source": p_regime_source,
                    "lift_source_regime": float(p_source_regime / base) if base and np.isfinite(base) else float("nan"),
                    "source_entropy": entropy,
                    "source_hhi": hhi,
                    "month_coverage": int(group["month"].nunique()),
                    "side_coverage": int(group["side_name"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def _group_path_metrics(group: pd.DataFrame) -> dict[str, Any]:
    u = _float_array(group["u"])
    ev = _float_array(group["ev_after_cost"])
    gross = np.clip(np.nan_to_num(_float_array(group["gross_u"]), nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    clean = np.clip(np.nan_to_num(_float_array(group["clean_exec"]), nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    dirty = _float_array(group["dirty_positive"])
    bad_mae = _float_array(group["bad_mae"])
    timeout = _float_array(group["timeout"])
    mae_r = _float_array(group["mae_r"])
    mfe_r = _float_array(group["mfe_r"])
    mfe_before = _float_array(group["mfe_before_mae_1r"])
    mae_before = _float_array(group["mae_before_mfe_1r"])
    cusum_good = _float_array(group["cusum_good_first"])
    cusum_bad = _float_array(group["cusum_bad_first"])
    max_adverse = _float_array(group["max_adverse_before_mfe_1r"])
    underwater = _float_array(group["underwater_bars_before_mfe_1r"])
    bars_to_exit = _float_array(group["bars_to_exit"])
    denom = float(gross.sum())
    ev_low, ev_high = _mean_ci95_array(ev)
    bad_mae_low, bad_mae_high = _rate_ci95_array(bad_mae)
    timeout_low, timeout_high = _rate_ci95_array(timeout)
    clean_low, clean_high = _rate_ci95_array(clean)
    mfe_first_low, mfe_first_high = _rate_ci95_array(mfe_before, observed=True)
    mae_first_low, mae_first_high = _rate_ci95_array(mae_before, observed=True)
    return {
        "rows": int(len(group)),
        "symbols": int(group["symbol"].nunique(dropna=True)),
        "mean_u": _nanmean_array(u),
        "median_u": _nanquantile_array(u, 0.50),
        "p10_u": _nanquantile_array(u, 0.10),
        "p25_u": _nanquantile_array(u, 0.25),
        "ev_after_cost": _nanmean_array(ev),
        "ev_after_cost_ci95_low": ev_low,
        "ev_after_cost_ci95_high": ev_high,
        "ev_weighted_clean_precision": float((clean * gross).sum() / denom) if denom > 0.0 else float("nan"),
        "clean_executable_rate": _rate_array(clean),
        "clean_executable_ci95_low": clean_low,
        "clean_executable_ci95_high": clean_high,
        "dirty_positive_rate": _rate_array(dirty),
        "bad_mae_rate": _rate_array(bad_mae),
        "bad_mae_ci95_low": bad_mae_low,
        "bad_mae_ci95_high": bad_mae_high,
        "timeout_rate": _rate_array(timeout),
        "timeout_ci95_low": timeout_low,
        "timeout_ci95_high": timeout_high,
        "mean_mae_r": _nanmean_array(mae_r),
        "p90_mae_r": _nanquantile_array(mae_r, 0.90),
        "mean_mfe_r": _nanmean_array(mfe_r),
        "p90_mfe_r": _nanquantile_array(mfe_r, 0.90),
        "mfe_before_mae_1r_rate": _rate_array(mfe_before, observed=True),
        "mfe_before_mae_1r_ci95_low": mfe_first_low,
        "mfe_before_mae_1r_ci95_high": mfe_first_high,
        "mae_1r_before_mfe_1r_rate": _rate_array(mae_before, observed=True),
        "mae_1r_before_mfe_1r_ci95_low": mae_first_low,
        "mae_1r_before_mfe_1r_ci95_high": mae_first_high,
        "cusum_good_first_rate": _rate_array(cusum_good, observed=True),
        "cusum_bad_first_rate": _rate_array(cusum_bad, observed=True),
        "max_adverse_before_mfe_1r": _nanmean_array(max_adverse),
        "underwater_bars_before_mfe_1r": _nanmean_array(underwater),
        "mean_bars_to_exit": _nanmean_array(bars_to_exit),
        "p90_bars_to_exit": _nanquantile_array(bars_to_exit, 0.90),
    }


def _outcome_matrix(frame: pd.DataFrame, *, regime_model: str, regime_col: str) -> pd.DataFrame:
    rows = []
    for (source, regime, side), group in frame.groupby(["source_tag", regime_col, "side_name"], dropna=False):
        row = {
            "regime_model": regime_model,
            "source_tag": str(source),
            "regime": str(regime),
            "side": str(side),
            "scope": "source_regime_side",
        }
        row.update(_group_path_metrics(group))
        row["month_coverage"] = int(group["month"].nunique())
        rows.append(row)
    for (source, regime), group in frame.groupby(["source_tag", regime_col], dropna=False):
        row = {
            "regime_model": regime_model,
            "source_tag": str(source),
            "regime": str(regime),
            "side": "all",
            "scope": "source_regime",
        }
        row.update(_group_path_metrics(group))
        row["month_coverage"] = int(group["month"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def _spearman(x: pd.Series, y: pd.Series) -> float:
    xx = _safe_num(x)
    yy = _safe_num(y)
    mask = xx.notna() & yy.notna()
    if int(mask.sum()) < 10:
        return float("nan")
    return float(xx.loc[mask].rank().corr(yy.loc[mask].rank()))


def _pearson(x: pd.Series, y: pd.Series) -> float:
    xx = _safe_num(x)
    yy = _safe_num(y)
    mask = xx.notna() & yy.notna()
    if int(mask.sum()) < 10:
        return float("nan")
    if int(xx.loc[mask].nunique(dropna=True)) < 2 or int(yy.loc[mask].nunique(dropna=True)) < 2:
        return float("nan")
    return float(xx.loc[mask].corr(yy.loc[mask]))


def _top_metrics(group: pd.DataFrame, frac: float) -> dict[str, Any]:
    if group.empty:
        return {}
    n = max(1, int(math.ceil(len(group) * float(frac))))
    scores = _float_array(group["score"])
    score_rank = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    if n >= len(group):
        order = np.argsort(-score_rank, kind="mergesort")
    else:
        candidate = np.argpartition(-score_rank, n - 1)[:n]
        order = candidate[np.argsort(-score_rank[candidate], kind="mergesort")]
    top = group.iloc[order]
    metrics = _group_path_metrics(top)
    return {
        f"top{int(frac * 100)}_rows": int(len(top)),
        f"top{int(frac * 100)}_ev": metrics["ev_after_cost"],
        f"top{int(frac * 100)}_clean_precision": metrics["clean_executable_rate"],
        f"top{int(frac * 100)}_bad_mae": metrics["bad_mae_rate"],
        f"top{int(frac * 100)}_timeout": metrics["timeout_rate"],
        f"top{int(frac * 100)}_mfe_before_mae": metrics["mfe_before_mae_1r_rate"],
        f"top{int(frac * 100)}_mae_before_mfe": metrics["mae_1r_before_mfe_1r_rate"],
        f"top{int(frac * 100)}_cusum_good_first": metrics["cusum_good_first_rate"],
        f"top{int(frac * 100)}_cusum_bad_first": metrics["cusum_bad_first_rate"],
        f"top{int(frac * 100)}_mean_bars_to_exit": metrics["mean_bars_to_exit"],
        f"top{int(frac * 100)}_ev_weighted_clean_precision": metrics["ev_weighted_clean_precision"],
    }


def _learnability_matrix(frame: pd.DataFrame, *, regime_model: str, regime_col: str, frontier_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (source, regime, side), group in frame.groupby(["source_tag", regime_col, "side_name"], dropna=False):
        frontier = group[group[frontier_col].eq(1)] if frontier_col in group.columns else group
        for scope, sample in (("eligible", group), ("frontier", frontier)):
            row = {
                "regime_model": regime_model,
                "source_tag": str(source),
                "regime": str(regime),
                "side": str(side),
                "scope": scope,
                "rows": int(len(sample)),
                "months": int(sample["month"].nunique()) if len(sample) else 0,
                "proxy_spearman_ic": _spearman(sample["score"], sample["u"]),
                "proxy_pearson_ic": _pearson(sample["score"], sample["u"]),
                "score_clean_gap": _safe_mean(sample.loc[sample["clean_exec"].eq(1.0), "score"])
                - _safe_mean(sample.loc[sample["dirty_positive"].eq(1.0), "score"]),
            }
            for frac in (0.05, 0.10, 0.20):
                row.update(_top_metrics(sample, frac))
            if len(sample):
                monthly = sample.groupby("month", dropna=False)["score"].count().astype(float)
                for month, month_group in sample.groupby("month", dropna=False):
                    monthly.loc[month] = _top_metrics(month_group, 0.10).get("top10_ev", np.nan)
                row["rank_stability_positive_months"] = int((monthly > 0.0).sum())
                row["rank_stability_months"] = int(monthly.notna().sum())
                row["rank_stability_worst_month_top10_ev"] = float(monthly.min()) if len(monthly.dropna()) else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def _compact_frontier_learnability(
    frame: pd.DataFrame,
    *,
    regime_model: str,
    regime_col: str,
    frontier_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (source, regime), group in frame.groupby(["source_tag", regime_col], dropna=False):
        sample = group[group[frontier_col].eq(1)] if frontier_col in group.columns else group
        row = {
            "regime_model": regime_model,
            "source_tag": str(source),
            "regime": str(regime),
            "scope": "frontier",
            "rows": int(len(sample)),
            "months": int(sample["month"].nunique()) if len(sample) else 0,
            "proxy_spearman_ic": _spearman(sample["score"], sample["u"]),
            "proxy_pearson_ic": _pearson(sample["score"], sample["u"]),
            "score_clean_gap": _safe_mean(sample.loc[sample["clean_exec"].eq(1.0), "score"])
            - _safe_mean(sample.loc[sample["dirty_positive"].eq(1.0), "score"]),
        }
        row.update(_top_metrics(sample, 0.10))
        if len(sample):
            monthly = []
            for _month, month_group in sample.groupby("month", dropna=False):
                monthly.append(_top_metrics(month_group, 0.10).get("top10_ev", np.nan))
            monthly_series = _safe_num(monthly)
            row["rank_stability_positive_months"] = int((monthly_series > 0.0).sum())
            row["rank_stability_months"] = int(monthly_series.notna().sum())
            row["rank_stability_worst_month_top10_ev"] = float(monthly_series.min()) if len(monthly_series.dropna()) else float("nan")
        else:
            row["rank_stability_positive_months"] = 0
            row["rank_stability_months"] = 0
            row["rank_stability_worst_month_top10_ev"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _fit_predict_month_holdout(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> np.ndarray | None:
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return None
    y = train["clean_exec"].astype(int)
    if int(y.nunique(dropna=True)) < 2:
        return None
    x_train = pd.get_dummies(train[cols].astype(str), dummy_na=True)
    x_test = pd.get_dummies(test[cols].astype(str), dummy_na=True).reindex(columns=x_train.columns, fill_value=0)
    model = LogisticRegression(max_iter=300, class_weight="balanced", solver="liblinear", random_state=17)
    model.fit(x_train, y)
    return model.predict_proba(x_test)[:, 1]


def _incremental_value_tests(frame: pd.DataFrame, candidate_map: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    months = sorted(str(m) for m in frame["month"].dropna().unique())
    if len(months) < 2:
        return pd.DataFrame(rows)
    tested_map = {k: v for k, v in candidate_map.items() if k in INCREMENTAL_VALUE_REGIME_ALLOWLIST}
    for regime_model, regime_col in tested_map.items():
        work = frame.copy()
        work["_source_regime_key"] = work["source_tag"].astype(str) + "||" + work[regime_col].astype(str)
        feature_sets = {
            "source_only": ["source_tag", "side_name"],
            "regime_only": [regime_col, "side_name"],
            "source_plus_regime": ["source_tag", regime_col, "side_name"],
            "source_x_regime": ["_source_regime_key", "side_name"],
        }
        for test_month in months:
            train = work[work["month"].astype(str) != test_month]
            test = work[work["month"].astype(str) == test_month]
            if train.empty or test.empty:
                continue
            for feature_set, cols in feature_sets.items():
                pred = _fit_predict_month_holdout(train, test, cols)
                if pred is None:
                    continue
                scored = test.copy()
                scored["_incremental_score"] = pred
                top = scored.sort_values("_incremental_score", ascending=False).head(max(1, int(math.ceil(len(scored) * 0.10))))
                metrics = _group_path_metrics(top)
                rows.append(
                    {
                        "regime_model": regime_model,
                        "feature_set": feature_set,
                        "test_month": test_month,
                        "train_rows": int(len(train)),
                        "test_rows": int(len(test)),
                        "top10_rows": int(len(top)),
                        "top10_ev": metrics["ev_after_cost"],
                        "top10_clean_precision": metrics["clean_executable_rate"],
                        "top10_ev_weighted_clean_precision": metrics["ev_weighted_clean_precision"],
                        "top10_bad_mae": metrics["bad_mae_rate"],
                        "top10_timeout": metrics["timeout_rate"],
                    }
                )
    fold = pd.DataFrame(rows)
    if fold.empty:
        return fold
    summary = (
        fold.groupby(["regime_model", "feature_set"], dropna=False)
        .agg(
            folds=("test_month", "nunique"),
            mean_top10_ev=("top10_ev", "mean"),
            worst_month_top10_ev=("top10_ev", "min"),
            mean_top10_clean_precision=("top10_clean_precision", "mean"),
            mean_top10_ev_weighted_clean_precision=("top10_ev_weighted_clean_precision", "mean"),
            mean_top10_bad_mae=("top10_bad_mae", "mean"),
            mean_top10_timeout=("top10_timeout", "mean"),
        )
        .reset_index()
    )
    source = summary[summary["feature_set"].eq("source_only")][
        ["regime_model", "mean_top10_ev", "mean_top10_clean_precision", "mean_top10_bad_mae"]
    ].rename(
        columns={
            "mean_top10_ev": "source_only_mean_top10_ev",
            "mean_top10_clean_precision": "source_only_mean_top10_clean_precision",
            "mean_top10_bad_mae": "source_only_mean_top10_bad_mae",
        }
    )
    summary = summary.merge(source, on="regime_model", how="left")
    summary["delta_top10_ev_vs_source"] = summary["mean_top10_ev"] - summary["source_only_mean_top10_ev"]
    summary["delta_top10_clean_precision_vs_source"] = (
        summary["mean_top10_clean_precision"] - summary["source_only_mean_top10_clean_precision"]
    )
    summary["delta_top10_bad_mae_vs_source"] = summary["mean_top10_bad_mae"] - summary["source_only_mean_top10_bad_mae"]
    summary["scope"] = "summary"
    fold["scope"] = "fold"
    return pd.concat([summary, fold], ignore_index=True, sort=False)


def _weighted_parent_stats(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key_values, group in frame.groupby(keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = {key: value for key, value in zip(keys, key_values)}
        n = _safe_num(group["rows"]).fillna(0.0).clip(lower=0.0)
        denom = float(n.sum())
        row["parent_rows"] = int(denom)
        for metric in metrics:
            values = _safe_num(group[metric])
            row[f"parent_{metric}"] = float((values * n).sum() / denom) if denom > 0.0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _apply_parent_shrinkage(
    frame: pd.DataFrame,
    *,
    parent_keys: list[str],
    metrics: list[str],
    k: float = 100.0,
) -> pd.DataFrame:
    out = frame.copy()
    parent = _weighted_parent_stats(out, parent_keys, metrics)
    out = out.merge(parent, on=parent_keys, how="left")
    n = _safe_num(out["rows"]).fillna(0.0).clip(lower=0.0)
    out["shrinkage_parent_keys"] = " + ".join(parent_keys)
    out["shrinkage_k"] = float(k)
    out["shrinkage_weight"] = n / (n + float(k))
    for metric in metrics:
        out[f"shrunk_{metric}"] = (
            out["shrinkage_weight"] * _safe_num(out[metric])
            + (1.0 - out["shrinkage_weight"]) * _safe_num(out[f"parent_{metric}"])
        )
        out[f"delta_{metric}_vs_parent"] = _safe_num(out[metric]) - _safe_num(out[f"parent_{metric}"])
    return out


def _simulate_policy_returns(frame: pd.DataFrame, policy: dict[str, Any], *, round_trip_cost: float) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    barrier = _first_present(frame, ("barrier", "__first_touch_barrier__"), default=np.nan).fillna(0.0).clip(lower=0.0)
    mae_r = _safe_num(frame["mae_r"]).fillna(0.0).clip(lower=0.0)
    mfe_r = _safe_num(frame["mfe_r"]).fillna(0.0).clip(lower=0.0)
    observed_u = _safe_num(frame["u"]).fillna(0.0)
    timeout = _safe_num(frame["timeout"]).fillna(0.0).clip(0.0, 1.0)
    mfe_first = _safe_num(frame["mfe_before_mae_1r"]).fillna(0.0).clip(0.0, 1.0)

    if policy["kind"] == "abstain":
        out["policy_u"] = 0.0
        out["policy_bad_mae"] = 0.0
        out["policy_timeout"] = 0.0
        out["policy_clean_exit"] = 0.0
        out["policy_mfe_before_mae"] = 0.0
        out["policy_stop"] = 0.0
        return out

    sl_r = float(policy.get("sl_r", 1.0))
    if policy["kind"] == "trailing":
        activation = float(policy["trail_start_r"])
        trail_gap = float(policy["trail_gap_r"])
        win = mfe_r.ge(activation) & ((mae_r.lt(sl_r)) | mfe_first.eq(1.0))
        stop = mae_r.ge(sl_r) & ~win
        realized_r = (mfe_r - trail_gap).clip(lower=0.0)
        gross = (realized_r * barrier).where(win, observed_u.clip(lower=-sl_r * barrier, upper=activation * barrier))
    else:
        tp_r = float(policy["tp_r"])
        win = mfe_r.ge(tp_r) & ((mae_r.lt(sl_r)) | mfe_first.eq(1.0))
        stop = mae_r.ge(sl_r) & ~win
        gross = pd.Series(0.0, index=frame.index, dtype=np.float64)
        gross.loc[win] = tp_r * barrier.loc[win]
        gross.loc[stop] = -sl_r * barrier.loc[stop]
        unresolved = ~(win | stop)
        gross.loc[unresolved] = observed_u.loc[unresolved].clip(
            lower=-sl_r * barrier.loc[unresolved],
            upper=tp_r * barrier.loc[unresolved],
        )

    out["policy_u"] = gross - float(round_trip_cost)
    out["policy_bad_mae"] = mae_r.ge(sl_r).astype(float)
    out["policy_timeout"] = (timeout.gt(0.0) & ~(win | stop)).astype(float)
    out["policy_clean_exit"] = (win & mae_r.lt(sl_r) & timeout.eq(0.0)).astype(float)
    out["policy_mfe_before_mae"] = win.astype(float)
    out["policy_stop"] = stop.astype(float)
    return out


def _execution_policy_matrix(
    frame: pd.DataFrame,
    candidate_map: dict[str, str],
    *,
    round_trip_cost: float,
    shrinkage_k: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sim_columns = []
    for policy in POLICY_MENU:
        sim = _simulate_policy_returns(frame, policy, round_trip_cost=float(round_trip_cost))
        policy_name = str(policy["policy"])
        for col in sim.columns:
            frame_col = f"__policy_{policy_name}_{col}"
            sim_columns.append(frame_col)
            frame[frame_col] = sim[col]

    tested_map = {k: v for k, v in candidate_map.items() if k in EXECUTION_POLICY_REGIME_ALLOWLIST}
    for regime_model, regime_col in tested_map.items():
        for policy in POLICY_MENU:
            policy_name = str(policy["policy"])
            metric_cols = {
                "policy_u": f"__policy_{policy_name}_policy_u",
                "policy_bad_mae": f"__policy_{policy_name}_policy_bad_mae",
                "policy_timeout": f"__policy_{policy_name}_policy_timeout",
                "policy_clean_exit": f"__policy_{policy_name}_policy_clean_exit",
                "policy_mfe_before_mae": f"__policy_{policy_name}_policy_mfe_before_mae",
                "policy_stop": f"__policy_{policy_name}_policy_stop",
            }
            for (source, regime, side), group in frame.groupby(["source_tag", regime_col, "side_name"], dropna=False):
                u = _safe_num(group[metric_cols["policy_u"]])
                rows.append(
                    {
                        "regime_model": regime_model,
                        "source_tag": str(source),
                        "regime": str(regime),
                        "side": str(side),
                        "policy": policy_name,
                        "policy_kind": str(policy["kind"]),
                        "support": int(len(group)),
                        "rows": int(len(group)),
                        "policy_ev": _safe_mean(u),
                        "policy_p10_u": _safe_quantile(u, 0.10),
                        "policy_bad_mae": _rate(group[metric_cols["policy_bad_mae"]]),
                        "policy_timeout": _rate(group[metric_cols["policy_timeout"]]),
                        "policy_clean_exit_rate": _rate(group[metric_cols["policy_clean_exit"]]),
                        "policy_mfe_before_mae": _rate(group[metric_cols["policy_mfe_before_mae"]]),
                        "policy_stop_rate": _rate(group[metric_cols["policy_stop"]]),
                        "month_coverage": int(group["month"].nunique()),
                        "simulation_mode": "conservative_proxy_from_prefeature_ledger_path_summaries",
                        "round_trip_cost": float(round_trip_cost),
                    }
                )
            for (source, regime), group in frame.groupby(["source_tag", regime_col], dropna=False):
                u = _safe_num(group[metric_cols["policy_u"]])
                rows.append(
                    {
                        "regime_model": regime_model,
                        "source_tag": str(source),
                        "regime": str(regime),
                        "side": "all",
                        "policy": policy_name,
                        "policy_kind": str(policy["kind"]),
                        "support": int(len(group)),
                        "rows": int(len(group)),
                        "policy_ev": _safe_mean(u),
                        "policy_p10_u": _safe_quantile(u, 0.10),
                        "policy_bad_mae": _rate(group[metric_cols["policy_bad_mae"]]),
                        "policy_timeout": _rate(group[metric_cols["policy_timeout"]]),
                        "policy_clean_exit_rate": _rate(group[metric_cols["policy_clean_exit"]]),
                        "policy_mfe_before_mae": _rate(group[metric_cols["policy_mfe_before_mae"]]),
                        "policy_stop_rate": _rate(group[metric_cols["policy_stop"]]),
                        "month_coverage": int(group["month"].nunique()),
                        "simulation_mode": "conservative_proxy_from_prefeature_ledger_path_summaries",
                        "round_trip_cost": float(round_trip_cost),
                    }
                )
    matrix = pd.DataFrame(rows)
    if matrix.empty:
        return matrix
    matrix = _apply_parent_shrinkage(
        matrix,
        parent_keys=["regime_model", "source_tag", "side", "policy"],
        metrics=["policy_ev", "policy_bad_mae", "policy_timeout", "policy_clean_exit_rate", "policy_p10_u"],
        k=float(shrinkage_k),
    )
    matrix["policy_objective"] = (
        _safe_num(matrix["shrunk_policy_ev"])
        - 0.010 * _safe_num(matrix["shrunk_policy_bad_mae"]).fillna(1.0)
        - 0.005 * _safe_num(matrix["shrunk_policy_timeout"]).fillna(1.0)
    )
    matrix["selected_policy"] = False
    for keys, group in matrix.groupby(["regime_model", "source_tag", "regime", "side"], dropna=False):
        best_idx = group["policy_objective"].astype(float).idxmax()
        matrix.loc[best_idx, "selected_policy"] = True
    matrix["recommended_sort"] = np.where(matrix["selected_policy"], _safe_num(matrix["policy_objective"]), np.nan)
    return matrix


def _score_regime_model(
    concentration: pd.DataFrame,
    outcome: pd.DataFrame,
    learnability: pd.DataFrame,
    *,
    regime_model: str,
) -> dict[str, Any]:
    conc = concentration[concentration["regime_model"].eq(regime_model)]
    out = outcome[(outcome["regime_model"].eq(regime_model)) & outcome["scope"].eq("source_regime")]
    learn = learnability[
        (learnability["regime_model"].eq(regime_model))
        & learnability["scope"].eq("frontier")
        & (learnability["rows"] >= 30)
    ]
    source_regime_counts = out.groupby("source_tag", dropna=False)["regime"].nunique(dropna=False) if len(out) else pd.Series(dtype=float)
    has_incremental_regime_split = bool(len(source_regime_counts) and int(source_regime_counts.max()) >= 2)
    source_conc = _nanmean_or_nan(
        [
            _safe_mean(conc["lift_source_regime"].abs().clip(upper=5.0)),
            _safe_mean(conc["source_hhi"]),
        ]
    )
    source_concentration_applicable = not str(regime_model).startswith("archetype_side_")
    if not source_concentration_applicable:
        source_conc = float("nan")
    path = _nanmean_or_nan(
        [
            _safe_mean(out.groupby("source_tag")["ev_after_cost"].std()),
            _safe_mean(out.groupby("source_tag")["bad_mae_rate"].std()),
            _safe_mean(out.groupby("source_tag")["mfe_before_mae_1r_rate"].std()),
        ]
    )
    frontier = _nanmean_or_nan(
        [
            _safe_mean(learn["top10_ev"]),
            _safe_mean(learn["top10_clean_precision"]),
            1.0 - _safe_mean(learn["top10_bad_mae"]),
            _safe_mean(learn["top10_mfe_before_mae"]),
            _safe_mean(learn["proxy_spearman_ic"].clip(lower=-0.2, upper=0.2)) + 0.2,
        ]
    )
    support = _nanmean_or_nan(
        [
            min(float((out["rows"] >= 100).mean()) if len(out) else 0.0, 1.0),
            min(float((out["month_coverage"] >= 2).mean()) if len(out) else 0.0, 1.0),
            min(float((learn["rows"] >= 50).mean()) if len(learn) else 0.0, 1.0),
        ]
    )
    weighted = [
        (source_conc, 0.10),
        (path, 0.35),
        (frontier, 0.35),
        (support, 0.20),
    ]
    denom = sum(weight for value, weight in weighted if np.isfinite(value))
    score = (
        float(sum(weight * value for value, weight in weighted if np.isfinite(value)) / denom)
        if denom > 0.0
        else float("nan")
    )
    if not has_incremental_regime_split:
        score = float("nan")
    return {
        "regime_model": regime_model,
        "has_incremental_regime_split": bool(has_incremental_regime_split),
        "source_concentration_applicable": bool(source_concentration_applicable),
        "source_concentration_score": source_conc,
        "path_outcome_interaction_score": path,
        "frontier_learnability_score": frontier,
        "stability_support_score": support,
        "regime_score": float(score),
    }


def _recommendations(
    outcome: pd.DataFrame,
    learnability: pd.DataFrame,
    execution_policy: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = outcome[outcome["scope"].eq("source_regime")].copy()
    learn = learnability[learnability["scope"].eq("frontier")].copy()
    keys = ["regime_model", "source_tag", "regime"]
    merged = out.merge(
        learn[
            keys
            + [
                "rows",
                "top10_ev",
                "top10_clean_precision",
                "top10_bad_mae",
                "top10_timeout",
                "top10_mfe_before_mae",
                "top10_cusum_good_first",
                "top10_cusum_bad_first",
                "top10_mean_bars_to_exit",
                "proxy_spearman_ic",
                "score_clean_gap",
                "rank_stability_positive_months",
                "rank_stability_months",
                "rank_stability_worst_month_top10_ev",
            ]
        ].rename(columns={"rows": "frontier_rows"}),
        on=keys,
        how="left",
    )
    merged = _apply_parent_shrinkage(
        merged,
        parent_keys=["regime_model", "source_tag"],
        metrics=["ev_after_cost", "bad_mae_rate", "timeout_rate", "clean_executable_rate"],
        k=100.0,
    )
    if execution_policy is not None and not execution_policy.empty:
        selected_policy = execution_policy[
            execution_policy["selected_policy"].eq(True) & execution_policy["side"].eq("all")
        ].copy()
        policy_cols = [
            "regime_model",
            "source_tag",
            "regime",
            "policy",
            "policy_objective",
            "shrunk_policy_ev",
            "shrunk_policy_bad_mae",
            "shrunk_policy_timeout",
            "shrunk_policy_clean_exit_rate",
            "policy_p10_u",
            "simulation_mode",
        ]
        merged = merged.merge(
            selected_policy[[col for col in policy_cols if col in selected_policy.columns]].rename(
                columns={
                    "policy": "recommended_execution_policy",
                    "policy_objective": "recommended_policy_objective",
                    "shrunk_policy_ev": "recommended_policy_ev",
                    "shrunk_policy_bad_mae": "recommended_policy_bad_mae",
                    "shrunk_policy_timeout": "recommended_policy_timeout",
                    "shrunk_policy_clean_exit_rate": "recommended_policy_clean_exit_rate",
                    "policy_p10_u": "recommended_policy_p10_u",
                }
            ),
            on=keys,
            how="left",
        )
    actions = []
    reasons = []
    confidences = []
    statuses = []
    meta_feature_roles = []
    sample_weight_multipliers = []
    threshold_deltas = []
    size_multipliers = []
    abstain_candidates = []
    label_conditioner_hints = []
    execution_policy_hints = []
    for row in merged.itertuples(index=False):
        support_ok = int(getattr(row, "rows")) >= 100 and int(getattr(row, "frontier_rows", 0) or 0) >= 30
        shrunk_ev = float(getattr(row, "shrunk_ev_after_cost", np.nan))
        shrunk_bad = float(getattr(row, "shrunk_bad_mae_rate", np.nan))
        shrunk_timeout = float(getattr(row, "shrunk_timeout_rate", np.nan))
        shrunk_clean = float(getattr(row, "shrunk_clean_executable_rate", np.nan))
        top10_ev = float(getattr(row, "top10_ev", np.nan))
        top10_bad = float(getattr(row, "top10_bad_mae", np.nan))
        top10_timeout = float(getattr(row, "top10_timeout", np.nan))
        top10_clean = float(getattr(row, "top10_clean_precision", np.nan))
        mfe_first = float(getattr(row, "mfe_before_mae_1r_rate", np.nan))
        mae_first = float(getattr(row, "mae_1r_before_mfe_1r_rate", np.nan))
        mean_bars = float(getattr(row, "mean_bars_to_exit", np.nan))
        path_ok = (
            shrunk_ev > 0.0
            and shrunk_bad <= 0.35
            and shrunk_timeout <= 0.12
        )
        learn_ok = (
            top10_ev > 0.0
            and top10_bad <= 0.30
            and top10_clean >= 0.45
        )
        stable_ok = int(getattr(row, "rank_stability_positive_months", 0) or 0) >= 2
        if not support_ok:
            actions.append("diagnostic_only")
            reasons.append("insufficient source x regime frontier support")
            confidences.append("low")
            statuses.append("not_promoted")
        elif path_ok and learn_ok and stable_ok:
            actions.append("feature_plus_upweight_or_policy_candidate")
            reasons.append("positive path quality and learnable frontier")
            confidences.append("medium")
            statuses.append("candidate")
        elif float(getattr(row, "shrunk_bad_mae_rate", 0.0)) > 0.45 or float(getattr(row, "top10_bad_mae", 0.0)) > 0.40:
            actions.append("downweight_or_meta_filter")
            reasons.append("dirty path quality / bad-MAE remains elevated")
            confidences.append("medium" if support_ok else "low")
            statuses.append("candidate_for_meta_filter")
        elif float(getattr(row, "shrunk_ev_after_cost", np.nan)) > 0.0 or float(getattr(row, "top10_ev", np.nan)) > 0.0:
            actions.append("feature_only")
            reasons.append("some positive signal but promotion gates are incomplete")
            confidences.append("low_medium")
            statuses.append("feature_only")
        else:
            actions.append("diagnostic_or_abstain_candidate")
            reasons.append("weak EV or weak frontier learnability")
            confidences.append("low_medium")
            statuses.append("not_promoted")

        dirty_risk = (
            (np.isfinite(shrunk_bad) and shrunk_bad > 0.50)
            or (np.isfinite(top10_bad) and top10_bad > 0.50)
            or (np.isfinite(mae_first) and mae_first > 0.35)
        )
        timeout_risk = (np.isfinite(shrunk_timeout) and shrunk_timeout > 0.12) or (
            np.isfinite(top10_timeout) and top10_timeout > 0.12
        )
        clean_path_signal = (
            np.isfinite(shrunk_ev)
            and shrunk_ev > 0.0
            and np.isfinite(shrunk_clean)
            and shrunk_clean >= 0.20
            and np.isfinite(mfe_first)
            and mfe_first >= 0.65
        )
        weak_or_negative = (
            (np.isfinite(shrunk_ev) and shrunk_ev <= 0.0)
            or (np.isfinite(top10_ev) and top10_ev <= 0.0)
        )
        abstain = bool(support_ok and weak_or_negative and (dirty_risk or timeout_risk))
        if support_ok and path_ok and learn_ok and stable_ok:
            meta_feature_roles.append("feature_plus_positive_weight_context")
            sample_weight_multipliers.append(1.25)
            threshold_deltas.append(-0.02)
            size_multipliers.append(1.10)
        elif abstain:
            meta_feature_roles.append("feature_plus_abstention_context")
            sample_weight_multipliers.append(0.25)
            threshold_deltas.append(0.20)
            size_multipliers.append(0.00)
        elif support_ok and dirty_risk and np.isfinite(shrunk_ev) and shrunk_ev > 0.0:
            meta_feature_roles.append("feature_plus_dirty_positive_filter_context")
            sample_weight_multipliers.append(0.50)
            threshold_deltas.append(0.10)
            size_multipliers.append(0.50)
        elif support_ok and clean_path_signal:
            meta_feature_roles.append("feature_plus_mild_positive_context")
            sample_weight_multipliers.append(1.10)
            threshold_deltas.append(-0.01)
            size_multipliers.append(1.00)
        elif support_ok:
            meta_feature_roles.append("feature_plus_neutral_context")
            sample_weight_multipliers.append(0.80 if dirty_risk or timeout_risk else 1.00)
            threshold_deltas.append(0.05 if dirty_risk or timeout_risk else 0.00)
            size_multipliers.append(0.75 if dirty_risk or timeout_risk else 1.00)
        else:
            meta_feature_roles.append("diagnostic_context_only")
            sample_weight_multipliers.append(1.00)
            threshold_deltas.append(0.00)
            size_multipliers.append(1.00)
        abstain_candidates.append(abstain)

        hints: list[str] = []
        if dirty_risk:
            hints.append("strict_bad_mae_penalty")
        if timeout_risk:
            hints.append("timeout_speed_penalty")
        if np.isfinite(mfe_first) and mfe_first < 0.55:
            hints.append("path_order_penalty")
        if np.isfinite(mae_first) and mae_first > 0.35:
            hints.append("mae_first_penalty")
        if np.isfinite(mean_bars) and mean_bars > 36.0:
            hints.append("slow_resolution_penalty")
        if clean_path_signal and not hints:
            hints.append("clean_path_context")
        label_conditioner_hints.append("|".join(hints) if hints else "none")

        policy = str(getattr(row, "recommended_execution_policy", "") or "")
        if policy and policy != "nan" and policy != "P0_abstain":
            execution_policy_hints.append(f"candidate_policy:{policy}")
        elif abstain:
            execution_policy_hints.append("candidate_policy:P0_abstain")
        else:
            execution_policy_hints.append("policy_not_promoted")
    merged["recommended_action"] = actions
    merged["confidence"] = confidences
    merged["reason"] = reasons
    merged["promotion_status"] = statuses
    merged["meta_feature_role"] = meta_feature_roles
    merged["meta_sample_weight_multiplier"] = sample_weight_multipliers
    merged["meta_threshold_delta"] = threshold_deltas
    merged["meta_size_multiplier"] = size_multipliers
    merged["meta_abstain_candidate"] = abstain_candidates
    merged["label_conditioner_hint"] = label_conditioner_hints
    merged["execution_policy_hint"] = execution_policy_hints
    return merged


def _summarize_selection(sample: pd.DataFrame, *, total_rows: int) -> dict[str, Any]:
    metrics = _group_path_metrics(sample) if len(sample) else {}
    return {
        "selected_rows": int(len(sample)),
        "eligible_rows": int(total_rows),
        "selected_frac": float(len(sample) / max(int(total_rows), 1)),
        "selected_ev": metrics.get("ev_after_cost", float("nan")),
        "selected_clean_precision": metrics.get("clean_executable_rate", float("nan")),
        "selected_ev_weighted_clean_precision": metrics.get("ev_weighted_clean_precision", float("nan")),
        "selected_bad_mae": metrics.get("bad_mae_rate", float("nan")),
        "selected_timeout": metrics.get("timeout_rate", float("nan")),
        "selected_mfe_before_mae_1r": metrics.get("mfe_before_mae_1r_rate", float("nan")),
        "selected_mae_1r_before_mfe_1r": metrics.get("mae_1r_before_mfe_1r_rate", float("nan")),
        "selected_cusum_good_first": metrics.get("cusum_good_first_rate", float("nan")),
        "selected_cusum_bad_first": metrics.get("cusum_bad_first_rate", float("nan")),
        "selected_mean_bars_to_exit": metrics.get("mean_bars_to_exit", float("nan")),
        "selected_p10_u": metrics.get("p10_u", float("nan")),
    }


def _apply_frozen_action_overlay(
    test: pd.DataFrame,
    recommendations: pd.DataFrame,
    *,
    regime_model: str,
    regime_col: str,
    top_frac: float,
) -> list[dict[str, Any]]:
    keys = ["source_tag", "_validation_regime"]
    action_cols = [
        "source_tag",
        "regime",
        "shrunk_ev_after_cost",
        "shrunk_bad_mae_rate",
        "shrunk_timeout_rate",
        "shrunk_clean_executable_rate",
        "mfe_before_mae_1r_rate",
        "mae_1r_before_mfe_1r_rate",
        "top10_ev",
        "top10_clean_precision",
        "top10_bad_mae",
        "top10_timeout",
        "top10_mfe_before_mae",
        "top10_mae_before_mfe",
        "meta_feature_role",
        "meta_sample_weight_multiplier",
        "meta_threshold_delta",
        "meta_size_multiplier",
        "meta_abstain_candidate",
        "recommended_action",
        "promotion_status",
        "label_conditioner_hint",
        "execution_policy_hint",
    ]
    available = [col for col in action_cols if col in recommendations.columns]
    action = recommendations[recommendations["regime_model"].eq(regime_model)][available].copy()
    if not action.empty:
        action = action.rename(columns={"regime": "_validation_regime"})
        action = action.sort_values(
            ["source_tag", "_validation_regime", "meta_abstain_candidate", "meta_threshold_delta"],
            ascending=[True, True, True, True],
        ).drop_duplicates(["source_tag", "_validation_regime"], keep="first")

    scored = test.copy()
    scored["_validation_regime"] = scored[regime_col].astype(str)
    if action.empty:
        for col, value in {
            "meta_feature_role": "prior_train_missing",
            "meta_sample_weight_multiplier": 1.0,
            "meta_threshold_delta": 0.0,
            "meta_size_multiplier": 1.0,
            "meta_abstain_candidate": False,
            "recommended_action": "feature_only",
            "promotion_status": "prior_train_missing",
            "label_conditioner_hint": "none",
            "execution_policy_hint": "policy_not_promoted",
        }.items():
            scored[col] = value
    else:
        scored = scored.merge(action, on=keys, how="left")
        scored["meta_feature_role"] = scored["meta_feature_role"].fillna("prior_train_missing")
        scored["meta_sample_weight_multiplier"] = _safe_num(scored["meta_sample_weight_multiplier"]).fillna(1.0)
        scored["meta_threshold_delta"] = _safe_num(scored["meta_threshold_delta"]).fillna(0.0)
        scored["meta_size_multiplier"] = _safe_num(scored["meta_size_multiplier"]).fillna(1.0)
        scored["meta_abstain_candidate"] = scored["meta_abstain_candidate"].map(
            lambda value: bool(value) if pd.notna(value) else False
        )
        scored["recommended_action"] = scored["recommended_action"].fillna("feature_only")
        scored["promotion_status"] = scored["promotion_status"].fillna("prior_train_missing")
        scored["label_conditioner_hint"] = scored["label_conditioner_hint"].fillna("none")
        scored["execution_policy_hint"] = scored["execution_policy_hint"].fillna("policy_not_promoted")
    for col, default in {
        "shrunk_ev_after_cost": 0.0,
        "shrunk_bad_mae_rate": 0.60,
        "shrunk_timeout_rate": 0.08,
        "shrunk_clean_executable_rate": 0.15,
        "mfe_before_mae_1r_rate": 0.65,
        "mae_1r_before_mfe_1r_rate": 0.25,
        "top10_ev": 0.0,
        "top10_clean_precision": 0.15,
        "top10_bad_mae": 0.60,
        "top10_timeout": 0.08,
        "top10_mfe_before_mae": 0.65,
        "top10_mae_before_mfe": 0.25,
    }.items():
        if col not in scored.columns:
            scored[col] = default
        scored[col] = _safe_num(scored[col]).fillna(float(default))

    n_select = max(1, int(math.ceil(len(scored) * float(top_frac)))) if len(scored) else 0
    score_rank = _safe_num(scored["score"]).rank(method="average", pct=True)
    size = _safe_num(scored["meta_size_multiplier"]).fillna(1.0).clip(lower=0.0, upper=2.0)
    sample_weight = _safe_num(scored["meta_sample_weight_multiplier"]).fillna(1.0).clip(lower=0.0, upper=2.0)
    threshold_delta = _safe_num(scored["meta_threshold_delta"]).fillna(0.0).clip(lower=-0.25, upper=0.50)
    prior_bad = pd.concat(
        [_safe_num(scored["shrunk_bad_mae_rate"]), _safe_num(scored["top10_bad_mae"])],
        axis=1,
    ).max(axis=1).fillna(0.60)
    prior_timeout = pd.concat(
        [_safe_num(scored["shrunk_timeout_rate"]), _safe_num(scored["top10_timeout"])],
        axis=1,
    ).max(axis=1).fillna(0.08)
    prior_clean = pd.concat(
        [_safe_num(scored["shrunk_clean_executable_rate"]), _safe_num(scored["top10_clean_precision"])],
        axis=1,
    ).max(axis=1).fillna(0.15)
    prior_mfe_first = pd.concat(
        [_safe_num(scored["mfe_before_mae_1r_rate"]), _safe_num(scored["top10_mfe_before_mae"])],
        axis=1,
    ).max(axis=1).fillna(0.65)
    prior_mae_first = pd.concat(
        [_safe_num(scored["mae_1r_before_mfe_1r_rate"]), _safe_num(scored["top10_mae_before_mfe"])],
        axis=1,
    ).max(axis=1).fillna(0.25)
    scored["_frozen_action_score"] = (
        score_rank
        - threshold_delta
        + 0.030 * np.log1p(size)
        + 0.015 * (sample_weight - 1.0)
    )
    scored["_frozen_path_score"] = (
        scored["_frozen_action_score"]
        - 0.350 * (prior_bad - 0.50).clip(lower=0.0)
        - 0.250 * (prior_timeout - 0.08).clip(lower=0.0)
        - 0.250 * (prior_mae_first - 0.25).clip(lower=0.0)
        + 0.120 * (prior_clean - 0.15).clip(lower=-0.15, upper=0.50)
        + 0.080 * (prior_mfe_first - 0.65).clip(lower=-0.65, upper=0.35)
    )
    scored["_frozen_weighted_u"] = _safe_num(scored["u"]) * size
    scored["_frozen_eligible"] = (~scored["meta_abstain_candidate"].astype(bool)) & size.gt(0.0)
    scored["_frozen_path_guard_eligible"] = (
        scored["_frozen_eligible"]
        & prior_bad.le(0.62)
        & prior_timeout.le(0.14)
        & prior_mae_first.le(0.38)
    )

    selections = {
        "baseline_score_top10": scored.sort_values("score", ascending=False).head(n_select),
        "frozen_threshold_adjusted_top10": scored.sort_values("_frozen_action_score", ascending=False).head(n_select),
        "frozen_abstain_threshold_adjusted_top10": scored[scored["_frozen_eligible"]]
        .sort_values("_frozen_action_score", ascending=False)
        .head(n_select),
        "frozen_path_penalized_top10": scored.sort_values("_frozen_path_score", ascending=False).head(n_select),
        "frozen_clean_guarded_top10": scored[scored["_frozen_path_guard_eligible"]]
        .sort_values("_frozen_path_score", ascending=False)
        .head(n_select),
    }
    rows: list[dict[str, Any]] = []
    for selector, selected in selections.items():
        row = {
            "regime_model": regime_model,
            "selector": selector,
            "top_frac": float(top_frac),
            "frozen_action_rows_matched": int(scored["meta_feature_role"].ne("prior_train_missing").sum()),
            "frozen_action_match_rate": float(scored["meta_feature_role"].ne("prior_train_missing").mean())
            if len(scored)
            else float("nan"),
            "abstain_candidate_rows": int(scored["meta_abstain_candidate"].astype(bool).sum()),
            "abstain_candidate_rate": float(scored["meta_abstain_candidate"].astype(bool).mean())
            if len(scored)
            else float("nan"),
            "mean_size_multiplier": _safe_mean(scored["meta_size_multiplier"]),
            "mean_threshold_delta": _safe_mean(scored["meta_threshold_delta"]),
            "path_guard_eligible_rows": int(scored["_frozen_path_guard_eligible"].sum()),
            "path_guard_eligible_rate": float(scored["_frozen_path_guard_eligible"].mean()) if len(scored) else float("nan"),
        }
        row.update(_summarize_selection(selected, total_rows=len(scored)))
        if len(selected):
            row["selected_weighted_ev"] = _safe_mean(selected["_frozen_weighted_u"])
            row["selected_abstain_candidate_rate"] = _rate(selected["meta_abstain_candidate"])
            row["selected_mean_size_multiplier"] = _safe_mean(selected["meta_size_multiplier"])
            row["selected_mean_threshold_delta"] = _safe_mean(selected["meta_threshold_delta"])
        else:
            row["selected_weighted_ev"] = float("nan")
            row["selected_abstain_candidate_rate"] = float("nan")
            row["selected_mean_size_multiplier"] = float("nan")
            row["selected_mean_threshold_delta"] = float("nan")
        rows.append(row)
    return rows


def _frozen_regime_action_validation(
    frame: pd.DataFrame,
    candidate_map: dict[str, str],
    *,
    frontier_col: str,
    round_trip_cost: float,
    policy_shrinkage_k: float,
    top_frac: float = 0.10,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    months = sorted(str(m) for m in frame["month"].dropna().unique())
    if len(months) < 2:
        return pd.DataFrame(rows)
    for test_month in months[1:]:
        train = frame[frame["month"].astype(str) < str(test_month)].copy()
        test = frame[frame["month"].astype(str) == str(test_month)].copy()
        if train.empty or test.empty:
            continue
        for regime_model, regime_col in candidate_map.items():
            train_outcome = _outcome_matrix(train, regime_model=regime_model, regime_col=regime_col)
            train_learnability = _compact_frontier_learnability(
                train,
                regime_model=regime_model,
                regime_col=regime_col,
                frontier_col=frontier_col,
            )
            train_recommendations = _recommendations(
                train_outcome,
                train_learnability,
                execution_policy=None,
            )
            fold_rows = _apply_frozen_action_overlay(
                test,
                train_recommendations,
                regime_model=regime_model,
                regime_col=regime_col,
                top_frac=float(top_frac),
            )
            for row in fold_rows:
                row["scope"] = "fold"
                row["test_month"] = str(test_month)
                row["train_months"] = ",".join(month for month in months if month < str(test_month))
                row["train_rows"] = int(len(train))
                row["test_rows"] = int(len(test))
                row["leakage_contract"] = "prior_months_train_only_actions_applied_to_heldout_month"
                rows.append(row)
    fold = pd.DataFrame(rows)
    if fold.empty:
        return fold
    baseline = fold[fold["selector"].eq("baseline_score_top10")][
        [
            "regime_model",
            "test_month",
            "selected_ev",
            "selected_weighted_ev",
            "selected_clean_precision",
            "selected_bad_mae",
            "selected_timeout",
            "selected_mfe_before_mae_1r",
            "selected_mae_1r_before_mfe_1r",
        ]
    ].rename(
        columns={
            "selected_ev": "baseline_selected_ev",
            "selected_weighted_ev": "baseline_selected_weighted_ev",
            "selected_clean_precision": "baseline_selected_clean_precision",
            "selected_bad_mae": "baseline_selected_bad_mae",
            "selected_timeout": "baseline_selected_timeout",
            "selected_mfe_before_mae_1r": "baseline_selected_mfe_before_mae_1r",
            "selected_mae_1r_before_mfe_1r": "baseline_selected_mae_1r_before_mfe_1r",
        }
    )
    fold = fold.merge(baseline, on=["regime_model", "test_month"], how="left")
    fold["delta_selected_ev_vs_baseline"] = _safe_num(fold["selected_ev"]) - _safe_num(fold["baseline_selected_ev"])
    fold["delta_selected_weighted_ev_vs_baseline"] = _safe_num(fold["selected_weighted_ev"]) - _safe_num(
        fold["baseline_selected_weighted_ev"]
    )
    fold["delta_clean_precision_vs_baseline"] = _safe_num(fold["selected_clean_precision"]) - _safe_num(
        fold["baseline_selected_clean_precision"]
    )
    fold["delta_bad_mae_vs_baseline"] = _safe_num(fold["selected_bad_mae"]) - _safe_num(fold["baseline_selected_bad_mae"])
    fold["delta_timeout_vs_baseline"] = _safe_num(fold["selected_timeout"]) - _safe_num(fold["baseline_selected_timeout"])
    fold["delta_mfe_before_mae_1r_vs_baseline"] = _safe_num(fold["selected_mfe_before_mae_1r"]) - _safe_num(
        fold["baseline_selected_mfe_before_mae_1r"]
    )
    fold["delta_mae_1r_before_mfe_1r_vs_baseline"] = _safe_num(fold["selected_mae_1r_before_mfe_1r"]) - _safe_num(
        fold["baseline_selected_mae_1r_before_mfe_1r"]
    )

    summary = (
        fold.groupby(["regime_model", "selector"], dropna=False)
        .agg(
            folds=("test_month", "nunique"),
            mean_selected_rows=("selected_rows", "mean"),
            mean_selected_frac=("selected_frac", "mean"),
            mean_selected_ev=("selected_ev", "mean"),
            worst_month_selected_ev=("selected_ev", "min"),
            positive_ev_folds=("selected_ev", lambda s: int((_safe_num(s) > 0.0).sum())),
            mean_selected_weighted_ev=("selected_weighted_ev", "mean"),
            mean_clean_precision=("selected_clean_precision", "mean"),
            mean_bad_mae=("selected_bad_mae", "mean"),
            mean_timeout=("selected_timeout", "mean"),
            mean_mfe_before_mae_1r=("selected_mfe_before_mae_1r", "mean"),
            mean_mae_1r_before_mfe_1r=("selected_mae_1r_before_mfe_1r", "mean"),
            mean_delta_selected_ev_vs_baseline=("delta_selected_ev_vs_baseline", "mean"),
            mean_delta_weighted_ev_vs_baseline=("delta_selected_weighted_ev_vs_baseline", "mean"),
            mean_delta_clean_precision_vs_baseline=("delta_clean_precision_vs_baseline", "mean"),
            mean_delta_bad_mae_vs_baseline=("delta_bad_mae_vs_baseline", "mean"),
            mean_delta_timeout_vs_baseline=("delta_timeout_vs_baseline", "mean"),
            mean_delta_mfe_before_mae_1r_vs_baseline=("delta_mfe_before_mae_1r_vs_baseline", "mean"),
            mean_delta_mae_1r_before_mfe_1r_vs_baseline=("delta_mae_1r_before_mfe_1r_vs_baseline", "mean"),
            mean_action_match_rate=("frozen_action_match_rate", "mean"),
            mean_abstain_candidate_rate=("abstain_candidate_rate", "mean"),
            mean_path_guard_eligible_rate=("path_guard_eligible_rate", "mean"),
        )
        .reset_index()
    )
    summary["scope"] = "summary"
    summary["leakage_contract"] = "prior_months_train_only_actions_applied_to_heldout_month"
    return pd.concat([summary, fold], ignore_index=True, sort=False)


def _regime_meta_promotion_decisions(
    frozen_action_validation: pd.DataFrame,
    regime_scores: pd.DataFrame,
) -> pd.DataFrame:
    if frozen_action_validation.empty or "scope" not in frozen_action_validation.columns:
        return pd.DataFrame()
    summary = frozen_action_validation[frozen_action_validation["scope"].eq("summary")].copy()
    if summary.empty:
        return pd.DataFrame()
    baseline = summary[summary["selector"].eq("baseline_score_top10")][
        [
            "regime_model",
            "mean_selected_ev",
            "worst_month_selected_ev",
            "mean_clean_precision",
            "mean_bad_mae",
            "mean_timeout",
            "mean_mfe_before_mae_1r",
            "mean_mae_1r_before_mfe_1r",
        ]
    ].rename(
        columns={
            "mean_selected_ev": "baseline_mean_selected_ev",
            "worst_month_selected_ev": "baseline_worst_month_selected_ev",
            "mean_clean_precision": "baseline_mean_clean_precision",
            "mean_bad_mae": "baseline_mean_bad_mae",
            "mean_timeout": "baseline_mean_timeout",
            "mean_mfe_before_mae_1r": "baseline_mean_mfe_before_mae_1r",
            "mean_mae_1r_before_mfe_1r": "baseline_mean_mae_1r_before_mfe_1r",
        }
    )
    work = summary[~summary["selector"].eq("baseline_score_top10")].merge(baseline, on="regime_model", how="left")
    if regime_scores is not None and not regime_scores.empty:
        score_cols = [
            col
            for col in [
                "regime_model",
                "regime_score",
                "path_outcome_interaction_score",
                "frontier_learnability_score",
                "stability_support_score",
            ]
            if col in regime_scores.columns
        ]
        work = work.merge(regime_scores[score_cols], on="regime_model", how="left")
    if work.empty:
        return work

    for col in [
        "folds",
        "positive_ev_folds",
        "mean_selected_ev",
        "worst_month_selected_ev",
        "mean_clean_precision",
        "mean_bad_mae",
        "mean_timeout",
        "mean_mfe_before_mae_1r",
        "mean_mae_1r_before_mfe_1r",
        "mean_delta_selected_ev_vs_baseline",
        "mean_delta_clean_precision_vs_baseline",
        "mean_delta_bad_mae_vs_baseline",
        "mean_delta_timeout_vs_baseline",
        "mean_delta_mfe_before_mae_1r_vs_baseline",
        "mean_delta_mae_1r_before_mfe_1r_vs_baseline",
        "mean_action_match_rate",
        "mean_path_guard_eligible_rate",
        "regime_score",
    ]:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = _safe_num(work[col])

    path_safe = (
        work["mean_delta_selected_ev_vs_baseline"].gt(0.0)
        & work["worst_month_selected_ev"].gt(0.0)
        & work["positive_ev_folds"].ge(work["folds"])
        & work["mean_delta_bad_mae_vs_baseline"].le(0.0)
        & work["mean_delta_clean_precision_vs_baseline"].ge(-0.005)
        & work["mean_delta_mae_1r_before_mfe_1r_vs_baseline"].le(0.0)
        & work["mean_timeout"].le(0.12)
        & work["mean_action_match_rate"].ge(0.80)
        & work["mean_path_guard_eligible_rate"].ge(0.15)
    )
    path_improving_but_timeout_worse = (
        work["mean_delta_selected_ev_vs_baseline"].gt(0.0)
        & work["worst_month_selected_ev"].gt(0.0)
        & work["mean_delta_bad_mae_vs_baseline"].le(0.0)
        & work["mean_delta_clean_precision_vs_baseline"].ge(-0.005)
        & work["mean_action_match_rate"].ge(0.80)
    )
    ev_only_dirty = (
        work["mean_delta_selected_ev_vs_baseline"].gt(0.0)
        & work["worst_month_selected_ev"].gt(0.0)
        & work["mean_delta_bad_mae_vs_baseline"].gt(0.0)
    )
    clean_but_ev_weaker = (
        work["mean_delta_bad_mae_vs_baseline"].lt(0.0)
        & work["mean_delta_selected_ev_vs_baseline"].le(0.0)
    )
    statuses = np.select(
        [path_safe, path_improving_but_timeout_worse, ev_only_dirty, clean_but_ev_weaker],
        [
            "promote_meta_path_context_candidate",
            "candidate_needs_timeout_tuning",
            "reject_ev_only_dirty_overlay",
            "diagnostic_cleaner_but_weaker_ev",
        ],
        default="diagnostic_no_promotion",
    )
    work["promotion_decision"] = statuses
    work["promotion_score"] = (
        _safe_num(work["mean_delta_selected_ev_vs_baseline"]).fillna(0.0)
        + 0.35 * _safe_num(work["mean_delta_clean_precision_vs_baseline"]).fillna(0.0)
        - 0.35 * _safe_num(work["mean_delta_bad_mae_vs_baseline"]).clip(lower=0.0).fillna(0.0)
        - 0.20 * _safe_num(work["mean_delta_timeout_vs_baseline"]).clip(lower=0.0).fillna(0.0)
        - 0.20 * _safe_num(work["mean_delta_mae_1r_before_mfe_1r_vs_baseline"]).clip(lower=0.0).fillna(0.0)
        + 0.05 * _safe_num(work["mean_path_guard_eligible_rate"]).fillna(0.0)
        + 0.03 * _safe_num(work["regime_score"]).fillna(0.0)
    )
    work["recommended_meta_integration"] = np.select(
        [
            work["promotion_decision"].eq("promote_meta_path_context_candidate"),
            work["promotion_decision"].eq("candidate_needs_timeout_tuning"),
            work["promotion_decision"].eq("reject_ev_only_dirty_overlay"),
            work["promotion_decision"].eq("diagnostic_cleaner_but_weaker_ev"),
        ],
        [
            "export_as_meta_feature_plus_path_threshold_context",
            "export_as_meta_feature_with_timeout_penalty_ablation",
            "feature_only_or_downweight_do_not_use_for_threshold_relaxation",
            "diagnostic_for_bad_mae_reduction_not_primary_selector",
        ],
        default="diagnostic_only",
    )
    work["promotion_reason"] = np.select(
        [
            work["promotion_decision"].eq("promote_meta_path_context_candidate"),
            work["promotion_decision"].eq("candidate_needs_timeout_tuning"),
            work["promotion_decision"].eq("reject_ev_only_dirty_overlay"),
            work["promotion_decision"].eq("diagnostic_cleaner_but_weaker_ev"),
        ],
        [
            "month-forward overlay improves EV while not worsening bad-MAE/path order",
            "EV and bad-MAE improve, but timeout penalty still needs tuning",
            "EV improves by selecting dirtier path-quality rows",
            "path quality improves but EV/worst-month evidence is weaker",
        ],
        default="does not beat baseline on required path-aware promotion criteria",
    )
    work["inference_safe_use"] = "use regime columns/features only; do not use train-only outcome-derived action labels directly"
    preferred = work.sort_values(["promotion_score", "mean_delta_selected_ev_vs_baseline"], ascending=[False, False])
    best_rows = preferred.groupby("regime_model", dropna=False).head(1).index
    work["best_selector_for_regime_model"] = work.index.isin(best_rows)
    columns = [
        "regime_model",
        "selector",
        "promotion_decision",
        "promotion_score",
        "recommended_meta_integration",
        "promotion_reason",
        "folds",
        "positive_ev_folds",
        "mean_selected_rows",
        "mean_selected_ev",
        "worst_month_selected_ev",
        "mean_clean_precision",
        "mean_bad_mae",
        "mean_timeout",
        "mean_mfe_before_mae_1r",
        "mean_mae_1r_before_mfe_1r",
        "mean_delta_selected_ev_vs_baseline",
        "mean_delta_clean_precision_vs_baseline",
        "mean_delta_bad_mae_vs_baseline",
        "mean_delta_timeout_vs_baseline",
        "mean_delta_mfe_before_mae_1r_vs_baseline",
        "mean_delta_mae_1r_before_mfe_1r_vs_baseline",
        "mean_action_match_rate",
        "mean_abstain_candidate_rate",
        "mean_path_guard_eligible_rate",
        "regime_score",
        "path_outcome_interaction_score",
        "frontier_learnability_score",
        "stability_support_score",
        "best_selector_for_regime_model",
        "inference_safe_use",
    ]
    return work[[col for col in columns if col in work.columns]].sort_values(
        ["promotion_decision", "promotion_score"],
        ascending=[True, False],
    )


def _build_meta_regime_integration_plan(
    promotion: pd.DataFrame,
    *,
    candidate_map: dict[str, str],
    feature_schema: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if promotion.empty:
        return pd.DataFrame(), {"rows": 0, "promoted_rows": 0, "status": "empty_promotion_table"}
    feature_cols = set(feature_schema["column"].astype(str)) if "column" in feature_schema.columns else set()
    rows: list[dict[str, Any]] = []
    best = promotion[promotion.get("best_selector_for_regime_model", False).astype(bool)].copy()
    promoted = promotion[promotion["promotion_decision"].eq("promote_meta_path_context_candidate")].copy()
    for row in pd.concat([promoted, best], ignore_index=True, sort=False).drop_duplicates(
        ["regime_model", "selector"], keep="first"
    ).itertuples(index=False):
        regime_model = str(getattr(row, "regime_model"))
        regime_col = str(candidate_map.get(regime_model, ""))
        regime_code_col = f"{regime_col}__code" if regime_col else ""
        exported = [col for col in (regime_col, regime_code_col) if col in feature_cols]
        decision = str(getattr(row, "promotion_decision", ""))
        is_promoted = decision == "promote_meta_path_context_candidate"
        is_dirty_reject = decision == "reject_ev_only_dirty_overlay"
        if is_promoted:
            integration_status = "promoted_for_meta_hpo"
            usage = "feature_plus_path_threshold_context"
            recommended_action = "include_features_and_test_path_guarded_threshold_modifier"
        elif is_dirty_reject:
            integration_status = "blocked_from_threshold_relaxation"
            usage = "feature_only_or_downweight_context"
            recommended_action = "do_not_use_for_threshold_relaxation; optionally downweight dirty contexts"
        else:
            integration_status = "diagnostic_context_only"
            usage = "feature_only"
            recommended_action = "keep as optional feature; do not alter thresholds without new evidence"
        rows.append(
            {
                "regime_model": regime_model,
                "selector": str(getattr(row, "selector", "")),
                "promotion_decision": decision,
                "promotion_score": float(getattr(row, "promotion_score", np.nan)),
                "integration_status": integration_status,
                "recommended_meta_usage": usage,
                "recommended_action": recommended_action,
                "regime_feature_column": regime_col,
                "regime_feature_code_column": regime_code_col,
                "exported_feature_columns": "|".join(exported),
                "exported_feature_count": int(len(exported)),
                "hard_gate_allowed": False,
                "threshold_modifier_allowed": bool(is_promoted),
                "size_modifier_allowed": False,
                "sample_weight_hint": "neutral_until_meta_hpo" if is_promoted else "none",
                "label_conditioner_hint": "path_order_and_bad_mae_context" if is_promoted else "none",
                "execution_policy_hint": "diagnostic_only_not_frozen_replay",
                "mean_selected_ev": float(getattr(row, "mean_selected_ev", np.nan)),
                "worst_month_selected_ev": float(getattr(row, "worst_month_selected_ev", np.nan)),
                "mean_bad_mae": float(getattr(row, "mean_bad_mae", np.nan)),
                "mean_timeout": float(getattr(row, "mean_timeout", np.nan)),
                "mean_delta_selected_ev_vs_baseline": float(getattr(row, "mean_delta_selected_ev_vs_baseline", np.nan)),
                "mean_delta_bad_mae_vs_baseline": float(getattr(row, "mean_delta_bad_mae_vs_baseline", np.nan)),
                "mean_delta_clean_precision_vs_baseline": float(
                    getattr(row, "mean_delta_clean_precision_vs_baseline", np.nan)
                ),
                "mean_path_guard_eligible_rate": float(getattr(row, "mean_path_guard_eligible_rate", np.nan)),
                "leakage_contract": "Use exported regime feature columns at meta inference; do not feed promotion/action labels as features.",
            }
        )
    plan = pd.DataFrame(rows)
    if plan.empty:
        return plan, {"rows": 0, "promoted_rows": 0, "status": "no_rows"}
    plan = plan.sort_values(["integration_status", "promotion_score"], ascending=[True, False])
    report = {
        "rows": int(len(plan)),
        "promoted_rows": int(plan["integration_status"].eq("promoted_for_meta_hpo").sum()),
        "blocked_from_threshold_relaxation_rows": int(
            plan["integration_status"].eq("blocked_from_threshold_relaxation").sum()
        ),
        "diagnostic_rows": int(plan["integration_status"].eq("diagnostic_context_only").sum()),
        "all_promoted_features_exported": bool(
            plan.loc[plan["integration_status"].eq("promoted_for_meta_hpo"), "exported_feature_count"].ge(1).all()
        ),
        "status": "available",
    }
    return plan, report


def _gate3_readiness_summary(
    frozen_action_validation: pd.DataFrame,
    promotion: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if frozen_action_validation.empty or "scope" not in frozen_action_validation.columns:
        return pd.DataFrame(), {"status": "missing_frozen_action_validation"}
    summary = frozen_action_validation[frozen_action_validation["scope"].eq("summary")].copy()
    if summary.empty:
        return pd.DataFrame(), {"status": "missing_summary_rows"}
    baseline = summary[summary["selector"].eq("baseline_score_top10")].head(1)
    candidates = []
    if not baseline.empty:
        candidates.append(("baseline_score_top10", baseline.iloc[0].to_dict(), "baseline"))
    if not promotion.empty:
        for row in promotion[promotion["promotion_decision"].eq("promote_meta_path_context_candidate")].itertuples(index=False):
            candidates.append((f"{getattr(row, 'regime_model')}::{getattr(row, 'selector')}", row._asdict(), "promoted"))
    for name, row, role in candidates:
        mean_ev = float(row.get("mean_selected_ev", np.nan))
        worst_ev = float(row.get("worst_month_selected_ev", np.nan))
        folds = float(row.get("folds", np.nan))
        positive_folds = float(row.get("positive_ev_folds", np.nan))
        bad = float(row.get("mean_bad_mae", np.nan))
        timeout = float(row.get("mean_timeout", np.nan))
        clean_precision = float(row.get("mean_clean_precision", np.nan))
        checks = {
            "mean_ev_positive": bool(np.isfinite(mean_ev) and mean_ev > 0.0),
            "worst_month_ev_positive": bool(np.isfinite(worst_ev) and worst_ev > 0.0),
            "positive_all_folds": bool(np.isfinite(folds) and np.isfinite(positive_folds) and positive_folds >= folds),
            "bad_mae_final_bar": bool(np.isfinite(bad) and bad <= 0.50),
            "timeout_final_bar": bool(np.isfinite(timeout) and timeout <= 0.12),
            "final_oracle_recall_available": False,
            "side_share_available": False,
        }
        status = "pass" if all(checks.values()) else "fail"
        if not checks["final_oracle_recall_available"] or not checks["side_share_available"]:
            status = "incomplete_evidence"
        if not checks["bad_mae_final_bar"]:
            status = "fail"
        rows.append(
            {
                "candidate": str(name),
                "role": role,
                "gate3_status": status,
                "mean_selected_ev": mean_ev,
                "worst_month_selected_ev": worst_ev,
                "positive_ev_folds": positive_folds,
                "folds": folds,
                "mean_clean_precision": clean_precision,
                "mean_bad_mae": bad,
                "mean_timeout": timeout,
                "mean_delta_selected_ev_vs_baseline": float(row.get("mean_delta_selected_ev_vs_baseline", 0.0)),
                "mean_delta_bad_mae_vs_baseline": float(row.get("mean_delta_bad_mae_vs_baseline", 0.0)),
                "mean_ev_positive": checks["mean_ev_positive"],
                "worst_month_ev_positive": checks["worst_month_ev_positive"],
                "positive_all_folds": checks["positive_all_folds"],
                "bad_mae_final_bar": checks["bad_mae_final_bar"],
                "timeout_final_bar": checks["timeout_final_bar"],
                "final_oracle_recall_available": checks["final_oracle_recall_available"],
                "side_share_available": checks["side_share_available"],
                "missing_evidence": "final_oracle_recall|side_share",
                "gate3_reason": "bad-MAE remains above final <=50% bar"
                if not checks["bad_mae_final_bar"]
                else "oracle recall and side-share evidence not present in this audit",
            }
        )
    table = pd.DataFrame(rows)
    report = {
        "rows": int(len(table)),
        "pass_rows": int(table["gate3_status"].eq("pass").sum()) if not table.empty else 0,
        "fail_rows": int(table["gate3_status"].eq("fail").sum()) if not table.empty else 0,
        "incomplete_evidence_rows": int(table["gate3_status"].eq("incomplete_evidence").sum()) if not table.empty else 0,
        "status": "available",
        "gate3_contract": "Uses frozen month-forward top10 evidence; final oracle recall and side-share must be supplied by downstream policy/meta evaluation.",
    }
    return table, report


FEATURE_EXPORT_FAMILIES = {
    "key_or_split",
    "regime_candidate_feature",
    "ctx_ae_gmm_state",
    "ctx_meta_feature",
    "ctx_vol_regime_feature",
    "ctx_raw_meta_feature",
    "semantic_source_tag",
    "semantic_source_score",
    "semantic_source_component",
    "prefit_path_risk_score",
    "prefit_score_or_rank",
}

FEATURE_EXPORT_EXTRA_COLUMNS = {
    "barrier",
    "prior_recent_source_strength",
    "selected_top10",
    "selected_top20",
    "selected_top30",
}

DERIVED_OUTCOME_COLUMNS = {
    "u",
    "cost",
    "ev_after_cost",
    "gross_u",
    "bad_mae",
    "timeout",
    "stop",
    "clean_exec",
    "dirty_positive",
    "mfe_before_mae_1r",
    "mae_before_mfe_1r",
    "cusum_good_first",
    "cusum_bad_first",
    "max_adverse_before_mfe_1r",
    "underwater_bars_before_mfe_1r",
    "bars_to_exit",
    "mae_r",
    "mfe_r",
}


def _is_direct_outcome_column(col: str, outcome_cols: set[str]) -> bool:
    lower = str(col).lower()
    if str(col) in outcome_cols or str(col) in DERIVED_OUTCOME_COLUMNS:
        return True
    if lower.startswith(("__first_touch", "__mfe_", "__mae_", "__max_adverse", "__underwater", "__policy_")):
        return True
    if lower.startswith(("__bars_to_", "__area_underwater", "__trailing_profit")):
        return True
    if "oracle" in lower:
        return True
    return False


def _categorical_codes(series: pd.Series) -> pd.Series:
    codes = pd.Categorical(series.astype(str).where(series.notna(), "missing")).codes
    return pd.Series(codes, index=series.index, dtype=np.int32)


def _build_meta_handoff_exports(
    frame: pd.DataFrame,
    recommendations: pd.DataFrame,
    *,
    outcome_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    outcome_set = set(str(c) for c in outcome_cols)
    feature_cols: list[str] = []
    excluded_direct_outcome: list[str] = []
    excluded_train_only: list[str] = []
    for col in frame.columns:
        name = str(col)
        if _is_direct_outcome_column(name, outcome_set):
            excluded_direct_outcome.append(name)
            continue
        family = _feature_family(name)
        if family in FEATURE_EXPORT_FAMILIES or name in FEATURE_EXPORT_EXTRA_COLUMNS:
            feature_cols.append(name)
        elif name.startswith("candidate_") or name.startswith("ctx_"):
            feature_cols.append(name)
        else:
            excluded_train_only.append(name)

    # Keep stable row identity columns first, then model/context columns.
    ordered = [c for c in ("timestamp", "symbol", "side_name", "side", "ctx_side", "month") if c in feature_cols]
    ordered.extend(c for c in feature_cols if c not in set(ordered))
    feature_export = frame.loc[:, ordered].copy()

    categorical_cols = [
        c
        for c in feature_export.columns
        if c in {"source_tag", "source_family", "candidate_regime_family"}
        or str(c).startswith("candidate_")
        or str(c).endswith("_tag")
    ]
    for col in categorical_cols:
        if col in feature_export.columns:
            feature_export[f"{col}__code"] = _categorical_codes(feature_export[col])

    schema_rows: list[dict[str, Any]] = []
    n = max(int(len(feature_export)), 1)
    for col in feature_export.columns:
        series = feature_export[col]
        schema_rows.append(
            {
                "column": str(col),
                "feature_family": _feature_family(str(col)),
                "dtype": str(series.dtype),
                "non_null_rows": int(series.notna().sum()),
                "missing_rate": float(1.0 - int(series.notna().sum()) / n),
                "numeric": bool(pd.api.types.is_numeric_dtype(series)),
                "export_role": "meta_prefeature_inference_safe",
            }
        )
    feature_schema = pd.DataFrame(schema_rows)

    action = recommendations.copy()
    if "delta_ev_after_cost_vs_parent" in action.columns:
        action["expected_delta_EV"] = _safe_num(action["delta_ev_after_cost_vs_parent"])
    if "delta_bad_mae_rate_vs_parent" in action.columns:
        action["expected_delta_bad_MAE"] = _safe_num(action["delta_bad_mae_rate_vs_parent"])
    if "delta_timeout_rate_vs_parent" in action.columns:
        action["expected_delta_timeout"] = _safe_num(action["delta_timeout_rate_vs_parent"])
    action["artifact_role"] = "train_only_regime_action_table"
    action["inference_safe"] = False
    action["leakage_note"] = (
        "Uses outcome-derived metrics and proxy execution simulations; freeze on outer-train before validation use."
    )

    export_report = {
        "feature_export_rows": int(len(feature_export)),
        "feature_export_columns": int(len(feature_export.columns)),
        "action_table_rows": int(len(action)),
        "action_table_columns": int(len(action.columns)),
        "direct_outcome_columns_excluded": sorted(set(excluded_direct_outcome)),
        "train_or_diagnostic_columns_excluded_from_feature_export": sorted(set(excluded_train_only)),
        "feature_export_families": sorted(set(feature_schema["feature_family"].astype(str))),
        "leakage_contract": {
            "feature_export": "pre-entry/live-computable plus prefit model scores and candidate regime descriptors",
            "action_table": "train-only outcome-derived recommendations; not an inference feature input",
        },
    }
    return feature_export, feature_schema, action, export_report


def _write_report(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    schema_summary: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    regime_scores: pd.DataFrame,
    incremental_value: pd.DataFrame,
    execution_policy: pd.DataFrame,
    recommendations: pd.DataFrame,
    frozen_action_validation: pd.DataFrame,
    meta_promotion_decisions: pd.DataFrame,
    meta_integration_plan: pd.DataFrame,
    gate3_readiness: pd.DataFrame,
) -> None:
    def table(df: pd.DataFrame, cols: list[str], limit: int = 20) -> str:
        if df.empty:
            return "No rows."
        view = df[[c for c in cols if c in df.columns]].head(limit).copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    incremental_summary = (
        incremental_value[incremental_value["scope"].eq("summary")]
        if "scope" in incremental_value.columns
        else pd.DataFrame()
    )
    frozen_summary = (
        frozen_action_validation[frozen_action_validation["scope"].eq("summary")]
        if "scope" in frozen_action_validation.columns
        else pd.DataFrame()
    )
    lines = [
        "# Regime Source Interaction Audit",
        "",
        f"Meta pre-feature-selection ledger: `{manifest['meta_feature_ledger_path']}`",
        f"Rows: `{manifest['input_report']['rows']}`; columns: `{manifest['input_report']['columns']}`; ctx columns: `{manifest['input_report']['ctx_columns']}`",
        f"Source tag source: `{manifest['source_report']['source_tag_source']}`",
        f"Semantic source status: `{manifest['source_report'].get('semantic_source_report', {}).get('status', 'unknown')}`",
        f"Path-order label source: `{manifest['input_report'].get('path_order_label_report', {}).get('path', 'disabled')}`",
        f"Path-order matched rows: `{manifest['input_report'].get('path_order_label_report', {}).get('matched_rows', 0)}`",
        "",
        "## Meta Input Schema",
        table(
            schema_summary,
            [
                "feature_family",
                "columns",
                "numeric_columns",
                "regime_input_columns",
                "outcome_eval_columns",
                "median_missing_rate",
            ],
            30,
        ),
        "",
        "## Candidate Regime Summary",
        table(
            candidate_summary,
            ["regime_model", "status", "regime_count", "min_regime_rows", "max_regime_rows", "entropy", "hhi", "month_coverage", "side_coverage"],
            20,
        ),
        "",
        "## Regime Scores",
        table(
            regime_scores.sort_values("regime_score", ascending=False),
            [
                "regime_model",
                "has_incremental_regime_split",
                "source_concentration_applicable",
                "regime_score",
                "source_concentration_score",
                "path_outcome_interaction_score",
                "frontier_learnability_score",
                "stability_support_score",
            ],
            20,
        ),
        "",
        "## Incremental Value Test",
        table(
            incremental_summary.sort_values(
                ["delta_top10_ev_vs_source", "delta_top10_clean_precision_vs_source"],
                ascending=[False, False],
            )
            if not incremental_summary.empty
            else incremental_summary,
            [
                "regime_model",
                "feature_set",
                "folds",
                "mean_top10_ev",
                "worst_month_top10_ev",
                "mean_top10_clean_precision",
                "mean_top10_bad_mae",
                "delta_top10_ev_vs_source",
                "delta_top10_clean_precision_vs_source",
                "delta_top10_bad_mae_vs_source",
            ],
            30,
        ),
        "",
        "## Frozen Month-Forward Action Validation",
        table(
            frozen_summary.sort_values(
                ["mean_delta_selected_ev_vs_baseline", "mean_delta_bad_mae_vs_baseline"],
                ascending=[False, True],
            )
            if not frozen_summary.empty
            else frozen_summary,
            [
                "regime_model",
                "selector",
                "folds",
                "mean_selected_rows",
                "mean_selected_ev",
                "worst_month_selected_ev",
                "positive_ev_folds",
                "mean_clean_precision",
                "mean_bad_mae",
                "mean_timeout",
                "mean_mfe_before_mae_1r",
                "mean_mae_1r_before_mfe_1r",
                "mean_delta_selected_ev_vs_baseline",
                "mean_delta_clean_precision_vs_baseline",
                "mean_delta_bad_mae_vs_baseline",
                "mean_delta_timeout_vs_baseline",
                "mean_action_match_rate",
                "mean_abstain_candidate_rate",
                "mean_path_guard_eligible_rate",
            ],
            40,
        ),
        "",
        "## Meta Promotion Decisions",
        table(
            meta_promotion_decisions.sort_values(["promotion_score"], ascending=[False])
            if not meta_promotion_decisions.empty
            else meta_promotion_decisions,
            [
                "regime_model",
                "selector",
                "promotion_decision",
                "promotion_score",
                "recommended_meta_integration",
                "folds",
                "positive_ev_folds",
                "mean_selected_ev",
                "worst_month_selected_ev",
                "mean_clean_precision",
                "mean_bad_mae",
                "mean_timeout",
                "mean_delta_selected_ev_vs_baseline",
                "mean_delta_clean_precision_vs_baseline",
                "mean_delta_bad_mae_vs_baseline",
                "mean_delta_timeout_vs_baseline",
                "mean_path_guard_eligible_rate",
                "regime_score",
                "best_selector_for_regime_model",
            ],
            40,
        ),
        "",
        "## Meta Integration Plan",
        table(
            meta_integration_plan.sort_values(["integration_status", "promotion_score"], ascending=[True, False])
            if not meta_integration_plan.empty
            else meta_integration_plan,
            [
                "regime_model",
                "selector",
                "integration_status",
                "recommended_meta_usage",
                "recommended_action",
                "regime_feature_column",
                "regime_feature_code_column",
                "exported_feature_count",
                "threshold_modifier_allowed",
                "hard_gate_allowed",
                "mean_selected_ev",
                "mean_bad_mae",
                "mean_timeout",
                "mean_delta_selected_ev_vs_baseline",
                "mean_delta_bad_mae_vs_baseline",
            ],
            40,
        ),
        "",
        "## Gate 3 Readiness",
        table(
            gate3_readiness.sort_values(["role", "mean_selected_ev"], ascending=[False, False])
            if not gate3_readiness.empty
            else gate3_readiness,
            [
                "candidate",
                "role",
                "gate3_status",
                "mean_selected_ev",
                "worst_month_selected_ev",
                "positive_ev_folds",
                "folds",
                "mean_bad_mae",
                "mean_timeout",
                "mean_delta_selected_ev_vs_baseline",
                "mean_delta_bad_mae_vs_baseline",
                "bad_mae_final_bar",
                "timeout_final_bar",
                "final_oracle_recall_available",
                "side_share_available",
                "gate3_reason",
            ],
            20,
        ),
        "",
        "## Top Policy Recommendations",
        table(
            recommendations.sort_values(["promotion_status", "top10_ev"], ascending=[True, False]),
            [
                "regime_model",
                "source_tag",
                "regime",
                "rows",
                "frontier_rows",
                "ev_after_cost",
                "ev_after_cost_ci95_low",
                "ev_after_cost_ci95_high",
                "shrunk_ev_after_cost",
                "bad_mae_rate",
                "bad_mae_ci95_low",
                "bad_mae_ci95_high",
                "shrunk_bad_mae_rate",
                "timeout_rate",
                "timeout_ci95_low",
                "timeout_ci95_high",
                "shrunk_timeout_rate",
                "mfe_before_mae_1r_rate",
                "mae_1r_before_mfe_1r_rate",
                "cusum_good_first_rate",
                "cusum_bad_first_rate",
                "mean_bars_to_exit",
                "top10_ev",
                "top10_clean_precision",
                "top10_bad_mae",
                "top10_cusum_good_first",
                "recommended_execution_policy",
                "recommended_policy_ev",
                "recommended_policy_bad_mae",
                "recommended_action",
                "meta_feature_role",
                "meta_sample_weight_multiplier",
                "meta_threshold_delta",
                "meta_size_multiplier",
                "meta_abstain_candidate",
                "label_conditioner_hint",
                "execution_policy_hint",
                "promotion_status",
                "reason",
            ],
            40,
        ),
        "",
        "## Execution Policy Menu",
        table(
            execution_policy[execution_policy["selected_policy"].eq(True)].sort_values(
                ["recommended_sort"], ascending=[False]
            )
            if "recommended_sort" in execution_policy.columns and not execution_policy.empty
            else execution_policy,
            [
                "regime_model",
                "source_tag",
                "regime",
                "side",
                "policy",
                "support",
                "policy_ev",
                "shrunk_policy_ev",
                "policy_bad_mae",
                "shrunk_policy_bad_mae",
                "policy_timeout",
                "shrunk_policy_timeout",
                "policy_clean_exit_rate",
                "policy_p10_u",
                "shrinkage_weight",
                "simulation_mode",
            ],
            40,
        ),
        "",
        "## Meta Handoff Exports",
        f"Feature export: `{manifest['outputs'].get('meta_regime_feature_export', '')}`",
        f"Feature columns: `{manifest.get('handoff_report', {}).get('feature_export_columns', 0)}`",
        f"Train-only action table: `{manifest['outputs'].get('train_only_regime_action_table', '')}`",
        f"Frozen action validation: `{manifest['outputs'].get('frozen_action_validation', '')}`",
        f"Meta promotion decisions: `{manifest['outputs'].get('meta_promotion_decisions', '')}`",
        f"Meta integration plan: `{manifest['outputs'].get('meta_integration_plan', '')}`",
        f"Gate 3 readiness: `{manifest['outputs'].get('gate3_readiness', '')}`",
        f"Direct outcome columns excluded: `{len(manifest.get('handoff_report', {}).get('direct_outcome_columns_excluded', []))}`",
        "",
        "## Notes",
        "- Regime inputs come from the persisted meta-layer candidate ledger before feature selection.",
        "- Outcome columns are used only for evaluation matrices, not to construct regime buckets except for explicitly precomputed risk-score features.",
        "- `archetype_side_*` regime candidates are built within source/archetype family x side buckets.",
        "- `archetype_side_aegmm_*` regimes scope AE/GMM cluster assignments to source/archetype family x side before scoring.",
        "- Underpowered source/archetype x side groups are pooled before local quantile binning to avoid tiny local regime cells.",
        "- Source concentration is marked not applicable for `archetype_side_*` candidates because the source/side context is embedded in those regime labels.",
        "- Confidence intervals are normal-approximation 95% intervals for EV means and binary path-quality rates.",
        "- CUSUM good/bad-first fields use explicit CUSUM columns when present; otherwise they fall back to MFE-before-MAE / MAE-before-MFE path-order flags.",
        "- Meta feature export excludes direct outcomes and train-only policy recommendations; the action table must be frozen on outer-train before validation use.",
        "- Frozen action validation learns source x regime action hints on prior months only, then applies those frozen hints to the held-out month.",
        "- Meta integration plan only references inference-safe exported regime feature columns; promotion/action labels remain train-only.",
        "- Gate 3 readiness is strict and marks missing oracle-recall/side-share evidence explicitly instead of treating it as passed.",
        "- Execution policy menu rows are conservative proxy simulations from available path summaries; they are not frozen replay evidence.",
        "- This audit is an evidence layer for meta handoff. It does not promote hard gates by itself.",
    ]
    (output_dir / "regime_source_interaction_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(
    *,
    meta_feature_ledger_path: Path,
    path_order_labels_path: Path | None,
    output_dir: Path,
    source_min_score: float,
    frontier_col: str,
    round_trip_cost: float,
    policy_shrinkage_k: float,
    skip_incremental_value: bool = False,
    skip_execution_policy: bool = False,
    skip_frozen_action_validation: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    merged, input_report = _standardize_meta_feature_ledger(meta_feature_ledger_path)
    merged, path_order_label_report = _merge_path_order_labels(merged, path_order_labels_path)
    input_report["path_order_label_report"] = path_order_label_report
    merged, semantic_source_report = _derive_semantic_source_scores(merged)
    input_report["semantic_source_report"] = semantic_source_report
    schema_audit = _feature_schema_audit(merged, outcome_cols=input_report["outcome_columns"])
    schema_summary = (
        schema_audit.groupby("feature_family", dropna=False)
        .agg(
            columns=("column", "count"),
            numeric_columns=("numeric", "sum"),
            regime_input_columns=("used_as_regime_input", "sum"),
            outcome_eval_columns=("used_as_outcome_eval", "sum"),
            median_missing_rate=("missing_rate", "median"),
        )
        .reset_index()
        .sort_values(["regime_input_columns", "columns"], ascending=[False, False])
    )
    input_report["feature_family_counts"] = {
        str(row.feature_family): int(row.columns) for row in schema_summary.itertuples(index=False)
    }
    merged, candidate_summary, source_report = _add_regime_candidates(merged, source_min_score=float(source_min_score))
    source_report["semantic_source_report"] = semantic_source_report
    merged = _metric_frame(merged)
    merged, leaf_candidate_summary, leaf_report, leaf_candidate_map = _add_oof_leaf_regime_candidates(merged)
    candidate_summary = pd.concat([candidate_summary, leaf_candidate_summary], ignore_index=True, sort=False)
    source_report["leaf_embedding_report"] = leaf_report
    input_report["path_order_observed_rows"] = {
        "mfe_before_mae_1r": int(_safe_num(merged["mfe_before_mae_1r"]).replace([np.inf, -np.inf], np.nan).notna().sum()),
        "mae_before_mfe_1r": int(_safe_num(merged["mae_before_mfe_1r"]).replace([np.inf, -np.inf], np.nan).notna().sum()),
        "cusum_good_first": int(_safe_num(merged["cusum_good_first"]).replace([np.inf, -np.inf], np.nan).notna().sum()),
        "cusum_bad_first": int(_safe_num(merged["cusum_bad_first"]).replace([np.inf, -np.inf], np.nan).notna().sum()),
        "bars_to_exit": int(_safe_num(merged["bars_to_exit"]).replace([np.inf, -np.inf], np.nan).notna().sum()),
    }

    candidate_map = {
        "observable_family": "candidate_regime_family",
        "base_score_decile": "candidate_base_score_decile",
        "archetype_side_base_score_decile": "candidate_archetype_side_base_score_decile",
        "spread_bin": "candidate_spread_bin",
        "liquidity_bin": "candidate_liquidity_bin",
        "archetype_side_liquidity_bin": "candidate_archetype_side_liquidity_bin",
        "activity_liquidity_bin": "candidate_activity_liquidity_bin",
        "archetype_side_activity_liquidity_bin": "candidate_archetype_side_activity_liquidity_bin",
        "volatility_bin": "candidate_volatility_bin",
        "archetype_side_volatility_bin": "candidate_archetype_side_volatility_bin",
        "volatility_zscore_bin": "candidate_volatility_zscore_bin",
        "directional_vol_imbalance_bin": "candidate_directional_vol_imbalance_bin",
        "archetype_side_directional_vol_imbalance_bin": "candidate_archetype_side_directional_vol_imbalance_bin",
        "market_dispersion_bin": "candidate_market_dispersion_bin",
        "archetype_side_market_dispersion_bin": "candidate_archetype_side_market_dispersion_bin",
        "volatility_shape_bin": "candidate_volatility_shape_bin",
        "aegmm_global_argmax": "candidate_aegmm_global_argmax",
        "archetype_side_aegmm_global_argmax": "candidate_archetype_side_aegmm_global_argmax",
        "aegmm_side_argmax": "candidate_aegmm_side_argmax",
        "archetype_side_aegmm_side_argmax": "candidate_archetype_side_aegmm_side_argmax",
        "aegmm_entropy_bin": "candidate_aegmm_entropy_bin",
        "archetype_side_aegmm_entropy_bin": "candidate_archetype_side_aegmm_entropy_bin",
        "aegmm_distance_bin": "candidate_aegmm_distance_bin",
        "reconstruction_bin": "candidate_reconstruction_bin",
        "archetype_side_reconstruction_bin": "candidate_archetype_side_reconstruction_bin",
        "bad_mae_score_bin": "candidate_bad_mae_score_bin",
        "archetype_side_bad_mae_score_bin": "candidate_archetype_side_bad_mae_score_bin",
        "timeout_score_bin": "candidate_timeout_score_bin",
        "archetype_side_timeout_score_bin": "candidate_archetype_side_timeout_score_bin",
        "execres_score_bin": "candidate_execres_score_bin",
        "archetype_side_execres_score_bin": "candidate_archetype_side_execres_score_bin",
        "exec_move_speed_bin": "candidate_exec_move_speed_bin",
        "archetype_side_exec_move_speed_bin": "candidate_archetype_side_exec_move_speed_bin",
        "exec_signal_to_spread_bin": "candidate_exec_signal_to_spread_bin",
        "archetype_side_exec_signal_to_spread_bin": "candidate_archetype_side_exec_signal_to_spread_bin",
        "exec_slow_resolution_risk_bin": "candidate_exec_slow_resolution_risk_bin",
        "archetype_side_exec_slow_resolution_risk_bin": "candidate_archetype_side_exec_slow_resolution_risk_bin",
        "exec_adverse_path_pressure_bin": "candidate_exec_adverse_path_pressure_bin",
        "archetype_side_exec_adverse_path_pressure_bin": "candidate_archetype_side_exec_adverse_path_pressure_bin",
        "exec_opportunity_pressure_bin": "candidate_exec_opportunity_pressure_bin",
        "archetype_side_exec_opportunity_pressure_bin": "candidate_archetype_side_exec_opportunity_pressure_bin",
    }
    candidate_map.update(leaf_candidate_map)

    evaluated_candidate_map = {
        name: col for name, col in candidate_map.items() if name in REGIME_MATRIX_ALLOWLIST
    }
    if not evaluated_candidate_map:
        evaluated_candidate_map = dict(candidate_map)

    concentration_frames = []
    outcome_frames = []
    learnability_frames = []
    for model_name, col in evaluated_candidate_map.items():
        concentration_frames.append(_source_concentration(merged, regime_model=model_name, regime_col=col))
        outcome_frames.append(_outcome_matrix(merged, regime_model=model_name, regime_col=col))
        learnability_frames.append(
            _learnability_matrix(merged, regime_model=model_name, regime_col=col, frontier_col=frontier_col)
        )

    concentration = pd.concat(concentration_frames, ignore_index=True)
    outcome = pd.concat(outcome_frames, ignore_index=True)
    learnability = pd.concat(learnability_frames, ignore_index=True)
    regime_scores = pd.DataFrame(
        [
            _score_regime_model(concentration, outcome, learnability, regime_model=model_name)
            for model_name in evaluated_candidate_map
        ]
    )
    incremental_value = (
        pd.DataFrame()
        if skip_incremental_value
        else _incremental_value_tests(merged, candidate_map)
    )
    execution_policy = (
        pd.DataFrame()
        if skip_execution_policy
        else _execution_policy_matrix(
            merged.copy(),
            candidate_map,
            round_trip_cost=float(round_trip_cost),
            shrinkage_k=float(policy_shrinkage_k),
        )
    )
    recommendations = _recommendations(outcome, learnability, execution_policy=execution_policy)
    frozen_candidate_map = dict(evaluated_candidate_map)
    frozen_action_validation = (
        pd.DataFrame()
        if skip_frozen_action_validation
        else _frozen_regime_action_validation(
            merged,
            frozen_candidate_map,
            frontier_col=frontier_col,
            round_trip_cost=float(round_trip_cost),
            policy_shrinkage_k=float(policy_shrinkage_k),
            top_frac=0.10,
        )
    )
    source_report["frozen_validation_candidate_models"] = {
        "evaluated_count": int(len(frozen_candidate_map)),
        "evaluated_models": sorted(frozen_candidate_map),
        "skipped_count": int(len(candidate_map) - len(frozen_candidate_map)),
        "skipped_models": sorted(name for name in candidate_map if name not in frozen_candidate_map),
        "contract": (
            "Candidate summary is generated for all discovered regimes; expensive source/outcome/learnability, "
            "policy, and frozen promotion diagnostics are limited to the compact stable candidate tournament."
        ),
    }
    meta_promotion_decisions = _regime_meta_promotion_decisions(frozen_action_validation, regime_scores)
    meta_feature_export, meta_feature_export_schema, train_only_action_table, handoff_report = _build_meta_handoff_exports(
        merged,
        recommendations,
        outcome_cols=input_report["outcome_columns"],
    )
    meta_integration_plan, integration_report = _build_meta_regime_integration_plan(
        meta_promotion_decisions,
        candidate_map=candidate_map,
        feature_schema=meta_feature_export_schema,
    )
    gate3_readiness, gate3_report = _gate3_readiness_summary(frozen_action_validation, meta_promotion_decisions)

    outputs = {
        "scored_ledger": output_dir / "regime_scored_base_ledger.parquet",
        "meta_regime_feature_export": output_dir / "meta_regime_feature_export.parquet",
        "meta_regime_feature_export_schema": output_dir / "meta_regime_feature_export_schema.csv",
        "train_only_regime_action_table": output_dir / "train_only_regime_action_table.csv",
        "meta_input_schema_audit": output_dir / "meta_input_schema_audit.csv",
        "meta_input_schema_summary": output_dir / "meta_input_schema_summary.csv",
        "candidate_summary": output_dir / "regime_candidate_summary.csv",
        "source_concentration": output_dir / "source_concentration_matrix.csv",
        "source_regime_outcome": output_dir / "source_regime_outcome_matrix.csv",
        "source_regime_learnability": output_dir / "source_regime_learnability_matrix.csv",
        "regime_scores": output_dir / "regime_usefulness_scores.csv",
        "incremental_value": output_dir / "regime_incremental_value_tests.csv",
        "execution_policy_matrix": output_dir / "execution_policy_matrix.csv",
        "recommendations": output_dir / "policy_recommendation_table.csv",
        "frozen_action_validation": output_dir / "frozen_month_forward_action_validation.csv",
        "meta_promotion_decisions": output_dir / "meta_regime_promotion_decisions.csv",
        "meta_integration_plan": output_dir / "meta_regime_integration_plan.csv",
        "gate3_readiness": output_dir / "regime_gate3_readiness_summary.csv",
        "manifest": output_dir / "manifest.json",
    }
    merged.to_parquet(outputs["scored_ledger"], index=False)
    meta_feature_export.to_parquet(outputs["meta_regime_feature_export"], index=False)
    meta_feature_export_schema.to_csv(outputs["meta_regime_feature_export_schema"], index=False)
    train_only_action_table.to_csv(outputs["train_only_regime_action_table"], index=False)
    schema_audit.to_csv(outputs["meta_input_schema_audit"], index=False)
    schema_summary.to_csv(outputs["meta_input_schema_summary"], index=False)
    candidate_summary.to_csv(outputs["candidate_summary"], index=False)
    concentration.to_csv(outputs["source_concentration"], index=False)
    outcome.to_csv(outputs["source_regime_outcome"], index=False)
    learnability.to_csv(outputs["source_regime_learnability"], index=False)
    regime_scores.to_csv(outputs["regime_scores"], index=False)
    incremental_value.to_csv(outputs["incremental_value"], index=False)
    execution_policy.to_csv(outputs["execution_policy_matrix"], index=False)
    recommendations.to_csv(outputs["recommendations"], index=False)
    frozen_action_validation.to_csv(outputs["frozen_action_validation"], index=False)
    meta_promotion_decisions.to_csv(outputs["meta_promotion_decisions"], index=False)
    meta_integration_plan.to_csv(outputs["meta_integration_plan"], index=False)
    gate3_readiness.to_csv(outputs["gate3_readiness"], index=False)

    manifest = {
        "scope": "regime_source_interaction_audit",
        "meta_feature_ledger_path": str(meta_feature_ledger_path),
        "path_order_labels_path": str(path_order_labels_path) if path_order_labels_path is not None else None,
        "output_dir": str(output_dir),
        "source_min_score": float(source_min_score),
        "frontier_col": str(frontier_col),
        "round_trip_cost": float(round_trip_cost),
        "policy_shrinkage_k": float(policy_shrinkage_k),
        "skip_incremental_value": bool(skip_incremental_value),
        "skip_execution_policy": bool(skip_execution_policy),
        "skip_frozen_action_validation": bool(skip_frozen_action_validation),
        "input_report": input_report,
        "source_report": source_report,
        "handoff_report": handoff_report,
        "candidate_models": candidate_map,
        "frozen_action_validation_report": {
            "rows": int(len(frozen_action_validation)),
            "summary_rows": int(frozen_action_validation["scope"].eq("summary").sum())
            if "scope" in frozen_action_validation.columns
            else 0,
            "fold_rows": int(frozen_action_validation["scope"].eq("fold").sum())
            if "scope" in frozen_action_validation.columns
            else 0,
            "contract": "prior-month train-only regime action hints applied to held-out month",
        },
        "meta_promotion_decision_report": {
            "rows": int(len(meta_promotion_decisions)),
            "promoted_rows": int(
                meta_promotion_decisions["promotion_decision"].eq("promote_meta_path_context_candidate").sum()
            )
            if "promotion_decision" in meta_promotion_decisions.columns
            else 0,
            "timeout_tuning_candidate_rows": int(
                meta_promotion_decisions["promotion_decision"].eq("candidate_needs_timeout_tuning").sum()
            )
            if "promotion_decision" in meta_promotion_decisions.columns
            else 0,
            "ev_only_dirty_rows": int(
                meta_promotion_decisions["promotion_decision"].eq("reject_ev_only_dirty_overlay").sum()
            )
            if "promotion_decision" in meta_promotion_decisions.columns
            else 0,
            "contract": "promotion based on frozen prior-month action validation; inference uses regime features, not outcome-derived labels",
        },
        "meta_integration_report": integration_report,
        "gate3_readiness_report": gate3_report,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    outputs["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_report(
        output_dir=output_dir,
        manifest=manifest,
        schema_summary=schema_summary,
        candidate_summary=candidate_summary,
        regime_scores=regime_scores,
        incremental_value=incremental_value,
        execution_policy=execution_policy,
        recommendations=recommendations,
        frozen_action_validation=frozen_action_validation,
        meta_promotion_decisions=meta_promotion_decisions,
        meta_integration_plan=meta_integration_plan,
        gate3_readiness=gate3_readiness,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-feature-ledger-path", type=Path, default=DEFAULT_META_FEATURE_LEDGER_PATH)
    parser.add_argument("--path-order-labels-path", type=Path, default=DEFAULT_PATH_ORDER_LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-min-score", type=float, default=0.50)
    parser.add_argument("--frontier-col", default="selected_top10")
    parser.add_argument("--round-trip-cost", type=float, default=0.0100)
    parser.add_argument("--policy-shrinkage-k", type=float, default=100.0)
    parser.add_argument("--skip-incremental-value", action="store_true")
    parser.add_argument("--skip-execution-policy", action="store_true")
    parser.add_argument("--skip-frozen-action-validation", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_audit(
        meta_feature_ledger_path=args.meta_feature_ledger_path,
        path_order_labels_path=args.path_order_labels_path,
        output_dir=args.output_dir,
        source_min_score=float(args.source_min_score),
        frontier_col=str(args.frontier_col),
        round_trip_cost=float(args.round_trip_cost),
        policy_shrinkage_k=float(args.policy_shrinkage_k),
        skip_incremental_value=bool(args.skip_incremental_value),
        skip_execution_policy=bool(args.skip_execution_policy),
        skip_frozen_action_validation=bool(args.skip_frozen_action_validation),
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
