#!/usr/bin/env python3
"""Run Ridge Position Sizer on the most recent training artifacts.

This script bridges the gap between meta model training and position sizing.
It loads OOF predictions from trained meta models and learns optimal combination
weights for position sizing using the RidgePositionSizer.

Usage:
    python -m extreme_price_movements.run_ridge_sizer --run-id 20260212_190000
    python -m extreme_price_movements.run_ridge_sizer  # Uses latest run

The script expects the following artifacts from a training run:
    - artifacts/{run_id}/meta_oof/meta_oof_*.parquet: OOF predictions from meta models
    - artifacts/{run_id}/trade_outcomes.parquet: Trade outcomes with entry/exit prices
    - artifacts/{run_id}/tpsl_params.json: Optimized TP/SL parameters (optional)
    - Price panel data for policy-aware labeling
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from extreme_price_movements.data_store import read_parquet_projected
from extreme_price_movements.ridge_position_sizer import (
    RidgePositionSizer,
    prepare_policy_params_from_tpsl_optimiser,
    prepare_trade_outcomes_from_labels,
    run_policy_aware_labeling_step,
    run_ridge_position_sizer_step,
)
from extreme_price_movements.training_utils import build_wide_tight_pair_features


def _filter_artifact_by_stage_view(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    view = cfg.get("_active_stage_view")
    if not view or df is None or df.empty:
        return df

    if "symbols" in view and view["symbols"] is not None:
        sym_col = (
            "__symbol__"
            if "__symbol__" in df.columns
            else "symbol"
            if "symbol" in df.columns
            else None
        )
        if sym_col:
            df = df[df[sym_col].isin(view["symbols"])]

    if view.get("allowed_start_ts") or view.get("allowed_end_ts"):
        ts_col = (
            "__ts__"
            if "__ts__" in df.columns
            else (
                "timestamp"
                if "timestamp" in df.columns
                else "t0"
                if "t0" in df.columns
                else None
            )
        )
        if ts_col:
            if view.get("allowed_start_ts"):
                df = df[
                    pd.to_datetime(df[ts_col], utc=True)
                    >= pd.to_datetime(view["allowed_start_ts"])
                ]
            if view.get("allowed_end_ts"):
                df = df[
                    pd.to_datetime(df[ts_col], utc=True)
                    <= pd.to_datetime(view["allowed_end_ts"])
                ]
    return df


from extreme_price_movements.entry_policy import (
    compute_entry_policy_decision,
    flatten_bucket_policy,
)
from extreme_price_movements.offline_optimisers.params_store import (
    load_inference_candidate_mask_params_per_bucket,
)
from extreme_price_movements.utils import log_pipeline_warning, tprint


def format_return_as_pct(ret: float) -> str:
    """Format return as percentage string.

    Args:
        ret: Return in decimal form (e.g., 0.01 for 1%)

    Returns:
        Formatted percentage string (e.g., "1.00%")
    """
    return f"{ret:.2%}"


def format_metric_float(val: float, decimals: int = 6) -> str:
    try:
        v = float(val)
    except Exception:
        return "N/A"
    if not np.isfinite(v):
        return "N/A"
    return f"{v:.{decimals}f}"


from extreme_price_movements.hf_data_loader import _load_existing_data


def _fill_nonfinite_oof_vector(values, neutral: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    if finite.any():
        fill = float(np.nanmedian(arr[finite]))
    else:
        fill = float(neutral)
    arr[~finite] = fill
    return arr


def find_latest_run_id(data_root: str) -> str:
    """Find the most recent run_id from artifacts directory.

    Args:
        data_root: Root directory for data/artifacts

    Returns:
        The most recent run_id string

    Raises:
        FileNotFoundError: If no artifacts directory or run directories exist
    """
    artifacts_dir = Path(data_root) / "artifacts"
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"No artifacts directory at {artifacts_dir}")

    import re

    _ts_pat = re.compile(r"^\d{8}_\d{6}.*$")
    run_dirs = sorted(
        [d for d in artifacts_dir.iterdir() if d.is_dir() and _ts_pat.match(d.name)],
        key=lambda x: x.name,
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError("No run directories found")

    return run_dirs[0].name


def find_best_feature_snapshot_ts(data_root: str, run_id: str) -> pd.Timestamp:
    requested_ts = pd.to_datetime(run_id, format="%Y%m%d_%H%M%S", utc=True)
    feature_root = Path(data_root) / "features"
    requested_dir = feature_root / run_id
    if requested_dir.exists():
        return requested_ts
    if not feature_root.exists():
        return requested_ts
    candidates = sorted([p.name for p in feature_root.iterdir() if p.is_dir()])
    if not candidates:
        return requested_ts
    chosen = candidates[-1]
    if chosen != run_id:
        tprint(
            f"Feature snapshot for {run_id} not found; "
            f"falling back to latest available feature cache {chosen}"
        )
    return pd.to_datetime(chosen, format="%Y%m%d_%H%M%S", utc=True)


# Metadata columns to exclude from features (Realized outcomes and target-related leaks)
_META_COLS = {
    "timestamp",
    "symbol",
    "return",
    "is_long",
    "index",
    "mae_ret",
    "mfe_ret",
    "duration",
    "u_policy",
    "u_policy_net",
    "exit_code",
    "label_policy_sl_atr_mult",
    "label_policy_tp_sl_ratio",
    "label_policy_max_hold_bars",
    "label_policy_giveback_pct",
    "atr_12_15m",
    "early_deadline",
    "early_mfe",
    "outcome_pnl",
    "outcome_sortino",
    "outcome_j",
}


def load_base_oof_predictions(data_root: str, run_id: str) -> Dict[str, pd.DataFrame]:
    """Load base model OOF predictions from a training run.

    Loads OOF predictions from the base training (before meta model).
    These are stored in data/artifacts/{run_id}/oof/ directory.

    Returns a dict keyed by bucket (e.g. 'long_mr') where each value is a
    DataFrame with base model prediction columns (base_H2, base_H4).
    """
    import re
    import re as re2

    base_oof_dir = Path(data_root) / "artifacts" / run_id / "oof"

    if not base_oof_dir.exists():
        tprint(
            f"WARNING: Base OOF directory not found at {base_oof_dir}, skipping base model predictions"
        )
        return {}

    # Load all parquet files (exclude _tight/_wide/_balanced variants)
    raw_dfs = {}
    for parquet_file in base_oof_dir.glob("oof_*.parquet"):
        if any(
            suffix in parquet_file.stem for suffix in ("_tight", "_wide", "_balanced")
        ):
            continue
        model_name = parquet_file.stem.replace("oof_", "")
        df = pd.read_parquet(parquet_file)
        raw_dfs[model_name] = df

    if not raw_dfs:
        tprint(
            f"WARNING: No base OOF parquet files found in {base_oof_dir}, skipping base model predictions"
        )
        return {}

    # Parse model names into (base_bucket, horizon, optional geometry archetype)
    # Patterns:
    #   long_mr_H2 -> (long_mr, H2)
    #   long_tf_H4_tight -> (long_tf, H4, tight)
    h_pat = re2.compile(r"^(.+)_H(\d+)(?:_(tight|balanced|wide))?$")
    buckets = {}
    for name, df in raw_dfs.items():
        m = h_pat.match(name)
        if m:
            bucket = m.group(1)
            _h = int(m.group(2))
            _variant = m.group(3)
            col_name = f"base_H{_h}" if not _variant else f"base_H{_h}_{_variant}"
        else:
            # Fallback: treat entire name as bucket
            bucket = name
            col_name = "base"

        if bucket not in buckets:
            buckets[bucket] = {}
        buckets[bucket][col_name] = df

    def _strip_side_prefix(name: str) -> str:
        return re.sub(r"^(long|short)_", "", name)

    # Merge side-prefixed buckets back into the canonical strategy id so a
    # single strategy can consume all head families that were exported under
    # side-specific filenames.
    canonical_buckets: Dict[str, List[Tuple[str, Dict[str, pd.DataFrame]]]] = {}
    for bucket, model_dfs in buckets.items():
        canonical = _strip_side_prefix(bucket)
        canonical_buckets.setdefault(canonical, []).append((bucket, model_dfs))

    result = {}
    for bucket, grouped_model_dfs in canonical_buckets.items():
        model_dfs: Dict[str, pd.DataFrame] = {}
        source_sides: set[str] = set()
        for source_bucket, source_model_dfs in grouped_model_dfs:
            if source_bucket.startswith("long_"):
                source_sides.add("long")
            elif source_bucket.startswith("short_"):
                source_sides.add("short")
            for col_name, mdf in source_model_dfs.items():
                # Keep the first occurrence for a family. In the current
                # artifact layout the side-specific files supply distinct
                # families (clf/mae/mfe/asym) while the canonical reg file
                # carries the base score.
                if col_name not in model_dfs:
                    model_dfs[col_name] = mdf

        # Use first available df as base (for length and metadata)
        base_df = next(iter(model_dfs.values()))
        n = len(base_df)
        combined = pd.DataFrame(index=range(n))

        # Add all base prediction columns
        for col_name, mdf in sorted(model_dfs.items()):
            if len(mdf) == n:
                # Use oof_prob (classifier) or oof_pred (regressor)
                if "oof_prob" in mdf.columns:
                    combined[col_name] = mdf["oof_prob"].values
                elif "oof_pred" in mdf.columns:
                    combined[col_name] = mdf["oof_pred"].values
                if "oof_sigma_trees" in mdf.columns:
                    if col_name.endswith("_wide"):
                        sigma_name = col_name.replace("_wide", "_sigma_wide")
                    elif col_name.endswith("_tight"):
                        sigma_name = col_name.replace("_tight", "_sigma_tight")
                    else:
                        sigma_name = f"{col_name}_sigma"
                    combined[sigma_name] = mdf["oof_sigma_trees"].values
                if "oof_sigma_robust" in mdf.columns:
                    if col_name.endswith("_wide"):
                        sigma_name = col_name.replace("_wide", "_robust_sigma_wide")
                    elif col_name.endswith("_tight"):
                        sigma_name = col_name.replace("_tight", "_robust_sigma_tight")
                    else:
                        sigma_name = f"{col_name}_robust_sigma"
                    combined[sigma_name] = mdf["oof_sigma_robust"].values
                tree_unc_cols = [c for c in mdf.columns if c.startswith("oof_tree_")]
                for tc in tree_unc_cols:
                    feat_name = tc.replace("oof_tree_", "")
                    combined[f"{col_name}_{feat_name}"] = mdf[tc].values

        pair_roots = sorted(
            {
                c[:-5]
                for c in combined.columns
                if c.endswith("_wide")
                and not c.endswith(("_sigma_wide", "_robust_sigma_wide"))
                and f"{c[:-5]}_tight" in combined.columns
            }
        )
        for root in pair_roots:
            wide_col = f"{root}_wide"
            tight_col = f"{root}_tight"
            sigma_wide_col = f"{root}_sigma_wide"
            sigma_tight_col = f"{root}_sigma_tight"
            robust_sigma_wide_col = f"{root}_robust_sigma_wide"
            robust_sigma_tight_col = f"{root}_robust_sigma_tight"
            pair_features = build_wide_tight_pair_features(
                combined[wide_col].values,
                combined[tight_col].values,
                base_name=root,
                sigma_wide=(
                    combined[sigma_wide_col].values
                    if sigma_wide_col in combined.columns
                    else None
                ),
                sigma_tight=(
                    combined[sigma_tight_col].values
                    if sigma_tight_col in combined.columns
                    else None
                ),
                robust_sigma_wide=(
                    combined[robust_sigma_wide_col].values
                    if robust_sigma_wide_col in combined.columns
                    else None
                ),
                robust_sigma_tight=(
                    combined[robust_sigma_tight_col].values
                    if robust_sigma_tight_col in combined.columns
                    else None
                ),
            )
            for col_name, values in pair_features.items():
                combined[col_name] = values

        # Add metadata and outcomes for joining and diagnostics
        meta_cols_to_pull = [
            "index",
            "timestamp",
            "symbol",
            "return",
            "is_long",
            "mae_ret",
            "mfe_ret",
            "duration",
            "exit_code",
            "y_ret",
            "y_bin",
        ]
        for col in meta_cols_to_pull:
            if col not in combined.columns:
                for mdf in model_dfs.values():
                    if col in mdf.columns:
                        combined[col] = mdf[col].values
                        break

        # Mapping alternative names
        if "return" not in combined.columns and "y_ret" in combined.columns:
            combined["return"] = combined["y_ret"]
        if "is_long" not in combined.columns and "y_bin" in combined.columns:
            combined["is_long"] = combined["y_bin"]
        elif "is_long" not in combined.columns:
            # Fallback based on bucket name
            if bucket.startswith("long"):
                combined["is_long"] = 1
            elif bucket.startswith("short"):
                combined["is_long"] = 0

        # Final metadata sanity check
        if "index" not in combined.columns:
            combined["index"] = range(n)

        result[bucket] = combined

    tprint(
        f"Loaded base OOF predictions for {len(result)} buckets: {list(result.keys())}"
    )
    return result


def load_meta_oof_predictions(
    data_root: str,
    run_id: str,
    *,
    require_meta_barrier_probs: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Load meta model OOF predictions from a training run.

    Handles per-horizon regressors from the canonical horizon set
    (e.g. long_mr_H1, long_mr_H2, long_mr_H4)
    and classifiers (e.g. long_mr_clf). Groups by base bucket and returns a
    DataFrame per bucket with columns like reg_H1/reg_H2/reg_H4, clf, plus
    agreement/disagreement features.

    Returns a dict keyed by bucket (e.g. 'long_mr') where each value is a
    DataFrame with prediction columns plus metadata.
    """
    import re

    meta_oof_dir = Path(data_root) / "artifacts" / run_id / "meta_oof"

    if not meta_oof_dir.exists():
        raise FileNotFoundError(f"No meta OOF directory at {meta_oof_dir}")

    # Load base OOF predictions first
    base_oofs = load_base_oof_predictions(data_root, run_id)

    # Load all parquet files
    raw_dfs = {}
    for parquet_file in meta_oof_dir.glob("meta_oof_*.parquet"):
        model_name = parquet_file.stem.replace("meta_oof_", "")
        df = pd.read_parquet(parquet_file)
        raw_dfs[model_name] = df

    if not raw_dfs:
        raise FileNotFoundError(f"No meta OOF parquet files found in {meta_oof_dir}")

    # Hard safety: when base TP-vs-SL excludes timeouts, sizing must use
    # meta classifier probabilities (p_sl/p_to/p_tp), not base-only outputs.
    _meta_prob_available = any(
        all(c in df.columns for c in ("oof_p_sl", "oof_p_to", "oof_p_tp"))
        for df in raw_dfs.values()
    )
    if not _meta_prob_available:
        msg = "Meta classifier probabilities (oof_p_sl/oof_p_to/oof_p_tp) not found in meta OOF artifacts."
        if require_meta_barrier_probs:
            raise RuntimeError(
                msg
                + " Policy-aligned ridge sizer runs require these barrier probabilities. "
                + "Rebuild meta_oof with classifier probability export enabled."
            )
        log_pipeline_warning(msg + " Continuing with regression/aux heads only.")
    else:
        tprint("Meta classifier probabilities found in meta OOF artifacts.")

    # Parse model names into (base_bucket, col_name)
    # Patterns: long_mr_H2 -> (long_mr, reg_H2), long_mr_clf -> (long_mr, clf),
    #           long_mr_utility -> (long_mr, utility), etc.
    _h_pat = re.compile(r"^(.+)_H(\d+)$")
    _tbm_pat = re.compile(r"^(.+)_(tbm_\d+_\d+)_h(\d+)$")
    _risk_pat = re.compile(r"^(.+)_(mae|mfe|asym)_h(\d+)$")
    _aux_heads = {"utility", "mae_q70", "mfe", "early_inval"}
    buckets = {}
    for name, df in raw_dfs.items():
        if name.endswith("_clf"):
            bucket = name[:-4]
            col_name = "clf"
        elif (_m_tbm := _tbm_pat.match(name)) is not None:
            bucket = _m_tbm.group(1)
            col_name = f"{_m_tbm.group(2)}_h{int(_m_tbm.group(3))}"
        elif (_m_risk := _risk_pat.match(name)) is not None:
            bucket = _m_risk.group(1)
            col_name = f"{_m_risk.group(2)}_h{int(_m_risk.group(3))}"
        elif name.endswith("_cal_reg"):
            bucket = name[:-8]
            col_name = "cal_reg"
        elif name.endswith("_reg"):
            bucket = name[:-4]
            col_name = "reg"
        elif any(name.endswith(f"_{h}") for h in _aux_heads):
            # Find which aux head it is
            h_suffix = next(h for h in _aux_heads if name.endswith(f"_{h}"))
            bucket = name[: -(len(h_suffix) + 1)]
            col_name = h_suffix
        else:
            m = _h_pat.match(name)
            if m:
                _h = int(m.group(2))
                bucket = m.group(1)
                col_name = f"reg_H{_h}"
            else:
                bucket = name
                col_name = "reg"
        if bucket not in buckets:
            buckets[bucket] = {}
        buckets[bucket][col_name] = df

    def _strip_side_prefix(name: str) -> str:
        return re.sub(r"^(long|short)_", "", name)

    # Merge side-prefixed buckets back into the canonical strategy id so a
    # single strategy can consume all head families that were exported under
    # side-specific filenames.
    canonical_buckets: Dict[str, List[Tuple[str, Dict[str, pd.DataFrame]]]] = {}
    for bucket, model_dfs in buckets.items():
        canonical = _strip_side_prefix(bucket)
        canonical_buckets.setdefault(canonical, []).append((bucket, model_dfs))

    result = {}
    for bucket, grouped_model_dfs in canonical_buckets.items():
        model_dfs: Dict[str, pd.DataFrame] = {}
        source_sides: set[str] = set()
        for source_bucket, source_model_dfs in grouped_model_dfs:
            if source_bucket.startswith("long_"):
                source_sides.add("long")
            elif source_bucket.startswith("short_"):
                source_sides.add("short")
            for col_name, mdf in source_model_dfs.items():
                # Keep the first occurrence for a family. In the current
                # artifact layout the side-specific files supply distinct
                # families (clf/mae/mfe/asym) while the canonical reg file
                # carries the base score.
                if col_name not in model_dfs:
                    model_dfs[col_name] = mdf

        # Use first available df as base (for length and metadata)
        base_df = next(iter(model_dfs.values()))
        n = len(base_df)
        combined = pd.DataFrame(index=range(n))

        # Add all prediction columns
        reg_cols = []
        for col_name, mdf in sorted(model_dfs.items()):
            if len(mdf) == n:
                combined[col_name] = _fill_nonfinite_oof_vector(
                    mdf["oof_pred"].values, neutral=0.0
                )
                if col_name.startswith("reg"):
                    reg_cols.append(col_name)

                # Preserve any extra numeric columns exported by the meta
                # OOF files so downstream sizers can use uncertainty heads
                # and auxiliary classifier outputs.
                for extra_col in mdf.columns:
                    if extra_col == "oof_pred" or extra_col in combined.columns:
                        continue
                    if pd.api.types.is_numeric_dtype(mdf[extra_col]):
                        combined[extra_col] = _fill_nonfinite_oof_vector(
                            mdf[extra_col].values, neutral=np.nan
                        )

        # Agreement/disagreement features across horizon regressors
        if len(reg_cols) >= 2:
            reg_vals = combined[reg_cols].values
            # Mean regressor prediction
            combined["reg_mean"] = np.nanmean(reg_vals, axis=1)
            # Std across regressors (disagreement)
            combined["reg_std"] = np.nanstd(reg_vals, axis=1)
            # Range (max - min)
            combined["reg_range"] = np.nanmax(reg_vals, axis=1) - np.nanmin(
                reg_vals, axis=1
            )
            # Sign agreement: fraction of regressors above median
            _med = np.nanmedian(reg_vals, axis=1, keepdims=True)
            combined["reg_sign_agree"] = np.nanmean(
                (reg_vals > _med).astype(float), axis=1
            )
            # Regressor-classifier agreement (if clf exists)
            if "clf" in combined.columns:
                _clf_high = (combined["clf"].values > 0.5).astype(float)
                _reg_high = (
                    combined["reg_mean"].values
                    > np.nanmedian(combined["reg_mean"].values)
                ).astype(float)
                combined["reg_clf_agree"] = (_clf_high == _reg_high).astype(float)
        elif len(reg_cols) == 1:
            combined["reg_mean"] = combined[reg_cols[0]].values
            combined["reg_std"] = 0.0

        tbm_cols = [c for c in combined.columns if c.startswith("tbm_")]
        if tbm_cols:
            tbm_vals = combined[tbm_cols].values
            combined["tbm_mean"] = np.nanmean(tbm_vals, axis=1)
            combined["tbm_std"] = np.nanstd(tbm_vals, axis=1)
        mae_cols = [c for c in combined.columns if c.startswith("mae_h")]
        if mae_cols:
            mae_vals = combined[mae_cols].values
            combined["mae_mean"] = np.nanmean(mae_vals, axis=1)
            combined["mae_std"] = np.nanstd(mae_vals, axis=1)
        mfe_cols = [c for c in combined.columns if c.startswith("mfe_h")]
        if mfe_cols:
            mfe_vals = combined[mfe_cols].values
            combined["mfe_mean"] = np.nanmean(mfe_vals, axis=1)
            combined["mfe_std"] = np.nanstd(mfe_vals, axis=1)
        asym_cols = [c for c in combined.columns if c.startswith("asym_h")]
        if asym_cols:
            asym_vals = combined[asym_cols].values
            combined["asym_mean"] = np.nanmean(asym_vals, axis=1)
            combined["asym_std"] = np.nanstd(asym_vals, axis=1)

        # -------------------------------------------------------------
        # Synthesize interaction features from auxiliary heads
        # -------------------------------------------------------------
        # Require base_df to have the auxiliary heads available
        if all(
            c in base_df.columns for c in ["oof_log_mfe_hat", "oof_log_mae_q70_hat"]
        ):
            _mfe_hat = base_df["oof_log_mfe_hat"].values
            _mae_hat = base_df["oof_log_mae_q70_hat"].values
            combined["risk_reward_ratio"] = _mfe_hat / (_mae_hat + 1e-6)
            combined["risk_adjusted_pred"] = (
                combined["reg_mean"].values - 0.5 * _mae_hat
            )

        if "oof_u_hat" in base_df.columns:
            _u_hat = base_df["oof_u_hat"].values
            # Sigmoid of utility
            _sigmoid_u = 1.0 / (1.0 + np.exp(-_u_hat))
            combined["high_utility_pred"] = combined["reg_mean"].values * _sigmoid_u
            combined["utility_disagreement"] = np.abs(
                combined["reg_mean"].values - _u_hat
            )

        # Pairwise agreement features for the main classifier/regressor.
        if "oof_u_hat" in combined.columns and "oof_p_move" in combined.columns:
            _u_hat = np.asarray(combined["oof_u_hat"].values, dtype=float)
            _p_move = np.asarray(combined["oof_p_move"].values, dtype=float)
            _eps = 1e-9
            combined["meta_avg"] = 0.5 * (_u_hat + _p_move)
            combined["meta_diff"] = _u_hat - _p_move
            combined["meta_abs_diff"] = np.abs(combined["meta_diff"].values)
            combined["meta_rel_diff"] = combined["meta_abs_diff"].values / (
                np.abs(_u_hat) + np.abs(_p_move) + _eps
            )
            combined["meta_agreement_strength"] = np.clip(
                1.0 - combined["meta_rel_diff"].values, 0.0, 1.0
            )
            if (
                "robust_sigma_meta_reg" in combined.columns
                and "robust_sigma_meta_clf" in combined.columns
            ):
                combined["avg_robust_sigma_meta"] = 0.5 * (
                    np.asarray(combined["robust_sigma_meta_reg"].values, dtype=float)
                    + np.asarray(combined["robust_sigma_meta_clf"].values, dtype=float)
                )
            if "cv_meta_reg" in combined.columns and "cv_meta_clf" in combined.columns:
                combined["avg_cv_meta"] = 0.5 * (
                    np.asarray(combined["cv_meta_reg"].values, dtype=float)
                    + np.asarray(combined["cv_meta_clf"].values, dtype=float)
                )
            if "avg_cv_meta" in combined.columns:
                combined["meta_reliability"] = 1.0 / (
                    1.0 + combined["avg_cv_meta"].values
                )

            # ── Cross-model uncertainty & composite edge features ──
            _reg = np.asarray(
                combined.get(
                    "reg_pred", combined.get("oof_u_hat", np.zeros(len(combined)))
                ).values,
                dtype=float,
            )
            _clf_c = np.asarray(
                combined.get("clf_center", np.zeros(len(combined))).values, dtype=float
            )
            _clf_pfx = np.asarray(
                combined.get("clf_prefix_std", np.full(len(combined), np.nan)).values,
                dtype=float,
            )
            _reg_pfx = np.asarray(
                combined.get("reg_prefix_std", np.full(len(combined), np.nan)).values,
                dtype=float,
            )
            _clf_sup = np.asarray(
                combined.get(
                    "clf_leaf_support_q25", np.full(len(combined), np.nan)
                ).values,
                dtype=float,
            )
            _reg_sup = np.asarray(
                combined.get(
                    "reg_leaf_support_q25", np.full(len(combined), np.nan)
                ).values,
                dtype=float,
            )
            _clf_iqr = np.asarray(
                combined.get(
                    "clf_leaf_target_iqr_mean", np.full(len(combined), np.nan)
                ).values,
                dtype=float,
            )
            _reg_iqr = np.asarray(
                combined.get(
                    "reg_leaf_target_iqr_mean", np.full(len(combined), np.nan)
                ).values,
                dtype=float,
            )

            combined["sign_agree"] = (np.sign(_reg) * np.sign(_clf_c)).astype(
                np.float32
            )
            combined["joint_confidence"] = (np.abs(_reg) * np.abs(_clf_c)).astype(
                np.float32
            )
            _reg_z = (_reg - np.nanmean(_reg)) / (np.nanstd(_reg) + 1e-9)
            _clf_z = (_clf_c - np.nanmean(_clf_c)) / (np.nanstd(_clf_c) + 1e-9)
            _min_z = np.minimum(np.abs(_reg_z), np.abs(_clf_z))
            _sum_z = np.abs(_reg_z) + np.abs(_clf_z) + 1e-9
            combined["conflict_score"] = (
                1.0 - np.sign(_reg) * np.sign(_clf_c) * (2.0 * _min_z / _sum_z)
            ).astype(np.float32)

            _pfx_clf_f = np.where(np.isfinite(_clf_pfx), _clf_pfx, 0.0)
            _pfx_reg_f = np.where(np.isfinite(_reg_pfx), _reg_pfx, 0.0)
            combined["joint_instability"] = (
                0.5 * _pfx_clf_f + 0.5 * _pfx_reg_f
            ).astype(np.float32)

            combined["edge_unc_pen"] = (
                _reg / (1.0 + _pfx_reg_f + _pfx_clf_f + 1e-12)
            ).astype(np.float32)

            _min_sup = np.minimum(
                np.where(np.isfinite(_clf_sup), _clf_sup, 0.0),
                np.where(np.isfinite(_reg_sup), _reg_sup, 0.0),
            )
            _support_score = np.clip(np.log1p(_min_sup) / 5.0, 0.0, 1.0).astype(
                np.float32
            )
            combined["edge_support_pen"] = (_reg * _support_score).astype(np.float32)

            _joint_leaf_noise = 0.5 * np.where(
                np.isfinite(_clf_iqr), _clf_iqr, 0.0
            ) + 0.5 * np.where(np.isfinite(_reg_iqr), _reg_iqr, 0.0)
            combined["edge_noise_pen"] = (
                _reg / (1.0 + 10.0 * _joint_leaf_noise + 1e-12)
            ).astype(np.float32)

        if "cal_reg" in combined.columns and "clf" in combined.columns:
            _cal_r = np.asarray(combined["cal_reg"].values, dtype=float)
            _clf_raw = np.asarray(combined["clf"].values, dtype=float)
            _cal_finite = np.where(np.isfinite(_cal_r), _cal_r, 0.0)
            combined["calibrated_p_move"] = np.clip(
                _clf_raw * (1.0 + _cal_finite), 0.0, 1.0
            ).astype(np.float32)
            combined["cal_residual"] = _cal_finite.astype(np.float32)
            combined["cal_abs_residual"] = np.abs(_cal_finite).astype(np.float32)
            combined["cal_adjusted_confidence"] = (
                np.abs(_cal_finite) * np.abs(_clf_raw - 0.5) * 2.0
            ).astype(np.float32)

        inferred_side = next(iter(source_sides), "")
        if inferred_side:
            combined.attrs["trade_side"] = inferred_side

        # Attach metadata and realized outcomes for diagnostics
        aux_cols = [
            "timestamp",
            "symbol",
            "return",
            "is_long",
            "index",
            "oof_p_sl",
            "oof_p_to",
            "oof_p_tp",
            "oof_u_hat",
            "oof_log_mae_q70_hat",
            "oof_log_mfe_hat",
            "oof_asym_hat",
            "oof_p_move",
            "mae_ret",
            "mfe_ret",
            "u_policy_net",
            "u_policy",
            "exit_code",
            "label_policy_sl_atr_mult",
            "label_policy_tp_sl_ratio",
            "label_policy_max_hold_bars",
            "label_policy_giveback_pct",
            "atr_12_15m",
        ]
        for col in aux_cols:
            # Check across all meta-model DataFrames for this bucket for the column
            col_data = None
            for m_df in model_dfs.values():
                if col in m_df.columns:
                    col_data = m_df[col].values
                    break

            if col_data is not None:
                if col in {"timestamp", "symbol"}:
                    combined[col] = col_data
                else:
                    combined[col] = _fill_nonfinite_oof_vector(col_data, neutral=0.0)

        # ---------------------------------------------------------------------
        # RECOVERY: Fetch missing timestamp/symbol from labels if possible
        # ---------------------------------------------------------------------
        if (
            "timestamp" not in combined.columns or "symbol" not in combined.columns
        ) and "index" in combined.columns:
            try:
                # Find a label file to join with
                labels_dir = Path(data_root) / "artifacts" / run_id / "labels"
                if labels_dir.exists():
                    # Prefer the bucket-specific training labels; generic labels
                    # can have unrelated row layouts and produce degenerate timestamp recovery.
                    bucket_label_files = sorted(
                        labels_dir.glob(f"train_{bucket}_*.parquet")
                    )
                    label_file = (
                        bucket_label_files[0]
                        if bucket_label_files
                        else next(labels_dir.glob("*.parquet"), None)
                    )

                    if label_file:
                        label_df = read_parquet_projected(
                            label_file,
                            ["__ts__", "__symbol__", "ts", "timestamp", "symbol"],
                        )
                        ts_col = (
                            "__ts__"
                            if "__ts__" in label_df.columns
                            else (
                                "ts"
                                if "ts" in label_df.columns
                                else (
                                    "timestamp"
                                    if "timestamp" in label_df.columns
                                    else None
                                )
                            )
                        )
                        sym_col = (
                            "__symbol__"
                            if "__symbol__" in label_df.columns
                            else ("symbol" if "symbol" in label_df.columns else None)
                        )
                        if len(label_df) >= n:
                            if "timestamp" not in combined.columns:
                                if ts_col is not None:
                                    combined["timestamp"] = combined["index"].map(
                                        label_df[ts_col]
                                    )
                            if "symbol" not in combined.columns and sym_col is not None:
                                combined["symbol"] = combined["index"].map(
                                    label_df[sym_col]
                                )
                        # Sanity check: if recovery still collapses to <= 2 days, discard it so downstream
                        # consumers don't mistake it for a real intraday sample.
                        if "timestamp" in combined.columns:
                            _ts_chk = pd.to_datetime(
                                combined["timestamp"], utc=True, errors="coerce"
                            )
                            if (
                                _ts_chk.notna().sum() > 0
                                and _ts_chk.dt.floor("D").nunique() <= 2
                            ):
                                tprint(
                                    f"  WARNING: Recovered timestamps for {bucket} from {label_file.name} look degenerate; dropping them"
                                )
                                combined = combined.drop(columns=["timestamp"])

                        if "timestamp" in combined.columns:
                            tprint(
                                f"  Successfully recovered timestamps for {bucket} from {label_file.name}"
                            )
            except Exception as e:
                tprint(f"  Warning: Metadata recovery failed for {bucket}: {e}")

        result[bucket] = combined

        # Merge base OOF predictions if available (same bucket only)
        # Base buckets: long_mr, long_tf, short_mr, short_tf
        # Meta buckets: long_mr_reg, long_mr_early_inval, etc.
        # Match by prefix (e.g., long_mr_reg -> long_mr)
        if base_oofs:
            for base_bucket, base_df in base_oofs.items():
                if bucket.startswith(base_bucket):
                    base_by_idx = (
                        base_df.set_index("index")
                        if "index" in base_df.columns
                        else base_df
                    )
                    # Merge base model features
                    base_cols = [c for c in base_df.columns if c.startswith("base_")]
                    if "index" in combined.columns:
                        for col in base_cols:
                            if col in base_by_idx.columns:
                                combined[col] = combined["index"].map(base_by_idx[col])
                        # Preserve base OOF prediction heads used downstream by the
                        # policy optimiser and diagnostics. These are bucket-aligned
                        # with the base OOF rows and must survive the meta merge.
                        for col in [
                            "oof_u_hat",
                            "oof_log_mae_q70_hat",
                            "oof_log_mfe_hat",
                            "oof_log_dur_hat",
                            "oof_p_move",
                            "oof_p_sl",
                            "oof_p_to",
                            "oof_p_tp",
                            "oof_asym_hat",
                        ]:
                            if (
                                col in base_by_idx.columns
                                and col not in combined.columns
                            ):
                                combined[col] = combined["index"].map(base_by_idx[col])

                    # Merge essential metadata if still missing
                    for meta_col in ["timestamp", "symbol"]:
                        if (
                            meta_col not in combined.columns
                            and meta_col in base_df.columns
                        ):
                            if "index" in combined.columns:
                                combined[meta_col] = combined["index"].map(
                                    base_by_idx[meta_col]
                                )
                            else:
                                combined[meta_col] = base_df[meta_col].values

                    tprint(
                        f"  Merged base OOF metadata/features into {bucket} (base: {base_bucket})"
                    )
                    break

        base_feature_cols = [c for c in combined.columns if c.startswith("base_")]
        if not base_feature_cols:
            tprint(
                f"  WARNING: {bucket} has no base model OOF features merged; ridge combiner will run meta-only for this bucket"
            )

    tprint(f"Loaded OOF predictions for {len(result)} buckets: {list(result.keys())}")

    # Inject required config-defined features into the inference data
    from extreme_price_movements.config import CFG
    from extreme_price_movements.data_store import load_features_selected

    try:
        from extreme_price_movements.pipeline_steps import (
            load_features_for_stage_or_all,
        )
    except Exception:
        from extreme_price_movements.slice_plan_store import (
            load_features_for_stage_or_all,
        )
    from extreme_price_movements.training import _fast_lookup

    cfg = dict(CFG)
    required_features = set()
    required_features.update(cfg.get("position_sizer_features", []))

    # Identify missing features across all buckets
    missing_feats = set()
    for bdf in result.values():
        missing_feats.update([f for f in required_features if f not in bdf.columns])

    # Remove known meta/prediction cols that aren't base features
    known_preds = {
        "score",
        "reg",
        "reg_mean",
        "reg_std",
        "reg_range",
        "utility",
        "mae_q70",
        "mfe",
        "early_inval",
        "oof_u_hat",
        "oof_log_mae_q70_hat",
        "oof_log_mfe_hat",
        "oof_asym_hat",
        "oof_p_move",
        "oof_p_sl",
        "oof_p_tp",
        "oof_p_time",
        "Upside",
        "Downside",
        "EdgeSharpe",
        "risk_reward_ratio",
        "high_utility_pred",
        "risk_adjusted_pred",
        "utility_disagreement",
        "base_H2",
        "base_H4",
    }
    missing_feats = missing_feats - known_preds

    if missing_feats:
        tprint(
            f"Position sizer fetching {len(missing_feats)} missing config features..."
        )
        all_syms = set()
        for bdf in result.values():
            if "symbol" in bdf.columns:
                all_syms.update(bdf["symbol"].unique())

        feats = load_features_for_stage_or_all(
            cfg,
            ts_sig=find_best_feature_snapshot_ts(data_root, run_id),
            root_dir=data_root,
            feature_keys=list(missing_feats),
            symbols=list(all_syms),
        )

        if feats:
            tprint(
                "Injecting fetched features into position sizer datasets (memory optimized)..."
            )

            # Identify missing features actually needed
            needed_keys = set()
            for bdf in result.values():
                needed_keys.update([k for k in missing_feats if k not in bdf.columns])

            actual_missing = [k for k in needed_keys if k in feats]
            tprint(f"Injecting {len(actual_missing)} features into sizer datasets...")

            # Temporary dict for new cols
            new_cols_per_bucket = {bucket: {} for bucket in result.keys()}

            for i, k in enumerate(actual_missing, 1):
                if i % 5 == 0:
                    tprint(
                        f"  Sizer feature injection progress: {i}/{len(actual_missing)}"
                    )

                feat_df = feats.pop(k)

                for bucket, bdf in result.items():
                    if "timestamp" not in bdf.columns or "symbol" not in bdf.columns:
                        continue
                    if k in bdf.columns:
                        continue

                    _ts = pd.to_datetime(bdf["timestamp"]).values
                    _sym = bdf["symbol"].values

                    v = np.nan_to_num(_fast_lookup(feat_df, _ts, _sym), nan=0.0).astype(
                        np.float32
                    )
                    new_cols_per_bucket[bucket][k] = v

                del feat_df

            for bucket, new_cols in new_cols_per_bucket.items():
                if new_cols:
                    bdf = result[bucket]
                    result[bucket] = pd.concat(
                        [bdf, pd.DataFrame(new_cols, index=bdf.index)], axis=1
                    )

            tprint("Position sizer feature injection complete.")
            del feats
            del new_cols_per_bucket
            import gc

            gc.collect()

    # These are metadata columns to exclude from features
    # Note: oof_u_hat, oof_log_mae_q70_hat, oof_log_mfe_hat, oof_log_dur_hat
    # are now included as features (meta model auxiliary predictions)
    for bk, bdf in result.items():
        pred_cols = [c for c in bdf.columns if c not in _META_COLS]
        tprint(f"  {bk}: {len(bdf)} samples, pred_cols={pred_cols}")
    return result


def _load_separate_outcomes(data_root, run_id):
    from pathlib import Path

    outcomes_path = Path(data_root) / "artifacts" / run_id / "trade_outcomes.parquet"
    if outcomes_path.exists():
        outcomes = pd.read_parquet(outcomes_path)
        tprint(f"Loaded trade outcomes from {outcomes_path}: {len(outcomes)} trades")
        return outcomes
    return None


def load_trade_outcomes(
    data_root: str, run_id: str, oof_df: pd.DataFrame
) -> pd.DataFrame:
    """Load or construct trade outcomes from OOF predictions data.

    The OOF predictions now include trade context (return, is_long, timestamp, symbol).
    This function constructs the trade outcomes DataFrame needed by the ridge sizer.

    Args:
        data_root: Root directory for data
        run_id: Training run identifier
        oof_df: DataFrame with OOF predictions and trade context

    Returns:
        DataFrame with columns [return, is_long] and optionally [timestamp, symbol]
    """
    # CRITICAL FIX: Detect if returns are in percentage points
    # Percentage-point returns have mean > 0.05 (5%)
    # Decimal returns for 15m bars typically have mean < 0.005 (0.5%)
    # Threshold set to 0.05 (5%) for 15m crypto bars - catches percentage points
    # while avoiding false positives from high-volatility decimal returns.
    raw_returns = np.asarray(oof_df["return"].values, dtype=np.float32)
    if np.abs(np.mean(raw_returns)) > 0.05:
        tprint(
            f"  WARNING: Returns appear to be in percentage points (mean={np.mean(raw_returns):.6f}). Converting to decimal."
        )
        raw_returns = raw_returns / 100.0

    # Keep returns as simple returns for PnL calculations
    # Conversion to log returns happens in RidgePositionSizer.fit
    outcomes = pd.DataFrame(
        {
            "return": raw_returns,
            "is_long": oof_df["is_long"].values if "is_long" in oof_df.columns else 1,
        }
    )
    if "timestamp" in oof_df.columns:
        outcomes["timestamp"] = oof_df["timestamp"].values
    if "symbol" in oof_df.columns:
        outcomes["symbol"] = oof_df["symbol"].values
    if "u_policy_net" in oof_df.columns:
        # CRITICAL FIX: Detect if u_policy_net is in percentage points
        raw_u_policy_net = np.asarray(oof_df["u_policy_net"].values, dtype=np.float32)
        if np.abs(np.mean(raw_u_policy_net)) > 0.05:
            tprint(
                f"  WARNING: u_policy_net appears to be in percentage points (mean={np.mean(raw_u_policy_net):.6f}). Converting to decimal."
            )
            raw_u_policy_net = raw_u_policy_net / 100.0
        outcomes["u_policy_net"] = raw_u_policy_net
    if "u_policy" in oof_df.columns:
        # CRITICAL FIX: Detect if u_policy is in percentage points
        raw_u_policy = np.asarray(oof_df["u_policy"].values, dtype=np.float32)
        if np.abs(np.mean(raw_u_policy)) > 0.05:
            tprint(
                f"  WARNING: u_policy appears to be in percentage points (mean={np.mean(raw_u_policy):.6f}). Converting to decimal."
            )
            raw_u_policy = raw_u_policy / 100.0
        outcomes["u_policy"] = raw_u_policy
    for c in oof_df.columns:
        if c.startswith("u_tbm_"):
            raw_u_tbm = np.asarray(oof_df[c].values, dtype=np.float32)
            if np.abs(np.mean(raw_u_tbm)) > 0.05:
                tprint(
                    f"  WARNING: {c} appears to be in percentage points (mean={np.mean(raw_u_tbm):.6f}). Converting to decimal."
                )
                raw_u_tbm = raw_u_tbm / 100.0
            outcomes[c] = raw_u_tbm
    if "exit_code" in oof_df.columns:
        outcomes["exit_code"] = oof_df["exit_code"].values
    if "entry_price" in oof_df.columns:
        outcomes["entry_price"] = oof_df["entry_price"].values

    # Triple-barrier labels: 0=SL, 1=TIME, 2=TP (cost-adjusted)
    _has_barrier_data = (
        "mfe_ret" in outcomes.columns
        and "mae_ret" in outcomes.columns
        and "return" in outcomes.columns
    )
    if _has_barrier_data:
        _mfe = np.abs(np.asarray(outcomes["mfe_ret"].values, dtype=np.float32))
        _mae = np.abs(np.asarray(outcomes["mae_ret"].values, dtype=np.float32))
        _barrier = np.clip(np.maximum(_mae * 2.5, 1e-4), 0.005, 0.2)
        _tp_mult = 0.50
        _sl_mult = 0.18
        if "label_policy_tp_sl_ratio" in outcomes.columns:
            _ratio = np.asarray(
                outcomes["label_policy_tp_sl_ratio"].values, dtype=np.float32
            )
            _ratio_clean = np.where(np.isfinite(_ratio) & (_ratio > 0), _ratio, 3.0)
            _tp_mult = np.clip(_ratio_clean * _sl_mult, 0.1, 2.0)
        _tp_dist = _tp_mult * _barrier - 0.003
        _sl_dist = _sl_mult * _barrier + 0.003
        _is_tp = _mfe >= np.maximum(_tp_dist, 1e-6)
        _is_sl = _mae >= np.maximum(_sl_dist, 1e-6)
        _tbm = np.ones(len(outcomes), dtype=np.int8)
        _tbm[_is_sl & ~_is_tp] = 0
        _tbm[_is_tp] = 2
        outcomes["tbm_label"] = _tbm

    # Policy-aware simulation columns
    policy_cols = [
        "label_policy_sl_atr_mult",
        "label_policy_tp_sl_ratio",
        "label_policy_max_hold_bars",
        "label_policy_giveback_pct",
        "atr_12_15m",
    ]
    for c in policy_cols:
        if c in oof_df.columns:
            outcomes[c] = oof_df[c].values

    # Copy aux diagnostic columns
    aux_cols = [
        "oof_u_hat",
        "oof_log_mae_q70_hat",
        "oof_log_mfe_hat",
        "oof_p_tp",
        "oof_p_sl",
        "oof_p_time",
        "mae_ret",
        "mfe_ret",
        "duration",
    ]
    for c in aux_cols:
        if c in oof_df.columns:
            outcomes[c] = oof_df[c].values

    # Preserve additional prediction columns that downstream policy code can use
    # as a fallback when the dedicated utility head is absent in older artifacts.
    prediction_cols = [
        "utility",
        "oof_pred",
        "oof_pred_oriented",
        "reg",
        "reg_mean",
        "reg_std",
        "clf",
        "oof_p_move",
        "oof_log_mae_q70_hat",
        "oof_log_mfe_hat",
        "oof_asym_hat",
    ]
    for c in prediction_cols:
        if c in oof_df.columns and c not in outcomes.columns:
            outcomes[c] = oof_df[c].values

    if "return" not in outcomes.columns:
        # Final fallback: try separate file
        sep_outcomes = _load_separate_outcomes(data_root, run_id)
        if sep_outcomes is not None:
            return sep_outcomes
        raise FileNotFoundError(
            f"No 'return' column in OOF and no trade_outcomes.parquet found for {run_id}"
        )

    tprint(f"Constructed base trade outcomes from OOF context: {len(outcomes)} trades")

    # Populate 15m path features directly
    if "symbol" in outcomes.columns and "timestamp" in outcomes.columns:
        tprint(
            "Populating 15m future paths for policy optimization from HF data cache..."
        )
        future_opens = []
        future_highs = []
        future_lows = []
        future_closes = []
        atr_15m = []

        if "mae_ret" in outcomes.columns:
            atr_source = outcomes["mae_ret"].values
        elif "atr" in outcomes.columns:
            atr_source = outcomes["atr"].values
        else:
            atr_source = np.full(len(outcomes), 0.02)

        # Iterate per symbol to efficiently load cached _load_existing_data
        outcomes_with_index = outcomes.copy()
        outcomes_with_index["_orig_idx"] = np.arange(len(outcomes))

        all_future_opens = np.empty(len(outcomes), dtype=object)
        all_future_highs = np.empty(len(outcomes), dtype=object)
        all_future_lows = np.empty(len(outcomes), dtype=object)
        all_future_closes = np.empty(len(outcomes), dtype=object)
        all_atr_15m = np.empty(len(outcomes), dtype=float)

        grouped = outcomes_with_index.groupby("symbol")
        for symbol, group in grouped:
            try:
                df_15m = _load_existing_data(symbol, allow_quote_fallback=True)
                orig_idx_arr = group["_orig_idx"].to_numpy(dtype=np.int64)
                if df_15m.empty:
                    # Fill empty lists
                    empty = np.array([], dtype=float)
                    all_future_opens[orig_idx_arr] = [empty] * len(orig_idx_arr)
                    all_future_highs[orig_idx_arr] = [empty] * len(orig_idx_arr)
                    all_future_lows[orig_idx_arr] = [empty] * len(orig_idx_arr)
                    all_future_closes[orig_idx_arr] = [empty] * len(orig_idx_arr)
                    all_atr_15m[orig_idx_arr] = atr_source[orig_idx_arr].astype(
                        float, copy=False
                    )
                    continue

                ts_series = pd.to_datetime(group["timestamp"])
                ts_values = (
                    ts_series.dt.tz_localize(None)
                    if ts_series.dt.tz is not None
                    else ts_series
                )
                df_15m_index = (
                    df_15m.index.tz_localize(None)
                    if df_15m.index.tz is not None
                    else df_15m.index
                )

                open_arr = df_15m["open"].values
                high_arr = df_15m["high"].values
                low_arr = df_15m["low"].values
                close_arr = df_15m["close"].values

                left_indices = df_15m_index.searchsorted(ts_values, side="left")
                valid = left_indices < len(df_15m)
                invalid_idx = orig_idx_arr[~valid]
                if len(invalid_idx) > 0:
                    empty = np.array([], dtype=float)
                    all_future_opens[invalid_idx] = [empty] * len(invalid_idx)
                    all_future_highs[invalid_idx] = [empty] * len(invalid_idx)
                    all_future_lows[invalid_idx] = [empty] * len(invalid_idx)
                    all_future_closes[invalid_idx] = [empty] * len(invalid_idx)
                    all_atr_15m[invalid_idx] = atr_source[invalid_idx].astype(
                        float, copy=False
                    )

                if np.any(valid):
                    valid_orig = orig_idx_arr[valid]
                    valid_left = left_indices[valid].astype(np.int64, copy=False)
                    valid_end = np.minimum(valid_left + 24, len(df_15m))
                    all_future_opens[valid_orig] = [
                        np.asarray(open_arr[s:e], dtype=np.float64)
                        for s, e in zip(valid_left, valid_end)
                    ]
                    all_future_highs[valid_orig] = [
                        np.asarray(high_arr[s:e], dtype=np.float64)
                        for s, e in zip(valid_left, valid_end)
                    ]
                    all_future_lows[valid_orig] = [
                        np.asarray(low_arr[s:e], dtype=np.float64)
                        for s, e in zip(valid_left, valid_end)
                    ]
                    all_future_closes[valid_orig] = [
                        np.asarray(close_arr[s:e], dtype=np.float64)
                        for s, e in zip(valid_left, valid_end)
                    ]

                    tr_full = np.full(len(close_arr), np.nan, dtype=np.float64)
                    if len(close_arr) > 1:
                        tr_full[1:] = np.maximum(
                            high_arr[1:] - low_arr[1:],
                            np.maximum(
                                np.abs(high_arr[1:] - close_arr[:-1]),
                                np.abs(low_arr[1:] - close_arr[:-1]),
                            ),
                        )
                    tr_roll = (
                        pd.Series(tr_full)
                        .rolling(12, min_periods=12)
                        .mean()
                        .to_numpy(dtype=np.float64)
                    )
                    atr_vals = atr_source[valid_orig].astype(np.float64, copy=False)
                    atr_mask = valid_left >= 13
                    if np.any(atr_mask):
                        idx_prev = valid_left[atr_mask] - 1
                        atr_calc = tr_roll[idx_prev] / np.maximum(
                            close_arr[idx_prev], 1e-12
                        )
                        atr_vals = atr_vals.copy()
                        atr_vals[atr_mask] = np.where(
                            np.isfinite(atr_calc), atr_calc, atr_vals[atr_mask]
                        )
                    all_atr_15m[valid_orig] = atr_vals

            except Exception as e:
                # On error, fill empty
                empty = np.array([], dtype=float)
                all_future_opens[orig_idx_arr] = [empty] * len(orig_idx_arr)
                all_future_highs[orig_idx_arr] = [empty] * len(orig_idx_arr)
                all_future_lows[orig_idx_arr] = [empty] * len(orig_idx_arr)
                all_future_closes[orig_idx_arr] = [empty] * len(orig_idx_arr)
                all_atr_15m[orig_idx_arr] = atr_source[orig_idx_arr].astype(
                    float, copy=False
                )

        outcomes["future_opens"] = all_future_opens.tolist()
        outcomes["future_highs"] = all_future_highs.tolist()
        outcomes["future_lows"] = all_future_lows.tolist()
        outcomes["future_closes"] = all_future_closes.tolist()
        outcomes["atr_12_15m"] = all_atr_15m.tolist()

        if "entry_price" not in outcomes.columns:
            # We don't have entry_price, use closes at `ts` as fallback if available
            entry_prices = np.full(len(outcomes), 1.0, dtype=float)
            for i, p in enumerate(outcomes["future_opens"]):
                if len(p) > 0:
                    entry_prices[i] = p[0]
            outcomes["entry_price"] = entry_prices
    return outcomes


def load_tpsl_params(data_root: str, run_id: str) -> Optional[Dict]:
    """Load optimized TP/SL parameters from tpsl_optimiser output.

    Args:
        data_root: Root directory for data
        run_id: Training run identifier

    Returns:
        Dict with TP/SL parameters, or None if not found
    """
    tpsl_path = Path(data_root) / "artifacts" / run_id / "tpsl_params.json"

    if tpsl_path.exists():
        with open(tpsl_path, "r") as f:
            params = json.load(f)
        tprint(f"Loaded TP/SL params from {tpsl_path}")
        return params

    # Try alternative location
    tpsl_path = Path(data_root) / "artifacts" / run_id / "tpsl" / "best_params.json"
    if tpsl_path.exists():
        with open(tpsl_path, "r") as f:
            params = json.load(f)
        tprint(f"Loaded TP/SL params from {tpsl_path}")
        return params

    tprint("No TP/SL params found, will use defaults")
    return None


def load_price_panel(data_root: str) -> Optional[Dict[str, pd.DataFrame]]:
    """Load price panel data for policy-aware labeling.

    Args:
        data_root: Root directory for data

    Returns:
        Dict with 'open', 'high', 'low', 'close' DataFrames, or None if not found
    """
    # Try common locations for price panel data
    panel_paths = [
        Path(data_root) / "price_panel.parquet",
        Path(data_root) / "ohlc_panel.parquet",
        Path(data_root) / "processed" / "price_panel.parquet",
    ]

    for panel_path in panel_paths:
        if panel_path.exists():
            tprint(f"Loading price panel from {panel_path}")
            panel_df = pd.read_parquet(panel_path)

            # Check if it's a multi-index format or wide format
            if isinstance(panel_df.index, pd.MultiIndex):
                # Long format: (timestamp, symbol) as index
                # Pivot to wide format
                panel_df = panel_df.reset_index()
                if "timestamp" in panel_df.columns and "symbol" in panel_df.columns:
                    price_panel = {}
                    for col in ["open", "high", "low", "close"]:
                        if col in panel_df.columns:
                            price_panel[col] = panel_df.pivot(
                                index="timestamp", columns="symbol", values=col
                            )
                    if len(price_panel) == 4:
                        tprint(
                            f"Loaded price panel: {len(price_panel['open'])} timestamps, "
                            f"{len(price_panel['open'].columns)} symbols"
                        )
                        return price_panel
            else:
                # Wide format or column-multiindex
                # Check for column structure
                if isinstance(panel_df.columns, pd.MultiIndex):
                    # Columns like (symbol, ohlc)
                    price_panel = {}
                    for ohlc in ["open", "high", "low", "close"]:
                        try:
                            price_panel[ohlc] = panel_df.xs(ohlc, level=1, axis=1)
                        except KeyError:
                            # Try level=0
                            price_panel[ohlc] = panel_df.xs(ohlc, level=0, axis=1)
                    if len(price_panel) == 4:
                        tprint(
                            f"Loaded price panel: {len(price_panel['open'])} timestamps"
                        )
                        return price_panel

    tprint(
        "Warning: No price panel found. Policy-aware labeling will not be available."
    )
    return None


def main():
    """Main entry point for the ridge position sizer runner."""
    parser = argparse.ArgumentParser(
        description="Run Ridge Position Sizer on training artifacts"
    )
    parser.add_argument(
        "--data-root", default="data", help="Data root directory (default: data)"
    )
    parser.add_argument(
        "--run-id", default=None, help="Training run ID (default: latest)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for sizer weights (default: artifacts/{run_id}/ridge_sizer)",
    )
    parser.add_argument(
        "--use-policy-labels",
        action="store_true",
        help="Use policy-aware labeling with TP/SL simulation",
    )
    parser.add_argument(
        "--max-hold-hours",
        type=int,
        default=24,
        help="Maximum holding period in hours (default: 24)",
    )
    parser.add_argument(
        "--cost-pct",
        type=float,
        default=0.0025,
        help="Transaction cost as decimal (default: 0.0025 = 0.25%%)",
    )
    parser.add_argument("--n-jobs", type=int, default=4, help="Number of parallel jobs")
    parser.add_argument("--patience", type=int, default=20, help="HPO patience")
    parser.add_argument(
        "--directions", nargs="+", default=["long", "short"], help="Directions to run"
    )
    parser.add_argument(
        "--buckets", nargs="+", default=None, help="Specific buckets to run"
    )
    parser.add_argument(
        "--n-trials", type=int, default=None, help="Maximum trials per HPO stage"
    )
    parser.add_argument(
        "--force-targets",
        nargs="+",
        default=None,
        help="Optional shortlist of training targets to force-test for this run",
    )
    args = parser.parse_args()

    tprint("=" * 80)
    tprint("RIDGE POSITION SIZER RUNNER")
    tprint("=" * 80)

    # Find run ID
    run_id = args.run_id or find_latest_run_id(args.data_root)
    tprint(f"Using run ID: {run_id}")

    # Load OOF predictions per bucket
    try:
        bucket_oofs = load_meta_oof_predictions(
            args.data_root,
            run_id,
            require_meta_barrier_probs=False,
        )
    except (FileNotFoundError, RuntimeError) as e:
        tprint(f"Error: {e}")
        tprint(
            "Meta model OOF predictions are incomplete for a policy-aligned ridge sizer run."
        )
        tprint(
            "Ensure training.py has been run with meta model training enabled and barrier probabilities exported."
        )
        return 1

    # Set up output directory
    output_dir = args.output_dir or os.path.join(
        args.data_root, "artifacts", run_id, "ridge_sizer"
    )
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Group buckets by direction using strategy definitions from final_rule_registry.csv
    # Strategies loaded via load_inference_candidate_mask_params_per_bucket() provide
    # strategy_id and trade_side for proper direction grouping.
    # -------------------------------------------------------------------------
    # Load strategies to get trade_side for each bucket
    strategies = load_inference_candidate_mask_params_per_bucket(
        top_n=2, ranking_metric="score_for_best_params"
    )
    strategy_side_map = {s["strategy_id"]: s["trade_side"] for s in strategies}

    direction_groups = {"long": {}, "short": {}}
    for bucket_name, oof_preds in bucket_oofs.items():
        # Look up trade_side from strategy map, fallback to legacy naming convention
        direction = strategy_side_map.get(bucket_name)
        if direction is None:
            # Legacy fallback: infer from bucket name
            direction = "long" if bucket_name.startswith("long") else "short"
        direction_groups[direction][bucket_name] = oof_preds

    all_direction_results = {}  # direction -> {weights, params, metrics, buckets}

    for direction, dir_buckets in direction_groups.items():
        if args.directions and direction not in args.directions:
            continue
        if not dir_buckets:
            continue
        tprint("=" * 80)
        tprint(
            f"Ridge Position Sizer — direction: {direction.upper()} ({list(dir_buckets.keys())})"
        )

        # Each bucket has its own event set (different lengths: long_mr=7551, long_tf=5167).
        # We train one Ridge per bucket within the direction, then store all weights
        # together in a single per-direction weight file. At inference the engine
        # applies the correct bucket's weights based on which bucket fired.
        dir_weights: Dict = {}
        dir_params: Dict = {}
        dir_metrics: Dict = {}
        last_best_params: Optional[Dict] = None

        for bucket_name, oof_preds in dir_buckets.items():
            if args.buckets and bucket_name not in args.buckets:
                continue
            try:
                trade_outcomes = load_trade_outcomes(args.data_root, run_id, oof_preds)
            except FileNotFoundError as e:
                tprint(f"  Skipping {bucket_name}: {e}")
                continue
            if "return" not in trade_outcomes.columns:
                tprint(f"  Skipping {bucket_name}: missing 'return' column")
                continue

            pred_cols = [c for c in oof_preds.columns if c not in _META_COLS]
            if not pred_cols:
                tprint(f"  Skipping {bucket_name}: no prediction columns")
                continue
            oof_pred_df = oof_preds[pred_cols].copy()
            tprint(f"  {bucket_name}: {len(oof_pred_df)} rows, features={pred_cols}")

            # Initialize entry policy config (will be set inside if block)
            _bp_cfg = {}

            # If optimize params are present for this bucket, align sizer rows to entry policy place-order mask.
            _run_models_path = (
                Path(args.data_root)
                / "artifacts"
                / run_id
                / "ridge_sizer"
                / "strategy_params.json"
            )
            if _run_models_path.exists():
                try:
                    _bp_blob = json.loads(_run_models_path.read_text())
                    _bp_buckets = _bp_blob.get("buckets", {})
                    _bp_cfg = flatten_bucket_policy(
                        _bp_buckets.get(bucket_name, {})
                        or _bp_buckets.get(bucket_name.lower(), {})
                        or _bp_buckets.get(bucket_name.upper(), {})
                    )
                    if _bp_cfg.get("entry_policy"):
                        _scores = np.asarray(
                            oof_pred_df[pred_cols[0]].values, dtype=float
                        )
                        _atr_vec = np.asarray(
                            trade_outcomes.get(
                                "mae_ret", pd.Series(np.full(len(trade_outcomes), 0.02))
                            ).values,
                            dtype=float,
                        )
                        _atr_vec = np.clip(
                            np.where(np.isfinite(_atr_vec), np.abs(_atr_vec), 0.02),
                            1e-4,
                            0.5,
                        )
                        _mask = np.ones(len(trade_outcomes), dtype=bool)
                        for _i in range(len(_mask)):
                            _pol = compute_entry_policy_decision(
                                entry_px=1.0,
                                atr_frac=float(_atr_vec[_i]),
                                score=float(_scores[_i]) if _i < len(_scores) else 0.0,
                                bucket_cfg=_bp_cfg,
                            )
                            _mask[_i] = bool(_pol.get("place_order", True))
                        trade_outcomes = trade_outcomes.loc[_mask].reset_index(
                            drop=True
                        )
                        oof_pred_df = oof_pred_df.loc[_mask].reset_index(drop=True)
                        tprint(
                            f"  {bucket_name}: policy mask kept {_mask.sum()}/{len(_mask)} rows"
                        )
                except Exception as _e_mask:
                    tprint(f"  {bucket_name}: policy mask skipped ({_e_mask})")

            timestamps = (
                trade_outcomes["timestamp"].values
                if "timestamp" in trade_outcomes.columns
                else None
            )
            symbols = (
                trade_outcomes["symbol"].values
                if "symbol" in trade_outcomes.columns
                else None
            )

            # Load candidate threshold config (from compare_candidate_thresholds.py)
            _candidate_cfg = None
            try:
                from extreme_price_movements.offline_optimisers.params_store import (
                    CANDIDATE_BEST_PARAMS_CSV,
                )

                if CANDIDATE_BEST_PARAMS_CSV.exists():
                    cand_df = pd.read_csv(CANDIDATE_BEST_PARAMS_CSV)
                    # Get bucket-specific params
                    bucket_row = (
                        cand_df[cand_df["bucket"] == bucket_name.upper()]
                        if "bucket" in cand_df.columns
                        else None
                    )
                    if bucket_row is not None and len(bucket_row) > 0:
                        _candidate_cfg = {
                            "extreme_price_pct": float(
                                bucket_row.iloc[0].get("extreme_price_pct", 0.0)
                            ),
                            "min_vol_zscore": float(
                                bucket_row.iloc[0].get("min_vol_zscore", -10.0)
                            ),
                        }
                        tprint(
                            f"  {bucket_name}: loaded candidate threshold config: {_candidate_cfg}"
                        )
            except Exception as _e_cand:
                tprint(
                    f"  {bucket_name}: candidate threshold config not loaded ({_e_cand})"
                )

            # Load entry policy config
            _entry_policy_cfg = _bp_cfg if _bp_cfg.get("entry_policy") else None

            try:
                sizer, metrics = run_ridge_position_sizer_step(
                    oof_preds=oof_pred_df,
                    trade_outcomes=trade_outcomes,
                    timestamps=timestamps,
                    cfg={
                        "cost_pct": args.cost_pct,
                        "sizer_n_jobs": 1,
                        "sizer_use_nested_cv": True,
                        "sizer_stage1_n_trials": args.n_trials,
                        "sizer_stage2_n_trials": args.n_trials,
                        "sizer_tree_hpo_trials": args.n_trials,
                        "patience": args.patience,
                        "sizer_stage1_cv_folds": 3,
                        "sizer_stage2_cv_folds": 3,
                        "sizer_stage2_lock_formula": False,
                        "sizer_target_train_fraction": 0.50,
                        "sizer_oos_fraction": 0.30,
                        "sizer_min_oos_days": 28,
                        "sizer_repeated_oos_splits": 2,
                        "sizer_max_fit_samples": 12000,
                        "sizer_forced_target_candidates": args.force_targets,
                        "label_policy_ab_tbm_grid": [
                            {
                                "name": "tbm_50_25_h8",
                                "tp_pct": 0.0050,
                                "sl_pct": 0.0025,
                                "max_hold_bars": 8,
                            },
                            {
                                "name": "tbm_50_25_h16",
                                "tp_pct": 0.0050,
                                "sl_pct": 0.0025,
                                "max_hold_bars": 16,
                            },
                            {
                                "name": "tbm_50_25_h24",
                                "tp_pct": 0.0050,
                                "sl_pct": 0.0025,
                                "max_hold_bars": 24,
                            },
                            {
                                "name": "tbm_100_50_h8",
                                "tp_pct": 0.0100,
                                "sl_pct": 0.0050,
                                "max_hold_bars": 8,
                            },
                            {
                                "name": "tbm_100_50_h16",
                                "tp_pct": 0.0100,
                                "sl_pct": 0.0050,
                                "max_hold_bars": 16,
                            },
                            {
                                "name": "tbm_100_50_h24",
                                "tp_pct": 0.0100,
                                "sl_pct": 0.0050,
                                "max_hold_bars": 24,
                            },
                        ],
                    },
                    save_model=True,
                    run_id=run_id,
                    symbols=symbols,
                    bucket_name=bucket_name,
                    entry_policy_config=_entry_policy_cfg,
                    candidate_threshold_config=_candidate_cfg,
                    warm_start_params=last_best_params,
                )
                last_best_params = sizer.best_params_
                bkt_weights = sizer.get_weights()
                # Prefix with bucket name so the combined manifest stays unambiguous
                for wname, wval in bkt_weights.items():
                    dir_weights[f"{bucket_name}_{wname}"] = wval
                dir_params[bucket_name] = sizer.best_params_
                dir_metrics[bucket_name] = metrics
                tprint(f"  {bucket_name} weights: {bkt_weights}")
            except Exception as e:
                tprint(f"  {bucket_name} failed: {e}")
                import traceback

                traceback.print_exc()
                continue

        if not dir_weights:
            tprint(f"  No weights produced for direction {direction}, skipping")
            continue

        # Save per-direction weight file
        dir_weights_path = os.path.join(output_dir, f"sizer_weights_{direction}.json")
        with open(dir_weights_path, "w") as f:
            json.dump(
                {
                    "direction": direction,
                    "weights": dir_weights,
                    "params_per_bucket": dir_params,
                    "buckets": list(dir_buckets.keys()),
                    "run_id": run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )
        tprint(f"  Saved {direction} sizer weights to {dir_weights_path}")

        all_direction_results[direction] = {
            "weights": dir_weights,
            "params": dir_params,
            "metrics": dir_metrics,
            "buckets": list(dir_buckets.keys()),
        }

    # Save combined manifest (backward-compatible: flattens all weights)
    all_weights = {}
    all_params = {}
    for direction, res in all_direction_results.items():
        all_weights.update(res["weights"])
        for bkt in res["buckets"]:
            all_params[bkt] = res["params"]

    weights_path = os.path.join(output_dir, "sizer_weights.json")
    with open(weights_path, "w") as f:
        json.dump(
            {
                "weights": all_weights,
                "params_per_bucket": all_params,
                "directions": {
                    d: {"buckets": r["buckets"], "params": r["params"]}
                    for d, r in all_direction_results.items()
                },
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )
    tprint(f"Saved combined manifest to {weights_path}")

    # Print summary
    tprint("=" * 80)
    tprint("RIDGE POSITION SIZER COMPLETE")
    tprint("=" * 80)
    for direction, res in all_direction_results.items():
        tprint(f"  {direction.upper()} sizer — buckets: {res['buckets']}")
        for name, w in res["weights"].items():
            tprint(f"    {name}: {w:.4f}")
    tprint(f"Output directory: {output_dir}")
    tprint("=" * 80)

    from extreme_price_movements.utils import dump_pipeline_warnings

    dump_pipeline_warnings(run_id)

    # Generate Automated Markdown Manifest
    try:
        report_path = Path(output_dir).parent / f"ridge_sizer_metrics_report.md"
        with open(report_path, "w") as f:
            f.write(f"# Ridge Position Sizer Metrics Report\n\n**Run ID:** {run_id}\n")

            for d, res in all_direction_results.items():
                f.write(f"\n## Direction: {d.upper()}\n")
                for bkt in res["buckets"]:
                    f.write(f"\n### Bucket: {bkt}\n")
                    bkt_metrics = res["metrics"].get(bkt, {})
                    best_p = bkt_metrics.get("best_params", {})
                    winner = best_p.get("race_winner", "Unknown")
                    f.write(f"- **Race Winner**: `{winner}`\n")
                    f.write(
                        f"- **Limit Offset**: `{'enabled' if bkt_metrics.get('limit_offset_enabled') else 'disabled'}`\n"
                    )
                    if bkt_metrics.get("selected_training_target_name"):
                        f.write(
                            f"- **Training Target**: `{bkt_metrics.get('selected_training_target_name')}` "
                            f"(family=`{bkt_metrics.get('selected_training_target_family', 'unknown')}`)\n"
                        )
                    f.write(
                        f"- **Ranking Gate (top_k_pct)**: `{best_p.get('ranking_top_k_pct', best_p.get('top_k_pct', 0.30)):.2%}`\n"
                    )
                    f.write(
                        f"- **Execution Top-K**: `{best_p.get('top_k_pct', 0.30):.2%}`\n"
                    )
                    f.write(
                        f"- **Position Sizing Formula**: `{best_p.get('sizing_formula', 'linear')}`\n"
                    )
                    f.write(f"- **Base Size**: `{best_p.get('base_size', 0.05):.2%}`\n")
                    f.write(
                        f"- **Rank Multiplier**: `{best_p.get('rank_multiplier', 0.10):.2%}`\n"
                    )
                    f.write(
                        f"- **Squash Function**: `{best_p.get('squash_fn', 'tanh')}`\n"
                    )
                    f.write(f"- **Squash k**: `{best_p.get('squash_k', 1.0):.2f}`\n")

                    f.write("- **CV Performance (Aligned Holdouts)**:\n")
                    if "cv_best_selector_column" in bkt_metrics:
                        f.write(
                            f"  - **CV Basis**: `{bkt_metrics.get('cv_best_selector_column')}`\n"
                        )
                    f.write(
                        f"  - **Total PnL**: {format_return_as_pct(bkt_metrics.get('cv_best_pnl_total', 0.0))}\n"
                    )
                    f.write(
                        f"  - **PnL/Day**: {format_return_as_pct(bkt_metrics.get('cv_best_pnl_per_day', 0.0))}\n"
                    )
                    f.write(
                        f"  - **Trades/Day**: {bkt_metrics.get('cv_best_trades_per_day', 0.0):.4f}\n"
                    )
                    f.write(
                        f"  - **Sortino**: {bkt_metrics.get('cv_best_sortino', 0.0):.4f}\n"
                    )
                    f.write(
                        f"  - **MaxDD**: {format_return_as_pct(bkt_metrics.get('cv_best_maxdd', 0.0))}\n"
                    )
                    f.write(
                        f"  - **WinRate**: {bkt_metrics.get('cv_best_winrate', 0.0):.2%}\n"
                    )
                    f.write(
                        f"  - **Profit Factor**: {bkt_metrics.get('cv_best_profit_factor', 0.0):.2f}\n"
                    )
                    f.write(
                        f"  - **Avg Win/Loss**: {format_return_as_pct(bkt_metrics.get('cv_best_avg_win', 0.0))} / {format_return_as_pct(bkt_metrics.get('cv_best_avg_loss', 0.0))}\n"
                    )
                    f.write(
                        f"  - **Ulcer Index**: {bkt_metrics.get('cv_best_ulcer', 0.0):.4f}\n"
                    )
                    f.write(
                        f"  - **Time Under Water**: {format_return_as_pct(bkt_metrics.get('cv_best_tuw', 0.0))}\n"
                    )

                    # Position Sizing Statistics
                    pos_sizing = bkt_metrics.get("cv_best_pos_sizing", {})
                    if pos_sizing:
                        f.write("- **Position Sizing (CV Best)**:\n")
                        f.write(
                            f"  - **Formula**: {pos_sizing.get('sizing_formula', 'N/A')}, Squash: {pos_sizing.get('squash_fn', 'N/A')}, Squash k: {pos_sizing.get('squash_k', 0):.2f}\n"
                        )
                        f.write(
                            f"  - **Base Size**: {format_return_as_pct(pos_sizing.get('base_size', 0))}\n"
                        )
                        f.write(
                            f"  - **Rank Multiplier**: {format_return_as_pct(pos_sizing.get('rank_multiplier', 0))}\n"
                        )
                        f.write(
                            f"  - **Max Position**: {format_return_as_pct(pos_sizing.get('max_position', 0))} (capped at {format_return_as_pct(pos_sizing.get('position_hard_cap', 0))})\n"
                        )
                        f.write(
                            f"  - **Average Size**: {format_return_as_pct(pos_sizing.get('avg', 0))}\n"
                        )
                        f.write(
                            f"  - **Median Size**: {format_return_as_pct(pos_sizing.get('median', 0))}\n"
                        )
                        f.write(
                            f"  - **Size Range**: [{format_return_as_pct(pos_sizing.get('min', 0))}, {format_return_as_pct(pos_sizing.get('max', 0))}]\n"
                        )
                        f.write(
                            f"  - **Std Dev**: {format_return_as_pct(pos_sizing.get('std', 0))}\n"
                        )
                        f.write(
                            f"  - **Zero Positions**: {pos_sizing.get('n_zero', 0)}\n"
                        )
                        f.write(
                            f"  - **Max Positions**: {pos_sizing.get('n_max', 0)}\n"
                        )

                    # OOF Rank Diagnostics (Fixed Span)
                    f.write("- **OOF Calibration (Top Rank Diagnostics)**:\n")
                    for top_q in [30, 20, 10]:
                        prefix = f"oof_top{top_q}"
                        if f"{prefix}_n_trades" in bkt_metrics:
                            f.write(
                                f"  - **Top {top_q}%**: Total PnL: {format_return_as_pct(bkt_metrics.get(f'{prefix}_pnl_total', 0.0))}, "
                                f"PnL/Day: {format_return_as_pct(bkt_metrics.get(f'{prefix}_pnl_per_day', 0.0))}, "
                                f"Trades/Day: {bkt_metrics.get(f'{prefix}_trades_per_day', 0.0):.2f}, "
                                f"Sortino: {bkt_metrics.get(f'{prefix}_sortino', 0.0):.4f}, "
                                f"MaxDD: {format_return_as_pct(bkt_metrics.get(f'{prefix}_maxdd', 0.0))}, "
                                f"WinRate: {bkt_metrics.get(f'{prefix}_win_rate', 0.0):.1%}, "
                                f"PF: {bkt_metrics.get(f'{prefix}_profit_factor', 0.0):.2f}, "
                                f"Avg Win/Loss: {format_return_as_pct(bkt_metrics.get(f'{prefix}_avg_win', 0.0))} / {format_return_as_pct(bkt_metrics.get(f'{prefix}_avg_loss', 0.0))}, "
                                f"Ulcer: {bkt_metrics.get(f'{prefix}_ulcer', 0.0):.4f}, "
                                f"TUW: {format_return_as_pct(bkt_metrics.get(f'{prefix}_time_under_water', 0.0))}, "
                                f"N: {bkt_metrics.get(f'{prefix}_n_trades', 0)}\n"
                            )

                    waterfall = bkt_metrics.get("alpha_retention_waterfall", {})
                    if waterfall:
                        f.write("- **Alpha Retention Waterfall**:\n")
                        f.write(
                            f"  - **Best Raw Feature**: `{waterfall.get('best_raw_feature', 'N/A')}`\n"
                        )
                        f.write(
                            f"  - **Best Raw Feature IC**: {waterfall.get('best_raw_feature_ic', 0.0):.4f}\n"
                        )
                        f.write(
                            f"  - **Combined Score IC**: {waterfall.get('combined_score_ic', 0.0):.4f}\n"
                        )
                        f.write(
                            f"  - **OOF PnL No Offset**: {format_return_as_pct(waterfall.get('oof_pnl_total_no_offset', 0.0))} total, {format_return_as_pct(waterfall.get('oof_pnl_per_day_no_offset', 0.0))}/day\n"
                        )
                        f.write(
                            f"  - **OOF PnL With Offset**: {format_return_as_pct(waterfall.get('oof_pnl_total_with_offset', 0.0))} total, {format_return_as_pct(waterfall.get('oof_pnl_per_day_with_offset', 0.0))}/day\n"
                        )

                    # Walk-Forward Validation Results (True OOS)
                    full_oos = bkt_metrics.get("full_oos_metrics", {})
                    oos = bkt_metrics.get("best_oos_metrics", {})
                    rep_oos_rows = bkt_metrics.get("repeated_oos_results", []) or []
                    if full_oos or oos:
                        f.write("- **Walk-Forward Validation (Out-of-Sample)**:\n")
                        if full_oos:
                            f.write("  - **Full OOS Holdout**:\n")
                            f.write(
                                f"    - Limit Offset: `{full_oos.get('limit_offset_mode', 'disabled')}`\n"
                            )
                            f.write(
                                f"    - Total PnL: {format_return_as_pct(full_oos.get('PnL_total', 0.0))}\n"
                            )
                            f.write(
                                f"    - PnL/Day: {format_return_as_pct(full_oos.get('PnL_per_day', 0.0))}\n"
                            )
                            f.write(
                                f"    - Trades/Day: {full_oos.get('Trades_per_day', 0.0):.4f}\n"
                            )
                            f.write(
                                f"    - N_selected: {full_oos.get('N_selected', 0)} across {full_oos.get('N_days', 0.0):.1f} days\n"
                            )
                            f.write("    - Trade Count Waterfall:\n")
                            f.write(
                                "      "
                                f"raw={int(full_oos.get('N_raw_candidates', 0))} "
                                f"({full_oos.get('Raw_candidates_per_day', 0.0):.2f}/day), "
                                f"finite={int(full_oos.get('N_finite_scores', 0))} "
                                f"({full_oos.get('Finite_scores_per_day', 0.0):.2f}/day), "
                                f"topk={int(full_oos.get('N_after_topk', 0))} "
                                f"({full_oos.get('Topk_candidates_per_day', 0.0):.2f}/day), "
                                f"sized={int(full_oos.get('N_after_size', 0))} "
                                f"({full_oos.get('Sized_candidates_per_day', 0.0):.2f}/day), "
                                f"overlap_kept={int(full_oos.get('N_after_overlap', 0))} "
                                f"({full_oos.get('Overlap_kept_per_day', 0.0):.2f}/day)\n"
                            )
                            f.write(
                                f"    - Sortino: {full_oos.get('Sortino', 0.0):.4f}\n"
                            )
                            f.write(
                                f"    - Profit Factor: {full_oos.get('ProfitFactor', 0.0):.2f}\n"
                            )
                        if "holdout_selector" in oos:
                            f.write("  - **Repeated OOS Summary**:\n")
                            f.write(
                                f"    - OOS Basis: `{oos.get('holdout_selector')}`\n"
                            )
                            f.write(
                                f"    - Limit Offset: `{oos.get('limit_offset_mode', 'disabled')}`\n"
                            )
                            f.write(
                                f"    - Total PnL: {format_return_as_pct(oos.get('PnL_total', 0.0))}\n"
                            )
                            f.write(
                                f"    - PnL/Day: {format_return_as_pct(oos.get('PnL_per_day', 0.0))}\n"
                            )
                            f.write(
                                f"    - Trades/Day: {oos.get('Trades_per_day', 0.0):.4f}\n"
                            )
                            f.write(f"    - N_selected: {oos.get('N_selected', 0)}\n")
                            f.write(
                                "    - Trade Count Waterfall: "
                                f"raw={int(oos.get('N_raw_candidates', 0))}, "
                                f"finite={int(oos.get('N_finite_scores', 0))}, "
                                f"topk={int(oos.get('N_after_topk', 0))}, "
                                f"sized={int(oos.get('N_after_size', 0))}, "
                                f"overlap_kept={int(oos.get('N_after_overlap', 0))}\n"
                            )
                            f.write(f"    - Sortino: {oos.get('Sortino', 0.0):.4f}\n")
                            f.write(
                                f"    - MaxDD: {format_return_as_pct(oos.get('MaxDD', 0.0))}\n"
                            )
                            f.write(f"    - WinRate: {oos.get('WinRate', 0.0):.2%}\n")
                            f.write(
                                f"    - Profit Factor: {oos.get('ProfitFactor', 0.0):.2f}\n"
                            )
                            f.write(
                                f"    - Avg Win/Loss: {format_return_as_pct(oos.get('AvgWin', 0.0))} / {format_return_as_pct(oos.get('AvgLoss', 0.0))}\n"
                            )
                            f.write(f"    - Ulcer Index: {oos.get('Ulcer', 0.0):.4f}\n")
                            f.write(
                                f"    - Time Under Water: {format_return_as_pct(oos.get('TUW', 0.0))}\n"
                            )
                            f.write(
                                f"    - No Offset PnL: {format_return_as_pct(oos.get('PnL_total_no_offset', 0.0))} total, {format_return_as_pct(oos.get('PnL_per_day_no_offset', 0.0))}/day\n"
                            )
                            f.write(
                                f"    - No Offset Objective: {oos.get('ObjectiveScore_no_offset', 0.0):.4f}\n"
                            )
                            if "repeated_min_selected_threshold" in oos:
                                f.write(
                                    f"    - Repeated Holdout Min-N Gate: {oos.get('repeated_min_selected_threshold', 0)} (pass={bool(oos.get('repeated_median_selected_ok', True))})\n"
                                )
                        if rep_oos_rows:
                            med_n = float(
                                np.median(
                                    [
                                        float(r.get("N_selected", 0.0))
                                        for r in rep_oos_rows
                                    ]
                                )
                            )
                            med_days = float(
                                np.median(
                                    [
                                        float(r.get("N_days", 0.0))
                                        for r in rep_oos_rows
                                        if "N_days" in r
                                    ]
                                )
                            )
                            f.write(
                                f"  - **Repeated Holdout Count**: {len(rep_oos_rows)}\n"
                            )
                            f.write(
                                f"  - **Repeated Median N_selected**: {med_n:.1f} across {med_days:.1f} days\n"
                            )

                        # OOS Per-Decile Diagnostics
                        f.write("- **OOS Per-Decile Diagnostics**:\n")
                        for top_q in [30, 20, 10]:
                            prefix = f"oos_top{top_q}"
                            if f"{prefix}_n_trades" in oos:
                                f.write(
                                    f"  - **Top {top_q}%**: Total PnL: {format_return_as_pct(oos.get(f'{prefix}_pnl_total', 0.0))}, "
                                    f"PnL/Day: {format_return_as_pct(oos.get(f'{prefix}_pnl_per_day', 0.0))}, "
                                    f"Trades/Day: {oos.get(f'{prefix}_trades_per_day', 0.0):.2f}, "
                                    f"Sortino: {oos.get(f'{prefix}_sortino', 0.0):.4f}, "
                                    f"MaxDD: {format_return_as_pct(oos.get(f'{prefix}_maxdd', 0.0))}, "
                                    f"WinRate: {oos.get(f'{prefix}_win_rate', 0.0):.1%}, "
                                    f"PF: {oos.get(f'{prefix}_profit_factor', 0.0):.2f}, "
                                    f"Avg Win/Loss: {format_return_as_pct(oos.get(f'{prefix}_avg_win', 0.0))} / {format_return_as_pct(oos.get(f'{prefix}_avg_loss', 0.0))}, "
                                    f"Ulcer: {oos.get(f'{prefix}_ulcer', 0.0):.4f}, "
                                    f"TUW: {format_return_as_pct(oos.get(f'{prefix}_time_under_water', 0.0))}, "
                                    f"N: {oos.get(f'{prefix}_n_trades', 0)}\n"
                                )

                    # Top Features
                    top_f = bkt_metrics.get("top_features", {})
                    if top_f:
                        f.write("- **Top 10 Sizer Features**:\n")
                        for fname, fval in top_f.items():
                            f.write(f"  - `{fname}`: {fval:.4f}\n")

                    # Feature Selection Diagnostics
                    fs_ridge = bkt_metrics.get("feature_selection_diag_ridge", {})
                    fs_tree = bkt_metrics.get("feature_selection_diag_tree", {})
                    if fs_ridge:
                        sel = fs_ridge.get("selected_features", [])
                        total = fs_ridge.get("total_features_input", len(sel))
                        f.write(
                            f"- **Feature Selection (Ridge)**: Kept {len(sel)}/{total} features.\n"
                        )
                        if len(sel) < total:
                            all_f = fs_ridge.get("all_features", [])
                            pruned = [f for f in all_f if f not in sel]
                            f.write(
                                f"  - *Pruned*: {', '.join(pruned[:10])}{'...' if len(pruned) > 10 else ''}\n"
                            )

                    if fs_tree and hasattr(fs_tree, "selected_features"):
                        sel = fs_tree.selected_features
                        f.write(
                            f"- **Feature Selection (Tree)**: Kept {len(sel)} features.\n"
                        )

                    # Label Stability (from Policy Optimizer)
                    low_opt = bkt_metrics.get("label_policy_optimizer", {})
                    if low_opt:
                        f.write("- **Label Stability (Sensitivity Analysis)**:\n")
                        f.write(
                            f"  - **Selected Policy**: {low_opt.get('best_policy_params', 'N/A')}\n"
                        )
                        f.write(
                            f"  - **J_stable**: {low_opt.get('best_j_stable', 0.0):.4f}\n"
                        )
                        f.write(
                            f"  - **TP Sweep Result**: {low_opt.get('tp_sensitivity', 'N/A')}\n"
                        )
                        ab = low_opt.get("ab_test", {})
                        if ab:
                            opt_ab = ab.get("optimized_policy_target", {})
                            best_tbm = ab.get("best_tbm_ridge_only", {})
                            tbm_rows = ab.get("tbm_ridge_only_candidates", [])
                            f.write("- **Policy Learnability A/B (Ridge Only)**:\n")
                            f.write(
                                f"  - **Winner**: `{ab.get('winner', 'N/A')}` (delta J_stable={ab.get('delta_j_stable', 0.0):.4f})\n"
                            )
                            opt_fin = opt_ab.get("financials_full", {})
                            f.write(
                                f"  - **Optimized Policy Target**: J_stable={opt_ab.get('j_stable', 0.0):.4f}, J_mean={opt_ab.get('j_mean', 0.0):.4f}, J_std={opt_ab.get('j_std', 0.0):.4f}, Expectancy={format_return_as_pct(opt_fin.get('expectancy', 0.0))}, WinRate={opt_fin.get('win_rate', 0.0):.2%}, PF={opt_fin.get('profit_factor', 0.0):.2f}, Avg Win/Loss={format_return_as_pct(opt_fin.get('avg_win', 0.0))} / {format_return_as_pct(opt_fin.get('avg_loss', 0.0))}\n"
                            )
                            if best_tbm:
                                best_fin = best_tbm.get("financials_full", {})
                                f.write(
                                    f"  - **Best TBM Ridge Baseline**: `{best_tbm.get('name', 'N/A')}` tp={best_tbm.get('tp_pct', 0.0):.2%}, sl={best_tbm.get('sl_pct', 0.0):.2%}, hold={best_tbm.get('max_hold_bars', 0)} bars, J_stable={best_tbm.get('j_stable', 0.0):.4f}, J_mean={best_tbm.get('j_mean', 0.0):.4f}, J_std={best_tbm.get('j_std', 0.0):.4f}, Expectancy={format_return_as_pct(best_fin.get('expectancy', 0.0))}, WinRate={best_fin.get('win_rate', 0.0):.2%}, PF={best_fin.get('profit_factor', 0.0):.2f}, Avg Win/Loss={format_return_as_pct(best_fin.get('avg_win', 0.0))} / {format_return_as_pct(best_fin.get('avg_loss', 0.0))}\n"
                                )
                            if tbm_rows:
                                f.write("  - **TBM Grid Leaderboard**:\n")
                                for row in sorted(
                                    tbm_rows,
                                    key=lambda r: float(r.get("j_stable", -1e9)),
                                    reverse=True,
                                )[:4]:
                                    fin = row.get("financials_full", {})
                                    f.write(
                                        f"    - `{row.get('name', 'tbm')}`: J_stable={row.get('j_stable', 0.0):.4f}, tp={row.get('tp_pct', 0.0):.2%}, sl={row.get('sl_pct', 0.0):.2%}, hold={row.get('max_hold_bars', 0)}, Expectancy={format_return_as_pct(fin.get('expectancy', 0.0))}, WinRate={fin.get('win_rate', 0.0):.2%}, PF={fin.get('profit_factor', 0.0):.2f}\n"
                                    )

                    tf_ab = bkt_metrics.get("target_family_ab", {})
                    if tf_ab:
                        f.write("- **Target Family A/B (Sizer Level)**:\n")
                        if tf_ab.get("status") == "ok":
                            winner = tf_ab.get("winner", {}) or {}
                            best_simpler = tf_ab.get("best_simpler", {}) or {}
                            f.write(
                                f"  - **Winner**: `{winner.get('target_name', 'N/A')}` family={winner.get('target_family', 'N/A')} score={winner.get('learnability_score', 0.0):.6f}, IC={winner.get('ridge_ic', 0.0):.4f}, TopQ Policy U={format_metric_float(winner.get('topq_policy_u_mean', 0.0), 6)}\n"
                            )
                            if best_simpler:
                                f.write(
                                    f"  - **Best Simpler Target**: `{best_simpler.get('target_name', 'N/A')}` family={best_simpler.get('target_family', 'N/A')} score={best_simpler.get('learnability_score', 0.0):.6f}, IC={best_simpler.get('ridge_ic', 0.0):.4f}, TopQ Policy U={format_metric_float(best_simpler.get('topq_policy_u_mean', 0.0), 6)}\n"
                                )
                            f.write("  - **Leaderboard**:\n")
                            for row in tf_ab.get("rows", [])[:6]:
                                f.write(
                                    f"    - `{row.get('target_name', 'N/A')}` ({row.get('target_family', 'N/A')}): score={row.get('learnability_score', 0.0):.6f}, IC={row.get('ridge_ic', 0.0):.4f}, TopQ Policy U={format_metric_float(row.get('topq_policy_u_mean', 0.0), 6)}, std={format_metric_float(row.get('topq_policy_u_std', 0.0), 6)}\n"
                                )
                        else:
                            f.write(
                                f"  - **Status**: `{tf_ab.get('status', 'unknown')}` ({tf_ab.get('reason', 'n/a')})\n"
                            )

            f.write(
                "\n\n---\n*Report generated with Bias Mitigation (2-Step CV Gating & 48h Purging + Walk-Forward OOS)*\n"
            )
        tprint(f"Metrics report manifested to {report_path}")
    except Exception as e:
        tprint(f"WARNING: Failed to generate markdown manifest: {e}")

    return 0


if __name__ == "__main__":
    exit(main())
