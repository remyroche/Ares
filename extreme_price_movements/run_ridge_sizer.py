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

from extreme_price_movements.ridge_position_sizer import (
    RidgePositionSizer,
    run_ridge_position_sizer_step,
    run_policy_aware_labeling_step,
    prepare_policy_params_from_tpsl_optimiser,
    prepare_trade_outcomes_from_labels,
)
from extreme_price_movements.utils import tprint


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
    _ts_pat = re.compile(r"^\d{8}_\d{6}$")
    run_dirs = sorted(
        [d for d in artifacts_dir.iterdir() if d.is_dir() and _ts_pat.match(d.name)],
        key=lambda x: x.name,
        reverse=True
    )
    if not run_dirs:
        raise FileNotFoundError("No run directories found")
    
    return run_dirs[0].name


def load_meta_oof_predictions(data_root: str, run_id: str) -> Dict[str, pd.DataFrame]:
    """Load meta model OOF predictions from a training run.
    
    Handles per-horizon regressors (e.g. long_mr_H2, long_mr_H4, long_mr_H8)
    and classifiers (e.g. long_mr_clf). Groups by base bucket and returns a
    DataFrame per bucket with columns: reg_H2, reg_H4, reg_H8, clf, plus
    agreement/disagreement features.
    
    Returns a dict keyed by bucket (e.g. 'long_mr') where each value is a
    DataFrame with prediction columns plus metadata.
    """
    import re
    meta_oof_dir = Path(data_root) / "artifacts" / run_id / "meta_oof"
    
    if not meta_oof_dir.exists():
        raise FileNotFoundError(f"No meta OOF directory at {meta_oof_dir}")
    
    # Load all parquet files
    raw_dfs = {}
    for parquet_file in meta_oof_dir.glob("meta_oof_*.parquet"):
        model_name = parquet_file.stem.replace("meta_oof_", "")
        df = pd.read_parquet(parquet_file)
        raw_dfs[model_name] = df
    
    if not raw_dfs:
        raise FileNotFoundError(f"No meta OOF parquet files found in {meta_oof_dir}")
    
    # Parse model names into (base_bucket, col_name)
    # Patterns: long_mr_H2 -> (long_mr, reg_H2), long_mr_clf -> (long_mr, clf),
    #           long_mr -> (long_mr, reg)  [legacy single-regressor format]
    _h_pat = re.compile(r'^(.+)_H(\d+)$')
    buckets = {}
    for name, df in raw_dfs.items():
        if name.endswith("_clf"):
            bucket = name[:-4]
            col_name = "clf"
        else:
            m = _h_pat.match(name)
            if m:
                bucket = m.group(1)
                col_name = f"reg_H{m.group(2)}"
            else:
                bucket = name
                col_name = "reg"
        if bucket not in buckets:
            buckets[bucket] = {}
        buckets[bucket][col_name] = df
    
    result = {}
    for bucket, model_dfs in buckets.items():
        # Use first available df as base (for length and metadata)
        base_df = next(iter(model_dfs.values()))
        n = len(base_df)
        combined = pd.DataFrame(index=range(n))
        
        # Add all prediction columns
        reg_cols = []
        for col_name, mdf in sorted(model_dfs.items()):
            if len(mdf) == n:
                combined[col_name] = mdf["oof_pred"].values
                if col_name.startswith("reg"):
                    reg_cols.append(col_name)
        
        # Agreement/disagreement features across horizon regressors
        if len(reg_cols) >= 2:
            reg_vals = combined[reg_cols].values
            # Mean regressor prediction
            combined["reg_mean"] = np.nanmean(reg_vals, axis=1)
            # Std across regressors (disagreement)
            combined["reg_std"] = np.nanstd(reg_vals, axis=1)
            # Range (max - min)
            combined["reg_range"] = np.nanmax(reg_vals, axis=1) - np.nanmin(reg_vals, axis=1)
            # Sign agreement: fraction of regressors above median
            _med = np.nanmedian(reg_vals, axis=1, keepdims=True)
            combined["reg_sign_agree"] = np.nanmean((reg_vals > _med).astype(float), axis=1)
            # Regressor-classifier agreement (if clf exists)
            if "clf" in combined.columns:
                _clf_high = (combined["clf"].values > 0.5).astype(float)
                _reg_high = (combined["reg_mean"].values > np.nanmedian(combined["reg_mean"].values)).astype(float)
                combined["reg_clf_agree"] = (_clf_high == _reg_high).astype(float)
        elif len(reg_cols) == 1:
            combined["reg_mean"] = combined[reg_cols[0]].values
            combined["reg_std"] = 0.0
        
        # Attach metadata from base
        for col in ["timestamp", "symbol", "return", "is_long", "index"]:
            if col in base_df.columns:
                combined[col] = base_df[col].values
        
        result[bucket] = combined
    
    tprint(f"Loaded OOF predictions for {len(result)} buckets: {list(result.keys())}")
    for bk, bdf in result.items():
        pred_cols = [c for c in bdf.columns if c not in ("timestamp", "symbol", "return", "is_long", "index")]
        tprint(f"  {bk}: {len(bdf)} samples, pred_cols={pred_cols}")
    return result


def load_trade_outcomes(data_root: str, run_id: str, oof_df: pd.DataFrame) -> pd.DataFrame:
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
    # Check if we have the return column directly in OOF data
    if "return" in oof_df.columns:
        outcomes = pd.DataFrame({
            "return": oof_df["return"].values,
            "is_long": oof_df["is_long"].values if "is_long" in oof_df.columns else 1,
        })
        if "timestamp" in oof_df.columns:
            outcomes["timestamp"] = oof_df["timestamp"].values
        if "symbol" in oof_df.columns:
            outcomes["symbol"] = oof_df["symbol"].values
        tprint(f"Constructed trade outcomes from OOF context: {len(outcomes)} trades")
        return outcomes
    
    # Fallback: try to load from separate file
    outcomes_path = Path(data_root) / "artifacts" / run_id / "trade_outcomes.parquet"
    
    if outcomes_path.exists():
        df = pd.read_parquet(outcomes_path)
        tprint(f"Loaded trade outcomes from {outcomes_path}: {len(df)} trades")
        return df
    
    raise FileNotFoundError(
        f"No trade outcomes found. The OOF predictions must include 'return' column, "
        f"or trade_outcomes.parquet must exist at {outcomes_path}"
    )


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
        with open(tpsl_path, 'r') as f:
            params = json.load(f)
        tprint(f"Loaded TP/SL params from {tpsl_path}")
        return params
    
    # Try alternative location
    tpsl_path = Path(data_root) / "artifacts" / run_id / "tpsl" / "best_params.json"
    if tpsl_path.exists():
        with open(tpsl_path, 'r') as f:
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
                if 'timestamp' in panel_df.columns and 'symbol' in panel_df.columns:
                    price_panel = {}
                    for col in ['open', 'high', 'low', 'close']:
                        if col in panel_df.columns:
                            price_panel[col] = panel_df.pivot(
                                index='timestamp', columns='symbol', values=col
                            )
                    if len(price_panel) == 4:
                        tprint(f"Loaded price panel: {len(price_panel['open'])} timestamps, "
                               f"{len(price_panel['open'].columns)} symbols")
                        return price_panel
            else:
                # Wide format or column-multiindex
                # Check for column structure
                if isinstance(panel_df.columns, pd.MultiIndex):
                    # Columns like (symbol, ohlc)
                    price_panel = {}
                    for ohlc in ['open', 'high', 'low', 'close']:
                        try:
                            price_panel[ohlc] = panel_df.xs(ohlc, level=1, axis=1)
                        except KeyError:
                            # Try level=0
                            price_panel[ohlc] = panel_df.xs(ohlc, level=0, axis=1)
                    if len(price_panel) == 4:
                        tprint(f"Loaded price panel: {len(price_panel['open'])} timestamps")
                        return price_panel
    
    tprint("Warning: No price panel found. Policy-aware labeling will not be available.")
    return None


def main():
    """Main entry point for the ridge position sizer runner."""
    parser = argparse.ArgumentParser(
        description="Run Ridge Position Sizer on training artifacts"
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Data root directory (default: data)"
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Training run ID (default: latest)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for sizer weights (default: artifacts/{run_id}/ridge_sizer)"
    )
    parser.add_argument(
        "--use-policy-labels",
        action="store_true",
        help="Use policy-aware labeling with TP/SL simulation"
    )
    parser.add_argument(
        "--max-hold-hours",
        type=int,
        default=24,
        help="Maximum holding period in hours (default: 24)"
    )
    parser.add_argument(
        "--cost-pct",
        type=float,
        default=0.0005,
        help="Transaction cost as decimal (default: 0.0005 = 0.05%%)"
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
        bucket_oofs = load_meta_oof_predictions(args.data_root, run_id)
    except FileNotFoundError as e:
        tprint(f"Error: {e}")
        tprint("Meta model OOF predictions not found.")
        tprint("Ensure training.py has been run with meta model training enabled.")
        return 1
    
    # Set up output directory
    output_dir = args.output_dir or os.path.join(
        args.data_root, "artifacts", run_id, "ridge_sizer"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    all_weights = {}
    all_params = {}
    
    for bucket_name, oof_preds in bucket_oofs.items():
        tprint("-" * 80)
        tprint(f"Running Ridge Position Sizer for bucket: {bucket_name}")
        tprint(f"  OOF shape: {oof_preds.shape}")
        
        # Build trade outcomes from OOF context
        try:
            trade_outcomes = load_trade_outcomes(args.data_root, run_id, oof_preds)
            tprint(f"  Trade outcomes shape: {trade_outcomes.shape}")
        except FileNotFoundError as e:
            tprint(f"  Skipping {bucket_name}: {e}")
            continue
        
        if "return" not in trade_outcomes.columns:
            tprint(f"  Skipping {bucket_name}: missing 'return' column")
            continue
        
        # Extract prediction columns (reg_H*, clf, agreement features)
        _meta_cols = {"timestamp", "symbol", "return", "is_long", "index"}
        pred_cols = [c for c in oof_preds.columns if c not in _meta_cols]
        if not pred_cols:
            tprint(f"  Skipping {bucket_name}: no prediction columns")
            continue
        oof_pred_df = oof_preds[pred_cols].copy()
        tprint(f"  Prediction features: {pred_cols}")
        
        timestamps = None
        if 'timestamp' in trade_outcomes.columns:
            timestamps = trade_outcomes['timestamp'].values
        
        try:
            sizer, metrics = run_ridge_position_sizer_step(
                oof_preds=oof_pred_df,
                trade_outcomes=trade_outcomes,
                timestamps=timestamps,
                cfg={'cost_pct': args.cost_pct},
                save_model=False,
                run_id=run_id,
            )
            weights = sizer.get_weights()
            # Prefix weights with bucket name
            for wname, wval in weights.items():
                all_weights[f"{bucket_name}_{wname}"] = wval
            all_params[bucket_name] = sizer.best_params_
            tprint(f"  {bucket_name} weights: {weights}")
        except Exception as e:
            tprint(f"  {bucket_name} failed: {e}")
            continue
    
    # Save combined weights
    weights_path = os.path.join(output_dir, "sizer_weights.json")
    with open(weights_path, 'w') as f:
        json.dump({
            'weights': all_weights,
            'params_per_bucket': all_params,
            'run_id': run_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)
    tprint(f"Saved weights to {weights_path}")
    
    # Print summary
    tprint("=" * 80)
    tprint("RIDGE POSITION SIZER COMPLETE")
    tprint("=" * 80)
    tprint(f"Buckets processed: {len(all_params)}")
    for name, w in all_weights.items():
        tprint(f"  {name}: {w:.4f}")
    tprint(f"Output directory: {output_dir}")
    tprint("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())
