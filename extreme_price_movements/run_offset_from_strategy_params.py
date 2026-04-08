#!/usr/bin/env python3
"""Run offset generator using strategy_params.json from Ridge sizer.

This script loads the per-strategy parameters from ridge_sizer/strategy_params.json
and runs the offset generator on the best strategy's trades.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from extreme_price_movements.simple_position_sizer import run_simple_position_sizer
from extreme_price_movements.simple_offset_generator import run_simple_offset_generator_from_sizer


def load_strategy_params(run_id: str, data_root: str = "data") -> dict:
    """Load strategy_params.json from ridge_sizer directory."""
    path = Path(data_root) / "artifacts" / run_id / "ridge_sizer" / "strategy_params.json"
    with open(path) as f:
        return json.load(f)


def load_meta_oof_for_strategy(strategy_id: str, run_id: str, data_root: str = "data") -> pd.DataFrame:
    """Load meta_oof parquet file for a specific strategy."""
    meta_oof_dir = Path(data_root) / "artifacts" / run_id / "meta_oof"
    
    # Find matching files - strategy_id may be prefixed with long_/short_
    # Use contains matching instead of startswith
    all_files = list(meta_oof_dir.glob("meta_oof_*.parquet"))
    matching_files = [f for f in all_files if strategy_id in f.name]
    
    if not matching_files:
        raise FileNotFoundError(f"No meta_oof files found for strategy {strategy_id}")
    
    # Prefer clf (classification) files, then reg (regression)
    clf_files = [f for f in matching_files if "_clf.parquet" in f.name]
    if clf_files:
        files = clf_files
    else:
        files = matching_files
    
    # Load the first matching file
    df = pd.read_parquet(files[0])
    return df


def run_offset_from_strategy_params(
    run_id: str,
    data_root: str = "data",
    use_best_only: bool = True,
    base_offset_ret: float = 0.0001,
    max_offset_ret: float = 0.0003,
    invert_offset: bool = True,
) -> dict:
    """Run offset generator using strategy_params.json.
    
    Args:
        run_id: The training run ID (e.g., "20260321_140000")
        data_root: Root data directory
        use_best_only: If True, only test the best strategy; if False, test all
        base_offset_ret: Minimum offset in return terms
        max_offset_ret: Maximum offset in return terms
        invert_offset: If True, higher confidence = lower offset
    
    Returns:
        Dictionary with results for each strategy tested
    """
    # Load strategy parameters
    params = load_strategy_params(run_id, data_root)
    
    if use_best_only:
        strategies_to_test = [params["best_strategy_id"]]
    else:
        strategies_to_test = list(params["buckets"].keys())
    
    results = {}
    
    for strategy_id in strategies_to_test:
        bucket_params = params["buckets"][strategy_id]
        
        print(f"\n{'='*70}")
        print(f"Testing strategy: {strategy_id}")
        print(f"Baseline PF: {bucket_params['profit_factor']:.2f}")
        print(f"Baseline Hit Rate: {bucket_params['hit_rate']*100:.1f}%")
        print(f"Threshold: {bucket_params['threshold_pct']:.1f}%")
        print(f"{'='*70}")
        
        # Load meta_oof data for this strategy
        try:
            df = load_meta_oof_for_strategy(strategy_id, run_id, data_root)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue
        
        # Prepare data for sizer
        # Use available columns - oof_pred is the main prediction
        feature_dict = {'base_h_0': df['oof_pred'].values}
        if 'oof_ev' in df.columns:
            feature_dict['base_h_1'] = df['oof_ev'].values
        else:
            # Use prediction as second feature if EV not available
            feature_dict['base_h_1'] = df['oof_pred'].values
        
        trade_outcomes = pd.DataFrame({
            'timestamp': df['timestamp'],
            'entry_price': np.ones(len(df)),
            'is_long': df['is_long'],
            'net_return': df['return'],
        })
        
        # Run sizer to get thresholds and confidence scores
        sizer_results = run_simple_position_sizer(
            feature_dict=feature_dict,
            trade_outcomes=trade_outcomes,
            y_raw_net_return=df['return'].values,
            y_downside=np.abs(np.minimum(df['return'].values, 0)),
            timestamps=df['timestamp'].values,
        )
        
        # Run offset generator
        offset_results = run_simple_offset_generator_from_sizer(
            sizer_results=sizer_results,
            trade_outcomes=trade_outcomes,
            use_raw_return_offset=True,
            base_offset_ret=base_offset_ret,
            max_offset_ret=max_offset_ret,
            invert_offset=invert_offset,
            offset_scaling='linear',
        )
        
        # Extract comparison
        baseline = offset_results['baseline_metrics']
        offset = offset_results['offset_metrics']
        
        print(f"\nResults:")
        print(f"  Hit Rate: {baseline['hit_rate']*100:.1f}% -> {offset['hit_rate']*100:.1f}% (Δ{(offset['hit_rate']-baseline['hit_rate'])*100:+.1f}%)")
        print(f"  PF: {baseline['profit_factor']:.2f} -> {offset['profit_factor']:.2f} (Δ{offset['profit_factor']-baseline['profit_factor']:+.2f})")
        print(f"  Sortino: {baseline['sortino']:.3f} -> {offset['sortino']:.3f}")
        print(f"  Fill Rate: {baseline['fill_rate']*100:.0f}% -> {offset['fill_rate']*100:.0f}%")
        print(f"  Trades: {baseline['n_executed']} -> {offset['n_executed']} (Δ{offset['n_executed']-baseline['n_executed']:+d})")
        
        results[strategy_id] = {
            'baseline': baseline,
            'offset': offset,
            'params': bucket_params,
        }
    
    return results


if __name__ == "__main__":
    # Run on the 20260321_140000 run
    results = run_offset_from_strategy_params(
        run_id="20260321_140000",
        use_best_only=True,  # Test only the best strategy
        base_offset_ret=0.0001,  # 10 bps
        max_offset_ret=0.0003,     # 30 bps
        invert_offset=True,       # High conf = lower offset
    )
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for strategy_id, res in results.items():
        baseline_pf = res['baseline']['profit_factor']
        offset_pf = res['offset']['profit_factor']
        print(f"{strategy_id[:50]:50} PF: {baseline_pf:.2f} -> {offset_pf:.2f}")
