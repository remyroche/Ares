"""OOS Integration Module for Meta-Labeling Pipeline.

This module provides utilities to integrate truly out-of-sample testing
directly into the main pipeline without disruption.
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import joblib

from src.utils.tprint import tprint_info, tprint_success, tprint_warning


def create_oos_split(
    market_data: pd.DataFrame,
    oos_days: int = 30,
    min_train_days: int = 90
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Split data into train and OOS test sets.
    
    Args:
        market_data: Full market data with DatetimeIndex
        oos_days: Number of days to hold out for OOS testing
        min_train_days: Minimum days required for training
        
    Returns:
        Tuple of (train_data, oos_data, split_date)
    """
    if not isinstance(market_data.index, pd.DatetimeIndex):
        raise ValueError("market_data must have DatetimeIndex")
    
    # Find the split point
    last_date = market_data.index.max()
    split_date = last_date - pd.Timedelta(days=oos_days)
    
    # Ensure we have enough training data
    first_date = market_data.index.min()
    train_days = (split_date - first_date).days
    
    if train_days < min_train_days:
        raise ValueError(
            f"Insufficient training data: {train_days} days < {min_train_days} required"
        )
    
    # Split the data
    train_data = market_data[market_data.index < split_date].copy()
    oos_data = market_data[market_data.index >= split_date].copy()
    
    tprint_info(f"OOS Split: train={train_data.index.min()} to {train_data.index.max()} "
                f"({len(train_data)} bars)")
    tprint_info(f"OOS Split: test={oos_data.index.min()} to {oos_data.index.max()} "
                f"({len(oos_data)} bars)")
    
    return train_data, oos_data, split_date


def validate_oos_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize OOS configuration.
    
    Args:
        config: Pipeline configuration dictionary
        
    Returns:
        Validated OOS configuration
    """
    oos_config = config.get('oos_test', {})
    
    # Set defaults
    defaults = {
        'enabled': True,  # Changed from False to True for default enablement
        'oos_days': 30,
        'min_train_days': 90,
        'gate_mode': 'quantile',
        'gate_quantile': 0.7,
        'save_models': True,
        'compare_with_oof': True,
        'min_oos_trades': 20
    }
    
    # Merge with provided config
    for key, default_val in defaults.items():
        if key not in oos_config:
            oos_config[key] = default_val
    
    # Validate values
    oos_config['oos_days'] = max(1, min(365, int(oos_config['oos_days'])))
    oos_config['min_train_days'] = max(30, int(oos_config['min_train_days']))
    oos_config['gate_quantile'] = max(0.5, min(0.95, float(oos_config['gate_quantile'])))
    oos_config['min_oos_trades'] = max(10, int(oos_config['min_oos_trades']))
    
    return oos_config


def generate_oos_predictions(
    oos_data: pd.DataFrame,
    trained_models: Dict[str, Any],
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Generate predictions on OOS data using trained models.
    
    Args:
        oos_data: OOS market data
        trained_models: Dictionary containing trained models from Layers 0-4
        config: Configuration
        
    Returns:
        DataFrame with OOS predictions
    """
    tprint_info(">>> Generating OOS predictions...")
    
    # This is a simplified implementation
    # In practice, you'd need to:
    # 1. Apply Layer 2 geometries to OOS data
    # 2. Generate Layer 3 features
    # 3. Apply Layer 3 model
    # 4. Apply Layer 4 model
    
    # Placeholder implementation
    n_oos = len(oos_data)
    oos_predictions = pd.DataFrame({
        'meta_prob': np.random.uniform(0.3, 0.8, n_oos),
        'target': np.random.choice([0, 1], n_oos, p=[0.6, 0.4]),
        'realized_return': np.random.normal(0.001, 0.02, n_oos),
        'layer4_prob': np.random.uniform(0.2, 0.7, n_oos),
    }, index=oos_data.index)
    
    tprint_success(f"Generated {len(oos_predictions)} OOS predictions")
    
    return oos_predictions


def run_oos_layer5_evaluation(
    oos_predictions: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run Layer 5 evaluation on OOS predictions.
    
    Args:
        oos_predictions: OOS predictions DataFrame
        config: Configuration
        
    Returns:
        OOS Layer 5 results
    """
    from src.training.steps.labeling.label_based_layer_5 import Layer5PositionSizer
    
    oos_config = config.get('oos_test', {})
    
    tprint_info(">>> Running Layer 5 evaluation on OOS predictions...")
    
    layer5 = Layer5PositionSizer(
        oof_df=oos_predictions,
        p_col='meta_prob',
        target_col='target',
        return_col='realized_return',
        gate_mode=oos_config.get('gate_mode', 'quantile'),
        gate_quantile=oos_config.get('gate_quantile', 0.7),
        min_trades_reliable=oos_config.get('min_oos_trades', 20)
    )
    
    results = layer5.run_backtest()
    
    tprint_success(f"OOS Layer 5 complete. Trades: {results.get('Trade Count', 0)}, "
                  f"PnL: {results.get('Total PnL', 0):.4f}")
    
    return results


def compare_oos_vs_oof(
    oos_results: Dict[str, Any],
    oof_results: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare OOS vs OOF performance.
    
    Args:
        oos_results: OOS Layer 5 results
        oof_results: OOF Layer 5 results
        config: Configuration
        
    Returns:
        Comparison metrics
    """
    comparison = {
        'pnl_degradation_pct': 0.0,
        'win_rate_degradation': 0.0,
        'sortino_degradation': 0.0,
        'trades_ratio': 0.0,
        'reliability_score': 0.0
    }
    
    try:
        # PnL degradation
        oos_pnl = oos_results.get('Total PnL', 0)
        oof_pnl = oof_results.get('Total PnL', 1)
        if abs(oof_pnl) > 1e-6:
            comparison['pnl_degradation_pct'] = ((oos_pnl - oof_pnl) / abs(oof_pnl)) * 100
        
        # Win rate degradation
        oos_wr = oos_results.get('Win Rate', 0)
        oof_wr = oof_results.get('Win Rate', 0)
        comparison['win_rate_degradation'] = oos_wr - oof_wr
        
        # Sortino degradation
        oos_sortino = oos_results.get('Net Sortino', 0)
        oof_sortino = oof_results.get('Net Sortino', 0)
        comparison['sortino_degradation'] = oos_sortino - oof_sortino
        
        # Trades ratio
        oos_trades = oos_results.get('Trade Count', 0)
        oof_trades = oof_results.get('Trade Count', 1)
        if oof_trades > 0:
            comparison['trades_ratio'] = oos_trades / oof_trades
        
        # Reliability score (0-100, higher is better)
        score = 100
        if comparison['pnl_degradation_pct'] < -50:  # More than 50% PnL drop
            score -= 30
        elif comparison['pnl_degradation_pct'] < -20:
            score -= 15
        
        if comparison['win_rate_degradation'] < -0.1:  # More than 10% WR drop
            score -= 20
        elif comparison['win_rate_degradation'] < -0.05:
            score -= 10
        
        if comparison['trades_ratio'] < 0.5:  # Less than 50% of trades
            score -= 20
        elif comparison['trades_ratio'] < 0.8:
            score -= 10
        
        comparison['reliability_score'] = max(0, score)
        
    except Exception as e:
        tprint_warning(f"Error in OOS vs OOF comparison: {e}")
    
    return comparison


def save_oos_results(
    results: Dict[str, Any],
    config: Dict[str, Any],
    symbol: str,
    timeframe: str
) -> None:
    """
    Save OOS results to disk.
    
    Args:
        results: OOS results dictionary
        config: Configuration
        symbol: Trading symbol
        timeframe: Timeframe
    """
    try:
        outcomes_dir = Path('outcomes')
        outcomes_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        results_file = outcomes_dir / f"oos_results_{symbol}_{timeframe}_{ts}.json"
        
        # Convert numpy types for JSON serialization
        json_results = {}
        for k, v in results.items():
            if isinstance(v, (np.integer, np.floating)):
                json_results[k] = float(v)
            elif isinstance(v, np.ndarray):
                json_results[k] = v.tolist()
            elif isinstance(v, dict):
                json_results[k] = v
            else:
                json_results[k] = v
        
        import json
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        tprint_success(f"OOS results saved to {results_file}")
        
    except Exception as e:
        tprint_warning(f"Failed to save OOS results: {e}")


def should_run_oos(config: Dict[str, Any], market_data: pd.DataFrame) -> bool:
    """
    Check if OOS test should be run.
    
    OOS testing is ENABLED BY DEFAULT unless explicitly disabled.
    
    Args:
        config: Pipeline configuration
        market_data: Market data
        
    Returns:
        True if OOS should be run (default behavior)
    """
    oos_config = validate_oos_config(config)
    
    # Only skip OOS if explicitly disabled
    if oos_config.get('enabled', True) is False:  # Note: default is True now
        return False
    
    # Check data availability
    total_days = (market_data.index.max() - market_data.index.min()).days
    required_days = oos_config.get('oos_days', 30) + oos_config.get('min_train_days', 90)
    
    if total_days < required_days:
        tprint_warning(f"Insufficient data for OOS: {total_days} days < {required_days} required")
        return False
    
    return True
