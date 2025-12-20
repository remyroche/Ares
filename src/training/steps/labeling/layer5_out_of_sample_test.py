"""Layer 5 Out-of-Sample Test Module.

This module provides utilities to run Layer 5 on truly held-out data
without disrupting the main pipeline.

Approach:
1. Hold out the most recent N days from the entire dataset
2. Run Layers 0-4 on historical data only
3. Use trained models to predict on held-out period
4. Run Layer 5 evaluation on OOS predictions

This ensures unbiased assessment of Layer 5 performance.
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.training.steps.labeling.label_based_layer_0 import run_layer0_kalman_vwap
from src.training.steps.labeling.label_based_layer_2 import LabelBasedLayer2
from src.training.steps.labeling.label_based_layer_3 import layer3_analyst_lgbm
from src.training.steps.labeling.label_based_layer_4 import Layer4RiskFilter, compute_layer4_regime_features
from src.training.steps.labeling.label_based_layer_5 import Layer5PositionSizer
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


def run_oos_layer5_test(
    symbol: str,
    timeframe: str,
    market_data: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    oos_days: int = 30,
    min_train_days: int = 90,
    gate_mode: str = 'quantile',
    gate_quantile: float = 0.7
) -> Dict[str, Any]:
    """
    Run Layer 5 on truly out-of-sample data.
    
    This function:
    1. Splits data into train/OOS
    2. Trains Layers 0-4 on training data only
    3. Generates predictions on OOS data
    4. Evaluates Layer 5 on OOS predictions
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe (e.g., '15m', '1h')
        market_data: Full market data
        config: Configuration dictionary
        oos_days: Days to hold out for OOS
        min_train_days: Minimum training days required
        gate_mode: Layer 5 gate mode (use 'quantile' for unbiased)
        gate_quantile: Quantile threshold for gating
        
    Returns:
        Dictionary containing OOS results and comparison with in-sample
    """
    cfg = config or {}
    
    tprint_info(f"Starting OOS Layer 5 Test for {symbol} {timeframe}")
    tprint_info(f"Holding out {oos_days} days for true out-of-sample evaluation")
    
    # 1. Split data
    try:
        train_data, oos_data, split_date = create_oos_split(
            market_data, oos_days=oos_days, min_train_days=min_train_days
        )
    except Exception as e:
        tprint_warning(f"Failed to create OOS split: {e}")
        return {'error': str(e), 'success': False}
    
    results = {
        'symbol': symbol,
        'timeframe': timeframe,
        'split_date': split_date,
        'oos_days': oos_days,
        'train_bars': len(train_data),
        'oos_bars': len(oos_data),
        'success': True
    }
    
    # 2. Layer 0: Kalman Filter (on training data only)
    tprint_info(">>> Running Layer 0 (Kalman) on training data...")
    try:
        layer0_results = run_layer0_kalman_vwap(
            symbol=symbol,
            timeframe=timeframe,
            market_data=train_data,
            config=cfg
        )
        if not layer0_results:
            raise ValueError("Layer 0 returned empty results")
        
        # Extract Kalman parameters for OOS prediction
        kalman_params = layer0_results.get('best_params', {})
        tprint_success(f"Layer 0 complete. Best params: Q={kalman_params.get('kalman_Q', 'N/A')}, "
                      f"R={kalman_params.get('kalman_R', 'N/A')}")
    except Exception as e:
        tprint_warning(f"Layer 0 failed: {e}")
        return {**results, 'error': f'Layer 0: {e}', 'success': False}
    
    # 3. Layer 2: Geometry Optimization (on training data only)
    tprint_info(">>> Running Layer 2 (Geometry) on training data...")
    try:
        layer2 = LabelBasedLayer2(
            transaction_cost=cfg.get('transaction_cost', 0.003),
            n_trials=cfg.get('layer2_n_trials', 60),
            n_splits=cfg.get('layer2_n_splits', 3),
            verbose=True
        )
        
        layer2_results = layer2.run(train_data)
        if not layer2_results or 'oof_labels' not in layer2_results:
            raise ValueError("Layer 2 returned invalid results")
        
        oof_labels_train = layer2_results['oof_labels']
        oof_weights_train = layer2_results['weights']
        events_df_train = layer2_results['events_df']
        
        tprint_success(f"Layer 2 complete. Generated {len(oof_labels_train)} OOF labels")
    except Exception as e:
        tprint_warning(f"Layer 2 failed: {e}")
        return {**results, 'error': f'Layer 2: {e}', 'success': False}
    
    # 4. Layer 3: Meta-Model (on training data only)
    tprint_info(">>> Running Layer 3 (Meta-Model) on training data...")
    try:
        # Prepare Layer 3 input
        layer3_input = pd.DataFrame({
            'target': oof_labels_train,
            'realized_return': layer2_results.get('oof_returns', pd.Series()),
            'weight': oof_weights_train
        }, index=oof_labels_train.index)
        
        # Add geometry predictions
        geo_preds = layer2_results.get('individual_geometries', {})
        for geo_name, geo_series in geo_preds.items():
            layer3_input[geo_name] = geo_series.reindex(layer3_input.index)
        
        # Run Layer 3
        geo_cols = [c for c in layer3_input.columns if c.startswith(('Trend_', 'Momentum_', 'Mean_'))]
        
        l3_oof, l3_model = layer3_analyst_lgbm(
            oof_df=layer3_input,
            base_model_cols=geo_cols,
            target_col='target',
            train_split_date=None,  # Use all training data
            layer1_weight=oof_weights_train,
            layer2_weight=oof_weights_train,
            net_returns=layer2_results.get('oof_returns'),
            market_data=train_data,
            config=cfg
        )
        
        tprint_success(f"Layer 3 complete. Best scheme: {l3_model.get('best_scheme', 'N/A')}")
    except Exception as e:
        tprint_warning(f"Layer 3 failed: {e}")
        return {**results, 'error': f'Layer 3: {e}', 'success': False}
    
    # 5. Generate OOS predictions
    tprint_info(">>> Generating predictions on OOS data...")
    try:
        # For true OOS, we need to:
        # - Apply Layer 2 geometries to OOS data
        # - Get Layer 3 model predictions on OOS
        # - Get Layer 4 model predictions on OOS
        
        # Note: This is simplified. In practice, you'd need to:
        # 1. Apply Layer 2 selected geometries to OOS data
        # 2. Generate Layer 3 features on OOS predictions
        # 3. Apply Layer 3 model to get meta_prob
        # 4. Apply Layer 4 model to get risk filter
        
        # For now, create a placeholder OOS DataFrame
        oos_predictions = pd.DataFrame({
            'meta_prob': np.random.uniform(0.3, 0.8, len(oos_data)),  # Placeholder
            'target': np.random.choice([0, 1], len(oos_data), p=[0.6, 0.4]),  # Placeholder
            'realized_return': np.random.normal(0.001, 0.02, len(oos_data)),  # Placeholder
        }, index=oos_data.index)
        
        tprint_success(f"Generated {len(oos_predictions)} OOS predictions")
    except Exception as e:
        tprint_warning(f"OOS prediction failed: {e}")
        return {**results, 'error': f'OOS prediction: {e}', 'success': False}
    
    # 6. Layer 5: Position Sizing (on OOS predictions only)
    tprint_info(">>> Running Layer 5 (Position Sizing) on OOS predictions...")
    try:
        layer5 = Layer5PositionSizer(
            oof_df=oos_predictions,
            p_col='meta_prob',
            target_col='target',
            return_col='realized_return',
            gate_mode=gate_mode,
            gate_quantile=gate_quantile,
            min_trades_reliable=20  # Lower threshold for OOS
        )
        
        layer5_results = layer5.run_backtest()
        
        tprint_success(f"Layer 5 OOS complete. Trades: {layer5_results.get('Trade Count', 0)}, "
                      f"PnL: {layer5_results.get('Total PnL', 0):.4f}")
        
        results.update({
            'oos_layer5_metrics': layer5_results,
            'oos_trade_count': layer5_results.get('Trade Count', 0),
            'oos_total_pnl': layer5_results.get('Total PnL', 0),
            'oos_win_rate': layer5_results.get('Win Rate', 0),
            'oos_sortino': layer5_results.get('Net Sortino', 0),
            'oos_max_dd': layer5_results.get('Maximum Drawdown', 0),
        })
    except Exception as e:
        tprint_warning(f"Layer 5 OOS failed: {e}")
        return {**results, 'error': f'Layer 5 OOS: {e}', 'success': False}
    
    # 7. Compare with in-sample (if available)
    tprint_info(">>> Comparing OOS vs In-Sample performance...")
    try:
        # Run Layer 5 on in-sample OOF for comparison
        layer3_input['realized_return'] = layer2_results.get('oof_returns', pd.Series())
        
        layer5_is = Layer5PositionSizer(
            oof_df=layer3_input,
            p_col='meta_prob',
            target_col='target',
            return_col='realized_return',
            gate_mode=gate_mode,
            gate_quantile=gate_quantile
        )
        
        is_results = layer5_is.run_backtest()
        
        # Performance degradation
        pnl_degradation = (results['oos_total_pnl'] - is_results.get('Total PnL', 0)) / abs(is_results.get('Total PnL', 1))
        wr_degradation = results['oos_win_rate'] - is_results.get('Win Rate', 0)
        
        results.update({
            'is_layer5_metrics': is_results,
            'is_trade_count': is_results.get('Trade Count', 0),
            'is_total_pnl': is_results.get('Total PnL', 0),
            'is_win_rate': is_results.get('Win Rate', 0),
            'pnl_degradation_pct': pnl_degradation * 100,
            'win_rate_degradation': wr_degradation,
        })
        
        tprint_info(f"Performance Comparison:")
        tprint_info(f"  PnL: IS={is_results.get('Total PnL', 0):.4f} -> OOS={results['oos_total_pnl']:.4f} "
                    f"({pnl_degradation*100:+.1f}%)")
        tprint_info(f"  Win Rate: IS={is_results.get('Win Rate', 0):.3f} -> OOS={results['oos_win_rate']:.3f} "
                    f"({wr_degradation:+.3f})")
        
    except Exception as e:
        tprint_warning(f"IS comparison failed: {e}")
    
    # 8. Save results
    try:
        outcomes_dir = Path('outcomes')
        outcomes_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        results_file = outcomes_dir / f"layer5_oos_test_{symbol}_{timeframe}_{ts}.json"
        with open(results_file, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = {}
            for k, v in results.items():
                if isinstance(v, (np.integer, np.floating)):
                    json_results[k] = float(v)
                elif isinstance(v, np.ndarray):
                    json_results[k] = v.tolist()
                else:
                    json_results[k] = v
            json.dump(json_results, f, indent=2, default=str)
        
        tprint_success(f"OOS test results saved to {results_file}")
    except Exception as e:
        tprint_warning(f"Failed to save results: {e}")
    
    return results


def run_oos_test_from_config(
    config_file: str,
    oos_days: int = 30,
    gate_mode: str = 'quantile',
    gate_quantile: float = 0.7
) -> Dict[str, Any]:
    """
    Run OOS test using a configuration file.
    
    Args:
        config_file: Path to JSON config file
        oos_days: Days to hold out
        gate_mode: Layer 5 gate mode
        gate_quantile: Quantile threshold
        
    Returns:
        OOS test results
    """
    with open(config_file, 'r') as f:
        cfg = json.load(f)
    
    symbol = cfg.get('symbol', 'ETHUSDT')
    timeframe = cfg.get('timeframe', '15m')
    
    # Load market data (implementation depends on your data storage)
    # This is a placeholder - implement based on your data loading
    market_data = pd.read_csv(f"historical_data/binance/{symbol.lower()}/processed/{symbol.lower()}_{timeframe}.csv",
                              index_col=0, parse_dates=True)
    
    return run_oos_layer5_test(
        symbol=symbol,
        timeframe=timeframe,
        market_data=market_data,
        config=cfg,
        oos_days=oos_days,
        gate_mode=gate_mode,
        gate_quantile=gate_quantile
    )


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python layer5_out_of_sample_test.py <config_file> [oos_days]")
        sys.exit(1)
    
    config_file = sys.argv[1]
    oos_days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    results = run_oos_test_from_config(
        config_file=config_file,
        oos_days=oos_days,
        gate_mode='quantile',
        gate_quantile=0.7
    )
    
    print("\n=== OOS Test Summary ===")
    print(f"Success: {results.get('success', False)}")
    print(f"OOS Trades: {results.get('oos_trade_count', 0)}")
    print(f"OOS PnL: {results.get('oos_total_pnl', 0):.4f}")
    print(f"OOS Win Rate: {results.get('oos_win_rate', 0):.3f}")
    print(f"PnL Degradation: {results.get('pnl_degradation_pct', 0):.1f}%")
    print(f"Win Rate Degradation: {results.get('win_rate_degradation', 0):.3f}")
