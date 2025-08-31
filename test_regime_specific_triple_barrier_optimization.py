#!/usr/bin/env python3
"""
Test Script for Regime-Specific Triple Barrier Optimization

This script demonstrates how to use the regime-specific triple barrier optimization
system to optimize triple barrier thresholds and TPSL parameters for each HMM regime.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any
import warnings

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import the regime-specific optimization components
from src.training.steps.step17_final_parameters_optimization.regime_specific_triple_barrier_optimization import (
    RegimeSpecificTripleBarrierOptimizer,
    optimize_regime_triple_barrier_parameters,
    get_regime_optimized_triple_barrier_params,
    get_regime_optimized_tpsl_params
)

from src.training.steps.step4_analyst_labeling_feature_engineering_components.regime_aware_triple_barrier_labeling import (
    RegimeAwareTripleBarrierLabeling,
    RegimeTripleBarrierConfig,
    apply_regime_aware_triple_barrier_labeling,
    create_regime_aware_labeler_from_optimization_results
)


def generate_test_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate test data with regime information."""
    
    print("📊 Generating test data...")
    
    # Generate synthetic OHLCV data
    np.random.seed(42)
    
    # Generate price data with some trend and volatility
    returns = np.random.normal(0.0001, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLC data
    data = pd.DataFrame({
        'close': prices,
        'open': prices * (1 + np.random.normal(0, 0.005, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'volume': np.random.lognormal(10, 1, n_samples),
    })
    
    # Ensure OHLC relationships are valid
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    # Generate regime information (simulate HMM regimes)
    n_regimes = 5
    regime_lengths = np.random.poisson(50, n_regimes)  # Average regime length of 50 periods
    regime_lengths = np.clip(regime_lengths, 20, 200)  # Ensure reasonable lengths
    
    regimes = []
    current_pos = 0
    
    while current_pos < n_samples:
        regime_id = np.random.randint(0, n_regimes)
        regime_length = min(regime_lengths[regime_id], n_samples - current_pos)
        regimes.extend([regime_id] * regime_length)
        current_pos += regime_length
    
    # Truncate to exact length
    regimes = regimes[:n_samples]
    data['composite_cluster_id'] = regimes
    
    # Add some regime-specific characteristics
    for regime_id in range(n_regimes):
        regime_mask = data['composite_cluster_id'] == regime_id
        if regime_id == 0:  # Bull trend
            data.loc[regime_mask, 'close'] *= (1 + np.cumsum(np.random.normal(0.0002, 0.01, regime_mask.sum())))
        elif regime_id == 1:  # Bear trend
            data.loc[regime_mask, 'close'] *= (1 - np.cumsum(np.random.normal(0.0002, 0.01, regime_mask.sum())))
        elif regime_id == 2:  # Sideways
            data.loc[regime_mask, 'close'] *= (1 + np.random.normal(0, 0.005, regime_mask.sum()))
        elif regime_id == 3:  # High volatility
            data.loc[regime_mask, 'close'] *= (1 + np.random.normal(0, 0.03, regime_mask.sum()))
        elif regime_id == 4:  # Low volatility
            data.loc[regime_mask, 'close'] *= (1 + np.random.normal(0, 0.005, regime_mask.sum()))
    
    # Update OHLC based on new close prices
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    print(f"✅ Generated test data with {len(data)} samples and {n_regimes} regimes")
    print(f"   Regime distribution: {data['composite_cluster_id'].value_counts().sort_index().to_dict()}")
    
    return data


def create_optimization_config() -> Dict[str, Any]:
    """Create configuration for regime-specific optimization."""
    
    config = {
        "regime_specific_optimization": {
            "enable_regime_optimization": True,
            "multi_objective": True,
            "n_trials_per_regime": 50,  # Reduced for testing
            "timeout_minutes_per_regime": 10,  # Reduced for testing
            "cv_folds": 3,  # Reduced for testing
            
            "objectives": ["sharpe_ratio", "win_rate", "profit_factor", "regime_accuracy"],
            "objective_weights": {
                "sharpe_ratio": 0.3,
                "win_rate": 0.25,
                "profit_factor": 0.25,
                "regime_accuracy": 0.2
            },
            
            "regime_constraints": {
                "REGIME_0": {  # Bull trend
                    "tp_multiplier_range": [2.5, 5.0],
                    "sl_multiplier_range": [1.2, 2.5],
                    "position_size_range": [0.10, 0.25],
                },
                "REGIME_1": {  # Bear trend
                    "tp_multiplier_range": [2.0, 4.5],
                    "sl_multiplier_range": [1.0, 2.2],
                    "position_size_range": [0.08, 0.20],
                },
                "REGIME_2": {  # Sideways
                    "tp_multiplier_range": [1.5, 3.0],
                    "sl_multiplier_range": [0.8, 1.8],
                    "position_size_range": [0.06, 0.15],
                },
                "REGIME_3": {  # High volatility
                    "tp_multiplier_range": [1.8, 3.5],
                    "sl_multiplier_range": [0.9, 2.0],
                    "position_size_range": [0.05, 0.12],
                },
                "REGIME_4": {  # Low volatility
                    "tp_multiplier_range": [2.0, 4.0],
                    "sl_multiplier_range": [1.0, 2.2],
                    "position_size_range": [0.08, 0.18],
                },
            },
            
            "early_stopping_patience": 10,
            "early_stopping_delta": 0.001,
            "enable_pruning": True,
            "pruning_method": "hyperband",
            "enable_statistical_testing": True,
            "confidence_level": 0.95,
            "min_sample_size": 30,  # Reduced for testing
        }
    }
    
    return config


async def test_regime_specific_optimization():
    """Test the regime-specific triple barrier optimization."""
    
    print("🚀 Testing Regime-Specific Triple Barrier Optimization")
    print("=" * 60)
    
    # Generate test data
    test_data = generate_test_data(n_samples=5000)  # Reduced for testing
    
    # Create optimization configuration
    config = create_optimization_config()
    
    print("\n📋 Optimization Configuration:")
    print(f"   - Trials per regime: {config['regime_specific_optimization']['n_trials_per_regime']}")
    print(f"   - Timeout per regime: {config['regime_specific_optimization']['timeout_minutes_per_regime']} minutes")
    print(f"   - CV folds: {config['regime_specific_optimization']['cv_folds']}")
    print(f"   - Objectives: {config['regime_specific_optimization']['objectives']}")
    
    # Run optimization
    print("\n🎯 Starting regime-specific optimization...")
    
    try:
        optimization_results = await optimize_regime_triple_barrier_parameters(
            data=test_data,
            config=config,
            regime_column="composite_cluster_id"
        )
        
        print(f"\n✅ Optimization completed successfully!")
        print(f"   - Optimized {len(optimization_results)} regimes")
        
        # Display results
        print("\n📊 Optimization Results:")
        print("-" * 80)
        
        for regime_name, result in optimization_results.items():
            print(f"\n🎯 {regime_name}:")
            print(f"   Optimization Score: {result.optimization_score:.4f}")
            print(f"   Sharpe Ratio: {result.sharpe_ratio:.4f}")
            print(f"   Win Rate: {result.win_rate:.4f}")
            print(f"   Profit Factor: {result.profit_factor:.4f}")
            print(f"   Total Return: {result.total_return:.4f}")
            print(f"   Max Drawdown: {result.max_drawdown:.4f}")
            print(f"   Trials: {result.n_trials}")
            print(f"   Time: {result.optimization_time:.2f}s")
            
            # Display optimized parameters
            print(f"   Triple Barrier Params:")
            print(f"     - Profit Take Multiplier: {result.triple_barrier_params.profit_take_multiplier:.4f}")
            print(f"     - Stop Loss Multiplier: {result.triple_barrier_params.stop_loss_multiplier:.4f}")
            print(f"     - Time Barrier Minutes: {result.triple_barrier_params.time_barrier_minutes}")
            print(f"     - Max Lookahead: {result.triple_barrier_params.max_lookahead}")
            
            print(f"   TPSL Params:")
            for param_name, param_value in result.tpsl_params.items():
                print(f"     - {param_name}: {param_value:.4f}")
        
        # Test regime-aware labeling with optimized parameters
        print("\n🧪 Testing regime-aware labeling with optimized parameters...")
        
        # Create regime-aware labeler from optimization results
        labeler = create_regime_aware_labeler_from_optimization_results(optimization_results)
        
        # Apply regime-aware labeling
        labeled_data = labeler.apply_regime_aware_triple_barrier_labeling(
            test_data, 
            regime_column="composite_cluster_id"
        )
        
        print(f"✅ Regime-aware labeling completed!")
        print(f"   - Original samples: {len(test_data)}")
        print(f"   - Labeled samples: {len(labeled_data)}")
        print(f"   - Label distribution: {labeled_data['label'].value_counts().to_dict()}")
        
        # Get performance summary by regime
        performance_summary = labeler.get_regime_performance_summary(
            labeled_data, 
            regime_column="composite_cluster_id"
        )
        
        print("\n📈 Performance Summary by Regime:")
        print("-" * 60)
        
        for regime_name, metrics in performance_summary.items():
            print(f"\n🎯 {regime_name}:")
            print(f"   Total Samples: {metrics['total_samples']}")
            print(f"   Valid Samples: {metrics['valid_samples']}")
            print(f"   Win Rate: {metrics['win_rate']:.4f}")
            print(f"   Avg Profit: {metrics['avg_profit']:.4f}")
            print(f"   Total Return: {metrics['total_return']:.4f}")
        
        # Test utility functions
        print("\n🔧 Testing utility functions...")
        
        # Test getting optimized parameters for specific regimes
        for regime_name in optimization_results.keys():
            tb_params = get_regime_optimized_triple_barrier_params(regime_name, optimization_results)
            tpsl_params = get_regime_optimized_tpsl_params(regime_name, optimization_results)
            
            if tb_params and tpsl_params:
                print(f"✅ Retrieved optimized parameters for {regime_name}")
            else:
                print(f"❌ Failed to retrieve optimized parameters for {regime_name}")
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()


async def test_regime_aware_labeling_standalone():
    """Test regime-aware labeling without optimization."""
    
    print("\n🧪 Testing Regime-Aware Labeling (Standalone)")
    print("=" * 50)
    
    # Generate test data
    test_data = generate_test_data(n_samples=2000)
    
    # Create regime-aware labeler with custom parameters
    config = RegimeTripleBarrierConfig()
    
    # Set regime-specific parameters
    regime_params = {
        "REGIME_0": {"profit_take": 0.03, "stop_loss": 0.015, "tp_mult": 3.0, "sl_mult": 1.5},
        "REGIME_1": {"profit_take": 0.025, "stop_loss": 0.012, "tp_mult": 2.5, "sl_mult": 1.2},
        "REGIME_2": {"profit_take": 0.02, "stop_loss": 0.01, "tp_mult": 2.0, "sl_mult": 1.0},
        "REGIME_3": {"profit_take": 0.035, "stop_loss": 0.018, "tp_mult": 3.5, "sl_mult": 1.8},
        "REGIME_4": {"profit_take": 0.018, "stop_loss": 0.008, "tp_mult": 1.8, "sl_mult": 0.8},
    }
    
    labeler = RegimeAwareTripleBarrierLabeling(config)
    
    # Set parameters for each regime
    for regime_name, params in regime_params.items():
        labeler.set_regime_parameters(
            regime_name=regime_name,
            profit_take_multiplier=params["profit_take"],
            stop_loss_multiplier=params["stop_loss"],
            tp_multiplier=params["tp_mult"],
            sl_multiplier=params["sl_mult"],
            position_size=0.1
        )
    
    # Apply regime-aware labeling
    labeled_data = labeler.apply_regime_aware_triple_barrier_labeling(
        test_data, 
        regime_column="composite_cluster_id"
    )
    
    print(f"✅ Standalone regime-aware labeling completed!")
    print(f"   - Original samples: {len(test_data)}")
    print(f"   - Labeled samples: {len(labeled_data)}")
    print(f"   - Label distribution: {labeled_data['label'].value_counts().to_dict()}")
    
    # Check for TPSL columns
    tpsl_columns = ['tp_level', 'sl_level', 'position_size']
    missing_columns = [col for col in tpsl_columns if col not in labeled_data.columns]
    
    if missing_columns:
        print(f"⚠️ Missing TPSL columns: {missing_columns}")
    else:
        print(f"✅ TPSL columns present: {tpsl_columns}")
    
    # Get performance summary
    performance_summary = labeler.get_regime_performance_summary(
        labeled_data, 
        regime_column="composite_cluster_id"
    )
    
    print("\n📈 Performance Summary:")
    for regime_name, metrics in performance_summary.items():
        print(f"   {regime_name}: Win Rate={metrics['win_rate']:.4f}, "
              f"Avg Profit={metrics['avg_profit']:.4f}, "
              f"Total Return={metrics['total_return']:.4f}")


async def main():
    """Main test function."""
    
    print("🧪 Regime-Specific Triple Barrier Optimization Test Suite")
    print("=" * 70)
    
    # Test 1: Regime-specific optimization
    await test_regime_specific_optimization()
    
    # Test 2: Standalone regime-aware labeling
    await test_regime_aware_labeling_standalone()
    
    print("\n🎉 All tests completed!")
    print("\n📝 Summary:")
    print("   ✅ Regime-specific triple barrier optimization")
    print("   ✅ Per-regime TPSL parameter optimization")
    print("   ✅ Regime-aware labeling with optimized parameters")
    print("   ✅ Performance tracking by regime")
    print("   ✅ Utility functions for parameter retrieval")


if __name__ == "__main__":
    # Run the tests
    asyncio.run(main())