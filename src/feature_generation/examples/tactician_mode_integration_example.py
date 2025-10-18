"""
Tactician Mode Integration Example

This example demonstrates how to use the feature bank in both regular and Tactician modes,
showing the conditional optimization behavior.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Import the feature bank and tactician optimization
from ..core.feature_bank import FeatureBank
from ..utils.optimization.tactician_feature_optimization import (
    TacticianFeatureOptimizer,
    generate_tactician_features_with_optimization,
    get_tactician_optimization_config
)

def example_regular_vs_tactician_mode():
    """
    Example showing the difference between regular and Tactician mode optimization.
    """
    print("🎯 Tactician Mode Integration Example")
    print("=" * 60)
    
    # 1. Create sample market data
    print("\n1. Creating sample market data...")
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='15T')
    n_samples = len(dates)
    
    # Generate realistic market data
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
    market_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    }, index=dates)
    
    print(f"   ✅ Created {len(market_data)} samples of market data")
    
    # 2. Create analyst outputs
    print("\n2. Creating analyst outputs...")
    analyst_outputs = {
        'analyst_oof_score': pd.Series(
            np.random.normal(0.5, 0.2, n_samples), 
            index=dates
        ),
        'analyst_confidence': pd.Series(
            np.random.uniform(0.3, 0.9, n_samples), 
            index=dates
        )
    }
    print(f"   ✅ Created analyst outputs: {list(analyst_outputs.keys())}")
    
    # 3. Create tactician targets
    print("\n3. Creating tactician targets...")
    tactician_targets = {
        'y_success': pd.Series(
            np.random.binomial(1, 0.3, n_samples), 
            index=dates
        ),
        'r_H': pd.Series(
            np.random.normal(0.001, 0.01, n_samples), 
            index=dates
        ),
        'time_to_hit': pd.Series(
            np.random.exponential(10, n_samples), 
            index=dates
        )
    }
    print(f"   ✅ Created tactician targets: {list(tactician_targets.keys())}")
    
    # 4. Create regime assignments
    print("\n4. Creating regime assignments...")
    regime_assignments = pd.Series(
        np.random.choice(['bull', 'bear', 'sideways'], n_samples, p=[0.4, 0.3, 0.3]),
        index=dates
    )
    print(f"   ✅ Created regimes: {regime_assignments.value_counts().to_dict()}")
    
    # 5. Initialize feature bank
    print("\n5. Initializing feature bank...")
    feature_bank = FeatureBank()
    print("   ✅ Feature bank initialized with both regular and Tactician optimizers")
    
    # 6. Regular mode feature generation
    print("\n6. Generating features in REGULAR mode...")
    print("   📊 Using standard lookback optimization (no complementary scoring)")
    
    regular_features = feature_bank.generate_features(
        data=market_data,
        categories=['returns', 'momentum', 'volume'],
        lookback_optimization=True,
        target_column='close',  # Simple target for regular mode
        # No tactician_mode, analyst_signals, or regime_series
    )
    
    print(f"   ✅ Generated {len(regular_features.columns)} features in regular mode")
    print(f"   📊 Feature columns: {list(regular_features.columns)[:5]}...")
    
    # 7. Tactician mode feature generation
    print("\n7. Generating features in TACTICIAN mode...")
    print("   🎯 Using complementary scoring with analyst signals and regime analysis")
    
    tactician_features = feature_bank.generate_features(
        data=market_data,
        categories=['returns', 'momentum', 'volume'],
        lookback_optimization=True,
        target_column='y_success',
        tactician_mode=True,  # Enable Tactician mode
        analyst_signals=analyst_outputs['analyst_oof_score'],
        regime_series=regime_assignments
    )
    
    print(f"   ✅ Generated {len(tactician_features.columns)} features in Tactician mode")
    print(f"   📊 Feature columns: {list(tactician_features.columns)[:5]}...")
    
    # 8. Using TacticianFeatureOptimizer directly
    print("\n8. Using TacticianFeatureOptimizer for advanced optimization...")
    
    # Configure tactician-specific optimization
    tactician_config = get_tactician_optimization_config(
        analyst_alignment_penalty=0.7,
        complementary_bonus=2.0,
        regime_consistency_weight=0.4
    )
    
    tactician_optimizer = TacticianFeatureOptimizer(tactician_config)
    
    # Generate features using the tactician optimizer
    tactician_features_advanced = tactician_optimizer.generate_tactician_features(
        feature_bank=feature_bank,
        data=market_data,
        tactician_targets=tactician_targets,
        analyst_outputs=analyst_outputs,
        regime_assignments=regime_assignments,
        categories=['returns', 'momentum', 'volume']
    )
    
    print(f"   ✅ Generated {len(tactician_features_advanced.columns)} features with advanced Tactician optimization")
    
    # 9. Using convenience function
    print("\n9. Using convenience function for Tactician feature generation...")
    
    tactician_features_convenience = generate_tactician_features_with_optimization(
        feature_bank=feature_bank,
        data=market_data,
        tactician_targets=tactician_targets,
        analyst_outputs=analyst_outputs,
        regime_assignments=regime_assignments,
        categories=['returns', 'momentum', 'volume'],
        config=tactician_config
    )
    
    print(f"   ✅ Generated {len(tactician_features_convenience.columns)} features using convenience function")
    
    # 10. Compare results
    print("\n10. Comparing results...")
    print(f"   📊 Regular mode features: {len(regular_features.columns)}")
    print(f"   🎯 Tactician mode features: {len(tactician_features.columns)}")
    print(f"   🧠 Advanced Tactician features: {len(tactician_features_advanced.columns)}")
    print(f"   🔧 Convenience function features: {len(tactician_features_convenience.columns)}")
    
    # Check if features are different (they should be due to different optimization)
    regular_cols = set(regular_features.columns)
    tactician_cols = set(tactician_features.columns)
    
    if regular_cols == tactician_cols:
        print("   ⚠️ Warning: Regular and Tactician features are identical (unexpected)")
    else:
        print("   ✅ Regular and Tactician features are different (expected due to different optimization)")
        print(f"   📊 Unique to regular: {len(regular_cols - tactician_cols)}")
        print(f"   📊 Unique to tactician: {len(tactician_cols - regular_cols)}")
    
    print("\n🎉 Tactician mode integration example completed!")
    print("\nKey Benefits:")
    print("✅ Conditional optimization based on mode")
    print("✅ Regular mode uses standard optimization")
    print("✅ Tactician mode uses complementary scoring")
    print("✅ Seamless integration with existing feature bank")
    print("✅ Advanced optimization capabilities for Tactician training")

def example_mode_selection_logic():
    """
    Example showing the mode selection logic in the feature bank.
    """
    print("\n🔍 Mode Selection Logic Example")
    print("=" * 40)
    
    print("Regular Mode (tactician_mode=False or not specified):")
    print("  - Uses self.lookback_optimizer (standard optimization)")
    print("  - No analyst signals or regime analysis")
    print("  - Standard correlation-based optimization")
    
    print("\nTactician Mode (tactician_mode=True):")
    print("  - Uses self.tactician_optimizer (complementary optimization)")
    print("  - Requires analyst_signals for complementary scoring")
    print("  - Uses regime_series for regime-invariant optimization")
    print("  - Advanced optimization with partial correlation")
    
    print("\nAutomatic Mode Detection:")
    print("  - Feature bank checks kwargs.get('tactician_mode', False)")
    print("  - If True and tactician_optimizer available → Tactician mode")
    print("  - Otherwise → Regular mode")
    
    print("\nIntegration Points:")
    print("  - TacticianFeatureOptimizer calls feature bank with tactician_mode=True")
    print("  - Regular feature generation uses default mode (False)")
    print("  - Explicit mode control via kwargs")

if __name__ == "__main__":
    example_regular_vs_tactician_mode()
    example_mode_selection_logic()
