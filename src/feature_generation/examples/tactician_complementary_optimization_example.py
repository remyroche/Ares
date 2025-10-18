"""
Tactician Complementary Feature Optimization Example

This example demonstrates how to use the new complementary lookback optimization
for tactician training, ensuring features provide complementary information
beyond what the analyst already knows.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Import the new complementary optimization system
from ..utils.optimization.tactician_feature_optimization import (
    TacticianFeatureOptimizer,
    optimize_tactician_features,
    get_tactician_optimization_config
)
from ..utils.optimization.complementary_lookback_optimizer import (
    ComplementaryOptimizationConfig
)

def example_tactician_complementary_optimization():
    """
    Example of using complementary optimization for tactician training.
    """
    print("🎯 Tactician Complementary Feature Optimization Example")
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
    
    # 2. Create analyst outputs (simulated)
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
    
    # 3. Create tactician targets (profit-aware labels)
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
        ),
        'direction': pd.Series(
            np.random.choice([-1, 1], n_samples), 
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
    
    # 5. Create sample feature generators (simplified)
    print("\n5. Setting up feature generators...")
    
    # For this example, we'll create mock generators
    class MockFeatureGenerator:
        def __init__(self, name: str, default_lookback: int = 20):
            self.name = name
            self.default_lookback = default_lookback
            self.config = type('Config', (), {'name': name, 'default_lookback': default_lookback})()
        
        def supports_lookback_optimization(self) -> bool:
            return True
        
        def generate_with_lookback(self, data: pd.DataFrame, lookback: int) -> type('Result', (), {'success': True, 'data': self._generate_feature(data, lookback)})():
            return type('Result', (), {'success': True, 'data': self._generate_feature(data, lookback)})()
        
        def generate(self, data: pd.DataFrame) -> type('Result', (), {'success': True, 'data': self._generate_feature(data, self.default_lookback)})():
            return type('Result', (), {'success': True, 'data': self._generate_feature(data, self.default_lookback)})()
        
        def _generate_feature(self, data: pd.DataFrame, lookback: int) -> pd.Series:
            """Generate a mock feature based on lookback."""
            if 'close' in data.columns:
                # Simple moving average as example
                return data['close'].rolling(window=lookback).mean()
            else:
                return pd.Series(np.random.normal(0, 1, len(data)), index=data.index)
    
    generators = [
        MockFeatureGenerator('returns_ma', 20),
        MockFeatureGenerator('volatility', 30),
        MockFeatureGenerator('momentum', 15),
        MockFeatureGenerator('volume_ma', 25)
    ]
    print(f"   ✅ Created {len(generators)} feature generators")
    
    # 6. Configure complementary optimization
    print("\n6. Configuring complementary optimization...")
    config = get_tactician_optimization_config(
        min_lookback=5,
        max_lookback=50,
        analyst_alignment_penalty=0.7,  # High penalty for analyst alignment
        complementary_bonus=2.0,        # High bonus for complementary info
        regime_consistency_weight=0.4,  # High weight for regime consistency
        temporal_stability_weight=0.3   # High weight for temporal stability
    )
    print(f"   ✅ Configuration: penalty={config.analyst_alignment_penalty}, bonus={config.complementary_bonus}")
    
    # 7. Perform complementary optimization
    print("\n7. Performing complementary optimization...")
    optimizer = TacticianFeatureOptimizer(config)
    
    optimal_lookbacks = optimizer.optimize_for_tactician_training(
        generators=generators,
        data=market_data,
        tactician_targets=tactician_targets,
        analyst_outputs=analyst_outputs,
        regime_assignments=regime_assignments
    )
    
    print(f"   ✅ Optimized {len(optimal_lookbacks)} features")
    for feature_name, lookback in optimal_lookbacks.items():
        print(f"      {feature_name}: {lookback} periods")
    
    # 8. Generate optimization report
    print("\n8. Generating optimization report...")
    report = optimizer.get_optimization_report(
        optimal_lookbacks=optimal_lookbacks,
        generators=generators,
        data=market_data,
        tactician_targets=tactician_targets,
        analyst_outputs=analyst_outputs,
        regime_assignments=regime_assignments
    )
    
    print("   📊 Optimization Report:")
    print(f"      Total features optimized: {report['optimization_summary']['total_features']}")
    print(f"      Average lookback: {report['optimization_summary']['lookback_distribution']['mean']:.1f}")
    
    # Complementary analysis
    comp_analysis = report.get('complementary_analysis', {})
    high_comp = comp_analysis.get('high_complementary_features', [])
    low_comp = comp_analysis.get('low_complementary_features', [])
    
    if high_comp:
        print(f"      High complementary features: {high_comp}")
    if low_comp:
        print(f"      Low complementary features: {low_comp}")
    
    # Regime analysis
    regime_analysis = report.get('regime_analysis', {})
    regime_consistency = regime_analysis.get('regime_consistency', {})
    if regime_consistency:
        avg_consistency = np.mean(list(regime_consistency.values()))
        print(f"      Average regime consistency: {avg_consistency:.3f}")
    
    # Recommendations
    recommendations = report.get('recommendations', [])
    if recommendations:
        print("   💡 Recommendations:")
        for rec in recommendations:
            print(f"      - {rec}")
    
    # 9. Demonstrate multi-target optimization
    print("\n9. Demonstrating multi-target optimization...")
    target_weights = {
        'y_success': 0.4,      # Primary: profit success
        'r_H': 0.3,            # Secondary: realized returns  
        'time_to_hit': 0.2,    # Tertiary: timing
        'direction': 0.1       # Quaternary: direction
    }
    
    multi_target_lookbacks = optimizer.optimize_with_multi_target_objectives(
        generators=generators,
        data=market_data,
        tactician_targets=tactician_targets,
        analyst_outputs=analyst_outputs,
        regime_assignments=regime_assignments,
        target_weights=target_weights
    )
    
    print(f"   ✅ Multi-target optimization completed")
    print("   📊 Multi-target vs Single-target comparison:")
    for feature_name in optimal_lookbacks.keys():
        single_lookback = optimal_lookbacks[feature_name]
        multi_lookback = multi_target_lookbacks.get(feature_name, single_lookback)
        print(f"      {feature_name}: {single_lookback} → {multi_lookback}")
    
    print("\n🎉 Complementary optimization example completed!")
    print("\nKey Benefits:")
    print("✅ Features optimized for complementary information beyond analyst")
    print("✅ Regime-invariant optimization for consistent performance")
    print("✅ Multi-objective optimization for tactician targets")
    print("✅ Comprehensive analysis and recommendations")

if __name__ == "__main__":
    example_tactician_complementary_optimization()
