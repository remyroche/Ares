"""
Example: NAS/TAS Target Meta Labeling System

This example demonstrates how to use the NAS/TAS integration system
to generate comprehensive meta labels for trading targets.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.analyst.nas_tas_target_integration import NAS_TAS_TargetIntegrationSystem
from src.analyst.trading_target_meta_labels import TradingTarget


def create_sample_market_data(n_periods: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create sample market data for testing."""
    np.random.seed(42)
    
    # Generate price data
    dates = pd.date_range(start='2023-01-01', periods=n_periods, freq='1H')
    
    # Generate realistic price movements
    returns = np.random.normal(0, 0.02, n_periods)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    price_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, n_periods)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_periods))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_periods)
    })
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    price_data['high'] = np.maximum(price_data['high'], np.maximum(price_data['open'], price_data['close']))
    price_data['low'] = np.minimum(price_data['low'], np.minimum(price_data['open'], price_data['close']))
    
    # Create volume data
    volume_data = pd.DataFrame({
        'timestamp': dates,
        'volume': price_data['volume']
    })
    
    return price_data, volume_data


def create_sample_regime_data(n_periods: int = 1000) -> pd.DataFrame:
    """Create sample regime data for testing."""
    np.random.seed(42)
    
    # Generate regime data (0: bear, 1: bull, 2: sideways)
    regimes = np.random.choice([0, 1, 2], n_periods, p=[0.3, 0.4, 0.3])
    
    # Add some persistence to regimes
    for i in range(1, len(regimes)):
        if np.random.random() < 0.9:  # 90% chance to stay in same regime
            regimes[i] = regimes[i-1]
    
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=n_periods, freq='1H'),
        'regime': regimes
    })


async def main():
    """Main example function."""
    print("🚀 NAS/TAS Target Meta Labeling Example")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating sample market data...")
    price_data, volume_data = create_sample_market_data(1000)
    regime_data = create_sample_regime_data(1000)
    
    print(f"   → Price data shape: {price_data.shape}")
    print(f"   → Volume data shape: {volume_data.shape}")
    print(f"   → Regime data shape: {regime_data.shape}")
    
    # Initialize the integration system
    print("\n🔧 Initializing NAS/TAS Target Integration System...")
    config = {
        "target_meta_labeling": {
            "breakout": {"volume_threshold": 1.5, "momentum_threshold": 0.01},
            "mean_reversion": {"volatility_threshold": 0.02, "rsi_threshold": 30},
            "trend_following": {"sma_periods": [20, 50], "momentum_threshold": 0.01},
            "rejection": {"wick_threshold": 0.01, "rsi_extreme": 30},
            "support_resistance": {"distance_threshold": 0.01, "volume_threshold": 1.5},
            "consolidation": {"bb_width_threshold": 0.05, "volatility_threshold": 0.02}
        },
        "meta_labeling": {
            "enable_analyst_labels": True,
            "enable_tactician_labels": True,
            "pattern_detection": {
                "volatility_threshold": 0.02,
                "momentum_threshold": 0.01,
                "volume_threshold": 1.5
            }
        },
        "ensemble": {
            "enabled_strategies": ["multi_horizon", "volatility_adjusted", "momentum_based"],
            "combination_method": "performance_weighted"
        },
        "nas": {
            "population_size": 50,
            "generations": 100,
            "enable_multi_objective": True
        },
        "tas": {
            "max_depth": 10,
            "min_samples_split": 20,
            "min_samples_leaf": 10
        }
    }
    
    integration_system = NAS_TAS_TargetIntegrationSystem(config)
    
    # Generate enhanced target labels
    print("\n🎯 Generating enhanced target meta labels...")
    enhanced_labels = await integration_system.generate_enhanced_target_labels(
        price_data=price_data,
        volume_data=volume_data,
        regime_data=regime_data
    )
    
    print(f"   → Generated {len(enhanced_labels)} enhanced target labels")
    
    # Display results
    print("\n📈 Target Meta Label Results:")
    print("-" * 40)
    
    for target_name, label in enhanced_labels.items():
        print(f"\n🎯 {target_name.upper()}:")
        print(f"   Signal Strength: {label.signal_strength:.3f}")
        print(f"   Confidence: {label.confidence:.3f}")
        print(f"   Probability: {label.probability:.3f}")
        print(f"   Time Horizon: {label.time_horizon}")
        print(f"   Risk Level: {label.risk_level}")
        print(f"   Setup Quality: {label.setup_quality:.3f}")
        print(f"   Overall Quality: {label.overall_quality_score:.3f}")
        
        # NAS features
        print(f"   NAS Architecture Score: {label.nas_architecture_score:.3f}")
        print(f"   NAS Regime Accuracy: {label.nas_regime_accuracy:.3f}")
        print(f"   NAS Trading Viability: {label.nas_trading_viability:.3f}")
        
        # TAS features
        print(f"   TAS Tree Depth: {label.tas_tree_depth:.1f}")
        print(f"   TAS Decision Quality: {label.tas_decision_quality:.3f}")
        print(f"   TAS Feature Importance: {label.tas_feature_importance:.3f}")
        
        # Combined features
        print(f"   Combined Meta Score: {label.combined_meta_score:.3f}")
        print(f"   NAS/TAS Synergy: {label.nas_tas_synergy_score:.3f}")
    
    # Get summary
    print("\n📊 Summary Statistics:")
    print("-" * 30)
    summary = integration_system.get_target_meta_label_summary(enhanced_labels)
    
    print(f"Total Targets: {summary['total_targets']}")
    print(f"Average Quality Score: {summary['average_quality_score']:.3f}")
    print(f"Average Confidence: {summary['average_confidence']:.3f}")
    print(f"Average Signal Strength: {summary['average_signal_strength']:.3f}")
    print(f"NAS/TAS Synergy: {summary['nas_tas_synergy']:.3f}")
    
    print("\n🏆 Top 3 Targets by Quality:")
    for i, (target_name, quality_score) in enumerate(summary['top_targets'], 1):
        print(f"   {i}. {target_name}: {quality_score:.3f}")
    
    # Save results
    print("\n💾 Saving enhanced labels...")
    output_path = "enhanced_target_labels.json"
    integration_system.save_enhanced_labels(enhanced_labels, output_path)
    print(f"   → Saved to {output_path}")
    
    # Demonstrate specific target analysis
    print("\n🔍 Detailed Analysis of Top Target:")
    top_target_name, top_quality = summary['top_targets'][0]
    top_label = enhanced_labels[top_target_name]
    
    print(f"\nTarget: {top_target_name}")
    print(f"Entry Conditions: {top_label.entry_conditions}")
    print(f"Exit Conditions: {top_label.exit_conditions}")
    print(f"Metadata: {top_label.metadata}")
    
    print("\n✅ Example completed successfully!")
    print("\nThis system provides:")
    print("   • Comprehensive trading target detection")
    print("   • NAS/TAS enhanced meta features")
    print("   • Traditional analyst/tactician labels")
    print("   • Ensemble combination methods")
    print("   • Quality scoring and ranking")
    print("   • Rich metadata for ML training")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())