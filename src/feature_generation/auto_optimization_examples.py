"""
Auto-Optimization Examples

This module demonstrates how to use the new auto-optimization features
in the feature generation system. It shows various ways to create and use
auto-optimized generators with different optimization strategies.

Usage:
    python auto_optimization_examples.py
"""

import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(rows: int = 1000) -> pd.DataFrame:
    """Create sample market data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=rows, freq='1min')

    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(rows) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(rows) * 0.1) + np.random.rand(rows) * 2,
        'low': 100 + np.cumsum(np.random.randn(rows) * 0.1) - np.random.rand(rows) * 2,
        'close': 100 + np.cumsum(np.random.randn(rows) * 0.1),
        'volume': np.random.randint(1000, 10000, rows)
    }, index=dates)

    # Ensure high >= low and high/low >= close
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))

    return data

def example_1_basic_auto_optimization():
    """Example 1: Basic auto-optimization with AutoOptimizedFeatureGenerator."""
    print("🚀 Example 1: Basic Auto-Optimization")
    print("=" * 60)

    # Create sample data
    data = create_sample_data(500)

    # Import auto-optimization components
    from src.feature_generation import (
        AutoOptimizedFeatureGenerator,
        FeatureConfig,
        FeatureCategory,
        AutoOptimizationConfig,
        OptimizationLevel
    )

    # Create feature config
    config = FeatureConfig(
        name="auto_optimized_sma",
        category=FeatureCategory.CUSTOM,
        description="Auto-optimized Simple Moving Average",
        required_columns=["close"],
        default_lookback=20
    )

    # Create auto-optimization config
    auto_opt_config = AutoOptimizationConfig(
        optimization_level=OptimizationLevel.BALANCED,
        enable_optimization_logging=True
    )

    # Create auto-optimized generator
    class AutoOptimizedSMAGenerator(AutoOptimizedFeatureGenerator):
        def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
            # Data is automatically optimized before reaching here
            lookback = kwargs.get('lookback', self.config.default_lookback)
            return data['close'].rolling(lookback).mean()

    generator = AutoOptimizedSMAGenerator(config, auto_opt_config)

    # Generate feature
    print("📊 Generating feature with auto-optimization...")
    start_time = time.time()
    result = generator.generate(data)
    generation_time = time.time() - start_time

    print(f"✅ Feature generated successfully: {result.success}")
    print(f"⏱️ Total time: {generation_time:.3f}s")
    print(f"📈 Feature data shape: {result.data.shape}")
    print(f"🔧 Optimization stats: {result.metadata.get('optimization_stats', {})}")

    # Show auto-optimization stats
    opt_stats = generator.get_auto_optimization_stats()
    print(f"📊 Auto-optimization stats:")
    print(f"   - Total optimizations: {opt_stats['total_optimizations']}")
    print(f"   - Average optimization time: {opt_stats['average_optimization_time']:.3f}s")
    print(f"   - Memory saved: {opt_stats['memory_savings_mb']:.2f}MB")
    print()

def example_2_optimization_strategies():
    """Example 2: Different optimization strategies."""
    print("🚀 Example 2: Optimization Strategies")
    print("=" * 60)

    # Create sample data
    data = create_sample_data(1000)

    from src.feature_generation import (
        AutoOptimizedFeatureGenerator,
        FeatureConfig,
        FeatureCategory,
        AutoOptimizationConfig,
        OptimizationLevel
    )

    # Create feature config
    config = FeatureConfig(
        name="strategy_test_sma",
        category=FeatureCategory.CUSTOM,
        description="SMA for testing optimization strategies",
        required_columns=["close"],
        default_lookback=20
    )

    class StrategyTestGenerator(AutoOptimizedFeatureGenerator):
        def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
            lookback = kwargs.get('lookback', self.config.default_lookback)
            return data['close'].rolling(lookback).mean()

    # Test different optimization strategies
    strategies = [
        ("conservative", OptimizationLevel.CONSERVATIVE),
        ("balanced", OptimizationLevel.BALANCED),
        ("aggressive", OptimizationLevel.AGGRESSIVE)
    ]

    for strategy_name, strategy_level in strategies:
        print(f"🔧 Testing {strategy_name} strategy...")

        # Create auto-optimization config
        auto_opt_config = AutoOptimizationConfig(
            optimization_level=strategy_level,
            enable_optimization_logging=False
        )

        # Create generator
        generator = StrategyTestGenerator(config, auto_opt_config)

        # Generate feature
        start_time = time.time()
        result = generator.generate(data)
        generation_time = time.time() - start_time

        # Get optimization stats
        opt_stats = generator.get_auto_optimization_stats()

        print(f"   ✅ {strategy_name.capitalize()}: {result.success}")
        print(f"   ⏱️ Time: {generation_time:.3f}s")
        print(f"   🔧 Optimizations applied: {opt_stats['strategy_stats']['optimizations_applied']}")
        print(f"   💾 Memory saved: {opt_stats['memory_savings_mb']:.2f}MB")
        print()

def example_3_factory_auto_optimization():
    """Example 3: Using GeneratorFactory with auto-optimization."""
    print("🚀 Example 3: Factory Auto-Optimization")
    print("=" * 60)

    # Create sample data
    data = create_sample_data(800)

    from src.feature_generation import GeneratorFactory, FeatureCategory

    # Get factory
    factory = GeneratorFactory()

    # Create auto-optimized generators with different strategies
    strategies = ["conservative", "balanced", "aggressive"]

    for strategy in strategies:
        print(f"🔧 Creating generator with {strategy} strategy...")

        generator = factory.create_auto_optimized_generator(
            name=f"sma_{strategy}",
            category=FeatureCategory.CUSTOM,
            required_columns=["close"],
            optimization_level=strategy
        )

        if generator:
            # Generate feature
            start_time = time.time()
            result = generator.generate(data)
            generation_time = time.time() - start_time

            print(f"   ✅ {strategy.capitalize()}: {result.success}")
            print(f"   ⏱️ Time: {generation_time:.3f}s")

            # Show optimization metadata
            if result.metadata and 'optimization_stats' in result.metadata:
                opt_stats = result.metadata['optimization_stats']
                print(f"   🔧 Strategy optimizations: {opt_stats['optimizations_applied']}")
        print()

def example_4_feature_bank_auto_optimization():
    """Example 4: Using FeatureBank with auto-optimization."""
    print("🚀 Example 4: FeatureBank Auto-Optimization")
    print("=" * 60)

    # Create sample data
    data = create_sample_data(600)

    from src.feature_generation import FeatureBank, FeatureCategory, FeatureBankConfig

    # Create feature bank with auto-optimization enabled
    config = FeatureBankConfig(
        enable_auto_optimization=True,
        default_optimization_level="balanced"
    )

    bank = FeatureBank(config)

    # Show feature bank summary
    summary = bank.get_feature_summary()
    print(f"📊 Feature Bank Summary:")
    print(f"   - Total generators: {summary['total_generators']}")
    print(f"   - Auto-optimization enabled: {summary['auto_optimization_enabled']}")
    print(f"   - Optimization level: {summary['optimization_level']}")
    print()

    # Generate features by category
    print("🔧 Generating momentum features with auto-optimization...")
    start_time = time.time()

    try:
        features = bank.generate_features_by_category(
            data=data,
            category=FeatureCategory.MOMENTUM
        )
        generation_time = time.time() - start_time

        print(f"✅ Generated {len(features.columns)} momentum features")
        print(f"⏱️ Generation time: {generation_time:.3f}s")
        print(f"📈 Features: {list(features.columns)[:5]}...")  # Show first 5 features

        # Show optimization stats
        opt_stats = bank.get_optimization_stats()
        print(f"📊 Optimization stats:")
        print(f"   - Auto-optimized generators: {opt_stats['auto_optimized_generators']}")
        print(f"   - Total optimizations: {opt_stats['total_optimizations']}")
        print(f"   - Memory saved: {opt_stats['memory_savings_mb']:.2f}MB")

    except Exception as e:
        print(f"❌ Error generating features: {e}")

    print()

def example_5_runtime_optimization_control():
    """Example 5: Runtime optimization control."""
    print("🚀 Example 5: Runtime Optimization Control")
    print("=" * 60)

    # Create sample data
    data = create_sample_data(400)

    from src.feature_generation import (
        AutoOptimizedFeatureGenerator,
        FeatureConfig,
        FeatureCategory,
        AutoOptimizationConfig,
        OptimizationLevel
    )

    # Create feature config
    config = FeatureConfig(
        name="runtime_control_sma",
        category=FeatureCategory.CUSTOM,
        description="SMA with runtime optimization control",
        required_columns=["close"],
        default_lookback=20
    )

    # Create auto-optimized generator
    generator = AutoOptimizedFeatureGenerator(config)

    # Test different optimization levels at runtime
    for level in ["conservative", "balanced", "aggressive"]:
        print(f"🔧 Changing optimization strategy to {level}...")

        # Change optimization strategy
        generator.set_optimization_strategy(level)

        # Generate feature
        start_time = time.time()
        result = generator.generate(data)
        generation_time = time.time() - start_time

        print(f"   ✅ {level.capitalize()}: {result.success}")
        print(f"   ⏱️ Time: {generation_time:.3f}s")

        # Show stats
        opt_stats = generator.get_auto_optimization_stats()
        print(f"   🔧 Total optimizations: {opt_stats['total_optimizations']}")
        print(f"   💾 Memory saved: {opt_stats['memory_savings_mb']:.2f}MB")
        print()

def example_6_custom_optimization_config():
    """Example 6: Custom optimization configuration."""
    print("🚀 Example 6: Custom Optimization Configuration")
    print("=" * 60)

    # Create sample data
    data = create_sample_data(1200)

    from src.feature_generation import (
        AutoOptimizedFeatureGenerator,
        FeatureConfig,
        FeatureCategory,
        AutoOptimizationConfig,
        OptimizationLevel
    )

    # Create custom optimization configuration
    custom_config = AutoOptimizationConfig(
        optimization_level=OptimizationLevel.AGGRESSIVE,
        enable_memory_optimization=True,
        memory_threshold_mb=50.0,
        enable_vectorbt_optimization=True,
        vectorbt_threshold=500,
        enable_rolling_optimization=True,
        enable_performance_monitoring=True,
        enable_optimization_logging=True
    )

    # Create feature config
    config = FeatureConfig(
        name="custom_optimized_sma",
        category=FeatureCategory.CUSTOM,
        description="SMA with custom optimization configuration",
        required_columns=["close"],
        default_lookback=20
    )

    class CustomOptimizedGenerator(AutoOptimizedFeatureGenerator):
        def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
            lookback = kwargs.get('lookback', self.config.default_lookback)
            return data['close'].rolling(lookback).mean()

    # Create generator with custom config
    generator = CustomOptimizedGenerator(config, custom_config)

    print("📊 Custom optimization configuration:")
    config_dict = custom_config.to_dict()
    for key, value in config_dict.items():
        print(f"   - {key}: {value}")
    print()

    # Generate feature
    print("🔧 Generating feature with custom optimization...")
    start_time = time.time()
    result = generator.generate(data)
    generation_time = time.time() - start_time

    print(f"✅ Feature generated: {result.success}")
    print(f"⏱️ Generation time: {generation_time:.3f}s")

    # Show detailed optimization stats
    opt_stats = generator.get_auto_optimization_stats()
    print(f"📊 Detailed optimization stats:")
    for key, value in opt_stats.items():
        print(f"   - {key}: {value}")
    print()

def example_7_batch_auto_optimization():
    """Example 7: Batch auto-optimization with factory."""
    print("🚀 Example 7: Batch Auto-Optimization")
    print("=" * 60)

    # Create sample data
    data = create_sample_data(700)

    from src.feature_generation import GeneratorFactory, FeatureCategory

    # Get factory
    factory = GeneratorFactory()

    # Create batch generator specifications
    generator_specs = [
        {
            'name': 'sma_20',
            'category': FeatureCategory.CUSTOM,
            'required_columns': ['close'],
            'optimization_level': 'conservative',
            'parameters': {'lookback': 20}
        },
        {
            'name': 'sma_50',
            'category': FeatureCategory.CUSTOM,
            'required_columns': ['close'],
            'optimization_level': 'balanced',
            'parameters': {'lookback': 50}
        },
        {
            'name': 'sma_100',
            'category': FeatureCategory.CUSTOM,
            'required_columns': ['close'],
            'optimization_level': 'aggressive',
            'parameters': {'lookback': 100}
        }
    ]

    print("🔧 Creating batch auto-optimized generators...")
    generators = factory.create_batch_auto_optimized_generators(generator_specs)

    print(f"✅ Created {len(generators)} auto-optimized generators")

    # Generate features with all generators
    print("📊 Generating features with batch generators...")
    start_time = time.time()

    results = []
    for generator in generators:
        result = generator.generate(data)
        results.append(result)

    generation_time = time.time() - start_time

    print(f"✅ Generated features with {len(results)} generators")
    print(f"⏱️ Total generation time: {generation_time:.3f}s")

    # Show results summary
    successful = sum(1 for r in results if r.success)
    print(f"📈 Successful generations: {successful}/{len(results)}")

    # Show optimization stats for each generator
    for i, generator in enumerate(generators):
        opt_stats = generator.get_auto_optimization_stats()
        print(f"   Generator {i+1} ({generator.config.name}):")
        print(f"     - Optimizations: {opt_stats['total_optimizations']}")
        print(f"     - Memory saved: {opt_stats['memory_savings_mb']:.2f}MB")
    print()

def main():
    """Run all auto-optimization examples."""
    print("🎯 Auto-Optimization Examples - Feature Generation System")
    print("=" * 80)
    print()

    try:
        example_1_basic_auto_optimization()
        example_2_optimization_strategies()
        example_3_factory_auto_optimization()
        example_4_feature_bank_auto_optimization()
        example_5_runtime_optimization_control()
        example_6_custom_optimization_config()
        example_7_batch_auto_optimization()

        print("🎉 All auto-optimization examples completed successfully!")
        print("=" * 80)
        print("📚 Auto-Optimization Features Demonstrated:")
        print("  ✅ Automatic optimization enabled by default")
        print("  ✅ Configurable optimization strategies (conservative, balanced, aggressive)")
        print("  ✅ Runtime optimization control")
        print("  ✅ Performance monitoring and statistics")
        print("  ✅ Factory pattern with auto-optimization")
        print("  ✅ FeatureBank integration with auto-optimization")
        print("  ✅ Custom optimization configuration")
        print("  ✅ Batch auto-optimization")
        print("  ✅ Backward compatibility maintained")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
