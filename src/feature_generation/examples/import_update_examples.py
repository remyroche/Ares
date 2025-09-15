"""
Import Update Examples

This module shows specific examples of how to update existing code
to use the new unified feature generation system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

# ============================================================================
# EXAMPLE 1: Updating Feature Engineering Orchestrator
# ============================================================================

def example_update_feature_engineering_orchestrator():
    """
    Example of how to update the FeatureEngineeringOrchestrator
    to use the new unified system.
    """
    print("=== Example 1: Updating FeatureEngineeringOrchestrator ===")
    
    # BEFORE: Old approach
    print("BEFORE (Old Approach):")
    print("""
    from src.analyst.feature_engineering_orchestrator import FeatureEngineeringOrchestrator
    
    config = {
        'enable_advanced_features': True,
        'enable_autoencoder_features': True,
        'enable_microstructure_features': True
    }
    
    orchestrator = FeatureEngineeringOrchestrator(config)
    features = await orchestrator.generate_all_features(df)
    """)
    
    # AFTER: New approach
    print("AFTER (New Approach):")
    print("""
    from src.feature_generation import (
        FeatureBank,
        RSIGenerator,
        MACDGenerator,
        BollingerBandsGenerator,
        CrossTimeframeInteractionGenerator,
        BaseCalculationType
    )
    
    # Initialize feature bank
    bank = FeatureBank()
    
    # Create generators with different base calculations
    rsi_returns = RSIGenerator(period=14, base_calculation=BaseCalculationType.PRICE_RETURNS)
    rsi_vwap = RSIGenerator(period=14, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20)
    
    macd_levels = MACDGenerator(fast=12, slow=26, signal=9, base_calculation=BaseCalculationType.PRICE_LEVELS)
    macd_vwap = MACDGenerator(fast=12, slow=26, signal=9, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20)
    
    # Generate features
    features = pd.DataFrame(index=df.index)
    features['rsi_returns'] = rsi_returns.generate(df)
    features['rsi_vwap'] = rsi_vwap.generate(df)
    features['macd_levels'] = macd_levels.generate(df)
    features['macd_vwap'] = macd_vwap.generate(df)
    
    # Add interaction features
    cross_timeframe = CrossTimeframeInteractionGenerator(5, 20, "ratio")
    features['cross_timeframe_ratio'] = cross_timeframe.generate(df)
    
    # Store in bank
    bank.add_features("enhanced_features", features)
    """)

def example_update_feature_generators():
    """
    Example of how to update the old FeatureGenerators class
    to use the new unified system.
    """
    print("\n=== Example 2: Updating FeatureGenerators ===")
    
    # BEFORE: Old approach
    print("BEFORE (Old Approach):")
    print("""
    from src.feature_engineering.feature_generators import FeatureGenerators
    
    feature_generators = FeatureGenerators()
    indicators_config = {
        'sma': [5, 10, 20, 50],
        'ema': [5, 10, 20, 50],
        'rsi': [14, 21],
        'macd': [(12, 26, 9)],
        'bb': [(20, 2), (20, 2.5)]
    }
    
    features = feature_generators.batch_technical_indicators(
        data=df,
        indicator_configs=indicators_config,
        use_gpu=True
    )
    """)
    
    # AFTER: New approach
    print("AFTER (New Approach):")
    print("""
    from src.feature_generation import (
        SMAGenerator,
        EMAGenerator,
        RSIGenerator,
        MACDGenerator,
        BollingerBandsGenerator,
        BaseCalculationType
    )
    
    # Create generators with different base calculations
    generators = []
    
    # SMA with different bases
    for period in [5, 10, 20, 50]:
        generators.append(SMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS))
        generators.append(SMAGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20))
    
    # RSI with different bases
    for period in [14, 21]:
        generators.append(RSIGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS))
        generators.append(RSIGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20))
    
    # MACD with different bases
    generators.append(MACDGenerator(fast=12, slow=26, signal=9, base_calculation=BaseCalculationType.PRICE_LEVELS))
    generators.append(MACDGenerator(fast=12, slow=26, signal=9, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20))
    
    # Bollinger Bands with different bases
    generators.append(BollingerBandsGenerator(period=20, std_dev=2.0, base_calculation=BaseCalculationType.PRICE_LEVELS))
    generators.append(BollingerBandsGenerator(period=20, std_dev=2.5, base_calculation=BaseCalculationType.PRICE_LEVELS))
    generators.append(BollingerBandsGenerator(period=20, std_dev=2.0, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20))
    
    # Generate features
    features = pd.DataFrame(index=df.index)
    for generator in generators:
        features[generator.name] = generator.generate(df)
    """)

def example_update_cross_timeframe_features():
    """
    Example of how to update cross-timeframe feature generation
    to use the new interaction features.
    """
    print("\n=== Example 3: Updating Cross-Timeframe Features ===")
    
    # BEFORE: Old approach
    print("BEFORE (Old Approach):")
    print("""
    from src.feature_engineering.cross_timeframe_interaction_features import (
        CrossTimeframeFeatureGenerator,
        CrossTimeframeConfig
    )
    
    config = CrossTimeframeConfig(
        momentum_timeframes=[1, 3, 5, 10, 15, 20],
        volatility_timeframes=[3, 5, 10, 15, 20, 30],
        volume_timeframes=[5, 10, 15, 30],
        rsi_periods=[3, 5, 10, 14, 21],
        macd_fast_periods=[3, 5, 8, 12],
        macd_slow_periods=[10, 15, 20, 26],
        bb_windows=[10, 15, 20],
        bb_stds=[1.0, 1.5, 2.0]
    )
    
    generator = CrossTimeframeFeatureGenerator(config)
    features = generator.generate_features(df)
    """)
    
    # AFTER: New approach
    print("AFTER (New Approach):")
    print("""
    from src.feature_generation import (
        CrossTimeframeInteractionGenerator,
        FeatureRatioGenerator,
        PolynomialFeatureGenerator,
        CorrelationInteractionGenerator,
        create_interaction_generators,
        RSIGenerator,
        MACDGenerator,
        BollingerBandsGenerator,
        BaseCalculationType
    )
    
    # Create interaction generators
    interaction_generators = create_interaction_generators({
        'cross_timeframe': {
            'short_periods': [1, 3, 5, 10, 15],
            'long_periods': [20, 30],
            'interaction_types': ['ratio', 'difference', 'product']
        },
        'feature_ratios': {
            'periods': [(5, 20), (10, 30)],
            'feature_types': ['sma', 'ema', 'volatility']
        },
        'polynomial': {
            'periods': [10, 20],
            'degrees': [2, 3],
            'feature_types': ['returns', 'volatility']
        },
        'correlation': {
            'combinations': [
                (5, 20, 'returns', 'volume'),
                (10, 30, 'volatility', 'returns')
            ]
        }
    })
    
    # Create individual feature generators with different bases
    feature_generators = []
    
    # RSI with different bases and periods
    for period in [3, 5, 10, 14, 21]:
        feature_generators.append(RSIGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS))
        feature_generators.append(RSIGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20))
    
    # MACD with different bases
    for fast in [3, 5, 8, 12]:
        for slow in [10, 15, 20, 26]:
            feature_generators.append(MACDGenerator(fast=fast, slow=slow, signal=9, base_calculation=BaseCalculationType.PRICE_LEVELS))
            feature_generators.append(MACDGenerator(fast=fast, slow=slow, signal=9, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20))
    
    # Bollinger Bands with different bases
    for window in [10, 15, 20]:
        for std in [1.0, 1.5, 2.0]:
            feature_generators.append(BollingerBandsGenerator(period=window, std_dev=std, base_calculation=BaseCalculationType.PRICE_LEVELS))
            feature_generators.append(BollingerBandsGenerator(period=window, std_dev=std, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20))
    
    # Generate all features
    features = pd.DataFrame(index=df.index)
    
    # Add interaction features
    for generator in interaction_generators:
        features[generator.name] = generator.generate(df)
    
    # Add individual feature features
    for generator in feature_generators:
        features[generator.name] = generator.generate(df)
    """)

def example_update_matrix_operations():
    """
    Example of how to update matrix operations usage
    to use the new integrated system.
    """
    print("\n=== Example 4: Updating Matrix Operations ===")
    
    # BEFORE: Old approach
    print("BEFORE (Old Approach):")
    print("""
    from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations
    
    matrix_ops = UnifiedMatrixOperations(
        enable_gpu=True,
        enable_memory_optimization=True,
        enable_parallel=True
    )
    
    # Manual matrix operations
    result = matrix_ops.matrix_multiply(A, B)
    correlation_matrix = matrix_ops.safe_correlation_matrix(data)
    """)
    
    # AFTER: New approach
    print("AFTER (New Approach):")
    print("""
    from src.feature_generation import (
        RSIGenerator,
        MACDGenerator,
        BollingerBandsGenerator,
        BaseCalculationType
    )
    
    # Matrix operations are automatically integrated into feature generators
    # No need to manually handle matrix operations
    
    # Create generators (matrix operations are used internally)
    rsi_generator = RSIGenerator(
        period=14,
        base_calculation=BaseCalculationType.PRICE_RETURNS
    )
    
    macd_generator = MACDGenerator(
        fast=12,
        slow=26,
        signal=9,
        base_calculation=BaseCalculationType.RETURNS_VWAP,
        vwap_period=20
    )
    
    # Generate features (matrix operations are automatically optimized)
    features = pd.DataFrame(index=df.index)
    features['rsi'] = rsi_generator.generate(df)
    features['macd'] = macd_generator.generate(df)
    """)

def example_update_optimization():
    """
    Example of how to update lookback optimization
    to use the new unified system.
    """
    print("\n=== Example 5: Updating Lookback Optimization ===")
    
    # BEFORE: Old approach
    print("BEFORE (Old Approach):")
    print("""
    from src.feature_engineering.feature_generation_optimization import (
        FeatureGenerationOptimizer,
        FeatureOptimizationConfig
    )
    
    config = FeatureOptimizationConfig(
        min_lookback=5,
        max_lookback=252,
        optimization_method=OptimizationMethod.CROSS_VALIDATION,
        cv_folds=5
    )
    
    optimizer = FeatureGenerationOptimizer(config)
    
    def rsi_generator(data, lookback):
        return calculate_rsi(data['close'], lookback)
    
    result = await optimizer.optimize_feature_lookback(
        data=df,
        feature_name="rsi",
        target_column="target",
        feature_generator=rsi_generator
    )
    """)
    
    # AFTER: New approach
    print("AFTER (New Approach):")
    print("""
    from src.feature_generation import (
        LookbackOptimizer,
        OptimizationConfig,
        RSIGenerator,
        BaseCalculationType
    )
    
    # Create optimizer
    config = OptimizationConfig(
        min_lookback=5,
        max_lookback=252,
        optimization_method=OptimizationMethod.CROSS_VALIDATION,
        cv_folds=5
    )
    
    optimizer = LookbackOptimizer(config)
    
    # Create RSI generator with different base calculations
    rsi_returns_generator = RSIGenerator(
        period=14,
        base_calculation=BaseCalculationType.PRICE_RETURNS
    )
    
    rsi_vwap_generator = RSIGenerator(
        period=14,
        base_calculation=BaseCalculationType.RETURNS_VWAP,
        vwap_period=20
    )
    
    # Optimize with different base calculations
    result_returns = await optimizer.optimize_feature(
        data=df,
        feature_name="rsi_returns",
        target_column="target",
        feature_generator_func=lambda data, lookback: rsi_returns_generator.generate(data)
    )
    
    result_vwap = await optimizer.optimize_feature(
        data=df,
        feature_name="rsi_vwap",
        target_column="target",
        feature_generator_func=lambda data, lookback: rsi_vwap_generator.generate(data)
    )
    """)

def example_complete_migration():
    """
    Example of a complete migration from old to new system.
    """
    print("\n=== Example 6: Complete Migration ===")
    
    # BEFORE: Complete old system
    print("BEFORE (Complete Old System):")
    print("""
    # Old imports
    from src.analyst.feature_engineering_orchestrator import FeatureEngineeringOrchestrator
    from src.feature_engineering.feature_generators import FeatureGenerators
    from src.feature_engineering.cross_timeframe_interaction_features import CrossTimeframeFeatureGenerator
    from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations
    
    # Old configuration
    config = {
        'enable_advanced_features': True,
        'enable_autoencoder_features': True,
        'enable_microstructure_features': True,
        'enable_cross_timeframe_features': True
    }
    
    # Old feature generation
    orchestrator = FeatureEngineeringOrchestrator(config)
    features = await orchestrator.generate_all_features(df)
    
    # Old matrix operations
    matrix_ops = UnifiedMatrixOperations()
    correlation_matrix = matrix_ops.safe_correlation_matrix(features)
    
    # Old cross-timeframe features
    cross_timeframe_config = CrossTimeframeConfig(
        momentum_timeframes=[1, 3, 5, 10, 15, 20],
        volatility_timeframes=[3, 5, 10, 15, 20, 30]
    )
    cross_timeframe_generator = CrossTimeframeFeatureGenerator(cross_timeframe_config)
    cross_timeframe_features = cross_timeframe_generator.generate_features(df)
    """)
    
    # AFTER: Complete new system
    print("AFTER (Complete New System):")
    print("""
    # New imports
    from src.feature_generation import (
        FeatureBank,
        RSIGenerator,
        MACDGenerator,
        BollingerBandsGenerator,
        SMAGenerator,
        EMAGenerator,
        CrossTimeframeInteractionGenerator,
        FeatureRatioGenerator,
        PolynomialFeatureGenerator,
        CorrelationInteractionGenerator,
        create_interaction_generators,
        BaseCalculationType,
        LookbackOptimizer,
        OptimizationConfig
    )
    
    # New feature generation with enhanced base calculations
    bank = FeatureBank()
    
    # Create generators with different base calculations
    generators = []
    
    # RSI with different bases
    for period in [14, 21]:
        generators.append(RSIGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS))
        generators.append(RSIGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20))
    
    # MACD with different bases
    generators.append(MACDGenerator(fast=12, slow=26, signal=9, base_calculation=BaseCalculationType.PRICE_LEVELS))
    generators.append(MACDGenerator(fast=12, slow=26, signal=9, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20))
    
    # Bollinger Bands with different bases
    generators.append(BollingerBandsGenerator(period=20, std_dev=2.0, base_calculation=BaseCalculationType.PRICE_LEVELS))
    generators.append(BollingerBandsGenerator(period=20, std_dev=2.0, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20))
    
    # Create interaction generators
    interaction_generators = create_interaction_generators({
        'cross_timeframe': {
            'short_periods': [1, 3, 5, 10, 15],
            'long_periods': [20, 30],
            'interaction_types': ['ratio', 'difference', 'product']
        },
        'feature_ratios': {
            'periods': [(5, 20), (10, 30)],
            'feature_types': ['sma', 'ema', 'volatility']
        },
        'polynomial': {
            'periods': [10, 20],
            'degrees': [2, 3],
            'feature_types': ['returns', 'volatility']
        },
        'correlation': {
            'combinations': [
                (5, 20, 'returns', 'volume'),
                (10, 30, 'volatility', 'returns')
            ]
        }
    })
    
    # Generate all features
    features = pd.DataFrame(index=df.index)
    
    # Add individual features
    for generator in generators:
        features[generator.name] = generator.generate(df)
    
    # Add interaction features
    for generator in interaction_generators:
        features[generator.name] = generator.generate(df)
    
    # Store in feature bank
    bank.add_features("enhanced_features", features)
    
    # Matrix operations are automatically integrated
    # No need for manual matrix operations
    
    # Lookback optimization with different base calculations
    optimizer = LookbackOptimizer(OptimizationConfig())
    
    # Optimize RSI with different bases
    rsi_returns_result = await optimizer.optimize_feature(
        data=df,
        feature_name="rsi_returns",
        target_column="target",
        feature_generator_func=lambda data, lookback: RSIGenerator(
            period=lookback,
            base_calculation=BaseCalculationType.PRICE_RETURNS
        ).generate(data)
    )
    
    rsi_vwap_result = await optimizer.optimize_feature(
        data=df,
        feature_name="rsi_vwap",
        target_column="target",
        feature_generator_func=lambda data, lookback: RSIGenerator(
            period=lookback,
            base_calculation=BaseCalculationType.RETURNS_VWAP,
            vwap_period=20
        ).generate(data)
    )
    """)

def run_all_import_examples():
    """Run all import update examples."""
    print("🔄 Import Update Examples")
    print("=" * 60)
    
    example_update_feature_engineering_orchestrator()
    example_update_feature_generators()
    example_update_cross_timeframe_features()
    example_update_matrix_operations()
    example_update_optimization()
    example_complete_migration()
    
    print("\n✅ All import update examples completed!")
    print("📚 Use these examples as a guide for updating your existing code")

if __name__ == "__main__":
    run_all_import_examples()