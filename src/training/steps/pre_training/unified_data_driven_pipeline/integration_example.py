"""
Integration Example for Enhanced Unified Data-Driven Pipeline

This script demonstrates how to use the enhanced unified pipeline that integrates
all the missing functionality from individual components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
import time
from pathlib import Path

# Import the enhanced pipeline
from .core.enhanced_unified_pipeline import (
    EnhancedUnifiedDataDrivenPipeline,
    EnhancedFeaturePipelineResult,
    create_enhanced_unified_pipeline,
    process_with_enhanced_pipeline
)

# Import configuration
from .core.config import UnifiedPipelineConfig, create_default_config

# Import individual components for comparison
from .core.economic_evaluator import (
    EconomicPeriodEvaluator, EconomicEvaluationConfig,
    create_economic_evaluator
)
from .core.intelligent_feature_selector import (
    IntelligentFeatureSelector, FeatureSelectionConfig,
    create_intelligent_feature_selector
)
from .core.modular_architecture import (
    ModularArchitecture, ValidationLevel, ErrorSeverity, ErrorCategory,
    create_modular_architecture
)
from .core.template_interaction_generator import (
    TemplateInteractionGenerator, TemplateConfig,
    create_template_interaction_generator
)
from .core.vectorbt_optimizer import (
    VectorBTOptimizer, VectorBTConfig,
    create_vectorbt_optimizer
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data(n_samples: int = 1000, n_features: int = 50) -> pd.DataFrame:
    """Create sample data for testing the enhanced pipeline."""
    np.random.seed(42)
    
    # Create time index
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='15T')
    
    # Create OHLCV data
    data = {
        'open': np.random.randn(n_samples).cumsum() + 100,
        'high': np.random.randn(n_samples).cumsum() + 102,
        'low': np.random.randn(n_samples).cumsum() + 98,
        'close': np.random.randn(n_samples).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, n_samples)
    }
    
    # Ensure high >= low and high >= close >= low
    for i in range(n_samples):
        data['high'][i] = max(data['open'][i], data['high'][i], data['close'][i])
        data['low'][i] = min(data['open'][i], data['low'][i], data['close'][i])
    
    # Create additional features
    close_prices = data['close']
    
    # Price-based features
    for period in [5, 10, 20, 50]:
        data[f'sma_{period}'] = close_prices.rolling(period).mean()
        data[f'ema_{period}'] = close_prices.ewm(span=period).mean()
        data[f'volatility_{period}'] = close_prices.rolling(period).std()
        data[f'momentum_{period}'] = close_prices.pct_change(period)
    
    # Volume features
    for period in [10, 20]:
        data[f'volume_sma_{period}'] = data['volume'].rolling(period).mean()
        data[f'volume_ratio_{period}'] = data['volume'] / data[f'volume_sma_{period}']
    
    # Technical indicators
    data['rsi_14'] = calculate_rsi(close_prices, 14)
    data['rsi_21'] = calculate_rsi(close_prices, 21)
    
    # Returns
    data['returns'] = close_prices.pct_change()
    data['log_returns'] = np.log(close_prices / close_prices.shift(1))
    
    # Volatility features
    data['volatility_rolling'] = data['returns'].rolling(20).std()
    data['volatility_ewm'] = data['returns'].ewm(span=20).std()
    
    # Create DataFrame
    df = pd.DataFrame(data, index=dates)
    
    # Add more random features to reach n_features
    remaining_features = n_features - len(df.columns)
    for i in range(remaining_features):
        df[f'feature_{i}'] = np.random.randn(n_samples)
    
    return df


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def create_sample_targets(data: pd.DataFrame, lookahead: int = 1) -> pd.Series:
    """Create sample targets for the pipeline."""
    # Simple target: future returns
    future_returns = data['close'].pct_change(lookahead).shift(-lookahead)
    return future_returns


def demonstrate_enhanced_pipeline():
    """Demonstrate the enhanced unified pipeline."""
    print("🚀 Enhanced Unified Data-Driven Pipeline Demo")
    print("=" * 60)
    
    # Create sample data
    print("📊 Creating sample data...")
    data = create_sample_data(n_samples=1000, n_features=100)
    targets = create_sample_targets(data, lookahead=1)
    
    print(f"✅ Data created: {data.shape[0]} samples, {data.shape[1]} features")
    print(f"✅ Targets created: {len(targets)} values")
    
    # Create enhanced pipeline configuration
    print("\n⚙️ Creating enhanced pipeline configuration...")
    config = create_default_config()
    config.enable_period_optimization = True
    config.enable_feature_lookback_optimization = True
    config.enable_interaction_generation = True
    config.enable_htf_interactions = True
    
    print("✅ Configuration created")
    
    # Initialize enhanced pipeline
    print("\n🔧 Initializing enhanced pipeline...")
    pipeline = create_enhanced_unified_pipeline(config)
    print("✅ Enhanced pipeline initialized")
    
    # Process data through enhanced pipeline
    print("\n🔄 Processing data through enhanced pipeline...")
    start_time = time.time()
    
    try:
        result = pipeline.process(data, targets)
        processing_time = time.time() - start_time
        
        print(f"✅ Pipeline processing completed in {processing_time:.3f}s")
        
        # Display results
        print("\n📈 Results Summary:")
        print(f"  Selected features: {len(result.selected_features)}")
        print(f"  Processing time: {result.processing_time:.3f}s")
        print(f"  CV splits: {result.n_cv_splits}")
        print(f"  Out-of-sample Sharpe: {result.out_of_sample_sharpe:.3f}")
        print(f"  Max drawdown: {result.max_drawdown:.3f}")
        print(f"  Stability score: {result.stability_score:.3f}")
        print(f"  Diversity score: {result.diversity_score:.3f}")
        
        # Display enhanced results
        print("\n🎯 Enhanced Results:")
        
        if result.economic_evaluation_result:
            print(f"  Economic evaluation: {result.economic_evaluation_result.successful_evaluations} successful")
            print(f"  Average Sharpe: {result.economic_evaluation_result.average_sharpe:.3f}")
            print(f"  Average drawdown: {result.economic_evaluation_result.average_drawdown:.3f}")
        
        if result.feature_preselection_result:
            print(f"  Feature pre-selection: {len(result.feature_preselection_result.selected_features)} features")
            print(f"  Categories: {list(result.feature_preselection_result.category_distribution.keys())}")
        
        if result.template_interaction_result:
            print(f"  Template interactions: {result.template_interaction_result['interaction_count']}")
        
        if result.modular_architecture_summary:
            print(f"  Modular architecture: {len(result.modular_architecture_summary)} components")
        
        # Display performance metrics
        print("\n⚡ Performance Metrics:")
        print(f"  VectorBT operations: {result.vectorbt_operations}")
        print(f"  Pandas fallbacks: {result.pandas_fallbacks}")
        print(f"  Cache hit rate: {result.cache_hit_rate:.3f}")
        print(f"  Memory usage: {result.memory_usage_mb:.1f} MB")
        
        # Get detailed performance summary
        performance_summary = pipeline.get_performance_summary()
        print(f"\n📊 Detailed Performance Summary:")
        for component, stats in performance_summary.items():
            print(f"  {component}: {stats}")
        
        return result
        
    except Exception as e:
        print(f"❌ Pipeline processing failed: {e}")
        logger.exception("Pipeline processing failed")
        return None


def demonstrate_individual_components():
    """Demonstrate individual enhanced components."""
    print("\n🔧 Individual Enhanced Components Demo")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data(n_samples=500, n_features=50)
    targets = create_sample_targets(data, lookahead=1)
    
    # 1. Economic Evaluator
    print("\n💰 Economic Evaluator Demo:")
    try:
        economic_config = EconomicEvaluationConfig(
            min_period=5,
            max_period=20,
            backtest_periods=50,
            min_backtest_periods=20
        )
        economic_evaluator = create_economic_evaluator(economic_config)
        
        candidate_periods = [5, 10, 15, 20]
        economic_result = economic_evaluator.evaluate_periods(data, candidate_periods, "15m")
        
        print(f"  Economic evaluation completed: {economic_result.successful_evaluations} successful")
        print(f"  Top periods: {economic_result.top_periods}")
        print(f"  Average Sharpe: {economic_result.average_sharpe:.3f}")
        
    except Exception as e:
        print(f"  ❌ Economic evaluator failed: {e}")
    
    # 2. Intelligent Feature Selector
    print("\n🎯 Intelligent Feature Selector Demo:")
    try:
        feature_config = FeatureSelectionConfig(
            target_feature_count=20,
            min_features_per_category=2,
            max_features_per_category=4
        )
        feature_selector = create_intelligent_feature_selector(feature_config)
        
        feature_result = feature_selector.select_features(data, targets)
        
        print(f"  Feature selection completed: {len(feature_result.selected_features)} features")
        print(f"  Categories: {list(feature_result.category_distribution.keys())}")
        print(f"  Selection time: {feature_result.selection_time:.3f}s")
        
    except Exception as e:
        print(f"  ❌ Feature selector failed: {e}")
    
    # 3. Modular Architecture
    print("\n🏗️ Modular Architecture Demo:")
    try:
        modular_arch = create_modular_architecture("DemoComponent")
        
        # Validate data
        validation_result = modular_arch.validate_inputs(data, ValidationLevel.STANDARD)
        print(f"  Data validation: {'✅ Valid' if validation_result.is_valid else '❌ Invalid'}")
        
        # Get system summary
        summary = modular_arch.get_system_summary()
        print(f"  System components: {len(summary)}")
        
    except Exception as e:
        print(f"  ❌ Modular architecture failed: {e}")
    
    # 4. Template Interaction Generator
    print("\n🎯 Template Interaction Generator Demo:")
    try:
        template_config = TemplateConfig(
            total_budget=20,
            core_budget=10,
            htf_aware_budget=10
        )
        template_generator = create_template_interaction_generator(template_config)
        
        # Create HTF features
        htf_features = {}
        close_prices = data['close']
        for period in [20, 50]:
            htf_features[f'htf_trend_{period}'] = close_prices.rolling(period).mean()
            htf_features[f'htf_vol_{period}'] = close_prices.rolling(period).std()
        
        template_interactions = template_generator.generate_interactions(htf_features, data, targets)
        
        print(f"  Template interactions generated: {len(template_interactions)}")
        print(f"  Interaction types: {set(i.interaction_type for i in template_interactions)}")
        
    except Exception as e:
        print(f"  ❌ Template generator failed: {e}")
    
    # 5. VectorBT Optimizer
    print("\n⚡ VectorBT Optimizer Demo:")
    try:
        vectorbt_config = VectorBTConfig(
            enable_vectorbt=True,
            enable_parallel=True,
            memory_efficient=True
        )
        vectorbt_optimizer = create_vectorbt_optimizer(vectorbt_config)
        
        # Test rolling operation
        rolling_result = vectorbt_optimizer.rolling_operation(
            data['close'], 'mean', window=20
        )
        
        print(f"  Rolling operation completed: {rolling_result.success}")
        print(f"  Execution time: {rolling_result.execution_time:.3f}s")
        print(f"  Optimization method: {rolling_result.optimization_method}")
        
        # Get performance summary
        perf_summary = vectorbt_optimizer.get_performance_summary()
        print(f"  VectorBT operations: {perf_summary['vectorbt_operations']}")
        print(f"  Pandas fallbacks: {perf_summary['pandas_fallbacks']}")
        
    except Exception as e:
        print(f"  ❌ VectorBT optimizer failed: {e}")


def compare_with_original_pipeline():
    """Compare enhanced pipeline with original pipeline."""
    print("\n🔄 Comparison with Original Pipeline")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data(n_samples=500, n_features=30)
    targets = create_sample_targets(data, lookahead=1)
    
    # Test enhanced pipeline
    print("Testing Enhanced Pipeline...")
    try:
        from .core.enhanced_unified_pipeline import create_enhanced_unified_pipeline
        
        enhanced_pipeline = create_enhanced_unified_pipeline()
        enhanced_start = time.time()
        enhanced_result = enhanced_pipeline.process(data, targets)
        enhanced_time = time.time() - enhanced_start
        
        print(f"✅ Enhanced pipeline: {enhanced_time:.3f}s, {len(enhanced_result.selected_features)} features")
        
        # Display enhanced features
        if enhanced_result.economic_evaluation_result:
            print(f"  Economic evaluation: ✅")
        if enhanced_result.feature_preselection_result:
            print(f"  Feature pre-selection: ✅")
        if enhanced_result.template_interaction_result:
            print(f"  Template interactions: ✅")
        if enhanced_result.modular_architecture_summary:
            print(f"  Modular architecture: ✅")
        
    except Exception as e:
        print(f"❌ Enhanced pipeline failed: {e}")
    
    # Test original pipeline (if available)
    print("\nTesting Original Pipeline...")
    try:
        from .core.unified_pipeline import create_unified_pipeline
        
        original_pipeline = create_unified_pipeline()
        original_start = time.time()
        original_result = original_pipeline.process(data, targets)
        original_time = time.time() - original_start
        
        print(f"✅ Original pipeline: {original_time:.3f}s, {len(original_result.selected_features)} features")
        
    except Exception as e:
        print(f"❌ Original pipeline failed: {e}")


def main():
    """Main demonstration function."""
    print("🚀 Enhanced Unified Data-Driven Pipeline Integration Demo")
    print("=" * 80)
    
    try:
        # Demonstrate enhanced pipeline
        result = demonstrate_enhanced_pipeline()
        
        if result:
            print("\n✅ Enhanced pipeline demonstration completed successfully!")
        else:
            print("\n❌ Enhanced pipeline demonstration failed!")
        
        # Demonstrate individual components
        demonstrate_individual_components()
        
        # Compare with original pipeline
        compare_with_original_pipeline()
        
        print("\n🎉 Integration demo completed!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.exception("Demo failed")


if __name__ == "__main__":
    main()