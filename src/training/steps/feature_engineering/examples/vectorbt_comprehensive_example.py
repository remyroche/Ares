"""
VectorBT Comprehensive Feature Engineering Example

This script demonstrates the complete VectorBT feature engineering pipeline
including feature generation, optimization, validation, and filtering.

Usage:
    python vectorbt_comprehensive_example.py
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import VectorBT components
from src.training.steps.feature_engineering.vectorbt_base import VectorBTConfig
from src.training.steps.feature_engineering.vectorbt_indicators_suite import VectorBTIndicatorSuite
from src.training.steps.feature_engineering.vectorbt_feature_registration import create_vectorbt_feature_registry
from src.training.steps.feature_engineering.vectorbt_optimization import create_vectorbt_optimizer, OptimizationMetric
from src.training.steps.feature_engineering.vectorbt_validation import create_vectorbt_validator

# Import VectorBT feature generators
from src.training.steps.feature_engineering.volatility.vectorbt_atr_volatility_ratio import VectorBTATRVolatilityRatioGenerator
from src.training.steps.feature_engineering.trend.vectorbt_trend_coherence import VectorBTTrendCoherenceGenerator
from src.training.steps.feature_engineering.price_action.vectorbt_bar_efficiency_ratio import VectorBTBarEfficiencyRatioGenerator
from src.training.steps.feature_engineering.price_action.vectorbt_close_location_value import VectorBTCloseLocationValueGenerator

# Import advanced filters
from src.training.steps.feature_engineering.filters.vectorbt_advanced_filters_15m import apply_vectorbt_advanced_filters_15m

from src.utils.tprint import tprint_info, tprint_success, tprint_error


def generate_sample_data(n_samples: int = 5000) -> pd.DataFrame:
    """Generate sample OHLCV data for demonstration."""
    tprint_info("📊 Generating sample market data")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate price data using geometric Brownian motion
    dt = 1/252  # Daily time step
    mu = 0.05   # Annual drift
    sigma = 0.2  # Annual volatility
    
    # Generate random walks
    random_walks = np.random.normal(0, 1, n_samples)
    price_changes = mu * dt + sigma * np.sqrt(dt) * random_walks
    
    # Calculate prices
    prices = 100 * np.exp(np.cumsum(price_changes))
    
    # Generate OHLCV data
    data = pd.DataFrame(index=pd.date_range('2020-01-01', periods=n_samples, freq='15min'))
    
    # Close prices
    data['close'] = prices
    
    # Generate high, low, open based on close
    data['high'] = data['close'] * (1 + np.abs(np.random.normal(0, 0.01, n_samples)))
    data['low'] = data['close'] * (1 - np.abs(np.random.normal(0, 0.01, n_samples)))
    data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
    
    # Ensure OHLC relationships are valid
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Generate volume data
    data['volume'] = np.random.lognormal(10, 1, n_samples).astype(int)
    
    tprint_success(f"✅ Generated {len(data)} samples of market data")
    return data


def demonstrate_basic_features(data: pd.DataFrame) -> None:
    """Demonstrate basic VectorBT feature generation."""
    tprint_info("🔧 Demonstrating basic VectorBT features")
    
    # Create VectorBT feature generators
    generators = {
        'ATR Volatility': VectorBTATRVolatilityRatioGenerator(lookback=4),
        'Trend Coherence': VectorBTTrendCoherenceGenerator(lookback=8),
        'Bar Efficiency': VectorBTBarEfficiencyRatioGenerator(lookback=3),
        'Close Location Value': VectorBTCloseLocationValueGenerator(lookback=8)
    }
    
    all_features = {}
    
    for name, generator in generators.items():
        tprint_info(f"  📊 Generating {name} features")
        
        try:
            # Generate features
            features = generator.generate_vectorbt_features(data)
            all_features[name] = features
            
            tprint_success(f"    ✅ Generated {len(features)} features")
            
            # Show feature statistics
            for feature_name, feature_data in features.items():
                if isinstance(feature_data, pd.Series) and pd.api.types.is_numeric_dtype(feature_data):
                    clean_data = feature_data.dropna()
                    if len(clean_data) > 0:
                        print(f"      {feature_name}: mean={clean_data.mean():.4f}, std={clean_data.std():.4f}")
        
        except Exception as e:
            tprint_error(f"    ❌ Error generating {name} features: {e}")
    
    return all_features


def demonstrate_technical_indicators(data: pd.DataFrame) -> None:
    """Demonstrate VectorBT technical indicators suite."""
    tprint_info("📈 Demonstrating VectorBT technical indicators suite")
    
    # Create indicator suite
    indicators = VectorBTIndicatorSuite()
    
    # Get all indicators
    tprint_info("  📊 Generating all technical indicators")
    all_indicators = indicators.get_all_indicators(data)
    
    tprint_success(f"  ✅ Generated {len(all_indicators)} technical indicators")
    
    # Show indicators by category
    categories = {
        'Trend': ['sma', 'ema', 'adx', 'psar', 'ichimoku'],
        'Momentum': ['rsi', 'macd', 'stoch', 'willr', 'cci'],
        'Volatility': ['atr', 'bb', 'kc', 'dc'],
        'Volume': ['vwap', 'obv', 'adl', 'mfi'],
        'Price Action': ['bar_efficiency', 'clv', 'price_position']
    }
    
    for category, patterns in categories.items():
        category_indicators = [name for name in all_indicators.keys() 
                             if any(pattern in name.lower() for pattern in patterns)]
        print(f"    {category}: {len(category_indicators)} indicators")
    
    return all_indicators


def demonstrate_optimization(data: pd.DataFrame) -> None:
    """Demonstrate VectorBT parameter optimization."""
    tprint_info("🔍 Demonstrating VectorBT parameter optimization")
    
    # Create optimizer
    optimizer = create_vectorbt_optimizer()
    
    # Create feature generator
    generator = VectorBTATRVolatilityRatioGenerator(lookback=4)
    
    # Define parameter ranges
    param_ranges = {
        'short_window': [3, 4, 5, 6],
        'long_window': [15, 20, 25, 30],
        'high_ratio_threshold': [1.2, 1.5, 1.8, 2.0]
    }
    
    tprint_info("  🔍 Running parameter optimization")
    
    try:
        # Optimize parameters
        result = optimizer.optimize_feature_parameters(
            generator, data, param_ranges, OptimizationMetric.SHARPE_RATIO
        )
        
        tprint_success(f"  ✅ Optimization completed in {result.optimization_time:.2f}s")
        print(f"    Best parameters: {result.best_parameters}")
        print(f"    Best score: {result.best_score:.4f}")
        print(f"    Iterations: {result.n_iterations}")
        print(f"    Convergence: {result.convergence_achieved}")
        
        # Show performance metrics
        print(f"    Sharpe ratio: {result.sharpe_ratio:.4f}")
        print(f"    Information ratio: {result.information_ratio:.4f}")
        print(f"    Max drawdown: {result.max_drawdown:.4f}")
        print(f"    Volatility: {result.volatility:.4f}")
        
    except Exception as e:
        tprint_error(f"  ❌ Optimization failed: {e}")


def demonstrate_validation(data: pd.DataFrame) -> None:
    """Demonstrate VectorBT feature validation."""
    tprint_info("🔍 Demonstrating VectorBT feature validation")
    
    # Create validator
    validator = create_vectorbt_validator()
    
    # Create feature generator
    generator = VectorBTTrendCoherenceGenerator(lookback=8)
    
    tprint_info("  🔍 Running comprehensive validation")
    
    try:
        # Validate feature
        validation_result = validator.validate_feature(generator, data)
        
        tprint_success(f"  ✅ Validation completed in {validation_result.validation_time:.2f}s")
        print(f"    Validation passed: {validation_result.validation_passed}")
        print(f"    Overall score: {validation_result.overall_score:.4f}")
        print(f"    Quality score: {validation_result.quality_score:.4f}")
        
        # Show detailed results
        print(f"    Statistical tests: {len(validation_result.statistical_tests)}")
        print(f"    Performance metrics: {len(validation_result.performance_metrics)}")
        print(f"    Stability metrics: {len(validation_result.stability_metrics)}")
        print(f"    CV mean: {validation_result.cv_mean:.4f} ± {validation_result.cv_std:.4f}")
        
        # Show recommendations
        if validation_result.recommendations:
            print(f"    Recommendations:")
            for rec in validation_result.recommendations:
                print(f"      - {rec}")
        
    except Exception as e:
        tprint_error(f"  ❌ Validation failed: {e}")


def demonstrate_advanced_filtering(data: pd.DataFrame) -> None:
    """Demonstrate VectorBT advanced filtering."""
    tprint_info("🔍 Demonstrating VectorBT advanced filtering")
    
    tprint_info("  🔍 Running advanced filters")
    
    try:
        # Apply VectorBT-enhanced filters
        filter_result = apply_vectorbt_advanced_filters_15m(data)
        
        tprint_success(f"  ✅ Filtering completed in {filter_result.processing_time:.2f}s")
        print(f"    Total samples: {filter_result.n_total_samples}")
        print(f"    Eligible samples: {filter_result.n_eligible_samples}")
        print(f"    Eligibility ratio: {filter_result.eligibility_ratio:.2%}")
        print(f"    Quality score: {filter_result.overall_quality_score:.4f}")
        print(f"    VectorBT optimization score: {filter_result.vectorbt_optimization_score:.4f}")
        
        # Show individual grades
        if filter_result.individual_grades:
            print(f"    Individual grades:")
            for filter_name, grade in filter_result.individual_grades.items():
                print(f"      {filter_name}: {grade.mean():.4f} ± {grade.std():.4f}")
        
        # Show VectorBT indicators
        if filter_result.vectorbt_indicators:
            print(f"    VectorBT indicators: {len(filter_result.vectorbt_indicators)}")
        
    except Exception as e:
        tprint_error(f"  ❌ Filtering failed: {e}")


def demonstrate_feature_registry(data: pd.DataFrame) -> None:
    """Demonstrate VectorBT feature registry."""
    tprint_info("📚 Demonstrating VectorBT feature registry")
    
    # Create feature registry
    registry = create_vectorbt_feature_registry()
    
    tprint_info("  📊 Registering indicator suite features")
    
    try:
        # Register indicator suite features
        registration_results = registry.register_indicator_suite_features(data)
        
        successful_registrations = sum(registration_results.values())
        total_registrations = len(registration_results)
        
        tprint_success(f"  ✅ Registered {successful_registrations}/{total_registrations} indicator features")
        
        # Get registered features
        features = registry.get_registered_features()
        print(f"    Total registered features: {len(features)}")
        
        # Get features by category
        categories = registry.get_feature_categories()
        for category, feature_list in categories.items():
            print(f"    {category}: {len(feature_list)} features")
        
        # Show some example features
        print(f"    Example features:")
        for i, (name, info) in enumerate(features.items()):
            if i < 5:  # Show first 5 features
                print(f"      {name}: {info.get('description', 'No description')}")
        
    except Exception as e:
        tprint_error(f"  ❌ Feature registry failed: {e}")


def demonstrate_performance_comparison(data: pd.DataFrame) -> None:
    """Demonstrate performance comparison between traditional and VectorBT methods."""
    tprint_info("⚡ Demonstrating performance comparison")
    
    import time
    
    # Traditional pandas approach
    tprint_info("  📊 Testing traditional pandas approach")
    start_time = time.time()
    
    # Simple RSI calculation
    close_prices = data['close']
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    traditional_rsi = 100 - (100 / (1 + rs))
    
    traditional_time = time.time() - start_time
    tprint_success(f"    ✅ Traditional RSI: {traditional_time:.4f}s")
    
    # VectorBT approach
    tprint_info("  📊 Testing VectorBT approach")
    start_time = time.time()
    
    from src.training.steps.feature_engineering.vectorbt_base import VectorBTTechnicalIndicators
    indicators = VectorBTTechnicalIndicators()
    vectorbt_rsi = indicators.vbt.RSI.run(close_prices, window=14).rsi
    
    vectorbt_time = time.time() - start_time
    tprint_success(f"    ✅ VectorBT RSI: {vectorbt_time:.4f}s")
    
    # Performance comparison
    speedup = traditional_time / vectorbt_time
    print(f"    Speedup: {speedup:.2f}x")
    print(f"    Time saved: {traditional_time - vectorbt_time:.4f}s")
    
    # Accuracy comparison
    correlation = traditional_rsi.corr(vectorbt_rsi)
    print(f"    Correlation: {correlation:.6f}")


def main():
    """Main demonstration function."""
    tprint_info("🚀 Starting VectorBT Comprehensive Feature Engineering Demo")
    
    try:
        # Generate sample data
        data = generate_sample_data(2000)  # Use smaller dataset for demo
        
        print(f"\n📊 Data Overview:")
        print(f"  Shape: {data.shape}")
        print(f"  Columns: {list(data.columns)}")
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
        print(f"  Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        
        # Demonstrate basic features
        print(f"\n🔧 Basic VectorBT Features:")
        basic_features = demonstrate_basic_features(data)
        
        # Demonstrate technical indicators
        print(f"\n📈 Technical Indicators Suite:")
        technical_indicators = demonstrate_technical_indicators(data)
        
        # Demonstrate optimization
        print(f"\n🔍 Parameter Optimization:")
        demonstrate_optimization(data)
        
        # Demonstrate validation
        print(f"\n🔍 Feature Validation:")
        demonstrate_validation(data)
        
        # Demonstrate advanced filtering
        print(f"\n🔍 Advanced Filtering:")
        demonstrate_advanced_filtering(data)
        
        # Demonstrate feature registry
        print(f"\n📚 Feature Registry:")
        demonstrate_feature_registry(data)
        
        # Demonstrate performance comparison
        print(f"\n⚡ Performance Comparison:")
        demonstrate_performance_comparison(data)
        
        tprint_success("🎉 VectorBT Comprehensive Demo completed successfully!")
        
        print(f"\n📋 Summary:")
        print(f"  ✅ Basic features: {len(basic_features)} generators")
        print(f"  ✅ Technical indicators: {len(technical_indicators)} indicators")
        print(f"  ✅ Parameter optimization: Completed")
        print(f"  ✅ Feature validation: Completed")
        print(f"  ✅ Advanced filtering: Completed")
        print(f"  ✅ Feature registry: Completed")
        print(f"  ✅ Performance comparison: Completed")
        
    except Exception as e:
        tprint_error(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run demonstration
    main()