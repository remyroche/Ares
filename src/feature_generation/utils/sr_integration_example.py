"""
SR Feature Integration Example

This module demonstrates how to integrate SR feature extraction with the parameter
optimization engine for maximum performance and accuracy.

Usage:
    python -m src.feature_generation.utils.sr_integration_example
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import time

# Import SR feature extraction components
from .sr_feature_extractor import (
    SRFeatureExtractor, SRFeatureConfig, get_sr_feature_extractor, extract_sr_features
)

# Import parameter optimization engine
try:
    from src.utils.sr_clustering.parameter_optimization_engine import (
        ParameterOptimizationEngine, ParameterOptimizationConfig, 
        get_parameter_optimization_engine
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    print("⚠️ Parameter optimization engine not available")

# Import enhanced feature engineering
from .step06_enhanced_feature_engineering_step import EnhancedFeatureEngineeringStep

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_market_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample market data for testing."""
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.normal(0, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    })
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data

def create_sample_sr_levels(data: pd.DataFrame) -> Dict[str, Any]:
    """Create sample SR levels for testing."""
    # Simple swing high/low detection
    window = 20
    highs = data['high'].rolling(window, center=True).max()
    lows = data['low'].rolling(window, center=True).min()
    
    # Find swing points
    swing_highs = data[data['high'] == highs]['high'].dropna().unique()
    swing_lows = data[data['low'] == lows]['low'].dropna().unique()
    
    # Filter by minimum touches (simplified)
    support_levels = swing_lows[:5].tolist()  # Top 5 support levels
    resistance_levels = swing_highs[:5].tolist()  # Top 5 resistance levels
    
    return {
        'support_levels': support_levels,
        'resistance_levels': resistance_levels,
        'quality_scores': {
            f'level_{level:.6f}': np.random.uniform(0.3, 0.9) 
            for level in support_levels + resistance_levels
        }
    }

def create_sample_regime_labels(data: pd.DataFrame) -> pd.Series:
    """Create sample regime labels for testing."""
    # Simple regime based on volatility
    returns = data['close'].pct_change()
    volatility = returns.rolling(20).std()
    
    # Create 3 regimes based on volatility
    regime_labels = pd.cut(volatility, bins=3, labels=['low_vol', 'med_vol', 'high_vol'])
    return regime_labels

def demonstrate_basic_sr_extraction():
    """Demonstrate basic SR feature extraction."""
    logger.info("🔧 Demonstrating basic SR feature extraction...")
    
    # Create sample data
    data = create_sample_market_data(500)
    sr_levels = create_sample_sr_levels(data)
    regime_labels = create_sample_regime_labels(data)
    
    # Create SR feature configuration
    sr_config = SRFeatureConfig(
        enable_basic_sr_features=True,
        enable_advanced_sr_features=True,
        enable_sr_bounce_signals=True,
        enable_sr_strength_calculation=True,
        enable_regime_aware_sr=True,
        use_pre_optimized_parameters=False  # Start without optimization
    )
    
    # Extract SR features
    start_time = time.time()
    sr_features = extract_sr_features(data, sr_levels, regime_labels, sr_config)
    extraction_time = time.time() - start_time
    
    logger.info(f"✅ Basic SR extraction completed in {extraction_time:.2f}s")
    logger.info(f"   Data shape: {data.shape}")
    logger.info(f"   SR features shape: {sr_features.shape}")
    logger.info(f"   SR feature columns: {list(sr_features.columns)}")
    
    return sr_features

def demonstrate_optimized_sr_extraction():
    """Demonstrate SR feature extraction with optimization."""
    if not OPTIMIZATION_AVAILABLE:
        logger.warning("⚠️ Optimization engine not available, skipping optimized extraction")
        return None
    
    logger.info("🎯 Demonstrating optimized SR feature extraction...")
    
    # Create sample data
    data = create_sample_market_data(1000)
    sr_levels = create_sample_sr_levels(data)
    regime_labels = create_sample_regime_labels(data)
    
    # Create optimized SR feature configuration
    sr_config = SRFeatureConfig(
        enable_basic_sr_features=True,
        enable_advanced_sr_features=True,
        enable_sr_bounce_signals=True,
        enable_sr_strength_calculation=True,
        enable_regime_aware_sr=True,
        use_pre_optimized_parameters=True,
        sr_detection_window=20,
        min_touches_required=3,
        touch_tolerance=0.002,
        min_bounce_strength=0.001,
        volume_threshold_multiplier=1.5
    )
    
    # Get SR feature extractor with optimization
    sr_extractor = get_sr_feature_extractor(sr_config)
    
    # Extract SR features with optimization
    start_time = time.time()
    sr_features = sr_extractor.extract_sr_features(data, sr_levels, regime_labels)
    extraction_time = time.time() - start_time
    
    logger.info(f"✅ Optimized SR extraction completed in {extraction_time:.2f}s")
    logger.info(f"   Data shape: {data.shape}")
    logger.info(f"   SR features shape: {sr_features.shape}")
    logger.info(f"   Optimized parameters: {sr_extractor.get_optimized_parameters()}")
    
    return sr_features

def demonstrate_feature_engineering_integration():
    """Demonstrate integration with enhanced feature engineering."""
    logger.info("🔗 Demonstrating feature engineering integration...")
    
    # Create sample data
    data = create_sample_market_data(1000)
    sr_levels = create_sample_sr_levels(data)
    regime_labels = create_sample_regime_labels(data)
    
    # Add regime labels to data
    data['regime_label'] = regime_labels
    
    # Create feature engineering configuration
    config = {
        'step06_feature_engineering': {
            'use_technical_indicators': True,
            'use_interaction_features': True,
            'use_regime_features': True,
            'use_sr_features': True,
            'use_dynamic_lookback': True,
            'chunk_size': 1000,
            'max_features': 200,
            'polynomial_degree': 2,
            'correlation_threshold': 0.95,
            'memory_limit_mb': 500,
            'sr_detection_window': 20,
            'min_touches_required': 3,
            'touch_tolerance': 0.002,
            'min_bounce_strength': 0.001,
            'volume_threshold_multiplier': 1.5,
            'use_pre_optimized_sr_parameters': True
        }
    }
    
    # Create enhanced feature engineering step
    feature_step = EnhancedFeatureEngineeringStep(config)
    
    # Simulate pipeline state with SR levels
    pipeline_state = {
        'train_data': data,
        'sr_levels': sr_levels
    }
    
    # Process data split
    start_time = time.time()
    processed_data = feature_step._process_data_split(data, 'train')
    processing_time = time.time() - start_time
    
    logger.info(f"✅ Feature engineering integration completed in {processing_time:.2f}s")
    logger.info(f"   Original data shape: {data.shape}")
    logger.info(f"   Processed data shape: {processed_data.shape}")
    
    # Count SR features
    sr_feature_cols = [col for col in processed_data.columns 
                      if any(sr_term in col.lower() for sr_term in 
                            ['support', 'resistance', 'pivot', 'swing', 'sr_', 'bounce'])]
    logger.info(f"   SR features created: {len(sr_feature_cols)}")
    logger.info(f"   SR feature examples: {sr_feature_cols[:5]}")
    
    return processed_data

def demonstrate_parameter_optimization():
    """Demonstrate parameter optimization for SR features."""
    if not OPTIMIZATION_AVAILABLE:
        logger.warning("⚠️ Optimization engine not available, skipping parameter optimization")
        return None
    
    logger.info("⚙️ Demonstrating parameter optimization...")
    
    # Create sample data
    data = create_sample_market_data(2000)
    sr_levels = create_sample_sr_levels(data)
    
    # Create optimization configuration
    opt_config = ParameterOptimizationConfig(
        optimization_method='adaptive_grid_search',
        n_trials=50,  # Reduced for demo
        cv_folds=3,
        objective_metric='quality_score_correlation',
        enable_hardware_optimization=True,
        enable_parallel_processing=True,
        max_parallel_workers=2
    )
    
    # Get optimization engine
    opt_engine = get_parameter_optimization_engine(opt_config)
    
    # Create mock backtest results for optimization
    # In real usage, these would come from actual backtesting
    mock_backtest_results = []
    for i in range(20):  # 20 mock results
        mock_result = type('MockResult', (), {
            'success_rate': np.random.uniform(0.3, 0.8),
            'avg_bounce_strength': np.random.uniform(0.001, 0.01),
            'total_volume_at_level': np.random.uniform(1000, 10000),
            'time_persistence': np.random.uniform(0.1, 0.9),
            'total_touches': np.random.randint(2, 10),
            'quality_score': np.random.uniform(0.2, 0.9)
        })()
        mock_backtest_results.append(mock_result)
    
    # Run optimization
    start_time = time.time()
    try:
        opt_result = opt_engine.optimize_parameters(mock_backtest_results, data)
        optimization_time = time.time() - start_time
        
        logger.info(f"✅ Parameter optimization completed in {optimization_time:.2f}s")
        logger.info(f"   Best score: {opt_result.best_score:.4f}")
        logger.info(f"   Best parameters: {opt_result.best_parameters}")
        logger.info(f"   Optimization method: {opt_result.optimization_method}")
        logger.info(f"   Number of trials: {opt_result.n_trials}")
        
        return opt_result
    except Exception as e:
        logger.error(f"❌ Parameter optimization failed: {e}")
        return None

def save_optimization_results(opt_result, file_path: str = "sr_optimization_results.json"):
    """Save optimization results to file."""
    if opt_result is None:
        return
    
    try:
        results_data = {
            'best_parameters': opt_result.best_parameters,
            'best_score': opt_result.best_score,
            'optimization_method': opt_result.optimization_method,
            'n_trials': opt_result.n_trials,
            'optimization_success': opt_result.optimization_success,
            'optimization_details': opt_result.optimization_details
        }
        
        with open(file_path, 'w') as f:
            import json

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
            json.dump(results_data, f, indent=2)
        
        logger.info(f"💾 Optimization results saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save optimization results: {e}")

def main():
    """Main demonstration function."""
    logger.info("🚀 Starting SR Feature Integration Demonstration")
    logger.info("=" * 60)
    
    try:
        # 1. Basic SR extraction
        logger.info("\n1. Basic SR Feature Extraction")
        logger.info("-" * 40)
        basic_features = demonstrate_basic_sr_extraction()
        
        # 2. Optimized SR extraction
        logger.info("\n2. Optimized SR Feature Extraction")
        logger.info("-" * 40)
        optimized_features = demonstrate_optimized_sr_extraction()
        
        # 3. Feature engineering integration
        logger.info("\n3. Feature Engineering Integration")
        logger.info("-" * 40)
        integrated_features = demonstrate_feature_engineering_integration()
        
        # 4. Parameter optimization
        logger.info("\n4. Parameter Optimization")
        logger.info("-" * 40)
        opt_result = demonstrate_parameter_optimization()
        
        # 5. Save results
        if opt_result:
            save_optimization_results(opt_result)
        
        logger.info("\n✅ All demonstrations completed successfully!")
        logger.info("=" * 60)
        
        # Summary
        logger.info("\n📊 Summary:")
        logger.info(f"   Basic SR features: {basic_features.shape[1] if basic_features is not None else 0}")
        logger.info(f"   Optimized SR features: {optimized_features.shape[1] if optimized_features is not None else 0}")
        logger.info(f"   Integrated features: {integrated_features.shape[1] if integrated_features is not None else 0}")
        logger.info(f"   Optimization completed: {opt_result is not None}")
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        raise

if __name__ == "__main__":
    main()
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
