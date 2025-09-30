"""
Enhanced TAS Regime Detection Integration Demo

This script demonstrates how to use the enhanced TAS regime detection system
with the integrated existing tools for optimization and validation.
"""

import numpy as np
import pandas as pd
import logging
import sys
import os

# Add the parent directory to the path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

from src.training.steps.market_analysis.tas_regime.core.tas_regime_config import TASRegimeConfig
from src.training.steps.market_analysis.tas_regime.core.tas_regime_detector import TASRegimeDetector
from src.utils.tprint import tprint_info, tprint_success, tprint_error, tprint_warning

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_sample_market_data(num_samples: int = 1000) -> pd.DataFrame:
    """Generates sample market data for demonstration."""
    tprint_info(f"Generating {num_samples} samples of market data...")
    np.random.seed(42)
    
    # Generate time series data with regime-like patterns
    timestamps = pd.date_range('2020-01-01', periods=num_samples, freq='15T')
    
    # Create regime-like patterns
    regime_length = num_samples // 8  # 8 regimes
    regimes = np.repeat(np.arange(8), regime_length)[:num_samples]
    
    # Generate OHLCV data with regime-specific characteristics
    data = []
    for i in range(num_samples):
        regime = regimes[i]
        
        # Regime-specific parameters
        if regime == 0:  # Bull market
            trend = 0.001
            volatility = 0.02
        elif regime == 1:  # Bear market
            trend = -0.001
            volatility = 0.03
        elif regime == 2:  # Sideways market
            trend = 0.0001
            volatility = 0.015
        elif regime == 3:  # High volatility
            trend = 0.0005
            volatility = 0.04
        elif regime == 4:  # Low volatility
            trend = 0.0002
            volatility = 0.01
        elif regime == 5:  # Trending up
            trend = 0.002
            volatility = 0.025
        elif regime == 6:  # Trending down
            trend = -0.002
            volatility = 0.025
        else:  # Random walk
            trend = 0.0001
            volatility = 0.02
        
        # Generate price data
        if i == 0:
            price = 100.0
        else:
            price = data[-1]['close'] * (1 + trend + np.random.normal(0, volatility))
        
        # Generate OHLCV
        high = price * (1 + abs(np.random.normal(0, volatility/2)))
        low = price * (1 - abs(np.random.normal(0, volatility/2)))
        open_price = data[-1]['close'] if i > 0 else price
        close = price
        volume = np.random.exponential(1000)
        
        data.append({
            'timestamp': timestamps[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add technical indicators
    df['returns'] = df['close'].pct_change()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['bb_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
    df['bb_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
    
    # Fill NaN values
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    tprint_success(f"Sample market data generated with shape: {df.shape}")
    return df

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def demonstrate_enhanced_tas_integration():
    """Demonstrate the enhanced TAS regime detection with integrated tools."""
    tprint_info("🚀 Starting Enhanced TAS Integration Demo")
    
    try:
        # Step 1: Generate sample data
        tprint_info("📊 Step 1: Generating sample market data...")
        market_data = generate_sample_market_data(num_samples=2000)
        timestamps = market_data['timestamp'].values
        features = market_data.drop(columns=['timestamp']).values
        
        # Step 2: Create enhanced TAS configuration
        tprint_info("⚙️ Step 2: Creating enhanced TAS configuration...")
        config = TASRegimeConfig.create_research_config()
        config.n_regimes = 8
        config.tree_depth = 8
        config.n_estimators = 500
        config.enable_hardware_optimization = True
        config.enable_matrix_optimization = True
        config.enable_memory_optimization = True
        
        # Add new optimization and validation parameters
        config.hpo_max_evals = 20  # For demo, keep HPO evals low
        config.cv_folds = 5
        config.cv_test_size = 0.2
        config.oos_test_size = 0.3
        config.oos_walk_forward = True
        config.oos_step_size = 0.1
        config.lookahead_prevention = True
        config.persistence_window = 50
        config.persistence_threshold = 0.7
        config.min_persistence_periods = 10
        config.significance_level = 0.05
        config.bootstrap_iterations = 1000
        
        try:
            config.validate_config()
            tprint_success("Configuration validated successfully.")
        except ValueError as e:
            tprint_error(f"Configuration validation failed: {e}")
            return
        
        # Step 3: Initialize the enhanced TAS Regime Detector
        tprint_info("🔧 Step 3: Initializing enhanced TAS detector...")
        detector = TASRegimeDetector(config=config)
        
        # Step 4: Demonstrate hyperparameter optimization
        tprint_info("🔬 Step 4: Demonstrating hyperparameter optimization...")
        try:
            optimization_result = detector.optimize_hyperparameters(
                market_data=features,
                timestamps=timestamps
            )
            
            if optimization_result.get('success', False):
                tprint_success("✅ Hyperparameter optimization completed!")
                tprint_info(f"Best parameters: {optimization_result.get('best_params', {})}")
                
                # Update config with optimized parameters
                best_params = optimization_result.get('best_params', {})
                if best_params:
                    config.n_regimes = best_params.get('n_regimes', config.n_regimes)
                    config.tree_depth = best_params.get('tree_depth', config.tree_depth)
                    config.n_estimators = best_params.get('n_estimators', config.n_estimators)
                    config.min_samples_split = best_params.get('min_samples_split', config.min_samples_split)
                    config.min_samples_leaf = best_params.get('min_samples_leaf', config.min_samples_leaf)
                    config.max_features = best_params.get('max_features', config.max_features)
                    
                    # Reinitialize detector with optimized parameters
                    detector = TASRegimeDetector(config=config)
                    tprint_info("🔄 Reinitialized detector with optimized parameters")
            else:
                tprint_warning("⚠️ Hyperparameter optimization failed or not available")
                
        except Exception as e:
            tprint_warning(f"⚠️ Hyperparameter optimization not available: {e}")
        
        # Step 5: Detect regimes with all enhancements enabled
        tprint_info("🎯 Step 5: Running regime detection with all enhancements...")
        try:
            result = detector.detect_regimes(
                market_data=features,
                timestamps=timestamps,
                optimize_performance=config.enable_hardware_optimization,
                enable_patchtst_enhancement=config.enable_patchtst_enhancement,
                enable_memory_optimization=config.enable_memory_optimization,
                enable_parallel_processing=config.enable_parallel_processing,
                enable_intelligent_caching=config.enable_intelligent_caching,
                enable_cross_validation=config.enable_cross_validation,
                enable_out_of_sample_validation=config.enable_out_of_sample_validation,
                enable_regime_persistence_analysis=config.enable_regime_persistence_analysis
            )
            
            if result.success:
                tprint_success("🎉 Enhanced TAS Regime Detection Completed Successfully!")
                tprint_info(f"Detected {result.regime_count} regimes.")
                tprint_info(f"Execution Time: {result.execution_time:.2f} seconds")
                tprint_info(f"Mean Economic Significance: {np.mean(result.economic_significance_scores):.3f}")
                tprint_info(f"Mean Trading Viability: {np.mean(result.trading_viability_scores):.3f}")
                tprint_info(f"Mean Regime Stability: {np.mean(result.regime_stability_scores):.3f}")
                
                # Display validation results if available
                if hasattr(result, 'cv_scores') and result.cv_scores:
                    tprint_info("Cross-validation results:")
                    for metric, value in result.cv_scores.items():
                        if isinstance(value, float):
                            tprint_info(f"  {metric}: {value:.3f}")
                
                if hasattr(result, 'oos_metrics') and result.oos_metrics:
                    tprint_info("Out-of-sample results:")
                    for metric, value in result.oos_metrics.items():
                        if isinstance(value, float):
                            tprint_info(f"  {metric}: {value:.3f}")
                
                if hasattr(result, 'persistence_analysis') and result.persistence_analysis:
                    tprint_info("Regime persistence analysis:")
                    if 'regime_durations' in result.persistence_analysis:
                        durations = result.persistence_analysis['regime_durations']
                        tprint_info(f"  Mean regime duration: {np.mean(durations):.1f} periods")
                        tprint_info(f"  Max regime duration: {np.max(durations)} periods")
                        tprint_info(f"  Min regime duration: {np.min(durations)} periods")
                
                # Demonstrate caching by running again
                tprint_info("🔄 Running regime detection again to demonstrate caching...")
                cached_result = detector.detect_regimes(
                    market_data=features,
                    timestamps=timestamps,
                    optimize_performance=config.enable_hardware_optimization,
                    enable_patchtst_enhancement=config.enable_patchtst_enhancement,
                    enable_memory_optimization=config.enable_memory_optimization,
                    enable_parallel_processing=config.enable_parallel_processing,
                    enable_intelligent_caching=config.enable_intelligent_caching,
                    enable_cross_validation=config.enable_cross_validation,
                    enable_out_of_sample_validation=config.enable_out_of_sample_validation,
                    enable_regime_persistence_analysis=config.enable_regime_persistence_analysis
                )
                
                if cached_result.success:
                    tprint_success(f"✅ Cached run completed in {cached_result.execution_time:.2f} seconds")
                    if hasattr(detector, 'cache_hits') and hasattr(detector, 'cache_misses'):
                        total_cache_requests = detector.cache_hits + detector.cache_misses
                        cache_hit_rate = detector.cache_hits / total_cache_requests if total_cache_requests > 0 else 0
                        tprint_info(f"Cache hit rate: {cache_hit_rate:.1%}")
                else:
                    tprint_error(f"❌ Cached run failed: {cached_result.error_message}")
                
            else:
                tprint_error(f"❌ Enhanced TAS Regime Detection Failed: {result.error_message}")
        
        except Exception as e:
            tprint_error(f"An unexpected error occurred during regime detection: {e}")
        
        tprint_info("Enhanced TAS Integration Demo completed successfully!")
        
    except Exception as e:
        tprint_error(f"Demo failed: {e}")
        raise

def main():
    """Main demonstration function."""
    try:
        demonstrate_enhanced_tas_integration()
    except Exception as e:
        tprint_error(f"Demo execution failed: {e}")
        raise

if __name__ == "__main__":
    main()