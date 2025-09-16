#!/usr/bin/env python3
"""
Enhanced HMM Clustering Usage Example

This script demonstrates how to use the enhanced HMM clustering system
with common utilities integration in the existing market analysis pipeline.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Import the enhanced HMM utilities
from hmm_executor import (
    create_enhanced_dependencies, 
    train_hmm_optimized,
    validate_hmm_model,
    calculate_regime_characteristics,
    calculate_feature_importance,
    save_hmm_model,
    load_hmm_model
)
from hmm_utils import (
    create_enhanced_hmm_utils,
    TechnicalIndicators
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample market data for demonstration."""
    np.random.seed(42)
    
    # Generate synthetic price data with multiple regimes
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
    
    # Create different market regimes
    regime_lengths = [200, 300, 250, 250]
    regimes = []
    
    for i, length in enumerate(regime_lengths):
        if i == 0:  # Bull market
            trend = 0.001
            volatility = 0.02
        elif i == 1:  # Bear market
            trend = -0.0005
            volatility = 0.03
        elif i == 2:  # Sideways market
            trend = 0.0001
            volatility = 0.015
        else:  # High volatility
            trend = 0.0002
            volatility = 0.04
        
        regime_data = np.random.normal(trend, volatility, length)
        regimes.extend(regime_data)
    
    regimes = regimes[:n_samples]
    
    # Generate price series
    prices = [100]
    for return_val in regimes:
        prices.append(prices[-1] * (1 + return_val))
    
    prices = prices[:n_samples]
    
    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    
    # Ensure OHLCV consistency
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    return data

def demonstrate_enhanced_hmm_clustering():
    """Demonstrate the enhanced HMM clustering with common utilities integration."""
    logger.info("Starting Enhanced HMM Clustering Demonstration")
    logger.info("=" * 60)
    
    try:
        # Create sample data
        logger.info("Creating sample market data...")
        data = create_sample_data(1000)
        logger.info(f"Created {len(data)} data points")
        
        # Initialize enhanced HMM utilities
        logger.info("Initializing enhanced HMM utilities...")
        hmm_utils = create_enhanced_hmm_utils()
        
        # Engineer features using enhanced utilities
        logger.info("Engineering features with enhanced utilities...")
        features = hmm_utils.engineer_features_enhanced(
            data=data,
            lookback_windows=[5, 10, 20, 50],
            technical_indicators=["rsi", "macd", "bollinger_bands", "atr"]
        )
        logger.info(f"Engineered {len(features.columns)} features")
        
        # Create enhanced dependencies
        logger.info("Creating enhanced dependencies...")
        deps = create_enhanced_dependencies(
            use_gpu=True,
            use_memory_optimization=True,
            use_cpu_optimization=True
        )
        
        # Train HMM model with enhanced utilities
        logger.info("Training HMM model with enhanced utilities...")
        hmm_result = train_hmm_optimized(
            features=features,
            n_components=4,
            covariance_type="full",
            n_iter=100,
            random_state=42,
            deps=deps
        )
        
        logger.info("✓ HMM model training completed")
        logger.info(f"  - Processing time: {hmm_result.get('processing_time', 0):.2f}s")
        logger.info(f"  - Used GPU: {hmm_result.get('used_gpu', False)}")
        logger.info(f"  - Model score: {hmm_result.get('score', 0):.4f}")
        
        # Validate model
        logger.info("Validating HMM model...")
        validation_result = validate_hmm_model(
            hmm_model=hmm_result['model'],
            features=features.values,
            n_components=4,
            logger=deps.logger
        )
        
        logger.info("✓ Model validation completed")
        logger.info(f"  - Converged: {validation_result['converged']}")
        logger.info(f"  - Issues: {len(validation_result['issues'])}")
        logger.info(f"  - Recommendations: {len(validation_result['recommendations'])}")
        
        if validation_result['quality_metrics']:
            logger.info("Quality metrics:")
            for metric, value in validation_result['quality_metrics'].items():
                logger.info(f"  - {metric}: {value:.4f}")
        
        # Calculate regime characteristics
        logger.info("Calculating regime characteristics...")
        regime_characteristics = calculate_regime_characteristics(
            features=features,
            state_sequence=hmm_result['state_sequence'],
            state_probs=hmm_result['state_probs'],
            logger=deps.logger
        )
        
        logger.info("✓ Regime characteristics calculated")
        for regime, char in regime_characteristics.items():
            logger.info(f"  - {regime}: {char['count']} samples ({char['percentage']:.1f}%)")
        
        # Calculate feature importance
        logger.info("Calculating feature importance...")
        feature_importance = calculate_feature_importance(
            features=features,
            state_sequence=hmm_result['state_sequence'],
            logger=deps.logger
        )
        
        logger.info("✓ Feature importance calculated")
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info("Top 5 most important features:")
            for feature, importance in top_features:
                logger.info(f"  - {feature}: {importance:.4f}")
        
        # Calculate enhanced regime metrics
        logger.info("Calculating enhanced regime metrics...")
        regime_metrics = hmm_utils.calculate_regime_metrics_enhanced(
            features=features,
            state_sequence=hmm_result['state_sequence'],
            state_probs=hmm_result['state_probs']
        )
        
        logger.info("✓ Enhanced regime metrics calculated")
        for metric, value in regime_metrics.items():
            logger.info(f"  - {metric}: {value:.4f}")
        
        # Save results
        logger.info("Saving results...")
        results = {
            'model': hmm_result['model'],
            'scaler': hmm_result['scaler'],
            'state_sequence': hmm_result['state_sequence'],
            'state_probs': hmm_result['state_probs'],
            'regime_characteristics': regime_characteristics,
            'feature_importance': feature_importance,
            'regime_metrics': regime_metrics,
            'validation_result': validation_result,
            'processing_time': hmm_result.get('processing_time', 0),
            'memory_usage': hmm_result.get('memory_usage', {})
        }
        
        # Save using enhanced utilities
        save_success = hmm_utils.save_results_enhanced(
            results=results,
            filepath="enhanced_hmm_results.json"
        )
        
        if save_success:
            logger.info("✓ Results saved successfully")
        else:
            logger.warning("✗ Failed to save results")
        
        # Test model loading
        logger.info("Testing model loading...")
        loaded_results = hmm_utils.load_results_enhanced("enhanced_hmm_results.json")
        
        if loaded_results:
            logger.info("✓ Model loaded successfully")
            logger.info(f"  - Loaded {len(loaded_results)} result components")
        else:
            logger.warning("✗ Failed to load model")
        
        logger.info("=" * 60)
        logger.info("Enhanced HMM Clustering Demonstration Completed Successfully!")
        
        return results
        
    except Exception as e:
        logger.error(f"Enhanced HMM clustering demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def demonstrate_technical_indicators():
    """Demonstrate enhanced technical indicators with common utilities."""
    logger.info("\nDemonstrating Enhanced Technical Indicators")
    logger.info("=" * 40)
    
    try:
        # Create sample price data
        data = create_sample_data(500)
        prices = data['close']
        
        # Test RSI calculation
        logger.info("Testing RSI calculation...")
        rsi = TechnicalIndicators.calculate_rsi(prices, window=14)
        logger.info(f"✓ RSI calculated: {len(rsi)} values, range: {rsi.min():.2f} - {rsi.max():.2f}")
        
        # Test MACD calculation
        logger.info("Testing MACD calculation...")
        macd_line, macd_signal, macd_hist = TechnicalIndicators.calculate_macd(prices)
        logger.info(f"✓ MACD calculated: {len(macd_line)} values")
        
        # Test Bollinger Bands calculation
        logger.info("Testing Bollinger Bands calculation...")
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(prices)
        logger.info(f"✓ Bollinger Bands calculated: {len(bb_upper)} values")
        
        # Test ATR calculation
        logger.info("Testing ATR calculation...")
        atr = TechnicalIndicators.calculate_atr(data)
        logger.info(f"✓ ATR calculated: {len(atr)} values")
        
        logger.info("✓ All technical indicators working correctly")
        
    except Exception as e:
        logger.error(f"Technical indicators demonstration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run demonstrations
    logger.info("Enhanced HMM Clustering with Common Utilities Integration")
    logger.info("=" * 70)
    
    # Demonstrate enhanced HMM clustering
    results = demonstrate_enhanced_hmm_clustering()
    
    # Demonstrate technical indicators
    demonstrate_technical_indicators()
    
    if results:
        print("\n✅ All demonstrations completed successfully!")
        print("The enhanced HMM clustering system is working correctly with common utilities integration.")
    else:
        print("\n❌ Some demonstrations failed. Please check the implementation.")