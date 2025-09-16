#!/usr/bin/env python3
"""
Triple Barrier Labeling Example Usage

This script demonstrates how to use the triple barrier labeling system for market analysis.
It shows various configurations, methods, and quality assessment techniques.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import triple barrier labeling components
from market_analysis.triple_barrier_labeling import (
    TripleBarrierLabeler, TripleBarrierConfig, LabelingMethod,
    RegimeAwareLabeler, RegimeAwareConfig,
    LabelQualityAssessment, QualityThresholds,
    LabelCrossValidator, CVConfig, CVMethod,
    MarketAnalysisUtils
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample market data for demonstration."""
    logger.info(f"📊 Creating sample market data with {n_samples} samples")
    
    # Generate synthetic price data
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_samples)  # 2% daily volatility
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLC data
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Ensure high >= low and high/low >= open/close
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    # Create datetime index
    start_date = datetime.now() - timedelta(days=n_samples)
    data.index = pd.date_range(start=start_date, periods=n_samples, freq='1H')
    
    logger.info(f"✅ Created sample data: {data.shape}")
    return data

def example_basic_triple_barrier():
    """Example of basic triple barrier labeling."""
    logger.info("🚀 Example 1: Basic Triple Barrier Labeling")
    
    # Create sample data
    data = create_sample_data(500)
    
    # Configure triple barrier
    config = TripleBarrierConfig(
        pt_mult=0.02,  # 2% profit target
        sl_mult=0.01,  # 1% stop loss
        min_holding_period=1,
        max_holding_period=50,
        transaction_cost=0.001
    )
    
    # Create labeler
    labeler = TripleBarrierLabeler(config)
    
    # Generate labels
    result = labeler.create_labels(
        data=data,
        method=LabelingMethod.TRIPLE_BARRIER
    )
    
    # Display results
    logger.info(f"✅ Generated {len(result.labels)} labels")
    logger.info(f"📊 Label distribution: {result.labels['label'].value_counts().to_dict()}")
    logger.info(f"💰 Profit range: {result.labels['profit_pct'].min():.4f} - {result.labels['profit_pct'].max():.4f}")
    logger.info(f"⏱️ Processing time: {result.processing_time:.3f}s")
    
    return result

def example_regime_aware_labeling():
    """Example of regime-aware triple barrier labeling."""
    logger.info("🚀 Example 2: Regime-Aware Triple Barrier Labeling")
    
    # Create sample data
    data = create_sample_data(1000)
    
    # Create regime data (simplified)
    regimes = []
    for i in range(len(data)):
        if i < 300:
            regimes.append('bull_market')
        elif i < 600:
            regimes.append('bear_market')
        else:
            regimes.append('sideways')
    
    regime_data = pd.DataFrame({'regime': regimes}, index=data.index)
    
    # Configure regime-aware labeling
    regime_config = RegimeAwareConfig(
        regime_detection_method="custom",
        regime_params={
            'bull_market': TripleBarrierConfig(pt_mult=0.03, sl_mult=0.015),
            'bear_market': TripleBarrierConfig(pt_mult=0.015, sl_mult=0.02),
            'sideways': TripleBarrierConfig(pt_mult=0.02, sl_mult=0.02)
        }
    )
    
    # Create regime-aware labeler
    regime_labeler = RegimeAwareLabeler(regime_config)
    
    # Generate regime-aware labels
    labels_df = regime_labeler.create_regime_aware_labels(
        data=data,
        regime_data=regime_data
    )
    
    # Display results
    logger.info(f"✅ Generated {len(labels_df)} regime-aware labels")
    logger.info(f"📊 Label distribution: {labels_df['label'].value_counts().to_dict()}")
    logger.info(f"📈 Regime distribution: {labels_df['regime'].value_counts().to_dict()}")
    
    return labels_df

def example_quality_assessment():
    """Example of label quality assessment."""
    logger.info("🚀 Example 3: Label Quality Assessment")
    
    # Generate labels first
    result = example_basic_triple_barrier()
    
    # Create quality assessor
    quality_assessor = LabelQualityAssessment()
    
    # Assess label quality
    quality_result = quality_assessor.assess_quality(
        labels_df=result.labels,
        original_data=create_sample_data(500)
    )
    
    # Display results
    logger.info(f"🎯 Overall quality: {quality_result.overall_quality:.3f} ({quality_result.quality_level.value})")
    logger.info(f"📊 Individual metrics:")
    for metric, score in quality_result.metric_scores.items():
        logger.info(f"   {metric}: {score:.3f}")
    
    if quality_result.warnings:
        logger.info("⚠️ Warnings:")
        for warning in quality_result.warnings:
            logger.info(f"   {warning}")
    
    if quality_result.recommendations:
        logger.info("💡 Recommendations:")
        for i, rec in enumerate(quality_result.recommendations, 1):
            logger.info(f"   {i}. {rec}")
    
    return quality_result

def example_cross_validation():
    """Example of cross-validation for labels."""
    logger.info("🚀 Example 4: Cross-Validation for Labels")
    
    # Generate labels and prepare features
    result = example_basic_triple_barrier()
    data = create_sample_data(500)
    
    # Prepare features
    feature_cols = ['open', 'high', 'low', 'close', 'volume']
    X = data[feature_cols]
    y = result.labels['label']
    
    # Configure cross-validation
    cv_config = CVConfig(
        method=CVMethod.TEMPORAL_CV,
        n_splits=5,
        models=['random_forest', 'logistic_regression']
    )
    
    # Create cross-validator
    cv_validator = LabelCrossValidator(cv_config)
    
    # Perform validation
    cv_result = cv_validator.validate_labels(X, y, result.labels)
    
    # Display results
    logger.info(f"🎯 Best model: {cv_result.best_model} (score: {cv_result.best_score:.3f})")
    logger.info(f"✅ Validation passed: {cv_result.validation_passed}")
    logger.info(f"📊 Mean scores:")
    for metric, score in cv_result.mean_scores.items():
        std_score = cv_result.std_scores.get(metric, 0.0)
        logger.info(f"   {metric}: {score:.3f} ± {std_score:.3f}")
    
    return cv_result

def example_market_analysis_utils():
    """Example of market analysis utilities."""
    logger.info("🚀 Example 5: Market Analysis Utilities")
    
    # Create sample data
    data = create_sample_data(200)
    
    # Initialize utilities
    utils = MarketAnalysisUtils()
    
    # Validate data
    validation = utils.validate_market_data(data)
    logger.info(f"✅ Data validation: {validation['is_valid']}")
    if validation['warnings']:
        for warning in validation['warnings']:
            logger.info(f"⚠️ {warning}")
    
    # Calculate technical indicators
    data_with_indicators = utils.calculate_technical_indicators(data)
    logger.info(f"📊 Added technical indicators: {list(data_with_indicators.columns)}")
    
    # Detect market regimes
    regime_data = utils.detect_market_regimes(data_with_indicators, method="volatility")
    logger.info(f"📈 Detected regimes: {regime_data['regime'].value_counts().to_dict()}")
    
    # Create analysis summary
    summary = utils.create_analysis_summary(data_with_indicators)
    logger.info(f"📋 Analysis summary created with {len(summary)} sections")
    
    return data_with_indicators, regime_data, summary

def example_comprehensive_workflow():
    """Example of comprehensive workflow combining all components."""
    logger.info("🚀 Example 6: Comprehensive Workflow")
    
    # Step 1: Load and prepare data
    data = create_sample_data(1000)
    utils = MarketAnalysisUtils()
    
    # Step 2: Calculate technical indicators
    data_with_indicators = utils.calculate_technical_indicators(data)
    
    # Step 3: Detect market regimes
    regime_data = utils.detect_market_regimes(data_with_indicators, method="volatility")
    
    # Step 4: Create regime-aware labels
    regime_config = RegimeAwareConfig()
    regime_labeler = RegimeAwareLabeler(regime_config)
    labels_df = regime_labeler.create_regime_aware_labels(
        data=data_with_indicators,
        regime_data=regime_data
    )
    
    # Step 5: Assess label quality
    quality_assessor = LabelQualityAssessment()
    quality_result = quality_assessor.assess_quality(
        labels_df=labels_df,
        original_data=data_with_indicators,
        regime_data=regime_data
    )
    
    # Step 6: Cross-validate labels
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd']
    X = data_with_indicators[feature_cols].fillna(0)
    y = labels_df['label']
    
    cv_config = CVConfig(method=CVMethod.TEMPORAL_CV, n_splits=5)
    cv_validator = LabelCrossValidator(cv_config)
    cv_result = cv_validator.validate_labels(X, y, labels_df)
    
    # Step 7: Generate comprehensive report
    logger.info("📋 COMPREHENSIVE ANALYSIS REPORT")
    logger.info("=" * 50)
    logger.info(f"Data samples: {len(data)}")
    logger.info(f"Labels generated: {len(labels_df)}")
    logger.info(f"Overall quality: {quality_result.overall_quality:.3f}")
    logger.info(f"Validation passed: {cv_result.validation_passed}")
    logger.info(f"Best model: {cv_result.best_model}")
    logger.info(f"Best score: {cv_result.best_score:.3f}")
    
    return {
        'data': data_with_indicators,
        'labels': labels_df,
        'regimes': regime_data,
        'quality': quality_result,
        'validation': cv_result
    }

def main():
    """Main function to run all examples."""
    logger.info("🚀 Starting Triple Barrier Labeling Examples")
    logger.info("=" * 60)
    
    try:
        # Run examples
        example_basic_triple_barrier()
        logger.info("")
        
        example_regime_aware_labeling()
        logger.info("")
        
        example_quality_assessment()
        logger.info("")
        
        example_cross_validation()
        logger.info("")
        
        example_market_analysis_utils()
        logger.info("")
        
        example_comprehensive_workflow()
        logger.info("")
        
        logger.info("✅ All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        raise

if __name__ == "__main__":
    main()