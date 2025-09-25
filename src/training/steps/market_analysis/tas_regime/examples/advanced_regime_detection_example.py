"""
Advanced Regime Detection and Qualification Example

This example demonstrates the complete implementation of unsupervised regime detection
and regime qualification for trading applications.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Import the implemented systems
from src.utils.nas_tas.unsupervised_regime_detection import (
    UnsupervisedRegimeDetector, RegimeDetectionConfig
)
from src.training.steps.market_analysis.tas_regime.regime_analysis.regime_qualification import (
    RegimeQualifier, RegimeQualificationConfig
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_market_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate sample market data for testing."""
    np.random.seed(42)
    
    # Generate time series
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1H')
    
    # Generate price data with different regimes
    prices = []
    volumes = []
    
    # Regime 1: Low volatility, uptrend
    regime1_length = n_samples // 3
    for i in range(regime1_length):
        price = 100 + i * 0.01 + np.random.normal(0, 0.5)
        volume = 1000 + np.random.normal(0, 100)
        prices.append(price)
        volumes.append(max(volume, 100))
    
    # Regime 2: High volatility, sideways
    regime2_length = n_samples // 3
    for i in range(regime2_length):
        price = prices[-1] + np.random.normal(0, 2.0)
        volume = 1500 + np.random.normal(0, 200)
        prices.append(price)
        volumes.append(max(volume, 100))
    
    # Regime 3: Medium volatility, downtrend
    regime3_length = n_samples - regime1_length - regime2_length
    for i in range(regime3_length):
        price = prices[-1] - 0.005 + np.random.normal(0, 1.0)
        volume = 800 + np.random.normal(0, 150)
        prices.append(price)
        volumes.append(max(volume, 100))
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p + np.random.uniform(0, 1) for p in prices],
        'low': [p - np.random.uniform(0, 1) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    return data


def demonstrate_unsupervised_regime_detection():
    """Demonstrate unsupervised regime detection capabilities."""
    logger.info("🔍 Demonstrating Unsupervised Regime Detection")
    
    # Generate sample data
    market_data = generate_sample_market_data(1000)
    logger.info(f"📊 Generated {len(market_data)} samples of market data")
    
    # Configure regime detection
    config = RegimeDetectionConfig(
        detection_method="hybrid",
        n_regimes=3,
        min_regime_duration=50,
        enable_multitimeframe=True,
        enable_transition_detection=True,
        enable_stability_analysis=True,
        enable_streaming=True
    )
    
    # Initialize detector
    detector = UnsupervisedRegimeDetector(config)
    
    # Detect regimes
    logger.info("🔍 Detecting regimes...")
    results = detector.detect_regimes(market_data)
    
    # Display results
    logger.info(f"📊 Detected {len(results['regimes'])} regimes")
    for regime_name, regime_info in results['regimes'].items():
        logger.info(f"  {regime_name}: {regime_info}")
    
    # Demonstrate real-time streaming
    logger.info("🚀 Starting real-time streaming detection...")
    detector.start_streaming_detection()
    
    # Simulate streaming data
    for i in range(5):
        new_data = market_data.iloc[i*10:(i+1)*10]
        detector.add_streaming_data(new_data)
        logger.info(f"📈 Added streaming data batch {i+1}")
    
    # Stop streaming
    detector.stop_streaming_detection()
    logger.info("🛑 Stopped streaming detection")
    
    # Demonstrate multi-timeframe detection
    logger.info("🔄 Demonstrating multi-timeframe detection...")
    
    # Create multi-timeframe data
    timeframes = {
        '1m': market_data.iloc[::10],  # Every 10th sample
        '5m': market_data.iloc[::50],  # Every 50th sample
        '15m': market_data.iloc[::150],  # Every 150th sample
    }
    
    multi_results = detector.detect_regimes_multitimeframe(timeframes)
    logger.info(f"📊 Multi-timeframe results: {len(multi_results['consensus_regimes'])} consensus regimes")
    
    return results


def demonstrate_regime_qualification():
    """Demonstrate regime qualification capabilities."""
    logger.info("✅ Demonstrating Regime Qualification")
    
    # Generate sample data
    market_data = generate_sample_market_data(1000)
    
    # Configure regime qualification
    config = RegimeQualificationConfig(
        min_regime_duration=50,
        min_economic_significance=0.1,
        min_sharpe_ratio=0.5,
        enable_normality_tests=True,
        enable_stationarity_tests=True,
        enable_autocorrelation_tests=True
    )
    
    # Initialize qualifier
    qualifier = RegimeQualifier(config)
    
    # First, detect regimes
    detection_config = RegimeDetectionConfig()
    detector = UnsupervisedRegimeDetector(detection_config)
    detection_results = detector.detect_regimes(market_data)
    
    # Qualify regimes
    logger.info("🔍 Qualifying detected regimes...")
    qualification_results = qualifier.qualify_regimes(detection_results, market_data)
    
    # Display qualification results
    logger.info(f"📊 Qualification Results:")
    logger.info(f"  Qualified: {qualification_results['n_qualified']}/{qualification_results['n_total']} regimes")
    logger.info(f"  Qualification Rate: {qualification_results['qualification_rate']:.2%}")
    
    # Show detailed qualification scores
    for regime_name, score in qualification_results['qualification_scores'].items():
        logger.info(f"  {regime_name}: {score:.3f}")
    
    # Demonstrate comprehensive quality scoring
    logger.info("🎯 Demonstrating comprehensive quality scoring...")
    
    for regime_name, regime_info in detection_results['regimes'].items():
        # Extract regime data
        regime_mask = detection_results['regime_labels'] == regime_info['regime_id']
        regime_data = market_data[regime_mask]
        
        # Calculate quality score
        quality_result = qualifier.calculate_regime_quality_score(regime_info, regime_data)
        
        logger.info(f"📊 {regime_name} Quality Score: {quality_result['quality_score']:.3f}")
        logger.info(f"  Status: {quality_result['qualification_status']}")
        logger.info(f"  Details: {quality_result['details']}")
    
    return qualification_results


def demonstrate_comprehensive_workflow():
    """Demonstrate complete workflow from detection to qualification."""
    logger.info("🚀 Demonstrating Complete Workflow")
    
    # Generate sample data
    market_data = generate_sample_market_data(1000)
    
    # Step 1: Configure systems
    detection_config = RegimeDetectionConfig(
        detection_method="hybrid",
        enable_multitimeframe=True,
        enable_transition_detection=True,
        enable_stability_analysis=True
    )
    
    qualification_config = RegimeQualificationConfig(
        min_regime_duration=50,
        min_economic_significance=0.1,
        min_sharpe_ratio=0.5,
        enable_normality_tests=True,
        enable_stationarity_tests=True
    )
    
    # Step 2: Initialize systems
    detector = UnsupervisedRegimeDetector(detection_config)
    qualifier = RegimeQualifier(qualification_config)
    
    # Step 3: Detect regimes
    logger.info("🔍 Step 1: Detecting regimes...")
    detection_results = detector.detect_regimes(market_data)
    
    # Step 4: Qualify regimes
    logger.info("✅ Step 2: Qualifying regimes...")
    qualification_results = qualifier.qualify_regimes(detection_results, market_data)
    
    # Step 5: Calculate quality scores
    logger.info("🎯 Step 3: Calculating quality scores...")
    quality_results = {}
    
    for regime_name, regime_info in detection_results['regimes'].items():
        if regime_name in qualification_results['qualified_regimes']:
            # Extract regime data
            regime_mask = detection_results['regime_labels'] == regime_info['regime_id']
            regime_data = market_data[regime_mask]
            
            # Calculate quality score
            quality_result = qualifier.calculate_regime_quality_score(regime_info, regime_data)
            quality_results[regime_name] = quality_result
    
    # Step 6: Generate comprehensive report
    logger.info("📊 Step 4: Generating comprehensive report...")
    
    report = {
        'detection_summary': {
            'n_regimes_detected': len(detection_results['regimes']),
            'detection_quality': detection_results['detection_quality'],
            'feature_importance': detection_results['feature_importance']
        },
        'qualification_summary': {
            'n_regimes_qualified': qualification_results['n_qualified'],
            'qualification_rate': qualification_results['qualification_rate'],
            'qualification_statistics': qualification_results['qualification_statistics']
        },
        'quality_scores': {
            regime_name: result['quality_score'] 
            for regime_name, result in quality_results.items()
        },
        'recommendations': generate_trading_recommendations(quality_results)
    }
    
    # Display report
    logger.info("📋 COMPREHENSIVE REPORT")
    logger.info(f"🔍 Detection: {report['detection_summary']['n_regimes_detected']} regimes detected")
    logger.info(f"✅ Qualification: {report['qualification_summary']['n_regimes_qualified']} regimes qualified")
    logger.info(f"🎯 Quality Scores: {report['quality_scores']}")
    logger.info(f"💡 Recommendations: {report['recommendations']}")
    
    return report


def generate_trading_recommendations(quality_results: dict) -> str:
    """Generate trading recommendations based on quality results."""
    if not quality_results:
        return "No qualified regimes for trading"
    
    # Find best regime
    best_regime = max(quality_results.items(), key=lambda x: x[1]['quality_score'])
    best_name, best_result = best_regime
    
    quality_score = best_result['quality_score']
    
    if quality_score >= 0.8:
        return f"Excellent trading opportunity in {best_name} (Score: {quality_score:.3f})"
    elif quality_score >= 0.6:
        return f"Good trading opportunity in {best_name} (Score: {quality_score:.3f})"
    elif quality_score >= 0.4:
        return f"Fair trading opportunity in {best_name} (Score: {quality_score:.3f})"
    else:
        return f"Poor trading opportunity in {best_name} (Score: {quality_score:.3f})"


def main():
    """Main demonstration function."""
    logger.info("🚀 Starting Advanced Regime Detection and Qualification Demo")
    
    try:
        # Demonstrate unsupervised regime detection
        detection_results = demonstrate_unsupervised_regime_detection()
        
        # Demonstrate regime qualification
        qualification_results = demonstrate_regime_qualification()
        
        # Demonstrate complete workflow
        workflow_results = demonstrate_comprehensive_workflow()
        
        logger.info("✅ Demo completed successfully!")
        
        return {
            'detection_results': detection_results,
            'qualification_results': qualification_results,
            'workflow_results': workflow_results
        }
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    results = main()
    print("\n🎉 Demo completed! Check the logs for detailed results.")