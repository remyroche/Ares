"""
Adaptive Threshold Learning Example

Demonstrates how the NAS Regime System learns data-driven thresholds
instead of using hardcoded values for economic significance, trading viability,
and regime stability.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the path
sys.path.append('/workspace/src')

# Import enhanced components
from training.steps.market_analysis.nas_regime.core.enhanced_perfect_nas_config import (
    EnhancedPerfectNASConfig, ThresholdLearningMode
)
from training.steps.market_analysis.nas_regime.core.perfect_nas_regime_detector import (
    PerfectNASRegimeDetector
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_realistic_market_data(n_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate realistic market data with different market conditions."""
    try:
        logger.info(f"📊 Generating realistic market data with {n_samples} samples...")
        
        # Generate timestamps
        start_time = datetime.now() - timedelta(days=n_samples//24)
        timestamps = [start_time + timedelta(hours=i) for i in range(n_samples)]
        
        # Generate OHLCV data with different market regimes
        data = []
        current_price = 100.0
        
        for i in range(n_samples):
            # Simulate different market conditions
            market_period = i // 200  # Change market condition every 200 samples
            
            if market_period % 5 == 0:  # Bull market - high returns, low volatility
                trend = 0.002
                volatility = 0.008
                volume_multiplier = 1.2
            elif market_period % 5 == 1:  # Bear market - negative returns, medium volatility
                trend = -0.0015
                volatility = 0.012
                volume_multiplier = 1.5
            elif market_period % 5 == 2:  # High volatility - high volatility, mixed returns
                trend = 0.0005
                volatility = 0.025
                volume_multiplier = 2.0
            elif market_period % 5 == 3:  # Low volatility - low volatility, small returns
                trend = 0.0003
                volatility = 0.003
                volume_multiplier = 0.8
            else:  # Normal market - moderate conditions
                trend = 0.0008
                volatility = 0.010
                volume_multiplier = 1.0
            
            # Generate price movement
            price_change = np.random.normal(trend, volatility)
            current_price *= (1 + price_change)
            
            # Generate OHLCV
            open_price = current_price
            close_price = open_price * (1 + price_change)
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, volatility/2)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, volatility/2)))
            volume = np.random.lognormal(10, 0.5) * volume_multiplier
            
            data.append([open_price, high_price, low_price, close_price, volume])
        
        market_data = np.array(data)
        timestamps = np.array(timestamps)
        
        logger.info(f"✅ Generated market data: {market_data.shape}")
        return market_data, timestamps
        
    except Exception as e:
        logger.error(f"❌ Market data generation failed: {e}")
        raise

def demonstrate_adaptive_thresholds():
    """Demonstrate adaptive threshold learning."""
    try:
        logger.info("🚀 Demonstrating Adaptive Threshold Learning")
        logger.info("=" * 60)
        
        # Step 1: Generate realistic market data
        logger.info("📊 Step 1: Generating realistic market data...")
        market_data, timestamps = generate_realistic_market_data(n_samples=1500)
        
        # Step 2: Create enhanced configuration with adaptive thresholds
        logger.info("⚙️ Step 2: Creating enhanced configuration with adaptive thresholds...")
        config = EnhancedPerfectNASConfig.create_adaptive_research_config()
        
        # Configure adaptive threshold learning
        config.adaptive_thresholds.learning_mode = ThresholdLearningMode.ADAPTIVE
        config.adaptive_thresholds.learning_frequency = 100  # Learn every 100 samples
        config.adaptive_thresholds.min_samples_for_learning = 200  # Minimum 200 samples
        config.adaptive_thresholds.enable_economic_learning = True
        config.adaptive_thresholds.enable_trading_learning = True
        config.adaptive_thresholds.enable_stability_learning = True
        
        # Enable market condition adaptation
        config.adaptive_thresholds.enable_volatility_adaptation = True
        config.adaptive_thresholds.enable_liquidity_adaptation = True
        config.adaptive_thresholds.enable_stress_adaptation = True
        config.adaptive_thresholds.enable_trend_adaptation = True
        
        logger.info(f"   Learning mode: {config.adaptive_thresholds.learning_mode.value}")
        logger.info(f"   Learning frequency: {config.adaptive_thresholds.learning_frequency}")
        logger.info(f"   Min samples for learning: {config.adaptive_thresholds.min_samples_for_learning}")
        
        # Step 3: Initialize detector with adaptive thresholds
        logger.info("🧠 Step 3: Initializing detector with adaptive thresholds...")
        detector = PerfectNASRegimeDetector(config)
        
        # Step 4: Demonstrate threshold learning
        logger.info("🎯 Step 4: Demonstrating adaptive threshold learning...")
        
        # Split data for learning and testing
        learning_data = market_data[:1000]
        learning_timestamps = timestamps[:1000]
        testing_data = market_data[1000:]
        testing_timestamps = timestamps[1000:]
        
        # Learn thresholds from historical data
        logger.info("📚 Learning thresholds from historical data...")
        learning_success = config.learn_thresholds(
            learning_data, np.array([]), learning_timestamps
        )
        
        if learning_success:
            logger.info("✅ Threshold learning successful!")
            
            # Get learned thresholds
            learned_thresholds = config.get_adaptive_thresholds()
            if learned_thresholds:
                logger.info("📊 Learned Adaptive Thresholds:")
                logger.info(f"   Economic Significance: {learned_thresholds.economic_significance_threshold:.3f}")
                logger.info(f"   Trading Viability: {learned_thresholds.trading_viability_threshold:.3f}")
                logger.info(f"   Regime Stability: {learned_thresholds.regime_stability_threshold:.3f}")
                logger.info(f"   Learning Confidence: {learned_thresholds.learning_confidence:.3f}")
                
                # Get confidence intervals
                confidence_intervals = config.get_threshold_confidence_intervals()
                logger.info("📈 Confidence Intervals:")
                for metric, (lower, upper) in confidence_intervals.items():
                    logger.info(f"   {metric}: [{lower:.3f}, {upper:.3f}]")
                
                # Get threshold explanations
                explanations = config.get_threshold_explanations()
                logger.info("💡 Threshold Explanations:")
                for metric, explanation in explanations.items():
                    logger.info(f"   {metric}: {explanation}")
        else:
            logger.warning("⚠️ Threshold learning failed, using fallback thresholds")
        
        # Step 5: Test with learned thresholds
        logger.info("🧪 Step 5: Testing regime detection with learned thresholds...")
        result = detector.detect_regimes(
            market_data=testing_data,
            timestamps=testing_timestamps,
            optimize_architecture=True,
            enable_meta_learning=True,
            learn_thresholds=False  # Don't learn again, use existing thresholds
        )
        
        # Step 6: Analyze results
        logger.info("📊 Step 6: Analyzing results with adaptive thresholds...")
        analyze_adaptive_results(result, config)
        
        # Step 7: Demonstrate threshold adaptation
        logger.info("🔄 Step 7: Demonstrating threshold adaptation...")
        demonstrate_threshold_adaptation(detector, config, testing_data, testing_timestamps)
        
        logger.info("✅ Adaptive threshold demonstration completed successfully!")
        return result
        
    except Exception as e:
        logger.error(f"❌ Adaptive threshold demonstration failed: {e}")
        raise

def analyze_adaptive_results(result, config: EnhancedPerfectNASConfig):
    """Analyze results with adaptive thresholds."""
    try:
        logger.info("📈 Adaptive Threshold Results Analysis:")
        logger.info("=" * 50)
        
        # Basic results
        logger.info(f"Success: {result.success}")
        logger.info(f"Execution time: {result.execution_time:.2f}s")
        logger.info(f"Regimes detected: {len(np.unique(result.regime_predictions))}")
        
        # Threshold information
        if result.metadata and 'adaptive_thresholds' in result.metadata:
            adaptive_info = result.metadata['adaptive_thresholds']
            logger.info(f"Adaptive thresholds enabled: {adaptive_info['enabled']}")
            logger.info(f"Learning mode: {adaptive_info['learning_mode']}")
            
            # Effective thresholds
            effective_thresholds = adaptive_info['effective_thresholds']
            logger.info("Effective Thresholds:")
            for metric, threshold in effective_thresholds.items():
                logger.info(f"   {metric}: {threshold:.3f}")
            
            # Confidence intervals
            confidence_intervals = adaptive_info['confidence_intervals']
            logger.info("Confidence Intervals:")
            for metric, (lower, upper) in confidence_intervals.items():
                logger.info(f"   {metric}: [{lower:.3f}, {upper:.3f}]")
        
        # Performance metrics
        economic_mean = np.mean(result.economic_significance_scores)
        trading_mean = np.mean(result.trading_viability_scores)
        stability_mean = np.mean(result.regime_stability_scores)
        
        logger.info(f"Economic significance: {economic_mean:.3f}")
        logger.info(f"Trading viability: {trading_mean:.3f}")
        logger.info(f"Regime stability: {stability_mean:.3f}")
        
        # Compare with learned thresholds
        if config.get_adaptive_thresholds():
            learned_thresholds = config.get_adaptive_thresholds()
            logger.info("Threshold Performance:")
            logger.info(f"   Economic: {economic_mean:.3f} vs threshold {learned_thresholds.economic_significance_threshold:.3f}")
            logger.info(f"   Trading: {trading_mean:.3f} vs threshold {learned_thresholds.trading_viability_threshold:.3f}")
            logger.info(f"   Stability: {stability_mean:.3f} vs threshold {learned_thresholds.regime_stability_threshold:.3f}")
        
    except Exception as e:
        logger.warning(f"Results analysis failed: {e}")

def demonstrate_threshold_adaptation(detector, config: EnhancedPerfectNASConfig,
                                  new_data: np.ndarray, new_timestamps: np.ndarray):
    """Demonstrate threshold adaptation with new data."""
    try:
        logger.info("🔄 Demonstrating threshold adaptation...")
        
        # Get current thresholds
        current_thresholds = config.get_effective_thresholds()
        logger.info("Current thresholds:")
        for metric, threshold in current_thresholds.items():
            logger.info(f"   {metric}: {threshold:.3f}")
        
        # Simulate new market conditions (different from training data)
        logger.info("📊 Simulating new market conditions...")
        
        # Create new data with different characteristics
        new_market_data = new_data.copy()
        # Add some noise to simulate market changes
        noise_factor = 0.1
        new_market_data[:, 3] *= (1 + np.random.normal(0, noise_factor, len(new_market_data)))  # Close prices
        
        # Update thresholds with new data
        logger.info("🧠 Updating thresholds with new market data...")
        update_success = config.update_thresholds(
            new_market_data, np.array([]), new_timestamps
        )
        
        if update_success:
            logger.info("✅ Threshold adaptation successful!")
            
            # Get updated thresholds
            updated_thresholds = config.get_effective_thresholds()
            logger.info("Updated thresholds:")
            for metric, threshold in updated_thresholds.items():
                old_threshold = current_thresholds[metric]
                change = threshold - old_threshold
                logger.info(f"   {metric}: {threshold:.3f} (change: {change:+.3f})")
            
            # Get updated explanations
            explanations = config.get_threshold_explanations()
            logger.info("Updated threshold explanations:")
            for metric, explanation in explanations.items():
                logger.info(f"   {metric}: {explanation}")
        else:
            logger.warning("⚠️ Threshold adaptation failed")
        
    except Exception as e:
        logger.warning(f"Threshold adaptation demonstration failed: {e}")

def compare_adaptive_vs_hardcoded():
    """Compare adaptive thresholds vs hardcoded thresholds."""
    try:
        logger.info("⚖️ Comparing Adaptive vs Hardcoded Thresholds")
        logger.info("=" * 50)
        
        # Generate test data
        market_data, timestamps = generate_realistic_market_data(n_samples=1000)
        
        # Test 1: Hardcoded thresholds
        logger.info("🔧 Test 1: Hardcoded thresholds...")
        hardcoded_config = EnhancedPerfectNASConfig()
        hardcoded_config.adaptive_thresholds.learning_mode = ThresholdLearningMode.DISABLED
        hardcoded_config.economic_significance_threshold = 0.8
        hardcoded_config.trading_viability_threshold = 0.7
        hardcoded_config.regime_stability_threshold = 0.8
        
        hardcoded_detector = PerfectNASRegimeDetector(hardcoded_config)
        hardcoded_result = hardcoded_detector.detect_regimes(
            market_data, timestamps, learn_thresholds=False
        )
        
        # Test 2: Adaptive thresholds
        logger.info("🧠 Test 2: Adaptive thresholds...")
        adaptive_config = EnhancedPerfectNASConfig.create_adaptive_research_config()
        adaptive_detector = PerfectNASRegimeDetector(adaptive_config)
        adaptive_result = adaptive_detector.detect_regimes(
            market_data, timestamps, learn_thresholds=True
        )
        
        # Compare results
        logger.info("📊 Comparison Results:")
        logger.info("=" * 30)
        
        # Execution time
        logger.info(f"Hardcoded execution time: {hardcoded_result.execution_time:.2f}s")
        logger.info(f"Adaptive execution time: {adaptive_result.execution_time:.2f}s")
        
        # Performance metrics
        hardcoded_economic = np.mean(hardcoded_result.economic_significance_scores)
        adaptive_economic = np.mean(adaptive_result.economic_significance_scores)
        logger.info(f"Hardcoded economic significance: {hardcoded_economic:.3f}")
        logger.info(f"Adaptive economic significance: {adaptive_economic:.3f}")
        
        hardcoded_trading = np.mean(hardcoded_result.trading_viability_scores)
        adaptive_trading = np.mean(adaptive_result.trading_viability_scores)
        logger.info(f"Hardcoded trading viability: {hardcoded_trading:.3f}")
        logger.info(f"Adaptive trading viability: {adaptive_trading:.3f}")
        
        hardcoded_stability = np.mean(hardcoded_result.regime_stability_scores)
        adaptive_stability = np.mean(adaptive_result.regime_stability_scores)
        logger.info(f"Hardcoded regime stability: {hardcoded_stability:.3f}")
        logger.info(f"Adaptive regime stability: {adaptive_stability:.3f}")
        
        # Threshold information
        if adaptive_result.metadata and 'adaptive_thresholds' in adaptive_result.metadata:
            adaptive_thresholds = adaptive_result.metadata['adaptive_thresholds']['effective_thresholds']
            logger.info("Adaptive learned thresholds:")
            for metric, threshold in adaptive_thresholds.items():
                logger.info(f"   {metric}: {threshold:.3f}")
        
        logger.info("✅ Comparison completed!")
        
    except Exception as e:
        logger.error(f"❌ Comparison failed: {e}")

if __name__ == "__main__":
    """Run the adaptive threshold learning demonstration."""
    try:
        logger.info("🚀 Starting Adaptive Threshold Learning Demonstration")
        logger.info("=" * 80)
        
        # Run main demonstration
        result = demonstrate_adaptive_thresholds()
        
        # Run comparison
        compare_adaptive_vs_hardcoded()
        
        logger.info("\n🏆 Adaptive Threshold Learning Demonstration Complete!")
        logger.info("🎯 Key Achievements:")
        logger.info("   ✅ Data-driven economic significance thresholds")
        logger.info("   ✅ Data-driven trading viability thresholds")
        logger.info("   ✅ Data-driven regime stability thresholds")
        logger.info("   ✅ Market condition adaptation")
        logger.info("   ✅ Confidence interval estimation")
        logger.info("   ✅ Threshold explanation generation")
        logger.info("   ✅ Continuous threshold adaptation")
        
        logger.info("\n💡 Benefits of Adaptive Thresholds:")
        logger.info("   - No more hardcoded values")
        logger.info("   - Thresholds adapt to market conditions")
        logger.info("   - Confidence intervals for uncertainty")
        logger.info("   - Explanations for transparency")
        logger.info("   - Continuous learning and adaptation")
        
    except Exception as e:
        logger.error(f"❌ Adaptive threshold demonstration failed: {e}")
        raise