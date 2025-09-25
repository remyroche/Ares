"""
Perfect NAS Regime System - Example Usage

This example demonstrates how to use the Perfect NAS Regime System
for advanced regime detection with economic significance and trading viability.
"""

import numpy as np
import pandas as pd
import torch
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Import Perfect NAS components
from ..core.perfect_nas_config import PerfectNASConfig, NeuralArchitectureType
from ..core.perfect_nas_regime_detector import PerfectNASRegimeDetector
from src.utils.nas_tas.shared_utils.unified_economic_evaluator import UnifiedEconomicSignificanceEvaluator as EconomicSignificanceEvaluator
from src.utils.nas_tas.shared_utils.unified_trading_viability_evaluator import UnifiedTradingViabilityEvaluator as TradingViabilityEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_market_data(n_samples: int = 1000, 
                               n_features: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample market data for demonstration."""
    try:
        # Generate realistic market data
        np.random.seed(42)
        
        # Generate timestamps
        start_time = datetime.now() - timedelta(days=n_samples//24)  # Assuming hourly data
        timestamps = [start_time + timedelta(hours=i) for i in range(n_samples)]
        
        # Generate OHLCV data with regime-like patterns
        data = []
        current_price = 100.0
        
        for i in range(n_samples):
            # Simulate different market regimes
            regime_period = i // 100  # Change regime every 100 samples
            
            if regime_period % 4 == 0:  # Bull market
                trend = 0.001
                volatility = 0.01
            elif regime_period % 4 == 1:  # Bear market
                trend = -0.001
                volatility = 0.015
            elif regime_period % 4 == 2:  # High volatility
                trend = 0.0005
                volatility = 0.02
            else:  # Low volatility
                trend = 0.0002
                volatility = 0.005
            
            # Generate price movement
            price_change = np.random.normal(trend, volatility)
            current_price *= (1 + price_change)
            
            # Generate OHLCV
            open_price = current_price
            high_price = open_price * (1 + abs(np.random.normal(0, volatility/2)))
            low_price = open_price * (1 - abs(np.random.normal(0, volatility/2)))
            close_price = open_price * (1 + price_change)
            volume = np.random.lognormal(10, 0.5)  # Log-normal volume
            
            data.append([open_price, high_price, low_price, close_price, volume])
        
        market_data = np.array(data)
        timestamps = np.array(timestamps)
        
        logger.info(f"✅ Generated sample market data: {market_data.shape}")
        return market_data, timestamps
        
    except Exception as e:
        logger.error(f"❌ Market data generation failed: {e}")
        raise

def demonstrate_perfect_nas_system():
    """Demonstrate the Perfect NAS Regime System."""
    try:
        logger.info("🚀 Starting Perfect NAS Regime System Demonstration")
        
        # Step 1: Generate sample data
        logger.info("📊 Generating sample market data...")
        market_data, timestamps = generate_sample_market_data(n_samples=1000)
        
        # Step 2: Create configuration
        logger.info("⚙️ Creating Perfect NAS configuration...")
        config = PerfectNASConfig.create_short_term_trading_config()
        config.primary_architecture = NeuralArchitectureType.HYBRID
        config.enable_neural_odes = True
        config.enable_vision_transformers = True
        config.enable_meta_learning = True
        config.n_regimes = 8
        config.population_size = 20  # Reduced for demo
        config.generations = 10     # Reduced for demo
        
        # Step 3: Initialize Perfect NAS Detector
        logger.info("🧠 Initializing Perfect NAS Regime Detector...")
        detector = PerfectNASRegimeDetector(config)
        
        # Step 4: Detect regimes
        logger.info("🎯 Detecting regimes with Perfect NAS system...")
        result = detector.detect_regimes(
            market_data=market_data,
            timestamps=timestamps,
            optimize_architecture=True,
            enable_meta_learning=True
        )
        
        # Step 5: Analyze results
        logger.info("📈 Analyzing Perfect NAS results...")
        analyze_perfect_nas_results(result, market_data, timestamps)
        
        # Step 6: Demonstrate individual components
        logger.info("🔍 Demonstrating individual components...")
        demonstrate_individual_components(market_data, timestamps, config)
        
        logger.info("✅ Perfect NAS Regime System demonstration completed successfully!")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Perfect NAS demonstration failed: {e}")
        raise

def analyze_perfect_nas_results(result, market_data: np.ndarray, timestamps: np.ndarray):
    """Analyze and display Perfect NAS results."""
    try:
        logger.info("📊 Perfect NAS Results Analysis:")
        logger.info("=" * 50)
        
        # Basic results
        logger.info(f"Success: {result.success}")
        logger.info(f"Execution time: {result.execution_time:.2f}s")
        logger.info(f"Regimes detected: {len(np.unique(result.regime_predictions))}")
        
        # Regime predictions
        unique_regimes, regime_counts = np.unique(result.regime_predictions, return_counts=True)
        logger.info(f"Regime distribution:")
        for regime, count in zip(unique_regimes, regime_counts):
            percentage = (count / len(result.regime_predictions)) * 100
            logger.info(f"  Regime {regime}: {count} samples ({percentage:.1f}%)")
        
        # Economic significance
        economic_mean = np.mean(result.economic_significance_scores)
        economic_std = np.std(result.economic_significance_scores)
        logger.info(f"Economic significance: {economic_mean:.3f} ± {economic_std:.3f}")
        
        # Trading viability
        trading_mean = np.mean(result.trading_viability_scores)
        trading_std = np.std(result.trading_viability_scores)
        logger.info(f"Trading viability: {trading_mean:.3f} ± {trading_std:.3f}")
        
        # Regime stability
        stability_mean = np.mean(result.regime_stability_scores)
        stability_std = np.std(result.regime_stability_scores)
        logger.info(f"Regime stability: {stability_mean:.3f} ± {stability_std:.3f}")
        
        # Micro-regimes
        if result.micro_regimes:
            micro_types = result.micro_regimes.get('types', [])
            unique_micro_types = set(micro_types)
            logger.info(f"Micro-regimes detected: {len(unique_micro_types)} types")
            logger.info(f"Micro-regime types: {list(unique_micro_types)}")
        
        # Uncertainty estimates
        if result.uncertainty_estimates is not None:
            uncertainty_mean = np.mean(result.uncertainty_estimates)
            uncertainty_std = np.std(result.uncertainty_estimates)
            logger.info(f"Uncertainty estimates: {uncertainty_mean:.3f} ± {uncertainty_std:.3f}")
        
        # Architecture performance
        if result.architecture_performance:
            arch_perf = result.architecture_performance
            logger.info(f"Architecture performance:")
            if 'best_architecture' in arch_perf:
                best_arch = arch_perf['best_architecture']
                logger.info(f"  Best fitness: {best_arch.fitness_score:.4f}")
                logger.info(f"  Architecture layers: {len(best_arch.layers)}")
                logger.info(f"  Parameters: {best_arch.parameters_count}")
        
        # Create visualizations
        create_perfect_nas_visualizations(result, market_data, timestamps)
        
    except Exception as e:
        logger.warning(f"Results analysis failed: {e}")

def demonstrate_individual_components(market_data: np.ndarray, timestamps: np.ndarray, config: PerfectNASConfig):
    """Demonstrate individual components of the Perfect NAS system."""
    try:
        # Economic Significance Evaluator
        logger.info("💰 Testing Economic Significance Evaluator...")
        economic_evaluator = EconomicSignificanceEvaluator(config.economic_config)
        
        # Create dummy regime predictions for testing
        dummy_predictions = np.random.randint(0, config.n_regimes, len(market_data))
        economic_scores = economic_evaluator.evaluate(market_data, dummy_predictions, timestamps)
        logger.info(f"Economic significance scores: {np.mean(economic_scores):.3f} ± {np.std(economic_scores):.3f}")
        
        # Trading Viability Evaluator
        logger.info("📈 Testing Trading Viability Evaluator...")
        trading_evaluator = TradingViabilityEvaluator(config.trading_config)
        trading_scores = trading_evaluator.evaluate(market_data, dummy_predictions, timestamps)
        logger.info(f"Trading viability scores: {np.mean(trading_scores):.3f} ± {np.std(trading_scores):.3f}")
        
        # Get detailed analysis
        economic_analysis = economic_evaluator.get_detailed_economic_analysis(market_data, dummy_predictions, timestamps)
        trading_analysis = trading_evaluator.get_detailed_trading_analysis(market_data, dummy_predictions, timestamps)
        
        logger.info(f"Economic analysis: {len(economic_analysis.get('regime_economic_profiles', {}))} regime profiles")
        logger.info(f"Trading analysis: {len(trading_analysis.get('regime_trading_profiles', {}))} regime profiles")
        
    except Exception as e:
        logger.warning(f"Individual component demonstration failed: {e}")

def create_perfect_nas_visualizations(result, market_data: np.ndarray, timestamps: np.ndarray):
    """Create visualizations for Perfect NAS results."""
    try:
        logger.info("📊 Creating Perfect NAS visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Perfect NAS Regime System Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Price data with regime colors
        ax1 = axes[0, 0]
        close_prices = market_data[:, 3]  # Close prices
        colors = plt.cm.Set3(result.regime_predictions / max(result.regime_predictions))
        
        ax1.scatter(range(len(close_prices)), close_prices, c=colors, alpha=0.6, s=1)
        ax1.set_title('Market Data with Regime Colors')
        ax1.set_xlabel('Time Index')
        ax1.set_ylabel('Close Price')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Economic significance over time
        ax2 = axes[0, 1]
        ax2.plot(result.economic_significance_scores, color='green', alpha=0.7)
        ax2.set_title('Economic Significance Over Time')
        ax2.set_xlabel('Time Index')
        ax2.set_ylabel('Economic Significance Score')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Trading viability over time
        ax3 = axes[1, 0]
        ax3.plot(result.trading_viability_scores, color='blue', alpha=0.7)
        ax3.set_title('Trading Viability Over Time')
        ax3.set_xlabel('Time Index')
        ax3.set_ylabel('Trading Viability Score')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Regime stability over time
        ax4 = axes[1, 1]
        ax4.plot(result.regime_stability_scores, color='red', alpha=0.7)
        ax4.set_title('Regime Stability Over Time')
        ax4.set_xlabel('Time Index')
        ax4.set_ylabel('Regime Stability Score')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('perfect_nas_results.png', dpi=300, bbox_inches='tight')
        logger.info("✅ Visualizations saved as 'perfect_nas_results.png'")
        
        # Create additional analysis plots
        create_additional_analysis_plots(result, market_data)
        
    except Exception as e:
        logger.warning(f"Visualization creation failed: {e}")

def create_additional_analysis_plots(result, market_data: np.ndarray):
    """Create additional analysis plots."""
    try:
        # Regime distribution pie chart
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Regime distribution
        unique_regimes, regime_counts = np.unique(result.regime_predictions, return_counts=True)
        axes[0].pie(regime_counts, labels=[f'Regime {r}' for r in unique_regimes], autopct='%1.1f%%')
        axes[0].set_title('Regime Distribution')
        
        # Score distributions
        scores_data = [
            result.economic_significance_scores,
            result.trading_viability_scores,
            result.regime_stability_scores
        ]
        scores_labels = ['Economic Significance', 'Trading Viability', 'Regime Stability']
        
        axes[1].boxplot(scores_data, labels=scores_labels)
        axes[1].set_title('Score Distributions')
        axes[1].set_ylabel('Score')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('perfect_nas_analysis.png', dpi=300, bbox_inches='tight')
        logger.info("✅ Additional analysis plots saved as 'perfect_nas_analysis.png'")
        
    except Exception as e:
        logger.warning(f"Additional analysis plots failed: {e}")

def run_performance_benchmark():
    """Run performance benchmark for Perfect NAS system."""
    try:
        logger.info("⚡ Running Perfect NAS Performance Benchmark...")
        
        # Test different data sizes
        data_sizes = [100, 500, 1000, 2000]
        results = {}
        
        for size in data_sizes:
            logger.info(f"Testing with {size} samples...")
            
            # Generate data
            market_data, timestamps = generate_sample_market_data(n_samples=size)
            
            # Create config
            config = PerfectNASConfig.create_production_config()
            config.population_size = 10  # Reduced for benchmark
            config.generations = 5       # Reduced for benchmark
            
            # Initialize detector
            detector = PerfectNASRegimeDetector(config)
            
            # Measure execution time
            import time
            start_time = time.time()
            
            result = detector.detect_regimes(
                market_data=market_data,
                timestamps=timestamps,
                optimize_architecture=True,
                enable_meta_learning=False  # Disable for benchmark
            )
            
            execution_time = time.time() - start_time
            
            results[size] = {
                'execution_time': execution_time,
                'success': result.success,
                'regimes_detected': len(np.unique(result.regime_predictions)),
                'economic_significance': np.mean(result.economic_significance_scores),
                'trading_viability': np.mean(result.trading_viability_scores)
            }
            
            logger.info(f"  Execution time: {execution_time:.2f}s")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Regimes detected: {len(np.unique(result.regime_predictions))}")
        
        # Display benchmark results
        logger.info("📊 Performance Benchmark Results:")
        logger.info("=" * 50)
        for size, metrics in results.items():
            logger.info(f"Data size {size}:")
            logger.info(f"  Execution time: {metrics['execution_time']:.2f}s")
            logger.info(f"  Success: {metrics['success']}")
            logger.info(f"  Regimes: {metrics['regimes_detected']}")
            logger.info(f"  Economic significance: {metrics['economic_significance']:.3f}")
            logger.info(f"  Trading viability: {metrics['trading_viability']:.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Performance benchmark failed: {e}")
        return {}

if __name__ == "__main__":
    """Run the Perfect NAS Regime System demonstration."""
    try:
        logger.info("🚀 Starting Perfect NAS Regime System Demo")
        
        # Run main demonstration
        result = demonstrate_perfect_nas_system()
        
        # Run performance benchmark
        benchmark_results = run_performance_benchmark()
        
        logger.info("✅ Perfect NAS Regime System demonstration completed!")
        logger.info("🎯 Key achievements:")
        logger.info("   - Advanced neural architectures (Neural ODEs, Vision Transformers)")
        logger.info("   - True NAS search with evolutionary algorithms")
        logger.info("   - Economic significance evaluation")
        logger.info("   - Trading viability assessment")
        logger.info("   - Meta-learning for regime adaptation")
        logger.info("   - Production-ready optimization")
        
    except Exception as e:
        logger.error(f"❌ Perfect NAS demonstration failed: {e}")
        raise