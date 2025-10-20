"""
Enhanced HDBSCAN Economic Profiling System - Example Usage

This example demonstrates the enhanced functionality added to the existing files:
- Enhanced probability calculation in hdbscan_clusterer.py
- Enhanced validation in economic_validator.py
- Enhanced integration in main_regime_discovery.py
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

# Import the enhanced existing components
from src.training.steps.market_analysis.hdbscan_clustering.main_regime_discovery import HDBSCANRegimeDiscovery
from src.training.steps.market_analysis.hdbscan_clustering.config.regime_discovery_config import RegimeDiscoveryConfig

# Import utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance, tprint_progress, tprint_timer,
    tprint_logged, LogLevel
)

logger = logging.getLogger(__name__)

def generate_sample_market_data(n_periods: int = 1000) -> pd.DataFrame:
    """Generate sample market data for demonstration."""
    try:
        tprint_info(f"Generating {n_periods} periods of sample market data")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate time index
        start_date = datetime.now() - timedelta(days=n_periods)
        dates = pd.date_range(start=start_date, periods=n_periods, freq='D')
        
        # Generate price data with regime changes
        base_price = 100.0
        prices = [base_price]
        
        # Define regime periods
        regime_periods = [
            (0, 200, 0.001, 0.02),    # Bull market: high return, low vol
            (200, 400, -0.0005, 0.03), # Bear market: negative return, high vol
            (400, 600, 0.0002, 0.015), # Sideways: low return, low vol
            (600, 800, 0.0008, 0.025), # Recovery: moderate return, moderate vol
            (800, 1000, 0.0001, 0.02)  # Consolidation: low return, low vol
        ]
        
        for i in range(1, n_periods):
            # Find current regime
            current_regime = None
            for start, end, mean_return, volatility in regime_periods:
                if start <= i < end:
                    current_regime = (mean_return, volatility)
                    break
            
            if current_regime is None:
                current_regime = (0.0001, 0.02)  # Default regime
            
            # Generate return
            daily_return = np.random.normal(current_regime[0], current_regime[1])
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)
        
        # Generate volume data (correlated with volatility)
        volumes = []
        for i in range(n_periods):
            # Find current regime for volume calculation
            current_regime = None
            for start, end, mean_return, volatility in regime_periods:
                if start <= i < end:
                    current_regime = (mean_return, volatility)
                    break
            
            if current_regime is None:
                current_regime = (0.0001, 0.02)
            
            # Volume is inversely related to volatility (simplified)
            base_volume = 1000000
            volume_multiplier = 1.0 + (0.03 - current_regime[1]) * 10
            volume = base_volume * volume_multiplier * np.random.uniform(0.8, 1.2)
            volumes.append(max(volume, 100000))  # Minimum volume
        
        # Create DataFrame
        market_data = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * np.random.uniform(1.0, 1.05) for p in prices],
            'low': [p * np.random.uniform(0.95, 1.0) for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        # Ensure high >= low
        market_data['high'] = np.maximum(market_data['high'], market_data['close'])
        market_data['low'] = np.minimum(market_data['low'], market_data['close'])
        
        tprint_success(f"Generated market data: {len(market_data)} periods")
        return market_data
        
    except Exception as e:
        tprint_error(f"Market data generation failed: {e}")
        raise

def demonstrate_enhanced_features():
    """Demonstrate the enhanced features of the existing system."""
    try:
        tprint_info("🚀 Starting Enhanced HDBSCAN Economic Profiling System Demo")
        
        # Generate sample data
        market_data = generate_sample_market_data(n_periods=500)
        
        # Create configuration with enhanced validation
        config = RegimeDiscoveryConfig(
            enable_validation=True,
            enable_optimization=True,
            enable_temporal_stabilization=True
        )
        
        # Create regime discovery system
        regime_discovery = HDBSCANRegimeDiscovery(config, use_optimized=True)
        
        # Fit the system
        tprint_info("🔧 Fitting the enhanced regime discovery system...")
        result = regime_discovery.fit(market_data)
        
        if result.success:
            tprint_success("✅ Enhanced regime discovery completed successfully!")
            
            # Display results
            print(f"\n📊 REGIME DISCOVERY RESULTS:")
            print(f"  Number of regimes: {len(set(result.labels)) - (1 if -1 in result.labels else 0)}")
            print(f"  Noise points: {np.sum(result.labels == -1)}")
            print(f"  Processing time: {result.processing_time:.2f} seconds")
            
            # Display economic profiles
            if result.economic_profiles:
                print(f"\n💰 ECONOMIC PROFILES:")
                for i, profile in enumerate(result.economic_profiles):
                    print(f"  Regime {i+1}: {profile.get('name', 'Unknown')}")
                    print(f"    Mean Return: {profile.get('key_stats', {}).get('mean_return', 0):.4f}")
                    print(f"    Volatility: {profile.get('key_stats', {}).get('volatility', 0):.4f}")
                    print(f"    Sharpe Ratio: {profile.get('key_stats', {}).get('sharpe_ratio', 0):.4f}")
            
            # Display validation metrics
            if result.validation_metrics:
                print(f"\n🔍 VALIDATION METRICS:")
                for key, value in result.validation_metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.3f}")
                    else:
                        print(f"  {key}: {value}")
            
            # Demonstrate enhanced prediction
            tprint_info("🔮 Demonstrating enhanced prediction with uncertainty...")
            
            # Use last 50 samples for prediction
            test_data = market_data.tail(50)
            prediction_result = regime_discovery.enhanced_predict_with_uncertainty(test_data)
            
            if prediction_result.get('success', False):
                tprint_success("✅ Enhanced prediction completed!")
                
                uncertainty_measures = prediction_result.get('uncertainty_measures', {})
                print(f"\n🎯 PREDICTION UNCERTAINTY MEASURES:")
                print(f"  Method Agreement: {uncertainty_measures.get('method_agreement', 0):.3f}")
                print(f"  Probability Variance: {uncertainty_measures.get('probability_variance', 0):.3f}")
                print(f"  Low Confidence Ratio: {uncertainty_measures.get('low_confidence_ratio', 0):.3f}")
                print(f"  Noise Ratio: {uncertainty_measures.get('noise_ratio', 0):.3f}")
            else:
                tprint_warning("⚠️ Enhanced prediction failed")
            
            # Demonstrate model persistence
            tprint_info("💾 Demonstrating model persistence...")
            
            # Save model
            save_success = regime_discovery.save_model("enhanced_model.pkl")
            if save_success:
                tprint_success("✅ Model saved successfully!")
                
                # Load model
                load_success = regime_discovery.load_model("enhanced_model.pkl")
                if load_success:
                    tprint_success("✅ Model loaded successfully!")
                else:
                    tprint_warning("⚠️ Model loading failed")
            else:
                tprint_warning("⚠️ Model saving failed")
            
            # Generate enhanced report
            tprint_info("📋 Generating enhanced report...")
            report = regime_discovery.generate_enhanced_report()
            print(f"\n{report}")
            
            # Save report to file
            with open("enhanced_hdbscan_report.txt", "w") as f:
                f.write(report)
            tprint_success("📄 Report saved to enhanced_hdbscan_report.txt")
            
        else:
            tprint_error(f"❌ Regime discovery failed: {result.error_message}")
        
    except Exception as e:
        tprint_error(f"❌ Demo failed: {e}")
        print(f"Error details: {e}")

def main():
    """Main function to run the enhanced HDBSCAN demo."""
    try:
        print("=" * 80)
        print("ENHANCED HDBSCAN ECONOMIC PROFILING SYSTEM - DEMO")
        print("=" * 80)
        print()
        print("This demo showcases the enhanced features added to existing files:")
        print("✅ Enhanced probability calculation in hdbscan_clusterer.py")
        print("✅ Enhanced validation in economic_validator.py")
        print("✅ Enhanced integration in main_regime_discovery.py")
        print("✅ Model persistence capabilities")
        print("✅ Uncertainty quantification")
        print("✅ Comprehensive reporting")
        print()
        
        demonstrate_enhanced_features()
        
        print("\n" + "=" * 80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check the logs for more details.")

if __name__ == "__main__":
    main()