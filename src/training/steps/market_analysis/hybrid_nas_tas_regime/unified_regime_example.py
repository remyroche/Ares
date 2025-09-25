"""
Unified Regime Detection System Example

This example demonstrates how to use the unified regime detection system
that combines TAS and NAS approaches with enhanced economic evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging

# Import unified regime system
from src.utils.ml_common.nas_tas_unified import (
    UnifiedRegimeDetector, UnifiedRegimeConfig, UnifiedRegimeResult,
    RegimeDetectionMethod, OptimizationStrategy, EconomicEvaluationMode
)

# Import integration layer
from .unified_regime_integration import UnifiedRegimeIntegration

# Import tprint for logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)

def create_sample_market_data(n_samples: int = 1000, n_features: int = 5) -> pd.DataFrame:
    """Create sample market data for demonstration."""
    np.random.seed(42)
    
    # Generate realistic OHLCV data
    base_price = 100.0
    prices = [base_price]
    
    for i in range(n_samples - 1):
        # Random walk with some trend
        change = np.random.normal(0, 0.02) + 0.0001  # Small upward bias
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.01))  # Prevent negative prices
    
    # Create OHLCV data
    data = []
    for i, close in enumerate(prices):
        open_price = prices[i-1] if i > 0 else close
        high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
        low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
        volume = np.random.lognormal(10, 0.5)  # Realistic volume distribution
        
        data.append([open_price, high, low, close, volume])
    
    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
    
    # Add timestamps
    df['timestamp'] = pd.date_range('2023-01-01', periods=len(df), freq='15T')
    
    return df

def demonstrate_unified_regime_detection():
    """Demonstrate unified regime detection with different configurations."""
    
    tprint_info("🚀 Starting Unified Regime Detection Demonstration")
    
    # Create sample data
    tprint_info("📊 Creating sample market data")
    market_data = create_sample_market_data(1000)
    tprint_success(f"✅ Created market data with {len(market_data)} samples")
    
    # Demonstrate different configurations
    configurations = {
        'short_term_trading': UnifiedRegimeConfig.create_short_term_trading_config(),
        'research': UnifiedRegimeConfig.create_research_config(),
        'production': UnifiedRegimeConfig.create_production_config(),
        'economic_focused': UnifiedRegimeConfig.create_economic_focused_config()
    }
    
    results = {}
    
    for config_name, config in configurations.items():
        tprint_info(f"🔧 Testing {config_name} configuration")
        
        try:
            # Initialize detector
            detector = UnifiedRegimeDetector(config)
            
            # Detect regimes
            result = detector.detect_regimes(
                market_data,
                timestamps=market_data['timestamp'].values,
                enable_adaptive_selection=True
            )
            
            results[config_name] = result
            
            # Log results
            if result.success:
                tprint_success(f"✅ {config_name} configuration completed successfully")
                tprint_info(f"   Execution time: {result.execution_time:.2f}s")
                tprint_info(f"   Regimes detected: {len(np.unique(result.regime_predictions))}")
                tprint_info(f"   Economic significance: {np.mean(result.economic_significance_scores):.3f}")
                tprint_info(f"   Trading viability: {np.mean(result.trading_viability_scores):.3f}")
                tprint_info(f"   Regime stability: {np.mean(result.regime_stability_scores):.3f}")
                
                # Show method used
                method = result.metadata.get('method', 'unknown')
                tprint_info(f"   Method used: {method}")
                
                if result.ensemble_weights:
                    tprint_info(f"   Ensemble weights: {result.ensemble_weights}")
            else:
                tprint_error(f"❌ {config_name} configuration failed: {result.error_message}")
        
        except Exception as e:
            tprint_error(f"❌ {config_name} configuration failed with exception: {e}")
            results[config_name] = None
    
    return results

def demonstrate_system_comparison():
    """Demonstrate comparison between different regime detection systems."""
    
    tprint_info("🔄 Starting System Comparison Demonstration")
    
    # Create sample data
    market_data = create_sample_market_data(500)  # Smaller dataset for faster comparison
    
    # Initialize integration layer
    integration = UnifiedRegimeIntegration()
    
    try:
        # Compare all systems
        comparison_results = integration.compare_all_systems(
            market_data,
            timestamps=market_data['timestamp'].values
        )
        
        # Display results
        tprint_success("✅ System comparison completed")
        
        summary = comparison_results.get('summary', {})
        tprint_info(f"📊 Summary:")
        tprint_info(f"   Total systems tested: {summary.get('total_systems_tested', 0)}")
        tprint_info(f"   Successful systems: {summary.get('successful_systems', 0)}")
        tprint_info(f"   Failed systems: {summary.get('failed_systems', 0)}")
        tprint_info(f"   Recommended system: {summary.get('recommended_system', 'None')}")
        
        # Show best performers
        if summary.get('best_accuracy'):
            best_acc = summary['best_accuracy']
            tprint_info(f"   Best accuracy: {best_acc['system']} ({best_acc['score']:.3f})")
        
        if summary.get('best_efficiency'):
            best_eff = summary['best_efficiency']
            tprint_info(f"   Best efficiency: {best_eff['system']} ({best_eff['score']:.3f})")
        
        if summary.get('best_economic_score'):
            best_econ = summary['best_economic_score']
            tprint_info(f"   Best economic score: {best_econ['system']} ({best_econ['score']:.3f})")
        
        # Show detailed results for each system
        for system_name, system_results in comparison_results.items():
            if system_name == 'summary' or 'error' in system_results:
                continue
            
            tprint_info(f"📊 {system_name.upper()} Results:")
            tprint_info(f"   Success: {system_results['success']}")
            tprint_info(f"   Execution time: {system_results['execution_time']:.2f}s")
            tprint_info(f"   Accuracy: {system_results['accuracy']:.3f}")
            tprint_info(f"   Efficiency: {system_results['efficiency']:.3f}")
            tprint_info(f"   Economic score: {system_results['economic_score']:.3f}")
        
        return comparison_results
        
    except Exception as e:
        tprint_error(f"❌ System comparison failed: {e}")
        return None

def demonstrate_adaptive_selection():
    """Demonstrate adaptive system selection based on performance."""
    
    tprint_info("🎯 Starting Adaptive Selection Demonstration")
    
    # Create multiple datasets to simulate different market conditions
    datasets = {
        'bull_market': create_sample_market_data(300, trend=0.001),
        'bear_market': create_sample_market_data(300, trend=-0.001),
        'sideways_market': create_sample_market_data(300, trend=0.0),
        'volatile_market': create_sample_market_data(300, volatility=0.05)
    }
    
    # Initialize integration with adaptive configuration
    config = UnifiedRegimeConfig.create_production_config()
    config.detection_method = RegimeDetectionMethod.ADAPTIVE_SELECTION
    integration = UnifiedRegimeIntegration(config)
    
    results = {}
    
    for market_type, data in datasets.items():
        tprint_info(f"📊 Testing adaptive selection on {market_type}")
        
        try:
            # Run recommended system
            result = integration.run_recommended_system(
                data,
                timestamps=data['timestamp'].values
            )
            
            results[market_type] = result
            
            # Show which system was recommended
            recommended = integration.recommend_best_system()
            tprint_success(f"✅ {market_type}: Recommended system was {recommended}")
            
            if hasattr(result, 'success') and result.success:
                tprint_info(f"   Execution time: {result.execution_time:.2f}s")
                tprint_info(f"   Regimes detected: {len(np.unique(result.regime_predictions))}")
        
        except Exception as e:
            tprint_error(f"❌ {market_type} adaptive selection failed: {e}")
            results[market_type] = None
    
    # Show final recommendations
    tprint_info("📊 Final System Recommendations:")
    metrics = integration.get_comparison_metrics()
    
    for system_name, system_metrics in metrics.items():
        if system_metrics['total_runs'] > 0:
            tprint_info(f"   {system_name.upper()}:")
            tprint_info(f"     Average accuracy: {system_metrics['avg_accuracy']:.3f}")
            tprint_info(f"     Average efficiency: {system_metrics['avg_efficiency']:.3f}")
            tprint_info(f"     Average economic score: {system_metrics['avg_economic_score']:.3f}")
            tprint_info(f"     Total runs: {system_metrics['total_runs']}")
    
    final_recommendation = integration.recommend_best_system()
    tprint_success(f"🎯 Final recommended system: {final_recommendation}")
    
    return results

def create_sample_market_data(n_samples: int, trend: float = 0.0, volatility: float = 0.02) -> pd.DataFrame:
    """Create sample market data with specified trend and volatility."""
    np.random.seed(42)
    
    base_price = 100.0
    prices = [base_price]
    
    for i in range(n_samples - 1):
        # Random walk with specified trend and volatility
        change = np.random.normal(trend, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.01))
    
    # Create OHLCV data
    data = []
    for i, close in enumerate(prices):
        open_price = prices[i-1] if i > 0 else close
        high = max(open_price, close) * (1 + abs(np.random.normal(0, volatility/2)))
        low = min(open_price, close) * (1 - abs(np.random.normal(0, volatility/2)))
        volume = np.random.lognormal(10, 0.5)
        
        data.append([open_price, high, low, close, volume])
    
    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.date_range('2023-01-01', periods=len(df), freq='15T')
    
    return df

def main():
    """Main demonstration function."""
    
    tprint("🚀 UNIFIED REGIME DETECTION SYSTEM DEMONSTRATION", color="cyan", bold=True)
    tprint("=" * 60, color="cyan")
    
    try:
        # Demonstrate unified regime detection
        tprint("\n📊 UNIFIED REGIME DETECTION DEMONSTRATION", color="blue", bold=True)
        unified_results = demonstrate_unified_regime_detection()
        
        # Demonstrate system comparison
        tprint("\n🔄 SYSTEM COMPARISON DEMONSTRATION", color="blue", bold=True)
        comparison_results = demonstrate_system_comparison()
        
        # Demonstrate adaptive selection
        tprint("\n🎯 ADAPTIVE SELECTION DEMONSTRATION", color="blue", bold=True)
        adaptive_results = demonstrate_adaptive_selection()
        
        # Summary
        tprint("\n✅ DEMONSTRATION COMPLETED SUCCESSFULLY", color="green", bold=True)
        tprint("=" * 60, color="green")
        
        tprint_info("📊 Summary of demonstrations:")
        tprint_info(f"   Unified regime detection: {'✅ Success' if unified_results else '❌ Failed'}")
        tprint_info(f"   System comparison: {'✅ Success' if comparison_results else '❌ Failed'}")
        tprint_info(f"   Adaptive selection: {'✅ Success' if adaptive_results else '❌ Failed'}")
        
        return {
            'unified_results': unified_results,
            'comparison_results': comparison_results,
            'adaptive_results': adaptive_results
        }
        
    except Exception as e:
        tprint_error(f"❌ Demonstration failed: {e}")
        return None

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demonstration
    results = main()
    
    if results:
        tprint_success("🎉 All demonstrations completed successfully!")
    else:
        tprint_error("💥 Demonstration failed!")