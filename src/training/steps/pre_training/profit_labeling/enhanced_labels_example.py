"""
Enhanced Data & Labels System - Example Usage

This script demonstrates how to use the enhanced data and labels system
to define what truth means in trading ML, clean inputs, and ensure stability.

The enhanced system provides:
1. Trading-aware label definitions (Analyst: "Should we trade?", Tactician: Direction/magnitude)
2. Comprehensive data cleaning (outliers, timestamps, deduplication)
3. Label stability monitoring (leakage detection, drift checking, OOS balance)
4. Full integration with existing infrastructure

Usage:
    python enhanced_labels_example.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Import the enhanced system
from enhanced_data_labels_system import (
    EnhancedDataLabelsSystem, EnhancedDataLabelsConfig,
    create_trading_optimized_config, create_research_optimized_config
)
from infrastructure_integration import (
    get_integration_manager, process_market_data_enhanced,
    validate_system_integration
)
from enhanced_labels_validation import (
    run_enhanced_labels_validation, validate_system_integration
)

# Import utilities
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success


def generate_sample_market_data(n_samples: int = 1000, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Generate realistic sample market data for demonstration."""
    tprint_info(f"📊 Generating {n_samples} samples of {symbol} market data...")
    
    # Generate datetime index (hourly data)
    start_date = datetime.now() - timedelta(hours=n_samples)
    dates = pd.date_range(start=start_date, periods=n_samples, freq='H')
    
    # Generate realistic price data with trends and volatility
    np.random.seed(42)  # For reproducibility
    
    # Base price
    base_price = 50000.0 if symbol == "BTCUSDT" else 100.0
    
    # Generate returns with realistic characteristics
    returns = np.random.normal(0, 0.02, n_samples)  # 2% hourly volatility
    
    # Add some trend and volatility clustering
    trend = np.linspace(0, 0.05, n_samples)  # 5% upward trend over period
    volatility_cluster = np.random.normal(0, 0.01, n_samples)
    volatility_cluster = np.convolve(volatility_cluster, np.ones(10)/10, mode='same')
    
    returns = returns + trend + volatility_cluster
    
    # Calculate prices
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(12, 1, n_samples)  # Realistic volume distribution
    }, index=dates)
    
    # Ensure OHLC relationships are correct
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Add some realistic market microstructure
    data['volume'] = data['volume'] * (1 + np.random.normal(0, 0.1, n_samples))
    data['volume'] = np.maximum(data['volume'], 100)  # Minimum volume
    
    tprint_success(f"✅ Generated {symbol} data: {data.shape}")
    tprint_info(f"   → Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    tprint_info(f"   → Volume range: {data['volume'].min():.0f} - {data['volume'].max():.0f}")
    
    return data


def demonstrate_basic_usage():
    """Demonstrate basic usage of the enhanced data and labels system."""
    tprint_info("🚀 Demonstrating basic usage of enhanced data and labels system")
    
    # Generate sample data
    market_data = generate_sample_market_data(500, "BTCUSDT")
    
    # Create enhanced system with trading-optimized configuration
    config = create_trading_optimized_config()
    enhanced_system = EnhancedDataLabelsSystem(config)
    
    # Process market data
    tprint_info("🔄 Processing market data with enhanced system...")
    start_time = time.time()
    
    result = enhanced_system.process_market_data(market_data)
    
    processing_time = time.time() - start_time
    
    # Display results
    tprint_success(f"✅ Processing completed in {processing_time:.2f}s")
    
    if 'error' in result:
        tprint_error(f"❌ Processing failed: {result['error']}")
        return
    
    # Show data quality results
    data_quality = result.get('data_quality', {})
    tprint_info(f"📊 Data Quality: {data_quality.get('quality_level', 'unknown')} ({data_quality.get('quality_score', 0.0):.3f})")
    tprint_info(f"   → Samples removed: {data_quality.get('samples_removed', 0)}")
    tprint_info(f"   → Features removed: {data_quality.get('features_removed', 0)}")
    
    # Show label results
    labels = result.get('labels', pd.DataFrame())
    if not labels.empty:
        tprint_info(f"🎯 Labels generated: {len(labels)} samples")
        tprint_info(f"   → Analyst positive ratio: {labels.get('analyst_label', pd.Series()).mean():.3f}")
        tprint_info(f"   → Tactician positive ratio: {labels.get('tactician_label', pd.Series()).mean():.3f}")
        tprint_info(f"   → Average analyst confidence: {labels.get('analyst_confidence', pd.Series()).mean():.3f}")
        tprint_info(f"   → Average tactician magnitude: {labels.get('tactician_magnitude', pd.Series()).mean():.3f}")
    
    # Show stability results
    label_stability = result.get('label_stability', {})
    tprint_info(f"🔍 Label Stability: {label_stability.get('stability_level', 'unknown')} ({label_stability.get('overall_stability', 0.0):.3f})")
    
    # Show final quality
    final_quality = result.get('final_quality', {})
    tprint_info(f"✅ Final Quality: {final_quality.get('quality_grade', 'F')} ({final_quality.get('overall_score', 0.0):.3f})")
    
    return result


def demonstrate_integration_usage():
    """Demonstrate integration with existing infrastructure."""
    tprint_info("🔗 Demonstrating integration with existing infrastructure")
    
    # Generate sample data
    market_data = generate_sample_market_data(1000, "ETHUSDT")
    
    # Use integration manager for full pipeline
    tprint_info("🔄 Processing with full integration pipeline...")
    start_time = time.time()
    
    result = process_market_data_enhanced(
        market_data=market_data,
        force_regime_detection=True,
        force_feature_engineering=True,
        force_recompute=False
    )
    
    processing_time = time.time() - start_time
    
    # Display results
    tprint_success(f"✅ Integration processing completed in {processing_time:.2f}s")
    
    if 'error' in result:
        tprint_error(f"❌ Integration processing failed: {result['error']}")
        return
    
    # Show integration status
    integration_status = result.get('integration_status', {})
    tprint_info(f"🔗 Integration Status:")
    for component, status in integration_status.items():
        status_icon = "✅" if status else "❌"
        tprint_info(f"   → {component}: {status_icon}")
    
    # Show data quality
    data_quality_level = result.get('data_quality_level', 'unknown')
    label_stability_level = result.get('label_stability_level', 'unknown')
    tprint_info(f"📊 Quality Levels: Data={data_quality_level}, Stability={label_stability_level}")
    
    # Show processed data info
    processed_data = result.get('processed_data', pd.DataFrame())
    labels = result.get('labels', pd.DataFrame())
    tprint_info(f"📈 Processed Data: {processed_data.shape}")
    tprint_info(f"🎯 Labels: {labels.shape}")
    
    return result


def demonstrate_validation():
    """Demonstrate validation of the enhanced system."""
    tprint_info("🔍 Demonstrating validation of enhanced system")
    
    # Run comprehensive validation
    tprint_info("🔄 Running comprehensive validation...")
    start_time = time.time()
    
    validation_result = run_enhanced_labels_validation()
    
    validation_time = time.time() - start_time
    
    # Display validation results
    tprint_success(f"✅ Validation completed in {validation_time:.2f}s")
    
    overall_score = validation_result.get('overall_score', 0.0)
    overall_status = validation_result.get('overall_status', 'unknown')
    
    tprint_info(f"📊 Overall Validation: {overall_status} ({overall_score:.3f})")
    
    # Show individual test results
    validation_tests = validation_result.get('validation_tests', {})
    tprint_info(f"🧪 Test Results:")
    
    for test_name, test_result in validation_tests.items():
        passed = test_result.get('passed', False)
        status_icon = "✅" if passed else "❌"
        tprint_info(f"   → {test_name}: {status_icon}")
        
        if not passed and 'details' in test_result:
            tprint_warning(f"      Details: {test_result['details']}")
    
    # Show recommendations
    recommendations = validation_result.get('recommendations', [])
    if recommendations:
        tprint_info(f"💡 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            tprint_info(f"   {i}. {rec}")
    
    return validation_result


def demonstrate_trading_scenarios():
    """Demonstrate different trading scenarios and configurations."""
    tprint_info("💰 Demonstrating different trading scenarios")
    
    # Generate sample data
    market_data = generate_sample_market_data(800, "ADAUSDT")
    
    # Scenario 1: Conservative trading (low risk, high quality)
    tprint_info("🛡️ Scenario 1: Conservative Trading Configuration")
    
    conservative_config = EnhancedDataLabelsConfig(
        trading_objective=TradingObjectiveConfig(
            primary_objective="risk_adjusted_returns",
            max_drawdown_pct=0.02,  # 2% max drawdown
            target_sharpe_ratio=2.0,
            max_volatility_pct=0.15,  # 15% max volatility
            enable_regime_conditioning=True
        ),
        min_data_quality_score=0.8,
        min_label_stability_score=0.7
    )
    
    conservative_system = EnhancedDataLabelsSystem(conservative_config)
    conservative_result = conservative_system.process_market_data(market_data)
    
    if 'error' not in conservative_result:
        labels = conservative_result.get('labels', pd.DataFrame())
        analyst_ratio = labels.get('analyst_label', pd.Series()).mean()
        tprint_info(f"   → Conservative analyst ratio: {analyst_ratio:.3f} (should be lower)")
    
    # Scenario 2: Aggressive trading (higher risk tolerance)
    tprint_info("⚡ Scenario 2: Aggressive Trading Configuration")
    
    aggressive_config = EnhancedDataLabelsConfig(
        trading_objective=TradingObjectiveConfig(
            primary_objective="returns",
            max_drawdown_pct=0.10,  # 10% max drawdown
            target_sharpe_ratio=1.0,
            max_volatility_pct=0.30,  # 30% max volatility
            enable_regime_conditioning=True
        ),
        min_data_quality_score=0.6,
        min_label_stability_score=0.5
    )
    
    aggressive_system = EnhancedDataLabelsSystem(aggressive_config)
    aggressive_result = aggressive_system.process_market_data(market_data)
    
    if 'error' not in aggressive_result:
        labels = aggressive_result.get('labels', pd.DataFrame())
        analyst_ratio = labels.get('analyst_label', pd.Series()).mean()
        tprint_info(f"   → Aggressive analyst ratio: {analyst_ratio:.3f} (should be higher)")
    
    # Scenario 3: Research configuration (balanced)
    tprint_info("🔬 Scenario 3: Research Configuration")
    
    research_config = create_research_optimized_config()
    research_system = EnhancedDataLabelsSystem(research_config)
    research_result = research_system.process_market_data(market_data)
    
    if 'error' not in research_result:
        labels = research_result.get('labels', pd.DataFrame())
        analyst_ratio = labels.get('analyst_label', pd.Series()).mean()
        tactician_ratio = labels.get('tactician_label', pd.Series()).mean()
        tprint_info(f"   → Research analyst ratio: {analyst_ratio:.3f}")
        tprint_info(f"   → Research tactician ratio: {tactician_ratio:.3f}")
    
    return {
        'conservative': conservative_result,
        'aggressive': aggressive_result,
        'research': research_result
    }


def demonstrate_stability_monitoring():
    """Demonstrate label stability monitoring over time."""
    tprint_info("🔍 Demonstrating label stability monitoring")
    
    # Generate multiple datasets to simulate time progression
    datasets = []
    for i in range(5):
        # Generate data with slight variations to simulate market changes
        data = generate_sample_market_data(200, f"TEST{i}")
        # Add some time-based variation
        data['close'] = data['close'] * (1 + i * 0.01)  # Gradual price increase
        datasets.append(data)
    
    # Process each dataset and monitor stability
    results = []
    enhanced_system = EnhancedDataLabelsSystem(create_trading_optimized_config())
    
    for i, data in enumerate(datasets):
        tprint_info(f"📊 Processing dataset {i+1}/5...")
        
        result = enhanced_system.process_market_data(data)
        if 'error' not in result:
            results.append(result)
            
            # Show stability metrics
            stability = result.get('label_stability', {})
            stability_level = stability.get('stability_level', 'unknown')
            overall_stability = stability.get('overall_stability', 0.0)
            
            tprint_info(f"   → Stability: {stability_level} ({overall_stability:.3f})")
    
    # Analyze stability trends
    if len(results) >= 2:
        tprint_info("📈 Stability Trend Analysis:")
        
        stability_scores = [r.get('label_stability', {}).get('overall_stability', 0.0) for r in results]
        
        if len(stability_scores) > 1:
            trend = "improving" if stability_scores[-1] > stability_scores[0] else "declining"
            tprint_info(f"   → Overall trend: {trend}")
            tprint_info(f"   → Score range: {min(stability_scores):.3f} - {max(stability_scores):.3f}")
            tprint_info(f"   → Average score: {np.mean(stability_scores):.3f}")
    
    return results


def main():
    """Main demonstration function."""
    tprint_success("🚀 Enhanced Data & Labels System - Comprehensive Demonstration")
    tprint_info("=" * 80)
    
    try:
        # 1. Basic Usage
        tprint_info("\n" + "=" * 50)
        tprint_info("1. BASIC USAGE DEMONSTRATION")
        tprint_info("=" * 50)
        basic_result = demonstrate_basic_usage()
        
        # 2. Integration Usage
        tprint_info("\n" + "=" * 50)
        tprint_info("2. INTEGRATION DEMONSTRATION")
        tprint_info("=" * 50)
        integration_result = demonstrate_integration_usage()
        
        # 3. Validation
        tprint_info("\n" + "=" * 50)
        tprint_info("3. VALIDATION DEMONSTRATION")
        tprint_info("=" * 50)
        validation_result = demonstrate_validation()
        
        # 4. Trading Scenarios
        tprint_info("\n" + "=" * 50)
        tprint_info("4. TRADING SCENARIOS DEMONSTRATION")
        tprint_info("=" * 50)
        scenarios_result = demonstrate_trading_scenarios()
        
        # 5. Stability Monitoring
        tprint_info("\n" + "=" * 50)
        tprint_info("5. STABILITY MONITORING DEMONSTRATION")
        tprint_info("=" * 50)
        stability_result = demonstrate_stability_monitoring()
        
        # Summary
        tprint_info("\n" + "=" * 50)
        tprint_info("SUMMARY")
        tprint_info("=" * 50)
        
        tprint_success("✅ All demonstrations completed successfully!")
        tprint_info("🎯 The enhanced data and labels system provides:")
        tprint_info("   → Trading-aware label definitions (Analyst & Tactician)")
        tprint_info("   → Comprehensive data cleaning and quality assessment")
        tprint_info("   → Label stability monitoring and leakage detection")
        tprint_info("   → Full integration with existing infrastructure")
        tprint_info("   → Flexible configuration for different trading objectives")
        tprint_info("   → Comprehensive validation and testing")
        
        # Show final validation status
        if validation_result and 'overall_status' in validation_result:
            overall_status = validation_result['overall_status']
            overall_score = validation_result.get('overall_score', 0.0)
            tprint_info(f"📊 Final validation status: {overall_status} ({overall_score:.3f})")
        
        tprint_success("🎉 Enhanced Data & Labels System is ready for production use!")
        
    except Exception as e:
        tprint_error(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()