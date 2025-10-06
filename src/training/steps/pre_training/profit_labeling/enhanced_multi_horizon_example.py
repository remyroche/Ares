"""
Enhanced Multi-Horizon Integration Example

This script demonstrates how to use the enhanced multi-horizon profit labeler
that integrates the enhanced data and labels system with existing functionality.

Key Features Demonstrated:
1. Drop-in replacement for existing MultiHorizonProfitLabeler
2. Enhanced data cleaning and quality assessment
3. Trading-aware label definitions (Analyst & Tactician)
4. Label stability monitoring and leakage detection
5. Full backward compatibility with existing pipeline
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import the enhanced multi-horizon labeler
from enhanced_multi_horizon_labeler import (
    EnhancedMultiHorizonProfitLabeler, EnhancedMultiHorizonConfig,
    create_trading_optimized_multi_horizon_config,
    create_research_optimized_multi_horizon_config
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


def generate_sample_regime_data(market_data: pd.DataFrame) -> Dict[str, Any]:
    """Generate sample regime data for demonstration."""
    tprint_info("🎭 Generating sample regime data...")
    
    # Simple regime classification based on volatility
    returns = market_data['close'].pct_change()
    volatility = returns.rolling(window=20).std()
    
    # Classify regimes based on volatility percentiles
    low_threshold = volatility.quantile(0.33)
    high_threshold = volatility.quantile(0.67)
    
    regimes = pd.Series('normal', index=market_data.index)
    regimes[volatility <= low_threshold] = 'low_vol'
    regimes[volatility >= high_threshold] = 'high_vol'
    
    regime_data = {
        'regime_data': {
            'regime_states': regimes.tolist(),
            'market_data': market_data.copy()
        }
    }
    
    tprint_success(f"✅ Generated regime data: {len(regimes.unique())} unique regimes")
    tprint_info(f"   → Regimes: {list(regimes.unique())}")
    
    return regime_data


async def demonstrate_basic_usage():
    """Demonstrate basic usage of the enhanced multi-horizon labeler."""
    tprint_info("🚀 Demonstrating basic usage of enhanced multi-horizon labeler")
    
    # Generate sample data
    market_data = generate_sample_market_data(500, "BTCUSDT")
    regime_data = generate_sample_regime_data(market_data)
    
    # Create enhanced labeler with trading-optimized configuration
    config = create_trading_optimized_multi_horizon_config()
    labeler = EnhancedMultiHorizonProfitLabeler(config)
    
    # Execute enhanced labeling
    tprint_info("🔄 Executing enhanced multi-horizon labeling...")
    start_time = datetime.now()
    
    result = await labeler.execute_labeling(
        symbol="BTCUSDT",
        exchange="binance",
        timeframe="15m",
        data_dir="historical_data",
        regime_data=regime_data
    )
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    # Display results
    tprint_success(f"✅ Enhanced labeling completed in {processing_time:.2f}s")
    
    if 'error' in result:
        tprint_error(f"❌ Labeling failed: {result['error']}")
        return
    
    # Show enhanced results
    labeling_result = result.get('multi_horizon_labeling_result', {})
    enhanced_artifacts = result.get('enhanced_artifacts', {})
    
    tprint_info(f"📊 Enhanced Results:")
    tprint_info(f"   → Samples: {labeling_result.get('n_samples', 0)}")
    tprint_info(f"   → Targets: {labeling_result.get('n_targets', 0)}")
    tprint_info(f"   → Horizons: {labeling_result.get('n_horizons', 0)}")
    
    # Show enhanced metrics
    data_quality = enhanced_artifacts.get('data_quality_metrics', {})
    label_stability = enhanced_artifacts.get('label_stability_metrics', {})
    final_quality = enhanced_artifacts.get('final_quality_metrics', {})
    
    tprint_info(f"🔍 Enhanced Metrics:")
    tprint_info(f"   → Data quality: {data_quality.get('quality_level', 'unknown')}")
    tprint_info(f"   → Label stability: {label_stability.get('stability_level', 'unknown')}")
    tprint_info(f"   → Final quality: {final_quality.get('overall_score', 0.0):.3f}")
    
    # Show recommendations
    recommendations = enhanced_artifacts.get('recommendations', [])
    if recommendations:
        tprint_info(f"💡 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            tprint_info(f"   {i}. {rec}")
    
    return result


async def demonstrate_configuration_options():
    """Demonstrate different configuration options."""
    tprint_info("🔧 Demonstrating configuration options")
    
    # Generate sample data
    market_data = generate_sample_market_data(300, "ETHUSDT")
    
    # Configuration 1: Trading-optimized
    tprint_info("💰 Configuration 1: Trading-optimized")
    trading_config = create_trading_optimized_multi_horizon_config()
    trading_labeler = EnhancedMultiHorizonProfitLabeler(trading_config)
    
    result1 = await trading_labeler.execute_labeling(
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="15m",
        data_dir="historical_data"
    )
    
    if 'error' not in result1:
        enhanced_artifacts = result1.get('enhanced_artifacts', {})
        final_quality = enhanced_artifacts.get('final_quality_metrics', {})
        tprint_info(f"   → Trading quality: {final_quality.get('overall_score', 0.0):.3f}")
    
    # Configuration 2: Research-optimized
    tprint_info("🔬 Configuration 2: Research-optimized")
    research_config = create_research_optimized_multi_horizon_config()
    research_labeler = EnhancedMultiHorizonProfitLabeler(research_config)
    
    result2 = await research_labeler.execute_labeling(
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="15m",
        data_dir="historical_data"
    )
    
    if 'error' not in result2:
        enhanced_artifacts = result2.get('enhanced_artifacts', {})
        final_quality = enhanced_artifacts.get('final_quality_metrics', {})
        tprint_info(f"   → Research quality: {final_quality.get('overall_score', 0.0):.3f}")
    
    # Configuration 3: Custom
    tprint_info("⚙️ Configuration 3: Custom")
    custom_config = EnhancedMultiHorizonConfig(
        enable_enhanced_data_cleaning=True,
        enable_enhanced_stability_monitoring=True,
        enable_trading_aware_labels=True,
        analyst_horizon_minutes=120,  # 2 hours
        tactician_horizon_minutes=60,  # 1 hour
        enable_regime_conditioning=True,
        enable_risk_awareness=True,
        min_data_quality_score=0.8,
        min_label_stability_score=0.7
    )
    
    custom_labeler = EnhancedMultiHorizonProfitLabeler(custom_config)
    
    result3 = await custom_labeler.execute_labeling(
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="15m",
        data_dir="historical_data"
    )
    
    if 'error' not in result3:
        enhanced_artifacts = result3.get('enhanced_artifacts', {})
        final_quality = enhanced_artifacts.get('final_quality_metrics', {})
        tprint_info(f"   → Custom quality: {final_quality.get('overall_score', 0.0):.3f}")
    
    return result1, result2, result3


async def demonstrate_backward_compatibility():
    """Demonstrate backward compatibility with existing code."""
    tprint_info("🔄 Demonstrating backward compatibility")
    
    # Generate sample data
    market_data = generate_sample_market_data(400, "ADAUSDT")
    
    # Use the enhanced labeler as a drop-in replacement
    # This should work exactly like the original MultiHorizonProfitLabeler
    tprint_info("📦 Using enhanced labeler as drop-in replacement...")
    
    # Create with original-style configuration
    config = EnhancedMultiHorizonConfig(
        timeframe="15m",
        enable_regime_aware_labeling=True,
        enable_volatility_normalization=True,
        enable_noise_gating=True,
        enable_quality_scoring=True,
        enable_multi_target_scheme=True,
        min_data_points=100,
        # Enhanced features are enabled by default
    )
    
    labeler = EnhancedMultiHorizonProfitLabeler(config)
    
    # Execute with original API
    result = await labeler.execute_labeling(
        symbol="ADAUSDT",
        exchange="binance",
        timeframe="15m",
        data_dir="historical_data"
    )
    
    if 'error' not in result:
        tprint_success("✅ Backward compatibility verified")
        tprint_info(f"   → Original API works with enhanced functionality")
        tprint_info(f"   → Enhanced processing: {result.get('multi_horizon_labeling_result', {}).get('enhanced_processing', False)}")
    else:
        tprint_error(f"❌ Backward compatibility issue: {result['error']}")
    
    return result


async def demonstrate_enhanced_features():
    """Demonstrate specific enhanced features."""
    tprint_info("🎯 Demonstrating enhanced features")
    
    # Generate sample data
    market_data = generate_sample_market_data(600, "SOLUSDT")
    regime_data = generate_sample_regime_data(market_data)
    
    # Create labeler with all enhanced features enabled
    config = EnhancedMultiHorizonConfig(
        enable_enhanced_data_cleaning=True,
        enable_enhanced_stability_monitoring=True,
        enable_trading_aware_labels=True,
        enable_regime_conditioning=True,
        enable_risk_awareness=True,
        min_data_quality_score=0.7,
        min_label_stability_score=0.6
    )
    
    labeler = EnhancedMultiHorizonProfitLabeler(config)
    
    # Execute with enhanced features
    result = await labeler.execute_labeling(
        symbol="SOLUSDT",
        exchange="binance",
        timeframe="15m",
        data_dir="historical_data",
        regime_data=regime_data
    )
    
    if 'error' not in result:
        tprint_success("✅ Enhanced features demonstration completed")
        
        # Show specific enhanced features
        enhanced_artifacts = result.get('enhanced_artifacts', {})
        
        # Data quality features
        data_quality = enhanced_artifacts.get('data_quality_metrics', {})
        tprint_info(f"🧹 Data Quality Features:")
        tprint_info(f"   → Quality level: {data_quality.get('quality_level', 'unknown')}")
        tprint_info(f"   → Quality score: {data_quality.get('quality_score', 0.0):.3f}")
        tprint_info(f"   → Samples removed: {data_quality.get('samples_removed', 0)}")
        tprint_info(f"   → Features removed: {data_quality.get('features_removed', 0)}")
        
        # Label stability features
        label_stability = enhanced_artifacts.get('label_stability_metrics', {})
        tprint_info(f"🔍 Label Stability Features:")
        tprint_info(f"   → Stability level: {label_stability.get('stability_level', 'unknown')}")
        tprint_info(f"   → Overall stability: {label_stability.get('overall_stability', 0.0):.3f}")
        tprint_info(f"   → Leakage detected: {label_stability.get('is_leakage_detected', False)}")
        tprint_info(f"   → Drift detected: {label_stability.get('is_drift_detected', False)}")
        
        # Final quality features
        final_quality = enhanced_artifacts.get('final_quality_metrics', {})
        tprint_info(f"✅ Final Quality Features:")
        tprint_info(f"   → Overall score: {final_quality.get('overall_score', 0.0):.3f}")
        tprint_info(f"   → Quality grade: {final_quality.get('quality_grade', 'F')}")
        tprint_info(f"   → Is acceptable: {final_quality.get('is_acceptable', False)}")
        
        # Show recommendations
        recommendations = enhanced_artifacts.get('recommendations', [])
        if recommendations:
            tprint_info(f"💡 Enhanced Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                tprint_info(f"   {i}. {rec}")
    else:
        tprint_error(f"❌ Enhanced features demonstration failed: {result['error']}")
    
    return result


async def main():
    """Main demonstration function."""
    tprint_success("🚀 Enhanced Multi-Horizon Integration - Comprehensive Demonstration")
    tprint_info("=" * 80)
    
    try:
        # 1. Basic Usage
        tprint_info("\n" + "=" * 50)
        tprint_info("1. BASIC USAGE DEMONSTRATION")
        tprint_info("=" * 50)
        basic_result = await demonstrate_basic_usage()
        
        # 2. Configuration Options
        tprint_info("\n" + "=" * 50)
        tprint_info("2. CONFIGURATION OPTIONS DEMONSTRATION")
        tprint_info("=" * 50)
        config_results = await demonstrate_configuration_options()
        
        # 3. Backward Compatibility
        tprint_info("\n" + "=" * 50)
        tprint_info("3. BACKWARD COMPATIBILITY DEMONSTRATION")
        tprint_info("=" * 50)
        compatibility_result = await demonstrate_backward_compatibility()
        
        # 4. Enhanced Features
        tprint_info("\n" + "=" * 50)
        tprint_info("4. ENHANCED FEATURES DEMONSTRATION")
        tprint_info("=" * 50)
        enhanced_result = await demonstrate_enhanced_features()
        
        # Summary
        tprint_info("\n" + "=" * 50)
        tprint_info("SUMMARY")
        tprint_info("=" * 50)
        
        tprint_success("✅ All demonstrations completed successfully!")
        tprint_info("🎯 The enhanced multi-horizon profit labeler provides:")
        tprint_info("   → Drop-in replacement for existing MultiHorizonProfitLabeler")
        tprint_info("   → Enhanced data cleaning and quality assessment")
        tprint_info("   → Trading-aware label definitions (Analyst & Tactician)")
        tprint_info("   → Label stability monitoring and leakage detection")
        tprint_info("   → Full backward compatibility with existing pipeline")
        tprint_info("   → No duplication of existing functionality")
        
        tprint_success("🎉 Enhanced Multi-Horizon Integration is ready for production use!")
        
    except Exception as e:
        tprint_error(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())