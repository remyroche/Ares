"""
Example usage of NAS-driven clustering for short-term trading.

This example demonstrates how to use the NAS clustering module
for regime detection in short-term trading scenarios.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Import NAS clustering components
from src.training.steps.market_analysis.nas_clustering import (
    NASOrchestrator,
    NASClusterer,
    NASClusteringConfig,
    NASFeatureExtractor,
    MicroRegimeDetector,
    NASOutputFormatter,
    NASMetrics
)


def create_sample_market_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample market data for testing."""
    np.random.seed(42)
    
    # Generate timestamps
    start_time = datetime.now() - timedelta(hours=24)
    timestamps = [start_time + timedelta(minutes=15*i) for i in range(n_samples)]
    
    # Generate OHLCV data
    base_price = 50000
    prices = [base_price]
    
    for i in range(1, n_samples):
        # Add some trend and volatility
        trend = np.sin(i * 0.1) * 0.001
        volatility = np.random.normal(0, 0.02)
        price_change = trend + volatility
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        # Generate OHLC from price
        open_price = price
        high_price = price * (1 + abs(np.random.normal(0, 0.01)))
        low_price = price * (1 - abs(np.random.normal(0, 0.01)))
        close_price = price * (1 + np.random.normal(0, 0.005))
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'timestamp': timestamps[i],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    return pd.DataFrame(data)


async def example_basic_usage():
    """Example of basic NAS clustering usage."""
    print("🚀 Example 1: Basic NAS Clustering Usage")
    print("=" * 50)
    
    # Create sample data
    market_data = create_sample_market_data(1000)
    timestamps = market_data['timestamp'].values
    data_array = market_data[['open', 'high', 'low', 'close', 'volume']].values
    
    # Initialize NAS orchestrator
    config = {
        'symbol': 'BTCUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'data_dir': 'historical_data',
        'nas_config': {
            'n_regimes': 12,
            'enable_micro_regime_detection': True,
            'economic_significance_threshold': 0.7,
            'trading_viability_threshold': 0.6
        }
    }
    
    orchestrator = NASOrchestrator(config)
    
    # Run NAS clustering
    results = await orchestrator.run_nas_clustering(
        data=data_array,
        timestamps=timestamps,
        symbol='BTCUSDT',
        exchange='binance',
        timeframe='15m'
    )
    
    # Print results
    if results['success']:
        print(f"✅ NAS clustering completed in {results['execution_time']:.2f}s")
        print(f"📊 Detected {len(np.unique(results['clustering_result'].labels))} regimes")
        print(f"🔍 Micro-regimes: {len(results['micro_regime_result'].micro_regime_types)}")
        print(f"💰 Economic significance: {np.mean(results['clustering_result'].economic_significance_scores):.3f}")
        print(f"📈 Trading viability: {np.mean(results['clustering_result'].trading_viability_scores):.3f}")
        
        # Save results
        orchestrator.save_results(results, 'output/nas_clustering_results')
        print("💾 Results saved to output/nas_clustering_results/")
    else:
        print(f"❌ NAS clustering failed: {results.get('error', 'Unknown error')}")


async def example_advanced_usage():
    """Example of advanced NAS clustering usage with custom configuration."""
    print("\n🧠 Example 2: Advanced NAS Clustering Usage")
    print("=" * 50)
    
    # Create sample data
    market_data = create_sample_market_data(2000)
    timestamps = market_data['timestamp'].values
    data_array = market_data[['open', 'high', 'low', 'close', 'volume']].values
    
    # Create custom NAS configuration
    nas_config = NASClusteringConfig.create_short_term_trading_config()
    nas_config.n_regimes = 15  # More regimes for detailed analysis
    nas_config.enable_micro_regime_detection = True
    nas_config.micro_regime_sensitivity = 0.8  # Higher sensitivity
    nas_config.economic_significance_threshold = 0.8  # Higher threshold
    nas_config.trading_viability_threshold = 0.7  # Higher threshold
    
    # Initialize components
    clusterer = NASClusterer(nas_config)
    feature_extractor = NASFeatureExtractor(nas_config.get_feature_config())
    micro_regime_detector = MicroRegimeDetector(nas_config.get_micro_regime_config())
    
    # Extract features
    print("📊 Extracting NAS features...")
    feature_result = feature_extractor.extract_features(data_array, timestamps)
    print(f"✅ Extracted {len(feature_result.feature_names)} features")
    
    # Detect micro-regimes
    print("🔍 Detecting micro-regimes...")
    micro_regime_result = micro_regime_detector.detect_micro_regimes(
        data_array, timestamps, feature_result.features
    )
    print(f"✅ Detected {len(micro_regime_result.micro_regime_types)} micro-regime types")
    
    # Perform NAS clustering
    print("🧠 Performing NAS clustering...")
    clustering_result = clusterer.cluster(
        data_array, timestamps, optimize_parameters=True, generate_report=True
    )
    
    if clustering_result.success:
        print(f"✅ NAS clustering completed in {clustering_result.execution_time:.2f}s")
        print(f"📊 Detected {len(np.unique(clustering_result.labels))} regimes")
        print(f"🎯 NAS Score: {clustering_result.quality_metrics.get('nas_score', 0.0):.3f}")
        print(f"📈 Silhouette Score: {clustering_result.quality_metrics.get('silhouette_score', 0.0):.3f}")
        
        # Calculate comprehensive metrics
        print("📊 Calculating comprehensive metrics...")
        metrics_calculator = NASMetrics({})
        metrics_result = metrics_calculator.calculate_metrics(
            feature_result.features,
            clustering_result.labels,
            clustering_result.economic_significance_scores,
            clustering_result.trading_viability_scores,
            micro_regime_result.detection_accuracy
        )
        
        print(f"💰 Economic Significance: {metrics_result.economic_significance_score:.3f}")
        print(f"📈 Trading Viability: {metrics_result.trading_viability_score:.3f}")
        print(f"🔒 Regime Stability: {metrics_result.regime_stability_score:.3f}")
        print(f"🔍 Regime Separation: {metrics_result.regime_separation_score:.3f}")
        print(f"🎯 Regime Consistency: {metrics_result.regime_consistency_score:.3f}")
        print(f"🔬 Micro-Regime Accuracy: {metrics_result.micro_regime_detection_accuracy:.3f}")
        
        # Evaluate clustering quality
        quality_evaluation = metrics_calculator.evaluate_clustering_quality(metrics_result)
        print(f"\n📊 Overall Quality: {quality_evaluation['overall_quality']}")
        print(f"📈 Overall Score: {quality_evaluation['overall_score']:.3f}")
        
        if quality_evaluation['recommendations']:
            print("\n💡 Recommendations:")
            for recommendation in quality_evaluation['recommendations']:
                print(f"   - {recommendation}")
    else:
        print(f"❌ NAS clustering failed: {clustering_result.error_message}")


async def example_pipeline_integration():
    """Example of pipeline integration with existing HMM clustering."""
    print("\n🔗 Example 3: Pipeline Integration")
    print("=" * 50)
    
    # Create sample data
    market_data = create_sample_market_data(1500)
    timestamps = market_data['timestamp'].values
    data_array = market_data[['open', 'high', 'low', 'close', 'volume']].values
    
    # Initialize NAS clustering component (compatible with existing pipeline)
    from src.training.steps.market_analysis.nas_clustering import NASClusteringComponent
    
    config = {
        'symbol': 'BTCUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'data_dir': 'historical_data',
        'nas_config': {
            'n_regimes': 12,
            'enable_micro_regime_detection': True,
            'economic_significance_threshold': 0.7,
            'trading_viability_threshold': 0.6
        }
    }
    
    # Simulate pipeline state
    pipeline_state = {
        'previous_clustering_result': None,
        'feature_extraction_completed': True,
        'data_preprocessing_completed': True
    }
    
    # Create NAS clustering component
    component = NASClusteringComponent(config)
    
    # Execute component (compatible with existing pipeline)
    print("🔗 Executing NAS clustering component...")
    result = await component.execute(data_array, pipeline_state)
    
    if result['success']:
        print(f"✅ Component execution completed in {result['execution_time']:.2f}s")
        print(f"📊 Pipeline compatible: {result.get('pipeline_compatible', False)}")
        print(f"🔗 HMM compatible: {result.get('hmm_compatible', False)}")
        print(f"📊 Regime data available: {result.get('regime_data_available', False)}")
        print(f"🤖 LM training ready: {result.get('lm_training_ready', False)}")
        
        # Access timestamped regime data for LM training
        if 'timestamped_regime_data' in result:
            timestamped_data = result['timestamped_regime_data']
            print(f"\n🤖 LM Training Data:")
            print(f"   - Regime sequences: {len(timestamped_data['regime_data']['regime_labels'])}")
            print(f"   - Micro-regime sequences: {len(timestamped_data['micro_regime_data']['micro_regimes'])}")
            print(f"   - Economic significance: {len(timestamped_data['economic_data']['economic_significance_scores'])}")
            print(f"   - Trading viability: {len(timestamped_data['economic_data']['trading_viability_scores'])}")
    else:
        print(f"❌ Component execution failed: {result.get('error', 'Unknown error')}")


async def example_micro_regime_analysis():
    """Example of micro-regime analysis."""
    print("\n🔬 Example 4: Micro-Regime Analysis")
    print("=" * 50)
    
    # Create sample data with different market conditions
    market_data = create_sample_market_data(2000)
    timestamps = market_data['timestamp'].values
    data_array = market_data[['open', 'high', 'low', 'close', 'volume']].values
    
    # Add some artificial micro-regime patterns
    # Add volume spikes
    volume_spike_indices = np.random.choice(len(data_array), 50, replace=False)
    for idx in volume_spike_indices:
        data_array[idx, 4] *= 3.0  # 3x volume spike
    
    # Add volatility spikes
    volatility_spike_indices = np.random.choice(len(data_array), 30, replace=False)
    for idx in volatility_spike_indices:
        data_array[idx, 1] *= 1.05  # 5% high spike
        data_array[idx, 2] *= 0.95  # 5% low spike
    
    # Initialize micro-regime detector
    micro_regime_config = {
        'enable_micro_regime_detection': True,
        'micro_regime_sensitivity': 0.7,
        'micro_regime_types': [
            'breakout', 'consolidation', 'reversal',
            'acceleration', 'volume_spike', 'volatility_spike'
        ],
        'micro_timeframe': '5m'
    }
    
    micro_regime_detector = MicroRegimeDetector(micro_regime_config)
    
    # Detect micro-regimes
    print("🔍 Detecting micro-regimes...")
    micro_regime_result = micro_regime_detector.detect_micro_regimes(
        data_array, timestamps
    )
    
    if micro_regime_result.micro_regimes.size > 0:
        print(f"✅ Micro-regime detection completed in {micro_regime_result.execution_time:.2f}s")
        print(f"🔍 Detection accuracy: {micro_regime_result.detection_accuracy:.3f}")
        print(f"📊 Micro-regime types: {[t.value for t in micro_regime_result.micro_regime_types]}")
        
        # Analyze micro-regime distribution
        unique_micro_regimes = np.unique(micro_regime_result.micro_regimes)
        print(f"\n📊 Micro-Regime Distribution:")
        for regime_id in unique_micro_regimes:
            count = np.sum(micro_regime_result.micro_regimes == regime_id)
            percentage = (count / len(micro_regime_result.micro_regimes)) * 100
            regime_type = micro_regime_result.micro_regime_types[regime_id].value if regime_id < len(micro_regime_result.micro_regime_types) else 'Unknown'
            print(f"   - {regime_type}: {count} ({percentage:.1f}%)")
        
        # Analyze micro-regime scores
        print(f"\n📈 Micro-Regime Quality Scores:")
        for regime_id in unique_micro_regimes:
            regime_mask = micro_regime_result.micro_regimes == regime_id
            regime_scores = micro_regime_result.micro_regime_scores[regime_mask]
            mean_score = np.mean(regime_scores)
            regime_type = micro_regime_result.micro_regime_types[regime_id].value if regime_id < len(micro_regime_result.micro_regime_types) else 'Unknown'
            print(f"   - {regime_type}: {mean_score:.3f}")
    else:
        print("❌ No micro-regimes detected")


async def main():
    """Run all examples."""
    print("🎯 NAS-Driven Clustering Examples")
    print("=" * 60)
    
    try:
        # Run examples
        await example_basic_usage()
        await example_advanced_usage()
        await example_pipeline_integration()
        await example_micro_regime_analysis()
        
        print("\n✅ All examples completed successfully!")
        print("📚 For more information, see the README.md file")
        
    except Exception as e:
        print(f"\n❌ Example execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())