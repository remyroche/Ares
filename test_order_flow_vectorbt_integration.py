#!/usr/bin/env python3
"""
Test script for VectorBT integration in order flow features.

This script validates that the VectorBT integration is working correctly
and provides performance comparisons.
"""

import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(n_points: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_points, freq='1min')
    
    # Generate realistic price data
    price = 100 + np.cumsum(np.random.randn(n_points) * 0.01)
    volume = np.random.lognormal(10, 1, n_points)
    
    data = pd.DataFrame({
        'open': price + np.random.randn(n_points) * 0.001,
        'high': price + np.abs(np.random.randn(n_points)) * 0.002,
        'low': price - np.abs(np.random.randn(n_points)) * 0.002,
        'close': price,
        'volume': volume,
        'bid': price - np.random.rand(n_points) * 0.001,
        'ask': price + np.random.rand(n_points) * 0.001,
        'market_buys': volume * np.random.rand(n_points),
        'market_sells': volume * np.random.rand(n_points)
    }, index=dates)
    
    return data

def test_legacy_generators(data: pd.DataFrame) -> Dict[str, Any]:
    """Test legacy order flow generators."""
    logger.info("Testing legacy order flow generators...")
    
    try:
        from src.feature_generation.categories.order_flow import (
            TakerBuyRatioGenerator,
            TakerSellRatioGenerator,
            MarketAggressionIndexGenerator,
            OrderFlowImbalanceGenerator
        )
        
        start_time = time.time()
        
        # Test individual generators
        generators = [
            TakerBuyRatioGenerator(window=20),
            TakerSellRatioGenerator(window=20),
            MarketAggressionIndexGenerator(window=20),
            OrderFlowImbalanceGenerator(window=20)
        ]
        
        results = {}
        for generator in generators:
            feature_name = generator.config.name
            result = generator._generate_feature(data)
            results[feature_name] = result
            logger.info(f"✅ Generated {feature_name}: {len(result)} points")
        
        processing_time = time.time() - start_time
        
        return {
            'success': True,
            'processing_time': processing_time,
            'features_generated': len(results),
            'feature_names': list(results.keys())
        }
        
    except Exception as e:
        logger.error(f"❌ Legacy generators failed: {e}")
        return {'success': False, 'error': str(e)}

def test_vectorbt_generators(data: pd.DataFrame) -> Dict[str, Any]:
    """Test VectorBT-optimized order flow generators."""
    logger.info("Testing VectorBT-optimized order flow generators...")
    
    try:
        from src.feature_generation.categories.vectorbt_order_flow import (
            VectorBTTakerBuyRatioGenerator,
            VectorBTTakerSellRatioGenerator,
            VectorBTMarketAggressionIndexGenerator,
            VectorBTOrderFlowImbalanceGenerator
        )
        
        start_time = time.time()
        
        # Test individual generators
        generators = [
            VectorBTTakerBuyRatioGenerator(window=20),
            VectorBTTakerSellRatioGenerator(window=20),
            VectorBTMarketAggressionIndexGenerator(window=20),
            VectorBTOrderFlowImbalanceGenerator(window=20)
        ]
        
        results = {}
        for generator in generators:
            feature_name = generator.config.name
            result = generator._generate_feature(data)
            results[feature_name] = result
            logger.info(f"✅ Generated {feature_name}: {len(result)} points")
        
        processing_time = time.time() - start_time
        
        return {
            'success': True,
            'processing_time': processing_time,
            'features_generated': len(results),
            'feature_names': list(results.keys())
        }
        
    except Exception as e:
        logger.error(f"❌ VectorBT generators failed: {e}")
        return {'success': False, 'error': str(e)}

def test_batch_processing(data: pd.DataFrame) -> Dict[str, Any]:
    """Test batch processing with unified vectorization manager."""
    logger.info("Testing batch processing with unified vectorization manager...")
    
    try:
        from src.feature_generation.categories.order_flow import process_order_flow_features_batch
        
        # Define batch processing configuration
        feature_configs = [
            {'name': 'taker_buy_ratio_5', 'type': 'taker_buy_ratio', 'window': 5, 'column': 'close'},
            {'name': 'taker_buy_ratio_20', 'type': 'taker_buy_ratio', 'window': 20, 'column': 'close'},
            {'name': 'taker_sell_ratio_5', 'type': 'taker_sell_ratio', 'window': 5, 'column': 'close'},
            {'name': 'taker_sell_ratio_20', 'type': 'taker_sell_ratio', 'window': 20, 'column': 'close'},
            {'name': 'market_aggression_10', 'type': 'market_aggression_index', 'window': 10, 'column': 'close'},
            {'name': 'market_aggression_20', 'type': 'market_aggression_index', 'window': 20, 'column': 'close'},
            {'name': 'order_flow_imbalance_10', 'type': 'order_flow_imbalance', 'window': 10, 'column': 'close'},
            {'name': 'order_flow_imbalance_20', 'type': 'order_flow_imbalance', 'window': 20, 'column': 'close'},
        ]
        
        start_time = time.time()
        
        # Process batch
        result_df = process_order_flow_features_batch(
            data, 
            feature_configs,
            enable_gpu=False,
            enable_parallel=True
        )
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Batch processing completed: {len(result_df.columns)} features in {processing_time:.3f}s")
        
        return {
            'success': True,
            'processing_time': processing_time,
            'features_generated': len(result_df.columns),
            'feature_names': list(result_df.columns),
            'data_shape': result_df.shape
        }
        
    except Exception as e:
        logger.error(f"❌ Batch processing failed: {e}")
        return {'success': False, 'error': str(e)}

def test_unified_vectorization_manager(data: pd.DataFrame) -> Dict[str, Any]:
    """Test the unified vectorization manager directly."""
    logger.info("Testing unified vectorization manager...")
    
    try:
        from src.feature_generation.categories.vectorbt_order_flow import create_unified_vectorization_manager
        
        # Create manager
        manager = create_unified_vectorization_manager(enable_gpu=False, enable_parallel=True)
        
        # Define operations
        operations_config = [
            {'name': 'close_mean_5', 'operation': 'mean', 'window': 5, 'column': 'close'},
            {'name': 'close_mean_20', 'operation': 'mean', 'window': 20, 'column': 'close'},
            {'name': 'close_std_5', 'operation': 'std', 'window': 5, 'column': 'close'},
            {'name': 'close_std_20', 'operation': 'std', 'window': 20, 'column': 'close'},
            {'name': 'volume_sum_5', 'operation': 'sum', 'window': 5, 'column': 'volume'},
            {'name': 'volume_sum_20', 'operation': 'sum', 'window': 20, 'column': 'volume'},
        ]
        
        start_time = time.time()
        
        # Process operations
        result_df = manager.process_batch_rolling_operations(data, operations_config)
        
        processing_time = time.time() - start_time
        
        # Get statistics
        stats = manager.get_batch_stats()
        
        logger.info(f"✅ Unified vectorization manager completed: {len(result_df.columns)} features in {processing_time:.3f}s")
        logger.info(f"📊 Stats: {stats}")
        
        return {
            'success': True,
            'processing_time': processing_time,
            'features_generated': len(result_df.columns),
            'feature_names': list(result_df.columns),
            'stats': stats
        }
        
    except Exception as e:
        logger.error(f"❌ Unified vectorization manager failed: {e}")
        return {'success': False, 'error': str(e)}

def main():
    """Main test function."""
    logger.info("🚀 Starting VectorBT integration test for order flow features...")
    
    # Create test data
    data = create_sample_data(1000)
    logger.info(f"📊 Created test data: {data.shape}")
    
    # Run tests
    test_results = {}
    
    # Test 1: Legacy generators
    test_results['legacy'] = test_legacy_generators(data)
    
    # Test 2: VectorBT generators
    test_results['vectorbt'] = test_vectorbt_generators(data)
    
    # Test 3: Batch processing
    test_results['batch'] = test_batch_processing(data)
    
    # Test 4: Unified vectorization manager
    test_results['unified'] = test_unified_vectorization_manager(data)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("📋 TEST SUMMARY")
    logger.info("="*60)
    
    for test_name, result in test_results.items():
        if result['success']:
            logger.info(f"✅ {test_name.upper()}: SUCCESS")
            if 'processing_time' in result:
                logger.info(f"   ⏱️  Processing time: {result['processing_time']:.3f}s")
            if 'features_generated' in result:
                logger.info(f"   📈 Features generated: {result['features_generated']}")
        else:
            logger.info(f"❌ {test_name.upper()}: FAILED")
            if 'error' in result:
                logger.info(f"   🚨 Error: {result['error']}")
    
    # Performance comparison
    if test_results['legacy']['success'] and test_results['vectorbt']['success']:
        legacy_time = test_results['legacy']['processing_time']
        vectorbt_time = test_results['vectorbt']['processing_time']
        speedup = legacy_time / vectorbt_time if vectorbt_time > 0 else 0
        
        logger.info(f"\n🚀 PERFORMANCE COMPARISON")
        logger.info(f"   Legacy: {legacy_time:.3f}s")
        logger.info(f"   VectorBT: {vectorbt_time:.3f}s")
        logger.info(f"   Speedup: {speedup:.2f}x")
    
    logger.info("\n🎉 VectorBT integration test completed!")

if __name__ == "__main__":
    main()