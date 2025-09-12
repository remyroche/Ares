#!/usr/bin/env python3
"""
Market Analysis Integration Test

This script tests the full integration of the three main market analysis components:
1. SR Parameter Optimization - Optimize SR detection levels
2. SR Detection - Detect Support/Resistance levels  
3. SR Clustering - Generate SR clusters

The test verifies that all components work together seamlessly and produce expected outputs.
"""

import asyncio
import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add workspace to path
sys.path.append('/workspace')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_market_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample market data for testing."""
    logger.info(f"Creating sample market data with {n_samples} samples")
    
    np.random.seed(42)  # For reproducible results
    
    # Generate realistic price data
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
    
    # Create price trend with some volatility
    base_price = 50000
    trend = np.cumsum(np.random.normal(0, 0.001, n_samples))  # Random walk
    noise = np.random.normal(0, 0.002, n_samples)  # Additional noise
    
    close_prices = base_price * (1 + trend + noise)
    
    # Generate OHLC data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices + np.random.normal(0, 0.001, n_samples) * close_prices,
        'high': close_prices + np.abs(np.random.normal(0, 0.005, n_samples)) * close_prices,
        'low': close_prices - np.abs(np.random.normal(0, 0.005, n_samples)) * close_prices,
        'close': close_prices,
        'volume': 1000 + np.random.randint(0, 9000, n_samples)
    })
    
    # Ensure high >= low and high >= open/close, low <= open/close
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    logger.info(f"Sample data created: {data.shape[0]} rows, {data.shape[1]} columns")
    logger.info(f"Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
    
    return data

async def test_sr_parameter_optimization():
    """Test SR parameter optimization component."""
    logger.info("🧪 Testing SR Parameter Optimization Component")
    
    try:
        # Import the market analysis sub-pipeline
        from src.training.steps.market_analysis.sub_pipeline import (
            get_market_analysis_sub_pipeline, 
            SubPipelineConfig, 
            ExecutionMode
        )
        
        # Create configuration for parameter optimization
        config = SubPipelineConfig(
            mode=ExecutionMode.LIGHT,  # Use light mode for testing
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="1h"
        )
        
        # Create pipeline
        pipeline = get_market_analysis_sub_pipeline(config)
        
        # Create sample market data
        market_data = create_sample_market_data(500)
        
        # Store data in pipeline state
        pipeline_state = {'dataframe': market_data}
        training_input = {'training_mode': 'light'}
        
        # Execute parameter optimization
        logger.info("🎯 Executing SR Parameter Optimization...")
        result = await pipeline.execute_sub_pipeline('sr_parameter_optimization', config)
        
        if result.success:
            logger.info("✅ SR Parameter Optimization completed successfully")
            
            # Check for expected artifacts
            artifacts = result.artifacts
            if 'optimized_parameters' in artifacts:
                optimized_params = artifacts['optimized_parameters']
                logger.info(f"📊 Optimized parameters: {len(optimized_params)} parameters")
                
                # Verify key parameters exist
                expected_params = [
                    'touch_tolerance', 'min_bounce_strength', 'volume_threshold_multiplier',
                    'min_touches_required', 'max_hold_time'
                ]
                
                for param in expected_params:
                    if param in optimized_params:
                        logger.info(f"   ✓ {param}: {optimized_params[param]}")
                    else:
                        logger.warning(f"   ⚠️ Missing parameter: {param}")
            
            if 'quality_thresholds' in artifacts:
                quality_thresholds = artifacts['quality_thresholds']
                logger.info(f"📊 Quality thresholds: {len(quality_thresholds)} thresholds")
            
            return result
            
        else:
            logger.error(f"❌ SR Parameter Optimization failed: {result.error}")
            return None
            
    except Exception as e:
        logger.error(f"❌ SR Parameter Optimization test failed: {e}")
        import traceback
        logger.error(f"Error details: {traceback.format_exc()}")
        return None

async def test_sr_detection():
    """Test SR detection component."""
    logger.info("🧪 Testing SR Detection Component")
    
    try:
        # Import the SR detection step
        from src.training.steps.market_analysis.sr_detection import SRDetectionStep
        
        # Create configuration for SR detection
        sr_config = {
            'sr_optimization': {
                'min_touches': 2,
                'tolerance_pct': 0.5,
                'lookback_periods': 100,
                'proximity_threshold': 0.002,
                'min_sr_ratio': 0.15,
                'max_sr_ratio': 0.30
            },
            'training_mode': 'light'
        }
        
        # Create SR detection step
        sr_detection_step = SRDetectionStep(sr_config)
        
        # Create sample market data
        market_data = create_sample_market_data(1000)
        
        # Prepare pipeline state
        pipeline_state = {'dataframe': market_data}
        training_input = {'training_mode': 'light'}
        
        # Execute SR detection
        logger.info("🎯 Executing SR Detection...")
        result = await sr_detection_step.execute(training_input, pipeline_state)
        
        if result['success']:
            logger.info("✅ SR Detection completed successfully")
            
            # Check for expected outputs
            sr_levels = result.get('sr_levels', {})
            
            if 'support_levels' in sr_levels:
                support_count = len(sr_levels['support_levels'])
                logger.info(f"📊 Support levels detected: {support_count}")
            
            if 'resistance_levels' in sr_levels:
                resistance_count = len(sr_levels['resistance_levels'])
                logger.info(f"📊 Resistance levels detected: {resistance_count}")
            
            if 'all_levels' in sr_levels:
                total_count = len(sr_levels['all_levels'])
                logger.info(f"📊 Total SR levels: {total_count}")
            
            return result
            
        else:
            logger.error(f"❌ SR Detection failed: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        logger.error(f"❌ SR Detection test failed: {e}")
        import traceback
        logger.error(f"Error details: {traceback.format_exc()}")
        return None

async def test_sr_clustering():
    """Test SR clustering component."""
    logger.info("🧪 Testing SR Clustering Component")
    
    try:
        # Import the SR clustering step
        from src.training.steps.market_analysis.sr_clustering import SRClusteringStep
        
        # Create configuration for SR clustering
        sr_config = {
            'sr_optimization': {
                'min_touches': 2,
                'tolerance_pct': 0.5,
                'lookback_periods': 100
            },
            'sr_clustering': {
                'min_levels_for_learning': 5,
                'quality_filter_threshold': 0.1,
                'proximity_adjustment_factor': 0.5
            },
            'training_mode': 'light'
        }
        
        # Create SR clustering step
        sr_clustering_step = SRClusteringStep(sr_config)
        
        # Create mock SR levels for testing
        mock_sr_levels = {
            'all_levels': [
                {'price': 50000, 'level_type': 'support', 'strength': 0.8, 'touch_count': 3},
                {'price': 51000, 'level_type': 'resistance', 'strength': 0.7, 'touch_count': 2},
                {'price': 49500, 'level_type': 'support', 'strength': 0.9, 'touch_count': 4},
                {'price': 52000, 'level_type': 'resistance', 'strength': 0.6, 'touch_count': 2},
                {'price': 50500, 'level_type': 'support', 'strength': 0.75, 'touch_count': 3}
            ]
        }
        
        # Prepare pipeline state
        pipeline_state = {'sr_levels': mock_sr_levels}
        training_input = {'training_mode': 'light'}
        
        # Execute SR clustering
        logger.info("🚀 Executing SR Clustering...")
        result = await sr_clustering_step.execute(training_input, pipeline_state)
        
        if result['success']:
            logger.info("✅ SR Clustering completed successfully")
            
            # Check for expected outputs
            clustered_levels = result.get('clustered_levels', {})
            
            if 'clusters' in clustered_levels:
                cluster_count = len(clustered_levels['clusters'])
                logger.info(f"📊 SR clusters created: {cluster_count}")
            
            if 'cluster_analysis' in clustered_levels:
                analysis = clustered_levels['cluster_analysis']
                total_levels = analysis.get('total_levels', 0)
                avg_cluster_size = analysis.get('average_cluster_size', 0)
                logger.info(f"📊 Cluster analysis: {total_levels} total levels, avg size: {avg_cluster_size:.2f}")
            
            return result
            
        else:
            logger.error(f"❌ SR Clustering failed: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        logger.error(f"❌ SR Clustering test failed: {e}")
        import traceback
        logger.error(f"Error details: {traceback.format_exc()}")
        return None

async def test_full_integration():
    """Test full integration of all three components."""
    logger.info("🧪 Testing Full Market Analysis Integration")
    
    try:
        # Import the market analysis sub-pipeline
        from src.training.steps.market_analysis.sub_pipeline import (
            get_market_analysis_sub_pipeline, 
            SubPipelineConfig, 
            ExecutionMode
        )
        
        # Create configuration
        config = SubPipelineConfig(
            mode=ExecutionMode.LIGHT,
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="1h"
        )
        
        # Create pipeline
        pipeline = get_market_analysis_sub_pipeline(config)
        
        # Create sample market data
        market_data = create_sample_market_data(1000)
        
        # Store data in pipeline state
        pipeline_state = {'dataframe': market_data}
        training_input = {'training_mode': 'light'}
        
        logger.info("🎯 Executing Full Market Analysis Pipeline...")
        logger.info("   Stage 1: SR Parameter Optimization")
        logger.info("   Stage 2: SR Detection (using optimized parameters)")
        logger.info("   Stage 3: SR Clustering (using optimized parameters)")
        
        # Execute full pipeline
        result = await pipeline.execute(training_input, pipeline_state)
        
        if result['success']:
            logger.info("✅ Full Market Analysis Pipeline completed successfully")
            
            # Check for expected outputs from all stages
            if 'optimized_parameters' in result:
                logger.info(f"📊 Parameter optimization: {len(result['optimized_parameters'])} parameters")
            
            if 'sr_levels' in result:
                sr_levels = result['sr_levels']
                if isinstance(sr_levels, list):
                    logger.info(f"📊 SR Detection: {len(sr_levels)} levels detected")
                elif isinstance(sr_levels, dict):
                    total_levels = len(sr_levels.get('all_levels', []))
                    logger.info(f"📊 SR Detection: {total_levels} levels detected")
            
            if 'clustered_levels' in result:
                clustered_levels = result['clustered_levels']
                cluster_count = len(clustered_levels.get('clusters', []))
                logger.info(f"📊 SR Clustering: {cluster_count} clusters created")
            
            return result
            
        else:
            logger.error(f"❌ Full Market Analysis Pipeline failed: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Full integration test failed: {e}")
        import traceback
        logger.error(f"Error details: {traceback.format_exc()}")
        return None

async def main():
    """Run all market analysis integration tests."""
    logger.info("🚀 Starting Market Analysis Integration Tests")
    logger.info("=" * 60)
    
    # Test results
    test_results = {}
    
    # Test individual components
    logger.info("\n1️⃣ Testing Individual Components")
    logger.info("-" * 40)
    
    # Test SR Parameter Optimization
    param_result = await test_sr_parameter_optimization()
    test_results['parameter_optimization'] = param_result is not None
    
    # Test SR Detection
    detection_result = await test_sr_detection()
    test_results['sr_detection'] = detection_result is not None
    
    # Test SR Clustering
    clustering_result = await test_sr_clustering()
    test_results['sr_clustering'] = clustering_result is not None
    
    # Test full integration
    logger.info("\n2️⃣ Testing Full Integration")
    logger.info("-" * 40)
    
    integration_result = await test_full_integration()
    test_results['full_integration'] = integration_result is not None
    
    # Summary
    logger.info("\n📊 Test Results Summary")
    logger.info("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! Market Analysis implementation is complete.")
        return True
    else:
        logger.warning(f"⚠️ {total_tests - passed_tests} tests failed. Review implementation.")
        return False

if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)