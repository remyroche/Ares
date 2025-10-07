#!/usr/bin/env python3
"""
Test script for the optimized interaction feature generation pipeline.

This script demonstrates the complete integration of the enhanced pipeline
with comprehensive logging, matrix operations, and hardware optimization.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.optimized_interaction_orchestrator import (
    OptimizedInteractionOrchestrator,
    OptimizedInteractionConfig
)
from src.utils.tprint import tprint_success, tprint_info, tprint_error

def create_sample_data(n_rows: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    tprint_info("📊 Creating sample market data...")
    
    # Generate timestamps
    start_date = datetime.now() - timedelta(days=30)
    timestamps = pd.date_range(start=start_date, periods=n_rows, freq='15min')
    
    # Generate realistic OHLCV data
    np.random.seed(42)
    base_price = 2000.0
    returns = np.random.normal(0, 0.02, n_rows)
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # Generate OHLCV
    data = []
    for i, price in enumerate(prices):
        # Add some volatility
        volatility = np.random.uniform(0.001, 0.01)
        high = price * (1 + volatility)
        low = price * (1 - volatility)
        open_price = prices[i-1] if i > 0 else price
        close = price
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'timestamp': timestamps[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    tprint_success(f"✅ Created sample data: {len(df)} rows, {len(df.columns)} columns")
    return df

def create_sample_targets(data: pd.DataFrame) -> dict:
    """Create sample target series for testing."""
    tprint_info("🎯 Creating sample targets...")
    
    # Create a simple return target
    returns = data['close'].pct_change().shift(-1)  # Next period return
    returns = returns.fillna(0)
    
    # Create binary target (positive/negative return)
    binary_target = (returns > 0).astype(int)
    
    targets = {
        1: returns,  # Continuous target
        2: binary_target  # Binary target
    }
    
    tprint_success(f"✅ Created {len(targets)} target series")
    return targets

async def test_pipeline():
    """Test the complete optimized interaction feature generation pipeline."""
    tprint_success("🚀 Starting optimized interaction feature generation test")
    
    try:
        # Create sample data
        data = create_sample_data(1000)
        targets = create_sample_targets(data)
        
        # Create configuration
        config = OptimizedInteractionConfig(
            symbol="ETHUSDT",
            exchange="binance",
            timeframe="15m",
            feature_budget_pre=50,  # Reduced for testing
            feature_budget_post=(20, 30),
            interactions_cap=10,
            enable_matrix_optimization=True,
            enable_hardware_optimization=True,
            enable_parallel_processing=True,
            verbose_logging=True,
            log_performance=True
        )
        
        tprint_info("🔧 Initializing orchestrator...")
        orchestrator = OptimizedInteractionOrchestrator(config)
        
        # Prepare training input
        training_input = {
            'data': data,
            'targets': targets
        }
        
        # Prepare pipeline state
        pipeline_state = {
            'targets': targets,
            'patch_features': {}
        }
        
        tprint_info("🏃 Running feature generation pipeline...")
        result = await orchestrator.generate_features(training_input, pipeline_state)
        
        # Display results
        tprint_success("🎉 Pipeline execution completed successfully!")
        tprint_info("📊 RESULTS SUMMARY:")
        tprint_info(f"   - Success: {result.success}")
        tprint_info(f"   - Execution time: {result.execution_time:.3f}s")
        tprint_info(f"   - Total features: {len(result.feature_names) if result.feature_names else 0}")
        tprint_info(f"   - Selected features: {len(result.selected_features) if result.selected_features else 0}")
        tprint_info(f"   - Interaction features: {result.interaction_features.shape[1] if not result.interaction_features.empty else 0}")
        tprint_info(f"   - Cross-timeframe features: {result.cross_timeframe_features.shape[1] if not result.cross_timeframe_features.empty else 0}")
        tprint_info(f"   - Memory usage: {result.memory_usage_mb:.2f} MB")
        
        if result.error_message:
            tprint_error(f"   - Error: {result.error_message}")
        
        # Display feature names (first 10)
        if result.feature_names:
            tprint_info("📋 Sample feature names:")
            for i, name in enumerate(result.feature_names[:10]):
                tprint_info(f"   {i+1}. {name}")
            if len(result.feature_names) > 10:
                tprint_info(f"   ... and {len(result.feature_names) - 10} more")
        
        # Display performance metrics
        if hasattr(result, 'artifacts') and result.artifacts:
            stage_results = result.artifacts.get('stage_results', {})
            if stage_results:
                tprint_info("⏱️ Stage execution times:")
                for stage_name, stage_data in stage_results.items():
                    if isinstance(stage_data, dict) and 'stage_time' in stage_data:
                        tprint_info(f"   - {stage_name}: {stage_data['stage_time']:.3f}s")
        
        return result
        
    except Exception as e:
        tprint_error(f"❌ Test failed: {e}")
        import traceback
        tprint_error(f"Traceback: {traceback.format_exc()}")
        return None

async def main():
    """Main test function."""
    tprint_success("🧪 Starting Optimized Interaction Feature Generation Test")
    tprint_info("=" * 60)
    
    # Run the test
    result = await test_pipeline()
    
    if result and result.success:
        tprint_success("✅ All tests passed successfully!")
        tprint_info("🎯 The optimized interaction feature generation pipeline is working correctly.")
    else:
        tprint_error("❌ Tests failed!")
        tprint_info("🔧 Please check the error messages above for debugging information.")
    
    tprint_info("=" * 60)
    tprint_success("🏁 Test completed")

if __name__ == "__main__":
    # Run the async test
    asyncio.run(main())