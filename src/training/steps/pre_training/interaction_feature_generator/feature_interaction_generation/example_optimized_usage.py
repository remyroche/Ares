"""
Example Usage of Optimized Interaction Feature Generation

This script demonstrates how to use the optimized interaction feature generation
pipeline with extensive logging, matrix operations, and hardware optimization.

Key Features Demonstrated:
- Complete pipeline execution
- Matrix operations optimization
- M1 hardware acceleration
- Extensive tprint logging
- Integration with sub_pipeline
- Performance monitoring
"""

import asyncio
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Import tprint utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, 
    tprint_debug, tprint_performance, tprint_progress
)

# Import the optimized orchestrator
from .optimized_interaction_orchestrator import (
    OptimizedInteractionOrchestrator, OptimizedInteractionConfig, generate_optimized_interaction_features
)

# Import the interactive component
from .interactive_feature_generation_component import (
    InteractiveFeatureGenerationComponent, InteractiveFeatureGenerationConfig, execute_interactive_feature_generation
)

# Import sub_pipeline
from ..sub_pipeline import PreTrainingSubPipeline, SubPipelineConfig, ExecutionMode


def create_sample_data(n_rows: int = 1000) -> pd.DataFrame:
    """Create sample market data for testing."""
    tprint_info("📊 Creating sample market data...")
    
    # Generate timestamps
    timestamps = pd.date_range(start='2024-01-01', periods=n_rows, freq='15min')
    
    # Generate OHLCV data
    np.random.seed(42)  # For reproducibility
    
    # Generate price data with some trend and volatility
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_rows)
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # Generate OHLC from prices
    high_multiplier = 1 + np.abs(np.random.normal(0, 0.01, n_rows))
    low_multiplier = 1 - np.abs(np.random.normal(0, 0.01, n_rows))
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices * (1 + np.random.normal(0, 0.005, n_rows)),
        'high': prices * high_multiplier,
        'low': prices * low_multiplier,
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_rows),
        'trade_count': np.random.poisson(50, n_rows),
        'bid': prices * (1 - np.random.uniform(0.0001, 0.001, n_rows)),
        'ask': prices * (1 + np.random.uniform(0.0001, 0.001, n_rows)),
        'bid_size': np.random.lognormal(8, 1, n_rows),
        'ask_size': np.random.lognormal(8, 1, n_rows)
    })
    
    # Set timestamp as index
    data.set_index('timestamp', inplace=True)
    
    tprint_success(f"✅ Created sample data: {data.shape[0]} rows, {data.shape[1]} columns")
    return data


def create_sample_targets(data: pd.DataFrame) -> Dict[int, pd.Series]:
    """Create sample targets for feature generation."""
    tprint_info("🎯 Creating sample targets...")
    
    # Generate simple targets based on future returns
    future_returns = data['close'].pct_change().shift(-1)  # 1-step ahead returns
    
    targets = {
        1: future_returns,  # 1-step target
        3: data['close'].pct_change().shift(-3),  # 3-step target
        5: data['close'].pct_change().shift(-5)   # 5-step target
    }
    
    tprint_success(f"✅ Created {len(targets)} target series")
    return targets


async def example_direct_orchestrator_usage():
    """Example of using the orchestrator directly."""
    tprint_success("🚀 Example 1: Direct Orchestrator Usage")
    
    # Create sample data
    data = create_sample_data(1000)
    targets = create_sample_targets(data)
    
    # Create configuration
    config = OptimizedInteractionConfig(
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="15m",
        feature_budget_pre=50,  # Smaller for demo
        interactions_cap=10,
        enable_matrix_optimization=True,
        enable_hardware_optimization=True,
        verbose_logging=True
    )
    
    # Create training input and pipeline state
    training_input = {
        'data': data,
        'targets': targets
    }
    
    pipeline_state = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'data_dir': 'historical_data'
    }
    
    # Execute feature generation
    tprint_info("🔧 Executing feature generation...")
    result = await generate_optimized_interaction_features(
        training_input, pipeline_state, config
    )
    
    # Display results
    if result.success:
        tprint_success("✅ Feature generation completed successfully!")
        tprint_info(f"📊 Generated {len(result.feature_names)} total features")
        tprint_info(f"🎯 Selected {len(result.selected_features)} features")
        tprint_info(f"🔗 Generated {len(result.interaction_features.columns)} interactions")
        tprint_info(f"⏰ Generated {len(result.cross_timeframe_features.columns)} cross-timeframe features")
        tprint_info(f"💾 Memory usage: {result.memory_usage_mb:.2f} MB")
        tprint_info(f"⏱️ Execution time: {result.execution_time:.3f}s")
        
        # Show some feature names
        tprint_debug("Sample feature names:")
        for i, name in enumerate(result.feature_names[:10]):
            tprint_debug(f"  {i+1}. {name}")
        
        if len(result.feature_names) > 10:
            tprint_debug(f"  ... and {len(result.feature_names) - 10} more")
    else:
        tprint_error(f"❌ Feature generation failed: {result.error_message}")


async def example_component_usage():
    """Example of using the component interface."""
    tprint_success("🚀 Example 2: Component Usage")
    
    # Create sample data
    data = create_sample_data(1000)
    targets = create_sample_targets(data)
    
    # Create component configuration
    config = RoadmapFeatureGenerationConfig(
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="15m",
        feature_budget_pre=50,
        interactions_cap=10,
        enable_matrix_optimization=True,
        enable_hardware_optimization=True,
        verbose_logging=True
    )
    
    # Create training input and pipeline state
    training_input = {
        'data': data,
        'targets': targets
    }
    
    pipeline_state = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'data_dir': 'historical_data'
    }
    
    # Execute using component
    tprint_info("🔧 Executing feature generation using component...")
    result = await execute_interactive_feature_generation(
        training_input, pipeline_state, config
    )
    
    # Display results
    if result.success:
        tprint_success("✅ Component execution completed successfully!")
        tprint_info(f"📊 Generated features: {result.artifacts['interactive_feature_generation_result']['feature_names']}")
        tprint_info(f"🎯 Selected features: {len(result.artifacts['interactive_feature_generation_result']['selected_features'])}")
        tprint_info(f"⏱️ Execution time: {result.execution_time:.3f}s")
        tprint_info(f"📁 Output files: {result.output_files}")
    else:
        tprint_error(f"❌ Component execution failed: {result.error_message}")


async def example_sub_pipeline_usage():
    """Example of using the sub_pipeline integration."""
    tprint_success("🚀 Example 3: Sub-Pipeline Integration")
    
    # Create sample data
    data = create_sample_data(1000)
    targets = create_sample_targets(data)
    
    # Create sub-pipeline configuration
    config = SubPipelineConfig(
        mode=ExecutionMode.FULL,
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="15m",
        data_dir="historical_data",
        parallel_processing=True,
        max_workers=2,
        custom_params={
            'feature_budget_pre': 50,
            'interactions_cap': 10,
            'enable_matrix_optimization': True,
            'enable_hardware_optimization': True,
            'verbose_logging': True
        }
    )
    
    # Create training input and pipeline state
    training_input = {
        'data': data,
        'targets': targets
    }
    
    pipeline_state = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'data_dir': 'historical_data'
    }
    
    # Execute sub-pipeline
    tprint_info("🔧 Executing sub-pipeline...")
    pipeline = PreTrainingSubPipeline()
    
    # Execute just the roadmap feature generation step
    result = await pipeline._execute_interactive_feature_generation(config)
    
    # Display results
    if result.success:
        tprint_success("✅ Sub-pipeline execution completed successfully!")
        tprint_info(f"📊 Status: {result.status.value}")
        tprint_info(f"⏱️ Duration: {result.duration_seconds:.3f}s")
        tprint_info(f"📁 Output files: {result.output_files}")
        tprint_info(f"📊 Metadata: {result.metadata}")
    else:
        tprint_error(f"❌ Sub-pipeline execution failed: {result.error_message}")


async def example_performance_comparison():
    """Example comparing performance with and without optimizations."""
    tprint_success("🚀 Example 4: Performance Comparison")
    
    # Create sample data
    data = create_sample_data(2000)  # Larger dataset for better comparison
    targets = create_sample_targets(data)
    
    training_input = {
        'data': data,
        'targets': targets
    }
    
    pipeline_state = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'data_dir': 'historical_data'
    }
    
    # Test with optimizations enabled
    tprint_info("🔧 Testing with optimizations enabled...")
    config_optimized = OptimizedInteractionConfig(
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="15m",
        feature_budget_pre=100,
        interactions_cap=15,
        enable_matrix_optimization=True,
        enable_hardware_optimization=True,
        verbose_logging=False  # Reduce logging for cleaner output
    )
    
    start_time = time.time()
    result_optimized = await generate_optimized_interaction_features(
        training_input, pipeline_state, config_optimized
    )
    optimized_time = time.time() - start_time
    
    # Test with optimizations disabled
    tprint_info("🔧 Testing with optimizations disabled...")
    config_unoptimized = OptimizedInteractionConfig(
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="15m",
        feature_budget_pre=100,
        interactions_cap=15,
        enable_matrix_optimization=False,
        enable_hardware_optimization=False,
        verbose_logging=False
    )
    
    start_time = time.time()
    result_unoptimized = await generate_optimized_interaction_features(
        training_input, pipeline_state, config_unoptimized
    )
    unoptimized_time = time.time() - start_time
    
    # Compare results
    tprint_success("📊 Performance Comparison Results:")
    tprint_info(f"With optimizations: {optimized_time:.3f}s")
    tprint_info(f"Without optimizations: {unoptimized_time:.3f}s")
    
    if optimized_time < unoptimized_time:
        speedup = unoptimized_time / optimized_time
        tprint_success(f"🚀 Speedup: {speedup:.2f}x faster with optimizations!")
    else:
        tprint_warning("⚠️ Optimizations did not provide speedup (may be due to small dataset)")
    
    # Compare memory usage
    if result_optimized.success and result_unoptimized.success:
        tprint_info(f"Memory usage (optimized): {result_optimized.memory_usage_mb:.2f} MB")
        tprint_info(f"Memory usage (unoptimized): {result_unoptimized.memory_usage_mb:.2f} MB")


async def main():
    """Main function to run all examples."""
    tprint_success("🎉 Starting Optimized Interaction Feature Generation Examples")
    tprint_info("=" * 80)
    
    try:
        # Example 1: Direct orchestrator usage
        await example_direct_orchestrator_usage()
        tprint_info("=" * 80)
        
        # Example 2: Component usage
        await example_component_usage()
        tprint_info("=" * 80)
        
        # Example 3: Sub-pipeline integration
        await example_sub_pipeline_usage()
        tprint_info("=" * 80)
        
        # Example 4: Performance comparison
        await example_performance_comparison()
        tprint_info("=" * 80)
        
        tprint_success("🎉 All examples completed successfully!")
        
    except Exception as e:
        tprint_error(f"❌ Example execution failed: {e}")
        raise


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())