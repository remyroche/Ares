"""
Candlestick Pattern Feature Generation - Usage Examples

This module demonstrates how to use the VectorBT-optimized candlestick pattern
feature generators for maximum performance and comprehensive pattern detection.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any
import time

# Import candlestick pattern generators
from ..categories.candlestick_pattern import (
    CandlestickPatternFeatureGenerator,
    VectorBTCandlestickPatternGenerator,
    VectorBTCandlestickPatternBatchProcessor,
    CandlestickPatternConfig,
    create_candlestick_pattern_generators,
    create_vectorbt_candlestick_generator,
    create_candlestick_batch_processor
)

# Import unified vectorization manager
from ..utils.unified_vectorization_manager import get_unified_vectorization_manager

logger = logging.getLogger(__name__)

def create_sample_ohlcv_data(n_periods: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(seed)
    
    # Generate realistic price data
    returns = np.random.normal(0, 0.02, n_periods)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = []
    for i, price in enumerate(prices):
        # Add some volatility
        volatility = np.random.uniform(0.005, 0.02)
        
        # Generate open, high, low, close
        open_price = price * (1 + np.random.normal(0, volatility/2))
        close_price = price * (1 + np.random.normal(0, volatility/2))
        
        high_price = max(open_price, close_price) * (1 + np.random.uniform(0, volatility))
        low_price = min(open_price, close_price) * (1 - np.random.uniform(0, volatility))
        
        # Generate volume
        volume = np.random.lognormal(10, 1)
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    # Create DataFrame with datetime index
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='1H')
    df = pd.DataFrame(data, index=dates)
    
    return df

def example_basic_pattern_detection():
    """Example: Basic candlestick pattern detection."""
    print("=== Basic Pattern Detection Example ===")
    
    # Create sample data
    data = create_sample_ohlcv_data(1000)
    print(f"Created sample data with {len(data)} periods")
    
    # Create pattern generator
    generator = CandlestickPatternFeatureGenerator()
    
    # Detect specific patterns
    patterns = ['doji', 'hammer', 'engulfing_bullish', 'engulfing_bearish']
    
    for pattern in patterns:
        start_time = time.time()
        result = generator._generate_feature(data, patterns=[pattern])
        execution_time = time.time() - start_time
        
        pattern_count = result.sum()
        print(f"{pattern}: {pattern_count} patterns detected in {execution_time:.3f}s")
    
    # Get performance stats
    stats = generator.get_pattern_stats()
    print(f"Performance stats: {stats}")

def example_vectorbt_optimized_detection():
    """Example: VectorBT-optimized pattern detection."""
    print("\n=== VectorBT-Optimized Detection Example ===")
    
    # Create sample data
    data = create_sample_ohlcv_data(5000)  # Larger dataset
    print(f"Created sample data with {len(data)} periods")
    
    # Create VectorBT-optimized generator
    pattern_config = CandlestickPatternConfig(
        enable_vectorbt=True,
        enable_batch_processing=True,
        enable_gpu_acceleration=False,  # Set to True if GPU available
        enable_memory_optimization=True,
        chunk_size=10000
    )
    
    generator = VectorBTCandlestickPatternGenerator(pattern_config=pattern_config)
    
    # Generate all patterns at once
    start_time = time.time()
    all_patterns = generator.generate_all_patterns(data)
    execution_time = time.time() - start_time
    
    print(f"Generated all patterns in {execution_time:.3f}s")
    print(f"Pattern columns: {list(all_patterns.columns)}")
    
    # Generate patterns with confidence scores
    start_time = time.time()
    patterns_with_confidence = generator.generate_patterns_with_confidence(
        data, patterns=['doji', 'hammer', 'engulfing_bullish']
    )
    execution_time = time.time() - start_time
    
    print(f"Generated patterns with confidence in {execution_time:.3f}s")
    print(f"Confidence columns: {list(patterns_with_confidence.columns)}")
    
    # Get performance stats
    stats = generator.get_pattern_stats()
    print(f"Performance stats: {stats}")

def example_batch_processing():
    """Example: Batch processing multiple pattern generators."""
    print("\n=== Batch Processing Example ===")
    
    # Create sample data
    data = create_sample_ohlcv_data(3000)
    print(f"Created sample data with {len(data)} periods")
    
    # Create multiple pattern configurations
    configs = [
        CandlestickPatternConfig(
            doji_threshold=0.05,  # More sensitive
            hammer_threshold=0.2,
            enable_batch_processing=True
        ),
        CandlestickPatternConfig(
            doji_threshold=0.15,  # Less sensitive
            hammer_threshold=0.4,
            enable_batch_processing=True
        )
    ]
    
    # Create batch processor
    batch_processor = create_candlestick_batch_processor(configs)
    
    # Define pattern lists for each generator
    pattern_lists = [
        ['doji', 'hammer', 'engulfing_bullish'],
        ['shooting_star', 'hanging_man', 'engulfing_bearish']
    ]
    
    # Process batch
    start_time = time.time()
    results = batch_processor.process_batch(data, pattern_lists)
    execution_time = time.time() - start_time
    
    print(f"Processed {len(results)} generators in {execution_time:.3f}s")
    
    for i, result in enumerate(results):
        print(f"Generator {i+1} results: {len(result)} patterns detected")
    
    # Get batch stats
    batch_stats = batch_processor.get_batch_stats()
    print(f"Batch stats: {batch_stats}")

def example_memory_optimization():
    """Example: Memory-optimized processing for large datasets."""
    print("\n=== Memory Optimization Example ===")
    
    # Create large dataset
    data = create_sample_ohlcv_data(10000)
    print(f"Created large dataset with {len(data)} periods")
    
    # Configure for memory optimization
    pattern_config = CandlestickPatternConfig(
        enable_memory_optimization=True,
        chunk_size=5000,  # Process in smaller chunks
        enable_vectorbt=True,
        enable_batch_processing=True
    )
    
    generator = VectorBTCandlestickPatternGenerator(pattern_config=pattern_config)
    
    # Process with memory optimization
    start_time = time.time()
    results = generator.generate_all_patterns(data)
    execution_time = time.time() - start_time
    
    print(f"Memory-optimized processing completed in {execution_time:.3f}s")
    print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Get memory stats from unified vectorization manager
    if hasattr(generator, 'vectorization_manager') and generator.vectorization_manager:
        vectorization_stats = generator.vectorization_manager.get_performance_stats()
        print(f"Vectorization stats: {vectorization_stats}")

def example_performance_comparison():
    """Example: Performance comparison between different approaches."""
    print("\n=== Performance Comparison Example ===")
    
    # Create test data
    data = create_sample_ohlcv_data(2000)
    patterns = ['doji', 'hammer', 'engulfing_bullish', 'engulfing_bearish']
    
    # Test 1: Basic generator
    print("Testing basic generator...")
    basic_generator = CandlestickPatternFeatureGenerator()
    
    start_time = time.time()
    basic_results = []
    for pattern in patterns:
        result = basic_generator._generate_feature(data, patterns=[pattern])
        basic_results.append(result)
    basic_time = time.time() - start_time
    
    print(f"Basic generator: {basic_time:.3f}s")
    
    # Test 2: VectorBT-optimized generator
    print("Testing VectorBT-optimized generator...")
    vectorbt_config = CandlestickPatternConfig(
        enable_vectorbt=True,
        enable_batch_processing=True,
        enable_memory_optimization=True
    )
    vectorbt_generator = VectorBTCandlestickPatternGenerator(pattern_config=vectorbt_config)
    
    start_time = time.time()
    vectorbt_results = vectorbt_generator.generate_all_patterns(data)
    vectorbt_time = time.time() - start_time
    
    print(f"VectorBT generator: {vectorbt_time:.3f}s")
    
    # Test 3: Batch processing
    print("Testing batch processing...")
    batch_processor = create_candlestick_batch_processor([vectorbt_config])
    
    start_time = time.time()
    batch_results = batch_processor.process_batch(data, [patterns])
    batch_time = time.time() - start_time
    
    print(f"Batch processor: {batch_time:.3f}s")
    
    # Performance summary
    print(f"\nPerformance Summary:")
    print(f"Basic generator: {basic_time:.3f}s")
    print(f"VectorBT generator: {vectorbt_time:.3f}s ({(basic_time/vectorbt_time):.2f}x speedup)")
    print(f"Batch processor: {batch_time:.3f}s ({(basic_time/batch_time):.2f}x speedup)")

def example_custom_pattern_configuration():
    """Example: Custom pattern configuration and thresholds."""
    print("\n=== Custom Pattern Configuration Example ===")
    
    # Create sample data
    data = create_sample_ohlcv_data(1000)
    
    # Custom configuration for more sensitive pattern detection
    sensitive_config = CandlestickPatternConfig(
        doji_threshold=0.05,      # More sensitive to doji
        hammer_threshold=0.2,     # More sensitive to hammer
        engulfing_threshold=0.05, # More sensitive to engulfing
        enable_vectorbt=True,
        enable_batch_processing=True
    )
    
    # Custom configuration for less sensitive pattern detection
    conservative_config = CandlestickPatternConfig(
        doji_threshold=0.2,       # Less sensitive to doji
        hammer_threshold=0.5,     # Less sensitive to hammer
        engulfing_threshold=0.2,  # Less sensitive to engulfing
        enable_vectorbt=True,
        enable_batch_processing=True
    )
    
    # Test both configurations
    for config_name, config in [("Sensitive", sensitive_config), ("Conservative", conservative_config)]:
        print(f"\nTesting {config_name} configuration:")
        
        generator = VectorBTCandlestickPatternGenerator(pattern_config=config)
        
        # Test doji detection
        doji_result = generator._detect_doji_pattern(data)
        doji_count = doji_result.sum()
        print(f"Doji patterns detected: {doji_count}")
        
        # Test hammer detection
        hammer_result = generator._detect_hammer_pattern(data)
        hammer_count = hammer_result.sum()
        print(f"Hammer patterns detected: {hammer_count}")

def main():
    """Run all examples."""
    print("🚀 Candlestick Pattern Feature Generation Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_basic_pattern_detection()
        example_vectorbt_optimized_detection()
        example_batch_processing()
        example_memory_optimization()
        example_performance_comparison()
        example_custom_pattern_configuration()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        logger.error(f"Example execution failed: {e}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run examples
    main()