"""
VectorBT Optimization Example

This script demonstrates the VectorBT optimizations integrated into the matrix operations module.
It shows how to use the optimized functions and compares performance with standard implementations.
"""

import numpy as np
import pandas as pd
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample data for testing."""
    # Create sample OHLCV data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic price data
    price = 100
    prices = [price]
    for _ in range(n_samples - 1):
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        price *= (1 + change)
        prices.append(price)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Ensure high >= low and high >= close >= low
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['close'])
    data['high'] = np.maximum(data['high'], data['open'])
    data['low'] = np.minimum(data['low'], data['open'])
    
    return data

def test_matrix_operations():
    """Test VectorBT-optimized matrix operations."""
    logger.info("🧮 Testing Matrix Operations with VectorBT Optimization")
    
    # Create sample matrices
    A = np.random.randn(500, 500)
    B = np.random.randn(500, 500)
    
    # Test matrix multiplication
    start_time = time.time()
    try:
        from .vectorbt_optimizations import vectorbt_matrix_multiply
        result = vectorbt_matrix_multiply(A, B)
        vectorbt_time = time.time() - start_time
        logger.info(f"✅ VectorBT matrix multiplication: {vectorbt_time:.4f}s, shape: {result.shape}")
    except ImportError:
        logger.warning("⚠️ VectorBT not available, using fallback")
        result = np.dot(A, B)
        vectorbt_time = time.time() - start_time
        logger.info(f"✅ Fallback matrix multiplication: {vectorbt_time:.4f}s, shape: {result.shape}")
    
    # Test correlation matrix
    data = np.random.randn(1000, 10)
    start_time = time.time()
    try:
        from .vectorbt_optimizations import vectorbt_correlation_matrix
        corr_matrix = vectorbt_correlation_matrix(data)
        corr_time = time.time() - start_time
        logger.info(f"✅ VectorBT correlation matrix: {corr_time:.4f}s, shape: {corr_matrix.shape}")
    except ImportError:
        logger.warning("⚠️ VectorBT not available, using fallback")
        corr_matrix = np.corrcoef(data.T)
        corr_time = time.time() - start_time
        logger.info(f"✅ Fallback correlation matrix: {corr_time:.4f}s, shape: {corr_matrix.shape}")

def test_trading_indicators():
    """Test VectorBT-optimized trading indicators."""
    logger.info("📊 Testing Trading Indicators with VectorBT Optimization")
    
    # Create sample data
    data = create_sample_data()
    
    # Test trading indicators
    start_time = time.time()
    try:
        from .vectorbt_optimizations import vectorbt_trading_indicators
        
        config = {
            'sma_periods': [9, 21, 50],
            'ema_periods': [12, 26],
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2.0
        }
        
        result = vectorbt_trading_indicators(data, config)
        indicators_time = time.time() - start_time
        
        new_columns = [col for col in result.columns if col not in data.columns]
        logger.info(f"✅ VectorBT trading indicators: {indicators_time:.4f}s, added {len(new_columns)} indicators")
        logger.info(f"   New indicators: {new_columns[:10]}...")  # Show first 10
        
    except ImportError:
        logger.warning("⚠️ VectorBT not available, using fallback")
        # Fallback to standard implementation
        from .vectorized_core import get_vectorized_processing_core
        core = get_vectorized_processing_core()
        result = core.compute_trading_indicators(data)
        indicators_time = time.time() - start_time
        
        new_columns = [col for col in result.columns if col not in data.columns]
        logger.info(f"✅ Fallback trading indicators: {indicators_time:.4f}s, added {len(new_columns)} indicators")

def test_rolling_features():
    """Test VectorBT-optimized rolling features."""
    logger.info("🔄 Testing Rolling Features with VectorBT Optimization")
    
    # Create sample data
    data = create_sample_data()
    
    # Test rolling features
    start_time = time.time()
    try:
        from .vectorbt_optimizations import vectorbt_rolling_features
        
        windows = [5, 10, 20]
        features = ['close', 'volume']
        
        result = vectorbt_rolling_features(data, windows, features)
        rolling_time = time.time() - start_time
        
        new_columns = [col for col in result.columns if col not in data.columns]
        logger.info(f"✅ VectorBT rolling features: {rolling_time:.4f}s, added {len(new_columns)} features")
        logger.info(f"   New features: {new_columns[:10]}...")  # Show first 10
        
    except ImportError:
        logger.warning("⚠️ VectorBT not available, using fallback")
        # Fallback to standard implementation
        from .vectorized_core import get_vectorized_processing_core
        core = get_vectorized_processing_core()
        result = core.vectorized_rolling_features(data, windows, features)
        rolling_time = time.time() - start_time
        
        new_columns = [col for col in result.columns if col not in data.columns]
        logger.info(f"✅ Fallback rolling features: {rolling_time:.4f}s, added {len(new_columns)} features")

def test_batch_processing():
    """Test VectorBT-optimized batch processing."""
    logger.info("⚡ Testing Batch Processing with VectorBT Optimization")
    
    # Create sample matrices for batch processing
    matrices_a = [np.random.randn(100, 100) for _ in range(10)]
    matrices_b = [np.random.randn(100, 100) for _ in range(10)]
    
    # Test batch matrix multiplication
    start_time = time.time()
    try:
        from .vectorbt_optimizations import vectorbt_batch_processing
        
        results = vectorbt_batch_processing(matrices_a, 'batch_matrix_multiply', matrices_b=matrices_b)
        batch_time = time.time() - start_time
        
        logger.info(f"✅ VectorBT batch processing: {batch_time:.4f}s, processed {len(results)} matrices")
        
    except ImportError:
        logger.warning("⚠️ VectorBT not available, using fallback")
        # Fallback to standard implementation
        from .batch_operations import batch_matrix_multiply
        results = batch_matrix_multiply(matrices_a, matrices_b)
        batch_time = time.time() - start_time
        
        logger.info(f"✅ Fallback batch processing: {batch_time:.4f}s, processed {len(results)} matrices")

def test_performance_comparison():
    """Compare performance between VectorBT and standard implementations."""
    logger.info("📈 Performance Comparison: VectorBT vs Standard Implementation")
    
    # Create larger dataset for meaningful comparison
    data = create_sample_data()
    data = pd.concat([data] * 5, ignore_index=True)  # 5000 samples
    
    # Test trading indicators performance
    config = {
        'sma_periods': [9, 21, 50, 200],
        'ema_periods': [12, 26, 50],
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9
    }
    
    # VectorBT implementation
    try:
        from .vectorbt_optimizations import vectorbt_trading_indicators
        
        start_time = time.time()
        vectorbt_result = vectorbt_trading_indicators(data, config)
        vectorbt_time = time.time() - start_time
        
        logger.info(f"🚀 VectorBT trading indicators: {vectorbt_time:.4f}s")
        
    except ImportError:
        logger.warning("⚠️ VectorBT not available for comparison")
        vectorbt_time = None
    
    # Standard implementation
    try:
        from .vectorized_core import get_vectorized_processing_core
        core = get_vectorized_processing_core()
        
        start_time = time.time()
        standard_result = core.compute_trading_indicators(data, config)
        standard_time = time.time() - start_time
        
        logger.info(f"🐌 Standard trading indicators: {standard_time:.4f}s")
        
        if vectorbt_time is not None:
            speedup = standard_time / vectorbt_time
            logger.info(f"⚡ VectorBT speedup: {speedup:.2f}x faster")
        
    except Exception as e:
        logger.error(f"❌ Standard implementation failed: {e}")

def main():
    """Run all VectorBT optimization tests."""
    logger.info("🚀 Starting VectorBT Optimization Tests")
    logger.info("=" * 50)
    
    try:
        test_matrix_operations()
        logger.info("")
        
        test_trading_indicators()
        logger.info("")
        
        test_rolling_features()
        logger.info("")
        
        test_batch_processing()
        logger.info("")
        
        test_performance_comparison()
        logger.info("")
        
        logger.info("✅ All VectorBT optimization tests completed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main()