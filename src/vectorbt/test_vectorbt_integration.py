"""
Test suite for production-ready VectorBT integration.

This module tests all VectorBT functionality used by the Ares system to ensure
production readiness and proper error handling.
"""

import pytest
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Import VectorBT modules
try:
    from src.vectorbt import (
        vbt, rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        scale, rank, zscore, winsorize, clip, quantile,
        Portfolio, PortfolioFactory, Returns,
        RSI, MACD, BBANDS, ATR, STOCH, SMA, EMA, BollingerBands,
        validate_vectorbt_installation, get_vectorbt_info,
        ProductionPortfolioFactory, ProductionRollingOperations,
        VectorBTError, VectorBTConfigurationError, VectorBTDataError, VectorBTComputationError
    )
    VECTORBT_AVAILABLE = True
except ImportError as e:
    VECTORBT_AVAILABLE = False
    pytest.skip(f"VectorBT not available: {e}", allow_module_level=True)

# Test data
@pytest.fixture
def sample_data():
    """Create sample financial data for testing."""
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.normal(0.0001, 0.02, 1000)  # 0.01% mean return, 2% volatility
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'close': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 1000))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 1000))),
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    # Ensure high >= low
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['close'])
    
    return data

@pytest.fixture
def sample_returns(sample_data):
    """Create sample returns data."""
    return sample_data['close'].pct_change().dropna()

class TestVectorBTInstallation:
    """Test VectorBT installation and configuration."""
    
    def test_vectorbt_available(self):
        """Test that VectorBT is available."""
        assert VECTORBT_AVAILABLE, "VectorBT should be available"
    
    def test_validate_installation(self):
        """Test VectorBT installation validation."""
        result = validate_vectorbt_installation()
        assert result is True, "VectorBT validation should pass"
    
    def test_get_vectorbt_info(self):
        """Test getting VectorBT information."""
        info = get_vectorbt_info()
        assert isinstance(info, dict), "Info should be a dictionary"
        assert 'version' in info, "Info should contain version"
        assert 'available' in info, "Info should contain availability status"
        assert info['available'] is True, "VectorBT should be available"

class TestRollingOperations:
    """Test rolling operations functionality."""
    
    def test_rolling_mean(self, sample_data):
        """Test rolling mean calculation."""
        result = rolling_mean(sample_data['close'], window=20)
        assert isinstance(result, pd.Series), "Result should be a Series"
        assert len(result) == len(sample_data), "Result should have same length"
        assert not result.isna().all(), "Result should not be all NaN"
    
    def test_rolling_std(self, sample_data):
        """Test rolling standard deviation calculation."""
        result = rolling_std(sample_data['close'], window=20)
        assert isinstance(result, pd.Series), "Result should be a Series"
        assert len(result) == len(sample_data), "Result should have same length"
        assert not result.isna().all(), "Result should not be all NaN"
    
    def test_rolling_apply(self, sample_data):
        """Test rolling apply functionality."""
        def custom_func(x):
            return x.max() - x.min()
        
        result = rolling_apply(sample_data['close'], custom_func, window=20)
        assert isinstance(result, pd.Series), "Result should be a Series"
        assert len(result) == len(sample_data), "Result should have same length"
    
    def test_rolling_corr(self, sample_data):
        """Test rolling correlation calculation."""
        # Create two correlated series
        series1 = sample_data['close']
        series2 = sample_data['close'].shift(1) + np.random.normal(0, 0.01, len(sample_data))
        
        result = rolling_corr(series1, series2, window=20)
        assert isinstance(result, pd.Series), "Result should be a Series"
        assert len(result) == len(sample_data), "Result should have same length"

class TestDataTransformations:
    """Test data transformation functions."""
    
    def test_scale(self, sample_data):
        """Test data scaling."""
        result = scale(sample_data['close'])
        assert isinstance(result, pd.Series), "Result should be a Series"
        assert abs(result.mean()) < 1e-10, "Scaled data should have mean ~0"
        assert abs(result.std() - 1.0) < 1e-10, "Scaled data should have std ~1"
    
    def test_rank(self, sample_data):
        """Test data ranking."""
        result = rank(sample_data['close'])
        assert isinstance(result, pd.Series), "Result should be a Series"
        assert result.min() >= 1, "Rank should start from 1"
        assert result.max() <= len(sample_data), "Rank should not exceed data length"
    
    def test_zscore(self, sample_data):
        """Test z-score calculation."""
        result = zscore(sample_data['close'])
        assert isinstance(result, pd.Series), "Result should be a Series"
        assert abs(result.mean()) < 1e-10, "Z-score should have mean ~0"
        assert abs(result.std() - 1.0) < 1e-10, "Z-score should have std ~1"
    
    def test_winsorize(self, sample_data):
        """Test data winsorization."""
        result = winsorize(sample_data['close'], limits=(0.05, 0.05))
        assert isinstance(result, pd.Series), "Result should be a Series"
        assert result.min() >= sample_data['close'].quantile(0.05), "Winsorized min should be >= 5th percentile"
        assert result.max() <= sample_data['close'].quantile(0.95), "Winsorized max should be <= 95th percentile"
    
    def test_clip(self, sample_data):
        """Test data clipping."""
        lower = sample_data['close'].quantile(0.1)
        upper = sample_data['close'].quantile(0.9)
        result = clip(sample_data['close'], lower=lower, upper=upper)
        assert isinstance(result, pd.Series), "Result should be a Series"
        assert result.min() >= lower, "Clipped data should respect lower bound"
        assert result.max() <= upper, "Clipped data should respect upper bound"

class TestTechnicalIndicators:
    """Test technical indicators."""
    
    def test_rsi(self, sample_data):
        """Test RSI calculation."""
        result = RSI.run(sample_data['close'])
        assert hasattr(result, 'rsi'), "RSI result should have rsi attribute"
        rsi_values = result.rsi
        assert isinstance(rsi_values, pd.Series), "RSI should be a Series"
        assert (rsi_values >= 0).all(), "RSI should be >= 0"
        assert (rsi_values <= 100).all(), "RSI should be <= 100"
    
    def test_macd(self, sample_data):
        """Test MACD calculation."""
        result = MACD.run(sample_data['close'])
        assert hasattr(result, 'macd'), "MACD result should have macd attribute"
        assert hasattr(result, 'signal'), "MACD result should have signal attribute"
        assert hasattr(result, 'histogram'), "MACD result should have histogram attribute"
    
    def test_bollinger_bands(self, sample_data):
        """Test Bollinger Bands calculation."""
        result = BBANDS.run(sample_data['close'])
        assert hasattr(result, 'upper'), "BBANDS result should have upper attribute"
        assert hasattr(result, 'middle'), "BBANDS result should have middle attribute"
        assert hasattr(result, 'lower'), "BBANDS result should have lower attribute"
        
        # Check that upper >= middle >= lower
        assert (result.upper >= result.middle).all(), "Upper band should be >= middle"
        assert (result.middle >= result.lower).all(), "Middle band should be >= lower"
    
    def test_atr(self, sample_data):
        """Test ATR calculation."""
        result = ATR.run(
            sample_data['high'], 
            sample_data['low'], 
            sample_data['close']
        )
        assert hasattr(result, 'atr'), "ATR result should have atr attribute"
        atr_values = result.atr
        assert isinstance(atr_values, pd.Series), "ATR should be a Series"
        assert (atr_values >= 0).all(), "ATR should be >= 0"
    
    def test_stochastic(self, sample_data):
        """Test Stochastic Oscillator calculation."""
        result = STOCH.run(
            sample_data['high'], 
            sample_data['low'], 
            sample_data['close']
        )
        assert hasattr(result, 'percent_k'), "STOCH result should have percent_k attribute"
        assert hasattr(result, 'percent_d'), "STOCH result should have percent_d attribute"
        
        k_values = result.percent_k
        d_values = result.percent_d
        assert (k_values >= 0).all(), "Stochastic %K should be >= 0"
        assert (k_values <= 100).all(), "Stochastic %K should be <= 100"
        assert (d_values >= 0).all(), "Stochastic %D should be >= 0"
        assert (d_values <= 100).all(), "Stochastic %D should be <= 100"

class TestPortfolioOperations:
    """Test portfolio operations."""
    
    def test_portfolio_from_returns(self, sample_returns):
        """Test portfolio creation from returns."""
        portfolio = PortfolioFactory.from_returns(sample_returns)
        assert isinstance(portfolio, Portfolio), "Result should be a Portfolio"
        assert hasattr(portfolio, 'returns'), "Portfolio should have returns attribute"
        assert hasattr(portfolio, 'total_return'), "Portfolio should have total_return method"
    
    def test_portfolio_from_signals(self, sample_data):
        """Test portfolio creation from signals."""
        # Create simple buy/sell signals
        entries = sample_data['close'] > sample_data['close'].rolling(20).mean()
        exits = sample_data['close'] < sample_data['close'].rolling(20).mean()
        
        portfolio = PortfolioFactory.from_signals(
            close=sample_data['close'],
            entries=entries,
            exits=exits
        )
        assert isinstance(portfolio, Portfolio), "Result should be a Portfolio"
        assert hasattr(portfolio, 'trades'), "Portfolio should have trades attribute"
        assert hasattr(portfolio, 'orders'), "Portfolio should have orders attribute"
    
    def test_production_portfolio_factory(self, sample_data, sample_returns):
        """Test production portfolio factory."""
        # Test from returns
        portfolio1 = ProductionPortfolioFactory.from_returns(sample_returns)
        assert isinstance(portfolio1, Portfolio), "Result should be a Portfolio"
        
        # Test from signals
        entries = sample_data['close'] > sample_data['close'].rolling(20).mean()
        exits = sample_data['close'] < sample_data['close'].rolling(20).mean()
        
        portfolio2 = ProductionPortfolioFactory.from_signals(
            close=sample_data['close'],
            entries=entries,
            exits=exits
        )
        assert isinstance(portfolio2, Portfolio), "Result should be a Portfolio"
    
    def test_portfolio_metrics(self, sample_returns):
        """Test portfolio metrics calculation."""
        portfolio = PortfolioFactory.from_returns(sample_returns)
        
        # Test total return
        total_return = portfolio.total_return()
        assert isinstance(total_return, (int, float)), "Total return should be numeric"
        
        # Test Sharpe ratio
        sharpe = portfolio.sharpe_ratio()
        assert isinstance(sharpe, (int, float)), "Sharpe ratio should be numeric"
        
        # Test max drawdown
        max_dd = portfolio.max_drawdown()
        assert isinstance(max_dd, (int, float)), "Max drawdown should be numeric"
        assert max_dd <= 0, "Max drawdown should be <= 0"

class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_vectorbt_data_error(self):
        """Test VectorBTDataError handling."""
        with pytest.raises(VectorBTDataError):
            ProductionPortfolioFactory.from_returns("invalid_data")
    
    def test_vectorbt_computation_error(self):
        """Test VectorBTComputationError handling."""
        # Create invalid data that should cause computation error
        invalid_data = pd.Series([np.nan] * 100)
        
        with pytest.raises(VectorBTComputationError):
            ProductionPortfolioFactory.from_returns(invalid_data)
    
    def test_production_rolling_operations_error(self):
        """Test production rolling operations error handling."""
        # Test with invalid function
        data = pd.Series([1, 2, 3, 4, 5])
        
        def invalid_func(x):
            raise ValueError("Test error")
        
        with pytest.raises(VectorBTComputationError):
            ProductionRollingOperations.safe_rolling_apply(
                data, invalid_func, window=3
            )

class TestPerformance:
    """Test performance and memory usage."""
    
    def test_large_dataset_performance(self):
        """Test performance with large dataset."""
        # Create large dataset
        large_data = pd.Series(np.random.randn(10000))
        
        # Time rolling operations
        import time
        start_time = time.time()
        
        result = rolling_mean(large_data, window=100)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        assert execution_time < 5.0, f"Rolling mean should complete in <5s, took {execution_time:.2f}s"
        assert isinstance(result, pd.Series), "Result should be a Series"
        assert len(result) == len(large_data), "Result should have same length"
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large operations."""
        # This is a basic test - in production, you'd use memory profiling
        large_data = pd.Series(np.random.randn(5000))
        
        # Perform multiple operations
        results = []
        for i in range(10):
            result = rolling_mean(large_data, window=50)
            results.append(result)
        
        # All operations should complete without memory issues
        assert len(results) == 10, "All operations should complete"
        assert all(isinstance(r, pd.Series) for r in results), "All results should be Series"

def run_tests() -> bool:
    """Run VectorBT integration tests."""
    try:
        import pytest
        result = pytest.main([__file__, "-v", "--tb=short"])
        return result == 0
    except Exception as e:
        print(f"Test execution failed: {e}")
        return False

if __name__ == "__main__":
    # Run tests
    success = run_tests()
    sys.exit(0 if success else 1)