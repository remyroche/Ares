"""
Pytest configuration and shared fixtures for Ares Trading System tests.

This module provides common fixtures and test utilities used across all test modules.
"""

import pytest
import sys
import os
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure testing environment
os.environ["TESTING"] = "True"
os.environ["TPRINT_MINIMAL"] = "1"  # Enable minimal tprint mode for tests

# Import tprint after path setup
try:
    from utils.tprint import tprint, tprint_logged, LogLevel
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    # Create mock tprint for testing
    def tprint(*args, **kwargs):
        print(f"[MOCK TPRINT] {' '.join(str(arg) for arg in args)}")
    
    def tprint_logged(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class LogLevel:
        DEBUG = "DEBUG"
        INFO = "INFO"
        WARNING = "WARNING"
        ERROR = "ERROR"
        SUCCESS = "SUCCESS"
        PROGRESS = "PROGRESS"
        PERFORMANCE = "PERFORMANCE"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_tprint():
    """Mock tprint for testing to avoid actual logging during tests."""
    with patch('utils.tprint.tprint') as mock_tprint:
        yield mock_tprint


@pytest.fixture
def sample_market_data():
    """Sample market data for testing trading components."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')
    n_periods = len(dates)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(100, 102, n_periods),
        'high': np.random.uniform(102, 105, n_periods),
        'low': np.random.uniform(98, 100, n_periods),
        'close': np.random.uniform(100, 102, n_periods),
        'volume': np.random.uniform(1000, 5000, n_periods),
        'symbol': 'BTCUSDT'
    })
    
    # Ensure high >= open >= low and high >= close >= low
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    return data


@pytest.fixture
def sample_trade_data():
    """Sample trade data for testing."""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='1H'),
        'symbol': ['BTCUSDT'] * 10,
        'side': ['buy', 'sell'] * 5,
        'price': np.random.uniform(100, 102, 10),
        'quantity': np.random.uniform(0.1, 1.0, 10),
        'execution_time': np.random.uniform(0.1, 2.0, 10),
        'status': ['filled'] * 10
    })


@pytest.fixture
def sample_portfolio_data():
    """Sample portfolio data for testing."""
    return {
        'total_value': 100000.0,
        'available_balance': 50000.0,
        'positions': {
            'BTCUSDT': {'quantity': 0.5, 'avg_price': 100.0, 'current_price': 101.0},
            'ETHUSDT': {'quantity': 2.0, 'avg_price': 50.0, 'current_price': 51.0}
        },
        'pnl': 2500.0,
        'pnl_percent': 2.5
    }


@pytest.fixture
def mock_exchange_api():
    """Mock exchange API for testing."""
    mock_api = Mock()
    
    # Mock market data methods
    mock_api.fetch_ohlcv.return_value = [
        [1640995200000, 100.0, 102.0, 98.0, 101.0, 1000.0],  # timestamp, open, high, low, close, volume
        [1640995260000, 101.0, 103.0, 99.0, 102.0, 1100.0]
    ]
    
    # Mock trading methods
    mock_api.create_order.return_value = {
        'id': 'test_order_123',
        'symbol': 'BTCUSDT',
        'side': 'buy',
        'type': 'limit',
        'price': 100.0,
        'amount': 0.1,
        'status': 'open',
        'filled': 0.0
    }
    
    mock_api.cancel_order.return_value = {
        'id': 'test_order_123',
        'status': 'canceled'
    }
    
    mock_api.fetch_order.return_value = {
        'id': 'test_order_123',
        'symbol': 'BTCUSDT',
        'side': 'buy',
        'type': 'limit',
        'price': 100.0,
        'amount': 0.1,
        'status': 'closed',
        'filled': 0.1
    }
    
    # Mock wallet methods
    mock_api.fetch_balance.return_value = {
        'free': {'USDT': 10000.0, 'BTC': 1.0},
        'used': {'USDT': 0.0, 'BTC': 0.0},
        'total': {'USDT': 10000.0, 'BTC': 1.0}
    }
    
    return mock_api


@pytest.fixture
def mock_database():
    """Mock database for testing."""
    mock_db = Mock()
    mock_db.execute.return_value = []
    mock_db.fetchall.return_value = []
    mock_db.commit.return_value = None
    mock_db.rollback.return_value = None
    return mock_db


@pytest.fixture
def test_config():
    """Test configuration for various components."""
    return {
        'trading': {
            'risk_limit': 0.02,
            'position_size': 0.1,
            'max_positions': 5,
            'stop_loss': 0.05,
            'take_profit': 0.1
        },
        'exchange': {
            'api_key': 'test_key',
            'api_secret': 'test_secret',
            'base_url': 'https://api.test.com',
            'timeout': 30
        },
        'data': {
            'lookback_period': 100,
            'timeframe': '1h',
            'symbols': ['BTCUSDT', 'ETHUSDT']
        },
        'logging': {
            'level': 'INFO',
            'file': 'test.log',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }


@pytest.fixture
def sample_signals():
    """Sample trading signals for testing."""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=20, freq='1H'),
        'symbol': ['BTCUSDT'] * 20,
        'signal': np.random.choice([-1, 0, 1], 20),
        'confidence': np.random.uniform(0.5, 1.0, 20),
        'price': np.random.uniform(100, 102, 20),
        'expected_return': np.random.uniform(-0.05, 0.05, 20)
    })


@pytest.fixture
def sample_models():
    """Sample model data for testing."""
    return {
        'model_type': 'random_forest',
        'features': ['close', 'volume', 'rsi', 'macd'],
        'target': 'future_return',
        'parameters': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        },
        'performance': {
            'accuracy': 0.75,
            'precision': 0.73,
            'recall': 0.76,
            'f1_score': 0.74
        }
    }


@pytest.fixture
def sample_error_data():
    """Sample error scenarios for testing."""
    return {
        'network_error': ConnectionError("Network connection failed"),
        'api_error': Exception("API rate limit exceeded"),
        'data_error': ValueError("Invalid data format"),
        'timeout_error': TimeoutError("Request timed out"),
        'auth_error': Exception("Authentication failed")
    }


class TestDataFactory:
    """Factory class for generating test data."""
    
    @staticmethod
    def create_ohlcv_data(n_periods: int = 100, 
                         start_date: str = '2024-01-01',
                         symbol: str = 'BTCUSDT') -> pd.DataFrame:
        """Create OHLCV test data."""
        dates = pd.date_range(start=start_date, periods=n_periods, freq='1H')
        
        # Generate realistic price movements
        price_changes = np.random.normal(0, 0.002, n_periods)  # 0.2% std deviation
        base_price = 100.0
        prices = base_price * (1 + np.cumsum(price_changes))
        
        # Create OHLCV data
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * np.random.uniform(1.0, 1.02, n_periods),
            'low': prices * np.random.uniform(0.98, 1.0, n_periods),
            'close': prices * np.random.uniform(0.99, 1.01, n_periods),
            'volume': np.random.uniform(1000, 10000, n_periods),
            'symbol': symbol
        })
        
        # Ensure proper OHLC relationships
        data['high'] = data[['open', 'close', 'high']].max(axis=1)
        data['low'] = data[['open', 'close', 'low']].min(axis=1)
        
        return data
    
    @staticmethod
    def create_trade_signals(n_signals: int = 50,
                            symbols: List[str] = None) -> pd.DataFrame:
        """Create trading signals test data."""
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT']
        
        dates = pd.date_range(start='2024-01-01', periods=n_signals, freq='1H')
        
        return pd.DataFrame({
            'timestamp': dates,
            'symbol': np.random.choice(symbols, n_signals),
            'signal': np.random.choice([-1, 0, 1], n_signals, p=[0.2, 0.6, 0.2]),
            'confidence': np.random.uniform(0.3, 0.95, n_signals),
            'price': np.random.uniform(50, 200, n_signals),
            'expected_return': np.random.uniform(-0.08, 0.08, n_signals),
            'model_confidence': np.random.uniform(0.4, 0.9, n_signals)
        })
    
    @staticmethod
    def create_portfolio_state(initial_balance: float = 100000.0) -> Dict[str, Any]:
        """Create portfolio state test data."""
        return {
            'total_value': initial_balance,
            'available_balance': initial_balance * 0.7,
            'positions': {
                'BTCUSDT': {
                    'quantity': 0.5,
                    'avg_price': 100.0,
                    'current_price': 105.0,
                    'pnl': 2.5
                }
            },
            'daily_pnl': 2500.0,
            'total_pnl': 2500.0,
            'win_rate': 0.65,
            'total_trades': 20,
            'last_updated': datetime.now()
        }


@pytest.fixture
def data_factory():
    """Test data factory fixture."""
    return TestDataFactory


# Custom pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "trading: marks tests as trading functionality"
    )
    config.addinivalue_line(
        "markers", "exchange: marks tests as exchange integrations"
    )
    config.addinivalue_line(
        "markers", "simulator: marks tests as trading simulator"
    )
    config.addinivalue_line(
        "markers", "data: marks tests as data handling and processing"
    )


# Test utilities
def assert_dataframe_equals(df1: pd.DataFrame, df2: pd.DataFrame, 
                           check_dtype: bool = False, **kwargs):
    """Assert that two DataFrames are equal, handling common test scenarios."""
    pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype, **kwargs)


def assert_series_equals(series1: pd.Series, series2: pd.Series,
                        check_dtype: bool = False, **kwargs):
    """Assert that two Series are equal, handling common test scenarios."""
    pd.testing.assert_series_equal(series1, series2, check_dtype=check_dtype, **kwargs)


def create_mock_coroutine(return_value=None):
    """Create a mock coroutine for async testing."""
    async def mock_coro(*args, **kwargs):
        return return_value
    return Mock(side_effect=mock_coro)


def skip_if_no_tprint():
    """Skip test if tprint is not available."""
    if not TPRINT_AVAILABLE:
        pytest.skip("tprint not available for testing")


# Performance testing utilities
class PerformanceTimer:
    """Context manager for timing test operations."""
    
    def __init__(self, max_duration: Optional[float] = None):
        self.max_duration = max_duration
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        
        if self.max_duration is not None and self.duration > self.max_duration:
            pytest.fail(f"Operation took {self.duration:.3f}s, max allowed: {self.max_duration:.3f}s")


# Import time at the end to avoid conflicts
import time
