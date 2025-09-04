"""Pytest configuration and fixtures for training system tests."""
import pytest
import asyncio
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any
import tempfile
import shutil
from copy import copy
from typing import Dict, List, Optional, Union, Any, Tuple

@pytest.fixture(scope='session')
def event_loop() -> None:
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope='session')
def test_data_dir() -> None:
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix='training_test_')
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide a sample configuration for testing."""
    return {'n_regimes': 3, 'lookback_years': 1, 'train_ratio': 0.7, 'val_ratio': 0.15, 'test_ratio': 0.15, 'feature_engineering_config': {'use_technical_indicators': True, 'use_interaction_features': True, 'use_regime_features': True, 'feature_selection': {'enabled': True, 'max_features': 50}}, 'matrix_operations_config': {'use_gpu': False, 'batch_size': 100, 'optimization_level': 'medium'}, 'labeling_config': {'use_triple_barrier': True, 'barrier_config': {'profit_taking': 0.02, 'stop_loss': 0.01, 'max_holding_period': 50}}}

@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    """Generate sample market data for testing."""
    np.random.seed(42)
    n_samples = 1000
    returns = np.random.normal(0.0001, 0.01, n_samples)
    price = 50000 * np.exp(np.cumsum(returns))
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1H')
    data = pd.DataFrame({'open': price * (1 + np.random.uniform(-0.001, 0.001, n_samples)), 'high': price * (1 + np.random.uniform(0, 0.005, n_samples)), 'low': price * (1 - np.random.uniform(0, 0.005, n_samples)), 'close': price, 'volume': np.random.uniform(100, 1000, n_samples)}, index=dates)
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    return data

@pytest.fixture
def sample_features(sample_market_data: Any) -> pd.DataFrame:
    """Generate sample features from market data."""
    data = sample_market_data.copy()
    data['feature_returns'] = data['close'].pct_change()
    data['feature_log_returns'] = np.log(data['close'] / data['close'].shift(1))
    data['feature_volatility'] = data['feature_returns'].rolling(20).std()
    data['feature_sma_10'] = data['close'].rolling(10).mean()
    data['feature_sma_20'] = data['close'].rolling(20).mean()
    data['feature_volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['feature_rsi'] = 100 - 100 / (1 + rs)
    data = data.dropna()
    return data

@pytest.fixture
def sample_regime_labels(sample_features: List[Any]) -> np.ndarray:
    """Generate sample regime labels."""
    n_samples = len(sample_features)
    regime_durations = [100, 150, 200, 150, 200, 200]
    regimes = []
    current_regime = 0
    for duration in regime_durations:
        regimes.extend([current_regime] * duration)
        current_regime = (current_regime + 1) % 3
    if len(regimes) > n_samples:
        regimes = regimes[:n_samples]
    else:
        regimes.extend([2] * (n_samples - len(regimes)))
    return np.array(regimes)

@pytest.fixture
def sample_pipeline_state(sample_features: List[Any], sample_regime_labels: List[Any]) -> Dict[str, Any]:
    """Create a sample pipeline state for testing."""
    return {'validated_data': sample_features, 'features': sample_features, 'regime_labels': sample_regime_labels, 'data_validation_results': {'data_quality_score': 95, 'has_required_columns': True, 'missing_data_pct': 0.0}, 'regime_characteristics': {'regime_0': {'count': 350, 'percentage': 35.0, 'volatility_20_mean': 0.01, 'returns_mean': 0.0001}, 'regime_1': {'count': 350, 'percentage': 35.0, 'volatility_20_mean': 0.015, 'returns_mean': -0.0001}, 'regime_2': {'count': 300, 'percentage': 30.0, 'volatility_20_mean': 0.02, 'returns_mean': 0.0002}}}

@pytest.fixture
def sample_training_input(test_data_dir: Any) -> Dict[str, Any]:
    """Create sample training input parameters."""
    return {'symbol': 'BTCUSDT', 'exchange': 'binance', 'timeframe': '1h', 'data_dir': str(test_data_dir / 'data'), 'output_dir': str(test_data_dir / 'output'), 'force_download': False}

class MockStep:
    """Mock step for testing pipeline integration."""

    def __init__(self, step_name: str, outputs: Dict[str, Any]) -> None:
        self.step_name = step_name
        self.outputs = outputs
        self.executed = False

    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute mock step."""
        self.executed = True
        pipeline_state.update(self.outputs)
        return pipeline_state

    def validate_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate inputs."""
        return (True, [])

    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate outputs."""
        return (True, [])

@pytest.fixture
def mock_steps() -> Dict[str, MockStep]:
    """Create mock steps for testing."""
    return {'01_data_collection': MockStep('data_collection', {'raw_market_data': 'path/to/data.parquet'}), '02_data_reading': MockStep('data_reading', {'validated_data': pd.DataFrame()}), '03_hmm_regime_discovery': MockStep('hmm_regime_discovery', {'regime_labels': np.array([0, 1, 2, 0, 1])})}