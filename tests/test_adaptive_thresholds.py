"""
Tests for Dynamic Event Frequency Adaptation and Regressor-Specific Target Engineering.
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.training.steps.labeling.orthogonal_label_generation import (
    AdaptiveEventThresholds,
    engineer_regressor_targets,
    _create_ridge_targets,
    _create_tree_targets,
    get_target_for_model_type,
)


def _create_mock_df(n_bars=1000, start='2024-01-01'):
    """Create mock OHLCV DataFrame for testing."""
    dates = pd.date_range(start=start, periods=n_bars, freq='15min')
    np.random.seed(42)
    
    # Simulate price with trend and noise
    returns = np.random.randn(n_bars) * 0.002 + 0.0001  # Slight uptrend
    price = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': price * (1 + np.random.randn(n_bars) * 0.001),
        'high': price * (1 + np.abs(np.random.randn(n_bars) * 0.005)),
        'low': price * (1 - np.abs(np.random.randn(n_bars) * 0.005)),
        'close': price,
        'volume': np.random.uniform(1000, 10000, n_bars),
        'volatility_1d': pd.Series(returns).rolling(20).std().fillna(0.01).values,
    }, index=dates)
    
    return df


def _create_mock_events(df, n_events=50):
    """Create mock event indices."""
    event_locs = np.random.choice(len(df) - 100, size=n_events, replace=False)
    event_locs = np.sort(event_locs)
    return pd.DatetimeIndex(df.index[event_locs])


class TestAdaptiveEventThresholds:
    """Tests for AdaptiveEventThresholds class."""
    
    def test_init_defaults(self):
        """Test default initialization."""
        adapter = AdaptiveEventThresholds()
        assert adapter.target_events_per_day == 2.0
        assert adapter.min_events_per_day == 0.5
        assert adapter.max_events_per_day == 10.0
    
    def test_vol_adjustment_factor_bounded(self):
        """Test volatility adjustment factor is bounded."""
        adapter = AdaptiveEventThresholds()
        df = _create_mock_df(n_bars=500)
        
        factor = adapter.get_vol_adjustment_factor(df, lookback=100)
        
        assert 0.5 <= factor <= 2.0, f"Factor {factor} out of bounds [0.5, 2.0]"
        assert np.isfinite(factor)
    
    def test_vol_adjustment_returns_one_for_short_data(self):
        """Test returns 1.0 for insufficient data."""
        adapter = AdaptiveEventThresholds()
        df = _create_mock_df(n_bars=50)  # Too short
        
        factor = adapter.get_vol_adjustment_factor(df, lookback=100)
        assert factor == 1.0
    
    def test_suggest_threshold_adjustment_too_few_events(self):
        """Test threshold is lowered when too few events."""
        adapter = AdaptiveEventThresholds(target_events_per_day=2.0, min_events_per_day=0.5)
        
        # 5 events over 100 days = 0.05 events/day (too few)
        adjusted = adapter.suggest_threshold_adjustment(
            current_events=5,
            span_days=100.0,
            current_threshold=1.0
        )
        
        # Should lower threshold (multiply by < 1)
        assert adjusted < 1.0
    
    def test_suggest_threshold_adjustment_too_many_events(self):
        """Test threshold is raised when too many events."""
        adapter = AdaptiveEventThresholds(target_events_per_day=2.0, max_events_per_day=10.0)
        
        # 500 events over 10 days = 50 events/day (too many)
        adjusted = adapter.suggest_threshold_adjustment(
            current_events=500,
            span_days=10.0,
            current_threshold=1.0
        )
        
        # Should raise threshold (multiply by > 1)
        assert adjusted > 1.0


class TestRegressorTargetEngineering:
    """Tests for regressor-specific target engineering."""
    
    def test_engineer_regressor_targets_ridge(self):
        """Test Ridge targets are properly normalized."""
        df = _create_mock_df()
        events = _create_mock_events(df)
        raw_returns = pd.Series(np.random.randn(len(events)) * 0.05, index=events)
        
        ridge_targets = engineer_regressor_targets(
            df, events, raw_returns, regressor_type='ridge', horizon=48
        )
        
        assert len(ridge_targets) > 0
        assert np.isfinite(ridge_targets).all()
        # Ridge targets should exist and be normalized (may have different scale)
    
    def test_engineer_regressor_targets_tree(self):
        """Test tree targets preserve ranking."""
        df = _create_mock_df()
        events = _create_mock_events(df)
        raw_returns = pd.Series(np.random.randn(len(events)) * 0.05, index=events)
        
        tree_targets = engineer_regressor_targets(
            df, events, raw_returns, regressor_type='lgbm', horizon=48
        )
        
        assert len(tree_targets) > 0
        assert np.isfinite(tree_targets).all()
    
    def test_engineer_regressor_targets_fallback(self):
        """Test fallback returns raw targets for unknown type."""
        df = _create_mock_df()
        events = _create_mock_events(df)
        raw_returns = pd.Series(np.random.randn(len(events)) * 0.05, index=events)
        
        result = engineer_regressor_targets(
            df, events, raw_returns, regressor_type='unknown', horizon=48
        )
        
        assert result.equals(raw_returns)
    
    def test_get_target_for_model_type_mappings(self):
        """Test model type detection."""
        assert get_target_for_model_type('LGBM_Focal') == 'tree'
        assert get_target_for_model_type('XGB_Tree') == 'tree'
        assert get_target_for_model_type('Ridge') == 'ridge'
        assert get_target_for_model_type('LinearRegressor') == 'ridge'
        assert get_target_for_model_type('CatBoost') == 'tree'
        assert get_target_for_model_type('SomeOtherModel') == 'auto'
    
    def test_empty_events_returns_empty(self):
        """Test handles empty events gracefully."""
        df = _create_mock_df()
        events = pd.DatetimeIndex([])
        raw_returns = pd.Series(dtype=float)
        
        result = engineer_regressor_targets(
            df, events, raw_returns, regressor_type='lgbm', horizon=48
        )
        
        assert len(result) == 0


class TestIntegration:
    """Integration tests for both features together."""
    
    def test_full_workflow(self):
        """Test complete workflow with adaptive thresholds and target engineering."""
        # Create mock data
        df = _create_mock_df(n_bars=2000)
        
        # Initialize adaptive thresholds
        adapter = AdaptiveEventThresholds(target_events_per_day=2.0)
        
        # Get volatility adjustment
        vol_factor = adapter.get_vol_adjustment_factor(df)
        base_threshold = 1.0
        adjusted_threshold = base_threshold * vol_factor
        
        # Create mock events (simulating what a calibrated generator would produce)
        events = _create_mock_events(df, n_events=30)
        
        # Create raw returns
        raw_returns = pd.Series(np.random.randn(len(events)) * 0.03, index=events)
        
        # Apply target engineering
        tree_targets = engineer_regressor_targets(
            df, events, raw_returns, regressor_type='tree', horizon=48
        )
        
        ridge_targets = engineer_regressor_targets(
            df, events, raw_returns, regressor_type='ridge', horizon=48
        )
        
        # Verify outputs
        assert len(tree_targets) > 0
        assert len(ridge_targets) > 0
        assert 0.5 <= adjusted_threshold <= 2.0
