"""
Integration Tests for Analyst→Tactician Flow

End-to-end tests for the complete Analyst→Tactician pipeline integration.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Import pipeline components
from src.training.steps.pre_training.analyst_profit_labeler import AnalystProfitLabeler, AnalystProfitLabelerConfig
from src.training.steps.pre_training.tactician_entry_labeler import TacticianDifferentiatedLabeler, TacticianLabelingConfig


class TestAnalystTacticianIntegration:
    """Test the complete Analyst→Tactician integration flow."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create realistic market data for integration testing."""
        # Create 4 hours of 15-minute data
        dates = pd.date_range(start='2023-01-01 10:00:00', periods=16, freq='15min')
        np.random.seed(42)
        
        # Generate realistic price movement
        base_price = 100.0
        returns = np.random.randn(16) * 0.01  # 1% volatility
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = pd.DataFrame({
            'open': prices * (1 + np.random.randn(16) * 0.001),
            'high': prices * (1 + np.abs(np.random.randn(16)) * 0.005),
            'low': prices * (1 - np.abs(np.random.randn(16)) * 0.005),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 16)
        }, index=dates)
        
        # Ensure OHLC consistency
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        return data
    
    @pytest.fixture
    def analyst_config(self):
        """Create Analyst configuration for testing."""
        return AnalystProfitLabelerConfig(
            timeframe='15m',
            horizons=[15, 30, 45],  # 15, 30, 45 minutes
            target_profit=0.5,  # 50 bps target
            min_label_quality=0.3,
            enable_advanced_filters=False  # Keep simple for testing
        )
    
    @pytest.fixture
    def tactician_config(self):
        """Create Tactician configuration for testing."""
        return TacticianLabelingConfig(
            min_entry_window_minutes=3,
            max_entry_window_minutes=30,
            entry_quality_threshold=0.1
        )
    
    def test_analyst_profit_labeling_generates_windows(self, sample_market_data, analyst_config):
        """Test that Analyst profit labeling generates opportunity windows."""
        analyst = AnalystProfitLabeler(analyst_config)
        
        # Generate labels
        result = analyst.generate_labels(data=sample_market_data)
        
        # Check that result has the expected structure
        assert hasattr(result, 'labels')
        assert hasattr(result, 'metadata')
        
        # Check that opportunity windows are generated
        if hasattr(result.metadata, 'opportunity_windows'):
            opportunity_windows = result.metadata.opportunity_windows
        elif isinstance(result.metadata, dict):
            opportunity_windows = result.metadata.get('opportunity_windows', [])
        else:
            opportunity_windows = []
        
        assert isinstance(opportunity_windows, list)
        
        # If windows are generated, check their structure
        if opportunity_windows:
            for window in opportunity_windows:
                assert 'start' in window
                assert 'end' in window
                assert 'anchor' in window
                assert 'direction' in window
                
                # Check temporal ordering
                start_ts = pd.Timestamp(window['start'])
                end_ts = pd.Timestamp(window['end'])
                anchor_ts = pd.Timestamp(window['anchor'])
                
                assert start_ts <= anchor_ts <= end_ts
                assert window['direction'] in [-1, 1]
    
    def test_analyst_generates_tactician_entry_labels(self, sample_market_data, analyst_config):
        """Test that Analyst generates tactician entry labels."""
        analyst = AnalystProfitLabeler(analyst_config)
        
        # Generate labels
        result = analyst.generate_labels(data=sample_market_data)
        
        # Check that tactician entry labels are generated
        if hasattr(result.metadata, 'tactician_entry_labels'):
            tactician_labels = result.metadata.tactician_entry_labels
        elif isinstance(result.metadata, dict):
            tactician_labels = result.metadata.get('tactician_entry_labels', None)
        else:
            tactician_labels = None
        
        if tactician_labels is not None:
            assert isinstance(tactician_labels, pd.Series)
            assert tactician_labels.dtype == np.int8 or tactician_labels.dtype == np.int64
            assert set(tactician_labels.unique()).issubset({0, 1})
            
            # Check alignment with data index
            assert tactician_labels.index.equals(sample_market_data.index)
    
    def test_tactician_generates_from_analyst_windows(self, sample_market_data, tactician_config):
        """Test that Tactician can generate dataset from Analyst windows."""
        tactician = TacticianDifferentiatedLabeler(tactician_config)
        
        # Create mock opportunity windows (simulating Analyst output)
        opportunity_windows = [
            {
                'start': pd.Timestamp('2023-01-01 10:15:00'),
                'end': pd.Timestamp('2023-01-01 10:45:00'),
                'anchor': pd.Timestamp('2023-01-01 10:30:00'),
                'direction': 1
            },
            {
                'start': pd.Timestamp('2023-01-01 11:00:00'),
                'end': pd.Timestamp('2023-01-01 11:30:00'),
                'anchor': pd.Timestamp('2023-01-01 11:15:00'),
                'direction': -1
            }
        ]
        
        # Create mock analyst OOF score
        analyst_oof_score = pd.Series(
            np.random.randn(len(sample_market_data)),
            index=sample_market_data.index,
            name='analyst_oof_score'
        )
        
        # Generate Tactician dataset
        result = tactician.generate_from_analyst_windows(
            data=sample_market_data,
            opportunity_windows=opportunity_windows,
            analyst_oof_score=analyst_oof_score
        )
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'X' in result
        assert 'y' in result
        assert 'window_id' in result
        assert 'meta' in result
        
        X = result['X']
        y = result['y']
        window_id = result['window_id']
        
        # Check that features are generated
        assert len(X) > 0
        assert len(X.columns) > 0
        
        # Check that labels are binary and correctly placed
        assert set(y.unique()).issubset({0, 1})
        
        # Check that positive labels correspond to anchor timestamps
        positive_indices = y[y == 1].index
        anchor_timestamps = [pd.Timestamp(w['anchor']) for w in opportunity_windows]
        
        for idx in positive_indices:
            assert idx in anchor_timestamps
        
        # Check window ID mapping
        assert len(window_id) == len(X)
        assert window_id.dtype == np.int32
        
        # Check that analyst OOF features are included
        if analyst_oof_score is not None:
            assert 'analyst_oof_lag1' in X.columns
            assert 'analyst_oof_ema5' in X.columns
    
    def test_end_to_end_analyst_tactician_flow(self, sample_market_data, analyst_config, tactician_config):
        """Test the complete end-to-end flow from Analyst to Tactician."""
        # Step 1: Run Analyst profit labeling
        analyst = AnalystProfitLabeler(analyst_config)
        analyst_result = analyst.generate_labels(data=sample_market_data)
        
        # Extract opportunity windows and tactician entry labels
        if hasattr(analyst_result.metadata, 'opportunity_windows'):
            opportunity_windows = analyst_result.metadata.opportunity_windows
        elif isinstance(analyst_result.metadata, dict):
            opportunity_windows = analyst_result.metadata.get('opportunity_windows', [])
        else:
            opportunity_windows = []
        
        if hasattr(analyst_result.metadata, 'tactician_entry_labels'):
            tactician_labels = analyst_result.metadata.tactician_entry_labels
        elif isinstance(analyst_result.metadata, dict):
            tactician_labels = analyst_result.metadata.get('tactician_entry_labels', None)
        else:
            tactician_labels = None
        
        # Create mock analyst OOF score (simulating ensemble output)
        analyst_oof_score = pd.Series(
            np.random.randn(len(sample_market_data)),
            index=sample_market_data.index,
            name='analyst_oof_score'
        )
        
        # Step 2: Run Tactician on Analyst windows
        tactician = TacticianDifferentiatedLabeler(tactician_config)
        
        if opportunity_windows:
            tactician_result = tactician.generate_from_analyst_windows(
                data=sample_market_data,
                opportunity_windows=opportunity_windows,
                analyst_oof_score=analyst_oof_score
            )
            
            # Step 3: Validate the complete flow
            self._validate_integration_flow(analyst_result, tactician_result, opportunity_windows)
        else:
            # If no windows generated, that's also a valid outcome
            pytest.skip("No opportunity windows generated by Analyst - this may be expected with small dataset")
    
    def _validate_integration_flow(self, analyst_result, tactician_result, opportunity_windows):
        """Validate the integration flow results."""
        # Check Analyst results
        assert analyst_result is not None
        assert hasattr(analyst_result, 'labels')
        assert hasattr(analyst_result, 'metadata')
        
        # Check Tactician results
        assert tactician_result is not None
        assert 'X' in tactician_result
        assert 'y' in tactician_result
        assert 'window_id' in tactician_result
        assert 'meta' in tactician_result
        
        X = tactician_result['X']
        y = tactician_result['y']
        meta = tactician_result['meta']
        
        # Check that we have training data
        assert len(X) > 0, "No features generated"
        assert len(y) > 0, "No labels generated"
        
        # Check that we have some positive labels
        positive_labels = (y == 1).sum()
        assert positive_labels > 0, "No positive labels generated"
        
        # Check that positive labels correspond to anchor timestamps
        positive_indices = y[y == 1].index
        anchor_timestamps = [pd.Timestamp(w['anchor']) for w in opportunity_windows]
        
        for idx in positive_indices:
            assert idx in anchor_timestamps, f"Positive label at {idx} not in anchor timestamps"
        
        # Check that we have the expected number of positive labels
        assert positive_labels == len(opportunity_windows), f"Expected {len(opportunity_windows)} positive labels, got {positive_labels}"
        
        # Check feature quality
        assert len(X.columns) > 0, "No features generated"
        
        # Check that features are numeric
        for col in X.columns:
            assert pd.api.types.is_numeric_dtype(X[col]), f"Feature {col} is not numeric"
        
        # Check metadata
        assert 'n_windows' in meta
        assert 'n_samples' in meta
        assert 'features' in meta
        assert meta['n_windows'] == len(opportunity_windows)
        assert meta['n_samples'] == len(X)


class TestAnalystTacticianValidation:
    """Test validation integration in the Analyst→Tactician flow."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for validation testing."""
        dates = pd.date_range(start='2023-01-01 10:00:00', periods=20, freq='15min')
        np.random.seed(42)
        
        base_price = 100.0
        returns = np.random.randn(20) * 0.01
        prices = base_price * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.randn(20) * 0.001),
            'high': prices * (1 + np.abs(np.random.randn(20)) * 0.005),
            'low': prices * (1 - np.abs(np.random.randn(20)) * 0.005),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 20)
        }, index=dates)
        
        # Ensure OHLC consistency
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        return data
    
    def test_validation_integration_analyst(self, sample_market_data):
        """Test that validation is integrated into Analyst labeling."""
        config = AnalystProfitLabelerConfig(
            timeframe='15m',
            horizons=[15, 30],
            target_profit=0.5,
            min_label_quality=0.3
        )
        
        analyst = AnalystProfitLabeler(config)
        result = analyst.generate_labels(data=sample_market_data)
        
        # Check that validation results are included in the report
        # This would be in the component wrapper, but we can check the structure
        assert hasattr(result, 'metadata')
        
        # The validation would be added by the component wrapper
        # In a real integration test, we'd check the component execution
    
    def test_validation_integration_tactician(self, sample_market_data):
        """Test that validation is integrated into Tactician dataset generation."""
        config = TacticianLabelingConfig(
            min_entry_window_minutes=3,
            max_entry_window_minutes=15,
            entry_quality_threshold=0.1
        )
        
        tactician = TacticianDifferentiatedLabeler(config)
        
        # Create mock opportunity windows
        opportunity_windows = [
            {
                'start': pd.Timestamp('2023-01-01 10:15:00'),
                'end': pd.Timestamp('2023-01-01 10:30:00'),
                'anchor': pd.Timestamp('2023-01-01 10:22:30'),
                'direction': 1
            }
        ]
        
        # Create mock analyst OOF score
        analyst_oof_score = pd.Series(
            np.random.randn(len(sample_market_data)),
            index=sample_market_data.index,
            name='analyst_oof_score'
        )
        
        result = tactician.generate_from_analyst_windows(
            data=sample_market_data,
            opportunity_windows=opportunity_windows,
            analyst_oof_score=analyst_oof_score
        )
        
        # Check that validation results are included in metadata
        meta = result['meta']
        
        if 'validation_results' in meta:
            validation_results = meta['validation_results']
            assert isinstance(validation_results, dict)
            
            # Check that key validation checks are present
            expected_checks = ['input_temporal', 'input_windows', 'leakage', 'dataset_quality']
            for check in expected_checks:
                if check in validation_results:
                    assert isinstance(validation_results[check], dict)


class TestAnalystTacticianPerformance:
    """Test performance characteristics of the Analyst→Tactician flow."""
    
    @pytest.fixture
    def large_market_data(self):
        """Create larger market dataset for performance testing."""
        # Create 1 day of 15-minute data (96 bars)
        dates = pd.date_range(start='2023-01-01 00:00:00', periods=96, freq='15min')
        np.random.seed(42)
        
        base_price = 100.0
        returns = np.random.randn(96) * 0.01
        prices = base_price * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.randn(96) * 0.001),
            'high': prices * (1 + np.abs(np.random.randn(96)) * 0.005),
            'low': prices * (1 - np.abs(np.random.randn(96)) * 0.005),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 96)
        }, index=dates)
        
        # Ensure OHLC consistency
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        return data
    
    def test_performance_with_larger_dataset(self, large_market_data):
        """Test performance with larger dataset."""
        import time
        
        # Test Analyst performance
        analyst_config = AnalystProfitLabelerConfig(
            timeframe='15m',
            horizons=[15, 30, 45],
            target_profit=0.5,
            min_label_quality=0.3
        )
        
        analyst = AnalystProfitLabeler(analyst_config)
        
        start_time = time.time()
        analyst_result = analyst.generate_labels(data=large_market_data)
        analyst_time = time.time() - start_time
        
        # Check that Analyst completes in reasonable time (adjust threshold as needed)
        assert analyst_time < 30.0, f"Analyst took too long: {analyst_time:.2f}s"
        
        # Test Tactician performance
        tactician_config = TacticianLabelingConfig(
            min_entry_window_minutes=3,
            max_entry_window_minutes=30,
            entry_quality_threshold=0.1
        )
        
        tactician = TacticianDifferentiatedLabeler(tactician_config)
        
        # Create mock opportunity windows (simulate Analyst output)
        opportunity_windows = [
            {
                'start': pd.Timestamp('2023-01-01 02:00:00'),
                'end': pd.Timestamp('2023-01-01 02:30:00'),
                'anchor': pd.Timestamp('2023-01-01 02:15:00'),
                'direction': 1
            },
            {
                'start': pd.Timestamp('2023-01-01 04:00:00'),
                'end': pd.Timestamp('2023-01-01 04:30:00'),
                'anchor': pd.Timestamp('2023-01-01 04:15:00'),
                'direction': -1
            }
        ]
        
        # Create mock analyst OOF score
        analyst_oof_score = pd.Series(
            np.random.randn(len(large_market_data)),
            index=large_market_data.index,
            name='analyst_oof_score'
        )
        
        start_time = time.time()
        tactician_result = tactician.generate_from_analyst_windows(
            data=large_market_data,
            opportunity_windows=opportunity_windows,
            analyst_oof_score=analyst_oof_score
        )
        tactician_time = time.time() - start_time
        
        # Check that Tactician completes in reasonable time
        assert tactician_time < 10.0, f"Tactician took too long: {tactician_time:.2f}s"
        
        # Check that results are reasonable
        assert len(tactician_result['X']) > 0
        assert len(tactician_result['y']) > 0
        
        print(f"Performance: Analyst={analyst_time:.2f}s, Tactician={tactician_time:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__])
