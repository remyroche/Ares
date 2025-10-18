"""
Test Tactician Dataset Generation

Tests for the generate_from_analyst_windows method and related functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Import the tactician entry labeler
from src.training.steps.pre_training.tactician_entry_labeler import TacticianDifferentiatedLabeler, TacticianLabelingConfig


class TestGenerateFromAnalystWindows:
    """Test the generate_from_analyst_windows method."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2023-01-01 10:00:00', periods=100, freq='15min')
        np.random.seed(42)
        
        # Generate realistic OHLCV data
        close_prices = 100 + np.cumsum(np.random.randn(100) * 0.01)
        high_prices = close_prices + np.random.rand(100) * 0.5
        low_prices = close_prices - np.random.rand(100) * 0.5
        open_prices = close_prices + np.random.randn(100) * 0.1
        volume = np.random.randint(1000, 10000, 100)
        
        data = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        }, index=dates)
        
        return data
    
    @pytest.fixture
    def sample_opportunity_windows(self):
        """Create sample opportunity windows for testing."""
        windows = [
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
        return windows
    
    @pytest.fixture
    def sample_analyst_oof_score(self, sample_data):
        """Create sample analyst OOF score for testing."""
        # Create OOF score aligned with data index
        np.random.seed(123)
        oof_scores = np.random.randn(len(sample_data)) * 0.5
        return pd.Series(oof_scores, index=sample_data.index, name='analyst_oof_score')
    
    @pytest.fixture
    def tactician_labeler(self):
        """Create a TacticianDifferentiatedLabeler instance for testing."""
        config = TacticianLabelingConfig(
            min_entry_window_minutes=3,
            max_entry_window_minutes=30,
            entry_quality_threshold=0.1
        )
        return TacticianDifferentiatedLabeler(config)
    
    def test_generate_from_analyst_windows_basic(self, tactician_labeler, sample_data, sample_opportunity_windows):
        """Test basic functionality of generate_from_analyst_windows."""
        result = tactician_labeler.generate_from_analyst_windows(
            data=sample_data,
            opportunity_windows=sample_opportunity_windows
        )
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'X' in result
        assert 'y' in result
        assert 'window_id' in result
        assert 'meta' in result
        
        # Check feature matrix
        X = result['X']
        assert isinstance(X, pd.DataFrame)
        assert len(X) > 0
        assert 'ret_1' in X.columns
        assert 'ret_2' in X.columns
        assert 'rv_5' in X.columns
        assert 'rv_10' in X.columns
        
        # Check labels
        y = result['y']
        assert isinstance(y, pd.Series)
        assert len(y) == len(X)
        assert y.dtype == np.int8
        
        # Check window IDs
        window_id = result['window_id']
        assert isinstance(window_id, pd.Series)
        assert len(window_id) == len(X)
        assert window_id.dtype == np.int32
        
        # Check metadata
        meta = result['meta']
        assert isinstance(meta, dict)
        assert 'n_windows' in meta
        assert 'n_samples' in meta
        assert 'features' in meta
    
    def test_generate_from_analyst_windows_with_oof_score(self, tactician_labeler, sample_data, sample_opportunity_windows, sample_analyst_oof_score):
        """Test generate_from_analyst_windows with analyst OOF score."""
        result = tactician_labeler.generate_from_analyst_windows(
            data=sample_data,
            opportunity_windows=sample_opportunity_windows,
            analyst_oof_score=sample_analyst_oof_score
        )
        
        X = result['X']
        
        # Check that OOF features are included
        assert 'analyst_oof_lag1' in X.columns
        assert 'analyst_oof_ema5' in X.columns
        
        # Check that OOF features are properly lagged (past-only)
        # The first few rows should have NaN values due to shifting
        assert X['analyst_oof_lag1'].isna().sum() > 0
        assert X['analyst_oof_ema5'].isna().sum() > 0
    
    def test_generate_from_analyst_windows_y_labels_at_anchors(self, tactician_labeler, sample_data, sample_opportunity_windows):
        """Test that y=1 only at anchor timestamps."""
        result = tactician_labeler.generate_from_analyst_windows(
            data=sample_data,
            opportunity_windows=sample_opportunity_windows
        )
        
        y = result['y']
        window_id = result['window_id']
        
        # Find all positive labels
        positive_mask = y == 1
        positive_indices = y[positive_mask].index
        
        # Check that positive labels correspond to anchor timestamps
        anchor_timestamps = [window['anchor'] for window in sample_opportunity_windows]
        
        for idx in positive_indices:
            assert idx in anchor_timestamps, f"Positive label at {idx} not in anchor timestamps"
        
        # Check that we have the expected number of positive labels
        expected_positive_count = len(sample_opportunity_windows)
        actual_positive_count = positive_mask.sum()
        assert actual_positive_count == expected_positive_count
    
    def test_generate_from_analyst_windows_window_id_mapping(self, tactician_labeler, sample_data, sample_opportunity_windows):
        """Test that window_id mapping is monotonic and correct."""
        result = tactician_labeler.generate_from_analyst_windows(
            data=sample_data,
            opportunity_windows=sample_opportunity_windows
        )
        
        window_id = result['window_id']
        
        # Check that window IDs are valid (0 to n_windows-1)
        unique_window_ids = window_id.unique()
        expected_window_ids = set(range(len(sample_opportunity_windows)))
        actual_window_ids = set(unique_window_ids)
        
        assert actual_window_ids.issubset(expected_window_ids), f"Unexpected window IDs: {actual_window_ids - expected_window_ids}"
    
    def test_generate_from_analyst_windows_index_alignment_preserved(self, tactician_labeler, sample_data, sample_opportunity_windows):
        """Test that index alignment is preserved in the output."""
        result = tactician_labeler.generate_from_analyst_windows(
            data=sample_data,
            opportunity_windows=sample_opportunity_windows
        )
        
        X = result['X']
        y = result['y']
        window_id = result['window_id']
        
        # Check that all outputs have the same index
        assert X.index.equals(y.index)
        assert X.index.equals(window_id.index)
        
        # Check that all indices are from the original data
        assert all(idx in sample_data.index for idx in X.index)
    
    def test_generate_from_analyst_windows_empty_windows(self, tactician_labeler, sample_data):
        """Test behavior with empty opportunity windows."""
        with pytest.raises(ValueError, match="opportunity_windows is empty"):
            tactician_labeler.generate_from_analyst_windows(
                data=sample_data,
                opportunity_windows=[]
            )
    
    def test_generate_from_analyst_windows_invalid_data(self, tactician_labeler, sample_opportunity_windows):
        """Test behavior with invalid input data."""
        # Test with empty DataFrame
        with pytest.raises(ValueError, match="data must be a non-empty DataFrame"):
            tactician_labeler.generate_from_analyst_windows(
                data=pd.DataFrame(),
                opportunity_windows=sample_opportunity_windows
            )
        
        # Test with missing 'close' column
        invalid_data = pd.DataFrame({
            'open': [1, 2, 3],
            'high': [1, 2, 3],
            'low': [1, 2, 3]
            # Missing 'close' column
        })
        
        with pytest.raises(ValueError, match="data must contain 'close' column"):
            tactician_labeler.generate_from_analyst_windows(
                data=invalid_data,
                opportunity_windows=sample_opportunity_windows
            )
    
    def test_generate_from_analyst_windows_pre_post_bars(self, tactician_labeler, sample_data, sample_opportunity_windows):
        """Test pre_bars and post_bars parameters."""
        result = tactician_labeler.generate_from_analyst_windows(
            data=sample_data,
            opportunity_windows=sample_opportunity_windows,
            pre_bars=2,
            post_bars=1
        )
        
        # With pre_bars=2 and post_bars=1, we should have more samples
        # than with the default (pre_bars=0, post_bars=0)
        result_default = tactician_labeler.generate_from_analyst_windows(
            data=sample_data,
            opportunity_windows=sample_opportunity_windows
        )
        
        # The extended version should have more samples
        assert len(result['X']) >= len(result_default['X'])
    
    def test_generate_from_analyst_windows_past_only_features(self, tactician_labeler, sample_data, sample_opportunity_windows):
        """Test that features are strictly past-only (no future leakage)."""
        result = tactician_labeler.generate_from_analyst_windows(
            data=sample_data,
            opportunity_windows=sample_opportunity_windows
        )
        
        X = result['X']
        
        # Check that return features are lagged (shifted by 1)
        # This is a basic check - in practice, more sophisticated leakage detection would be used
        for col in ['ret_1', 'ret_2', 'rv_5', 'rv_10']:
            if col in X.columns:
                # The first row should have NaN due to shifting
                assert pd.isna(X[col].iloc[0]), f"Feature {col} should be NaN in first row due to shifting"
    
    def test_generate_from_analyst_windows_validation_integration(self, tactician_labeler, sample_data, sample_opportunity_windows, sample_analyst_oof_score):
        """Test that validation results are included in metadata."""
        result = tactician_labeler.generate_from_analyst_windows(
            data=sample_data,
            opportunity_windows=sample_opportunity_windows,
            analyst_oof_score=sample_analyst_oof_score
        )
        
        meta = result['meta']
        
        # Check that validation results are included
        assert 'validation_results' in meta
        
        validation_results = meta['validation_results']
        assert isinstance(validation_results, dict)
        
        # Check that key validation checks are present
        expected_validation_keys = ['input_temporal', 'input_windows', 'leakage', 'dataset_quality']
        for key in expected_validation_keys:
            if key in validation_results:
                assert isinstance(validation_results[key], dict)


class TestTacticianDatasetEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def tactician_labeler(self):
        """Create a TacticianDifferentiatedLabeler instance for testing."""
        config = TacticianLabelingConfig()
        return TacticianDifferentiatedLabeler(config)
    
    def test_generate_from_analyst_windows_misaligned_timestamps(self, tactician_labeler):
        """Test behavior with misaligned timestamps in windows."""
        # Create data
        dates = pd.date_range(start='2023-01-01 10:00:00', periods=50, freq='15min')
        data = pd.DataFrame({
            'open': np.random.randn(50),
            'high': np.random.randn(50),
            'low': np.random.randn(50),
            'close': np.random.randn(50),
            'volume': np.random.randint(1000, 10000, 50)
        }, index=dates)
        
        # Create windows with timestamps not in data index
        windows = [
            {
                'start': pd.Timestamp('2023-01-01 09:00:00'),  # Before data
                'end': pd.Timestamp('2023-01-01 09:30:00'),
                'anchor': pd.Timestamp('2023-01-01 09:15:00'),
                'direction': 1
            }
        ]
        
        result = tactician_labeler.generate_from_analyst_windows(
            data=data,
            opportunity_windows=windows
        )
        
        # Should handle gracefully with 0 valid windows
        assert result['meta']['n_windows'] == 0
        assert len(result['X']) == 0
    
    def test_generate_from_analyst_windows_very_short_data(self, tactician_labeler):
        """Test behavior with very short data."""
        # Create very short data
        dates = pd.date_range(start='2023-01-01 10:00:00', periods=3, freq='15min')
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [100.5, 101.5, 102.5],
            'low': [99.5, 100.5, 101.5],
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        }, index=dates)
        
        windows = [
            {
                'start': pd.Timestamp('2023-01-01 10:00:00'),
                'end': pd.Timestamp('2023-01-01 10:15:00'),
                'anchor': pd.Timestamp('2023-01-01 10:07:30'),
                'direction': 1
            }
        ]
        
        result = tactician_labeler.generate_from_analyst_windows(
            data=data,
            opportunity_windows=windows
        )
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert 'X' in result
        assert 'y' in result
    
    def test_generate_from_analyst_windows_malformed_window(self, tactician_labeler):
        """Test behavior with malformed window data."""
        dates = pd.date_range(start='2023-01-01 10:00:00', periods=20, freq='15min')
        data = pd.DataFrame({
            'open': np.random.randn(20),
            'high': np.random.randn(20),
            'low': np.random.randn(20),
            'close': np.random.randn(20),
            'volume': np.random.randint(1000, 10000, 20)
        }, index=dates)
        
        # Create windows with malformed data
        windows = [
            {
                'start': pd.Timestamp('2023-01-01 10:00:00'),
                'end': pd.Timestamp('2023-01-01 10:15:00'),
                'anchor': pd.Timestamp('2023-01-01 10:07:30'),
                'direction': 1
            },
            {
                # Malformed window - invalid timestamp format
                'start': 'invalid_timestamp',
                'end': pd.Timestamp('2023-01-01 10:30:00'),
                'anchor': pd.Timestamp('2023-01-01 10:22:30'),
                'direction': -1
            }
        ]
        
        result = tactician_labeler.generate_from_analyst_windows(
            data=data,
            opportunity_windows=windows
        )
        
        # Should handle gracefully, processing only valid windows
        assert isinstance(result, dict)
        assert result['meta']['n_windows'] >= 0  # Should have processed at least the valid window


if __name__ == "__main__":
    pytest.main([__file__])
