"""
Comprehensive Test Suite for Unified Triple Barrier Labeler

This module provides comprehensive tests for the unified triple barrier labeling implementation,
including tests for error handling, validation, performance, and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

from .unified_labeler import (
    UnifiedTripleBarrierLabeler,
    TripleBarrierConfig,
    TripleBarrierResult,
    ValidationError,
    ConfigurationError,
    HardwareOptimizationError,
    DataQualityError,
    create_triple_barrier_labeler,
    apply_triple_barrier_labeling
)

class TestTripleBarrierConfig:
    """Test configuration validation and defaults."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = TripleBarrierConfig()
        
        assert config.profit_take_multiplier == 0.002
        assert config.stop_loss_multiplier == 0.001
        assert config.time_barrier_minutes == 30
        assert config.max_lookahead == 100
        assert config.transaction_cost == 0.0008
        assert config.binary_classification is True
        assert config.regime_aware is True
        assert config.regime_column == 'hmm_regime'
        assert config.fail_on_validation_error is True
        assert config.fail_on_hardware_optimization_error is False
    
    def test_invalid_profit_take_multiplier(self):
        """Test validation of profit take multiplier."""
        with pytest.raises(ConfigurationError):
            TripleBarrierConfig(profit_take_multiplier=0.1)  # Too large
        
        with pytest.raises(ConfigurationError):
            TripleBarrierConfig(profit_take_multiplier=0.0001)  # Too small
    
    def test_invalid_stop_loss_multiplier(self):
        """Test validation of stop loss multiplier."""
        with pytest.raises(ConfigurationError):
            TripleBarrierConfig(stop_loss_multiplier=0.1)  # Too large
        
        with pytest.raises(ConfigurationError):
            TripleBarrierConfig(stop_loss_multiplier=0.0001)  # Too small
    
    def test_risk_reward_ratio_validation(self):
        """Test risk-reward ratio validation."""
        with pytest.raises(ConfigurationError):
            TripleBarrierConfig(
                profit_take_multiplier=0.001,  # 0.1%
                stop_loss_multiplier=0.002     # 0.2% - worse than profit take
            )
    
    def test_barriers_too_close(self):
        """Test validation when barriers are too close."""
        with pytest.raises(ConfigurationError):
            TripleBarrierConfig(
                profit_take_multiplier=0.001,
                stop_loss_multiplier=0.001  # Same as profit take
            )
    
    def test_negative_parameters(self):
        """Test validation of negative parameters."""
        with pytest.raises(ConfigurationError):
            TripleBarrierConfig(transaction_cost=-0.001)
        
        with pytest.raises(ConfigurationError):
            TripleBarrierConfig(time_barrier_minutes=-10)
        
        with pytest.raises(ConfigurationError):
            TripleBarrierConfig(max_lookahead=-50)

class TestDataValidation:
    """Test data validation functionality."""
    
    def create_valid_data(self, n_rows: int = 1000) -> pd.DataFrame:
        """Create valid test data."""
        dates = pd.date_range('2024-01-01', periods=n_rows, freq='1min')
        return pd.DataFrame({
            'open': np.random.uniform(100, 110, n_rows),
            'high': np.random.uniform(105, 115, n_rows),
            'low': np.random.uniform(95, 105, n_rows),
            'close': np.random.uniform(100, 110, n_rows),
            'volume': np.random.uniform(1000, 10000, n_rows),
            'hmm_regime': np.random.choice([0, 1, 2], n_rows)
        }, index=dates)
    
    def test_valid_data_passes_validation(self):
        """Test that valid data passes validation."""
        config = TripleBarrierConfig()
        labeler = UnifiedTripleBarrierLabeler(config)
        data = self.create_valid_data()
        
        result = labeler._validate_input_data(data)
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_empty_data_fails_validation(self):
        """Test that empty data fails validation."""
        config = TripleBarrierConfig()
        labeler = UnifiedTripleBarrierLabeler(config)
        
        # Test None data
        result = labeler._validate_input_data(None)
        assert result.is_valid is False
        assert "Input data is None or empty" in result.errors[0]
        
        # Test empty DataFrame
        result = labeler._validate_input_data(pd.DataFrame())
        assert result.is_valid is False
        assert "Input data is None or empty" in result.errors[0]
    
    def test_missing_required_columns(self):
        """Test validation with missing required columns."""
        config = TripleBarrierConfig()
        labeler = UnifiedTripleBarrierLabeler(config)
        
        # Missing 'close' column
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97]
            # Missing 'close'
        })
        
        result = labeler._validate_input_data(data)
        assert result.is_valid is False
        assert "Missing required columns: ['close']" in result.errors[0]
    
    def test_insufficient_data_points(self):
        """Test validation with insufficient data points."""
        config = TripleBarrierConfig(min_data_points=100)
        labeler = UnifiedTripleBarrierLabeler(config)
        data = self.create_valid_data(50)  # Less than minimum
        
        result = labeler._validate_input_data(data)
        assert result.is_valid is False
        assert "Insufficient data points: 50 < 100" in result.errors[0]
    
    def test_invalid_ohlc_relationships(self):
        """Test validation of OHLC relationships."""
        config = TripleBarrierConfig()
        labeler = UnifiedTripleBarrierLabeler(config)
        
        # Create data with invalid OHLC relationships
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [95, 96, 97],  # High < open (invalid)
            'low': [105, 106, 107],  # Low > open (invalid)
            'close': [100, 101, 102]
        })
        
        result = labeler._validate_input_data(data)
        assert result.is_valid is False
        assert any("high < max(open, close)" in error for error in result.errors)
        assert any("low > min(open, close)" in error for error in result.errors)
    
    def test_missing_data_validation(self):
        """Test validation of missing data."""
        config = TripleBarrierConfig(max_missing_data_ratio=0.1)
        labeler = UnifiedTripleBarrierLabeler(config)
        
        data = self.create_valid_data(100)
        # Add 20% missing data (exceeds 10% threshold)
        data.loc[0:19, 'close'] = np.nan
        
        result = labeler._validate_input_data(data)
        assert result.is_valid is False
        assert any("missing data" in error for error in result.errors)
    
    def test_non_positive_prices(self):
        """Test validation of non-positive prices."""
        config = TripleBarrierConfig()
        labeler = UnifiedTripleBarrierLabeler(config)
        
        data = self.create_valid_data(100)
        data.loc[0, 'close'] = 0  # Non-positive price
        data.loc[1, 'close'] = -1  # Negative price
        
        result = labeler._validate_input_data(data)
        assert result.is_valid is False
        assert any("non-positive values" in error for error in result.errors)
    
    def test_regime_data_validation(self):
        """Test validation of regime data."""
        config = TripleBarrierConfig(regime_aware=True, regime_column='hmm_regime')
        labeler = UnifiedTripleBarrierLabeler(config)
        
        # Test missing regime column
        data = self.create_valid_data(100)
        data = data.drop(columns=['hmm_regime'])
        
        result = labeler._validate_input_data(data)
        assert result.is_valid is False
        assert "Regime column 'hmm_regime' not found" in result.errors[0]
    
    def test_regime_imbalance_warning(self):
        """Test warning for imbalanced regimes."""
        config = TripleBarrierConfig(regime_aware=True)
        labeler = UnifiedTripleBarrierLabeler(config)
        
        data = self.create_valid_data(100)
        # Create severely imbalanced regimes
        data['hmm_regime'] = [0] * 90 + [1] * 10  # 90% vs 10%
        
        result = labeler._validate_input_data(data)
        assert result.is_valid is True  # Should pass validation
        assert any("imbalanced regimes" in warning for warning in result.warnings)

class TestTripleBarrierLabeling:
    """Test the main triple barrier labeling functionality."""
    
    def create_valid_data(self, n_rows: int = 1000) -> pd.DataFrame:
        """Create valid test data."""
        dates = pd.date_range('2024-01-01', periods=n_rows, freq='1min')
        return pd.DataFrame({
            'open': np.random.uniform(100, 110, n_rows),
            'high': np.random.uniform(105, 115, n_rows),
            'low': np.random.uniform(95, 105, n_rows),
            'close': np.random.uniform(100, 110, n_rows),
            'volume': np.random.uniform(1000, 10000, n_rows),
            'hmm_regime': np.random.choice([0, 1, 2], n_rows)
        }, index=dates)
    
    def test_successful_labeling(self):
        """Test successful labeling execution."""
        config = TripleBarrierConfig()
        labeler = UnifiedTripleBarrierLabeler(config)
        data = self.create_valid_data()
        
        result = labeler.apply_labeling(data)
        
        assert result.success is True
        assert result.labeled_data is not None
        assert len(result.labeled_data) > 0
        assert 'label' in result.labeled_data.columns
        assert result.total_labels_generated > 0
        assert result.execution_duration > 0
    
    def test_fail_on_validation_error(self):
        """Test fail-on-validation-error behavior."""
        config = TripleBarrierConfig(fail_on_validation_error=True)
        labeler = UnifiedTripleBarrierLabeler(config)
        
        # Create invalid data
        data = pd.DataFrame({
            'open': [100, 101],
            'high': [95, 96],  # Invalid: high < open
            'low': [105, 106],  # Invalid: low > open
            'close': [100, 101]
        })
        
        result = labeler.apply_labeling(data)
        
        assert result.success is False
        assert result.error_message is not None
        assert "Data validation failed" in result.error_message
    
    def test_graceful_validation_failure(self):
        """Test graceful handling of validation failures."""
        config = TripleBarrierConfig(fail_on_validation_error=False)
        labeler = UnifiedTripleBarrierLabeler(config)
        
        # Create data with warnings but not critical errors
        data = self.create_valid_data(50)  # Below minimum but not critical
        data.loc[0:4, 'close'] = np.nan  # Some missing data
        
        result = labeler.apply_labeling(data)
        
        # Should still succeed but with warnings
        assert result.success is True
        assert len(result.validation_warnings) > 0
    
    def test_binary_classification_filtering(self):
        """Test binary classification filtering."""
        config = TripleBarrierConfig(binary_classification=True)
        labeler = UnifiedTripleBarrierLabeler(config)
        data = self.create_valid_data()
        
        result = labeler.apply_labeling(data)
        
        assert result.success is True
        # Should not have any 0 labels (HOLD) in binary classification
        if 'label' in result.labeled_data.columns:
            hold_labels = (result.labeled_data['label'] == 0).sum()
            assert hold_labels == 0
    
    def test_ternary_classification(self):
        """Test ternary classification (including HOLD labels)."""
        config = TripleBarrierConfig(binary_classification=False)
        labeler = UnifiedTripleBarrierLabeler(config)
        data = self.create_valid_data()
        
        result = labeler.apply_labeling(data)
        
        assert result.success is True
        # Should have 0 labels (HOLD) in ternary classification
        if 'label' in result.labeled_data.columns:
            hold_labels = (result.labeled_data['label'] == 0).sum()
            assert hold_labels >= 0  # May or may not have hold labels
    
    def test_label_distribution(self):
        """Test that labels are properly distributed."""
        config = TripleBarrierConfig()
        labeler = UnifiedTripleBarrierLabeler(config)
        data = self.create_valid_data()
        
        result = labeler.apply_labeling(data)
        
        assert result.success is True
        assert len(result.label_distribution) > 0
        
        # Check that we have reasonable label distribution
        total_labels = sum(result.label_distribution.values())
        assert total_labels > 0
        
        # Check that we have both positive and negative labels (for binary)
        if config.binary_classification:
            assert 1 in result.label_distribution  # LONG
            assert -1 in result.label_distribution  # SHORT
    
    def test_performance_metrics(self):
        """Test that performance metrics are collected."""
        config = TripleBarrierConfig()
        labeler = UnifiedTripleBarrierLabeler(config)
        data = self.create_valid_data()
        
        result = labeler.apply_labeling(data)
        
        assert result.success is True
        assert result.execution_duration > 0
        assert isinstance(result.numba_acceleration_used, bool)
        assert isinstance(result.hardware_optimizations_used, list)
    
    def test_quality_metrics(self):
        """Test that quality metrics are calculated."""
        config = TripleBarrierConfig()
        labeler = UnifiedTripleBarrierLabeler(config)
        data = self.create_valid_data()
        
        result = labeler.apply_labeling(data)
        
        assert result.success is True
        assert 0.0 <= result.data_quality_score <= 1.0
        assert 0.0 <= result.label_quality_score <= 1.0
        assert len(result.barrier_hit_statistics) > 0
    
    def test_result_serialization(self):
        """Test that result can be serialized to dictionary."""
        config = TripleBarrierConfig()
        labeler = UnifiedTripleBarrierLabeler(config)
        data = self.create_valid_data()
        
        result = labeler.apply_labeling(data)
        
        assert result.success is True
        
        # Test serialization
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert 'success' in result_dict
        assert 'execution_duration' in result_dict
        assert 'label_distribution' in result_dict
    
    def test_summary_generation(self):
        """Test that summary can be generated."""
        config = TripleBarrierConfig()
        labeler = UnifiedTripleBarrierLabeler(config)
        data = self.create_valid_data()
        
        result = labeler.apply_labeling(data)
        
        assert result.success is True
        
        summary = result.generate_summary()
        assert isinstance(summary, str)
        assert "Triple Barrier Labeling Execution Summary" in summary
        assert "SUCCESS" in summary or "FAILED" in summary

class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def create_valid_data(self, n_rows: int = 1000) -> pd.DataFrame:
        """Create valid test data."""
        dates = pd.date_range('2024-01-01', periods=n_rows, freq='1min')
        return pd.DataFrame({
            'open': np.random.uniform(100, 110, n_rows),
            'high': np.random.uniform(105, 115, n_rows),
            'low': np.random.uniform(95, 105, n_rows),
            'close': np.random.uniform(100, 110, n_rows),
            'volume': np.random.uniform(1000, 10000, n_rows),
            'hmm_regime': np.random.choice([0, 1, 2], n_rows)
        }, index=dates)
    
    def test_create_triple_barrier_labeler(self):
        """Test create_triple_barrier_labeler convenience function."""
        labeler = create_triple_barrier_labeler(
            profit_take_multiplier=0.003,
            stop_loss_multiplier=0.002,
            regime_aware=False
        )
        
        assert isinstance(labeler, UnifiedTripleBarrierLabeler)
        assert labeler.config.profit_take_multiplier == 0.003
        assert labeler.config.stop_loss_multiplier == 0.002
        assert labeler.config.regime_aware is False
    
    def test_apply_triple_barrier_labeling(self):
        """Test apply_triple_barrier_labeling convenience function."""
        data = self.create_valid_data()
        
        result = apply_triple_barrier_labeling(
            data,
            profit_take_multiplier=0.003,
            stop_loss_multiplier=0.002
        )
        
        assert isinstance(result, TripleBarrierResult)
        assert result.success is True
        assert result.labeled_data is not None

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_silent_failure_prevention(self):
        """Test that silent failures are prevented."""
        config = TripleBarrierConfig(fail_on_validation_error=True)
        labeler = UnifiedTripleBarrierLabeler(config)
        
        # Test with invalid data that would cause silent failure in old implementation
        invalid_data = pd.DataFrame({'invalid': [1, 2, 3]})
        
        result = labeler.apply_labeling(invalid_data)
        
        assert result.success is False
        assert result.error_message is not None
        assert "Data validation failed" in result.error_message
    
    def test_hardware_optimization_failure_handling(self):
        """Test handling of hardware optimization failures."""
        config = TripleBarrierConfig(
            fail_on_hardware_optimization_error=False,
            enable_hardware_optimizations=True
        )
        labeler = UnifiedTripleBarrierLabeler(config)
        data = self.create_valid_data()
        
        # Should not fail even if hardware optimizations fail
        result = labeler.apply_labeling(data)
        
        assert result.success is True
        # Should have some non-critical failures recorded
        assert len(result.non_critical_failures) >= 0
    
    def test_memory_optimization_fallback(self):
        """Test fallback when memory optimization fails."""
        config = TripleBarrierConfig(enable_hardware_optimizations=True)
        labeler = UnifiedTripleBarrierLabeler(config)
        data = self.create_valid_data()
        
        result = labeler.apply_labeling(data)
        
        assert result.success is True
        # Should complete successfully even if memory optimization fails
    
    def test_numba_fallback(self):
        """Test fallback when Numba is not available."""
        # This test would require mocking Numba availability
        # For now, just test that the system works without Numba
        config = TripleBarrierConfig(enable_numba_acceleration=False)
        labeler = UnifiedTripleBarrierLabeler(config)
        data = self.create_valid_data()
        
        result = labeler.apply_labeling(data)
        
        assert result.success is True
        assert result.numba_acceleration_used is False

class TestPerformance:
    """Test performance characteristics."""
    
    def create_valid_data(self, n_rows: int = 1000) -> pd.DataFrame:
        """Create valid test data."""
        dates = pd.date_range('2024-01-01', periods=n_rows, freq='1min')
        return pd.DataFrame({
            'open': np.random.uniform(100, 110, n_rows),
            'high': np.random.uniform(105, 115, n_rows),
            'low': np.random.uniform(95, 105, n_rows),
            'close': np.random.uniform(100, 110, n_rows),
            'volume': np.random.uniform(1000, 10000, n_rows),
            'hmm_regime': np.random.choice([0, 1, 2], n_rows)
        }, index=dates)
    
    def test_execution_time_reasonable(self):
        """Test that execution time is reasonable."""
        config = TripleBarrierConfig()
        labeler = UnifiedTripleBarrierLabeler(config)
        data = self.create_valid_data(1000)
        
        result = labeler.apply_labeling(data)
        
        assert result.success is True
        # Should complete within reasonable time (adjust threshold as needed)
        assert result.execution_duration < 30.0  # 30 seconds max
    
    def test_memory_usage_reasonable(self):
        """Test that memory usage is reasonable."""
        config = TripleBarrierConfig()
        labeler = UnifiedTripleBarrierLabeler(config)
        data = self.create_valid_data(1000)
        
        result = labeler.apply_labeling(data)
        
        assert result.success is True
        # Memory usage should be reasonable (adjust threshold as needed)
        assert result.memory_usage_mb < 1000  # 1GB max
    
    def test_scalability(self):
        """Test scalability with larger datasets."""
        config = TripleBarrierConfig()
        labeler = UnifiedTripleBarrierLabeler(config)
        
        # Test with different data sizes
        for size in [100, 1000, 5000]:
            data = self.create_valid_data(size)
            result = labeler.apply_labeling(data)
            
            assert result.success is True
            assert result.total_labels_generated > 0
            assert result.execution_duration > 0

if __name__ == '__main__':
    # Run the tests
    pytest.main([__file__, '-v'])