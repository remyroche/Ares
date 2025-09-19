#!/usr/bin/env python3
"""
Comprehensive error path testing for regime data splitting module.

This test suite covers error scenarios, edge cases, and failure modes
to ensure robust error handling and graceful degradation.
"""

import sys
import os
import asyncio
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../'))

from src.training.steps.market_analysis.regime_data_splitting.component import (
    RegimeDataSplittingComponent, RegimeSplittingStatus, RegimeSplittingMetrics
)
from src.training.steps.market_analysis.regime_data_splitting.enhanced import (
    RegimeDataSplittingEnhanced, HMMRegimeTagger
)
from src.training.steps.market_analysis.regime_data_splitting.validator import (
    Step4RegimeDataSplittingValidator
)
from src.training.steps.market_analysis.regime_data_splitting.error_handling_utils import (
    ErrorCategory, ErrorSeverity
)
from src.training.steps.market_analysis.regime_data_splitting.validation_config import (
    ValidationConfiguration, get_unified_validator
)
from src.training.steps.market_analysis.regime_data_splitting.resource_management import (
    get_resource_manager, reset_resource_manager
)
from src.training.steps.market_analysis.components.base_component import ComponentConfig


class TestDataCreator:
    """Helper class to create test data for various scenarios."""
    
    @staticmethod
    def create_valid_market_data(n_rows: int = 1000) -> pd.DataFrame:
        """Create valid market data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=n_rows, freq='1H')
        np.random.seed(42)
        
        # Generate realistic price data
        base_price = 100.0
        price_changes = np.random.normal(0, 0.01, n_rows).cumsum()
        close_prices = base_price * (1 + price_changes)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices * (1 + np.random.uniform(-0.001, 0.001, n_rows)),
            'high': close_prices * (1 + np.random.uniform(0, 0.01, n_rows)),
            'low': close_prices * (1 - np.random.uniform(0, 0.01, n_rows)),
            'close': close_prices,
            'volume': np.random.uniform(1000, 10000, n_rows)
        })
        
        return data
    
    @staticmethod
    def create_invalid_market_data_scenarios():
        """Create various invalid market data scenarios."""
        scenarios = {}
        
        # Empty DataFrame
        scenarios['empty'] = pd.DataFrame()
        
        # DataFrame with missing columns
        scenarios['missing_columns'] = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1H'),
            'close': np.random.uniform(90, 110, 100)
            # Missing OHLCV columns
        })
        
        # DataFrame with null values
        valid_data = TestDataCreator.create_valid_market_data(100)
        scenarios['with_nulls'] = valid_data.copy()
        scenarios['with_nulls'].loc[10:20, 'close'] = np.nan
        scenarios['with_nulls'].loc[30:40, 'volume'] = np.nan
        
        # DataFrame with infinite values
        scenarios['with_infinites'] = valid_data.copy()
        scenarios['with_infinites'].loc[10, 'high'] = np.inf
        scenarios['with_infinites'].loc[20, 'low'] = -np.inf
        
        # DataFrame with negative prices
        scenarios['negative_prices'] = valid_data.copy()
        scenarios['negative_prices'].loc[10:15, 'close'] = -50.0
        
        # DataFrame with duplicate timestamps
        scenarios['duplicate_timestamps'] = valid_data.copy()
        scenarios['duplicate_timestamps'].loc[50:55, 'timestamp'] = scenarios['duplicate_timestamps'].loc[45, 'timestamp']
        
        # DataFrame with non-monotonic timestamps
        scenarios['non_monotonic'] = valid_data.copy()
        scenarios['non_monotonic'] = scenarios['non_monotonic'].sample(frac=1).reset_index(drop=True)
        
        return scenarios
    
    @staticmethod
    def create_regime_discovery_scenarios():
        """Create various regime discovery result scenarios."""
        scenarios = {}
        
        # Valid regime discovery
        scenarios['valid'] = {
            'regime_states': np.random.randint(0, 3, 1000),
            'regime_probabilities': np.random.dirichlet([1, 1, 1], 1000),
            'regime_means': np.array([[0.1, 0.5], [-0.05, 0.8], [0.01, 0.3]]),
            'regime_covariances': [np.eye(2) * 0.1, np.eye(2) * 0.2, np.eye(2) * 0.05]
        }
        
        # Empty regime discovery
        scenarios['empty'] = {}
        
        # Mismatched lengths
        scenarios['length_mismatch'] = {
            'regime_states': np.random.randint(0, 3, 500),  # Different length
            'regime_probabilities': np.random.dirichlet([1, 1, 1], 1000)
        }
        
        # Single regime (invalid)
        scenarios['single_regime'] = {
            'regime_states': np.zeros(1000, dtype=int),
            'regime_probabilities': np.ones((1000, 1))
        }
        
        # Too many regimes
        scenarios['too_many_regimes'] = {
            'regime_states': np.random.randint(0, 50, 1000),
            'regime_probabilities': np.random.dirichlet([1] * 50, 1000)
        }
        
        # Invalid regime values
        scenarios['invalid_values'] = {
            'regime_states': np.random.uniform(-5, 5, 1000),  # Should be integers
            'regime_probabilities': np.random.uniform(-0.1, 1.1, (1000, 3))  # Invalid probabilities
        }
        
        return scenarios


class TestRegimeDataSplittingComponent:
    """Test suite for RegimeDataSplittingComponent error paths."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ComponentConfig(
            symbol="TESTUSDT",
            exchange="test_exchange",
            timeframe="1h"
        )
        reset_resource_manager()
    
    def teardown_method(self):
        """Clean up after tests."""
        reset_resource_manager()
    
    def test_initialization_with_missing_dependencies(self):
        """Test component initialization with missing dependencies."""
        # Mock missing numpy
        with patch.dict('sys.modules', {'numpy': None}):
            with pytest.raises(ImportError):
                RegimeDataSplittingComponent(self.config)
    
    def test_initialization_with_invalid_config(self):
        """Test component initialization with invalid configuration."""
        # Test with None config
        component = RegimeDataSplittingComponent(None)
        assert component.config is not None
        
        # Test with config missing required fields
        invalid_config = ComponentConfig(symbol=None, exchange=None, timeframe=None)
        component = RegimeDataSplittingComponent(invalid_config)
        assert component.config is not None
    
    @pytest.mark.asyncio
    async def test_execute_with_none_data(self):
        """Test execute method with None data."""
        component = RegimeDataSplittingComponent(self.config)
        pipeline_state = {}
        
        result = await component.execute(None, pipeline_state)
        
        assert not result.success
        assert "Input data is None" in result.error_message
        assert result.artifacts['regime_splitting_report']['status'] == RegimeSplittingStatus.FAILED.value
    
    @pytest.mark.asyncio
    async def test_execute_with_invalid_pipeline_state(self):
        """Test execute method with invalid pipeline state."""
        component = RegimeDataSplittingComponent(self.config)
        market_data = TestDataCreator.create_valid_market_data(100)
        
        # Test with None pipeline state
        result = await component.execute(market_data, None)
        assert not result.success
        assert "Pipeline state must be a dictionary" in result.error_message
        
        # Test with non-dict pipeline state
        result = await component.execute(market_data, "invalid")
        assert not result.success
        assert "Pipeline state must be a dictionary" in result.error_message
    
    @pytest.mark.asyncio
    async def test_execute_with_missing_config_fields(self):
        """Test execute method with missing configuration fields."""
        invalid_config = ComponentConfig(symbol=None, exchange=None, timeframe="1h")
        component = RegimeDataSplittingComponent(invalid_config)
        market_data = TestDataCreator.create_valid_market_data(100)
        pipeline_state = {}
        
        result = await component.execute(market_data, pipeline_state)
        
        assert not result.success
        assert "Symbol and exchange must be configured" in result.error_message
    
    @pytest.mark.asyncio
    async def test_load_and_prepare_data_error_scenarios(self):
        """Test data loading with various error scenarios."""
        component = RegimeDataSplittingComponent(self.config)
        
        invalid_scenarios = TestDataCreator.create_invalid_market_data_scenarios()
        
        for scenario_name, data in invalid_scenarios.items():
            result = component._load_and_prepare_data(data)
            
            if scenario_name == 'empty':
                assert result is None
            else:
                # Should handle gracefully with warnings
                if result is not None:
                    assert isinstance(result, pd.DataFrame)
    
    def test_regime_discovery_extraction_errors(self):
        """Test regime discovery result extraction with error scenarios."""
        component = RegimeDataSplittingComponent(self.config)
        
        # Empty pipeline state
        result = component._get_regime_discovery_results({})
        assert result is None
        
        # Pipeline state with empty regime discovery
        result = component._get_regime_discovery_results({'hmm_regime_discovery_result': {}})
        assert result is None
        
        # Pipeline state with None regime discovery
        result = component._get_regime_discovery_results({'hmm_regime_discovery_result': None})
        assert result is None
    
    @pytest.mark.asyncio
    async def test_data_alignment_error_scenarios(self):
        """Test data alignment with various error scenarios."""
        component = RegimeDataSplittingComponent(self.config)
        
        market_data = TestDataCreator.create_valid_market_data(1000)
        regime_scenarios = TestDataCreator.create_regime_discovery_scenarios()
        
        from src.training.steps.market_analysis.regime_data_splitting.component import RegimeSplittingReport, RegimeSplittingStatus
        report = RegimeSplittingReport(status=RegimeSplittingStatus.IN_PROGRESS)
        
        # Test with length mismatch
        result = await component._perform_regime_splitting(
            market_data, regime_scenarios['length_mismatch'], report
        )
        
        # Should handle gracefully with warnings about data loss
        assert len(report.warnings) > 0 or not result['success']
    
    def test_validation_error_scenarios(self):
        """Test validation with various error scenarios."""
        component = RegimeDataSplittingComponent(self.config)
        
        # Test with invalid splitting results
        invalid_results = [
            {'success': False, 'data': None, 'errors': ['Test error']},
            {'success': True, 'data': None},
            {'success': True, 'data': {'market_data': None, 'regime_states': None}},
        ]
        
        from src.training.steps.market_analysis.regime_data_splitting.component import RegimeSplittingReport, RegimeSplittingStatus
        report = RegimeSplittingReport(status=RegimeSplittingStatus.IN_PROGRESS)
        
        for invalid_result in invalid_results:
            validation_result = component._validate_splitting_results(invalid_result, report)
            assert not validation_result['valid']
            assert len(validation_result['errors']) > 0
    
    def test_resource_management_error_scenarios(self):
        """Test resource management error scenarios."""
        component = RegimeDataSplittingComponent(self.config)
        
        # Test cleanup with resource manager errors
        with patch.object(component.resource_manager, 'cleanup', side_effect=Exception("Cleanup error")):
            cleanup_result = component.cleanup()
            assert cleanup_result['status'] == 'failed'
            assert 'error' in cleanup_result
        
        # Test resource metrics with errors
        with patch.object(component.resource_manager, 'get_resource_metrics', side_effect=Exception("Metrics error")):
            metrics_result = component.get_resource_metrics()
            assert 'error' in metrics_result


class TestRegimeDataSplittingEnhanced:
    """Test suite for RegimeDataSplittingEnhanced error paths."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'symbol': 'TESTUSDT',
            'exchange': 'test_exchange',
            'timeframe': '1h',
            'data_dir': 'test_data'
        }
        reset_resource_manager()
    
    def teardown_method(self):
        """Clean up after tests."""
        reset_resource_manager()
    
    @pytest.mark.asyncio
    async def test_execute_with_invalid_training_input(self):
        """Test execute method with invalid training input."""
        splitter = RegimeDataSplittingEnhanced(self.config)
        
        # Test with None training input
        result = await splitter.execute(None, {})
        assert not result['step04_regime_data_splitting_completed']
        assert 'error_message' in result
        
        # Test with invalid training input structure
        result = await splitter.execute("invalid", {})
        assert not result['step04_regime_data_splitting_completed']
        assert 'error_message' in result
    
    @pytest.mark.asyncio
    async def test_execute_with_missing_data_files(self):
        """Test execute method when data files are missing."""
        splitter = RegimeDataSplittingEnhanced(self.config)
        
        training_input = {
            'symbol': 'TESTUSDT',
            'exchange': 'test_exchange',
            'timeframe': '1h',
            'data_dir': '/nonexistent/path'
        }
        
        result = await splitter.execute(training_input, {})
        assert not result['step04_regime_data_splitting_completed']
        assert 'error_message' in result
    
    def test_hmm_regime_tagger_error_scenarios(self):
        """Test HMMRegimeTagger with various error scenarios."""
        tagger = HMMRegimeTagger(self.config)
        
        # Test with None market data
        with pytest.raises(ValueError, match="market_data is None"):
            tagger.tag_regimes_with_models(None)
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="market_data is empty"):
            tagger.tag_regimes_with_models(empty_df)
        
        # Test with non-DataFrame input
        with pytest.raises(ValueError, match="market_data must be a DataFrame"):
            tagger.tag_regimes_with_models("invalid")
        
        # Test with invalid use_ensemble parameter
        valid_df = TestDataCreator.create_valid_market_data(100)
        with pytest.raises(ValueError, match="use_ensemble must be a boolean"):
            tagger.tag_regimes_with_models(valid_df, use_ensemble="invalid")
    
    def test_feature_creation_error_scenarios(self):
        """Test feature creation with error scenarios."""
        tagger = HMMRegimeTagger(self.config)
        
        # Test with feature generator not initialized
        tagger.feature_generator = None
        
        valid_df = TestDataCreator.create_valid_market_data(100)
        with pytest.raises(ValueError, match="Feature generator not initialized"):
            tagger.create_features_for_tagging(valid_df)
    
    @pytest.mark.asyncio
    async def test_data_alignment_critical_loss(self):
        """Test data alignment with critical data loss."""
        splitter = RegimeDataSplittingEnhanced(self.config)
        
        # Mock scenario with critical data loss (>20%)
        with patch.object(splitter, '_load_and_validate_market_data') as mock_load:
            mock_load.return_value = TestDataCreator.create_valid_market_data(1000)
            
            # Create scenario with significant length mismatch
            training_input = {
                'symbol': 'TESTUSDT',
                'exchange': 'test_exchange',
                'timeframe': '1h',
                'data_dir': 'test_data'
            }
            
            pipeline_state = {
                'hmm_regime_discovery_result': {
                    'regime_states': np.random.randint(0, 3, 100),  # Much shorter
                    'regime_probabilities': np.random.dirichlet([1, 1, 1], 100)
                }
            }
            
            result = await splitter.execute(training_input, pipeline_state)
            assert not result['step04_regime_data_splitting_completed']
            assert 'error_message' in result


class TestValidationErrorPaths:
    """Test suite for validation error paths."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {'validation_config': {}}
        self.validator = Step4RegimeDataSplittingValidator(self.config)
    
    @pytest.mark.asyncio
    async def test_validate_with_missing_files(self):
        """Test validation with missing files."""
        result = await self.validator.validate_step4_regime_data_splitting(
            'TESTUSDT', 'test_exchange', '/nonexistent/path', {}
        )
        assert not result
    
    def test_validate_regime_file_errors(self):
        """Test regime file validation errors."""
        # Test with non-existent file
        non_existent_file = Path('/nonexistent/file.parquet')
        result = self.validator._validate_regime_file(non_existent_file)
        assert not result
        
        # Test with corrupted file (mock)
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(b'corrupted data')
        
        try:
            result = self.validator._validate_regime_file(temp_path)
            assert not result
        finally:
            temp_path.unlink()
    
    def test_validate_statistics_file_errors(self):
        """Test statistics file validation errors."""
        # Test with invalid JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write('invalid json content')
        
        try:
            result = self.validator._validate_statistics_file(temp_path)
            assert not result
        finally:
            temp_path.unlink()


class TestUnifiedValidationErrors:
    """Test suite for unified validation error scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validation_config = ValidationConfiguration()
        self.unified_validator = get_unified_validator(self.validation_config)
    
    def test_data_quality_validation_failures(self):
        """Test data quality validation failures."""
        # Test with metrics that should fail validation
        bad_metrics = {
            'data_quality_score': 0.5,  # Below threshold
            'missing_data_percentage': 15.0,  # Above threshold
            'data_alignment_loss': 25.0  # Critical level
        }
        
        result = self.unified_validator.validate_data_quality(bad_metrics)
        assert not result['passed']
        assert len(result['errors']) > 0 or len(result['warnings']) > 0
    
    def test_regime_quality_validation_failures(self):
        """Test regime quality validation failures."""
        # Test with metrics that should fail validation
        bad_metrics = {
            'regime_count': 1,  # Too few regimes
            'regime_continuity_score': 0.3,  # Below threshold
            'regime_confidence_score': 0.5  # Below threshold
        }
        
        result = self.unified_validator.validate_regime_quality(bad_metrics)
        assert not result['passed']
        assert len(result['errors']) > 0
    
    def test_performance_validation_failures(self):
        """Test performance validation failures."""
        # Test with metrics that should fail validation
        bad_metrics = {
            'execution_time': 400.0,  # Above threshold
            'memory_usage': 2500.0  # Above threshold
        }
        
        result = self.unified_validator.validate_performance(bad_metrics)
        assert not result['passed']
        assert len(result['warnings']) > 0


class TestErrorHandlingIntegration:
    """Integration tests for error handling across components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        reset_resource_manager()
    
    def teardown_method(self):
        """Clean up after tests."""
        reset_resource_manager()
    
    @pytest.mark.asyncio
    async def test_end_to_end_error_propagation(self):
        """Test error propagation through the entire pipeline."""
        config = ComponentConfig(
            symbol="TESTUSDT",
            exchange="test_exchange",
            timeframe="1h"
        )
        
        component = RegimeDataSplittingComponent(config)
        
        # Create scenario that should fail at multiple points
        invalid_data = None
        invalid_pipeline_state = None
        
        result = await component.execute(invalid_data, invalid_pipeline_state)
        
        # Should fail gracefully with proper error reporting
        assert not result.success
        assert result.error_message is not None
        assert 'regime_splitting_report' in result.artifacts
        assert result.artifacts['regime_splitting_report']['status'] == RegimeSplittingStatus.FAILED.value
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_on_errors(self):
        """Test that resources are properly cleaned up even when errors occur."""
        config = ComponentConfig(
            symbol="TESTUSDT",
            exchange="test_exchange",
            timeframe="1h"
        )
        
        component = RegimeDataSplittingComponent(config)
        
        # Force an error during processing
        with patch.object(component, '_perform_regime_splitting', side_effect=Exception("Processing error")):
            market_data = TestDataCreator.create_valid_market_data(100)
            pipeline_state = {'hmm_regime_discovery_result': TestDataCreator.create_regime_discovery_scenarios()['valid']}
            
            result = await component.execute(market_data, pipeline_state)
            
            # Should fail but still perform cleanup
            assert not result.success
            
            # Verify cleanup was attempted
            cleanup_result = component.cleanup()
            assert 'cleanup_functions_executed' in cleanup_result or 'status' in cleanup_result


def run_comprehensive_error_tests():
    """Run all comprehensive error path tests."""
    print("🧪 Running Comprehensive Error Path Tests")
    print("=" * 60)
    
    # Run pytest with verbose output
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "--capture=no"
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n✅ All error path tests passed!")
        print("🎯 The regime data splitting module handles errors gracefully")
    else:
        print(f"\n❌ Some tests failed (exit code: {exit_code})")
        print("🔧 Review the test output above for details")
    
    return exit_code == 0


if __name__ == "__main__":
    success = run_comprehensive_error_tests()
    sys.exit(0 if success else 1)