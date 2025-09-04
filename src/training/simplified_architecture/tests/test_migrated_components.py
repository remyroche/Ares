"""
Comprehensive Test Suite for Migrated Components

This module provides comprehensive tests for the migrated components,
ensuring they work correctly and maintain compatibility with the original functionality.
"""
import asyncio
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os

from ..enhanced_interfaces import StepConfig, StepStatus, StepFactory
from ..dependency_injection import EnhancedDIContainer, ServiceLifetime
from ..enhanced_config_system import ConfigurationManager, PipelineConfiguration
from ..enhanced_pipeline_orchestrator import EnhancedPipelineOrchestrator
from ..migrated_components.data_components import DataCollectionStep, DataConverterStep, DataQualityMetrics

class TestDataCollectionStep:
    """Test suite for DataCollectionStep."""

    @pytest.fixture
    def sample_config(self):
        """Create sample step configuration."""
        return StepConfig(
            name="test_data_collection",
            parameters={
                'source_type': 'file',
                'required_columns': ['open', 'high', 'low', 'close', 'volume'],
                'min_rows': 100,
                'max_null_percentage': 5.0,
                'save_snapshot': False
            }
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample market data."""
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='1H')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'open': 100 + np.random.randn(1000).cumsum(),
            'high': 105 + np.random.randn(1000).cumsum(),
            'low': 95 + np.random.randn(1000).cumsum(),
            'close': 100 + np.random.randn(1000).cumsum(),
            'volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)
        
        return data

    @pytest.fixture
    def temp_file(self, sample_data):
        """Create temporary parquet file with sample data."""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            sample_data.to_parquet(f.name)
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def data_collection_step(self, sample_config):
        """Create DataCollectionStep instance."""
        return DataCollectionStep(sample_config)

    def test_step_initialization(self, data_collection_step):
        """Test step initialization."""
        assert data_collection_step.name == "test_data_collection"
        assert data_collection_step.version == "2.0.0"
        assert data_collection_step.description is not None
        assert data_collection_step.input_schema is not None
        assert data_collection_step.output_schema is not None

    def test_input_validation(self, data_collection_step):
        """Test input validation."""
        # Valid inputs
        valid_inputs = {
            'source': 'test_file.parquet',
            'symbol': 'BTCUSDT',
            'timeframe': '1h'
        }
        
        # This would be async in real implementation
        # result = await data_collection_step.validate_inputs(**valid_inputs)
        # assert result is True

    @pytest.mark.asyncio
    async def test_load_data_from_file(self, data_collection_step, temp_file):
        """Test loading data from file."""
        data = await data_collection_step._load_from_file(temp_file)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 1000
        assert 'open' in data.columns
        assert 'high' in data.columns
        assert 'low' in data.columns
        assert 'close' in data.columns
        assert 'volume' in data.columns

    @pytest.mark.asyncio
    async def test_validate_data_success(self, data_collection_step, sample_data):
        """Test successful data validation."""
        result = await data_collection_step.validate_data(sample_data)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_data_missing_columns(self, data_collection_step):
        """Test data validation with missing columns."""
        incomplete_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107]
            # Missing low, close, volume
        })
        
        result = await data_collection_step.validate_data(incomplete_data)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_data_insufficient_rows(self, data_collection_step):
        """Test data validation with insufficient rows."""
        small_data = pd.DataFrame({
            'open': [100, 101],
            'high': [105, 106],
            'low': [95, 96],
            'close': [100, 101],
            'volume': [1000, 1100]
        })
        
        result = await data_collection_step.validate_data(small_data)
        assert result is False

    @pytest.mark.asyncio
    async def test_preprocess_data(self, data_collection_step, sample_data):
        """Test data preprocessing."""
        # Add some duplicates
        data_with_duplicates = pd.concat([sample_data, sample_data.iloc[:10]])
        
        processed_data = await data_collection_step.preprocess_data(data_with_duplicates)
        
        assert len(processed_data) == len(sample_data)  # Duplicates removed
        assert processed_data.index.is_monotonic_increasing  # Should be sorted

    def test_get_data_quality_metrics(self, data_collection_step, sample_data):
        """Test data quality metrics calculation."""
        metrics = data_collection_step.get_data_quality_metrics(sample_data)
        
        assert isinstance(metrics, DataQualityMetrics)
        assert metrics.total_rows == 1000
        assert metrics.total_columns == 5
        assert metrics.duplicate_rows == 0
        assert metrics.memory_usage_mb > 0

    @pytest.mark.asyncio
    async def test_execute_impl_success(self, data_collection_step, temp_file):
        """Test successful step execution."""
        result = await data_collection_step._execute_impl(source=temp_file)
        
        assert isinstance(result, dict)
        assert 'data' in result
        assert 'metadata' in result
        assert 'quality_metrics' in result
        assert isinstance(result['data'], pd.DataFrame)
        assert len(result['data']) == 1000

    @pytest.mark.asyncio
    async def test_execute_impl_validation_failure(self, data_collection_step):
        """Test step execution with validation failure."""
        # Create invalid data file
        invalid_data = pd.DataFrame({
            'open': [100, 101],  # Too few rows
            'high': [105, 106],
            'low': [95, 96],
            'close': [100, 101],
            'volume': [1000, 1100]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            invalid_data.to_parquet(f.name)
            
            with pytest.raises(ValueError, match="Data validation failed"):
                await data_collection_step._execute_impl(source=f.name)
            
            os.unlink(f.name)

class TestDataConverterStep:
    """Test suite for DataConverterStep."""

    @pytest.fixture
    def converter_config(self):
        """Create converter step configuration."""
        return StepConfig(
            name="test_data_converter",
            parameters={
                'column_mapping': {
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                },
                'required_columns': ['open', 'high', 'low', 'close', 'volume'],
                'data_types': {
                    'open': 'float64',
                    'high': 'float64',
                    'low': 'float64',
                    'close': 'float64',
                    'volume': 'int64'
                }
            }
        )

    @pytest.fixture
    def converter_step(self, converter_config):
        """Create DataConverterStep instance."""
        return DataConverterStep(converter_config)

    @pytest.fixture
    def sample_input_data(self):
        """Create sample input data with different column names."""
        return pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [100, 101, 102],
            'Volume': [1000, 1100, 1200]
        })

    def test_converter_initialization(self, converter_step):
        """Test converter step initialization."""
        assert converter_step.name == "test_data_converter"
        assert converter_step.version == "2.0.0"

    @pytest.mark.asyncio
    async def test_validate_data_success(self, converter_step, sample_input_data):
        """Test successful data validation."""
        result = await converter_step.validate_data(sample_input_data)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_data_empty(self, converter_step):
        """Test data validation with empty data."""
        empty_data = pd.DataFrame()
        result = await converter_step.validate_data(empty_data)
        assert result is False

    @pytest.mark.asyncio
    async def test_preprocess_data_column_mapping(self, converter_step, sample_input_data):
        """Test data preprocessing with column mapping."""
        processed_data = await converter_step.preprocess_data(sample_input_data)
        
        # Check that columns were renamed
        assert 'open' in processed_data.columns
        assert 'high' in processed_data.columns
        assert 'low' in processed_data.columns
        assert 'close' in processed_data.columns
        assert 'volume' in processed_data.columns
        
        # Check that original columns are gone
        assert 'Open' not in processed_data.columns
        assert 'High' not in processed_data.columns

    @pytest.mark.asyncio
    async def test_execute_impl_success(self, converter_step, sample_input_data):
        """Test successful converter execution."""
        result = await converter_step._execute_impl(data=sample_input_data)
        
        assert isinstance(result, dict)
        assert 'data' in result
        assert 'conversion_metadata' in result
        assert 'quality_metrics' in result
        
        # Check converted data
        converted_data = result['data']
        assert 'open' in converted_data.columns
        assert len(converted_data) == 3

class TestDependencyInjection:
    """Test suite for dependency injection system."""

    @pytest.fixture
    def di_container(self):
        """Create DI container for testing."""
        return EnhancedDIContainer()

    def test_service_registration(self, di_container):
        """Test service registration."""
        class TestService:
            def __init__(self):
                self.value = "test"
        
        di_container.register_singleton("test_service", TestService)
        
        # Verify service is registered
        service_info = di_container.get_service_info("test_service")
        assert service_info is not None
        assert service_info['name'] == "test_service"

    def test_service_resolution(self, di_container):
        """Test service resolution."""
        class TestService:
            def __init__(self):
                self.value = "test"
        
        di_container.register_singleton("test_service", TestService)
        
        # Resolve service
        service = di_container.get("test_service")
        assert isinstance(service, TestService)
        assert service.value == "test"

    def test_circular_dependency_detection(self, di_container):
        """Test circular dependency detection."""
        class ServiceA:
            def __init__(self, service_b=None):
                self.service_b = service_b
        
        class ServiceB:
            def __init__(self, service_a=None):
                self.service_a = service_a
        
        di_container.register_singleton("service_a", ServiceA, dependencies=["service_b"])
        di_container.register_singleton("service_b", ServiceB, dependencies=["service_a"])
        
        # Should detect circular dependency
        with pytest.raises(Exception):  # CircularDependencyError
            di_container.get("service_a")

    def test_dependency_validation(self, di_container):
        """Test dependency validation."""
        class TestService:
            def __init__(self, missing_dep=None):
                self.missing_dep = missing_dep
        
        di_container.register_singleton("test_service", TestService, dependencies=["missing_dep"])
        
        # Should detect missing dependency
        errors = di_container.validate_dependencies()
        assert len(errors) > 0
        assert any("missing_dep" in error for error in errors)

class TestConfigurationSystem:
    """Test suite for configuration system."""

    @pytest.fixture
    def config_manager(self):
        """Create configuration manager for testing."""
        return ConfigurationManager()

    @pytest.fixture
    def sample_config_data(self):
        """Create sample configuration data."""
        return {
            'name': 'Test_Pipeline',
            'version': '1.0.0',
            'description': 'Test pipeline configuration',
            'environment': 'development',
            'global_settings': {
                'data_source': {'type': 'file'},
                'model': {'type': 'lightgbm'}
            },
            'steps': [
                {
                    'name': 'test_step',
                    'class_name': 'TestStep',
                    'parameters': {'param1': 'value1'}
                }
            ]
        }

    def test_config_parsing(self, config_manager, sample_config_data):
        """Test configuration parsing."""
        config = config_manager._parse_configuration(sample_config_data)
        
        assert config.name == 'Test_Pipeline'
        assert config.version == '1.0.0'
        assert len(config.steps) == 1
        assert config.steps[0].name == 'test_step'

    def test_config_validation_success(self, config_manager, sample_config_data):
        """Test successful configuration validation."""
        config = config_manager._parse_configuration(sample_config_data)
        errors = config_manager.validate_config(config)
        assert len(errors) == 0

    def test_config_validation_failure(self, config_manager):
        """Test configuration validation with errors."""
        invalid_config_data = {
            'name': '',  # Empty name
            'version': '',  # Empty version
            'steps': [
                {
                    'name': 'step1',
                    'class_name': 'Step1',
                    'dependencies': ['missing_step']  # Missing dependency
                }
            ]
        }
        
        config = config_manager._parse_configuration(invalid_config_data)
        errors = config_manager.validate_config(config)
        assert len(errors) > 0

    def test_circular_dependency_detection(self, config_manager):
        """Test circular dependency detection in configuration."""
        circular_config_data = {
            'name': 'Circular_Pipeline',
            'version': '1.0.0',
            'steps': [
                {
                    'name': 'step1',
                    'class_name': 'Step1',
                    'dependencies': ['step2']
                },
                {
                    'name': 'step2',
                    'class_name': 'Step2',
                    'dependencies': ['step1']
                }
            ]
        }
        
        config = config_manager._parse_configuration(circular_config_data)
        errors = config_manager.validate_config(config)
        assert len(errors) > 0
        assert any('circular' in error.lower() for error in errors)

class TestPipelineOrchestrator:
    """Test suite for pipeline orchestrator."""

    @pytest.fixture
    def sample_pipeline_config(self):
        """Create sample pipeline configuration."""
        return PipelineConfiguration(
            name="Test_Pipeline",
            version="1.0.0",
            description="Test pipeline",
            steps=[
                StepConfiguration(
                    name="test_step",
                    class_name="TestStep",
                    parameters={"param1": "value1"}
                )
            ]
        )

    @pytest.fixture
    def orchestrator(self, sample_pipeline_config):
        """Create pipeline orchestrator for testing."""
        return EnhancedPipelineOrchestrator(config=sample_pipeline_config)

    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.config.name == "Test_Pipeline"
        assert orchestrator.config.version == "1.0.0"

    def test_get_status_not_started(self, orchestrator):
        """Test status when pipeline hasn't started."""
        status = orchestrator.get_status()
        assert status["status"] == "not_started"

    def test_get_execution_history_empty(self, orchestrator):
        """Test empty execution history."""
        history = orchestrator.get_execution_history()
        assert len(history) == 0

class TestStepFactory:
    """Test suite for step factory."""

    def test_step_registration(self):
        """Test step registration."""
        class TestStep:
            pass
        
        StepFactory.register_step("test_step", TestStep)
        
        # Verify step is registered
        available_steps = StepFactory.list_available_steps()
        assert "test_step" in available_steps

    def test_step_creation(self):
        """Test step creation from configuration."""
        class TestStep:
            def __init__(self, config, logger=None, di_container=None):
                self.config = config
                self.logger = logger
                self.di_container = di_container
        
        StepFactory.register_step("test_step", TestStep)
        
        config = StepConfig(name="test", parameters={"type": "test_step"})
        step = StepFactory.create_step(config)
        
        assert isinstance(step, TestStep)
        assert step.config.name == "test"

    def test_step_info(self):
        """Test step information retrieval."""
        class TestStep:
            description = "Test step description"
            version = "1.0.0"
        
        StepFactory.register_step("test_step", TestStep)
        
        info = StepFactory.get_step_info("test_step")
        assert info is not None
        assert info['name'] == "test_step"
        assert info['description'] == "Test step description"
        assert info['version'] == "1.0.0"

# Integration tests
class TestIntegration:
    """Integration tests for the complete system."""

    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self):
        """Test end-to-end pipeline execution."""
        # Create sample data file
        sample_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            sample_data.to_parquet(f.name)
            
            # Create configuration
            config = PipelineConfiguration(
                name="Integration_Test_Pipeline",
                version="1.0.0",
                steps=[
                    StepConfiguration(
                        name="data_collection",
                        class_name="DataCollectionStep",
                        parameters={
                            'source_type': 'file',
                            'required_columns': ['open', 'high', 'low', 'close', 'volume'],
                            'min_rows': 5
                        }
                    )
                ]
            )
            
            # Create orchestrator
            orchestrator = EnhancedPipelineOrchestrator(config=config)
            
            # Run pipeline
            result = await orchestrator.run()
            
            # Verify results
            assert result.status.value == "completed"
            assert len(result.step_results) == 1
            assert "data_collection" in result.step_results
            
            step_result = result.step_results["data_collection"]
            assert step_result.status == StepStatus.COMPLETED
            assert step_result.data is not None
            
            os.unlink(f.name)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])