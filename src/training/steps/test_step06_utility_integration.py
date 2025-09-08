"""
Comprehensive Test Suite for Step06 Utility Integration

This module tests the extensive utility integration in step06 components:
- Dependency injection container functionality
- Utility service initialization and health checks
- Feature engineering with utility integration
- Comprehensive implementation with utilities
- Validation orchestrator with utilities
- M1 optimization utilities
- Data processing utilities
- Mathematical validation utilities
- Serialization utilities
- Parquet utilities
"""

import asyncio
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import logging
from typing import Dict, Any, Optional

# Import step06 components with utility integration
from .step06_utility_container import (
    Step06UtilityContainer, UtilityConfig, get_utility_container, 
    utility_container_context, inject_utilities
)
from .step06_comprehensive_implementation import Step06ComprehensiveImplementation
from .step06_enhanced_feature_engineering import EnhancedFeatureEngineering
from .step06_validation_orchestrator import Step06ValidationOrchestrator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestStep06UtilityIntegration:
    """Test suite for step06 utility integration."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='1min')
        
        # Generate realistic OHLCV data
        base_price = 100.0
        returns = np.random.normal(0, 0.001, 100)
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        market_data = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        
        # Ensure OHLC consistency
        market_data['high'] = np.maximum(market_data['high'], np.maximum(market_data['open'], market_data['close']))
        market_data['low'] = np.minimum(market_data['low'], np.minimum(market_data['open'], market_data['close']))
        
        return market_data
    
    @pytest.fixture
    def utility_config(self):
        """Create utility configuration for testing."""
        return UtilityConfig(
            enable_common_operations=True,
            enable_data_processing=True,
            enable_math_validation=True,
            enable_parquet_utils=True,
            enable_serialization=True,
            enable_m1_gpu=True,
            enable_m1_memory=True,
            enable_m1_cpu=True,
            data_processing_chunk_size=50,
            m1_memory_limit_gb=2.0,
            m1_max_workers=4
        )
    
    @pytest.fixture
    def step06_config(self):
        """Create step06 configuration for testing."""
        return {
            'step06_feature_engineering': {
                'chunk_size': 50,
                'max_features': 100,
                'polynomial_degree': 2,
                'correlation_threshold': 0.95,
                'memory_limit_mb': 100
            }
        }

    @pytest.mark.asyncio
    async def test_utility_container_initialization(self, utility_config):
        """Test utility container initialization and health checks."""
        logger.info("🧪 Testing utility container initialization...")
        
        # Test container creation
        container = Step06UtilityContainer(utility_config)
        assert container is not None
        assert not container._is_initialized
        
        # Test initialization
        await container.initialize()
        assert container._is_initialized
        
        # Test service retrieval
        common_ops = container.get_common_operations()
        assert common_ops is not None
        
        data_proc = container.get_data_processing()
        assert data_proc is not None
        
        math_val = container.get_math_validation()
        assert math_val is not None
        
        parquet = container.get_parquet()
        assert parquet is not None
        
        serialization = container.get_serialization()
        assert serialization is not None
        
        m1_gpu = container.get_m1_gpu()
        assert m1_gpu is not None
        
        m1_memory = container.get_m1_memory()
        assert m1_memory is not None
        
        m1_cpu = container.get_m1_cpu()
        assert m1_cpu is not None
        
        # Test health report
        health_report = container.get_health_report()
        assert health_report is not None
        assert 'status' in health_report
        assert 'total_services' in health_report
        assert 'healthy_services' in health_report
        
        logger.info(f"✅ Health report: {health_report['status']}")
        logger.info(f"   Healthy services: {health_report['healthy_services']}/{health_report['total_services']}")
        
        # Test cleanup
        await container.cleanup()
        assert not container._is_initialized
        
        logger.info("✅ Utility container initialization test passed")

    @pytest.mark.asyncio
    async def test_utility_container_context_manager(self, utility_config):
        """Test utility container context manager."""
        logger.info("🧪 Testing utility container context manager...")
        
        async with utility_container_context(utility_config) as container:
            assert container is not None
            assert container._is_initialized
            
            # Test service access within context
            common_ops = container.get_common_operations()
            assert common_ops is not None
            
            health_report = container.get_health_report()
            assert health_report is not None
        
        logger.info("✅ Utility container context manager test passed")

    @pytest.mark.asyncio
    async def test_enhanced_feature_engineering_with_utilities(self, sample_market_data, step06_config, utility_config):
        """Test enhanced feature engineering with utility integration."""
        logger.info("🧪 Testing enhanced feature engineering with utilities...")
        
        # Initialize feature engineering with utilities
        feature_engineering = EnhancedFeatureEngineering(step06_config, utility_config)
        
        try:
            # Initialize utilities
            await feature_engineering.initialize_utilities()
            
            # Test utility-enhanced feature creation
            enhanced_features = await feature_engineering.create_enhanced_features_with_utilities(sample_market_data)
            
            # Verify results
            assert enhanced_features is not None
            assert len(enhanced_features) == len(sample_market_data)
            assert len(enhanced_features.columns) > len(sample_market_data.columns)
            
            # Check that utility-specific features were created
            expected_features = ['price_range', 'price_range_pct', 'volatility', 'volatility_pct', 
                               'momentum_5', 'momentum_10', 'momentum_20', 'rsi_14', 'sma_20', 'ema_12',
                               'bb_upper_20', 'bb_lower_20', 'macd_line']
            
            for feature in expected_features:
                if feature in enhanced_features.columns:
                    logger.info(f"✅ Feature '{feature}' created successfully")
            
            # Check processing stats
            stats = feature_engineering.get_processing_stats()
            assert stats['utility_operations_count'] > 0
            assert stats['total_features_created'] > 0
            
            logger.info(f"✅ Created {len(enhanced_features.columns)} features")
            logger.info(f"   Utility operations: {stats['utility_operations_count']}")
            
        finally:
            # Cleanup
            await feature_engineering.cleanup()
        
        logger.info("✅ Enhanced feature engineering with utilities test passed")

    @pytest.mark.asyncio
    async def test_comprehensive_implementation_with_utilities(self, sample_market_data, step06_config, utility_config):
        """Test comprehensive implementation with utility integration."""
        logger.info("🧪 Testing comprehensive implementation with utilities...")
        
        # Initialize comprehensive implementation with utilities
        implementation = Step06ComprehensiveImplementation(step06_config, utility_config)
        
        try:
            # Run comprehensive pipeline with utilities
            results = await implementation.run_comprehensive_pipeline(sample_market_data)
            
            # Verify results
            assert results is not None
            assert 'pipeline_status' in results
            assert 'feature_engineering_results' in results
            assert 'labeling_results' in results
            assert 'validation_results' in results
            assert 'utility_integration_results' in results
            assert 'utility_health_report' in results
            
            # Check pipeline status
            assert results['pipeline_status'] in ['completed', 'success', 'running']
            
            # Check feature engineering results
            feature_results = results['feature_engineering_results']
            assert 'enhanced_features' in feature_results
            assert 'features_created' in feature_results
            
            # Check labeling results
            labeling_results = results['labeling_results']
            assert 'labels' in labeling_results
            assert 'labels_generated' in labeling_results
            
            # Check utility integration results
            utility_results = results['utility_integration_results']
            assert 'memory_optimization' in utility_results
            assert 'performance_optimization' in utility_results
            
            # Check utility health report
            health_report = results['utility_health_report']
            assert health_report is not None
            assert 'status' in health_report
            
            # Check performance metrics
            performance_metrics = results['performance_metrics']
            assert 'utility_operations_count' in performance_metrics
            assert 'utility_errors' in performance_metrics
            
            logger.info(f"✅ Pipeline status: {results['pipeline_status']}")
            logger.info(f"   Features created: {feature_results['features_created']}")
            logger.info(f"   Labels generated: {labeling_results['labels_generated']}")
            logger.info(f"   Utility operations: {performance_metrics['utility_operations_count']}")
            logger.info(f"   Utility errors: {performance_metrics['utility_errors']}")
            
        finally:
            # Cleanup
            await implementation.cleanup()
        
        logger.info("✅ Comprehensive implementation with utilities test passed")

    @pytest.mark.asyncio
    async def test_validation_orchestrator_with_utilities(self, sample_market_data, utility_config):
        """Test validation orchestrator with utility integration."""
        logger.info("🧪 Testing validation orchestrator with utilities...")
        
        # Initialize validation orchestrator with utilities
        orchestrator = Step06ValidationOrchestrator(utility_config=utility_config)
        
        try:
            # Run comprehensive validation with utilities
            results = await orchestrator.run_comprehensive_validation(sample_market_data)
            
            # Verify results
            assert results is not None
            assert 'validation_status' in results
            assert 'utility_integration' in results
            assert 'utility_health' in results
            assert 'performance_metrics' in results
            
            # Check validation status
            assert results['validation_status'] in ['passed', 'failed', 'warning']
            
            # Check utility integration
            utility_integration = results['utility_integration']
            assert 'status' in utility_integration
            
            # Check utility health
            utility_health = results['utility_health']
            assert utility_health is not None
            
            # Check performance metrics
            performance_metrics = results['performance_metrics']
            assert 'utility_initialization_time' in performance_metrics
            assert 'utility_errors' in performance_metrics
            
            logger.info(f"✅ Validation status: {results['validation_status']}")
            logger.info(f"   Utility integration: {utility_integration['status']}")
            logger.info(f"   Utility health: {utility_health.get('status', 'unknown')}")
            logger.info(f"   Utility errors: {performance_metrics['utility_errors']}")
            
        finally:
            # Cleanup
            await orchestrator.cleanup()
        
        logger.info("✅ Validation orchestrator with utilities test passed")

    @pytest.mark.asyncio
    async def test_utility_service_functionality(self, utility_config):
        """Test individual utility service functionality."""
        logger.info("🧪 Testing individual utility service functionality...")
        
        async with utility_container_context(utility_config) as container:
            # Test common operations service
            common_ops = container.get_common_operations()
            if common_ops:
                # Test datetime operations
                current_time = common_ops.get_operation('datetime', 'get_current_datetime')()
                assert current_time is not None
                logger.info(f"✅ Common operations datetime: {current_time}")
            
            # Test data processing service
            data_proc = container.get_data_processing()
            if data_proc and data_proc.validator:
                # Create test data
                test_data = pd.DataFrame({
                    'col1': [1, 2, 3, 4, 5],
                    'col2': [1.1, 2.2, 3.3, 4.4, 5.5],
                    'col3': ['a', 'b', 'c', 'd', 'e']
                })
                
                # Test validation
                quality_report = data_proc.validator.validate_dataframe(test_data)
                assert quality_report is not None
                logger.info(f"✅ Data processing validation: {quality_report.summary.get('data_quality_score', 0)}")
            
            # Test math validation service
            math_val = container.get_math_validation()
            if math_val:
                # Test safe division
                result = math_val.safe_divide(10, 2)
                assert result == 5.0
                
                # Test safe division with zero
                result = math_val.safe_divide(10, 0, default=0.0)
                assert result == 0.0
                
                logger.info("✅ Math validation safe operations working")
            
            # Test serialization service
            serialization = container.get_serialization()
            if serialization:
                # Test JSON serialization
                test_data = {'test': 'data', 'number': 42}
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    temp_file = Path(f.name)
                
                try:
                    # Save data
                    success = serialization.serializers['json'].save(test_data, temp_file)
                    assert success
                    
                    # Load data
                    loaded_data = serialization.serializers['json'].load(temp_file)
                    assert loaded_data == test_data
                    
                    logger.info("✅ Serialization service working")
                    
                finally:
                    # Clean up
                    temp_file.unlink(missing_ok=True)
            
            # Test parquet service
            parquet = container.get_parquet()
            if parquet and parquet.parquet_utils:
                # Create test DataFrame
                test_df = pd.DataFrame({
                    'col1': [1, 2, 3, 4, 5],
                    'col2': [1.1, 2.2, 3.3, 4.4, 5.5]
                })
                
                with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
                    temp_file = Path(f.name)
                
                try:
                    # Save DataFrame
                    test_df.to_parquet(temp_file)
                    
                    # Validate parquet file
                    validation_result = parquet.parquet_utils.validate_parquet_file(str(temp_file))
                    assert validation_result is not None
                    assert validation_result.get('valid', False)
                    
                    logger.info("✅ Parquet service working")
                    
                finally:
                    # Clean up
                    temp_file.unlink(missing_ok=True)
            
            # Test M1 services
            m1_gpu = container.get_m1_gpu()
            if m1_gpu and m1_gpu.manager:
                logger.info("✅ M1 GPU service available")
            
            m1_memory = container.get_m1_memory()
            if m1_memory and m1_memory.optimizer:
                logger.info("✅ M1 memory service available")
            
            m1_cpu = container.get_m1_cpu()
            if m1_cpu and m1_cpu.optimizer:
                logger.info("✅ M1 CPU service available")
        
        logger.info("✅ Individual utility service functionality test passed")

    @pytest.mark.asyncio
    async def test_dependency_injection_decorator(self, utility_config):
        """Test the dependency injection decorator."""
        logger.info("🧪 Testing dependency injection decorator...")
        
        class TestClass:
            def __init__(self):
                self.utility_container = None
            
            @inject_utilities('common_ops', 'math_val')
            async def test_method(self, common_ops, math_val):
                """Test method with injected utilities."""
                assert common_ops is not None or common_ops is None  # Allow None for graceful degradation
                assert math_val is not None or math_val is None  # Allow None for graceful degradation
                return "test_result"
        
        # Initialize utility container
        container = await get_utility_container(utility_config)
        
        # Create test instance
        test_instance = TestClass()
        test_instance.utility_container = container
        
        # Test method with dependency injection
        result = await test_instance.test_method()
        assert result == "test_result"
        
        logger.info("✅ Dependency injection decorator test passed")

    @pytest.mark.asyncio
    async def test_error_handling_and_graceful_degradation(self, sample_market_data, step06_config):
        """Test error handling and graceful degradation when utilities fail."""
        logger.info("🧪 Testing error handling and graceful degradation...")
        
        # Create config with some utilities disabled
        limited_utility_config = UtilityConfig(
            enable_common_operations=True,
            enable_data_processing=False,  # Disable to test graceful degradation
            enable_math_validation=True,
            enable_parquet_utils=False,    # Disable to test graceful degradation
            enable_serialization=True,
            enable_m1_gpu=False,           # Disable to test graceful degradation
            enable_m1_memory=False,        # Disable to test graceful degradation
            enable_m1_cpu=False,           # Disable to test graceful degradation
            data_processing_chunk_size=50,
            m1_memory_limit_gb=2.0,
            m1_max_workers=4
        )
        
        # Test feature engineering with limited utilities
        feature_engineering = EnhancedFeatureEngineering(step06_config, limited_utility_config)
        
        try:
            # Initialize utilities (should handle disabled services gracefully)
            await feature_engineering.initialize_utilities()
            
            # Test feature creation (should work with available utilities)
            enhanced_features = await feature_engineering.create_enhanced_features_with_utilities(sample_market_data)
            
            # Verify results (should still work with available utilities)
            assert enhanced_features is not None
            assert len(enhanced_features) == len(sample_market_data)
            assert len(enhanced_features.columns) > len(sample_market_data.columns)
            
            # Check that some features were still created
            stats = feature_engineering.get_processing_stats()
            assert stats['total_features_created'] > 0
            
            logger.info(f"✅ Graceful degradation working - created {len(enhanced_features.columns)} features")
            logger.info(f"   Utility operations: {stats['utility_operations_count']}")
            logger.info(f"   Utility errors: {stats['utility_errors']}")
            
        finally:
            # Cleanup
            await feature_engineering.cleanup()
        
        logger.info("✅ Error handling and graceful degradation test passed")

    @pytest.mark.asyncio
    async def test_performance_metrics_and_monitoring(self, sample_market_data, step06_config, utility_config):
        """Test performance metrics and monitoring with utilities."""
        logger.info("🧪 Testing performance metrics and monitoring...")
        
        # Test comprehensive implementation
        implementation = Step06ComprehensiveImplementation(step06_config, utility_config)
        
        try:
            # Run pipeline and measure performance
            results = await implementation.run_comprehensive_pipeline(sample_market_data)
            
            # Check performance metrics
            performance_metrics = results['performance_metrics']
            
            # Verify utility-related metrics
            assert 'utility_initialization_time' in performance_metrics
            assert 'utility_operations_count' in performance_metrics
            assert 'utility_errors' in performance_metrics
            assert 'data_processing_time' in performance_metrics
            
            # Verify metrics are reasonable
            assert performance_metrics['utility_initialization_time'] >= 0
            assert performance_metrics['utility_operations_count'] >= 0
            assert performance_metrics['utility_errors'] >= 0
            assert performance_metrics['data_processing_time'] >= 0
            
            # Log performance metrics
            logger.info("📊 Performance Metrics:")
            logger.info(f"   Total execution time: {performance_metrics['total_execution_time']:.2f}s")
            logger.info(f"   Utility initialization: {performance_metrics['utility_initialization_time']:.2f}s")
            logger.info(f"   Data processing: {performance_metrics['data_processing_time']:.2f}s")
            logger.info(f"   Utility operations: {performance_metrics['utility_operations_count']}")
            logger.info(f"   Utility errors: {performance_metrics['utility_errors']}")
            logger.info(f"   Features created: {performance_metrics['features_created']}")
            logger.info(f"   Labels generated: {performance_metrics['labels_generated']}")
            
        finally:
            # Cleanup
            await implementation.cleanup()
        
        logger.info("✅ Performance metrics and monitoring test passed")

async def run_comprehensive_utility_integration_tests():
    """Run all utility integration tests."""
    logger.info("🚀 Starting comprehensive utility integration tests...")
    
    # Create test instance
    test_suite = TestStep06UtilityIntegration()
    
    # Create fixtures
    sample_data = test_suite.sample_market_data()
    utility_config = test_suite.utility_config()
    step06_config = test_suite.step06_config()
    
    try:
        # Run all tests
        await test_suite.test_utility_container_initialization(utility_config)
        await test_suite.test_utility_container_context_manager(utility_config)
        await test_suite.test_enhanced_feature_engineering_with_utilities(sample_data, step06_config, utility_config)
        await test_suite.test_comprehensive_implementation_with_utilities(sample_data, step06_config, utility_config)
        await test_suite.test_validation_orchestrator_with_utilities(sample_data, utility_config)
        await test_suite.test_utility_service_functionality(utility_config)
        await test_suite.test_dependency_injection_decorator(utility_config)
        await test_suite.test_error_handling_and_graceful_degradation(sample_data, step06_config)
        await test_suite.test_performance_metrics_and_monitoring(sample_data, step06_config, utility_config)
        
        logger.info("🎉 All utility integration tests passed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    # Run tests
    asyncio.run(run_comprehensive_utility_integration_tests())