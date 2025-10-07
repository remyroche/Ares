"""
Integration Test for Optimized Interaction Feature Generation

This module provides comprehensive integration tests to ensure the optimized
interaction feature generation pipeline works correctly with all components
and maintains consistency with the sub_pipeline architecture.

Test Coverage:
- Direct orchestrator usage
- Component interface
- Sub-pipeline integration
- Matrix operations optimization
- Hardware acceleration
- Error handling and recovery
- Performance validation
- Memory usage validation
"""

import asyncio
import time
import pandas as pd
import numpy as np
import pytest
from typing import Dict, Any, List
from pathlib import Path

# Import tprint utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, 
    tprint_debug, tprint_performance, tprint_progress
)

# Import the components to test
from .optimized_interaction_orchestrator import (
    OptimizedInteractionOrchestrator, OptimizedInteractionConfig, generate_optimized_interaction_features
)

from .roadmap_feature_generation_component import (
    RoadmapFeatureGenerationComponent, RoadmapFeatureGenerationConfig, execute_roadmap_feature_generation
)

from ..sub_pipeline import PreTrainingSubPipeline, SubPipelineConfig, ExecutionMode


class TestDataGenerator:
    """Generate test data for integration tests."""
    
    @staticmethod
    def create_market_data(n_rows: int = 1000, symbol: str = "ETHUSDT") -> pd.DataFrame:
        """Create realistic market data for testing."""
        tprint_debug(f"Creating market data: {n_rows} rows for {symbol}")
        
        # Generate timestamps
        timestamps = pd.date_range(start='2024-01-01', periods=n_rows, freq='15min')
        
        # Generate price data with realistic patterns
        np.random.seed(42)
        base_price = 2000.0 if symbol == "ETHUSDT" else 100.0
        
        # Generate returns with some autocorrelation
        returns = np.random.normal(0, 0.02, n_rows)
        for i in range(1, len(returns)):
            returns[i] = 0.1 * returns[i-1] + 0.9 * returns[i]
        
        # Generate prices
        prices = [base_price]
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        prices = np.array(prices)
        
        # Generate OHLC from prices
        high_multiplier = 1 + np.abs(np.random.normal(0, 0.01, n_rows))
        low_multiplier = 1 - np.abs(np.random.normal(0, 0.01, n_rows))
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices * (1 + np.random.normal(0, 0.005, n_rows)),
            'high': prices * high_multiplier,
            'low': prices * low_multiplier,
            'close': prices,
            'volume': np.random.lognormal(10, 1, n_rows),
            'trade_count': np.random.poisson(50, n_rows),
            'bid': prices * (1 - np.random.uniform(0.0001, 0.001, n_rows)),
            'ask': prices * (1 + np.random.uniform(0.0001, 0.001, n_rows)),
            'bid_size': np.random.lognormal(8, 1, n_rows),
            'ask_size': np.random.lognormal(8, 1, n_rows)
        })
        
        # Set timestamp as index
        data.set_index('timestamp', inplace=True)
        
        return data
    
    @staticmethod
    def create_targets(data: pd.DataFrame) -> Dict[int, pd.Series]:
        """Create realistic targets for testing."""
        tprint_debug("Creating target series")
        
        # Generate targets based on future returns
        future_returns = data['close'].pct_change().shift(-1)
        
        targets = {
            1: future_returns,  # 1-step target
            3: data['close'].pct_change().shift(-3),  # 3-step target
            5: data['close'].pct_change().shift(-5)   # 5-step target
        }
        
        return targets


class TestOptimizedInteractionOrchestrator:
    """Test the optimized interaction orchestrator."""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        tprint_info("🧪 Testing orchestrator initialization...")
        
        config = OptimizedInteractionConfig(
            symbol="ETHUSDT",
            exchange="binance",
            timeframe="15m",
            feature_budget_pre=50,
            interactions_cap=10
        )
        
        orchestrator = OptimizedInteractionOrchestrator(config)
        
        assert orchestrator.config.symbol == "ETHUSDT"
        assert orchestrator.config.exchange == "binance"
        assert orchestrator.config.timeframe == "15m"
        assert orchestrator.config.feature_budget_pre == 50
        assert orchestrator.config.interactions_cap == 10
        
        tprint_success("✅ Orchestrator initialization test passed")
    
    @pytest.mark.asyncio
    async def test_orchestrator_execution(self):
        """Test orchestrator execution with sample data."""
        tprint_info("🧪 Testing orchestrator execution...")
        
        # Create test data
        data = TestDataGenerator.create_market_data(500)
        targets = TestDataGenerator.create_targets(data)
        
        # Create configuration
        config = OptimizedInteractionConfig(
            symbol="ETHUSDT",
            exchange="binance",
            timeframe="15m",
            feature_budget_pre=30,
            interactions_cap=5,
            enable_matrix_optimization=True,
            enable_hardware_optimization=True,
            verbose_logging=False  # Reduce logging for tests
        )
        
        # Create training input and pipeline state
        training_input = {
            'data': data,
            'targets': targets
        }
        
        pipeline_state = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframe': '15m',
            'data_dir': 'historical_data'
        }
        
        # Execute feature generation
        result = await generate_optimized_interaction_features(
            training_input, pipeline_state, config
        )
        
        # Validate results
        assert result.success, f"Feature generation failed: {result.error_message}"
        assert len(result.feature_names) > 0, "No features generated"
        assert len(result.selected_features) > 0, "No features selected"
        assert isinstance(result.features, pd.DataFrame), "Features should be DataFrame"
        assert isinstance(result.interaction_features, pd.DataFrame), "Interactions should be DataFrame"
        assert isinstance(result.cross_timeframe_features, pd.DataFrame), "Cross-timeframe features should be DataFrame"
        assert result.execution_time > 0, "Execution time should be positive"
        assert result.memory_usage_mb >= 0, "Memory usage should be non-negative"
        
        tprint_success("✅ Orchestrator execution test passed")
    
    @pytest.mark.asyncio
    async def test_orchestrator_error_handling(self):
        """Test orchestrator error handling."""
        tprint_info("🧪 Testing orchestrator error handling...")
        
        # Test with invalid data
        config = OptimizedInteractionConfig(
            symbol="ETHUSDT",
            exchange="binance",
            timeframe="15m",
            feature_budget_pre=30,
            interactions_cap=5,
            verbose_logging=False
        )
        
        # Invalid training input
        training_input = {
            'data': None,  # Invalid data
            'targets': {}
        }
        
        pipeline_state = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframe': '15m'
        }
        
        result = await generate_optimized_interaction_features(
            training_input, pipeline_state, config
        )
        
        # Should fail gracefully
        assert not result.success, "Should fail with invalid data"
        assert result.error_message is not None, "Should have error message"
        
        tprint_success("✅ Orchestrator error handling test passed")


class TestRoadmapFeatureGenerationComponent:
    """Test the roadmap feature generation component."""
    
    def test_component_initialization(self):
        """Test component initialization."""
        tprint_info("🧪 Testing component initialization...")
        
        config = RoadmapFeatureGenerationConfig(
            symbol="ETHUSDT",
            exchange="binance",
            timeframe="15m",
            feature_budget_pre=50,
            interactions_cap=10
        )
        
        component = RoadmapFeatureGenerationComponent(config)
        
        assert component.config.symbol == "ETHUSDT"
        assert component.config.exchange == "binance"
        assert component.config.timeframe == "15m"
        assert component.config.feature_budget_pre == 50
        assert component.config.interactions_cap == 10
        
        tprint_success("✅ Component initialization test passed")
    
    @pytest.mark.asyncio
    async def test_component_execution(self):
        """Test component execution."""
        tprint_info("🧪 Testing component execution...")
        
        # Create test data
        data = TestDataGenerator.create_market_data(500)
        targets = TestDataGenerator.create_targets(data)
        
        # Create configuration
        config = RoadmapFeatureGenerationConfig(
            symbol="ETHUSDT",
            exchange="binance",
            timeframe="15m",
            feature_budget_pre=30,
            interactions_cap=5,
            enable_matrix_optimization=True,
            enable_hardware_optimization=True,
            verbose_logging=False
        )
        
        # Create training input and pipeline state
        training_input = {
            'data': data,
            'targets': targets
        }
        
        pipeline_state = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframe': '15m',
            'data_dir': 'historical_data'
        }
        
        # Execute component
        result = await execute_roadmap_feature_generation(
            training_input, pipeline_state, config
        )
        
        # Validate results
        assert result.success, f"Component execution failed: {result.error_message}"
        assert 'roadmap_feature_generation_result' in result.artifacts, "Should have roadmap result in artifacts"
        assert result.execution_time > 0, "Execution time should be positive"
        assert len(result.output_files) > 0, "Should have output files"
        assert len(result.metadata) > 0, "Should have metadata"
        
        # Validate roadmap result
        roadmap_result = result.artifacts['roadmap_feature_generation_result']
        assert 'features' in roadmap_result, "Should have features"
        assert 'feature_names' in roadmap_result, "Should have feature names"
        assert 'selected_features' in roadmap_result, "Should have selected features"
        
        tprint_success("✅ Component execution test passed")
    
    def test_component_info(self):
        """Test component info retrieval."""
        tprint_info("🧪 Testing component info...")
        
        config = RoadmapFeatureGenerationConfig()
        component = RoadmapFeatureGenerationComponent(config)
        
        info = component.get_component_info()
        
        assert 'name' in info, "Should have name"
        assert 'description' in info, "Should have description"
        assert 'version' in info, "Should have version"
        assert 'dependencies' in info, "Should have dependencies"
        assert 'config' in info, "Should have config"
        assert 'capabilities' in info, "Should have capabilities"
        
        assert info['name'] == 'roadmap_feature_generation'
        assert isinstance(info['capabilities'], list)
        assert len(info['capabilities']) > 0
        
        tprint_success("✅ Component info test passed")


class TestSubPipelineIntegration:
    """Test sub-pipeline integration."""
    
    @pytest.mark.asyncio
    async def test_sub_pipeline_integration(self):
        """Test integration with sub-pipeline."""
        tprint_info("🧪 Testing sub-pipeline integration...")
        
        # Create test data
        data = TestDataGenerator.create_market_data(500)
        targets = TestDataGenerator.create_targets(data)
        
        # Create sub-pipeline configuration
        config = SubPipelineConfig(
            mode=ExecutionMode.FULL,
            symbol="ETHUSDT",
            exchange="binance",
            timeframe="15m",
            data_dir="historical_data",
            parallel_processing=True,
            max_workers=2,
            custom_params={
                'feature_budget_pre': 30,
                'interactions_cap': 5,
                'enable_matrix_optimization': True,
                'enable_hardware_optimization': True,
                'verbose_logging': False
            }
        )
        
        # Create training input and pipeline state
        training_input = {
            'data': data,
            'targets': targets
        }
        
        pipeline_state = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframe': '15m',
            'data_dir': 'historical_data'
        }
        
        # Execute sub-pipeline
        pipeline = PreTrainingSubPipeline()
        result = await pipeline._execute_roadmap_feature_generation(config)
        
        # Validate results
        assert result.success, f"Sub-pipeline execution failed: {result.error_message}"
        assert result.status.value == 'completed', f"Status should be completed, got {result.status.value}"
        assert result.duration_seconds > 0, "Duration should be positive"
        assert len(result.artifacts) > 0, "Should have artifacts"
        assert len(result.output_files) > 0, "Should have output files"
        assert len(result.metadata) > 0, "Should have metadata"
        
        tprint_success("✅ Sub-pipeline integration test passed")


class TestPerformanceValidation:
    """Test performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self):
        """Test performance metrics collection."""
        tprint_info("🧪 Testing performance metrics...")
        
        # Create test data
        data = TestDataGenerator.create_market_data(1000)
        targets = TestDataGenerator.create_targets(data)
        
        # Create configuration
        config = OptimizedInteractionConfig(
            symbol="ETHUSDT",
            exchange="binance",
            timeframe="15m",
            feature_budget_pre=50,
            interactions_cap=10,
            enable_matrix_optimization=True,
            enable_hardware_optimization=True,
            verbose_logging=False
        )
        
        # Create training input and pipeline state
        training_input = {
            'data': data,
            'targets': targets
        }
        
        pipeline_state = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframe': '15m',
            'data_dir': 'historical_data'
        }
        
        # Execute and measure performance
        start_time = time.time()
        result = await generate_optimized_interaction_features(
            training_input, pipeline_state, config
        )
        end_time = time.time()
        
        # Validate performance
        assert result.success, f"Feature generation failed: {result.error_message}"
        assert result.execution_time > 0, "Execution time should be positive"
        assert result.execution_time <= (end_time - start_time) + 1.0, "Execution time should be reasonable"
        assert result.memory_usage_mb >= 0, "Memory usage should be non-negative"
        
        # Check stage results
        if hasattr(result, 'stage_results') and result.stage_results:
            for stage, stage_result in result.stage_results.items():
                assert 'stage_time' in stage_result, f"Stage {stage} should have stage_time"
                assert stage_result['stage_time'] > 0, f"Stage {stage} time should be positive"
        
        tprint_success("✅ Performance metrics test passed")
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage validation."""
        tprint_info("🧪 Testing memory usage...")
        
        # Create test data
        data = TestDataGenerator.create_market_data(2000)  # Larger dataset
        targets = TestDataGenerator.create_targets(data)
        
        # Create configuration
        config = OptimizedInteractionConfig(
            symbol="ETHUSDT",
            exchange="binance",
            timeframe="15m",
            feature_budget_pre=100,
            interactions_cap=15,
            enable_matrix_optimization=True,
            enable_hardware_optimization=True,
            verbose_logging=False
        )
        
        # Create training input and pipeline state
        training_input = {
            'data': data,
            'targets': targets
        }
        
        pipeline_state = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframe': '15m',
            'data_dir': 'historical_data'
        }
        
        # Execute
        result = await generate_optimized_interaction_features(
            training_input, pipeline_state, config
        )
        
        # Validate memory usage
        assert result.success, f"Feature generation failed: {result.error_message}"
        assert result.memory_usage_mb > 0, "Memory usage should be positive"
        assert result.memory_usage_mb < 1000, "Memory usage should be reasonable (< 1GB)"
        
        # Check that features are reasonable size
        if hasattr(result, 'features') and not result.features.empty:
            feature_memory = result.features.memory_usage(deep=True).sum() / 1024 / 1024
            assert feature_memory > 0, "Feature memory usage should be positive"
            assert feature_memory < 500, "Feature memory usage should be reasonable"
        
        tprint_success("✅ Memory usage test passed")


class TestErrorHandling:
    """Test error handling and recovery."""
    
    @pytest.mark.asyncio
    async def test_invalid_data_handling(self):
        """Test handling of invalid data."""
        tprint_info("🧪 Testing invalid data handling...")
        
        config = OptimizedInteractionConfig(
            symbol="ETHUSDT",
            exchange="binance",
            timeframe="15m",
            feature_budget_pre=30,
            interactions_cap=5,
            verbose_logging=False
        )
        
        # Test with empty data
        training_input = {
            'data': pd.DataFrame(),
            'targets': {}
        }
        
        pipeline_state = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframe': '15m'
        }
        
        result = await generate_optimized_interaction_features(
            training_input, pipeline_state, config
        )
        
        assert not result.success, "Should fail with empty data"
        assert result.error_message is not None, "Should have error message"
        
        tprint_success("✅ Invalid data handling test passed")
    
    @pytest.mark.asyncio
    async def test_missing_columns_handling(self):
        """Test handling of missing required columns."""
        tprint_info("🧪 Testing missing columns handling...")
        
        config = OptimizedInteractionConfig(
            symbol="ETHUSDT",
            exchange="binance",
            timeframe="15m",
            feature_budget_pre=30,
            interactions_cap=5,
            verbose_logging=False
        )
        
        # Test with missing columns
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='15min'),
            'open': np.random.randn(100),
            # Missing high, low, close, volume
        })
        data.set_index('timestamp', inplace=True)
        
        training_input = {
            'data': data,
            'targets': {}
        }
        
        pipeline_state = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframe': '15m'
        }
        
        result = await generate_optimized_interaction_features(
            training_input, pipeline_state, config
        )
        
        assert not result.success, "Should fail with missing columns"
        assert result.error_message is not None, "Should have error message"
        assert "Missing required columns" in result.error_message, "Should mention missing columns"
        
        tprint_success("✅ Missing columns handling test passed")


async def run_all_tests():
    """Run all integration tests."""
    tprint_success("🧪 Starting Integration Tests for Optimized Interaction Feature Generation")
    tprint_info("=" * 80)
    
    test_classes = [
        TestOptimizedInteractionOrchestrator,
        TestRoadmapFeatureGenerationComponent,
        TestSubPipelineIntegration,
        TestPerformanceValidation,
        TestErrorHandling
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        tprint_info(f"🔧 Running {test_class.__name__}...")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            test_name = f"{test_class.__name__}.{test_method}"
            
            try:
                tprint_debug(f"  Running {test_name}...")
                
                # Create test instance
                test_instance = test_class()
                
                # Run test method
                if asyncio.iscoroutinefunction(getattr(test_instance, test_method)):
                    await getattr(test_instance, test_method)()
                else:
                    getattr(test_instance, test_method)()
                
                passed_tests += 1
                tprint_debug(f"  ✅ {test_name} passed")
                
            except Exception as e:
                failed_tests += 1
                tprint_error(f"  ❌ {test_name} failed: {e}")
    
    # Summary
    tprint_info("=" * 80)
    tprint_success(f"🎉 Integration Tests Completed!")
    tprint_info(f"📊 Total tests: {total_tests}")
    tprint_info(f"✅ Passed: {passed_tests}")
    tprint_info(f"❌ Failed: {failed_tests}")
    tprint_info(f"📈 Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests > 0:
        tprint_warning(f"⚠️ {failed_tests} tests failed - check logs for details")
        return False
    else:
        tprint_success("🎉 All tests passed!")
        return True


if __name__ == "__main__":
    # Run all tests
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)