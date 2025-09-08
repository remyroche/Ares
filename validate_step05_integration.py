#!/usr/bin/env python3
"""
Validation script for Step05 utility integration.

This script performs a comprehensive validation of the Step05 utility integration
to ensure all utilities are properly connected and functioning.
"""

import asyncio
import sys
import logging
from pathlib import Path
import tempfile
import shutil
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.training.steps.step05_optimized_integrated import Step05OptimizedIntegrated
from src.training.steps.step05_dependency_injection import (
    Step05DependencyContainer, 
    UtilityConfig, 
    initialize_step05_utilities
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_data(temp_dir: Path) -> None:
    """Create test data for validation."""
    logger.info("📊 Creating test data...")
    
    # Create data directory structure
    data_dir = temp_dir / "data" / "processed_data" / "binance" / "BTCUSDT"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample triple barrier data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
        'open': np.random.uniform(100, 200, n_samples),
        'high': np.random.uniform(100, 200, n_samples),
        'low': np.random.uniform(100, 200, n_samples),
        'close': np.random.uniform(100, 200, n_samples),
        'volume': np.random.uniform(1000, 10000, n_samples),
        'hmm_regime': np.random.choice([0, 1, 2], n_samples),
        'returns': np.random.normal(0, 0.01, n_samples),
        'volatility': np.random.uniform(0.01, 0.05, n_samples),
        'upper_barrier': np.random.uniform(0.001, 0.01, n_samples),
        'lower_barrier': np.random.uniform(-0.01, -0.001, n_samples),
        'time_barrier': np.random.randint(10, 60, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Save as parquet
    output_file = data_dir / "BTCUSDT_binance_1h_triple_barrier.parquet"
    df.to_parquet(output_file)
    
    logger.info(f"✅ Created test data: {output_file}")
    logger.info(f"   Shape: {df.shape}")
    logger.info(f"   Columns: {list(df.columns)}")
    
    return str(data_dir.parent.parent.parent)  # Return base data directory


def test_dependency_injection():
    """Test dependency injection container."""
    logger.info("🔧 Testing dependency injection container...")
    
    try:
        # Test utility config
        config = UtilityConfig(
            enable_gpu_optimization=True,
            enable_memory_optimization=True,
            enable_cpu_optimization=True,
            enable_math_validation=True,
            enable_data_validation=True,
            enable_serialization=True,
            memory_limit_gb=4.0,
            max_workers=2,
            gpu_memory_threshold=0.7,
            log_level='INFO'
        )
        
        # Initialize container
        container = initialize_step05_utilities(config)
        logger.info("✅ Dependency injection container initialized")
        
        # Test health check
        health_status = container.health_check()
        logger.info(f"✅ Health check: {health_status['overall_health']}")
        logger.info(f"   Total utilities: {health_status['total_utilities']}")
        logger.info(f"   Healthy utilities: {health_status['healthy_utilities']}")
        
        # Test utility summary
        summary = container.get_utility_summary()
        logger.info(f"✅ Utility summary: {len(summary)} categories")
        
        for category, info in summary.items():
            logger.info(f"   • {category}: {info['total_utilities']} utilities")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dependency injection test failed: {e}")
        return False


def test_utility_categories():
    """Test individual utility categories."""
    logger.info("🧪 Testing utility categories...")
    
    try:
        config = UtilityConfig()
        container = initialize_step05_utilities(config)
        
        categories = [
            'common_operations', 'common_utilities', 'math_validation',
            'parquet_utils', 'serialization_utils', 'data_processing_utils',
            'm1_gpu_utils', 'm1_memory_utils', 'm1_cpu_utils'
        ]
        
        for category in categories:
            if container.has_category(category):
                category_utils = container.get_category(category)
                logger.info(f"✅ {category}: {len(category_utils)} utilities available")
            else:
                logger.error(f"❌ {category}: Not found")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Utility categories test failed: {e}")
        return False


def test_utility_functions():
    """Test specific utility functions."""
    logger.info("⚙️ Testing utility functions...")
    
    try:
        config = UtilityConfig()
        container = initialize_step05_utilities(config)
        
        # Test common operations
        common_ops = container.get_category('common_operations')
        current_time = common_ops['datetime_ops']['get_current_datetime']()
        formatted_time = common_ops['datetime_ops']['format_datetime'](current_time)
        logger.info(f"✅ Datetime operations: {formatted_time}")
        
        # Test string operations
        test_string = "Hello World"
        lower_string = common_ops['string_ops']['safe_lower'](test_string)
        upper_string = common_ops['string_ops']['safe_upper'](test_string)
        logger.info(f"✅ String operations: '{lower_string}' -> '{upper_string}'")
        
        # Test math operations
        safe_float = common_ops['math_ops']['safe_float']("123.45", 0.0)
        safe_int = common_ops['math_ops']['safe_int']("123", 0)
        logger.info(f"✅ Math operations: {safe_float}, {safe_int}")
        
        # Test math validation
        math_validation = container.get_category('math_validation')
        safe_divide = math_validation['safe_math_ops']['safe_divide'](10, 2, 0.0)
        logger.info(f"✅ Math validation: 10/2 = {safe_divide}")
        
        # Test validation
        math_validation['validation_ops']['validate_positive'](5.0, "test_value")
        logger.info("✅ Validation operations: Positive validation passed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Utility functions test failed: {e}")
        return False


async def test_step05_integration(data_dir: str):
    """Test Step05 integration with utilities."""
    logger.info("🚀 Testing Step05 integration...")
    
    try:
        # Create configuration
        config = {
            'SYMBOL': 'BTCUSDT',
            'EXCHANGE': 'binance',
            'TIMEFRAME': '1h',
            'DATA_DIR': data_dir,
            'vectorized_labelling_orchestrator': {
                'auto_recalculate_hmm_barriers': True,
                'hmm_barrier_regime_column': 'hmm_regime',
                'time_barrier_minutes': 30,
                'max_lookahead': 100,
                'profit_take_multiplier': 0.002,
                'stop_loss_multiplier': 0.001
            },
            'transaction_costs': {
                'maker_fee': 0.001,
                'taker_fee': 0.001,
                'slippage_bps': 2.0,
                'funding_rate': 0.0001
            },
            'memory': {
                'thresholds': {
                    'warning_mb': 1000.0,
                    'critical_mb': 2000.0,
                    'max_memory_mb': 4000.0
                },
                'optimization_strategies': {
                    'dtype_optimization': True,
                    'categorical_optimization': True,
                    'sparse_optimization': True,
                    'chunk_processing': True,
                    'garbage_collection': True
                }
            },
            'streaming': {
                'chunk_size': 10000,
                'max_memory_mb': 1000.0,
                'overlap_rows': 100,
                'enable_compression': True,
                'enable_parallel_processing': False,
                'max_workers': 4,
                'progress_reporting_interval': 10
            },
            # Utility integration configuration
            'enable_gpu_optimization': True,
            'enable_memory_optimization': True,
            'enable_cpu_optimization': True,
            'enable_math_validation': True,
            'enable_data_validation': True,
            'enable_serialization': True,
            'memory_limit_gb': 4.0,
            'max_workers': 2,
            'gpu_memory_threshold': 0.7,
            'log_level': 'INFO'
        }
        
        # Initialize Step05
        step = Step05OptimizedIntegrated(config)
        logger.info("✅ Step05 initialized with utility integration")
        
        # Test initialization
        await step.initialize()
        logger.info("✅ Step05 async initialization completed")
        
        # Test utility references
        assert hasattr(step, 'utils'), "Utils container not found"
        assert hasattr(step, 'common_ops'), "Common operations not found"
        assert hasattr(step, 'math_validation'), "Math validation not found"
        assert hasattr(step, 'parquet_utils'), "Parquet utils not found"
        assert hasattr(step, 'serialization_utils'), "Serialization utils not found"
        assert hasattr(step, 'data_processing_utils'), "Data processing utils not found"
        assert hasattr(step, 'm1_gpu_utils'), "M1 GPU utils not found"
        assert hasattr(step, 'm1_memory_utils'), "M1 memory utils not found"
        assert hasattr(step, 'm1_cpu_utils'), "M1 CPU utils not found"
        logger.info("✅ All utility references properly set")
        
        # Test performance metrics
        assert 'gpu_operations' in step.performance_metrics
        assert 'cpu_parallel_operations' in step.performance_metrics
        assert 'math_validation_operations' in step.performance_metrics
        assert 'data_processing_operations' in step.performance_metrics
        assert 'serialization_operations' in step.performance_metrics
        logger.info("✅ Performance metrics include utility tracking")
        
        # Test utility health
        health_status = step.utils.health_check()
        logger.info(f"✅ Utility health check: {health_status['overall_health']}")
        
        # Test performance summary
        performance_summary = step.get_performance_summary()
        assert 'utility_integration' in performance_summary
        assert 'm1_optimization' in performance_summary
        assert 'utility_usage_metrics' in performance_summary
        logger.info("✅ Performance summary includes utility metrics")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Step05 integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main validation function."""
    logger.info("🎯 Starting Step05 Utility Integration Validation")
    logger.info("=" * 60)
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            # Create test data
            data_dir = create_test_data(temp_path)
            
            # Run tests
            tests = [
                ("Dependency Injection", test_dependency_injection),
                ("Utility Categories", test_utility_categories),
                ("Utility Functions", test_utility_functions),
                ("Step05 Integration", lambda: test_step05_integration(data_dir))
            ]
            
            results = []
            for test_name, test_func in tests:
                logger.info(f"\n🧪 Running {test_name} test...")
                if asyncio.iscoroutinefunction(test_func):
                    result = await test_func()
                else:
                    result = test_func()
                results.append((test_name, result))
                logger.info(f"{'✅' if result else '❌'} {test_name}: {'PASSED' if result else 'FAILED'}")
            
            # Summary
            logger.info("\n" + "=" * 60)
            logger.info("📊 VALIDATION SUMMARY")
            logger.info("=" * 60)
            
            passed = sum(1 for _, result in results if result)
            total = len(results)
            
            for test_name, result in results:
                status = "✅ PASSED" if result else "❌ FAILED"
                logger.info(f"{status} {test_name}")
            
            logger.info(f"\n🎯 Overall Result: {passed}/{total} tests passed")
            
            if passed == total:
                logger.info("🎉 ALL TESTS PASSED! Step05 utility integration is working correctly.")
                return True
            else:
                logger.error(f"⚠️ {total - passed} tests failed. Please check the issues above.")
                return False
                
        except Exception as e:
            logger.error(f"❌ Validation failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == '__main__':
    success = asyncio.run(main())
    sys.exit(0 if success else 1)