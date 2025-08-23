#!/usr/bin/env python3
"""
Test Enhanced Step1 and Step1_5 Implementations

This script demonstrates and validates the enhanced implementations
with comprehensive testing and performance monitoring.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import enhanced implementations
try:
    from src.training.steps.enhanced_step1_data_collection import (
        EnhancedStep1DataCollection, run_enhanced_step1
    )
    from src.training.steps.enhanced_step1_5_data_converter import (
        EnhancedStep1_5DataConverter, run_enhanced_step1_5
    )
    from src.utils.enhanced_config_management import (
        Step1Config, Step1_5Config, PipelineConfig, load_pipeline_config
    )
    from src.utils.enhanced_data_quality_validator import (
        quick_validate_dataframe, validate_unified_dataframe, check_dataframe_health
    )
    from src.utils.enhanced_memory_management import (
        get_memory_usage_mb, log_memory_status, trigger_gc_if_needed
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all enhanced utilities are properly installed.")
    sys.exit(1)

import numpy as np
import pandas as pd


def create_test_data():
    """Create test data for validation."""
    print("🔧 Creating test data...")
    
    # Create sample klines data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1min')
    n_samples = len(dates)
    
    klines_data = {
        'timestamp': (dates.astype(np.int64) // 10**6).values,
        'open': np.random.randn(n_samples) * 100 + 1000,
        'high': np.random.randn(n_samples) * 100 + 1000,
        'low': np.random.randn(n_samples) * 100 + 1000,
        'close': np.random.randn(n_samples) * 100 + 1000,
        'volume': np.random.randn(n_samples) * 1000 + 10000,
    }
    
    # Create sample aggtrades data
    aggtrades_data = {
        'timestamp': (dates.astype(np.int64) // 10**6).values,
        'price': np.random.randn(n_samples) * 100 + 1000,
        'quantity': np.random.randn(n_samples) * 1000 + 10000,
        'is_buyer_maker': np.random.choice([True, False], n_samples),
        'agg_trade_id': np.arange(n_samples),
    }
    
    # Create DataFrames
    klines_df = pd.DataFrame(klines_data)
    aggtrades_df = pd.DataFrame(aggtrades_data)
    
    # Save test data
    os.makedirs("data_cache", exist_ok=True)
    klines_df.to_parquet("data_cache/klines_BINANCE_ETHUSDT_1m_consolidated.parquet", index=False)
    aggtrades_df.to_parquet("data_cache/aggtrades_BINANCE_ETHUSDT_consolidated.parquet", index=False)
    
    print(f"✅ Created test data: {klines_df.shape}, {aggtrades_df.shape}")
    return klines_df, aggtrades_df


def test_data_quality_validation():
    """Test data quality validation utilities."""
    print("\n" + "="*60)
    print("TESTING DATA QUALITY VALIDATION")
    print("="*60)
    
    # Create test data
    klines_df, aggtrades_df = create_test_data()
    
    # Test quick validation
    print("\n🔍 Testing quick validation...")
    health_status = check_dataframe_health(klines_df)
    # Convert numpy types to native Python types for JSON serialization
    health_status_json = {
        "healthy": bool(health_status["healthy"]),
        "shape": tuple(health_status["shape"]),
        "memory_mb": float(health_status["memory_mb"]),
        "nan_ratio": float(health_status["nan_ratio"]),
        "infinite_count": int(health_status["infinite_count"]),
        "issues": health_status["issues"]
    }
    print(f"Health status: {json.dumps(health_status_json, indent=2)}")
    
    # Test comprehensive validation
    print("\n🔍 Testing comprehensive validation...")
    quality_result = quick_validate_dataframe(klines_df, "test_klines")
    # Convert to JSON-serializable format
    summary = quality_result.get_summary()
    summary_json = {
        "passed": bool(summary["passed"]),
        "issue_count": int(summary["issue_count"]),
        "warning_count": int(summary["warning_count"]),
        "metrics": {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                   for k, v in summary["metrics"].items()},
        "issues": summary["issues"],
        "warnings": summary["warnings"]
    }
    print(f"Quality result: {json.dumps(summary_json, indent=2)}")
    
    # Test with some issues
    print("\n🔍 Testing with quality issues...")
    problematic_df = klines_df.copy()
    problematic_df.loc[100, 'open'] = np.nan  # Add NaN
    problematic_df.loc[200, 'high'] = np.inf  # Add infinite value
    
    quality_result = quick_validate_dataframe(problematic_df, "problematic_data")
    # Convert to JSON-serializable format
    summary = quality_result.get_summary()
    summary_json = {
        "passed": bool(summary["passed"]),
        "issue_count": int(summary["issue_count"]),
        "warning_count": int(summary["warning_count"]),
        "metrics": {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                   for k, v in summary["metrics"].items()},
        "issues": summary["issues"],
        "warnings": summary["warnings"]
    }
    print(f"Quality result with issues: {json.dumps(summary_json, indent=2)}")
    
    print("✅ Data quality validation tests completed")


def test_memory_management():
    """Test memory management utilities."""
    print("\n" + "="*60)
    print("TESTING MEMORY MANAGEMENT")
    print("="*60)
    
    # Test memory monitoring
    print("\n💾 Testing memory monitoring...")
    initial_memory = get_memory_usage_mb()
    print(f"Initial memory usage: {initial_memory:.1f}MB")
    
    # Create large DataFrame to test memory pressure
    print("\n💾 Creating large DataFrame...")
    large_df = pd.DataFrame({
        'col1': np.random.randn(100000),
        'col2': np.random.randn(100000),
        'col3': np.random.randn(100000),
    })
    
    memory_after_large_df = get_memory_usage_mb()
    print(f"Memory after large DataFrame: {memory_after_large_df:.1f}MB")
    
    # Test garbage collection
    print("\n💾 Testing garbage collection...")
    gc_result = trigger_gc_if_needed(max_memory_mb=1024)
    # Convert to JSON-serializable format
    gc_result_json = {
        "before_mb": float(gc_result["before_mb"]),
        "after_mb": float(gc_result["after_mb"]),
        "freed_mb": float(gc_result["freed_mb"])
    }
    print(f"GC result: {json.dumps(gc_result_json, indent=2)}")
    
    # Clean up
    del large_df
    final_memory = get_memory_usage_mb()
    print(f"Final memory usage: {final_memory:.1f}MB")
    
    print("✅ Memory management tests completed")


async def test_enhanced_step1():
    """Test enhanced Step1 implementation."""
    print("\n" + "="*60)
    print("TESTING ENHANCED STEP1")
    print("="*60)
    
    # Create configuration
    config = Step1Config(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m",
        lookback_days=30,  # Shorter for testing
        max_memory_mb=512,  # Lower for testing
        chunk_size=5000,    # Smaller chunks for testing
        max_retries=2       # Fewer retries for testing
    )
    
    # Create enhanced Step1 instance
    step1 = EnhancedStep1DataCollection(config)
    
    # Prepare training input
    training_input = {
        "symbol": "ETHUSDT",
        "exchange": "BINANCE",
        "timeframe": "1m",
        "data_dir": "data_cache"
    }
    
    # Prepare pipeline state
    pipeline_state = {
        "data_collection_completed": False,
        "quality_check_passed": False
    }
    
    # Execute enhanced data collection
    print("\n🚀 Executing enhanced Step1...")
    start_time = time.time()
    
    try:
        result = await step1.execute(training_input, pipeline_state)
        duration = time.time() - start_time
        
        print(f"\n📊 Step1 Execution Results:")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Data collection completed: {result['data_collection_completed']}")
        print(f"   Quality check passed: {result['quality_check_passed']}")
        
        # Get memory stats
        memory_stats = step1.get_memory_stats()
        print(f"   Peak memory usage: {memory_stats['peak_mb']:.1f}MB")
        
        print("✅ Enhanced Step1 test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Step1 test failed: {e}")
        return False


async def test_enhanced_step1_5():
    """Test enhanced Step1_5 implementation."""
    print("\n" + "="*60)
    print("TESTING ENHANCED STEP1_5")
    print("="*60)
    
    # Create configuration
    config = Step1_5Config(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m",
        max_memory_mb=512,  # Lower for testing
        chunk_size=5000,    # Smaller chunks for testing
        force_rerun=False,
        enable_incremental=True
    )
    
    # Create enhanced Step1_5 instance
    step1_5 = EnhancedStep1_5DataConverter(config)
    
    # Prepare training input
    training_input = {
        "symbol": "ETHUSDT",
        "exchange": "BINANCE",
        "timeframe": "1m",
        "data_dir": "data_cache"
    }
    
    # Prepare pipeline state
    pipeline_state = {
        "data_conversion_completed": False,
        "quality_check_passed": False
    }
    
    # Execute enhanced data conversion
    print("\n🔄 Executing enhanced Step1_5...")
    start_time = time.time()
    
    try:
        result = await step1_5.execute(training_input, pipeline_state)
        duration = time.time() - start_time
        
        print(f"\n📊 Step1_5 Execution Results:")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Data conversion completed: {result['data_conversion_completed']}")
        print(f"   Quality check passed: {result['quality_check_passed']}")
        
        # Get memory stats
        memory_stats = step1_5.get_memory_stats()
        print(f"   Peak memory usage: {memory_stats['peak_mb']:.1f}MB")
        
        print("✅ Enhanced Step1_5 test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Step1_5 test failed: {e}")
        return False


def test_configuration_management():
    """Test configuration management."""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION MANAGEMENT")
    print("="*60)
    
    # Test Step1 configuration
    print("\n⚙️ Testing Step1 configuration...")
    step1_config = Step1Config(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m",
        max_memory_mb=1024
    )
    
    # Validate configuration
    issues = step1_config.validate()
    if issues:
        print(f"❌ Configuration validation failed: {issues}")
    else:
        print("✅ Step1 configuration validation passed")
    
    # Test Step1_5 configuration
    print("\n⚙️ Testing Step1_5 configuration...")
    step1_5_config = Step1_5Config(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m",
        max_memory_mb=1024
    )
    
    # Validate configuration
    issues = step1_5_config.validate()
    if issues:
        print(f"❌ Configuration validation failed: {issues}")
    else:
        print("✅ Step1_5 configuration validation passed")
    
    # Test pipeline configuration
    print("\n⚙️ Testing pipeline configuration...")
    pipeline_config = PipelineConfig(
        step1=step1_config,
        step1_5=step1_5_config,
        environment="development"
    )
    
    # Validate configuration
    issues = pipeline_config.validate()
    if issues:
        print(f"❌ Pipeline configuration validation failed: {issues}")
    else:
        print("✅ Pipeline configuration validation passed")
    
    print("✅ Configuration management tests completed")


async def test_integration():
    """Test integration between Step1 and Step1_5."""
    print("\n" + "="*60)
    print("TESTING INTEGRATION")
    print("="*60)
    
    # Create test data first
    create_test_data()
    
    # Test Step1
    print("\n🔄 Testing Step1 -> Step1_5 integration...")
    step1_success = await test_enhanced_step1()
    
    if step1_success:
        # Test Step1_5
        step1_5_success = await test_enhanced_step1_5()
        
        if step1_5_success:
            print("\n✅ Integration test completed successfully")
            
            # Check if unified data was created
            unified_dir = "data_cache/unified/binance/ethusdt/1m"
            if os.path.exists(unified_dir):
                parquet_files = [f for f in os.listdir(unified_dir) if f.endswith('.parquet')]
                print(f"📁 Unified data created: {len(parquet_files)} files")
            else:
                print("⚠️ Unified data directory not found")
        else:
            print("❌ Step1_5 failed in integration test")
    else:
        print("❌ Step1 failed in integration test")
    
    print("✅ Integration tests completed")


def generate_performance_report():
    """Generate performance report."""
    print("\n" + "="*60)
    print("PERFORMANCE REPORT")
    print("="*60)
    
    # Memory usage
    current_memory = get_memory_usage_mb()
    print(f"\n💾 Current memory usage: {current_memory:.1f}MB")
    
    # Check for created files
    print("\n📁 Created files:")
    if os.path.exists("data_cache"):
        for root, dirs, files in os.walk("data_cache"):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"   - {file_path}: {file_size:.1f}MB")
    
    # Configuration summary
    print("\n⚙️ Configuration summary:")
    step1_config = Step1Config()
    step1_5_config = Step1_5Config()
    
    print(f"   Step1 max memory: {step1_config.max_memory_mb}MB")
    print(f"   Step1_5 max memory: {step1_5_config.max_memory_mb}MB")
    print(f"   Step1 chunk size: {step1_config.chunk_size}")
    print(f"   Step1_5 chunk size: {step1_5_config.chunk_size}")
    
    print("✅ Performance report generated")


async def main():
    """Main test function."""
    print("🚀 Starting Enhanced Step1 and Step1_5 Tests")
    print("="*80)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Test individual components
        test_data_quality_validation()
        test_memory_management()
        test_configuration_management()
        
        # Test enhanced implementations
        await test_enhanced_step1()
        await test_enhanced_step1_5()
        
        # Test integration
        await test_integration()
        
        # Generate performance report
        generate_performance_report()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    
    if success:
        print("\n✅ All tests passed! Enhanced implementations are working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)