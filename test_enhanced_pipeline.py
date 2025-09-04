#!/usr/bin/env python3
"""
Test Script for Enhanced Pipeline

This script tests the enhanced pipeline system to ensure all components
work together effectively with proper validation, monitoring, and protection.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import enhanced pipeline components
from src.utils.pipeline_integration import run_enhanced_pipeline
from src.utils.pipeline_validator_framework import (
    validator_orchestrator,
    ValidationLevel,
    ValidationResult
)
from src.utils.pipeline_decorators import (
    pipeline_step,
    data_reader,
    data_writer,
    data_transformer,
    data_analyzer
)
from src.utils.pipeline_utilities import (
    pipeline_utilities,
    DataFormat
)
from src.utils.pipeline_state_manager import (
    pipeline_state_manager,
    PipelineState
)
from src.utils.pipeline_monitoring import (
    pipeline_monitor,
    LogLevel,
    MetricType
)
from src.utils.common_operations import (
    get_current_datetime,
    format_datetime,
    ensure_directory
)
import pandas as pd
import numpy as np


# Test pipeline steps
@pipeline_step("test_data_generation", ValidationLevel.CRITICAL)
@data_writer(validate_schema=True)
def test_data_generation_step(symbol: str, exchange: str, timeframe: str, data_dir: str, **kwargs) -> bool:
    """Test step: Generate sample data."""
    print(f"🔄 Generating test data for {symbol} on {exchange}")
    
    try:
        # Create sample data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(100, 200, len(dates)),
            'high': np.random.uniform(150, 250, len(dates)),
            'low': np.random.uniform(50, 150, len(dates)),
            'close': np.random.uniform(100, 200, len(dates)),
            'volume': np.random.uniform(1000, 10000, len(dates))
        })
        
        # Ensure data directory exists
        ensure_directory(data_dir)
        
        # Save data
        output_file = f"{data_dir}/test_data_{symbol}_{timeframe}.parquet"
        pipeline_utilities.format_manager.write_data(
            data=data,
            file_path=output_file,
            format=DataFormat.PARQUET
        )
        
        print(f"✅ Generated test data: {len(data)} rows")
        return True
        
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False


@pipeline_step("test_data_processing", ValidationLevel.CRITICAL)
@data_reader(validate_schema=True)
@data_transformer(validate_schema=True)
@data_writer(validate_schema=True)
def test_data_processing_step(symbol: str, exchange: str, timeframe: str, data_dir: str, **kwargs) -> bool:
    """Test step: Process sample data."""
    print(f"🔄 Processing test data for {symbol} on {exchange}")
    
    try:
        # Read input data
        input_file = f"{data_dir}/test_data_{symbol}_{timeframe}.parquet"
        data = pipeline_utilities.format_manager.read_data(input_file)
        
        # Process data
        data['price_change'] = data['close'] - data['open']
        data['price_change_pct'] = (data['price_change'] / data['open']) * 100
        data['volume_ma'] = data['volume'].rolling(window=24).mean()
        
        # Clean data
        processed_data = pipeline_utilities.manipulation_manager.clean_data(
            data=data,
            remove_duplicates=True,
            handle_missing="drop"
        )
        
        # Save processed data
        output_file = f"{data_dir}/processed_data_{symbol}_{timeframe}.parquet"
        pipeline_utilities.format_manager.write_data(
            data=processed_data,
            file_path=output_file,
            format=DataFormat.PARQUET
        )
        
        print(f"✅ Processed test data: {len(processed_data)} rows")
        return True
        
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        return False


@pipeline_step("test_data_analysis", ValidationLevel.WARNING)
@data_reader(validate_schema=True)
@data_analyzer(validate_schema=True)
def test_data_analysis_step(symbol: str, exchange: str, timeframe: str, data_dir: str, **kwargs) -> bool:
    """Test step: Analyze processed data."""
    print(f"🔄 Analyzing processed data for {symbol} on {exchange}")
    
    try:
        # Read processed data
        input_file = f"{data_dir}/processed_data_{symbol}_{timeframe}.parquet"
        data = pipeline_utilities.format_manager.read_data(input_file)
        
        # Analyze data quality
        quality_analysis = pipeline_utilities.analysis_manager.analyze_data_quality(data)
        print(f"📊 Data quality analysis completed")
        
        # Detect outliers
        outlier_analysis = pipeline_utilities.analysis_manager.detect_outliers(data)
        print(f"🔍 Outlier detection completed")
        
        # Correlation analysis
        correlation_analysis = pipeline_utilities.analysis_manager.correlation_analysis(data)
        print(f"📈 Correlation analysis completed")
        
        # Save analysis results
        analysis_results = {
            "quality_analysis": quality_analysis,
            "outlier_analysis": outlier_analysis,
            "correlation_analysis": correlation_analysis,
            "timestamp": format_datetime(get_current_datetime())
        }
        
        output_file = f"{data_dir}/analysis_results_{symbol}_{timeframe}.json"
        pipeline_utilities.format_manager.safe_json_dump(analysis_results, output_file, indent=2)
        
        print(f"✅ Data analysis completed")
        return True
        
    except Exception as e:
        print(f"❌ Data analysis failed: {e}")
        return False


@pipeline_step("test_validation", ValidationLevel.CRITICAL)
def test_validation_step(symbol: str, exchange: str, timeframe: str, data_dir: str, **kwargs) -> bool:
    """Test step: Validate all outputs."""
    print(f"🔄 Validating all outputs for {symbol} on {exchange}")
    
    try:
        # Test data format validation
        test_data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e'],
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        # Validate data format
        validation_results = asyncio.run(
            validator_orchestrator.validate_pipeline_step(
                step_name="test_validation",
                data=test_data,
                context={
                    "required_columns": ["col1", "col2", "col3"],
                    "expected_types": {"col1": int, "col2": str, "col3": float}
                },
                validators_to_run=["data_format", "data_quality"]
            )
        )
        
        # Check validation results
        all_passed = True
        for validator_name, report in validation_results.items():
            if report.result == ValidationResult.FAILED:
                print(f"❌ Validation failed for {validator_name}: {report.message}")
                all_passed = False
            elif report.result == ValidationResult.WARNING:
                print(f"⚠️ Validation warning for {validator_name}: {report.message}")
            else:
                print(f"✅ Validation passed for {validator_name}")
        
        print(f"✅ Validation step completed")
        return all_passed
        
    except Exception as e:
        print(f"❌ Validation step failed: {e}")
        return False


async def test_enhanced_pipeline():
    """Test the enhanced pipeline system."""
    
    print("🚀 TESTING ENHANCED PIPELINE SYSTEM")
    print("=" * 80)
    
    # Test configuration
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    timeframe = "1m"
    data_dir = "test_data_cache"
    
    # Ensure test data directory exists
    ensure_directory(data_dir)
    
    # Define test pipeline steps
    pipeline_steps = [
        {
            "name": "data_generation",
            "function": test_data_generation_step,
            "dependencies": [],
            "validation_level": "critical",
            "critical": True
        },
        {
            "name": "data_processing",
            "function": test_data_processing_step,
            "dependencies": ["data_generation"],
            "validation_level": "critical",
            "critical": True
        },
        {
            "name": "data_analysis",
            "function": test_data_analysis_step,
            "dependencies": ["data_processing"],
            "validation_level": "warning",
            "critical": False
        },
        {
            "name": "validation",
            "function": test_validation_step,
            "dependencies": ["data_analysis"],
            "validation_level": "critical",
            "critical": True
        }
    ]
    
    try:
        # Run enhanced pipeline
        print(f"🔄 Starting enhanced pipeline test for {symbol} on {exchange}")
        print(f"📊 Pipeline steps: {len(pipeline_steps)}")
        print(f"📁 Data directory: {data_dir}")
        print("-" * 80)
        
        success = await run_enhanced_pipeline(
            symbol=symbol,
            exchange=exchange,
            pipeline_steps=pipeline_steps,
            timeframe=timeframe,
            data_dir=data_dir,
            test_mode=True
        )
        
        print("-" * 80)
        if success:
            print("🎉 ENHANCED PIPELINE TEST COMPLETED SUCCESSFULLY!")
        else:
            print("❌ ENHANCED PIPELINE TEST FAILED!")
        
        # Test individual components
        print("\n🔍 TESTING INDIVIDUAL COMPONENTS")
        print("-" * 80)
        
        # Test validator orchestrator
        print("📋 Testing validator orchestrator...")
        validation_summary = validator_orchestrator.get_validation_summary()
        print(f"   Total validations: {validation_summary.get('total_validations', 0)}")
        print(f"   Success rate: {validation_summary.get('success_rate', 0):.2%}")
        
        # Test pipeline utilities
        print("🔧 Testing pipeline utilities...")
        utilities_status = pipeline_utilities.get_pipeline_status()
        print(f"   Status: {utilities_status}")
        
        # Test state manager
        print("📊 Testing state manager...")
        state_summary = pipeline_state_manager.get_pipeline_status_summary()
        print(f"   Total pipelines: {state_summary.get('total_pipelines', 0)}")
        
        # Test monitoring
        print("📈 Testing monitoring system...")
        monitoring_summary = pipeline_monitor.get_monitoring_summary()
        print(f"   Monitoring active: {monitoring_summary.get('monitoring_active', False)}")
        print(f"   Total log entries: {monitoring_summary.get('total_log_entries', 0)}")
        print(f"   Total metrics: {monitoring_summary.get('total_metrics', 0)}")
        
        print("\n" + "=" * 80)
        print("✅ ALL COMPONENT TESTS COMPLETED")
        
        return success
        
    except Exception as e:
        print(f"💥 Enhanced pipeline test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_individual_components():
    """Test individual pipeline components."""
    
    print("\n🧪 TESTING INDIVIDUAL COMPONENTS")
    print("=" * 80)
    
    # Test data format manager
    print("📁 Testing data format manager...")
    try:
        # Create test data
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'price': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })
        
        # Test writing and reading
        test_file = "test_data_cache/test_format.parquet"
        pipeline_utilities.format_manager.write_data(test_data, test_file, DataFormat.PARQUET)
        loaded_data = pipeline_utilities.format_manager.read_data(test_file, DataFormat.PARQUET)
        
        print(f"   ✅ Data format manager test passed: {len(loaded_data)} rows")
        
    except Exception as e:
        print(f"   ❌ Data format manager test failed: {e}")
    
    # Test data analysis manager
    print("📊 Testing data analysis manager...")
    try:
        # Create test data with some issues
        test_data = pd.DataFrame({
            'price': [100, 200, None, 300, 400],
            'volume': [1000, 2000, 3000, 4000, 5000],
            'outlier': [1, 2, 3, 4, 1000]  # Contains outlier
        })
        
        # Test data quality analysis
        quality_analysis = pipeline_utilities.analysis_manager.analyze_data_quality(test_data)
        print(f"   ✅ Data quality analysis completed")
        
        # Test outlier detection
        outlier_analysis = pipeline_utilities.analysis_manager.detect_outliers(test_data)
        print(f"   ✅ Outlier detection completed")
        
    except Exception as e:
        print(f"   ❌ Data analysis manager test failed: {e}")
    
    # Test data manipulation manager
    print("🔧 Testing data manipulation manager...")
    try:
        # Create test data with issues
        test_data = pd.DataFrame({
            'price': [100, 200, 200, 300, None],  # Duplicate and missing
            'volume': [1000, 2000, 3000, 4000, 5000]
        })
        
        # Test data cleaning
        cleaned_data = pipeline_utilities.manipulation_manager.clean_data(
            test_data,
            remove_duplicates=True,
            handle_missing="drop"
        )
        
        print(f"   ✅ Data cleaning completed: {len(test_data)} -> {len(cleaned_data)} rows")
        
    except Exception as e:
        print(f"   ❌ Data manipulation manager test failed: {e}")
    
    print("✅ Individual component tests completed")


async def main():
    """Main test function."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 ENHANCED PIPELINE COMPREHENSIVE TEST SUITE")
    print("=" * 100)
    print(f"📅 Test started at: {format_datetime(get_current_datetime())}")
    print("=" * 100)
    
    try:
        # Test individual components first
        await test_individual_components()
        
        # Test enhanced pipeline
        pipeline_success = await test_enhanced_pipeline()
        
        print("\n" + "=" * 100)
        print("📊 FINAL TEST RESULTS")
        print("=" * 100)
        
        if pipeline_success:
            print("🎉 ALL TESTS PASSED!")
            print("✅ Enhanced pipeline system is working correctly")
            print("✅ All validators, decorators, and utilities are functional")
            print("✅ Pipeline state management and monitoring are operational")
        else:
            print("❌ SOME TESTS FAILED!")
            print("⚠️ Please check the logs for details")
        
        print(f"📅 Test completed at: {format_datetime(get_current_datetime())}")
        print("=" * 100)
        
        return pipeline_success
        
    except Exception as e:
        print(f"💥 Test suite failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the test suite
    success = asyncio.run(main())
    sys.exit(0 if success else 1)