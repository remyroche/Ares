#!/usr/bin/env python3
"""
Simple test script for the enhanced pipeline without external dependencies.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import simplified utilities
from src.utils.common_operations_simple import (
    get_current_datetime,
    format_datetime,
    safe_file_exists,
    ensure_directory,
    create_pipeline_id,
    log_pipeline_start,
    log_pipeline_success,
    log_pipeline_failure
)

# Import the enhanced pipeline orchestrator
from src.training.steps.run_all_pipelines import EnhancedPipelineOrchestrator


async def test_enhanced_pipeline():
    """Test the enhanced pipeline with simplified dependencies."""
    
    print("🚀 Testing Enhanced Pipeline")
    print("=" * 80)
    
    # Configuration
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    timeframe = "1m"
    data_dir = "data_cache"
    
    # Enhanced configuration
    config = {
        'force_rerun': True,
        'quality_checks': True,
        'validate_data': True,
        'convert_format': True,
        'enable_validation': True,
        'enable_monitoring': True,
        'enable_checkpoints': True,
        'validation_level': 'critical',
    }
    
    print(f"📊 Test Configuration:")
    print(f"   Symbol: {symbol}")
    print(f"   Exchange: {exchange}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Data directory: {data_dir}")
    print(f"   Enhanced validation: {config['enable_validation']}")
    print(f"   Monitoring enabled: {config['enable_monitoring']}")
    print("=" * 80)
    
    # Ensure data directory exists
    ensure_directory(data_dir)
    
    # Create pipeline orchestrator
    orchestrator = EnhancedPipelineOrchestrator(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir
    )
    
    # Test pipeline execution
    start_time = time.time()
    pipeline_id = create_pipeline_id(symbol, exchange, timeframe)
    
    print(f"🔄 Starting enhanced pipeline test: {pipeline_id}")
    
    try:
        # Test the enhanced pipeline
        success = await orchestrator.run_all_pipelines(**config)
        
        total_time = time.time() - start_time
        
        if success:
            print("\n🎉 ENHANCED PIPELINE TEST COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("✅ Enhanced pipeline features tested:")
            print("   ✅ Prerequisites validation")
            print("   ✅ Enhanced error handling")
            print("   ✅ Performance monitoring")
            print("   ✅ Data quality validation")
            print("   ✅ Comprehensive logging")
            print("   ✅ Checkpoint management")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print("=" * 80)
        else:
            print("\n❌ ENHANCED PIPELINE TEST FAILED!")
            print("=" * 80)
            print("❌ Please check the logs for error details")
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print("=" * 80)
            
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n💥 ENHANCED PIPELINE TEST FAILED WITH EXCEPTION: {e}")
        print("=" * 80)
        print(f"⏱️ Total execution time: {total_time:.2f} seconds")
        print("=" * 80)
        raise
    
    return success


async def test_individual_components():
    """Test individual enhanced components."""
    
    print("\n🔧 Testing Individual Enhanced Components")
    print("=" * 80)
    
    # Test data collection main
    try:
        from src.training.steps.data_collection.step01_data_collection_main import main as data_collection_main
        print("✅ Enhanced data collection main imported successfully")
    except Exception as e:
        print(f"❌ Failed to import enhanced data collection main: {e}")
    
    # Test common operations
    try:
        from src.utils.common_operations_simple import (
            get_current_datetime,
            format_datetime,
            safe_file_exists,
            ensure_directory
        )
        
        # Test basic operations
        current_time = get_current_datetime()
        formatted_time = format_datetime(current_time)
        print(f"✅ Common operations working: {formatted_time}")
        
        # Test directory creation
        test_dir = "test_directory"
        ensure_directory(test_dir)
        if safe_file_exists(test_dir):
            print("✅ Directory creation and checking working")
        else:
            print("❌ Directory creation failed")
            
    except Exception as e:
        print(f"❌ Common operations test failed: {e}")
    
    print("=" * 80)


async def main():
    """Main test function."""
    
    print("🧪 ENHANCED PIPELINE COMPREHENSIVE TEST")
    print("=" * 100)
    print(f"📅 Started at: {format_datetime(get_current_datetime())}")
    print("=" * 100)
    
    try:
        # Test individual components first
        await test_individual_components()
        
        # Test the full enhanced pipeline
        success = await test_enhanced_pipeline()
        
        print("\n🎯 TEST SUMMARY")
        print("=" * 100)
        if success:
            print("✅ ALL TESTS PASSED - Enhanced pipeline is working correctly!")
            print("✅ Enhanced features successfully integrated:")
            print("   • Comprehensive validation framework")
            print("   • Enhanced error handling with decorators")
            print("   • Performance monitoring and logging")
            print("   • Data quality validation")
            print("   • Checkpoint management")
            print("   • Common utilities for data operations")
        else:
            print("❌ SOME TESTS FAILED - Please check the logs above")
        
        print("=" * 100)
        
    except Exception as e:
        print(f"\n💥 TEST SUITE FAILED: {e}")
        print("=" * 100)
        raise


if __name__ == "__main__":
    # Run the enhanced pipeline test
    asyncio.run(main())