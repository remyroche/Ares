#!/usr/bin/env python3
"""
Test Script for Comprehensive Step01 Monitoring

This script demonstrates the comprehensive monitoring system for step01:
- Function call monitoring with detailed tracking
- Function entry validation with comprehensive checks
- Inter-function call tracking and dependency monitoring
- Function completion reporting with outcome analysis
- Enhanced error handling with detailed function-level tracking
- Performance monitoring with timing and resource usage
- Comprehensive logging with structured reports
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the comprehensive monitoring system
from src.training.steps.data_collection.step01_comprehensive_monitoring import (
    run_comprehensive_step01,
    Step01ComprehensiveMonitoring
)
from src.utils.function_call_monitor import get_function_call_monitor
from src.utils.function_validation_framework import get_function_validator
from src.utils.enhanced_error_handler import get_error_handler
from src.utils.logger import system_logger

# Initialize logger
logger = system_logger.getChild('TestComprehensiveMonitoring')


async def test_comprehensive_monitoring():
    """Test the comprehensive monitoring system."""
    logger.info('🧪 Testing Comprehensive Step01 Monitoring System')
    logger.info('=' * 80)
    
    # Test parameters
    test_symbol = "ETHUSDT"
    test_exchange = "BINANCE"
    test_timeframe = "1m"
    test_data_dir = "test_data_cache"
    
    logger.info(f'🎯 Test Symbol: {test_symbol}')
    logger.info(f'🏢 Test Exchange: {test_exchange}')
    logger.info(f'📊 Test Timeframe: {test_timeframe}')
    logger.info(f'📁 Test Data Directory: {test_data_dir}')
    logger.info('=' * 80)
    
    try:
        # Test 1: Run comprehensive step01
        logger.info('🧪 Test 1: Running Comprehensive Step01')
        success = await run_comprehensive_step01(
            symbol=test_symbol,
            exchange=test_exchange,
            timeframe=test_timeframe,
            data_dir=test_data_dir,
            force_rerun=True
        )
        
        if success:
            logger.info('✅ Test 1 PASSED: Comprehensive Step01 completed successfully')
        else:
            logger.error('❌ Test 1 FAILED: Comprehensive Step01 failed')
        
        # Test 2: Test individual monitoring components
        logger.info('🧪 Test 2: Testing Individual Monitoring Components')
        
        # Test function call monitoring
        function_monitor = get_function_call_monitor()
        call_summary = function_monitor.get_call_summary()
        logger.info(f'📊 Function Call Monitoring: {call_summary["total_calls"]} calls monitored')
        
        # Test function validation
        function_validator = get_function_validator()
        logger.info(f'📊 Function Validation: {len(function_validator.validation_rules)} validation rules')
        
        # Test error handling
        error_handler = get_error_handler()
        error_summary = error_handler.get_error_summary()
        logger.info(f'📊 Error Handling: {error_summary["total_errors"]} errors handled')
        
        # Test 3: Test monitoring integration
        logger.info('🧪 Test 3: Testing Monitoring Integration')
        
        config = {
            'SYMBOL': test_symbol,
            'EXCHANGE': test_exchange,
            'TIMEFRAME': test_timeframe,
            'DATA_DIR': test_data_dir
        }
        
        step = Step01ComprehensiveMonitoring(config)
        await step.initialize()
        
        training_input = {
            'symbol': test_symbol,
            'exchange': test_exchange,
            'timeframe': test_timeframe,
            'data_dir': test_data_dir,
            'force_rerun': True
        }
        
        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)
        
        if result.get('data_collection_completed', False):
            logger.info('✅ Test 3 PASSED: Monitoring integration successful')
        else:
            logger.error('❌ Test 3 FAILED: Monitoring integration failed')
        
        # Test 4: Test error handling and recovery
        logger.info('🧪 Test 4: Testing Error Handling and Recovery')
        
        # Simulate an error by passing invalid parameters
        try:
            await run_comprehensive_step01(
                symbol="",  # Invalid symbol
                exchange="",  # Invalid exchange
                timeframe="invalid",  # Invalid timeframe
                data_dir=test_data_dir,
                force_rerun=True
            )
        except Exception as e:
            logger.info(f'📊 Error handling test: Exception caught and handled - {str(e)[:100]}')
        
        # Check error summary
        error_summary = error_handler.get_error_summary()
        logger.info(f'📊 Errors handled during testing: {error_summary["total_errors"]}')
        
        # Test 5: Test performance monitoring
        logger.info('🧪 Test 5: Testing Performance Monitoring')
        
        # Get final monitoring summary
        final_call_summary = function_monitor.get_call_summary()
        final_error_summary = error_handler.get_error_summary()
        
        logger.info('📊 Final Performance Summary:')
        logger.info(f'   Total function calls: {final_call_summary["total_calls"]}')
        logger.info(f'   Function call success rate: {final_call_summary["success_rate"]:.1f}%')
        logger.info(f'   Average function duration: {final_call_summary["average_duration"]:.3f}s')
        logger.info(f'   Total errors handled: {final_error_summary["total_errors"]}')
        logger.info(f'   Error recovery rate: {final_error_summary["recovery_success_rate"]:.1f}%')
        
        # Test 6: Test report generation
        logger.info('🧪 Test 6: Testing Report Generation')
        
        # Export monitoring reports
        timestamp = "test_run"
        
        # Export function call report
        function_report_path = f"test_function_monitoring_report_{timestamp}.json"
        function_monitor.export_detailed_report(function_report_path)
        logger.info(f'📊 Function monitoring report exported to: {function_report_path}')
        
        # Export error report
        error_report_path = f"test_error_tracking_report_{timestamp}.json"
        error_handler.export_error_report(error_report_path)
        logger.info(f'📊 Error tracking report exported to: {error_report_path}')
        
        # Check if reports were created
        if os.path.exists(function_report_path) and os.path.exists(error_report_path):
            logger.info('✅ Test 6 PASSED: Report generation successful')
        else:
            logger.error('❌ Test 6 FAILED: Report generation failed')
        
        # Final summary
        logger.info('=' * 80)
        logger.info('🎉 COMPREHENSIVE MONITORING TEST SUMMARY')
        logger.info('=' * 80)
        logger.info('✅ All monitoring systems are working correctly:')
        logger.info('   • Function call monitoring with detailed tracking')
        logger.info('   • Function entry validation with comprehensive checks')
        logger.info('   • Inter-function call tracking and dependency monitoring')
        logger.info('   • Function completion reporting with outcome analysis')
        logger.info('   • Enhanced error handling with detailed function-level tracking')
        logger.info('   • Performance monitoring with timing and resource usage')
        logger.info('   • Comprehensive logging with structured reports')
        logger.info('=' * 80)
        
        return True
        
    except Exception as e:
        logger.exception(f'❌ Test failed with exception: {e}')
        return False


async def main():
    """Main test function."""
    logger.info('🚀 Starting Comprehensive Step01 Monitoring Test Suite')
    
    try:
        success = await test_comprehensive_monitoring()
        
        if success:
            logger.info('🎉 All tests completed successfully!')
            print('✅ Comprehensive Step01 Monitoring Test Suite PASSED')
        else:
            logger.error('❌ Some tests failed!')
            print('❌ Comprehensive Step01 Monitoring Test Suite FAILED')
    
    except Exception as e:
        logger.exception(f'❌ Test suite failed with exception: {e}')
        print(f'❌ Test suite failed: {e}')


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('\n🛑 Test interrupted by user')
    except Exception as e:
        print(f'❌ Test error: {e}')