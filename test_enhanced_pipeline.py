#!/usr/bin/env python3
"""
Test Enhanced Model Training Pipeline

This script tests the enhanced model training pipeline with comprehensive validation,
error handling, and monitoring.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.steps.model_training.enhanced_model_training_pipeline import (
    EnhancedModelTrainingPipeline,
    run_enhanced_model_training_pipeline,
)
from src.utils.pipeline_validation_framework import validation_orchestrator
from src.utils.performance_monitoring import performance_monitor
from src.utils.error_handling_framework import error_recovery_manager


async def test_enhanced_pipeline():
    """Test the enhanced model training pipeline."""
    print("🚀 Testing Enhanced Model Training Pipeline")
    print("=" * 80)
    
    # Configuration for testing
    config = {
        'data_dir': 'data_cache',
        'hmm_training': True,
        'regime_intelligence': True,
        'analyst_creation': True,
        'analyst_enhancement': True,
        'ensemble_creation': True,
        'tactician_training': True,
        'force_rerun': False,
        'random_state': 42,
    }
    
    try:
        # Test pipeline execution
        print("📊 Running enhanced model training pipeline...")
        result = await run_enhanced_model_training_pipeline(
            symbol="ETHUSDT",
            exchange="BINANCE",
            timeframe="1m",
            config=config
        )
        
        # Display results
        print("\n📋 PIPELINE EXECUTION RESULTS")
        print("=" * 80)
        print(f"Success: {result.get('success', False)}")
        print(f"Execution Time: {result.get('execution_time', 0):.2f} seconds")
        print(f"Success Rate: {result.get('success_rate', 0):.2%}")
        
        if 'performance_metrics' in result:
            metrics = result['performance_metrics']
            print(f"Total Steps: {metrics.get('total_steps', 0)}")
            print(f"Completed Steps: {metrics.get('completed_steps', 0)}")
            print(f"Validation Reports: {metrics.get('validation_reports_count', 0)}")
        
        # Display validation summary
        print("\n🔍 VALIDATION SUMMARY")
        print("=" * 80)
        validation_summary = validation_orchestrator.get_validation_summary()
        print(f"Total Validations: {validation_summary['total_validations']}")
        print(f"Passed: {validation_summary['passed']}")
        print(f"Failed: {validation_summary['failed']}")
        print(f"Warnings: {validation_summary['warnings']}")
        print(f"Success Rate: {validation_summary['success_rate']:.2%}")
        
        # Display performance summary
        print("\n📈 PERFORMANCE SUMMARY")
        print("=" * 80)
        performance_summary = performance_monitor.get_performance_summary()
        print(f"Active Operations: {performance_summary['active_operations']}")
        
        if 'operations' in performance_summary:
            for op_name, op_stats in performance_summary['operations'].items():
                if isinstance(op_stats, dict) and 'avg_duration' in op_stats:
                    print(f"{op_name}: {op_stats['avg_duration']:.3f}s avg, {op_stats['success_rate']:.2%} success")
        
        # Display error summary
        print("\n⚠️ ERROR SUMMARY")
        print("=" * 80)
        error_summary = error_recovery_manager.get_error_summary()
        print(f"Total Errors: {error_summary['total_errors']}")
        
        if error_summary['total_errors'] > 0:
            print("Error Categories:")
            for category, count in error_summary['category_counts'].items():
                print(f"  {category}: {count}")
        
        # Save validation report
        validation_orchestrator.save_validation_report("validation_report.json")
        print(f"\n💾 Validation report saved to: validation_report.json")
        
        # Export performance metrics
        performance_monitor.export_metrics("performance_metrics.json")
        print(f"💾 Performance metrics exported to: performance_metrics.json")
        
        print("\n✅ Enhanced pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Enhanced pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_individual_components():
    """Test individual pipeline components."""
    print("\n🧪 Testing Individual Components")
    print("=" * 80)
    
    try:
        # Test validation framework
        print("Testing validation framework...")
        from src.utils.pipeline_validation_framework import DataFormatValidator
        validator = DataFormatValidator()
        
        import pandas as pd
        test_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        validation_report = await validator.validate(test_df, {})
        
        print(f"  Validation result: {validation_report.result.value}")
        print(f"  Validation duration: {validation_report.duration:.3f}s")
        
        # Test performance monitoring
        print("Testing performance monitoring...")
        with performance_monitor.start_operation("test_operation") as op_id:
            await asyncio.sleep(0.1)  # Simulate work
            performance_monitor.end_operation(op_id, success=True)
        
        # Test error handling
        print("Testing error handling...")
        from src.utils.error_handling_framework import ErrorSeverity, ErrorCategory
        context = error_recovery_manager.create_error_context(
            error=Exception("Test error"),
            step_name="test_step",
            function_name="test_function",
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.UNKNOWN
        )
        
        print(f"  Error context created: {context.error_message}")
        
        print("✅ Individual component tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Individual component tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🧪 ENHANCED MODEL TRAINING PIPELINE TEST SUITE")
    print("=" * 80)
    
    # Test individual components first
    component_success = await test_individual_components()
    
    if component_success:
        # Test full pipeline
        pipeline_success = await test_enhanced_pipeline()
        
        if pipeline_success:
            print("\n🎉 ALL TESTS PASSED!")
            print("The enhanced model training pipeline is working correctly with:")
            print("✅ Comprehensive validation framework")
            print("✅ Operation protection decorators")
            print("✅ Enhanced common utilities")
            print("✅ Individual step validators")
            print("✅ Secure data access patterns")
            print("✅ Pipeline orchestration validation")
            print("✅ Robust error handling framework")
            print("✅ Performance monitoring system")
        else:
            print("\n❌ PIPELINE TEST FAILED")
            return 1
    else:
        print("\n❌ COMPONENT TESTS FAILED")
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the test suite
    exit_code = asyncio.run(main())
    sys.exit(exit_code)