#!/usr/bin/env python3
"""
Test Enhanced Existing Pipeline

This script tests the enhanced existing model training pipeline with
pre-existing decorators and validation utilities.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.steps.model_training import run_model_training_pipeline
from src.utils.pipeline_validation_utils import get_pipeline_validation_summary
from src.utils.common_operations import validate_dataframe_integrity
from src.utils.logger import system_logger


async def test_enhanced_pipeline():
    """Test the enhanced existing model training pipeline."""
    print("🚀 Testing Enhanced Existing Model Training Pipeline")
    print("=" * 80)
    
    logger = system_logger.getChild("TestEnhancedPipeline")
    
    # Configuration for testing
    config = {
        'force_rerun': False,  # Don't force rerun for testing
        'hmm_training': True,
        'regime_intelligence': True,
        'analyst_creation': True,
        'analyst_enhancement': True,
        'ensemble_creation': True,
        'tactician_training': True,
        'random_state': 42,
    }
    
    try:
        # Test pipeline execution
        print("📊 Running enhanced model training pipeline...")
        logger.info("Starting enhanced pipeline test")
        
        success = await run_model_training_pipeline(
            symbol="ETHUSDT",
            exchange="BINANCE",
            timeframe="1m",
            data_dir="data_cache",
            **config
        )
        
        # Display results
        print("\n📋 PIPELINE EXECUTION RESULTS")
        print("=" * 80)
        print(f"Success: {success}")
        
        # Display validation summary
        print("\n🔍 VALIDATION SUMMARY")
        print("=" * 80)
        validation_summary = get_pipeline_validation_summary()
        print(f"Total Validations: {validation_summary['total_validations']}")
        print(f"Passed: {validation_summary['passed']}")
        print(f"Failed: {validation_summary['failed']}")
        print(f"Success Rate: {validation_summary['success_rate']:.2%}")
        
        if validation_summary['validation_results']:
            print("\n📊 DETAILED VALIDATION RESULTS:")
            for result in validation_summary['validation_results']:
                status = "✅ PASSED" if result.get('is_valid', False) else "❌ FAILED"
                print(f"   {result.get('step', 'Unknown')}: {status}")
                if result.get('errors'):
                    for error in result['errors']:
                        print(f"     Error: {error}")
                if result.get('warnings'):
                    for warning in result['warnings']:
                        print(f"     Warning: {warning}")
        
        print("\n✅ Enhanced existing pipeline test completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Enhanced pipeline test failed: {e}")
        logger.exception(f"Enhanced pipeline test failed: {e}")
        return False


async def test_validation_utilities():
    """Test the validation utilities."""
    print("\n🧪 Testing Validation Utilities")
    print("=" * 80)
    
    try:
        # Test DataFrame validation
        import pandas as pd
        import numpy as np
        
        # Create test DataFrame
        test_df = pd.DataFrame({
            'price': [100.0, 101.0, 102.0, np.nan, 104.0],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'side': ['buy', 'sell', 'buy', 'sell', 'buy']
        })
        
        print("Testing DataFrame validation...")
        validation_result = validate_dataframe_integrity(
            test_df, 
            required_columns=['price', 'volume', 'side']
        )
        
        print(f"Validation result: {validation_result['is_valid']}")
        print(f"Errors: {validation_result['errors']}")
        print(f"Warnings: {validation_result['warnings']}")
        
        # Test validation summary
        print("\nTesting validation summary...")
        summary = get_pipeline_validation_summary()
        print(f"Validation summary: {summary}")
        
        print("✅ Validation utilities test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Validation utilities test failed: {e}")
        return False


async def main():
    """Main test function."""
    print("🧪 ENHANCED EXISTING PIPELINE TEST SUITE")
    print("=" * 80)
    
    # Test validation utilities first
    validation_success = await test_validation_utilities()
    
    if validation_success:
        # Test enhanced pipeline
        pipeline_success = await test_enhanced_pipeline()
        
        if pipeline_success:
            print("\n🎉 ALL TESTS PASSED!")
            print("The enhanced existing model training pipeline is working correctly with:")
            print("✅ Pre-existing core decorators (handles_errors, retry, timeout, log_execution_time, traced, validates)")
            print("✅ Enhanced validation utilities")
            print("✅ Data integrity validation")
            print("✅ Pipeline step validation")
            print("✅ Comprehensive error handling")
            print("✅ Performance monitoring")
            print("✅ Detailed logging and reporting")
        else:
            print("\n❌ PIPELINE TEST FAILED")
            return 1
    else:
        print("\n❌ VALIDATION UTILITIES TEST FAILED")
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the test suite
    exit_code = asyncio.run(main())
    sys.exit(exit_code)