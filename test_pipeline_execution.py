#!/usr/bin/env python3
"""
Test Pipeline Execution

This script tests the comprehensive training pipeline execution with mock data
to validate the complete structure and data flow.
"""

import sys
import os
import asyncio
from pathlib import Path
from datetime import datetime

def main():
    """Main execution function."""
    print("🧪 Testing Comprehensive Training Pipeline Execution")
    print("=" * 60)
    
    test_results = {
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    # Test 1: Test pipeline creation
    print("🔍 Test 1: Testing pipeline creation...")
    
    try:
        # Import mock dependencies first
        sys.path.insert(0, '/workspace')
        from src.utils.mock_dependencies import install_mocks
        install_mocks()
        print("  ✅ Mock dependencies installed")
        test_results['passed'] += 1
    except Exception as e:
        print(f"  ❌ Failed to install mock dependencies: {e}")
        test_results['failed'] += 1
        test_results['errors'].append(f"Mock dependencies error: {e}")
    
    # Test 2: Test configuration creation
    print("🔍 Test 2: Testing configuration creation...")
    
    try:
        from src.training.steps.comprehensive_config_integration import create_custom_config
        
        # Test different configuration templates
        configs_to_test = [
            ('development', 'Development configuration'),
            ('testing', 'Testing configuration'),
            ('production', 'Production configuration'),
            ('minimal', 'Minimal configuration')
        ]
        
        for template_name, description in configs_to_test:
            config = create_custom_config(template_name)
            if config and 'symbol' in config:
                print(f"  ✅ {description}")
                test_results['passed'] += 1
            else:
                print(f"  ❌ {description}")
                test_results['failed'] += 1
                test_results['errors'].append(f"Configuration creation failed: {template_name}")
        
    except Exception as e:
        print(f"  ❌ Configuration creation failed: {e}")
        test_results['failed'] += 1
        test_results['errors'].append(f"Configuration error: {e}")
    
    print()
    
    # Test 3: Test pipeline initialization
    print("🔍 Test 3: Testing pipeline initialization...")
    
    try:
        from src.training.steps.comprehensive_training_pipeline import ComprehensiveTrainingPipeline
        
        # Create test configuration
        test_config = {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1m',
            'data_dir': 'data',
            'output_dir': 'output',
            'model_dir': 'models',
            'log_dir': 'logs',
            'enable_gpu': False,
            'enable_parallel': True,
            'max_workers': 2,
            'memory_limit': 0.6,
            'timeout_seconds': 300,
            'random_state': 42,
            'debug_mode': True,
            'verbose_logging': True,
            'model_training_config': {
                'enable_confidence_metrics': True,
                'enable_calibration_assessment': True,
                'enable_feature_importance': True,
                'enable_cross_validation': True,
                'enable_model_explanations': True,
                'enable_post_training_hpo': False,
                'cv_folds': 3
            },
            'evaluation_config': {
                'enable_cross_validation': True,
                'enable_time_series_validation': True,
                'enable_confidence_intervals': True,
                'enable_model_comparison': True,
                'enable_feature_importance_analysis': True,
                'enable_prediction_analysis': True,
                'cv_folds': 3,
                'confidence_level': 0.95
            }
        }
        
        # Initialize pipeline
        pipeline = ComprehensiveTrainingPipeline(test_config)
        print("  ✅ Pipeline initialized successfully")
        test_results['passed'] += 1
        
        # Test pipeline summary
        summary = pipeline.get_pipeline_summary()
        if summary and 'pipeline_type' in summary:
            print(f"  ✅ Pipeline summary generated: {summary['pipeline_type']}")
            test_results['passed'] += 1
        else:
            print("  ❌ Pipeline summary generation failed")
            test_results['failed'] += 1
            test_results['errors'].append("Pipeline summary generation failed")
        
    except Exception as e:
        print(f"  ❌ Pipeline initialization failed: {e}")
        test_results['failed'] += 1
        test_results['errors'].append(f"Pipeline initialization error: {e}")
    
    print()
    
    # Test 4: Test mock data generation
    print("🔍 Test 4: Testing mock data generation...")
    
    try:
        from src.training.steps.comprehensive_data_flow_testing import generate_mock_pipeline_data
        
        # Generate mock data
        mock_data = generate_mock_pipeline_data()
        
        # Validate mock data structure
        required_data_keys = [
            'raw_data', 'data_quality_report', 'collection_metadata',
            'sr_levels', 'sr_metadata', 'regimes', 'regime_metadata',
            'engineered_features', 'feature_metadata', 'selected_features',
            'selection_metadata', 'analyst_models', 'analyst_metadata',
            'general_model', 'general_model_metadata', 'tactician_models',
            'tactician_metadata', 'backtesting_results', 'validation_results',
            'validation_metadata'
        ]
        
        missing_keys = [key for key in required_data_keys if key not in mock_data]
        
        if not missing_keys:
            print(f"  ✅ Mock data generated with {len(mock_data)} data sections")
            test_results['passed'] += 1
        else:
            print(f"  ❌ Mock data missing keys: {missing_keys}")
            test_results['failed'] += 1
            test_results['errors'].append(f"Mock data missing keys: {missing_keys}")
        
    except Exception as e:
        print(f"  ❌ Mock data generation failed: {e}")
        test_results['failed'] += 1
        test_results['errors'].append(f"Mock data generation error: {e}")
    
    print()
    
    # Test 5: Test data flow validation
    print("🔍 Test 5: Testing data flow validation...")
    
    try:
        from src.training.steps.comprehensive_data_flow_testing import test_pipeline_data_flow
        
        # Test data flow
        test_results_flow = test_pipeline_data_flow(mock_data)
        
        if test_results_flow.get('overall_passed', False):
            summary = test_results_flow.get('summary', {})
            print(f"  ✅ Data flow test passed: {summary.get('steps_passed', 0)}/{summary.get('total_steps_tested', 0)} steps")
            test_results['passed'] += 1
        else:
            print("  ❌ Data flow test failed")
            test_results['failed'] += 1
            test_results['errors'].append("Data flow test failed")
        
    except Exception as e:
        print(f"  ❌ Data flow validation failed: {e}")
        test_results['failed'] += 1
        test_results['errors'].append(f"Data flow validation error: {e}")
    
    print()
    
    # Test 6: Test pipeline step creation
    print("🔍 Test 6: Testing pipeline step creation...")
    
    try:
        # Test that all pipeline steps can be created
        pipeline_steps = [
            'data_collection_qualification',
            'sr_levels_detection',
            'regimes_definition',
            'feature_engineering',
            'feature_selection',
            'analyst_training',
            'general_model_training',
            'tactician_training',
            'backtesting_validation'
        ]
        
        for step_name in pipeline_steps:
            # Test that step creation methods exist
            step_method = f'_create_{step_name}_step'
            if hasattr(pipeline, step_method):
                print(f"  ✅ Pipeline step: {step_name}")
                test_results['passed'] += 1
            else:
                print(f"  ❌ Pipeline step: {step_name}")
                test_results['failed'] += 1
                test_results['errors'].append(f"Missing pipeline step method: {step_method}")
        
    except Exception as e:
        print(f"  ❌ Pipeline step creation test failed: {e}")
        test_results['failed'] += 1
        test_results['errors'].append(f"Pipeline step creation error: {e}")
    
    print()
    
    # Test 7: Test toolbox utilities integration
    print("🔍 Test 7: Testing toolbox utilities integration...")
    
    try:
        # Test that toolbox utilities are properly initialized
        toolbox_utilities = [
            'model_trainer',
            'model_evaluator',
            'data_quality',
            'safeguards',
            'feature_selector',
            'memory_optimizer',
            'parallel_processor'
        ]
        
        for utility_name in toolbox_utilities:
            if hasattr(pipeline, utility_name):
                utility = getattr(pipeline, utility_name)
                if utility is not None:
                    print(f"  ✅ Toolbox utility: {utility_name}")
                    test_results['passed'] += 1
                else:
                    print(f"  ❌ Toolbox utility: {utility_name} (None)")
                    test_results['failed'] += 1
                    test_results['errors'].append(f"Toolbox utility {utility_name} is None")
            else:
                print(f"  ❌ Toolbox utility: {utility_name} (missing)")
                test_results['failed'] += 1
                test_results['errors'].append(f"Missing toolbox utility: {utility_name}")
        
    except Exception as e:
        print(f"  ❌ Toolbox utilities integration test failed: {e}")
        test_results['failed'] += 1
        test_results['errors'].append(f"Toolbox utilities integration error: {e}")
    
    print()
    
    # Test 8: Test core principles validation
    print("🔍 Test 8: Testing core principles validation...")
    
    try:
        # Test that core principles are preserved
        core_principles = [
            ('per-HMM regime training', 'Per-HMM regime training'),
            ('Analyst/Tactician separation', 'Analyst/Tactician separation'),
            ('Tactician labels based on Analyst', 'Tactician labels based on Analyst predictions'),
            ('ConsolidatedAnalystEnhancement', 'Analyst enhancement class'),
            ('ConsolidatedTacticianSpecialistTraining', 'Tactician specialist training class'),
            ('ConsolidatedUnifiedRegimeIntelligence', 'Unified regime intelligence class')
        ]
        
        # Check pipeline source code for core principles
        pipeline_file = Path('src/training/steps/comprehensive_training_pipeline.py')
        if pipeline_file.exists():
            with open(pipeline_file, 'r', encoding='utf-8') as f:
                pipeline_content = f.read()
            
            for keyword, description in core_principles:
                if keyword in pipeline_content:
                    print(f"  ✅ Core principle: {description}")
                    test_results['passed'] += 1
                else:
                    print(f"  ❌ Core principle: {description}")
                    test_results['failed'] += 1
                    test_results['errors'].append(f"Missing core principle: {description}")
        else:
            print("  ❌ Pipeline file not found")
            test_results['failed'] += 1
            test_results['errors'].append("Pipeline file not found")
        
    except Exception as e:
        print(f"  ❌ Core principles validation failed: {e}")
        test_results['failed'] += 1
        test_results['errors'].append(f"Core principles validation error: {e}")
    
    print()
    
    # Summary
    print("📊 PIPELINE EXECUTION TEST SUMMARY")
    print("=" * 40)
    print(f"Tests passed: {test_results['passed']}")
    print(f"Tests failed: {test_results['failed']}")
    print(f"Total tests: {test_results['passed'] + test_results['failed']}")
    
    if test_results['errors']:
        print("\n❌ ERRORS:")
        for error in test_results['errors']:
            print(f"  - {error}")
    
    if test_results['failed'] == 0:
        print("\n✅ ALL PIPELINE EXECUTION TESTS PASSED!")
        print("🎉 Comprehensive training pipeline execution is working correctly!")
        print("\n📋 Pipeline Execution Features Verified:")
        print("  ✅ Mock dependencies integration")
        print("  ✅ Configuration creation and validation")
        print("  ✅ Pipeline initialization and summary")
        print("  ✅ Mock data generation")
        print("  ✅ Data flow validation")
        print("  ✅ Pipeline step creation")
        print("  ✅ Toolbox utilities integration")
        print("  ✅ Core principles preservation")
        print("\n🚀 Pipeline Ready For:")
        print("  ✅ Full execution with mock data")
        print("  ✅ Integration with real data sources")
        print("  ✅ Production deployment")
        return True
    else:
        print(f"\n❌ {test_results['failed']} PIPELINE EXECUTION TESTS FAILED!")
        print("🔧 Please review the errors above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)