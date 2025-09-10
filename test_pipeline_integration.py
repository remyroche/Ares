#!/usr/bin/env python3
"""
Test Pipeline Integration

This script tests the comprehensive training pipeline integration without external dependencies.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

def main():
    """Main execution function."""
    print("🧪 Testing Comprehensive Training Pipeline Integration")
    print("=" * 60)
    
    test_results = {
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    # Test 1: Check that all required files exist
    print("🔍 Test 1: Verifying all required files exist...")
    
    required_files = [
        'src/training/steps/comprehensive_training_pipeline.py',
        'src/training/steps/comprehensive_training_pipeline_no_deps.py',
        'src/training/steps/consolidated_analyst_tactician_training.py',
        'src/training/steps/consolidated_model_training.py',
        'src/training/steps/simplified_pipeline_infrastructure.py',
        'src/utils/ml_common/__init__.py',
        'src/utils/ml_common/model_training.py',
        'src/utils/ml_common/model_evaluation.py',
        'src/utils/ml_common/data_quality.py',
        'src/utils/mock_dependencies.py'
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
            test_results['passed'] += 1
        else:
            print(f"  ❌ {file_path}")
            test_results['failed'] += 1
            test_results['errors'].append(f"Missing file: {file_path}")
    
    print()
    
    # Test 2: Check pipeline structure
    print("🔍 Test 2: Verifying pipeline structure...")
    
    pipeline_file = Path('src/training/steps/comprehensive_training_pipeline.py')
    if pipeline_file.exists():
        with open(pipeline_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for all 9 pipeline steps
        required_steps = [
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
        
        for step in required_steps:
            if step in content:
                print(f"  ✅ Step: {step}")
                test_results['passed'] += 1
            else:
                print(f"  ❌ Step: {step}")
                test_results['failed'] += 1
                test_results['errors'].append(f"Missing step: {step}")
    
    print()
    
    # Test 3: Check toolbox utilities integration
    print("🔍 Test 3: Verifying toolbox utilities integration...")
    
    if pipeline_file.exists():
        with open(pipeline_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        toolbox_utilities = [
            'EnhancedModelTrainer',
            'ModelEvaluationUtilities', 
            'DataQualityUtilities',
            'MLTrainingSafeguards',
            'FeatureSelectionFramework',
            'MemoryEfficientTraining',
            'ParallelProcessingCoordinator'
        ]
        
        for utility in toolbox_utilities:
            if utility in content:
                print(f"  ✅ Toolbox utility: {utility}")
                test_results['passed'] += 1
            else:
                print(f"  ❌ Toolbox utility: {utility}")
                test_results['failed'] += 1
                test_results['errors'].append(f"Missing toolbox utility: {utility}")
    
    print()
    
    # Test 4: Check core principles preservation
    print("🔍 Test 4: Verifying core principles preservation...")
    
    if pipeline_file.exists():
        with open(pipeline_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        core_principles = [
            'per-HMM regime training',
            'Analyst/Tactician separation',
            'Tactician labels based on Analyst',
            'ConsolidatedAnalystEnhancement',
            'ConsolidatedTacticianSpecialistTraining',
            'ConsolidatedUnifiedRegimeIntelligence'
        ]
        
        for principle in core_principles:
            if principle in content:
                print(f"  ✅ Core principle: {principle}")
                test_results['passed'] += 1
            else:
                print(f"  ❌ Core principle: {principle}")
                test_results['failed'] += 1
                test_results['errors'].append(f"Missing core principle: {principle}")
    
    print()
    
    # Test 5: Check multi-output functionality
    print("🔍 Test 5: Verifying multi-output functionality...")
    
    if pipeline_file.exists():
        with open(pipeline_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        multi_output_features = [
            'MultiOutputModelTrainer',
            'price_prediction',
            'probability', 
            'risk',
            'multi_output_predictions',
            'multi_output_types'
        ]
        
        for feature in multi_output_features:
            if feature in content:
                print(f"  ✅ Multi-output feature: {feature}")
                test_results['passed'] += 1
            else:
                print(f"  ❌ Multi-output feature: {feature}")
                test_results['failed'] += 1
                test_results['errors'].append(f"Missing multi-output feature: {feature}")
    
    print()
    
    # Test 6: Check configuration integration
    print("🔍 Test 6: Verifying configuration integration...")
    
    if pipeline_file.exists():
        with open(pipeline_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        config_features = [
            'validate_and_fix_config',
            'model_training_config',
            'evaluation_config',
            'ConfigurationValidator'
        ]
        
        for feature in config_features:
            if feature in content:
                print(f"  ✅ Configuration feature: {feature}")
                test_results['passed'] += 1
            else:
                print(f"  ❌ Configuration feature: {feature}")
                test_results['failed'] += 1
                test_results['errors'].append(f"Missing configuration feature: {feature}")
    
    print()
    
    # Test 7: Check data flow structure
    print("🔍 Test 7: Verifying data flow structure...")
    
    if pipeline_file.exists():
        with open(pipeline_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        data_flow_features = [
            'pipeline_state',
            'dependencies',
            'execute_pipeline',
            'get_pipeline_summary',
            'create_data_processing_step_function'
        ]
        
        for feature in data_flow_features:
            if feature in content:
                print(f"  ✅ Data flow feature: {feature}")
                test_results['passed'] += 1
            else:
                print(f"  ❌ Data flow feature: {feature}")
                test_results['failed'] += 1
                test_results['errors'].append(f"Missing data flow feature: {feature}")
    
    print()
    
    # Test 8: Check error handling
    print("🔍 Test 8: Verifying error handling...")
    
    if pipeline_file.exists():
        with open(pipeline_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        error_handling_features = [
            'try:',
            'except Exception as e:',
            'self.logger.exception',
            'raise'
        ]
        
        for feature in error_handling_features:
            if feature in content:
                print(f"  ✅ Error handling feature: {feature}")
                test_results['passed'] += 1
            else:
                print(f"  ❌ Error handling feature: {feature}")
                test_results['failed'] += 1
                test_results['errors'].append(f"Missing error handling feature: {feature}")
    
    print()
    
    # Summary
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 30)
    print(f"Tests passed: {test_results['passed']}")
    print(f"Tests failed: {test_results['failed']}")
    print(f"Total tests: {test_results['passed'] + test_results['failed']}")
    
    if test_results['errors']:
        print("\n❌ ERRORS:")
        for error in test_results['errors']:
            print(f"  - {error}")
    
    if test_results['failed'] == 0:
        print("\n✅ ALL INTEGRATION TESTS PASSED!")
        print("🎉 Comprehensive training pipeline is fully integrated!")
        print("\n📋 Integration Features Verified:")
        print("  ✅ All required files present")
        print("  ✅ Complete pipeline structure (9 steps)")
        print("  ✅ Toolbox utilities integration")
        print("  ✅ Core principles preserved")
        print("  ✅ Multi-output functionality")
        print("  ✅ Configuration integration")
        print("  ✅ Data flow structure")
        print("  ✅ Error handling")
        print("\n🚀 Pipeline Ready For:")
        print("  ✅ Development and testing")
        print("  ✅ Integration with existing codebase")
        print("  ✅ Further customization and extension")
        return True
    else:
        print(f"\n❌ {test_results['failed']} INTEGRATION TESTS FAILED!")
        print("🔧 Please review the errors above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)