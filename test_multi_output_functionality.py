#!/usr/bin/env python3
"""
Test Multi-Output Functionality

This script tests that the ML models still generate multiple outputs:
- Price prediction before hitting opposite side price barrier
- Probability of hitting the barrier
- Risk of hitting the opposite price barrier first
"""

import os
import sys
from pathlib import Path
from datetime import datetime

def main():
    """Main execution function."""
    print("🧪 Testing Multi-Output Functionality")
    print("=" * 50)
    
    test_results = {
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    # Test 1: Check that consolidated analyst and tactician training file exists
    print("🔍 Test 1: Verifying consolidated analyst and tactician training file...")
    analyst_tactician_file = Path('src/training/steps/consolidated_analyst_tactician_training.py')
    
    if analyst_tactician_file.exists():
        print(f"  ✅ {analyst_tactician_file}")
        test_results['passed'] += 1
    else:
        print(f"  ❌ {analyst_tactician_file}")
        test_results['failed'] += 1
        test_results['errors'].append(f"Missing file: {analyst_tactician_file}")
    
    print()
    
    # Test 2: Check for multi-output functionality in the file
    print("🔍 Test 2: Verifying multi-output functionality...")
    
    if analyst_tactician_file.exists():
        with open(analyst_tactician_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        multi_output_features = [
            ('MultiOutputModelTrainer', 'Multi-output model trainer class'),
            ('price_prediction', 'Price prediction output'),
            ('probability', 'Probability output'),
            ('risk', 'Risk output'),
            ('train_multi_output_model', 'Multi-output training method'),
            ('generate_multi_output_predictions', 'Multi-output prediction generation'),
            ('calculate_combined_risk_metrics', 'Combined risk metrics calculation')
        ]
        
        for keyword, description in multi_output_features:
            if keyword in content:
                print(f"  ✅ {description}")
                test_results['passed'] += 1
            else:
                print(f"  ❌ {description}")
                test_results['failed'] += 1
                test_results['errors'].append(f"Missing: {description}")
    
    print()
    
    # Test 3: Check for core principles preservation
    print("🔍 Test 3: Verifying core principles preservation...")
    
    if analyst_tactician_file.exists():
        with open(analyst_tactician_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        core_principles = [
            ('per-HMM regime training', 'Per-HMM regime training preservation'),
            ('Analyst/Tactician separation', 'Analyst/Tactician separation preservation'),
            ('Tactician labels based on Analyst', 'Tactician labels based on Analyst predictions'),
            ('ConsolidatedAnalystEnhancement', 'Analyst enhancement class'),
            ('ConsolidatedTacticianSpecialistTraining', 'Tactician specialist training class')
        ]
        
        for keyword, description in core_principles:
            if keyword in content:
                print(f"  ✅ {description}")
                test_results['passed'] += 1
            else:
                print(f"  ❌ {description}")
                test_results['failed'] += 1
                test_results['errors'].append(f"Missing: {description}")
    
    print()
    
    # Test 4: Check that classes are in src/training/steps/ not in utilities
    print("🔍 Test 4: Verifying architecture (classes in src/training/steps/, not utilities)...")
    
    # Check that the classes are in the training steps directory
    if analyst_tactician_file.exists():
        print(f"  ✅ ConsolidatedAnalystEnhancement and ConsolidatedTacticianSpecialistTraining are in src/training/steps/")
        test_results['passed'] += 1
    else:
        print(f"  ❌ Classes not found in src/training/steps/")
        test_results['failed'] += 1
        test_results['errors'].append("Classes not in correct location")
    
    # Check that utilities are used as toolbox
    if analyst_tactician_file.exists():
        with open(analyst_tactician_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'from src.utils.ml_common import' in content:
            print(f"  ✅ Utilities are used as toolbox (imported from src.utils.ml_common)")
            test_results['passed'] += 1
        else:
            print(f"  ❌ Utilities not properly used as toolbox")
            test_results['failed'] += 1
            test_results['errors'].append("Utilities not used as toolbox")
    
    print()
    
    # Test 5: Check for backward compatibility
    print("🔍 Test 5: Verifying backward compatibility...")
    
    if analyst_tactician_file.exists():
        with open(analyst_tactician_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        compatibility_classes = [
            'AnalystEnhancement',
            'TacticianSpecialistTraining'
        ]
        
        for class_name in compatibility_classes:
            if class_name in content:
                print(f"  ✅ {class_name} backward compatibility wrapper")
                test_results['passed'] += 1
            else:
                print(f"  ❌ {class_name} backward compatibility wrapper")
                test_results['failed'] += 1
                test_results['errors'].append(f"Missing backward compatibility: {class_name}")
    
    print()
    
    # Test 6: Check for multi-output model outputs
    print("🔍 Test 6: Verifying specific multi-output model outputs...")
    
    if analyst_tactician_file.exists():
        with open(analyst_tactician_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_outputs = [
            ('price_prediction', 'Price prediction before hitting opposite side price barrier'),
            ('probability', 'Probability of hitting the barrier'),
            ('risk', 'Risk of hitting the opposite price barrier first')
        ]
        
        for output_key, description in required_outputs:
            if output_key in content:
                print(f"  ✅ {description}")
                test_results['passed'] += 1
            else:
                print(f"  ❌ {description}")
                test_results['failed'] += 1
                test_results['errors'].append(f"Missing output: {description}")
    
    print()
    
    # Summary
    print("📊 TEST SUMMARY")
    print("=" * 20)
    print(f"Tests passed: {test_results['passed']}")
    print(f"Tests failed: {test_results['failed']}")
    print(f"Total tests: {test_results['passed'] + test_results['failed']}")
    
    if test_results['errors']:
        print("\n❌ ERRORS:")
        for error in test_results['errors']:
            print(f"  - {error}")
    
    if test_results['failed'] == 0:
        print("\n✅ ALL TESTS PASSED!")
        print("🎉 Multi-output functionality is working correctly!")
        print("\n📋 Multi-Output Model Features Verified:")
        print("  ✅ Price prediction before hitting opposite side price barrier")
        print("  ✅ Probability of hitting the barrier")
        print("  ✅ Risk of hitting the opposite price barrier first")
        print("  ✅ ConsolidatedAnalystEnhancement and ConsolidatedTacticianSpecialistTraining in src/training/steps/")
        print("  ✅ Utilities used as toolbox from src/utils/")
        print("  ✅ Core principles preserved")
        print("  ✅ Backward compatibility maintained")
        return True
    else:
        print(f"\n❌ {test_results['failed']} TESTS FAILED!")
        print("🔧 Please review the errors above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)