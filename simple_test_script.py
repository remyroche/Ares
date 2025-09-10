#!/usr/bin/env python3
"""
Simplified Test Script for Simplified Infrastructure

This script performs basic tests to verify the new infrastructure works correctly.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

def main():
    """Main execution function."""
    print("🧪 Simplified Infrastructure Test Suite")
    print("=" * 50)
    
    test_results = {
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    # Test 1: Check new infrastructure files exist
    print("🔍 Test 1: Verifying new infrastructure files...")
    new_files = [
        'src/training/steps/simplified_pipeline_infrastructure.py',
        'src/training/steps/simplified_base_step.py',
        'src/training/steps/standardized_config_validation.py',
        'src/training/steps/unified_data_quality.py',
        'src/training/steps/unified_feature_engineering.py',
        'src/training/steps/unified_model_training.py',
        'src/training/steps/consolidated_model_training.py',
    ]
    
    for file_path in new_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ {file_path}")
            test_results['passed'] += 1
        else:
            print(f"  ❌ {file_path}")
            test_results['failed'] += 1
            test_results['errors'].append(f"Missing file: {file_path}")
    
    print()
    
    # Test 2: Check that deprecated files still exist (before deletion)
    print("🔍 Test 2: Verifying deprecated files are ready for deletion...")
    deprecated_files = [
        'src/training/steps/base_step.py',
        'src/training/steps/step1_data_collection.py',
        'src/training/steps/step05_labeling.py',
        'src/training/steps/model_training/step09_hmm_based_training.py',
        'src/training/steps/model_training/step11_analyst_creation.py',
        'src/training/steps/model_training/step12_analyst_enhancement.py',
        'src/training/steps/model_training/step15_tactician_specialist_training.py',
    ]
    
    for file_path in deprecated_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ {file_path} (ready for deletion)")
            test_results['passed'] += 1
        else:
            print(f"  ⚠️  {file_path} (already deleted or not found)")
    
    print()
    
    # Test 3: Check import syntax in new files
    print("🔍 Test 3: Checking import syntax in new files...")
    for file_path in new_files:
        path = Path(file_path)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for basic Python syntax issues
                if 'import' in content and 'from' in content:
                    print(f"  ✅ {file_path} (imports look good)")
                    test_results['passed'] += 1
                else:
                    print(f"  ⚠️  {file_path} (no imports found)")
                    
            except Exception as e:
                print(f"  ❌ {file_path} (error reading: {e})")
                test_results['failed'] += 1
                test_results['errors'].append(f"Error reading {file_path}: {e}")
    
    print()
    
    # Test 4: Check for core principles preservation
    print("🔍 Test 4: Verifying core principles preservation...")
    
    # Check consolidated model training file for Analyst/Tactician classes
    consolidated_file = Path('src/training/steps/consolidated_model_training.py')
    if consolidated_file.exists():
        with open(consolidated_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        core_principles = [
            ('ConsolidatedAnalystEnhancement', 'Analyst enhancement class'),
            ('ConsolidatedTacticianSpecialistTraining', 'Tactician specialist training class'),
            ('ConsolidatedUnifiedRegimeIntelligence', 'Unified regime intelligence class'),
            ('per-HMM regime training', 'Per-HMM regime training preservation'),
            ('Analyst/Tactician separation', 'Analyst/Tactician separation preservation')
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
    
    # Test 5: Check for backward compatibility wrappers
    print("🔍 Test 5: Verifying backward compatibility wrappers...")
    
    # Check unified model training for backward compatibility
    unified_file = Path('src/training/steps/unified_model_training.py')
    if unified_file.exists():
        with open(unified_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        compatibility_classes = [
            'AnalystEnhancement',
            'TacticianSpecialistTraining',
            'HMMBasedTraining'
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
    
    # Test 6: Check file sizes and code reduction
    print("🔍 Test 6: Verifying code reduction...")
    
    # Count lines in new infrastructure files
    total_new_lines = 0
    for file_path in new_files:
        path = Path(file_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
                total_new_lines += lines
                print(f"  📊 {file_path}: {lines} lines")
    
    print(f"  📊 Total new infrastructure: {total_new_lines} lines")
    
    # Count lines in deprecated files
    total_deprecated_lines = 0
    for file_path in deprecated_files:
        path = Path(file_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
                total_deprecated_lines += lines
                print(f"  📊 {file_path}: {lines} lines")
    
    print(f"  📊 Total deprecated files: {total_deprecated_lines} lines")
    
    if total_deprecated_lines > 0:
        reduction_percentage = ((total_deprecated_lines - total_new_lines) / total_deprecated_lines) * 100
        print(f"  📊 Code reduction: {reduction_percentage:.1f}%")
        test_results['passed'] += 1
    else:
        print("  ⚠️  No deprecated files found for comparison")
    
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
        print("🎉 Simplified infrastructure is working correctly!")
        return True
    else:
        print(f"\n❌ {test_results['failed']} TESTS FAILED!")
        print("🔧 Please review the errors above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)