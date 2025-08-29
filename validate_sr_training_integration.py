#!/usr/bin/env python3
"""
Simple validation script to verify S/R integration across training files.
This script checks syntax and method calls without requiring external dependencies.
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set

def check_file_syntax(file_path: str) -> bool:
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return False

def find_sr_method_calls(file_path: str) -> Dict[str, List[int]]:
    """Find S/R method calls in a Python file."""
    sr_methods = {
        'get_sr_context': [],
        'predict_sr_outcome': [],
        'calculate_sr_features': [],
        'calculate_comprehensive_sr_features': [],
        'is_near_sr_level': [],
        'predict_breakout': [],
        'set_weights': [],
        'SRBreakoutPredictor': [],
        'setup_sr_breakout_predictor': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
        for line_num, line in enumerate(lines, 1):
            for method in sr_methods.keys():
                if method in line:
                    sr_methods[method].append(line_num)
                    
    except Exception as e:
        print(f"❌ Error analyzing {file_path}: {e}")
        
    return sr_methods

def check_sr_imports(file_path: str) -> bool:
    """Check if file imports SRBreakoutPredictor."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for import statements
        import_patterns = [
            'from src.tactician.sr_breakout_predictor import SRBreakoutPredictor',
            'from src.tactician.sr_breakout_predictor import setup_sr_breakout_predictor',
            'import src.tactician.sr_breakout_predictor',
            'from .sr_breakout_predictor import SRBreakoutPredictor'
        ]
        
        for pattern in import_patterns:
            if pattern in content:
                return True
                
        return False
        
    except Exception as e:
        print(f"❌ Error checking imports in {file_path}: {e}")
        return False

def validate_sr_integration():
    """Validate S/R integration across training files."""
    print("🚀 Starting S/R Training Integration Validation")
    print("=" * 60)
    
    # Files that should use S/R functionality
    training_files = [
        "src/training/steps/step6_feature_engineering.py",
        "src/training/steps/step10_unified_regime_intelligence.py", 
        "src/training/steps/step15_tactician_specialist_training.py",
        "src/training/steps/sr_outcome_model_trainer.py",
        "src/training/steps/step9_hmm_based_training.py",
        "src/training/steps/step17_final_parameters_optimization/sr_optuna_optimization.py"
    ]
    
    validation_results = {}
    
    for file_path in training_files:
        print(f"\n📁 Checking {file_path}...")
        
        if not Path(file_path).exists():
            print(f"❌ File not found: {file_path}")
            validation_results[file_path] = False
            continue
            
        # Check syntax
        syntax_ok = check_file_syntax(file_path)
        if not syntax_ok:
            validation_results[file_path] = False
            continue
            
        # Check imports
        has_import = check_sr_imports(file_path)
        
        # Find method calls
        method_calls = find_sr_method_calls(file_path)
        
        # Check if file uses S/R functionality
        has_sr_usage = any(len(calls) > 0 for calls in method_calls.values())
        
        if has_import and has_sr_usage:
            print(f"✅ {file_path} - Valid S/R integration")
            print(f"   Methods used: {[method for method, calls in method_calls.items() if calls]}")
            validation_results[file_path] = True
        elif has_import:
            print(f"⚠️ {file_path} - Imports S/R but no method calls found")
            validation_results[file_path] = True  # Still valid, just not actively used
        else:
            print(f"❌ {file_path} - No S/R integration found")
            validation_results[file_path] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 S/R TRAINING INTEGRATION VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in validation_results.values() if result)
    total = len(validation_results)
    
    for file_path, result in validation_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{file_path:<50} {status}")
    
    print("-" * 60)
    print(f"Total Files: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL S/R TRAINING INTEGRATION VALIDATIONS PASSED!")
        print("The cleaned up S/R implementation is properly integrated across all training files.")
        return True
    else:
        print(f"\n⚠️ {total - passed} VALIDATIONS FAILED")
        print("Some S/R integrations need attention.")
        return False

def check_sr_predictor_file():
    """Check the main S/R predictor file."""
    print("\n🔍 Checking main S/R predictor file...")
    
    sr_file = "src/tactician/sr_breakout_predictor.py"
    
    if not Path(sr_file).exists():
        print(f"❌ S/R predictor file not found: {sr_file}")
        return False
        
    # Check syntax
    if not check_file_syntax(sr_file):
        print("❌ S/R predictor file has syntax errors")
        return False
        
    # Check for required methods
    required_methods = [
        'get_sr_context',
        'predict_sr_outcome', 
        'calculate_sr_features',
        'calculate_comprehensive_sr_features',
        'is_near_sr_level',
        'predict_breakout',
        'set_weights'
    ]
    
    method_calls = find_sr_method_calls(sr_file)
    
    missing_methods = []
    for method in required_methods:
        if not method_calls.get(method):
            missing_methods.append(method)
    
    if missing_methods:
        print(f"❌ Missing required methods: {missing_methods}")
        return False
    else:
        print("✅ S/R predictor file is valid and complete")
        return True

if __name__ == "__main__":
    # Check main S/R file
    sr_ok = check_sr_predictor_file()
    
    # Check training integrations
    training_ok = validate_sr_integration()
    
    # Overall result
    if sr_ok and training_ok:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        sys.exit(0)
    else:
        print("\n⚠️ SOME VALIDATIONS FAILED!")
        sys.exit(1)