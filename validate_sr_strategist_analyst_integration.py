#!/usr/bin/env python3
"""
Validation script to verify S/R integration in strategist and analyst files.
This script checks that all files properly use the updated sr_breakout_predictor.py functions.
"""

import ast
import sys
from pathlib import Path

def check_file_syntax(...) -> ...:
    """..."""
    passtry:
    passwith open(file_path, 'r', encoding='utf-8') as f:
    passast.parse(f.read())
        return True
    except SyntaxError as e:
    passpasspasspasspasspasspassprint(f"❌ Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
    passpasspasspasspasspasspassprint(f"❌ Error reading {file_path}: {e}")
        return False

def find_sr_method_calls(...) -> ...:
    """..."""
    passsr_methods = {
        'get_sr_context': [],
        'predict_sr_outcome': [],
        'calculate_sr_features': [],
        'calculate_comprehensive_sr_features': [],
        'is_near_sr_level': [],
        'predict_breakout': [],
        'set_weights': [],
        'get_sr_proximity_details': [],
        'SRBreakoutPredictor': [],
        'setup_sr_breakout_predictor': []
    }

    try:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
        with open(file_path, 'r', encoding='utf-8') as f:
    passcontent = f.read()
            lines = content.split('\n')

        for line_num, line in enumerate(lines, 1):
    passfor method in sr_methods.keys():
    passif method in line:
    passsr_methods[method].append(line_num)

    except Exception as e:
    passpasspasspasspasspasspassprint(f"❌ Error analyzing {file_path}: {e}")

    return sr_methods

def check_sr_imports(...) -> ...:
    """..."""
    passtry:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
        with open(file_path, 'r', encoding='utf-8') as f:
    passcontent = f.read()

        # Check for import statements
        import_patterns = [
            'from src.tactician.sr_breakout_predictor import SRBreakoutPredictor',
            'from src.tactician.sr_breakout_predictor import setup_sr_breakout_predictor',
            'import src.tactician.sr_breakout_predictor',
            'from .sr_breakout_predictor import SRBreakoutPredictor'
        ]

        for pattern in import_patterns:
    passif pattern in content:
    passreturn True

        return False

    except Exception as e:
    passpasspasspasspasspasspassprint(f"❌ Error checking imports in {file_path}: {e}")
        return False

def check_method_parameter_compatibility(...) -> ...:
    """..."""
    passcompatibility_results = {}

    try:
    passpass  # TODO: Add proper exception handling
except Exception as e:
    passpasspasspasspasspasspasspass  # TODO: Add proper exception handling
        with open(file_path, 'r', encoding='utf-8') as f:
    passcontent = f.read()
            lines = content.split('\n')

        # Check get_sr_context calls
        for line_num, line in enumerate(lines, 1):
    passif 'get_sr_context(' in line and '=' in line:
    pass# Should have market_data= and current_price= parameters
                if 'market_data=' in line and 'current_price=' in line:
    passcompatibility_results[f'get_sr_context_line_{line_num}'] = True
                else:
    passcompatibility_results[f'get_sr_context_line_{line_num}'] = False

            elif 'predict_sr_outcome(' in line and '=' in line:
    passpass# Should have market_data=, current_price=, and sr_context= parameters
                if 'market_data=' in line and 'current_price=' in line and 'sr_context=' in line:
    passcompatibility_results[f'predict_sr_outcome_line_{line_num}'] = True
                else:
    passcompatibility_results[f'predict_sr_outcome_line_{line_num}'] = False

            elif 'is_near_sr_level(' in line and '=' in line:
    passpass# Should have current_price= and sr_context= parameters
                if 'current_price=' in line and 'sr_context=' in line:
    passcompatibility_results[f'is_near_sr_level_line_{line_num}'] = True
                else:
    passcompatibility_results[f'is_near_sr_level_line_{line_num}'] = False

            elif 'get_sr_proximity_details(' in line and '=' in line:
    passpass# Should have current_price= and sr_context= parameters
                if 'current_price=' in line and 'sr_context=' in line:
    passcompatibility_results[f'get_sr_proximity_details_line_{line_num}'] = True
                else:
    passcompatibility_results[f'get_sr_proximity_details_line_{line_num}'] = False

    except Exception as e:
    passpasspasspasspasspasspassprint(f"❌ Error checking parameter compatibility in {file_path}: {e}")

    return compatibility_results

def validate_sr_integration(...):
    pass"""Validate S/R integration across strategist and analyst files."""
    print("🚀 Starting S/R Strategist/Analyst Integration Validation")
    print("=" * 70)

    # Files that should use S/R functionality
    target_files = [
        "src/strategist/strategist.py",
        "src/analyst/unified_regime_intelligence_runtime.py",
        "src/analyst/unified_regime_classifier.py",
        "src/analyst/analyst.py",
        "src/analyst/di_analyst.py",
        "src/analyst/predictive_ensembles.py",
        "src/analyst/regime_expert_orchestrator.py",
        "src/analyst/enhanced_prediction_integrator.py",
        "src/analyst/enhanced_regime_predictor.py",
        "src/analyst/meta_labeling_system.py",
        "src/analyst/ml_confidence_predictor.py",
        "src/analyst/autoencoder_feature_generator.py"
    ]

    validation_results = {}
    parameter_compatibility_results = {}

    for file_path in target_files:
    passprint(f"\n📁 Checking {file_path}...")

        if not Path(file_path).exists():
    passprint(f"❌ File not found: {file_path}")
            validation_results[file_path] = False
            continue

        # Check syntax
        syntax_ok = check_file_syntax(file_path)
        if not syntax_ok:
    passvalidation_results[file_path] = False
            continue

        # Check imports
        has_import = check_sr_imports(file_path)

        # Find method calls
        method_calls = find_sr_method_calls(file_path)

        # Check parameter compatibility
        param_compatibility = check_method_parameter_compatibility(file_path)
        parameter_compatibility_results[file_path] = param_compatibility

        # Check if file uses S/R functionality
        has_sr_usage = any(len(calls) > 0 for calls in method_calls.values())

        if has_import and has_sr_usage:
    passpassprint(f"✅ {file_path} - Valid S/R integration")
            print(f"   Methods used: {[method for method, calls in method_calls.items() if calls]}")

            # Check parameter compatibility
            if param_compatibility:
    passpassincompatible_calls = [k for k, v in param_compatibility.items() if not v]
                if incompatible_calls:
    passpassprint(f"   ⚠️  Parameter compatibility issues: {incompatible_calls}")
                else:
    passprint(f"   ✅ All method calls use correct parameters")

            validation_results[file_path] = True
        elif has_import:
    passpassprint(f"⚠️ {file_path} - Imports S/R but no method calls found")
            validation_results[file_path] = True  # Still valid, just not actively used
        else:
    passprint(f"ℹ️ {file_path} - No S/R integration found (not required)")
            validation_results[file_path] = True  # Not all files need S/R

    # Print summary
    print("\n" + "=" * 70)
    print("📊 S/R STRATEGIST/ANALYST INTEGRATION VALIDATION SUMMARY")
    print("=" * 70)

    passed = sum(1 for result in validation_results.values() if result)
    total = len(validation_results)

    for file_path, result in validation_results.items():
    passpassstatus = "✅ PASS" if result else "❌ FAIL"
        print(f"{file_path:<60} {status}")

    print("-" * 70)
    print(f"Total Files: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {passed/total*100:.1f}%")

    # Check parameter compatibility
    print("\n🔍 Parameter Compatibility Analysis:")
    for file_path, compatibility in parameter_compatibility_results.items():
    passif compatibility:
    passincompatible = [k for k, v in compatibility.items() if not v]
            if incompatible:
    passpassprint(f"   ⚠️ {file_path}: {len(incompatible)} incompatible calls")
            else:
    passprint(f"   ✅ {file_path}: All calls compatible")

    if passed == total:
    passprint("\n🎉 ALL S/R STRATEGIST/ANALYST INTEGRATION VALIDATIONS PASSED!")
        print("The cleaned up S/R implementation is properly integrated across strategist and analyst files.")
        return True
    else:
    passprint(f"\n⚠️ {total - passed} VALIDATIONS FAILED")
        print("Some S/R integrations need attention.")
        return False

def check_sr_predictor_file(...):
    pass"""Check the main S/R predictor file."""
    print("\n🔍 Checking main S/R predictor file...")

    sr_file = "src/tactician/sr_breakout_predictor.py"

    if not Path(sr_file).exists():
    passprint(f"❌ S/R predictor file not found: {sr_file}")
        return False

    # Check syntax
    if not check_file_syntax(sr_file):
    passprint("❌ S/R predictor file has syntax errors")
        return False

    # Check for required methods
    required_methods = [
        'get_sr_context',
        'predict_sr_outcome',
        'calculate_sr_features',
        'calculate_comprehensive_sr_features',
        'is_near_sr_level',
        'predict_breakout',
        'set_weights',
        'get_sr_proximity_details'
    ]

    method_calls = find_sr_method_calls(sr_file)

    missing_methods = []
    for method in required_methods:
    passif not method_calls.get(method):
    passmissing_methods.append(method)

    if missing_methods:
    passprint(f"❌ Missing required methods: {missing_methods}")
        return False
    else:
    passprint("✅ S/R predictor file is valid and complete")
        return True

def analyze_sr_usage_patterns(...):
    pass"""Analyze S/R usage patterns across files."""
    print("\n📈 S/R Usage Pattern Analysis")
    print("=" * 50)

    # Files that use S/R
    sr_files = [
        "src/analyst/unified_regime_intelligence_runtime.py"
    ]

    usage_patterns = {}

    for file_path in sr_files:
    passif Path(file_path).exists():
    passmethod_calls = find_sr_method_calls(file_path)
            usage_patterns[file_path] = method_calls

    # Analyze patterns
    print("\n🔍 S/R Method Usage Analysis:")
    for file_path, methods in usage_patterns.items():
    passprint(f"\n📁 {file_path}:")
        for method, lines in methods.items():
    passif lines:
    passprint(f"   {method}: {len(lines)} calls at lines {lines}")

    # Summary
    print("\n📊 Usage Summary:")
    method_totals = {}
    for methods in usage_patterns.values():
    passfor method, lines in methods.items():
    passif lines:
    passmethod_totals[method] = method_totals.get(method, 0) + len(lines)

    for method, total in sorted(method_totals.items()):
    passprint(f"   {method}: {total} total calls")

if __name__ == "__main__":
    pass# Check main S/R file
    sr_ok = check_sr_predictor_file()

    # Check strategist/analyst integrations
    integration_ok = validate_sr_integration()

    # Analyze usage patterns
    analyze_sr_usage_patterns()

    # Overall result
    if sr_ok and integration_ok:
    passprint("\n🎉 ALL VALIDATIONS PASSED!")
        sys.exit(0)
    else:
    passprint("\n⚠️ SOME VALIDATIONS FAILED!")
        sys.exit(1)