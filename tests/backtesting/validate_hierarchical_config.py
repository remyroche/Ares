"""
Minimal Validation Script for Hierarchical Optimization Configuration
======================================================================

This script validates the configuration structure by directly reading
the config file without importing the full system.

Author: Ares Trading System
Date: 2025-10-31
"""

import sys
import ast
from pathlib import Path


def validate_config_file():
    """Validate the hierarchical optimization config file structure."""
    
    config_file = Path(__file__).parent.parent.parent / "src" / "training" / "steps" / "backtesting" / "hierarchical_optimization_config.py"
    
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        return False
    
    print(f"✅ Config file exists: {config_file.name}")
    
    # Read file content
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Check for required constants
    required_constants = [
        'STAGE_1_GROUPS',
        'STAGE_2_GROUPS',
        'STAGE_3_GROUPS',
        'STAGE_4_GROUPS',
        'STAGE_5_GROUPS',
        'STAGE_CONFIGURATIONS',
        'FINAL_REFINEMENT_CONFIG'
    ]
    
    for const in required_constants:
        if const in content:
            print(f"✅ Found constant: {const}")
        else:
            print(f"❌ Missing constant: {const}")
            return False
    
    # Check for required functions
    required_functions = [
        'create_hierarchical_optimizer',
        'create_objective_function_for_hierarchical_optimization',
        'get_total_parameter_count',
        'get_total_expected_trials'
    ]
    
    for func in required_functions:
        if f"def {func}" in content:
            print(f"✅ Found function: {func}")
        else:
            print(f"❌ Missing function: {func}")
            return False
    
    # Count STAGE groups
    stage_counts = {
        'STAGE_1_GROUPS': content.count('ParameterGroup(') if 'STAGE_1_GROUPS' in content else 0,
        'STAGE_2_GROUPS': 0,
        'STAGE_3_GROUPS': 0,
        'STAGE_4_GROUPS': 0,
        'STAGE_5_GROUPS': 0
    }
    
    # Parse to count groups per stage (rough estimate)
    stage_1_start = content.find('STAGE_1_GROUPS = [')
    stage_2_start = content.find('STAGE_2_GROUPS = [')
    
    if stage_1_start > 0 and stage_2_start > 0:
        stage_1_section = content[stage_1_start:stage_2_start]
        stage_1_count = stage_1_section.count('ParameterGroup(')
        print(f"✅ STAGE_1_GROUPS contains ~{stage_1_count} groups")
    
    # Check for regime-aware parameters
    regime_patterns = [
        'trending_',
        'ranging_',
        'high_vol_'
    ]
    
    regime_count = sum(content.count(pattern) for pattern in regime_patterns)
    print(f"✅ Found {regime_count} regime-aware parameters")
    
    # Check for removed parameters (should NOT exist)
    removed_params = [
        'micro_immediate_long_threshold',
        'micro_immediate_short_threshold',
        'analyst_tcn_weight',
        'tactician_xgboost_weight',
        'confidence_degradation_window'
    ]
    
    found_removed = []
    for param in removed_params:
        if f"'{param}'" in content or f'"{param}"' in content:
            found_removed.append(param)
    
    if found_removed:
        print(f"⚠️  Found obsolete parameters: {found_removed}")
    else:
        print(f"✅ No obsolete parameters found")
    
    # Check for unified parameters (should exist)
    unified_params = [
        'volatility_sl_scaling',
        'volatility_tp_scaling',
        'volatility_position_scaling',
        'trailing_log_confidence_weight',
        'trailing_uncertainty_multiplier'
    ]
    
    found_unified = []
    for param in unified_params:
        if f"'{param}'" in content or f'"{param}"' in content:
            found_unified.append(param)
    
    print(f"✅ Found {len(found_unified)}/{len(unified_params)} unified parameters")
    
    # Check for custom_balanced_score usage
    if 'custom_balanced_score' in content:
        print(f"✅ Uses custom_balanced_score as objective")
    else:
        print(f"⚠️  custom_balanced_score not found in config")
    
    # Check for algorithm types
    algorithms = ['TPE', 'BOHB', 'COARSE_GRID', 'FINE_GRID']
    for algo in algorithms:
        if algo in content:
            print(f"✅ Uses {algo} algorithm")
    
    return True


def validate_integration():
    """Validate integration with final_parameters_optimization.py."""
    
    main_file = Path(__file__).parent.parent.parent / "src" / "training" / "steps" / "backtesting" / "final_parameters_optimization.py"
    
    if not main_file.exists():
        print(f"❌ Main file not found: {main_file}")
        return False
    
    print(f"✅ Main file exists: {main_file.name}")
    
    with open(main_file, 'r') as f:
        content = f.read()
    
    # Check for hierarchical optimization integration
    if 'use_hierarchical_optimization' in content:
        print(f"✅ Found use_hierarchical_optimization flag")
    else:
        print(f"❌ Missing use_hierarchical_optimization flag")
        return False
    
    # Check for helper methods
    helper_methods = [
        '_prepare_data_for_hierarchical_optimization',
        '_run_backtest_for_hierarchical_optimization',
        '_convert_hierarchical_to_category_format'
    ]
    
    for method in helper_methods:
        if f"def {method}" in content:
            print(f"✅ Found helper method: {method}")
        else:
            print(f"❌ Missing helper method: {method}")
            return False
    
    # Check for import of hierarchical config
    if 'from src.training.steps.backtesting.hierarchical_optimization_config import' in content:
        print(f"✅ Imports hierarchical_optimization_config")
    else:
        print(f"❌ Does not import hierarchical_optimization_config")
        return False
    
    return True


def main():
    """Run all validations."""
    print("=" * 80)
    print("HIERARCHICAL OPTIMIZATION CONFIGURATION VALIDATION")
    print("=" * 80)
    print()
    
    print("─" * 80)
    print("1. CONFIGURATION FILE VALIDATION")
    print("─" * 80)
    config_valid = validate_config_file()
    
    print()
    print("─" * 80)
    print("2. INTEGRATION VALIDATION")
    print("─" * 80)
    integration_valid = validate_integration()
    
    print()
    print("=" * 80)
    if config_valid and integration_valid:
        print("✅ ALL VALIDATIONS PASSED")
        print("=" * 80)
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())

