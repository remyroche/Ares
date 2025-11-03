"""
Simple Unit Tests for Hierarchical Optimization Configuration
==============================================================

Lightweight tests that verify the configuration structure without
requiring full system initialization.

Author: Ares Trading System
Date: 2025-10-31
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_imports():
    """Test that we can import the hierarchical optimization config."""
    try:
        from src.training.steps.backtesting.hierarchical_optimization_config import (
            STAGE_1_GROUPS,
            STAGE_2_GROUPS,
            STAGE_3_GROUPS,
            STAGE_4_GROUPS,
            STAGE_5_GROUPS,
            STAGE_CONFIGURATIONS,
            get_total_parameter_count,
            get_total_expected_trials,
        )
        print("✅ Successfully imported hierarchical optimization config")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_parameter_groups_exist():
    """Test that all parameter groups are defined."""
    from src.training.steps.backtesting.hierarchical_optimization_config import (
        STAGE_1_GROUPS,
        STAGE_2_GROUPS,
        STAGE_3_GROUPS,
        STAGE_4_GROUPS,
        STAGE_5_GROUPS,
    )
    
    assert len(STAGE_1_GROUPS) == 2, f"Expected 2 Stage 1 groups, got {len(STAGE_1_GROUPS)}"
    assert len(STAGE_2_GROUPS) == 1, f"Expected 1 Stage 2 group, got {len(STAGE_2_GROUPS)}"
    assert len(STAGE_3_GROUPS) == 2, f"Expected 2 Stage 3 groups, got {len(STAGE_3_GROUPS)}"
    assert len(STAGE_4_GROUPS) == 1, f"Expected 1 Stage 4 group, got {len(STAGE_4_GROUPS)}"
    assert len(STAGE_5_GROUPS) == 1, f"Expected 1 Stage 5 group, got {len(STAGE_5_GROUPS)}"
    
    total = len(STAGE_1_GROUPS) + len(STAGE_2_GROUPS) + len(STAGE_3_GROUPS) + len(STAGE_4_GROUPS) + len(STAGE_5_GROUPS)
    assert total == 7, f"Expected 7 total groups, got {total}"
    
    print(f"✅ All 7 parameter groups exist")
    return True


def test_group_names():
    """Test that groups have correct names."""
    from src.training.steps.backtesting.hierarchical_optimization_config import (
        STAGE_1_GROUPS,
        STAGE_2_GROUPS,
        STAGE_3_GROUPS,
        STAGE_4_GROUPS,
        STAGE_5_GROUPS,
    )
    
    expected_names = [
        "core_confidence",
        "entry_timing",
        "position_sizing_leverage",
        "unified_tpsl",
        "trailing_framework",
        "time_confidence_decay",
        "regime_intelligence"
    ]
    
    all_groups = STAGE_1_GROUPS + STAGE_2_GROUPS + STAGE_3_GROUPS + STAGE_4_GROUPS + STAGE_5_GROUPS
    actual_names = [g.name for g in all_groups]
    
    assert actual_names == expected_names, f"Group names mismatch: {actual_names} vs {expected_names}"
    print(f"✅ All group names correct")
    return True


def test_parameter_count():
    """Test that parameter count is reduced."""
    from src.training.steps.backtesting.hierarchical_optimization_config import get_total_parameter_count
    
    total = get_total_parameter_count()
    assert 40 <= total <= 50, f"Expected ~45 parameters, got {total}"
    
    reduction = (150 - total) / 150
    assert reduction >= 0.65, f"Expected 65%+ reduction, got {reduction:.1%}"
    
    print(f"✅ Parameter count: {total} (reduced from 150+, {reduction:.1%} reduction)")
    return True


def test_core_confidence_parameters():
    """Test core confidence group structure."""
    from src.training.steps.backtesting.hierarchical_optimization_config import STAGE_1_GROUPS
    
    core_conf = STAGE_1_GROUPS[0]
    assert core_conf.name == "core_confidence"
    
    # Test critical parameters exist
    required_params = [
        'tactician_confidence_threshold',
        'exit_confidence_threshold',
        'directional_confidence_min'
    ]
    
    for param in required_params:
        assert param in core_conf.params, f"Missing required parameter: {param}"
    
    # Test regime-aware parameters exist
    regime_params = [
        'trending_entry_threshold_multiplier',
        'ranging_entry_threshold_multiplier',
        'high_vol_entry_threshold_multiplier'
    ]
    
    for param in regime_params:
        assert param in core_conf.params, f"Missing regime parameter: {param}"
    
    print(f"✅ Core confidence has {len(core_conf.params)} parameters with regime awareness")
    return True


def test_obsolete_parameters_removed():
    """Test that obsolete parameters were removed."""
    from src.training.steps.backtesting.hierarchical_optimization_config import (
        STAGE_1_GROUPS,
        STAGE_2_GROUPS,
        STAGE_3_GROUPS,
        STAGE_4_GROUPS,
        STAGE_5_GROUPS,
    )
    
    all_groups = STAGE_1_GROUPS + STAGE_2_GROUPS + STAGE_3_GROUPS + STAGE_4_GROUPS + STAGE_5_GROUPS
    
    # Parameters that should NOT exist
    removed_params = [
        'micro_immediate_long_threshold',
        'micro_immediate_short_threshold',
        'exit_micro_immediate_long_threshold',
        'exit_micro_immediate_short_threshold',
        'analyst_tcn_weight',
        'tactician_xgboost_weight',
        'confidence_degradation_window'
    ]
    
    for group in all_groups:
        for param in removed_params:
            assert param not in group.params, f"Obsolete parameter '{param}' found in group '{group.name}'"
    
    print(f"✅ All obsolete parameters removed")
    return True


def test_unified_parameters():
    """Test that parameters were properly unified."""
    from src.training.steps.backtesting.hierarchical_optimization_config import STAGE_3_GROUPS
    
    tpsl = STAGE_3_GROUPS[0]  # unified_tpsl
    
    # Should have unified volatility scaling
    assert 'volatility_sl_scaling' in tpsl.params
    assert 'volatility_tp_scaling' in tpsl.params
    
    # Should NOT have separate low/normal/high vol parameters
    assert 'low_vol_sl_atr' not in tpsl.params
    assert 'normal_vol_sl_atr' not in tpsl.params
    assert 'high_vol_sl_atr' not in tpsl.params
    
    print(f"✅ Parameters properly unified (volatility regime: 12 → 3)")
    return True


def test_trailing_log_space():
    """Test that trailing uses log-space combination."""
    from src.training.steps.backtesting.hierarchical_optimization_config import STAGE_3_GROUPS
    
    trailing = STAGE_3_GROUPS[1]  # trailing_framework
    
    # Should have log-space weights
    assert 'trailing_log_confidence_weight' in trailing.params
    assert 'trailing_log_uncertainty_weight' in trailing.params
    assert 'trailing_log_volatility_weight' in trailing.params
    assert 'trailing_log_regime_weight' in trailing.params
    
    # Should have uncertainty multiplier
    assert 'trailing_uncertainty_multiplier' in trailing.params
    
    print(f"✅ Trailing framework uses log-space combination with uncertainty multiplier")
    return True


def test_dependencies():
    """Test that dependencies are properly defined."""
    from src.training.steps.backtesting.hierarchical_optimization_config import (
        STAGE_1_GROUPS,
        STAGE_2_GROUPS,
        STAGE_3_GROUPS,
    )
    
    # Core confidence has no dependencies
    assert STAGE_1_GROUPS[0].depends_on == []
    
    # Entry timing depends on core confidence
    assert "core_confidence" in STAGE_1_GROUPS[1].depends_on
    
    # Position sizing depends on core confidence
    assert "core_confidence" in STAGE_2_GROUPS[0].depends_on
    
    # Trailing depends on unified_tpsl
    assert "unified_tpsl" in STAGE_3_GROUPS[1].depends_on
    
    print(f"✅ Dependencies properly defined")
    return True


def test_stage_configurations():
    """Test that stage configurations are defined for all groups."""
    from src.training.steps.backtesting.hierarchical_optimization_config import STAGE_CONFIGURATIONS
    
    expected_groups = [
        "core_confidence",
        "entry_timing",
        "position_sizing_leverage",
        "unified_tpsl",
        "trailing_framework",
        "time_confidence_decay",
        "regime_intelligence"
    ]
    
    for group_name in expected_groups:
        assert group_name in STAGE_CONFIGURATIONS, f"Missing config for {group_name}"
        config = STAGE_CONFIGURATIONS[group_name]
        assert 'algorithm' in config
        assert 'n_trials' in config
        assert 'justification' in config
    
    print(f"✅ All 7 groups have stage configurations")
    return True


def test_algorithm_selection():
    """Test that algorithms are properly assigned."""
    from src.training.steps.backtesting.hierarchical_optimization_config import STAGE_CONFIGURATIONS
    
    # Core confidence should use TPE
    assert STAGE_CONFIGURATIONS["core_confidence"]["algorithm"] == "TPE"
    
    # Entry timing should use staged
    assert "Staged" in STAGE_CONFIGURATIONS["entry_timing"]["algorithm"]
    
    # Trailing should use BOHB
    assert STAGE_CONFIGURATIONS["trailing_framework"]["algorithm"] == "BOHB"
    
    # Time decay should use hybrid
    assert "Hybrid" in STAGE_CONFIGURATIONS["time_confidence_decay"]["algorithm"]
    
    print(f"✅ Algorithms properly assigned based on parameter nature")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("HIERARCHICAL OPTIMIZATION CONFIGURATION TESTS")
    print("=" * 80)
    print()
    
    tests = [
        ("Import Test", test_imports),
        ("Parameter Groups Exist", test_parameter_groups_exist),
        ("Group Names", test_group_names),
        ("Parameter Count", test_parameter_count),
        ("Core Confidence Parameters", test_core_confidence_parameters),
        ("Obsolete Parameters Removed", test_obsolete_parameters_removed),
        ("Unified Parameters", test_unified_parameters),
        ("Trailing Log-Space", test_trailing_log_space),
        ("Dependencies", test_dependencies),
        ("Stage Configurations", test_stage_configurations),
        ("Algorithm Selection", test_algorithm_selection),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            print(f"\n{'─' * 80}")
            print(f"Running: {name}")
            print(f"{'─' * 80}")
            result = test_func()
            if result:
                passed += 1
                print(f"✅ PASSED: {name}")
            else:
                failed += 1
                print(f"❌ FAILED: {name}")
        except AssertionError as e:
            failed += 1
            print(f"❌ FAILED: {name}")
            print(f"   Reason: {e}")
        except Exception as e:
            failed += 1
            print(f"❌ ERROR: {name}")
            print(f"   Error: {e}")
    
    print()
    print("=" * 80)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = run_all_tests()
    sys.exit(0 if failed == 0 else 1)

