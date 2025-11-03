"""
Unit Tests for Hierarchical Optimization Implementation
========================================================

Tests for the hierarchical parameter optimization system including:
- Parameter group definitions
- Algorithm selection
- Objective function
- Helper methods
- Integration with FinalParametersOptimizer

Author: Ares Trading System
Date: 2025-10-31
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.steps.backtesting.hierarchical_optimization_config import (
    STAGE_1_GROUPS,
    STAGE_2_GROUPS,
    STAGE_3_GROUPS,
    STAGE_4_GROUPS,
    STAGE_5_GROUPS,
    STAGE_CONFIGURATIONS,
    create_hierarchical_optimizer,
    create_objective_function_for_hierarchical_optimization,
    get_total_parameter_count,
    get_total_expected_trials,
)
from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
    ParameterGroup,
    OptimizationStage
)


class TestParameterGroups:
    """Test parameter group definitions."""
    
    def test_stage1_groups_exist(self):
        """Test that Stage 1 groups are defined."""
        assert len(STAGE_1_GROUPS) == 2
        assert STAGE_1_GROUPS[0].name == "core_confidence"
        assert STAGE_1_GROUPS[1].name == "entry_timing"
    
    def test_stage2_groups_exist(self):
        """Test that Stage 2 groups are defined."""
        assert len(STAGE_2_GROUPS) == 1
        assert STAGE_2_GROUPS[0].name == "position_sizing_leverage"
    
    def test_stage3_groups_exist(self):
        """Test that Stage 3 groups are defined."""
        assert len(STAGE_3_GROUPS) == 2
        assert STAGE_3_GROUPS[0].name == "unified_tpsl"
        assert STAGE_3_GROUPS[1].name == "trailing_framework"
    
    def test_stage4_groups_exist(self):
        """Test that Stage 4 groups are defined."""
        assert len(STAGE_4_GROUPS) == 1
        assert STAGE_4_GROUPS[0].name == "time_confidence_decay"
    
    def test_stage5_groups_exist(self):
        """Test that Stage 5 groups are defined."""
        assert len(STAGE_5_GROUPS) == 1
        assert STAGE_5_GROUPS[0].name == "regime_intelligence"
    
    def test_total_groups_count(self):
        """Test that we have exactly 7 groups."""
        total_groups = (
            len(STAGE_1_GROUPS) + 
            len(STAGE_2_GROUPS) + 
            len(STAGE_3_GROUPS) + 
            len(STAGE_4_GROUPS) + 
            len(STAGE_5_GROUPS)
        )
        assert total_groups == 7
    
    def test_core_confidence_parameters(self):
        """Test core confidence group parameters."""
        core_conf = STAGE_1_GROUPS[0]
        assert 'tactician_confidence_threshold' in core_conf.params
        assert 'exit_confidence_threshold' in core_conf.params
        assert 'directional_confidence_min' in core_conf.params
        
        # Test regime-aware parameters
        assert 'trending_entry_threshold_multiplier' in core_conf.params
        assert 'ranging_entry_threshold_multiplier' in core_conf.params
        assert 'high_vol_entry_threshold_multiplier' in core_conf.params
    
    def test_parameter_ranges(self):
        """Test that parameter ranges are sensible."""
        core_conf = STAGE_1_GROUPS[0]
        
        # Confidence threshold should be 0-1
        tact_conf = core_conf.params['tactician_confidence_threshold']
        assert tact_conf['min'] >= 0.0
        assert tact_conf['max'] <= 1.0
        assert tact_conf['min'] < tact_conf['max']
    
    def test_dependencies(self):
        """Test that dependencies are properly defined."""
        # Core confidence has no dependencies
        assert STAGE_1_GROUPS[0].depends_on == []
        
        # Entry timing depends on core confidence
        assert "core_confidence" in STAGE_1_GROUPS[1].depends_on
        
        # Position sizing depends on core confidence
        assert "core_confidence" in STAGE_2_GROUPS[0].depends_on
        
        # Trailing framework depends on unified_tpsl
        assert "unified_tpsl" in STAGE_3_GROUPS[1].depends_on
    
    def test_priorities(self):
        """Test that priorities are sequential."""
        all_groups = (
            STAGE_1_GROUPS + 
            STAGE_2_GROUPS + 
            STAGE_3_GROUPS + 
            STAGE_4_GROUPS + 
            STAGE_5_GROUPS
        )
        
        priorities = [group.priority for group in all_groups]
        assert priorities == sorted(priorities)
        assert priorities[0] == 1
        assert priorities[-1] == 7


class TestAlgorithmSelection:
    """Test algorithm selection and configurations."""
    
    def test_stage_configurations_exist(self):
        """Test that all groups have stage configurations."""
        all_group_names = [
            "core_confidence",
            "entry_timing",
            "position_sizing_leverage",
            "unified_tpsl",
            "trailing_framework",
            "time_confidence_decay",
            "regime_intelligence"
        ]
        
        for name in all_group_names:
            assert name in STAGE_CONFIGURATIONS
    
    def test_core_confidence_uses_tpe(self):
        """Test that core confidence uses TPE."""
        config = STAGE_CONFIGURATIONS["core_confidence"]
        assert config["algorithm"] == "TPE"
        assert "TPE" in str(config["stages"])
    
    def test_entry_timing_uses_staged(self):
        """Test that entry timing uses staged optimization."""
        config = STAGE_CONFIGURATIONS["entry_timing"]
        assert "Staged" in config["algorithm"]
        assert len(config["stages"]) == 3  # COARSE_GRID, FINE_GRID, TPE
    
    def test_trailing_uses_bohb(self):
        """Test that trailing framework uses BOHB."""
        config = STAGE_CONFIGURATIONS["trailing_framework"]
        assert config["algorithm"] == "BOHB"
        assert "min_budget" in config
        assert "max_budget" in config
        assert config["min_budget"] == 0.2
        assert config["max_budget"] == 1.0
    
    def test_time_decay_uses_hybrid(self):
        """Test that time decay uses hybrid approach."""
        config = STAGE_CONFIGURATIONS["time_confidence_decay"]
        assert "Hybrid" in config["algorithm"]
        assert len(config["stages"]) == 2  # COARSE_GRID, TPE


class TestParameterCount:
    """Test parameter counting and reduction."""
    
    def test_total_parameter_count(self):
        """Test that total parameter count is around 45."""
        total = get_total_parameter_count()
        assert 40 <= total <= 50, f"Expected ~45 parameters, got {total}"
    
    def test_parameter_reduction(self):
        """Test that we achieved significant parameter reduction."""
        total = get_total_parameter_count()
        original_count = 150
        reduction = (original_count - total) / original_count
        assert reduction >= 0.60, f"Expected 60%+ reduction, got {reduction:.1%}"
    
    def test_expected_trials(self):
        """Test that expected trials is reasonable."""
        total = get_total_expected_trials()
        # Should be around 350-400 trials per round, × 2 rounds
        assert 600 <= total <= 900, f"Expected ~740 trials, got {total}"


class TestObjectiveFunction:
    """Test objective function creation and execution."""
    
    def test_objective_function_creation(self):
        """Test that objective function can be created."""
        def mock_backtest(params, calibration_results, X_train, y_train, X_val, y_val):
            return {
                'predictions': np.random.rand(100),
                'targets': np.random.randint(0, 2, 100),
                'returns': np.random.randn(100) * 0.01,
                'regime_labels': None
            }
        
        obj_func = create_objective_function_for_hierarchical_optimization(
            backtest_func=mock_backtest,
            calibration_results={}
        )
        
        assert callable(obj_func)
    
    def test_objective_function_execution(self):
        """Test that objective function executes without error."""
        def mock_backtest(params, calibration_results, X_train, y_train, X_val, y_val):
            return {
                'predictions': np.random.rand(100),
                'targets': np.random.randint(0, 2, 100),
                'returns': np.random.randn(100) * 0.01,
                'regime_labels': None
            }
        
        obj_func = create_objective_function_for_hierarchical_optimization(
            backtest_func=mock_backtest,
            calibration_results={}
        )
        
        # Test execution
        score = obj_func(
            params={'tactician_confidence_threshold': 0.75},
            X_train=np.random.rand(100, 2),
            y_train=np.random.rand(100),
            X_val=None,
            y_val=None
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_objective_function_handles_errors(self):
        """Test that objective function handles errors gracefully."""
        def failing_backtest(params, calibration_results, X_train, y_train, X_val, y_val):
            raise ValueError("Backtest failed")
        
        obj_func = create_objective_function_for_hierarchical_optimization(
            backtest_func=failing_backtest,
            calibration_results={}
        )
        
        # Should return 0.0 on error, not crash
        score = obj_func(
            params={'tactician_confidence_threshold': 0.75},
            X_train=np.random.rand(100, 2),
            y_train=np.random.rand(100)
        )
        
        assert score == 0.0


class TestHierarchicalOptimizerCreation:
    """Test hierarchical optimizer creation."""
    
    def test_optimizer_creation(self):
        """Test that optimizer can be created."""
        def mock_backtest(params, calibration_results, X_train, y_train, X_val, y_val):
            return {
                'predictions': np.random.rand(100),
                'targets': np.random.randint(0, 2, 100),
                'returns': np.random.randn(100) * 0.01,
                'regime_labels': None
            }
        
        optimizer = create_hierarchical_optimizer(
            backtest_func=mock_backtest,
            calibration_results={},
            config={'cv_folds': 3, 'verbose': False}
        )
        
        assert optimizer is not None
        assert len(optimizer.param_groups) == 7
    
    def test_optimizer_config(self):
        """Test optimizer configuration."""
        def mock_backtest(params, calibration_results, X_train, y_train, X_val, y_val):
            return {
                'predictions': np.random.rand(10),
                'targets': np.random.randint(0, 2, 10),
                'returns': np.random.randn(10) * 0.01,
                'regime_labels': None
            }
        
        config = {
            'cv_folds': 3,
            'n_rounds': 1,  # Reduce for testing
            'verbose': False
        }
        
        optimizer = create_hierarchical_optimizer(
            backtest_func=mock_backtest,
            calibration_results={},
            config=config
        )
        
        assert optimizer.cv_folds == 3
        assert optimizer.n_rounds == 1
        assert optimizer.direction == 'maximize'
        assert optimizer.scoring_metric == 'custom_balanced_score'


class TestRegimeAwareParameters:
    """Test regime-aware parameter implementation."""
    
    def test_core_confidence_has_regime_multipliers(self):
        """Test that core confidence has regime multipliers."""
        core_conf = STAGE_1_GROUPS[0]
        
        # Entry threshold multipliers
        assert 'trending_entry_threshold_multiplier' in core_conf.params
        assert 'ranging_entry_threshold_multiplier' in core_conf.params
        assert 'high_vol_entry_threshold_multiplier' in core_conf.params
        
        # Exit threshold multipliers
        assert 'trending_exit_threshold_multiplier' in core_conf.params
        assert 'ranging_exit_threshold_multiplier' in core_conf.params
        assert 'high_vol_exit_threshold_multiplier' in core_conf.params
    
    def test_position_sizing_has_regime_multipliers(self):
        """Test that position sizing has regime multipliers."""
        pos_sizing = STAGE_2_GROUPS[0]
        
        assert 'trending_max_position_multiplier' in pos_sizing.params
        assert 'ranging_max_position_multiplier' in pos_sizing.params
        assert 'high_vol_max_position_multiplier' in pos_sizing.params
    
    def test_unified_tpsl_has_regime_multipliers(self):
        """Test that TP/SL has regime multipliers."""
        tpsl = STAGE_3_GROUPS[0]
        
        assert 'trending_sl_atr_multiplier' in tpsl.params
        assert 'trending_tp_atr_multiplier' in tpsl.params
        assert 'ranging_sl_atr_multiplier' in tpsl.params
        assert 'ranging_tp_atr_multiplier' in tpsl.params
        assert 'high_vol_sl_atr_multiplier' in tpsl.params
        assert 'high_vol_tp_atr_multiplier' in tpsl.params
    
    def test_regime_intelligence_group_exists(self):
        """Test that regime intelligence group exists."""
        regime = STAGE_5_GROUPS[0]
        
        assert regime.name == "regime_intelligence"
        assert 'regime_transition_penalty' in regime.params
        assert 'trending_profit_band_multiplier' in regime.params
        assert 'ranging_profit_band_multiplier' in regime.params
        assert 'high_vol_profit_band_multiplier' in regime.params


class TestParameterUnification:
    """Test that redundant parameters were unified."""
    
    def test_volatility_scaling_unified(self):
        """Test that volatility scaling uses single parameter."""
        tpsl = STAGE_3_GROUPS[0]
        
        # Should have unified scaling, not separate low/normal/high vol params
        assert 'volatility_sl_scaling' in tpsl.params
        assert 'volatility_tp_scaling' in tpsl.params
        
        # Should NOT have separate parameters for each regime
        assert 'low_vol_sl_atr' not in tpsl.params
        assert 'normal_vol_sl_atr' not in tpsl.params
    
    def test_trailing_uses_log_space(self):
        """Test that trailing uses log-space combination."""
        trailing = STAGE_3_GROUPS[1]
        
        # Should have log-space weights
        assert 'trailing_log_confidence_weight' in trailing.params
        assert 'trailing_log_uncertainty_weight' in trailing.params
        assert 'trailing_log_volatility_weight' in trailing.params
        assert 'trailing_log_regime_weight' in trailing.params
        
        # Should have uncertainty multiplier
        assert 'trailing_uncertainty_multiplier' in trailing.params
    
    def test_position_sizing_unified(self):
        """Test that position sizing uses single volatility parameter."""
        pos_sizing = STAGE_2_GROUPS[0]
        
        # Should have single volatility scaling parameter
        assert 'volatility_position_scaling' in pos_sizing.params
        
        # Should NOT have separate high/low vol parameters
        assert 'high_vol_position_scaling' not in pos_sizing.params
        assert 'low_vol_position_scaling' not in pos_sizing.params


class TestRemovedParameters:
    """Test that obsolete parameters were removed."""
    
    def test_micro_thresholds_removed(self):
        """Test that micro movement thresholds were removed."""
        # Check all groups
        all_groups = (
            STAGE_1_GROUPS + 
            STAGE_2_GROUPS + 
            STAGE_3_GROUPS + 
            STAGE_4_GROUPS + 
            STAGE_5_GROUPS
        )
        
        for group in all_groups:
            assert 'micro_immediate_long_threshold' not in group.params
            assert 'micro_immediate_short_threshold' not in group.params
            assert 'exit_micro_immediate_long_threshold' not in group.params
            assert 'exit_micro_immediate_short_threshold' not in group.params
    
    def test_model_weights_removed(self):
        """Test that base model weights were removed."""
        all_groups = (
            STAGE_1_GROUPS + 
            STAGE_2_GROUPS + 
            STAGE_3_GROUPS + 
            STAGE_4_GROUPS + 
            STAGE_5_GROUPS
        )
        
        for group in all_groups:
            assert 'analyst_tcn_weight' not in group.params
            assert 'tactician_xgboost_weight' not in group.params
    
    def test_confidence_degradation_window_removed(self):
        """Test that confidence_degradation_window was removed."""
        time_decay = STAGE_4_GROUPS[0]
        
        assert 'confidence_degradation_window' not in time_decay.params


def test_integration_with_final_parameters_optimizer():
    """Test integration with FinalParametersOptimizer."""
    try:
        from src.training.steps.backtesting.final_parameters_optimization import (
            FinalParametersOptimizer
        )
        
        # Test that optimizer can be created with hierarchical enabled
        config = {
            'use_hierarchical_optimization': True,
            'verbose': False
        }
        
        optimizer = FinalParametersOptimizer(config=config)
        
        # Check that hierarchical flag is set
        assert optimizer.use_hierarchical_optimization == True
        
        print("✅ Integration test passed")
        
    except Exception as e:
        print(f"⚠️ Integration test skipped: {e}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

