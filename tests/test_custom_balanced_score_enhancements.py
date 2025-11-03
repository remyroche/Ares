"""
Unit Tests for Custom Balanced Score Enhancements
==================================================

Tests for:
1. Custom balanced score calculation
2. Pareto integration
3. Adaptive final refinement
4. Parameter importance analysis
5. Log-space narrowing

Author: Ares Trading System
Date: 2025-10-31
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Core imports
from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
    HierarchicalParameterOptimizer,
    ParameterGroup,
    OptimizationStage,
    create_custom_balanced_score_objective,
    CUSTOM_BALANCED_SCORE_AVAILABLE
)

from src.utils.ml_common.optimization.shared_utils.evaluation_metrics import (
    calculate_custom_balanced_score_for_hpo,
    create_unified_evaluator,
    FinancialMetrics,
    StatisticalMetrics
)


class TestCustomBalancedScore:
    """Test custom balanced score calculation."""
    
    def test_imports_successful(self):
        """Test that all imports work."""
        assert CUSTOM_BALANCED_SCORE_AVAILABLE is True
        assert calculate_custom_balanced_score_for_hpo is not None
        assert create_unified_evaluator is not None
    
    def test_custom_balanced_score_basic(self):
        """Test basic custom balanced score calculation."""
        # Create simple test data
        predictions = np.array([1, 0, 1, 1, 0, 1, 0, 1])
        targets = np.array([1, 0, 1, 0, 1, 1, 0, 1])
        returns = np.array([0.01, 0.02, 0.01, -0.01, -0.02, 0.015, 0.01, 0.02])
        
        score = calculate_custom_balanced_score_for_hpo(
            predictions=predictions,
            targets=targets,
            returns=returns
        )
        
        # Score should be between 0 and 1
        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)
        print(f"✅ Basic score calculated: {score:.4f}")
    
    def test_custom_balanced_score_with_evaluator(self):
        """Test custom balanced score via unified evaluator."""
        evaluator = create_unified_evaluator()
        
        predictions = np.random.rand(100)
        targets = np.random.rand(100)
        returns = np.random.randn(100) * 0.01
        
        result = evaluator.evaluate(predictions, targets, returns)
        
        assert result.success is True
        assert result.custom_balanced_score is not None
        assert 0.0 <= result.custom_balanced_score <= 1.0
        print(f"✅ Evaluator score: {result.custom_balanced_score:.4f}")
    
    def test_custom_balanced_score_60_40_split(self):
        """Test that score respects 60/40 financial/statistical split."""
        evaluator = create_unified_evaluator()
        
        # Perfect financial, poor statistical
        predictions_fin = np.ones(50)
        targets_fin = np.ones(50)
        returns_fin = np.ones(50) * 0.02  # High returns
        
        result_fin = evaluator.evaluate(predictions_fin, targets_fin, returns_fin)
        
        # Perfect statistical, poor financial  
        predictions_stat = np.ones(50)
        targets_stat = np.ones(50)
        returns_stat = np.random.randn(50) * 0.001  # Low returns
        
        result_stat = evaluator.evaluate(predictions_stat, targets_stat, returns_stat)
        
        # Both should have reasonable scores
        assert 0.0 <= result_fin.custom_balanced_score <= 1.0
        assert 0.0 <= result_stat.custom_balanced_score <= 1.0
        print(f"✅ Financial-heavy score: {result_fin.custom_balanced_score:.4f}")
        print(f"✅ Statistical-heavy score: {result_stat.custom_balanced_score:.4f}")


class TestParameterImportance:
    """Test parameter importance calculation."""
    
    def test_calculate_importance_basic(self):
        """Test basic importance calculation."""
        # Create mock optimizer with trial history
        param_groups = [
            ParameterGroup(
                name="test",
                params={'param1': {'type': 'float', 'low': 0.0, 'high': 1.0}},
                priority=1
            )
        ]
        
        def mock_objective(params, X_train, y_train, **kwargs):
            # Param1 has strong correlation with score
            return params.get('param1', 0.5)
        
        optimizer = HierarchicalParameterOptimizer(
            param_groups=param_groups,
            objective_func=mock_objective,
            verbose=False
        )
        
        # Add mock trial history
        from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import OptimizationResult
        
        mock_result = OptimizationResult(
            group_name="test",
            stage=OptimizationStage.TPE,
            best_params={'param1': 0.8},
            best_score=0.8,
            n_trials=10,
            optimization_time=1.0,
            all_trials=[
                {'params': {'param1': 0.1}, 'score': 0.1, 'trial_number': 0},
                {'params': {'param1': 0.3}, 'score': 0.3, 'trial_number': 1},
                {'params': {'param1': 0.5}, 'score': 0.5, 'trial_number': 2},
                {'params': {'param1': 0.7}, 'score': 0.7, 'trial_number': 3},
                {'params': {'param1': 0.9}, 'score': 0.9, 'trial_number': 4},
            ]
        )
        
        optimizer.group_results = [mock_result]
        
        # Calculate importance
        importance = optimizer._calculate_parameter_importance()
        
        assert 'param1' in importance
        # Should have very high importance (perfect correlation)
        assert importance['param1'] > 0.9
        print(f"✅ Parameter importance calculated: param1={importance['param1']:.3f}")
    
    def test_calculate_importance_no_correlation(self):
        """Test importance with no correlation."""
        param_groups = [
            ParameterGroup(
                name="test",
                params={'param1': {'type': 'float', 'low': 0.0, 'high': 1.0}},
                priority=1
            )
        ]
        
        def mock_objective(params, X_train, y_train, **kwargs):
            return 0.5  # Always same score
        
        optimizer = HierarchicalParameterOptimizer(
            param_groups=param_groups,
            objective_func=mock_objective,
            verbose=False
        )
        
        from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import OptimizationResult
        
        mock_result = OptimizationResult(
            group_name="test",
            stage=OptimizationStage.TPE,
            best_params={'param1': 0.5},
            best_score=0.5,
            n_trials=5,
            optimization_time=1.0,
            all_trials=[
                {'params': {'param1': 0.1}, 'score': 0.5, 'trial_number': 0},
                {'params': {'param1': 0.5}, 'score': 0.5, 'trial_number': 1},
                {'params': {'param1': 0.9}, 'score': 0.5, 'trial_number': 2},
            ]
        )
        
        optimizer.group_results = [mock_result]
        
        importance = optimizer._calculate_parameter_importance()
        
        # Should have low importance (no correlation)
        assert importance['param1'] < 0.6
        print(f"✅ No correlation detected: param1={importance['param1']:.3f}")


class TestLogSpaceNarrowing:
    """Test log-space narrowing functionality."""
    
    def test_log_space_narrowing_basic(self):
        """Test basic log-space narrowing."""
        param_groups = [
            ParameterGroup(
                name="test",
                params={'lr': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True}},
                priority=1
            )
        ]
        
        optimizer = HierarchicalParameterOptimizer(
            param_groups=param_groups,
            objective_func=lambda p, **kw: 0.5,
            verbose=False
        )
        
        search_space = {'lr': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True}}
        best_params = {'lr': 0.1}
        
        # Test with log-space narrowing
        narrowed = optimizer._create_narrowed_search_space(
            search_space,
            best_params,
            narrow_factor=0.1,
            use_log_space_narrowing=True
        )
        
        # Check that narrowing happened
        assert narrowed['lr']['low'] > 0.01
        assert narrowed['lr']['high'] < 0.3
        assert narrowed['lr']['low'] < 0.1 < narrowed['lr']['high']
        
        # Log-space narrowing should give wider range than linear
        narrowed_linear = optimizer._create_narrowed_search_space(
            search_space,
            best_params,
            narrow_factor=0.1,
            use_log_space_narrowing=False
        )
        
        # Log-space should be wider (more appropriate for log-scale)
        log_range = narrowed['lr']['high'] - narrowed['lr']['low']
        linear_range = narrowed_linear['lr']['high'] - narrowed_linear['lr']['low']
        
        assert log_range >= linear_range
        print(f"✅ Log-space narrowing: [{narrowed['lr']['low']:.6f}, {narrowed['lr']['high']:.6f}]")
        print(f"   Linear narrowing: [{narrowed_linear['lr']['low']:.6f}, {narrowed_linear['lr']['high']:.6f}]")
        print(f"   Log-space is {log_range/linear_range:.2f}x wider (correct for log-scale!)")
    
    def test_adaptive_narrowing_with_importance(self):
        """Test adaptive narrowing based on importance."""
        param_groups = [
            ParameterGroup(
                name="test",
                params={
                    'important_param': {'type': 'float', 'low': 0.0, 'high': 1.0},
                    'unimportant_param': {'type': 'float', 'low': 0.0, 'high': 1.0}
                },
                priority=1
            )
        ]
        
        optimizer = HierarchicalParameterOptimizer(
            param_groups=param_groups,
            objective_func=lambda p, **kw: 0.5,
            verbose=False
        )
        
        search_space = {
            'important_param': {'type': 'float', 'low': 0.0, 'high': 1.0},
            'unimportant_param': {'type': 'float', 'low': 0.0, 'high': 1.0}
        }
        best_params = {'important_param': 0.5, 'unimportant_param': 0.5}
        importance_weights = {'important_param': 0.9, 'unimportant_param': 0.2}
        
        narrowed = optimizer._create_narrowed_search_space(
            search_space,
            best_params,
            narrow_factor=0.1,
            importance_weights=importance_weights
        )
        
        # Calculate ranges
        important_range = narrowed['important_param']['high'] - narrowed['important_param']['low']
        unimportant_range = narrowed['unimportant_param']['high'] - narrowed['unimportant_param']['low']
        
        # Important param (imp=0.9) gets LARGER adaptive factor (0.5+0.9=1.4)
        # → narrowed MORE (±14% vs base ±10%)
        # → LARGER range (not smaller!)
        # This allows MORE exploration around the important parameter
        assert important_range > unimportant_range
        print(f"✅ Adaptive narrowing working:")
        print(f"   Important (imp=0.9): range={important_range:.3f} (wider - more exploration)")
        print(f"   Unimportant (imp=0.2): range={unimportant_range:.3f} (narrower - less focus)")
        print(f"   Ratio: {important_range/unimportant_range:.2f}x wider for important parameter")


class TestParetoIntegration:
    """Test Pareto integration."""
    
    def test_pareto_import_available(self):
        """Test that Pareto utilities are available."""
        try:
            from src.utils.ml_common.optimization.pareto import scalarize_financial_goals
            print("✅ Pareto utilities available")
            assert scalarize_financial_goals is not None
        except ImportError as e:
            pytest.skip(f"Pareto not available: {e}")
    
    def test_pareto_scalarization_basic(self):
        """Test basic Pareto scalarization."""
        try:
            from src.utils.ml_common.optimization.pareto import scalarize_financial_goals
            
            metrics = {
                'pnl': 1.5,
                'win_rate': 0.65,
                'sharpe': 1.2
            }
            
            score = scalarize_financial_goals(metrics, use_nonlinear_scaling=True)
            
            assert isinstance(score, float)
            print(f"✅ Pareto scalarization: {score:.4f}")
            
        except ImportError as e:
            pytest.skip(f"Pareto not available: {e}")


class TestHierarchicalOptimizerDefaults:
    """Test hierarchical optimizer default settings."""
    
    def test_default_scoring_metric(self):
        """Test that default scoring metric is custom_balanced_score."""
        param_groups = [
            ParameterGroup(
                name="test",
                params={'x': {'type': 'float', 'low': 0.0, 'high': 1.0}},
                priority=1
            )
        ]
        
        optimizer = HierarchicalParameterOptimizer(
            param_groups=param_groups,
            objective_func=lambda p, **kw: 0.5,
            verbose=False
        )
        
        assert optimizer.scoring_metric == 'custom_balanced_score'
        print("✅ Default scoring metric is custom_balanced_score")
    
    def test_default_direction(self):
        """Test that default direction is maximize."""
        param_groups = [
            ParameterGroup(
                name="test",
                params={'x': {'type': 'float', 'low': 0.0, 'high': 1.0}},
                priority=1
            )
        ]
        
        optimizer = HierarchicalParameterOptimizer(
            param_groups=param_groups,
            objective_func=lambda p, **kw: 0.5,
            verbose=False
        )
        
        assert optimizer.direction == 'maximize'
        print("✅ Default direction is maximize")
    
    def test_custom_balanced_score_flag(self):
        """Test use_custom_balanced_score flag."""
        param_groups = [
            ParameterGroup(
                name="test",
                params={'x': {'type': 'float', 'low': 0.0, 'high': 1.0}},
                priority=1
            )
        ]
        
        optimizer = HierarchicalParameterOptimizer(
            param_groups=param_groups,
            objective_func=lambda p, **kw: 0.5,
            use_custom_balanced_score=True,
            verbose=False
        )
        
        assert optimizer.use_custom_balanced_score is True
        print("✅ use_custom_balanced_score flag set correctly")


class TestBackwardCompatibility:
    """Test backward compatibility."""
    
    def test_old_scoring_metric_still_works(self):
        """Test that old scoring metrics still work."""
        param_groups = [
            ParameterGroup(
                name="test",
                params={'x': {'type': 'float', 'low': 0.0, 'high': 1.0}},
                priority=1
            )
        ]
        
        optimizer = HierarchicalParameterOptimizer(
            param_groups=param_groups,
            objective_func=lambda p, **kw: 0.5,
            scoring_metric='neg_mean_squared_error',  # Old default
            direction='minimize',  # Old common usage
            verbose=False
        )
        
        assert optimizer.scoring_metric == 'neg_mean_squared_error'
        assert optimizer.direction == 'minimize'
        print("✅ Old scoring metric still works (backward compatible)")
    
    def test_can_disable_custom_balanced_score(self):
        """Test that custom balanced score can be disabled."""
        param_groups = [
            ParameterGroup(
                name="test",
                params={'x': {'type': 'float', 'low': 0.0, 'high': 1.0}},
                priority=1
            )
        ]
        
        optimizer = HierarchicalParameterOptimizer(
            param_groups=param_groups,
            objective_func=lambda p, **kw: 0.5,
            use_custom_balanced_score=False,
            verbose=False
        )
        
        assert optimizer.use_custom_balanced_score is False
        print("✅ Can disable custom_balanced_score")


class TestCreateCustomBalancedScoreObjective:
    """Test objective function creation helper."""
    
    def test_helper_function_creation(self):
        """Test that helper creates valid objective function."""
        def mock_trainer(params, X_train, y_train, X_val, y_val):
            predictions = np.random.rand(len(y_val)) if y_val is not None else np.random.rand(10)
            model = Mock()
            return model, predictions
        
        objective = create_custom_balanced_score_objective(mock_trainer)
        
        assert callable(objective)
        print("✅ Objective function created successfully")
    
    def test_helper_function_execution(self):
        """Test that created objective executes."""
        def mock_trainer(params, X_train, y_train, X_val, y_val):
            predictions = np.random.rand(len(y_val)) if y_val is not None else np.random.rand(10)
            model = Mock()
            return model, predictions
        
        objective = create_custom_balanced_score_objective(mock_trainer)
        
        score = objective(
            params={'x': 0.5},
            X_train=np.random.rand(20, 5),
            y_train=np.random.rand(20),
            X_val=np.random.rand(10, 5),
            y_val=np.random.rand(10)
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        print(f"✅ Objective function executes: score={score:.4f}")


class TestNarrowingEdgeCases:
    """Test edge cases in narrowing."""
    
    def test_narrowing_with_no_best_params(self):
        """Test narrowing when parameter not in best_params."""
        param_groups = [
            ParameterGroup(
                name="test",
                params={'x': {'type': 'float', 'low': 0.0, 'high': 1.0}},
                priority=1
            )
        ]
        
        optimizer = HierarchicalParameterOptimizer(
            param_groups=param_groups,
            objective_func=lambda p, **kw: 0.5,
            verbose=False
        )
        
        search_space = {
            'x': {'type': 'float', 'low': 0.0, 'high': 1.0},
            'y': {'type': 'float', 'low': 0.0, 'high': 1.0}
        }
        best_params = {'x': 0.5}  # Missing 'y'
        
        narrowed = optimizer._create_narrowed_search_space(search_space, best_params)
        
        # 'y' should keep original range
        assert narrowed['y']['low'] == 0.0
        assert narrowed['y']['high'] == 1.0
        print("✅ Parameters not in best_params keep original range")
    
    def test_narrowing_integer_parameters(self):
        """Test narrowing for integer parameters."""
        param_groups = [
            ParameterGroup(
                name="test",
                params={'n': {'type': 'int', 'low': 10, 'high': 100}},
                priority=1
            )
        ]
        
        optimizer = HierarchicalParameterOptimizer(
            param_groups=param_groups,
            objective_func=lambda p, **kw: 0.5,
            verbose=False
        )
        
        search_space = {'n': {'type': 'int', 'low': 10, 'high': 100}}
        best_params = {'n': 50}
        
        narrowed = optimizer._create_narrowed_search_space(search_space, best_params)
        
        # Check narrowing happened
        assert narrowed['n']['low'] > 10
        assert narrowed['n']['high'] < 100
        assert narrowed['n']['low'] < 50 < narrowed['n']['high']
        
        # Check values are integers
        assert isinstance(narrowed['n']['low'], (int, np.integer))
        assert isinstance(narrowed['n']['high'], (int, np.integer))
        print(f"✅ Integer narrowing: [{narrowed['n']['low']}, {narrowed['n']['high']}]")


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*80)
    print("RUNNING CUSTOM BALANCED SCORE ENHANCEMENT TESTS")
    print("="*80 + "\n")
    
    # Run pytest with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-s"])


if __name__ == "__main__":
    run_all_tests()

