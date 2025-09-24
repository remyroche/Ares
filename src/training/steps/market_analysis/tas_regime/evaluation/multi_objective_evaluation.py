"""
Multi-objective Evaluation for TAS Tree Architecture

Uses the unified evaluation framework for multi-objective evaluation.
"""

from ...hybrid_nas_tas_regime.shared_utils.unified_evaluation_framework import (
    UnifiedEvaluationFramework, EvaluationConfig
)
from ...hybrid_nas_tas_regime.shared_utils.unified_architecture_config import ArchitectureType

class MultiObjectiveEvaluator:
    """Multi-objective evaluator using unified framework."""
    
    def __init__(self, config: EvaluationConfig = None):
        self.evaluator = UnifiedEvaluationFramework(
            architecture_type=ArchitectureType.TAS,
            config=config or EvaluationConfig()
        )
    
    def evaluate_multi_objective(self, model, X_test, y_test, objectives, **kwargs):
        """Evaluate model with multiple objectives using unified framework."""
        return self.evaluator.evaluate_multi_objective(model, X_test, y_test, objectives, **kwargs)

class TreeMultiObjectiveEvaluator:
    """Tree multi-objective evaluator using unified framework."""
    
    def __init__(self, config: EvaluationConfig = None):
        self.evaluator = UnifiedEvaluationFramework(
            architecture_type=ArchitectureType.TAS,
            config=config or EvaluationConfig()
        )
    
    def evaluate_multi_objective(self, model, X_test, y_test, objectives, **kwargs):
        """Evaluate tree model with multiple objectives using unified framework."""
        return self.evaluator.evaluate_multi_objective(model, X_test, y_test, objectives, **kwargs)

class TreeParetoEvaluator:
    """Tree Pareto evaluator using unified framework."""
    
    def __init__(self, config: EvaluationConfig = None):
        self.evaluator = UnifiedEvaluationFramework(
            architecture_type=ArchitectureType.TAS,
            config=config or EvaluationConfig()
        )
    
    def evaluate_pareto(self, model, X_test, y_test, objectives, **kwargs):
        """Evaluate tree model for Pareto optimality using unified framework."""
        return self.evaluator.evaluate_pareto(model, X_test, y_test, objectives, **kwargs)
