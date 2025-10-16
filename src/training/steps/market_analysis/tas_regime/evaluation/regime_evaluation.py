"""
Regime Evaluation for TAS Tree Architecture

Uses the unified evaluation framework for regime evaluation.
"""

from ...hybrid_nas_tas_regime.shared_utils.unified_evaluation_framework import (
    UnifiedEvaluationFramework, EvaluationConfig
)
from ...hybrid_nas_tas_regime.shared_utils.unified_architecture_config import ArchitectureType

class TreeRegimeEvaluator:
    """Tree regime evaluator using unified framework."""

    def __init__(self, config: EvaluationConfig = None):
        self.evaluator = UnifiedEvaluationFramework(
            architecture_type=ArchitectureType.TAS,
            config=config or EvaluationConfig()
        )

    def evaluate_regime(self, model, X_test, y_test, **kwargs):
        """Evaluate tree model for regime detection using unified framework."""
        return self.evaluator.evaluate_model(model, X_test, y_test, **kwargs)

class TreeRegimePerformanceAnalyzer:
    """Tree regime performance analyzer using unified framework."""

    def __init__(self, config: EvaluationConfig = None):
        self.evaluator = UnifiedEvaluationFramework(
            architecture_type=ArchitectureType.TAS,
            config=config or EvaluationConfig()
        )

    def analyze_performance(self, model, X_test, y_test, **kwargs):
        """Analyze tree regime performance using unified framework."""
        return self.evaluator.evaluate_model(model, X_test, y_test, **kwargs)
