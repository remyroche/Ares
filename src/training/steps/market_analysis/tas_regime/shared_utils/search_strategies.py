"""
Search Strategies

Uses the unified evaluation framework for search strategy evaluation.
"""

from ...hybrid_nas_tas_regime.shared_utils.unified_evaluation_framework import (
    UnifiedEvaluationFramework, EvaluationConfig
)
from ...hybrid_nas_tas_regime.shared_utils.unified_architecture_config import ArchitectureType

class SearchStrategies:
    """Search strategies using unified framework."""

    def __init__(self, config: EvaluationConfig = None):
        self.evaluator = UnifiedEvaluationFramework(
            architecture_type=ArchitectureType.TAS,
            config=config or EvaluationConfig()
        )

    def evaluate_strategy(self, strategy, X_test, y_test, **kwargs):
        """Evaluate search strategy using unified framework."""
        return self.evaluator.evaluate_model(strategy, X_test, y_test, **kwargs)
