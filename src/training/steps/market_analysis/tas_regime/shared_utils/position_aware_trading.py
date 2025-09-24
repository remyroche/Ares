"""
Position Aware Trading

Uses the unified evaluation framework for position-aware trading evaluation.
"""

from ...hybrid_nas_tas_regime.shared_utils.unified_evaluation_framework import (
    UnifiedEvaluationFramework, EvaluationConfig
)
from ...hybrid_nas_tas_regime.shared_utils.unified_architecture_config import ArchitectureType

class PositionAwareTrading:
    """Position aware trading using unified framework."""
    
    def __init__(self, config: EvaluationConfig = None):
        self.evaluator = UnifiedEvaluationFramework(
            architecture_type=ArchitectureType.TAS,
            config=config or EvaluationConfig()
        )
    
    def evaluate_trading(self, model, X_test, y_test, positions, **kwargs):
        """Evaluate position-aware trading using unified framework."""
        return self.evaluator.evaluate_trading_performance(model, X_test, y_test, positions, **kwargs)
