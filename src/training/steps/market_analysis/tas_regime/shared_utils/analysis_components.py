"""
Analysis Components

Uses the unified evaluation framework for analysis components.
"""

from ...hybrid_nas_tas_regime.shared_utils.unified_evaluation_framework import (
    UnifiedEvaluationFramework, EvaluationConfig
)
from ...hybrid_nas_tas_regime.shared_utils.unified_architecture_config import ArchitectureType

class AnalysisComponents:
    """Analysis components using unified framework."""
    
    def __init__(self, config: EvaluationConfig = None):
        self.evaluator = UnifiedEvaluationFramework(
            architecture_type=ArchitectureType.TAS,
            config=config or EvaluationConfig()
        )
    
    def analyze_components(self, components, X_test, y_test, **kwargs):
        """Analyze components using unified framework."""
        return self.evaluator.evaluate_model(components, X_test, y_test, **kwargs)
