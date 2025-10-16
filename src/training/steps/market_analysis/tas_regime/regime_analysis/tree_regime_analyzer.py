"""
Tree Regime Analyzer

Uses the unified evaluation framework for regime analysis.
"""

from ...hybrid_nas_tas_regime.shared_utils.unified_evaluation_framework import (
    UnifiedEvaluationFramework, EvaluationConfig
)
from ...hybrid_nas_tas_regime.shared_utils.unified_architecture_config import ArchitectureType

class TreeRegimeAnalyzer:
    """Tree regime analyzer using unified framework."""

    def __init__(self, config: EvaluationConfig = None):
        self.evaluator = UnifiedEvaluationFramework(
            architecture_type=ArchitectureType.TAS,
            config=config or EvaluationConfig()
        )

    def analyze_regimes(self, model, X_test, y_test, **kwargs):
        """Analyze regimes using unified framework."""
        return self.evaluator.evaluate_regime_performance(model, X_test, y_test, **kwargs)

class TreeRegimeDetector:
    """Tree regime detector using unified framework."""

    def __init__(self, config: EvaluationConfig = None):
        self.evaluator = UnifiedEvaluationFramework(
            architecture_type=ArchitectureType.TAS,
            config=config or EvaluationConfig()
        )

    def detect_regimes(self, model, X_test, y_test, **kwargs):
        """Detect regimes using unified framework."""
        return self.evaluator.evaluate_regime_detection(model, X_test, y_test, **kwargs)

    def predict_regimes(self, model, X_test, **kwargs):
        """Predict regimes using unified framework."""
        return self.evaluator.predict_regime(model, X_test, **kwargs)

class TreeRegimeClassifier:
    """Tree regime classifier using unified framework."""

    def __init__(self, config: EvaluationConfig = None):
        self.evaluator = UnifiedEvaluationFramework(
            architecture_type=ArchitectureType.TAS,
            config=config or EvaluationConfig()
        )

    def classify_regimes(self, model, X_test, y_test, **kwargs):
        """Classify regimes using unified framework."""
        return self.evaluator.evaluate_regime_classification(model, X_test, y_test, **kwargs)

    def predict_regime_classes(self, model, X_test, **kwargs):
        """Predict regime classes using unified framework."""
        return self.evaluator.predict_regime_class(model, X_test, **kwargs)
