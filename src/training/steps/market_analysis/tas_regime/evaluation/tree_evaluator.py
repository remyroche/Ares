"""
Tree Evaluator

Uses the unified evaluation framework for TAS tree architectures.
"""

from ...hybrid_nas_tas_regime.shared_utils.unified_evaluation_framework import (
    UnifiedEvaluationFramework, EvaluationConfig, EvaluationType
)
from ...hybrid_nas_tas_regime.shared_utils.unified_architecture_config import ArchitectureType

class TreeEvaluator:
    """Tree evaluator using unified evaluation framework."""

    def __init__(self, config: EvaluationConfig = None):
        self.evaluator = UnifiedEvaluationFramework(
            architecture_type=ArchitectureType.TAS,
            config=config or EvaluationConfig()
        )

    def evaluate(self, model, X_test, y_test, **kwargs):
        """Evaluate tree model using unified framework."""
        return self.evaluator.evaluate_model(model, X_test, y_test, **kwargs)

class TreePerformanceEvaluator:
    """Tree performance evaluator using unified framework."""

    def __init__(self, config: EvaluationConfig = None):
        self.evaluator = UnifiedEvaluationFramework(
            architecture_type=ArchitectureType.TAS,
            config=config or EvaluationConfig()
        )

    def evaluate_performance(self, model, X_test, y_test, **kwargs):
        """Evaluate tree performance using unified framework."""
        return self.evaluator.evaluate_model(model, X_test, y_test, **kwargs)

class TreeBenchmarkEvaluator:
    """Tree benchmark evaluator using unified framework."""

    def __init__(self, config: EvaluationConfig = None):
        self.evaluator = UnifiedEvaluationFramework(
            architecture_type=ArchitectureType.TAS,
            config=config or EvaluationConfig()
        )

    def evaluate_benchmark(self, model, X_test, y_test, benchmark_model=None, **kwargs):
        """Evaluate tree benchmark using unified framework."""
        if benchmark_model:
            return self.evaluator.evaluate_benchmark(model, X_test, y_test, benchmark_model, **kwargs)
        else:
            return self.evaluator.evaluate_model(model, X_test, y_test, **kwargs)
