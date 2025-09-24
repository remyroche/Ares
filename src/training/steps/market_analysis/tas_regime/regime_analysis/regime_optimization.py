"""
Regime Optimization for TAS Tree Architecture

Uses the unified evaluation framework for regime optimization.
"""

from ...hybrid_nas_tas_regime.shared_utils.unified_evaluation_framework import (
    UnifiedEvaluationFramework, EvaluationConfig
)
from ...hybrid_nas_tas_regime.shared_utils.unified_architecture_config import ArchitectureType


class TreeRegimeOptimizer:
    """Tree regime optimizer using unified framework."""
    
    def __init__(self, config: EvaluationConfig = None):
        self.evaluator = UnifiedEvaluationFramework(
            architecture_type=ArchitectureType.TAS,
            config=config or EvaluationConfig()
        )
    
    def optimize_regimes(self, model, X_test, y_test, **kwargs):
        """Optimize regimes using unified framework."""
        return self.evaluator.optimize_regime_performance(model, X_test, y_test, **kwargs)
    
    def tune_regime_parameters(self, model, X_test, y_test, **kwargs):
        """Tune regime parameters using unified framework."""
        return self.evaluator.tune_regime_parameters(model, X_test, y_test, **kwargs)


class TreeRegimeSelector:
    """Tree regime selector using unified framework."""
    
    def __init__(self, config: EvaluationConfig = None):
        self.evaluator = UnifiedEvaluationFramework(
            architecture_type=ArchitectureType.TAS,
            config=config or EvaluationConfig()
        )
    
    def select_regimes(self, model, X_test, y_test, **kwargs):
        """Select regimes using unified framework."""
        return self.evaluator.select_regime_features(model, X_test, y_test, **kwargs)
    
    def rank_regimes(self, model, X_test, y_test, **kwargs):
        """Rank regimes using unified framework."""
        return self.evaluator.rank_regime_features(model, X_test, y_test, **kwargs)


class TreeRegimeAdapter:
    """Tree regime adapter using unified framework."""
    
    def __init__(self, config: EvaluationConfig = None):
        self.evaluator = UnifiedEvaluationFramework(
            architecture_type=ArchitectureType.TAS,
            config=config or EvaluationConfig()
        )
    
    def adapt_regimes(self, model, X_test, y_test, **kwargs):
        """Adapt regimes using unified framework."""
        return self.evaluator.adapt_regime_features(model, X_test, y_test, **kwargs)
    
    def transform_regimes(self, model, X_test, y_test, **kwargs):
        """Transform regimes using unified framework."""
        return self.evaluator.transform_regime_features(model, X_test, y_test, **kwargs)
