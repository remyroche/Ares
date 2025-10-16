"""
Regime Reporting for TAS Tree Architecture

Uses the unified evaluation framework for regime reporting and visualization.
"""

from ...hybrid_nas_tas_regime.shared_utils.unified_evaluation_framework import (
    UnifiedEvaluationFramework, EvaluationConfig
)
from ...hybrid_nas_tas_regime.shared_utils.unified_architecture_config import ArchitectureType

class TreeRegimeReporter:
    """Tree regime reporter using unified framework."""

    def __init__(self, config: EvaluationConfig = None):
        self.evaluator = UnifiedEvaluationFramework(
            architecture_type=ArchitectureType.TAS,
            config=config or EvaluationConfig()
        )

    def generate_report(self, model, X_test, y_test, **kwargs):
        """Generate regime report using unified framework."""
        return self.evaluator.generate_regime_report(model, X_test, y_test, **kwargs)

    def export_results(self, model, X_test, y_test, **kwargs):
        """Export regime results using unified framework."""
        return self.evaluator.export_regime_results(model, X_test, y_test, **kwargs)

class TreeRegimeVisualizer:
    """Tree regime visualizer using unified framework."""

    def __init__(self, config: EvaluationConfig = None):
        self.evaluator = UnifiedEvaluationFramework(
            architecture_type=ArchitectureType.TAS,
            config=config or EvaluationConfig()
        )

    def visualize_regimes(self, model, X_test, y_test, **kwargs):
        """Visualize regimes using unified framework."""
        return self.evaluator.visualize_regime_performance(model, X_test, y_test, **kwargs)

    def plot_regime_analysis(self, model, X_test, y_test, **kwargs):
        """Plot regime analysis using unified framework."""
        return self.evaluator.plot_regime_analysis(model, X_test, y_test, **kwargs)

class TreeRegimeDashboard:
    """Tree regime dashboard using unified framework."""

    def __init__(self, config: EvaluationConfig = None):
        self.evaluator = UnifiedEvaluationFramework(
            architecture_type=ArchitectureType.TAS,
            config=config or EvaluationConfig()
        )

    def create_dashboard(self, model, X_test, y_test, **kwargs):
        """Create regime dashboard using unified framework."""
        return self.evaluator.create_regime_dashboard(model, X_test, y_test, **kwargs)

    def update_dashboard(self, model, X_test, y_test, **kwargs):
        """Update regime dashboard using unified framework."""
        return self.evaluator.update_regime_dashboard(model, X_test, y_test, **kwargs)
