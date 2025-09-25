"""Public exports for the market analysis training step."""

from __future__ import annotations

from typing import Any, Dict, List

from .multi_horizon_profit_labeler import (
    MultiHorizonProfitLabeler,
    MultiHorizonConfig,
    create_multi_horizon_labeler,
    apply_multi_horizon_labeling,
)
from .multi_horizon_sub_pipeline_adapter import (
    MultiHorizonSubPipelineAdapter,
    execute_multi_horizon_labeling_step,
)
from .gradient_flow_analysis import (
    GradientFlowAnalyzer,
    GradientFlowAnalysis,
    analyze_gradient_flow_benefits,
)
from .enhanced_multi_horizon_pipeline import (
    EnhancedMultiHorizonPipeline,
    EnhancedPipelineConfig,
    execute_enhanced_multi_horizon_labeling,
    get_optimal_configurations_for_training,
)
from .pid_based_feature_generation import (
    PIDBasedFeatureOrchestrator,
    OrchestratorConfig,
    InteractionFeatureGenerator,
    InteractionConfig,
    PolynomialFeatureGenerator,
    PolynomialConfig,
    CrossTimeframeFeatureGenerator,
    CrossTimeframeConfig,
    OptimizedLookbackIntegration,
    FeatureSelectionMechanism,
    FeatureSelectionConfig,
    SelectionStrategy,
)

MODULE_NAME = "MARKET_ANALYSIS"
__version__ = "1.0.0"
__author__ = "Market Analysis Team"
__email__ = "market-analysis@example.com"

_COMPONENT_EXPORTS = [
    "MultiHorizonProfitLabeler",
    "MultiHorizonConfig",
    "create_multi_horizon_labeler",
    "apply_multi_horizon_labeling",
    "MultiHorizonSubPipelineAdapter",
    "execute_multi_horizon_labeling_step",
    "GradientFlowAnalyzer",
    "GradientFlowAnalysis",
    "analyze_gradient_flow_benefits",
    "EnhancedMultiHorizonPipeline",
    "EnhancedPipelineConfig",
    "execute_enhanced_multi_horizon_labeling",
    "get_optimal_configurations_for_training",
    "PIDBasedFeatureOrchestrator",
    "OrchestratorConfig",
    "InteractionFeatureGenerator",
    "InteractionConfig",
    "PolynomialFeatureGenerator",
    "PolynomialConfig",
    "CrossTimeframeFeatureGenerator",
    "CrossTimeframeConfig",
    "OptimizedLookbackIntegration",
    "FeatureSelectionMechanism",
    "FeatureSelectionConfig",
    "SelectionStrategy",
]

__all__ = _COMPONENT_EXPORTS + ["get_module_info", "quick_start_example"]


def get_module_info() -> Dict[str, Any]:
    """Return descriptive metadata about the module."""

    return {
        "name": MODULE_NAME,
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "components": list(_COMPONENT_EXPORTS),
        "description": (
            "Comprehensive market analysis with regime-aware labeling, validation, and feature "
            "engineering support."
        ),
    }


def quick_start_example(
    num_samples: int = 240,
    seed: int | None = None,
) -> "pd.DataFrame":
    """Generate a small synthetic dataset and apply multi-horizon labeling.

    Parameters
    ----------
    num_samples:
        Number of synthetic rows to generate. Defaults to 240 (~one trading day of minute data).
    seed:
        Optional random seed for reproducible output.

    Returns
    -------
    pandas.DataFrame
        The labeled dataset produced by :func:`apply_multi_horizon_labeling`.
    """

    import numpy as np
    import pandas as pd

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=num_samples, freq="1min")

    base_price = 100.0
    price = base_price
    rows: List[Dict[str, float]] = []

    for _ in range(num_samples):
        price *= 1 + rng.normal(0, 0.0008)
        close_price = price * (1 + rng.normal(0, 0.0005))
        high_price = max(price, close_price) * (1 + rng.uniform(0, 0.002))
        low_price = min(price, close_price) * (1 - rng.uniform(0, 0.002))

        rows.append(
            {
                "open": price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": rng.uniform(1_000, 10_000),
                "hmm_regime": int(rng.choice([0, 1, 2], p=[0.4, 0.4, 0.2])),
            }
        )

    dataset = pd.DataFrame(rows, index=dates)
    return apply_multi_horizon_labeling(dataset)
