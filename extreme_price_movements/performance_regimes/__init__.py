"""Performance-regime market-state experts.

This package keeps discovery, pruning, archetype learning, and portfolio
calibration fold-local.  The modules are intentionally import-light so tests
and offline scripts can exercise the leakage contract without loading the
production training stack.
"""

from extreme_price_movements.performance_regimes.labels import (
    StrategyPerformanceLabelBundle,
    StrategyPerformanceLabels,
    build_strategy_performance_labels,
)
from extreme_price_movements.performance_regimes.feature_matrix import (
    FeatureMatrixPipelineArtifact,
    MarketStateFeatureMatrix,
    apply_frozen_feature_pipeline,
    build_market_state_feature_matrix,
    fit_market_state_feature_pipeline,
)
from extreme_price_movements.performance_regimes.first_stage_models import (
    FirstStageLGBMConfig,
    FirstStageModelBundle,
    TimeSeriesSplitSpec,
    train_first_stage_bad_good_models,
)
from extreme_price_movements.performance_regimes.archetype_experts import (
    FrozenArchetypeExpertScores,
    score_frozen_archetype_experts,
)
from extreme_price_movements.performance_regimes.leaf_extraction import (
    extract_score_prune_leaves,
)
from extreme_price_movements.performance_regimes.portfolio_calibration import (
    apply_portfolio_actions,
    score_frozen_portfolio_calibrator,
    threshold_archetype_scores_for_modulation,
    train_portfolio_calibrator,
)

__all__ = [
    "FeatureMatrixPipelineArtifact",
    "FirstStageLGBMConfig",
    "FirstStageModelBundle",
    "FrozenArchetypeExpertScores",
    "MarketStateFeatureMatrix",
    "StrategyPerformanceLabelBundle",
    "StrategyPerformanceLabels",
    "TimeSeriesSplitSpec",
    "apply_frozen_feature_pipeline",
    "apply_portfolio_actions",
    "build_market_state_feature_matrix",
    "build_strategy_performance_labels",
    "extract_score_prune_leaves",
    "fit_market_state_feature_pipeline",
    "score_frozen_archetype_experts",
    "score_frozen_portfolio_calibrator",
    "threshold_archetype_scores_for_modulation",
    "train_portfolio_calibrator",
    "train_first_stage_bad_good_models",
]
