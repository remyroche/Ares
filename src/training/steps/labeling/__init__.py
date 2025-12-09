"""
Labeling Module for Step05.

This module consolidates all labeling-specific functionality including:
- Regime-aware triple barrier labeling components
- Meta-labeling with ensemble models (LGBM + XGBoost + RF)
- Hyperparameter optimization for labeling parameters
- Triple barrier validation framework
- Data validation for feature generation
- SNR diagnostics for label quality assessment
- Meta-gated backtesting for labeling evaluation
- LGBM-based feature selection for meta-labeling (2025-12-08)
"""

from .labeling_components import (
    RegimeAwareLabeling,
)

from .feature_generation_meta_labeling_step import (
    FeatureGenerationMetaLabelingStep,
    compute_realized_returns,
    kalman_smooth_labels,
    fit_probability_to_return_mapping,
    translate_to_targets_with_isotonic,
    generate_primary_signals,
    create_meta_features,
    compute_learnability_score,
    compute_label_entropy_score,
    generate_diagnostics_report,
    compute_vol_scaled_returns_for_events,
    create_quantile_labels_from_vol_scaled_returns,
    # NEW: Volatility-scaled labeling functions (2025-12-09)
    compute_ema_volatility,
    compute_regime_metrics,
    compute_target_positive_fraction,
    compute_dynamic_threshold_k,
    create_volatility_scaled_labels,
    create_volatility_scaled_labels_for_events,
    DEFAULT_PROFIT_THRESHOLD,
    DEFAULT_STOP_THRESHOLD,
    DEFAULT_TRANSACTION_COST,
)

try:
    from .meta_labeling_hpo_experiment_step import (
        MetaLabelingHPOExperimentStep,
    )
except Exception:
    # If the optional meta-labeling HPO step fails to import (e.g. due to
    # environment-specific issues), degrade gracefully so that core labeling
    # and training steps continue to function.
    MetaLabelingHPOExperimentStep = None

from .triple_barrier_validator import (
    TripleBarrierValidator,
    ValidationResult,
    ValidationReport,
)

from .feature_generation_data_validation_step import (
    FeatureGenerationDataValidationStep,
)

from .meta_gated_backtest_step import (
    MetaGatedBacktestStep,
)

# LGBM Feature Selection (2025-12-08)
from .lgbm_feature_selection import (
    lgbm_feature_selection_pipeline,
    select_features_lgbm_for_meta_labeling,
    select_features_by_importance_lgbm,
    FeatureSetPersistence,
    iterative_lgbm_importance_selection,
    permutation_importance_rfe,
    correlation_pruning,
    FEATURE_SELECTION_CONFIG,
    DEFAULT_LGBM_PARAMS,
)

# Winning Feature Set Selection (2025-12-08)
from .winning_feature_set_selector import (
    determine_winning_feature_set,
    compute_winning_score,
    compute_composite_score,  # Backward compat wrapper
    run_winning_feature_set_selection,
    METRIC_WEIGHTS,
)

__all__ = [
    "RegimeAwareLabeling",
    "FeatureGenerationMetaLabelingStep",
    "MetaLabelingHPOExperimentStep",
    "TripleBarrierValidator",
    "ValidationResult",
    "ValidationReport",
    "FeatureGenerationDataValidationStep",
    "MetaGatedBacktestStep",
    "compute_realized_returns",
    "kalman_smooth_labels",
    "fit_probability_to_return_mapping",
    "translate_to_targets_with_isotonic",
    "generate_primary_signals",
    "create_meta_features",
    "compute_learnability_score",
    "compute_label_entropy_score",
    "generate_diagnostics_report",
    "compute_vol_scaled_returns_for_events",
    "create_quantile_labels_from_vol_scaled_returns",
    # NEW: Volatility-scaled labeling functions (2025-12-09)
    "compute_ema_volatility",
    "compute_regime_metrics",
    "compute_target_positive_fraction",
    "compute_dynamic_threshold_k",
    "create_volatility_scaled_labels",
    "create_volatility_scaled_labels_for_events",
    "DEFAULT_PROFIT_THRESHOLD",
    "DEFAULT_STOP_THRESHOLD",
    "DEFAULT_TRANSACTION_COST",
    # LGBM Feature Selection (2025-12-08)
    "lgbm_feature_selection_pipeline",
    "select_features_lgbm_for_meta_labeling",
    "select_features_by_importance_lgbm",
    "FeatureSetPersistence",
    "iterative_lgbm_importance_selection",
    "permutation_importance_rfe",
    "correlation_pruning",
    "FEATURE_SELECTION_CONFIG",
    "DEFAULT_LGBM_PARAMS",
    # Winning Feature Set Selection (2025-12-08)
    "determine_winning_feature_set",
    "compute_winning_score",
    "compute_composite_score",
    "run_winning_feature_set_selection",
    "METRIC_WEIGHTS",
]
