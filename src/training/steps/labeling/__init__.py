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
    DEFAULT_PROFIT_THRESHOLD,
    DEFAULT_STOP_THRESHOLD,
    DEFAULT_TRANSACTION_COST,
)

from .meta_labeling_hpo_experiment_step import (
    MetaLabelingHPOExperimentStep,
)

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
    "DEFAULT_PROFIT_THRESHOLD",
    "DEFAULT_STOP_THRESHOLD",
    "DEFAULT_TRANSACTION_COST",
]
