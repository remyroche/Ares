"""
Labeling Module for Step05.

This module consolidates all labeling-specific functionality including:
- Regime-aware triple barrier labeling components
- Meta-labeling with ensemble models (LGBM + XGBoost + RF)
- Hyperparameter optimization for labeling parameters (weighted and non-weighted)
- Triple barrier validation framework
- Data validation for feature generation
- SNR diagnostics for label quality assessment
- Meta-gated backtesting for labeling evaluation
- LGBM-based feature selection for meta-labeling (2025-12-08)
- Sample weighting for meta-labeling (2025-12-11)
"""

from .feature_generation_data_validation_step import (
    FeatureGenerationDataValidationStep,
)

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
    # NEW: Label generation diagnostics (SNR-based, 2025-12-09)
    LabelGenerationReport,
    compute_label_generation_report,
    monitor_label_density_by_stage,
    DEFAULT_PROFIT_THRESHOLD,
    DEFAULT_STOP_THRESHOLD,
    DEFAULT_TRANSACTION_COST,
)

from .generate_weights_per_label import (
    generate_weights_per_label,
    compute_horizon_consistency,
    compute_uniqueness,
    run_layer1_optimization,
)

# Sample weighting utilities (2025-12-11)
# from .generate_weights_per_label import (
#     generate_weights_per_label,
#     compute_horizon_consistency,
#     compute_uniqueness,
#     run_layer1_optimization,
# )

# Meta-labeling HPO steps
# CANONICAL: meta_labeling_hpo_sample_weighted is the primary HPO entry point
try:
    from .meta_labeling_hpo_sample_weighted import (
        MetaLabelingHPOSampleWeightedStep,
    )
    # Alias for backward compatibility
    MetaLabelingHPOExperimentStep = MetaLabelingHPOSampleWeightedStep
except Exception:
    MetaLabelingHPOSampleWeightedStep = None
    # Fallback to original if sample-weighted version fails
    try:
        from .meta_labeling_hpo_experiment_step import (
            MetaLabelingHPOExperimentStep,
        )
    except Exception:
        MetaLabelingHPOExperimentStep = None

# Layer-specific HPO modules (refactored from meta_labeling_hpo_sample_weighted.py)
# Each module handles one layer of the hierarchical optimization:
# - Layer 0: Kalman/RTS smoother optimization
# - Layer 1: Sample weighting optimization
# - Layer 2: Trading parameters optimization
# - Layer 3: Model hyperparameters optimization
try:
    from .meta_labeling_weighted_hpo_0 import (
        LAYER0_KALMAN_SEARCH_SPACE,
        run_layer0_kalman_optimization,
        run_committee_pre_step,
    )
    from .meta_labeling_weighted_hpo_1 import (
        DEFAULT_WEIGHTING_PARAMS,
        compute_committee_weight_factors,
        run_layer1_weighting_optimization,
    )
    from .meta_labeling_weighted_hpo_2 import (
        get_layer2_search_space,
        compute_regime_conditional_barrier_geometry,
        save_layer2_results,
    )
    from .meta_labeling_weighted_hpo_3 import (
        get_layer3_search_space,
        get_lgbm_params_from_trial,
        compute_layer3_cv_metrics,
        save_layer3_results,
    )
except Exception:
    # Layer modules are optional - main coordinator still has all logic
    pass

# Weighted meta-labeling production step (2025-12-11)
try:
    from .weighted_meta_labeling_step import (
        WeightedMetaLabelingStep,
    )
except Exception:
    WeightedMetaLabelingStep = None

from .triple_barrier_validator import (
    TripleBarrierValidator,
    ValidationResult,
    ValidationReport,
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
    "MetaLabelingHPOSampleWeightedStep",
    "WeightedMetaLabelingStep",
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
    # Sample weighting utilities (2025-12-11)
    "generate_weights_per_label",
    "compute_horizon_consistency",
    "compute_uniqueness",
    "run_layer1_optimization",
    # Volatility-scaled labeling functions (2025-12-09)
    "compute_ema_volatility",
    "compute_regime_metrics",
    "compute_target_positive_fraction",
    "compute_dynamic_threshold_k",
    "create_volatility_scaled_labels",
    "create_volatility_scaled_labels_for_events",
    # Label generation diagnostics (SNR-based, 2025-12-09)
    "LabelGenerationReport",
    "compute_label_generation_report",
    "monitor_label_density_by_stage",
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
