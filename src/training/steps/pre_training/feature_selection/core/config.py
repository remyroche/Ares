"""
Configuration classes for feature selection.

This module contains all configuration dataclasses used in the feature selection
pipeline, including base configurations, model-specific settings, and validation parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from src.utils.tprint import tprint_debug


@dataclass
class BaseFeatureSelectionConfig:
    """Base configuration for multi-stage feature selection."""
    # Stage targets
    initial_features: int = 120
    stage_1_target: int = 100
    stage_2_target: int = 80
    stage_3_target: int = 60

    # Model configuration
    model_type: str = 'regime_detection'
    target_features: int = 60  # Default target set to 60
    min_features: int = 60
    max_features: int = 100
    priority_categories: List[str] = field(default_factory=lambda: ['volatility', 'structural', 'volume_regime', 'statistical'])

    # VectorBT optimization settings
    enable_vectorbt_optimization: bool = True
    vectorbt_memory_efficient: bool = True
    vectorbt_chunk_size: int = 1000
    vectorbt_enable_parallel: bool = True
    vectorbt_enable_gpu: bool = False

    # Output settings
    save_models: bool = True
    save_analysis: bool = True
    output_directory: str = "outcomes/market_analysis"
    verbose: bool = True


@dataclass
class ModelSpecificConfig:
    """Model-specific parameters."""
    # RandomForest parameters
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    rf_min_samples_split: int = 5
    rf_random_state: int = 42

    # SHAP parameters
    shap_sample_size: int = 1000
    shap_max_features: int = 200

    # LightGBM parameters
    lightgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 10,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42
    })


@dataclass
class QualityThresholdsConfig:
    """Quality thresholds for feature selection."""
    min_feature_importance: float = 0.002
    min_correlation_threshold: float = 0.90
    min_variance_threshold: float = 0.005
    model_correlation_threshold: float = 0.85
    model_importance_threshold: float = 0.003
    
    # Regime-specific thresholds
    regime_importance_threshold: float = 0.005
    regime_correlation_threshold: float = 0.80
    regime_variance_threshold: float = 0.001


@dataclass
class ValidationConfig:
    """Cross-validation and trading-aware evaluation."""
    cv_folds: int = 5
    cv_scoring: str = 'neg_mean_squared_error'
    label_horizon_minutes: int = 30
    purge_minutes: Optional[int] = None
    embargo_minutes: Optional[int] = None
    
    # Trading parameters
    trading_cost: float = 0.0005
    trading_horizon: int = 252
    turnover_penalty: float = 0.0
    ic_method: str = 'spearman'
    market_impact_coefficient: float = 0.1
    capacity_limit_usd: float = 1_000_000.0
    max_turnover_annual: float = 50.0
    min_sharpe_to_turnover_ratio: float = 0.1

    # Uncertainty & calibration
    enable_uncertainty_reporting: bool = True
    reliability_bins: int = 10
    confidence_coverage_target: float = 0.9


@dataclass
class AdvancedSelectionConfig:
    """Advanced feature selection parameters."""
    # Selection methods
    selection_methods: List[str] = field(default_factory=lambda: [
        'mrmr', 'lasso', 'correlation_filtering', 'rfe', 'variance_filtering', 'mutual_info'
    ])

    # Directional feature selection
    direction_mode: str = 'both'
    separate_directional_features: bool = True
    directional_feature_prefixes: Dict[str, str] = field(default_factory=lambda: {
        'long': 'long_',
        'short': 'short_'
    })

    # Regime-aware selection
    enable_regime_aware_selection: bool = True
    regime_clustering_threshold: float = 0.7
    regime_separation_bonus: float = 0.1
    regime_focus_weights: Dict[str, float] = field(default_factory=lambda: {
        'volatility': 0.35,
        'structural': 0.25,
        'volume_regime': 0.20,
        'statistical': 0.20
    })

    # Multi-criteria selection
    enable_multi_criteria_selection: bool = True
    criteria_weights: Dict[str, float] = field(default_factory=lambda: {
        'importance': 0.30,
        'correlation': 0.20,
        'variance': 0.15,
        'regime_separation': 0.25,
        'temporal_stability': 0.10
    })
    
    # Stage-specific scoring weights
    stage_1_weights: Dict[str, float] = field(default_factory=lambda: {
        'mrmr': 0.7,
        'spearman': 0.3
    })
    stage_3_weights: Dict[str, float] = field(default_factory=lambda: {
        'ensemble': 0.6,
        'stability': 0.4
    })


@dataclass
class NewPipelineConfig:
    """Configuration for the new multi-stage pipeline."""
    
    # Pipeline stages
    enable_new_pipeline: bool = True
    
    # Stage 1: mRMR + Spearman combination
    stage1_mrmr_weight: float = 0.7
    stage1_spearman_weight: float = 0.3
    stage1_target_ratio: float = 0.5  # Select top 50% above target
    
    # Stage 2: Progressive refinement
    stage2_enable_progressive_refinement: bool = True
    
    # Bootstrap stability and CV threshold
    stage2_bootstrap_cv_threshold: int = 40  # Use bootstrap stability and CV when 40+ features away from target
    
    # LGBM-SHAP configuration
    lgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'verbose': -1
    })
    
    # SHAP configuration
    shap_sample_size: int = 1000
    shap_max_features: int = 200
    shap_explainer_type: str = 'tree'  # 'tree', 'linear', 'kernel'
    
    # LASSO ensemble configuration
    lasso_alpha_range: Tuple[float, float] = (0.001, 1.0)
    lasso_cv_folds: int = 5
    lasso_n_alphas: int = 50
    
    # RFE configuration
    rfe_step_size: float = 0.10  # Remove 10% of features above target in each RFE round
    rfe_min_features: int = 10
    rfe_cv_folds: int = 3
    rfe_early_stopping: bool = True
    rfe_early_stopping_patience: int = 3
    rfe_use_percentage_step: bool = True  # Use percentage-based step size instead of fixed
    
    # Bootstrap stability configuration
    bootstrap_n_samples: int = 100
    bootstrap_sample_ratio: float = 0.8
    stability_threshold: float = 0.6
    
    # Cross-validation configuration
    cv_folds: int = 5
    cv_scoring: str = 'neg_mean_squared_error'
    
    # Ensemble weights
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        'lgbm_shap': 0.4,
        'lasso_ensemble': 0.3,
        'rfe': 0.2,
        'bootstrap_stability': 0.1
    })
    
    # VectorBT optimization
    enable_vectorbt_optimization: bool = True
    vectorbt_chunk_size: int = 1000
    vectorbt_enable_parallel: bool = True

    # Entropy stability filtering
    enable_entropy_balancing: bool = True
    entropy_num_slices: int = 12
    entropy_min_slice_size: int = 100
    entropy_variance_threshold: float = 0.12
    entropy_max_bins: int = 15
    entropy_min_unique_values: int = 5
    entropy_use_time_index: bool = True

    # RFE parameters
    enable_rfe: bool = True
    rfe_step_size: float = 0.1
    rfe_min_features: int = 10
    rfe_cv_folds: int = 3
    rfe_early_stopping: bool = True
    rfe_early_stopping_patience: int = 3

    # Chunked processing
    enable_chunked_processing: bool = True
    chunk_size: int = 1000
    max_chunks: int = 10
    chunk_overlap: int = 50


@dataclass
class FeatureSelectionConfig(BaseFeatureSelectionConfig):
    """Complete configuration combining all sub-configs."""
    
    def __post_init__(self):
        """Initialize sub-configurations."""
        tprint_debug("🔧 Initializing feature selection configuration sub-configs")
        self.model_config = ModelSpecificConfig()
        self.quality_config = QualityThresholdsConfig()
        self.validation_config = ValidationConfig()
        self.advanced_config = AdvancedSelectionConfig()
        self.new_pipeline_config = NewPipelineConfig()
        
        # Merge parameters from sub-configs for backward compatibility
        for config in [self.model_config, self.quality_config, self.validation_config, self.advanced_config, self.new_pipeline_config]:
            for attr, value in config.__dict__.items():
                if not hasattr(self, attr):
                    setattr(self, attr, value)


@dataclass
class FeatureSelectionResult:
    """Result of feature selection analysis."""
    # Selected features
    selected_features: List[str]
    feature_importance: Dict[str, float]
    feature_scores: Dict[str, Dict[str, float]]
    
    # Performance metrics
    performance_metrics: Dict[str, Any]
    validation_scores: Dict[str, float]
    
    # Configuration used
    config_used: FeatureSelectionConfig
    
    # Metadata
    execution_time: float
    memory_usage: Dict[str, Any]
    vectorbt_stats: Optional[Dict[str, Any]] = None
    
    # Stage results
    stage_results: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Success flag
    success: bool = True
    error_message: Optional[str] = None