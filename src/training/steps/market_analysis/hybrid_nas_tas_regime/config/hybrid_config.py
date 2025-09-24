"""
Hybrid NAS TAS Regime Configuration

Comprehensive configuration system for hybrid regime detection that combines
TAS and NAS regime detection with economic and financial relevance.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import numpy as np


class RegimeType(Enum):
    """Regime types for hybrid detection."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"
    CRISIS = "crisis"
    NORMAL = "normal"
    UNKNOWN = "unknown"


class EconomicRegimeType(Enum):
    """Economic regime types."""
    EXPANSION = "expansion"
    RECESSION = "recession"
    RECOVERY = "recovery"
    STAGNATION = "stagnation"
    INFLATION = "inflation"
    DEFLATION = "deflation"
    STAGFLATION = "stagflation"
    BOOM = "boom"
    BUST = "bust"


class FinancialRegimeType(Enum):
    """Financial regime types."""
    LIQUIDITY_ABUNDANT = "liquidity_abundant"
    LIQUIDITY_CRUNCH = "liquidity_crunch"
    CREDIT_EASY = "credit_easy"
    CREDIT_TIGHT = "credit_tight"
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    FLIGHT_TO_QUALITY = "flight_to_quality"
    SPECULATION = "speculation"


class ClusteringMethod(Enum):
    """Clustering methods for regime detection."""
    KMEANS = "kmeans"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"
    SPECTRAL = "spectral"
    BIRCH = "birch"
    AFFINITY_PROPAGATION = "affinity_propagation"
    MEAN_SHIFT = "mean_shift"
    OPTICS = "optics"
    HYBRID = "hybrid"


class IntegrationStrategy(Enum):
    """Integration strategies for TAS and NAS."""
    WEIGHTED_AVERAGE = "weighted_average"
    ENSEMBLE = "ensemble"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"
    META_LEARNING = "meta_learning"


@dataclass
class HybridNASConfig:
    """Configuration for NAS regime detection integration."""
    
    # NAS model settings
    nas_model_types: List[str] = field(default_factory=lambda: [
        "neural_ode", "vision_transformer", "state_space_model", "lstm", "gru"
    ])
    nas_architecture_types: List[str] = field(default_factory=lambda: [
        "continuous_time", "transformer", "recurrent", "hybrid"
    ])
    
    # NAS search settings
    nas_search_strategy: str = "evolutionary"
    nas_search_budget: int = 100
    nas_search_time_limit: int = 3600
    
    # NAS optimization
    nas_optimization_objectives: List[str] = field(default_factory=lambda: [
        "accuracy", "robustness", "efficiency", "economic_significance"
    ])
    nas_objective_weights: List[float] = field(default_factory=lambda: [0.4, 0.2, 0.2, 0.2])
    
    # NAS regime detection
    nas_regime_detection_enabled: bool = True
    nas_regime_stability_threshold: float = 0.7
    nas_regime_transition_threshold: float = 0.5
    nas_micro_regime_detection: bool = True
    nas_micro_regime_sensitivity: float = 0.7
    
    # NAS economic evaluation
    nas_economic_significance_threshold: float = 0.7
    nas_trading_viability_threshold: float = 0.6
    nas_regime_adaptation_speed: float = 0.8
    
    # NAS meta-learning
    nas_meta_learning_enabled: bool = True
    nas_adaptation_history_length: int = 100
    nas_regime_similarity_threshold: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'nas_model_types': self.nas_model_types,
            'nas_architecture_types': self.nas_architecture_types,
            'nas_search_strategy': self.nas_search_strategy,
            'nas_search_budget': self.nas_search_budget,
            'nas_search_time_limit': self.nas_search_time_limit,
            'nas_optimization_objectives': self.nas_optimization_objectives,
            'nas_objective_weights': self.nas_objective_weights,
            'nas_regime_detection_enabled': self.nas_regime_detection_enabled,
            'nas_regime_stability_threshold': self.nas_regime_stability_threshold,
            'nas_regime_transition_threshold': self.nas_regime_transition_threshold,
            'nas_micro_regime_detection': self.nas_micro_regime_detection,
            'nas_micro_regime_sensitivity': self.nas_micro_regime_sensitivity,
            'nas_economic_significance_threshold': self.nas_economic_significance_threshold,
            'nas_trading_viability_threshold': self.nas_trading_viability_threshold,
            'nas_regime_adaptation_speed': self.nas_regime_adaptation_speed,
            'nas_meta_learning_enabled': self.nas_meta_learning_enabled,
            'nas_adaptation_history_length': self.nas_adaptation_history_length,
            'nas_regime_similarity_threshold': self.nas_regime_similarity_threshold
        }


@dataclass
class HybridTASConfig:
    """Configuration for TAS regime detection integration."""
    
    # TAS model settings
    tas_model_types: List[str] = field(default_factory=lambda: [
        "random_forest", "xgboost", "lightgbm", "extra_trees", "gradient_boosting"
    ])
    tas_architecture_types: List[str] = field(default_factory=lambda: [
        "tree_only", "cvlSA_tree", "hybrid_tree_neural", "ensemble_hierarchical"
    ])
    
    # TAS search settings
    tas_search_strategy: str = "bayesian"
    tas_search_budget: int = 100
    tas_search_time_limit: int = 3600
    
    # TAS optimization
    tas_optimization_objectives: List[str] = field(default_factory=lambda: [
        "accuracy", "robustness", "efficiency", "interpretability"
    ])
    tas_objective_weights: List[float] = field(default_factory=lambda: [0.3, 0.2, 0.2, 0.3])
    
    # TAS regime detection
    tas_regime_detection_enabled: bool = True
    tas_regime_stability_threshold: float = 0.6
    tas_regime_transition_threshold: float = 0.4
    tas_micro_regime_detection: bool = True
    tas_micro_regime_sensitivity: float = 0.6
    
    # TAS economic evaluation
    tas_economic_significance_threshold: float = 0.6
    tas_trading_viability_threshold: float = 0.5
    tas_regime_adaptation_speed: float = 0.7
    
    # TAS tree constraints
    tas_min_trees: int = 10
    tas_max_trees: int = 1000
    tas_min_depth: int = 1
    tas_max_depth: int = 20
    tas_min_samples_split: int = 2
    tas_max_samples_split: int = 1000
    tas_min_samples_leaf: int = 1
    tas_max_samples_leaf: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'tas_model_types': self.tas_model_types,
            'tas_architecture_types': self.tas_architecture_types,
            'tas_search_strategy': self.tas_search_strategy,
            'tas_search_budget': self.tas_search_budget,
            'tas_search_time_limit': self.tas_search_time_limit,
            'tas_optimization_objectives': self.tas_optimization_objectives,
            'tas_objective_weights': self.tas_objective_weights,
            'tas_regime_detection_enabled': self.tas_regime_detection_enabled,
            'tas_regime_stability_threshold': self.tas_regime_stability_threshold,
            'tas_regime_transition_threshold': self.tas_regime_transition_threshold,
            'tas_micro_regime_detection': self.tas_micro_regime_detection,
            'tas_micro_regime_sensitivity': self.tas_micro_regime_sensitivity,
            'tas_economic_significance_threshold': self.tas_economic_significance_threshold,
            'tas_trading_viability_threshold': self.tas_trading_viability_threshold,
            'tas_regime_adaptation_speed': self.tas_regime_adaptation_speed,
            'tas_min_trees': self.tas_min_trees,
            'tas_max_trees': self.tas_max_trees,
            'tas_min_depth': self.tas_min_depth,
            'tas_max_depth': self.tas_max_depth,
            'tas_min_samples_split': self.tas_min_samples_split,
            'tas_max_samples_split': self.tas_max_samples_split,
            'tas_min_samples_leaf': self.tas_min_samples_leaf,
            'tas_max_samples_leaf': self.tas_max_samples_leaf
        }


@dataclass
class HybridRegimeConfig:
    """Main configuration for hybrid regime detection."""
    
    # Integration settings
    integration_strategy: IntegrationStrategy = IntegrationStrategy.ADAPTIVE
    nas_weight: float = 0.6
    tas_weight: float = 0.4
    adaptive_weighting: bool = True
    weight_adaptation_rate: float = 0.1
    
    # Regime detection settings
    n_regimes: int = 12
    min_regime_duration: int = 15  # minutes
    max_regime_duration: int = 180  # minutes
    regime_stability_threshold: float = 0.7
    regime_transition_threshold: float = 0.5
    
    # Clustering settings
    clustering_method: ClusteringMethod = ClusteringMethod.HYBRID
    clustering_metrics: List[str] = field(default_factory=lambda: [
        "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"
    ])
    clustering_n_init: int = 10
    clustering_max_iter: int = 300
    clustering_tolerance: float = 1e-4
    
    # Economic and financial modeling
    economic_modeling_enabled: bool = True
    financial_modeling_enabled: bool = True
    economic_significance_threshold: float = 0.7
    financial_significance_threshold: float = 0.6
    trading_viability_threshold: float = 0.6
    
    # Micro-regime detection
    micro_regime_detection: bool = True
    micro_regime_sensitivity: float = 0.7
    micro_regime_types: List[str] = field(default_factory=lambda: [
        "breakout", "consolidation", "reversal", "acceleration", "deceleration",
        "volume_spike", "volatility_spike", "momentum_shift", "liquidity_change"
    ])
    
    # Performance settings
    enable_gpu_acceleration: bool = True
    enable_parallel_processing: bool = True
    n_jobs: int = -1
    memory_limit_gb: float = 8.0
    batch_size: int = 1000
    
    # Output settings
    save_results: bool = True
    save_models: bool = True
    output_dir: str = "hybrid_regime_results"
    verbose: bool = True
    
    # Validation settings
    validation_method: str = "holdout"  # "holdout", "cross_validation", "time_series_split"
    validation_split: float = 0.2
    cv_folds: int = 5
    time_series_gap: int = 0
    
    # Timeframe settings
    primary_timeframe: str = "15m"
    micro_timeframe: str = "5m"
    regime_detection_window: int = 100
    adaptation_interval_minutes: int = 15
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'integration_strategy': self.integration_strategy.value,
            'nas_weight': self.nas_weight,
            'tas_weight': self.tas_weight,
            'adaptive_weighting': self.adaptive_weighting,
            'weight_adaptation_rate': self.weight_adaptation_rate,
            'n_regimes': self.n_regimes,
            'min_regime_duration': self.min_regime_duration,
            'max_regime_duration': self.max_regime_duration,
            'regime_stability_threshold': self.regime_stability_threshold,
            'regime_transition_threshold': self.regime_transition_threshold,
            'clustering_method': self.clustering_method.value,
            'clustering_metrics': self.clustering_metrics,
            'clustering_n_init': self.clustering_n_init,
            'clustering_max_iter': self.clustering_max_iter,
            'clustering_tolerance': self.clustering_tolerance,
            'economic_modeling_enabled': self.economic_modeling_enabled,
            'financial_modeling_enabled': self.financial_modeling_enabled,
            'economic_significance_threshold': self.economic_significance_threshold,
            'financial_significance_threshold': self.financial_significance_threshold,
            'trading_viability_threshold': self.trading_viability_threshold,
            'micro_regime_detection': self.micro_regime_detection,
            'micro_regime_sensitivity': self.micro_regime_sensitivity,
            'micro_regime_types': self.micro_regime_types,
            'enable_gpu_acceleration': self.enable_gpu_acceleration,
            'enable_parallel_processing': self.enable_parallel_processing,
            'n_jobs': self.n_jobs,
            'memory_limit_gb': self.memory_limit_gb,
            'batch_size': self.batch_size,
            'save_results': self.save_results,
            'save_models': self.save_models,
            'output_dir': self.output_dir,
            'verbose': self.verbose,
            'validation_method': self.validation_method,
            'validation_split': self.validation_split,
            'cv_folds': self.cv_folds,
            'time_series_gap': self.time_series_gap,
            'primary_timeframe': self.primary_timeframe,
            'micro_timeframe': self.micro_timeframe,
            'regime_detection_window': self.regime_detection_window,
            'adaptation_interval_minutes': self.adaptation_interval_minutes
        }


@dataclass
class HybridIntegrationConfig:
    """Configuration for TAS and NAS integration."""
    
    # Integration method
    integration_method: str = "weighted_ensemble"  # "weighted_ensemble", "hierarchical", "adaptive"
    ensemble_method: str = "voting"  # "voting", "stacking", "blending"
    
    # Weight settings
    nas_weight: float = 0.6
    tas_weight: float = 0.4
    adaptive_weights: bool = True
    weight_learning_rate: float = 0.01
    
    # Performance thresholds
    min_performance_threshold: float = 0.5
    max_performance_threshold: float = 0.95
    performance_window: int = 50
    
    # Regime consistency
    regime_consistency_threshold: float = 0.7
    regime_agreement_threshold: float = 0.6
    regime_disagreement_penalty: float = 0.1
    
    # Uncertainty handling
    uncertainty_weighting: bool = True
    uncertainty_threshold: float = 0.3
    confidence_weighting: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'integration_method': self.integration_method,
            'ensemble_method': self.ensemble_method,
            'nas_weight': self.nas_weight,
            'tas_weight': self.tas_weight,
            'adaptive_weights': self.adaptive_weights,
            'weight_learning_rate': self.weight_learning_rate,
            'min_performance_threshold': self.min_performance_threshold,
            'max_performance_threshold': self.max_performance_threshold,
            'performance_window': self.performance_window,
            'regime_consistency_threshold': self.regime_consistency_threshold,
            'regime_agreement_threshold': self.regime_agreement_threshold,
            'regime_disagreement_penalty': self.regime_disagreement_penalty,
            'uncertainty_weighting': self.uncertainty_weighting,
            'uncertainty_threshold': self.uncertainty_threshold,
            'confidence_weighting': self.confidence_weighting
        }


@dataclass
class HybridClusteringConfig:
    """Configuration for hybrid clustering."""
    
    # Clustering methods
    primary_clustering_method: ClusteringMethod = ClusteringMethod.GAUSSIAN_MIXTURE
    secondary_clustering_methods: List[ClusteringMethod] = field(default_factory=lambda: [
        ClusteringMethod.KMEANS, ClusteringMethod.HIERARCHICAL
    ])
    
    # Clustering parameters
    n_clusters_range: Tuple[int, int] = (3, 20)
    clustering_metrics: List[str] = field(default_factory=lambda: [
        "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"
    ])
    
    # Ensemble clustering
    ensemble_clustering: bool = True
    ensemble_method: str = "consensus"  # "consensus", "voting", "stacking"
    consensus_threshold: float = 0.6
    
    # Regime-specific clustering
    regime_specific_clustering: bool = True
    regime_clustering_weights: Dict[str, float] = field(default_factory=lambda: {
        "economic": 0.4,
        "financial": 0.3,
        "technical": 0.3
    })
    
    # Clustering validation
    clustering_validation: bool = True
    validation_metrics: List[str] = field(default_factory=lambda: [
        "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'primary_clustering_method': self.primary_clustering_method.value,
            'secondary_clustering_methods': [m.value for m in self.secondary_clustering_methods],
            'n_clusters_range': self.n_clusters_range,
            'clustering_metrics': self.clustering_metrics,
            'ensemble_clustering': self.ensemble_clustering,
            'ensemble_method': self.ensemble_method,
            'consensus_threshold': self.consensus_threshold,
            'regime_specific_clustering': self.regime_specific_clustering,
            'regime_clustering_weights': self.regime_clustering_weights,
            'clustering_validation': self.clustering_validation,
            'validation_metrics': self.validation_metrics
        }


@dataclass
class HybridModelingConfig:
    """Configuration for hybrid regime modeling."""
    
    # Economic modeling
    economic_modeling_enabled: bool = True
    economic_indicators: List[str] = field(default_factory=lambda: [
        "gdp_growth", "inflation_rate", "unemployment_rate", "interest_rate",
        "money_supply", "consumer_confidence", "business_confidence"
    ])
    economic_regime_types: List[EconomicRegimeType] = field(default_factory=lambda: [
        EconomicRegimeType.EXPANSION, EconomicRegimeType.RECESSION,
        EconomicRegimeType.RECOVERY, EconomicRegimeType.STAGNATION
    ])
    
    # Financial modeling
    financial_modeling_enabled: bool = True
    financial_indicators: List[str] = field(default_factory=lambda: [
        "liquidity_ratio", "credit_spread", "volatility_index", "risk_appetite",
        "market_sentiment", "institutional_flows", "retail_flows"
    ])
    financial_regime_types: List[FinancialRegimeType] = field(default_factory=lambda: [
        FinancialRegimeType.RISK_ON, FinancialRegimeType.RISK_OFF,
        FinancialRegimeType.LIQUIDITY_ABUNDANT, FinancialRegimeType.LIQUIDITY_CRUNCH
    ])
    
    # Model settings
    model_complexity: str = "medium"  # "low", "medium", "high"
    model_regularization: float = 0.01
    model_validation: bool = True
    model_uncertainty_quantification: bool = True
    
    # Regime transition modeling
    transition_modeling: bool = True
    transition_probability_threshold: float = 0.1
    transition_smoothness: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'economic_modeling_enabled': self.economic_modeling_enabled,
            'economic_indicators': self.economic_indicators,
            'economic_regime_types': [t.value for t in self.economic_regime_types],
            'financial_modeling_enabled': self.financial_modeling_enabled,
            'financial_indicators': self.financial_indicators,
            'financial_regime_types': [t.value for t in self.financial_regime_types],
            'model_complexity': self.model_complexity,
            'model_regularization': self.model_regularization,
            'model_validation': self.model_validation,
            'model_uncertainty_quantification': self.model_uncertainty_quantification,
            'transition_modeling': self.transition_modeling,
            'transition_probability_threshold': self.transition_probability_threshold,
            'transition_smoothness': self.transition_smoothness
        }


@dataclass
class HybridTaggingConfig:
    """Configuration for hybrid regime tagging."""
    
    # Tagging methods
    tagging_method: str = "ensemble"  # "ensemble", "hierarchical", "adaptive"
    tagging_confidence_threshold: float = 0.7
    tagging_uncertainty_threshold: float = 0.3
    
    # Tag types
    primary_tags: List[str] = field(default_factory=lambda: [
        "regime_type", "economic_regime", "financial_regime", "confidence_level"
    ])
    secondary_tags: List[str] = field(default_factory=lambda: [
        "volatility_level", "trend_direction", "momentum_strength", "liquidity_level"
    ])
    
    # Tag validation
    tag_validation: bool = True
    tag_consistency_check: bool = True
    tag_consistency_threshold: float = 0.8
    
    # Tag persistence
    tag_persistence: bool = True
    tag_history_length: int = 100
    tag_smoothing: bool = True
    tag_smoothing_window: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'tagging_method': self.tagging_method,
            'tagging_confidence_threshold': self.tagging_confidence_threshold,
            'tagging_uncertainty_threshold': self.tagging_uncertainty_threshold,
            'primary_tags': self.primary_tags,
            'secondary_tags': self.secondary_tags,
            'tag_validation': self.tag_validation,
            'tag_consistency_check': self.tag_consistency_check,
            'tag_consistency_threshold': self.tag_consistency_threshold,
            'tag_persistence': self.tag_persistence,
            'tag_history_length': self.tag_history_length,
            'tag_smoothing': self.tag_smoothing,
            'tag_smoothing_window': self.tag_smoothing_window
        }


# Configuration presets
def create_quick_hybrid_config() -> HybridRegimeConfig:
    """Create a quick hybrid configuration for fast testing."""
    return HybridRegimeConfig(
        n_regimes=6,
        min_regime_duration=5,
        max_regime_duration=60,
        clustering_method=ClusteringMethod.KMEANS,
        economic_modeling_enabled=False,
        financial_modeling_enabled=False,
        micro_regime_detection=False,
        enable_gpu_acceleration=False,
        save_results=False,
        verbose=False
    )


def create_comprehensive_hybrid_config() -> HybridRegimeConfig:
    """Create a comprehensive hybrid configuration for production use."""
    return HybridRegimeConfig(
        n_regimes=15,
        min_regime_duration=15,
        max_regime_duration=180,
        clustering_method=ClusteringMethod.HYBRID,
        economic_modeling_enabled=True,
        financial_modeling_enabled=True,
        micro_regime_detection=True,
        enable_gpu_acceleration=True,
        save_results=True,
        verbose=True
    )


def create_economic_focused_config() -> HybridRegimeConfig:
    """Create a configuration focused on economic regime detection."""
    return HybridRegimeConfig(
        n_regimes=10,
        economic_modeling_enabled=True,
        financial_modeling_enabled=False,
        economic_significance_threshold=0.8,
        financial_significance_threshold=0.5,
        trading_viability_threshold=0.7
    )


def create_financial_focused_config() -> HybridRegimeConfig:
    """Create a configuration focused on financial regime detection."""
    return HybridRegimeConfig(
        n_regimes=10,
        economic_modeling_enabled=False,
        financial_modeling_enabled=True,
        economic_significance_threshold=0.5,
        financial_significance_threshold=0.8,
        trading_viability_threshold=0.7
    )