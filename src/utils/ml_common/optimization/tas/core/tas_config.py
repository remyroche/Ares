"""
Advanced Trading Architecture Search Configuration

This module provides comprehensive configuration for Trading-TAS with:
- Micro-regime detection
- Economic significance validation
- Hardware acceleration
- Multi-objective optimization with advanced constraints
- Neural architecture integration
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import numpy as np


class TASArchitectureType(Enum):
    """TAS architecture types for advanced trading."""
    TREE_ONLY = "tree_only"
    CVLSA_TREE = "cvlSA_tree"  # Cascade Variable Length Selection Architecture
    HYBRID_TREE_NEURAL = "hybrid_tree_neural"
    NEURAL_ONLY = "neural_only"
    ENSEMBLE_HIERARCHICAL = "ensemble_hierarchical"
    META_LEARNING = "meta_learning"


class MicroRegimeType(Enum):
    """Micro-regime types for subtle market changes."""
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"
    REVERSAL = "reversal"
    ACCELERATION = "acceleration"
    DECELERATION = "deceleration"
    VOLUME_SPIKE = "volume_spike"
    VOLATILITY_SPIKE = "volatility_spike"
    MOMENTUM_SHIFT = "momentum_shift"
    LIQUIDITY_CHANGE = "liquidity_change"


class TradingObjective(Enum):
    """Advanced trading-specific optimization objectives."""
    PROFITABILITY = "profitability"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    REGIME_STABILITY = "regime_stability"
    ADAPTATION_SPEED = "adaptation_speed"
    ROBUSTNESS = "robustness"
    TRANSACTION_COSTS = "transaction_costs"
    ECONOMIC_SIGNIFICANCE = "economic_significance"
    TRADING_VIABILITY = "trading_viability"
    MICRO_REGIME_ACCURACY = "micro_regime_accuracy"
    PREDICTION_CONFIDENCE = "prediction_confidence"


class MarketRegime(Enum):
    """Advanced market regime types for trading."""
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
    # Micro-regimes
    BREAKOUT_MICRO = "breakout_micro"
    CONSOLIDATION_MICRO = "consolidation_micro"
    REVERSAL_MICRO = "reversal_micro"
    ACCELERATION_MICRO = "acceleration_micro"
    DECELERATION_MICRO = "deceleration_micro"


@dataclass
class TASConfig:
    """Advanced configuration for Trading Architecture Search."""

    # Architecture type and components
    architecture_type: TASArchitectureType = TASArchitectureType.HYBRID_TREE_NEURAL
    enable_micro_regime_detection: bool = True
    enable_neural_components: bool = True
    enable_hierarchical_ensembles: bool = True
    enable_meta_learning: bool = True

    # Timeframe configuration
    primary_timeframe: str = "15m"
    micro_timeframe: str = "5m"
    regime_detection_window: int = 100  # Data points for regime detection
    adaptation_interval_minutes: int = 15

    # Regime configuration
    n_regimes: int = 12
    min_regime_duration: int = 15  # Minimum 15 minutes
    max_regime_duration: int = 180  # Maximum 3 hours
    data_driven_regimes: bool = True
    regime_stability_threshold: float = 0.7

    # Micro-regime configuration
    micro_regime_types: List[MicroRegimeType] = field(default_factory=lambda: [
        MicroRegimeType.BREAKOUT,
        MicroRegimeType.CONSOLIDATION,
        MicroRegimeType.REVERSAL,
        MicroRegimeType.ACCELERATION,
        MicroRegimeType.VOLUME_SPIKE,
        MicroRegimeType.VOLATILITY_SPIKE
    ])
    micro_regime_sensitivity: float = 0.7
    micro_regime_detection_threshold: float = 0.6

    # Multi-objective optimization
    trading_objectives: List[TradingObjective] = field(default_factory=lambda: [
        TradingObjective.PROFITABILITY,
        TradingObjective.SHARPE_RATIO,
        TradingObjective.ROBUSTNESS,
        TradingObjective.ECONOMIC_SIGNIFICANCE,
        TradingObjective.TRADING_VIABILITY
    ])
    objective_weights: List[float] = field(default_factory=lambda: [0.25, 0.2, 0.15, 0.2, 0.2])

    # Economic significance and validation
    economic_significance_threshold: float = 0.7
    trading_viability_threshold: float = 0.6
    regime_transition_cost: float = 0.05
    min_position_size: float = 0.01
    max_position_size: float = 0.1

    # Risk management
    max_drawdown_threshold: float = 0.15
    risk_adjusted_return_threshold: float = 0.1
    transaction_cost_penalty: float = 0.001
    slippage_assumption: float = 0.0005

    # Model configuration
    min_model_confidence: float = 0.6
    max_model_complexity: int = 100
    preferred_model_types: List[str] = field(default_factory=lambda: [
        'RandomForest', 'XGBoost', 'LightGBM', 'ExtraTrees',
        'NeuralNetwork', 'LSTM', 'Attention', 'NeuralODE'
    ])

    # Advanced search parameters
    search_space_config: Dict[str, Any] = field(default_factory=dict)
    enable_bayesian_optimization: bool = True
    enable_evolutionary_search: bool = True
    enable_random_search: bool = False
    n_search_iterations: int = 50
    population_size: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8

    # CVLSA-specific parameters
    enable_cvlSA_architecture: bool = True
    cvlSA_cascade_depth: int = 3
    cvlSA_variable_selection_methods: List[str] = field(default_factory=lambda: [
        'variance_threshold', 'mutual_information', 'tree_importance',
        'correlation_filter', 'recursive_elimination'
    ])
    cvlSA_feature_ensemble_method: str = "intersection"
    cvlSA_optimization_objective: str = "cascade_efficiency"

    # Hardware acceleration
    enable_hardware_acceleration: bool = True
    enable_gpu_acceleration: bool = True
    enable_batch_processing: bool = True
    batch_size: int = 1000
    max_memory_usage: float = 0.8

    # Meta-learning
    meta_learning_enabled: bool = True
    regime_similarity_threshold: float = 0.8
    adaptation_history_length: int = 100
    transfer_learning_enabled: bool = True

    # Performance tracking
    enable_performance_tracking: bool = True
    performance_tracking_interval: int = 60
    save_model_snapshots: bool = True
    enable_uncertainty_quantification: bool = True

    # Integration settings
    integrate_with_nas_clustering: bool = True
    use_existing_regime_detection: bool = True
    output_format: str = "comprehensive"

    # Validation settings
    validation_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create_advanced_trading_config(cls) -> 'TASConfig':
        """Create configuration optimized for advanced trading."""
        return cls(
            architecture_type=TASArchitectureType.HYBRID_TREE_NEURAL,
            enable_micro_regime_detection=True,
            enable_neural_components=True,
            enable_hierarchical_ensembles=True,
            enable_meta_learning=True,
            primary_timeframe="15m",
            micro_timeframe="5m",
            n_regimes=12,
            min_regime_duration=15,
            max_regime_duration=180,
            data_driven_regimes=True,
            regime_stability_threshold=0.7,
            micro_regime_sensitivity=0.7,
            micro_regime_detection_threshold=0.6,
            trading_objectives=[
                TradingObjective.PROFITABILITY,
                TradingObjective.SHARPE_RATIO,
                TradingObjective.ROBUSTNESS,
                TradingObjective.ECONOMIC_SIGNIFICANCE,
                TradingObjective.TRADING_VIABILITY,
                TradingObjective.MICRO_REGIME_ACCURACY
            ],
            objective_weights=[0.25, 0.2, 0.15, 0.2, 0.15, 0.05],
            economic_significance_threshold=0.7,
            trading_viability_threshold=0.6,
            regime_transition_cost=0.05,
            enable_hardware_acceleration=True,
            enable_gpu_acceleration=True,
            enable_batch_processing=True,
            batch_size=1000,
            meta_learning_enabled=True,
            regime_similarity_threshold=0.8,
            enable_uncertainty_quantification=True,
            integrate_with_nas_clustering=True,
            enable_cvlSA_architecture=True,
            cvlSA_cascade_depth=3,
            search_space_config={
                'tree_search_space': {
                    'min_depth': [3, 5, 8, 10, 15],
                    'max_depth': [5, 10, 15, 20, 25],
                    'min_trees': [50, 100, 200, 300, 500],
                    'max_trees': [100, 200, 400, 600, 800],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 5, 10],
                    'max_features': ['sqrt', 'log2', 'auto', 0.3, 0.5, 0.8]
                },
                'neural_search_space': {
                    'hidden_dims': [
                        [32], [64], [128], [256],
                        [64, 32], [128, 64], [256, 128],
                        [128, 64, 32], [256, 128, 64]
                    ],
                    'activation_functions': ['relu', 'tanh', 'leaky_relu', 'elu', 'gelu', 'swish'],
                    'dropout_rates': [0.0, 0.1, 0.2, 0.3, 0.4],
                    'use_lstm': [True, False],
                    'use_attention': [True, False],
                    'use_batch_norm': [True, False]
                }
            },
            validation_config={
                'min_regime_stability': 0.6,
                'min_economic_significance': 0.7,
                'min_trading_viability': 0.6,
                'max_regime_volatility': 0.3,
                'min_prediction_confidence': 0.6,
                'max_model_complexity': 100
            }
        )

    @classmethod
    def create_cvlSA_tree_config(cls) -> 'TASConfig':
        """Create configuration optimized for CVLSA tree architecture."""
        return cls(
            architecture_type=TASArchitectureType.CVLSA_TREE,
            enable_micro_regime_detection=True,
            enable_neural_components=False,  # Tree-only
            enable_hierarchical_ensembles=True,
            enable_meta_learning=True,
            primary_timeframe="15m",
            micro_timeframe="5m",
            n_regimes=12,
            min_regime_duration=15,
            max_regime_duration=180,
            data_driven_regimes=True,
            regime_stability_threshold=0.7,
            micro_regime_sensitivity=0.8,  # Higher sensitivity for CVLSA
            micro_regime_detection_threshold=0.7,
            trading_objectives=[
                TradingObjective.PROFITABILITY,
                TradingObjective.SHARPE_RATIO,
                TradingObjective.ROBUSTNESS,
                TradingObjective.ECONOMIC_SIGNIFICANCE,
                TradingObjective.TRADING_VIABILITY,
                TradingObjective.MICRO_REGIME_ACCURACY
            ],
            objective_weights=[0.25, 0.2, 0.15, 0.2, 0.15, 0.05],
            economic_significance_threshold=0.7,
            trading_viability_threshold=0.6,
            regime_transition_cost=0.05,
            enable_hardware_acceleration=False,  # Tree-focused, no GPU needed
            enable_gpu_acceleration=False,
            enable_batch_processing=True,
            batch_size=500,  # Smaller batches for tree models
            meta_learning_enabled=True,
            regime_similarity_threshold=0.85,  # Higher similarity for cascade
            enable_uncertainty_quantification=True,
            integrate_with_nas_clustering=True,
            enable_cvlSA_architecture=True,
            cvlSA_cascade_depth=3,
            cvlSA_variable_selection_methods=[
                'variance_threshold',
                'mutual_information',
                'tree_importance',
                'correlation_filter',
                'recursive_elimination'
            ],
            cvlSA_feature_ensemble_method="intersection",
            cvlSA_optimization_objective="cascade_efficiency",
            search_space_config={
                'tree_search_space': {
                    'min_depth': [3, 5, 8, 10, 15],
                    'max_depth': [5, 10, 15, 20, 25],
                    'min_trees': [50, 100, 200, 300, 500],
                    'max_trees': [100, 200, 400, 600, 800],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 5, 10],
                    'max_features': ['sqrt', 'log2', 'auto', 0.3, 0.5, 0.8],
                    'splitting_strategies': [
                        'gini', 'entropy', 'log_loss',
                        'xgb_gbtree', 'xgb_dart',
                        'lgb_gbdt', 'lgb_rf', 'lgb_dart'
                    ]
                },
                'cvlSA_search_space': {
                    'cascade_depths': [2, 3, 4, 5],
                    'ensemble_methods': ['voting', 'stacking', 'weighted_voting'],
                    'feature_selection_methods': [
                        'variance_threshold', 'mutual_information', 'tree_importance'
                    ],
                    'optimization_objectives': ['accuracy', 'efficiency', 'robustness']
                }
            },
            validation_config={
                'min_regime_stability': 0.6,
                'min_economic_significance': 0.7,
                'min_trading_viability': 0.6,
                'max_regime_volatility': 0.3,
                'min_prediction_confidence': 0.6,
                'max_model_complexity': 100,
                'min_cascade_efficiency': 0.7,
                'min_variable_selection_accuracy': 0.8
            }
        )

    @classmethod
    def create_tree_only_config(cls) -> 'TASConfig':
        """Create configuration for tree-only architectures."""
        return cls(
            architecture_type=TASArchitectureType.TREE_ONLY,
            enable_micro_regime_detection=True,
            enable_neural_components=False,
            enable_hierarchical_ensembles=True,
            enable_meta_learning=False,  # Simpler tree-only approach
            primary_timeframe="15m",
            micro_timeframe="5m",
            n_regimes=8,  # Fewer regimes for tree-only
            min_regime_duration=15,
            max_regime_duration=180,
            data_driven_regimes=True,
            regime_stability_threshold=0.6,
            micro_regime_sensitivity=0.6,
            micro_regime_detection_threshold=0.5,
            trading_objectives=[
                TradingObjective.PROFITABILITY,
                TradingObjective.SHARPE_RATIO,
                TradingObjective.ROBUSTNESS
            ],
            objective_weights=[0.4, 0.3, 0.3],
            economic_significance_threshold=0.6,  # Lower threshold for tree-only
            trading_viability_threshold=0.5,
            regime_transition_cost=0.05,
            enable_hardware_acceleration=False,
            enable_gpu_acceleration=False,
            enable_batch_processing=True,
            batch_size=500,
            meta_learning_enabled=False,
            regime_similarity_threshold=0.7,
            enable_uncertainty_quantification=True,
            integrate_with_nas_clustering=False,  # Tree-only, no neural integration
            enable_cvlSA_architecture=False,  # Use standard tree architecture
            search_space_config={
                'tree_search_space': {
                    'min_depth': [3, 5, 8, 10],
                    'max_depth': [5, 10, 15, 20],
                    'min_trees': [50, 100, 200],
                    'max_trees': [100, 200, 400],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 5],
                    'max_features': ['sqrt', 'log2', 'auto']
                }
            },
            validation_config={
                'min_regime_stability': 0.5,
                'min_economic_significance': 0.6,
                'min_trading_viability': 0.5,
                'max_regime_volatility': 0.4,
                'min_prediction_confidence': 0.5,
                'max_model_complexity': 50
            }
        )

    def get_tree_search_space(self) -> Dict[str, Any]:
        """Get tree-specific search space configuration."""
        return self.search_space_config.get('tree_search_space', {})

    def get_neural_search_space(self) -> Dict[str, Any]:
        """Get neural-specific search space configuration."""
        return self.search_space_config.get('neural_search_space', {})

    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration."""
        return {
            'regime_stability_threshold': self.regime_stability_threshold,
            'economic_significance_threshold': self.economic_significance_threshold,
            'trading_viability_threshold': self.trading_viability_threshold,
            'min_model_confidence': self.min_model_confidence,
            'max_model_complexity': self.max_model_complexity,
            **self.validation_config
        }