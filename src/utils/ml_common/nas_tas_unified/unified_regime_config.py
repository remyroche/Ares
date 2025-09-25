"""
Unified Regime Detection Configuration

This module provides a unified configuration system that combines the best aspects
of both TAS (Tree Architecture Search) and NAS (Neural Architecture Search) regime
detection systems with enhanced economic significance and trading viability evaluation.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class RegimeDetectionMethod(Enum):
    """Unified regime detection methods."""
    TAS_ONLY = "tas_only"
    NAS_ONLY = "nas_only"
    HYBRID_TAS_NAS = "hybrid_tas_nas"
    ADAPTIVE_SELECTION = "adaptive_selection"

class OptimizationStrategy(Enum):
    """Optimization strategies for regime detection."""
    PERFORMANCE_FIRST = "performance_first"
    ACCURACY_FIRST = "accuracy_first"
    BALANCED = "balanced"
    ECONOMIC_FOCUSED = "economic_focused"

class EconomicEvaluationMode(Enum):
    """Economic evaluation modes."""
    BASIC = "basic"
    ADVANCED = "advanced"
    POSITION_AWARE = "position_aware"
    REAL_TIME = "real_time"

@dataclass
class UnifiedRegimeConfig:
    """Unified configuration for regime detection combining TAS and NAS approaches."""
    
    # System identification
    system_name: str = "Unified TAS-NAS Regime Detection System"
    version: str = "1.0.0"
    
    # Core regime detection settings
    detection_method: RegimeDetectionMethod = RegimeDetectionMethod.HYBRID_TAS_NAS
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    n_regimes: int = 8
    primary_timeframe: str = "15m"
    min_regime_samples: int = 50
    max_regime_samples: int = 10000
    
    # TAS-specific configuration
    tas_config: Dict[str, Any] = field(default_factory=lambda: {
        'primary_architecture': 'hybrid_tree',
        'tree_depth': 6,
        'n_estimators': 1000,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'enable_clvsa_enhancement': True,
        'enable_statistical_methods': True,
        'enable_bootstrap_analysis': True,
        'bootstrap_iterations': 1000,
        'enable_meta_learning': True,
        'adaptation_rate': 0.1,
        'memory_size': 1000
    })
    
    # NAS-specific configuration
    nas_config: Dict[str, Any] = field(default_factory=lambda: {
        'primary_architecture': 'hybrid',
        'enable_neural_odes': True,
        'enable_vision_transformers': True,
        'enable_state_space_models': True,
        'enable_meta_learning': True,
        'search_strategy': 'evolutionary',
        'population_size': 50,
        'generations': 100,
        'mutation_rate': 0.1,
        'crossover_rate': 0.8,
        'elite_size': 5,
        'enable_uncertainty_quantification': True,
        'enable_multi_scale_analysis': True
    })
    
    # Economic evaluation configuration
    economic_evaluation: EconomicEvaluationMode = EconomicEvaluationMode.POSITION_AWARE
    economic_significance_threshold: float = 0.7
    trading_viability_threshold: float = 0.6
    risk_adjusted_return_threshold: float = 0.1
    
    # Economic evaluation parameters
    economic_params: Dict[str, Any] = field(default_factory=lambda: {
        'price_impact_weight': 0.3,
        'volume_significance_weight': 0.2,
        'volatility_impact_weight': 0.2,
        'trend_consistency_weight': 0.15,
        'market_efficiency_weight': 0.15,
        'minimum_profit_threshold': 0.001,
        'transaction_cost': 0.001,
        'position_holding_periods': [1, 5, 10, 20],
        'risk_free_rate': 0.02,
        'win_rate_thresholds': {
            'excellent': 0.7,
            'good': 0.6,
            'acceptable': 0.5,
            'poor': 0.4
        }
    })
    
    # Regime stability and transition settings
    regime_stability_threshold: float = 0.8
    transition_accuracy_threshold: float = 0.85
    transition_detection_sensitivity: float = 0.7
    enable_micro_regime_detection: bool = True
    micro_regime_sensitivity: float = 0.7
    
    # Performance and execution settings
    max_execution_time: float = 300.0  # seconds
    target_accuracy: float = 0.85
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    enable_checkpointing: bool = True
    checkpoint_interval: int = 10
    
    # Hardware optimization
    enable_hardware_optimization: bool = True
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_matrix_optimization: bool = True
    optimization_level: str = "maximum"
    
    # Multi-timeframe settings
    trading_timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h"])
    regime_detection_timeframe: str = "15m"
    enable_multi_timeframe_training: bool = True
    
    # Multi-objective optimization
    enable_multi_objective: bool = True
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        "regime_accuracy": 0.25,
        "economic_significance": 0.25,
        "trading_viability": 0.25,
        "regime_stability": 0.15,
        "computational_efficiency": 0.1
    })
    
    # Data quality and validation
    feature_importance_threshold: float = 0.1
    correlation_threshold: float = 0.8
    outlier_detection_enabled: bool = True
    data_quality_threshold: float = 0.8
    
    # Logging and monitoring
    log_level: str = "INFO"
    enable_profiling: bool = True
    enable_visualization: bool = True
    save_results: bool = True
    results_directory: str = "unified_regime_results"
    
    # Adaptive selection parameters
    adaptive_selection_params: Dict[str, Any] = field(default_factory=lambda: {
        'performance_window': 100,
        'accuracy_weight': 0.4,
        'efficiency_weight': 0.3,
        'economic_weight': 0.3,
        'switch_threshold': 0.1,
        'minimum_evaluations': 50
    })
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._setup_logging()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        try:
            # Validate regime count
            if not (2 <= self.n_regimes <= 20):
                raise ValueError(f"n_regimes must be between 2 and 20, got {self.n_regimes}")
            
            # Validate thresholds
            thresholds = [
                ('economic_significance_threshold', self.economic_significance_threshold),
                ('trading_viability_threshold', self.trading_viability_threshold),
                ('regime_stability_threshold', self.regime_stability_threshold),
                ('transition_accuracy_threshold', self.transition_accuracy_threshold)
            ]
            
            for name, value in thresholds:
                if not (0.0 <= value <= 1.0):
                    raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")
            
            # Validate timeframes
            if self.regime_detection_timeframe not in self.trading_timeframes:
                raise ValueError(f"regime_detection_timeframe must be in trading_timeframes")
            
            # Validate objective weights
            total_weight = sum(self.objective_weights.values())
            if abs(total_weight - 1.0) > 1e-6:
                logger.warning(f"Objective weights sum to {total_weight}, normalizing to 1.0")
                for obj in self.objective_weights:
                    self.objective_weights[obj] /= total_weight
            
            # Validate sample limits
            if self.min_regime_samples >= self.max_regime_samples:
                raise ValueError("min_regime_samples must be less than max_regime_samples")
            
            logger.info("✅ Unified regime configuration validation passed")
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            raise
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def get_tas_config(self) -> Dict[str, Any]:
        """Get TAS-specific configuration."""
        return self.tas_config.copy()
    
    def get_nas_config(self) -> Dict[str, Any]:
        """Get NAS-specific configuration."""
        return self.nas_config.copy()
    
    def get_economic_config(self) -> Dict[str, Any]:
        """Get economic evaluation configuration."""
        return {
            'mode': self.economic_evaluation.value,
            'thresholds': {
                'economic_significance': self.economic_significance_threshold,
                'trading_viability': self.trading_viability_threshold,
                'risk_adjusted_return': self.risk_adjusted_return_threshold
            },
            'parameters': self.economic_params.copy()
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance and execution configuration."""
        return {
            'max_execution_time': self.max_execution_time,
            'target_accuracy': self.target_accuracy,
            'enable_early_stopping': self.enable_early_stopping,
            'early_stopping_patience': self.early_stopping_patience,
            'enable_checkpointing': self.enable_checkpointing,
            'checkpoint_interval': self.checkpoint_interval,
            'hardware_optimization': {
                'enabled': self.enable_hardware_optimization,
                'gpu_acceleration': self.enable_gpu_acceleration,
                'memory_optimization': self.enable_memory_optimization,
                'matrix_optimization': self.enable_matrix_optimization,
                'level': self.optimization_level
            }
        }
    
    def get_multi_objective_config(self) -> Dict[str, Any]:
        """Get multi-objective optimization configuration."""
        return {
            'enabled': self.enable_multi_objective,
            'weights': self.objective_weights.copy(),
            'adaptive_selection': self.adaptive_selection_params.copy()
        }
    
    def should_use_tas(self, performance_metrics: Optional[Dict[str, Any]] = None) -> bool:
        """Determine if TAS should be used based on current configuration and performance."""
        if self.detection_method == RegimeDetectionMethod.TAS_ONLY:
            return True
        elif self.detection_method == RegimeDetectionMethod.NAS_ONLY:
            return False
        elif self.detection_method == RegimeDetectionMethod.HYBRID_TAS_NAS:
            return True  # Use both
        elif self.detection_method == RegimeDetectionMethod.ADAPTIVE_SELECTION:
            if performance_metrics is None:
                return True  # Default to TAS if no metrics available
            
            # Simple adaptive logic based on accuracy and efficiency
            tas_accuracy = performance_metrics.get('tas_accuracy', 0.5)
            nas_accuracy = performance_metrics.get('nas_accuracy', 0.5)
            tas_efficiency = performance_metrics.get('tas_efficiency', 0.5)
            nas_efficiency = performance_metrics.get('nas_efficiency', 0.5)
            
            tas_score = (tas_accuracy * 0.6) + (tas_efficiency * 0.4)
            nas_score = (nas_accuracy * 0.6) + (nas_efficiency * 0.4)
            
            return tas_score >= nas_score
        
        return True
    
    def should_use_nas(self, performance_metrics: Optional[Dict[str, Any]] = None) -> bool:
        """Determine if NAS should be used based on current configuration and performance."""
        if self.detection_method == RegimeDetectionMethod.NAS_ONLY:
            return True
        elif self.detection_method == RegimeDetectionMethod.TAS_ONLY:
            return False
        elif self.detection_method == RegimeDetectionMethod.HYBRID_TAS_NAS:
            return True  # Use both
        elif self.detection_method == RegimeDetectionMethod.ADAPTIVE_SELECTION:
            return not self.should_use_tas(performance_metrics)
        
        return False
    
    @classmethod
    def create_short_term_trading_config(cls) -> 'UnifiedRegimeConfig':
        """Create configuration optimized for short-term trading."""
        config = cls()
        config.detection_method = RegimeDetectionMethod.HYBRID_TAS_NAS
        config.optimization_strategy = OptimizationStrategy.PERFORMANCE_FIRST
        config.n_regimes = 6
        config.primary_timeframe = "15m"
        config.economic_evaluation = EconomicEvaluationMode.POSITION_AWARE
        config.max_execution_time = 120.0
        
        # Optimize TAS for speed
        config.tas_config.update({
            'tree_depth': 8,
            'n_estimators': 1500,
            'enable_clvsa_enhancement': True,
            'enable_meta_learning': True
        })
        
        # Optimize NAS for speed
        config.nas_config.update({
            'population_size': 30,
            'generations': 50,
            'enable_uncertainty_quantification': True
        })
        
        return config
    
    @classmethod
    def create_research_config(cls) -> 'UnifiedRegimeConfig':
        """Create configuration optimized for research and experimentation."""
        config = cls()
        config.detection_method = RegimeDetectionMethod.HYBRID_TAS_NAS
        config.optimization_strategy = OptimizationStrategy.ACCURACY_FIRST
        config.n_regimes = 12
        config.economic_evaluation = EconomicEvaluationMode.ADVANCED
        config.max_execution_time = 600.0
        config.enable_profiling = True
        config.enable_visualization = True
        
        # Maximize TAS capabilities
        config.tas_config.update({
            'tree_depth': 10,
            'n_estimators': 2000,
            'enable_statistical_methods': True,
            'enable_bootstrap_analysis': True,
            'bootstrap_iterations': 2000
        })
        
        # Maximize NAS capabilities
        config.nas_config.update({
            'population_size': 100,
            'generations': 200,
            'enable_neural_odes': True,
            'enable_vision_transformers': True,
            'enable_state_space_models': True
        })
        
        return config
    
    @classmethod
    def create_production_config(cls) -> 'UnifiedRegimeConfig':
        """Create configuration optimized for production deployment."""
        config = cls()
        config.detection_method = RegimeDetectionMethod.ADAPTIVE_SELECTION
        config.optimization_strategy = OptimizationStrategy.BALANCED
        config.n_regimes = 8
        config.economic_evaluation = EconomicEvaluationMode.POSITION_AWARE
        config.max_execution_time = 60.0
        config.enable_early_stopping = True
        config.log_level = "WARNING"
        config.enable_profiling = False
        
        # Optimize for production
        config.tas_config.update({
            'tree_depth': 6,
            'n_estimators': 1000,
            'enable_clvsa_enhancement': True,
            'enable_early_stopping': True
        })
        
        config.nas_config.update({
            'population_size': 20,
            'generations': 30,
            'enable_early_stopping': True
        })
        
        return config
    
    @classmethod
    def create_economic_focused_config(cls) -> 'UnifiedRegimeConfig':
        """Create configuration optimized for economic significance and trading viability."""
        config = cls()
        config.detection_method = RegimeDetectionMethod.HYBRID_TAS_NAS
        config.optimization_strategy = OptimizationStrategy.ECONOMIC_FOCUSED
        config.economic_evaluation = EconomicEvaluationMode.POSITION_AWARE
        
        # Adjust objective weights for economic focus
        config.objective_weights.update({
            "economic_significance": 0.4,
            "trading_viability": 0.4,
            "regime_accuracy": 0.15,
            "regime_stability": 0.05
        })
        
        # Optimize for economic evaluation
        config.tas_config.update({
            'enable_economic_evaluation': True,
            'enable_meta_learning': True
        })
        
        config.nas_config.update({
            'enable_uncertainty_quantification': True,
            'enable_multi_scale_analysis': True
        })
        
        return config