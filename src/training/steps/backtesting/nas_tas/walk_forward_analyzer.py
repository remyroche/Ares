"""
Walk-Forward Analysis for NAS-TAS

This module provides a simplified interface to the consolidated walk-forward analyzer
for NAS-TAS models, maintaining backward compatibility while delegating to the
consolidated implementation.
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Callable
import logging
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

# Import the consolidated walk-forward analyzer
from src.utils.nas_tas.walk_forward_analyzer import (
    WalkForwardAnalyzer as ConsolidatedWalkForwardAnalyzer,
    WalkForwardConfig as ConsolidatedWalkForwardConfig,
    WalkForwardResult as ConsolidatedWalkForwardResult,
    WalkForwardMode as ConsolidatedWalkForwardMode,
    ValidationMetric as ConsolidatedValidationMetric
)

logger = logging.getLogger(__name__)


# Re-export enums for backward compatibility
WalkForwardMode = ConsolidatedWalkForwardMode
ValidationMetric = ConsolidatedValidationMetric


# Legacy configuration class for backward compatibility
@dataclass
class WalkForwardConfig:
    """Legacy configuration for walk-forward analysis - maps to consolidated config."""
    
    # Walk-forward settings
    mode: WalkForwardMode = WalkForwardMode.EXPANDING_WINDOW
    initial_training_size: int = 1000  # Initial training window size
    validation_size: int = 100         # Validation window size
    step_size: int = 50               # Step size for moving window
    
    # Regime-aware settings
    enable_regime_aware_validation: bool = True
    regime_change_threshold: float = 0.3  # Threshold for regime change detection
    min_regime_samples: int = 50      # Minimum samples per regime
    
    # Model retraining
    enable_model_retraining: bool = True
    retraining_frequency: int = 10    # Retrain every N steps
    enable_incremental_learning: bool = True
    incremental_learning_rate: float = 0.01
    
    # Performance tracking
    validation_metrics: List[ValidationMetric] = field(default_factory=lambda: [
        ValidationMetric.ACCURACY,
        ValidationMetric.F1_SCORE,
        ValidationMetric.SHARPE_RATIO
    ])
    performance_threshold: float = 0.6  # Minimum performance threshold
    degradation_threshold: float = 0.1  # Performance degradation threshold
    
    # Data handling
    enable_data_preprocessing: bool = True
    enable_feature_engineering: bool = True
    enable_data_validation: bool = True
    
    # Output settings
    save_results: bool = True
    results_path: str = "walk_forward_results"
    enable_detailed_logging: bool = True
    enable_visualization: bool = True
    
    # Advanced features
    enable_ensemble_validation: bool = True
    enable_uncertainty_quantification: bool = True
    enable_regime_transition_analysis: bool = True
    
    def __post_init__(self):
        """Convert to consolidated config after initialization."""
        self.consolidated_config = ConsolidatedWalkForwardConfig(
            mode=self.mode,
            initial_training_size=self.initial_training_size,
            validation_size=self.validation_size,
            step_size=self.step_size,
            enable_regime_aware_validation=self.enable_regime_aware_validation,
            regime_change_threshold=self.regime_change_threshold,
            min_regime_samples=self.min_regime_samples,
            enable_model_retraining=self.enable_model_retraining,
            retraining_frequency=self.retraining_frequency,
            enable_incremental_learning=self.enable_incremental_learning,
            incremental_learning_rate=self.incremental_learning_rate,
            validation_metrics=self.validation_metrics,
            performance_threshold=self.performance_threshold,
            degradation_threshold=self.degradation_threshold,
            enable_data_preprocessing=self.enable_data_preprocessing,
            enable_feature_engineering=self.enable_feature_engineering,
            enable_data_validation=self.enable_data_validation,
            save_results=self.save_results,
            results_path=self.results_path,
            enable_detailed_logging=self.enable_detailed_logging,
            enable_visualization=self.enable_visualization,
            enable_ensemble_validation=self.enable_ensemble_validation,
            enable_uncertainty_quantification=self.enable_uncertainty_quantification,
            enable_regime_transition_analysis=self.enable_regime_transition_analysis
        )


# Legacy result class for backward compatibility
@dataclass
class WalkForwardResult:
    """Legacy result class - wraps consolidated result."""
    
    # Basic results
    success: bool
    execution_time: float
    total_folds: int
    successful_folds: int
    
    # Performance metrics
    overall_performance: Dict[str, float]
    fold_performance: List[Dict[str, Any]]
    regime_performance: Dict[int, Dict[str, float]]
    
    # Model evolution
    model_evolution: List[Dict[str, Any]]
    retraining_events: List[Dict[str, Any]]
    
    # Regime analysis
    regime_transitions: List[Dict[str, Any]]
    regime_stability: Dict[int, float]
    
    # Validation insights
    performance_trends: Dict[str, str]
    degradation_events: List[Dict[str, Any]]
    improvement_events: List[Dict[str, Any]]
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    configuration: Dict[str, Any] = field(default_factory=dict)
    data_statistics: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(self, consolidated_result: ConsolidatedWalkForwardResult):
        """Initialize from consolidated result."""
        self.success = consolidated_result.success
        self.execution_time = consolidated_result.execution_time
        self.total_folds = consolidated_result.total_folds
        self.successful_folds = consolidated_result.successful_folds
        self.overall_performance = consolidated_result.overall_performance
        self.fold_performance = consolidated_result.fold_performance
        self.regime_performance = consolidated_result.regime_performance
        self.model_evolution = consolidated_result.model_evolution
        self.retraining_events = consolidated_result.retraining_events
        self.regime_transitions = consolidated_result.regime_transitions
        self.regime_stability = consolidated_result.regime_stability
        self.performance_trends = consolidated_result.performance_trends
        self.degradation_events = consolidated_result.degradation_events
        self.improvement_events = consolidated_result.improvement_events
        self.error_message = consolidated_result.error_message
        self.warnings = consolidated_result.warnings
        self.configuration = consolidated_result.configuration
        self.data_statistics = consolidated_result.data_statistics


class WalkForwardAnalyzer:
    """
    Legacy walk-forward analyzer for NAS-TAS models.
    
    This class provides backward compatibility by wrapping the consolidated
    walk-forward analyzer from src.utils.nas_tas.walk_forward_analyzer
    """
    
    def __init__(self, config: WalkForwardConfig):
        """Initialize walk-forward analyzer.
        
        Args:
            config: Walk-forward configuration
        """
        self.legacy_config = config
        self.consolidated_analyzer = ConsolidatedWalkForwardAnalyzer(config.consolidated_config)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info("✅ Legacy Walk-Forward Analyzer initialized (NAS-TAS - using consolidated analyzer)")
        self.logger.info(f"   Mode: {config.mode.value}")
        self.logger.info(f"   Initial training size: {config.initial_training_size}")
        self.logger.info(f"   Validation size: {config.validation_size}")
        self.logger.info(f"   Step size: {config.step_size}")
    
    def register_models(self, 
                       regime_models: Dict[int, Dict[str, Any]],
                       ensemble_models: Optional[Dict[str, Any]] = None):
        """
        Register models for walk-forward analysis.
        
        Args:
            regime_models: Dictionary of regime_id -> {model_type: model_info}
            ensemble_models: Optional ensemble models
        """
        self.logger.info("📝 Registering models for walk-forward analysis (delegating to consolidated analyzer)")
        return self.consolidated_analyzer.register_models(regime_models, ensemble_models)
    
    def run_walk_forward_analysis(self, 
                                market_data: pd.DataFrame,
                                target_variable: str = 'close',
                                feature_columns: Optional[List[str]] = None) -> WalkForwardResult:
        """
        Run comprehensive walk-forward analysis.
        
        Args:
            market_data: Historical market data
            target_variable: Target variable for prediction
            feature_columns: List of feature columns
            
        Returns:
            WalkForwardResult with complete analysis results
        """
        self.logger.info("🚀 Starting legacy walk-forward analysis (NAS-TAS - delegating to consolidated analyzer)")
        
        try:
            # Run consolidated analysis
            consolidated_result = self.consolidated_analyzer.run_walk_forward_analysis(
                market_data=market_data,
                target_variable=target_variable,
                feature_columns=feature_columns
            )
            
            # Wrap result in legacy interface
            legacy_result = WalkForwardResult(consolidated_result)
            
            self.logger.info(f"✅ Legacy walk-forward analysis completed")
            self.logger.info(f"   Total folds: {legacy_result.total_folds}")
            self.logger.info(f"   Successful folds: {legacy_result.successful_folds}")
            self.logger.info(f"   Success rate: {legacy_result.successful_folds/legacy_result.total_folds:.2%}")
            
            return legacy_result
            
        except Exception as e:
            self.logger.error(f"❌ Legacy walk-forward analysis failed: {e}")
            raise
    
    def get_walk_forward_summary(self) -> Dict[str, Any]:
        """Get summary of walk-forward analysis."""
        return self.consolidated_analyzer.get_walk_forward_summary()