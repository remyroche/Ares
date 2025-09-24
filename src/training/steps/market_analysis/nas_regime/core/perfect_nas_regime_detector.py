"""
Perfect NAS Regime Detector

The ultimate regime detection system that combines:
- Advanced neural architectures (Neural ODEs, Vision Transformers)
- True NAS search with evolutionary algorithms
- Economic significance evaluation
- Trading viability assessment
- Meta-learning for regime adaptation
- Production optimization
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from dataclasses import dataclass

# Enhanced-only implementation imports

# Import new components
from .perfect_nas_config import PerfectNASConfig, NeuralArchitectureType
from .enhanced_perfect_nas_config import EnhancedPerfectNASConfig, ThresholdLearningMode
from .hybrid_architecture import HybridRegimeArchitecture

# Import enhanced integrations
from .enhanced_perfect_nas_regime_detector import EnhancedPerfectNASRegimeDetector
from .enhanced_matrix_operations import EnhancedMatrixOperations
from .enhanced_ml_common_integration import EnhancedMLCommonIntegration, MLCommonConfig
from .enhanced_nas_clustering_integration import EnhancedNASClusteringIntegration, NASClusteringConfig
from .enhanced_nas_modeling_integration import EnhancedNASModelingIntegration, NASModelingConfig

# Enhanced-only implementation with full tool integration

# Import shared utilities from hybrid regime system
try:
    from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.search_strategies import SearchStrategyManager, SearchStrategyConfig
    from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.analysis_components import SharedClusteringUtilities
    from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.position_aware_trading import PositionAwareTradingAnalyzer, PositionAwareConfig
    SHARED_UTILITIES_AVAILABLE = True
    POSITION_AWARE_AVAILABLE = True
except ImportError:
    SHARED_UTILITIES_AVAILABLE = False
    POSITION_AWARE_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class PerfectNASResult:
    """Result from Perfect NAS Regime Detection."""
    success: bool
    regime_predictions: np.ndarray
    regime_probabilities: np.ndarray
    economic_significance_scores: np.ndarray
    trading_viability_scores: np.ndarray
    regime_stability_scores: np.ndarray
    transition_probabilities: np.ndarray
    micro_regimes: Optional[Dict[str, Any]] = None
    architecture_performance: Optional[Dict[str, Any]] = None
    uncertainty_estimates: Optional[np.ndarray] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None
    error_message: Optional[str] = None

class PerfectNASRegimeDetector:
    """
    Perfect NAS Regime Detector - The ultimate regime qualification system.
    
    Combines the best of both nas_modeling and nas_clustering systems with
    enhanced economic significance and trading viability evaluation.
    Now includes full integration with existing tools infrastructure and
    adaptive threshold learning for data-driven thresholds.
    """
    
    def __init__(self, config: Union[PerfectNASConfig, EnhancedPerfectNASConfig]):
        """Initialize Perfect NAS Regime Detector (Enhanced Mode Only).

        Args:
            config: Perfect NAS configuration with full tool integration and adaptive thresholds
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Check if using enhanced configuration with adaptive thresholds
        self.using_adaptive_thresholds = isinstance(config, EnhancedPerfectNASConfig)
        
        # Initialize enhanced detector with full tool integration
        self.enhanced_detector = EnhancedPerfectNASRegimeDetector(config)

        # Initialize shared utilities
        self._initialize_shared_utilities()
        self._initialize_position_aware_analyzer()

        self.logger.info(f"✅ Enhanced Perfect NAS Regime Detector initialized with full tool integration")

        self.logger.info(f"   Architecture: {config.primary_architecture.value}")
        self.logger.info(f"   Neural ODEs: {config.enable_neural_odes}")
        self.logger.info(f"   Vision Transformers: {config.enable_vision_transformers}")
        self.logger.info(f"   Meta-learning: {config.enable_meta_learning}")
        self.logger.info(f"   Search Strategy: {config.search_strategy.value}")
        self.logger.info(f"   Maximum Advancement: Enabled")
        
        if self.using_adaptive_thresholds:
            self.logger.info(f"   Adaptive Thresholds: {config.adaptive_thresholds.learning_mode.value}")
            self.logger.info(f"   Economic Learning: {config.adaptive_thresholds.enable_economic_learning}")
            self.logger.info(f"   Trading Learning: {config.adaptive_thresholds.enable_trading_learning}")
            self.logger.info(f"   Stability Learning: {config.adaptive_thresholds.enable_stability_learning}")

    def _initialize_shared_utilities(self):
        """Initialize shared utilities from hybrid regime system."""
        if not SHARED_UTILITIES_AVAILABLE:
            self.logger.warning("⚠️ Shared utilities not available")
            self.search_strategy_manager = None
            self.shared_clustering = None
            return

        try:
            # Initialize search strategy manager
            search_config = SearchStrategyConfig(
                max_iterations=50,
                n_initial_points=10,
                acquisition_function="expected_improvement",
                exploration_weight=0.1,
                convergence_threshold=1e-6,
                parallel_evaluations=1,
                random_state=42,
                use_bayesian_optimization=True,
                use_grid_optimization=True
            )
            self.search_strategy_manager = SearchStrategyManager(search_config)

            # Initialize shared clustering utilities
            self.shared_clustering = SharedClusteringUtilities()

            self.logger.info("✅ Shared utilities initialized")
        except Exception as e:
            self.logger.warning(f"Shared utilities initialization failed: {e}")
            self.search_strategy_manager = None
            self.shared_clustering = None

    def _initialize_position_aware_analyzer(self):
        """Initialize position-aware trading analyzer."""
        if not POSITION_AWARE_AVAILABLE:
            self.position_analyzer = None
            return

        try:
            position_config = PositionAwareConfig(
                minimum_profit_threshold=0.001,
                transaction_cost=0.001,
                position_holding_periods=[1, 5, 10, 20],
                risk_free_rate=0.02,
                win_rate_thresholds={
                    'excellent': 0.7,
                    'good': 0.6,
                    'acceptable': 0.5,
                    'poor': 0.4
                }
            )
            self.position_analyzer = PositionAwareTradingAnalyzer(position_config)
            self.logger.info("✅ Position-aware trading analyzer initialized")
        except Exception as e:
            self.logger.warning(f"Position-aware analyzer initialization failed: {e}")
            self.position_analyzer = None

    def detect_regimes(self,
                      market_data: Union[pd.DataFrame, np.ndarray],
                      timestamps: Optional[np.ndarray] = None,
                      optimize_architecture: bool = True,
                      enable_meta_learning: bool = True,
                      learn_thresholds: bool = True) -> PerfectNASResult:
        """
        Detect market regimes using Enhanced Perfect NAS system with full tool integration
        and adaptive threshold learning.

        Args:
            market_data: Market data (OHLCV)
            timestamps: Optional timestamps
            optimize_architecture: Whether to optimize architecture
            enable_meta_learning: Whether to use meta-learning adaptation
            learn_thresholds: Whether to learn adaptive thresholds

        Returns:
            PerfectNASResult with regime detection results
        """
        # Learn adaptive thresholds if enabled
        if (self.using_adaptive_thresholds and 
            learn_thresholds and 
            self.config.should_learn_thresholds(len(market_data))):
            
            self.logger.info("🧠 Learning adaptive thresholds from data...")
            threshold_learning_success = self.config.learn_thresholds(
                market_data, np.array([]), timestamps
            )
            
            if threshold_learning_success:
                self.logger.info("✅ Adaptive thresholds learned successfully")
                # Get threshold explanations
                explanations = self.config.get_threshold_explanations()
                for metric, explanation in explanations.items():
                    self.logger.info(f"   {metric}: {explanation}")
            else:
                self.logger.warning("⚠️ Adaptive threshold learning failed, using fallback thresholds")
        
        # Use enhanced detector with full tool integration (only mode available)
        enhanced_result = self.enhanced_detector.detect_regimes(
            market_data, timestamps, optimize_architecture, enable_meta_learning
        )

        # Convert enhanced result to standard result
        result = PerfectNASResult(
            success=enhanced_result.success,
            regime_predictions=enhanced_result.regime_predictions,
            regime_probabilities=enhanced_result.regime_probabilities,
            economic_significance_scores=enhanced_result.economic_significance_scores,
            trading_viability_scores=enhanced_result.trading_viability_scores,
            regime_stability_scores=enhanced_result.regime_stability_scores,
            transition_probabilities=enhanced_result.transition_probabilities,
            micro_regimes=enhanced_result.micro_regimes,
            architecture_performance=enhanced_result.architecture_performance,
            uncertainty_estimates=enhanced_result.uncertainty_estimates,
            execution_time=enhanced_result.execution_time,
            metadata=enhanced_result.metadata,
            error_message=enhanced_result.error_message
        )
        
        # Add adaptive threshold information to metadata
        if self.using_adaptive_thresholds and result.metadata:
            result.metadata['adaptive_thresholds'] = {
                'enabled': True,
                'learning_mode': self.config.adaptive_thresholds.learning_mode.value,
                'effective_thresholds': self.config.get_effective_thresholds(),
                'confidence_intervals': self.config.get_threshold_confidence_intervals(),
                'threshold_explanations': self.config.get_threshold_explanations()
            }
        
        return result