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
from .hybrid_architecture import HybridRegimeArchitecture

# Import enhanced integrations
from .enhanced_perfect_nas_regime_detector import EnhancedPerfectNASRegimeDetector
from .enhanced_matrix_operations import EnhancedMatrixOperations
from .enhanced_ml_common_integration import EnhancedMLCommonIntegration, MLCommonConfig
from .enhanced_nas_clustering_integration import EnhancedNASClusteringIntegration, NASClusteringConfig
from .enhanced_nas_modeling_integration import EnhancedNASModelingIntegration, NASModelingConfig

# Enhanced-only implementation with full tool integration

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
    Now includes full integration with existing tools infrastructure.
    """
    
    def __init__(self, config: PerfectNASConfig):
        """Initialize Perfect NAS Regime Detector (Enhanced Mode Only).

        Args:
            config: Perfect NAS configuration with full tool integration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize enhanced detector with full tool integration
        self.enhanced_detector = EnhancedPerfectNASRegimeDetector(config)
        self.logger.info(f"✅ Enhanced Perfect NAS Regime Detector initialized with full tool integration")

        self.logger.info(f"   Architecture: {config.primary_architecture.value}")
        self.logger.info(f"   Neural ODEs: {config.enable_neural_odes}")
        self.logger.info(f"   Vision Transformers: {config.enable_vision_transformers}")
        self.logger.info(f"   Meta-learning: {config.enable_meta_learning}")
        self.logger.info(f"   Search Strategy: {config.search_strategy.value}")
        self.logger.info(f"   Maximum Advancement: Enabled")
    
    def detect_regimes(self,
                      market_data: Union[pd.DataFrame, np.ndarray],
                      timestamps: Optional[np.ndarray] = None,
                      optimize_architecture: bool = True,
                      enable_meta_learning: bool = True) -> PerfectNASResult:
        """
        Detect market regimes using Enhanced Perfect NAS system with full tool integration.

        Args:
            market_data: Market data (OHLCV)
            timestamps: Optional timestamps
            optimize_architecture: Whether to optimize architecture
            enable_meta_learning: Whether to use meta-learning adaptation

        Returns:
            PerfectNASResult with regime detection results
        """
        # Use enhanced detector with full tool integration (only mode available)
        enhanced_result = self.enhanced_detector.detect_regimes(
            market_data, timestamps, optimize_architecture, enable_meta_learning
        )

        # Convert enhanced result to standard result
        return PerfectNASResult(
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