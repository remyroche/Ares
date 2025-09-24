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
from .enhanced_data_operations import EnhancedDataOperations

# Import enhanced utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, ensure_directory,
    safe_file_exists, timed_operation, format_bytes,
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    integrate_with_m1_optimizers, memory_checkpoint, gpu_context
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, validate_finite,
    validate_positive, validate_range, safe_correlation,
    validate_numeric_array, MathValidationError
)
from src.utils.serialization_utils import UniversalSerializer

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
        
        # Initialize enhanced utilities
        self._initialize_enhanced_utilities()
        
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
    
    def _initialize_enhanced_utilities(self):
        """Initialize enhanced utility components."""
        try:
            # Initialize serialization
            self.serializer = UniversalSerializer()
            
            # Initialize M1 optimizations
            self.m1_integration = integrate_with_m1_optimizers()
            self.gpu_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
            
            # Initialize enhanced data operations
            self.data_operations = EnhancedDataOperations(
                data_dir="nas_regime_data",
                enable_validation=True
            )
            
            # Initialize enhanced matrix operations
            self.matrix_operations = EnhancedMatrixOperations(
                enable_gpu=True,
                enable_optimization=True,
                enable_m1_optimization=True
            )
            
            # Initialize enhanced ML common integration
            ml_config = MLCommonConfig(
                enable_validation=True,
                enable_feature_selection=True,
                enable_ensemble_methods=True,
                enable_evaluation=True,
                enable_optimization=True,
                enable_hardware_optimization=True,
                enable_m1_optimization=True,
                enable_serialization=True,
                math_validation_level='standard',
                enable_safe_math=True,
                enable_performance_monitoring=True
            )
            self.ml_common_integration = EnhancedMLCommonIntegration(ml_config)
            
            self.logger.info("✅ Enhanced utilities initialized successfully")
            self.logger.info(f"   M1 Integration: {'✅ Available' if self.m1_integration.get('success', False) else '❌ Not available'}")
            self.logger.info(f"   Data Operations: ✅ Initialized")
            self.logger.info(f"   Matrix Operations: ✅ Initialized")
            self.logger.info(f"   ML Common Integration: ✅ Initialized")
            
        except Exception as e:
            self.logger.warning(f"Enhanced utilities initialization failed: {e}")
            # Initialize fallback components
            self.serializer = None
            self.m1_integration = None
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.data_operations = None
            self.matrix_operations = None
            self.ml_common_integration = None

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

    @timed_operation
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
        try:
            # Use memory checkpoint for large datasets
            with memory_checkpoint("regime_detection"):
                # Pre-process market data using enhanced data operations
                if isinstance(market_data, pd.DataFrame):
                    # Validate market data
                    if self.data_operations:
                        validation_result = self.data_operations.validate_market_data(market_data)
                        if not validation_result['is_valid']:
                            self.logger.warning(f"Market data validation failed: {validation_result.get('errors', [])}")
                        
                        # Process market data for enhanced features
                        processed_data = self.data_operations.process_market_data(market_data)
                    else:
                        processed_data = market_data
                    
                    # Convert to numpy array for processing
                    market_data_array = processed_data.values
                else:
                    market_data_array = market_data
                    processed_data = None
                
                # Validate numeric data
                validate_numeric_array(market_data_array, "market_data")
                
                # Learn adaptive thresholds if enabled
                if (self.using_adaptive_thresholds and 
                    learn_thresholds and 
                    self.config.should_learn_thresholds(len(market_data_array))):
                    
                    self.logger.info("🧠 Learning adaptive thresholds from data...")
                    threshold_learning_success = self.config.learn_thresholds(
                        market_data_array, np.array([]), timestamps
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
                    market_data_array, timestamps, optimize_architecture, enable_meta_learning
                )
                
        except Exception as e:
            self.logger.error(f"Enhanced regime detection failed: {e}")
            # Return error result
            return PerfectNASResult(
                success=False,
                regime_predictions=np.array([]),
                regime_probabilities=np.array([]),
                economic_significance_scores=np.array([]),
                trading_viability_scores=np.array([]),
                regime_stability_scores=np.array([]),
                transition_probabilities=np.array([]),
                execution_time=0.0,
                metadata={'error': str(e)},
                error_message=str(e)
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
        
        # Add enhanced utilities information to metadata
        if result.metadata:
            result.metadata['enhanced_utilities'] = {
                'data_operations_available': self.data_operations is not None,
                'matrix_operations_available': self.matrix_operations is not None,
                'ml_common_integration_available': self.ml_common_integration is not None,
                'm1_integration_available': self.m1_integration is not None and self.m1_integration.get('success', False),
                'serialization_available': self.serializer is not None
            }
        
        return result
    
    def save_detector_state(self, filepath: str) -> bool:
        """Save detector state using enhanced serialization."""
        try:
            if not self.serializer:
                self.logger.warning("Serialization not available")
                return False
            
            state = {
                'config': self.config,
                'using_adaptive_thresholds': self.using_adaptive_thresholds,
                'enhanced_utilities_status': {
                    'data_operations': self.data_operations is not None,
                    'matrix_operations': self.matrix_operations is not None,
                    'ml_common_integration': self.ml_common_integration is not None,
                    'm1_integration': self.m1_integration is not None,
                    'serialization': self.serializer is not None
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Save state
            success = self.serializer.save(state, filepath)
            
            if success:
                self.logger.info(f"✅ Detector state saved to {filepath}")
            else:
                self.logger.error(f"Failed to save detector state to {filepath}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to save detector state: {e}")
            return False
    
    def load_detector_state(self, filepath: str) -> bool:
        """Load detector state using enhanced serialization."""
        try:
            if not self.serializer:
                self.logger.warning("Serialization not available")
                return False
            
            state = self.serializer.load(filepath)
            if state is None:
                self.logger.error(f"Failed to load state from {filepath}")
                return False
            
            # Restore configuration
            if 'config' in state:
                self.config = state['config']
                self.using_adaptive_thresholds = state.get('using_adaptive_thresholds', False)
            
            self.logger.info(f"✅ Detector state loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load detector state: {e}")
            return False