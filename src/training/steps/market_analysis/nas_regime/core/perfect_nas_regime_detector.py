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
from datetime import datetime
from dataclasses import dataclass
from src.utils.tprint import (tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_progress, tprint_performance, tprint_timer)

# Import unified utilities
try:
    from src.utils.ml_common.nas_tas_unified import (
        UnifiedRegimeDetector, UnifiedRegimeConfig, UnifiedRegimeResult,
        RegimeDetectionMethod
    )
    from src.utils.common_operations import (
        CommonUtilities, memory_checkpoint, gpu_context, timed_operation
    )
    UNIFIED_UTILITIES_AVAILABLE = True
except ImportError:
    UNIFIED_UTILITIES_AVAILABLE = False

# Import enhanced integrations (keep for compatibility)
from .enhanced_perfect_nas_regime_detector import EnhancedPerfectNASRegimeDetector
from .enhanced_matrix_operations import EnhancedMatrixOperations
from .enhanced_ml_common_integration import EnhancedMLCommonIntegration, MLCommonConfig
from .enhanced_nas_clustering_integration import EnhancedNASClusteringIntegration, NASClusteringConfig
from .enhanced_nas_modeling_integration import EnhancedNASModelingIntegration, NASModelingConfig
from .enhanced_data_operations import EnhancedDataOperations

# Import configuration
from .perfect_nas_config import PerfectNASConfig, NeuralArchitectureType
from .enhanced_perfect_nas_config import EnhancedPerfectNASConfig, ThresholdLearningMode

# Keep only essential imports for legacy compatibility
from src.utils.math_validation import (
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
        tprint("🚀 [PERFECT_NAS_REGIME_DETECTOR] Initializing Perfect NAS Regime Detector", color="cyan", bold=True)
        tprint(f"📊 [PERFECT_NAS_REGIME_DETECTOR] Architecture: {config.primary_architecture.value}", color="blue")
        tprint(f"📊 [PERFECT_NAS_REGIME_DETECTOR] Neural ODEs: {config.enable_neural_odes}", color="blue")
        tprint(f"📊 [PERFECT_NAS_REGIME_DETECTOR] Vision Transformers: {config.enable_vision_transformers}", color="blue")
        tprint(f"📊 [PERFECT_NAS_REGIME_DETECTOR] Meta-learning: {config.enable_meta_learning}", color="blue")
        tprint(f"📊 [PERFECT_NAS_REGIME_DETECTOR] Search Strategy: {config.search_strategy.value}", color="blue")
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Check if using enhanced configuration with adaptive thresholds
        self.using_adaptive_thresholds = isinstance(config, EnhancedPerfectNASConfig)
        tprint(f"📊 [PERFECT_NAS_REGIME_DETECTOR] Using adaptive thresholds: {self.using_adaptive_thresholds}", color="blue")
        
        # Initialize unified utilities if available
        if UNIFIED_UTILITIES_AVAILABLE:
            tprint("🔧 [PERFECT_NAS_REGIME_DETECTOR] Initializing unified utilities", color="yellow")
            try:
                self.common_utils = CommonUtilities()
                self.unified_detector = UnifiedRegimeDetector(self._create_unified_config())
                tprint_success("✅ Unified utilities initialized")
                self.logger.info("✅ Unified utilities initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Unified utilities initialization failed: {e}")
                self.logger.warning(f"Unified utilities initialization failed: {e}")
                self.common_utils = None
                self.unified_detector = None
        else:
            tprint_warning("⚠️ Unified utilities not available, using enhanced detector")
            self.unified_detector = None
        
        # Initialize enhanced detector as fallback
        tprint("🧠 [PERFECT_NAS_REGIME_DETECTOR] Initializing enhanced detector", color="yellow")
        self.enhanced_detector = EnhancedPerfectNASRegimeDetector(config)

        tprint("✅ [PERFECT_NAS_REGIME_DETECTOR] Perfect NAS Regime Detector initialized successfully", color="green")
        self.logger.info(f"✅ Enhanced Perfect NAS Regime Detector initialized with full tool integration")

        self.logger.info(f"   Architecture: {config.primary_architecture.value}")
        self.logger.info(f"   Neural ODEs: {config.enable_neural_odes}")
        self.logger.info(f"   Vision Transformers: {config.enable_vision_transformers}")
        self.logger.info(f"   Meta-learning: {config.enable_meta_learning}")
        self.logger.info(f"   Search Strategy: {config.search_strategy.value}")
        self.logger.info(f"   Maximum Advancement: Enabled")
        
        if self.using_adaptive_thresholds:
            tprint(f"📊 [PERFECT_NAS_REGIME_DETECTOR] Adaptive Thresholds: {config.adaptive_thresholds.learning_mode.value}", color="blue")
            tprint(f"📊 [PERFECT_NAS_REGIME_DETECTOR] Economic Learning: {config.adaptive_thresholds.enable_economic_learning}", color="blue")
            tprint(f"📊 [PERFECT_NAS_REGIME_DETECTOR] Trading Learning: {config.adaptive_thresholds.enable_trading_learning}", color="blue")
            tprint(f"📊 [PERFECT_NAS_REGIME_DETECTOR] Stability Learning: {config.adaptive_thresholds.enable_stability_learning}", color="blue")
            self.logger.info(f"   Adaptive Thresholds: {config.adaptive_thresholds.learning_mode.value}")
            self.logger.info(f"   Economic Learning: {config.adaptive_thresholds.enable_economic_learning}")
            self.logger.info(f"   Trading Learning: {config.adaptive_thresholds.enable_trading_learning}")
            self.logger.info(f"   Stability Learning: {config.adaptive_thresholds.enable_stability_learning}")

    def _create_unified_config(self) -> UnifiedRegimeConfig:
        """Create unified configuration from NAS configuration."""
        return UnifiedRegimeConfig(
            detection_method=RegimeDetectionMethod.NAS_ONLY,
            n_regimes=self.config.n_regimes,
            primary_timeframe=self.config.primary_timeframe,
            min_regime_samples=self.config.min_regime_duration,
            max_regime_samples=self.config.max_regime_duration,
            economic_significance_threshold=self.config.economic_significance_threshold,
            trading_viability_threshold=self.config.trading_viability_threshold,
            max_execution_time=self.config.max_execution_time,
            enable_hardware_optimization=self.config.hardware_config.enable_gpu_acceleration
        )

    def _convert_unified_to_nas_result(self, unified_result: UnifiedRegimeResult) -> PerfectNASResult:
        """Convert unified result to NAS result format."""
        return PerfectNASResult(
            success=unified_result.success,
            regime_predictions=unified_result.regime_predictions,
            regime_probabilities=unified_result.regime_probabilities,
            economic_significance_scores=unified_result.economic_significance_scores,
            trading_viability_scores=unified_result.trading_viability_scores,
            regime_stability_scores=unified_result.regime_stability_scores,
            transition_probabilities=unified_result.transition_probabilities,
            micro_regimes=unified_result.micro_regimes,
            uncertainty_estimates=unified_result.uncertainty_estimates,
            execution_time=unified_result.execution_time,
            metadata=unified_result.metadata,
            error_message=unified_result.error_message
        )

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
            # Use unified detector if available
            if self.unified_detector:
                tprint_info("🧠 Using unified regime detector")
                unified_result = self.unified_detector.detect_regimes(market_data, timestamps)
                return self._convert_unified_to_nas_result(unified_result)

            # Fallback to enhanced detector
            tprint_info("🔄 Using enhanced NAS regime detection")
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
                
                # Validate numeric data using enhanced validation
                validate_numeric_array(market_data_array, "market_data")
                
                # Additional validation using ML common integration if available
                if self.ml_common_integration:
                    ml_validation = self.ml_common_integration.validate_data(market_data_array, 'market_data')
                    if not ml_validation['is_valid']:
                        self.logger.warning(f"ML validation failed: {ml_validation.get('errors', [])}")
                
                # Pre-process data using enhanced matrix operations if available
                if self.matrix_operations:
                    with gpu_context("data_preprocessing"):
                        # Normalize data for better regime detection
                        normalized_data = self.matrix_operations.normalize_data(market_data_array, method='robust')
                        
                        # Calculate enhanced features for regime detection
                        enhanced_features = self.matrix_operations.calculate_enhanced_features(normalized_data, window=20)
                        
                        # Combine original data with enhanced features
                        if enhanced_features:
                            feature_arrays = []
                            for feature_name, feature_data in enhanced_features.items():
                                if feature_data.ndim == 1:
                                    feature_arrays.append(feature_data.reshape(-1, 1))
                                else:
                                    feature_arrays.append(feature_data)
                            
                            if feature_arrays:
                                enhanced_features_array = np.concatenate(feature_arrays, axis=1)
                                market_data_array = np.concatenate([market_data_array, enhanced_features_array], axis=1)
                                self.logger.info(f"✅ Enhanced data with {len(enhanced_features)} feature types")
                            else:
                                self.logger.info("⚠️ No enhanced features generated, using original data")
                else:
                    self.logger.info("⚠️ Matrix operations not available, using original data")
                
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

        # Post-process results using enhanced tools
        if enhanced_result.success and self.matrix_operations:
            try:
                with gpu_context("result_postprocessing"):
                    # Calculate regime stability using enhanced matrix operations
                    if enhanced_result.regime_predictions is not None and len(enhanced_result.regime_predictions) > 0:
                        enhanced_stability = self.matrix_operations.calculate_regime_stability(
                            enhanced_result.regime_predictions, timestamps if timestamps is not None else np.arange(len(enhanced_result.regime_predictions))
                        )
                        
                        # Update stability scores if calculated
                        if enhanced_stability is not None:
                            enhanced_result.regime_stability_scores = enhanced_stability
                            self.logger.info("✅ Enhanced regime stability calculated")
                    
                    # Calculate transition probabilities using enhanced operations
                    if enhanced_result.regime_predictions is not None:
                        n_regimes = len(np.unique(enhanced_result.regime_predictions))
                        enhanced_transitions = self.matrix_operations.calculate_transition_probabilities(
                            enhanced_result.regime_predictions, n_regimes
                        )
                        
                        # Update transition probabilities if calculated
                        if enhanced_transitions is not None:
                            enhanced_result.transition_probabilities = enhanced_transitions
                            self.logger.info("✅ Enhanced transition probabilities calculated")
                            
            except Exception as e:
                self.logger.warning(f"Enhanced post-processing failed: {e}")
        
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
    
    def load_market_data(self, symbol: str, interval: str, 
                        start_date=None, end_date=None, data_type: str = "processed") -> Optional[pd.DataFrame]:
        """Load market data using enhanced data operations."""
        try:
            # Validate inputs
            if not symbol or not isinstance(symbol, str):
                tprint_error(f"❌ Invalid symbol: {symbol}")
                raise ValueError(f"Symbol must be a non-empty string, got: {symbol}")
            
            if not interval or not isinstance(interval, str):
                tprint_error(f"❌ Invalid interval: {interval}")
                raise ValueError(f"Interval must be a non-empty string, got: {interval}")
            
            if data_type not in ["processed", "raw", "enhanced"]:
                tprint_warning(f"⚠️ Unknown data_type '{data_type}', using 'processed'")
                data_type = "processed"
            
            tprint_info(f"📊 Loading market data: {symbol} {interval} ({data_type})")
            
            if self.data_operations:
                data = self.data_operations.load_market_data(symbol, interval, start_date, end_date, data_type)
                if data is not None:
                    tprint_success(f"✅ Loaded {len(data)} records for {symbol} {interval}")
                    self.logger.info(f"✅ Loaded {len(data)} records for {symbol} {interval}")
                else:
                    tprint_warning(f"⚠️ No data returned for {symbol} {interval}")
                    self.logger.warning(f"No data returned for {symbol} {interval}")
                return data
            else:
                tprint_warning("⚠️ Enhanced data operations not available")
                self.logger.warning("Enhanced data operations not available")
                return None
                
        except Exception as e:
            tprint_error(f"❌ Failed to load market data for {symbol} {interval}: {e}")
            self.logger.error(f"Failed to load market data for {symbol} {interval}: {e}")
            return None
    
    def get_data_quality_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get data quality report using enhanced data operations."""
        try:
            if self.data_operations:
                return self.data_operations.get_data_quality_report(data)
            else:
                return {'error': 'Enhanced data operations not available'}
        except Exception as e:
            return {'error': str(e)}
    
    def save_processed_data(self, data: pd.DataFrame, symbol: str, interval: str) -> bool:
        """Save processed data using enhanced data operations."""
        try:
            if self.data_operations:
                return self.data_operations.save_processed_data(data, symbol, interval, "processed")
            else:
                self.logger.warning("Enhanced data operations not available")
                return False
        except Exception as e:
            self.logger.error(f"Failed to save processed data: {e}")
            return False
    
    def get_enhanced_features(self, data: np.ndarray, window: int = 20) -> Dict[str, np.ndarray]:
        """Get enhanced features using matrix operations."""
        try:
            if self.matrix_operations:
                return self.matrix_operations.calculate_enhanced_features(data, window)
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Failed to calculate enhanced features: {e}")
            return {}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from all enhanced utilities."""
        metrics = {
            'enhanced_utilities_status': {
                'data_operations': self.data_operations is not None,
                'matrix_operations': self.matrix_operations is not None,
                'ml_common_integration': self.ml_common_integration is not None,
                'm1_integration': self.m1_integration is not None and self.m1_integration.get('success', False),
                'serialization': self.serializer is not None
            }
        }
        
        # Add matrix operations metrics if available
        if self.matrix_operations:
            matrix_metrics = self.matrix_operations.get_performance_metrics()
            metrics['matrix_operations_metrics'] = matrix_metrics
        
        return metrics