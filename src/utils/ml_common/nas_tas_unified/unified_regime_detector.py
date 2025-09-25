"""
Unified Regime Detection System

This module provides a unified regime detection system that combines the best aspects
of both TAS (Tree Architecture Search) and NAS (Neural Architecture Search) regime
detection with enhanced economic significance and trading viability evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from dataclasses import dataclass
from datetime import datetime

# Import tprint for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import unified configuration
from .unified_regime_config import (
    UnifiedRegimeConfig, RegimeDetectionMethod, OptimizationStrategy, EconomicEvaluationMode
)

# Import performance optimizer
from .performance_optimizer import (
    PerformanceOptimizer, optimize_performance, get_performance_optimizer
)

# Import TAS components
try:
    from src.training.steps.market_analysis.tas_regime.core.tas_regime_detector import (
        TASRegimeDetector, TASRegimeResult
    )
    from src.training.steps.market_analysis.tas_regime.core.tas_regime_config import (
        TASRegimeConfig, TASArchitectureType
    )
    TAS_AVAILABLE = True
except ImportError:
    TAS_AVAILABLE = False

# Import NAS components
try:
    from src.training.steps.market_analysis.nas_regime.core.perfect_nas_regime_detector import (
        PerfectNASRegimeDetector, PerfectNASResult
    )
    from src.training.steps.market_analysis.nas_regime.core.perfect_nas_config import (
        PerfectNASConfig, NeuralArchitectureType
    )
    NAS_AVAILABLE = True
except ImportError:
    NAS_AVAILABLE = False

# Import enhanced utilities
try:
    from src.utils.common_operations import (
        safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
        calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
        safe_apply_function, create_summary_statistics, safe_drop_columns,
        safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
        get_dataframe_info, safe_filter_dataframe, create_data_quality_report,
        optimize_dataframe_dtypes, safe_to_parquet, safe_read_parquet,
        align_dataframes, validate_dataframe_schema, guard_dataframe_nulls,
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        integrate_with_m1_optimizers, memory_checkpoint, gpu_context,
        optimize_memory, get_memory_usage, validate_file_path, get_file_size,
        check_disk_space, CommonUtilities, timed_operation
    )
    ENHANCED_UTILITIES_AVAILABLE = True
except ImportError:
    ENHANCED_UTILITIES_AVAILABLE = False

# Import position-aware trading analysis
try:
    from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.position_aware_trading import (
        PositionAwareTradingAnalyzer, PositionAwareConfig
    )
    POSITION_AWARE_AVAILABLE = True
except ImportError:
    POSITION_AWARE_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class UnifiedRegimeResult:
    """Unified result from regime detection combining TAS and NAS approaches."""
    success: bool
    regime_predictions: np.ndarray
    regime_probabilities: np.ndarray
    economic_significance_scores: np.ndarray
    trading_viability_scores: np.ndarray
    regime_stability_scores: np.ndarray
    transition_probabilities: np.ndarray
    micro_regimes: Optional[Dict[str, Any]] = None
    tas_results: Optional[TASRegimeResult] = None
    nas_results: Optional[PerfectNASResult] = None
    ensemble_weights: Optional[Dict[str, float]] = None
    uncertainty_estimates: Optional[np.ndarray] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None
    error_message: Optional[str] = None

class UnifiedRegimeDetector:
    """
    Unified Regime Detection System.
    
    Combines TAS and NAS regime detection approaches with adaptive selection
    and enhanced economic significance evaluation.
    """
    
    def __init__(self, config: UnifiedRegimeConfig):
        """Initialize Unified Regime Detector."""
        tprint_info("🚀 Initializing Unified Regime Detection System")
        tprint_info(f"📊 Detection Method: {config.detection_method.value}")
        tprint_info(f"📊 Optimization Strategy: {config.optimization_strategy.value}")
        tprint_info(f"📊 Economic Evaluation: {config.economic_evaluation.value}")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize performance optimizer
        self.performance_optimizer = get_performance_optimizer()
        tprint_success("⚡ Performance optimizer initialized")
        
        # Initialize enhanced utilities
        self._initialize_enhanced_utilities()
        
        # Initialize TAS detector if available and needed
        self.tas_detector = None
        if TAS_AVAILABLE and config.should_use_tas():
            tprint_info("🌲 Initializing TAS Regime Detector")
            self._initialize_tas_detector()
        
        # Initialize NAS detector if available and needed
        self.nas_detector = None
        if NAS_AVAILABLE and config.should_use_nas():
            tprint_info("🧠 Initializing NAS Regime Detector")
            self._initialize_nas_detector()
        
        # Initialize position-aware analyzer
        self._initialize_position_aware_analyzer()
        
        # Performance tracking
        self.performance_metrics = {
            'tas_accuracy': 0.5,
            'nas_accuracy': 0.5,
            'tas_efficiency': 0.5,
            'nas_efficiency': 0.5,
            'total_runs': 0,
            'last_update': datetime.now()
        }
        
        tprint_success("✅ Unified Regime Detection System initialized successfully")
        self.logger.info("✅ Unified Regime Detection System initialized successfully")
    
    def _initialize_enhanced_utilities(self):
        """Initialize enhanced utility components."""
        try:
            if ENHANCED_UTILITIES_AVAILABLE:
                self.common_utils = CommonUtilities()
                self.m1_integration = integrate_with_m1_optimizers()
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
                
                tprint_success("✅ Enhanced utilities initialized")
                self.logger.info("✅ Enhanced utilities initialized")
            else:
                tprint_warning("⚠️ Enhanced utilities not available")
                self.common_utils = None
                self.m1_integration = None
                
        except Exception as e:
            tprint_warning(f"⚠️ Enhanced utilities initialization failed: {e}")
            self.logger.warning(f"Enhanced utilities initialization failed: {e}")
            self.common_utils = None
            self.m1_integration = None
    
    def _initialize_tas_detector(self):
        """Initialize TAS regime detector."""
        try:
            # Convert unified config to TAS config
            tas_config = self._create_tas_config()
            self.tas_detector = TASRegimeDetector(tas_config)
            
            tprint_success("✅ TAS Regime Detector initialized")
            self.logger.info("✅ TAS Regime Detector initialized")
            
        except Exception as e:
            tprint_warning(f"⚠️ TAS detector initialization failed: {e}")
            self.logger.warning(f"TAS detector initialization failed: {e}")
            self.tas_detector = None
    
    def _initialize_nas_detector(self):
        """Initialize NAS regime detector."""
        try:
            # Convert unified config to NAS config
            nas_config = self._create_nas_config()
            self.nas_detector = PerfectNASRegimeDetector(nas_config)
            
            tprint_success("✅ NAS Regime Detector initialized")
            self.logger.info("✅ NAS Regime Detector initialized")
            
        except Exception as e:
            tprint_warning(f"⚠️ NAS detector initialization failed: {e}")
            self.logger.warning(f"NAS detector initialization failed: {e}")
            self.nas_detector = None
    
    def _initialize_position_aware_analyzer(self):
        """Initialize position-aware trading analyzer."""
        if not POSITION_AWARE_AVAILABLE:
            self.position_analyzer = None
            return
        
        try:
            economic_config = self.config.get_economic_config()
            params = economic_config['parameters']
            
            position_config = PositionAwareConfig(
                minimum_profit_threshold=params.get('minimum_profit_threshold', 0.001),
                transaction_cost=params.get('transaction_cost', 0.001),
                position_holding_periods=params.get('position_holding_periods', [1, 5, 10, 20]),
                risk_free_rate=params.get('risk_free_rate', 0.02),
                win_rate_thresholds=params.get('win_rate_thresholds', {
                    'excellent': 0.7,
                    'good': 0.6,
                    'acceptable': 0.5,
                    'poor': 0.4
                })
            )
            self.position_analyzer = PositionAwareTradingAnalyzer(position_config)
            
            tprint_success("✅ Position-aware trading analyzer initialized")
            self.logger.info("✅ Position-aware trading analyzer initialized")
            
        except Exception as e:
            tprint_warning(f"⚠️ Position-aware analyzer initialization failed: {e}")
            self.logger.warning(f"Position-aware analyzer initialization failed: {e}")
            self.position_analyzer = None
    
    def _create_tas_config(self) -> TASRegimeConfig:
        """Create TAS configuration from unified config."""
        if not TAS_AVAILABLE:
            raise ImportError("TAS components not available")
        
        tas_params = self.config.get_tas_config()
        
        # Map unified config to TAS config
        tas_config = TASRegimeConfig(
            n_regimes=self.config.n_regimes,
            primary_timeframe=self.config.primary_timeframe,
            min_regime_samples=self.config.min_regime_samples,
            max_regime_samples=self.config.max_regime_samples,
            primary_architecture=TASArchitectureType.HYBRID_TREE,
            tree_depth=tas_params.get('tree_depth', 6),
            n_estimators=tas_params.get('n_estimators', 1000),
            min_samples_split=tas_params.get('min_samples_split', 10),
            min_samples_leaf=tas_params.get('min_samples_leaf', 5),
            max_features=tas_params.get('max_features', 'sqrt'),
            enable_statistical_methods=tas_params.get('enable_statistical_methods', True),
            enable_bootstrap_analysis=tas_params.get('enable_bootstrap_analysis', True),
            bootstrap_iterations=tas_params.get('bootstrap_iterations', 1000),
            enable_clvsa_enhancement=tas_params.get('enable_clvsa_enhancement', True),
            enable_meta_learning=tas_params.get('enable_meta_learning', True),
            adaptation_rate=tas_params.get('adaptation_rate', 0.1),
            memory_size=tas_params.get('memory_size', 1000),
            enable_economic_evaluation=True,
            economic_significance_threshold=self.config.economic_significance_threshold,
            trading_viability_threshold=self.config.trading_viability_threshold,
            enable_hardware_optimization=self.config.enable_hardware_optimization,
            enable_matrix_optimization=self.config.enable_matrix_optimization,
            enable_memory_optimization=self.config.enable_memory_optimization,
            optimization_level=tas_params.get('optimization_level', 'maximum'),
            max_execution_time=self.config.max_execution_time,
            enable_early_stopping=self.config.enable_early_stopping
        )
        
        return tas_config
    
    def _create_nas_config(self) -> PerfectNASConfig:
        """Create NAS configuration from unified config."""
        if not NAS_AVAILABLE:
            raise ImportError("NAS components not available")
        
        nas_params = self.config.get_nas_config()
        
        # Map unified config to NAS config
        nas_config = PerfectNASConfig(
            n_regimes=self.config.n_regimes,
            primary_timeframe=self.config.primary_timeframe,
            min_regime_duration=self.config.min_regime_samples,
            max_regime_duration=self.config.max_regime_samples,
            primary_architecture=NeuralArchitectureType.HYBRID,
            enable_neural_odes=nas_params.get('enable_neural_odes', True),
            enable_vision_transformers=nas_params.get('enable_vision_transformers', True),
            enable_state_space_models=nas_params.get('enable_state_space_models', True),
            enable_meta_learning=nas_params.get('enable_meta_learning', True),
            search_strategy=nas_params.get('search_strategy', 'evolutionary'),
            population_size=nas_params.get('population_size', 50),
            generations=nas_params.get('generations', 100),
            mutation_rate=nas_params.get('mutation_rate', 0.1),
            crossover_rate=nas_params.get('crossover_rate', 0.8),
            elite_size=nas_params.get('elite_size', 5),
            enable_uncertainty_quantification=nas_params.get('enable_uncertainty_quantification', True),
            enable_multi_scale_analysis=nas_params.get('enable_multi_scale_analysis', True),
            accuracy_threshold=self.config.target_accuracy,
            economic_significance_threshold=self.config.economic_significance_threshold,
            trading_viability_threshold=self.config.trading_viability_threshold,
            regime_stability_threshold=self.config.regime_stability_threshold,
            transition_accuracy_threshold=self.config.transition_accuracy_threshold,
            max_execution_time=self.config.max_execution_time,
            enable_early_stopping=self.config.enable_early_stopping,
            early_stopping_patience=self.config.early_stopping_patience,
            enable_checkpointing=self.config.enable_checkpointing,
            checkpoint_interval=self.config.checkpoint_interval,
            enable_gpu_acceleration=self.config.enable_gpu_acceleration,
            enable_memory_optimization=self.config.enable_memory_optimization
        )
        
        return nas_config
    
    @optimize_performance(enable_cache=True, enable_gpu=True, max_memory_gb=8.0)
    @timed_operation
    def detect_regimes(self,
                      market_data: Union[pd.DataFrame, np.ndarray],
                      timestamps: Optional[np.ndarray] = None,
                      enable_adaptive_selection: bool = True) -> UnifiedRegimeResult:
        """
        Detect market regimes using unified TAS-NAS system.
        
        Args:
            market_data: Market data (OHLCV)
            timestamps: Optional timestamps
            enable_adaptive_selection: Whether to use adaptive method selection
            
        Returns:
            UnifiedRegimeResult with regime detection results
        """
        start_time = time.time()
        
        try:
            tprint_info("🚀 Starting Unified Regime Detection")
            
            # Validate inputs
            if market_data is None or (hasattr(market_data, '__len__') and len(market_data) == 0):
                raise ValueError("Market data cannot be None or empty")
            
            # Use memory checkpoint for large datasets
            if ENHANCED_UTILITIES_AVAILABLE and self.memory_optimizer:
                with memory_checkpoint("unified_regime_detection"):
                    result = self._perform_unified_detection(
                        market_data, timestamps, enable_adaptive_selection
                    )
            else:
                result = self._perform_unified_detection(
                    market_data, timestamps, enable_adaptive_selection
                )
            
            # Update performance metrics
            self._update_performance_metrics(result)
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            tprint_success(f"✅ Unified regime detection completed in {execution_time:.2f}s")
            self.logger.info(f"✅ Unified regime detection completed in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ Unified regime detection failed: {e}")
            self.logger.error(f"❌ Unified regime detection failed: {e}")
            
            return UnifiedRegimeResult(
                success=False,
                regime_predictions=np.array([]),
                regime_probabilities=np.array([]),
                economic_significance_scores=np.array([]),
                trading_viability_scores=np.array([]),
                regime_stability_scores=np.array([]),
                transition_probabilities=np.array([]),
                execution_time=execution_time,
                error_message=str(e),
                metadata={'error': str(e)}
            )
    
    def _perform_unified_detection(self,
                                 market_data: Union[pd.DataFrame, np.ndarray],
                                 timestamps: Optional[np.ndarray],
                                 enable_adaptive_selection: bool) -> UnifiedRegimeResult:
        """Perform the actual unified regime detection."""
        
        # Determine which methods to use
        use_tas = self.config.should_use_tas(self.performance_metrics if enable_adaptive_selection else None)
        use_nas = self.config.should_use_nas(self.performance_metrics if enable_adaptive_selection else None)
        
        tprint_info(f"📊 Using TAS: {use_tas}, Using NAS: {use_nas}")
        
        tas_results = None
        nas_results = None
        
        # Run TAS detection if enabled
        if use_tas and self.tas_detector:
            try:
                tprint_info("🌲 Running TAS regime detection")
                tas_results = self.tas_detector.detect_regimes(
                    market_data, timestamps,
                    optimize_performance=True,
                    enable_clvsa_enhancement=True
                )
                tprint_success("✅ TAS regime detection completed")
            except Exception as e:
                tprint_error(f"❌ TAS detection failed: {e}")
                self.logger.error(f"TAS detection failed: {e}")
                tas_results = None
        
        # Run NAS detection if enabled
        if use_nas and self.nas_detector:
            try:
                tprint_info("🧠 Running NAS regime detection")
                nas_results = self.nas_detector.detect_regimes(
                    market_data, timestamps,
                    optimize_architecture=True,
                    enable_meta_learning=True
                )
                tprint_success("✅ NAS regime detection completed")
            except Exception as e:
                tprint_error(f"❌ NAS detection failed: {e}")
                self.logger.error(f"NAS detection failed: {e}")
                nas_results = None
        
        # Combine results
        if tas_results is not None and nas_results is not None:
            # Both methods succeeded - ensemble the results
            return self._ensemble_results(tas_results, nas_results)
        elif tas_results is not None:
            # Only TAS succeeded
            return self._convert_tas_result(tas_results)
        elif nas_results is not None:
            # Only NAS succeeded
            return self._convert_nas_result(nas_results)
        else:
            # Both methods failed
            raise RuntimeError("Both TAS and NAS regime detection failed")
    
    def _ensemble_results(self, tas_results: TASRegimeResult, nas_results: PerfectNASResult) -> UnifiedRegimeResult:
        """Ensemble TAS and NAS results."""
        try:
            tprint_info("🔄 Ensembling TAS and NAS results")
            
            # Calculate ensemble weights based on performance
            tas_weight = 0.6  # Default weight for TAS
            nas_weight = 0.4  # Default weight for NAS
            
            if self.config.optimization_strategy == OptimizationStrategy.ACCURACY_FIRST:
                # Weight by accuracy if available
                if hasattr(tas_results, 'metadata') and 'accuracy' in tas_results.metadata:
                    tas_weight = tas_results.metadata['accuracy']
                if hasattr(nas_results, 'metadata') and 'accuracy' in nas_results.metadata:
                    nas_weight = nas_results.metadata['accuracy']
                
                # Normalize weights
                total_weight = tas_weight + nas_weight
                if total_weight > 0:
                    tas_weight /= total_weight
                    nas_weight /= total_weight
                else:
                    tas_weight = 0.6
                    nas_weight = 0.4
            
            ensemble_weights = {'tas': tas_weight, 'nas': nas_weight}
            
            # Ensemble predictions (weighted average)
            if len(tas_results.regime_predictions) == len(nas_results.regime_predictions):
                ensemble_predictions = (
                    tas_weight * tas_results.regime_predictions + 
                    nas_weight * nas_results.regime_predictions
                ).astype(int)
            else:
                # Use the longer prediction array and pad the shorter one
                max_len = max(len(tas_results.regime_predictions), len(nas_results.regime_predictions))
                tas_pred = np.pad(tas_results.regime_predictions, (0, max_len - len(tas_results.regime_predictions)), 'constant')
                nas_pred = np.pad(nas_results.regime_predictions, (0, max_len - len(nas_results.regime_predictions)), 'constant')
                ensemble_predictions = (tas_weight * tas_pred + nas_weight * nas_pred).astype(int)
            
            # Ensemble probabilities (weighted average)
            if (tas_results.regime_probabilities.shape == nas_results.regime_probabilities.shape):
                ensemble_probabilities = (
                    tas_weight * tas_results.regime_probabilities + 
                    nas_weight * nas_results.regime_probabilities
                )
            else:
                # Use the larger probability array and pad the smaller one
                max_shape = (max(tas_results.regime_probabilities.shape[0], nas_results.regime_probabilities.shape[0]),
                           max(tas_results.regime_probabilities.shape[1], nas_results.regime_probabilities.shape[1]))
                tas_prob = np.pad(tas_results.regime_probabilities, 
                                ((0, max_shape[0] - tas_results.regime_probabilities.shape[0]),
                                 (0, max_shape[1] - tas_results.regime_probabilities.shape[1])), 'constant')
                nas_prob = np.pad(nas_results.regime_probabilities,
                                ((0, max_shape[0] - nas_results.regime_probabilities.shape[0]),
                                 (0, max_shape[1] - nas_results.regime_probabilities.shape[1])), 'constant')
                ensemble_probabilities = tas_weight * tas_prob + nas_weight * nas_prob
            
            # Ensemble other metrics
            ensemble_economic = (
                tas_weight * tas_results.economic_significance_scores + 
                nas_weight * nas_results.economic_significance_scores
            )
            ensemble_trading = (
                tas_weight * tas_results.trading_viability_scores + 
                nas_weight * nas_results.trading_viability_scores
            )
            ensemble_stability = (
                tas_weight * tas_results.regime_stability_scores + 
                nas_weight * nas_results.regime_stability_scores
            )
            
            # Calculate ensemble transition probabilities
            ensemble_transitions = self._ensemble_transition_probabilities(
                tas_results.transition_probabilities, 
                nas_results.transition_probabilities,
                tas_weight, nas_weight
            )
            
            result = UnifiedRegimeResult(
                success=True,
                regime_predictions=ensemble_predictions,
                regime_probabilities=ensemble_probabilities,
                economic_significance_scores=ensemble_economic,
                trading_viability_scores=ensemble_trading,
                regime_stability_scores=ensemble_stability,
                transition_probabilities=ensemble_transitions,
                tas_results=tas_results,
                nas_results=nas_results,
                ensemble_weights=ensemble_weights,
                metadata={
                    'method': 'ensemble',
                    'tas_weight': tas_weight,
                    'nas_weight': nas_weight,
                    'tas_success': tas_results.success,
                    'nas_success': nas_results.success,
                    'system': 'Unified TAS-NAS Regime Detection'
                }
            )
            
            tprint_success("✅ Results ensemble completed successfully")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Results ensemble failed: {e}")
            self.logger.error(f"Results ensemble failed: {e}")
            raise
    
    def _ensemble_transition_probabilities(self, tas_transitions: np.ndarray, 
                                         nas_transitions: np.ndarray,
                                         tas_weight: float, nas_weight: float) -> np.ndarray:
        """Ensemble transition probability matrices."""
        try:
            if tas_transitions.shape == nas_transitions.shape:
                return tas_weight * tas_transitions + nas_weight * nas_transitions
            else:
                # Use the larger matrix and pad the smaller one
                max_shape = max(tas_transitions.shape, nas_transitions.shape)
                tas_padded = np.pad(tas_transitions, 
                                  ((0, max_shape[0] - tas_transitions.shape[0]),
                                   (0, max_shape[1] - tas_transitions.shape[1])), 'constant')
                nas_padded = np.pad(nas_transitions,
                                  ((0, max_shape[0] - nas_transitions.shape[0]),
                                   (0, max_shape[1] - nas_transitions.shape[1])), 'constant')
                return tas_weight * tas_padded + nas_weight * nas_padded
        except Exception as e:
            tprint_warning(f"⚠️ Transition probability ensemble failed: {e}")
            return tas_transitions if tas_transitions is not None else nas_transitions
    
    def _convert_tas_result(self, tas_results: TASRegimeResult) -> UnifiedRegimeResult:
        """Convert TAS result to unified result format."""
        return UnifiedRegimeResult(
            success=tas_results.success,
            regime_predictions=tas_results.regime_predictions,
            regime_probabilities=tas_results.regime_probabilities,
            economic_significance_scores=tas_results.economic_significance_scores,
            trading_viability_scores=tas_results.trading_viability_scores,
            regime_stability_scores=tas_results.regime_stability_scores,
            transition_probabilities=tas_results.transition_probabilities,
            micro_regimes=tas_results.micro_regimes,
            tas_results=tas_results,
            uncertainty_estimates=tas_results.uncertainty_estimates,
            metadata={
                'method': 'tas_only',
                'system': 'TAS Regime Detection',
                'tas_success': tas_results.success
            }
        )
    
    def _convert_nas_result(self, nas_results: PerfectNASResult) -> UnifiedRegimeResult:
        """Convert NAS result to unified result format."""
        return UnifiedRegimeResult(
            success=nas_results.success,
            regime_predictions=nas_results.regime_predictions,
            regime_probabilities=nas_results.regime_probabilities,
            economic_significance_scores=nas_results.economic_significance_scores,
            trading_viability_scores=nas_results.trading_viability_scores,
            regime_stability_scores=nas_results.regime_stability_scores,
            transition_probabilities=nas_results.transition_probabilities,
            micro_regimes=nas_results.micro_regimes,
            nas_results=nas_results,
            uncertainty_estimates=nas_results.uncertainty_estimates,
            metadata={
                'method': 'nas_only',
                'system': 'NAS Regime Detection',
                'nas_success': nas_results.success
            }
        )
    
    def _update_performance_metrics(self, result: UnifiedRegimeResult):
        """Update performance metrics based on results."""
        try:
            self.performance_metrics['total_runs'] += 1
            self.performance_metrics['last_update'] = datetime.now()
            
            # Update accuracy metrics if available
            if result.tas_results is not None:
                # Calculate TAS accuracy (simplified)
                tas_accuracy = np.mean(result.tas_results.regime_stability_scores)
                self.performance_metrics['tas_accuracy'] = tas_accuracy
            
            if result.nas_results is not None:
                # Calculate NAS accuracy (simplified)
                nas_accuracy = np.mean(result.nas_results.regime_stability_scores)
                self.performance_metrics['nas_accuracy'] = nas_accuracy
            
            # Update efficiency metrics (simplified)
            if result.tas_results is not None:
                self.performance_metrics['tas_efficiency'] = 1.0 / (result.tas_results.execution_time + 1e-8)
            
            if result.nas_results is not None:
                self.performance_metrics['nas_efficiency'] = 1.0 / (result.nas_results.execution_time + 1e-8)
            
            tprint_debug(f"📊 Performance metrics updated: TAS acc={self.performance_metrics['tas_accuracy']:.3f}, "
                        f"NAS acc={self.performance_metrics['nas_accuracy']:.3f}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Performance metrics update failed: {e}")
            self.logger.warning(f"Performance metrics update failed: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = self.performance_metrics.copy()
        
        # Add performance optimizer stats
        if self.performance_optimizer:
            optimizer_stats = self.performance_optimizer.get_performance_stats()
            metrics['optimizer_stats'] = optimizer_stats
        
        return metrics
    
    def reset_performance_metrics(self):
        """Reset performance metrics."""
        self.performance_metrics = {
            'tas_accuracy': 0.5,
            'nas_accuracy': 0.5,
            'tas_efficiency': 0.5,
            'nas_efficiency': 0.5,
            'total_runs': 0,
            'last_update': datetime.now()
        }
        tprint_info("📊 Performance metrics reset")
    
    def save_results(self, result: UnifiedRegimeResult, filepath: str):
        """Save unified results to file."""
        try:
            import pickle
            from pathlib import Path
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(result, f)
            
            tprint_success(f"✅ Unified results saved to {filepath}")
            self.logger.info(f"✅ Unified results saved to {filepath}")
            
        except Exception as e:
            tprint_error(f"❌ Failed to save unified results: {e}")
            self.logger.error(f"❌ Failed to save unified results: {e}")
    
    def load_results(self, filepath: str) -> UnifiedRegimeResult:
        """Load unified results from file."""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                result = pickle.load(f)
            
            tprint_success(f"✅ Unified results loaded from {filepath}")
            self.logger.info(f"✅ Unified results loaded from {filepath}")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Failed to load unified results: {e}")
            self.logger.error(f"❌ Failed to load unified results: {e}")
            raise