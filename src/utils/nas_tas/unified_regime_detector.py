"""
Unified Regime Detector

Provides a unified interface for both TAS and NAS regime detection systems,
eliminating code duplication and providing consistent error handling and logging.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager

# Import tprint for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import unified components
from .unified_regime_config import UnifiedRegimeConfig, RegimeSystemType, ArchitectureType
from .unified_result import UnifiedRegimeResult

# Import enhanced utility tools
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
        check_disk_space, CommonUtilities
    )
    COMMON_UTILITIES_AVAILABLE = True
except ImportError:
    COMMON_UTILITIES_AVAILABLE = False

from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, safe_correlation,
    safe_covariance, safe_mean, safe_std, safe_percentile,
    validate_correlation_matrix, safe_matrix_inverse, math_safe,
    MathValidation, MathValidationError
)

from src.utils.matrix_operations.unified_operations import (
    UnifiedMatrixOperations, get_unified_matrix_operations,
    safe_matrix_multiply, safe_correlation_matrix, safe_matrix_inverse
)

from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

# Import hardware optimization tools
try:
    from src.utils.hardware.unified_hardware_manager import (
        UnifiedHardwareManager, HardwareConfig, WorkloadType, OptimizationLevel
    )
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False

# Import shared utilities from hybrid regime system
try:
    from src.utils.nas_tas.shared_utils.search_strategies import SearchStrategyManager, SearchStrategyConfig
    from src.utils.nas_tas.shared_utils.analysis_components import SharedClusteringUtilities, AnalysisComponentConfig
    from src.utils.nas_tas.shared_utils.position_aware_trading import PositionAwareTradingAnalyzer, PositionAwareConfig
    SHARED_UTILITIES_AVAILABLE = True
    POSITION_AWARE_AVAILABLE = True
except ImportError:
    SHARED_UTILITIES_AVAILABLE = False
    POSITION_AWARE_AVAILABLE = False

# Import TAS-specific components
try:
    from src.training.steps.market_analysis.tas_regime.core.tas_regime_detector import TASRegimeDetector
    from src.training.steps.market_analysis.tas_regime.core.tas_regime_config import TASRegimeConfig
    TAS_AVAILABLE = True
except ImportError:
    TAS_AVAILABLE = False

# Import NAS-specific components
try:
    from src.training.steps.market_analysis.nas_regime.core.perfect_nas_regime_detector import PerfectNASRegimeDetector
    from src.training.steps.market_analysis.nas_regime.core.perfect_nas_config import PerfectNASConfig
    NAS_AVAILABLE = True
except ImportError:
    NAS_AVAILABLE = False

logger = logging.getLogger(__name__)

class UnifiedRegimeDetector:
    """
    Unified Regime Detector that combines TAS and NAS systems.
    
    Provides a single interface for regime detection that can use either
    TAS (Tree-based Advanced Statistics) or NAS (Neural Architecture Search)
    systems, or both in hybrid mode.
    """
    
    def __init__(self, config: UnifiedRegimeConfig):
        """Initialize Unified Regime Detector."""
        tprint_info("🚀 Initializing Unified Regime Detector")
        tprint_debug(f"Configuration: {config}")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize enhanced utility tools
        tprint_info("🔧 Initializing enhanced utility tools...")
        self._initialize_enhanced_utility_tools()
        
        # Initialize system-specific detectors
        tprint_info("🛠️ Initializing system-specific detectors...")
        self._initialize_system_detectors()
        
        # Initialize shared utilities
        tprint_info("🔗 Initializing shared utilities...")
        self._initialize_shared_utilities()
        
        tprint_success("✅ Unified Regime Detector initialized successfully")
        self.logger.info("✅ Unified Regime Detector initialized successfully")
        self.logger.info(f"🛠️ System type: {config.system_type.value}")
        self.logger.info(f"🛠️ Architecture: {config.primary_architecture.value}")
        self.logger.info(f"🛠️ TAS available: {TAS_AVAILABLE}")
        self.logger.info(f"🛠️ NAS available: {NAS_AVAILABLE}")
    
    def _initialize_enhanced_utility_tools(self):
        """Initialize enhanced utility tools for unified regime detection."""
        tprint_debug("🔧 Starting enhanced utility tools initialization...")
        try:
            # Initialize common utilities
            if COMMON_UTILITIES_AVAILABLE:
                tprint_debug("📦 Creating common utilities...")
                self.common_utils = CommonUtilities()
                tprint_success("✅ Common utilities initialized")
                self.logger.info("✅ Common utilities initialized")
            else:
                tprint_warning("⚠️ Common utilities not available")
                self.common_utils = None
            
            # Initialize math validation
            tprint_debug("🧮 Creating math validator...")
            self.math_validator = MathValidation()
            tprint_success("✅ Math validation initialized")
            self.logger.info("✅ Math validation initialized")
            
            # Initialize matrix operations
            tprint_debug("🔢 Creating enhanced matrix operations...")
            self.enhanced_matrix_ops = get_unified_matrix_operations(
                enable_gpu=True,
                enable_memory_optimization=True,
                enable_parallel=True
            )
            tprint_success("✅ Enhanced matrix operations initialized")
            self.logger.info("✅ Enhanced matrix operations initialized")
            
            # Initialize serialization
            tprint_debug("💾 Creating enhanced serializer...")
            self.enhanced_serializer = UniversalSerializer()
            tprint_success("✅ Enhanced serialization initialized")
            self.logger.info("✅ Enhanced serialization initialized")
            
            # Initialize hardware optimization
            self._initialize_hardware_optimization()
            
        except Exception as e:
            tprint_error(f"❌ Enhanced utility tools initialization failed: {e}")
            self.logger.error(f"❌ Enhanced utility tools initialization failed: {e}")
            # Set fallback values
            self.common_utils = None
            self.math_validator = None
            self.enhanced_matrix_ops = None
            self.enhanced_serializer = None
            self.hardware_manager = None
    
    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization components."""
        tprint_debug("💻 Starting hardware optimization initialization...")
        if not HARDWARE_AVAILABLE:
            tprint_warning("⚠️ Hardware optimization not available")
            self.hardware_manager = None
            return
        
        try:
            tprint_debug("⚙️ Creating hardware configuration...")
            hardware_config = HardwareConfig(
                cpu_optimization_level=OptimizationLevel.AGGRESSIVE,
                gpu_optimization_level=OptimizationLevel.AGGRESSIVE,
                memory_optimization_level=OptimizationLevel.AGGRESSIVE,
                enable_adaptive_optimization=True,
                enable_learning=True,
                auto_tuning_enabled=True
            )
            tprint_debug("🏗️ Creating unified hardware manager...")
            self.hardware_manager = UnifiedHardwareManager(hardware_config)
            tprint_success("✅ Hardware optimization initialized")
            self.logger.info("✅ Hardware optimization initialized")
        except Exception as e:
            tprint_warning(f"⚠️ Hardware optimization initialization failed: {e}")
            self.logger.warning(f"Hardware optimization initialization failed: {e}")
            self.hardware_manager = None
    
    def _initialize_system_detectors(self):
        """Initialize system-specific detectors based on configuration."""
        tprint_debug("🛠️ Starting system-specific detector initialization...")
        
        # Initialize TAS detector if needed
        if (self.config.system_type in [RegimeSystemType.TAS, RegimeSystemType.HYBRID, RegimeSystemType.UNIFIED] 
            and TAS_AVAILABLE):
            try:
                tprint_debug("🌲 Initializing TAS detector...")
                tas_config = self._create_tas_config()
                self.tas_detector = TASRegimeDetector(tas_config)
                tprint_success("✅ TAS detector initialized")
                self.logger.info("✅ TAS detector initialized")
            except Exception as e:
                tprint_warning(f"⚠️ TAS detector initialization failed: {e}")
                self.logger.warning(f"TAS detector initialization failed: {e}")
                self.tas_detector = None
        else:
            self.tas_detector = None
            if not TAS_AVAILABLE:
                tprint_warning("⚠️ TAS system not available")
        
        # Initialize NAS detector if needed
        if (self.config.system_type in [RegimeSystemType.NAS, RegimeSystemType.HYBRID, RegimeSystemType.UNIFIED] 
            and NAS_AVAILABLE):
            try:
                tprint_debug("🧠 Initializing NAS detector...")
                nas_config = self._create_nas_config()
                self.nas_detector = PerfectNASRegimeDetector(nas_config)
                tprint_success("✅ NAS detector initialized")
                self.logger.info("✅ NAS detector initialized")
            except Exception as e:
                tprint_warning(f"⚠️ NAS detector initialization failed: {e}")
                self.logger.warning(f"NAS detector initialization failed: {e}")
                self.nas_detector = None
        else:
            self.nas_detector = None
            if not NAS_AVAILABLE:
                tprint_warning("⚠️ NAS system not available")
    
    def _create_tas_config(self) -> TASRegimeConfig:
        """Create TAS configuration from unified config."""
        if not TAS_AVAILABLE:
            raise ImportError("TAS system not available")
        
        # Convert unified config to TAS config
        tas_config = TASRegimeConfig()
        tas_config.n_regimes = self.config.n_regimes
        tas_config.primary_timeframe = self.config.primary_timeframe
        tas_config.tree_depth = self.config.tree_depth
        tas_config.n_estimators = self.config.n_estimators
        tas_config.min_samples_split = self.config.min_samples_split
        tas_config.min_samples_leaf = self.config.min_samples_leaf
        tas_config.max_features = self.config.max_features
        tas_config.enable_statistical_methods = self.config.enable_statistical_methods
        tas_config.enable_bootstrap_analysis = self.config.enable_bootstrap_analysis
        tas_config.bootstrap_iterations = self.config.bootstrap_iterations
        tas_config.enable_clvsa_enhancement = self.config.enable_clvsa_enhancement
        tas_config.enable_regime_adaptation = self.config.enable_regime_adaptation
        tas_config.enable_uncertainty_quantification = self.config.enable_uncertainty_quantification
        tas_config.enable_multi_scale_analysis = self.config.enable_multi_scale_analysis
        tas_config.enable_hardware_optimization = self.config.enable_hardware_optimization
        tas_config.enable_matrix_optimization = self.config.enable_matrix_optimization
        tas_config.enable_memory_optimization = self.config.enable_memory_optimization
        tas_config.enable_economic_evaluation = self.config.enable_economic_evaluation
        tas_config.economic_significance_threshold = self.config.economic_significance_threshold
        tas_config.trading_viability_threshold = self.config.trading_viability_threshold
        tas_config.enable_meta_learning = self.config.enable_meta_learning
        tas_config.adaptation_rate = self.config.meta_learning_config.adaptation_steps / 100.0
        tas_config.memory_size = self.config.meta_learning_config.memory_size
        
        return tas_config
    
    def _create_nas_config(self) -> PerfectNASConfig:
        """Create NAS configuration from unified config."""
        if not NAS_AVAILABLE:
            raise ImportError("NAS system not available")
        
        # Convert unified config to NAS config
        nas_config = PerfectNASConfig()
        nas_config.n_regimes = self.config.n_regimes
        nas_config.primary_timeframe = self.config.primary_timeframe
        nas_config.enable_neural_odes = self.config.enable_neural_odes
        nas_config.enable_vision_transformers = self.config.enable_vision_transformers
        nas_config.enable_state_space_models = self.config.enable_state_space_models
        nas_config.enable_meta_learning = self.config.enable_meta_learning
        nas_config.population_size = self.config.population_size
        nas_config.generations = self.config.generations
        nas_config.mutation_rate = self.config.mutation_rate
        nas_config.crossover_rate = self.config.crossover_rate
        nas_config.elite_size = self.config.elite_size
        nas_config.accuracy_threshold = self.config.accuracy_threshold
        nas_config.economic_significance_threshold = self.config.economic_significance_threshold
        nas_config.trading_viability_threshold = self.config.trading_viability_threshold
        nas_config.regime_stability_threshold = self.config.regime_stability_threshold
        nas_config.transition_accuracy_threshold = self.config.transition_accuracy_threshold
        nas_config.max_execution_time = self.config.max_execution_time
        nas_config.enable_early_stopping = self.config.enable_early_stopping
        nas_config.early_stopping_patience = self.config.early_stopping_patience
        nas_config.enable_checkpointing = self.config.enable_checkpointing
        nas_config.checkpoint_interval = self.config.checkpoint_interval
        
        return nas_config
    
    def _initialize_shared_utilities(self):
        """Initialize shared utilities from hybrid regime system."""
        if not SHARED_UTILITIES_AVAILABLE:
            self.logger.warning("⚠️ Shared utilities not available")
            self.search_strategy_manager = None
            self.shared_clustering = None
            self.position_analyzer = None
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
            
            # Initialize position-aware analyzer
            if POSITION_AWARE_AVAILABLE:
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
            else:
                self.position_analyzer = None
            
            self.logger.info("✅ Shared utilities initialized")
        except Exception as e:
            self.logger.warning(f"Shared utilities initialization failed: {e}")
            self.search_strategy_manager = None
            self.shared_clustering = None
            self.position_analyzer = None
    
    def detect_regimes(self,
                      market_data: Union[pd.DataFrame, np.ndarray],
                      timestamps: Optional[np.ndarray] = None,
                      optimize_performance: bool = True,
                      enable_hybrid_mode: bool = True) -> UnifiedRegimeResult:
        """
        Detect market regimes using unified system.
        
        Args:
            market_data: Market data (OHLCV)
            timestamps: Optional timestamps
            optimize_performance: Whether to use hardware optimization
            enable_hybrid_mode: Whether to use hybrid TAS-NAS mode
            
        Returns:
            UnifiedRegimeResult with regime detection results
        """
        start_time = time.time()
        
        try:
            self.logger.info("🚀 Starting unified regime detection")
            tprint_info("🚀 Starting unified regime detection")
            
            # Hardware optimization context
            with self._hardware_optimization_context():
                # Prepare and enhance data
                processed_data, processed_timestamps = self._prepare_and_enhance_data(
                    market_data, timestamps
                )
                
                # Determine detection strategy
                if self.config.system_type == RegimeSystemType.TAS and self.tas_detector:
                    result = self._detect_with_tas(processed_data, processed_timestamps, optimize_performance)
                elif self.config.system_type == RegimeSystemType.NAS and self.nas_detector:
                    result = self._detect_with_nas(processed_data, processed_timestamps, optimize_performance)
                elif self.config.system_type in [RegimeSystemType.HYBRID, RegimeSystemType.UNIFIED] and enable_hybrid_mode:
                    result = self._detect_with_hybrid(processed_data, processed_timestamps, optimize_performance)
                else:
                    # Fallback to available system
                    if self.tas_detector:
                        result = self._detect_with_tas(processed_data, processed_timestamps, optimize_performance)
                    elif self.nas_detector:
                        result = self._detect_with_nas(processed_data, processed_timestamps, optimize_performance)
                    else:
                        raise RuntimeError("No regime detection system available")
                
                # Post-process results
                result = self._post_process_results(result, processed_data, processed_timestamps)
                
            # Create unified result
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            result.system_type = self.config.system_type.value
            result.architecture_used = self.config.primary_architecture.value
            
            self.logger.info(f"✅ Unified regime detection completed in {execution_time:.2f}s")
            self._log_unified_results_summary(result)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ Unified regime detection failed: {e}")
            tprint_debug(f"Error context: {locals()}")
            tprint_warning(f"Execution time before failure: {execution_time:.2f}s")
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
    
    def _detect_with_tas(self, data: np.ndarray, timestamps: np.ndarray, optimize_performance: bool) -> UnifiedRegimeResult:
        """Detect regimes using TAS system."""
        try:
            tprint_info("🌲 Using TAS regime detection")
            tas_result = self.tas_detector.detect_regimes(
                data, timestamps, optimize_performance, enable_clvsa_enhancement=True
            )
            
            # Convert TAS result to unified result
            return UnifiedRegimeResult(
                success=tas_result.success,
                regime_predictions=tas_result.regime_predictions,
                regime_probabilities=tas_result.regime_probabilities,
                economic_significance_scores=tas_result.economic_significance_scores,
                trading_viability_scores=tas_result.trading_viability_scores,
                regime_stability_scores=tas_result.regime_stability_scores,
                transition_probabilities=tas_result.transition_probabilities,
                micro_regimes=tas_result.micro_regimes,
                performance_metrics=tas_result.tree_performance_metrics,
                uncertainty_estimates=tas_result.uncertainty_estimates,
                enhanced_features=tas_result.clvsa_enhanced_features,
                metadata=tas_result.metadata,
                error_message=tas_result.error_message
            )
            
        except Exception as e:
            tprint_error(f"❌ TAS regime detection failed: {e}")
            self.logger.error(f"TAS regime detection failed: {e}")
            raise
    
    def _detect_with_nas(self, data: np.ndarray, timestamps: np.ndarray, optimize_performance: bool) -> UnifiedRegimeResult:
        """Detect regimes using NAS system."""
        try:
            tprint_info("🧠 Using NAS regime detection")
            nas_result = self.nas_detector.detect_regimes(
                data, timestamps, optimize_architecture=optimize_performance, 
                enable_meta_learning=True, learn_thresholds=True
            )
            
            # Convert NAS result to unified result
            return UnifiedRegimeResult(
                success=nas_result.success,
                regime_predictions=nas_result.regime_predictions,
                regime_probabilities=nas_result.regime_probabilities,
                economic_significance_scores=nas_result.economic_significance_scores,
                trading_viability_scores=nas_result.trading_viability_scores,
                regime_stability_scores=nas_result.regime_stability_scores,
                transition_probabilities=nas_result.transition_probabilities,
                micro_regimes=nas_result.micro_regimes,
                performance_metrics=nas_result.architecture_performance,
                uncertainty_estimates=nas_result.uncertainty_estimates,
                enhanced_features=None,  # NAS doesn't have CLVSA features
                metadata=nas_result.metadata,
                error_message=nas_result.error_message
            )
            
        except Exception as e:
            tprint_error(f"❌ NAS regime detection failed: {e}")
            self.logger.error(f"NAS regime detection failed: {e}")
            raise
    
    def _detect_with_hybrid(self, data: np.ndarray, timestamps: np.ndarray, optimize_performance: bool) -> UnifiedRegimeResult:
        """Detect regimes using hybrid TAS-NAS system."""
        try:
            tprint_info("🔀 Using hybrid TAS-NAS regime detection")
            
            # Get results from both systems
            tas_result = None
            nas_result = None
            
            if self.tas_detector:
                try:
                    tas_result = self._detect_with_tas(data, timestamps, optimize_performance)
                except Exception as e:
                    tprint_warning(f"⚠️ TAS detection failed in hybrid mode: {e}")
                    tas_result = None
            
            if self.nas_detector:
                try:
                    nas_result = self._detect_with_nas(data, timestamps, optimize_performance)
                except Exception as e:
                    tprint_warning(f"⚠️ NAS detection failed in hybrid mode: {e}")
                    nas_result = None
            
            # Combine results
            if tas_result and nas_result:
                return self._combine_hybrid_results(tas_result, nas_result)
            elif tas_result:
                tprint_info("📊 Using TAS result (NAS failed)")
                return tas_result
            elif nas_result:
                tprint_info("📊 Using NAS result (TAS failed)")
                return nas_result
            else:
                raise RuntimeError("Both TAS and NAS detection failed")
                
        except Exception as e:
            tprint_error(f"❌ Hybrid regime detection failed: {e}")
            self.logger.error(f"Hybrid regime detection failed: {e}")
            raise
    
    def _combine_hybrid_results(self, tas_result: UnifiedRegimeResult, nas_result: UnifiedRegimeResult) -> UnifiedRegimeResult:
        """Combine TAS and NAS results using weighted averaging."""
        try:
            # Weighted combination based on configuration
            tas_weight = self.config.tas_base_weight
            nas_weight = self.config.nas_base_weight
            
            # Normalize weights
            total_weight = tas_weight + nas_weight
            tas_weight /= total_weight
            nas_weight /= total_weight
            
            # Combine predictions using weighted voting
            combined_predictions = np.round(
                tas_weight * tas_result.regime_predictions + 
                nas_weight * nas_result.regime_predictions
            ).astype(int)
            
            # Combine probabilities
            combined_probabilities = (
                tas_weight * tas_result.regime_probabilities + 
                nas_weight * nas_result.regime_probabilities
            )
            
            # Combine scores
            combined_economic = (
                tas_weight * tas_result.economic_significance_scores + 
                nas_weight * nas_result.economic_significance_scores
            )
            
            combined_trading = (
                tas_weight * tas_result.trading_viability_scores + 
                nas_weight * nas_result.trading_viability_scores
            )
            
            combined_stability = (
                tas_weight * tas_result.regime_stability_scores + 
                nas_weight * nas_result.regime_stability_scores
            )
            
            # Combine transition probabilities
            combined_transitions = (
                tas_weight * tas_result.transition_probabilities + 
                nas_weight * nas_result.transition_probabilities
            )
            
            # Create combined metadata
            combined_metadata = {
                'hybrid_mode': True,
                'tas_weight': tas_weight,
                'nas_weight': nas_weight,
                'tas_metadata': tas_result.metadata,
                'nas_metadata': nas_result.metadata
            }
            
            return UnifiedRegimeResult(
                success=True,
                regime_predictions=combined_predictions,
                regime_probabilities=combined_probabilities,
                economic_significance_scores=combined_economic,
                trading_viability_scores=combined_trading,
                regime_stability_scores=combined_stability,
                transition_probabilities=combined_transitions,
                micro_regimes=tas_result.micro_regimes or nas_result.micro_regimes,
                performance_metrics={
                    'tas_metrics': tas_result.performance_metrics,
                    'nas_metrics': nas_result.performance_metrics,
                    'combination_method': 'weighted_average'
                },
                uncertainty_estimates=tas_result.uncertainty_estimates or nas_result.uncertainty_estimates,
                enhanced_features=tas_result.enhanced_features,
                metadata=combined_metadata
            )
            
        except Exception as e:
            tprint_error(f"❌ Hybrid result combination failed: {e}")
            self.logger.error(f"Hybrid result combination failed: {e}")
            raise
    
    def _prepare_and_enhance_data(self, market_data: Union[pd.DataFrame, np.ndarray],
                                 timestamps: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and enhance market data with optimizations."""
        try:
            if isinstance(market_data, pd.DataFrame):
                data_array = market_data.values
                if timestamps is None and 'timestamp' in market_data.columns:
                    timestamps = market_data['timestamp'].values
            else:
                data_array = market_data
                if timestamps is None:
                    timestamps = np.arange(len(data_array))
            
            # Apply matrix optimizations
            if self.enhanced_matrix_ops:
                data_array = self.enhanced_matrix_ops.normalize_data(data_array)
            
            return data_array, timestamps
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise
    
    def _post_process_results(self, result: UnifiedRegimeResult, data: np.ndarray, timestamps: np.ndarray) -> UnifiedRegimeResult:
        """Post-process results with enhanced utilities."""
        try:
            # Calculate enhanced stability if matrix operations available
            if self.enhanced_matrix_ops and result.success:
                try:
                    enhanced_stability = self.enhanced_matrix_ops.calculate_regime_stability(
                        result.regime_predictions, timestamps
                    )
                    if enhanced_stability is not None:
                        result.regime_stability_scores = enhanced_stability
                        self.logger.info("✅ Enhanced regime stability calculated")
                except Exception as e:
                    self.logger.warning(f"Enhanced stability calculation failed: {e}")
            
            # Calculate enhanced transition probabilities
            if self.enhanced_matrix_ops and result.success:
                try:
                    n_regimes = len(np.unique(result.regime_predictions))
                    enhanced_transitions = self.enhanced_matrix_ops.calculate_transition_probabilities(
                        result.regime_predictions, n_regimes
                    )
                    if enhanced_transitions is not None:
                        result.transition_probabilities = enhanced_transitions
                        self.logger.info("✅ Enhanced transition probabilities calculated")
                except Exception as e:
                    self.logger.warning(f"Enhanced transition calculation failed: {e}")
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Post-processing failed: {e}")
            return result
    
    @contextmanager
    def _hardware_optimization_context(self):
        """Context manager for hardware optimization."""
        if self.hardware_manager:
            try:
                self.hardware_manager.start_optimization(WorkloadType.ML_TRAINING)
                yield
            finally:
                self.hardware_manager.stop_optimization()
        else:
            yield
    
    def _log_unified_results_summary(self, result: UnifiedRegimeResult):
        """Log summary of unified results."""
        try:
            self.logger.info("📊 Unified Regime Detection Results Summary:")
            self.logger.info(f"   Success: {result.success}")
            self.logger.info(f"   Execution time: {result.execution_time:.2f}s")
            self.logger.info(f"   System type: {result.system_type}")
            self.logger.info(f"   Architecture: {result.architecture_used}")
            
            if result.success:
                self.logger.info(f"   Regimes detected: {len(np.unique(result.regime_predictions))}")
                self.logger.info(f"   Economic significance: {np.mean(result.economic_significance_scores):.3f}")
                self.logger.info(f"   Trading viability: {np.mean(result.trading_viability_scores):.3f}")
                self.logger.info(f"   Regime stability: {np.mean(result.regime_stability_scores):.3f}")
            
            # Tool integration status
            if HARDWARE_AVAILABLE:
                self.logger.info("   Hardware optimization: ✅ Enabled")
            if COMMON_UTILITIES_AVAILABLE:
                self.logger.info("   Common utilities: ✅ Available")
            if SHARED_UTILITIES_AVAILABLE:
                self.logger.info("   Shared utilities: ✅ Available")
            if TAS_AVAILABLE:
                self.logger.info("   TAS system: ✅ Available")
            if NAS_AVAILABLE:
                self.logger.info("   NAS system: ✅ Available")
                
        except Exception as e:
            self.logger.warning(f"Results summary logging failed: {e}")
    
    def save_results(self, result: UnifiedRegimeResult, filepath: str):
        """Save unified results to file."""
        try:
            if not self.enhanced_serializer:
                self.logger.warning("Serialization not available")
                return False
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            success = self.enhanced_serializer.save(result.to_dict(), filepath)
            
            if success:
                self.logger.info(f"✅ Unified results saved to {filepath}")
            else:
                self.logger.error(f"Failed to save unified results to {filepath}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save unified results: {e}")
            return False
    
    def load_results(self, filepath: str) -> Optional[UnifiedRegimeResult]:
        """Load unified results from file."""
        try:
            if not self.enhanced_serializer:
                self.logger.warning("Serialization not available")
                return None
            
            data = self.enhanced_serializer.load(filepath)
            if data is None:
                self.logger.error(f"Failed to load results from {filepath}")
                return None
            
            result = UnifiedRegimeResult.from_dict(data)
            self.logger.info(f"✅ Unified results loaded from {filepath}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load unified results: {e}")
            return None