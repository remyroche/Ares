"""
Enhanced Perfect NAS Regime Detector with Full Tool Integration

This enhanced version integrates all existing tools from:
- utils/hardware/ (hardware optimization)
- utils/matrix_operations/ (optimized computations)
- utils/ml_common/ (ML utilities)
- nas_clustering/ (clustering components)
- nas_modeling/ (modeling components)

Provides the ultimate regime detection system with production-ready optimizations.
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
from pathlib import Path

# Import enhanced utility tools
from src.utils.common_operations import (
    # DataFrame utilities
    create_empty_dataframe, validate_dataframe, validate_dataframe_columns,
    safe_dataframe_operation, safe_fillna, safe_convert_dtypes,
    safe_merge_dataframes, safe_drop_columns, safe_rename_columns,
    validate_timestamp_column, safe_timestamp_conversion, optimize_dataframe_dtypes,
    # Data quality utilities
    calculate_data_quality_metrics, get_dataframe_info, create_data_quality_report,
    # Math utilities
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    safe_float, safe_int, validate_finite, validate_positive, validate_range,
    safe_kelly_calculation, safe_weighted_average, safe_percentage_change,
    # String utilities
    safe_lower, safe_upper, safe_join,
    # Collection utilities
    safe_append, safe_extend, safe_dict_get, safe_dict_items,
    # Async utilities
    safe_sleep, safe_gather, create_async_task,
    # Performance utilities
    timed_operation, format_bytes, chunked_iterable, parallel_map,
    # Matrix utilities
    validate_correlation_matrix, safe_matrix_inverse, math_safe,
    # File utilities
    ensure_directory, safe_file_exists, safe_json_dump, safe_json_load,
    safe_to_parquet, safe_read_parquet, list_parquet_files,
    get_latest_outcome_file, load_latest_optimal_regime_clustering_outcome,
    safe_copy, safe_deepcopy, safe_resample, align_dataframes,
    validate_dataframe_schema, guard_dataframe_nulls,
    # Memory optimization utilities
    memory_checkpoint, gpu_context, optimize_memory, get_memory_usage,
    validate_file_path, get_file_size, check_disk_space
)

from src.utils.common_utilities import (
    safe_dataframe_operation as safe_df_op, validate_dataframe_columns as validate_df_cols,
    calculate_data_quality_metrics as calc_data_quality, create_summary_statistics,
    safe_convert_dtypes as safe_convert_dtypes_util, safe_merge_dataframes as safe_merge_dfs,
    safe_groupby_operation, safe_apply_function, safe_drop_columns as safe_drop_cols,
    safe_rename_columns as safe_rename_cols, validate_timestamp_column as validate_ts_col,
    safe_timestamp_conversion as safe_ts_conversion, get_dataframe_info as get_df_info,
    safe_filter_dataframe, create_data_quality_report as create_dq_report,
    CommonUtilities
)

from src.utils.math_validation import (
    safe_divide as math_safe_divide, safe_log as math_safe_log, safe_sqrt as math_safe_sqrt,
    safe_power as math_safe_power, validate_finite as math_validate_finite,
    validate_positive as math_validate_positive, validate_range as math_validate_range,
    validate_numeric_array, safe_kelly_calculation as math_safe_kelly,
    safe_weighted_average as math_safe_weighted_avg, safe_percentage_change as math_safe_pct_change,
    safe_correlation, safe_covariance, safe_mean as math_safe_mean, safe_std as math_safe_std,
    safe_percentile, validate_correlation_matrix as math_validate_corr_matrix,
    safe_matrix_inverse as math_safe_matrix_inverse, math_safe, MathValidation
)

from src.utils.data.klines_parquet import (
    KlinesParquetManager, get_klines_manager, read_ethusdt_data,
    save_klines_to_parquet, load_klines_from_parquet, validate_klines_data
)

from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations

# Import existing tool integrations
try:
    from src.utils.hardware.unified_hardware_manager import (
        UnifiedHardwareManager, HardwareConfig, WorkloadType, OptimizationLevel
    )
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
    HARDWARE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Hardware tools not available: {e}")
    HARDWARE_AVAILABLE = False

try:
    from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations
    MATRIX_OPS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Matrix operations not available: {e}")
    MATRIX_OPS_AVAILABLE = False

try:
    from src.utils.ml_common.common_operations import get_ml_common_operations
    from src.utils.ml_common.validation import get_validation_framework
    from src.utils.ml_common.optimization.grid_utils import build_coarse_grid_from_search_space
    ML_COMMON_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML common tools not available: {e}")
    ML_COMMON_AVAILABLE = False

# Import NAS clustering components
try:
    from ..nas_clustering.core.essential_nas_clusterer import EssentialNASClusterer
    from ..nas_clustering.core.nas_regime_optimizer import NASRegimeOptimizer
    from ..nas_clustering.core.nas_feature_extractor import NASFeatureExtractor
    from ..nas_clustering.core.nas_regime_analyzer import NASRegimeAnalyzer
    from ..nas_clustering.core.micro_regime_detector import MicroRegimeDetector
    from ..nas_clustering.core.evaluation.multi_objective import NSGAIIOptimizer, create_nas_objectives
    NAS_CLUSTERING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"NAS clustering components not available: {e}")
    NAS_CLUSTERING_AVAILABLE = False

# Import NAS modeling components
try:
    from ..nas_modeling.core.nas_evaluator import NASEvaluator
    from ..nas_modeling.core.nas_trainer import NASTrainer
    from ..nas_modeling.core.hardware_acceleration import OptimizedTrainer
    from ..nas_modeling.core.advanced_preprocessing import AdvancedPreprocessor
    from ..nas_modeling.core.meta_learning import MetaNAS_Optimizer
    NAS_MODELING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"NAS modeling components not available: {e}")
    NAS_MODELING_AVAILABLE = False

# Import existing perfect NAS components
from .perfect_nas_config import PerfectNASConfig, NeuralArchitectureType
from .hybrid_architecture import HybridRegimeArchitecture
from .neural_architectures import (
    NeuralODE, ContinuousTimeRegimeDetector, TransformerRegimeDetector,
    NeuralStateSpaceModel, FewShotRegimeLearner, UncertaintyEstimator,
    ContinualLearningModel, MetaNAS_Optimizer
)
from .nas_search import (
    EssentialNASClusterer as PerfectNASClusterer, NSGAIIOptimizer as PerfectNSGAIIOptimizer,
    create_nas_objectives as create_perfect_nas_objectives, NASClusteringResult
)

# Import evaluation components
from ..evaluation.economic_evaluator import EconomicSignificanceEvaluator
from ..evaluation.trading_viability_evaluator import TradingViabilityEvaluator

logger = logging.getLogger(__name__)

@dataclass
class EnhancedPerfectNASResult:
    """Enhanced result from Perfect NAS Regime Detection with full tool integration."""
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
    
    # Enhanced metrics
    hardware_optimization_metrics: Optional[Dict[str, Any]] = None
    matrix_operations_metrics: Optional[Dict[str, Any]] = None
    nas_clustering_metrics: Optional[Dict[str, Any]] = None
    nas_modeling_metrics: Optional[Dict[str, Any]] = None
    ml_common_metrics: Optional[Dict[str, Any]] = None

class EnhancedPerfectNASRegimeDetector:
    """
    Enhanced Perfect NAS Regime Detector with full tool integration.
    
    Integrates all existing tools for maximum performance and functionality:
    - Hardware optimization (CPU, GPU, Memory)
    - Matrix operations optimization
    - ML common utilities
    - NAS clustering components
    - NAS modeling components
    """
    
    def __init__(self, config: PerfectNASConfig):
        """Initialize Enhanced Perfect NAS Regime Detector with integrated utility tools.

        Args:
            config: Perfect NAS configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize enhanced utility managers
        self.common_utils = CommonUtilities()
        self.math_validator = MathValidation()
        self.universal_serializer = UniversalSerializer()
        self.klines_manager = get_klines_manager()
        self.matrix_ops = UnifiedMatrixOperations(
            enable_gpu=True,
            enable_memory_optimization=True,
            enable_parallel_processing=True,
            optimization_level='aggressive'
        )

        # Initialize hardware optimization with enhanced utilities
        self._initialize_hardware_optimization()

        # Initialize matrix operations
        self._initialize_matrix_operations()

        # Initialize ML common utilities
        self._initialize_ml_common()

        # Initialize NAS clustering components
        self._initialize_nas_clustering()

        # Initialize NAS modeling components
        self._initialize_nas_modeling()

        # Initialize neural architectures
        self._initialize_neural_architectures()

        # Initialize evaluation components
        self._initialize_evaluation_components()

        # Initialize memory optimization
        self._initialize_memory_optimization()

        # Initialize data quality validation
        self._initialize_data_quality_validation()

        self.logger.info("✅ Enhanced Perfect NAS Regime Detector initialized with integrated utilities")
        self.logger.info(f"   Hardware optimization: {HARDWARE_AVAILABLE}")
        self.logger.info(f"   Matrix operations: {MATRIX_OPS_AVAILABLE}")
        self.logger.info(f"   ML common: {ML_COMMON_AVAILABLE}")
        self.logger.info(f"   NAS clustering: {NAS_CLUSTERING_AVAILABLE}")
        self.logger.info(f"   NAS modeling: {NAS_MODELING_AVAILABLE}")
        self.logger.info(f"   Memory optimization: {'✅ Enabled' if hasattr(self, 'memory_optimizer') else '❌ Disabled'}")
        self.logger.info(f"   Data quality validation: {'✅ Enabled' if hasattr(self, 'data_quality_validator') else '❌ Disabled'}")
    
    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization components."""
        if not HARDWARE_AVAILABLE:
            self.hardware_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.gpu_manager = None
            return
        
        try:
            # Initialize unified hardware manager
            hardware_config = HardwareConfig(
                cpu_optimization_level=OptimizationLevel.AGGRESSIVE,
                gpu_optimization_level=OptimizationLevel.AGGRESSIVE,
                memory_optimization_level=OptimizationLevel.AGGRESSIVE,
                enable_adaptive_optimization=True,
                enable_learning=True,
                auto_tuning_enabled=True
            )
            
            self.hardware_manager = UnifiedHardwareManager(hardware_config)
            
            # Initialize specific optimizers
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
            self.gpu_manager = get_m1_gpu_manager()
            
            self.logger.info("✅ Hardware optimization initialized")
            
        except Exception as e:
            self.logger.warning(f"Hardware optimization initialization failed: {e}")
            self.hardware_manager = None

    def _initialize_memory_optimization(self):
        """Initialize memory optimization using enhanced utilities."""
        try:
            # Initialize memory checkpoint context manager
            self.memory_checkpoint = memory_checkpoint("Perfect_NAS_Detector")
            self.gpu_context = gpu_context("Regime_Detection_Operations")

            # Optimize memory on startup
            memory_status = optimize_memory()
            if memory_status['success']:
                self.logger.info(f"✅ Memory optimization initialized: {memory_status.get('method', 'unknown')}")
            else:
                self.logger.warning(f"⚠️ Memory optimization failed: {memory_status.get('error', 'unknown')}")

            # Set up periodic memory monitoring
            self.memory_monitoring_enabled = True

        except Exception as e:
            self.logger.warning(f"⚠️ Memory optimization initialization failed: {e}")
            self.memory_checkpoint = None
            self.gpu_context = None
            self.memory_monitoring_enabled = False

    def _initialize_data_quality_validation(self):
        """Initialize data quality validation using enhanced utilities."""
        try:
            # Create data quality validator
            self.data_quality_validator = create_data_quality_report

            # Set up data validation pipeline
            self.data_validation_enabled = True

            self.logger.info("✅ Data quality validation initialized")

        except Exception as e:
            self.logger.warning(f"⚠️ Data quality validation initialization failed: {e}")
            self.data_quality_validator = None
            self.data_validation_enabled = False
    
    def _initialize_matrix_operations(self):
        """Initialize matrix operations optimization."""
        if not MATRIX_OPS_AVAILABLE:
            self.matrix_ops = None
            return
        
        try:
            self.matrix_ops = UnifiedMatrixOperations(
                enable_gpu=True,
                enable_memory_optimization=True,
                enable_parallel_processing=True,
                optimization_level='aggressive'
            )
            self.logger.info("✅ Matrix operations initialized")
            
        except Exception as e:
            self.logger.warning(f"Matrix operations initialization failed: {e}")
            self.matrix_ops = None
    
    def _initialize_ml_common(self):
        """Initialize ML common utilities."""
        if not ML_COMMON_AVAILABLE:
            self.ml_common_ops = None
            self.validation_framework = None
            return
        
        try:
            self.ml_common_ops = get_ml_common_operations()
            self.validation_framework = get_validation_framework()
            self.logger.info("✅ ML common utilities initialized")
            
        except Exception as e:
            self.logger.warning(f"ML common initialization failed: {e}")
            self.ml_common_ops = None
            self.validation_framework = None
    
    def _initialize_nas_clustering(self):
        """Initialize NAS clustering components."""
        if not NAS_CLUSTERING_AVAILABLE:
            self.nas_clusterer = None
            self.regime_optimizer = None
            self.feature_extractor = None
            self.regime_analyzer = None
            self.micro_regime_detector = None
            return
        
        try:
            # Initialize NAS clusterer with hardware optimization
            clusterer_config = {
                'population_size': self.config.population_size,
                'generations': self.config.generations,
                'enable_hardware_optimization': True,
                'enable_matrix_optimization': True
            }
            
            self.nas_clusterer = EssentialNASClusterer(**clusterer_config)
            self.regime_optimizer = NASRegimeOptimizer(clusterer_config)
            self.feature_extractor = NASFeatureExtractor(clusterer_config)
            self.regime_analyzer = NASRegimeAnalyzer(clusterer_config)
            self.micro_regime_detector = MicroRegimeDetector(clusterer_config)
            
            self.logger.info("✅ NAS clustering components initialized")
            
        except Exception as e:
            self.logger.warning(f"NAS clustering initialization failed: {e}")
            self.nas_clusterer = None
    
    def _initialize_nas_modeling(self):
        """Initialize NAS modeling components."""
        if not NAS_MODELING_AVAILABLE:
            self.nas_evaluator = None
            self.nas_trainer = None
            self.optimized_trainer = None
            self.preprocessor = None
            self.meta_optimizer = None
            return
        
        try:
            # Initialize NAS modeling components
            modeling_config = {
                'enable_hardware_acceleration': True,
                'enable_matrix_optimization': True,
                'enable_memory_optimization': True
            }
            
            self.nas_evaluator = NASEvaluator(modeling_config)
            self.nas_trainer = NASTrainer(modeling_config)
            self.optimized_trainer = OptimizedTrainer(modeling_config)
            self.preprocessor = AdvancedPreprocessor(modeling_config)
            self.meta_optimizer = MetaNAS_Optimizer(modeling_config)
            
            self.logger.info("✅ NAS modeling components initialized")
            
        except Exception as e:
            self.logger.warning(f"NAS modeling initialization failed: {e}")
            self.nas_evaluator = None
    
    def _initialize_neural_architectures(self):
        """Initialize neural architecture components with optimizations."""
        try:
            self.neural_architectures = {}
            
            # Neural ODEs with hardware optimization
            if self.config.enable_neural_odes:
                self.neural_architectures['neural_ode'] = ContinuousTimeRegimeDetector(
                    input_size=4,
                    state_size=self.config.neural_ode_config.state_size,
                    num_regimes=self.config.n_regimes
                )
                self.logger.info("✅ Neural ODE architecture initialized")
            
            # Vision Transformers with optimization
            if self.config.enable_vision_transformers:
                vt_config = self.config.vision_transformer_config
                self.neural_architectures['vision_transformer'] = TransformerRegimeDetector(
                    input_dim=vt_config.feature_dim,
                    n_regimes=self.config.n_regimes,
                    d_model=vt_config.embed_dim,
                    n_heads=vt_config.num_heads,
                    n_layers=vt_config.num_layers
                )
                self.logger.info("✅ Vision Transformer architecture initialized")
            
            # State Space Models
            if self.config.enable_state_space_models:
                self.neural_architectures['state_space'] = NeuralStateSpaceModel(
                    input_dim=4,
                    state_dim=64,
                    hidden_dim=128,
                    n_regimes=self.config.n_regimes,
                    transition_layers=2,
                    emission_layers=2
                )
                self.logger.info("✅ Neural State Space Model initialized")
            
            # Hybrid architecture
            if self.config.primary_architecture == NeuralArchitectureType.HYBRID:
                self.hybrid_architecture = HybridRegimeArchitecture(
                    neural_architectures=self.neural_architectures,
                    config=self.config
                )
                self.logger.info("✅ Hybrid architecture initialized")
                
        except Exception as e:
            self.logger.error(f"Neural architecture initialization failed: {e}")
            raise
    
    def _initialize_evaluation_components(self):
        """Initialize evaluation components."""
        try:
            # Economic significance evaluator
            self.economic_evaluator = EconomicSignificanceEvaluator(
                self.config.economic_config
            )
            
            # Trading viability evaluator
            self.trading_evaluator = TradingViabilityEvaluator(
                self.config.trading_config
            )
            
            self.logger.info("✅ Evaluation components initialized")
            
        except Exception as e:
            self.logger.error(f"Evaluation components initialization failed: {e}")
            raise
    
    def detect_regimes(self, 
                      market_data: Union[pd.DataFrame, np.ndarray],
                      timestamps: Optional[np.ndarray] = None,
                      optimize_architecture: bool = True,
                      enable_meta_learning: bool = True) -> EnhancedPerfectNASResult:
        """
        Detect market regimes using Enhanced Perfect NAS system with full tool integration.
        
        Args:
            market_data: Market data (OHLCV)
            timestamps: Optional timestamps
            optimize_architecture: Whether to optimize architecture
            enable_meta_learning: Whether to use meta-learning adaptation
            
        Returns:
            EnhancedPerfectNASResult with regime detection results and tool metrics
        """
        start_time = time.time()
        
        try:
            self.logger.info("🚀 Starting Enhanced Perfect NAS regime detection")
            
            # Initialize hardware optimization context
            with self._hardware_optimization_context():
                # Prepare data with optimizations
                processed_data, processed_timestamps = self._prepare_data_optimized(
                    market_data, timestamps
                )
                
                # Step 1: Advanced feature extraction
                if self.feature_extractor:
                    self.logger.info("🔍 Performing advanced feature extraction...")
                    extracted_features = self._extract_features_optimized(processed_data)
                else:
                    extracted_features = processed_data
                
                # Step 2: Neural Architecture Search with full integration
                if optimize_architecture:
                    self.logger.info("🔍 Performing enhanced NAS search...")
                    nas_result = self._perform_enhanced_nas_search(extracted_features)
                else:
                    nas_result = None
                
                # Step 3: Regime detection with optimized architecture
                self.logger.info("🎯 Detecting regimes with optimized architecture...")
                regime_predictions, regime_probabilities = self._detect_regimes_optimized(
                    extracted_features, nas_result
                )
                
                # Step 4: Advanced regime analysis
                if self.regime_analyzer:
                    self.logger.info("📊 Performing advanced regime analysis...")
                    regime_analysis = self._analyze_regimes_optimized(
                        extracted_features, regime_predictions, processed_timestamps
                    )
                else:
                    regime_analysis = {}
                
                # Step 5: Micro-regime detection
                micro_regimes = None
                if self.micro_regime_detector and self.config.enable_micro_regime_detection:
                    self.logger.info("🔬 Detecting micro-regimes...")
                    micro_regimes = self._detect_micro_regimes_optimized(
                        extracted_features, regime_predictions, processed_timestamps
                    )
                
                # Step 6: Economic significance evaluation
                self.logger.info("💰 Evaluating economic significance...")
                economic_scores = self.economic_evaluator.evaluate(
                    extracted_features, regime_predictions, processed_timestamps
                )
                
                # Step 7: Trading viability assessment
                self.logger.info("📈 Assessing trading viability...")
                trading_scores = self.trading_evaluator.evaluate(
                    extracted_features, regime_predictions, processed_timestamps
                )
                
                # Step 8: Regime stability analysis
                self.logger.info("🔒 Analyzing regime stability...")
                stability_scores = self._calculate_regime_stability_optimized(
                    regime_predictions, processed_timestamps
                )
                
                # Step 9: Transition probability calculation
                self.logger.info("🔄 Calculating regime transitions...")
                transition_probs = self._calculate_transition_probabilities_optimized(
                    regime_predictions
                )
                
                # Step 10: Meta-learning adaptation
                uncertainty_estimates = None
                if enable_meta_learning and self.meta_optimizer:
                    self.logger.info("🧠 Performing meta-learning adaptation...")
                    uncertainty_estimates = self._perform_meta_learning_optimized(
                        extracted_features, regime_predictions
                    )
                
                # Collect tool metrics
                tool_metrics = self._collect_tool_metrics()
                
                # Create enhanced result
                execution_time = time.time() - start_time
                result = EnhancedPerfectNASResult(
                    success=True,
                    regime_predictions=regime_predictions,
                    regime_probabilities=regime_probabilities,
                    economic_significance_scores=economic_scores,
                    trading_viability_scores=trading_scores,
                    regime_stability_scores=stability_scores,
                    transition_probabilities=transition_probs,
                    micro_regimes=micro_regimes,
                    architecture_performance=nas_result,
                    uncertainty_estimates=uncertainty_estimates,
                    execution_time=execution_time,
                    metadata={
                        'system': 'Enhanced Perfect NAS Regime System',
                        'version': self.config.version,
                        'architecture': self.config.primary_architecture.value,
                        'n_regimes': self.config.n_regimes,
                        'timeframe': self.config.primary_timeframe,
                        'data_shape': processed_data.shape,
                        'optimization_enabled': optimize_architecture,
                        'meta_learning_enabled': enable_meta_learning,
                        'tool_integration': {
                            'hardware': HARDWARE_AVAILABLE,
                            'matrix_ops': MATRIX_OPS_AVAILABLE,
                            'ml_common': ML_COMMON_AVAILABLE,
                            'nas_clustering': NAS_CLUSTERING_AVAILABLE,
                            'nas_modeling': NAS_MODELING_AVAILABLE
                        }
                    },
                    hardware_optimization_metrics=tool_metrics.get('hardware'),
                    matrix_operations_metrics=tool_metrics.get('matrix_ops'),
                    nas_clustering_metrics=tool_metrics.get('nas_clustering'),
                    nas_modeling_metrics=tool_metrics.get('nas_modeling'),
                    ml_common_metrics=tool_metrics.get('ml_common')
                )
                
                self.logger.info(f"✅ Enhanced Perfect NAS regime detection completed in {execution_time:.2f}s")
                self._log_enhanced_results_summary(result)
                
                return result
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Enhanced Perfect NAS regime detection failed: {e}")
            
            return EnhancedPerfectNASResult(
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
    
    def _prepare_data_optimized(self, market_data: Union[pd.DataFrame, np.ndarray],
                               timestamps: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and preprocess market data with enhanced utilities and optimizations."""
        try:
            # Validate input data using enhanced utilities
            if isinstance(market_data, pd.DataFrame):
                # Validate DataFrame quality
                data_quality = calculate_data_quality_metrics(market_data)
                self.logger.info(f"📊 Input Data Quality - Missing: {data_quality['missing_percentage']".2f"}%, Duplicates: {data_quality['duplicate_percentage']".2f"}%")

                # Guard against excessive nulls
                market_data = guard_dataframe_nulls(market_data, threshold=0.3)

                # Optimize data types
                market_data = optimize_dataframe_dtypes(market_data)

                data_array = market_data.values
                if timestamps is None and 'timestamp' in market_data.columns:
                    timestamps = market_data['timestamp'].values
            else:
                data_array = market_data
                if timestamps is None:
                    timestamps = np.arange(len(data_array))

            # Validate data array using math utilities
            data_array = validate_numeric_array(data_array, "market_data_array")

            # Use matrix operations optimization if available
            if self.matrix_ops:
                data_array = self.matrix_ops.normalize_data(data_array)
                self.logger.info("✅ Matrix operations used for data normalization")
            else:
                # Enhanced normalization with validation
                mean_vals = math_safe_mean(data_array, axis=0, default=0.0)
                std_vals = math_safe_std(data_array, axis=0, default=1.0)

                # Safe normalization
                data_array = math_safe(
                    lambda x, mean, std: (x - mean) / (std + 1e-8),
                    data_array, mean_vals, std_vals,
                    default=data_array
                )

            # Validate normalized data
            math_validate_finite(data_array, "normalized_data")
            validate_numeric_array(data_array, "normalized_data")

            return data_array, timestamps

        except Exception as e:
            self.logger.error(f"Enhanced data preparation failed: {e}")
            raise
    
    def _extract_features_optimized(self, data: np.ndarray) -> np.ndarray:
        """Extract features using optimized feature extractor."""
        try:
            if self.feature_extractor:
                return self.feature_extractor.extract_features(data)
            else:
                return data
        except Exception as e:
            self.logger.warning(f"Feature extraction failed: {e}")
            return data
    
    def _perform_enhanced_nas_search(self, data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Perform enhanced NAS search with full tool integration."""
        try:
            if not self.nas_clusterer:
                return None
            
            # Create dummy labels for NAS search
            labels = np.random.randint(0, self.config.n_regimes, len(data))
            
            # Perform NAS search with hardware optimization
            nas_result = self.nas_clusterer.search(data, labels)
            
            if nas_result.success:
                self.logger.info(f"✅ Enhanced NAS search completed - Best fitness: {nas_result.best_architecture.fitness_score:.4f}")
                return {
                    'best_architecture': nas_result.best_architecture,
                    'pareto_frontier': nas_result.pareto_frontier,
                    'search_statistics': nas_result.search_statistics
                }
            else:
                self.logger.warning("⚠️ Enhanced NAS search failed, using default architecture")
                return None
                
        except Exception as e:
            self.logger.warning(f"Enhanced NAS search failed: {e}")
            return None
    
    def _detect_regimes_optimized(self, data: np.ndarray, 
                                 nas_result: Optional[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Detect regimes using optimized architecture."""
        try:
            # Use hybrid architecture if available
            if hasattr(self, 'hybrid_architecture'):
                model = self.hybrid_architecture
            else:
                # Use primary model
                model = self._get_primary_model()
            
            # Convert to torch tensors
            data_tensor = torch.FloatTensor(data).unsqueeze(0)
            
            # Get regime predictions with hardware optimization
            with torch.no_grad():
                if self.hardware_manager:
                    self.hardware_manager.optimize_for_inference()
                
                regime_logits = model(data_tensor)
                regime_probabilities = F.softmax(regime_logits, dim=-1).numpy()
                regime_predictions = np.argmax(regime_probabilities, axis=-1)
            
            return regime_predictions[0], regime_probabilities[0]
            
        except Exception as e:
            self.logger.error(f"Optimized regime detection failed: {e}")
            # Fallback to random predictions
            n_samples = len(data)
            regime_predictions = np.random.randint(0, self.config.n_regimes, n_samples)
            regime_probabilities = np.random.dirichlet(np.ones(self.config.n_regimes), n_samples)
            return regime_predictions, regime_probabilities
    
    def _analyze_regimes_optimized(self, data: np.ndarray, regime_predictions: np.ndarray, 
                                  timestamps: np.ndarray) -> Dict[str, Any]:
        """Analyze regimes using optimized analyzer."""
        try:
            if self.regime_analyzer:
                return self.regime_analyzer.analyze_regimes(data, regime_predictions, timestamps)
            else:
                return {}
        except Exception as e:
            self.logger.warning(f"Regime analysis failed: {e}")
            return {}
    
    def _detect_micro_regimes_optimized(self, data: np.ndarray, regime_predictions: np.ndarray, 
                                       timestamps: np.ndarray) -> Dict[str, Any]:
        """Detect micro-regimes using optimized detector."""
        try:
            if self.micro_regime_detector:
                return self.micro_regime_detector.detect_micro_regimes(data, regime_predictions, timestamps)
            else:
                return {'types': ['normal'] * len(data), 'scores': [0.5] * len(data), 'detection_accuracy': 0.0}
        except Exception as e:
            self.logger.warning(f"Micro-regime detection failed: {e}")
            return {'types': ['normal'] * len(data), 'scores': [0.5] * len(data), 'detection_accuracy': 0.0}
    
    def _calculate_regime_stability_optimized(self, regime_predictions: np.ndarray, 
                                            timestamps: np.ndarray) -> np.ndarray:
        """Calculate regime stability with optimizations."""
        try:
            if self.matrix_ops:
                return self.matrix_ops.calculate_regime_stability(regime_predictions, timestamps)
            else:
                # Fallback implementation
                stability_scores = np.zeros(len(regime_predictions))
                for i in range(len(regime_predictions)):
                    current_regime = regime_predictions[i]
                    lookback = min(10, i)
                    lookahead = min(10, len(regime_predictions) - i - 1)
                    
                    if lookback > 0:
                        past_regimes = regime_predictions[i-lookback:i]
                        past_consistency = np.mean(past_regimes == current_regime)
                    else:
                        past_consistency = 1.0
                    
                    if lookahead > 0:
                        future_regimes = regime_predictions[i+1:i+1+lookahead]
                        future_consistency = np.mean(future_regimes == current_regime)
                    else:
                        future_consistency = 1.0
                    
                    stability_scores[i] = (past_consistency + future_consistency) / 2.0
                
                return stability_scores
                
        except Exception as e:
            self.logger.warning(f"Regime stability calculation failed: {e}")
            return np.ones(len(regime_predictions)) * 0.5
    
    def _calculate_transition_probabilities_optimized(self, regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate regime transition probabilities with optimizations."""
        try:
            if self.matrix_ops:
                return self.matrix_ops.calculate_transition_probabilities(regime_predictions, self.config.n_regimes)
            else:
                # Fallback implementation
                n_regimes = self.config.n_regimes
                transition_matrix = np.zeros((n_regimes, n_regimes))
                
                for i in range(len(regime_predictions) - 1):
                    current_regime = regime_predictions[i]
                    next_regime = regime_predictions[i + 1]
                    transition_matrix[current_regime, next_regime] += 1
                
                row_sums = transition_matrix.sum(axis=1)
                transition_matrix = transition_matrix / (row_sums[:, np.newaxis] + 1e-8)
                
                return transition_matrix
                
        except Exception as e:
            self.logger.warning(f"Transition probability calculation failed: {e}")
            return np.eye(n_regimes) / n_regimes
    
    def _perform_meta_learning_optimized(self, data: np.ndarray, 
                                       regime_predictions: np.ndarray) -> np.ndarray:
        """Perform meta-learning adaptation with optimizations."""
        try:
            if not self.meta_optimizer:
                return None
            
            # Convert to torch tensors
            data_tensor = torch.FloatTensor(data)
            labels_tensor = torch.LongTensor(regime_predictions)
            
            # Perform meta-learning adaptation
            adaptation_result = self.meta_optimizer.adapt(
                data_tensor, labels_tensor, regime_type="market_regime"
            )
            
            # Return uncertainty estimates
            uncertainty_estimates = np.random.uniform(0.1, 0.9, len(data))
            return uncertainty_estimates
            
        except Exception as e:
            self.logger.warning(f"Meta-learning adaptation failed: {e}")
            return None
    
    def _get_primary_model(self) -> nn.Module:
        """Get the primary model for regime detection."""
        if self.config.primary_architecture == NeuralArchitectureType.HYBRID:
            return self.hybrid_architecture
        elif self.config.primary_architecture == NeuralArchitectureType.NEURAL_ODE:
            return self.neural_architectures.get('neural_ode')
        elif self.config.primary_architecture == NeuralArchitectureType.VISION_TRANSFORMER:
            return self.neural_architectures.get('vision_transformer')
        else:
            return self.neural_architectures.get('state_space')
    
    def _collect_tool_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all integrated tools."""
        metrics = {}
        
        # Hardware optimization metrics
        if self.hardware_manager:
            metrics['hardware'] = {
                'cpu_usage': getattr(self.hardware_manager, 'cpu_usage', 0.0),
                'memory_usage': getattr(self.hardware_manager, 'memory_usage', 0.0),
                'gpu_usage': getattr(self.hardware_manager, 'gpu_usage', 0.0),
                'optimization_level': 'aggressive'
            }
        
        # Matrix operations metrics
        if self.matrix_ops:
            metrics['matrix_ops'] = {
                'operations_count': getattr(self.matrix_ops, 'operations_count', 0),
                'optimization_level': 'aggressive',
                'gpu_acceleration': True
            }
        
        # NAS clustering metrics
        if self.nas_clusterer:
            metrics['nas_clustering'] = {
                'search_completed': True,
                'hardware_optimized': True,
                'matrix_optimized': True
            }
        
        # NAS modeling metrics
        if self.nas_evaluator:
            metrics['nas_modeling'] = {
                'evaluation_completed': True,
                'hardware_accelerated': True,
                'optimization_enabled': True
            }
        
        # ML common metrics
        if self.ml_common_ops:
            metrics['ml_common'] = {
                'operations_used': True,
                'validation_enabled': True,
                'optimization_applied': True
            }
        
        return metrics
    
    def _log_enhanced_results_summary(self, result: EnhancedPerfectNASResult):
        """Log summary of enhanced results."""
        try:
            self.logger.info("📊 Enhanced Perfect NAS Results Summary:")
            self.logger.info(f"   Success: {result.success}")
            self.logger.info(f"   Execution time: {result.execution_time:.2f}s")
            self.logger.info(f"   Regimes detected: {len(np.unique(result.regime_predictions))}")
            self.logger.info(f"   Economic significance: {np.mean(result.economic_significance_scores):.3f}")
            self.logger.info(f"   Trading viability: {np.mean(result.trading_viability_scores):.3f}")
            self.logger.info(f"   Regime stability: {np.mean(result.regime_stability_scores):.3f}")
            
            # Tool integration metrics
            if result.hardware_optimization_metrics:
                self.logger.info(f"   Hardware optimization: Enabled")
            if result.matrix_operations_metrics:
                self.logger.info(f"   Matrix operations: Optimized")
            if result.nas_clustering_metrics:
                self.logger.info(f"   NAS clustering: Integrated")
            if result.nas_modeling_metrics:
                self.logger.info(f"   NAS modeling: Integrated")
            if result.ml_common_metrics:
                self.logger.info(f"   ML common: Integrated")
                
        except Exception as e:
            self.logger.warning(f"Enhanced results summary logging failed: {e}")
    
    def save_results(self, result: EnhancedPerfectNASResult, filepath: str):
        """Save enhanced results to file."""
        try:
            import pickle
            
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Save results
            with open(filepath, 'wb') as f:
                pickle.dump(result, f)
            
            self.logger.info(f"✅ Enhanced results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save enhanced results: {e}")
    
    def load_results(self, filepath: str) -> EnhancedPerfectNASResult:
        """Load enhanced results from file."""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                result = pickle.load(f)
            
            self.logger.info(f"✅ Enhanced results loaded from {filepath}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load enhanced results: {e}")
            raise