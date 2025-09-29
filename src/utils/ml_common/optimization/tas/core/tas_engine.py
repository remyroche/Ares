"""
Advanced Tree Architecture Search Engine

Main engine for tree-based architecture search with advanced capabilities
including meta-learning, hardware optimization, and regime-aware search.
Extensively integrated with utility modules for optimal performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
from enum import Enum

# Extensive use of common utilities
from ....common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_summary_statistics, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, create_data_quality_report,
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    validate_finite, validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, optimize_dataframe_dtypes,
    safe_to_parquet, safe_read_parquet, integrate_with_m1_optimizers,
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    cleanup_m1_optimizers, memory_checkpoint, gpu_context, optimize_memory,
    get_memory_usage, safe_copy, safe_deepcopy, safe_resample, align_dataframes,
    validate_dataframe_schema, guard_dataframe_nulls, timed_operation,
    format_bytes, parallel_map, chunked_iterable, safe_rolling, safe_groupby_operation,
    safe_apply_function as co_safe_apply_function, create_summary_statistics as co_create_summary_statistics
)

from ....common_utilities import (
    CommonUtilities, safe_dataframe_operation as cu_safe_dataframe_operation,
    validate_dataframe_columns as cu_validate_dataframe_columns,
    calculate_data_quality_metrics as cu_calculate_data_quality_metrics,
    safe_merge_dataframes as cu_safe_merge_dataframes,
    safe_groupby_operation as cu_safe_groupby_operation,
    safe_apply_function as cu_safe_apply_function,
    create_summary_statistics as cu_create_summary_statistics,
    safe_drop_columns as cu_safe_drop_columns,
    safe_rename_columns as cu_safe_rename_columns,
    validate_timestamp_column as cu_validate_timestamp_column,
    safe_timestamp_conversion as cu_safe_timestamp_conversion,
    get_dataframe_info as cu_get_dataframe_info,
    safe_filter_dataframe as cu_safe_filter_dataframe,
    create_data_quality_report as cu_create_data_quality_report
)

from ....math_validation import (
    MathValidation, safe_divide as mv_safe_divide, safe_log as mv_safe_log,
    safe_sqrt as mv_safe_sqrt, safe_power as mv_safe_power,
    validate_finite as mv_validate_finite, validate_positive as mv_validate_positive,
    validate_range as mv_validate_range, safe_kelly_calculation as mv_safe_kelly_calculation,
    safe_weighted_average as mv_safe_weighted_average, safe_percentage_change as mv_safe_percentage_change,
    safe_correlation, safe_covariance, safe_mean as mv_safe_mean, safe_std as mv_safe_std,
    safe_percentile, validate_correlation_matrix, safe_matrix_inverse, math_safe,
    validate_numeric_array
)

from ....tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_structured,
    tprint_with_level, tprint_timer, tprint_logged, configure_tprint,
    get_tprint_config, tprint_context, LogLevel
)

from ....data.klines_parquet import (
    KlinesParquetManager, get_klines_manager, read_ethusdt_data,
    save_klines_to_parquet, load_klines_from_parquet, validate_klines_data
)

from ....serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

# Import data processing utilities
from ....data.processing.data_processing import DataProcessor
from ....data.basic_returns_engineer import BasicReturnsEngineer
from ....data.feature_engineer import FeatureEngineer
from ....data.gap_detector import GapDetector
from ....data.unified_data_utils import UnifiedDataUtils

# Import matrix operations
from ....matrix_operations.unified_operations import MatrixOperations
from ....matrix_operations.enhanced_operations import EnhancedMatrixOperations
from ....matrix_operations.batch_operations import BatchMatrixOperations
from ....matrix_operations.vectorized_core import VectorizedCore
from ....matrix_operations.convenience import MatrixConvenience

# Import hardware utilities
from ....hardware.m1_gpu_utils import M1GPUManager, is_m1_available, is_mps_available
from ....hardware.m1_memory_optimizer import M1MemoryOptimizer
from ....hardware.m1_cpu_optimizer import M1CPUOptimizer

# Import TAS components
from .tas_config import TASConfig, TASSearchConfig, TASOptimizationConfig
from .tas_result import TASResult, TASSearchResult, TASOptimizationResult
from .tree_architecture import TreeArchitecture, TreeArchitectureCandidate
from .search_space import TreeSearchSpace

# Import advanced components
from ..meta_learning.tree_meta_learning import TreeMetaLearning, TreeMAML
from ..search.evolutionary_search import EvolutionaryTreeSearch
from ..search.bayesian_search import BayesianTreeSearch
from ..search.rl_search import RLTreeSearch
from ..optimization.hardware_optimization import TreeHardwareOptimizer
from ..uncertainty.uncertainty_estimation import TreeUncertaintyEstimator
from ..regime_analysis.tree_regime_analyzer import TreeRegimeAnalyzer
from ..adaptation.real_time_adaptation import TreeRealTimeAdapter
from ..evaluation.tree_evaluator import TreeEvaluator

# Import shared utilities
from ...shared_utils.evolutionary_search import (
    EvolutionaryAlgorithmManager, EvolutionaryConfig, EvolutionaryResult,
    create_evolutionary_algorithm_manager
)
from ...shared_utils.feature_engineering import (
    UnifiedFeatureEngineer, FeatureConfig, FeatureEngineeringResult,
    create_unified_feature_engineer
)
from ...shared_utils.evaluation_metrics import (
    UnifiedEvaluator, UnifiedEvaluationResult,
    create_unified_evaluator
)

# Import enhanced TAS components
from ..models.enhanced_tree_models import (
    EnhancedTreeModelFactory, TreeModelConfig, TreeModelResult,
    TreeModelEvaluator, create_model_ensemble
)
from ..automl.tree_automl import (
    TreeAutoMLManager, AutoMLConfig, AutoMLResult,
    create_tree_automl_manager
)
from ...shared_utils.advanced_metrics import (
    AdvancedEvaluator, AdvancedEvaluationResult,
    create_advanced_evaluator
)
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Search strategies for tree architecture search."""
    RANDOM = "random"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT = "reinforcement"
    META_LEARNING = "meta_learning"
    HYBRID = "hybrid"


class OptimizationMode(Enum):
    """Optimization modes for TAS."""
    SINGLE_OBJECTIVE = "single_objective"
    MULTI_OBJECTIVE = "multi_objective"
    REGIME_AWARE = "regime_aware"
    REAL_TIME = "real_time"
    CONTINUAL = "continual"


@dataclass
class TASEngineConfig:
    """Configuration for the TAS engine."""
    
    # Base configuration
    base_config: TASConfig = field(default_factory=TASConfig)
    search_config: TASSearchConfig = field(default_factory=TASSearchConfig)
    optimization_config: TASOptimizationConfig = field(default_factory=TASOptimizationConfig)
    
    # Advanced features
    enable_meta_learning: bool = True
    enable_hardware_optimization: bool = True
    enable_uncertainty_estimation: bool = True
    enable_regime_analysis: bool = True
    enable_real_time_adaptation: bool = True
    enable_continual_learning: bool = True
    
    # Enhanced TAS features
    enable_enhanced_models: bool = True
    enable_automl: bool = True
    enable_evolutionary_search: bool = True
    enable_advanced_metrics: bool = True
    enable_feature_engineering: bool = True
    enable_ensemble: bool = True
    
    # Model types for enhanced TAS
    model_types: List[str] = field(default_factory=lambda: [
        "xgboost", "lightgbm", "catboost", "random_forest", "extra_trees"
    ])
    
    # AutoML settings
    automl_method: str = "optuna"  # "optuna", "grid", "random", "bayesian"
    max_automl_trials: int = 100
    automl_timeout: int = 3600
    
    # Evolutionary search settings
    evolutionary_algorithm: str = "nsga2"  # "nsga2", "spea2", "ga"
    population_size: int = 50
    max_generations: int = 100
    
    # Advanced evaluation settings
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        "risk_adjusted", "regime_aware", "economic_significance", "trading_viability"
    ])
    
    # Feature engineering settings
    feature_selection_method: str = "mutual_info"  # "mutual_info", "f_score", "rfe", "embedded"
    max_features: int = 100
    feature_importance_threshold: float = 0.01
    
    # Multi-objective optimization
    enable_multi_objective: bool = True
    objectives: List[str] = field(default_factory=lambda: [
        "accuracy", "robustness", "efficiency", "interpretability"
    ])
    objective_weights: List[float] = field(default_factory=lambda: [0.4, 0.2, 0.2, 0.2])
    
    # Search strategy
    search_strategy: SearchStrategy = SearchStrategy.HYBRID
    optimization_mode: OptimizationMode = OptimizationMode.REGIME_AWARE
    
    # Performance settings
    max_search_time: int = 3600  # 1 hour
    max_evaluations: int = 1000
    parallel_evaluations: int = 4
    memory_limit_gb: float = 8.0
    
    # Output settings
    save_results: bool = True
    save_models: bool = True
    output_dir: str = "tas_results"
    verbose: bool = True


class TreeArchitectureSearchEngine:
    """
    Advanced Tree Architecture Search Engine.
    
    Provides comprehensive tree-based architecture search with advanced capabilities
    including meta-learning, hardware optimization, uncertainty estimation,
    regime analysis, and real-time adaptation.
    """
    
    def __init__(self, config: TASEngineConfig):
        """Initialize the TAS engine with extensive utility integration.
        
        Args:
            config: TAS engine configuration
        """
        tprint_info("🚀 Initializing Advanced Tree Architecture Search Engine with extensive utility integration")
        tprint_debug(f"Configuration: {config}")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize utility classes extensively
        tprint_debug("🔧 Initializing utility classes")
        self.common_ops = CommonUtilities()
        self.math_validator = MathValidation()
        self.klines_manager = get_klines_manager()
        self.serializer = UniversalSerializer()
        
        # Initialize data processing utilities
        tprint_debug("🔧 Initializing data processing utilities")
        self.data_processor = DataProcessor()
        self.returns_engineer = BasicReturnsEngineer()
        self.feature_engineer = FeatureEngineer()
        self.gap_detector = GapDetector()
        self.unified_data_utils = UnifiedDataUtils()
        
        # Initialize matrix operations
        tprint_debug("🔧 Initializing matrix operations")
        self.matrix_ops = MatrixOperations()
        self.enhanced_matrix_ops = EnhancedMatrixOperations()
        self.batch_matrix_ops = BatchMatrixOperations()
        self.vectorized_core = VectorizedCore()
        self.matrix_convenience = MatrixConvenience()
        
        # Initialize M1 hardware optimizations
        tprint_debug("🔧 Initializing M1 hardware optimizations")
        self.m1_integration = integrate_with_m1_optimizers()
        if self.m1_integration['success']:
            tprint_success("✅ M1 integration successful")
            self.gpu_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
        else:
            tprint_warning("⚠️ M1 integration failed, using fallback")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
        
        # Initialize core components
        tprint_info("🔧 Initializing core components...")
        tprint_debug("🌳 Creating search space...")
        self.search_space = TreeSearchSpace(config.base_config)
        tprint_success("✅ Search space created")
        
        tprint_debug("📊 Creating evaluator...")
        self.evaluator = TreeEvaluator(config.base_config)
        tprint_success("✅ Evaluator created")
        
        # Initialize advanced components
        tprint_info("⚡ Initializing advanced components...")
        self._initialize_advanced_components()
        
        # Initialize enhanced TAS components
        tprint_info("🚀 Initializing enhanced TAS components...")
        self._initialize_enhanced_components()
        
        # Search state
        tprint_debug("📊 Initializing search state...")
        self.search_history = []
        self.best_architectures = []
        self.current_search = None
        self.performance_monitor = None
        tprint_success("✅ Search state initialized")
        
        tprint_success("✅ Advanced TAS Engine initialized with extensive utility integration")
        tprint_info(f"🔍 Search strategy: {config.search_strategy.value}")
        tprint_info(f"⚙️ Optimization mode: {config.optimization_mode.value}")
        tprint_info(f"🧠 Meta-learning: {config.enable_meta_learning}")
        tprint_info(f"🖥️ Hardware optimization: {config.enable_hardware_optimization}")
        tprint_info(f"🎯 Uncertainty estimation: {config.enable_uncertainty_estimation}")
        tprint_info(f"📊 Regime analysis: {config.enable_regime_analysis}")
        tprint_info(f"⚡ Real-time adaptation: {config.enable_real_time_adaptation}")
        self.logger.info("✅ Advanced TAS Engine initialized")
        self.logger.info(f"🔍 Search strategy: {config.search_strategy.value}")
        self.logger.info(f"⚙️ Optimization mode: {config.optimization_mode.value}")
        self.logger.info(f"🧠 Meta-learning: {config.enable_meta_learning}")
        self.logger.info(f"🖥️ Hardware optimization: {config.enable_hardware_optimization}")
        self.logger.info(f"🎯 Uncertainty estimation: {config.enable_uncertainty_estimation}")
        self.logger.info(f"📊 Regime analysis: {config.enable_regime_analysis}")
        self.logger.info(f"⚡ Real-time adaptation: {config.enable_real_time_adaptation}")
    
    def _initialize_advanced_components(self):
        """Initialize advanced TAS components."""
        tprint_debug("🔧 Initializing advanced TAS components...")
        try:
            # Meta-learning components
            if self.config.enable_meta_learning:
                tprint_debug("🧠 Initializing meta-learning components...")
                self.meta_learner = TreeMetaLearning(self.config.base_config)
                self.maml = TreeMAML(self.config.base_config)
                tprint_success("✅ Meta-learning components initialized")
                self.logger.info("✅ Meta-learning components initialized")
            
            # Hardware optimization
            if self.config.enable_hardware_optimization:
                self.hardware_optimizer = TreeHardwareOptimizer(self.config.base_config)
                self.logger.info("✅ Hardware optimization initialized")
            
            # Uncertainty estimation
            if self.config.enable_uncertainty_estimation:
                self.uncertainty_estimator = TreeUncertaintyEstimator(self.config.base_config)
                self.logger.info("✅ Uncertainty estimation initialized")
            
            # Regime analysis
            if self.config.enable_regime_analysis:
                self.regime_analyzer = TreeRegimeAnalyzer(self.config.base_config)
                self.logger.info("✅ Regime analysis initialized")
            
            # Real-time adaptation
            if self.config.enable_real_time_adaptation:
                self.real_time_adapter = TreeRealTimeAdapter(self.config.base_config)
                self.performance_monitor = TreePerformanceMonitor(self.config.base_config)
                self.logger.info("✅ Real-time adaptation initialized")
            
            # Search strategies
            self._initialize_search_strategies()
            
        except Exception as e:
            self.logger.error(f"❌ Advanced components initialization failed: {e}")
            raise
    
    def _initialize_enhanced_components(self):
        """Initialize enhanced TAS components."""
        tprint_debug("🔧 Initializing enhanced TAS components...")
        try:
            # Enhanced models
            if self.config.enable_enhanced_models:
                tprint_debug("🌳 Initializing enhanced tree models...")
                self.enhanced_model_factory = EnhancedTreeModelFactory()
                self.enhanced_model_evaluator = TreeModelEvaluator()
                tprint_success("✅ Enhanced tree models initialized")
                self.logger.info("✅ Enhanced tree models initialized")
            
            # AutoML
            if self.config.enable_automl:
                tprint_debug("🤖 Initializing AutoML...")
                automl_config = AutoMLConfig(
                    optimization_method=self.config.automl_method,
                    max_trials=self.config.max_automl_trials,
                    timeout_seconds=self.config.automl_timeout,
                    model_types=self.config.model_types,
                    enable_ensemble=self.config.enable_ensemble
                )
                self.automl_manager = create_tree_automl_manager(automl_config)
                tprint_success("✅ AutoML initialized")
                self.logger.info("✅ AutoML initialized")
            
            # Evolutionary search
            if self.config.enable_evolutionary_search:
                tprint_debug("🧬 Initializing evolutionary search...")
                evolutionary_config = EvolutionaryConfig(
                    population_size=self.config.population_size,
                    max_generations=self.config.max_generations,
                    use_nsga2=self.config.evolutionary_algorithm == "nsga2",
                    use_spea2=self.config.evolutionary_algorithm == "spea2",
                    use_genetic_algorithm=self.config.evolutionary_algorithm == "ga"
                )
                self.evolutionary_manager = create_evolutionary_algorithm_manager(evolutionary_config)
                tprint_success("✅ Evolutionary search initialized")
                self.logger.info("✅ Evolutionary search initialized")
            
            # Advanced evaluation
            if self.config.enable_advanced_metrics:
                tprint_debug("📊 Initializing advanced evaluation...")
                self.advanced_evaluator = create_advanced_evaluator()
                tprint_success("✅ Advanced evaluation initialized")
                self.logger.info("✅ Advanced evaluation initialized")
            
            # Feature engineering
            if self.config.enable_feature_engineering:
                tprint_debug("🔧 Initializing feature engineering...")
                feature_config = FeatureConfig(
                    enable_technical_indicators=True,
                    enable_feature_selection=True,
                    feature_selection_method=self.config.feature_selection_method,
                    max_features=self.config.max_features,
                    feature_importance_threshold=self.config.feature_importance_threshold
                )
                self.feature_engineer = create_unified_feature_engineer(feature_config)
                tprint_success("✅ Feature engineering initialized")
                self.logger.info("✅ Feature engineering initialized")
            
            # Unified evaluation
            tprint_debug("📊 Initializing unified evaluation...")
            self.unified_evaluator = create_unified_evaluator()
            tprint_success("✅ Unified evaluation initialized")
            self.logger.info("✅ Unified evaluation initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced components initialization failed: {e}")
            raise
    
    def _initialize_search_strategies(self):
        """Initialize search strategies."""
        try:
            self.search_strategies = {}
            
            # Bayesian search
            self.search_strategies[SearchStrategy.BAYESIAN] = BayesianTreeSearch(
                self.config.search_config
            )
            
            # Evolutionary search
            self.search_strategies[SearchStrategy.EVOLUTIONARY] = EvolutionaryTreeSearch(
                self.config.search_config
            )
            
            # Reinforcement learning search
            self.search_strategies[SearchStrategy.REINFORCEMENT] = RLTreeSearch(
                self.config.search_config
            )
            
            self.logger.info("✅ Search strategies initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Search strategies initialization failed: {e}")
            raise
    
    @tprint_timer("Tree Architecture Search")
    def search(self,
               train_data: Tuple[np.ndarray, np.ndarray],
               validation_data: Tuple[np.ndarray, np.ndarray],
               test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
               regime_data: Optional[Dict[str, Any]] = None,
               search_strategy: Optional[SearchStrategy] = None,
               optimization_mode: Optional[OptimizationMode] = None) -> TASResult:
        """
        Perform advanced tree architecture search with extensive utility integration.
        
        Args:
            train_data: Training data (X, y)
            validation_data: Validation data (X, y)
            test_data: Optional test data (X, y)
            regime_data: Optional regime information
            search_strategy: Search strategy to use
            optimization_mode: Optimization mode to use
            
        Returns:
            TASResult with search results
        """
        start_time = time.time()
        tprint_info("🚀 Starting advanced tree architecture search with extensive utility integration")
        
        # Validate input data using math validation utilities
        try:
            X_train, y_train = train_data
            X_val, y_val = validation_data
            
            # Validate data arrays
            X_train = validate_numeric_array(X_train, "training_features")
            y_train = validate_numeric_array(y_train, "training_targets")
            X_val = validate_numeric_array(X_val, "validation_features")
            y_val = validate_numeric_array(y_val, "validation_targets")
            
            if test_data is not None:
                X_test, y_test = test_data
                X_test = validate_numeric_array(X_test, "test_features")
                y_test = validate_numeric_array(y_test, "test_targets")
                test_data = (X_test, y_test)
            
            # Update train and validation data with validated arrays
            train_data = (X_train, y_train)
            validation_data = (X_val, y_val)
            
        except Exception as e:
            tprint_error(f"❌ Data validation failed: {e}")
            return TASResult.create_error_result(f"Data validation failed: {e}")
        
        # Use provided strategy or default
        strategy = search_strategy or self.config.search_strategy
        mode = optimization_mode or self.config.optimization_mode
        
        tprint_info(f"🔍 Using search strategy: {strategy.value}")
        tprint_info(f"⚙️ Using optimization mode: {mode.value}")
        
        # Log data quality metrics
        tprint_debug("📊 Logging data quality metrics")
        train_quality = self._calculate_data_quality_metrics(train_data)
        val_quality = self._calculate_data_quality_metrics(validation_data)
        tprint_structured({"train_quality": train_quality, "validation_quality": val_quality}, LogLevel.DEBUG)
        
        try:
            # Use M1 GPU context if available for search
            with gpu_context("tree_architecture_search") if self.gpu_manager else memory_checkpoint("tree_architecture_search"):
                # Prepare search environment with utility integration
                search_env = self._prepare_search_environment(
                    train_data, validation_data, test_data, regime_data
                )
            
            # Select search strategy
            searcher = self._select_search_strategy(strategy)
            
            # Perform search based on optimization mode
            if mode == OptimizationMode.SINGLE_OBJECTIVE:
                result = self._single_objective_search(searcher, search_env)
            elif mode == OptimizationMode.MULTI_OBJECTIVE:
                result = self._multi_objective_search(searcher, search_env)
            elif mode == OptimizationMode.REGIME_AWARE:
                result = self._regime_aware_search(searcher, search_env)
            elif mode == OptimizationMode.REAL_TIME:
                result = self._real_time_search(searcher, search_env)
            elif mode == OptimizationMode.CONTINUAL:
                result = self._continual_search(searcher, search_env)
            else:
                raise ValueError(f"Unknown optimization mode: {mode}")
            
            # Post-process results
            result = self._post_process_results(result, search_env)
            
            # Save results if requested
            if self.config.save_results:
                self._save_search_results(result)
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            self.logger.info(f"✅ Advanced TAS completed in {execution_time:.2f}s")
            self.logger.info(f"🏆 Best architecture: {result.best_architecture}")
            self.logger.info(f"🎯 Best score: {result.best_score:.4f}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Advanced TAS failed: {e}")
            
            return TASResult(
                best_architecture=None,
                best_score=0.0,
                search_history=[],
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _prepare_search_environment(self,
                                   train_data: Tuple[np.ndarray, np.ndarray],
                                   validation_data: Tuple[np.ndarray, np.ndarray],
                                   test_data: Optional[Tuple[np.ndarray, np.ndarray]],
                                   regime_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare search environment with all necessary components."""
        try:
            search_env = {
                'train_data': train_data,
                'validation_data': validation_data,
                'test_data': test_data,
                'regime_data': regime_data,
                'search_space': self.search_space,
                'evaluator': self.evaluator
            }
            
            # Add advanced components if enabled
            if self.config.enable_meta_learning:
                search_env['meta_learner'] = self.meta_learner
                search_env['maml'] = self.maml
            
            if self.config.enable_hardware_optimization:
                search_env['hardware_optimizer'] = self.hardware_optimizer
            
            if self.config.enable_uncertainty_estimation:
                search_env['uncertainty_estimator'] = self.uncertainty_estimator
            
            if self.config.enable_regime_analysis:
                search_env['regime_analyzer'] = self.regime_analyzer
            
            if self.config.enable_real_time_adaptation:
                search_env['real_time_adapter'] = self.real_time_adapter
                search_env['performance_monitor'] = self.performance_monitor
            
            # Add enhanced TAS components if enabled
            if self.config.enable_enhanced_models:
                search_env['enhanced_model_factory'] = self.enhanced_model_factory
                search_env['enhanced_model_evaluator'] = self.enhanced_model_evaluator
            
            if self.config.enable_automl:
                search_env['automl_manager'] = self.automl_manager
            
            if self.config.enable_evolutionary_search:
                search_env['evolutionary_manager'] = self.evolutionary_manager
            
            if self.config.enable_advanced_metrics:
                search_env['advanced_evaluator'] = self.advanced_evaluator
            
            # Add shared utilities
            if self.config.enable_feature_engineering:
                search_env['feature_engineer'] = self.feature_engineer
            
            search_env['unified_evaluator'] = self.unified_evaluator
            
            return search_env
            
        except Exception as e:
            self.logger.error(f"❌ Search environment preparation failed: {e}")
            raise
    
    def _select_search_strategy(self, strategy: SearchStrategy):
        """Select search strategy."""
        if strategy == SearchStrategy.HYBRID:
            # Use multiple strategies in hybrid mode
            return {
                'bayesian': self.search_strategies[SearchStrategy.BAYESIAN],
                'evolutionary': self.search_strategies[SearchStrategy.EVOLUTIONARY],
                'reinforcement': self.search_strategies[SearchStrategy.REINFORCEMENT]
            }
        else:
            return self.search_strategies[strategy]
    
    def _single_objective_search(self, searcher, search_env: Dict[str, Any]) -> TASResult:
        """Perform single-objective search."""
        self.logger.info("🎯 Performing single-objective search")
        
        if isinstance(searcher, dict):  # Hybrid mode
            # Use Bayesian search for single-objective
            searcher = searcher['bayesian']
        
        return searcher.search(
            search_env['train_data'],
            search_env['validation_data'],
            search_env['test_data']
        )
    
    def _multi_objective_search(self, searcher, search_env: Dict[str, Any]) -> TASResult:
        """Perform multi-objective search."""
        self.logger.info("🎯 Performing multi-objective search")
        
        if isinstance(searcher, dict):  # Hybrid mode
            # Use evolutionary search for multi-objective
            searcher = searcher['evolutionary']
        
        return searcher.multi_objective_search(
            search_env['train_data'],
            search_env['validation_data'],
            search_env['test_data']
        )
    
    def _regime_aware_search(self, searcher, search_env: Dict[str, Any]) -> TASResult:
        """Perform regime-aware search."""
        self.logger.info("🎯 Performing regime-aware search")
        
        if not self.config.enable_regime_analysis:
            self.logger.warning("⚠️ Regime analysis not enabled, falling back to single-objective search")
            return self._single_objective_search(searcher, search_env)
        
        # Use regime analyzer for regime-aware search
        regime_analyzer = search_env['regime_analyzer']
        
        # Analyze regimes
        regime_analysis = regime_analyzer.analyze_regimes(
            search_env['train_data'],
            search_env['regime_data']
        )
        
        # Perform regime-specific search
        if isinstance(searcher, dict):  # Hybrid mode
            # Use Bayesian search for regime-aware
            searcher = searcher['bayesian']
        
        return searcher.regime_aware_search(
            search_env['train_data'],
            search_env['validation_data'],
            search_env['test_data'],
            regime_analysis
        )
    
    def _real_time_search(self, searcher, search_env: Dict[str, Any]) -> TASResult:
        """Perform real-time search."""
        self.logger.info("🎯 Performing real-time search")
        
        if not self.config.enable_real_time_adaptation:
            self.logger.warning("⚠️ Real-time adaptation not enabled, falling back to single-objective search")
            return self._single_objective_search(searcher, search_env)
        
        # Use real-time adapter
        real_time_adapter = search_env['real_time_adapter']
        
        return real_time_adapter.real_time_search(
            search_env['train_data'],
            search_env['validation_data'],
            search_env['test_data']
        )
    
    def _continual_search(self, searcher, search_env: Dict[str, Any]) -> TASResult:
        """Perform continual search."""
        self.logger.info("🎯 Performing continual search")
        
        if not self.config.enable_continual_learning:
            self.logger.warning("⚠️ Continual learning not enabled, falling back to single-objective search")
            return self._single_objective_search(searcher, search_env)
        
        # Use meta-learning for continual search
        if self.config.enable_meta_learning:
            meta_learner = search_env['meta_learner']
            return meta_learner.continual_search(
                search_env['train_data'],
                search_env['validation_data'],
                search_env['test_data']
            )
        else:
            return self._single_objective_search(searcher, search_env)
    
    def _post_process_results(self, result: TASResult, search_env: Dict[str, Any]) -> TASResult:
        """Post-process search results."""
        try:
            # Add uncertainty estimates if enabled
            if self.config.enable_uncertainty_estimation and result.best_architecture:
                uncertainty_estimator = search_env['uncertainty_estimator']
                uncertainty = uncertainty_estimator.estimate_uncertainty(
                    result.best_architecture,
                    search_env['validation_data']
                )
                result.uncertainty_estimates = uncertainty
            
            # Add regime analysis if enabled
            if self.config.enable_regime_analysis and result.best_architecture:
                regime_analyzer = search_env['regime_analyzer']
                regime_analysis = regime_analyzer.analyze_architecture_regimes(
                    result.best_architecture,
                    search_env['train_data']
                )
                result.regime_analysis = regime_analysis
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Post-processing failed: {e}")
            self.logger.warning("⚠️ Post-processing failed - returning results without uncertainty estimates and regime analysis")
            return result
    
    def _save_search_results(self, result: TASResult):
        """Save search results."""
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save result
            result_file = output_dir / "tas_result.json"
            with open(result_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            
            # Save best architecture if available
            if result.best_architecture and self.config.save_models:
                model_file = output_dir / "best_architecture.json"
                with open(model_file, 'w') as f:
                    json.dump(result.best_architecture.to_dict(), f, indent=2, default=str)
            
            self.logger.info(f"💾 Results saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save search results: {e}")
            self.logger.warning("⚠️ Results will only be available in memory - consider checking disk space and permissions")
    
    def adapt_to_new_data(self,
                          new_data: Tuple[np.ndarray, np.ndarray],
                          current_architecture: TreeArchitectureCandidate) -> TreeArchitectureCandidate:
        """
        Adapt current architecture to new data.
        
        Args:
            new_data: New data for adaptation
            current_architecture: Current best architecture
            
        Returns:
            Adapted architecture
        """
        self.logger.info("🔄 Adapting architecture to new data")
        
        try:
            if self.config.enable_meta_learning:
                # Use meta-learning for adaptation
                adapted_architecture = self.meta_learner.adapt_architecture(
                    current_architecture,
                    new_data
                )
                self.logger.info("✅ Architecture adapted using meta-learning")
                return adapted_architecture
            
            elif self.config.enable_real_time_adaptation:
                # Use real-time adaptation
                adapted_architecture = self.real_time_adapter.adapt_architecture(
                    current_architecture,
                    new_data
                )
                self.logger.info("✅ Architecture adapted using real-time adaptation")
                return adapted_architecture
            
            else:
                # Fallback to simple retraining
                self.logger.warning("⚠️ No adaptation method available, returning current architecture")
                return current_architecture
                
        except Exception as e:
            self.logger.error(f"❌ Architecture adaptation failed: {e}")
            return current_architecture
    
    def _calculate_data_quality_metrics(self, data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """Calculate data quality metrics using utility functions.
        
        Args:
            data: Tuple of (X, y) arrays
            
        Returns:
            Dictionary with data quality metrics
        """
        try:
            X, y = data
            
            # Calculate basic statistics using math validation utilities
            metrics = {
                'n_samples': len(X),
                'n_features': X.shape[1] if len(X.shape) > 1 else 1,
                'feature_mean': safe_mean(X.flatten()),
                'feature_std': safe_std(X.flatten()),
                'target_mean': safe_mean(y),
                'target_std': safe_std(y),
                'feature_min': np.min(X),
                'feature_max': np.max(X),
                'target_min': np.min(y),
                'target_max': np.max(y),
                'feature_nan_count': np.isnan(X).sum(),
                'target_nan_count': np.isnan(y).sum(),
                'feature_inf_count': np.isinf(X).sum(),
                'target_inf_count': np.isinf(y).sum()
            }
            
            # Calculate correlation if possible
            if len(X.shape) > 1 and X.shape[1] > 1:
                # Calculate feature correlations
                feature_correlations = []
                for i in range(min(5, X.shape[1])):  # Sample first 5 features
                    for j in range(i+1, min(5, X.shape[1])):
                        corr = safe_correlation(X[:, i], X[:, j])
                        feature_correlations.append(corr)
                
                if feature_correlations:
                    metrics['avg_feature_correlation'] = safe_mean(np.array(feature_correlations))
            
            return metrics
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating data quality metrics: {e}")
            return {'error': str(e)}
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics."""
        try:
            if not self.search_history:
                return {
                    'total_searches': 0,
                    'best_score': 0.0,
                    'average_execution_time': 0.0,
                    'search_strategies_used': [],
                    'optimization_modes_used': []
                }

            # Safely extract scores and times
            valid_scores = [r.best_score for r in self.search_history if hasattr(r, 'best_score') and r.best_score is not None]
            valid_times = [r.execution_time for r in self.search_history if hasattr(r, 'execution_time') and r.execution_time is not None and r.execution_time > 0]
            valid_strategies = [r.search_strategy for r in self.search_history if hasattr(r, 'search_strategy') and r.search_strategy]
            valid_modes = [r.optimization_mode for r in self.search_history if hasattr(r, 'optimization_mode') and r.optimization_mode]

            return {
                'total_searches': len(self.search_history),
                'best_score': max(valid_scores) if valid_scores else 0.0,
                'average_execution_time': np.mean(valid_times) if valid_times else 0.0,
                'search_strategies_used': list(set(valid_strategies)),
                'optimization_modes_used': list(set(valid_modes))
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate search statistics: {e}")
            return {
                'total_searches': len(self.search_history),
                'best_score': 0.0,
                'average_execution_time': 0.0,
                'search_strategies_used': [],
                'optimization_modes_used': [],
                'error': str(e)
            }


# Convenience functions
def create_tas_engine(config: Optional[TASEngineConfig] = None) -> TreeArchitectureSearchEngine:
    """Create a TAS engine with default configuration."""
    if config is None:
        config = TASEngineConfig()
    return TreeArchitectureSearchEngine(config)


def quick_search(train_data: Tuple[np.ndarray, np.ndarray],
                validation_data: Tuple[np.ndarray, np.ndarray],
                test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                search_strategy: SearchStrategy = SearchStrategy.BAYESIAN,
                optimization_mode: OptimizationMode = OptimizationMode.SINGLE_OBJECTIVE) -> TASResult:
    """
    Quick tree architecture search with default settings.
    
    Args:
        train_data: Training data
        validation_data: Validation data
        test_data: Optional test data
        search_strategy: Search strategy
        optimization_mode: Optimization mode
        
    Returns:
        TAS search result
    """
    config = TASEngineConfig(
        search_strategy=search_strategy,
        optimization_mode=optimization_mode,
        enable_meta_learning=False,
        enable_hardware_optimization=False,
        enable_uncertainty_estimation=False,
        enable_regime_analysis=False,
        enable_real_time_adaptation=False
    )
    
    engine = TreeArchitectureSearchEngine(config)
    return engine.search(train_data, validation_data, test_data)