"""
Advanced Tree Architecture Search Engine

Main engine for tree-based architecture search with advanced capabilities
including meta-learning, hardware optimization, and regime-aware search.
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

# Import common utilities
from src.utils.common_operations import (
    get_logger, setup_basic_logging, safe_log_metric, safe_log_params, safe_log_artifact,
    get_current_datetime, format_datetime, safe_json_dump, safe_json_load, ensure_directory,
    validate_dataframe, validate_dataframe_columns, safe_fillna, safe_convert_dtypes,
    optimize_dataframe_dtypes, calculate_data_quality_metrics, get_dataframe_info,
    create_data_quality_report, safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    safe_float, safe_int, validate_finite, validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, safe_lower, safe_upper, safe_join,
    safe_dict_get, safe_dict_items, timed_operation, format_bytes, parallel_map,
    validate_correlation_matrix, safe_matrix_inverse, math_safe, MathValidationError,
    safe_rolling, safe_groupby_operation, safe_apply_function, safe_filter_dataframe,
    create_summary_statistics, safe_to_parquet, safe_read_parquet, list_parquet_files,
    get_latest_outcome_file, load_latest_optimal_regime_clustering_outcome,
    safe_copy, safe_deepcopy, safe_resample, align_dataframes, validate_dataframe_schema,
    validate_file_size, guard_dataframe_nulls, secure_file_path, with_tracing_span,
    sanitize_string, memory_checkpoint, gpu_context, optimize_memory, get_memory_usage,
    validate_file_path, get_file_size, check_disk_space
)

# Import math validation utilities
from src.utils.math_validation import (
    safe_divide as math_safe_divide, safe_log as math_safe_log, safe_sqrt as math_safe_sqrt,
    safe_power as math_safe_power, validate_finite as math_validate_finite,
    validate_positive as math_validate_positive, validate_range as math_validate_range,
    validate_numeric_array, safe_kelly_calculation as math_safe_kelly_calculation,
    safe_weighted_average as math_safe_weighted_average, safe_percentage_change as math_safe_percentage_change,
    safe_correlation, safe_covariance, safe_mean as math_safe_mean, safe_std as math_safe_std,
    safe_percentile, validate_correlation_matrix as math_validate_correlation_matrix,
    safe_matrix_inverse as math_safe_matrix_inverse, math_safe, MathValidation as MathValidationClass,
    MathValidationError as MathValidationErrorClass
)

# Import serialization utilities
from src.utils.serialization_utils import JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer

# Import ML common utilities
from src.utils.ml_common.validation.cross_validation import CrossValidator
from src.utils.ml_common.validation.overfitting_detection import OverfittingDetector
from src.utils.ml_common.optimization.hyperparameter_optimization import HyperparameterOptimizer
from src.utils.ml_common.optimization.grid_search import GridSearch
from src.utils.ml_common.optimization.bayesian_optimization import BayesianOptimizer
from src.utils.ml_common.data_leakage_detection import DataLeakageDetector

# Import data utilities
from src.utils.data.unified_data_utils import UnifiedDataUtils
from src.utils.data.feature_engineer import FeatureEngineer
from src.utils.data.quality.data_quality import DataQualityChecker

# Import matrix operations
from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations
from src.utils.matrix_operations.batch_operations import BatchMatrixOperations
from src.utils.matrix_operations.vectorized_core import VectorizedMatrixCore

# Import M1 optimization utilities
from src.utils.hardware.m1_gpu_utils import (
    get_m1_gpu_manager, is_m1_available, is_mps_available, optimize_dataframe_for_m1,
    create_m1_optimized_array, m1_backtesting_simulate, m1_monte_carlo_simulate
)
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer

# Import TAS components
from .tas_config import TASConfig, TASSearchConfig, TASOptimizationConfig
from .tas_result import TASResult, TASSearchResult, TASOptimizationResult
from .tree_architecture import TreeArchitecture, TreeArchitectureCandidate
from .search_space import TreeSearchSpace

# Import unified utilities
from ...hybrid_nas_tas_regime.shared_utils import (
    UnifiedEconomicSignificanceEvaluator, EconomicEvaluationConfig,
    UnifiedTradingViabilityEvaluator, TradingViabilityConfig,
    UnifiedMultiObjectiveOptimizer, OptimizationConfig,
    UnifiedHardwareOptimizer, HardwareConfig,
    UnifiedRegimeAnalyzer, RegimeAnalysisConfig,
    UnifiedValidationSystem, ValidationConfig,
    UnifiedConfigManager, UnifiedRegimeConfig,
    create_unified_economic_evaluator, quick_economic_evaluation,
    create_unified_trading_viability_evaluator, quick_trading_viability_evaluation,
    create_unified_multi_objective_optimizer, quick_multi_objective_optimization,
    create_unified_hardware_optimizer, quick_hardware_optimization,
    create_unified_regime_analyzer, quick_regime_analysis,
    create_unified_config_manager, load_config_from_file, create_environment_config,
    create_unified_validation_system, quick_validation
)

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
    Advanced Tree Architecture Search Engine with Common Utilities Integration.

    Provides comprehensive tree-based architecture search with advanced capabilities
    including meta-learning, hardware optimization, uncertainty estimation,
    regime analysis, and real-time adaptation. Now enhanced with common utilities
    for improved reliability, performance, and maintainability.
    """

    def __init__(self, config: TASEngineConfig):
        """Initialize the TAS engine with common utilities.

        Args:
            config: TAS engine configuration
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        # Initialize common utility managers
        self._initialize_common_utilities()

        # Initialize core components with utilities
        self.search_space = TreeSearchSpace(config.base_config)
        self.evaluator = TreeEvaluator(config.base_config)

        # Initialize advanced components
        self._initialize_advanced_components()

        # Search state
        self.search_history = []
        self.best_architectures = []
        self.current_search = None
        self.performance_monitor = None

        # Initialize utility instances
        self.math_validator = MathValidationClass()
        self.unified_matrix_ops = UnifiedMatrixOperations()
        self.batch_matrix_ops = BatchMatrixOperations()
        self.vectorized_matrix_core = VectorizedMatrixCore()
        self.unified_data_utils = UnifiedDataUtils()
        self.feature_engineer = FeatureEngineer()
        self.data_quality_checker = DataQualityChecker()

        # Initialize M1 optimization
        self.m1_gpu_manager = get_m1_gpu_manager()
        self.m1_memory_optimizer = get_m1_memory_optimizer()
        self.m1_cpu_optimizer = get_m1_cpu_optimizer()

        # Setup logging
        setup_basic_logging()
        safe_log_params({
            'search_strategy': config.search_strategy.value,
            'optimization_mode': config.optimization_mode.value,
            'enable_meta_learning': config.enable_meta_learning,
            'enable_hardware_optimization': config.enable_hardware_optimization,
            'enable_uncertainty_estimation': config.enable_uncertainty_estimation,
            'enable_regime_analysis': config.enable_regime_analysis,
            'enable_real_time_adaptation': config.enable_real_time_adaptation
        })

        self.logger.info("✅ Advanced TAS Engine initialized with common utilities")
        safe_log_metric('initialization_time', time.time())

    def _initialize_common_utilities(self):
        """Initialize common utility managers for enhanced functionality."""
        try:
            # Initialize cross-validation utility
            self.cross_validator = CrossValidator()

            # Initialize overfitting detection
            self.overfitting_detector = OverfittingDetector()

            # Initialize hyperparameter optimization
            self.hyperparameter_optimizer = HyperparameterOptimizer()

            # Initialize grid search
            self.grid_search = GridSearch()

            # Initialize Bayesian optimization
            self.bayesian_optimizer = BayesianOptimizer()

            # Initialize data leakage detection
            self.data_leakage_detector = DataLeakageDetector()

            # Initialize universal serializer
            self.universal_serializer = UniversalSerializer()

            # Initialize memory optimization context manager
            self.memory_checkpoint_context = memory_checkpoint
            self.gpu_context_manager = gpu_context

            self.logger.info("✅ Common utilities initialized successfully")

        except Exception as e:
            self.logger.warning(f"⚠️ Some common utilities failed to initialize: {e}")

    def _initialize_advanced_components(self):
        """Initialize advanced TAS components."""
        try:
            # Meta-learning components
            if self.config.enable_meta_learning:
                self.meta_learner = TreeMetaLearning(self.config.base_config)
                self.maml = TreeMAML(self.config.base_config)
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
    
    @timed_operation
    def search(self,
               train_data: Tuple[np.ndarray, np.ndarray],
               validation_data: Tuple[np.ndarray, np.ndarray],
               test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
               regime_data: Optional[Dict[str, Any]] = None,
               search_strategy: Optional[SearchStrategy] = None,
               optimization_mode: Optional[OptimizationMode] = None) -> TASResult:
        """
        Perform advanced tree architecture search with common utilities.

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
        start_time = get_current_datetime()
        self.logger.info("🚀 Starting advanced tree architecture search with common utilities")

        # Use provided strategy or default
        strategy = search_strategy or self.config.search_strategy
        mode = optimization_mode or self.config.optimization_mode

        safe_log_params({
            'search_strategy': strategy.value,
            'optimization_mode': mode.value,
            'train_data_shape': train_data[0].shape if train_data else None,
            'validation_data_shape': validation_data[0].shape if validation_data else None,
            'test_data_available': test_data is not None,
            'regime_data_available': regime_data is not None
        })

        try:
            # Validate input data using common utilities
            self._validate_search_data(train_data, validation_data, test_data)

            # Check for data leakage
            if not self.data_leakage_detector.detect_data_leakage(
                train_data[0], validation_data[0], test_data[0] if test_data else None
            ):
                self.logger.warning("⚠️ Potential data leakage detected")

            # Optimize data for M1 if available
            if is_m1_available():
                train_data = self._optimize_data_for_m1(train_data)
                validation_data = self._optimize_data_for_m1(validation_data)
                if test_data:
                    test_data = self._optimize_data_for_m1(test_data)
            # Prepare search environment
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
            self.logger.warning(f"⚠️ Post-processing failed: {e}")
            return result
    
    def _save_search_results(self, result: TASResult):
        """Save search results using common utilities."""
        try:
            output_dir = Path(self.config.output_dir)
            ensure_directory(output_dir)

            # Use universal serializer to save result
            result_file = output_dir / "tas_result.json"
            success = self.universal_serializer.save(result.to_dict(), str(result_file))
            if not success:
                raise Exception("Failed to save result with universal serializer")

            # Save best architecture if available using serialization utilities
            if result.best_architecture and self.config.save_models:
                model_file = output_dir / "best_architecture.pkl"
                success = PickleSerializer.save(result.best_architecture, str(model_file))
                if not success:
                    self.logger.warning("⚠️ Failed to save best architecture")

            # Log metrics
            safe_log_metric('results_saved', 1)
            safe_log_artifact('tas_results', str(output_dir))
            self.logger.info(f"💾 Results saved to {output_dir}")

        except Exception as e:
            self.logger.warning(f"⚠️ Could not save results: {e}")

    def save_engine_state(self, filepath: str) -> bool:
        """Save the entire engine state using common utilities."""
        try:
            output_dir = Path(filepath).parent
            ensure_directory(output_dir)

            # Create state dictionary with safe operations
            state = {
                'config': self.config.__dict__ if hasattr(self.config, '__dict__') else vars(self.config),
                'search_history': [safe_dict_get(r.__dict__, 'to_dict', lambda: {})() for r in self.search_history],
                'best_architectures': [safe_dict_get(a.__dict__, 'to_dict', lambda: {})() for a in self.best_architectures],
                'current_search': safe_dict_get(vars(self), 'current_search', None),
                'performance_monitor': safe_dict_get(vars(self), 'performance_monitor', None),
                'timestamp': format_datetime(get_current_datetime()),
                'memory_usage': format_bytes(get_memory_usage()),
                'm1_available': is_m1_available(),
                'mps_available': is_mps_available()
            }

            # Use universal serializer
            success = self.universal_serializer.save(state, filepath)
            if success:
                safe_log_metric('engine_state_saved', 1)
                self.logger.info(f"✅ Engine state saved to {filepath}")
                return True
            else:
                raise Exception("Universal serializer failed")

        except Exception as e:
            self.logger.error(f"❌ Failed to save engine state: {e}")
            return False

    def load_engine_state(self, filepath: str) -> bool:
        """Load engine state using common utilities."""
        try:
            state = self.universal_serializer.load(filepath)
            if state is None:
                raise Exception("Failed to load state with universal serializer")

            # Restore state safely
            if 'config' in state:
                self.config = TASEngineConfig(**state['config'])
            if 'search_history' in state:
                self.search_history = state['search_history']
            if 'best_architectures' in state:
                self.best_architectures = state['best_architectures']

            safe_log_metric('engine_state_loaded', 1)
            self.logger.info(f"✅ Engine state loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to load engine state: {e}")
            return False
    
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
    
    def _validate_search_data(self,
                             train_data: Tuple[np.ndarray, np.ndarray],
                             validation_data: Tuple[np.ndarray, np.ndarray],
                             test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> bool:
        """Validate search data using common utilities."""
        try:
            X_train, y_train = train_data
            X_val, y_val = validation_data

            # Validate data quality
            train_df = pd.DataFrame(X_train)
            val_df = pd.DataFrame(X_val)

            # Check for missing values and data quality issues
            train_quality = calculate_data_quality_metrics(train_df)
            val_quality = calculate_data_quality_metrics(val_df)

            if train_quality.get('missing_percentage', 0) > 50:
                self.logger.warning(f"⚠️ High missing values in training data: {train_quality['missing_percentage']".2f"}%")
            if val_quality.get('missing_percentage', 0) > 50:
                self.logger.warning(f"⚠️ High missing values in validation data: {val_quality['missing_percentage']".2f"}%")

            # Check for class imbalance if classification
            if len(np.unique(y_train)) < 10:  # Likely classification
                train_class_dist = pd.Series(y_train).value_counts(normalize=True)
                val_class_dist = pd.Series(y_val).value_counts(normalize=True)

                if train_class_dist.min() < 0.01:
                    self.logger.warning("⚠️ Potential class imbalance in training data")
                if val_class_dist.min() < 0.01:
                    self.logger.warning("⚠️ Potential class imbalance in validation data")

            return True

        except Exception as e:
            self.logger.error(f"❌ Data validation failed: {e}")
            return False

    def _optimize_data_for_m1(self, data: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Optimize data for M1 hardware using common utilities."""
        try:
            X, y = data

            # Create optimized arrays for M1
            X_optimized = create_m1_optimized_array(X, dtype=np.float32)
            y_optimized = create_m1_optimized_array(y, dtype=np.float32)

            return (X_optimized, y_optimized)

        except Exception as e:
            self.logger.warning(f"⚠️ M1 optimization failed: {e}")
            return data

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics using common utilities."""
        try:
            if not self.search_history:
                return {
                    'total_searches': 0,
                    'best_score': 0.0,
                    'average_execution_time': 0.0,
                    'search_strategies_used': [],
                    'optimization_modes_used': [],
                    'memory_usage': format_bytes(get_memory_usage()),
                    'data_quality': 'No data available'
                }

            # Calculate statistics using safe math operations
            scores = [r.best_score for r in self.search_history]
            execution_times = [r.execution_time for r in self.search_history]
            strategies = list(set([r.search_strategy for r in self.search_history]))
            modes = list(set([r.optimization_mode for r in self.search_history]))

            return {
                'total_searches': len(self.search_history),
                'best_score': safe_float(max(scores)) if scores else 0.0,
                'average_execution_time': safe_float(np.mean(execution_times)) if execution_times else 0.0,
                'search_strategies_used': strategies,
                'optimization_modes_used': modes,
                'memory_usage': format_bytes(get_memory_usage()),
                'avg_score': safe_float(np.mean(scores)) if scores else 0.0,
                'score_std': safe_float(np.std(scores)) if scores else 0.0,
                'min_score': safe_float(min(scores)) if scores else 0.0,
                'max_score': safe_float(max(scores)) if scores else 0.0
            }

        except Exception as e:
            self.logger.error(f"❌ Error calculating search statistics: {e}")
            return {
                'total_searches': len(self.search_history),
                'error': str(e),
                'memory_usage': format_bytes(get_memory_usage())
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