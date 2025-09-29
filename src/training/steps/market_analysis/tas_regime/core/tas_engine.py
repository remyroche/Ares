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
from contextlib import contextmanager

# Import tprint for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import centralized TAS utilities
from src.utils.nas_tas.core.tas_engine import TASEngine
from src.utils.nas_tas.optimization.strategy_search import StrategySearchOptimizer, StrategySearchConfig
try:
    from src.utils.pipeline_results_manager import pipeline_results_manager
except ImportError:
    # Fallback for when the import path is not available
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))
    from src.utils.pipeline_results_manager import pipeline_results_manager

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

# Import enhanced utility tools
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

from src.utils.data.klines_parquet import (
    KlinesParquetManager, get_klines_manager, read_ethusdt_data,
    save_klines_to_parquet, load_klines_from_parquet, validate_klines_data
)

# Import ML common utilities
try:
    from src.utils.ml_common.common_operations import get_ml_common_operations
    from src.utils.ml_common.validation import get_validation_framework
    from src.utils.lookahead_bias_detector import LookaheadBiasDetector
    from src.utils.ml_common.optimization.overfitting_prevention import OverfittingDetector
    from src.utils.ml_common.validation.cv import CrossValidationManager
    from src.utils.ml_common.optimization.hpo_utils import HyperparameterOptimizer
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False

# Import advanced components
try:
    from ..meta_learning.tree_meta_learning import TreeMetaLearning, TreeMAML
    META_LEARNING_AVAILABLE = True
except ImportError:
    META_LEARNING_AVAILABLE = False

try:
    from ..search.evolutionary_search import EvolutionaryTreeSearch
    EVOLUTIONARY_SEARCH_AVAILABLE = True
except ImportError:
    EVOLUTIONARY_SEARCH_AVAILABLE = False

try:
    from ..search.bayesian_search import BayesianTreeSearch
    BAYESIAN_SEARCH_AVAILABLE = True
except ImportError:
    BAYESIAN_SEARCH_AVAILABLE = False

try:
    from ..search.rl_search import RLTreeSearch
    RL_SEARCH_AVAILABLE = True
except ImportError:
    RL_SEARCH_AVAILABLE = False

try:
    from ..optimization.enhanced_hardware_optimization import TreeHardwareOptimizer
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False

try:
    from ..uncertainty.uncertainty_estimation import TreeUncertaintyEstimator
    UNCERTAINTY_ESTIMATION_AVAILABLE = True
except ImportError:
    UNCERTAINTY_ESTIMATION_AVAILABLE = False

try:
    from ..regime_analysis.tree_regime_analyzer import TreeRegimeAnalyzer
    REGIME_ANALYSIS_AVAILABLE = True
except ImportError:
    REGIME_ANALYSIS_AVAILABLE = False

try:
    from ..adaptation.real_time_adaptation import TreeRealTimeAdapter
    REAL_TIME_ADAPTATION_AVAILABLE = True
except ImportError:
    REAL_TIME_ADAPTATION_AVAILABLE = False

try:
    from ..evaluation.tree_evaluator import TreeEvaluator
    TREE_EVALUATOR_AVAILABLE = True
except ImportError:
    TREE_EVALUATOR_AVAILABLE = False

try:
    from ..adaptation.real_time_adaptation import TreePerformanceMonitor
    PERFORMANCE_MONITOR_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITOR_AVAILABLE = False

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
    output_dir: str = "outcomes"
    verbose: bool = True


class TreeArchitectureSearchEngine:
    """
    Advanced Tree Architecture Search Engine.
    
    Provides comprehensive tree-based architecture search with advanced capabilities
    including meta-learning, hardware optimization, uncertainty estimation,
    regime analysis, and real-time adaptation.
    """
    
    def __init__(self, config: TASEngineConfig):
        """Initialize the TAS engine.
        
        Args:
            config: TAS engine configuration
        """
        tprint_info("🚀 Initializing Tree Architecture Search Engine")
        tprint_debug(f"Configuration: {config}")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize utility tools
        tprint_info("🔧 Initializing utility tools...")
        self._initialize_utility_tools()
        
        # Initialize core components
        tprint_info("🧩 Initializing core components...")
        tprint_debug("🌳 Creating search space...")
        self.search_space = TreeSearchSpace(config.base_config)
        tprint_debug("📊 Creating evaluator...")
        self.evaluator = TreeEvaluator(config.base_config)
        
        # Initialize advanced components
        tprint_info("⚡ Initializing advanced components...")
        self._initialize_advanced_components()
        
        # Search state
        tprint_debug("📊 Initializing search state...")
        self.search_history = []
        self.best_architectures = []
        self.current_search = None
        self.performance_monitor = None
        
        self.logger.info("✅ Advanced TAS Engine initialized with enhanced utilities")
        self.logger.info(f"🔍 Search strategy: {config.search_strategy.value}")
        self.logger.info(f"⚙️ Optimization mode: {config.optimization_mode.value}")
        self.logger.info(f"🧠 Meta-learning: {config.enable_meta_learning}")
        self.logger.info(f"🖥️ Hardware optimization: {config.enable_hardware_optimization}")
        self.logger.info(f"🎯 Uncertainty estimation: {config.enable_uncertainty_estimation}")
        self.logger.info(f"📊 Regime analysis: {config.enable_regime_analysis}")
        self.logger.info(f"⚡ Real-time adaptation: {config.enable_real_time_adaptation}")
        self.logger.info(f"🛠️ Utility tools: {self._get_utility_status()}")
    
    def _initialize_utility_tools(self):
        """Initialize enhanced utility tools."""
        tprint_debug("🔧 Starting utility tools initialization...")
        try:
            # Initialize common utilities
            tprint_debug("📦 Creating common utilities...")
            self.common_utils = CommonUtilities()
            tprint_success("✅ Common utilities initialized")
            self.logger.info("✅ Common utilities initialized")
            
            # Initialize math validation
            tprint_debug("🧮 Creating math validator...")
            self.math_validator = MathValidation()
            tprint_success("✅ Math validation initialized")
            self.logger.info("✅ Math validation initialized")
            
            # Initialize matrix operations
            tprint_debug("🔢 Creating matrix operations...")
            self.matrix_ops = get_unified_matrix_operations(
                enable_gpu=True,
                enable_memory_optimization=True,
                enable_parallel=True
            )
            tprint_success("✅ Matrix operations initialized")
            self.logger.info("✅ Matrix operations initialized")
            
            # Initialize serialization
            tprint_debug("💾 Creating serializer...")
            self.serializer = UniversalSerializer()
            tprint_success("✅ Serialization utilities initialized")
            self.logger.info("✅ Serialization utilities initialized")
            
            # Initialize data management
            tprint_debug("📊 Creating klines manager...")
            self.klines_manager = get_klines_manager()
            tprint_success("✅ Klines data manager initialized")
            self.logger.info("✅ Klines data manager initialized")
            
            # Initialize ML common utilities
            tprint_debug("🤖 Checking ML common availability...")
            if ML_COMMON_AVAILABLE:
                tprint_debug("🔧 Creating ML common operations...")
                self.ml_common_ops = get_ml_common_operations()
                tprint_debug("🛡️ Creating validation framework...")
                self.validation_framework = get_validation_framework()
                tprint_debug("🔍 Creating lookahead detector...")
                self.lookahead_detector = LookaheadBiasDetector()
                tprint_debug("⚠️ Creating overfitting detector...")
                self.overfitting_detector = OverfittingDetector()
                tprint_debug("📊 Creating CV manager...")
                self.cv_manager = CrossValidationManager()
                tprint_debug("🎯 Creating HPO optimizer...")
                self.hpo_optimizer = HyperparameterOptimizer()
                tprint_success("✅ ML common utilities initialized")
                self.logger.info("✅ ML common utilities initialized")
            else:
                tprint_warning("⚠️ ML common utilities not available")
                self.ml_common_ops = None
                self.validation_framework = None
                self.lookahead_detector = None
                self.overfitting_detector = None
                self.cv_manager = None
                self.hpo_optimizer = None
                self.logger.warning("⚠️ ML common utilities not available")
            
            # Initialize M1 optimizations
            tprint_debug("🍎 Initializing M1 optimizations...")
            self._initialize_m1_optimizations()
            
        except Exception as e:
            tprint_error(f"❌ Utility tools initialization failed: {e}")
            self.logger.error(f"❌ Utility tools initialization failed: {e}")
            # Set fallback values
            self.common_utils = None
            self.math_validator = None
            self.matrix_ops = None
            self.serializer = None
            self.klines_manager = None
            self.ml_common_ops = None
            self.validation_framework = None
            self.lookahead_detector = None
            self.overfitting_detector = None
            self.cv_manager = None
            self.hpo_optimizer = None
    
    def _initialize_m1_optimizations(self):
        """Initialize M1 hardware optimizations."""
        tprint_debug("🍎 Starting M1 optimizations initialization...")
        try:
            # Get M1 optimizers
            tprint_debug("🎮 Getting M1 GPU manager...")
            self.gpu_manager = get_m1_gpu_manager()
            tprint_debug("💾 Getting M1 memory optimizer...")
            self.memory_optimizer = get_m1_memory_optimizer()
            tprint_debug("⚡ Getting M1 CPU optimizer...")
            self.cpu_optimizer = get_m1_cpu_optimizer()
            
            # Integrate M1 optimizations
            tprint_debug("🔗 Integrating M1 optimizations...")
            integration_result = integrate_with_m1_optimizers()
            if integration_result.get('success', False):
                tprint_success("✅ M1 optimizations integrated successfully")
                tprint_info(f"   GPU Manager: {integration_result.get('gpu_manager', False)}")
                tprint_info(f"   Memory Optimizer: {integration_result.get('memory_optimizer', False)}")
                tprint_info(f"   CPU Optimizer: {integration_result.get('cpu_optimizer', False)}")
                self.logger.info("✅ M1 optimizations integrated successfully")
                self.logger.info(f"   GPU Manager: {integration_result.get('gpu_manager', False)}")
                self.logger.info(f"   Memory Optimizer: {integration_result.get('memory_optimizer', False)}")
                self.logger.info(f"   CPU Optimizer: {integration_result.get('cpu_optimizer', False)}")
            else:
                tprint_warning("⚠️ M1 optimizations integration failed")
                self.logger.warning("⚠️ M1 optimizations integration failed")
                
        except Exception as e:
            tprint_warning(f"⚠️ M1 optimizations initialization failed: {e}")
            self.logger.warning(f"⚠️ M1 optimizations initialization failed: {e}")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
    
    def _get_utility_status(self) -> str:
        """Get status of utility tools."""
        status = []
        if self.common_utils: status.append("CommonOps")
        if self.math_validator: status.append("MathVal")
        if self.matrix_ops: status.append("MatrixOps")
        if self.serializer: status.append("Serialization")
        if self.klines_manager: status.append("DataManager")
        if self.ml_common_ops: status.append("MLCommon")
        if self.gpu_manager: status.append("M1GPU")
        if self.memory_optimizer: status.append("M1Memory")
        if self.cpu_optimizer: status.append("M1CPU")
        return ", ".join(status) if status else "None"
    
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
    
    def search(self,
               train_data: Tuple[np.ndarray, np.ndarray],
               validation_data: Tuple[np.ndarray, np.ndarray],
               test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
               regime_data: Optional[Dict[str, Any]] = None,
               search_strategy: Optional[SearchStrategy] = None,
               optimization_mode: Optional[OptimizationMode] = None) -> TASResult:
        """
        Perform advanced tree architecture search using centralized TAS utilities.
        
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
        self.logger.info("🚀 Starting enhanced tree architecture search with centralized utilities")
        
        try:
            # Create search configuration
            search_config = StrategySearchConfig(
                max_iterations=100,
                population_size=50,
                enable_parallel_processing=True,
                max_workers=4
            )
            
            # Initialize centralized TAS engine
            tas_engine = TASEngine()
            strategy_optimizer = StrategySearchOptimizer(search_config)
            
            # Convert data to DataFrame format for centralized optimizer
            X_train, y_train = train_data
            X_val, y_val = validation_data
            
            # Create DataFrame for search
            train_df = pd.DataFrame(X_train)
            train_df['target'] = y_train
            
            # Define search space
            search_space = {
                'entry_threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                'exit_threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                'risk_factor': [0.5, 1.0, 1.5, 2.0],
                'position_size': [0.05, 0.1, 0.15, 0.2, 0.25]
            }
            
            # Perform search using centralized optimizer
            results = tas_engine.search_strategies(
                data=train_df,
                search_space=search_space,
                optimization_method="bayesian_tpe",
                n_trials=50,
                include_regime_specific=True
            )
            
            if results and 'best_strategy' in results:
                self.logger.info("✅ Strategy search completed successfully")
                # Convert results to TAS result format
                return TASResult(
                    best_architecture=None,  # Tree architectures not applicable
                    best_score=results.get('best_score', 0.0),
                    search_time=results.get('search_time', 0.0),
                    search_history=results.get('trials', []),
                    regime_analysis=results.get('regime_analysis', {}),
                    performance_metrics=results.get('performance_metrics', {})
                )
            else:
                self.logger.warning("⚠️ No strategies found, returning empty result")
                return TASResult()
                
        except Exception as e:
            self.logger.error(f"❌ TAS search failed: {e}")
            return TASResult()
    
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
        """Save search results using pipeline results manager."""
        try:
            # Convert result to dictionary format
            result_dict = result.to_dict()
            
            # Use pipeline results manager to save to outcomes/ directory
            filepath = pipeline_results_manager.save_tas_results(
                tas_result=result_dict,
                symbol=getattr(self, 'symbol', None),
                timeframe=getattr(self, 'timeframe', None),
                additional_metadata={
                    'search_iterations': len(result.search_history) if hasattr(result, 'search_history') else 0,
                    'best_architecture_type': result.best_architecture.architecture_type if result.best_architecture else 'unknown'
                }
            )
            
            # Save best architecture separately if available and configured
            if result.best_architecture and self.config.save_models:
                architecture_data = result.best_architecture.to_dict()
                architecture_filepath = pipeline_results_manager.save_generic_results(
                    result_data=architecture_data,
                    result_type='tas_best_architecture',
                    symbol=getattr(self, 'symbol', None),
                    timeframe=getattr(self, 'timeframe', None),
                    additional_metadata={
                        'architecture_type': result.best_architecture.architecture_type,
                        'performance_score': result.best_performance
                    }
                )
                self.logger.info(f"💾 Best architecture saved to {architecture_filepath}")
            
            self.logger.info(f"ℹ️ TAS results saved to {filepath}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not save results: {e}")
    
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
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics."""
        return {
            'total_searches': len(self.search_history),
            'best_score': max([r.best_score for r in self.search_history]) if self.search_history else 0.0,
            'average_execution_time': np.mean([r.execution_time for r in self.search_history]) if self.search_history else 0.0,
            'search_strategies_used': list(set([r.search_strategy for r in self.search_history])),
            'optimization_modes_used': list(set([r.optimization_mode for r in self.search_history]))
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


# Enhanced utility methods for TAS engine
def _enhanced_data_preparation(self, train_data, validation_data, test_data):
    """Enhanced data preparation using utility tools."""
    try:
        self.logger.info("🔧 Enhanced data preparation with utility tools")
        
        # Convert to DataFrames for utility operations
        train_X, train_y = train_data
        val_X, val_y = validation_data
        
        # Create DataFrames
        train_df = pd.DataFrame(train_X)
        train_df['target'] = train_y
        
        val_df = pd.DataFrame(val_X)
        val_df['target'] = val_y
        
        # Apply data quality checks
        if self.common_utils:
            # Validate data quality
            train_quality = self.common_utils.get_data_summary(train_df)
            val_quality = self.common_utils.get_data_summary(val_df)
            
            self.logger.info(f"📊 Training data quality: {train_quality.get('shape', 'Unknown')}")
            self.logger.info(f"📊 Validation data quality: {val_quality.get('shape', 'Unknown')}")
            
            # Guard against nulls
            train_df = guard_dataframe_nulls(train_df)
            val_df = guard_dataframe_nulls(val_df)
        
        # Apply math validation
        if self.math_validator:
            # Validate numeric columns
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'target':
                    train_df[col] = train_df[col].apply(
                        lambda x: self.math_validator.validate_finite(x, f"train_{col}")
                    )
                    val_df[col] = val_df[col].apply(
                        lambda x: self.math_validator.validate_finite(x, f"val_{col}")
                    )
        
        # Apply matrix optimizations
        if self.matrix_ops:
            # Optimize dataframes
            train_df = self.matrix_ops.optimize_dataframe(train_df)
            val_df = self.matrix_ops.optimize_dataframe(val_df)
        
        # Convert back to numpy arrays
        train_X_enhanced = train_df.drop('target', axis=1).values
        train_y_enhanced = train_df['target'].values
        val_X_enhanced = val_df.drop('target', axis=1).values
        val_y_enhanced = val_df['target'].values
        
        # Handle test data if provided
        if test_data is not None:
            test_X, test_y = test_data
            test_df = pd.DataFrame(test_X)
            test_df['target'] = test_y
            
            if self.common_utils:
                test_df = guard_dataframe_nulls(test_df)
            
            if self.matrix_ops:
                test_df = self.matrix_ops.optimize_dataframe(test_df)
            
            test_X_enhanced = test_df.drop('target', axis=1).values
            test_y_enhanced = test_df['target'].values
            test_data_enhanced = (test_X_enhanced, test_y_enhanced)
        else:
            test_data_enhanced = None
        
        self.logger.info("✅ Enhanced data preparation completed")
        return (train_X_enhanced, train_y_enhanced), (val_X_enhanced, val_y_enhanced), test_data_enhanced
        
    except Exception as e:
        self.logger.warning(f"⚠️ Enhanced data preparation failed: {e}")
        return train_data, validation_data, test_data


def _prepare_enhanced_search_environment(self, train_data, validation_data, test_data, regime_data):
    """Prepare enhanced search environment with utility tools."""
    try:
        search_env = {
            'train_data': train_data,
            'validation_data': validation_data,
            'test_data': test_data,
            'regime_data': regime_data,
            'search_space': self.search_space,
            'evaluator': self.evaluator,
            'utility_tools': {
                'common_utils': self.common_utils,
                'math_validator': self.math_validator,
                'matrix_ops': self.matrix_ops,
                'serializer': self.serializer,
                'klines_manager': self.klines_manager,
                'ml_common_ops': self.ml_common_ops,
                'validation_framework': self.validation_framework,
                'lookahead_detector': self.lookahead_detector,
                'overfitting_detector': self.overfitting_detector,
                'cv_manager': self.cv_manager,
                'hpo_optimizer': self.hpo_optimizer
            }
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
        self.logger.error(f"❌ Enhanced search environment preparation failed: {e}")
        raise


@contextmanager
def _hardware_optimization_context(self):
    """Context manager for hardware optimization."""
    try:
        if self.memory_optimizer:
            with memory_checkpoint("tas_search"):
                if self.gpu_manager:
                    with gpu_context("tas_search"):
                        yield
                else:
                    yield
        else:
            yield
    except Exception as e:
        self.logger.warning(f"⚠️ Hardware optimization context failed: {e}")
        yield


def _enhanced_post_process_results(self, result, search_env):
    """Enhanced post-processing with utility tools."""
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
        
        # Add lookahead bias detection
        if self.lookahead_detector:
            lookahead_analysis = self.lookahead_detector.detect_lookahead_bias(
                search_env['train_data'], search_env['validation_data']
            )
            result.lookahead_analysis = lookahead_analysis
        
        # Add overfitting detection
        if self.overfitting_detector:
            overfitting_analysis = self.overfitting_detector.detect_overfitting(
                search_env['train_data'], search_env['validation_data']
            )
            result.overfitting_analysis = overfitting_analysis
        
        # Add cross-validation results
        if self.cv_manager:
            cv_results = self.cv_manager.perform_cv_analysis(
                search_env['train_data'], search_env['validation_data']
            )
            result.cv_results = cv_results
        
        return result
        
    except Exception as e:
        self.logger.warning(f"⚠️ Enhanced post-processing failed: {e}")
        return result


def _enhanced_save_search_results(self, result):
    """Enhanced save search results with utility tools."""
    try:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use enhanced serialization
        if self.serializer:
            # Save result with enhanced serialization
            result_file = output_dir / "enhanced_tas_result.json"
            result_dict = result.to_dict()
            
            # Add utility tool information
            result_dict['utility_tools_used'] = self._get_utility_status()
            result_dict['enhanced_features'] = {
                'data_quality_checks': self.common_utils is not None,
                'math_validation': self.math_validator is not None,
                'matrix_optimization': self.matrix_ops is not None,
                'ml_common_integration': self.ml_common_ops is not None,
                'm1_optimization': self.gpu_manager is not None
            }
            
            self.serializer.save(result_dict, str(result_file))
            self.logger.info(f"💾 Enhanced results saved to {result_file}")
        
        # Save best architecture if available
        if result.best_architecture and self.config.save_models:
            model_file = output_dir / "best_architecture.json"
            if self.serializer:
                self.serializer.save(result.best_architecture.to_dict(), str(model_file))
            else:
                # Fallback to JSON
                with open(model_file, 'w') as f:
                    json.dump(result.best_architecture.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"💾 Enhanced results saved to {output_dir}")
        
    except Exception as e:
        self.logger.warning(f"⚠️ Enhanced save failed: {e}")
        # Fallback to original save method
        self._save_search_results(result)