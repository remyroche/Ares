"""
Bayesian Lookback Period Optimization with TPE and Intelligent Pruning

This module implements advanced Bayesian optimization to find optimal lookback periods
for feature parameters based on:
1. Mutual Information (MI) maximization for the first lookback period
2. Low correlation & high mutual importance for the second lookback period

Key Features:
- Tree-structured Parzen Estimator (TPE) for intelligent parameter search
- Intelligent pruning strategies to stop unpromising trials early
- Multi-objective optimization for correlation and mutual information
- Transfer learning capabilities for similar parameters
- Real-time optimization monitoring and analytics
"""

import asyncio
import logging
import time
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd


# Import tprint for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import Optuna for Bayesian optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available - using fallback optimization")

# Import mutual information utilities (sklearn)
try:
    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
    from sklearn.metrics import mutual_info_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Sklearn not available - using fallback methods")

# Import correlation utilities (scipy)
try:
    from scipy.stats import pearsonr, spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available - using numpy correlation fallback")

# Import advanced feature selection tools
try:
    from src.training.utils.feature_selection.selection_methods import (
        MRMRSelector, ElasticNetStabilitySelector, CorrelationBasedFilter,
        RecursiveFeatureEliminator, FeatureImportanceRanker
    )
    from src.training.utils.feature_selection.partial_information_decomposition import (
        PartialInformationDecomposition, PIDConfig, PIDMeasure
    )
    from src.training.utils.feature_selection.quality_metrics import QualityMetricsCalculator
    ADVANCED_FEATURE_SELECTION_AVAILABLE = True
except ImportError as e:
    ADVANCED_FEATURE_SELECTION_AVAILABLE = False
    logging.warning(f"Advanced feature selection tools not available: {e}")

# Import common operations for enhanced functionality
try:
    from src.utils.common_operations import (
        safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
        validate_finite, get_memory_usage, timed_operation
    )
    COMMON_OPERATIONS_AVAILABLE = True
except ImportError as e:
    COMMON_OPERATIONS_AVAILABLE = False
    logging.warning(f"Common operations not available: {e}")

# Import math validation utilities for safe operations
try:
    from src.utils.math_validation import (
        safe_correlation, safe_mean as math_safe_mean, safe_std as math_safe_std,
        validate_finite as math_validate_finite, validate_positive, validate_range,
        safe_percentile, math_safe
    )
    from src.utils.core.common import safe_list_get
    MATH_VALIDATION_AVAILABLE = True
except ImportError as e:
    MATH_VALIDATION_AVAILABLE = False
    logging.warning(f"Math validation utilities not available: {e}")
    
    # Fallback implementations
    def safe_list_get(lst, index, default=None):
        try:
            return lst[index] if lst and 0 <= index < len(lst) else default
        except (IndexError, TypeError):
            return default
    
    def safe_correlation(x, y, default=0.0):
        try:
            if len(x) != len(y) or len(x) < 2:
                return default
            corr = np.corrcoef(x, y)[0, 1]
            return corr if np.isfinite(corr) else default
        except Exception as e:
                    tprint_debug(f"🔍 Operation failed: {e}")
            return default

# Import matrix operations for advanced computations
try:
    from src.utils.matrix_operations import (
        safe_correlation_matrix, correlation_matrix_gpu, 
        batch_correlation_analysis, optimize_matrix_operation_with_hardware
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False
    logging.warning("Matrix operations not available - using fallback methods")

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class LookbackOptimizationConfig:
    """Configuration for improved two-step grid + TPE lookback period optimization."""
    
    # Optimization Strategy
    optimization_method: str = "two_step_grid_tpe"  # "two_step_grid_tpe" (no fallbacks)
    
    # Grid Search Configuration
    coarse_grid_size: int = 5          # 5x5 = 25 combinations
    fine_grid_size: int = 5            # 5x5 = 25 combinations
    top_k_coarse_candidates: int = 6   # Top 6 from coarse grid
    top_k_fine_candidates: int = 4     # Top 4 from fine grid
    
    # TPE Configuration
    tpe_trials: int = 25               # TPE fine-tuning trials (reduced from 50)
    tpe_timeout: Optional[int] = None
    n_startup_trials: int = 10
    n_warmup_steps: int = 5
    interval_steps: int = 1
    
    # Sampler/Pruner configuration
    sampler_type: str = "tpe"  # currently supports 'tpe'
    pruner_type: str = "median"  # 'median', 'successive_halving', or 'none'
    
    # Lookback Period Constraints
    min_lookback: int = 5
    max_lookback: int = 100
    lookback_step: int = 1
    
    # Refinement Factors
    coarse_refinement_factor: float = 0.3  # 30% of original range
    fine_refinement_factor: float = 0.2    # 20% of refined range
    
    # Advanced Feature Selection Methods
    first_lookback_method: str = "mutual_info"  # Method for first lookback period
    second_lookback_method: str = "mrmr"  # Method for second lookback period (mRMR only)
    quality_assessment: bool = True  # Enable comprehensive quality metrics
    
    # Correlation and MI Constraints
    max_correlation_threshold: float = 0.7  # Maximum correlation between lookback periods
    min_mutual_info_threshold: float = 0.1  # Minimum mutual information for second period
    correlation_method: str = "pearson"  # "pearson", "spearman", "kendall"
    
    # Multi-objective Weights
    first_lookback_weight: float = 0.4  # Weight for first lookback period (MI)
    second_lookback_weight: float = 0.4  # Weight for second lookback period (mRMR)
    correlation_weight: float = 0.2  # Weight for low correlation between periods
    
    def __post_init__(self):
        """Validate configuration parameters."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters with comprehensive checks."""
        errors = []
        
        # Validate lookback constraints
        if self.min_lookback <= 0:
            errors.append("min_lookback must be positive")
        if self.max_lookback <= self.min_lookback:
            errors.append("max_lookback must be greater than min_lookback")
        if self.lookback_step <= 0:
            errors.append("lookback_step must be positive")
        if self.max_lookback - self.min_lookback < self.lookback_step:
            errors.append("lookback range too small for given step size")
            
        # Validate grid sizes
        if self.coarse_grid_size <= 0:
            errors.append("coarse_grid_size must be positive")
        if self.fine_grid_size <= 0:
            errors.append("fine_grid_size must be positive")
        if self.coarse_grid_size > 20:
            errors.append("coarse_grid_size too large (max 20)")
        if self.fine_grid_size > 20:
            errors.append("fine_grid_size too large (max 20)")
            
        # Validate candidate selection
        if self.top_k_coarse_candidates <= 0:
            errors.append("top_k_coarse_candidates must be positive")
        if self.top_k_fine_candidates <= 0:
            errors.append("top_k_fine_candidates must be positive")
        if self.top_k_coarse_candidates > self.coarse_grid_size ** 2:
            errors.append("top_k_coarse_candidates exceeds total coarse combinations")
        if self.top_k_fine_candidates > self.fine_grid_size ** 2:
            errors.append("top_k_fine_candidates exceeds total fine combinations")
            
        # Validate TPE configuration
        if self.tpe_trials <= 0:
            errors.append("tpe_trials must be positive")
        if self.n_startup_trials < 0:
            errors.append("n_startup_trials must be non-negative")
        if self.n_warmup_steps < 0:
            errors.append("n_warmup_steps must be non-negative")
        if self.interval_steps <= 0:
            errors.append("interval_steps must be positive")
            
        # Validate refinement factors
        if not 0 < self.coarse_refinement_factor <= 1:
            errors.append("coarse_refinement_factor must be between 0 and 1")
        if not 0 < self.fine_refinement_factor <= 1:
            errors.append("fine_refinement_factor must be between 0 and 1")
            
        # Validate thresholds
        if not 0 <= self.max_correlation_threshold <= 1:
            errors.append("max_correlation_threshold must be between 0 and 1")
        if self.min_mutual_info_threshold < 0:
            errors.append("min_mutual_info_threshold must be non-negative")
            
        # Validate weights
        if self.first_lookback_weight < 0:
            errors.append("first_lookback_weight must be non-negative")
        if self.second_lookback_weight < 0:
            errors.append("second_lookback_weight must be non-negative")
        if self.correlation_weight < 0:
            errors.append("correlation_weight must be non-negative")
            
        # Check weight sum (should be reasonable)
        total_weight = self.first_lookback_weight + self.second_lookback_weight + self.correlation_weight
        if total_weight <= 0:
            errors.append("sum of weights must be positive")
        if total_weight > 2.0:
            errors.append("sum of weights seems too large (> 2.0)")
            
        # Validate method names
        valid_methods = ["mutual_info", "mrmr", "elastic_net", "pid", "feature_importance"]
        if self.first_lookback_method not in valid_methods:
            errors.append(f"invalid first_lookback_method: {self.first_lookback_method}")
        if self.second_lookback_method not in valid_methods:
            errors.append(f"invalid second_lookback_method: {self.second_lookback_method}")
            
        valid_correlation_methods = ["pearson", "spearman", "kendall"]
        if self.correlation_method not in valid_correlation_methods:
            errors.append(f"invalid correlation_method: {self.correlation_method}")
            
        # Validate sampler/pruner types
        valid_samplers = ["tpe"]
        if self.sampler_type not in valid_samplers:
            errors.append(f"invalid sampler_type: {self.sampler_type}")
            
        valid_pruners = ["median", "successive_halving", "none"]
        if self.pruner_type not in valid_pruners:
            errors.append(f"invalid pruner_type: {self.pruner_type}")
            
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    # Advanced Feature Selection Parameters
    mrmr_config: Dict[str, Any] = field(default_factory=lambda: {
        'relevance_method': 'mutual_info',
        'redundancy_method': 'correlation',
        'n_neighbors': 3
    })
    
    elastic_net_config: Dict[str, Any] = field(default_factory=lambda: {
        'n_bootstraps': 20,
        'bootstrap_fraction': 0.8,
        'stability_threshold': 0.6,
        'alpha_range': (0.001, 1.0),
        'l1_ratio_range': (0.1, 0.9),
        'cv_folds': 5
    })
    
    pid_config: Dict[str, Any] = field(default_factory=lambda: {
        'method': 'bivariate',
        'pid_measures': ['i_min', 'i_ccs'],
        'discretization_method': 'adaptive',
        'n_bins': 10,
        'enable_parallel': True
    })
    
    quality_metrics_config: Dict[str, Any] = field(default_factory=lambda: {
        'redundancy_weight': 0.2,
        'relevance_weight': 0.3,
        'stability_weight': 0.2,
        'interpretability_weight': 0.1,
        'performance_weight': 0.2,
        'correlation_threshold': 0.8,
        'performance_threshold': 0.7
    })
    
    # Advanced Features
    enable_pruning: bool = True
    enable_parallel: bool = True
    n_jobs: int = -1
    random_state: int = 42
    
    # Performance Monitoring
    enable_monitoring: bool = True
    save_intermediate_results: bool = True
    results_directory: str = "lookback_optimization_results"
    
    # Memory and Performance
    memory_limit_gb: float = 8.0
    enable_memory_optimization: bool = True
    cache_trials: bool = True

@dataclass
class LookbackOptimizationResult:
    """Result of lookback period optimization."""
    
    # Primary Results
    first_lookback_period: int
    second_lookback_period: Optional[int]
    
    # Basic Mutual Information Scores
    first_mi_score: float
    second_mi_score: Optional[float]
    combined_mi_score: float
    
    # Advanced Feature Selection Scores
    first_mrmr_score: Optional[float] = None
    second_mrmr_score: Optional[float] = None
    first_elastic_net_score: Optional[float] = None
    second_elastic_net_score: Optional[float] = None
    first_pid_score: Optional[float] = None
    second_pid_score: Optional[float] = None
    first_importance_score: Optional[float] = None
    second_importance_score: Optional[float] = None
    
    # Quality Metrics
    quality_metrics: Optional[Dict[str, Any]] = None
    overall_quality_score: Optional[float] = None
    
    # Correlation Analysis
    correlation_between_periods: Optional[float] = None
    correlation_method: str = "pearson"
    
    # Advanced Redundancy Analysis
    redundancy_analysis: Optional[Dict[str, Any]] = None
    stability_analysis: Optional[Dict[str, Any]] = None
    
    # Optimization Metrics
    optimization_time: float = 0.0
    n_trials: int = 0
    n_successful_trials: int = 0
    n_pruned_trials: int = 0
    
    # Performance Metrics
    best_score: float = 0.0
    convergence_rate: float = 0.0
    parameter_importance: Dict[str, float] = field(default_factory=dict)
    
    # Feature Selection Method Used
    relevance_method_used: str = "mutual_info"
    redundancy_method_used: str = "correlation"
    
    # Additional Information
    optimization_method: str = "bayesian"
    config: Optional[LookbackOptimizationConfig] = None
    all_trials: List[Dict[str, Any]] = field(default_factory=list)
    convergence_history: List[Dict[str, Any]] = field(default_factory=list)

class MRMRLookbackOptimizer:
    """
    MRMR Lookback Period Optimizer using MI + mRMR approach.
    
    Optimizes lookback periods for feature parameters based on:
    1. Mutual Information (MI) maximization for the first lookback period
    2. mRMR (minimum Redundancy Maximum Relevance) for the second lookback period
    """
    
    def __init__(self, config: Optional[LookbackOptimizationConfig] = None):
        """Initialize the MRMR lookback optimizer."""
        self.config = config or LookbackOptimizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize optimization components
        self.study = None
        self.optimization_history = []
        self.performance_metrics = {}
        
        # Initialize Optuna components
        if OPTUNA_AVAILABLE:
            self._initialize_optuna()
        else:
            self.logger.warning("Optuna not available - using fallback optimization")
        
        # Initialize performance tracking
        self._initialize_performance_tracking()
        
        # Initialize advanced feature selection tools
        self._initialize_advanced_feature_selection()
        
        self.logger.info("🔧 MRMRLookbackOptimizer initialized")
        self.logger.info(f"📊 Optimization method: {self.config.optimization_method}")
        self.logger.info(f"📊 Lookback range: {self.config.min_lookback}-{self.config.max_lookback}")
        self.logger.info(f"📊 First lookback method: {self.config.first_lookback_method}")
        self.logger.info(f"📊 Second lookback method: {self.config.second_lookback_method}")
        self.logger.info(f"📊 Advanced feature selection available: {ADVANCED_FEATURE_SELECTION_AVAILABLE}")
    
    def _initialize_optuna(self):
        """Initialize Optuna study with TPE sampler and intelligent pruning."""
        # Create TPE sampler with optimal settings
        sampler = TPESampler(
            n_startup_trials=self.config.n_startup_trials,
            n_ei_candidates=24,
            seed=self.config.random_state
        )
        
        # Create pruner if enabled
        if self.config.enable_pruning:
            pruner = MedianPruner(
                n_startup_trials=self.config.n_startup_trials,
                n_warmup_steps=self.config.n_warmup_steps,
                interval_steps=self.config.interval_steps
            )
        else:
            pruner = None
        
        # Create study for multi-objective optimization
        self.study = optuna.create_study(
            directions=["maximize", "minimize"],  # Maximize MI, minimize correlation
            sampler=sampler,
            pruner=pruner
        )
        
        self.logger.info("✅ Optuna study initialized with TPE sampler and intelligent pruning")
    
    def _initialize_performance_tracking(self):
        """Initialize performance tracking components."""
        self.performance_metrics = {
            'total_trials': 0,
            'successful_trials': 0,
            'pruned_trials': 0,
            'optimization_time': 0.0,
            'best_score': -np.inf,
            'convergence_rate': 0.0,
            'memory_usage': 0.0
        }
        
        self.convergence_history = []
        self.parameter_importance = {}
    
    def _initialize_advanced_feature_selection(self):
        """Initialize advanced feature selection tools."""
        self.advanced_selectors = {}
        
        if not ADVANCED_FEATURE_SELECTION_AVAILABLE:
            self.logger.warning("Advanced feature selection tools not available")
            return
        
        try:
            # Initialize mRMR selector for second lookback period
            if self.config.second_lookback_method == "mrmr":
                self.advanced_selectors['mrmr'] = MRMRSelector(self.config.mrmr_config)
                self.logger.info("✅ mRMR selector initialized for second lookback period")
            
            # Initialize quality metrics calculator
            if self.config.quality_assessment:
                self.advanced_selectors['quality_metrics'] = QualityMetricsCalculator(self.config.quality_metrics_config)
                self.logger.info("✅ Quality metrics calculator initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize some advanced feature selection tools: {e}")
            # Continue with available tools
    
    def optimize_lookback_periods(self, 
                                 data: pd.DataFrame,
                                 feature_name: str,
                                 target_column: str,
                                 parameter_type: str = "technical_indicator") -> LookbackOptimizationResult:
        """
        Optimize lookback periods for a specific feature parameter.
        
        Args:
            data: Input data with features and target
            feature_name: Name of the feature to optimize
            target_column: Name of the target column
            parameter_type: Type of parameter ("technical_indicator", "moving_average", etc.)
            
        Returns:
            LookbackOptimizationResult with optimal lookback periods
        """
        start_time = time.time()
        self.logger.info(f"🔍 Starting lookback optimization for {feature_name}")
        
        # Validate input data
        if not self._validate_input_data(data, feature_name, target_column):
            raise ValueError("Invalid input data for optimization")
        
        # Create objective function
        def objective(trial):
            return self._lookback_objective(trial, data, feature_name, target_column, parameter_type)
        
        # Run two-step grid + TPE optimization
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for two-step grid + TPE optimization. Please install optuna.")
        
        # Step 1: Coarse 5x5 Grid Search
        coarse_results = self._coarse_grid_search_5x5(data, feature_name, target_column)
        
        # Step 2: Fine 5x5 Grid Search
        if not coarse_results or not coarse_results.get('top_candidates'):
            raise ValueError("Coarse grid search failed to produce valid candidates")
        fine_results = self._fine_grid_search_5x5(data, feature_name, target_column, coarse_results)
        
        # Step 3: TPE Fine-tuning
        if not fine_results or not fine_results.get('top_candidates'):
            raise ValueError("Fine grid search failed to produce valid candidates")
        result = self._tpe_fine_tuning(data, feature_name, target_column, fine_results, start_time)
        
        # Update performance metrics
        self._update_performance_metrics(result)
        
        # Save results if enabled
        if self.config.save_intermediate_results:
            self._save_optimization_results(result, feature_name)
        
        self.logger.info(f"✅ Lookback optimization completed for {feature_name}")
        self.logger.info(f"📊 First lookback: {result.first_lookback_period} (MI: {result.first_mi_score:.4f})")
        if result.second_lookback_period:
            self.logger.info(f"📊 Second lookback: {result.second_lookback_period} (MI: {result.second_mi_score:.4f})")
            self.logger.info(f"📊 Correlation: {result.correlation_between_periods:.4f}")
        
        return result
    
    def _lookback_objective(self, 
                           trial: optuna.Trial,
                           data: pd.DataFrame,
                           feature_name: str,
                           target_column: str,
                           parameter_type: str) -> Tuple[float, float]:
        """
        Enhanced objective function for lookback period optimization using advanced feature selection.
        
        Returns:
            Tuple of (combined_score, penalty_score)
        """
        # Suggest first lookback period
        first_lookback = trial.suggest_int(
            'first_lookback',
            self.config.min_lookback,
            self.config.max_lookback,
            step=self.config.lookback_step
        )
        
        # Suggest second lookback period (optional)
        second_lookback = trial.suggest_int(
            'second_lookback',
            self.config.min_lookback,
            self.config.max_lookback,
            step=self.config.lookback_step
        )
        
        # Ensure second lookback is different from first
        if second_lookback == first_lookback:
            second_lookback = trial.suggest_int(
                'second_lookback_alt',
                self.config.min_lookback,
                self.config.max_lookback,
                step=self.config.lookback_step
            )
        
        # Calculate first lookback period score (using basic MI)
        first_relevance_score = self._calculate_mutual_information(
            data, feature_name, target_column, first_lookback, parameter_type
        )
        
        # Calculate second lookback period score (using mRMR)
        second_relevance_score = self._calculate_second_lookback_mrmr_score(
            data, feature_name, target_column, second_lookback, first_lookback, parameter_type
        )
        
        # Calculate correlation between the two lookback periods
        correlation_penalty = self._calculate_correlation_between_periods(
            data, feature_name, first_lookback, second_lookback, parameter_type
        )
        
        # Calculate quality metrics if enabled
        quality_score = 0.0
        if self.config.quality_assessment and 'quality_metrics' in self.advanced_selectors:
            quality_score = self._calculate_quality_score(
                data, feature_name, target_column, first_lookback, second_lookback, parameter_type
            )
        
        # Calculate combined score with weights
        combined_score = (
            self.config.first_lookback_weight * first_relevance_score +
            self.config.second_lookback_weight * second_relevance_score +
            (1.0 - self.config.first_lookback_weight - self.config.second_lookback_weight) * quality_score
        )
        
        # Calculate penalty score (correlation penalty)
        penalty_score = self.config.correlation_weight * correlation_penalty
        
        # Advanced redundancy penalty (for diagnostics)
        redundancy_penalty = self._calculate_advanced_redundancy_penalty(
            data, feature_name, first_lookback, second_lookback, parameter_type
        )
        
        # Set user attributes for analysis
        trial.set_user_attr("first_lookback", first_lookback)
        trial.set_user_attr("second_lookback", second_lookback)
        trial.set_user_attr("first_relevance_score", first_relevance_score)
        trial.set_user_attr("second_relevance_score", second_relevance_score)
        trial.set_user_attr("correlation_penalty", correlation_penalty)
        trial.set_user_attr("quality_score", quality_score)
        trial.set_user_attr("combined_score", combined_score)
        trial.set_user_attr("penalty_score", penalty_score)
        
        return combined_score, penalty_score
    
    def _calculate_mutual_information(self, 
                                    data: pd.DataFrame,
                                    feature_name: str,
                                    target_column: str,
                                    lookback_period: int,
                                    parameter_type: str) -> float:
        """Calculate mutual information for a specific lookback period with safe operations."""
        try:
            # Generate feature with lookback period
            feature_values = self._generate_feature_with_lookback(
                data, feature_name, lookback_period, parameter_type
            )
            
            # Get target values
            target_values = data[target_column].values
            
            # Safely align arrays
            if len(feature_values) == 0 or len(target_values) == 0:
                return 0.0
                
            min_length = min(len(feature_values), len(target_values))
            feature_values = feature_values[:min_length]
            target_values = target_values[:min_length]
            
            # Remove NaN values using safe operations
            if MATH_VALIDATION_AVAILABLE:
                # Use safe validation
                try:
                    feature_clean = feature_values[np.isfinite(feature_values)]
                    target_clean = target_values[np.isfinite(target_values)]
                    
                    # Align cleaned arrays
                    valid_mask = np.isfinite(feature_values) & np.isfinite(target_values)
                    feature_clean = feature_values[valid_mask]
                    target_clean = target_values[valid_mask]
                except Exception:
                    return 0.0
            else:
                # Fallback NaN removal
                mask = ~(np.isnan(feature_values) | np.isnan(target_values))
                feature_clean = feature_values[mask]
                target_clean = target_values[mask]
            
            # Check minimum data requirement (increased from 10 to 30 for reliability)
            min_samples = max(30, lookback_period * 2)
            if len(feature_clean) < min_samples:
                self.logger.debug(f"Insufficient data for MI calculation: {len(feature_clean)} < {min_samples}")
                return 0.0
            
            # Calculate mutual information with safe operations
            if SKLEARN_AVAILABLE:
                try:
                    # Use sklearn for continuous target
                    if data[target_column].dtype in ['float64', 'int64']:
                        mi_scores = mutual_info_regression(
                            feature_clean.reshape(-1, 1), 
                            target_clean,
                            random_state=self.config.random_state
                        )
                        # Safe array access
                        mi_score = safe_list_get(mi_scores, 0, 0.0)
                    else:
                        # For categorical target
                        mi_scores = mutual_info_classif(
                            feature_clean.reshape(-1, 1), 
                            target_clean,
                            random_state=self.config.random_state
                        )
                        # Safe array access
                        mi_score = safe_list_get(mi_scores, 0, 0.0)
                except Exception as e:
                    self.logger.debug(f"Sklearn MI calculation failed: {e}")
                    mi_score = 0.0
            else:
                mi_score = 0.0
            
            # Fallback to safe correlation if MI failed
            if mi_score == 0.0:
                if MATH_VALIDATION_AVAILABLE:
                    correlation = safe_correlation(feature_clean, target_clean, default=0.0)
                else:
                    try:
                        corr_matrix = np.corrcoef(feature_clean, target_clean)
                        if corr_matrix.shape == (2, 2):
                            correlation = corr_matrix[0, 1]
                            correlation = correlation if np.isfinite(correlation) else 0.0
                        else:
                            correlation = 0.0
                    except Exception:
                        correlation = 0.0
                
                mi_score = abs(correlation)
            
            # Validate result
            if MATH_VALIDATION_AVAILABLE:
                try:
                    mi_score = math_validate_finite(mi_score, "mutual_information")
                    mi_score = max(0.0, mi_score)  # Ensure non-negative
                except Exception:
                    mi_score = 0.0
            
            return float(mi_score)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate MI for lookback {lookback_period}: {e}")
            return 0.0
    
    def _calculate_second_lookback_mrmr_score(self, 
                                            data: pd.DataFrame,
                                            feature_name: str,
                                            target_column: str,
                                            second_lookback: int,
                                            first_lookback: int,
                                            parameter_type: str) -> float:
        """
        Calculate mRMR score for the second lookback period.
        This considers both relevance to target and redundancy with the first lookback period.
        """
        try:
            if self.config.second_lookback_method != "mrmr" or 'mrmr' not in self.advanced_selectors:
                # Fallback to basic mutual information
                return self._calculate_mutual_information(
                    data, feature_name, target_column, second_lookback, parameter_type
                )
            
            # Generate features for both lookback periods
            first_feature = self._generate_feature_with_lookback(
                data, feature_name, first_lookback, parameter_type
            )
            second_feature = self._generate_feature_with_lookback(
                data, feature_name, second_lookback, parameter_type
            )
            
            # Get target values
            target_values = data[target_column].values
            
            # Ensure same length and remove NaN values
            min_length = min(len(first_feature), len(second_feature), len(target_values))
            first_feature = first_feature[:min_length]
            second_feature = second_feature[:min_length]
            target_values = target_values[:min_length]
            
            mask = ~(np.isnan(first_feature) | np.isnan(second_feature) | np.isnan(target_values))
            first_feature = first_feature[mask]
            second_feature = second_feature[mask]
            target_values = target_values[mask]
            
            if len(first_feature) < 10:
                return 0.0
            
            # Create feature matrix with both lookback periods
            X = np.column_stack([first_feature, second_feature])
            feature_names = [f"{feature_name}_lookback_{first_lookback}", f"{feature_name}_lookback_{second_lookback}"]
            
            # Use mRMR to select features (we want the second feature)
            result = self.advanced_selectors['mrmr'].select_features(X, target_values, feature_names, 2)
            
            if result['success'] and result['scores']:
                # Get the mRMR score for the second lookback period
                second_feature_name = f"{feature_name}_lookback_{second_lookback}"
                if second_feature_name in result['scores']:
                    return result['scores'][second_feature_name]
                else:
                    # If second feature not selected, return 0 (low relevance/redundancy)
                    return 0.0
            else:
                # Fallback to basic mutual information
                return self._calculate_mutual_information(
                    data, feature_name, target_column, second_lookback, parameter_type
                )
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate mRMR score for second lookback period: {e}")
            # Fallback to basic mutual information
            return self._calculate_mutual_information(
                data, feature_name, target_column, second_lookback, parameter_type
            )
    
    def _calculate_advanced_relevance_score(self, 
                                          data: pd.DataFrame,
                                          feature_name: str,
                                          target_column: str,
                                          lookback_period: int,
                                          parameter_type: str) -> float:
        """Calculate advanced relevance score using the configured method."""
        try:
            # Generate feature with lookback period
            feature_values = self._generate_feature_with_lookback(
                data, feature_name, lookback_period, parameter_type
            )
            
            # Get target values
            target_values = data[target_column].values
            
            # Ensure same length and remove NaN values
            min_length = min(len(feature_values), len(target_values))
            feature_values = feature_values[:min_length]
            target_values = target_values[:min_length]
            
            mask = ~(np.isnan(feature_values) | np.isnan(target_values))
            feature_values = feature_values[mask]
            target_values = target_values[mask]
            
            if len(feature_values) < 10:
                return 0.0
            
            # Use configured relevance method
            if self.config.relevance_method == "mrmr" and 'mrmr' in self.advanced_selectors:
                return self._calculate_mrmr_relevance_score(feature_values, target_values)
            elif self.config.relevance_method == "elastic_net" and 'elastic_net' in self.advanced_selectors:
                return self._calculate_elastic_net_relevance_score(feature_values, target_values)
            elif self.config.relevance_method == "pid" and 'pid' in self.advanced_selectors:
                return self._calculate_pid_relevance_score(feature_values, target_values)
            elif self.config.relevance_method == "feature_importance" and 'feature_importance' in self.advanced_selectors:
                return self._calculate_importance_relevance_score(feature_values, target_values)
            else:
                # Fallback to basic mutual information
                return self._calculate_mutual_information(
                    data, feature_name, target_column, lookback_period, parameter_type
                )
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate advanced relevance score: {e}")
            return 0.0
    
    def _calculate_advanced_redundancy_penalty(self, 
                                             data: pd.DataFrame,
                                             feature_name: str,
                                             first_lookback: int,
                                             second_lookback: int,
                                             parameter_type: str) -> float:
        """Calculate advanced redundancy penalty using the configured method."""
        try:
            # Generate features for both lookback periods
            first_feature = self._generate_feature_with_lookback(
                data, feature_name, first_lookback, parameter_type
            )
            second_feature = self._generate_feature_with_lookback(
                data, feature_name, second_lookback, parameter_type
            )
            
            # Ensure same length and remove NaN values
            min_length = min(len(first_feature), len(second_feature))
            first_feature = first_feature[:min_length]
            second_feature = second_feature[:min_length]
            
            mask = ~(np.isnan(first_feature) | np.isnan(second_feature))
            first_feature = first_feature[mask]
            second_feature = second_feature[mask]
            
            if len(first_feature) < 10:
                return 1.0  # High penalty for insufficient data
            
            # Use configured redundancy method
            if self.config.redundancy_method == "elastic_net" and 'elastic_net' in self.advanced_selectors:
                return self._calculate_elastic_net_redundancy_penalty(first_feature, second_feature)
            elif self.config.redundancy_method == "mrmr" and 'mrmr' in self.advanced_selectors:
                return self._calculate_mrmr_redundancy_penalty(first_feature, second_feature)
            elif self.config.redundancy_method == "pid" and 'pid' in self.advanced_selectors:
                return self._calculate_pid_redundancy_penalty(first_feature, second_feature)
            else:
                # Fallback to basic correlation
                return self._calculate_correlation_between_periods(
                    data, feature_name, first_lookback, second_lookback, parameter_type
                )
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate advanced redundancy penalty: {e}")
            return 1.0  # High penalty for errors
    
    def _calculate_quality_score(self, 
                               data: pd.DataFrame,
                               feature_name: str,
                               target_column: str,
                               first_lookback: int,
                               second_lookback: int,
                               parameter_type: str) -> float:
        """Calculate quality score using comprehensive quality metrics."""
        try:
            if 'quality_metrics' not in self.advanced_selectors:
                return 0.0
            
            # Generate features for both lookback periods
            first_feature = self._generate_feature_with_lookback(
                data, feature_name, first_lookback, parameter_type
            )
            second_feature = self._generate_feature_with_lookback(
                data, feature_name, second_lookback, parameter_type
            )
            
            # Get target values
            target_values = data[target_column].values
            
            # Ensure same length and remove NaN values
            min_length = min(len(first_feature), len(second_feature), len(target_values))
            first_feature = first_feature[:min_length]
            second_feature = second_feature[:min_length]
            target_values = target_values[:min_length]
            
            mask = ~(np.isnan(first_feature) | np.isnan(second_feature) | np.isnan(target_values))
            first_feature = first_feature[mask]
            second_feature = second_feature[mask]
            target_values = target_values[mask]
            
            if len(first_feature) < 10:
                return 0.0
            
            # Create feature matrix with both lookback periods
            X = np.column_stack([first_feature, second_feature])
            feature_names = [f"{feature_name}_lookback_{first_lookback}", f"{feature_name}_lookback_{second_lookback}"]
            
            # Calculate quality metrics
            quality_result = self.advanced_selectors['quality_metrics'].calculate_comprehensive_quality_metrics(
                X, target_values, feature_names, feature_names
            )
            
            return quality_result.get('overall_quality_score', 0.0)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate quality score: {e}")
            return 0.0
    
    def _calculate_mrmr_relevance_score(self, feature_values: np.ndarray, target_values: np.ndarray) -> float:
        """Calculate mRMR relevance score."""
        try:
            # Create feature matrix
            X = feature_values.reshape(-1, 1)
            feature_names = ['feature']
            
            # Use mRMR selector to get relevance score
            result = self.advanced_selectors['mrmr'].select_features(X, target_values, feature_names, 1)
            
            if result['success'] and result['scores']:
                return list(result['scores'].values())[0]
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate mRMR relevance score: {e}")
            return 0.0
    
    def _calculate_elastic_net_relevance_score(self, feature_values: np.ndarray, target_values: np.ndarray) -> float:
        """Calculate Elastic Net relevance score."""
        try:
            # Create feature matrix
            X = feature_values.reshape(-1, 1)
            feature_names = ['feature']
            
            # Use Elastic Net selector to get stability score
            result = self.advanced_selectors['elastic_net'].select_features(X, target_values, feature_names)
            
            if result['success'] and result['stability_scores']:
                return list(result['stability_scores'].values())[0]
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate Elastic Net relevance score: {e}")
            return 0.0
    
    def _calculate_pid_relevance_score(self, feature_values: np.ndarray, target_values: np.ndarray) -> float:
        """Calculate PID relevance score."""
        try:
            # Use PID analyzer to get information decomposition
            # This is a simplified version - full PID analysis would be more complex
            result = self.advanced_selectors['pid'].analyze_information_decomposition(
                feature_values, target_values
            )
            
            if result and 'total_information' in result:
                return result['total_information']
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate PID relevance score: {e}")
            return 0.0
    
    def _calculate_importance_relevance_score(self, feature_values: np.ndarray, target_values: np.ndarray) -> float:
        """Calculate feature importance relevance score."""
        try:
            # Create feature matrix
            X = feature_values.reshape(-1, 1)
            feature_names = ['feature']
            
            # Use feature importance ranker
            result = self.advanced_selectors['feature_importance'].select_features(X, target_values, feature_names, 1)
            
            if result['success'] and result['importance_scores']:
                return list(result['importance_scores'].values())[0]
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate importance relevance score: {e}")
            return 0.0
    
    def _calculate_elastic_net_redundancy_penalty(self, first_feature: np.ndarray, second_feature: np.ndarray) -> float:
        """Calculate Elastic Net redundancy penalty."""
        try:
            # Create feature matrix with both features
            X = np.column_stack([first_feature, second_feature])
            feature_names = ['feature1', 'feature2']
            
            # Use Elastic Net selector to analyze redundancy
            result = self.advanced_selectors['elastic_net'].select_features(X, first_feature, feature_names)
            
            if result['success'] and result['all_stability_scores']:
                # Higher stability scores indicate more redundancy
                scores = list(result['all_stability_scores'].values())
                return np.mean(scores) if scores else 0.0
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate Elastic Net redundancy penalty: {e}")
            return 0.0
    
    def _calculate_mrmr_redundancy_penalty(self, first_feature: np.ndarray, second_feature: np.ndarray) -> float:
        """Calculate mRMR redundancy penalty."""
        try:
            # Create feature matrix with both features
            X = np.column_stack([first_feature, second_feature])
            feature_names = ['feature1', 'feature2']
            
            # Use mRMR selector to analyze redundancy
            result = self.advanced_selectors['mrmr'].select_features(X, first_feature, feature_names, 2)
            
            if result['success'] and result['scores']:
                # Calculate redundancy as inverse of mRMR scores
                scores = list(result['scores'].values())
                return 1.0 - np.mean(scores) if scores else 0.0
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate mRMR redundancy penalty: {e}")
            return 0.0
    
    def _calculate_pid_redundancy_penalty(self, first_feature: np.ndarray, second_feature: np.ndarray) -> float:
        """Calculate PID redundancy penalty."""
        try:
            # Use PID analyzer to get redundancy information
            result = self.advanced_selectors['pid'].analyze_information_decomposition(
                first_feature, second_feature
            )
            
            if result and 'redundant_information' in result:
                return result['redundant_information']
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate PID redundancy penalty: {e}")
            return 0.0
    
    def _calculate_correlation_between_periods(self,
                                             data: pd.DataFrame,
                                             feature_name: str,
                                             first_lookback: int,
                                             second_lookback: int,
                                             parameter_type: str) -> float:
        """Calculate correlation between two lookback periods with safe operations."""
        try:
            # Generate features for both lookback periods
            first_feature = self._generate_feature_with_lookback(
                data, feature_name, first_lookback, parameter_type
            )
            second_feature = self._generate_feature_with_lookback(
                data, feature_name, second_lookback, parameter_type
            )
            
            # Validate feature arrays
            if len(first_feature) == 0 or len(second_feature) == 0:
                return 1.0  # High penalty for empty arrays
            
            # Safely align arrays
            min_length = min(len(first_feature), len(second_feature))
            first_feature = first_feature[:min_length]
            second_feature = second_feature[:min_length]
            
            # Remove NaN/infinite values with safe operations
            if MATH_VALIDATION_AVAILABLE:
                valid_mask = np.isfinite(first_feature) & np.isfinite(second_feature)
                if not np.any(valid_mask):
                    return 1.0  # High penalty for no valid data
                    
                first_clean = first_feature[valid_mask]
                second_clean = second_feature[valid_mask]
            else:
                # Fallback NaN removal
                mask = ~(np.isnan(first_feature) | np.isnan(second_feature))
                if not np.any(mask):
                    return 1.0
                first_clean = first_feature[mask]
                second_clean = second_feature[mask]
            
            # Check minimum data requirement
            min_samples = max(30, max(first_lookback, second_lookback) * 2)
            if len(first_clean) < min_samples:
                self.logger.debug(f"Insufficient data for correlation: {len(first_clean)} < {min_samples}")
                return 1.0  # High penalty for insufficient data
            
            # Calculate correlation with safe operations
            correlation = 0.0
            
            if MATH_VALIDATION_AVAILABLE:
                # Use safe correlation from math_validation
                correlation = safe_correlation(first_clean, second_clean, default=1.0)
            elif SCIPY_AVAILABLE:
                try:
                    if self.config.correlation_method == "pearson":
                        correlation, p_value = pearsonr(first_clean, second_clean)
                        # Check if correlation is significant (optional)
                        if not np.isfinite(correlation):
                            correlation = 1.0
                    elif self.config.correlation_method == "spearman":
                        correlation, p_value = spearmanr(first_clean, second_clean)
                        if not np.isfinite(correlation):
                            correlation = 1.0
                    else:
                        corr_matrix = np.corrcoef(first_clean, second_clean)
                        if corr_matrix.shape == (2, 2):
                            correlation = corr_matrix[0, 1]
                        else:
                            correlation = 1.0
                except Exception as e:
                    self.logger.debug(f"SciPy correlation calculation failed: {e}")
                    correlation = 1.0
            else:
                # Fallback to numpy correlation
                try:
                    corr_matrix = np.corrcoef(first_clean, second_clean)
                    if corr_matrix.shape == (2, 2):
                        correlation = corr_matrix[0, 1]
                        correlation = correlation if np.isfinite(correlation) else 1.0
                    else:
                        correlation = 1.0
                except Exception as e:
                    self.logger.debug(f"Numpy correlation calculation failed: {e}")
                    correlation = 1.0
            
            # Validate and return absolute correlation
            if MATH_VALIDATION_AVAILABLE:
                try:
                    correlation = math_validate_finite(correlation, "correlation")
                except Exception:
                    correlation = 1.0
            
            # Return absolute correlation (we want low correlation penalty)
            result = abs(float(correlation))
            return min(result, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate correlation between periods: {e}")
            return 1.0  # High correlation penalty for errors
    
    def _generate_feature_with_lookback(self,
                                      data: pd.DataFrame,
                                      feature_name: str,
                                      lookback_period: int,
                                      parameter_type: str) -> np.ndarray:
        """Generate feature values with specific lookback period."""
        try:
            if parameter_type == "technical_indicator":
                # For technical indicators, use rolling window
                if feature_name in data.columns:
                    return data[feature_name].rolling(window=lookback_period).mean().values
                else:
                    # Generate basic technical indicator
                    return data['close'].rolling(window=lookback_period).mean().values
            
            elif parameter_type == "moving_average":
                # For moving averages
                return data['close'].rolling(window=lookback_period).mean().values
            
            elif parameter_type == "volatility":
                # For volatility indicators
                returns = data['close'].pct_change()
                return returns.rolling(window=lookback_period).std().values
            
            elif parameter_type == "momentum":
                # For momentum indicators
                return data['close'].pct_change(periods=lookback_period).values
            
            else:
                # Default to simple moving average
                return data['close'].rolling(window=lookback_period).mean().values
                
        except Exception as e:
            self.logger.warning(f"Failed to generate feature with lookback {lookback_period}: {e}")
            return np.full(len(data), np.nan)
    
    def _validate_input_data(self, data: pd.DataFrame, feature_name: str, target_column: str) -> bool:
        """Validate input data for optimization."""
        try:
            # Check if data is not empty
            if data.empty:
                self.logger.error("Input data is empty")
                return False
            
            # Check if target column exists
            if target_column not in data.columns:
                self.logger.error(f"Target column '{target_column}' not found in data")
                return False
            
            # Check if we have enough data points
            if len(data) < self.config.max_lookback * 2:
                self.logger.warning(f"Insufficient data points: {len(data)} < {self.config.max_lookback * 2}")
                return False
            
            # Check for required columns
            required_columns = ['close']
            for col in required_columns:
                if col not in data.columns:
                    self.logger.error(f"Required column '{col}' not found in data")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            return False
    
    def _extract_optimization_results(self,
                                    data: pd.DataFrame,
                                    feature_name: str,
                                    target_column: str,
                                    start_time: float) -> LookbackOptimizationResult:
        """Extract results from Optuna optimization."""
        optimization_time = time.time() - start_time
        
        # Safely get best trial
        if not self.study or not self.study.trials:
            raise ValueError("No optimization trials completed")
            
        try:
            best_trial = self.study.best_trial
        except Exception as e:
            self.logger.error(f"Failed to get best trial: {e}")
            raise ValueError("No valid best trial found")
        
        # Safely extract parameters
        first_lookback = best_trial.params.get('first_lookback')
        second_lookback = best_trial.params.get('second_lookback')
        
        if first_lookback is None or second_lookback is None:
            raise ValueError("Missing required parameters in best trial")
        
        # Calculate final scores
        first_mi_score = self._calculate_mutual_information(
            data, feature_name, target_column, first_lookback, "technical_indicator"
        )
        second_mrmr_score = self._calculate_second_lookback_mrmr_score(
            data, feature_name, target_column, second_lookback, first_lookback, "technical_indicator"
        )
        
        # Calculate correlation
        correlation = self._calculate_correlation_between_periods(
            data, feature_name, first_lookback, second_lookback, "technical_indicator"
        )
        
        # Calculate combined score with safe division
        if MATH_VALIDATION_AVAILABLE:
            combined_mi_score = safe_divide(first_mi_score + second_mrmr_score, 2.0, 0.0)
        else:
            combined_mi_score = (first_mi_score + second_mrmr_score) / 2 if (first_mi_score + second_mrmr_score) > 0 else 0.0
        
        # Get parameter importance with safe operations
        parameter_importance = {}
        if OPTUNA_AVAILABLE:
            try:
                parameter_importance = optuna.importance.get_param_importances(self.study)
            except Exception as e:
                self.logger.debug(f"Failed to get parameter importance: {e}")
                parameter_importance = {}
        
        # Safely collect all trials
        all_trials = []
        try:
            for trial in self.study.trials:
                trial_info = {
                    'params': trial.params,
                    'values': trial.values if trial.values else [],
                    'state': trial.state.name if hasattr(trial.state, 'name') else str(trial.state),
                    'user_attrs': trial.user_attrs if trial.user_attrs else {}
                }
                all_trials.append(trial_info)
        except Exception as e:
            self.logger.warning(f"Failed to collect trial information: {e}")
            all_trials = []
        
        # Calculate convergence rate
        convergence_rate = self._calculate_convergence_rate()
        
        return LookbackOptimizationResult(
            first_lookback_period=first_lookback,
            second_lookback_period=second_lookback,
            first_mi_score=first_mi_score,
            second_mi_score=second_mrmr_score,  # This is actually mRMR score
            combined_mi_score=combined_mi_score,
            second_mrmr_score=second_mrmr_score,  # Store mRMR score separately
            correlation_between_periods=correlation,
            correlation_method=self.config.correlation_method,
            optimization_time=optimization_time,
            n_trials=len(self.study.trials),
            n_successful_trials=len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            n_pruned_trials=len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            best_score=safe_list_get(best_trial.values, 0, 0.0),
            convergence_rate=convergence_rate,
            parameter_importance=parameter_importance,
            relevance_method_used=self.config.first_lookback_method,
            redundancy_method_used=self.config.second_lookback_method,
            optimization_method=self.config.optimization_method,
            config=self.config,
            all_trials=all_trials,
            convergence_history=self.convergence_history
        )
    
    def _coarse_grid_search_5x5(self,
                             data: pd.DataFrame,
                             feature_name: str,
                             target_column: str) -> Dict[str, Any]:
        """Step 1: Coarse 5x5 grid search to identify promising regions."""
        self.logger.info("🔍 Step 1: Coarse 5x5 grid search...")
        
        # Create 5x5 grid
        first_lookback_values = np.linspace(
            self.config.min_lookback, 
            self.config.max_lookback, 
            self.config.coarse_grid_size, 
            dtype=int
        )
        second_lookback_values = np.linspace(
            self.config.min_lookback, 
            self.config.max_lookback, 
            self.config.coarse_grid_size, 
            dtype=int
        )
        
        results = []
        
        # Evaluate all 5x5 = 25 combinations
        for first_lookback in first_lookback_values:
            for second_lookback in second_lookback_values:
                if first_lookback == second_lookback:
                    continue
                    
                # Calculate scores
                first_mi_score = self._calculate_mutual_information(
                    data, feature_name, target_column, first_lookback, "technical_indicator"
                )
                second_mi_score = self._calculate_mutual_information(
                    data, feature_name, target_column, second_lookback, "technical_indicator"
                )
                correlation = self._calculate_correlation_between_periods(
                    data, feature_name, first_lookback, second_lookback, "technical_indicator"
                )
                
                # Calculate combined score
                combined_score = (
                    self.config.first_lookback_weight * first_mi_score +
                    self.config.second_lookback_weight * second_mi_score -
                    self.config.correlation_weight * abs(correlation)
                )
                
                results.append({
                    'first_lookback': first_lookback,
                    'second_lookback': second_lookback,
                    'first_mi_score': first_mi_score,
                    'second_mi_score': second_mi_score,
                    'correlation': correlation,
                    'combined_score': combined_score
                })
        
        # Sort by combined score and return top candidates
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        top_candidates = results[:self.config.top_k_coarse_candidates]
        
        self.logger.info(f"📊 Coarse grid search completed: {len(results)} combinations evaluated")
        
        # Safe access to top candidate for logging
        best_candidate = safe_list_get(top_candidates, 0, None)
        if best_candidate:
            self.logger.info(f"📊 Top coarse candidate: {best_candidate['first_lookback']}, {best_candidate['second_lookback']} (score: {best_candidate['combined_score']:.4f})")
        else:
            self.logger.warning("📊 No valid coarse candidates found")
        
        return {
            'all_results': results,
            'top_candidates': top_candidates,
            'best_candidate': best_candidate
        }
    
    def _fine_grid_search_5x5(self, data: pd.DataFrame, feature_name: str, target_column: str, coarse_results: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Fine 5x5 grid search around best coarse candidates."""
        
        if not coarse_results['top_candidates']:
            raise ValueError("No coarse candidates available for fine grid search")
        
        # Calculate refined ranges around best coarse candidates
        best_coarse = coarse_results['best_candidate']
        
        # Create refined ranges (30% of original range around best candidate)
        first_range = self._calculate_refined_range(
            best_coarse['first_lookback'], 
            self.config.min_lookback, 
            self.config.max_lookback, 
            self.config.coarse_refinement_factor
        )
        second_range = self._calculate_refined_range(
            best_coarse['second_lookback'], 
            self.config.min_lookback, 
            self.config.max_lookback, 
            self.config.coarse_refinement_factor
        )
        
        # Create fine 5x5 grid
        first_lookback_values = np.linspace(first_range[0], first_range[1], self.config.fine_grid_size, dtype=int)
        second_lookback_values = np.linspace(second_range[0], second_range[1], self.config.fine_grid_size, dtype=int)
        
        results = []
        
        # Evaluate all 5x5 = 25 combinations in refined space
        for first_lookback in first_lookback_values:
            for second_lookback in second_lookback_values:
                if first_lookback == second_lookback:
                    continue
                    
                # Calculate scores (same as coarse grid)
                first_mi_score = self._calculate_mutual_information(
                    data, feature_name, target_column, first_lookback, "technical_indicator"
                )
                second_mi_score = self._calculate_mutual_information(
                    data, feature_name, target_column, second_lookback, "technical_indicator"
                )
                correlation = self._calculate_correlation_between_periods(
                    data, feature_name, first_lookback, second_lookback, "technical_indicator"
                )
                
                combined_score = (
                    self.config.first_lookback_weight * first_mi_score +
                    self.config.second_lookback_weight * second_mi_score -
                    self.config.correlation_weight * abs(correlation)
                )
                
                results.append({
                    'first_lookback': first_lookback,
                    'second_lookback': second_lookback,
                    'first_mi_score': first_mi_score,
                    'second_mi_score': second_mi_score,
                    'correlation': correlation,
                    'combined_score': combined_score
                })
        
        # Sort by combined score and return top candidates
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        top_candidates = results[:self.config.top_k_fine_candidates]
        
        self.logger.info(f"📊 Fine grid search completed: {len(results)} combinations evaluated")
        
        # Safe access to top candidate for logging
        best_candidate = safe_list_get(top_candidates, 0, None)
        if best_candidate:
            self.logger.info(f"📊 Top fine candidate: {best_candidate['first_lookback']}, {best_candidate['second_lookback']} (score: {best_candidate['combined_score']:.4f})")
        else:
            self.logger.warning("📊 No valid fine candidates found")
        
        return {
            'all_results': results,
            'top_candidates': top_candidates,
            'best_candidate': best_candidate,
            'refined_ranges': {
                'first_range': first_range,
                'second_range': second_range
            }
        }
    
    def _tpe_fine_tuning(self, data: pd.DataFrame, feature_name: str, target_column: str, fine_results: Dict[str, Any], start_time: float) -> LookbackOptimizationResult:
        """Step 3: TPE fine-tuning around best fine candidates."""
        
        if not fine_results['top_candidates']:
            raise ValueError("No fine candidates available for TPE fine-tuning")
        
        best_fine = fine_results['best_candidate']
        
        # Calculate ultra-refined ranges (20% of fine range around best candidate)
        first_range = self._calculate_refined_range(
            best_fine['first_lookback'], 
            fine_results['refined_ranges']['first_range'][0], 
            fine_results['refined_ranges']['first_range'][1], 
            self.config.fine_refinement_factor
        )
        second_range = self._calculate_refined_range(
            best_fine['second_lookback'], 
            fine_results['refined_ranges']['second_range'][0], 
            fine_results['refined_ranges']['second_range'][1], 
            self.config.fine_refinement_factor
        )
        
        # Create TPE study with refined ranges
        study = optuna.create_study(
            sampler=TPESampler(
                n_startup_trials=self.config.n_startup_trials,
                n_ei_candidates=24,
                gamma=lambda x: min(0.25, 1.0 / np.sqrt(x)),
                prior_weight=1.0,
                consider_magic_clip=True,
                consider_endpoints=True
            ),
            pruner=MedianPruner(
                n_startup_trials=self.config.n_startup_trials,
                n_warmup_steps=self.config.n_warmup_steps,
                interval_steps=self.config.interval_steps
            ),
            direction='maximize'
        )
        
        # Define objective function with refined ranges
        def objective(trial):
            first_lookback = trial.suggest_int('first_lookback', first_range[0], first_range[1])
            second_lookback = trial.suggest_int('second_lookback', second_range[0], second_range[1])
            
            if first_lookback == second_lookback:
                return float('-inf')
            
            # Calculate scores
            first_mi_score = self._calculate_mutual_information(
                data, feature_name, target_column, first_lookback, "technical_indicator"
            )
            second_mi_score = self._calculate_mutual_information(
                data, feature_name, target_column, second_lookback, "technical_indicator"
            )
            correlation = self._calculate_correlation_between_periods(
                data, feature_name, first_lookback, second_lookback, "technical_indicator"
            )
            
            # Calculate combined score
            combined_score = (
                self.config.first_lookback_weight * first_mi_score +
                self.config.second_lookback_weight * second_mi_score -
                self.config.correlation_weight * abs(correlation)
            )
            
            # Set user attributes
            trial.set_user_attr("first_lookback", first_lookback)
            trial.set_user_attr("second_lookback", second_lookback)
            trial.set_user_attr("first_mi_score", first_mi_score)
            trial.set_user_attr("second_mi_score", second_mi_score)
            trial.set_user_attr("correlation", correlation)
            
            return combined_score
        
        # Run TPE optimization
        study.optimize(objective, n_trials=self.config.tpe_trials, timeout=self.config.tpe_timeout)
        
        # Extract best result
        best_trial = study.best_trial
        best_params = best_trial.params
        
        self.logger.info(f"📊 TPE fine-tuning completed: {len(study.trials)} trials")
        self.logger.info(f"📊 Final result: {best_params['first_lookback']}, {best_params['second_lookback']} (score: {best_trial.value:.4f})")
        
        # Create result object
        result = LookbackOptimizationResult(
            first_lookback_period=best_params['first_lookback'],
            second_lookback_period=best_params['second_lookback'],
            first_mi_score=best_trial.user_attrs['first_mi_score'],
            second_mi_score=best_trial.user_attrs['second_mi_score'],
            combined_mi_score=best_trial.value,
            correlation_between_periods=best_trial.user_attrs['correlation'],
            optimization_time=time.time() - start_time,
            n_trials=len(study.trials),
            best_score=best_trial.value,
            optimization_method="two_step_grid_tpe"
        )
        
        return result
    
    def _calculate_refined_range(self, center: int, min_val: int, max_val: int, refinement_factor: float) -> Tuple[int, int]:
        """Calculate refined range around center point with validation."""
        range_size = max_val - min_val
        refined_size = max(1, range_size * refinement_factor)  # Ensure minimum size of 1
        
        new_min = max(min_val, int(center - refined_size / 2))
        new_max = min(max_val, int(center + refined_size / 2))
        
        # Ensure we have a valid range
        if new_min >= new_max:
            # Fallback to a small range around the center
            new_min = max(min_val, center - 2)
            new_max = min(max_val, center + 2)
        
        # Final validation
        if new_min >= new_max:
            new_min = min_val
            new_max = max_val
        
        return (new_min, new_max)
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate of optimization with safe division."""
        if not self.convergence_history or len(self.convergence_history) < 2:
            return 0.0
        
        # Calculate improvement rate over time
        improvements = 0
        for i in range(1, len(self.convergence_history)):
            try:
                current_score = self.convergence_history[i].get('best_score', 0)
                previous_score = self.convergence_history[i-1].get('best_score', 0)
                if current_score > previous_score:
                    improvements += 1
            except (KeyError, IndexError, TypeError):
                continue
        
        # Safe division
        total_comparisons = len(self.convergence_history) - 1
        if total_comparisons <= 0:
            return 0.0
        
        if MATH_VALIDATION_AVAILABLE:
            return safe_divide(improvements, total_comparisons, 0.0)
        else:
            return improvements / total_comparisons if total_comparisons > 0 else 0.0
    
    def _update_performance_metrics(self, result: LookbackOptimizationResult):
        """Update performance metrics."""
        self.performance_metrics.update({
            'total_trials': result.n_trials,
            'successful_trials': result.n_successful_trials,
            'pruned_trials': result.n_pruned_trials,
            'optimization_time': result.optimization_time,
            'best_score': result.best_score,
            'convergence_rate': result.convergence_rate,
            'memory_usage': get_memory_usage() if COMMON_OPERATIONS_AVAILABLE else 0.0
        })
    
    def _save_optimization_results(self, result: LookbackOptimizationResult, feature_name: str):
        """Save optimization results to file."""
        try:
            # Create results directory
            results_dir = Path(self.config.results_directory)
            results_dir.mkdir(exist_ok=True)
            
            # Save results
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{feature_name}_lookback_optimization_{timestamp}.json"
            filepath = results_dir / filename
            
            # Convert result to dictionary
            result_dict = {
                'first_lookback_period': result.first_lookback_period,
                'second_lookback_period': result.second_lookback_period,
                'first_mi_score': result.first_mi_score,
                'second_mi_score': result.second_mi_score,
                'combined_mi_score': result.combined_mi_score,
                'correlation_between_periods': result.correlation_between_periods,
                'correlation_method': result.correlation_method,
                'optimization_time': result.optimization_time,
                'n_trials': result.n_trials,
                'n_successful_trials': result.n_successful_trials,
                'n_pruned_trials': result.n_pruned_trials,
                'best_score': result.best_score,
                'convergence_rate': result.convergence_rate,
                'parameter_importance': result.parameter_importance,
                'optimization_method': result.optimization_method,
                'config': self.config.__dict__,
                'performance_metrics': self.performance_metrics
            }
            
            with open(filepath, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            self.logger.info(f"✅ Optimization results saved to {filepath}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save optimization results: {e}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization performance."""
        return {
            'performance_metrics': self.performance_metrics,
            'convergence_history': self.convergence_history,
            'parameter_importance': self.parameter_importance,
            'optimization_config': self.config.__dict__
        }

# Convenience function for easy usage
def optimize_lookback_periods(data: pd.DataFrame,
                            feature_name: str,
                            target_column: str,
                            config: Optional[LookbackOptimizationConfig] = None) -> LookbackOptimizationResult:
    """
    Convenience function to optimize lookback periods for a feature.
    
    Args:
        data: Input data with features and target
        feature_name: Name of the feature to optimize
        target_column: Name of the target column
        config: Optional configuration for optimization
        
    Returns:
        LookbackOptimizationResult with optimal lookback periods
    """
    optimizer = MRMRLookbackOptimizer(config)
    return optimizer.optimize_lookback_periods(data, feature_name, target_column)