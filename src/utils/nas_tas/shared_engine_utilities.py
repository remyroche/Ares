"""
Shared Engine Utilities for NAS and TAS

This module provides shared utilities that eliminate redundancy between
NAS and TAS engines while maintaining their specialized functionality.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# Import common utilities
from ...common_operations import (
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
    format_bytes, parallel_map, chunked_iterable
)

from ...math_validation import (
    MathValidation, safe_divide as mv_safe_divide, safe_log as mv_safe_log,
    safe_sqrt as mv_safe_sqrt, safe_power as mv_safe_power,
    validate_finite as mv_validate_finite, validate_positive as mv_validate_positive,
    validate_range as mv_validate_range, safe_kelly_calculation as mv_safe_kelly_calculation,
    safe_weighted_average as mv_safe_weighted_average, safe_percentage_change as mv_safe_percentage_change,
    safe_correlation, safe_covariance, safe_mean as mv_safe_mean, safe_std as mv_safe_std,
    safe_percentile, validate_correlation_matrix, safe_matrix_inverse, math_safe
)

from ...tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_structured,
    tprint_with_level, tprint_timer, tprint_logged, configure_tprint,
    get_tprint_config, tprint_context, LogLevel
)

from ...data.klines_parquet import (
    KlinesParquetManager, get_klines_manager, read_ethusdt_data,
    save_klines_to_parquet, load_klines_from_parquet, validate_klines_data
)

from ...serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

from ...matrix_operations.unified_operations import MatrixOperations
from ...matrix_operations.enhanced_operations import EnhancedMatrixOperations
from ...matrix_operations.batch_operations import BatchMatrixOperations
from ...matrix_operations.vectorized_core import VectorizedCore

from ...hardware.m1_gpu_utils import M1GPUManager, is_m1_available, is_mps_available
from ...hardware.m1_memory_optimizer import M1MemoryOptimizer
from ...hardware.m1_cpu_optimizer import M1CPUOptimizer

# Import ML common optimization utilities
from ..ml_common.optimization.bayesian_entry_timing_optimizer import BayesianEntryTimingOptimizer
from ..ml_common.optimization.grid_utils import GridSearchOptimizer
from ..ml_common.optimization.hpo_utils import HPOUtils
from ..ml_common.optimization.hierarchical_hpo import HierarchicalHPO

logger = logging.getLogger(__name__)


class EngineType(Enum):
    """Types of engines."""
    NAS = "nas"
    TAS = "tas"
    HYBRID = "hybrid"


class SearchMethod(Enum):
    """Search methods."""
    BAYESIAN_TPE = "bayesian_tpe"
    GRID = "grid"
    HIERARCHICAL = "hierarchical"


@dataclass
class SearchResult:
    """Unified search result structure."""
    method: str
    n_trials: int
    trials: List[Dict[str, Any]]
    best_solution: Optional[Dict[str, Any]]
    best_score: float
    search_time: float
    performance_metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataLoadingConfig:
    """Configuration for data loading."""
    symbol: str = "ETHUSDT"
    interval: str = "1m"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    data_type: str = "processed"
    apply_feature_engineering: bool = True
    validate_data: bool = True
    optimize_dtypes: bool = True
    guard_nulls: bool = True


class SharedDataLoader:
    """Shared data loading utility for both NAS and TAS engines."""
    
    def __init__(self, engine_type: EngineType):
        """Initialize shared data loader.
        
        Args:
            engine_type: Type of engine (NAS/TAS)
        """
        self.engine_type = engine_type
        self.klines_manager = get_klines_manager()
        self.logger = logging.getLogger(f"SharedDataLoader_{engine_type.value}")
        
        # Initialize M1 optimizations
        self.m1_integration = integrate_with_m1_optimizers()
        if self.m1_integration['success']:
            self.gpu_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
        else:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
    
    @tprint_timer("Data Loading")
    def load_data(self, config: DataLoadingConfig) -> Optional[pd.DataFrame]:
        """Load data using shared logic.
        
        Args:
            config: Data loading configuration
            
        Returns:
            Loaded and processed DataFrame or None if loading fails
        """
        tprint_info(f"📊 Loading data for {config.symbol} {config.interval}")
        
        try:
            # Load data using klines parquet manager
            with memory_checkpoint("data_loading"):
                data = self.klines_manager.read_data(
                    symbol=config.symbol,
                    interval=config.interval,
                    start_date=config.start_date,
                    end_date=config.end_date,
                    data_type=config.data_type
                )
            
            if data is None or data.empty:
                tprint_error(f"❌ No data loaded for {config.symbol} {config.interval}")
                return None
            
            tprint_info(f"📊 Loaded {len(data)} records")
            
            # Validate data if requested
            if config.validate_data:
                tprint_debug("🔍 Validating data quality")
                validation_result = validate_klines_data(data)
                
                if not validation_result['valid']:
                    tprint_error(f"❌ Data validation failed: {validation_result['errors']}")
                    return None
                
                # Apply data quality metrics
                quality_metrics = calculate_data_quality_metrics(data)
                tprint_info(f"📈 Data quality metrics: {quality_metrics}")
            
            # Apply engine-specific processing
            if self.engine_type == EngineType.TAS and config.apply_feature_engineering:
                data = self._apply_tas_processing(data)
            elif self.engine_type == EngineType.NAS:
                data = self._apply_nas_processing(data)
            
            # Optimize data types if requested
            if config.optimize_dtypes:
                tprint_debug("🔧 Optimizing data types")
                data = optimize_dataframe_dtypes(data)
            
            # Guard against null values if requested
            if config.guard_nulls:
                data = guard_dataframe_nulls(data, threshold=0.1)
            
            tprint_success(f"✅ Data loaded and processed: {len(data)} records")
            return data
            
        except Exception as e:
            tprint_error(f"❌ Error loading data: {e}")
            self.logger.exception("Data loading error")
            return None
    
    def _apply_tas_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply TAS-specific data processing."""
        # This would include feature engineering, returns calculation, etc.
        # For now, return data as-is
        return data
    
    def _apply_nas_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply NAS-specific data processing."""
        # This would include neural network specific preprocessing
        # For now, return data as-is
        return data


class SharedSearchFramework:
    """Shared search framework for both NAS and TAS engines."""
    
    def __init__(self, engine_type: EngineType):
        """Initialize shared search framework.
        
        Args:
            engine_type: Type of engine (NAS/TAS)
        """
        self.engine_type = engine_type
        self.logger = logging.getLogger(f"SharedSearchFramework_{engine_type.value}")
        
        # Initialize optimization components
        self.bayesian_optimizer = BayesianEntryTimingOptimizer()
        self.grid_optimizer = GridSearchOptimizer()
        self.hpo_utils = HPOUtils()
        self.hierarchical_hpo = HierarchicalHPO()
        
        # Initialize M1 optimizations
        self.m1_integration = integrate_with_m1_optimizers()
        if self.m1_integration['success']:
            self.gpu_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
        else:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
    
    @tprint_timer("Search Execution")
    def execute_search(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        optimization_method: str = "bayesian_tpe",
        n_trials: int = 100,
        evaluation_function: Callable = None,
        additional_params: Dict[str, Any] = None
    ) -> SearchResult:
        """Execute search using shared framework.
        
        Args:
            data: Input data for search
            search_space: Search space definition
            optimization_method: Optimization method to use
            n_trials: Number of trials to run
            evaluation_function: Function to evaluate solutions
            additional_params: Additional parameters for evaluation
            
        Returns:
            Search result with best solution and metrics
        """
        tprint_info(f"🔍 Starting {self.engine_type.value} search with {optimization_method}")
        
        try:
            # Validate input data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not validate_dataframe_columns(data, required_columns):
                tprint_error(f"❌ Invalid data columns for {self.engine_type.value} search")
                return self._create_empty_result(optimization_method, n_trials)
            
            # Initialize search results
            search_results = SearchResult(
                method=optimization_method,
                n_trials=n_trials,
                trials=[],
                best_solution=None,
                best_score=-np.inf,
                search_time=0,
                performance_metrics={}
            )
            
            start_time = time.time()
            
            # Use M1 GPU context if available
            with gpu_context(f"{self.engine_type.value}_search") if self.gpu_manager else memory_checkpoint(f"{self.engine_type.value}_search"):
                
                if optimization_method == SearchMethod.BAYESIAN_TPE.value:
                    tprint_info("🧠 Using Bayesian TPE optimization")
                    best_solution, best_score, trials = self._bayesian_search(
                        data, search_space, n_trials, evaluation_function, additional_params
                    )
                elif optimization_method == SearchMethod.GRID.value:
                    tprint_info("🔧 Using Grid Search optimization")
                    best_solution, best_score, trials = self._grid_search(
                        data, search_space, n_trials, evaluation_function, additional_params
                    )
                elif optimization_method == SearchMethod.HIERARCHICAL.value:
                    tprint_info("🏗️ Using Hierarchical HPO optimization")
                    best_solution, best_score, trials = self._hierarchical_search(
                        data, search_space, n_trials, evaluation_function, additional_params
                    )
                else:
                    tprint_error(f"❌ Unknown optimization method: {optimization_method}")
                    return self._create_empty_result(optimization_method, n_trials)
                
                search_results.best_solution = best_solution
                search_results.best_score = best_score
                search_results.trials = trials
            
            search_time = time.time() - start_time
            search_results.search_time = search_time
            
            # Calculate performance metrics
            search_results.performance_metrics = self._calculate_search_metrics(trials)
            
            tprint_success(f"✅ {self.engine_type.value} search completed in {search_time:.2f}s")
            tprint_info(f"🏆 Best score: {best_score:.4f}")
            
            return search_results
            
        except Exception as e:
            tprint_error(f"❌ Error in {self.engine_type.value} search: {e}")
            self.logger.exception("Search error")
            return self._create_empty_result(optimization_method, n_trials)
    
    def _bayesian_search(
        self, 
        data: pd.DataFrame, 
        search_space: Dict[str, Any], 
        n_trials: int,
        evaluation_function: Callable,
        additional_params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """Perform Bayesian TPE search."""
        tprint_debug(f"🧠 Starting Bayesian TPE {self.engine_type.value} search")
        
        trials = []
        best_score = -np.inf
        best_solution = None
        
        try:
            # Configure Bayesian optimizer
            self.bayesian_optimizer.configure(
                search_space=search_space,
                n_trials=n_trials,
                random_state=42
            )
            
            for trial_idx in range(n_trials):
                tprint_progress(trial_idx, n_trials, f"Bayesian TPE trial {trial_idx}")
                
                # Get next trial parameters
                trial_params = self.bayesian_optimizer.suggest()
                
                # Evaluate solution
                with tprint_timer(f"Trial {trial_idx} evaluation"):
                    score = evaluation_function(data, trial_params, additional_params)
                
                # Record trial
                trial_result = {
                    'trial_idx': trial_idx,
                    'params': trial_params,
                    'score': score,
                    'timestamp': time.time()
                }
                trials.append(trial_result)
                
                # Update best if improved
                if score > best_score:
                    best_score = score
                    best_solution = trial_params.copy()
                    tprint_info(f"🏆 New best score: {best_score:.4f} at trial {trial_idx}")
                
                # Update optimizer
                self.bayesian_optimizer.update(trial_params, score)
                
                # Memory optimization
                if trial_idx % 10 == 0:
                    optimize_memory()
            
            tprint_success(f"✅ Bayesian {self.engine_type.value} search completed: {len(trials)} trials")
            return best_solution, best_score, trials
            
        except Exception as e:
            tprint_error(f"❌ Error in Bayesian {self.engine_type.value} search: {e}")
            return {}, -np.inf, []
    
    def _grid_search(
        self, 
        data: pd.DataFrame, 
        search_space: Dict[str, Any], 
        n_trials: int,
        evaluation_function: Callable,
        additional_params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """Perform Grid Search."""
        tprint_debug(f"🔧 Starting Grid Search {self.engine_type.value} search")
        
        trials = []
        best_score = -np.inf
        best_solution = None
        
        try:
            # Generate grid parameters
            grid_params = self.grid_optimizer.generate_grid(search_space, max_trials=n_trials)
            
            total_trials = len(grid_params)
            tprint_info(f"🔧 Grid search: {total_trials} parameter combinations")
            
            for trial_idx, params in enumerate(grid_params):
                tprint_progress(trial_idx, total_trials, f"Grid search trial {trial_idx}")
                
                # Evaluate solution
                with tprint_timer(f"Grid trial {trial_idx} evaluation"):
                    score = evaluation_function(data, params, additional_params)
                
                # Record trial
                trial_result = {
                    'trial_idx': trial_idx,
                    'params': params,
                    'score': score,
                    'timestamp': time.time()
                }
                trials.append(trial_result)
                
                # Update best if improved
                if score > best_score:
                    best_score = score
                    best_solution = params.copy()
                    tprint_info(f"🏆 New best score: {best_score:.4f} at trial {trial_idx}")
                
                # Memory optimization
                if trial_idx % 10 == 0:
                    optimize_memory()
            
            tprint_success(f"✅ Grid {self.engine_type.value} search completed: {len(trials)} trials")
            return best_solution, best_score, trials
            
        except Exception as e:
            tprint_error(f"❌ Error in Grid {self.engine_type.value} search: {e}")
            return {}, -np.inf, []
    
    def _hierarchical_search(
        self, 
        data: pd.DataFrame, 
        search_space: Dict[str, Any], 
        n_trials: int,
        evaluation_function: Callable,
        additional_params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """Perform Hierarchical HPO search."""
        tprint_debug(f"🏗️ Starting Hierarchical HPO {self.engine_type.value} search")
        
        trials = []
        best_score = -np.inf
        best_solution = None
        
        try:
            # Configure hierarchical HPO
            self.hierarchical_hpo.configure(
                search_space=search_space,
                n_trials=n_trials,
                hierarchy_levels=3
            )
            
            for trial_idx in range(n_trials):
                tprint_progress(trial_idx, n_trials, f"Hierarchical HPO trial {trial_idx}")
                
                # Get next trial parameters
                trial_params = self.hierarchical_hpo.suggest()
                
                # Evaluate solution
                with tprint_timer(f"Hierarchical trial {trial_idx} evaluation"):
                    score = evaluation_function(data, trial_params, additional_params)
                
                # Record trial
                trial_result = {
                    'trial_idx': trial_idx,
                    'params': trial_params,
                    'score': score,
                    'timestamp': time.time()
                }
                trials.append(trial_result)
                
                # Update best if improved
                if score > best_score:
                    best_score = score
                    best_solution = trial_params.copy()
                    tprint_info(f"🏆 New best score: {best_score:.4f} at trial {trial_idx}")
                
                # Update hierarchical HPO
                self.hierarchical_hpo.update(trial_params, score)
                
                # Memory optimization
                if trial_idx % 10 == 0:
                    optimize_memory()
            
            tprint_success(f"✅ Hierarchical {self.engine_type.value} search completed: {len(trials)} trials")
            return best_solution, best_score, trials
            
        except Exception as e:
            tprint_error(f"❌ Error in Hierarchical {self.engine_type.value} search: {e}")
            return {}, -np.inf, []
    
    def _calculate_search_metrics(self, trials: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate search performance metrics."""
        try:
            if not trials:
                return {}
            
            # Extract scores
            scores = [trial['score'] for trial in trials]
            scores_array = np.array(scores)
            
            # Calculate metrics using math validation utilities
            metrics = {
                'mean_score': safe_mean(scores_array),
                'std_score': safe_std(scores_array),
                'max_score': np.max(scores_array),
                'min_score': np.min(scores_array),
                'median_score': safe_percentile(scores_array, 50.0),
                'q25_score': safe_percentile(scores_array, 25.0),
                'q75_score': safe_percentile(scores_array, 75.0),
                'improvement_rate': self._calculate_improvement_rate(scores),
                'convergence_metric': self._calculate_convergence_metric(scores)
            }
            
            return metrics
            
        except Exception as e:
            tprint_error(f"❌ Error calculating search metrics: {e}")
            return {}
    
    def _calculate_improvement_rate(self, scores: List[float]) -> float:
        """Calculate improvement rate."""
        try:
            if len(scores) < 2:
                return 0.0
            
            improvements = 0
            for i in range(1, len(scores)):
                if scores[i] > scores[i-1]:
                    improvements += 1
            
            return safe_divide(improvements, len(scores) - 1)
            
        except Exception:
            return 0.0
    
    def _calculate_convergence_metric(self, scores: List[float]) -> float:
        """Calculate convergence metric."""
        try:
            if len(scores) < 10:
                return 0.0
            
            # Use last 20% of trials for convergence analysis
            last_portion = max(1, len(scores) // 5)
            recent_scores = scores[-last_portion:]
            
            # Calculate coefficient of variation
            mean_score = safe_mean(np.array(recent_scores))
            std_score = safe_std(np.array(recent_scores))
            
            if mean_score == 0:
                return 0.0
            
            cv = safe_divide(std_score, abs(mean_score))
            return 1.0 - cv  # Lower CV means better convergence
            
        except Exception:
            return 0.0
    
    def _create_empty_result(self, method: str, n_trials: int) -> SearchResult:
        """Create empty search result."""
        return SearchResult(
            method=method,
            n_trials=n_trials,
            trials=[],
            best_solution=None,
            best_score=-np.inf,
            search_time=0,
            performance_metrics={}
        )


class SharedFeatureMatrixBuilder:
    """Shared feature matrix builder for both NAS and TAS engines."""
    
    def __init__(self, engine_type: EngineType):
        """Initialize shared feature matrix builder.
        
        Args:
            engine_type: Type of engine (NAS/TAS)
        """
        self.engine_type = engine_type
        self.logger = logging.getLogger(f"SharedFeatureMatrixBuilder_{engine_type.value}")
        
        # Initialize matrix operations
        self.matrix_ops = MatrixOperations()
        self.enhanced_matrix_ops = EnhancedMatrixOperations()
        self.batch_matrix_ops = BatchMatrixOperations()
        self.vectorized_core = VectorizedCore()
    
    def create_feature_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """Create feature matrix using shared logic.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Feature matrix as numpy array
        """
        try:
            # Extract numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            feature_data = data[numeric_cols].values
            
            # Handle NaN values
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply engine-specific feature engineering
            if self.engine_type == EngineType.NAS:
                feature_matrix = self._create_nas_features(feature_data)
            elif self.engine_type == EngineType.TAS:
                feature_matrix = self._create_tas_features(feature_data)
            else:
                feature_matrix = self._create_basic_features(feature_data)
            
            return feature_matrix
            
        except Exception as e:
            tprint_error(f"❌ Error creating feature matrix: {e}")
            return np.array([])
    
    def _create_nas_features(self, feature_data: np.ndarray) -> np.ndarray:
        """Create NAS-specific features."""
        # Normalize features
        normalized_features = self.matrix_ops.normalize_matrix(feature_data)
        
        # Add polynomial features
        polynomial_features = self.enhanced_matrix_ops.add_polynomial_features(
            normalized_features, degree=2
        )
        
        return polynomial_features
    
    def _create_tas_features(self, feature_data: np.ndarray) -> np.ndarray:
        """Create TAS-specific features."""
        # Normalize features
        normalized_features = self.matrix_ops.normalize_matrix(feature_data)
        
        # Add technical indicator features
        technical_features = self.enhanced_matrix_ops.add_technical_features(
            normalized_features
        )
        
        return technical_features
    
    def _create_basic_features(self, feature_data: np.ndarray) -> np.ndarray:
        """Create basic features."""
        # Normalize features
        normalized_features = self.matrix_ops.normalize_matrix(feature_data)
        
        return normalized_features


class SharedEvaluationFramework:
    """Shared evaluation framework for both NAS and TAS engines."""
    
    def __init__(self, engine_type: EngineType):
        """Initialize shared evaluation framework.
        
        Args:
            engine_type: Type of engine (NAS/TAS)
        """
        self.engine_type = engine_type
        self.logger = logging.getLogger(f"SharedEvaluationFramework_{engine_type.value}")
        
        # Initialize matrix operations
        self.matrix_ops = MatrixOperations()
        self.enhanced_matrix_ops = EnhancedMatrixOperations()
        self.vectorized_core = VectorizedCore()
        
        # Initialize M1 optimizations
        self.m1_integration = integrate_with_m1_optimizers()
        if self.m1_integration['success']:
            self.gpu_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
        else:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
    
    @tprint_timer("Solution Evaluation")
    def evaluate_solution(
        self,
        data: pd.DataFrame,
        solution_params: Dict[str, Any],
        additional_params: Dict[str, Any] = None
    ) -> float:
        """Evaluate solution using shared framework.
        
        Args:
            data: Input data for evaluation
            solution_params: Solution parameters to evaluate
            additional_params: Additional parameters for evaluation
            
        Returns:
            Solution performance score
        """
        try:
            # Validate solution parameters
            validated_params = {}
            for param, value in solution_params.items():
                try:
                    if isinstance(value, (int, float)):
                        validated_value = validate_finite(value, param)
                        validated_params[param] = validated_value
                    else:
                        validated_params[param] = value
                except ValueError as e:
                    tprint_warning(f"⚠️ Invalid parameter {param}: {e}")
                    continue
            
            # Prepare data for evaluation
            with memory_checkpoint("data_preparation"):
                # Create feature matrix using matrix operations
                feature_matrix = self._create_feature_matrix(data)
                
                # Validate feature matrix
                if not validate_correlation_matrix(feature_matrix):
                    tprint_warning("⚠️ Invalid feature matrix correlation structure")
                    return 0.0
            
            # Simulate solution evaluation
            with gpu_context("solution_evaluation") if self.gpu_manager else memory_checkpoint("solution_evaluation"):
                # Use matrix operations for evaluation
                score = self._compute_solution_score(feature_matrix, validated_params, additional_params)
            
            # Validate score
            score = validate_finite(score, "solution_score")
            
            tprint_debug(f"🔍 Solution evaluation score: {score:.4f}")
            return score
            
        except Exception as e:
            tprint_error(f"❌ Error evaluating solution: {e}")
            return 0.0
    
    def _create_feature_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """Create feature matrix for evaluation."""
        try:
            # Extract numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            feature_data = data[numeric_cols].values
            
            # Handle NaN values
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Use matrix operations for feature engineering
            normalized_features = self.matrix_ops.normalize_matrix(feature_data)
            
            return normalized_features
            
        except Exception as e:
            tprint_error(f"❌ Error creating feature matrix: {e}")
            return np.array([])
    
    def _compute_solution_score(
        self, 
        feature_matrix: np.ndarray, 
        params: Dict[str, Any],
        additional_params: Dict[str, Any] = None
    ) -> float:
        """Compute solution score using matrix operations."""
        try:
            # Extract key parameters
            if self.engine_type == EngineType.NAS:
                complexity = params.get('complexity', 1.0)
                depth = params.get('depth', 1)
                width = params.get('width', 1)
                
                # Compute base score using matrix operations
                base_score = self.vectorized_core.compute_performance_metric(
                    feature_matrix, complexity, depth, width
                )
                
                # Apply parameter-based adjustments
                complexity_factor = safe_power(complexity, 0.5)
                depth_factor = safe_log(depth + 1)
                width_factor = safe_sqrt(width)
                
                # Combine factors
                adjusted_score = safe_weighted_average(
                    [base_score, complexity_factor, depth_factor, width_factor],
                    [0.7, 0.1, 0.1, 0.1]
                )
                
            elif self.engine_type == EngineType.TAS:
                entry_threshold = params.get('entry_threshold', 0.5)
                exit_threshold = params.get('exit_threshold', 0.5)
                risk_factor = params.get('risk_factor', 1.0)
                position_size = params.get('position_size', 0.1)
                
                # Compute base score using matrix operations
                base_score = self.vectorized_core.compute_strategy_performance(
                    feature_matrix, entry_threshold, exit_threshold
                )
                
                # Apply parameter-based adjustments
                risk_adjustment = safe_power(risk_factor, 0.5)
                position_adjustment = safe_sqrt(position_size)
                
                # Combine factors
                adjusted_score = safe_weighted_average(
                    [base_score, risk_adjustment, position_adjustment],
                    [0.8, 0.1, 0.1]
                )
            
            else:
                # Generic scoring
                adjusted_score = np.mean(feature_matrix)
            
            return adjusted_score
            
        except Exception as e:
            tprint_error(f"❌ Error computing solution score: {e}")
            return 0.0


# Factory functions for creating shared utilities
def create_shared_data_loader(engine_type: EngineType) -> SharedDataLoader:
    """Create a shared data loader for the specified engine type."""
    return SharedDataLoader(engine_type)


def create_shared_search_framework(engine_type: EngineType) -> SharedSearchFramework:
    """Create a shared search framework for the specified engine type."""
    return SharedSearchFramework(engine_type)


def create_shared_feature_matrix_builder(engine_type: EngineType) -> SharedFeatureMatrixBuilder:
    """Create a shared feature matrix builder for the specified engine type."""
    return SharedFeatureMatrixBuilder(engine_type)


def create_shared_evaluation_framework(engine_type: EngineType) -> SharedEvaluationFramework:
    """Create a shared evaluation framework for the specified engine type."""
    return SharedEvaluationFramework(engine_type)


# Convenience function for creating all shared utilities
def create_shared_utilities(engine_type: EngineType) -> Dict[str, Any]:
    """Create all shared utilities for the specified engine type.
    
    Args:
        engine_type: Type of engine (NAS/TAS)
        
    Returns:
        Dictionary containing all shared utilities
    """
    return {
        'data_loader': create_shared_data_loader(engine_type),
        'search_framework': create_shared_search_framework(engine_type),
        'feature_matrix_builder': create_shared_feature_matrix_builder(engine_type),
        'evaluation_framework': create_shared_evaluation_framework(engine_type)
    }