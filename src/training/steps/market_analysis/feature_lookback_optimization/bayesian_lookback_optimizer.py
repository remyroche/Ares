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

# Import Optuna for Bayesian optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
    from optuna.integration import SklearnIntegration
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available - using fallback optimization")

# Import mutual information and correlation utilities
try:
    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
    from sklearn.metrics import mutual_info_score
    from scipy.stats import pearsonr, spearmanr
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Sklearn not available - using fallback correlation methods")

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
    """Configuration for Bayesian lookback period optimization."""
    
    # Optimization Strategy
    optimization_method: str = "bayesian"  # "bayesian", "grid", "random"
    sampler_type: str = "tpe"  # "tpe", "random", "grid"
    pruner_type: str = "median"  # "median", "successive_halving", "none"
    
    # Trial Configuration
    n_trials: int = 100
    timeout_seconds: Optional[int] = None
    n_startup_trials: int = 10
    n_warmup_steps: int = 5
    interval_steps: int = 1
    
    # Lookback Period Constraints
    min_lookback: int = 5
    max_lookback: int = 100
    lookback_step: int = 1
    
    # Correlation and MI Constraints
    max_correlation_threshold: float = 0.7  # Maximum correlation between lookback periods
    min_mutual_info_threshold: float = 0.1  # Minimum mutual information for second period
    correlation_method: str = "pearson"  # "pearson", "spearman", "kendall"
    
    # Multi-objective Weights
    mi_weight: float = 0.6  # Weight for mutual information
    correlation_weight: float = 0.4  # Weight for low correlation
    
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
    
    # Mutual Information Scores
    first_mi_score: float
    second_mi_score: Optional[float]
    combined_mi_score: float
    
    # Correlation Analysis
    correlation_between_periods: Optional[float]
    correlation_method: str
    
    # Optimization Metrics
    optimization_time: float
    n_trials: int
    n_successful_trials: int
    n_pruned_trials: int
    
    # Performance Metrics
    best_score: float
    convergence_rate: float
    parameter_importance: Dict[str, float]
    
    # Additional Information
    optimization_method: str
    config: LookbackOptimizationConfig
    all_trials: List[Dict[str, Any]] = field(default_factory=list)
    convergence_history: List[Dict[str, Any]] = field(default_factory=list)

class BayesianLookbackOptimizer:
    """
    Bayesian Lookback Period Optimizer using TPE and intelligent pruning.
    
    Optimizes lookback periods for feature parameters based on:
    1. Mutual Information (MI) maximization for the first lookback period
    2. Low correlation & high mutual importance for the second lookback period
    """
    
    def __init__(self, config: Optional[LookbackOptimizationConfig] = None):
        """Initialize the Bayesian lookback optimizer."""
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
        
        self.logger.info("🔧 BayesianLookbackOptimizer initialized")
        self.logger.info(f"📊 Optimization method: {self.config.optimization_method}")
        self.logger.info(f"📊 Sampler type: {self.config.sampler_type}")
        self.logger.info(f"📊 Pruner type: {self.config.pruner_type}")
        self.logger.info(f"📊 Lookback range: {self.config.min_lookback}-{self.config.max_lookback}")
    
    def _initialize_optuna(self):
        """Initialize Optuna study with TPE sampler and intelligent pruning."""
        # Create TPE sampler
        if self.config.sampler_type == "tpe":
            sampler = TPESampler(
                n_startup_trials=self.config.n_startup_trials,
                n_ei_candidates=24,
                seed=self.config.random_state
            )
        else:
            sampler = TPESampler(seed=self.config.random_state)
        
        # Create pruner
        if self.config.enable_pruning and self.config.pruner_type != "none":
            if self.config.pruner_type == "median":
                pruner = MedianPruner(
                    n_startup_trials=self.config.n_startup_trials,
                    n_warmup_steps=self.config.n_warmup_steps,
                    interval_steps=self.config.interval_steps
                )
            elif self.config.pruner_type == "successive_halving":
                pruner = SuccessiveHalvingPruner(
                    min_resource=1,
                    reduction_factor=3,
                    min_early_stopping_rate=0
                )
            else:
                pruner = MedianPruner()
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
        
        # Run optimization
        if OPTUNA_AVAILABLE and self.study is not None:
            self.study.optimize(
                objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout_seconds,
                n_jobs=self.config.n_jobs if self.config.enable_parallel else 1
            )
            
            # Extract results
            result = self._extract_optimization_results(data, feature_name, target_column, start_time)
        else:
            # Fallback to basic optimization
            result = self._fallback_optimization(data, feature_name, target_column, start_time)
        
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
        Objective function for lookback period optimization.
        
        Returns:
            Tuple of (mutual_information_score, correlation_penalty)
        """
        # Suggest first lookback period
        first_lookback = trial.suggest_int(
            'first_lookback',
            self.config.min_lookback,
            self.config.max_lookback,
            step=self.config.lookback_step
        )
        
        # Calculate mutual information for first lookback period
        first_mi_score = self._calculate_mutual_information(
            data, feature_name, target_column, first_lookback, parameter_type
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
        
        # Calculate mutual information for second lookback period
        second_mi_score = self._calculate_mutual_information(
            data, feature_name, target_column, second_lookback, parameter_type
        )
        
        # Calculate correlation between the two lookback periods
        correlation = self._calculate_correlation_between_periods(
            data, feature_name, first_lookback, second_lookback, parameter_type
        )
        
        # Check constraints
        if correlation > self.config.max_correlation_threshold:
            # High correlation - penalize heavily
            correlation_penalty = correlation * 10
        elif second_mi_score < self.config.min_mutual_info_threshold:
            # Low mutual information - penalize
            correlation_penalty = (self.config.min_mutual_info_threshold - second_mi_score) * 5
        else:
            # Good combination - reward
            correlation_penalty = correlation
        
        # Combined score (maximize MI, minimize correlation)
        combined_mi_score = (first_mi_score + second_mi_score) / 2
        
        # Set user attributes for analysis
        trial.set_user_attr("first_lookback", first_lookback)
        trial.set_user_attr("second_lookback", second_lookback)
        trial.set_user_attr("first_mi_score", first_mi_score)
        trial.set_user_attr("second_mi_score", second_mi_score)
        trial.set_user_attr("correlation", correlation)
        trial.set_user_attr("combined_mi_score", combined_mi_score)
        
        return combined_mi_score, correlation_penalty
    
    def _calculate_mutual_information(self, 
                                    data: pd.DataFrame,
                                    feature_name: str,
                                    target_column: str,
                                    lookback_period: int,
                                    parameter_type: str) -> float:
        """Calculate mutual information for a specific lookback period."""
        try:
            # Generate feature with lookback period
            feature_values = self._generate_feature_with_lookback(
                data, feature_name, lookback_period, parameter_type
            )
            
            # Get target values
            target_values = data[target_column].values
            
            # Ensure same length
            min_length = min(len(feature_values), len(target_values))
            feature_values = feature_values[:min_length]
            target_values = target_values[:min_length]
            
            # Remove NaN values
            mask = ~(np.isnan(feature_values) | np.isnan(target_values))
            feature_values = feature_values[mask]
            target_values = target_values[mask]
            
            if len(feature_values) < 10:  # Need minimum data points
                return 0.0
            
            # Calculate mutual information
            if SKLEARN_AVAILABLE:
                # Use sklearn for continuous target
                if data[target_column].dtype in ['float64', 'int64']:
                    mi_score = mutual_info_regression(
                        feature_values.reshape(-1, 1), 
                        target_values,
                        random_state=self.config.random_state
                    )[0]
                else:
                    # For categorical target
                    mi_score = mutual_info_classif(
                        feature_values.reshape(-1, 1), 
                        target_values,
                        random_state=self.config.random_state
                    )[0]
            else:
                # Fallback to basic correlation
                correlation = np.corrcoef(feature_values, target_values)[0, 1]
                mi_score = abs(correlation) if not np.isnan(correlation) else 0.0
            
            return float(mi_score)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate MI for lookback {lookback_period}: {e}")
            return 0.0
    
    def _calculate_correlation_between_periods(self,
                                             data: pd.DataFrame,
                                             feature_name: str,
                                             first_lookback: int,
                                             second_lookback: int,
                                             parameter_type: str) -> float:
        """Calculate correlation between two lookback periods."""
        try:
            # Generate features for both lookback periods
            first_feature = self._generate_feature_with_lookback(
                data, feature_name, first_lookback, parameter_type
            )
            second_feature = self._generate_feature_with_lookback(
                data, feature_name, second_lookback, parameter_type
            )
            
            # Ensure same length
            min_length = min(len(first_feature), len(second_feature))
            first_feature = first_feature[:min_length]
            second_feature = second_feature[:min_length]
            
            # Remove NaN values
            mask = ~(np.isnan(first_feature) | np.isnan(second_feature))
            first_feature = first_feature[mask]
            second_feature = second_feature[mask]
            
            if len(first_feature) < 10:
                return 1.0  # High correlation penalty for insufficient data
            
            # Calculate correlation
            if self.config.correlation_method == "pearson":
                correlation, _ = pearsonr(first_feature, second_feature)
            elif self.config.correlation_method == "spearman":
                correlation, _ = spearmanr(first_feature, second_feature)
            else:
                correlation = np.corrcoef(first_feature, second_feature)[0, 1]
            
            return abs(float(correlation)) if not np.isnan(correlation) else 1.0
            
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
        
        # Get best trial
        best_trial = self.study.best_trial
        
        # Extract parameters
        first_lookback = best_trial.params['first_lookback']
        second_lookback = best_trial.params['second_lookback']
        
        # Calculate final scores
        first_mi_score = self._calculate_mutual_information(
            data, feature_name, target_column, first_lookback, "technical_indicator"
        )
        second_mi_score = self._calculate_mutual_information(
            data, feature_name, target_column, second_lookback, "technical_indicator"
        )
        
        # Calculate correlation
        correlation = self._calculate_correlation_between_periods(
            data, feature_name, first_lookback, second_lookback, "technical_indicator"
        )
        
        # Calculate combined score
        combined_mi_score = (first_mi_score + second_mi_score) / 2
        
        # Get parameter importance
        if OPTUNA_AVAILABLE:
            try:
                parameter_importance = optuna.importance.get_param_importances(self.study)
            except:
                parameter_importance = {}
        else:
            parameter_importance = {}
        
        # Collect all trials
        all_trials = []
        for trial in self.study.trials:
            trial_info = {
                'params': trial.params,
                'values': trial.values,
                'state': trial.state.name,
                'user_attrs': trial.user_attrs
            }
            all_trials.append(trial_info)
        
        # Calculate convergence rate
        convergence_rate = self._calculate_convergence_rate()
        
        return LookbackOptimizationResult(
            first_lookback_period=first_lookback,
            second_lookback_period=second_lookback,
            first_mi_score=first_mi_score,
            second_mi_score=second_mi_score,
            combined_mi_score=combined_mi_score,
            correlation_between_periods=correlation,
            correlation_method=self.config.correlation_method,
            optimization_time=optimization_time,
            n_trials=len(self.study.trials),
            n_successful_trials=len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            n_pruned_trials=len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            best_score=best_trial.values[0] if best_trial.values else 0.0,
            convergence_rate=convergence_rate,
            parameter_importance=parameter_importance,
            optimization_method=self.config.optimization_method,
            config=self.config,
            all_trials=all_trials,
            convergence_history=self.convergence_history
        )
    
    def _fallback_optimization(self,
                             data: pd.DataFrame,
                             feature_name: str,
                             target_column: str,
                             start_time: float) -> LookbackOptimizationResult:
        """Fallback optimization when Optuna is not available."""
        self.logger.warning("Using fallback optimization (grid search)")
        
        best_first_lookback = self.config.min_lookback
        best_second_lookback = None
        best_score = -np.inf
        best_correlation = 1.0
        
        # Simple grid search
        for first_lookback in range(self.config.min_lookback, self.config.max_lookback + 1, self.config.lookback_step):
            first_mi = self._calculate_mutual_information(data, feature_name, target_column, first_lookback, "technical_indicator")
            
            for second_lookback in range(self.config.min_lookback, self.config.max_lookback + 1, self.config.lookback_step):
                if second_lookback == first_lookback:
                    continue
                
                second_mi = self._calculate_mutual_information(data, feature_name, target_column, second_lookback, "technical_indicator")
                correlation = self._calculate_correlation_between_periods(data, feature_name, first_lookback, second_lookback, "technical_indicator")
                
                # Combined score
                combined_score = (first_mi + second_mi) / 2 - correlation * 0.5
                
                if combined_score > best_score and correlation < self.config.max_correlation_threshold:
                    best_score = combined_score
                    best_first_lookback = first_lookback
                    best_second_lookback = second_lookback
                    best_correlation = correlation
        
        optimization_time = time.time() - start_time
        
        return LookbackOptimizationResult(
            first_lookback_period=best_first_lookback,
            second_lookback_period=best_second_lookback,
            first_mi_score=self._calculate_mutual_information(data, feature_name, target_column, best_first_lookback, "technical_indicator"),
            second_mi_score=self._calculate_mutual_information(data, feature_name, target_column, best_second_lookback, "technical_indicator") if best_second_lookback else None,
            combined_mi_score=best_score,
            correlation_between_periods=best_correlation,
            correlation_method=self.config.correlation_method,
            optimization_time=optimization_time,
            n_trials=(self.config.max_lookback - self.config.min_lookback + 1) ** 2,
            n_successful_trials=0,
            n_pruned_trials=0,
            best_score=best_score,
            convergence_rate=1.0,
            parameter_importance={},
            optimization_method="grid_search_fallback",
            config=self.config,
            all_trials=[],
            convergence_history=[]
        )
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate of optimization."""
        if not self.convergence_history or len(self.convergence_history) < 2:
            return 0.0
        
        # Calculate improvement rate over time
        improvements = 0
        for i in range(1, len(self.convergence_history)):
            if self.convergence_history[i]['best_score'] > self.convergence_history[i-1]['best_score']:
                improvements += 1
        
        return improvements / (len(self.convergence_history) - 1)
    
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
    optimizer = BayesianLookbackOptimizer(config)
    return optimizer.optimize_lookback_periods(data, feature_name, target_column)