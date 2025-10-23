"""
Data-Driven Temporal Window Size Optimization

This module provides optimization of temporal window sizes and smoothing parameters
using various strategies including Bayesian TPE, volatility adaptation, and
economic validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import logging
from dataclasses import dataclass
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings

# Import optimization utilities with BOHB for clustering optimization (Phase 3 Migration)
try:
    from src.utils.ml_common.optimization.bohb_optimizer import (
        BOHBOptimizer, BOHBConfig, BOHBResult
    )
    BOHB_AVAILABLE = True
    logging.info("✅ BOHB optimizer loaded for temporal window optimization")
except ImportError as e:
    BOHB_AVAILABLE = False
    BOHBOptimizer = None
    BOHBConfig = None
    BOHBResult = None
    logging.warning(f"BOHB optimizer not available: {e}")

# Import Bayesian TPE as fallback for simple cases
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer, OptimizationConfig
    )
    BAYESIAN_TPE_AVAILABLE = True
    logging.info("✅ Bayesian TPE optimizer loaded as fallback")
except ImportError as e:
    BAYESIAN_TPE_AVAILABLE = False
    BayesianTPEOptimizer = None
    OptimizationConfig = None
    logging.warning(f"Bayesian TPE optimizer not available: {e}")

OPTIMIZATION_AVAILABLE = BOHB_AVAILABLE or BAYESIAN_TPE_AVAILABLE

from ..config.data_driven_config import (
    TemporalWindowConfig, ValidationMetric, OptimizationStrategy
)

logger = logging.getLogger(__name__)


@dataclass
class TemporalWindowResult:
    """Result of temporal window optimization."""
    optimal_windows: Dict[str, int]
    optimization_score: float
    validation_scores: Dict[str, float]
    volatility_adaptation: Dict[str, Any]
    optimization_history: List[Dict[str, Any]]
    convergence_info: Dict[str, Any]
    metadata: Dict[str, Any]


class DataDrivenTemporalWindowOptimizer:
    """
    Data-driven optimizer for temporal window sizes and smoothing parameters.
    
    Replaces hardcoded window sizes (window_size=300, smoothing_window=5) with
    data-driven optimization based on volatility, clustering stability, and economic metrics.
    """
    
    def __init__(self, config: TemporalWindowConfig):
        """
        Initialize the temporal window optimizer.
        
        Args:
            config: Configuration for temporal window optimization
        """
        self.config = config
        self.optimization_history = []
        self.best_windows = None
        self.best_score = -np.inf
        
    def optimize_windows(self, 
                        market_data: pd.DataFrame,
                        clustering_func: Callable,
                        economic_validation_func: Optional[Callable] = None) -> TemporalWindowResult:
        """
        Optimize temporal window sizes using the specified strategy.
        
        Args:
            market_data: Market data for volatility analysis
            clustering_func: Function that performs clustering given window parameters
            economic_validation_func: Optional function for economic validation
            
        Returns:
            TemporalWindowResult with optimal window sizes and metadata
        """
        try:
            logger.info("⏰ Starting data-driven temporal window optimization...")
            
            # Calculate volatility characteristics
            volatility_info = self._analyze_volatility(market_data)
            logger.info(f"📊 Volatility analysis: {volatility_info}")
            
            # Optimize windows based on strategy
            if self.config.optimization_strategy == OptimizationStrategy.BAYESIAN_TPE:
                optimal_windows, optimization_info = self._optimize_with_tpe(
                    market_data, volatility_info, clustering_func, economic_validation_func
                )
            elif self.config.optimization_strategy == OptimizationStrategy.GRID_SEARCH:
                optimal_windows, optimization_info = self._optimize_with_grid_search(
                    market_data, volatility_info, clustering_func, economic_validation_func
                )
            elif self.config.optimization_strategy == OptimizationStrategy.RANDOM_SEARCH:
                optimal_windows, optimization_info = self._optimize_with_random_search(
                    market_data, volatility_info, clustering_func, economic_validation_func
                )
            elif self.config.optimization_strategy == OptimizationStrategy.ADAPTIVE:
                optimal_windows, optimization_info = self._optimize_adaptively(
                    market_data, volatility_info, clustering_func, economic_validation_func
                )
            else:
                raise ValueError(f"Unknown optimization strategy: {self.config.optimization_strategy}")
            
            # Validate optimal windows
            validation_scores = self._validate_windows(
                optimal_windows, market_data, clustering_func
            )
            
            # Calculate volatility adaptation info
            volatility_adaptation = self._calculate_volatility_adaptation(
                optimal_windows, volatility_info
            )
            
            # Create result
            result = TemporalWindowResult(
                optimal_windows=optimal_windows,
                optimization_score=optimization_info.get('best_score', 0.0),
                validation_scores=validation_scores,
                volatility_adaptation=volatility_adaptation,
                optimization_history=self.optimization_history,
                convergence_info=optimization_info,
                metadata={
                    'config': self.config.__dict__,
                    'n_samples': len(market_data),
                    'volatility_info': volatility_info
                }
            )
            
            logger.info(f"✅ Temporal window optimization completed. Best score: {result.optimization_score:.4f}")
            logger.info(f"📈 Optimal windows: {optimal_windows}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Temporal window optimization failed: {e}")
            raise
    
    def _analyze_volatility(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility characteristics of the market data."""
        try:
            if 'close' not in market_data.columns:
                return {'volatility_regime': 'unknown', 'avg_volatility': 0.0, 'volatility_std': 0.0}
            
            # Calculate returns
            returns = market_data['close'].pct_change().dropna()
            
            # Calculate rolling volatility
            vol_window = min(self.config.volatility_lookback, len(returns) // 4)
            rolling_vol = returns.rolling(window=vol_window).std()
            
            # Calculate volatility statistics
            avg_volatility = rolling_vol.mean()
            volatility_std = rolling_vol.std()
            
            # Determine volatility regime
            if avg_volatility > self.config.high_volatility_threshold:
                volatility_regime = 'high'
            elif avg_volatility < self.config.low_volatility_threshold:
                volatility_regime = 'low'
            else:
                volatility_regime = 'medium'
            
            # Calculate volatility persistence
            vol_autocorr = rolling_vol.autocorr(lag=1)
            
            return {
                'volatility_regime': volatility_regime,
                'avg_volatility': avg_volatility,
                'volatility_std': volatility_std,
                'volatility_autocorr': vol_autocorr,
                'volatility_percentiles': {
                    '25th': rolling_vol.quantile(0.25),
                    '50th': rolling_vol.quantile(0.50),
                    '75th': rolling_vol.quantile(0.75),
                    '90th': rolling_vol.quantile(0.90)
                }
            }
            
        except Exception as e:
            logger.warning(f"Volatility analysis failed: {e}")
            return {'volatility_regime': 'unknown', 'avg_volatility': 0.0, 'volatility_std': 0.0}
    
    def _optimize_with_tpe(self, 
                          market_data: pd.DataFrame,
                          volatility_info: Dict[str, Any],
                          clustering_func: Callable,
                          economic_validation_func: Optional[Callable]) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """Optimize windows using Bayesian TPE."""
        if not OPTIMIZATION_AVAILABLE:
            raise ImportError("Bayesian TPE optimizer not available")
        
        def objective(trial):
            # Sample window sizes
            window_size = trial.suggest_int(
                'window_size',
                self.config.window_size_range[0],
                self.config.window_size_range[1]
            )
            smoothing_window = trial.suggest_int(
                'smoothing_window',
                self.config.smoothing_window_range[0],
                self.config.smoothing_window_range[1]
            )
            
            windows = {
                'window_size': window_size,
                'smoothing_window': smoothing_window
            }
            
            # Apply windows and evaluate
            try:
                score = self._evaluate_windows(windows, market_data, volatility_info, 
                                             clustering_func, economic_validation_func)
                
                # Store trial info
                self.optimization_history.append({
                    'trial': len(self.optimization_history),
                    'windows': windows.copy(),
                    'score': score,
                    'timestamp': pd.Timestamp.now()
                })
                
                return score
                
            except Exception as e:
                logger.debug(f"Trial failed: {e}")
                return -np.inf
        
        # Run optimization with BOHB (Phase 3: Clustering Migration)
        if BOHB_AVAILABLE:
            try:
                # Create BOHB configuration for temporal window optimization
                bohb_config = BOHBConfig(
                    n_trials=self.config.n_trials,
                    timeout=self.config.timeout_seconds,
                    direction='maximize',
                    metric_name='quality_score',
                    resource_name='iteration',  # Use iterations as resource for multi-fidelity
                    min_resource=1,  # Minimum iterations
                    max_resource=3,  # Maximum iterations
                    reduction_factor=2,  # Successive halving factor
                    n_startup_trials=self.config.n_startup_trials,
                    pruner_type='hyperband',  # Use Hyperband pruning
                    enable_hardware_optimization=True,
                    enable_vectorbt_optimization=True,
                    enable_explainability=True,
                    enable_cv=True,
                    enable_oof_stacking=False,  # Not needed for temporal windows
                    seed=42
                )

                # Define multi-fidelity objective function for BOHB
                def bohb_objective(params: Dict[str, Any], resource: int = None) -> float:
                    """Multi-fidelity objective function for BOHB temporal window optimization."""
                    try:
                        # Use resource (iterations) for multi-fidelity evaluation
                        if resource and resource < 3:
                            # Limit iterations based on resource level
                            limited_windows = {
                                'window_size': int(params.get('window_size', 300)),
                                'smoothing_window': int(params.get('smoothing_window', 5))
                            }
                            return objective(limited_windows, resource)
                        else:
                            return objective(params)
                    except Exception as e:
                        logger.debug(f"BOHB objective function failed: {e}")
                        return -np.inf

                # Create and run BOHB optimizer
                optimizer = BOHBOptimizer(bohb_config)
                result = optimizer.optimize(bohb_objective, search_space)

                if result.success:
                    best_params = result.best_params
                    best_score = result.best_value
                    logger.info("✅ BOHB temporal window optimization completed successfully")
                    logger.info(f"📊 Resource efficiency: {result.resource_efficiency:.2f}x")
                else:
                    logger.warning(f"⚠️ BOHB optimization failed: {result.error_message}")
                    # Fall through to TPE fallback
                    raise Exception("BOHB optimization failed")

            except Exception as e:
                logger.warning(f"⚠️ BOHB optimization error: {e}, falling back to TPE")
                # Fall through to TPE fallback

        # Fallback to Bayesian TPE with enhanced early stopping
        if BAYESIAN_TPE_AVAILABLE:
            try:
                logger.info("🔄 Falling back to enhanced Bayesian TPE optimization...")
                
                # Create enhanced optimization configuration with aggressive early stopping
                opt_config = OptimizationConfig(
                    n_trials=self.config.n_trials,
                    timeout=self.config.timeout_seconds,
                    n_startup_trials=self.config.n_startup_trials,
                    direction='maximize',
                    metric_name='quality_score',
                    early_stopping_patience=3,  # More aggressive early stopping
                    early_stopping_threshold=0.001,  # Stricter threshold
                    enable_pruner=True,  # Enable trial-level pruning
                    pruner_type='hyperband',  # Use Hyperband pruner
                    adaptive_patience=True,  # Enable adaptive patience
                    confidence_based_stopping=True,  # Enable confidence-based stopping
                    seed=42
                )

                optimizer = BayesianTPEOptimizer(opt_config)
                best_params, best_score = optimizer.optimize(objective)
                logger.info("✅ Enhanced Bayesian TPE optimization completed successfully")

            except Exception as e:
                logger.error(f"❌ Enhanced Bayesian TPE optimization error: {e}")
                raise
        else:
            logger.error("❌ No optimizers available")
            raise RuntimeError("No optimizers available for temporal window optimization")
        
        # Extract optimal windows
        optimal_windows = {
            'window_size': int(best_params.get('window_size', 300)),
            'smoothing_window': int(best_params.get('smoothing_window', 5))
        }
        
        return optimal_windows, {'best_score': best_score, 'n_trials': len(self.optimization_history)}
    
    def _optimize_with_grid_search(self, 
                                  market_data: pd.DataFrame,
                                  volatility_info: Dict[str, Any],
                                  clustering_func: Callable,
                                  economic_validation_func: Optional[Callable]) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """Optimize windows using grid search."""
        # Create grid of window combinations
        window_values = np.linspace(
            self.config.window_size_range[0],
            self.config.window_size_range[1],
            5, dtype=int
        )
        smoothing_values = np.linspace(
            self.config.smoothing_window_range[0],
            self.config.smoothing_window_range[1],
            5, dtype=int
        )
        
        best_score = -np.inf
        best_windows = None
        
        # Generate all combinations
        for window_size, smoothing_window in itertools.product(window_values, smoothing_values):
            windows = {
                'window_size': int(window_size),
                'smoothing_window': int(smoothing_window)
            }
            
            # Apply windows and evaluate
            try:
                score = self._evaluate_windows(windows, market_data, volatility_info,
                                             clustering_func, economic_validation_func)
                
                if score > best_score:
                    best_score = score
                    best_windows = windows.copy()
                
                self.optimization_history.append({
                    'trial': len(self.optimization_history),
                    'windows': windows.copy(),
                    'score': score,
                    'timestamp': pd.Timestamp.now()
                })
                
            except Exception as e:
                logger.debug(f"Grid search trial failed: {e}")
                continue
        
        return best_windows or {
            'window_size': 300,
            'smoothing_window': 5
        }, {'best_score': best_score}
    
    def _optimize_with_random_search(self, 
                                   market_data: pd.DataFrame,
                                   volatility_info: Dict[str, Any],
                                   clustering_func: Callable,
                                   economic_validation_func: Optional[Callable]) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """Optimize windows using random search."""
        best_score = -np.inf
        best_windows = None
        
        for trial in range(self.config.n_trials):
            # Sample random windows
            windows = {
                'window_size': np.random.randint(*self.config.window_size_range),
                'smoothing_window': np.random.randint(*self.config.smoothing_window_range)
            }
            
            # Apply windows and evaluate
            try:
                score = self._evaluate_windows(windows, market_data, volatility_info,
                                             clustering_func, economic_validation_func)
                
                if score > best_score:
                    best_score = score
                    best_windows = windows.copy()
                
                self.optimization_history.append({
                    'trial': trial,
                    'windows': windows.copy(),
                    'score': score,
                    'timestamp': pd.Timestamp.now()
                })
                
            except Exception as e:
                logger.debug(f"Random search trial failed: {e}")
                continue
        
        return best_windows or {
            'window_size': 300,
            'smoothing_window': 5
        }, {'best_score': best_score}
    
    def _optimize_adaptively(self, 
                           market_data: pd.DataFrame,
                           volatility_info: Dict[str, Any],
                           clustering_func: Callable,
                           economic_validation_func: Optional[Callable]) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """Optimize windows adaptively based on volatility characteristics."""
        try:
            # Estimate optimal window size based on volatility
            volatility_regime = volatility_info.get('volatility_regime', 'medium')
            avg_volatility = volatility_info.get('avg_volatility', 0.01)
            
            if volatility_regime == 'high':
                # High volatility: use shorter windows for faster adaptation
                window_size = int(self.config.window_size_range[0] + 
                                0.3 * (self.config.window_size_range[1] - self.config.window_size_range[0]))
                smoothing_window = int(self.config.smoothing_window_range[0] + 
                                    0.2 * (self.config.smoothing_window_range[1] - self.config.smoothing_window_range[0]))
            elif volatility_regime == 'low':
                # Low volatility: use longer windows for stability
                window_size = int(self.config.window_size_range[0] + 
                                0.8 * (self.config.window_size_range[1] - self.config.window_size_range[0]))
                smoothing_window = int(self.config.smoothing_window_range[0] + 
                                    0.7 * (self.config.smoothing_window_range[1] - self.config.smoothing_window_range[0]))
            else:
                # Medium volatility: use moderate windows
                window_size = int(self.config.window_size_range[0] + 
                                0.5 * (self.config.window_size_range[1] - self.config.window_size_range[0]))
                smoothing_window = int(self.config.smoothing_window_range[0] + 
                                    0.5 * (self.config.smoothing_window_range[1] - self.config.smoothing_window_range[0]))
            
            # Apply constraints
            window_size = np.clip(window_size, self.config.min_window_size, self.config.max_window_size)
            smoothing_window = np.clip(smoothing_window, self.config.min_smoothing_window, self.config.max_smoothing_window)
            
            windows = {
                'window_size': window_size,
                'smoothing_window': smoothing_window
            }
            
            # Fine-tune with local optimization
            def objective(windows_array):
                window_size, smoothing_window = windows_array
                temp_windows = {
                    'window_size': int(window_size),
                    'smoothing_window': int(smoothing_window)
                }
                
                try:
                    score = self._evaluate_windows(temp_windows, market_data, volatility_info,
                                                 clustering_func, economic_validation_func)
                    return -score  # Minimize negative score
                except:
                    return np.inf
            
            # Initial windows
            initial_windows = np.array([window_size, smoothing_window])
            
            # Bounds
            bounds = [
                (self.config.min_window_size, self.config.max_window_size),
                (self.config.min_smoothing_window, self.config.max_smoothing_window)
            ]
            
            # Optimize
            result = minimize(objective, initial_windows, method='L-BFGS-B', bounds=bounds)
            
            optimal_windows = {
                'window_size': int(result.x[0]),
                'smoothing_window': int(result.x[1])
            }
            
            return optimal_windows, {'best_score': -result.fun, 'converged': result.success}
            
        except Exception as e:
            logger.warning(f"Adaptive optimization failed: {e}")
            return {
                'window_size': 300,
                'smoothing_window': 5
            }, {'best_score': 0.0, 'converged': False}
    
    def _evaluate_windows(self, 
                         windows: Dict[str, int],
                         market_data: pd.DataFrame,
                         volatility_info: Dict[str, Any],
                         clustering_func: Callable,
                         economic_validation_func: Optional[Callable]) -> float:
        """Evaluate window configuration."""
        try:
            # Apply windows to clustering
            cluster_labels = clustering_func(market_data, windows)
            
            # Calculate quality metrics
            quality_score = self._calculate_quality_score(cluster_labels, volatility_info)
            
            # Add economic validation if available
            if economic_validation_func and self.config.enable_economic_validation:
                try:
                    economic_score = economic_validation_func(market_data, cluster_labels, windows)
                    quality_score += self.config.economic_weight * economic_score
                except Exception as e:
                    logger.debug(f"Economic validation failed: {e}")
            
            # Add volatility adaptation bonus
            if self.config.enable_volatility_adaptation:
                adaptation_bonus = self._calculate_volatility_adaptation_bonus(windows, volatility_info)
                quality_score += adaptation_bonus
            
            return quality_score
            
        except Exception as e:
            logger.debug(f"Window evaluation failed: {e}")
            return -np.inf
    
    def _calculate_quality_score(self, cluster_labels: np.ndarray, volatility_info: Dict[str, Any]) -> float:
        """Calculate quality score for window evaluation."""
        try:
            # Remove noise points
            valid_mask = cluster_labels != -1
            if valid_mask.sum() < 2:
                return -np.inf
            
            valid_labels = cluster_labels[valid_mask]
            
            if len(set(valid_labels)) < 2:
                return -np.inf
            
            # Calculate primary metric
            if self.config.primary_metric == ValidationMetric.STABILITY_INDEX:
                primary_score = self._calculate_stability_index(valid_labels)
            elif self.config.primary_metric == ValidationMetric.SILHOUETTE:
                # For temporal windows, we focus on stability rather than silhouette
                primary_score = self._calculate_stability_index(valid_labels)
            else:
                primary_score = self._calculate_stability_index(valid_labels)
            
            # Calculate secondary metrics
            secondary_scores = []
            for metric in self.config.secondary_metrics:
                if metric == ValidationMetric.STABILITY_INDEX:
                    secondary_scores.append(self._calculate_stability_index(valid_labels))
                elif metric == ValidationMetric.SILHOUETTE:
                    # Use a simplified stability measure
                    secondary_scores.append(self._calculate_stability_index(valid_labels))
            
            # Combine scores
            combined_score = primary_score
            if secondary_scores:
                combined_score += 0.3 * np.mean(secondary_scores)
            
            return combined_score
            
        except Exception as e:
            logger.debug(f"Quality score calculation failed: {e}")
            return -np.inf
    
    def _calculate_stability_index(self, cluster_labels: np.ndarray) -> float:
        """Calculate stability index for temporal clustering."""
        try:
            # Calculate regime persistence (how often labels change)
            label_changes = np.sum(np.diff(cluster_labels) != 0)
            total_periods = len(cluster_labels) - 1
            
            if total_periods == 0:
                return 0.0
            
            # Stability = 1 - (change_rate)
            change_rate = label_changes / total_periods
            stability = 1.0 - change_rate
            
            # Add bonus for reasonable cluster count
            n_clusters = len(set(cluster_labels))
            if 2 <= n_clusters <= 8:
                stability += 0.1
            
            return np.clip(stability, 0.0, 1.0)
            
        except Exception as e:
            logger.debug(f"Stability index calculation failed: {e}")
            return 0.0
    
    def _calculate_volatility_adaptation_bonus(self, 
                                             windows: Dict[str, int],
                                             volatility_info: Dict[str, Any]) -> float:
        """Calculate bonus for volatility-adaptive window selection."""
        try:
            if not self.config.enable_volatility_adaptation:
                return 0.0
            
            volatility_regime = volatility_info.get('volatility_regime', 'medium')
            window_size = windows['window_size']
            smoothing_window = windows['smoothing_window']
            
            bonus = 0.0
            
            # High volatility: prefer shorter windows
            if volatility_regime == 'high':
                if window_size < self.config.window_size_range[0] + 0.4 * (self.config.window_size_range[1] - self.config.window_size_range[0]):
                    bonus += 0.1
                if smoothing_window < self.config.smoothing_window_range[0] + 0.3 * (self.config.smoothing_window_range[1] - self.config.smoothing_window_range[0]):
                    bonus += 0.05
            
            # Low volatility: prefer longer windows
            elif volatility_regime == 'low':
                if window_size > self.config.window_size_range[0] + 0.6 * (self.config.window_size_range[1] - self.config.window_size_range[0]):
                    bonus += 0.1
                if smoothing_window > self.config.smoothing_window_range[0] + 0.5 * (self.config.smoothing_window_range[1] - self.config.smoothing_window_range[0]):
                    bonus += 0.05
            
            return bonus
            
        except Exception as e:
            logger.debug(f"Volatility adaptation bonus calculation failed: {e}")
            return 0.0
    
    def _validate_windows(self, 
                         windows: Dict[str, int],
                         market_data: pd.DataFrame,
                         clustering_func: Callable) -> Dict[str, float]:
        """Validate optimal windows."""
        try:
            # Apply windows
            cluster_labels = clustering_func(market_data, windows)
            
            # Calculate validation metrics
            valid_mask = cluster_labels != -1
            if valid_mask.sum() < 2:
                return {'stability_index': 0.0, 'cluster_count': 0, 'change_rate': 1.0}
            
            valid_labels = cluster_labels[valid_mask]
            
            if len(set(valid_labels)) < 2:
                return {'stability_index': 0.0, 'cluster_count': 0, 'change_rate': 1.0}
            
            stability_index = self._calculate_stability_index(valid_labels)
            cluster_count = len(set(valid_labels))
            
            # Calculate change rate
            label_changes = np.sum(np.diff(valid_labels) != 0)
            total_periods = len(valid_labels) - 1
            change_rate = label_changes / max(total_periods, 1)
            
            return {
                'stability_index': stability_index,
                'cluster_count': cluster_count,
                'change_rate': change_rate
            }
            
        except Exception as e:
            logger.warning(f"Window validation failed: {e}")
            return {'stability_index': 0.0, 'cluster_count': 0, 'change_rate': 1.0}
    
    def _calculate_volatility_adaptation(self, 
                                       windows: Dict[str, int],
                                       volatility_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate volatility adaptation information."""
        try:
            volatility_regime = volatility_info.get('volatility_regime', 'unknown')
            avg_volatility = volatility_info.get('avg_volatility', 0.0)
            
            # Determine if windows are appropriate for volatility regime
            window_size = windows['window_size']
            smoothing_window = windows['smoothing_window']
            
            adaptation_score = 0.0
            recommendations = []
            
            if volatility_regime == 'high':
                if window_size > self.config.window_size_range[0] + 0.5 * (self.config.window_size_range[1] - self.config.window_size_range[0]):
                    recommendations.append("Consider shorter window for high volatility")
                else:
                    adaptation_score += 0.5
                
                if smoothing_window > self.config.smoothing_window_range[0] + 0.4 * (self.config.smoothing_window_range[1] - self.config.smoothing_window_range[0]):
                    recommendations.append("Consider shorter smoothing for high volatility")
                else:
                    adaptation_score += 0.5
            
            elif volatility_regime == 'low':
                if window_size < self.config.window_size_range[0] + 0.5 * (self.config.window_size_range[1] - self.config.window_size_range[0]):
                    recommendations.append("Consider longer window for low volatility")
                else:
                    adaptation_score += 0.5
                
                if smoothing_window < self.config.smoothing_window_range[0] + 0.4 * (self.config.smoothing_window_range[1] - self.config.smoothing_window_range[0]):
                    recommendations.append("Consider longer smoothing for low volatility")
                else:
                    adaptation_score += 0.5
            
            return {
                'volatility_regime': volatility_regime,
                'avg_volatility': avg_volatility,
                'adaptation_score': adaptation_score,
                'recommendations': recommendations,
                'windows': windows
            }
            
        except Exception as e:
            logger.warning(f"Volatility adaptation calculation failed: {e}")
            return {'volatility_regime': 'unknown', 'adaptation_score': 0.0, 'recommendations': []}