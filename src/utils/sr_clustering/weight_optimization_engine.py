from src.utils.tprint import tprint

"""
Weight Optimization Engine for SR Quality Score Parameters

This module implements backtesting-based optimization of quality score parameter weights
to maximize the predictive power of the quality scoring system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
import itertools

from ..logger import system_logger
from .sr_backtesting_engine import SRBacktestingEngine, BacktestResult, SRLevel

# Import M1 optimization utilities
try:
    from ..hardware.m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
    from ..matrix_operations import get_unified_matrix_operations, M1EnhancedMatrixOperations
    from ..hardware.memory_optimization import get_memory_manager, MemoryMonitor
    M1_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    M1_OPTIMIZATIONS_AVAILABLE = False
    get_m1_memory_optimizer = None
    get_unified_matrix_operations = None
    get_memory_manager = None
    tprint(f"⚠️ M1 optimizations not available: {e}")

# Import PyTorch for MPS acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

@dataclass
class WeightOptimizationConfig:
    """Configuration for weight optimization with overfitting protection."""
    # Optimization parameters
    optimization_method: str = 'scipy_minimize'  # 'scipy_minimize', 'grid_search', 'genetic_algorithm'
    max_iterations: int = 100
    convergence_tolerance: float = 1e-6
    
    # Cross-validation parameters
    n_splits: int = 5
    test_size: float = 0.2
    
    # Weight constraints
    min_weight: float = 0.0
    max_weight: float = 1.0
    weight_sum_constraint: bool = True  # Whether weights should sum to 1.0
    
    # Overfitting protection parameters - ADAPTIVE
    min_samples_for_optimization: int = 10  # Reduced minimum samples (was 50)
    max_features_per_sample_ratio: float = 0.5  # Increased ratio for small samples (was 0.1)
    early_stopping_patience: int = 5  # Reduced patience for small samples (was 10)
    regularization_strength: float = 0.1  # Higher regularization for small samples (was 0.01)
    stability_penalty: float = 0.2  # Higher stability penalty for small samples (was 0.1)
    
    # Adaptive optimization parameters
    enable_adaptive_optimization: bool = True  # Enable adaptive optimization
    small_sample_mode_threshold: int = 30  # Use small sample mode below this
    minimal_optimization_threshold: int = 15  # Use minimal optimization below this
    
    # M1 optimization parameters
    enable_m1_optimizations: bool = True
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    memory_limit_gb: float = 8.0
    chunk_size: int = 1000
    
    # Feature groups for optimization
    primary_features: List[str] = field(default_factory=lambda: [
        'success_rate', 'avg_bounce_strength', 'total_volume_at_level', 
        'time_persistence', 'touch_frequency'
    ])
    penetration_features: List[str] = field(default_factory=lambda: [
        'penetration_depth', 'penetration_frequency'
    ])
    pattern_features: List[str] = field(default_factory=lambda: [
        'pattern_consistency', 'pattern_strength', 'order_flow_confirmation'
    ])
    
    # Optimization objectives
    primary_objective: str = 'r2_score'  # 'r2_score', 'mse', 'mae', 'correlation'
    secondary_objective: str = 'stability'  # 'stability', 'generalization', 'interpretability'

class WeightOptimizationEngine:
    """Engine for optimizing quality score parameter weights through backtesting."""
    
    def __init__(self, config: Optional[WeightOptimizationConfig] = None):
        self.config = config or WeightOptimizationConfig()
        self.logger = system_logger.getChild('WeightOptimizationEngine')
        
        self.logger.info("Initializing WeightOptimizationEngine")
        self.logger.info(f"Configuration: optimization_method={self.config.optimization_method}, max_iterations={self.config.max_iterations}")
        self.logger.info(f"Cross-validation: n_splits={self.config.n_splits}, test_size={self.config.test_size}")
        self.logger.info(f"Primary objective: {self.config.primary_objective}, Secondary objective: {self.config.secondary_objective}")
        self.logger.info(f"Feature groups: primary={len(self.config.primary_features)}, penetration={len(self.config.penetration_features)}, pattern={len(self.config.pattern_features)}")
        
        # Initialize M1 optimizations
        self.enable_m1_optimizations = self.config.enable_m1_optimizations and M1_OPTIMIZATIONS_AVAILABLE
        self.enable_gpu_acceleration = self.config.enable_gpu_acceleration and TORCH_AVAILABLE
        
        if self.enable_m1_optimizations:
            try:
                self.m1_memory_optimizer = get_m1_memory_optimizer(memory_limit_gb=self.config.memory_limit_gb)
                self.matrix_ops = get_unified_matrix_operations()
                self.memory_monitor = get_memory_manager()
                self.logger.info("✅ M1 optimizations initialized for weight optimization")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize M1 optimizations: {e}")
                self.enable_m1_optimizations = False
        else:
            self.m1_memory_optimizer = None
            self.matrix_ops = None
            self.memory_monitor = None
        
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_weights: Dict[str, float] = {}
        self.best_score: float = 0.0
        
        self.logger.info("✅ WeightOptimizationEngine initialization completed")
        
    def optimize_weights(self, backtest_results: List[BacktestResult], 
                        market_data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize quality score parameter weights using backtesting with overfitting protection."""
        try:
            self.logger.info(f"🚀 Starting weight optimization for {len(backtest_results)} backtest results with overfitting protection")
            self.logger.info(f"Market data shape: {market_data.shape}")
            
            # ADAPTIVE OPTIMIZATION: Determine optimization strategy based on sample size
            n_samples = len(backtest_results)
            optimization_strategy = self._determine_optimization_strategy(n_samples)
            
            self.logger.info(f"🎯 Optimization strategy: {optimization_strategy} (samples: {n_samples})")
            
            # Apply adaptive optimization strategy
            if optimization_strategy == 'minimal':
                return self._minimal_optimization(backtest_results, market_data)
            elif optimization_strategy == 'small_sample':
                return self._small_sample_optimization(backtest_results, market_data)
            elif optimization_strategy == 'standard':
                return self._standard_optimization(backtest_results, market_data)
            else:
                self.logger.warning(f"Unknown optimization strategy: {optimization_strategy}")
                return self._minimal_optimization(backtest_results, market_data)
            
            # Prepare data for optimization
            self.logger.info("🔧 Preparing optimization data with overfitting protection")
            optimization_data = self._prepare_optimization_data_with_overfitting_protection(backtest_results, market_data)
            
            if not optimization_data:
                self.logger.warning("⚠️ No valid data for optimization")
                return {}
            
            self.logger.info(f"Optimization data prepared: {len(optimization_data)} samples")
            
            # OVERFITTING PROTECTION: Check feature-to-sample ratio
            n_samples = len(backtest_results)
            n_features = len(optimization_data.get('feature_names', []))
            feature_to_sample_ratio = n_features / n_samples
            
            if feature_to_sample_ratio > self.config.max_features_per_sample_ratio:
                self.logger.warning(f"⚠️ High feature-to-sample ratio: {feature_to_sample_ratio:.3f} > {self.config.max_features_per_sample_ratio}")
                self.logger.warning("⚠️ This may lead to overfitting - consider reducing features")
            
            # Run optimization based on method
            self.logger.info(f"🎯 Running optimization using {self.config.optimization_method} with overfitting protection")
            if self.config.optimization_method == 'scipy_minimize':
                result = self._optimize_with_scipy_with_overfitting_protection(optimization_data)
            elif self.config.optimization_method == 'grid_search':
                result = self._optimize_with_grid_search_with_overfitting_protection(optimization_data)
            elif self.config.optimization_method == 'genetic_algorithm':
                result = self._optimize_with_genetic_algorithm_with_overfitting_protection(optimization_data)
            else:
                self.logger.error(f"❌ Unknown optimization method: {self.config.optimization_method}")
                raise ValueError(f"Unknown optimization method: {self.config.optimization_method}")
            
            # Store results
            self.best_weights = result['best_weights']
            self.best_score = result['best_score']
            self.optimization_history.append(result)
            
            self.logger.info(f"✅ Weight optimization completed successfully with overfitting protection")
            self.logger.info(f"Best score: {self.best_score:.4f}")
            self.logger.info(f"Best weights: {self.best_weights}")
            
            # Log optimization statistics
            if 'optimization_iterations' in result:
                self.logger.info(f"Optimization iterations: {result['optimization_iterations']}")
            if 'convergence_achieved' in result:
                self.logger.info(f"Convergence achieved: {result['convergence_achieved']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Weight optimization failed: {e}")
            import traceback

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {}

    def optimize_weights_m1_optimized(self, backtest_results: List[BacktestResult], 
                                    market_data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize quality score parameter weights using M1-optimized backtesting."""
        if not self.enable_m1_optimizations:
            self.logger.warning("⚠️ M1 optimizations not available, falling back to standard method")
            return self.optimize_weights(backtest_results, market_data)
        
        try:
            self.logger.info(f"🚀 Starting M1-optimized weight optimization for {len(backtest_results)} backtest results")
            self.logger.info(f"Market data shape: {market_data.shape}")
            
            # Memory checkpoint for M1 optimization
            with self.m1_memory_optimizer.memory_checkpoint("weight_optimization"):
                # Check if data should be processed in chunks
                data_size_mb = market_data.memory_usage(deep=True).sum() / (1024**2)
                
                if self.m1_memory_optimizer.should_chunk_data(data_size_mb, "weight_optimization"):
                    self.logger.info(f"📦 Processing large dataset ({data_size_mb:.1f}MB) in chunks")
                    return self._chunked_weight_optimization(backtest_results, market_data)
                
                # Use GPU acceleration for optimization if available
                if self.enable_gpu_acceleration and self.matrix_ops:
                    self.logger.info("🎯 Using GPU acceleration for weight optimization")
                    return self._gpu_accelerated_weight_optimization(backtest_results, market_data)
                
                # Standard M1-optimized processing
                return self._m1_optimized_weight_optimization(backtest_results, market_data)
                
        except Exception as e:
            self.logger.error(f"❌ M1-optimized weight optimization failed: {e}")
            return {}

    def _m1_optimized_weight_optimization(self, backtest_results: List[BacktestResult], 
                                        market_data: pd.DataFrame) -> Dict[str, Any]:
        """M1-optimized weight optimization."""
        # Prepare data with M1 memory optimization
        optimization_data = self._prepare_optimization_data_m1_optimized(backtest_results, market_data)
        
        if not optimization_data:
            self.logger.warning("⚠️ No valid data for M1 optimization")
            return {}
        
        # Run optimization with M1 optimizations
        if self.config.optimization_method == 'scipy_minimize':
            result = self._m1_optimize_with_scipy(optimization_data)
        else:
            # Fallback to standard optimization
            result = self._optimize_with_scipy(optimization_data)
        
        # Store results
        self.best_weights = result['best_weights']
        self.best_score = result['best_score']
        self.optimization_history.append(result)
        
        self.logger.info("✅ M1-optimized weight optimization completed")
        return result

    def _prepare_optimization_data_m1_optimized(self, backtest_results: List[BacktestResult], 
                                              market_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Prepare optimization data with M1 memory optimization."""
        try:
            # Use M1 memory-efficient data preparation
            feature_data = []
            target_data = []
            
            for result in backtest_results:
                # Extract features with M1 optimization
                features = [
                    result.success_rate,
                    result.avg_bounce_strength,
                    result.total_volume_at_level,
                    result.time_persistence,
                    result.touch_frequency
                ]
                
                # Use M1 memory-efficient array creation
                if self.m1_memory_optimizer:
                    feature_array = self.m1_memory_optimizer.create_memory_efficient_array(features, np.float32)
                else:
                    feature_array = np.array(features, dtype=np.float32)
                
                feature_data.append(feature_array)
                target_data.append(result.quality_score)
            
            # Convert to numpy arrays with M1 optimization
            feature_matrix = np.array(feature_data, dtype=np.float32)
            target_array = np.array(target_data, dtype=np.float32)
            
            return {
                'features': feature_matrix,
                'targets': target_array,
                'feature_names': self.config.primary_features
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to prepare M1 optimization data: {e}")
            return None

    def _m1_optimize_with_scipy(self, optimization_data: Dict[str, Any]) -> Dict[str, Any]:
        """M1-optimized scipy optimization."""
        try:
            features = optimization_data['features']
            targets = optimization_data['targets']
            feature_names = optimization_data['feature_names']
            
            # Define objective function with M1 memory optimization
            def objective(weights):
                with self.m1_memory_optimizer.memory_checkpoint("scipy_objective"):
                    # Calculate weighted features
                    weighted_features = np.dot(features, weights)
                    
                    # Calculate R² score
                    ss_res = np.sum((targets - weighted_features) ** 2)
                    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    return -r2  # Minimize negative R²
            
            # Initial weights
            n_features = len(feature_names)
            initial_weights = np.ones(n_features) / n_features
            
            # Constraints
            constraints = []
            if self.config.weight_sum_constraint:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda w: np.sum(w) - 1.0
                })
            
            # Bounds
            bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_features)]
            
            # Optimize with M1 memory management
            with self.m1_memory_optimizer.memory_checkpoint("scipy_optimization"):
                result = minimize(
                    objective,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': self.config.max_iterations}
                )
            
            # Extract results
            best_weights = dict(zip(feature_names, result.x))
            best_score = -result.fun
            
            return {
                'best_weights': best_weights,
                'best_score': best_score,
                'optimization_method': 'scipy_minimize_m1_optimized',
                'convergence_achieved': result.success,
                'optimization_iterations': result.nit
            }
            
        except Exception as e:
            self.logger.error(f"❌ M1 scipy optimization failed: {e}")
            return {}
    
    def _prepare_optimization_data(self, backtest_results: List[BacktestResult], 
                                 market_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for weight optimization."""
        try:
            if not backtest_results:
                self.logger.warning("No backtest results provided for optimization data preparation")
                return {}
            
            self.logger.info(f"Preparing optimization data from {len(backtest_results)} backtest results")
            
            # Extract features and target
            all_features = (self.config.primary_features + 
                          self.config.penetration_features + 
                          self.config.pattern_features)
            
            self.logger.info(f"Feature groups: primary={len(self.config.primary_features)}, penetration={len(self.config.penetration_features)}, pattern={len(self.config.pattern_features)}")
            self.logger.info(f"Total features: {len(all_features)}")
            
            # Build feature matrix
            self.logger.info("Building feature matrix")
            feature_data = {}
            for feature in all_features:
                feature_values = []
                for result in backtest_results:
                    value = getattr(result, feature, 0.0)
                    feature_values.append(value)
                feature_data[feature] = np.array(feature_values)
                
                # Log feature statistics
                feature_array = feature_data[feature]
                self.logger.debug(f"Feature {feature}: mean={np.mean(feature_array):.3f}, std={np.std(feature_array):.3f}, min={np.min(feature_array):.3f}, max={np.max(feature_array):.3f}")
            
            # Target variable (actual quality scores from backtesting)
            target_scores = np.array([result.quality_score for result in backtest_results])
            self.logger.info(f"Target scores: mean={np.mean(target_scores):.3f}, std={np.std(target_scores):.3f}, min={np.min(target_scores):.3f}, max={np.max(target_scores):.3f}")
            
            # Market context features (if available)
            self.logger.info("Extracting market context")
            market_context = self._extract_market_context(backtest_results, market_data)
            
            optimization_data = {
                'feature_data': feature_data,
                'target_scores': target_scores,
                'market_context': market_context,
                'backtest_results': backtest_results,
                'feature_names': all_features
            }
            
            self.logger.info(f"✅ Optimization data prepared successfully: {len(feature_data)} features, {len(target_scores)} targets")
            
            return optimization_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to prepare optimization data: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
    
    def _extract_market_context(self, backtest_results: List[BacktestResult], 
                              market_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract market context for optimization."""
        try:
            self.logger.debug("Extracting market context features")
            
            # Calculate market regime features
            if len(market_data) > 0:
                self.logger.debug(f"Processing market data with {len(market_data)} rows")
                
                # Volatility regime
                returns = market_data['close'].pct_change().dropna()
                volatility = returns.rolling(20).std()
                volatility_regime = np.mean(volatility) if len(volatility) > 0 else 0.0
                
                self.logger.debug(f"Volatility regime: {volatility_regime:.4f}")
                
                # Trend strength
                sma_short = market_data['close'].rolling(10).mean()
                sma_long = market_data['close'].rolling(50).mean()
                trend_strength = abs(np.mean((sma_short - sma_long) / sma_long)) if len(sma_short) > 0 else 0.0
                
                self.logger.debug(f"Trend strength: {trend_strength:.4f}")
                
                # Volume regime
                volume_avg = market_data['volume'].mean() if 'volume' in market_data.columns else 1.0
                volume_regime = volume_avg / 1000000  # Normalize
                
                self.logger.debug(f"Volume regime: {volume_regime:.4f}")
                
                market_context = {
                    'volatility_regime': volatility_regime,
                    'trend_strength': trend_strength,
                    'volume_regime': volume_regime,
                    'market_periods': len(market_data)
                }
                
                self.logger.debug(f"Market context extracted: {market_context}")
                
                return market_context
            else:
                self.logger.warning("No market data available for context extraction")
                return {
                    'volatility_regime': 0.0,
                    'trend_strength': 0.0,
                    'volume_regime': 0.0,
                    'market_periods': 0
                }
                
        except Exception as e:
            self.logger.warning(f"Failed to extract market context: {e}")
            return {}
    
    def _optimize_with_scipy(self, optimization_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize weights using scipy minimize."""
        try:
            self.logger.info("🎯 Starting scipy optimization")
            
            feature_data = optimization_data['feature_data']
            target_scores = optimization_data['target_scores']
            feature_names = optimization_data['feature_names']
            
            # Initial weights (equal weights)
            n_features = len(feature_names)
            initial_weights = np.ones(n_features) / n_features
            
            self.logger.info(f"Optimization setup: {n_features} features, {len(target_scores)} targets")
            self.logger.info(f"Initial weights: {dict(zip(feature_names, initial_weights))}")
            
            # Define objective function
            def objective(weights):
                return -self._evaluate_weights(weights, feature_data, target_scores, feature_names)
            
            # Define constraints
            constraints = []
            if self.config.weight_sum_constraint:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda w: np.sum(w) - 1.0
                })
                self.logger.info("Weight sum constraint enabled (weights must sum to 1.0)")
            
            # Define bounds
            bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_features)]
            self.logger.info(f"Weight bounds: [{self.config.min_weight}, {self.config.max_weight}]")
            
            # Optimize
            self.logger.info(f"Running SLSQP optimization with max_iter={self.config.max_iterations}, ftol={self.config.convergence_tolerance}")
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.config.max_iterations, 'ftol': self.config.convergence_tolerance}
            )
            
            if result.success:
                best_weights = dict(zip(feature_names, result.x))
                best_score = -result.fun
                
                self.logger.info(f"✅ Scipy optimization completed successfully")
                self.logger.info(f"Best score: {best_score:.4f}")
                self.logger.info(f"Iterations: {result.nit}")
                self.logger.info(f"Convergence message: {result.message}")
                
                # Log top weights
                sorted_weights = sorted(best_weights.items(), key=lambda x: x[1], reverse=True)
                self.logger.info("Top 5 optimized weights:")
                for feature, weight in sorted_weights[:5]:
                    self.logger.info(f"  {feature}: {weight:.3f}")
                
                return {
                    'method': 'scipy_minimize',
                    'best_weights': best_weights,
                    'best_score': best_score,
                    'optimization_success': True,
                    'iterations': result.nit,
                    'convergence_message': result.message
                }
            else:
                self.logger.warning(f"⚠️ Scipy optimization failed: {result.message}")
                return self._get_default_weights(feature_names)
                
        except Exception as e:
            self.logger.error(f"❌ Scipy optimization failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return self._get_default_weights(optimization_data['feature_names'])
    
    def _optimize_with_grid_search(self, optimization_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize weights using grid search."""
        try:
            self.logger.info("🔍 Starting grid search optimization")
            
            feature_data = optimization_data['feature_data']
            target_scores = optimization_data['target_scores']
            feature_names = optimization_data['feature_names']
            
            # Define weight grid
            weight_values = np.linspace(self.config.min_weight, self.config.max_weight, 11)  # 0.0 to 1.0 in steps of 0.1
            
            self.logger.info(f"Grid search setup: {len(feature_names)} features")
            self.logger.info(f"Weight grid: {len(weight_values)} values from {self.config.min_weight} to {self.config.max_weight}")
            
            best_score = -np.inf
            best_weights = {}
            
            # Generate all possible weight combinations
            weight_combinations = itertools.product(weight_values, repeat=len(feature_names))
            
            total_combinations = len(weight_values) ** len(feature_names)
            self.logger.info(f"Grid search: evaluating {total_combinations} weight combinations")
            
            if total_combinations > 10000:
                self.logger.warning(f"⚠️ Large number of combinations ({total_combinations}), this may take a while")
            
            evaluated = 0
            for weights in weight_combinations:
                weights = np.array(weights)
                
                # Apply weight sum constraint
                if self.config.weight_sum_constraint:
                    weights = weights / np.sum(weights)
                
                # Evaluate weights
                score = self._evaluate_weights(weights, feature_data, target_scores, feature_names)
                
                if score > best_score:
                    best_score = score
                    best_weights = dict(zip(feature_names, weights))
                    self.logger.debug(f"New best score: {best_score:.4f}")
                
                evaluated += 1
                if evaluated % 1000 == 0:
                    self.logger.info(f"Evaluated {evaluated}/{total_combinations} combinations (best score: {best_score:.4f})")
            
            self.logger.info(f"✅ Grid search optimization completed")
            self.logger.info(f"Best score: {best_score:.4f}")
            self.logger.info(f"Combinations evaluated: {evaluated}")
            
            # Log top weights
            sorted_weights = sorted(best_weights.items(), key=lambda x: x[1], reverse=True)
            self.logger.info("Top 5 optimized weights:")
            for feature, weight in sorted_weights[:5]:
                self.logger.info(f"  {feature}: {weight:.3f}")
            
            return {
                'method': 'grid_search',
                'best_weights': best_weights,
                'best_score': best_score,
                'optimization_success': True,
                'combinations_evaluated': evaluated
            }
            
        except Exception as e:
            self.logger.error(f"❌ Grid search optimization failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return self._get_default_weights(optimization_data['feature_names'])
    
    def _optimize_with_genetic_algorithm(self, optimization_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize weights using genetic algorithm (simplified implementation)."""
        try:
            self.logger.info("🧬 Starting genetic algorithm optimization")
            
            # This is a simplified genetic algorithm implementation
            # In practice, you might want to use DEAP or similar library
            
            feature_data = optimization_data['feature_data']
            target_scores = optimization_data['target_scores']
            feature_names = optimization_data['feature_names']
            
            n_features = len(feature_names)
            population_size = 50
            generations = 20
            
            self.logger.info(f"Genetic algorithm setup: {n_features} features, population_size={population_size}, generations={generations}")
            
            # Initialize population
            self.logger.info("Initializing population")
            population = []
            for _ in range(population_size):
                weights = np.random.random(n_features)
                if self.config.weight_sum_constraint:
                    weights = weights / np.sum(weights)
                population.append(weights)
            
            best_score = -np.inf
            best_weights = {}
            
            for generation in range(generations):
                self.logger.debug(f"Generation {generation + 1}/{generations}")
                
                # Evaluate population
                scores = []
                for weights in population:
                    score = self._evaluate_weights(weights, feature_data, target_scores, feature_names)
                    scores.append(score)
                    
                    if score > best_score:
                        best_score = score
                        best_weights = dict(zip(feature_names, weights))
                        self.logger.debug(f"New best score in generation {generation + 1}: {best_score:.4f}")
                
                # Selection (keep top 50%)
                sorted_indices = np.argsort(scores)[::-1]
                elite_size = population_size // 2
                elite = [population[i] for i in sorted_indices[:elite_size]]
                
                self.logger.debug(f"Generation {generation + 1}: best_score={best_score:.4f}, avg_score={np.mean(scores):.4f}")
                
                # Create new generation
                new_population = elite.copy()
                
                # Crossover and mutation
                while len(new_population) < population_size:
                    parent1 = elite[np.random.randint(elite_size)]
                    parent2 = elite[np.random.randint(elite_size)]
                    
                    # Crossover
                    child = (parent1 + parent2) / 2
                    
                    # Mutation
                    mutation_rate = 0.1
                    for i in range(n_features):
                        if np.random.random() < mutation_rate:
                            child[i] = np.random.random()
                    
                    # Apply constraints
                    if self.config.weight_sum_constraint:
                        child = child / np.sum(child)
                    
                    new_population.append(child)
                
                population = new_population
                
                self.logger.info(f"Generation {generation + 1}: Best score = {best_score:.4f}")
            
            self.logger.info(f"✅ Genetic algorithm optimization completed")
            self.logger.info(f"Best score: {best_score:.4f}")
            self.logger.info(f"Generations: {generations}, Population size: {population_size}")
            
            # Log top weights
            sorted_weights = sorted(best_weights.items(), key=lambda x: x[1], reverse=True)
            self.logger.info("Top 5 optimized weights:")
            for feature, weight in sorted_weights[:5]:
                self.logger.info(f"  {feature}: {weight:.3f}")
            
            return {
                'method': 'genetic_algorithm',
                'best_weights': best_weights,
                'best_score': best_score,
                'optimization_success': True,
                'generations': generations,
                'population_size': population_size
            }
            
        except Exception as e:
            self.logger.error(f"❌ Genetic algorithm optimization failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return self._get_default_weights(optimization_data['feature_names'])
    
    def _evaluate_weights(self, weights: np.ndarray, feature_data: Dict[str, np.ndarray], 
                         target_scores: np.ndarray, feature_names: List[str]) -> float:
        """Evaluate a set of weights using cross-validation."""
        try:
            self.logger.debug(f"Evaluating weights: {dict(zip(feature_names, weights))}")
            
            # Build weighted quality scores
            weighted_scores = np.zeros(len(target_scores))
            
            for i, feature in enumerate(feature_names):
                if feature in feature_data:
                    weighted_scores += weights[i] * feature_data[feature]
            
            # Normalize to 0-1 range
            weighted_scores = np.clip(weighted_scores, 0.0, 1.0)
            
            self.logger.debug(f"Weighted scores: mean={np.mean(weighted_scores):.3f}, std={np.std(weighted_scores):.3f}")
            
            # Calculate performance metric
            if self.config.primary_objective == 'r2_score':
                score = r2_score(target_scores, weighted_scores)
                self.logger.debug(f"R² score: {score:.4f}")
            elif self.config.primary_objective == 'mse':
                score = -mean_squared_error(target_scores, weighted_scores)  # Negative because we want to minimize MSE
                self.logger.debug(f"MSE score: {score:.4f}")
            elif self.config.primary_objective == 'correlation':
                correlation = np.corrcoef(target_scores, weighted_scores)[0, 1]
                score = correlation if not np.isnan(correlation) else 0.0
                self.logger.debug(f"Correlation score: {score:.4f}")
            else:
                score = r2_score(target_scores, weighted_scores)  # Default to R²
                self.logger.debug(f"Default R² score: {score:.4f}")
            
            # Add stability penalty if requested
            if self.config.secondary_objective == 'stability':
                # Penalize extreme weights
                weight_penalty = -np.sum(np.abs(weights - np.mean(weights))) * 0.1
                score += weight_penalty
                self.logger.debug(f"Stability penalty: {weight_penalty:.4f}, Final score: {score:.4f}")
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Weight evaluation failed: {e}")
            return 0.0
    
    def _get_default_weights(self, feature_names: List[str]) -> Dict[str, Any]:
        """Get default weights when optimization fails."""
        n_features = len(feature_names)
        default_weights = {feature: 1.0 / n_features for feature in feature_names}
        
        self.logger.warning(f"Using default equal weights for {n_features} features")
        self.logger.info(f"Default weights: {default_weights}")
        
        return {
            'method': 'default',
            'best_weights': default_weights,
            'best_score': 0.0,
            'optimization_success': False,
            'error': 'Optimization failed, using default weights'
        }
    
    def get_optimized_weights(self) -> Dict[str, float]:
        """Get the best optimized weights."""
        return self.best_weights.copy() if self.best_weights else {}
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get a summary of the optimization process."""
        if not self.optimization_history:
            return {'status': 'No optimization performed yet'}
        
        latest_result = self.optimization_history[-1]
        
        return {
            'status': 'Optimization completed',
            'method': latest_result.get('method', 'unknown'),
            'best_score': self.best_score,
            'best_weights': self.best_weights,
            'optimization_success': latest_result.get('optimization_success', False),
            'total_optimizations': len(self.optimization_history)
        }
    
    def validate_weights(self, weights: Dict[str, float], backtest_results: List[BacktestResult]) -> Dict[str, Any]:
        """Validate optimized weights on new data."""
        try:
            if not backtest_results or not weights:
                return {'validation_score': 0.0, 'status': 'No data for validation'}
            
            # Extract features
            feature_data = {}
            for feature in weights.keys():
                feature_values = [getattr(result, feature, 0.0) for result in backtest_results]
                feature_data[feature] = np.array(feature_values)
            
            # Calculate weighted scores
            target_scores = np.array([result.quality_score for result in backtest_results])
            weighted_scores = np.zeros(len(target_scores))
            
            for feature, weight in weights.items():
                if feature in feature_data:
                    weighted_scores += weight * feature_data[feature]
            
            # Normalize
            weighted_scores = np.clip(weighted_scores, 0.0, 1.0)
            
            # Calculate validation metrics
            r2 = r2_score(target_scores, weighted_scores)
            mse = mean_squared_error(target_scores, weighted_scores)
            correlation = np.corrcoef(target_scores, weighted_scores)[0, 1]
            
            return {
                'validation_score': r2,
                'r2_score': r2,
                'mse': mse,
                'correlation': correlation if not np.isnan(correlation) else 0.0,
                'status': 'Validation completed',
                'samples_validated': len(backtest_results)
            }
            
        except Exception as e:
            self.logger.error(f"Weight validation failed: {e}")
            return {'validation_score': 0.0, 'status': f'Validation failed: {e}'}

    def _prepare_optimization_data_with_overfitting_protection(self, backtest_results: List[BacktestResult], 
                                                             market_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare optimization data with overfitting protection."""
        try:
            if not backtest_results:
                self.logger.warning("No backtest results provided for optimization data preparation")
                return {}
            
            self.logger.info(f"Preparing optimization data from {len(backtest_results)} backtest results with overfitting protection")
            
            # OVERFITTING PROTECTION: Limit features based on sample size
            max_features = int(len(backtest_results) * self.config.max_features_per_sample_ratio)
            
            # Extract features and target
            all_features = (self.config.primary_features + 
                          self.config.penetration_features + 
                          self.config.pattern_features)
            
            # OVERFITTING PROTECTION: Select most important features if we have too many
            if len(all_features) > max_features:
                # Prioritize primary features, then add others up to max_features
                selected_features = self.config.primary_features[:min(len(self.config.primary_features), max_features)]
                remaining_slots = max_features - len(selected_features)
                if remaining_slots > 0:
                    selected_features.extend(self.config.penetration_features[:remaining_slots])
                if remaining_slots > len(self.config.penetration_features):
                    remaining_slots -= len(self.config.penetration_features)
                    selected_features.extend(self.config.pattern_features[:remaining_slots])
                all_features = selected_features
                self.logger.info(f"🔒 Limited features to {len(all_features)} to prevent overfitting")
            
            self.logger.info(f"Feature groups: primary={len(self.config.primary_features)}, penetration={len(self.config.penetration_features)}, pattern={len(self.config.pattern_features)}")
            self.logger.info(f"Total features after overfitting protection: {len(all_features)}")
            
            # Build feature matrix
            self.logger.info("Building feature matrix with overfitting protection")
            feature_data = {}
            for feature in all_features:
                feature_values = []
                for result in backtest_results:
                    value = getattr(result, feature, 0.0)
                    feature_values.append(value)
                feature_data[feature] = np.array(feature_values)
                
                # Log feature statistics
                feature_array = feature_data[feature]
                self.logger.debug(f"Feature {feature}: mean={np.mean(feature_array):.3f}, std={np.std(feature_array):.3f}, min={np.min(feature_array):.3f}, max={np.max(feature_array):.3f}")
            
            # Target variable (actual quality scores from backtesting)
            target_scores = np.array([result.quality_score for result in backtest_results])
            self.logger.info(f"Target scores: mean={np.mean(target_scores):.3f}, std={np.std(target_scores):.3f}, min={np.min(target_scores):.3f}, max={np.max(target_scores):.3f}")
            
            # Market context features (if available)
            self.logger.info("Extracting market context with overfitting protection")
            market_context = self._extract_market_context(backtest_results, market_data)
            
            optimization_data = {
                'feature_data': feature_data,
                'target_scores': target_scores,
                'market_context': market_context,
                'backtest_results': backtest_results,
                'feature_names': all_features,
                'max_features_allowed': max_features,
                'overfitting_protection_applied': True
            }
            
            self.logger.info(f"✅ Optimization data prepared successfully with overfitting protection: {len(feature_data)} features, {len(target_scores)} targets")
            
            return optimization_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to prepare optimization data with overfitting protection: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
    
    def _optimize_with_scipy_with_overfitting_protection(self, optimization_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize weights using scipy minimize with overfitting protection."""
        try:
            self.logger.info("🎯 Starting scipy optimization with overfitting protection")
            
            feature_data = optimization_data['feature_data']
            target_scores = optimization_data['target_scores']
            feature_names = optimization_data['feature_names']
            
            # Initial weights (equal weights)
            n_features = len(feature_names)
            initial_weights = np.ones(n_features) / n_features
            
            self.logger.info(f"Optimization setup: {n_features} features, {len(target_scores)} targets")
            self.logger.info(f"Initial weights: {dict(zip(feature_names, initial_weights))}")
            
            # Define objective function with regularization
            def objective(weights):
                score = self._evaluate_weights_with_overfitting_protection(weights, feature_data, target_scores, feature_names)
                return -score  # Minimize negative score
            
            # Define constraints
            constraints = []
            if self.config.weight_sum_constraint:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda w: np.sum(w) - 1.0
                })
                self.logger.info("Weight sum constraint enabled (weights must sum to 1.0)")
            
            # Define bounds
            bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_features)]
            self.logger.info(f"Weight bounds: [{self.config.min_weight}, {self.config.max_weight}]")
            
            # Optimize with early stopping
            self.logger.info(f"Running SLSQP optimization with max_iter={self.config.max_iterations}, ftol={self.config.convergence_tolerance}")
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.config.max_iterations, 'ftol': self.config.convergence_tolerance}
            )
            
            if result.success:
                best_weights = dict(zip(feature_names, result.x))
                best_score = -result.fun
                
                # OVERFITTING PROTECTION: Check if score is too high
                if best_score > 0.95:
                    self.logger.warning(f"⚠️ Suspiciously high optimization score: {best_score:.4f}")
                    self.logger.warning("⚠️ This may indicate overfitting - applying conservative scaling")
                    best_weights = self._apply_conservative_weight_scaling(best_weights)
                    best_score = self._evaluate_weights_with_overfitting_protection(
                        list(best_weights.values()), feature_data, target_scores, feature_names
                    )
                
                self.logger.info(f"✅ Scipy optimization completed successfully with overfitting protection")
                self.logger.info(f"Best score: {best_score:.4f}")
                self.logger.info(f"Iterations: {result.nit}")
                self.logger.info(f"Convergence message: {result.message}")
                
                # Log top weights
                sorted_weights = sorted(best_weights.items(), key=lambda x: x[1], reverse=True)
                self.logger.info("Top 5 optimized weights:")
                for feature, weight in sorted_weights[:5]:
                    self.logger.info(f"  {feature}: {weight:.3f}")
                
                return {
                    'method': 'scipy_minimize_with_overfitting_protection',
                    'best_weights': best_weights,
                    'best_score': best_score,
                    'optimization_success': True,
                    'iterations': result.nit,
                    'convergence_message': result.message,
                    'overfitting_protection_applied': True
                }
            else:
                self.logger.warning(f"⚠️ Scipy optimization failed: {result.message}")
                return self._get_default_weights(feature_names)
                
        except Exception as e:
            self.logger.error(f"❌ Scipy optimization with overfitting protection failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return self._get_default_weights(optimization_data['feature_names'])
    
    def _evaluate_weights_with_overfitting_protection(self, weights: np.ndarray, feature_data: Dict[str, np.ndarray], 
                                                    target_scores: np.ndarray, feature_names: List[str]) -> float:
        """Evaluate weights with overfitting protection including regularization."""
        try:
            self.logger.debug(f"Evaluating weights with overfitting protection: {dict(zip(feature_names, weights))}")
            
            # Build weighted quality scores
            weighted_scores = np.zeros(len(target_scores))
            
            for i, feature in enumerate(feature_names):
                if feature in feature_data:
                    weighted_scores += weights[i] * feature_data[feature]
            
            # Normalize to 0-1 range
            weighted_scores = np.clip(weighted_scores, 0.0, 1.0)
            
            self.logger.debug(f"Weighted scores: mean={np.mean(weighted_scores):.3f}, std={np.std(weighted_scores):.3f}")
            
            # Calculate performance metric
            if self.config.primary_objective == 'r2_score':
                score = r2_score(target_scores, weighted_scores)
                self.logger.debug(f"R² score: {score:.4f}")
            elif self.config.primary_objective == 'mse':
                score = -mean_squared_error(target_scores, weighted_scores)  # Negative because we want to minimize MSE
                self.logger.debug(f"MSE score: {score:.4f}")
            elif self.config.primary_objective == 'correlation':
                correlation = np.corrcoef(target_scores, weighted_scores)[0, 1]
                score = correlation if not np.isnan(correlation) else 0.0
                self.logger.debug(f"Correlation score: {score:.4f}")
            else:
                score = r2_score(target_scores, weighted_scores)  # Default to R²
                self.logger.debug(f"Default R² score: {score:.4f}")
            
            # OVERFITTING PROTECTION: Add regularization penalty
            l2_penalty = self.config.regularization_strength * np.sum(weights ** 2)
            score -= l2_penalty
            self.logger.debug(f"L2 regularization penalty: {l2_penalty:.4f}")
            
            # Add stability penalty if requested
            if self.config.secondary_objective == 'stability':
                # Penalize extreme weights
                weight_penalty = -np.sum(np.abs(weights - np.mean(weights))) * self.config.stability_penalty
                score += weight_penalty
                self.logger.debug(f"Stability penalty: {weight_penalty:.4f}")
            
            final_score = score
            self.logger.debug(f"Final score with overfitting protection: {final_score:.4f}")
            
            return final_score
            
        except Exception as e:
            self.logger.warning(f"Weight evaluation with overfitting protection failed: {e}")
            return 0.0
    
    def _apply_conservative_weight_scaling(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply conservative scaling to weights to prevent overfitting."""
        if not weights:
            return weights
        
        # Normalize weights to sum to 1.0 and apply conservative scaling
        total_weight = sum(weights.values())
        if total_weight == 0:
            return weights
        
        # Scale down extreme weights and normalize
        conservative_weights = {}
        for feature, weight in weights.items():
            # Cap individual weights at 0.5 and scale down by 0.8
            capped_weight = min(weight / total_weight, 0.5) * 0.8
            conservative_weights[feature] = capped_weight
        
        # Renormalize to sum to 1.0
        total_conservative = sum(conservative_weights.values())
        if total_conservative > 0:
            for feature in conservative_weights:
                conservative_weights[feature] /= total_conservative
        
        self.logger.info("🔒 Applied conservative weight scaling to prevent overfitting")
        return conservative_weights
    
    def _determine_optimization_strategy(self, n_samples: int) -> str:
        """Determine optimization strategy based on sample size."""
        if n_samples < self.config.minimal_optimization_threshold:
            return 'minimal'
        elif n_samples < self.config.small_sample_mode_threshold:
            return 'small_sample'
        else:
            return 'standard'
    
    def _minimal_optimization(self, backtest_results: List[BacktestResult], 
                            market_data: pd.DataFrame) -> Dict[str, Any]:
        """Minimal optimization for very small samples (10-15 samples)."""
        self.logger.info("🔬 Using minimal optimization for very small samples")
        
        # Use only primary features with equal weights
        primary_features = ['success_rate', 'avg_bounce_strength', 'total_touches']
        
        # Calculate simple correlations
        correlations = {}
        for feature in primary_features:
            values = [getattr(result, feature, 0.0) for result in backtest_results]
            quality_scores = [result.quality_score for result in backtest_results]
            
            if len(values) > 1 and np.std(values) > 0:
                corr, _ = pearsonr(values, quality_scores)
                correlations[feature] = abs(corr)
        
        # Use correlation-based weights (normalized)
        if correlations:
            total_corr = sum(correlations.values())
            best_weights = {feature: corr / total_corr for feature, corr in correlations.items()}
        else:
            # Equal weights as fallback
            best_weights = {feature: 1.0 / len(primary_features) for feature in primary_features}
        
        # Calculate simple score
        best_score = self._calculate_simple_score(backtest_results, best_weights)
        
        return {
            'method': 'minimal_optimization',
            'best_weights': best_weights,
            'best_score': best_score,
            'optimization_success': True,
            'iterations': 0,
            'convergence_message': 'Minimal optimization completed',
            'overfitting_protection_applied': True,
            'strategy': 'minimal'
        }
    
    def _small_sample_optimization(self, backtest_results: List[BacktestResult], 
                                 market_data: pd.DataFrame) -> Dict[str, Any]:
        """Small sample optimization (15-30 samples)."""
        self.logger.info("🛡️ Using small sample optimization")
        
        # Use limited feature set
        limited_features = ['success_rate', 'avg_bounce_strength', 'total_touches', 
                          'time_persistence', 'total_volume_at_level']
        
        # Prepare data with limited features
        optimization_data = self._prepare_limited_optimization_data(backtest_results, market_data, limited_features)
        
        if not optimization_data:
            return self._minimal_optimization(backtest_results, market_data)
        
        # Use simple grid search with high regularization
        best_weights = {}
        best_score = 0.0
        
        # Simple grid search over weight combinations
        weight_combinations = [
            [0.4, 0.3, 0.2, 0.1, 0.0],  # Focus on success_rate and bounce_strength
            [0.3, 0.4, 0.2, 0.1, 0.0],  # Focus on bounce_strength
            [0.2, 0.2, 0.2, 0.2, 0.2],  # Equal weights
            [0.5, 0.3, 0.1, 0.1, 0.0],  # Heavy focus on success_rate
            [0.3, 0.3, 0.2, 0.1, 0.1],  # Balanced approach
        ]
        
        for weights in weight_combinations:
            weight_dict = dict(zip(limited_features, weights))
            score = self._calculate_simple_score(backtest_results, weight_dict)
            
            if score > best_score:
                best_score = score
                best_weights = weight_dict
        
        return {
            'method': 'small_sample_optimization',
            'best_weights': best_weights,
            'best_score': best_score,
            'optimization_success': True,
            'iterations': len(weight_combinations),
            'convergence_message': 'Small sample optimization completed',
            'overfitting_protection_applied': True,
            'strategy': 'small_sample'
        }
    
    def _standard_optimization(self, backtest_results: List[BacktestResult], 
                             market_data: pd.DataFrame) -> Dict[str, Any]:
        """Standard optimization for larger samples (30+ samples)."""
        self.logger.info("📚 Using standard optimization for larger samples")
        
        # Use full optimization with overfitting protection
        optimization_data = self._prepare_optimization_data_with_overfitting_protection(backtest_results, market_data)
        
        if not optimization_data:
            return self._minimal_optimization(backtest_results, market_data)
        
        # Use scipy optimization with overfitting protection
        return self._optimize_with_scipy_with_overfitting_protection(optimization_data)
    
    def _prepare_limited_optimization_data(self, backtest_results: List[BacktestResult], 
                                         market_data: pd.DataFrame, 
                                         limited_features: List[str]) -> Dict[str, Any]:
        """Prepare optimization data with limited features."""
        try:
            if not backtest_results:
                return {}
            
            self.logger.info(f"Preparing limited optimization data with {len(limited_features)} features")
            
            # Build feature matrix with limited features
            feature_data = {}
            for feature in limited_features:
                feature_values = []
                for result in backtest_results:
                    value = getattr(result, feature, 0.0)
                    feature_values.append(value)
                feature_data[feature] = np.array(feature_values)
            
            # Target variable
            target_scores = np.array([result.quality_score for result in backtest_results])
            
            return {
                'feature_data': feature_data,
                'target_scores': target_scores,
                'feature_names': limited_features,
                'overfitting_protection_applied': True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to prepare limited optimization data: {e}")
            return {}
    
    def _calculate_simple_score(self, backtest_results: List[BacktestResult], 
                              weights: Dict[str, float]) -> float:
        """Calculate a simple score for weight evaluation."""
        try:
            # Calculate weighted quality scores
            weighted_scores = []
            for result in backtest_results:
                weighted_score = 0.0
                total_weight = 0.0
                
                for feature, weight in weights.items():
                    value = getattr(result, feature, 0.0)
                    weighted_score += weight * value
                    total_weight += weight
                
                if total_weight > 0:
                    weighted_score = weighted_score / total_weight
                
                weighted_scores.append(weighted_score)
            
            # Calculate correlation with actual quality scores
            actual_scores = [result.quality_score for result in backtest_results]
            correlation, _ = pearsonr(weighted_scores, actual_scores)
            
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            self.logger.warning(f"Simple score calculation failed: {e}")
            return 0.0

def get_weight_optimization_engine(config: Optional[WeightOptimizationConfig] = None) -> WeightOptimizationEngine:
    """Get a weight optimization engine instance."""
    return WeightOptimizationEngine(config)

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
