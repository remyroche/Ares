"""
Feature selection optimization strategies.

This module contains optimization strategies for feature selection including
bayesian optimization, hyperparameter tuning, and performance optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
import time
from dataclasses import dataclass

from src.utils.tprint import tprint_debug, tprint_info, tprint_warning, tprint_success

# Import bayesian optimization utilities
try:
    from src.utils.ml_common.optimization.bayesian_entry_timing_optimizer import (
        BayesianEntryTimingOptimizer, EntryTimingConfig
    )
    from src.utils.ml_common.optimization.hpo_utils import HyperparameterOptimization
    BAYESIAN_OPTIMIZATION_AVAILABLE = True
except ImportError:
    BAYESIAN_OPTIMIZATION_AVAILABLE = False

# Import hardware optimization tools
try:
    from src.utils.hardware import (
        get_unified_hardware_manager,
        get_adaptive_optimization_engine,
        get_advanced_memory_optimizer,
        WorkloadType
    )
    from src.utils.hardware.advanced_memory_optimizer import AdvancedM1MemoryOptimizer, MemoryStrategy
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False


@dataclass
class OptimizationResult:
    """Result of feature selection optimization."""
    best_params: Dict[str, Any]
    best_score: float
    optimization_time: float
    n_trials: int
    convergence_info: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class FeatureSelectionOptimizer:
    """Optimization strategies for feature selection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("FeatureSelectionOptimizer")
        
        # Initialize hardware optimization tools
        self._initialize_hardware_tools()
        
        # Initialize bayesian optimization tools
        self._initialize_bayesian_tools()
    
    def _initialize_hardware_tools(self):
        """Initialize hardware optimization tools."""
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            try:
                self.hardware_manager = get_unified_hardware_manager()
                self.adaptive_engine = get_adaptive_optimization_engine()
                self.memory_optimizer = get_advanced_memory_optimizer()
                
                # Initialize advanced memory optimizer
                self.advanced_memory_optimizer = AdvancedM1MemoryOptimizer(
                    memory_limit_gb=8.0,
                    strategy=MemoryStrategy.AGGRESSIVE
                )
                
                tprint_success("✅ Hardware optimization tools initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Hardware optimization tools not available: {e}")
                self.hardware_manager = None
                self.adaptive_engine = None
                self.memory_optimizer = None
                self.advanced_memory_optimizer = None
        else:
            self.hardware_manager = None
            self.adaptive_engine = None
            self.memory_optimizer = None
            self.advanced_memory_optimizer = None
    
    def _initialize_bayesian_tools(self):
        """Initialize bayesian optimization tools."""
        if BAYESIAN_OPTIMIZATION_AVAILABLE:
            try:
                self.bayesian_optimizer = BayesianEntryTimingOptimizer(
                    EntryTimingConfig(
                        n_trials=100,
                        timeout_minutes=30
                    )
                )
                
                self.hpo_utils = HyperparameterOptimization(
                    config={
                        'enable_parallel': True,
                        'max_workers': 4,
                        'enable_monitoring': True,
                        'convergence': {
                            'improvement_threshold': 0.001,
                            'patience_trials': 20,
                            'min_trials': 10
                        }
                    }
                )
                
                tprint_success("✅ Bayesian optimization tools initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Bayesian optimization tools not available: {e}")
                self.bayesian_optimizer = None
                self.hpo_utils = None
        else:
            self.bayesian_optimizer = None
            self.hpo_utils = None
    
    def optimize_feature_selection_config(
        self,
        base_config: Dict[str, Any],
        data: pd.DataFrame,
        target: pd.Series,
        optimization_objective: Optional[Callable] = None
    ) -> OptimizationResult:
        """
        Optimize feature selection configuration using bayesian optimization.
        
        Args:
            base_config: Base feature selection configuration
            data: Feature data for optimization
            target: Target variable
            optimization_objective: Custom objective function
            
        Returns:
            OptimizationResult with optimized parameters
        """
        tprint_info("🔬 Starting bayesian optimization for feature selection parameters")
        start_time = time.time()
        
        try:
            if not BAYESIAN_OPTIMIZATION_AVAILABLE or self.bayesian_optimizer is None:
                tprint_warning("⚠️ Bayesian optimization not available, using base config")
                return OptimizationResult(
                    best_params=base_config,
                    best_score=0.5,
                    optimization_time=0.0,
                    n_trials=0,
                    convergence_info={},
                    success=False,
                    error_message="Bayesian optimization not available"
                )
            
            # Define optimization objective function
            def objective(trial):
                """Objective function for bayesian optimization."""
                tprint_debug(f"   🎯 Running optimization trial {trial.number}")
                
                # Sample feature selection parameters
                n_features_stage1 = trial.suggest_int('n_features_stage1', 80, 120)
                n_features_stage2 = trial.suggest_int('n_features_stage2', 60, 100)
                n_features_final = trial.suggest_int('n_features_final', 40, 80)
                
                # Sample selection criteria weights
                correlation_weight = trial.suggest_float('correlation_weight', 0.1, 1.0)
                importance_weight = trial.suggest_float('importance_weight', 0.1, 1.0)
                stability_weight = trial.suggest_float('stability_weight', 0.1, 1.0)
                
                # Sample processing parameters
                use_parallel = trial.suggest_categorical('use_parallel', [True, False])
                memory_efficient = trial.suggest_categorical('memory_efficient', [True, False])
                
                # Create temporary config for evaluation
                temp_config = base_config.copy()
                temp_config.update({
                    'stage_reductions': [n_features_stage1, n_features_stage2, n_features_final],
                    'selection_weights': {
                        'correlation': correlation_weight,
                        'importance': importance_weight,
                        'stability': stability_weight
                    },
                    'use_parallel': use_parallel,
                    'memory_efficient': memory_efficient,
                    'bayesian_optimized': True
                })
                
                # Evaluate configuration
                if optimization_objective:
                    score = optimization_objective(temp_config, data, target)
                else:
                    score = self._evaluate_config_performance(temp_config, data, target)
                
                tprint_debug(f"   📊 Trial {trial.number} score: {score:.4f}")
                return score
            
            # Run bayesian optimization
            tprint_info("🎯 Running bayesian optimization for feature selection parameters")
            study = self.bayesian_optimizer.create_study(
                study_name="feature_selection_optimization",
                direction="maximize"
            )
            
            # Run optimization with timeout and early stopping
            best_params = self.bayesian_optimizer.optimize(
                objective=objective,
                study=study,
                n_trials=50,  # Reduced for demo, increase in production
                timeout_minutes=5.0
            )
            
            optimization_time = time.time() - start_time
            
            # Create result
            result = OptimizationResult(
                best_params=best_params,
                best_score=study.best_value,
                optimization_time=optimization_time,
                n_trials=len(study.trials),
                convergence_info={
                    'best_value': study.best_value,
                    'n_trials': len(study.trials),
                    'study_name': study.study_name
                },
                success=True
            )
            
            tprint_success(
                f"✅ Bayesian optimization completed: best score {study.best_value:.4f}, "
                f"optimized parameters: {len(best_params)}"
            )
            
            return result
            
        except Exception as e:
            tprint_warning(f"❌ Bayesian optimization failed: {e}")
            return OptimizationResult(
                best_params=base_config,
                best_score=0.5,
                optimization_time=time.time() - start_time,
                n_trials=0,
                convergence_info={},
                success=False,
                error_message=str(e)
            )
    
    def _evaluate_config_performance(
        self, 
        config: Dict[str, Any], 
        data: pd.DataFrame, 
        target: pd.Series
    ) -> float:
        """Evaluate configuration performance (simplified implementation)."""
        tprint_debug("📊 Evaluating configuration performance")
        
        try:
            # Simple heuristic scoring based on configuration parameters
            score = 0.5  # Base score
            
            # Reward balanced stage reductions
            stages = config.get('stage_reductions', [120, 100, 80, 60])
            if len(stages) >= 3:
                reduction_balance = 1.0 - (max(stages) - min(stages)) / max(stages)
                score += reduction_balance * 0.3
            
            # Reward use of parallel processing
            if config.get('use_parallel', False):
                score += 0.1
            
            # Reward memory efficiency
            if config.get('memory_efficient', False):
                score += 0.1
            
            # Reward balanced selection weights
            weights = config.get('selection_weights', {})
            if weights:
                weight_values = list(weights.values())
                if weight_values:
                    weight_balance = 1.0 - np.std(weight_values) / (np.mean(weight_values) + 1e-10)
                    score += weight_balance * 0.1
            
            tprint_debug(f"   📊 Configuration score: {score:.4f}")
            return min(1.0, score)
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Error evaluating config performance: {e}")
            return 0.5  # Default score on error
    
    def get_optimal_hardware_config(self, workload_type: str = 'feature_selection') -> Dict[str, Any]:
        """Get optimal hardware configuration for feature selection."""
        tprint_debug(f"🛠️ Getting optimal hardware config for {workload_type}")
        
        try:
            if self.hardware_manager:
                config = self.hardware_manager.get_optimal_config(workload_type)
                tprint_debug(f"   ✅ Hardware config: {config}")
                return config
            else:
                # Fallback to default configuration
                default_config = {
                    'use_gpu': False,
                    'batch_size': 1000,
                    'num_threads': 4,
                    'memory_limit_gb': 8.0
                }
                tprint_warning("   ⚠️ Using default hardware config")
                return default_config
        except Exception as e:
            tprint_warning(f"   ⚠️ Failed to get hardware config: {e}")
            return {
                'use_gpu': False,
                'batch_size': 1000,
                'num_threads': 4,
                'memory_limit_gb': 8.0
            }
    
    def get_adaptive_strategy(
        self, 
        workload_type: str = 'feature_selection',
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get adaptive optimization strategy."""
        tprint_debug(f"🎯 Getting adaptive strategy for {workload_type}")
        
        try:
            if self.adaptive_engine and context:
                strategy = self.adaptive_engine.get_optimal_strategy(workload_type, context)
                tprint_debug(f"   ✅ Adaptive strategy: {strategy}")
                return strategy
            else:
                # Fallback to default strategy
                default_strategy = {
                    'batch_size': 1000,
                    'parallel_workers': 4,
                    'use_gpu': False,
                    'memory_limit_mb': 2048,
                    'hardware_accelerated': True,
                    'memory_efficient': True,
                    'parallel_processing': False
                }
                tprint_warning("   ⚠️ Using default adaptive strategy")
                return default_strategy
        except Exception as e:
            tprint_warning(f"   ⚠️ Failed to get adaptive strategy: {e}")
            return {
                'batch_size': 1000,
                'parallel_workers': 4,
                'use_gpu': False,
                'memory_limit_mb': 2048,
                'hardware_accelerated': True,
                'memory_efficient': True,
                'parallel_processing': False
            }
    
    def monitor_memory_pressure(self) -> Dict[str, Any]:
        """Monitor memory pressure and return statistics."""
        tprint_debug("🧠 Monitoring memory pressure")
        
        try:
            if self.memory_optimizer and hasattr(self.memory_optimizer, 'get_memory_pressure'):
                pressure = self.memory_optimizer.get_memory_pressure()
                cleanup_triggered = pressure > 0.8  # 80% threshold
                
                stats = {
                    'pressure': pressure,
                    'cleanup_triggered': cleanup_triggered,
                    'recommendations': ['Reduce batch size', 'Enable memory optimization'] if cleanup_triggered else []
                }
                
                tprint_debug(f"   📊 Memory pressure: {pressure:.2f}, cleanup triggered: {cleanup_triggered}")
                return stats
            else:
                tprint_debug("   ℹ️ Memory pressure monitoring not available")
                return {'pressure': 0.0, 'cleanup_triggered': False, 'recommendations': []}
        except Exception as e:
            tprint_warning(f"   ⚠️ Memory pressure monitoring failed: {e}")
            return {'pressure': 0.0, 'cleanup_triggered': False, 'recommendations': []}
    
    def perform_aggressive_memory_cleanup(self, force_cleanup: bool = False) -> Dict[str, Any]:
        """Perform aggressive memory cleanup."""
        tprint_info("🧹 Performing aggressive memory cleanup")
        
        try:
            if self.advanced_memory_optimizer:
                result = self.advanced_memory_optimizer.cleanup(force=force_cleanup)
                tprint_success(f"   ✅ Memory cleanup completed: {result.get('memory_freed_mb', 0):.1f}MB freed")
                return result
            else:
                tprint_warning("   ⚠️ Advanced memory optimizer not available")
                return {'memory_freed_mb': 0.0, 'success': False}
        except Exception as e:
            tprint_warning(f"   ⚠️ Memory cleanup failed: {e}")
            return {'memory_freed_mb': 0.0, 'success': False}
    
    def optimize_feature_selection_pipeline(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        base_config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], OptimizationResult]:
        """
        Optimize the entire feature selection pipeline.
        
        Args:
            X: Feature matrix
            y: Target variable
            base_config: Base configuration
            
        Returns:
            Tuple of (optimized_config, optimization_result)
        """
        tprint_info("🚀 Optimizing feature selection pipeline")
        
        try:
            # Get hardware configuration
            hardware_config = self.get_optimal_hardware_config()
            
            # Get adaptive strategy
            context = {
                'data_shape': X.shape,
                'hardware_config': hardware_config,
                'memory_pressure': self.monitor_memory_pressure()['pressure']
            }
            adaptive_strategy = self.get_adaptive_strategy(context=context)
            
            # Run bayesian optimization
            optimization_result = self.optimize_feature_selection_config(
                base_config, X, y
            )
            
            # Combine all optimizations
            optimized_config = base_config.copy()
            optimized_config.update({
                'hardware_config': hardware_config,
                'adaptive_strategy': adaptive_strategy,
                'bayesian_optimization': optimization_result.best_params,
                'optimization_metadata': {
                    'optimization_time': optimization_result.optimization_time,
                    'n_trials': optimization_result.n_trials,
                    'best_score': optimization_result.best_score
                }
            })
            
            tprint_success("✅ Feature selection pipeline optimization completed")
            return optimized_config, optimization_result
            
        except Exception as e:
            tprint_warning(f"⚠️ Pipeline optimization failed: {e}")
            return base_config, OptimizationResult(
                best_params=base_config,
                best_score=0.5,
                optimization_time=0.0,
                n_trials=0,
                convergence_info={},
                success=False,
                error_message=str(e)
            )