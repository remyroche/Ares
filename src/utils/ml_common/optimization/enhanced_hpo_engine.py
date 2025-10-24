"""
Enhanced HPO Engine with Multi-Objective, Early Stopping, and Warm Starting

This module provides a comprehensive HPO engine that integrates all the
enhanced features: multi-objective optimization, early stopping, warm starting,
and concurrent model optimization.

Enhancement: Complete HPO system integration
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import json
from pathlib import Path

# Import existing components
from .core.hpo_engine import HPOEngine, OptimizationContext
from .core.optimization_strategy import OptimizationStrategy
from .validation import HPOConfig
from .results import HPOResult

# Import new enhanced components
from .multi_objective_optimizer import (
    MultiObjectiveOptimizer, MultiObjectiveConfig, 
    create_multi_objective_optimizer, create_accuracy_efficiency_objectives
)
from .enhanced_early_stopping_integration import (
    EarlyStoppingIntegration, EarlyStoppingIntegrationConfig,
    create_early_stopping_integration
)
from .warm_starting_system import (
    WarmStartManager, WarmStartConfig, WarmStartData,
    create_warm_start_manager, create_warm_start_data_from_hpo_result
)

logger = logging.getLogger(__name__)


@dataclass
class EnhancedHPOConfig:
    """Configuration for enhanced HPO engine."""
    
    # Base HPO configuration
    base_config: HPOConfig
    
    # Multi-objective optimization
    enable_multi_objective: bool = False
    multi_objective_config: Optional[MultiObjectiveConfig] = None
    objective_functions: List[Callable] = None
    
    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_config: Optional[EarlyStoppingIntegrationConfig] = None
    
    # Warm starting
    enable_warm_start: bool = True
    warm_start_config: Optional[WarmStartConfig] = None
    
    # Concurrent optimization
    enable_concurrent_optimization: bool = False
    max_concurrent_models: int = 3
    concurrent_strategy: str = 'thread'  # 'thread' or 'process'
    
    # Performance tracking
    enable_performance_tracking: bool = True
    save_optimization_history: bool = True
    optimization_history_file: str = "optimization_history.json"
    
    # Resource management
    max_memory_usage_gb: float = 8.0
    max_cpu_usage_percent: float = 80.0


class EnhancedHPOEngine:
    """Enhanced HPO engine with all advanced features."""
    
    def __init__(self, config: EnhancedHPOConfig):
        self.config = config
        self.base_engine = HPOEngine(config.base_config)
        
        # Initialize enhanced components
        self.multi_objective_optimizer = None
        self.early_stopping_integration = None
        self.warm_start_manager = None
        
        # Performance tracking
        self.optimization_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Enhanced HPO engine initialized")
    
    def _initialize_components(self):
        """Initialize enhanced components."""
        # Initialize multi-objective optimizer
        if self.config.enable_multi_objective:
            if self.config.multi_objective_config:
                self.multi_objective_optimizer = MultiObjectiveOptimizer(
                    self.config.multi_objective_config
                )
            else:
                self.multi_objective_optimizer = create_multi_objective_optimizer()
            
            # Add default objectives if none provided
            if not self.config.objective_functions:
                objectives = create_accuracy_efficiency_objectives()
                for obj in objectives:
                    self.multi_objective_optimizer.add_objective(
                        obj.name, obj.function, obj.weight, obj.direction
                    )
        
        # Initialize early stopping integration
        if self.config.enable_early_stopping:
            if self.config.early_stopping_config:
                self.early_stopping_integration = EarlyStoppingIntegration(
                    self.config.early_stopping_config
                )
            else:
                self.early_stopping_integration = create_early_stopping_integration()
        
        # Initialize warm start manager
        if self.config.enable_warm_start:
            if self.config.warm_start_config:
                self.warm_start_manager = WarmStartManager(self.config.warm_start_config)
            else:
                self.warm_start_manager = create_warm_start_manager()
    
    def optimize_single_model(self, 
                            model_factory: Callable,
                            X: Any,
                            y: Any,
                            search_space: Dict[str, Any],
                            model_name: str = "unknown",
                            use_warm_start: bool = True,
                            use_early_stopping: bool = True) -> HPOResult:
        """
        Optimize a single model with enhanced features.
        
        Args:
            model_factory: Function to create model instances
            X: Training features
            y: Training targets
            search_space: Parameter search space
            model_name: Name of the model
            use_warm_start: Whether to use warm starting
            use_early_stopping: Whether to use early stopping
            
        Returns:
            HPOResult with optimization results
        """
        start_time = time.time()
        logger.info(f"Starting enhanced optimization for {model_name}")
        
        # Warm start if enabled
        warm_start_data = None
        if use_warm_start and self.warm_start_manager:
            warm_start_data = self._get_warm_start_data(
                model_name, search_space, X, y
            )
        
        # Choose optimization strategy
        if self.config.enable_multi_objective and self.multi_objective_optimizer:
            result = self._optimize_multi_objective(
                model_factory, X, y, search_space, model_name, warm_start_data
            )
        else:
            result = self._optimize_single_objective(
                model_factory, X, y, search_space, model_name, 
                warm_start_data, use_early_stopping
            )
        
        # Save warm start data
        if use_warm_start and self.warm_start_manager:
            self._save_warm_start_data(result, model_name, search_space, X, y)
        
        # Track performance
        if self.config.enable_performance_tracking:
            self._track_optimization_performance(result, model_name, time.time() - start_time)
        
        logger.info(f"Enhanced optimization completed for {model_name} in {result.optimization_time:.2f}s")
        return result
    
    def optimize_multiple_models(self,
                               model_configs: List[Dict[str, Any]],
                               X: Any,
                               y: Any,
                               search_spaces: List[Dict[str, Any]],
                               use_concurrent: bool = True) -> List[HPOResult]:
        """
        Optimize multiple models concurrently or sequentially.
        
        Args:
            model_configs: List of model configurations
            X: Training features
            y: Training targets
            search_spaces: List of search spaces for each model
            use_concurrent: Whether to use concurrent optimization
            
        Returns:
            List of HPOResult objects
        """
        if len(model_configs) != len(search_spaces):
            raise ValueError("Number of model configs must match number of search spaces")
        
        logger.info(f"Starting optimization for {len(model_configs)} models")
        
        if use_concurrent and self.config.enable_concurrent_optimization:
            return self._optimize_concurrent(model_configs, X, y, search_spaces)
        else:
            return self._optimize_sequential(model_configs, X, y, search_spaces)
    
    def _optimize_multi_objective(self,
                                model_factory: Callable,
                                X: Any,
                                y: Any,
                                search_space: Dict[str, Any],
                                model_name: str,
                                warm_start_data: Optional[Dict[str, Any]]) -> HPOResult:
        """Perform multi-objective optimization."""
        logger.info(f"Performing multi-objective optimization for {model_name}")
        
        # Run multi-objective optimization
        results = self.multi_objective_optimizer.optimize(
            search_space=search_space,
            model_factory=model_factory,
            X=X,
            y=y,
            warm_start_data=warm_start_data
        )
        
        # Convert to HPOResult format
        best_solution = results['diverse_solutions'][0] if results['diverse_solutions'] else None
        if not best_solution:
            raise ValueError("No solutions found in multi-objective optimization")
        
        return HPOResult(
            best_params=best_solution['params'],
            best_score=best_solution['objectives'][0],  # Primary objective
            n_trials=results['n_trials'],
            trial_results=results['pareto_front'],
            strategy='multi_objective',
            optimization_time=results['optimization_time'],
            metadata={
                'pareto_front': results['pareto_front'],
                'diverse_solutions': results['diverse_solutions'],
                'metrics': results['metrics'],
                'objective_names': results['objective_names']
            }
        )
    
    def _optimize_single_objective(self,
                                 model_factory: Callable,
                                 X: Any,
                                 y: Any,
                                 search_space: Dict[str, Any],
                                 model_name: str,
                                 warm_start_data: Optional[Dict[str, Any]],
                                 use_early_stopping: bool) -> HPOResult:
        """Perform single-objective optimization with enhanced features."""
        # Use base engine for single-objective optimization
        result = self.base_engine.optimize(
            model_factory=model_factory,
            X=X,
            y=y,
            search_space=search_space,
            model_name=model_name
        )
        
        # Add early stopping information if available
        if use_early_stopping and self.early_stopping_integration:
            early_stopping_info = self.early_stopping_integration.get_early_stopping_summary()
            result.metadata = result.metadata or {}
            result.metadata['early_stopping'] = early_stopping_info
        
        return result
    
    def _optimize_concurrent(self,
                           model_configs: List[Dict[str, Any]],
                           X: Any,
                           y: Any,
                           search_spaces: List[Dict[str, Any]]) -> List[HPOResult]:
        """Optimize multiple models concurrently."""
        logger.info(f"Starting concurrent optimization with {self.config.max_concurrent_models} workers")
        
        # Prepare optimization tasks
        tasks = []
        for i, (model_config, search_space) in enumerate(zip(model_configs, search_spaces)):
            task = {
                'model_factory': model_config['model_factory'],
                'model_name': model_config.get('model_name', f'model_{i}'),
                'search_space': search_space,
                'use_warm_start': model_config.get('use_warm_start', True),
                'use_early_stopping': model_config.get('use_early_stopping', True)
            }
            tasks.append(task)
        
        # Choose executor - prefer ProcessPoolExecutor for better parallelism with sklearn
        if self.config.concurrent_strategy == 'thread':
            executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_models)
        else:
            # Default to process-based parallelism for better performance with sklearn
            executor = ProcessPoolExecutor(max_workers=self.config.max_concurrent_models)
        
        # Execute optimizations with interleaved results
        results = []
        try:
            with executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(self._optimize_single_model_task, task, X, y): task
                    for task in tasks
                }
                
                # Process results as they complete
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        results.append(result)
                        logger.info(f"Completed optimization for {task['model_name']}")
                    except Exception as e:
                        logger.error(f"Concurrent optimization failed for {task['model_name']}: {e}")
                        results.append(None)
        except Exception as e:
            logger.error(f"Concurrent optimization error: {e}")
            # Fallback to sequential
            return self._optimize_sequential(model_configs, X, y, search_spaces)
        
        # Filter out None results
        results = [r for r in results if r is not None]
        
        logger.info(f"Concurrent optimization completed: {len(results)} successful results")
        return results
    
    def _optimize_single_model_task(self, task: Dict[str, Any], X: Any, y: Any) -> HPOResult:
        """Single model optimization task for concurrent execution."""
        return self.optimize_single_model(
            model_factory=task['model_factory'],
            X=X,
            y=y,
            search_space=task['search_space'],
            model_name=task['model_name'],
            use_warm_start=task['use_warm_start'],
            use_early_stopping=task['use_early_stopping']
        )
    
    def _optimize_sequential(self,
                           model_configs: List[Dict[str, Any]],
                           X: Any,
                           y: Any,
                           search_spaces: List[Dict[str, Any]]) -> List[HPOResult]:
        """Optimize multiple models sequentially."""
        logger.info("Starting sequential optimization")
        
        results = []
        for i, (model_config, search_space) in enumerate(zip(model_configs, search_spaces)):
            try:
                result = self.optimize_single_model(
                    model_factory=model_config['model_factory'],
                    X=X,
                    y=y,
                    search_space=search_space,
                    model_name=model_config.get('model_name', f'model_{i}'),
                    use_warm_start=model_config.get('use_warm_start', True),
                    use_early_stopping=model_config.get('use_early_stopping', True)
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Sequential optimization failed for model {i}: {e}")
                results.append(None)
        
        # Filter out None results
        results = [r for r in results if r is not None]
        
        logger.info(f"Sequential optimization completed: {len(results)} successful results")
        return results
    
    def _get_warm_start_data(self,
                           model_name: str,
                           search_space: Dict[str, Any],
                           X: Any,
                           y: Any) -> Optional[Dict[str, Any]]:
        """Get warm start data for optimization."""
        if not self.warm_start_manager:
            return None
        
        # Create dataset hash
        dataset_hash = self._create_dataset_hash(X, y)
        
        # Find similar optimizations
        similar_data = self.warm_start_manager.find_similar_optimizations(
            model_name, search_space, dataset_hash
        )
        
        if similar_data:
            # Create warm start parameters
            warm_start_params = self.warm_start_manager.create_warm_start_parameters(
                similar_data, search_space
            )
            
            return {
                'warm_start_params': warm_start_params,
                'similar_data': similar_data
            }
        
        return None
    
    def _save_warm_start_data(self,
                            result: HPOResult,
                            model_name: str,
                            search_space: Dict[str, Any],
                            X: Any,
                            y: Any):
        """Save optimization result as warm start data."""
        if not self.warm_start_manager:
            return
        
        # Create dataset hash
        dataset_hash = self._create_dataset_hash(X, y)
        
        # Create warm start data
        warm_start_data = create_warm_start_data_from_hpo_result(
            result, model_name, result.strategy, search_space, dataset_hash
        )
        
        # Add to warm start manager
        self.warm_start_manager.add_warm_start_data(warm_start_data)
    
    def _create_dataset_hash(self, X: Any, y: Any) -> str:
        """Create hash of dataset for warm starting."""
        import hashlib
        
        # Create hash from dataset characteristics
        if hasattr(X, 'shape'):
            x_info = f"{X.shape}_{X.dtype}"
        else:
            x_info = str(type(X))
        
        if hasattr(y, 'shape'):
            y_info = f"{y.shape}_{y.dtype}"
        else:
            y_info = str(type(y))
        
        dataset_info = f"{x_info}_{y_info}"
        return hashlib.md5(dataset_info.encode()).hexdigest()
    
    def _track_optimization_performance(self, result: HPOResult, model_name: str, total_time: float):
        """Track optimization performance metrics."""
        performance_entry = {
            'timestamp': time.time(),
            'model_name': model_name,
            'strategy': result.strategy,
            'best_score': result.best_score,
            'n_trials': result.n_trials,
            'optimization_time': result.optimization_time,
            'total_time': total_time,
            'early_stopped': getattr(result, 'early_stopped', False),
            'warm_started': getattr(result, 'warm_started', False)
        }
        
        self.optimization_history.append(performance_entry)
        
        # Save to file if enabled
        if self.config.save_optimization_history:
            self._save_optimization_history()
    
    def _save_optimization_history(self):
        """Save optimization history to file."""
        try:
            with open(self.config.optimization_history_file, 'w') as f:
                json.dump(self.optimization_history, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save optimization history: {e}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary."""
        summary = {
            'total_optimizations': len(self.optimization_history),
            'strategies_used': list(set(entry['strategy'] for entry in self.optimization_history)),
            'avg_optimization_time': np.mean([entry['optimization_time'] for entry in self.optimization_history]),
            'avg_best_score': np.mean([entry['best_score'] for entry in self.optimization_history]),
            'early_stopping_summary': {},
            'warm_start_summary': {}
        }
        
        # Add early stopping summary
        if self.early_stopping_integration:
            summary['early_stopping_summary'] = self.early_stopping_integration.get_early_stopping_summary()
        
        # Add warm start summary
        if self.warm_start_manager:
            summary['warm_start_summary'] = self.warm_start_manager.get_warm_start_summary()
        
        return summary
    
    def save_results(self, filepath: str, results: Union[HPOResult, List[HPOResult]]):
        """Save optimization results to file."""
        try:
            if isinstance(results, list):
                results_data = [self._result_to_dict(r) for r in results]
            else:
                results_data = self._result_to_dict(results)
            
            with open(filepath, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            logger.info(f"Results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _result_to_dict(self, result: HPOResult) -> Dict[str, Any]:
        """Convert HPOResult to dictionary."""
        return {
            'best_params': result.best_params,
            'best_score': result.best_score,
            'n_trials': result.n_trials,
            'strategy': result.strategy,
            'optimization_time': result.optimization_time,
            'trial_results': result.trial_results,
            'metadata': result.metadata
        }


# Convenience functions
def create_enhanced_hpo_engine(
    base_config: Optional[HPOConfig] = None,
    enable_multi_objective: bool = False,
    enable_early_stopping: bool = True,
    enable_warm_start: bool = True,
    enable_concurrent: bool = False,
    **kwargs
) -> EnhancedHPOEngine:
    """Create enhanced HPO engine with default settings."""
    if base_config is None:
        base_config = HPOConfig()
    
    config = EnhancedHPOConfig(
        base_config=base_config,
        enable_multi_objective=enable_multi_objective,
        enable_early_stopping=enable_early_stopping,
        enable_warm_start=enable_warm_start,
        enable_concurrent_optimization=enable_concurrent,
        **kwargs
    )
    
    return EnhancedHPOEngine(config)


def create_multi_model_optimization_config(
    model_types: List[str],
    search_spaces: List[Dict[str, Any]],
    model_factories: List[Callable],
    use_concurrent: bool = True
) -> List[Dict[str, Any]]:
    """Create configuration for multi-model optimization."""
    if len(model_types) != len(search_spaces) or len(model_types) != len(model_factories):
        raise ValueError("All lists must have the same length")
    
    configs = []
    for i, (model_type, search_space, model_factory) in enumerate(zip(model_types, search_spaces, model_factories)):
        config = {
            'model_name': f"{model_type}_{i}",
            'model_factory': model_factory,
            'use_warm_start': True,
            'use_early_stopping': True
        }
        configs.append(config)
    
    return configs