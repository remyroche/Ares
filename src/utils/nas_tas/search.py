#!/usr/bin/env python3
"""
Unified Search Algorithms

This module provides unified search algorithms combining Bayesian TPE optimization
with tree-specific search strategies, consolidating search capabilities for both NAS and TAS.

Key Features:
- Integration with existing bayesian_tpe_optimizer.py
- Tree-specific search strategies for TAS
- Neural architecture search for NAS
- Unified search interface
- Multi-objective optimization support
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from abc import ABC, abstractmethod

# Import Bayesian TPE optimizer
try:
    from src.utils.nas_tas.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer, BayesianTPEConfig, optimize_with_bayesian_tpe
    )
    BAYESIAN_TPE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Bayesian TPE optimizer not available: {e}")
    BAYESIAN_TPE_AVAILABLE = False

# Import utility modules
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success
    )
    UTILITY_MODULES_AVAILABLE = True
except ImportError:
    UTILITY_MODULES_AVAILABLE = False
    # Fallback functions
    def tprint(*args, **kwargs):
        print(*args, **kwargs)
    def tprint_info(*args, **kwargs):
        print("INFO:", *args, **kwargs)
    def tprint_warning(*args, **kwargs):
        print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs):
        print("ERROR:", *args, **kwargs)
    def tprint_success(*args, **kwargs):
        print("SUCCESS:", *args, **kwargs)

logger = logging.getLogger(__name__)


class SearchStrategy(ABC):
    """Abstract base class for search strategies."""
    
    @abstractmethod
    def search(self, objective_function: Callable, search_space: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Perform search optimization."""
        pass


class BayesianTPEStrategy(SearchStrategy):
    """Bayesian TPE search strategy using existing optimizer."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Bayesian TPE strategy."""
        self.config = config
        
    def search(self, objective_function: Callable, search_space: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Perform Bayesian TPE search."""
        if not BAYESIAN_TPE_AVAILABLE:
            tprint_warning("Bayesian TPE not available, falling back to random search")
            return self._random_search(objective_function, search_space, **kwargs)
        
        try:
            # Use existing Bayesian TPE optimizer
            n_trials = self.config.get('n_trials', 50)
            timeout_seconds = self.config.get('timeout_seconds', None)
            
            # Create TPE config
            tpe_config = BayesianTPEConfig(
                n_trials=n_trials,
                timeout_seconds=timeout_seconds,
                enable_early_stopping=self.config.get('enable_early_stopping', True),
                early_stopping_patience=self.config.get('early_stopping_patience', 10)
            )
            
            # Run optimization
            result = optimize_with_bayesian_tpe(
                objective_function=objective_function,
                search_space=search_space,
                config=tpe_config
            )
            
            tprint_success(f"Bayesian TPE search completed with {len(result.get('trials', []))} trials")
            return result
            
        except Exception as e:
            tprint_error(f"Bayesian TPE search failed: {e}")
            return self._random_search(objective_function, search_space, **kwargs)
    
    def _random_search(self, objective_function: Callable, search_space: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Fallback random search."""
        n_trials = self.config.get('n_trials', 10)
        best_score = float('-inf')
        best_params = None
        
        for trial in range(n_trials):
            # Sample random parameters
            params = self._sample_random_params(search_space)
            
            try:
                score = objective_function(params)
                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception as e:
                tprint_warning(f"Trial {trial} failed: {e}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': n_trials,
            'strategy': 'random_search'
        }
    
    def _sample_random_params(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample random parameters from search space."""
        params = {}
        
        for param_name, param_config in search_space.items():
            if isinstance(param_config, dict):
                param_type = param_config.get('type', 'float')
                
                if param_type == 'float':
                    low = param_config.get('low', 0.0)
                    high = param_config.get('high', 1.0)
                    params[param_name] = np.random.uniform(low, high)
                
                elif param_type == 'int':
                    low = param_config.get('low', 1)
                    high = param_config.get('high', 10)
                    params[param_name] = np.random.randint(low, high + 1)
                
                elif param_type == 'categorical':
                    choices = param_config.get('choices', [])
                    params[param_name] = np.random.choice(choices)
            else:
                # Simple list of choices
                params[param_name] = np.random.choice(param_config)
        
        return params


class TreeSearchStrategy(SearchStrategy):
    """Tree-specific search strategy for TAS."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize tree search strategy."""
        self.config = config
        
    def search(self, objective_function: Callable, search_space: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Perform tree-specific search."""
        try:
            # Tree-specific search parameters
            max_depth = search_space.get('max_depth', {'type': 'int', 'low': 3, 'high': 15})
            n_estimators = search_space.get('n_estimators', {'type': 'int', 'low': 50, 'high': 500})
            
            # Grid search for tree parameters
            depth_values = list(range(3, 16, 2)) if isinstance(max_depth, dict) else max_depth
            estimator_values = list(range(50, 501, 50)) if isinstance(n_estimators, dict) else n_estimators
            
            best_score = float('-inf')
            best_params = None
            
            for depth in depth_values:
                for n_est in estimator_values:
                    params = {
                        'max_depth': depth,
                        'n_estimators': n_est,
                        'random_state': 42
                    }
                    
                    try:
                        score = objective_function(params)
                        if score > best_score:
                            best_score = score
                            best_params = params
                    except Exception as e:
                        tprint_warning(f"Tree search trial failed: {e}")
            
            tprint_success(f"Tree search completed with best score: {best_score:.4f}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'strategy': 'tree_search'
            }
            
        except Exception as e:
            tprint_error(f"Tree search failed: {e}")
            return {'best_params': None, 'best_score': 0.0, 'strategy': 'tree_search', 'error': str(e)}


class NeuralArchitectureSearchStrategy(SearchStrategy):
    """Neural architecture search strategy for NAS."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize neural architecture search strategy."""
        self.config = config
        
    def search(self, objective_function: Callable, search_space: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Perform neural architecture search."""
        try:
            # NAS-specific parameters
            n_layers = search_space.get('n_layers', {'type': 'int', 'low': 2, 'high': 8})
            n_neurons = search_space.get('n_neurons', {'type': 'int', 'low': 32, 'high': 512})
            activation = search_space.get('activation', ['relu', 'tanh', 'sigmoid'])
            dropout = search_space.get('dropout', {'type': 'float', 'low': 0.0, 'high': 0.5})
            
            best_score = float('-inf')
            best_params = None
            
            # Random architecture sampling
            n_trials = self.config.get('n_trials', 20)
            
            for trial in range(n_trials):
                # Sample architecture
                if isinstance(n_layers, dict):
                    layers = np.random.randint(n_layers['low'], n_layers['high'] + 1)
                else:
                    layers = np.random.choice(n_layers)
                
                if isinstance(n_neurons, dict):
                    neurons = np.random.randint(n_neurons['low'], n_neurons['high'] + 1)
                else:
                    neurons = np.random.choice(n_neurons)
                
                if isinstance(activation, list):
                    act = np.random.choice(activation)
                else:
                    act = activation
                
                if isinstance(dropout, dict):
                    drop = np.random.uniform(dropout['low'], dropout['high'])
                else:
                    drop = np.random.choice(dropout)
                
                params = {
                    'n_layers': layers,
                    'n_neurons': neurons,
                    'activation': act,
                    'dropout': drop,
                    'learning_rate': np.random.uniform(1e-5, 1e-2)
                }
                
                try:
                    score = objective_function(params)
                    if score > best_score:
                        best_score = score
                        best_params = params
                except Exception as e:
                    tprint_warning(f"NAS trial {trial} failed: {e}")
            
            tprint_success(f"Neural architecture search completed with best score: {best_score:.4f}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'strategy': 'neural_architecture_search'
            }
            
        except Exception as e:
            tprint_error(f"Neural architecture search failed: {e}")
            return {'best_params': None, 'best_score': 0.0, 'strategy': 'neural_architecture_search', 'error': str(e)}


class UnifiedSearchEngine:
    """Unified search engine combining different strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize unified search engine."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize search strategies
        self.strategies = {
            'bayesian_tpe': BayesianTPEStrategy(config),
            'tree_search': TreeSearchStrategy(config),
            'neural_architecture_search': NeuralArchitectureSearchStrategy(config)
        }
    
    def search(self, 
               objective_function: Callable,
               search_space: Dict[str, Any],
               strategy: str = 'bayesian_tpe',
               **kwargs) -> Dict[str, Any]:
        """Perform search using specified strategy."""
        
        if strategy not in self.strategies:
            tprint_warning(f"Strategy {strategy} not available, using bayesian_tpe")
            strategy = 'bayesian_tpe'
        
        tprint_info(f"Starting {strategy} search...")
        
        try:
            result = self.strategies[strategy].search(objective_function, search_space, **kwargs)
            
            # Add metadata
            result['strategy_used'] = strategy
            result['search_space'] = search_space
            result['config'] = self.config
            
            tprint_success(f"Search completed using {strategy}")
            return result
            
        except Exception as e:
            tprint_error(f"Search failed: {e}")
            return {
                'best_params': None,
                'best_score': 0.0,
                'strategy_used': strategy,
                'error': str(e)
            }
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available search strategies."""
        return list(self.strategies.keys())
    
    def add_custom_strategy(self, name: str, strategy: SearchStrategy):
        """Add custom search strategy."""
        self.strategies[name] = strategy
        tprint_info(f"Added custom strategy: {name}")