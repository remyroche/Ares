"""
Strategy Search Optimizer

This module provides unified strategy search capabilities that consolidate
strategy search logic previously scattered across NAS and TAS implementations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

from ..core.nas_engine import NASEngine
from ..core.tas_engine import TASEngine
from ..config.base_config import UnifiedArchitectureConfig, ArchitectureType


@dataclass
class StrategySearchConfig:
    """Configuration for strategy search optimization."""
    
    # Search parameters
    max_iterations: int = 100
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    
    # Strategy settings
    strategy_types: List[str] = field(default_factory=lambda: [
        "momentum", "mean_reversion", "breakout", "arbitrage", "regime_aware"
    ])
    enable_ensemble_strategies: bool = True
    ensemble_method: str = "weighted_average"
    
    # Evaluation settings
    evaluation_metric: str = "sharpe_ratio"
    backtest_periods: int = 252  # 1 year of daily data
    validation_split: float = 0.2
    
    # Risk management
    max_position_size: float = 0.1
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    max_drawdown_limit: float = 0.15
    
    # Optimization settings
    enable_early_stopping: bool = True
    early_stopping_patience: int = 20
    min_improvement_threshold: float = 0.001
    
    # Parallel processing
    enable_parallel_processing: bool = True
    max_workers: int = 4
    
    # Output settings
    save_results: bool = True
    results_path: str = "strategy_search_results"
    enable_visualization: bool = True


@dataclass
class StrategySearchResult:
    """Result from strategy search."""
    
    # Search results
    best_strategy: Dict[str, Any]
    best_score: float
    search_history: List[Dict[str, Any]]
    
    # Performance metrics
    total_iterations: int
    convergence_iteration: int
    search_time: float
    
    # Strategy details
    strategy_type: str
    risk_score: float
    complexity_score: float
    
    # Backtesting results
    backtest_results: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    search_timestamp: datetime
    configuration: Dict[str, Any]


class StrategySearchOptimizer:
    """
    Unified strategy search optimizer for both NAS and TAS systems.
    
    This class consolidates strategy search logic that was previously
    scattered across different implementations, providing a unified interface
    for both neural and tree-based strategy search.
    """
    
    def __init__(self, config: StrategySearchConfig):
        """Initialize strategy search optimizer.
        
        Args:
            config: Strategy search configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize search engines
        self.nas_engine = None
        self.tas_engine = None
        
        # Search state
        self.search_history = []
        self.best_strategy = None
        self.best_score = -np.inf
        
        tprint_info("Strategy Search Optimizer initialized")
    
    def initialize_engines(self, unified_config: UnifiedArchitectureConfig):
        """Initialize NAS and TAS engines based on configuration.
        
        Args:
            unified_config: Unified architecture configuration
        """
        try:
            # Initialize NAS engine if needed
            if unified_config.architecture_type in [ArchitectureType.NEURAL_ONLY, ArchitectureType.HYBRID_NEURAL_TREE]:
                self.nas_engine = NASEngine(unified_config.__dict__)
                tprint_success("NAS engine initialized for strategy search")
            
            # Initialize TAS engine if needed
            if unified_config.architecture_type in [ArchitectureType.TREE_ONLY, ArchitectureType.HYBRID_NEURAL_TREE]:
                self.tas_engine = TASEngine(unified_config.__dict__)
                tprint_success("TAS engine initialized for strategy search")
            
        except Exception as e:
            tprint_error(f"Engine initialization failed: {e}")
            raise
    
    async def search_strategies(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        unified_config: UnifiedArchitectureConfig
    ) -> StrategySearchResult:
        """Search for optimal strategies using unified approach.
        
        Args:
            data: Input data for strategy search
            search_space: Strategy search space
            unified_config: Unified architecture configuration
            
        Returns:
            StrategySearchResult with search results
        """
        start_time = datetime.now()
        tprint_info("Starting unified strategy search")
        
        try:
            # Initialize engines
            self.initialize_engines(unified_config)
            
            # Perform search based on architecture type
            if unified_config.architecture_type == ArchitectureType.NEURAL_ONLY:
                result = await self._search_neural_strategies(data, search_space)
            elif unified_config.architecture_type == ArchitectureType.TREE_ONLY:
                result = await self._search_tree_strategies(data, search_space)
            elif unified_config.architecture_type == ArchitectureType.HYBRID_NEURAL_TREE:
                result = await self._search_hybrid_strategies(data, search_space)
            else:
                raise ValueError(f"Unsupported architecture type: {unified_config.architecture_type}")
            
            # Calculate search metrics
            search_time = (datetime.now() - start_time).total_seconds()
            result.search_time = search_time
            result.total_iterations = len(self.search_history)
            result.convergence_iteration = self._find_convergence_point()
            
            # Perform backtesting
            if self.config.backtest_periods > 0:
                result.backtest_results = await self._backtest_strategy(
                    result.best_strategy, data
                )
            
            tprint_success(f"Strategy search completed in {search_time:.2f}s")
            tprint_info(f"Best score: {result.best_score:.4f}")
            
            return result
            
        except Exception as e:
            tprint_error(f"Strategy search failed: {e}")
            raise
    
    async def _search_neural_strategies(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any]
    ) -> StrategySearchResult:
        """Search for neural-based strategies using NAS engine."""
        tprint_info("Searching neural strategies")
        
        if not self.nas_engine:
            raise ValueError("NAS engine not initialized")
        
        # Use NAS engine for strategy search
        nas_results = self.nas_engine.search_architectures(
            data=data,
            search_space=search_space,
            optimization_method="bayesian_tpe",
            n_trials=self.config.max_iterations
        )
        
        return StrategySearchResult(
            best_strategy=nas_results.get('best_architecture', {}),
            best_score=nas_results.get('best_score', 0.0),
            search_history=nas_results.get('trials', []),
            total_iterations=0,
            convergence_iteration=0,
            search_time=0.0,
            strategy_type="neural",
            risk_score=0.0,
            complexity_score=0.0,
            search_timestamp=datetime.now(),
            configuration=self.config.__dict__
        )
    
    async def _search_tree_strategies(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any]
    ) -> StrategySearchResult:
        """Search for tree-based strategies using TAS engine."""
        tprint_info("Searching tree strategies")
        
        if not self.tas_engine:
            raise ValueError("TAS engine not initialized")
        
        # Use TAS engine for strategy search
        tas_results = self.tas_engine.search_strategies(
            data=data,
            search_space=search_space,
            optimization_method="bayesian_tpe",
            n_trials=self.config.max_iterations,
            include_regime_specific=True
        )
        
        return StrategySearchResult(
            best_strategy=tas_results.get('best_strategy', {}),
            best_score=tas_results.get('best_score', 0.0),
            search_history=tas_results.get('trials', []),
            total_iterations=0,
            convergence_iteration=0,
            search_time=0.0,
            strategy_type="tree",
            risk_score=0.0,
            complexity_score=0.0,
            search_timestamp=datetime.now(),
            configuration=self.config.__dict__
        )
    
    async def _search_hybrid_strategies(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any]
    ) -> StrategySearchResult:
        """Search for hybrid strategies combining neural and tree components."""
        tprint_info("Searching hybrid strategies")
        
        if not self.nas_engine or not self.tas_engine:
            raise ValueError("Both NAS and TAS engines required for hybrid search")
        
        # Search neural strategies
        nas_results = await self._search_neural_strategies(data, search_space)
        
        # Search tree strategies
        tas_results = await self._search_tree_strategies(data, search_space)
        
        # Combine results for hybrid strategy
        hybrid_strategy = {
            'neural_strategy': nas_results.best_strategy,
            'tree_strategy': tas_results.best_strategy,
            'ensemble_method': self.config.ensemble_method,
            'neural_weight': 0.6,
            'tree_weight': 0.4,
            'strategy_type': 'hybrid'
        }
        
        # Calculate hybrid score
        hybrid_score = (nas_results.best_score * 0.6 + tas_results.best_score * 0.4)
        
        return StrategySearchResult(
            best_strategy=hybrid_strategy,
            best_score=hybrid_score,
            search_history=nas_results.search_history + tas_results.search_history,
            total_iterations=0,
            convergence_iteration=0,
            search_time=0.0,
            strategy_type="hybrid",
            risk_score=0.0,
            complexity_score=0.0,
            search_timestamp=datetime.now(),
            configuration=self.config.__dict__
        )
    
    async def _backtest_strategy(
        self,
        strategy: Dict[str, Any],
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """Backtest the best strategy."""
        tprint_info("Backtesting strategy")
        
        try:
            # Simple backtesting implementation
            # In a real implementation, this would use the backtesting engine
            
            # Calculate basic metrics
            returns = data['close'].pct_change().dropna()
            
            # Calculate performance metrics
            total_return = (data['close'].iloc[-1] / data['close'].iloc[0]) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = (returns.mean() * 252) / volatility if volatility > 0 else 0
            
            # Calculate drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            return {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': (returns > 0).mean()
            }
            
        except Exception as e:
            tprint_warning(f"Backtesting failed: {e}")
            return {}
    
    def _find_convergence_point(self) -> int:
        """Find the iteration where convergence occurred."""
        if not self.search_history:
            return 0
        
        # Simple convergence detection based on score improvement
        scores = [trial.get('score', 0.0) for trial in self.search_history]
        if len(scores) < 10:
            return len(scores)
        
        # Find point where improvement becomes minimal
        for i in range(10, len(scores)):
            recent_scores = scores[i-10:i]
            if np.std(recent_scores) < self.config.min_improvement_threshold:
                return i
        
        return len(scores)
    
    def get_search_summary(self) -> Dict[str, Any]:
        """Get summary of search results."""
        return {
            'total_iterations': len(self.search_history),
            'best_score': self.best_score,
            'convergence_iteration': self._find_convergence_point(),
            'search_efficiency': self._calculate_search_efficiency()
        }
    
    def _calculate_search_efficiency(self) -> float:
        """Calculate search efficiency metric."""
        if not self.search_history:
            return 0.0
        
        # Calculate improvement rate
        improvements = 0
        for i in range(1, len(self.search_history)):
            if self.search_history[i].get('score', 0.0) > self.search_history[i-1].get('score', 0.0):
                improvements += 1
        
        return improvements / max(1, len(self.search_history) - 1)


# Convenience function for quick strategy search
async def search_optimal_strategy(
    data: pd.DataFrame,
    search_space: Dict[str, Any],
    architecture_type: ArchitectureType = ArchitectureType.HYBRID_NEURAL_TREE,
    config: Optional[StrategySearchConfig] = None
) -> StrategySearchResult:
    """Search for optimal strategy with default configuration.
    
    Args:
        data: Input data for search
        search_space: Strategy search space
        architecture_type: Type of architecture to search for
        config: Optional search configuration
        
    Returns:
        StrategySearchResult with search results
    """
    if config is None:
        config = StrategySearchConfig()
    
    # Create unified configuration
    unified_config = UnifiedArchitectureConfig(
        architecture_type=architecture_type,
        optimization_mode=OptimizationMode.REGIME_AWARE
    )
    
    # Initialize optimizer
    optimizer = StrategySearchOptimizer(config)
    
    # Perform search
    result = await optimizer.search_strategies(data, search_space, unified_config)
    
    return result