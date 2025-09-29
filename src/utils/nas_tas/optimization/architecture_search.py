"""
Architecture Search Optimizer

This module provides unified architecture search capabilities that consolidate
search logic previously scattered across NAS and TAS implementations.
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
class ArchitectureSearchConfig:
    """Configuration for architecture search optimization."""
    
    # Search parameters
    max_iterations: int = 100
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    
    # Evaluation settings
    evaluation_metric: str = "f1_score"
    validation_split: float = 0.2
    cv_folds: int = 5
    
    # Optimization settings
    enable_early_stopping: bool = True
    early_stopping_patience: int = 20
    min_improvement_threshold: float = 0.001
    
    # Parallel processing
    enable_parallel_processing: bool = True
    max_workers: int = 4
    
    # Output settings
    save_results: bool = True
    results_path: str = "architecture_search_results"
    enable_visualization: bool = True


@dataclass
class ArchitectureSearchResult:
    """Result from architecture search."""
    
    # Search results
    best_architecture: Dict[str, Any]
    best_score: float
    search_history: List[Dict[str, Any]]
    
    # Performance metrics
    total_iterations: int
    convergence_iteration: int
    search_time: float
    
    # Architecture details
    architecture_type: ArchitectureType
    complexity_score: float
    efficiency_score: float
    
    # Metadata
    search_timestamp: datetime
    configuration: Dict[str, Any]


class ArchitectureSearchOptimizer:
    """
    Unified architecture search optimizer for both NAS and TAS systems.
    
    This class consolidates architecture search logic that was previously
    scattered across different implementations, providing a unified interface
    for both neural and tree architecture search.
    """
    
    def __init__(self, config: ArchitectureSearchConfig):
        """Initialize architecture search optimizer.
        
        Args:
            config: Architecture search configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize search engines
        self.nas_engine = None
        self.tas_engine = None
        
        # Search state
        self.search_history = []
        self.best_architecture = None
        self.best_score = -np.inf
        
        tprint_info("Architecture Search Optimizer initialized")
    
    def initialize_engines(self, unified_config: UnifiedArchitectureConfig):
        """Initialize NAS and TAS engines based on configuration.
        
        Args:
            unified_config: Unified architecture configuration
        """
        try:
            # Initialize NAS engine if needed
            if unified_config.architecture_type in [ArchitectureType.NEURAL_ONLY, ArchitectureType.HYBRID_NEURAL_TREE]:
                self.nas_engine = NASEngine(unified_config.__dict__)
                tprint_success("NAS engine initialized")
            
            # Initialize TAS engine if needed
            if unified_config.architecture_type in [ArchitectureType.TREE_ONLY, ArchitectureType.HYBRID_NEURAL_TREE]:
                self.tas_engine = TASEngine(unified_config.__dict__)
                tprint_success("TAS engine initialized")
            
        except Exception as e:
            tprint_error(f"Engine initialization failed: {e}")
            raise
    
    async def search_architectures(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        unified_config: UnifiedArchitectureConfig
    ) -> ArchitectureSearchResult:
        """Search for optimal architectures using unified approach.
        
        Args:
            data: Input data for architecture search
            search_space: Architecture search space
            unified_config: Unified architecture configuration
            
        Returns:
            ArchitectureSearchResult with search results
        """
        start_time = datetime.now()
        tprint_info("Starting unified architecture search")
        
        try:
            # Initialize engines
            self.initialize_engines(unified_config)
            
            # Perform search based on architecture type
            if unified_config.architecture_type == ArchitectureType.NEURAL_ONLY:
                result = await self._search_neural_architectures(data, search_space)
            elif unified_config.architecture_type == ArchitectureType.TREE_ONLY:
                result = await self._search_tree_architectures(data, search_space)
            elif unified_config.architecture_type == ArchitectureType.HYBRID_NEURAL_TREE:
                result = await self._search_hybrid_architectures(data, search_space)
            else:
                raise ValueError(f"Unsupported architecture type: {unified_config.architecture_type}")
            
            # Calculate search metrics
            search_time = (datetime.now() - start_time).total_seconds()
            result.search_time = search_time
            result.total_iterations = len(self.search_history)
            result.convergence_iteration = self._find_convergence_point()
            
            tprint_success(f"Architecture search completed in {search_time:.2f}s")
            tprint_info(f"Best score: {result.best_score:.4f}")
            
            return result
            
        except Exception as e:
            tprint_error(f"Architecture search failed: {e}")
            raise
    
    async def _search_neural_architectures(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any]
    ) -> ArchitectureSearchResult:
        """Search for neural architectures using NAS engine."""
        tprint_info("Searching neural architectures")
        
        if not self.nas_engine:
            raise ValueError("NAS engine not initialized")
        
        # Use NAS engine for architecture search
        nas_results = self.nas_engine.search_architectures(
            data=data,
            search_space=search_space,
            optimization_method="bayesian_tpe",
            n_trials=self.config.max_iterations
        )
        
        return ArchitectureSearchResult(
            best_architecture=nas_results.get('best_architecture', {}),
            best_score=nas_results.get('best_score', 0.0),
            search_history=nas_results.get('trials', []),
            total_iterations=0,
            convergence_iteration=0,
            search_time=0.0,
            architecture_type=ArchitectureType.NEURAL_ONLY,
            complexity_score=0.0,
            efficiency_score=0.0,
            search_timestamp=datetime.now(),
            configuration=self.config.__dict__
        )
    
    async def _search_tree_architectures(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any]
    ) -> ArchitectureSearchResult:
        """Search for tree architectures using TAS engine."""
        tprint_info("Searching tree architectures")
        
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
        
        return ArchitectureSearchResult(
            best_architecture=tas_results.get('best_strategy', {}),
            best_score=tas_results.get('best_score', 0.0),
            search_history=tas_results.get('trials', []),
            total_iterations=0,
            convergence_iteration=0,
            search_time=0.0,
            architecture_type=ArchitectureType.TREE_ONLY,
            complexity_score=0.0,
            efficiency_score=0.0,
            search_timestamp=datetime.now(),
            configuration=self.config.__dict__
        )
    
    async def _search_hybrid_architectures(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any]
    ) -> ArchitectureSearchResult:
        """Search for hybrid architectures combining neural and tree components."""
        tprint_info("Searching hybrid architectures")
        
        if not self.nas_engine or not self.tas_engine:
            raise ValueError("Both NAS and TAS engines required for hybrid search")
        
        # Search neural architectures
        nas_results = await self._search_neural_architectures(data, search_space)
        
        # Search tree architectures
        tas_results = await self._search_tree_architectures(data, search_space)
        
        # Combine results for hybrid architecture
        hybrid_architecture = {
            'neural_config': nas_results.best_architecture,
            'tree_config': tas_results.best_architecture,
            'ensemble_method': 'weighted_average',
            'neural_weight': 0.6,
            'tree_weight': 0.4
        }
        
        # Calculate hybrid score
        hybrid_score = (nas_results.best_score * 0.6 + tas_results.best_score * 0.4)
        
        return ArchitectureSearchResult(
            best_architecture=hybrid_architecture,
            best_score=hybrid_score,
            search_history=nas_results.search_history + tas_results.search_history,
            total_iterations=0,
            convergence_iteration=0,
            search_time=0.0,
            architecture_type=ArchitectureType.HYBRID_NEURAL_TREE,
            complexity_score=0.0,
            efficiency_score=0.0,
            search_timestamp=datetime.now(),
            configuration=self.config.__dict__
        )
    
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


# Convenience function for quick architecture search
async def search_optimal_architecture(
    data: pd.DataFrame,
    search_space: Dict[str, Any],
    architecture_type: ArchitectureType = ArchitectureType.HYBRID_NEURAL_TREE,
    config: Optional[ArchitectureSearchConfig] = None
) -> ArchitectureSearchResult:
    """Search for optimal architecture with default configuration.
    
    Args:
        data: Input data for search
        search_space: Architecture search space
        architecture_type: Type of architecture to search for
        config: Optional search configuration
        
    Returns:
        ArchitectureSearchResult with search results
    """
    if config is None:
        config = ArchitectureSearchConfig()
    
    # Create unified configuration
    unified_config = UnifiedArchitectureConfig(
        architecture_type=architecture_type,
        optimization_mode=OptimizationMode.REGIME_AWARE
    )
    
    # Initialize optimizer
    optimizer = ArchitectureSearchOptimizer(config)
    
    # Perform search
    result = await optimizer.search_architectures(data, search_space, unified_config)
    
    return result