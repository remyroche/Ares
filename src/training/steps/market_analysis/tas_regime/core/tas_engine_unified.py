"""
Advanced Tree Architecture Search Engine - Unified System Version

This module provides a tree architecture search engine that uses the unified system
for all operations while maintaining backward compatibility with the original interface.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime
import warnings

# Import unified system components
from src.utils.nas_tas.shared_utils import (
    UnifiedSearchEngine, SearchConfig, SearchResult, SearchStrategy, ArchitectureType,
    UnifiedMultiObjectiveOptimizer, UnifiedMultiObjectiveConfig, OptimizationAlgorithm,
    UnifiedEconomicEvaluator, EconomicEvaluationConfig,
    UnifiedRegimeDetector, RegimeDetectionConfig,
    UnifiedUtilities, UnifiedUtilityConfig,
    UnifiedConfig, create_unified_system,
    LegacyTASEngineAdapter  # For backward compatibility
)

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class SearchStrategy(Enum):
    """Available search strategies (legacy enum for compatibility)."""
    RANDOM = "random"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    ENHANCED_BAYESIAN = "enhanced_bayesian"
    ADAPTIVE_EVOLUTIONARY = "adaptive_evolutionary"
    HYBRID = "hybrid"

@dataclass
class TASConfig:
    """Configuration for TAS engine (legacy compatibility)."""
    search_strategy: SearchStrategy = SearchStrategy.EVOLUTIONARY
    population_size: int = 50
    max_iterations: int = 100
    elite_size: int = 5
    
    # Hardware optimization
    enable_parallel_processing: bool = True
    n_jobs: int = -1
    enable_gpu_acceleration: bool = False  # Trees don't typically use GPU
    memory_limit_gb: Optional[float] = None
    
    # Monitoring and logging
    enable_logging: bool = True
    log_level: str = 'INFO'
    save_intermediate_results: bool = True
    checkpoint_frequency: int = 10
    
    # Advanced features
    enable_early_stopping: bool = True
    early_stopping_patience: int = 20
    convergence_threshold: float = 0.01
    enable_adaptive_parameters: bool = True

class TASEngine:
    """
    Tree Architecture Search Engine - Now Using Unified System
    
    This engine now uses the unified system for all operations while maintaining
    the same interface for backward compatibility.
    """
    
    def __init__(self, config: Optional[TASConfig] = None):
        """Initialize the TAS Engine using unified system."""
        self.config = config or TASConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize unified system
        self._initialize_unified_system()
        
        # Performance tracking
        self.search_history = []
        self.performance_metrics = {}
        
        tprint_info("🚀 TAS Engine initialized using unified system")
        tprint_info(f"   Architecture type: {ArchitectureType.TREE.value}")
        tprint_info(f"   Search strategy: {SearchStrategy.EVOLUTIONARY.value}")
        tprint_info(f"   Max iterations: {self.config.max_iterations}")
        tprint_info(f"   Population size: {self.config.population_size}")
    
    def _initialize_unified_system(self):
        """Initialize unified system components."""
        try:
            # Create unified configuration
            unified_config = UnifiedConfig()
            
            # Configure for tree architecture search
            unified_config.search_config.architecture_type = ArchitectureType.TREE
            unified_config.search_config.search_strategy = SearchStrategy.EVOLUTIONARY
            unified_config.search_config.max_iterations = self.config.max_iterations
            unified_config.search_config.population_size = self.config.population_size
            unified_config.search_config.elite_size = self.config.elite_size
            unified_config.search_config.enable_parallel_processing = self.config.enable_parallel_processing
            unified_config.search_config.n_jobs = self.config.n_jobs
            unified_config.search_config.enable_gpu_acceleration = self.config.enable_gpu_acceleration
            unified_config.search_config.memory_limit_gb = self.config.memory_limit_gb
            
            # Configure optimization
            unified_config.optimization_config.algorithm = OptimizationAlgorithm.NSGA2
            unified_config.optimization_config.objectives = ['accuracy', 'interpretability', 'speed']
            unified_config.optimization_config.objective_weights = [0.4, 0.4, 0.2]
            unified_config.optimization_config.max_iterations = self.config.max_iterations
            unified_config.optimization_config.population_size = self.config.population_size
            
            # Configure evaluation
            unified_config.evaluation_config.architecture_type = ArchitectureType.TREE
            unified_config.evaluation_config.evaluation_types = ['economic_significance', 'performance_metrics']
            
            # Configure regime detection
            unified_config.regime_detection_config.architecture_type = ArchitectureType.TREE
            unified_config.regime_detection_config.method = 'tree_based'
            
            # Configure utilities
            unified_config.utility_config.enable_data_validation = True
            unified_config.utility_config.enable_memory_optimization = True
            unified_config.utility_config.enable_hardware_optimization = True
            unified_config.utility_config.memory_limit_gb = self.config.memory_limit_gb
            
            # Create unified system
            self.unified_system = create_unified_system(unified_config)
            
            # Get unified components
            self.search_engine = self.unified_system['search_engine']
            self.optimizer = self.unified_system['optimizer']
            self.evaluator = self.unified_system['evaluator']
            self.regime_detector = self.unified_system['regime_detector']
            self.utilities = self.unified_system['utilities']
            
            tprint_success("✅ Unified system initialized for TAS")
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize unified system: {e}")
            raise
    
    def search(self, search_space: Dict[str, Any], objective_function: Callable) -> Dict[str, Any]:
        """
        Perform tree architecture search.
        
        Args:
            search_space: Dictionary defining the search space
            objective_function: Function to evaluate architectures
            
        Returns:
            Dictionary containing search results
        """
        try:
            tprint_info("🔍 Starting tree architecture search...")
            start_time = time.time()
            
            # Use unified search engine
            result = self.search_engine.search(search_space, objective_function)
            
            # Convert unified result to legacy format
            legacy_result = {
                'best_parameters': result.best_architecture,
                'best_score': max(result.best_scores.values()) if result.best_scores else 0.0,
                'search_history': result.optimization_history,
                'convergence_achieved': result.convergence_achieved,
                'execution_time': result.execution_time,
                'success': result.success,
                'search_strategy': result.search_strategy.value,
                'architecture_type': result.architecture_type.value
            }
            
            # Update performance metrics
            self._update_performance_metrics(legacy_result)
            
            # Save search history
            self.search_history.append(legacy_result)
            
            tprint_success(f"✅ Tree architecture search completed in {legacy_result['execution_time']:.2f}s")
            tprint_info(f"   Best score: {legacy_result['best_score']:.4f}")
            tprint_info(f"   Convergence: {'Yes' if legacy_result['convergence_achieved'] else 'No'}")
            
            return legacy_result
            
        except Exception as e:
            tprint_error(f"❌ Tree architecture search failed: {e}")
            return {
                'best_parameters': {},
                'best_score': 0.0,
                'search_history': [],
                'convergence_achieved': False,
                'execution_time': 0.0,
                'success': False,
                'error_message': str(e)
            }
    
    def optimize(self, objective_functions: Dict[str, Callable], 
                parameter_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        Perform multi-objective optimization.
        
        Args:
            objective_functions: Dictionary mapping objective names to functions
            parameter_bounds: Dictionary defining parameter bounds
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            tprint_info("🎯 Starting multi-objective optimization...")
            start_time = time.time()
            
            # Use unified optimizer
            result = self.optimizer.optimize(objective_functions, parameter_bounds)
            
            # Convert unified result to legacy format
            legacy_result = {
                'best_parameters': result.best_parameters,
                'best_scores': result.best_scores,
                'pareto_frontier': [sol.parameters for sol in result.pareto_frontier],
                'optimization_history': result.optimization_history,
                'convergence_achieved': result.convergence_achieved,
                'execution_time': result.execution_time,
                'success': result.success,
                'algorithm': result.algorithm.value,
                'hypervolume': result.hypervolume
            }
            
            tprint_success(f"✅ Multi-objective optimization completed in {legacy_result['execution_time']:.2f}s")
            tprint_info(f"   Best scores: {legacy_result['best_scores']}")
            tprint_info(f"   Pareto frontier size: {len(legacy_result['pareto_frontier'])}")
            
            return legacy_result
            
        except Exception as e:
            tprint_error(f"❌ Multi-objective optimization failed: {e}")
            return {
                'best_parameters': {},
                'best_scores': {},
                'pareto_frontier': [],
                'optimization_history': [],
                'convergence_achieved': False,
                'execution_time': 0.0,
                'success': False,
                'error_message': str(e)
            }
    
    def evaluate(self, predictions: np.ndarray, market_data: pd.DataFrame, 
                returns: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate economic significance and trading viability.
        
        Args:
            predictions: Model predictions/regime labels
            market_data: Market data (OHLCV)
            returns: Market returns
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            tprint_info("💰 Starting economic evaluation...")
            start_time = time.time()
            
            # Use unified evaluator
            result = self.evaluator.evaluate(predictions, market_data, returns)
            
            # Convert unified result to legacy format
            legacy_result = {
                'economic_significance': result.economic_metrics.economic_significance,
                'trading_viability': result.economic_metrics.trading_viability,
                'overall_score': result.economic_metrics.overall_score,
                'sharpe_ratio': result.economic_metrics.sharpe_ratio,
                'max_drawdown': result.economic_metrics.max_drawdown,
                'volatility': result.economic_metrics.volatility,
                'return_ratio': result.economic_metrics.return_ratio,
                'sortino_ratio': result.economic_metrics.sortino_ratio,
                'calmar_ratio': result.economic_metrics.calmar_ratio,
                'information_ratio': result.economic_metrics.information_ratio,
                'trading_frequency': result.economic_metrics.trading_frequency,
                'win_rate': result.economic_metrics.win_rate,
                'profit_factor': result.economic_metrics.profit_factor,
                'evaluation_summary': result.evaluation_summary,
                'execution_time': result.total_evaluation_time,
                'success': result.success
            }
            
            tprint_success(f"✅ Economic evaluation completed in {legacy_result['execution_time']:.2f}s")
            tprint_info(f"   Economic significance: {legacy_result['economic_significance']:.4f}")
            tprint_info(f"   Trading viability: {legacy_result['trading_viability']:.4f}")
            tprint_info(f"   Overall score: {legacy_result['overall_score']:.4f}")
            
            return legacy_result
            
        except Exception as e:
            tprint_error(f"❌ Economic evaluation failed: {e}")
            return {
                'economic_significance': 0.0,
                'trading_viability': 0.0,
                'overall_score': 0.0,
                'execution_time': 0.0,
                'success': False,
                'error_message': str(e)
            }
    
    def detect_regimes(self, data: pd.DataFrame, features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect regimes in the data.
        
        Args:
            data: Input data for regime detection
            features: List of feature columns to use
            
        Returns:
            Dictionary containing regime detection results
        """
        try:
            tprint_info("🔍 Starting regime detection...")
            start_time = time.time()
            
            # Use unified regime detector
            result = self.regime_detector.detect_regimes(data, features)
            
            # Convert unified result to legacy format
            legacy_result = {
                'regime_labels': result.regime_labels,
                'n_regimes': result.n_regimes,
                'regime_infos': [
                    {
                        'regime_id': ri.regime_id,
                        'start_index': ri.start_index,
                        'end_index': ri.end_index,
                        'duration': ri.duration,
                        'stability': ri.stability,
                        'separation': ri.separation,
                        'characteristics': ri.characteristics
                    }
                    for ri in result.regime_infos
                ],
                'regime_quality_score': result.regime_quality_score,
                'stability_scores': result.stability_scores,
                'separation_scores': result.separation_scores,
                'detection_time': result.detection_time,
                'success': result.success
            }
            
            tprint_success(f"✅ Regime detection completed in {legacy_result['detection_time']:.2f}s")
            tprint_info(f"   Detected regimes: {legacy_result['n_regimes']}")
            tprint_info(f"   Quality score: {legacy_result['regime_quality_score']:.4f}")
            
            return legacy_result
            
        except Exception as e:
            tprint_error(f"❌ Regime detection failed: {e}")
            return {
                'regime_labels': np.array([]),
                'n_regimes': 0,
                'regime_infos': [],
                'regime_quality_score': 0.0,
                'detection_time': 0.0,
                'success': False,
                'error_message': str(e)
            }
    
    def validate_data(self, data: Union[pd.DataFrame, np.ndarray], 
                     data_type: str, architecture_type: str) -> Dict[str, Any]:
        """
        Validate data for the specified architecture type.
        
        Args:
            data: Data to validate
            data_type: Type of data
            architecture_type: Type of architecture
            
        Returns:
            Validation result dictionary
        """
        try:
            tprint_info(f"🔍 Validating {data_type} data for {architecture_type} architecture...")
            
            # Use unified utilities
            result = self.utilities.validate_data(data, data_type, architecture_type)
            
            tprint_success(f"✅ Data validation completed - Score: {result['data_quality_score']:.4f}")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Data validation failed: {e}")
            return {
                'is_valid': False,
                'warnings': [],
                'errors': [str(e)],
                'data_quality_score': 0.0,
                'recommendations': ['Fix validation errors before proceeding']
            }
    
    def optimize_data(self, data: Union[pd.DataFrame, np.ndarray], 
                     architecture_type: str, data_type: str) -> Union[pd.DataFrame, np.ndarray]:
        """
        Optimize data for the specified architecture type.
        
        Args:
            data: Data to optimize
            architecture_type: Type of architecture
            data_type: Type of data
            
        Returns:
            Optimized data
        """
        try:
            tprint_info(f"⚡ Optimizing data for {architecture_type} architecture...")
            
            # Use unified utilities
            optimized_data = self.utilities.optimize_data(data, architecture_type, data_type)
            
            tprint_success("✅ Data optimization completed")
            
            return optimized_data
            
        except Exception as e:
            tprint_error(f"❌ Data optimization failed: {e}")
            return data  # Return original data on failure
    
    def _update_performance_metrics(self, result: Dict[str, Any]):
        """Update performance metrics."""
        if 'search_performance' not in self.performance_metrics:
            self.performance_metrics['search_performance'] = {
                'total_searches': 0,
                'successful_searches': 0,
                'avg_execution_time': 0.0,
                'avg_best_score': 0.0,
                'convergence_rate': 0.0
            }
        
        metrics = self.performance_metrics['search_performance']
        metrics['total_searches'] += 1
        
        if result['success']:
            metrics['successful_searches'] += 1
        
        # Update averages
        total = metrics['total_searches']
        successful = metrics['successful_searches']
        
        if successful > 0:
            metrics['avg_execution_time'] = (
                (metrics['avg_execution_time'] * (successful - 1) + result['execution_time']) / successful
            )
            
            metrics['avg_best_score'] = (
                (metrics['avg_best_score'] * (successful - 1) + result['best_score']) / successful
            )
            
            metrics['convergence_rate'] = successful / total if total > 0 else 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {
            'search_performance': self.performance_metrics.get('search_performance', {}),
            'search_history_length': len(self.search_history),
            'unified_system_summary': self.unified_system['config'].get_performance_summary() if hasattr(self.unified_system['config'], 'get_performance_summary') else {}
        }
        
        return summary
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'engine_type': 'TASEngine',
            'architecture_type': 'tree',
            'unified_system_version': '1.0.0',
            'search_strategy': self.config.search_strategy.value,
            'max_iterations': self.config.max_iterations,
            'population_size': self.config.population_size,
            'components': {
                'search_engine': 'UnifiedSearchEngine',
                'optimizer': 'UnifiedMultiObjectiveOptimizer',
                'evaluator': 'UnifiedEconomicEvaluator',
                'regime_detector': 'UnifiedRegimeDetector',
                'utilities': 'UnifiedUtilities'
            }
        }

# Convenience functions
def create_tas_engine(config: Optional[TASConfig] = None) -> TASEngine:
    """Create a TAS engine with specified configuration."""
    return TASEngine(config)

def quick_tas_search(search_space: Dict[str, Any], 
                    objective_function: Callable,
                    max_iterations: int = 100) -> Dict[str, Any]:
    """Quick TAS search using default configuration."""
    config = TASConfig(max_iterations=max_iterations)
    engine = TASEngine(config)
    return engine.search(search_space, objective_function)

# Export main classes and functions
__all__ = [
    'TASEngine',
    'TASConfig',
    'SearchStrategy',
    'create_tas_engine',
    'quick_tas_search'
]