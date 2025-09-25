"""
Hybrid NAS-TAS Regime Detector - Unified System Version

This module provides a hybrid regime detector that uses the unified system
for all operations while maintaining backward compatibility with the original interface.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import time
from datetime import datetime
from dataclasses import dataclass
import warnings

# Import unified system components
from src.utils.nas_tas.shared_utils import (
    UnifiedSearchEngine, SearchConfig, SearchResult, SearchStrategy, ArchitectureType,
    UnifiedMultiObjectiveOptimizer, UnifiedMultiObjectiveConfig, OptimizationAlgorithm,
    UnifiedEconomicEvaluator, EconomicEvaluationConfig,
    UnifiedRegimeDetector, RegimeDetectionConfig,
    UnifiedUtilities, UnifiedUtilityConfig,
    UnifiedConfig, create_unified_system
)

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class HybridRegimeDetectorConfig:
    """Configuration for hybrid regime detector."""
    
    # Core detection parameters
    n_regimes: int = 3
    min_regime_duration: int = 10
    
    # Architecture weights
    nas_weight: float = 0.5
    tas_weight: float = 0.5
    
    # Clustering parameters
    clustering_algorithm: str = "kmeans"
    n_clusters: int = 3
    random_state: int = 42
    
    # Hardware optimization
    enable_parallel_processing: bool = True
    n_jobs: int = -1
    memory_limit_gb: Optional[float] = None
    
    # Monitoring and logging
    enable_logging: bool = True
    log_level: str = 'INFO'
    save_intermediate_results: bool = True

class HybridRegimeDetector:
    """
    Hybrid NAS-TAS Regime Detector - Now Using Unified System
    
    This detector combines neural and tree architecture search approaches
    using the unified system for all operations.
    """
    
    def __init__(self, config: Optional[HybridRegimeDetectorConfig] = None):
        """Initialize the Hybrid Regime Detector using unified system."""
        self.config = config or HybridRegimeDetectorConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize unified system
        self._initialize_unified_system()
        
        # Performance tracking
        self.detection_history = []
        self.performance_metrics = {}
        
        tprint_info("🚀 Hybrid Regime Detector initialized using unified system")
        tprint_info(f"   Architecture type: {ArchitectureType.HYBRID.value}")
        tprint_info(f"   Target regimes: {self.config.n_regimes}")
        tprint_info(f"   NAS weight: {self.config.nas_weight}")
        tprint_info(f"   TAS weight: {self.config.tas_weight}")
    
    def _initialize_unified_system(self):
        """Initialize unified system components."""
        try:
            # Create unified configuration
            unified_config = UnifiedConfig()
            
            # Configure for hybrid architecture search
            unified_config.search_config.architecture_type = ArchitectureType.HYBRID
            unified_config.search_config.search_strategy = SearchStrategy.HYBRID
            unified_config.search_config.max_iterations = 100
            unified_config.search_config.population_size = 50
            unified_config.search_config.enable_parallel_processing = self.config.enable_parallel_processing
            unified_config.search_config.n_jobs = self.config.n_jobs
            unified_config.search_config.memory_limit_gb = self.config.memory_limit_gb
            
            # Configure optimization
            unified_config.optimization_config.algorithm = OptimizationAlgorithm.HYBRID
            unified_config.optimization_config.objectives = ['accuracy', 'efficiency', 'robustness', 'interpretability']
            unified_config.optimization_config.objective_weights = [0.3, 0.25, 0.25, 0.2]
            
            # Configure evaluation
            unified_config.evaluation_config.architecture_type = ArchitectureType.HYBRID
            unified_config.evaluation_config.evaluation_types = ['economic_significance', 'trading_viability', 'performance_metrics']
            
            # Configure regime detection
            unified_config.regime_detection_config.architecture_type = ArchitectureType.HYBRID
            unified_config.regime_detection_config.method = 'hybrid'
            unified_config.regime_detection_config.n_regimes = self.config.n_regimes
            unified_config.regime_detection_config.min_regime_duration = self.config.min_regime_duration
            unified_config.regime_detection_config.clustering_algorithm = self.config.clustering_algorithm
            unified_config.regime_detection_config.n_clusters = self.config.n_clusters
            unified_config.regime_detection_config.random_state = self.config.random_state
            
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
            
            tprint_success("✅ Unified system initialized for Hybrid Regime Detector")
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize unified system: {e}")
            raise
    
    def detect_regimes(self, data: pd.DataFrame, features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect regimes using hybrid NAS-TAS approach.
        
        Args:
            data: Input data for regime detection
            features: List of feature columns to use
            
        Returns:
            Dictionary containing regime detection results
        """
        try:
            tprint_info("🔍 Starting hybrid regime detection...")
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
                'success': result.success,
                'method': result.method.value,
                'architecture_type': result.architecture_type.value
            }
            
            # Update performance metrics
            self._update_performance_metrics(legacy_result)
            
            # Save detection history
            self.detection_history.append(legacy_result)
            
            tprint_success(f"✅ Hybrid regime detection completed in {legacy_result['detection_time']:.2f}s")
            tprint_info(f"   Detected regimes: {legacy_result['n_regimes']}")
            tprint_info(f"   Quality score: {legacy_result['regime_quality_score']:.4f}")
            
            return legacy_result
            
        except Exception as e:
            tprint_error(f"❌ Hybrid regime detection failed: {e}")
            return {
                'regime_labels': np.array([]),
                'n_regimes': 0,
                'regime_infos': [],
                'regime_quality_score': 0.0,
                'detection_time': 0.0,
                'success': False,
                'error_message': str(e)
            }
    
    def search_architectures(self, search_space: Dict[str, Any], objective_function: Callable) -> Dict[str, Any]:
        """
        Search for optimal architectures using hybrid approach.
        
        Args:
            search_space: Dictionary defining the search space
            objective_function: Function to evaluate architectures
            
        Returns:
            Dictionary containing search results
        """
        try:
            tprint_info("🔍 Starting hybrid architecture search...")
            start_time = time.time()
            
            # Use unified search engine
            result = self.search_engine.search(search_space, objective_function)
            
            # Convert unified result to legacy format
            legacy_result = {
                'best_architecture': result.best_architecture,
                'best_score': max(result.best_scores.values()) if result.best_scores else 0.0,
                'optimization_history': result.optimization_history,
                'convergence_achieved': result.convergence_achieved,
                'execution_time': result.execution_time,
                'success': result.success,
                'search_strategy': result.search_strategy.value,
                'architecture_type': result.architecture_type.value
            }
            
            tprint_success(f"✅ Hybrid architecture search completed in {legacy_result['execution_time']:.2f}s")
            tprint_info(f"   Best score: {legacy_result['best_score']:.4f}")
            tprint_info(f"   Convergence: {'Yes' if legacy_result['convergence_achieved'] else 'No'}")
            
            return legacy_result
            
        except Exception as e:
            tprint_error(f"❌ Hybrid architecture search failed: {e}")
            return {
                'best_architecture': {},
                'best_score': 0.0,
                'optimization_history': [],
                'convergence_achieved': False,
                'execution_time': 0.0,
                'success': False,
                'error_message': str(e)
            }
    
    def optimize_objectives(self, objective_functions: Dict[str, Callable], 
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
            tprint_info("🎯 Starting hybrid multi-objective optimization...")
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
            
            tprint_success(f"✅ Hybrid multi-objective optimization completed in {legacy_result['execution_time']:.2f}s")
            tprint_info(f"   Best scores: {legacy_result['best_scores']}")
            tprint_info(f"   Pareto frontier size: {len(legacy_result['pareto_frontier'])}")
            
            return legacy_result
            
        except Exception as e:
            tprint_error(f"❌ Hybrid multi-objective optimization failed: {e}")
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
    
    def evaluate_performance(self, predictions: np.ndarray, market_data: pd.DataFrame, 
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
            tprint_info("💰 Starting hybrid economic evaluation...")
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
            
            tprint_success(f"✅ Hybrid economic evaluation completed in {legacy_result['execution_time']:.2f}s")
            tprint_info(f"   Economic significance: {legacy_result['economic_significance']:.4f}")
            tprint_info(f"   Trading viability: {legacy_result['trading_viability']:.4f}")
            tprint_info(f"   Overall score: {legacy_result['overall_score']:.4f}")
            
            return legacy_result
            
        except Exception as e:
            tprint_error(f"❌ Hybrid economic evaluation failed: {e}")
            return {
                'economic_significance': 0.0,
                'trading_viability': 0.0,
                'overall_score': 0.0,
                'execution_time': 0.0,
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
        if 'detection_performance' not in self.performance_metrics:
            self.performance_metrics['detection_performance'] = {
                'total_detections': 0,
                'successful_detections': 0,
                'avg_execution_time': 0.0,
                'avg_quality_score': 0.0,
                'success_rate': 0.0
            }
        
        metrics = self.performance_metrics['detection_performance']
        metrics['total_detections'] += 1
        
        if result['success']:
            metrics['successful_detections'] += 1
        
        # Update averages
        total = metrics['total_detections']
        successful = metrics['successful_detections']
        
        if successful > 0:
            metrics['avg_execution_time'] = (
                (metrics['avg_execution_time'] * (successful - 1) + result['detection_time']) / successful
            )
            
            metrics['avg_quality_score'] = (
                (metrics['avg_quality_score'] * (successful - 1) + result['regime_quality_score']) / successful
            )
            
            metrics['success_rate'] = successful / total if total > 0 else 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {
            'detection_performance': self.performance_metrics.get('detection_performance', {}),
            'detection_history_length': len(self.detection_history),
            'unified_system_summary': self.unified_system['config'].get_performance_summary() if hasattr(self.unified_system['config'], 'get_performance_summary') else {}
        }
        
        return summary
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'detector_type': 'HybridRegimeDetector',
            'architecture_type': 'hybrid',
            'unified_system_version': '1.0.0',
            'n_regimes': self.config.n_regimes,
            'nas_weight': self.config.nas_weight,
            'tas_weight': self.config.tas_weight,
            'components': {
                'search_engine': 'UnifiedSearchEngine',
                'optimizer': 'UnifiedMultiObjectiveOptimizer',
                'evaluator': 'UnifiedEconomicEvaluator',
                'regime_detector': 'UnifiedRegimeDetector',
                'utilities': 'UnifiedUtilities'
            }
        }

# Convenience functions
def create_hybrid_regime_detector(config: Optional[HybridRegimeDetectorConfig] = None) -> HybridRegimeDetector:
    """Create a hybrid regime detector with specified configuration."""
    return HybridRegimeDetector(config)

def quick_hybrid_regime_detection(data: pd.DataFrame, 
                                features: Optional[List[str]] = None,
                                n_regimes: int = 3) -> Dict[str, Any]:
    """Quick hybrid regime detection using default configuration."""
    config = HybridRegimeDetectorConfig(n_regimes=n_regimes)
    detector = HybridRegimeDetector(config)
    return detector.detect_regimes(data, features)

# Export main classes and functions
__all__ = [
    'HybridRegimeDetector',
    'HybridRegimeDetectorConfig',
    'create_hybrid_regime_detector',
    'quick_hybrid_regime_detection'
]