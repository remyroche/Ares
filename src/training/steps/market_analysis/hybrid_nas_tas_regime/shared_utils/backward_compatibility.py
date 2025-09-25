"""
Backward Compatibility Layer for NAS and TAS Systems

This module provides backward compatibility for existing NAS and TAS components
to work with the new unified system. It includes adapters and wrappers that
maintain the original interfaces while using the unified components underneath.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import logging
import warnings
from functools import wraps

# Import unified components
from .unified_search_engine import (
    UnifiedSearchEngine, SearchConfig, SearchResult, SearchStrategy, ArchitectureType
)
from .unified_multi_objective_optimizer import (
    UnifiedMultiObjectiveOptimizer, UnifiedMultiObjectiveConfig, UnifiedOptimizationResult, OptimizationAlgorithm
)
from .unified_economic_evaluator import (
    UnifiedEconomicEvaluator, EconomicEvaluationConfig, EconomicEvaluationResult
)
from .unified_regime_detector import (
    UnifiedRegimeDetector, RegimeDetectionConfig, RegimeDetectionResult
)
from .unified_utilities import (
    UnifiedUtilities, UnifiedUtilityConfig
)
from .unified_config import (
    UnifiedConfig, config_manager
)

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Legacy imports for backward compatibility
try:
    # NAS legacy imports
    from src.training.steps.market_analysis.nas_regime.core.enhanced_nas_engine import EnhancedNASEngine
    from src.training.steps.market_analysis.nas_regime.core.nas_search import NASSearch
    from src.training.steps.market_analysis.nas_regime.optimization.multi_objective_optimizer import TradingMultiObjectiveOptimizer as NASOptimizer
    from src.training.steps.market_analysis.nas_regime.evaluation.economic_evaluator import EconomicEvaluator as NASEvaluator
    from src.training.steps.market_analysis.nas_clustering.core.nas_regime_analyzer import NASRegimeAnalyzer
    
    # TAS legacy imports
    from src.training.steps.market_analysis.tas_regime.core.tas_engine import TASEngine
    from src.training.steps.market_analysis.tas_regime.search.advanced_search import AdvancedSearch
    from src.training.steps.market_analysis.tas_regime.search.multi_objective_search import MultiObjectiveSearch
    from src.training.steps.market_analysis.tas_regime.evaluation.tree_evaluator import TreeEvaluator
    from src.training.steps.market_analysis.tas_regime.regime_analysis.tree_regime_analyzer import TreeRegimeAnalyzer
    
    # Hybrid legacy imports
    from src.training.steps.market_analysis.hybrid_nas_tas_regime.core.hybrid_regime_detector import HybridRegimeDetector
    from src.training.steps.market_analysis.hybrid_nas_tas_regime.core.multi_objective_optimizer import TradingMultiObjectiveOptimizer as HybridOptimizer
    from src.training.steps.market_analysis.hybrid_nas_tas_regime.evaluation.economic_evaluator import EconomicEvaluator as HybridEvaluator
    
    LEGACY_IMPORTS_AVAILABLE = True
    
except ImportError as e:
    logger.warning(f"Legacy imports not available: {e}")
    LEGACY_IMPORTS_AVAILABLE = False

def deprecated_warning(message: str):
    """Decorator to show deprecation warnings."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated. {message}",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator

class LegacyNASEngineAdapter:
    """Adapter for legacy NAS engine to use unified search engine."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize adapter."""
        self.config = config or {}
        
        # Convert legacy config to unified config
        unified_config = self._convert_legacy_config()
        
        # Initialize unified engine
        self.unified_engine = UnifiedSearchEngine(unified_config)
        
        print("🔄 NAS Engine Adapter initialized - using unified search engine")
    
    def _convert_legacy_config(self) -> SearchConfig:
        """Convert legacy NAS config to unified config."""
        return SearchConfig(
            architecture_type=ArchitectureType.NEURAL,
            search_strategy=SearchStrategy.ENHANCED_BAYESIAN,
            max_iterations=self.config.get('max_iterations', 100),
            population_size=self.config.get('population_size', 50),
            elite_size=self.config.get('elite_size', 5),
            bayesian_config=self.config.get('bayesian_config', {}),
            evolutionary_config=self.config.get('evolutionary_config', {}),
            enable_parallel_processing=self.config.get('enable_parallel_processing', True),
            n_jobs=self.config.get('n_jobs', -1)
        )
    
    @deprecated_warning("Use UnifiedSearchEngine directly for new code")
    def search(self, search_space: Dict[str, Any], objective_function: Callable) -> Dict[str, Any]:
        """Legacy search method."""
        result = self.unified_engine.search(search_space, objective_function)
        
        # Convert unified result to legacy format
        return {
            'best_architecture': result.best_architecture,
            'best_score': max(result.best_scores.values()) if result.best_scores else 0.0,
            'optimization_history': result.optimization_history,
            'convergence_achieved': result.convergence_achieved,
            'execution_time': result.execution_time,
            'success': result.success
        }
    
    @deprecated_warning("Use UnifiedSearchEngine directly for new code")
    def optimize(self, *args, **kwargs) -> Dict[str, Any]:
        """Legacy optimize method."""
        return self.search(*args, **kwargs)

class LegacyTASEngineAdapter:
    """Adapter for legacy TAS engine to use unified search engine."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize adapter."""
        self.config = config or {}
        
        # Convert legacy config to unified config
        unified_config = self._convert_legacy_config()
        
        # Initialize unified engine
        self.unified_engine = UnifiedSearchEngine(unified_config)
        
        print("🔄 TAS Engine Adapter initialized - using unified search engine")
    
    def _convert_legacy_config(self) -> SearchConfig:
        """Convert legacy TAS config to unified config."""
        return SearchConfig(
            architecture_type=ArchitectureType.TREE,
            search_strategy=SearchStrategy.EVOLUTIONARY,
            max_iterations=self.config.get('max_iterations', 100),
            population_size=self.config.get('population_size', 50),
            elite_size=self.config.get('elite_size', 5),
            evolutionary_config=self.config.get('evolutionary_config', {}),
            enable_parallel_processing=self.config.get('enable_parallel_processing', True),
            n_jobs=self.config.get('n_jobs', -1)
        )
    
    @deprecated_warning("Use UnifiedSearchEngine directly for new code")
    def search(self, search_space: Dict[str, Any], objective_function: Callable) -> Dict[str, Any]:
        """Legacy search method."""
        result = self.unified_engine.search(search_space, objective_function)
        
        # Convert unified result to legacy format
        return {
            'best_parameters': result.best_architecture,
            'best_score': max(result.best_scores.values()) if result.best_scores else 0.0,
            'search_history': result.optimization_history,
            'convergence_achieved': result.convergence_achieved,
            'execution_time': result.execution_time,
            'success': result.success
        }
    
    @deprecated_warning("Use UnifiedSearchEngine directly for new code")
    def optimize(self, *args, **kwargs) -> Dict[str, Any]:
        """Legacy optimize method."""
        return self.search(*args, **kwargs)

class LegacyMultiObjectiveOptimizerAdapter:
    """Adapter for legacy multi-objective optimizer."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize adapter."""
        self.config = config or {}
        
        # Convert legacy config to unified config
        unified_config = self._convert_legacy_config()
        
        # Initialize unified optimizer
        self.unified_optimizer = UnifiedMultiObjectiveOptimizer(unified_config)
        
        print("🔄 Multi-Objective Optimizer Adapter initialized - using unified optimizer")
    
    def _convert_legacy_config(self) -> UnifiedMultiObjectiveConfig:
        """Convert legacy config to unified config."""
        return UnifiedMultiObjectiveConfig(
            algorithm=OptimizationAlgorithm.NSGA2,
            objectives=self.config.get('objectives', ['accuracy', 'efficiency', 'profitability']),
            objective_weights=self.config.get('objective_weights', [0.4, 0.3, 0.3]),
            max_iterations=self.config.get('max_iterations', 100),
            population_size=self.config.get('population_size', 50),
            convergence_threshold=self.config.get('convergence_threshold', 0.01),
            convergence_patience=self.config.get('convergence_patience', 20)
        )
    
    @deprecated_warning("Use UnifiedMultiObjectiveOptimizer directly for new code")
    def optimize(self, objective_functions: Dict[str, Callable], 
                parameter_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Legacy optimize method."""
        result = self.unified_optimizer.optimize(objective_functions, parameter_bounds)
        
        # Convert unified result to legacy format
        return {
            'best_parameters': result.best_parameters,
            'best_scores': result.best_scores,
            'pareto_frontier': [sol.parameters for sol in result.pareto_frontier],
            'optimization_history': result.optimization_history,
            'convergence_achieved': result.convergence_achieved,
            'execution_time': result.execution_time,
            'success': result.success
        }

class LegacyEconomicEvaluatorAdapter:
    """Adapter for legacy economic evaluator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize adapter."""
        self.config = config or {}
        
        # Convert legacy config to unified config
        unified_config = self._convert_legacy_config()
        
        # Initialize unified evaluator
        self.unified_evaluator = UnifiedEconomicEvaluator(unified_config)
        
        print("🔄 Economic Evaluator Adapter initialized - using unified evaluator")
    
    def _convert_legacy_config(self) -> EconomicEvaluationConfig:
        """Convert legacy config to unified config."""
        return EconomicEvaluationConfig(
            evaluation_types=self.config.get('evaluation_types', ['economic_significance', 'trading_viability']),
            significance_threshold=self.config.get('significance_threshold', 0.05),
            min_regime_duration=self.config.get('min_regime_duration', 10),
            risk_free_rate=self.config.get('risk_free_rate', 0.02),
            enable_logging=self.config.get('enable_logging', True)
        )
    
    @deprecated_warning("Use UnifiedEconomicEvaluator directly for new code")
    def evaluate(self, predictions: np.ndarray, market_data: pd.DataFrame, 
                returns: np.ndarray) -> Dict[str, Any]:
        """Legacy evaluate method."""
        result = self.unified_evaluator.evaluate(predictions, market_data, returns)
        
        # Convert unified result to legacy format
        return {
            'economic_significance': result.economic_metrics.economic_significance,
            'trading_viability': result.economic_metrics.trading_viability,
            'overall_score': result.economic_metrics.overall_score,
            'sharpe_ratio': result.economic_metrics.sharpe_ratio,
            'max_drawdown': result.economic_metrics.max_drawdown,
            'volatility': result.economic_metrics.volatility,
            'evaluation_summary': result.evaluation_summary,
            'execution_time': result.total_evaluation_time,
            'success': result.success
        }

class LegacyRegimeDetectorAdapter:
    """Adapter for legacy regime detector."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize adapter."""
        self.config = config or {}
        
        # Convert legacy config to unified config
        unified_config = self._convert_legacy_config()
        
        # Initialize unified detector
        self.unified_detector = UnifiedRegimeDetector(unified_config)
        
        print("🔄 Regime Detector Adapter initialized - using unified detector")
    
    def _convert_legacy_config(self) -> RegimeDetectionConfig:
        """Convert legacy config to unified config."""
        return RegimeDetectionConfig(
            method=self.config.get('method', 'hybrid'),
            n_regimes=self.config.get('n_regimes', 3),
            min_regime_duration=self.config.get('min_regime_duration', 10),
            clustering_algorithm=self.config.get('clustering_algorithm', 'kmeans'),
            n_clusters=self.config.get('n_clusters', 3),
            random_state=self.config.get('random_state', 42)
        )
    
    @deprecated_warning("Use UnifiedRegimeDetector directly for new code")
    def detect_regimes(self, data: pd.DataFrame, features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Legacy detect_regimes method."""
        result = self.unified_detector.detect_regimes(data, features)
        
        # Convert unified result to legacy format
        return {
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
            'detection_time': result.detection_time,
            'success': result.success
        }

class LegacyUtilitiesAdapter:
    """Adapter for legacy utilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize adapter."""
        self.config = config or {}
        
        # Convert legacy config to unified config
        unified_config = self._convert_legacy_config()
        
        # Initialize unified utilities
        self.unified_utilities = UnifiedUtilities(unified_config)
        
        print("🔄 Utilities Adapter initialized - using unified utilities")
    
    def _convert_legacy_config(self) -> UnifiedUtilityConfig:
        """Convert legacy config to unified config."""
        return UnifiedUtilityConfig(
            enable_data_validation=self.config.get('enable_data_validation', True),
            enable_memory_optimization=self.config.get('enable_memory_optimization', True),
            enable_hardware_optimization=self.config.get('enable_hardware_optimization', True),
            memory_limit_gb=self.config.get('memory_limit_gb'),
            strict_validation=self.config.get('strict_validation', False),
            auto_fix_issues=self.config.get('auto_fix_issues', True),
            validation_threshold=self.config.get('validation_threshold', 0.95),
            enable_logging=self.config.get('enable_logging', True)
        )
    
    @deprecated_warning("Use UnifiedUtilities directly for new code")
    def validate_data(self, data: Union[pd.DataFrame, np.ndarray], 
                     data_type: str, architecture_type: str) -> Dict[str, Any]:
        """Legacy validate_data method."""
        from .unified_utilities import DataType, ArchitectureType as UnifiedArchitectureType
        
        # Convert string types to enums
        data_type_enum = DataType(data_type) if isinstance(data_type, str) else data_type
        arch_type_enum = UnifiedArchitectureType(architecture_type) if isinstance(architecture_type, str) else architecture_type
        
        return self.unified_utilities.validate_data(data, data_type_enum, arch_type_enum)
    
    @deprecated_warning("Use UnifiedUtilities directly for new code")
    def optimize_data(self, data: Union[pd.DataFrame, np.ndarray], 
                     architecture_type: str, data_type: str) -> Union[pd.DataFrame, np.ndarray]:
        """Legacy optimize_data method."""
        from .unified_utilities import DataType, ArchitectureType as UnifiedArchitectureType
        
        # Convert string types to enums
        data_type_enum = DataType(data_type) if isinstance(data_type, str) else data_type
        arch_type_enum = UnifiedArchitectureType(architecture_type) if isinstance(architecture_type, str) else architecture_type
        
        return self.unified_utilities.optimize_data(data, arch_type_enum, data_type_enum)

# Legacy class aliases for backward compatibility
if LEGACY_IMPORTS_AVAILABLE:
    # NAS legacy classes
    class EnhancedNASEngine(LegacyNASEngineAdapter):
        """Legacy NAS engine using unified system."""
        pass
    
    class NASSearch(LegacyNASEngineAdapter):
        """Legacy NAS search using unified system."""
        pass
    
    class TradingMultiObjectiveOptimizer(LegacyMultiObjectiveOptimizerAdapter):
        """Legacy multi-objective optimizer using unified system."""
        pass
    
    class EconomicEvaluator(LegacyEconomicEvaluatorAdapter):
        """Legacy economic evaluator using unified system."""
        pass
    
    class NASRegimeAnalyzer(LegacyRegimeDetectorAdapter):
        """Legacy NAS regime analyzer using unified system."""
        pass
    
    # TAS legacy classes
    class TASEngine(LegacyTASEngineAdapter):
        """Legacy TAS engine using unified system."""
        pass
    
    class AdvancedSearch(LegacyTASEngineAdapter):
        """Legacy TAS advanced search using unified system."""
        pass
    
    class MultiObjectiveSearch(LegacyMultiObjectiveOptimizerAdapter):
        """Legacy TAS multi-objective search using unified system."""
        pass
    
    class TreeEvaluator(LegacyEconomicEvaluatorAdapter):
        """Legacy tree evaluator using unified system."""
        pass
    
    class TreeRegimeAnalyzer(LegacyRegimeDetectorAdapter):
        """Legacy tree regime analyzer using unified system."""
        pass
    
    # Hybrid legacy classes
    class HybridRegimeDetector(LegacyRegimeDetectorAdapter):
        """Legacy hybrid regime detector using unified system."""
        pass
    
    class HybridMultiObjectiveOptimizer(LegacyMultiObjectiveOptimizerAdapter):
        """Legacy hybrid multi-objective optimizer using unified system."""
        pass
    
    class HybridEconomicEvaluator(LegacyEconomicEvaluatorAdapter):
        """Legacy hybrid economic evaluator using unified system."""
        pass

# Migration helper functions
def migrate_config_to_unified(legacy_config: Dict[str, Any], 
                            component_type: str) -> Dict[str, Any]:
    """Migrate legacy configuration to unified format."""
    migration_map = {
        'nas_engine': {
            'max_iterations': 'max_iterations',
            'population_size': 'population_size',
            'elite_size': 'elite_size',
            'bayesian_config': 'bayesian_config',
            'evolutionary_config': 'evolutionary_config',
            'enable_parallel_processing': 'enable_parallel_processing',
            'n_jobs': 'n_jobs'
        },
        'tas_engine': {
            'max_iterations': 'max_iterations',
            'population_size': 'population_size',
            'elite_size': 'elite_size',
            'evolutionary_config': 'evolutionary_config',
            'enable_parallel_processing': 'enable_parallel_processing',
            'n_jobs': 'n_jobs'
        },
        'multi_objective_optimizer': {
            'objectives': 'objectives',
            'objective_weights': 'objective_weights',
            'max_iterations': 'max_iterations',
            'population_size': 'population_size',
            'convergence_threshold': 'convergence_threshold',
            'convergence_patience': 'convergence_patience'
        },
        'economic_evaluator': {
            'evaluation_types': 'evaluation_types',
            'significance_threshold': 'significance_threshold',
            'min_regime_duration': 'min_regime_duration',
            'risk_free_rate': 'risk_free_rate',
            'enable_logging': 'enable_logging'
        },
        'regime_detector': {
            'method': 'method',
            'n_regimes': 'n_regimes',
            'min_regime_duration': 'min_regime_duration',
            'clustering_algorithm': 'clustering_algorithm',
            'n_clusters': 'n_clusters',
            'random_state': 'random_state'
        },
        'utilities': {
            'enable_data_validation': 'enable_data_validation',
            'enable_memory_optimization': 'enable_memory_optimization',
            'enable_hardware_optimization': 'enable_hardware_optimization',
            'memory_limit_gb': 'memory_limit_gb',
            'strict_validation': 'strict_validation',
            'auto_fix_issues': 'auto_fix_issues',
            'validation_threshold': 'validation_threshold',
            'enable_logging': 'enable_logging'
        }
    }
    
    if component_type not in migration_map:
        raise ValueError(f"Unknown component type: {component_type}")
    
    mapping = migration_map[component_type]
    unified_config = {}
    
    for legacy_key, unified_key in mapping.items():
        if legacy_key in legacy_config:
            unified_config[unified_key] = legacy_config[legacy_key]
    
    return unified_config

def create_legacy_component(component_type: str, config: Optional[Dict[str, Any]] = None):
    """Create a legacy component using the unified system."""
    component_map = {
        'nas_engine': LegacyNASEngineAdapter,
        'tas_engine': LegacyTASEngineAdapter,
        'multi_objective_optimizer': LegacyMultiObjectiveOptimizerAdapter,
        'economic_evaluator': LegacyEconomicEvaluatorAdapter,
        'regime_detector': LegacyRegimeDetectorAdapter,
        'utilities': LegacyUtilitiesAdapter
    }
    
    if component_type not in component_map:
        raise ValueError(f"Unknown component type: {component_type}")
    
    return component_map[component_type](config)

def get_migration_guide() -> Dict[str, Any]:
    """Get migration guide for transitioning to unified system."""
    return {
        'overview': 'This guide helps migrate from legacy NAS/TAS components to the unified system',
        'migration_steps': [
            '1. Update imports to use unified components',
            '2. Migrate configuration using migrate_config_to_unified()',
            '3. Update method calls to use unified interfaces',
            '4. Test with backward compatibility adapters first',
            '5. Gradually replace legacy components'
        ],
        'component_mappings': {
            'EnhancedNASEngine': 'UnifiedSearchEngine with ArchitectureType.NEURAL',
            'TASEngine': 'UnifiedSearchEngine with ArchitectureType.TREE',
            'TradingMultiObjectiveOptimizer': 'UnifiedMultiObjectiveOptimizer',
            'EconomicEvaluator': 'UnifiedEconomicEvaluator',
            'NASRegimeAnalyzer': 'UnifiedRegimeDetector',
            'TreeRegimeAnalyzer': 'UnifiedRegimeDetector',
            'HybridRegimeDetector': 'UnifiedRegimeDetector with ArchitectureType.HYBRID'
        },
        'benefits': [
            'Unified interface across all components',
            'Better performance and optimization',
            'Improved error handling and logging',
            'Enhanced configurability',
            'Future-proof architecture'
        ],
        'backward_compatibility': 'Legacy components are still supported through adapters',
        'deprecation_timeline': 'Legacy components will be deprecated in future versions'
    }

# Export main classes and functions
__all__ = [
    'LegacyNASEngineAdapter',
    'LegacyTASEngineAdapter',
    'LegacyMultiObjectiveOptimizerAdapter',
    'LegacyEconomicEvaluatorAdapter',
    'LegacyRegimeDetectorAdapter',
    'LegacyUtilitiesAdapter',
    'migrate_config_to_unified',
    'create_legacy_component',
    'get_migration_guide',
    'deprecated_warning'
]

# Export legacy classes if imports are available
if LEGACY_IMPORTS_AVAILABLE:
    __all__.extend([
        'EnhancedNASEngine',
        'NASSearch',
        'TradingMultiObjectiveOptimizer',
        'EconomicEvaluator',
        'NASRegimeAnalyzer',
        'TASEngine',
        'AdvancedSearch',
        'MultiObjectiveSearch',
        'TreeEvaluator',
        'TreeRegimeAnalyzer',
        'HybridRegimeDetector',
        'HybridMultiObjectiveOptimizer',
        'HybridEconomicEvaluator'
    ])