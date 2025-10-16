"""
Advanced Tree Architecture Search (TAS) System

A comprehensive system for tree-based architecture search with advanced capabilities:
- Meta-learning and few-shot learning
- Hardware optimization and acceleration
- Advanced search strategies
- Uncertainty estimation
- Continual learning
- Regime analysis and reporting
- Multi-objective optimization
- Real-time adaptation
- Trading-specific optimizations
- Micro-regime detection
- Economic significance validation

This system provides tree-based alternatives to neural architecture search
while maintaining the same level of sophistication and capabilities, with
specialized support for financial trading applications.
"""

# Core TAS components
from .core.tas_engine import TreeArchitectureSearchEngine
from .core.tas_config import TASConfig, TASSearchConfig, TASOptimizationConfig
from .core.tas_result import TASResult, TASSearchResult, TASOptimizationResult

# Trading-specific components
from .core.tas_config import TASArchitectureType, TradingObjective, MarketRegime, MicroRegimeType
from .core.advanced_tas_search import AdvancedTradingArchitectureSearch, AdvancedTASResult
from .core.tree_cvlSA_architecture import TreeCVLSASearch, CVLSAResult

# Meta-learning components
from .meta_learning.tree_meta_learning import TreeMetaLearning, TreeMAML, TreePrototypicalNetwork
from .meta_learning import FewShotTreeLearner, TreeFewShotAdapter, ContinualTreeLearner, TreeEpisodicMemory

# Search strategies
from .search.evolutionary_search import EvolutionaryTreeSearch, TreeGeneticAlgorithm
from .search.bayesian_search import BayesianTreeSearch, TreeBayesianOptimizer
from .search.rl_search import RLTreeSearch, TreeReinforcementLearner
from .search.multi_objective_search import MultiObjectiveTreeSearch, TreeNSGA2

# Optimization components
from .optimization.enhanced_hardware_optimization import TreeHardwareOptimizer, TreeMatrixOperations
from .optimization import TreeMemoryOptimizer, TreeCacheManager
from .optimization import TreeParallelOptimizer, TreeDistributedSearch

# Uncertainty estimation
from .uncertainty.uncertainty_estimation import TreeUncertaintyEstimator, TreeEnsembleUncertainty
from .uncertainty.confidence_scoring import TreeConfidenceScorer, TreeReliabilityEstimator
from .uncertainty.robustness_analysis import TreeRobustnessAnalyzer, TreeAdversarialTesting

# Regime analysis
from .regime_analysis.tree_regime_analyzer import TreeRegimeAnalyzer, TreeRegimeDetector
from .regime_analysis.regime_optimization import TreeRegimeOptimizer, TreeRegimeSelector
from .regime_analysis.regime_reporting import TreeRegimeReporter, TreeRegimeVisualizer

# Trading-specific components
from .components.micro_regime_detector import MicroRegimeDetector, MicroRegimeDetectionResult
from .components.neural_architecture import TASNeuralModel, NeuralArchitectureConfig

# Adaptation
from .adaptation.real_time_adaptation import TreeRealTimeAdapter, TreePerformanceMonitor
from .adaptation.dynamic_optimization import TreeDynamicOptimizer, TreeAdaptiveSearch
from .adaptation.performance_tracking import TreePerformanceTracker, TreeMetricsCollector

# Evaluation
from .evaluation.tree_evaluator import TreeEvaluator, TreePerformanceEvaluator
from .evaluation.multi_objective_evaluation import TreeMultiObjectiveEvaluator
from .evaluation.regime_evaluation import TreeRegimeEvaluator, TreeRegimePerformanceAnalyzer
from .evaluation.tas_evaluator import TASEvaluator, EvaluationResult

# Utilities
from .utils.tree_utils import TreeUtils, TreeArchitectureUtils
from .utils.visualization import TreeVisualizer, TreeArchitectureVisualizer
from .utils.logging import TreeLogger, TreeSearchLogger

# Convenience functions
from .core.advanced_tas_search import optimize_advanced_trading_architecture
from .core.tree_cvlSA_architecture import optimize_cvlSA_architecture

__version__ = "2.0.0"
__author__ = "Advanced TAS Team"

# Package configuration
DEFAULT_CONFIG = None

# Main exports
__all__ = [
    # Core components
    'TreeArchitectureSearchEngine',
    'TASConfig', 'TASSearchConfig', 'TASOptimizationConfig',
    'TASResult', 'TASSearchResult', 'TASOptimizationResult',

    # Trading-specific components
    'TASArchitectureType', 'TradingObjective', 'MarketRegime', 'MicroRegimeType',
    'AdvancedTradingArchitectureSearch', 'AdvancedTASResult',
    'TreeCVLSASearch', 'CVLSAResult',
    'MicroRegimeDetector', 'MicroRegimeDetectionResult',
    'TASNeuralModel', 'NeuralArchitectureConfig',

    # Meta-learning
    'TreeMetaLearning', 'TreeMAML', 'TreePrototypicalNetwork',
    'FewShotTreeLearner', 'TreeFewShotAdapter',
    'ContinualTreeLearner', 'TreeEpisodicMemory',

    # Search strategies
    'EvolutionaryTreeSearch', 'TreeGeneticAlgorithm',
    'BayesianTreeSearch', 'TreeBayesianOptimizer',
    'RLTreeSearch', 'TreeReinforcementLearner',
    'MultiObjectiveTreeSearch', 'TreeNSGA2',

    # Optimization
    'TreeHardwareOptimizer', 'TreeMatrixOperations',
    'TreeMemoryOptimizer', 'TreeCacheManager',
    'TreeParallelOptimizer', 'TreeDistributedSearch',

    # Uncertainty
    'TreeUncertaintyEstimator', 'TreeEnsembleUncertainty',
    'TreeConfidenceScorer', 'TreeReliabilityEstimator',
    'TreeRobustnessAnalyzer', 'TreeAdversarialTesting',

    # Regime analysis
    'TreeRegimeAnalyzer', 'TreeRegimeDetector',
    'TreeRegimeOptimizer', 'TreeRegimeSelector',
    'TreeRegimeReporter', 'TreeRegimeVisualizer',

    # Adaptation
    'TreeRealTimeAdapter', 'TreePerformanceMonitor',
    'TreeDynamicOptimizer', 'TreeAdaptiveSearch',
    'TreePerformanceTracker', 'TreeMetricsCollector',

    # Evaluation
    'TreeEvaluator', 'TreePerformanceEvaluator',
    'TreeMultiObjectiveEvaluator',
    'TreeRegimeEvaluator', 'TreeRegimePerformanceAnalyzer',
    'TASEvaluator', 'EvaluationResult',

    # Utilities
    'TreeUtils', 'TreeArchitectureUtils',
    'TreeVisualizer', 'TreeArchitectureVisualizer',
    'TreeLogger', 'TreeSearchLogger',

    # Convenience functions
    'optimize_advanced_trading_architecture',
    'optimize_cvlSA_architecture',

    # Configuration
    'DEFAULT_CONFIG'
]
