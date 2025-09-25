"""
Unified TAS-NAS Regime Detection System

This module provides a unified regime detection system that combines the best aspects
of both TAS (Tree Architecture Search) and NAS (Neural Architecture Search) regime
detection with enhanced economic significance and trading viability evaluation.
"""

from .unified_regime_config import (
    UnifiedRegimeConfig,
    RegimeDetectionMethod,
    OptimizationStrategy,
    EconomicEvaluationMode
)

from .unified_regime_detector import (
    UnifiedRegimeDetector,
    UnifiedRegimeResult
)

from .performance_optimizer import (
    PerformanceOptimizer,
    PerformanceCache,
    GPUAccelerator,
    MemoryOptimizer,
    optimize_performance,
    get_performance_optimizer
)

from .real_time_monitor import (
    RealTimeRegimeMonitor,
    RegimeChangeEvent,
    RealTimeMetrics,
    DataStreamProcessor,
    RegimeChangeDetector,
    PerformanceMonitor,
    create_real_time_monitor
)

from .confidence_scoring import (
    TreeConfidenceScorer,
    TreeReliabilityEstimator,
    TreeCalibrationScorer,
    ConfidenceConfig
)

# Training-related utilities
from .regime_aware_trainer import (
    RegimeAwareTrainer,
    RegimeAwareTrainingConfig,
    RegimeTrainingResult,
    ModelType,
    RegimeTrainingStrategy
)

from .training_orchestrator import (
    TrainingOrchestrator,
    OrchestratorConfig,
    OrchestrationResult,
    OrchestrationMode
)

from .model_selector import (
    ModelSelector,
    ModelSelectionConfig,
    ModelSelectionResult,
    SelectionStrategy,
    RoutingMethod
)

from .model_manager import (
    ModelManager,
    ModelManagerConfig,
    ModelMetadata,
    ModelDeploymentResult,
    ModelStatus,
    DeploymentStrategy
)

from .performance_tracker import (
    PerformanceTracker,
    PerformanceConfig,
    PerformanceRecord,
    PerformanceAlert,
    PerformanceReport,
    PerformanceMetric,
    AlertType
)
# Import NAS modules
from .nas import (
    NeuralArchitectureSearch,
    ArchitectureConfig,
    ArchitectureCandidate,
    ArchitectureSearchSpace,
    search_neural_architecture,
    AdaptiveRegimeNAS,
    AdaptiveRegimeNASConfig,
    RegimeDetector
)

# Import TAS modules
from .tas import (
    TreeBasedArchitectureSearch,
    TreeArchitectureConfig,
    TreeArchitectureCandidate,
    TreeArchitectureSearchSpace,
    search_tree_architecture,
    PureTreeNAS,
    PureTreeNASConfig,
    UnsupervisedTreeNAS,
    UnsupervisedTreeNASConfig,
    RegimeTradingTreeNAS,
    RegimeTradingTreeNASConfig,
    TradingTreeArchitectureSearch,
    TradingTASConfig,
    TradingRegime,
    TradingTASResult,
    TradingObjective,
    MarketRegime
)

# Import search space utilities
from .search_space import (
    create_default_nas_search_space,
    create_tree_search_space,
    SearchSpace,
    SearchSpaceConfig,
    ParameterRange,
    SearchSpaceType,
    OptimizationStrategy
)

# Import risk analysis
from .risk_analysis.risk_analysis import (
    RiskAnalyzer,
    RiskConfig,
    RiskResult,
    RiskMetric
)

# Import backtesting engine
from .backtesting_engine import (
    BacktestingEngine,
    BacktestingConfig,
    BacktestingResult,
    BacktestingMode
)

# Import evolutionary search
from .evolutionary_search import (
    EvolutionaryTreeSearch,
    TreeGeneticAlgorithm,
    TreeNSGA2,
    EvolutionaryConfig
)

# Import uncertainty estimation
from .uncertainty_estimation import (
    TreeUncertaintyEstimator,
    TreeEnsembleUncertainty,
    TreeBayesianUncertainty,
    UncertaintyConfig
)

# Import unified evaluator
from .unified_evaluator import (
    UnifiedEvaluator,
    EvaluationConfig,
    EvaluationResult,
    ModelType,
    EvaluationMode,
    MetricType
)

# Import unified hardware manager
from .unified_hardware import (
    UnifiedHardwareManager,
    HardwareAccelerationConfig,
    WorkloadType,
    OptimizationLevel,
    PerformanceMetrics,
    create_unified_hardware_manager,
    get_hardware_manager
)

# Import Hybrid NAS System
from .hybrid_nas_system import (
    HybridNASSystem,
    HybridNASConfig,
    HybridArchitectureCandidate,
    optimize_hybrid_architecture,
    analyze_data_characteristics
)


__all__ = [
    'UnifiedRegimeConfig',
    'RegimeDetectionMethod',
    'OptimizationStrategy',
    'create_default_nas_search_space',
    'create_tree_search_space',
    'SearchSpace',
    'SearchSpaceConfig',
    'ParameterRange',
    'SearchSpaceType',
    'RiskAnalyzer',
    'RiskConfig',
    'RiskResult',
    'RiskMetric',
    'BacktestingEngine',
    'BacktestingConfig', 
    'BacktestingResult',
    'BacktestingMode',
    'EvolutionaryTreeSearch',
    'TreeGeneticAlgorithm',
    'TreeNSGA2',
    'EvolutionaryConfig',
    'TreeUncertaintyEstimator',
    'TreeEnsembleUncertainty',
    'TreeBayesianUncertainty',
    'UncertaintyConfig',
    'TreeConfidenceScorer',
    'TreeReliabilityEstimator',
    'TreeCalibrationScorer',
    'ConfidenceConfig',
    'UnifiedEvaluator',
    'EvaluationConfig',
    'EvaluationResult',
    'ModelType',
    'EvaluationMode',
    'MetricType',
    'UnifiedHardwareManager',
    'HardwareAccelerationConfig',
    'WorkloadType',
    'OptimizationLevel',
    'PerformanceMetrics',
    'create_unified_hardware_manager',
    'get_hardware_manager',
    
    # NAS modules
    'NeuralArchitectureSearch',
    'ArchitectureConfig',
    'ArchitectureCandidate',
    'ArchitectureSearchSpace',
    'search_neural_architecture',
    'AdaptiveRegimeNAS',
    'AdaptiveRegimeNASConfig',
    'RegimeDetector',
    
    # TAS modules
    'TreeBasedArchitectureSearch',
    'TreeArchitectureConfig',
    'TreeArchitectureCandidate',
    'TreeArchitectureSearchSpace',
    'search_tree_architecture',
    'PureTreeNAS',
    'PureTreeNASConfig',
    'UnsupervisedTreeNAS',
    'UnsupervisedTreeNASConfig',
    'RegimeTradingTreeNAS',
    'RegimeTradingTreeNASConfig',
    'TradingTreeArchitectureSearch',
    'TradingTASConfig',
    'TradingRegime',
    'TradingTASResult',
    'TradingObjective',
    'MarketRegime',
    
    # Hybrid NAS System
    'HybridNASSystem',
    'HybridNASConfig',
    'HybridArchitectureCandidate',
    'optimize_hybrid_architecture',
    'analyze_data_characteristics',
  
    # Training-related utilities
    'RegimeAwareTrainer',
    'RegimeAwareTrainingConfig',
    'RegimeTrainingResult',
    'ModelType',
    'RegimeTrainingStrategy',
    'TrainingOrchestrator',
    'OrchestratorConfig',
    'OrchestrationResult',
    'OrchestrationMode',
    'ModelSelector',
    'ModelSelectionConfig',
    'ModelSelectionResult',
    'SelectionStrategy',
    'RoutingMethod',
    'ModelManager',
    'ModelManagerConfig',
    'ModelMetadata',
    'ModelDeploymentResult',
    'ModelStatus',
    'DeploymentStrategy',
    'PerformanceTracker',
    'PerformanceConfig',
    'PerformanceRecord',
    'PerformanceAlert',
    'PerformanceReport',
    'PerformanceMetric',
    'AlertType',
    'EconomicEvaluationMode',
    'UnifiedRegimeDetector',
    'UnifiedRegimeResult',
    'PerformanceOptimizer',
    'PerformanceCache',
    'GPUAccelerator',
    'MemoryOptimizer',
    'optimize_performance',
    'get_performance_optimizer',
    'RealTimeRegimeMonitor',
    'RegimeChangeEvent',
    'RealTimeMetrics',
    'DataStreamProcessor',
    'RegimeChangeDetector',
    'PerformanceMonitor',
    'create_real_time_monitor'
]

__version__ = "1.0.0"
__author__ = "Unified Regime Detection System"