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

__all__ = [
    'UnifiedRegimeConfig',
    'RegimeDetectionMethod',
    'OptimizationStrategy',
    'create_default_nas_search_space',
    'create_tree_search_space',
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
