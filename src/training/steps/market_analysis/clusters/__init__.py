"""
Clustering modules for NAS-TAS regime detection.

This package contains the refactored clustering components organized by:
- Main = orchestration (clustering_orchestrator, clustering_service)
- Services = high-level managers (feature_service, hardware_service, optimization_service)
- Features = data preparation (step1_feature_preparation, features/)
- Hardware = resource optimization (gpu_manager, m1_optimizer, memory_manager, performance_monitor)
- Utils = helpers & validation (clustering_utils, data_validator, risk_mitigation, validation_framework)
- Clustering = core algorithms + metrics + optimization (engine, metrics, optimizer, iterative_optimization, step files)
"""

# Main - Orchestration
from .clustering_orchestrator import ClusteringOrchestrator
from .clustering_service import ClusteringService, ClusteringResult

# Services - High-level managers
from .feature_service import FeatureService, FeaturePreparationResult
from .hardware_service import HardwareService
from .optimization_service import OptimizationService

# Features - Data preparation
from .step1_feature_preparation import FeaturePreparationStep, ClusteringContext
from .features import FeatureSelector, FeaturePreprocessor, FeatureAnalyzer
from .features import FeatureSelectorConfig, FeaturePreprocessorConfig, FeatureAnalyzerConfig

# Hardware - Resource optimization
from .gpu_manager import GPUManager, get_gpu_manager
from .m1_optimizer import M1Optimizer, get_m1_optimizer
from .memory_manager import MemoryManager, get_memory_manager
from .performance_monitor import PerformanceMonitor, performance_monitor

# Utils - Helpers & validation
from .clustering_utils import ClusteringUtils
from .data_validator import DataValidator, validate_data_comprehensive
from .risk_mitigation import RiskMitigationSystem, PRODUCTION_RISK_CONFIG
from .validation_framework import ClusteringValidator, ValidationConfig, ValidationResults

# Clustering - Core algorithms + metrics + optimization
from .engine import ClusteringEngine, EngineConfig
from .metrics import ClusteringMetrics, MetricsConfig, MetricsReport, MetricResult
from .optimizer import ClusteringOptimizer, OptimizerConfig
from .iterative_optimization import IterativeOptimization, ClusteringStats
from .step2_initial_clustering import InitialClusteringStep
from .step8_validation import ValidationStep
from .step9_results_consolidation import ResultsConsolidationStep
from .step10_comprehensive_reporting import ComprehensiveReporter
from .nas_tas_clustering_refactored import NASTASClusteringComponent, NASTASClusteringConfig

__all__ = [
    # Main - Orchestration
    'ClusteringOrchestrator',
    'ClusteringService',
    'ClusteringResult',

    # Services - High-level managers
    'FeatureService',
    'FeaturePreparationResult',
    'HardwareService',
    'OptimizationService',

    # Features - Data preparation
    'FeaturePreparationStep',
    'ClusteringContext',
    'FeatureSelector',
    'FeaturePreprocessor',
    'FeatureAnalyzer',
    'FeatureSelectorConfig',
    'FeaturePreprocessorConfig',
    'FeatureAnalyzerConfig',

    # Hardware - Resource optimization
    'GPUManager',
    'get_gpu_manager',
    'M1Optimizer',
    'get_m1_optimizer',
    'MemoryManager',
    'get_memory_manager',
    'PerformanceMonitor',
    'performance_monitor',

    # Utils - Helpers & validation
    'ClusteringUtils',
    'DataValidator',
    'validate_data_comprehensive',
    'RiskMitigationSystem',
    'PRODUCTION_RISK_CONFIG',
    'ClusteringValidator',
    'ValidationConfig',
    'ValidationResults',

    # Clustering - Core algorithms + metrics + optimization
    'ClusteringEngine',
    'EngineConfig',
    'ClusteringMetrics',
    'MetricsConfig',
    'MetricsReport',
    'MetricResult',
    'ClusteringOptimizer',
    'OptimizerConfig',
    'IterativeOptimization',
    'ClusteringStats',
    'InitialClusteringStep',
    'ValidationStep',
    'ResultsConsolidationStep',
    'ComprehensiveReporter',
    'NASTASClusteringComponent',
    'NASTASClusteringConfig'
]
