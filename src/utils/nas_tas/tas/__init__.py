"""
TAS to NAS Parity Enhancement Module

This module provides comprehensive enhancements to bring the TAS (Tensor Adaptive System)
to full parity with state-of-the-art NAS (Neural Architecture Search) systems.

Key Components:
- Hardware acceleration for tree-based CLVSA models
- Real-time optimization engine
- Architecture diversity expansion
- Advanced meta-learning capabilities
- Continuous adaptation system
- Comprehensive testing framework
"""

# Import all major components
from .hardware import (
    TreeHardwareAccelerator,
    CLVSAHardwareOptimizer,
    HardwareAccelerationConfig,
    create_tree_hardware_accelerator,
    create_cvlsa_hardware_optimizer
)

from .realtime import (
    RealTimeOptimizationEngine,
    PerformanceMonitor,
    AdaptationEngine,
    RealTimeOptimizationConfig,
    create_realtime_optimization_engine,
    create_performance_monitor,
    create_adaptation_engine
)

from .architecture import (
    TreeArchitectureFactory,
    TreeArchitectureEvaluator,
    TreeArchitectureSelector,
    TreeArchitectureType,
    TreeArchitectureConfig,
    TreeArchitectureCandidate,
    create_tree_architecture_factory,
    create_tree_architecture_evaluator,
    create_tree_architecture_selector
)

from .meta_learning import (
    AdvancedMetaLearningSystem,
    AdvancedMAML,
    CrossDomainMetaLearning,
    MetaLearningMethod,
    AdvancedMetaLearningConfig,
    MetaTask,
    MetaLearningResult,
    create_advanced_meta_learning_system,
    create_advanced_maml,
    create_cross_domain_meta_learning
)

from .adaptation import (
    ContinuousAdaptationSystem,
    RegimeChangeDetector,
    PerformanceAdaptationTrigger,
    CLVSAAdaptationEngine,
    ContinuousAdaptationConfig,
    AdaptationTrigger,
    AdaptationResult,
    create_continuous_adaptation_system,
    create_regime_change_detector,
    create_cvlsa_adaptation_engine
)

from .testing import (
    ComprehensiveTestingFramework,
    UnitTestSuite,
    IntegrationTestSuite,
    NASBenchmarkSuite,
    TestingConfig,
    TestResult,
    BenchmarkResult,
    create_comprehensive_testing_framework,
    create_unit_test_suite,
    create_integration_test_suite,
    create_nas_benchmark_suite
)

__all__ = [
    # Hardware acceleration
    'TreeHardwareAccelerator',
    'CLVSAHardwareOptimizer',
    'HardwareAccelerationConfig',
    'create_tree_hardware_accelerator',
    'create_cvlsa_hardware_optimizer',
    
    # Real-time optimization
    'RealTimeOptimizationEngine',
    'PerformanceMonitor',
    'AdaptationEngine',
    'RealTimeOptimizationConfig',
    'create_realtime_optimization_engine',
    'create_performance_monitor',
    'create_adaptation_engine',
    
    # Architecture diversity
    'TreeArchitectureFactory',
    'TreeArchitectureEvaluator',
    'TreeArchitectureSelector',
    'TreeArchitectureType',
    'TreeArchitectureConfig',
    'TreeArchitectureCandidate',
    'create_tree_architecture_factory',
    'create_tree_architecture_evaluator',
    'create_tree_architecture_selector',
    
    # Meta-learning
    'AdvancedMetaLearningSystem',
    'AdvancedMAML',
    'CrossDomainMetaLearning',
    'MetaLearningMethod',
    'AdvancedMetaLearningConfig',
    'MetaTask',
    'MetaLearningResult',
    'create_advanced_meta_learning_system',
    'create_advanced_maml',
    'create_cross_domain_meta_learning',
    
    # Continuous adaptation
    'ContinuousAdaptationSystem',
    'RegimeChangeDetector',
    'PerformanceAdaptationTrigger',
    'CLVSAAdaptationEngine',
    'ContinuousAdaptationConfig',
    'AdaptationTrigger',
    'AdaptationResult',
    'create_continuous_adaptation_system',
    'create_regime_change_detector',
    'create_cvlsa_adaptation_engine',
    
    # Testing framework
    'ComprehensiveTestingFramework',
    'UnitTestSuite',
    'IntegrationTestSuite',
    'NASBenchmarkSuite',
    'TestingConfig',
    'TestResult',
    'BenchmarkResult',
    'create_comprehensive_testing_framework',
    'create_unit_test_suite',
    'create_integration_test_suite',
    'create_nas_benchmark_suite'
]