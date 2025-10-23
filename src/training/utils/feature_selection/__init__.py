"""
Feature Selection Framework - Modular Components

This package provides a comprehensive, modular feature selection framework
that has been refactored from the original monolithic implementation.

Components:
- base_framework: Core framework initialization and configuration
- data_validation: Data quality checks and validation utilities
- selection_methods: Individual feature selection algorithms
- stability_analysis: Stability validation and analysis
- performance_monitoring: Performance tracking and optimization
- quality_metrics: Feature selection quality assessment
- temporal_analysis: Time-based feature analysis
- causal_analysis: Causal inference and filtering
- partial_information_decompositor: PID-based feature engineering and interaction analysis
- main_framework: Main orchestrator that combines all components

Usage:
    from src.training.utils.feature_selection import FeatureSelectionFramework

    # Initialize the framework
    framework = FeatureSelectionFramework(config=your_config)

    # Run comprehensive feature selection
    results = framework.run_comprehensive_feature_selection(X, y, feature_names)
"""

from .main_framework import FeatureSelectionFramework
from .base_framework import BaseFeatureSelectionFramework
from .data_validation import DataValidator
from .selection_methods import (
    MRMRSelector,
    StabilityWeightedSelector,
    RecursiveFeatureEliminator,
    FeatureImportanceRanker,
    CompositeFeatureScorer,
    CrossValidatedSelector,
    ElasticNetStabilitySelector,
    TreeBasedEnsembleSelector
)
from .stability_analysis import StabilityAnalyzer
from .performance_monitoring import PerformanceMonitor
from .quality_metrics import QualityMetricsCalculator
from .temporal_analysis import TemporalAnalyzer
from .causal_analysis import CausalAnalyzer
from .partial_information_decompositor import (
    PartialInformationDecompositor,
    PIDConfig,
    PIDResult
)

__all__ = [
    'FeatureSelectionFramework',
    'BaseFeatureSelectionFramework',
    'DataValidator',
    'MRMRSelector',
    'StabilityWeightedSelector',
    'RecursiveFeatureEliminator',
    'FeatureImportanceRanker',
    'CompositeFeatureScorer',
    'CrossValidatedSelector',
    'ElasticNetStabilitySelector',
    'TreeBasedEnsembleSelector',
    'StabilityAnalyzer',
    'PerformanceMonitor',
    'QualityMetricsCalculator',
    'TemporalAnalyzer',
    'CausalAnalyzer',
    'PartialInformationDecompositor',
    'PIDConfig',
    'PIDResult'
]

__version__ = "2.0.0"
__author__ = "Feature Selection Framework Team"
