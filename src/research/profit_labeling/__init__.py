"""
Enhanced Multi-Horizon Profit Labeling Research Framework

A comprehensive research framework for analyzing and optimizing the multi-horizon profit 
labeling system from a fully data-driven perspective. This framework provides tools to:

1. Analyze labeling heuristics and their effectiveness
2. Validate labeling quality and consistency 
3. Optimize labeling parameters systematically
4. Visualize labeling patterns and performance
5. Compare different labeling strategies
6. ML-based label quality assessment and enhancement
7. Adaptive market regime-aware labeling
8. Ensemble labeling approaches for robustness
9. Dynamic target and horizon optimization
10. Advanced statistical validation methods
11. Contextual feature engineering
12. Backtesting-integrated validation

Key Components:
- HeuristicAnalyzer: Analyzes the effectiveness of profit labeling heuristics
- LabelingValidator: Validates labeling quality and consistency
- ParameterOptimizer: Optimizes labeling parameters using systematic approaches
- LabelingVisualizer: Comprehensive visualization system for labeling analysis
- ResearchRunner: Main runner for research workflows and experiments

Enhanced Components:
- MLLabelQualityAssessor: ML-based label quality assessment and enhancement
- AdaptiveLabelingStrategy: Market regime-aware adaptive labeling
- AdvancedStatisticalValidator: Comprehensive statistical validation
- EnsembleLabelingSystem: Multiple strategy ensemble approaches
- DynamicTargetOptimizer: Data-driven target and horizon discovery
- ContextualFeatureEngineer: Rich market context feature engineering
- BacktestingIntegratedValidator: Trading performance-based validation
- EnhancedMultiHorizonProfitLabeler: Integrated enhanced labeling system

Usage:
    from research.profit_labeling import (
        # Original components
        HeuristicAnalyzer,
        LabelingValidator, 
        ParameterOptimizer,
        LabelingVisualizer,
        ResearchRunner,
        
        # Enhanced components
        MLLabelQualityAssessor,
        AdaptiveLabelingStrategy,
        AdvancedStatisticalValidator,
        EnsembleLabelingSystem,
        EnhancedMultiHorizonProfitLabeler,
        
        # Convenience functions
        generate_fully_enhanced_labels,
        create_enhanced_labeler
    )
"""

from src.utils.tprint import tprint

tprint("🔧 Loading enhanced multi-horizon profit labeling research framework...")

# Original components
from .heuristic_analyzer import HeuristicAnalyzer, HeuristicAnalysisConfig
from .labeling_validator import LabelingValidator, ValidationConfig
from .parameter_optimizer import ParameterOptimizer, OptimizationConfig
from .labeling_visualizer import LabelingVisualizer, VisualizationConfig
from .research_runner import ResearchRunner, ResearchConfig

# Enhanced components
from .ml_label_quality_assessor import (
    MLLabelQualityAssessor, 
    MLQualityAssessmentConfig,
    assess_label_quality_ml,
    enhance_labels_with_ml
)
from .adaptive_labeling_strategy import (
    AdaptiveLabelingStrategy,
    AdaptiveLabelingConfig,
    MarketRegimeDetector,
    ContextualParameterOptimizer,
    create_adaptive_labeling_strategy,
    get_regime_adaptive_config
)
from .advanced_statistical_validator import (
    AdvancedStatisticalValidator,
    AdvancedValidationConfig,
    validate_labels_advanced,
    generate_advanced_validation_report
)
from .ensemble_labeling_system import (
    EnsembleLabelingSystem,
    EnsembleLabelingConfig,
    create_ensemble_labeling_system,
    generate_ensemble_labels
)
from .dynamic_target_optimizer import (
    JointTargetHorizonOptimizer,
    DynamicOptimizationConfig,
    DynamicTargetDiscovery,
    DynamicHorizonOptimizer,
    discover_optimal_targets_and_horizons,
    create_optimized_multi_horizon_config
)
from .contextual_feature_labeling import (
    ContextualFeatureEngineer,
    ContextualFeatureConfig,
    engineer_contextual_features,
    create_feature_enhanced_labels
)
from .backtesting_integrated_validator import (
    BacktestingIntegratedValidator,
    BacktestingConfig,
    validate_labels_through_backtesting,
    generate_backtesting_validation_report
)
from .enhanced_multi_horizon_labeler import (
    EnhancedMultiHorizonProfitLabeler,
    EnhancedLabelingConfig,
    EnhancementLevel,
    create_enhanced_labeler,
    generate_fully_enhanced_labels,
    enhance_existing_labeler
)
from .real_time_monitor import (
    RealTimeLabelingMonitor,
    MonitoringConfig,
    LabelingPerformanceTracker,
    LabelingDriftDetector,
    AutoRecalibrator,
    create_real_time_monitor,
    monitor_labeling_quality
)
from .bonus_penalty_optimizer import (
    BonusPenaltyOptimizer,
    BonusPenaltyOptimizationConfig,
    DataDrivenQualityScorer,
    ModifiedMultiHorizonLabeler,
    RegimeSpecificBonusPenaltyOptimizer,
    optimize_bonus_penalty_parameters,
    create_optimized_labeler,
    get_optimal_bonus_penalty_config
)

tprint("📋 Setting up module exports...")
__all__ = [
    # Original components
    'HeuristicAnalyzer',
    'HeuristicAnalysisConfig',
    'LabelingValidator', 
    'ValidationConfig',
    'ParameterOptimizer',
    'OptimizationConfig',
    'LabelingVisualizer',
    'VisualizationConfig',
    'ResearchRunner',
    'ResearchConfig',
    
    # Enhanced components
    'MLLabelQualityAssessor',
    'MLQualityAssessmentConfig',
    'AdaptiveLabelingStrategy',
    'AdaptiveLabelingConfig',
    'MarketRegimeDetector',
    'ContextualParameterOptimizer',
    'AdvancedStatisticalValidator',
    'AdvancedValidationConfig',
    'EnsembleLabelingSystem',
    'EnsembleLabelingConfig',
    'JointTargetHorizonOptimizer',
    'DynamicOptimizationConfig',
    'DynamicTargetDiscovery',
    'DynamicHorizonOptimizer',
    'ContextualFeatureEngineer',
    'ContextualFeatureConfig',
    'BacktestingIntegratedValidator',
    'BacktestingConfig',
    'EnhancedMultiHorizonProfitLabeler',
    'EnhancedLabelingConfig',
    'EnhancementLevel',
    'RealTimeLabelingMonitor',
    'MonitoringConfig',
    'LabelingPerformanceTracker',
    'LabelingDriftDetector',
    'AutoRecalibrator',
    'BonusPenaltyOptimizer',
    'BonusPenaltyOptimizationConfig',
    'DataDrivenQualityScorer',
    'ModifiedMultiHorizonLabeler',
    'RegimeSpecificBonusPenaltyOptimizer',
    
    # Convenience functions
    'assess_label_quality_ml',
    'enhance_labels_with_ml',
    'create_adaptive_labeling_strategy',
    'get_regime_adaptive_config',
    'validate_labels_advanced',
    'generate_advanced_validation_report',
    'create_ensemble_labeling_system',
    'generate_ensemble_labels',
    'discover_optimal_targets_and_horizons',
    'create_optimized_multi_horizon_config',
    'engineer_contextual_features',
    'create_feature_enhanced_labels',
    'validate_labels_through_backtesting',
    'generate_backtesting_validation_report',
    'create_enhanced_labeler',
    'generate_fully_enhanced_labels',
    'enhance_existing_labeler',
    'create_real_time_monitor',
    'monitor_labeling_quality',
    'optimize_bonus_penalty_parameters',
    'create_optimized_labeler',
    'get_optimal_bonus_penalty_config'
]

__version__ = '1.0.0'
__author__ = 'Ares Trading System'
__description__ = 'Multi-Horizon Profit Labeling Research Framework'
tprint("✅ Enhanced multi-horizon profit labeling research framework fully loaded")
