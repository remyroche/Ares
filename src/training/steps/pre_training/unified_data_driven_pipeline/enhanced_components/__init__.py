"""
Enhanced Components for UnifiedDataDrivenPipeline

This package contains sophisticated components that integrate advanced functionality
from FeatureLookbackOptimizationComponent into the UnifiedDataDrivenPipeline.

Components:
- sophisticated_lookback_optimizer: Advanced optimization algorithms
- multi_horizon_integration: Multi-horizon profit labeling integration
- comprehensive_validation: Comprehensive validation system
- advanced_lookback_optimizer: Advanced lookback optimization (legacy)
- modular_architecture: Modular architecture system
- economic_evaluator: Economic evaluation components
- intelligent_feature_selector: Intelligent feature selection
- template_interaction_generator: Template-based interaction generation
- vectorbt_optimizer: VectorBT optimization components
- vectorbt_enhancements: VectorBT enhancement utilities
- feature_bank_integration: Feature bank integration
- htf_template_system: Higher timeframe template system
- enhanced_feature_generator: Enhanced feature generation
- enhanced_unified_pipeline: Enhanced unified pipeline implementation
"""

# Import sophisticated components
from .sophisticated_lookback_optimizer import (
    SophisticatedLookbackOptimizer,
    SophisticatedOptimizationResult,
    OptimizationDirection,
    create_sophisticated_lookback_optimizer
)

from .multi_horizon_integration import (
    MultiHorizonIntegration,
    MultiHorizonIntegrationResult,
    TargetDirection,
    TargetColumnInfo,
    create_multi_horizon_integration
)

from .comprehensive_validation import (
    ComprehensiveValidator,
    ValidationLevel,
    ErrorSeverity,
    ErrorCategory,
    ValidationSummary,
    PerformanceValidationResult,
    create_comprehensive_validator
)

# Import existing enhanced components
from .advanced_lookback_optimizer import (
    AdvancedLookbackOptimizer,
    OptimizationResult,
    LookbackConstraints,
    create_advanced_lookback_optimizer
)

from .modular_architecture import (
    ModularArchitecture,
    ValidationLevel as ModularValidationLevel,
    ErrorSeverity as ModularErrorSeverity,
    ErrorCategory as ModularErrorCategory,
    create_modular_architecture
)

from .economic_evaluator import (
    EconomicPeriodEvaluator,
    EconomicEvaluationConfig,
    EconomicPeriodEvaluationResult,
    create_economic_evaluator
)

from .intelligent_feature_selector import (
    IntelligentFeatureSelector,
    FeatureSelectionConfig,
    FeatureSelectionResult,
    create_intelligent_feature_selector
)

from .template_interaction_generator import (
    TemplateInteractionGenerator,
    TemplateConfig,
    create_template_interaction_generator
)

from .vectorbt_optimizer import (
    VectorBTOptimizer,
    VectorBTConfig,
    create_vectorbt_optimizer
)

from .vectorbt_enhancements import (
    VectorBTEnhancements,
    create_vectorbt_enhancements
)

from .feature_bank_integration import (
    FeatureBankIntegration,
    create_feature_bank_integration
)

from .htf_template_system import (
    HTFTemplateSystem,
    create_htf_template_system
)

from .enhanced_feature_generator import (
    EnhancedFeatureGenerator,
    create_enhanced_feature_generator
)

from .enhanced_unified_pipeline import (
    EnhancedUnifiedDataDrivenPipeline,
    EnhancedFeaturePipelineResult,
    create_enhanced_unified_pipeline,
    process_with_enhanced_pipeline
)

# Export all components
__all__ = [
    # Sophisticated components
    'SophisticatedLookbackOptimizer',
    'SophisticatedOptimizationResult',
    'OptimizationDirection',
    'create_sophisticated_lookback_optimizer',
    
    'MultiHorizonIntegration',
    'MultiHorizonIntegrationResult',
    'TargetDirection',
    'TargetColumnInfo',
    'create_multi_horizon_integration',
    
    'ComprehensiveValidator',
    'ValidationLevel',
    'ErrorSeverity',
    'ErrorCategory',
    'ValidationSummary',
    'PerformanceValidationResult',
    'create_comprehensive_validator',
    
    # Existing enhanced components
    'AdvancedLookbackOptimizer',
    'OptimizationResult',
    'LookbackConstraints',
    'create_advanced_lookback_optimizer',
    
    'ModularArchitecture',
    'ModularValidationLevel',
    'ModularErrorSeverity',
    'ModularErrorCategory',
    'create_modular_architecture',
    
    'EconomicPeriodEvaluator',
    'EconomicEvaluationConfig',
    'EconomicPeriodEvaluationResult',
    'create_economic_evaluator',
    
    'IntelligentFeatureSelector',
    'FeatureSelectionConfig',
    'FeatureSelectionResult',
    'create_intelligent_feature_selector',
    
    'TemplateInteractionGenerator',
    'TemplateConfig',
    'create_template_interaction_generator',
    
    'VectorBTOptimizer',
    'VectorBTConfig',
    'create_vectorbt_optimizer',
    
    'VectorBTEnhancements',
    'create_vectorbt_enhancements',
    
    'FeatureBankIntegration',
    'create_feature_bank_integration',
    
    'HTFTemplateSystem',
    'create_htf_template_system',
    
    'EnhancedFeatureGenerator',
    'create_enhanced_feature_generator',
    
    'EnhancedUnifiedDataDrivenPipeline',
    'EnhancedFeaturePipelineResult',
    'create_enhanced_unified_pipeline',
    'process_with_enhanced_pipeline'
]