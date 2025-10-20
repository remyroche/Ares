"""
Optimization module for regime discovery system.

This module provides optimized components for the HDBSCAN clustering pipeline:
- Feature Extraction Optimization
- Preprocessing Pipeline Optimization  
- Dimensionality Reduction Optimization
- HDBSCAN Clustering Optimization
- Post-Processing Optimization
- Unified Pipeline
"""

# Import optimized components
from .optimized_feature_extractor import (
    OptimizedFeatureExtractor,
    FeatureExtractionConfig,
    create_optimized_feature_extractor
)

from .optimized_preprocessor import (
    OptimizedPreprocessor,
    PreprocessingConfig,
    create_optimized_preprocessor
)

from .optimized_dimensionality_reducer import (
    OptimizedDimensionalityReducer,
    DimensionalityReductionConfig,
    create_optimized_dimensionality_reducer
)

from .optimized_hdbscan_clusterer import (
    OptimizedHDBSCANClusterer,
    HDBSCANConfig,
    create_optimized_hdbscan_clusterer
)

from .optimized_post_processor import (
    OptimizedPostProcessor,
    PostProcessingConfig,
    create_optimized_post_processor
)

from .optimized_hdbscan_pipeline import (
    OptimizedHDBSCANPipeline,
    OptimizedHDBSCANPipelineConfig,
    create_optimized_hdbscan_pipeline
)

# Enhanced optimization components
from .enhanced_memory_optimizer import (
    EnhancedMemoryOptimizer,
    MemoryOptimizationConfig,
    create_enhanced_memory_optimizer
)

from .enhanced_hyperparameter_optimizer import (
    EnhancedHyperparameterOptimizer,
    HDBSCANHyperparameterConfig,
    create_enhanced_hyperparameter_optimizer
)

from .enhanced_vectorized_processor import (
    EnhancedVectorizedProcessor,
    VectorizedProcessingConfig,
    create_enhanced_vectorized_processor
)

from .enhanced_hdbscan_integration import (
    EnhancedHDBSCANIntegration,
    EnhancedHDBSCANConfig,
    create_enhanced_hdbscan_integration
)

# Efficient regime feature selection
from .efficient_regime_feature_selector import (
    EfficientRegimeFeatureSelector,
    EfficientMRMRSelector,
    EfficientLASSOSelector,
    RegimeFeatureImportanceScorer,
    EfficientFeatureSelectionConfig,
    create_efficient_regime_feature_selector
)

# Optimized regime feature processing
from .optimized_regime_feature_processor import (
    OptimizedRegimeFeatureProcessor,
    OptimizedRegimeFeatureProcessorConfig,
    create_optimized_regime_feature_processor
)

# HDBSCAN regime optimization
from .hdbscan_regime_optimizer import (
    HDBSCANRegimeOptimizer,
    HDBSCANRegimeOptimizerConfig,
    create_hdbscan_regime_optimizer
)

# Optimized HDBSCAN regime discovery
from .optimized_hdbscan_regime_discovery import (
    OptimizedHDBSCANRegimeDiscovery,
    OptimizedRegimeResult,
    create_optimized_hdbscan_regime_discovery
)

# Feature usage guide
from .feature_usage_guide import (
    HDBSCANFeatureUsageGuide,
    get_hdbscan_feature_usage_guide
)

# Features common integration
from .features_common_integration import (
    FeaturesCommonHDBSCANIntegration,
    FeaturesCommonIntegrationConfig,
    create_features_common_hdbscan_integration
)

from .features_common_benefits_analysis import (
    FeaturesCommonBenefitsAnalysis,
    get_features_common_benefits_analysis
)

__all__ = [
    # Feature extraction optimization
    'OptimizedFeatureExtractor',
    'FeatureExtractionConfig',
    'create_optimized_feature_extractor',
    
    # Preprocessing optimization
    'OptimizedPreprocessor',
    'PreprocessingConfig',
    'create_optimized_preprocessor',
    
    # Dimensionality reduction optimization
    'OptimizedDimensionalityReducer',
    'DimensionalityReductionConfig',
    'create_optimized_dimensionality_reducer',
    
    # HDBSCAN clustering optimization
    'OptimizedHDBSCANClusterer',
    'HDBSCANConfig',
    'create_optimized_hdbscan_clusterer',
    
    # Post-processing optimization
    'OptimizedPostProcessor',
    'PostProcessingConfig',
    'create_optimized_post_processor',
    
    # Unified pipeline
    'OptimizedHDBSCANPipeline',
    'OptimizedHDBSCANPipelineConfig',
    'create_optimized_hdbscan_pipeline',
    
    # Enhanced optimization components
    'EnhancedMemoryOptimizer',
    'MemoryOptimizationConfig',
    'create_enhanced_memory_optimizer',
    'EnhancedHyperparameterOptimizer',
    'HDBSCANHyperparameterConfig',
    'create_enhanced_hyperparameter_optimizer',
    'EnhancedVectorizedProcessor',
    'VectorizedProcessingConfig',
    'create_enhanced_vectorized_processor',
    'EnhancedHDBSCANIntegration',
    'EnhancedHDBSCANConfig',
    'create_enhanced_hdbscan_integration',
    'HDBSCANFeatureUsageGuide',
    'get_hdbscan_feature_usage_guide',
    'FeaturesCommonHDBSCANIntegration',
    'FeaturesCommonIntegrationConfig',
    'create_features_common_hdbscan_integration',
    'FeaturesCommonBenefitsAnalysis',
    'get_features_common_benefits_analysis',
    
    # Efficient regime feature selection
    'EfficientRegimeFeatureSelector',
    'EfficientMRMRSelector',
    'EfficientLASSOSelector',
    'RegimeFeatureImportanceScorer',
    'EfficientFeatureSelectionConfig',
    'create_efficient_regime_feature_selector',
    
    # Optimized regime feature processing
    'OptimizedRegimeFeatureProcessor',
    'OptimizedRegimeFeatureProcessorConfig',
    'create_optimized_regime_feature_processor',
    
    # HDBSCAN regime optimization
    'HDBSCANRegimeOptimizer',
    'HDBSCANRegimeOptimizerConfig',
    'create_hdbscan_regime_optimizer',
    
    # Optimized HDBSCAN regime discovery
    'OptimizedHDBSCANRegimeDiscovery',
    'OptimizedRegimeResult',
    'create_optimized_hdbscan_regime_discovery'
]