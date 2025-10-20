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

# Enhanced HDBSCAN integration (kept for compatibility)
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

# Features common benefits analysis (kept for reference)
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
    
    # Enhanced HDBSCAN integration
    'EnhancedHDBSCANIntegration',
    'EnhancedHDBSCANConfig',
    'create_enhanced_hdbscan_integration',
    'HDBSCANFeatureUsageGuide',
    'get_hdbscan_feature_usage_guide',
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