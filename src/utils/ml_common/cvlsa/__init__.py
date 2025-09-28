"""
Enhanced PatchTST (Patch Time Series Transformer) Architecture

This package contains the complete implementation of the Enhanced PatchTST architecture
with advanced features for financial machine learning applications.

Key Components:
- cvlsa_architecture: Core PatchTST implementation with cross-view attention
- cvlsa_integration: Integration with tree-based models
- adaptive_cascade_architecture: Adaptive cascade system with genetic optimization
- enhanced_variable_selection: Enhanced variable selection with parallel processing
- improved_feature_engineering: Advanced feature engineering with domain knowledge
- performance_memory_management: Performance & memory management
- robust_error_handling: Robust error handling and validation
- advanced_monitoring_analytics: Monitoring and analytics system
- configuration_simplification: Configuration profiles and auto-configuration

Quick Start:
    from src.utils.ml_common.cvlsa.cvlsa_architecture import create_enhanced_patchtst_model
    from src.utils.ml_common.cvlsa.configuration_simplification import create_configuration_simplification
    
    # Auto-configure the system
    config_simplifier = create_configuration_simplification()
    auto_result = config_simplifier.auto_configure(X, y, use_case='research')
    
    # Create and use PatchTST model
    patchtst_model = create_enhanced_patchtst_model(auto_result.config)
"""

# Core PatchTST components
from .cvlsa_architecture import (
    EnhancedPatchTSTConfig,
    CrossViewAttention,
    MultiScaleTemporalAttention,
    MemoryEfficientPatchTST,
    BayesianHyperparameterOptimizer,
    EnhancedPatchTSTTrainer,
    create_enhanced_patchtst_model,
    create_patchtst_config
)

from .cvlsa_integration import (
    PatchTSTTreeModel,
    PatchTSTFeatureExtractor,
    create_default_patchtst_tree_model,
    create_patchtst_feature_extractor
)

# Adaptive cascade architecture
from .adaptive_cascade_architecture import (
    CascadeLevel,
    GeneticOptimizationConfig,
    AdaptiveCascadeArchitecture,
    create_adaptive_cascade
)

# Enhanced variable selection
from .enhanced_variable_selection import (
    SelectionMethod,
    VariableSelectionConfig,
    EnhancedVariableSelector,
    create_enhanced_variable_selector
)

# Improved feature engineering
from .improved_feature_engineering import (
    FeatureEngineeringConfig,
    ImprovedFeatureEngineer,
    create_improved_feature_engineer
)

# Performance and memory management
from .performance_memory_management import (
    ResourceConfig,
    ModelCache,
    ResourceMonitor,
    IncrementalLearner,
    PerformanceMemoryManager,
    create_performance_memory_manager
)

# Robust error handling
from .robust_error_handling import (
    ErrorSeverity,
    ValidationError,
    ErrorReport,
    ValidationConfig,
    InputValidator,
    ErrorRecovery,
    RobustErrorHandler,
    robust_operation,
    create_robust_error_handler
)

# Advanced monitoring and analytics
from .advanced_monitoring_analytics import (
    ExperimentConfig,
    Experiment,
    ExperimentTracker,
    PerformanceAnalytics,
    RealTimeMonitor,
    AdvancedMonitoringAnalytics,
    create_advanced_monitoring_analytics
)

# Configuration simplification
from .configuration_simplification import (
    ConfigurationProfile,
    ConfigurationProfileData,
    AutoConfigurationResult,
    ConfigurationProfiles,
    AutoConfiguration,
    ConfigurationValidator,
    ConfigurationSimplification,
    create_configuration_simplification
)

__version__ = "1.0.0"
__author__ = "Enhanced PatchTST Team"
__description__ = "Enhanced PatchTST Architecture for Financial Machine Learning"

# Main exports for easy access
__all__ = [
    # Core PatchTST
    "EnhancedPatchTSTConfig",
    "CrossViewAttention", 
    "MultiScaleTemporalAttention",
    "MemoryEfficientPatchTST",
    "EnhancedPatchTSTTrainer",
    "create_enhanced_patchtst_model",
    
    # Integration
    "PatchTSTTreeModel",
    "PatchTSTFeatureExtractor",
    "create_default_patchtst_tree_model",
    
    # Adaptive cascade
    "AdaptiveCascadeArchitecture",
    "create_adaptive_cascade",
    
    # Variable selection
    "EnhancedVariableSelector",
    "create_enhanced_variable_selector",
    
    # Feature engineering
    "ImprovedFeatureEngineer",
    "create_improved_feature_engineer",
    
    # Performance management
    "PerformanceMemoryManager",
    "create_performance_memory_manager",
    
    # Error handling
    "RobustErrorHandler",
    "create_robust_error_handler",
    
    # Monitoring
    "AdvancedMonitoringAnalytics",
    "create_advanced_monitoring_analytics",
    
    # Configuration
    "ConfigurationSimplification",
    "create_configuration_simplification"
]