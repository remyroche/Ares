"""
Data-Driven Lookback Optimization System

This package implements a comprehensive three-stage Bayesian optimization system
for selecting optimal lookback periods for feature families, replacing hardcoded
ceilings with data-driven inference while maintaining production constraints.

Key Features:
- Stage 1: IC Surface Estimation with HAC standard errors and spline fitting
- Stage 2: Walk-Forward Stability Testing with purged cross-validation
- Stage 3: Hierarchical Bayesian Shrinkage across families and symbols
- Cost-aware optimization with CPU, staleness, and uncertainty penalties
- Hysteresis and simplicity priors for stable production deployment
- Support for both discrete and blended lookback approaches

Usage:
    from src.training.steps.pre_training.interaction_feature_generator import (
        LookbackOptimizationOrchestrator,
        create_default_config,
        create_development_config,
        create_production_config
    )
    
    # Create configuration
    config = create_production_config()
    
    # Initialize orchestrator
    orchestrator = LookbackOptimizationOrchestrator(config)
    
    # Run optimization
    result = orchestrator.optimize_lookbacks(data, targets, feature_names)
    
    # Generate report
    report = orchestrator.generate_comprehensive_report(result)
"""

# Import main orchestrator
from .orchestrator import LookbackOptimizationOrchestrator, OptimizationResult

# Import configuration utilities
from .config import (
    LookbackOptimizationConfig,
    create_default_config,
    create_development_config,
    create_production_config,
    FamilyType,
    OptimizationMode
)

# Import individual stage modules for advanced usage
from .ic_surface import ICSurfaceEstimator, ICSurfaceResult
from .wf_stability import StabilityTester, StabilityResult, MultiFamilyStabilityTester
from .hierarchical import HierarchicalBayesianShrinkage, HierarchicalResult, MultiSymbolHierarchicalShrinkage
from .decision import LookbackDecisionMaker, DecisionResult, MultiFamilyDecisionMaker
from .feature_families import MultiFamilyFeatureGenerator, FeatureResult

# Version information
__version__ = "1.0.0"
__author__ = "Ares Trading System"
__description__ = "Data-driven lookback optimization with Bayesian shrinkage"

# Public API
__all__ = [
    # Main orchestrator
    "LookbackOptimizationOrchestrator",
    "OptimizationResult",
    
    # Configuration
    "LookbackOptimizationConfig",
    "create_default_config",
    "create_development_config", 
    "create_production_config",
    "FamilyType",
    "OptimizationMode",
    
    # Stage 1: IC Surface Estimation
    "ICSurfaceEstimator",
    "ICSurfaceResult",
    
    # Stage 2: Walk-Forward Stability
    "StabilityTester",
    "StabilityResult",
    "MultiFamilyStabilityTester",
    
    # Stage 3: Hierarchical Shrinkage
    "HierarchicalBayesianShrinkage",
    "HierarchicalResult",
    "MultiSymbolHierarchicalShrinkage",
    
    # Decision Making
    "LookbackDecisionMaker",
    "DecisionResult",
    "MultiFamilyDecisionMaker",
    
    # Feature Generation
    "MultiFamilyFeatureGenerator",
    "FeatureResult",
]

# Feature Interaction Generation Module
# This module provides the end-to-end interactive feature generation system
# that replaces PID-based feature generation with a comprehensive approach.

# Key Components:
# - InteractiveFeatureGenerationComponent: Main component for interactive generation
# - End-to-end interactive system with all modules
# - Complete feature engineering pipeline
# - Validation, monitoring, and deployment systems

from .interactive_feature_generation_component import (
    InteractiveFeatureGenerationComponent,
    InteractiveFeatureGenerationStatus
)

from .end_to_end_roadmap import (
    EndToEndRoadmapSystem,
    SystemConfig,
    SystemResult,
    create_end_to_end_system,
    run_end_to_end_pipeline
)

__all__ = [
    'InteractiveFeatureGenerationComponent',
    'InteractiveFeatureGenerationStatus',
    'EndToEndRoadmapSystem',
    'SystemConfig',
    'SystemResult',
    'create_end_to_end_system',
    'run_end_to_end_pipeline'
]

__version__ = '1.0.0'
__author__ = 'End-to-End Roadmap System'
__description__ = 'Comprehensive end-to-end roadmap feature generation system'
