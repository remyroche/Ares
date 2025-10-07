"""
Feature Interaction Generation Module

This module provides the end-to-end roadmap feature generation system
that replaces PID-based feature generation with a comprehensive approach.

Key Components:
- RoadmapFeatureGenerationComponent: Main component for roadmap generation
- End-to-end roadmap system with all modules
- Complete feature engineering pipeline
- Validation, monitoring, and deployment systems
"""

from .roadmap_feature_generation_component import (
    RoadmapFeatureGenerationComponent,
    RoadmapStatus
)

from .end_to_end_roadmap import (
    EndToEndRoadmapSystem,
    SystemConfig,
    SystemResult,
    create_end_to_end_system,
    run_end_to_end_pipeline
)

__all__ = [
    'RoadmapFeatureGenerationComponent',
    'RoadmapStatus',
    'EndToEndRoadmapSystem',
    'SystemConfig',
    'SystemResult',
    'create_end_to_end_system',
    'run_end_to_end_pipeline'
]

__version__ = '1.0.0'
__author__ = 'End-to-End Roadmap System'
__description__ = 'Comprehensive end-to-end roadmap feature generation system'