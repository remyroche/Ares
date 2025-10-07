"""
End-to-End Roadmap Generation Module

This module provides comprehensive end-to-end roadmap feature generation
that replaces the PID-based feature generation system.

Key Components:
- EndToEndRoadmapComponent: Main component for roadmap generation
- System contracts and configuration
- Data contracts and validation
- Feature registry with parent features
- Transform system with EW-Z, TOD Rank, Signed-log, Winsorization
- Lookback selection with hysteresis
- Interaction engine with 15 locked interactions
- Patch/GRU model integration
- Assembly DAG orchestration
- Walk-forward validation
- Monitoring and retrain decision tree
- CI/CD gates and tests
- Rollout plan with shadow/canary/full deployment
"""

from .end_to_end_roadmap_component import (
    EndToEndRoadmapComponent,
    RoadmapStatus
)

__all__ = [
    'EndToEndRoadmapComponent',
    'RoadmapStatus'
]

__version__ = '1.0.0'
__author__ = 'End-to-End Roadmap System'
__description__ = 'Comprehensive end-to-end roadmap feature generation system'