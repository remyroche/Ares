"""
Ablation Studies Framework for UnifiedDataDrivenPipeline

Provides comprehensive ablation study capabilities to validate the contribution
of different pipeline components and configurations.

Key Features:
- Systematic ablation of pipeline components
- Statistical significance testing
- Performance delta calculations
- Comprehensive reporting
"""

from .ablation_framework import (
    AblationStudyFramework,
    AblationStudyConfig,
    AblationResult,
    AblationDelta,
    AblationReport
)

from .component_ablation import (
    ComponentAblationConfig,
    ComponentAblationStudy,
    AblationComponent
)

from .statistical_ablation import (
    StatisticalAblationConfig,
    StatisticalAblationStudy,
    AblationStatisticalTest
)

from .performance_ablation import (
    PerformanceAblationConfig,
    PerformanceAblationStudy,
    AblationPerformanceMetrics
)

__all__ = [
    # Main framework
    'AblationStudyFramework',
    'AblationStudyConfig', 
    'AblationResult',
    'AblationDelta',
    'AblationReport',
    
    # Component ablation
    'ComponentAblationConfig',
    'ComponentAblationStudy',
    'AblationComponent',
    
    # Statistical ablation
    'StatisticalAblationConfig',
    'StatisticalAblationStudy',
    'AblationStatisticalTest',
    
    # Performance ablation
    'PerformanceAblationConfig',
    'PerformanceAblationStudy',
    'AblationPerformanceMetrics'
]