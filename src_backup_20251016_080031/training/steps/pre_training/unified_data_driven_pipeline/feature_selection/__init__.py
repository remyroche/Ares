"""
Feature Selection Module

Multi-objective feature selection with explicit objectives.
"""

from .multi_objective_selector import (
    MultiObjectiveFeatureSelector,
    ObjectiveFunction,
    OutOfSampleSharpeObjective,
    DrawdownObjective,
    TurnoverObjective,
    StabilityObjective,
    DiversityObjective,
    MutualInformationObjective,
    ProfitCenteredObjective,
    create_default_objectives,
    create_performance_objectives,
    create_stability_objectives,
    create_balanced_objectives
)

__all__ = [
    'MultiObjectiveFeatureSelector',
    'ObjectiveFunction',
    'OutOfSampleSharpeObjective',
    'DrawdownObjective',
    'TurnoverObjective',
    'StabilityObjective',
    'DiversityObjective',
    'MutualInformationObjective',
    'ProfitCenteredObjective',
    'create_default_objectives',
    'create_performance_objectives',
    'create_stability_objectives',
    'create_balanced_objectives'
]