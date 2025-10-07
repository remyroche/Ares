"""
Data-Driven Feature Selection System

This module implements a budgeted experimental design approach to feature selection
that treats each feature generator as an arm with expected utility, cost, and
availability constraints. The system uses a two-phase gating process to efficiently
select the most promising features for the lookback optimization system.

Key Components:
- Phase 1: Cheap probes to estimate predictive value and stability
- Phase 2: Rich probes with rigorous data-driven lookback optimization
- Budgeted selection: Knapsack-style optimization under compute/latency constraints
- Interaction generation: Only for selected parent features
- Final model selection: Stability selection with FDR control

Usage:
    from src.training.steps.pre_training.interaction_feature_generator.data_driven_feature_selection import DataDrivenFeatureSelector
    
    selector = DataDrivenFeatureSelector()
    selected_features = await selector.select_features(market_data, targets, budget_config)
"""

from .feature_selector import DataDrivenFeatureSelector
from .phase1_cheap_probes import Phase1CheapProbes
from .phase2_rich_probes import Phase2RichProbes
from .budgeted_selection import BudgetedFeatureSelection
from .interaction_generator import InteractionFeatureGenerator
from .final_model_selection import FinalModelSelection
from .config import DataDrivenFeatureSelectionConfig, BudgetConfig, Phase1Config, Phase2Config
from .utils import FeatureGeneratorWrapper, CostEstimator, UtilityEstimator

__all__ = [
    'DataDrivenFeatureSelector',
    'Phase1CheapProbes',
    'Phase2RichProbes', 
    'BudgetedFeatureSelection',
    'InteractionFeatureGenerator',
    'FinalModelSelection',
    'DataDrivenFeatureSelectionConfig',
    'BudgetConfig',
    'Phase1Config',
    'Phase2Config',
    'FeatureGeneratorWrapper',
    'CostEstimator',
    'UtilityEstimator'
]

__version__ = "1.0.0"