"""
Cross-Timeframe Feature Generation System

A comprehensive pipeline for generating and optimizing higher temporal frequency (HTF) features
with regime-aware optimization, cost-aware materialization, and statistical selection.

Key Components:
- Phase-1: HTF probe stage with coarse adaptive grids
- Phase-2: Optimization with local grids and IC surface fitting
- Regime segmentation and change-point detection
- EHU vs RIH assignment logic
- Knapsack selection with correlation constraints
- HTF-aware interaction templates
- Statistical selection with stability selection and FDR
- Walk-forward evaluation system
"""

from .pipeline import CrossTimeframePipeline
from .phase1_probe import Phase1HTFProbe
from .phase2_optimization import Phase2Optimization
from .regime_segmentation import RegimeSegmentation
from .scoring_system import AdaptiveScoringSystem
from .ehu_rih_assignment import EHU_RIH_Assignment
from .knapsack_selection import KnapsackSelection, CrossTimeframeKnapsackSelectionResult
from .htf_materialization import HTFMaterialization
from .statistical_selection import StatisticalSelection, CrossTimeframeStatisticalSelectionResult
from .evaluation import WalkForwardEvaluation
from .monitoring import MonitoringSystem
from .staleness_curve import StalenessCurveCalculator, StalenessCurve, StalenessSummary

__all__ = [
    'CrossTimeframePipeline',
    'Phase1HTFProbe',
    'Phase2Optimization', 
    'RegimeSegmentation',
    'AdaptiveScoringSystem',
    'EHU_RIH_Assignment',
    'KnapsackSelection',
    'CrossTimeframeKnapsackSelectionResult',
    'HTFMaterialization',
    'StatisticalSelection',
    'CrossTimeframeStatisticalSelectionResult',
    'WalkForwardEvaluation',
    'MonitoringSystem',
    'StalenessCurveCalculator',
    'StalenessCurve',
    'StalenessSummary'
]