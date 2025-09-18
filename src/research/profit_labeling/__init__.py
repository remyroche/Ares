"""
Multi-Horizon Profit Labeling Research Framework

A comprehensive research framework for analyzing and optimizing the multi-horizon profit 
labeling system from a data-driven perspective. This framework provides tools to:

1. Analyze labeling heuristics and their effectiveness
2. Validate labeling quality and consistency 
3. Optimize labeling parameters systematically
4. Visualize labeling patterns and performance
5. Compare different labeling strategies

Key Components:
- HeuristicAnalyzer: Analyzes the effectiveness of profit labeling heuristics
- LabelingValidator: Validates labeling quality and consistency
- ParameterOptimizer: Optimizes labeling parameters using systematic approaches
- LabelingVisualizer: Comprehensive visualization system for labeling analysis
- ResearchRunner: Main runner for research workflows and experiments

Usage:
    from src.research.profit_labeling import (
        HeuristicAnalyzer,
        LabelingValidator, 
        ParameterOptimizer,
        LabelingVisualizer,
        ResearchRunner
    )
"""

from .heuristic_analyzer import HeuristicAnalyzer, HeuristicAnalysisConfig
from .labeling_validator import LabelingValidator, ValidationConfig
from .parameter_optimizer import ParameterOptimizer, OptimizationConfig
from .labeling_visualizer import LabelingVisualizer, VisualizationConfig
from .research_runner import ResearchRunner, ResearchConfig

__all__ = [
    'HeuristicAnalyzer',
    'HeuristicAnalysisConfig',
    'LabelingValidator', 
    'ValidationConfig',
    'ParameterOptimizer',
    'OptimizationConfig',
    'LabelingVisualizer',
    'VisualizationConfig',
    'ResearchRunner',
    'ResearchConfig'
]

__version__ = '1.0.0'
__author__ = 'Ares Trading System'
__description__ = 'Multi-Horizon Profit Labeling Research Framework'
