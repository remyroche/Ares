"""
Backtesting Steps Module.

This module registers all backtesting steps for autonomous execution.
"""

from src.training.steps.base_step import step_registry
from .final_parameters_optimization import FinalParametersOptimizer
from .real_parameters_optimization import RealParametersOptimizer

# Register backtesting steps
step_registry.register("final_parameters_optimization", FinalParametersOptimizer)
step_registry.register("real_parameters_optimization", RealParametersOptimizer)