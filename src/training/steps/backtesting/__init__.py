"""
Backtesting Steps Module.

This module registers all backtesting steps for autonomous execution.
"""

from src.training.steps.base_step import step_registry
from .final_parameters_optimization import FinalParametersOptimizer
from .real_parameters_optimization import RealParametersOptimizer
from .basic_backtesting_pre import BasicBacktestingPreStep
from .basic_backtesting_post import BasicBacktestingPostStep

# Register backtesting steps
step_registry.register("final_parameters_optimization", FinalParametersOptimizer)
step_registry.register("real_parameters_optimization", RealParametersOptimizer)
step_registry.register("basic_backtesting_pre", BasicBacktestingPreStep)
step_registry.register("basic_backtesting_post", BasicBacktestingPostStep)