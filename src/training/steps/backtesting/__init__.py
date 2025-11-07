"""
Backtesting Steps Module.

This module registers all backtesting steps for autonomous execution.
"""

from src.training.steps.base_step import step_registry

# Import backtesting steps (order matters for dependencies)
from .basic_backtesting_pre_step import BasicBacktestingPreStep
from .basic_backtesting_post_step import BasicBacktestingPostStep
from .monte_carlo_simulation_step import MonteCarloSimulationStep
from .ab_testing_step import ABTestingStep
from .reporting_step import ReportingStep
from .final_parameters_optimization import FinalParametersOptimizer
from .real_parameters_optimization import RealParametersOptimizer

# Register backtesting steps (these will be registered again when imported, but that's safe)
# The individual step files also auto-register themselves
step_registry.register("basic_backtesting_pre", BasicBacktestingPreStep)
step_registry.register("basic_backtesting_post", BasicBacktestingPostStep)
step_registry.register("monte_carlo_simulation", MonteCarloSimulationStep)
step_registry.register("ab_testing", ABTestingStep)
step_registry.register("reporting", ReportingStep)
step_registry.register("final_parameters_optimization", FinalParametersOptimizer)
step_registry.register("real_parameters_optimization", RealParametersOptimizer)