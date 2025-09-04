#!/usr/bin/env python3
"""Backtesting Package for Trading Pipeline.

This package contains all the components for backtesting:
- Walk forward validation per regime
- Monte Carlo validation per regime
- A/B testing per regime
- Model saving and persistence
"""

from .step18_walk_forward_validation_per_regime import WalkForwardValidationPerRegimeStep
from .step18_walk_forward_validation_validator import WalkForwardValidationValidator
from .step19_monte_carlo_validation_per_regime import MonteCarloValidationPerRegimeStep
from .step19_monte_carlo_validation_validator import MonteCarloValidationValidator
from .step20_ab_testing_per_regime import ABTestingPerRegimeStep
from .step20_ab_testing_validator import ABTestingValidator
from .step21_saving import SavingStep
from .step21_saving_per_regime import PerRegimeSavingStep
from .step21_saving_validator import SavingValidator

# Main pipeline function
async def run_backtesting_pipeline(symbol, exchange, timeframe, data_dir, **config):
    """Run the complete backtesting pipeline."""
    try:
        # Step 1: Walk Forward Validation (if enabled)
        if config.get('walk_forward_validation', True):
            walk_forward = WalkForwardValidationPerRegimeStep()
            await walk_forward.validate_walk_forward(symbol, exchange, timeframe, data_dir)
        
        # Step 2: Monte Carlo Validation (if enabled)
        if config.get('monte_carlo_validation', True):
            monte_carlo = MonteCarloValidationPerRegimeStep()
            await monte_carlo.validate_monte_carlo(symbol, exchange, timeframe, data_dir)
        
        # Step 3: A/B Testing (if enabled)
        if config.get('ab_testing', True):
            ab_tester = ABTestingPerRegimeStep()
            await ab_tester.run_ab_testing(symbol, exchange, timeframe, data_dir)
        
        # Step 4: Model Saving (if enabled)
        if config.get('model_saving', True):
            model_saver = SavingStep()
            await model_saver.save_models(symbol, exchange, timeframe, data_dir)
        
        return True
        
    except Exception as e:
        print(f"Backtesting pipeline failed: {e}")
        return False

__all__ = [
    'WalkForwardValidationPerRegimeStep',
    'WalkForwardValidationValidator',
    'MonteCarloValidationPerRegimeStep',
    'MonteCarloValidationValidator',
    'ABTestingPerRegimeStep',
    'ABTestingValidator',
    'SavingStep',
    'PerRegimeSavingStep',
    'SavingValidator',
    'run_backtesting_pipeline'
]