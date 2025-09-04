#!/usr/bin/env python3
"""Optimization Package for Trading Pipeline.

This package contains all the components for optimization:
- Confidence calibration per regime
- Final parameters optimization
- Parameter optimization wrapper
"""

from .step16_confidence_calibration_per_regime import ConfidenceCalibrationPerRegimeStep
from .step16_confidence_calibration_validator import ConfidenceCalibrationValidator
from .step17_final_parameters_optimization_new import FinalParametersOptimizationStep
from .step17_final_parameters_optimization_per_regime import PerRegimeFinalParametersOptimizationStep
from .step17_final_parameters_optimization_validator import FinalParametersOptimizationValidator
from .step17_parameter_optimization_wrapper import ParameterOptimizationWrapper

# Main pipeline function
async def run_optimisation_pipeline(symbol, exchange, timeframe, data_dir, **config):
    """Run the complete optimization pipeline."""
    try:
        # Step 1: Confidence Calibration (if enabled)
        if config.get('confidence_calibration', True):
            confidence_calibrator = ConfidenceCalibrationPerRegimeStep()
            await confidence_calibrator.calibrate_confidence(symbol, exchange, timeframe, data_dir)
        
        # Step 2: Final Parameters Optimization (if enabled)
        if config.get('parameter_optimization', True):
            param_optimizer = FinalParametersOptimizationStep()
            await param_optimizer.optimize_parameters(symbol, exchange, timeframe, data_dir)
        
        return True
        
    except Exception as e:
        print(f"Optimization pipeline failed: {e}")
        return False

__all__ = [
    'ConfidenceCalibrationPerRegimeStep',
    'ConfidenceCalibrationValidator',
    'FinalParametersOptimizationStep',
    'PerRegimeFinalParametersOptimizationStep',
    'FinalParametersOptimizationValidator',
    'ParameterOptimizationWrapper',
    'run_optimisation_pipeline'
]