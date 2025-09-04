#!/usr/bin/env python3
"""Backtesting Package for Trading Pipeline.

This package contains all the components for backtesting:
- Enhanced backtesting pipeline with validation and error handling
- Walk forward validation per regime
- Monte Carlo validation per regime
- A/B testing per regime
- Model saving and persistence
- Comprehensive validation framework
- Data formatting and access protection decorators
- Common utilities for data operations and error handling
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

# Enhanced backtesting components
from .enhanced_backtesting_pipeline import (
    EnhancedBacktestingPipeline,
    BacktestingConfig,
    run_enhanced_backtesting_pipeline
)
from .validation_framework import (
    BacktestingValidationOrchestrator,
    ValidationResult,
    ValidationStatus
)
from .step_validators import StepValidationOrchestrator
from .decorators import (
    BacktestingDecorators,
    DataFormattingDecorator,
    AnalysisProtectionDecorator,
    DataAccessProtectionDecorator,
    PerformanceMonitoringDecorator
)
from .common_utilities import (
    DataOperationUtilities,
    ErrorHandlingUtilities,
    PipelineManagementUtilities,
    ConfigurationUtilities,
    LoggingUtilities
)

# Main pipeline function (enhanced version)
async def run_backtesting_pipeline(symbol, exchange, timeframe, data_dir, **config):
    """Run the complete enhanced backtesting pipeline."""
    try:
        # Use enhanced backtesting pipeline
        pipeline_config = BacktestingConfig(
            symbol=symbol,
            exchange=exchange,
            data_dir=data_dir,
            timeframe=timeframe,
            enable_validation=config.get('enable_validation', True),
            strict_mode=config.get('strict_mode', True),
            initial_capital=config.get('initial_capital', 10000.0),
            commission=config.get('commission', 0.001),
            slippage=config.get('slippage', 0.0005)
        )
        
        pipeline = EnhancedBacktestingPipeline(pipeline_config)
        success = await pipeline.run_complete_pipeline()
        
        if success:
            print("✅ Enhanced backtesting pipeline completed successfully")
        else:
            print("❌ Enhanced backtesting pipeline failed")
        
        return success
        
    except Exception as e:
        print(f"Enhanced backtesting pipeline failed: {e}")
        return False

# Legacy pipeline function (for backward compatibility)
async def run_legacy_backtesting_pipeline(symbol, exchange, timeframe, data_dir, **config):
    """Run the legacy backtesting pipeline."""
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
        print(f"Legacy backtesting pipeline failed: {e}")
        return False

__all__ = [
    # Enhanced components
    'EnhancedBacktestingPipeline',
    'BacktestingConfig',
    'run_enhanced_backtesting_pipeline',
    'BacktestingValidationOrchestrator',
    'ValidationResult',
    'ValidationStatus',
    'StepValidationOrchestrator',
    'BacktestingDecorators',
    'DataFormattingDecorator',
    'AnalysisProtectionDecorator',
    'DataAccessProtectionDecorator',
    'PerformanceMonitoringDecorator',
    'DataOperationUtilities',
    'ErrorHandlingUtilities',
    'PipelineManagementUtilities',
    'ConfigurationUtilities',
    'LoggingUtilities',
    
    # Legacy components
    'WalkForwardValidationPerRegimeStep',
    'WalkForwardValidationValidator',
    'MonteCarloValidationPerRegimeStep',
    'MonteCarloValidationValidator',
    'ABTestingPerRegimeStep',
    'ABTestingValidator',
    'SavingStep',
    'PerRegimeSavingStep',
    'SavingValidator',
    
    # Main functions
    'run_backtesting_pipeline',
    'run_legacy_backtesting_pipeline'
]