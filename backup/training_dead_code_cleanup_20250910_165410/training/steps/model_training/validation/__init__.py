"""Validation package for model training steps."""

# Import from consolidated backtesting step
try:
    from src.training.steps.backtesting import ConsolidatedBacktestingStep
    from src.utils.common_ml.backtesting import ABTestingEngine
    
    # Create compatibility aliases
    ABTestingStep = ABTestingEngine
    run_step = None  # Use ConsolidatedBacktestingStep instead
    
    __all__ = ['ABTestingStep', 'run_step', 'ConsolidatedBacktestingStep']
except ImportError:
    # Fallback if consolidated backtesting is not available
    ABTestingStep = None
    run_step = None
    ConsolidatedBacktestingStep = None
    
    __all__ = ['ABTestingStep', 'run_step', 'ConsolidatedBacktestingStep']