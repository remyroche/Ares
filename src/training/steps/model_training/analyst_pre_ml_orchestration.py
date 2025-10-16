"""
Analyst Pre-ML Orchestration - Import Redirect

This module redirects to the actual implementation in models_training directory.
The pre_training subpipeline has been updated to use the new unified approach.
"""

# Import from the actual implementation
from ..models_training.analyst_pre_ml_orchestration import (
    AnalystPreMLConfig,
    AnalystPreMLResult,
    AnalystPreMLOrchestrator,
    execute_analyst_pre_ml_orchestration,
    OrchestrationPhase
)

# Re-export for compatibility
__all__ = [
    'AnalystPreMLConfig',
    'AnalystPreMLResult', 
    'AnalystPreMLOrchestrator',
    'execute_analyst_pre_ml_orchestration',
    'OrchestrationPhase'
]
