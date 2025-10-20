"""
Training-specific artifact manager utilities.

This module re-exports the main artifact manager functions for training modules.
"""

from src.utils.artifact_manager import (
    ArtifactManager, 
    get_analyst_context, 
    setup_enhanced_artifact_manager, 
    get_pretraining_artifact_manager
)

def get_step_context_from_config(config: dict) -> dict:
    """Get step context from configuration."""
    return {
        'symbol': config.get('symbol', 'UNKNOWN'),
        'timeframe': config.get('timeframe', '15m'),
        'exchange': config.get('exchange', 'binance'),
        'execution_mode': config.get('execution_mode', 'light'),
        'step_name': config.get('step_name', 'unknown')
    }

__all__ = [
    'ArtifactManager', 
    'get_analyst_context', 
    'setup_enhanced_artifact_manager', 
    'get_pretraining_artifact_manager',
    'get_step_context_from_config'
]