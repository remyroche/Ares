"""
Enhanced Artifact Manager Integration Helper

This module provides helper functions to integrate the enhanced artifact manager
across all pre-training steps with proper context setting and file management.
"""

from typing import Optional, Dict, Any
from datetime import datetime
import logging

from src.training.steps.pre_training.utils.artifact_manager import (
    get_pretraining_artifact_manager,
    ArtifactConfig
)

def setup_enhanced_artifact_manager(
    symbol: Optional[str] = None,
    exchange: Optional[str] = None,
    direction: str = "long",
    model: str = "Analyst",
    information: str = "pre_training",
    datetime: Optional[datetime] = None
) -> Any:
    """
    Set up the enhanced artifact manager with proper context.
    
    Args:
        symbol: Trading symbol (e.g., 'ETHUSDT')
        exchange: Exchange name (e.g., 'binance')
        direction: Trading direction ('long' or 'short')
        model: Model type ('Analyst' or 'Tactician')
        information: Information prefix for file naming
        datetime: Specific datetime, defaults to current time
    
    Returns:
        Configured artifact manager instance
    """
    # Get artifact manager instance
    am = get_pretraining_artifact_manager()
    
    # Configure enhanced settings
    am.config = ArtifactConfig(
        include_symbol_in_filename=True,
        include_exchange_in_filename=True,
        include_datetime_in_filename=True,
        include_information_in_filename=True,
        include_direction_in_filename=True,
        include_model_in_filename=True,
        use_joint_parquet_format=True,
        generate_json_metadata=True
    )
    
    # Set context with enhanced parameters
    am.set_context(
        symbol=symbol,
        exchange=exchange,
        direction=direction,
        model=model,
        information=information,
        datetime=datetime or datetime.utcnow()
    )
    
    return am

def get_step_context_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract context information from step configuration.
    
    Args:
        config: Step configuration dictionary
        
    Returns:
        Context dictionary with symbol, exchange, direction, model
    """
    return {
        'symbol': config.get('symbol'),
        'exchange': config.get('exchange'),
        'direction': config.get('direction', 'long'),
        'model': config.get('model', 'Analyst'),
        'information': config.get('information', 'pre_training')
    }

def log_artifact_operation(operation: str, step_name: str, key: str, success: bool = True):
    """
    Log artifact operations with enhanced formatting.
    
    Args:
        operation: Operation type (save, load, etc.)
        step_name: Name of the step
        key: Artifact key
        success: Whether operation was successful
    """
    status = "✅" if success else "❌"
    logger = logging.getLogger(__name__)
    logger.info(f"{status} {operation}: {step_name}/{key}")

def create_enhanced_step_wrapper(step_class):
    """
    Create an enhanced wrapper for pre-training steps that automatically
    sets up the artifact manager context.
    
    Args:
        step_class: The step class to wrap
        
    Returns:
        Enhanced step class with artifact manager integration
    """
    class EnhancedStepWrapper(step_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._artifact_manager = None
            self._context_set = False
        
        def _setup_artifact_manager(self, **context_kwargs):
            """Set up the enhanced artifact manager with context."""
            if not self._context_set:
                self._artifact_manager = setup_enhanced_artifact_manager(**context_kwargs)
                self._context_set = True
            return self._artifact_manager
        
        def get_artifact_manager(self):
            """Get the configured artifact manager."""
            if not self._artifact_manager:
                # Extract context from step configuration
                context = get_step_context_from_config(self.config)
                self._setup_artifact_manager(**context)
            return self._artifact_manager

    try:
        phase_methods = [
            "_phase3_1_shallow_sweep",
            "_phase3_2_deeper_refinement",
            "_phase3_3_interaction_discovery",
        ]
        for method_name in phase_methods:
            if hasattr(step_class, method_name) and not hasattr(EnhancedStepWrapper, method_name):
                setattr(EnhancedStepWrapper, method_name, getattr(step_class, method_name))
    except Exception:
        pass

    return EnhancedStepWrapper

# Pre-configured step contexts for different model types
ANALYST_STEP_CONTEXT = {
    'direction': 'long',
    'model': 'Analyst',
    'information': 'pre_training'
}

TACTICIAN_STEP_CONTEXT = {
    'direction': 'long',  # Can be overridden
    'model': 'Tactician',
    'information': 'pre_training'
}

def get_analyst_context(symbol: str, exchange: str, **overrides):
    """Get context for Analyst model steps."""
    context = ANALYST_STEP_CONTEXT.copy()
    context.update({
        'symbol': symbol,
        'exchange': exchange,
        **overrides
    })
    return context

def get_tactician_context(symbol: str, exchange: str, direction: str = "long", **overrides):
    """Get context for Tactician model steps."""
    context = TACTICIAN_STEP_CONTEXT.copy()
    context.update({
        'symbol': symbol,
        'exchange': exchange,
        'direction': direction,
        **overrides
    })
    return context
