"""Step 10 Configuration Module.

This module handles all configuration management for the unified regime
intelligence system, including model parameters, training settings,
and external integrations.
"""

from .step10_config import Step10Config, DEFAULT_CONFIG, create_step10_config

__all__ = [
    'Step10Config',
    'DEFAULT_CONFIG',
    'create_step10_config',
]
