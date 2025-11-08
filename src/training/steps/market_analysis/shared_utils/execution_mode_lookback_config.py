"""
Execution Mode Lookback Configuration

This module provides centralized utilities for extracting lookback period configurations
based on execution modes (full, light, blank) from the ares_launcher.

These parameters control the data window sizes and computational intensity for each
component in the market analysis pipeline.
"""

import logging
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger(__name__)

class ExecutionMode(Enum):
    """Execution modes matching ares_launcher."""
    FULL = "full"
    LIGHT = "light"
    BLANK = "blank"

@dataclass
class LookbackConfiguration:
    """Configuration for lookback periods based on execution mode."""
    # Feature lookback optimization parameters
    optimization_window_days: int
    optimization_sample_size: int
    optimization_max_features: int

    # PID-based feature generation parameters
    pid_generation_window_days: int
    pid_interaction_features: int
    pid_polynomial_features: int
    pid_cross_timeframe_features: int

    # Multi-horizon profit labeling parameters
    labeling_window_days: int
    labeling_horizons_count: int
    labeling_sample_size: int

    # Final feature selection parameters
    selection_window_days: int
    selection_stage_targets: Tuple[int, int, int, int]  # (120→100→80→60)

    # General parameters
    data_intensity_percentage: float
    computational_complexity: str

class ExecutionModeLookbackConfig:
    """Centralized configuration for lookback periods based on execution modes."""

    def __init__(self):
        """Initialize the execution mode configuration manager."""
        logger.info("🔧 Initializing Execution Mode Lookback Configuration")

        # Define configurations for each execution mode
        self._configurations = {
            ExecutionMode.FULL: LookbackConfiguration(
                # Feature lookback optimization - Full intensity
                optimization_window_days=1460,  # ~4 years of daily data
                optimization_sample_size=100000,  # Full sample size
                optimization_max_features=80,  # Maximum features to optimize

                # PID-based feature generation - Full complexity
                pid_generation_window_days=1460,  # ~4 years
                pid_interaction_features=100,  # Full interaction features
                pid_polynomial_features=50,  # Full polynomial features
                pid_cross_timeframe_features=50,  # Full cross-timeframe features

                # Multi-horizon profit labeling - Full analysis
                labeling_window_days=1460,  # ~4 years for comprehensive labeling
                labeling_horizons_count=20,  # Multiple horizons for rich signals
                labeling_sample_size=100000,  # Full sample for probability estimation

                # Final feature selection - Full pipeline
                selection_window_days=1460,  # ~4 years for feature selection
                selection_stage_targets=(120, 100, 80, 60),  # Complete 4-stage reduction

                # General parameters
                data_intensity_percentage=100.0,
                computational_complexity="maximum"
            ),

            ExecutionMode.LIGHT: LookbackConfiguration(
                # Feature lookback optimization - Light intensity
                optimization_window_days=10,  # 10 days for quick optimization
                optimization_sample_size=1000,  # Reduced sample size
                optimization_max_features=80,  # Keep all features

                # PID-based feature generation - Light complexity
                pid_generation_window_days=10,  # 10 days
                pid_interaction_features=100,  # Keep all interaction features
                pid_polynomial_features=50,  # Keep all polynomial features
                pid_cross_timeframe_features=50,  # Keep all cross-timeframe features

                # Multi-horizon profit labeling - Light analysis
                labeling_window_days=10,  # 10 days for quick labeling
                labeling_horizons_count=5,  # Reduced horizons
                labeling_sample_size=1000,  # Reduced sample size

                # Final feature selection - Light pipeline
                selection_window_days=10,  # 10 days for quick selection
                selection_stage_targets=(120, 100, 80, 60),  # Keep consistent stage targets

                # General parameters
                data_intensity_percentage=5.0,
                computational_complexity="light"
            ),

            ExecutionMode.BLANK: LookbackConfiguration(
                # Feature lookback optimization - Minimal intensity
                optimization_window_days=180,  # 180 days for validation
                optimization_sample_size=500,  # Minimal sample size
                optimization_max_features=80,  # Keep all features

                # PID-based feature generation - Minimal complexity
                pid_generation_window_days=180,  # 180 days
                pid_interaction_features=100,  # Keep all interaction features
                pid_polynomial_features=50,  # Keep all polynomial features
                pid_cross_timeframe_features=50,  # Keep all cross-timeframe features

                # Multi-horizon profit labeling - Minimal analysis
                labeling_window_days=180,  # 180 days for validation
                labeling_horizons_count=3,  # Minimal horizons
                labeling_sample_size=500,  # Minimal sample size

                # Final feature selection - Minimal pipeline
                selection_window_days=180,  # 180 days for validation
                selection_stage_targets=(120, 100, 80, 60),  # Keep consistent stage targets

                # General parameters
                data_intensity_percentage=10.0,
                computational_complexity="minimal"
            )
        }

        logger.info("✅ Execution Mode Lookback Configuration initialized")

    def get_configuration(self, mode: str) -> LookbackConfiguration:
        """
        Get lookback configuration for a specific execution mode.

        Args:
            mode: Execution mode ('full', 'light', 'blank')

        Returns:
            LookbackConfiguration for the specified mode
        """
        try:
            execution_mode = ExecutionMode(mode.lower())
            config = self._configurations[execution_mode]

            logger.info(f"📊 Retrieved configuration for mode '{mode}':")
            logger.info(f"   - Optimization window: {config.optimization_window_days} days")
            logger.info(f"   - PID generation window: {config.pid_generation_window_days} days")
            logger.info(f"   - Labeling window: {config.labeling_window_days} days")
            logger.info(f"   - Selection window: {config.selection_window_days} days")
            logger.info(f"   - Data intensity: {config.data_intensity_percentage}%")

            return config

        except KeyError:
            logger.warning(f"⚠️ Unknown execution mode '{mode}', defaulting to LIGHT")
            return self._configurations[ExecutionMode.LIGHT]

    def extract_from_pipeline_config(self, pipeline_config: Dict[str, Any]) -> LookbackConfiguration:
        """
        Extract execution mode from pipeline configuration and return corresponding lookback config.

        Args:
            pipeline_config: Pipeline configuration dictionary

        Returns:
            LookbackConfiguration based on pipeline execution mode
        """
        # Try to extract execution mode from various possible locations in config
        execution_mode = None

        # Check for explicit execution mode in config
        if 'execution_mode' in pipeline_config:
            execution_mode = pipeline_config['execution_mode']
        elif 'mode' in pipeline_config:
            execution_mode = pipeline_config['mode']
        elif hasattr(pipeline_config, 'mode'):
            execution_mode = pipeline_config.mode.value if hasattr(pipeline_config.mode, 'value') else str(pipeline_config.mode)
        else:
            # Try to infer from intensity percentage
            intensity = pipeline_config.get('intensity_percentage', 100)
            if intensity >= 90:
                execution_mode = 'full'
            elif intensity >= 20:
                execution_mode = 'light'
            else:
                execution_mode = 'blank'

        logger.info(f"🔍 Extracted execution mode '{execution_mode}' from pipeline config")
        return self.get_configuration(execution_mode)

    def get_optimization_parameters(self, mode: str) -> Dict[str, Any]:
        """Get optimization-specific parameters for a mode."""
        config = self.get_configuration(mode)
        return {
            'window_days': config.optimization_window_days,
            'sample_size': config.optimization_sample_size,
            'max_features': config.optimization_max_features
        }

    def get_pid_generation_parameters(self, mode: str) -> Dict[str, Any]:
        """Get PID generation-specific parameters for a mode."""
        config = self.get_configuration(mode)
        return {
            'window_days': config.pid_generation_window_days,
            'interaction_features': config.pid_interaction_features,
            'polynomial_features': config.pid_polynomial_features,
            'cross_timeframe_features': config.pid_cross_timeframe_features
        }

    def get_labeling_parameters(self, mode: str) -> Dict[str, Any]:
        """Get labeling-specific parameters for a mode."""
        config = self.get_configuration(mode)
        return {
            'window_days': config.labeling_window_days,
            'horizons_count': config.labeling_horizons_count,
            'sample_size': config.labeling_sample_size
        }

    def get_selection_parameters(self, mode: str) -> Dict[str, Any]:
        """Get selection-specific parameters for a mode."""
        config = self.get_configuration(mode)
        return {
            'window_days': config.selection_window_days,
            'stage_targets': config.selection_stage_targets
        }

# Global instance for easy access
execution_mode_config = ExecutionModeLookbackConfig()

def get_execution_mode_config() -> ExecutionModeLookbackConfig:
    """Get the global execution mode configuration instance."""
    return execution_mode_config

def get_lookback_config_from_pipeline(pipeline_config: Dict[str, Any]) -> LookbackConfiguration:
    """
    Convenience function to extract lookback configuration from pipeline config.

    Args:
        pipeline_config: Pipeline configuration dictionary

    Returns:
        LookbackConfiguration based on the pipeline's execution mode
    """
    return execution_mode_config.extract_from_pipeline_config(pipeline_config)

def get_component_lookback_params(component_name: str, mode: str) -> Dict[str, Any]:
    """
    Get lookback parameters for a specific component.

    Args:
        component_name: Name of the component ('optimization', 'pid_generation', 'labeling', 'selection')
        mode: Execution mode ('full', 'light', 'blank')

    Returns:
        Dictionary of lookback parameters for the component
    """
    param_getters = {
        'optimization': execution_mode_config.get_optimization_parameters,
        'pid_generation': execution_mode_config.get_pid_generation_parameters,
        'labeling': execution_mode_config.get_labeling_parameters,
        'selection': execution_mode_config.get_selection_parameters
    }

    getter = param_getters.get(component_name)
    if getter:
        return getter(mode)
    else:
        logger.warning(f"⚠️ Unknown component '{component_name}', returning empty parameters")
        return {}
