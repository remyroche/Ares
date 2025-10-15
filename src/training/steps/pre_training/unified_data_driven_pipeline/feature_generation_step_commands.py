"""
Feature Generation Step Commands for Ares Launcher Integration

This module provides command handlers for all feature generation steps
that can be launched individually from ares_launcher.py.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass

from src.training.steps.pre_training.unified_data_driven_pipeline.steps import (
    handle_feature_generation_data_validation_step,
    handle_feature_generation_feature_generation_step,
    handle_feature_generation_feature_selection_step,
    handle_feature_generation_period_optimization_step,
    handle_feature_generation_lookback_optimization_step,
    handle_feature_generation_interaction_generation_step,
    handle_feature_generation_vectorization_step,
    handle_feature_generation_labeling_integration_step,
    handle_feature_generation_final_validation_step,
    DataValidationResult,
    FeatureGenerationResult,
    FeatureSelectionResult,
    PeriodOptimizationResult,
    LookbackOptimizationResult,
    InteractionGenerationResult,
    VectorizationResult,
    LabelingIntegrationResult,
    FinalValidationResult
)


@dataclass
class FeatureGenerationStepConfig:
    """Configuration for feature generation step commands."""
    
    # Core parameters (from ares_launcher)
    symbol: str = "ETHUSDT"
    timeframe: str = "15m"
    direction: str = "longs"
    intensity: str = "blank"
    lookback_days: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    exchange: str = "binance"
    
    # Additional parameters
    custom_overrides: Optional[Dict[str, Any]] = None
    artifacts_dir: str = "artifacts"
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.custom_overrides is None:
            self.custom_overrides = {}


class FeatureGenerationStepCommandHandler:
    """Handler for feature generation step commands from ares_launcher."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the command handler.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def create_config(self, **kwargs) -> FeatureGenerationStepConfig:
        """Create configuration from command arguments.
        
        Args:
            **kwargs: Command arguments
            
        Returns:
            FeatureGenerationStepConfig instance
        """
        return FeatureGenerationStepConfig(**kwargs)
    
    # Data Validation Step
    async def handle_data_validation_step(self, **kwargs) -> DataValidationResult:
        """Handle feature_generation_data_validation_step command."""
        config = self.create_config(**kwargs)
        return await handle_feature_generation_data_validation_step(
            symbol=config.symbol,
            timeframe=config.timeframe,
            direction=config.direction,
            intensity=config.intensity,
            lookback_days=config.lookback_days,
            start_date=config.start_date,
            end_date=config.end_date,
            exchange=config.exchange,
            custom_overrides=config.custom_overrides
        )
    
    # Feature Generation Step
    async def handle_feature_generation_step(self, **kwargs) -> FeatureGenerationResult:
        """Handle feature_generation_feature_generation_step command."""
        config = self.create_config(**kwargs)
        return await handle_feature_generation_feature_generation_step(
            symbol=config.symbol,
            timeframe=config.timeframe,
            direction=config.direction,
            intensity=config.intensity,
            lookback_days=config.lookback_days,
            start_date=config.start_date,
            end_date=config.end_date,
            exchange=config.exchange,
            custom_overrides=config.custom_overrides
        )
    
    # Feature Selection Step
    async def handle_feature_selection_step(self, **kwargs) -> FeatureSelectionResult:
        """Handle feature_generation_feature_selection_step command."""
        config = self.create_config(**kwargs)
        return await handle_feature_generation_feature_selection_step(
            symbol=config.symbol,
            timeframe=config.timeframe,
            direction=config.direction,
            intensity=config.intensity,
            lookback_days=config.lookback_days,
            start_date=config.start_date,
            end_date=config.end_date,
            exchange=config.exchange,
            custom_overrides=config.custom_overrides
        )
    
    # Period Optimization Step
    async def handle_period_optimization_step(self, **kwargs) -> PeriodOptimizationResult:
        """Handle feature_generation_period_optimization_step command."""
        config = self.create_config(**kwargs)
        return await handle_feature_generation_period_optimization_step(
            symbol=config.symbol,
            timeframe=config.timeframe,
            direction=config.direction,
            intensity=config.intensity,
            lookback_days=config.lookback_days,
            start_date=config.start_date,
            end_date=config.end_date,
            exchange=config.exchange,
            custom_overrides=config.custom_overrides
        )
    
    # Lookback Optimization Step
    async def handle_lookback_optimization_step(self, **kwargs) -> LookbackOptimizationResult:
        """Handle feature_generation_lookback_optimization_step command."""
        config = self.create_config(**kwargs)
        return await handle_feature_generation_lookback_optimization_step(
            symbol=config.symbol,
            timeframe=config.timeframe,
            direction=config.direction,
            intensity=config.intensity,
            lookback_days=config.lookback_days,
            start_date=config.start_date,
            end_date=config.end_date,
            exchange=config.exchange,
            custom_overrides=config.custom_overrides
        )
    
    # Interaction Generation Step
    async def handle_interaction_generation_step(self, **kwargs) -> InteractionGenerationResult:
        """Handle feature_generation_interaction_generation_step command."""
        config = self.create_config(**kwargs)
        return await handle_feature_generation_interaction_generation_step(
            symbol=config.symbol,
            timeframe=config.timeframe,
            direction=config.direction,
            intensity=config.intensity,
            lookback_days=config.lookback_days,
            start_date=config.start_date,
            end_date=config.end_date,
            exchange=config.exchange,
            custom_overrides=config.custom_overrides
        )
    
    # Vectorization Step
    async def handle_vectorization_step(self, **kwargs) -> VectorizationResult:
        """Handle feature_generation_vectorization_step command."""
        config = self.create_config(**kwargs)
        return await handle_feature_generation_vectorization_step(
            symbol=config.symbol,
            timeframe=config.timeframe,
            direction=config.direction,
            intensity=config.intensity,
            lookback_days=config.lookback_days,
            start_date=config.start_date,
            end_date=config.end_date,
            exchange=config.exchange,
            custom_overrides=config.custom_overrides
        )
    
    # Labeling Integration Step
    async def handle_labeling_integration_step(self, **kwargs) -> LabelingIntegrationResult:
        """Handle feature_generation_labeling_integration_step command."""
        config = self.create_config(**kwargs)
        return await handle_feature_generation_labeling_integration_step(
            symbol=config.symbol,
            timeframe=config.timeframe,
            direction=config.direction,
            intensity=config.intensity,
            lookback_days=config.lookback_days,
            start_date=config.start_date,
            end_date=config.end_date,
            exchange=config.exchange,
            custom_overrides=config.custom_overrides
        )
    
    # Final Validation Step
    async def handle_final_validation_step(self, **kwargs) -> FinalValidationResult:
        """Handle feature_generation_final_validation_step command."""
        config = self.create_config(**kwargs)
        return await handle_feature_generation_final_validation_step(
            symbol=config.symbol,
            timeframe=config.timeframe,
            direction=config.direction,
            intensity=config.intensity,
            lookback_days=config.lookback_days,
            start_date=config.start_date,
            end_date=config.end_date,
            exchange=config.exchange,
            custom_overrides=config.custom_overrides
        )


# Convenience functions for direct command handling
async def handle_feature_generation_data_validation_step_command(**kwargs) -> DataValidationResult:
    """Handle feature_generation_data_validation_step command."""
    handler = FeatureGenerationStepCommandHandler()
    return await handler.handle_data_validation_step(**kwargs)


async def handle_feature_generation_feature_generation_step_command(**kwargs) -> FeatureGenerationResult:
    """Handle feature_generation_feature_generation_step command."""
    handler = FeatureGenerationStepCommandHandler()
    return await handler.handle_feature_generation_step(**kwargs)


async def handle_feature_generation_feature_selection_step_command(**kwargs) -> FeatureSelectionResult:
    """Handle feature_generation_feature_selection_step command."""
    handler = FeatureGenerationStepCommandHandler()
    return await handler.handle_feature_selection_step(**kwargs)


async def handle_feature_generation_period_optimization_step_command(**kwargs) -> PeriodOptimizationResult:
    """Handle feature_generation_period_optimization_step command."""
    handler = FeatureGenerationStepCommandHandler()
    return await handler.handle_period_optimization_step(**kwargs)


async def handle_feature_generation_lookback_optimization_step_command(**kwargs) -> LookbackOptimizationResult:
    """Handle feature_generation_lookback_optimization_step command."""
    handler = FeatureGenerationStepCommandHandler()
    return await handler.handle_lookback_optimization_step(**kwargs)


async def handle_feature_generation_interaction_generation_step_command(**kwargs) -> InteractionGenerationResult:
    """Handle feature_generation_interaction_generation_step command."""
    handler = FeatureGenerationStepCommandHandler()
    return await handler.handle_interaction_generation_step(**kwargs)


async def handle_feature_generation_vectorization_step_command(**kwargs) -> VectorizationResult:
    """Handle feature_generation_vectorization_step command."""
    handler = FeatureGenerationStepCommandHandler()
    return await handler.handle_vectorization_step(**kwargs)


async def handle_feature_generation_labeling_integration_step_command(**kwargs) -> LabelingIntegrationResult:
    """Handle feature_generation_labeling_integration_step command."""
    handler = FeatureGenerationStepCommandHandler()
    return await handler.handle_labeling_integration_step(**kwargs)


async def handle_feature_generation_final_validation_step_command(**kwargs) -> FinalValidationResult:
    """Handle feature_generation_final_validation_step command."""
    handler = FeatureGenerationStepCommandHandler()
    return await handler.handle_final_validation_step(**kwargs)


# Export all command handlers
__all__ = [
    "FeatureGenerationStepCommandHandler",
    "FeatureGenerationStepConfig",
    "handle_feature_generation_data_validation_step_command",
    "handle_feature_generation_feature_generation_step_command",
    "handle_feature_generation_feature_selection_step_command",
    "handle_feature_generation_period_optimization_step_command",
    "handle_feature_generation_lookback_optimization_step_command",
    "handle_feature_generation_interaction_generation_step_command",
    "handle_feature_generation_vectorization_step_command",
    "handle_feature_generation_labeling_integration_step_command",
    "handle_feature_generation_final_validation_step_command"
]