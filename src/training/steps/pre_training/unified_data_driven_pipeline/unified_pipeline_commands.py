"""
Unified Pipeline Commands for Ares Launcher Integration

This module provides command handlers for the existing ares_launcher.py commands:
- --unified-pipeline-analyst
- --unified-pipeline-tactician
- --unified-pipeline-analyst-short
- --unified-pipeline-tactician-long

The difference between tactician and analyst is in the labels used to qualify the financial data.
"""

import argparse
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path

from .core.simplified_config import (
    create_full_config, create_blank_config, create_light_config,
    create_config_by_intensity, PipelineIntensity
)
from .refactored_pipeline import (
    RefactoredUnifiedPipeline, create_refactored_pipeline
)


@dataclass
class UnifiedPipelineCommandConfig:
    """Configuration for unified pipeline commands."""
    
    # Command type
    command_type: str  # 'analyst', 'tactician', 'analyst-short', 'tactician-long'
    
    # Symbol and timeframe
    symbol: str = "ETHUSDT"
    timeframe: str = "15m"
    
    # Pipeline intensity (from ares_launcher execution mode)
    intensity: str = "blank"  # Default to 25% intensity
    
    # Direction settings (from ares_launcher --direction)
    direction: str = "longs"  # 'longs', 'shorts', 'both' (ares_launcher format)
    
    # Lookback settings (from ares_launcher mode config)
    lookback_days: Optional[int] = None  # From ares_launcher mode config
    
    # Date range settings (from ares_launcher)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Exchange settings (from ares_launcher)
    exchange: str = "binance"
    
    # Output settings
    output_dir: Optional[str] = None
    save_results: bool = True
    
    # Logging
    log_level: str = "INFO"
    
    # Custom overrides
    custom_overrides: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.custom_overrides is None:
            self.custom_overrides = {}
        
        # Set direction based on command type (convert to ares_launcher format)
        if 'short' in self.command_type:
            self.direction = 'shorts'
        elif 'long' in self.command_type:
            self.direction = 'longs'
        else:
            self.direction = 'longs'  # Default for analyst/tactician


class UnifiedPipelineCommandHandler:
    """Handler for unified pipeline commands from ares_launcher."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the command handler.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def handle_analyst_command(self, 
                              symbol: str = "ETHUSDT",
                              timeframe: str = "15m",
                              direction: str = "longs",
                              intensity: str = "blank",
                              lookback_days: Optional[int] = None,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              exchange: str = "binance",
                              custom_overrides: Optional[Dict[str, Any]] = None,
                              **kwargs) -> RefactoredUnifiedPipeline:
        """Handle --unified-pipeline-analyst command.
        
        Args:
            symbol: Trading symbol (default: ETHUSDT)
            timeframe: Data timeframe (default: 15m)
            direction: Direction type (default: longs)
            intensity: Pipeline intensity (default: blank)
            lookback_days: Lookback period in days (optional)
            start_date: Start date for data (optional)
            end_date: End date for data (optional)
            exchange: Exchange name (default: binance)
            custom_overrides: Custom configuration overrides
            **kwargs: Additional arguments
            
        Returns:
            Configured RefactoredUnifiedPipeline for analyst mode
        """
        config = UnifiedPipelineCommandConfig(
            command_type="analyst",
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            intensity=intensity,
            lookback_days=lookback_days,
            start_date=start_date,
            end_date=end_date,
            exchange=exchange,
            custom_overrides=custom_overrides or {}
        )
        
        # Set analyst-specific overrides
        analyst_overrides = {
            'labeling_system': 'tactician_analyst',
            'labeling_type': 'analyst',
            'enable_labeling_optimization': True,
            'labeling_quality_threshold': 0.7,
            'symbol': config.symbol,
            'timeframe': config.timeframe,
            'direction': config.direction,
            'exchange': config.exchange
        }
        
        # Add lookback and date settings if provided
        if config.lookback_days is not None:
            analyst_overrides['lookback_days'] = config.lookback_days
        if config.start_date is not None:
            analyst_overrides['start_date'] = config.start_date
        if config.end_date is not None:
            analyst_overrides['end_date'] = config.end_date
        
        # Merge with custom overrides
        final_overrides = {**analyst_overrides, **config.custom_overrides}
        
        # Create pipeline
        pipeline = create_refactored_pipeline(
            intensity=config.intensity,
            custom_overrides=final_overrides,
            logger=self.logger
        )
        
        self.logger.info(f"Created analyst pipeline for {symbol} with {intensity} intensity")
        return pipeline
    
    def handle_tactician_command(self, 
                                symbol: str = "ETHUSDT",
                                timeframe: str = "15m",
                                direction: str = "longs",
                                intensity: str = "blank",
                                lookback_days: Optional[int] = None,
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None,
                                exchange: str = "binance",
                                custom_overrides: Optional[Dict[str, Any]] = None,
                                **kwargs) -> RefactoredUnifiedPipeline:
        """Handle --unified-pipeline-tactician command.
        
        Args:
            symbol: Trading symbol (default: ETHUSDT)
            timeframe: Data timeframe (default: 15m)
            direction: Direction type (default: longs)
            intensity: Pipeline intensity (default: blank)
            lookback_days: Lookback period in days (optional)
            start_date: Start date for data (optional)
            end_date: End date for data (optional)
            exchange: Exchange name (default: binance)
            custom_overrides: Custom configuration overrides
            **kwargs: Additional arguments
            
        Returns:
            Configured RefactoredUnifiedPipeline for tactician mode
        """
        config = UnifiedPipelineCommandConfig(
            command_type="tactician",
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            intensity=intensity,
            lookback_days=lookback_days,
            start_date=start_date,
            end_date=end_date,
            exchange=exchange,
            custom_overrides=custom_overrides or {}
        )
        
        # Set tactician-specific overrides
        tactician_overrides = {
            'labeling_system': 'tactician_analyst',
            'labeling_type': 'tactician',
            'enable_labeling_optimization': True,
            'labeling_quality_threshold': 0.7,
            'symbol': config.symbol,
            'timeframe': config.timeframe,
            'direction': config.direction,
            'exchange': config.exchange
        }
        
        # Add lookback and date settings if provided
        if config.lookback_days is not None:
            tactician_overrides['lookback_days'] = config.lookback_days
        if config.start_date is not None:
            tactician_overrides['start_date'] = config.start_date
        if config.end_date is not None:
            tactician_overrides['end_date'] = config.end_date
        
        # Merge with custom overrides
        final_overrides = {**tactician_overrides, **config.custom_overrides}
        
        # Create pipeline
        pipeline = create_refactored_pipeline(
            intensity=config.intensity,
            custom_overrides=final_overrides,
            logger=self.logger
        )
        
        self.logger.info(f"Created tactician pipeline for {symbol} with {intensity} intensity")
        return pipeline
    
    def handle_analyst_short_command(self, 
                                   symbol: str = "ETHUSDT",
                                   intensity: str = "blank",
                                   custom_overrides: Optional[Dict[str, Any]] = None,
                                   **kwargs) -> RefactoredUnifiedPipeline:
        """Handle --unified-pipeline-analyst-short command.
        
        Args:
            symbol: Trading symbol (default: ETHUSDT)
            intensity: Pipeline intensity (default: blank)
            custom_overrides: Custom configuration overrides
            **kwargs: Additional arguments
            
        Returns:
            Configured RefactoredUnifiedPipeline for analyst short mode
        """
        config = UnifiedPipelineCommandConfig(
            command_type="analyst-short",
            symbol=symbol,
            intensity=intensity,
            custom_overrides=custom_overrides or {}
        )
        
        # Set analyst-short-specific overrides
        analyst_short_overrides = {
            'labeling_system': 'tactician_analyst',
            'labeling_type': 'analyst',
            'enable_labeling_optimization': True,
            'labeling_quality_threshold': 0.7,
            'optimization_direction': 'short'
        }
        
        # Merge with custom overrides
        final_overrides = {**analyst_short_overrides, **config.custom_overrides}
        
        # Create pipeline
        pipeline = create_refactored_pipeline(
            intensity=config.intensity,
            custom_overrides=final_overrides,
            logger=self.logger
        )
        
        self.logger.info(f"Created analyst-short pipeline for {symbol} with {intensity} intensity")
        return pipeline
    
    def handle_tactician_long_command(self, 
                                    symbol: str = "ETHUSDT",
                                    intensity: str = "blank",
                                    custom_overrides: Optional[Dict[str, Any]] = None,
                                    **kwargs) -> RefactoredUnifiedPipeline:
        """Handle --unified-pipeline-tactician-long command.
        
        Args:
            symbol: Trading symbol (default: ETHUSDT)
            intensity: Pipeline intensity (default: blank)
            custom_overrides: Custom configuration overrides
            **kwargs: Additional arguments
            
        Returns:
            Configured RefactoredUnifiedPipeline for tactician long mode
        """
        config = UnifiedPipelineCommandConfig(
            command_type="tactician-long",
            symbol=symbol,
            intensity=intensity,
            custom_overrides=custom_overrides or {}
        )
        
        # Set tactician-long-specific overrides
        tactician_long_overrides = {
            'labeling_system': 'tactician_analyst',
            'labeling_type': 'tactician',
            'enable_labeling_optimization': True,
            'labeling_quality_threshold': 0.7,
            'optimization_direction': 'long'
        }
        
        # Merge with custom overrides
        final_overrides = {**tactician_long_overrides, **config.custom_overrides}
        
        # Create pipeline
        pipeline = create_refactored_pipeline(
            intensity=config.intensity,
            custom_overrides=final_overrides,
            logger=self.logger
        )
        
        self.logger.info(f"Created tactician-long pipeline for {symbol} with {intensity} intensity")
        return pipeline
    
    def get_command_info(self, command_type: str) -> Dict[str, Any]:
        """Get information about a specific command type.
        
        Args:
            command_type: Type of command ('analyst', 'tactician', etc.)
            
        Returns:
            Dictionary with command information
        """
        command_info = {
            'analyst': {
                'description': 'Analyst mode - "Should we trade?" based on expected PnL > fees + slippage',
                'labeling_type': 'analyst',
                'direction': 'long',
                'use_case': 'Long-term position analysis'
            },
            'tactician': {
                'description': 'Tactician mode - Direction/magnitude based on max favorable/adverse excursion',
                'labeling_type': 'tactician',
                'direction': 'long',
                'use_case': 'Short-term tactical analysis'
            },
            'analyst-short': {
                'description': 'Analyst mode for short positions',
                'labeling_type': 'analyst',
                'direction': 'short',
                'use_case': 'Short position analysis'
            },
            'tactician-long': {
                'description': 'Tactician mode for long positions',
                'labeling_type': 'tactician',
                'direction': 'long',
                'use_case': 'Long tactical analysis'
            }
        }
        
        return command_info.get(command_type, {})
    
    def list_available_commands(self) -> Dict[str, Dict[str, Any]]:
        """List all available unified pipeline commands.
        
        Returns:
            Dictionary of available commands with their information
        """
        commands = {}
        for command_type in ['analyst', 'tactician', 'analyst-short', 'tactician-long']:
            commands[command_type] = self.get_command_info(command_type)
        return commands


def create_unified_pipeline_command_handler(logger: Optional[logging.Logger] = None) -> UnifiedPipelineCommandHandler:
    """Create a unified pipeline command handler.
    
    Args:
        logger: Optional logger instance
        
    Returns:
        UnifiedPipelineCommandHandler instance
    """
    return UnifiedPipelineCommandHandler(logger)


# Convenience functions for direct command handling
def handle_unified_pipeline_analyst(symbol: str = "ETHUSDT", 
                                  timeframe: str = "15m",
                                  direction: str = "longs",
                                  intensity: str = "blank",
                                  lookback_days: Optional[int] = None,
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None,
                                  exchange: str = "binance",
                                  **kwargs) -> RefactoredUnifiedPipeline:
    """Handle unified pipeline analyst command.
    
    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        direction: Direction type
        intensity: Pipeline intensity
        lookback_days: Lookback period in days
        start_date: Start date for data
        end_date: End date for data
        exchange: Exchange name
        **kwargs: Additional arguments
        
    Returns:
        Configured pipeline for analyst mode
    """
    handler = create_unified_pipeline_command_handler()
    return handler.handle_analyst_command(
        symbol, timeframe, direction, intensity, 
        lookback_days, start_date, end_date, exchange, **kwargs
    )


def handle_unified_pipeline_tactician(symbol: str = "ETHUSDT", 
                                     timeframe: str = "15m",
                                     direction: str = "longs",
                                     intensity: str = "blank",
                                     lookback_days: Optional[int] = None,
                                     start_date: Optional[str] = None,
                                     end_date: Optional[str] = None,
                                     exchange: str = "binance",
                                     **kwargs) -> RefactoredUnifiedPipeline:
    """Handle unified pipeline tactician command.
    
    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        direction: Direction type
        intensity: Pipeline intensity
        lookback_days: Lookback period in days
        start_date: Start date for data
        end_date: End date for data
        exchange: Exchange name
        **kwargs: Additional arguments
        
    Returns:
        Configured pipeline for tactician mode
    """
    handler = create_unified_pipeline_command_handler()
    return handler.handle_tactician_command(
        symbol, timeframe, direction, intensity, 
        lookback_days, start_date, end_date, exchange, **kwargs
    )


def handle_unified_pipeline_analyst_short(symbol: str = "ETHUSDT", 
                                        intensity: str = "blank",
                                        **kwargs) -> RefactoredUnifiedPipeline:
    """Handle unified pipeline analyst short command.
    
    Args:
        symbol: Trading symbol
        intensity: Pipeline intensity
        **kwargs: Additional arguments
        
    Returns:
        Configured pipeline for analyst short mode
    """
    handler = create_unified_pipeline_command_handler()
    return handler.handle_analyst_short_command(symbol, intensity, **kwargs)


def handle_unified_pipeline_tactician_long(symbol: str = "ETHUSDT", 
                                         intensity: str = "blank",
                                         **kwargs) -> RefactoredUnifiedPipeline:
    """Handle unified pipeline tactician long command.
    
    Args:
        symbol: Trading symbol
        intensity: Pipeline intensity
        **kwargs: Additional arguments
        
    Returns:
        Configured pipeline for tactician long mode
    """
    handler = create_unified_pipeline_command_handler()
    return handler.handle_tactician_long_command(symbol, intensity, **kwargs)


# Export main classes and functions
__all__ = [
    'UnifiedPipelineCommandHandler',
    'UnifiedPipelineCommandConfig',
    'create_unified_pipeline_command_handler',
    'handle_unified_pipeline_analyst',
    'handle_unified_pipeline_tactician',
    'handle_unified_pipeline_analyst_short',
    'handle_unified_pipeline_tactician_long'
]