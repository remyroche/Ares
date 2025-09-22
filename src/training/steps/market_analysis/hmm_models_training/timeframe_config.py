"""
Timeframe Configuration - Single Source of Truth

Centralized timeframe configuration for HMM discovery, clustering, ML training, and ensemble training.
Ensures consistency across all HMM pipeline components.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TimeframeConfig:
    """Centralized timeframe configuration for HMM pipeline."""
    
    # Primary timeframe for HMM operations
    primary_timeframe: str = "15m"
    
    # Supported timeframes for validation
    supported_timeframes: List[str] = None
    
    # Cross-timeframe features configuration
    enable_cross_timeframe_features: bool = True
    cross_timeframe_list: List[str] = None
    
    # Timeframe validation settings
    strict_timeframe_validation: bool = True
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.supported_timeframes is None:
            self.supported_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        
        if self.cross_timeframe_list is None:
            self.cross_timeframe_list = ["5m", "30m", "1h"]
    
    def validate_timeframe(self, timeframe: str) -> bool:
        """
        Validate if timeframe is supported.
        
        Args:
            timeframe: Timeframe to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not timeframe:
            logger.error("Empty timeframe provided")
            return False
            
        if timeframe not in self.supported_timeframes:
            logger.error(f"Unsupported timeframe: {timeframe}. Supported: {self.supported_timeframes}")
            return False
            
        return True
    
    def get_primary_timeframe(self) -> str:
        """Get the primary timeframe for HMM operations."""
        return self.primary_timeframe
    
    def get_cross_timeframes(self) -> List[str]:
        """Get list of cross-timeframe features."""
        if not self.enable_cross_timeframe_features:
            return []
        return self.cross_timeframe_list
    
    def is_cross_timeframe_enabled(self) -> bool:
        """Check if cross-timeframe features are enabled."""
        return self.enable_cross_timeframe_features
    
    def get_all_timeframes(self) -> List[str]:
        """Get all timeframes (primary + cross-timeframes)."""
        all_timeframes = [self.primary_timeframe]
        if self.enable_cross_timeframe_features:
            all_timeframes.extend(self.cross_timeframe_list)
        return list(set(all_timeframes))  # Remove duplicates


# Global timeframe configuration instance
DEFAULT_TIMEFRAME_CONFIG = TimeframeConfig()

def get_timeframe_config() -> TimeframeConfig:
    """
    Get the global timeframe configuration.
    
    Returns:
        TimeframeConfig: Global timeframe configuration instance
    """
    return DEFAULT_TIMEFRAME_CONFIG

def set_timeframe_config(config: TimeframeConfig) -> None:
    """
    Set the global timeframe configuration.
    
    Args:
        config: New timeframe configuration
    """
    global DEFAULT_TIMEFRAME_CONFIG
    DEFAULT_TIMEFRAME_CONFIG = config
    logger.info(f"Timeframe configuration updated: primary={config.primary_timeframe}")

def validate_timeframe_consistency(timeframe: str, component_name: str) -> bool:
    """
    Validate timeframe consistency across components.
    
    Args:
        timeframe: Timeframe to validate
        component_name: Name of the component for logging
        
    Returns:
        bool: True if consistent, False otherwise
    """
    config = get_timeframe_config()
    
    if not config.validate_timeframe(timeframe):
        logger.error(f"Timeframe validation failed for {component_name}: {timeframe}")
        return False
    
    if config.strict_timeframe_validation and timeframe != config.primary_timeframe:
        logger.warning(f"Timeframe mismatch in {component_name}: {timeframe} != {config.primary_timeframe}")
        return False
    
    return True

def get_primary_timeframe() -> str:
    """
    Get the primary timeframe for HMM operations.
    
    Returns:
        str: Primary timeframe
    """
    return get_timeframe_config().get_primary_timeframe()

def get_cross_timeframes() -> List[str]:
    """
    Get cross-timeframe features list.
    
    Returns:
        List[str]: List of cross-timeframes
    """
    return get_timeframe_config().get_cross_timeframes()

def is_cross_timeframe_enabled() -> bool:
    """
    Check if cross-timeframe features are enabled.
    
    Returns:
        bool: True if enabled, False otherwise
    """
    return get_timeframe_config().is_cross_timeframe_enabled()

# Convenience functions for backward compatibility
def get_timeframe() -> str:
    """Get primary timeframe (backward compatibility)."""
    return get_primary_timeframe()

def validate_timeframe(timeframe: str) -> bool:
    """Validate timeframe (backward compatibility)."""
    return get_timeframe_config().validate_timeframe(timeframe)