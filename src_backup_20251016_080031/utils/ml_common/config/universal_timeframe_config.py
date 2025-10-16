"""
Universal Timeframe Configuration for ML Common

Centralized timeframe configuration system that can be used across all ML models
to ensure consistency and prevent timeframe-related issues.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from src.common.config.loader import save_to_file as _unified_save_to_file, load_from_file as _unified_load_from_file

logger = logging.getLogger(__name__)

@dataclass
class UniversalTimeframeConfig:
    """Universal timeframe configuration for all ML models."""
    
    # Primary timeframe for ML operations
    primary_timeframe: str = "15m"
    
    # Supported timeframes for validation
    supported_timeframes: List[str] = None
    
    # Cross-timeframe features configuration
    enable_cross_timeframe_features: bool = True
    cross_timeframe_list: List[str] = None
    
    # Timeframe validation settings
    strict_timeframe_validation: bool = True
    auto_validate_timeframes: bool = True
    
    # Model-specific timeframe overrides
    model_timeframe_overrides: Dict[str, str] = None
    
    # Configuration metadata
    config_version: str = "1.0.0"
    last_updated: str = None
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.supported_timeframes is None:
            self.supported_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        
        if self.cross_timeframe_list is None:
            self.cross_timeframe_list = ["5m", "30m", "1h"]
        
        if self.model_timeframe_overrides is None:
            self.model_timeframe_overrides = {}
        
        if self.last_updated is None:
            from datetime import datetime
            self.last_updated = datetime.now().isoformat()
    
    def validate_timeframe(self, timeframe: str, model_type: str = "unknown") -> bool:
        """
        Validate if timeframe is supported for a specific model.
        
        Args:
            timeframe: Timeframe to validate
            model_type: Type of model for validation
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not timeframe:
            logger.error("Empty timeframe provided")
            return False
        
        # Check if timeframe is in supported list
        if timeframe not in self.supported_timeframes:
            logger.error(f"Unsupported timeframe: {timeframe}. Supported: {self.supported_timeframes}")
            return False
        
        # Check for model-specific overrides
        if model_type in self.model_timeframe_overrides:
            expected_timeframe = self.model_timeframe_overrides[model_type]
            if timeframe != expected_timeframe:
                logger.warning(f"Model {model_type} expects timeframe {expected_timeframe}, got {timeframe}")
                if self.strict_timeframe_validation:
                    return False
        
        return True
    
    def get_primary_timeframe(self) -> str:
        """Get the primary timeframe for ML operations."""
        return self.primary_timeframe
    
    def get_cross_timeframes(self) -> List[str]:
        """Get list of cross-timeframe features."""
        if not self.enable_cross_timeframe_features:
            return []
        return self.cross_timeframe_list
    
    def get_all_timeframes(self) -> List[str]:
        """Get all timeframes (primary + cross-timeframes)."""
        all_timeframes = [self.primary_timeframe]
        if self.enable_cross_timeframe_features:
            all_timeframes.extend(self.cross_timeframe_list)
        return list(set(all_timeframes))  # Remove duplicates
    
    def set_model_timeframe(self, model_type: str, timeframe: str) -> bool:
        """
        Set timeframe for a specific model type.
        
        Args:
            model_type: Type of model
            timeframe: Timeframe to set
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.validate_timeframe(timeframe, model_type):
            return False
        
        self.model_timeframe_overrides[model_type] = timeframe
        logger.info(f"Set timeframe for {model_type}: {timeframe}")
        return True
    
    def get_model_timeframe(self, model_type: str) -> str:
        """
        Get timeframe for a specific model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            str: Timeframe for the model
        """
        if model_type in self.model_timeframe_overrides:
            return self.model_timeframe_overrides[model_type]
        return self.primary_timeframe
    
    def is_cross_timeframe_enabled(self) -> bool:
        """Check if cross-timeframe features are enabled."""
        return self.enable_cross_timeframe_features
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'primary_timeframe': self.primary_timeframe,
            'supported_timeframes': self.supported_timeframes,
            'enable_cross_timeframe_features': self.enable_cross_timeframe_features,
            'cross_timeframe_list': self.cross_timeframe_list,
            'strict_timeframe_validation': self.strict_timeframe_validation,
            'auto_validate_timeframes': self.auto_validate_timeframes,
            'model_timeframe_overrides': self.model_timeframe_overrides,
            'config_version': self.config_version,
            'last_updated': self.last_updated
        }
    
    def save_config(self, filepath: str) -> bool:
        """
        Save configuration to file.
        
        Args:
            filepath: Path to save configuration
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            _unified_save_to_file(self, filepath)
            logger.info(f"Configuration saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    @classmethod
    def load_config(cls, filepath: str) -> 'UniversalTimeframeConfig':
        """
        Load configuration from file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            UniversalTimeframeConfig: Loaded configuration
        """
        try:
            config = _unified_load_from_file(filepath, cls)
            logger.info(f"Configuration loaded from {filepath}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return cls()  # Return default configuration

class UniversalTimeframeManager:
    """Universal timeframe manager for all ML models."""
    
    def __init__(self, config: Optional[UniversalTimeframeConfig] = None):
        """
        Initialize universal timeframe manager.
        
        Args:
            config: Timeframe configuration
        """
        self.config = config or UniversalTimeframeConfig()
        self.validation_history = []
    
    def validate_timeframe_consistency(self, 
                                     timeframe: str, 
                                     model_type: str = "unknown",
                                     component_name: str = "unknown") -> bool:
        """
        Validate timeframe consistency across components.
        
        Args:
            timeframe: Timeframe to validate
            model_type: Type of model
            component_name: Name of the component for logging
            
        Returns:
            bool: True if consistent, False otherwise
        """
        if not self.config.auto_validate_timeframes:
            return True
        
        is_valid = self.config.validate_timeframe(timeframe, model_type)
        
        # Track validation history
        self.validation_history.append({
            'timestamp': self.config.last_updated,
            'timeframe': timeframe,
            'model_type': model_type,
            'component_name': component_name,
            'is_valid': is_valid
        })
        
        if not is_valid:
            logger.error(f"Timeframe validation failed for {component_name}: {timeframe}")
            return False
        
        if self.config.strict_timeframe_validation and timeframe != self.config.primary_timeframe:
            logger.warning(f"Timeframe mismatch in {component_name}: {timeframe} != {self.config.primary_timeframe}")
            return False
        
        return True
    
    def get_timeframe_for_model(self, model_type: str) -> str:
        """
        Get appropriate timeframe for a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            str: Appropriate timeframe
        """
        return self.config.get_model_timeframe(model_type)
    
    def get_cross_timeframes_for_model(self, model_type: str) -> List[str]:
        """
        Get cross-timeframes for a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            List[str]: Cross-timeframes for the model
        """
        if not self.config.enable_cross_timeframe_features:
            return []
        
        # Filter cross-timeframes based on model type
        base_timeframe = self.get_timeframe_for_model(model_type)
        cross_timeframes = self.config.get_cross_timeframes()
        
        # Ensure cross-timeframes are different from base timeframe
        return [tf for tf in cross_timeframes if tf != base_timeframe]
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of timeframe validations."""
        if not self.validation_history:
            return {'message': 'No validations performed'}
        
        total_validations = len(self.validation_history)
        successful_validations = sum(1 for v in self.validation_history if v['is_valid'])
        success_rate = successful_validations / total_validations
        
        # Group by model type
        model_type_counts = {}
        for validation in self.validation_history:
            model_type = validation['model_type']
            model_type_counts[model_type] = model_type_counts.get(model_type, 0) + 1
        
        return {
            'total_validations': total_validations,
            'successful_validations': successful_validations,
            'success_rate': success_rate,
            'model_type_distribution': model_type_counts,
            'primary_timeframe': self.config.primary_timeframe,
            'supported_timeframes': self.config.supported_timeframes
        }

# Global instances for easy access
DEFAULT_TIMEFRAME_CONFIG = UniversalTimeframeConfig()
DEFAULT_TIMEFRAME_MANAGER = UniversalTimeframeManager(DEFAULT_TIMEFRAME_CONFIG)

def get_timeframe_config() -> UniversalTimeframeConfig:
    """Get the default timeframe configuration."""
    return DEFAULT_TIMEFRAME_CONFIG

def get_timeframe_manager() -> UniversalTimeframeManager:
    """Get the default timeframe manager."""
    return DEFAULT_TIMEFRAME_MANAGER

def set_timeframe_config(config: UniversalTimeframeConfig) -> None:
    """
    Set the global timeframe configuration.
    
    Args:
        config: New timeframe configuration
    """
    global DEFAULT_TIMEFRAME_CONFIG, DEFAULT_TIMEFRAME_MANAGER
    DEFAULT_TIMEFRAME_CONFIG = config
    DEFAULT_TIMEFRAME_MANAGER = UniversalTimeframeManager(config)
    logger.info(f"Timeframe configuration updated: primary={config.primary_timeframe}")

def validate_timeframe_consistency(timeframe: str, 
                                 model_type: str = "unknown",
                                 component_name: str = "unknown") -> bool:
    """
    Validate timeframe consistency across components.
    
    Args:
        timeframe: Timeframe to validate
        model_type: Type of model
        component_name: Name of the component for logging
        
    Returns:
        bool: True if consistent, False otherwise
    """
    return DEFAULT_TIMEFRAME_MANAGER.validate_timeframe_consistency(
        timeframe, model_type, component_name
    )

def get_primary_timeframe() -> str:
    """
    Get the primary timeframe for ML operations.
    
    Returns:
        str: Primary timeframe
    """
    return DEFAULT_TIMEFRAME_CONFIG.get_primary_timeframe()

def get_cross_timeframes() -> List[str]:
    """
    Get cross-timeframe features list.
    
    Returns:
        List[str]: List of cross-timeframes
    """
    return DEFAULT_TIMEFRAME_CONFIG.get_cross_timeframes()

def is_cross_timeframe_enabled() -> bool:
    """
    Check if cross-timeframe features are enabled.
    
    Returns:
        bool: True if enabled, False otherwise
    """
    return DEFAULT_TIMEFRAME_CONFIG.is_cross_timeframe_enabled()

# Convenience functions for backward compatibility
def get_timeframe() -> str:
    """Get primary timeframe (backward compatibility)."""
    return get_primary_timeframe()

def validate_timeframe(timeframe: str) -> bool:
    """Validate timeframe (backward compatibility)."""
    return DEFAULT_TIMEFRAME_CONFIG.validate_timeframe(timeframe)