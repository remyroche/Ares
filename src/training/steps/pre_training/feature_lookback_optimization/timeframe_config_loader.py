"""
Timeframe-Aware Configuration Loader for Feature Lookback Optimization

This module provides intelligent configuration loading based on timeframe,
ensuring optimal parameters for 5m, 15m, and 60m timeframes.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error


@dataclass
class TimeframeConfig:
    """Configuration for a specific timeframe."""
    timeframe: str
    base_period_minutes: float
    min_lookback: int
    max_lookback: int
    lookback_step: int
    cv_folds: int
    max_optimization_time: int
    min_samples_for_ic: int
    min_ic_threshold: float
    max_workers: int
    chunk_size: int
    label_definition_type: str
    multi_target_scheme: Dict[str, Any]


class TimeframeConfigLoader:
    """Loads and manages timeframe-specific configurations."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the configuration loader."""
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent
        self.timeframe_configs = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all available timeframe configurations."""
        tprint("📋 Loading timeframe-specific configurations...")
        
        # Define available timeframes and their config files
        timeframe_files = {
            '5m': 'feature_lookback_optimization_optimized_config.yaml',
            '15m': 'feature_lookback_optimization_15m_config.yaml',
            '60m': 'feature_lookback_optimization_60m_config.yaml'
        }
        
        for timeframe, filename in timeframe_files.items():
            config_path = self.config_dir / filename
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config_data = yaml.safe_load(f)
                    
                    # Extract the main configuration
                    main_config = config_data.get('optimized_feature_lookback_optimization', {})
                    
                    # Create TimeframeConfig object
                    timeframe_config = TimeframeConfig(
                        timeframe=timeframe,
                        base_period_minutes=main_config.get('base_period_minutes', 15.0),
                        min_lookback=main_config.get('min_lookback', 5),
                        max_lookback=main_config.get('max_lookback', 50),
                        lookback_step=main_config.get('lookback_step', 1),
                        cv_folds=main_config.get('cv_folds', 5),
                        max_optimization_time=main_config.get('max_optimization_time', 300),
                        min_samples_for_ic=main_config.get('min_samples_for_ic', 100),
                        min_ic_threshold=main_config.get('min_ic_threshold', 0.01),
                        max_workers=main_config.get('max_workers', 4),
                        chunk_size=main_config.get('chunk_size', 1000),
                        label_definition_type=main_config.get('label_definition_type', 'tactician'),
                        multi_target_scheme=main_config.get('multi_target_scheme', {})
                    )
                    
                    self.timeframe_configs[timeframe] = {
                        'config': timeframe_config,
                        'full_config': config_data
                    }
                    
                    tprint_success(f"✅ Loaded {timeframe} configuration")
                    
                except Exception as e:
                    tprint_error(f"❌ Failed to load {timeframe} configuration: {e}")
            else:
                tprint_warning(f"⚠️ Configuration file not found: {config_path}")
    
    def get_config_for_timeframe(self, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get the full configuration for a specific timeframe."""
        # Normalize timeframe
        normalized_timeframe = self._normalize_timeframe(timeframe)
        
        if normalized_timeframe in self.timeframe_configs:
            return self.timeframe_configs[normalized_timeframe]['full_config']
        
        # Fallback to 15m if timeframe not found
        tprint_warning(f"⚠️ Timeframe {timeframe} not found, falling back to 15m")
        return self.timeframe_configs.get('15m', {}).get('full_config')
    
    def get_timeframe_config(self, timeframe: str) -> Optional[TimeframeConfig]:
        """Get the TimeframeConfig object for a specific timeframe."""
        normalized_timeframe = self._normalize_timeframe(timeframe)
        
        if normalized_timeframe in self.timeframe_configs:
            return self.timeframe_configs[normalized_timeframe]['config']
        
        # Fallback to 15m if timeframe not found
        tprint_warning(f"⚠️ Timeframe {timeframe} not found, falling back to 15m")
        return self.timeframe_configs.get('15m', {}).get('config')
    
    def _normalize_timeframe(self, timeframe: str) -> str:
        """Normalize timeframe string to standard format."""
        if not timeframe:
            return '15m'
        
        timeframe = timeframe.lower().strip()
        
        # Handle various formats
        if timeframe in ['5m', '5min', '5_min', '5_minute']:
            return '5m'
        elif timeframe in ['15m', '15min', '15_min', '15_minute']:
            return '15m'
        elif timeframe in ['60m', '60min', '60_min', '60_minute', '1h', '1hour', '1_hour']:
            return '60m'
        else:
            # Default to 15m for unknown timeframes
            tprint_warning(f"⚠️ Unknown timeframe format: {timeframe}, defaulting to 15m")
            return '15m'
    
    def get_available_timeframes(self) -> list:
        """Get list of available timeframes."""
        return list(self.timeframe_configs.keys())
    
    def validate_timeframe_config(self, timeframe: str) -> bool:
        """Validate that a timeframe configuration is complete."""
        normalized_timeframe = self._normalize_timeframe(timeframe)
        
        if normalized_timeframe not in self.timeframe_configs:
            return False
        
        config = self.timeframe_configs[normalized_timeframe]['config']
        
        # Check required fields
        required_fields = [
            'timeframe', 'base_period_minutes', 'min_lookback', 'max_lookback',
            'lookback_step', 'cv_folds', 'max_optimization_time'
        ]
        
        for field in required_fields:
            if not hasattr(config, field) or getattr(config, field) is None:
                tprint_error(f"❌ Missing required field {field} in {timeframe} config")
                return False
        
        # Validate ranges
        if config.min_lookback >= config.max_lookback:
            tprint_error(f"❌ Invalid lookback range in {timeframe} config: min >= max")
            return False
        
        if config.base_period_minutes <= 0:
            tprint_error(f"❌ Invalid base period in {timeframe} config: {config.base_period_minutes}")
            return False
        
        tprint_success(f"✅ {timeframe} configuration is valid")
        return True
    
    def get_optimized_parameters(self, timeframe: str) -> Dict[str, Any]:
        """Get optimized parameters for a specific timeframe."""
        config = self.get_timeframe_config(timeframe)
        if not config:
            return {}
        
        return {
            'timeframe': config.timeframe,
            'base_period_minutes': config.base_period_minutes,
            'min_lookback': config.min_lookback,
            'max_lookback': config.max_lookback,
            'lookback_step': config.lookback_step,
            'cv_folds': config.cv_folds,
            'max_optimization_time': config.max_optimization_time,
            'min_samples_for_ic': config.min_samples_for_ic,
            'min_ic_threshold': config.min_ic_threshold,
            'max_workers': config.max_workers,
            'chunk_size': config.chunk_size,
            'label_definition_type': config.label_definition_type,
            'multi_target_scheme': config.multi_target_scheme
        }
    
    def print_timeframe_summary(self, timeframe: str):
        """Print a summary of the configuration for a specific timeframe."""
        config = self.get_timeframe_config(timeframe)
        if not config:
            tprint_error(f"❌ No configuration found for timeframe: {timeframe}")
            return
        
        tprint_info(f"📊 Configuration Summary for {timeframe.upper()}:")
        tprint_info(f"   → Base period: {config.base_period_minutes} minutes")
        tprint_info(f"   → Lookback range: {config.min_lookback} - {config.max_lookback} periods")
        tprint_info(f"   → Lookback step: {config.lookback_step}")
        tprint_info(f"   → CV folds: {config.cv_folds}")
        tprint_info(f"   → Max optimization time: {config.max_optimization_time}s")
        tprint_info(f"   → Min samples for IC: {config.min_samples_for_ic}")
        tprint_info(f"   → Min IC threshold: {config.min_ic_threshold}")
        tprint_info(f"   → Max workers: {config.max_workers}")
        tprint_info(f"   → Chunk size: {config.chunk_size}")
        tprint_info(f"   → Label type: {config.label_definition_type}")
        
        if config.multi_target_scheme:
            tprint_info(f"   → Multi-target scheme: {config.multi_target_scheme}")


# Global instance for easy access
_timeframe_loader = None

def get_timeframe_config_loader() -> TimeframeConfigLoader:
    """Get the global timeframe configuration loader instance."""
    global _timeframe_loader
    if _timeframe_loader is None:
        _timeframe_loader = TimeframeConfigLoader()
    return _timeframe_loader

def load_config_for_timeframe(timeframe: str) -> Dict[str, Any]:
    """Convenience function to load configuration for a specific timeframe."""
    loader = get_timeframe_config_loader()
    return loader.get_config_for_timeframe(timeframe)

def get_optimized_parameters(timeframe: str) -> Dict[str, Any]:
    """Convenience function to get optimized parameters for a specific timeframe."""
    loader = get_timeframe_config_loader()
    return loader.get_optimized_parameters(timeframe)

def validate_timeframe(timeframe: str) -> bool:
    """Convenience function to validate a timeframe configuration."""
    loader = get_timeframe_config_loader()
    return loader.validate_timeframe_config(timeframe)