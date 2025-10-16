from dataclasses import dataclass
from enum import Enum
from src.utils.tprint import tprint
from typing import Dict, Any, Optional

class OptimizationLevel(Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"

@dataclass
class AutoOptimizationConfig:
    """Configuration for automatic optimization."""
    # Core settings
    enable_auto_optimization: bool = True
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED

    # Memory optimization
    enable_memory_optimization: bool = True
    memory_threshold_mb: float = 100.0
    enable_data_compression: bool = True
    enable_chunked_processing: bool = True
    chunk_size: int = 10000

    # VectorBT optimization
    enable_vectorbt_optimization: bool = True
    vectorbt_threshold: int = 1000
    enable_gpu_acceleration: bool = False

    # Rolling operations optimization
    enable_rolling_optimization: bool = True
    enable_rolling_cache: bool = True
    rolling_cache_size: int = 100

    # Performance monitoring
    enable_performance_monitoring: bool = True
    enable_optimization_logging: bool = False

    # Strategy-specific settings
    conservative_settings: Dict[str, Any] = None
    balanced_settings: Dict[str, Any] = None
    aggressive_settings: Dict[str, Any] = None

    def __post_init__(self):
        tprint("🔧 Initializing AutoOptimizationConfig...")

        try:
            if self.conservative_settings is None:
                tprint("📝 Setting up conservative optimization settings...")
                self.conservative_settings = {
                    'enable_memory_optimization': True,
                    'enable_vectorbt_optimization': False,
                    'enable_rolling_optimization': False,
                    'memory_threshold_mb': 500.0
                }
                tprint("✅ Conservative settings configured")

            if self.balanced_settings is None:
                tprint("📝 Setting up balanced optimization settings...")
                self.balanced_settings = {
                    'enable_memory_optimization': True,
                    'enable_vectorbt_optimization': True,
                    'enable_rolling_optimization': True,
                    'memory_threshold_mb': 100.0,
                    'vectorbt_threshold': 1000
                }
                tprint("✅ Balanced settings configured")

            if self.aggressive_settings is None:
                tprint("📝 Setting up aggressive optimization settings...")
                self.aggressive_settings = {
                    'enable_memory_optimization': True,
                    'enable_vectorbt_optimization': True,
                    'enable_rolling_optimization': True,
                    'enable_data_compression': True,
                    'enable_chunked_processing': True,
                    'memory_threshold_mb': 50.0,
                    'vectorbt_threshold': 500
                }
                tprint("✅ Aggressive settings configured")

            tprint(f"🎯 AutoOptimizationConfig initialized with {self.optimization_level.value} strategy")

        except Exception as e:
            tprint(f"❌ Error initializing AutoOptimizationConfig: {e}")
            raise

    def get_settings_for_level(self) -> Dict[str, Any]:
        """Get settings for current optimization level."""
        try:
            tprint(f"🔍 Getting settings for {self.optimization_level.value} optimization level...")

            if self.optimization_level == OptimizationLevel.CONSERVATIVE:
                tprint("📊 Returning conservative settings")
                return self.conservative_settings
            elif self.optimization_level == OptimizationLevel.BALANCED:
                tprint("📊 Returning balanced settings")
                return self.balanced_settings
            elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
                tprint("📊 Returning aggressive settings")
                return self.aggressive_settings
            else:
                tprint("⚠️ Unknown optimization level, defaulting to balanced")
                return self.balanced_settings

        except Exception as e:
            tprint(f"❌ Error getting settings for level: {e}")
            tprint("🔄 Falling back to balanced settings")
            return self.balanced_settings

    def apply_level_settings(self):
        """Apply settings based on current optimization level."""
        try:
            tprint(f"🔧 Applying {self.optimization_level.value} level settings...")
            level_settings = self.get_settings_for_level()

            applied_count = 0
            for key, value in level_settings.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    applied_count += 1
                    tprint(f"   ✅ Applied {key} = {value}")
                else:
                    tprint(f"   ⚠️ Setting {key} not found in config, skipping")

            tprint(f"✅ Applied {applied_count} settings for {self.optimization_level.value} level")

        except Exception as e:
            tprint(f"❌ Error applying level settings: {e}")
            raise

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'enable_auto_optimization': self.enable_auto_optimization,
            'optimization_level': self.optimization_level.value,
            'enable_memory_optimization': self.enable_memory_optimization,
            'memory_threshold_mb': self.memory_threshold_mb,
            'enable_data_compression': self.enable_data_compression,
            'enable_chunked_processing': self.enable_chunked_processing,
            'chunk_size': self.chunk_size,
            'enable_vectorbt_optimization': self.enable_vectorbt_optimization,
            'vectorbt_threshold': self.vectorbt_threshold,
            'enable_gpu_acceleration': self.enable_gpu_acceleration,
            'enable_rolling_optimization': self.enable_rolling_optimization,
            'enable_rolling_cache': self.enable_rolling_cache,
            'rolling_cache_size': self.rolling_cache_size,
            'enable_performance_monitoring': self.enable_performance_monitoring,
            'enable_optimization_logging': self.enable_optimization_logging
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AutoOptimizationConfig':
        """Create configuration from dictionary."""
        # Extract optimization level
        if 'optimization_level' in config_dict:
            if isinstance(config_dict['optimization_level'], str):
                config_dict['optimization_level'] = OptimizationLevel(config_dict['optimization_level'])

        return cls(**config_dict)
