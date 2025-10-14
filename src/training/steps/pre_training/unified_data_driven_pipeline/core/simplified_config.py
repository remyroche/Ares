"""
Simplified Configuration Presets for Unified Data-Driven Pipeline

This module provides simplified configuration presets for common use cases:
- full: Complete pipeline with all features (100% intensity)
- blank: Reduced pipeline with 25% intensity (same pipeline but lower iterations, less features)
- light: Minimal pipeline with 10% intensity (essential features only)
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from .config import (
    UnifiedPipelineConfig, 
    create_default_config,
    PeriodOptimizationConfig,
    LookbackOptimizationConfig,
    InteractionGenerationConfig,
    FeatureSelectionConfig,
    VectorizationConfig,
    PerformanceConfig,
    MultiObjectiveConfig,
    TimeSeriesCVConfig,
    WalkForwardConfig,
    GuardrailConfig,
    OptimizationStrategy
)


class PipelineIntensity(Enum):
    """Pipeline intensity levels."""
    FULL = "full"      # 100% intensity - all features
    BLANK = "blank"    # 25% intensity - reduced but comprehensive
    LIGHT = "light"    # 10% intensity - essential features only


@dataclass
class SimplifiedConfig:
    """Simplified configuration with intensity-based presets."""
    
    intensity: PipelineIntensity = PipelineIntensity.FULL
    custom_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Apply intensity-based configuration after initialization."""
        self._apply_intensity_config()
        self._apply_custom_overrides()
    
    def _apply_intensity_config(self) -> None:
        """Apply configuration based on intensity level."""
        if self.intensity == PipelineIntensity.FULL:
            self._apply_full_config()
        elif self.intensity == PipelineIntensity.BLANK:
            self._apply_blank_config()
        elif self.intensity == PipelineIntensity.LIGHT:
            self._apply_light_config()
    
    def _apply_full_config(self) -> None:
        """Apply full intensity configuration (100%)."""
        self.config = create_default_config()
        # Full configuration uses all default settings
    
    def _apply_blank_config(self) -> None:
        """Apply blank intensity configuration (25% - reduced but comprehensive)."""
        self.config = create_default_config()
        
        # Reduce period optimization intensity
        self.config.period_optimization.max_period = 63  # 3 months instead of 1 year
        self.config.period_optimization.period_step = 2  # Larger steps
        self.config.period_optimization.max_computation_time = 300.0  # 5 minutes instead of 10
        
        # Reduce lookback optimization intensity
        self.config.lookback_optimization.max_lookback = 50  # Reduced from 100
        self.config.lookback_optimization.step_size = 10  # Larger steps
        self.config.lookback_optimization.bayesian_trials = 25  # Reduced from 50
        self.config.lookback_optimization.max_computation_time = 300.0  # 5 minutes
        
        # Reduce interaction generation intensity
        self.config.interaction_generation.max_interactions = 50  # Reduced from 100
        self.config.interaction_generation.batch_size = 500  # Smaller batches
        self.config.interaction_generation.htf_interaction_ratio = 0.2  # Reduced HTF interactions
        
        # Reduce feature selection intensity
        self.config.feature_selection.multi_objective.max_features = 25  # Reduced from 50
        self.config.feature_selection.multi_objective.min_features = 3  # Reduced from 5
        self.config.feature_selection.multi_objective.max_generations = 50  # Reduced from 100
        self.config.feature_selection.multi_objective.population_size = 25  # Reduced from 50
        self.config.feature_selection.cv_config.n_splits = 3  # Reduced from 5
        self.config.feature_selection.cv_config.test_size = 0.3  # Larger test set
        self.config.feature_selection.max_computation_time = 300.0  # 5 minutes
        
        # Reduce vectorization intensity
        self.config.vectorization.chunk_size = 500  # Smaller chunks
        self.config.vectorization.cache_size = 500  # Smaller cache
        
        # Reduce performance monitoring
        self.config.performance.enable_profiling = False  # Disable profiling
        self.config.performance.log_level = 'WARNING'  # Less verbose logging
    
    def _apply_light_config(self) -> None:
        """Apply light intensity configuration (10% - same pipeline structure, minimal parameters)."""
        self.config = create_default_config()
        
        # Light period optimization - same structure, reduced parameters
        self.config.period_optimization.max_period = 21  # 1 month instead of 1 year
        self.config.period_optimization.period_step = 3  # Larger steps
        self.config.period_optimization.max_computation_time = 120.0  # 2 minutes instead of 10
        self.config.period_optimization.enable_parallel = True  # Keep parallel processing
        
        # Light lookback optimization - same structure, reduced parameters
        self.config.lookback_optimization.max_lookback = 20  # Reduced from 100
        self.config.lookback_optimization.step_size = 5  # Larger steps
        self.config.lookback_optimization.bayesian_trials = 10  # Reduced from 50
        self.config.lookback_optimization.enable_bayesian_optimization = True  # Keep enabled
        self.config.lookback_optimization.max_computation_time = 120.0  # 2 minutes
        
        # Light interaction generation - same structure, reduced parameters
        self.config.interaction_generation.max_interactions = 20  # Reduced from 100
        self.config.interaction_generation.enable_htf_interactions = True  # Keep enabled
        self.config.interaction_generation.htf_interaction_ratio = 0.1  # Reduced HTF ratio
        self.config.interaction_generation.batch_size = 200  # Smaller batches
        
        # Light feature selection - same structure, reduced parameters
        self.config.feature_selection.multi_objective.max_features = 10  # Reduced from 50
        self.config.feature_selection.multi_objective.min_features = 2  # Reduced from 5
        self.config.feature_selection.multi_objective.max_generations = 20  # Reduced from 100
        self.config.feature_selection.multi_objective.population_size = 10  # Reduced from 50
        self.config.feature_selection.cv_config.n_splits = 2  # Reduced from 5
        self.config.feature_selection.cv_config.test_size = 0.4  # Larger test set
        self.config.feature_selection.max_computation_time = 120.0  # 2 minutes
        
        # Light vectorization - same structure, reduced parameters
        self.config.vectorization.chunk_size = 200  # Smaller chunks
        self.config.vectorization.cache_size = 200  # Smaller cache
        self.config.vectorization.enable_gpu = False  # Disable GPU for simplicity
        self.config.vectorization.enable_parallel = True  # Keep parallel processing
        
        # Light performance monitoring - same structure, reduced verbosity
        self.config.performance.enable_performance_tracking = True  # Keep enabled
        self.config.performance.enable_memory_monitoring = True  # Keep enabled
        self.config.performance.enable_profiling = False  # Disable profiling
        self.config.performance.log_level = 'WARNING'  # Less verbose logging
        
        # Keep all advanced features enabled but with reduced parameters
        self.config.enable_nested_cv = True
        self.config.enable_direction_optimization = True
        self.config.enable_bayesian_optimization = True
        self.config.enable_advanced_caching = True
        self.config.enable_regularization = True
    
    def _apply_custom_overrides(self) -> None:
        """Apply custom configuration overrides."""
        for key, value in self.custom_overrides.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                # Try to apply to sub-configurations
                self._apply_nested_override(key, value)
    
    def _apply_nested_override(self, key: str, value: Any) -> None:
        """Apply override to nested configuration objects."""
        # Split key by dots to navigate nested structure
        keys = key.split('.')
        current = self.config
        
        try:
            # Navigate to the parent object
            for k in keys[:-1]:
                current = getattr(current, k)
            
            # Set the final value
            setattr(current, keys[-1], value)
        except AttributeError:
            # If the path doesn't exist, skip this override
            pass
    
    def get_config(self) -> UnifiedPipelineConfig:
        """Get the configured UnifiedPipelineConfig."""
        return self.config
    
    def get_intensity_description(self) -> str:
        """Get a description of the current intensity level."""
        descriptions = {
            PipelineIntensity.FULL: "Full pipeline with all features enabled (100% intensity)",
            PipelineIntensity.BLANK: "Reduced pipeline with 25% intensity - same pipeline structure but lower iterations and fewer features",
            PipelineIntensity.LIGHT: "Light pipeline with 10% intensity - same pipeline structure but minimal parameters for fast processing"
        }
        return descriptions[self.intensity]


def create_full_config(custom_overrides: Optional[Dict[str, Any]] = None) -> UnifiedPipelineConfig:
    """Create a full intensity configuration (100%)."""
    simplified_config = SimplifiedConfig(
        intensity=PipelineIntensity.FULL,
        custom_overrides=custom_overrides or {}
    )
    return simplified_config.get_config()


def create_blank_config(custom_overrides: Optional[Dict[str, Any]] = None) -> UnifiedPipelineConfig:
    """Create a blank intensity configuration (25%)."""
    simplified_config = SimplifiedConfig(
        intensity=PipelineIntensity.BLANK,
        custom_overrides=custom_overrides or {}
    )
    return simplified_config.get_config()


def create_light_config(custom_overrides: Optional[Dict[str, Any]] = None) -> UnifiedPipelineConfig:
    """Create a light intensity configuration (10%)."""
    simplified_config = SimplifiedConfig(
        intensity=PipelineIntensity.LIGHT,
        custom_overrides=custom_overrides or {}
    )
    return simplified_config.get_config()


def create_config_by_intensity(intensity: str, custom_overrides: Optional[Dict[str, Any]] = None) -> UnifiedPipelineConfig:
    """Create configuration by intensity string."""
    intensity_map = {
        'full': PipelineIntensity.FULL,
        'blank': PipelineIntensity.BLANK,
        'light': PipelineIntensity.LIGHT
    }
    
    if intensity not in intensity_map:
        raise ValueError(f"Invalid intensity: {intensity}. Must be one of: {list(intensity_map.keys())}")
    
    simplified_config = SimplifiedConfig(
        intensity=intensity_map[intensity],
        custom_overrides=custom_overrides or {}
    )
    return simplified_config.get_config()


# Convenience functions for backward compatibility
def get_simplified_config(intensity: str = "full", **overrides) -> UnifiedPipelineConfig:
    """Get simplified configuration with optional overrides."""
    return create_config_by_intensity(intensity, overrides)


def list_available_intensities() -> Dict[str, str]:
    """List available intensity levels with descriptions."""
    return {
        'full': 'Full pipeline with all features enabled (100% intensity)',
        'blank': 'Reduced pipeline with 25% intensity - same pipeline structure but lower iterations and fewer features',
        'light': 'Light pipeline with 10% intensity - same pipeline structure but minimal parameters for fast processing'
    }