"""
Modular Feature Selection System

This package provides a comprehensive, modular feature selection system
with hardware optimizations, configuration management, and validation.

Key Components:
- Core: Pipeline, selector, and optimizer modules
- Hardware: Memory management and VectorBT optimizations
- Config: Configuration loading and model profiles
- Validation: Data and result validation utilities
"""

from .core import (
    FeatureSelector,
    FeatureSelectionOptimizer,
    FeatureSelectionConfig,
    FeatureSelectionResult
)

from .core.multi_stage_pipeline import (
    MultiStageFeatureSelectionPipeline,
    run_multi_stage_feature_selection
)

from .hardware import (
    MemoryManager,
    VectorBTManager,
    PerformanceMonitor
)

from .config import (
    ConfigLoader,
    ModelProfileManager,
    ConfigValidator
)

from .validation import (
    DataValidator,
    ResultValidator,
    SchemaValidator
)

# Main pipeline function
def run_final_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    symbol: str = "BTCUSDT",
    exchange: str = "binance", 
    timeframe: str = "15m",
    config: Optional[Dict[str, Any]] = None
) -> FeatureSelectionResult:
    """
    Run final feature selection using the modular system.
    
    Args:
        X: Feature matrix
        y: Target variable
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        config: Optional configuration
        
    Returns:
        FeatureSelectionResult with selected features
    """
    from .core.config import FeatureSelectionConfig
    
    # Create feature selection config
    fs_config = FeatureSelectionConfig()
    if config:
        for key, value in config.items():
            if hasattr(fs_config, key):
                setattr(fs_config, key, value)
    
    # Use the new pipeline
    return run_multi_stage_feature_selection(X, y, symbol, exchange, timeframe, fs_config)

__all__ = [
    # Core modules
    'FeatureSelector', 
    'FeatureSelectionOptimizer',
    'FeatureSelectionConfig',
    'FeatureSelectionResult',
    
    # Multi-stage pipeline
    'MultiStageFeatureSelectionPipeline',
    'run_multi_stage_feature_selection',
    
    # Hardware modules
    'MemoryManager',
    'VectorBTManager',
    'PerformanceMonitor',
    
    # Config modules
    'ConfigLoader',
    'ModelProfileManager',
    'ConfigValidator',
    
    # Validation modules
    'DataValidator',
    'ResultValidator',
    'SchemaValidator',
    
    # Main function
    'run_final_feature_selection'
]