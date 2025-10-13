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
    MultiStageFeatureSelector,
    FeatureSelector,
    FeatureSelectionOptimizer,
    FeatureSelectionConfig,
    FeatureSelectionResult
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
    from .core.pipeline import MultiStageFeatureSelector
    from .core.config import FeatureSelectionConfig
    
    # Load configuration
    config_loader = ConfigLoader()
    config_result = config_loader.load_config('feature_selection', 'default')
    
    if not config_result.success:
        raise ValueError(f"Failed to load configuration: {config_result.error_message}")
    
    # Create feature selection config
    fs_config = FeatureSelectionConfig(**config_result.config)
    
    # Initialize selector
    selector = MultiStageFeatureSelector(fs_config)
    
    # Run feature selection
    result = selector.select_features(X, y, symbol, exchange, timeframe)
    
    return result

__all__ = [
    # Core modules
    'MultiStageFeatureSelector',
    'FeatureSelector', 
    'FeatureSelectionOptimizer',
    'FeatureSelectionConfig',
    'FeatureSelectionResult',
    
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