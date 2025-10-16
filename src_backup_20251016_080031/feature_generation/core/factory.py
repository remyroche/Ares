"""
Factory Functions for Feature Generation

This module provides factory functions for creating and accessing
feature generators and the feature bank.
"""

import logging
from typing import Optional, List, Union

from .feature_bank import FeatureBank, get_global_feature_bank
from .feature_generator import FeatureGenerator, FeatureCategory
from .feature_registry import FeatureRegistry

logger = logging.getLogger(__name__)

def get_feature_bank(config: Optional[dict] = None) -> FeatureBank:
    """
    Get a feature bank instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Feature bank instance
    """
    if config is None:
        return get_global_feature_bank()
    
    from .feature_bank import FeatureBankConfig
    bank_config = FeatureBankConfig(**config)
    return FeatureBank(bank_config)

def get_feature_generator(name: str, bank: Optional[FeatureBank] = None) -> Optional[FeatureGenerator]:
    """
    Get a feature generator by name.
    
    Args:
        name: Generator name
        bank: Optional feature bank (uses global if None)
        
    Returns:
        Feature generator or None if not found
    """
    if bank is None:
        bank = get_global_feature_bank()
    
    return bank.get_generator_by_name(name)

def register_feature_generator(generator: FeatureGenerator, bank: Optional[FeatureBank] = None) -> None:
    """
    Register a feature generator.
    
    Args:
        generator: Feature generator to register
        bank: Optional feature bank (uses global if None)
    """
    if bank is None:
        bank = get_global_feature_bank()
    
    bank.register_generator(generator)

def list_available_features(category: Optional[Union[str, FeatureCategory]] = None, 
                          bank: Optional[FeatureBank] = None) -> List[str]:
    """
    List available features.
    
    Args:
        category: Optional category filter
        bank: Optional feature bank (uses global if None)
        
    Returns:
        List of feature names
    """
    if bank is None:
        bank = get_global_feature_bank()
    
    if category is not None:
        if isinstance(category, str):
            try:
                category = FeatureCategory(category)
            except ValueError:
                logger.warning(f"Invalid category: {category}")
                return []
        
        return bank.list_features(category)
    
    return bank.list_features()

def list_available_categories(bank: Optional[FeatureBank] = None) -> List[FeatureCategory]:
    """
    List available categories.
    
    Args:
        bank: Optional feature bank (uses global if None)
        
    Returns:
        List of categories
    """
    if bank is None:
        bank = get_global_feature_bank()
    
    return bank.list_categories()

def create_feature_bank_with_defaults() -> FeatureBank:
    """
    Create a feature bank with default generators registered.
    
    Returns:
        Feature bank with default generators
    """
    bank = FeatureBank()
    
    # Register default generators from categories
    try:
        from ..categories import (
            ReturnsFeatureGenerator,
            MomentumFeatureGenerator,
            VolumeFeatureGenerator,
            VolatilityFeatureGenerator,
            TrendFeatureGenerator,
            OscillatorFeatureGenerator,
            SupportResistanceFeatureGenerator,
            CandlestickPatternFeatureGenerator
        )
        
        # Create and register default generators
        default_generators = [
            ReturnsFeatureGenerator.create_default(),
            MomentumFeatureGenerator.create_default(),
            VolumeFeatureGenerator.create_default(),
            VolatilityFeatureGenerator.create_default(),
            TrendFeatureGenerator.create_default(),
            OscillatorFeatureGenerator.create_default(),
            SupportResistanceFeatureGenerator.create_default(),
            CandlestickPatternFeatureGenerator.create_default()
        ]
        
        for generator in default_generators:
            if generator:
                bank.register_generator(generator)
        
        logger.info(f"Registered {len(default_generators)} default generators")
        
    except ImportError as e:
        logger.warning(f"Could not import default generators: {e}")
    
    return bank

def get_or_create_feature_bank(config: Optional[dict] = None) -> FeatureBank:
    """
    Get existing feature bank or create a new one with defaults.
    
    Args:
        config: Optional configuration
        
    Returns:
        Feature bank instance
    """
    if config is None:
        return get_global_feature_bank()
    
    return create_feature_bank_with_defaults()

def _initialize_default_bank() -> None:
    """
    Initialize the default feature bank with all available generators.
    This function is called during module initialization.
    """
    try:
        bank = create_feature_bank_with_defaults()
        from .feature_bank import set_global_feature_bank
        set_global_feature_bank(bank)
        return bank
    except Exception as e:
        logger.error(f"Failed to create feature bank: {e}")
        return None

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.indicators.basic import RSI, MACD, ATR, BBANDS, STOCH, OBV, MA
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    RSI = None
    MACD = None
    ATR = None
    BBANDS = None
    STOCH = None
    OBV = None
    MA = None
    import warnings
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:
    
    cp = None

def search_features(query: str, 
                   category: Optional[Union[str, FeatureCategory]] = None,
                   bank: Optional[FeatureBank] = None) -> List[str]:
    """
    Search for features by name or description.
    
    Args:
        query: Search query
        category: Optional category filter
        bank: Optional feature bank (uses global if None)
        
    Returns:
        List of matching feature names
    """
    if bank is None:
        bank = get_global_feature_bank()
    
    # Get all features
    all_features = bank.list_features(category)
    
    # Filter by query
    query_lower = query.lower()
    matching_features = [
        name for name in all_features
        if query_lower in name.lower()
    ]
    
    return matching_features

def get_feature_info(name: str, bank: Optional[FeatureBank] = None) -> Optional[dict]:
    """
    Get detailed information about a feature.
    
    Args:
        name: Feature name
        bank: Optional feature bank (uses global if None)
        
    Returns:
        Dictionary with feature information or None if not found
    """
    generator = get_feature_generator(name, bank)
    if generator is None:
        return None
    
    config = generator.config
    return {
        'name': config.name,
        'category': config.category.value,
        'description': config.description,
        'required_columns': config.required_columns,
        'optional_columns': config.optional_columns,
        'default_lookback': config.default_lookback,
        'min_lookback': config.min_lookback,
        'max_lookback': config.max_lookback,
        'parameters': config.parameters,
        'dependencies': config.dependencies,
        'matrix_optimized': config.matrix_optimized,
        'gpu_accelerated': config.gpu_accelerated,
        'supports_lookback_optimization': generator.supports_lookback_optimization(),
        'performance_stats': generator.get_performance_stats()
    }

def validate_feature_requirements(data_columns: List[str], 
                                feature_names: List[str],
                                bank: Optional[FeatureBank] = None) -> dict:
    """
    Validate that data has required columns for specified features.
    
    Args:
        data_columns: Available data columns
        feature_names: List of feature names to validate
        bank: Optional feature bank (uses global if None)
        
    Returns:
        Dictionary with validation results
    """
    if bank is None:
        bank = get_global_feature_bank()
    
    results = {
        'valid_features': [],
        'invalid_features': [],
        'missing_columns': {},
        'warnings': []
    }
    
    data_columns_set = set(data_columns)
    
    for feature_name in feature_names:
        generator = bank.get_generator_by_name(feature_name)
        if generator is None:
            results['invalid_features'].append(feature_name)
            results['warnings'].append(f"Feature not found: {feature_name}")
            continue
        
        config = generator.config
        missing_columns = set(config.required_columns) - data_columns_set
        
        if missing_columns:
            results['invalid_features'].append(feature_name)
            results['missing_columns'][feature_name] = list(missing_columns)
        else:
            results['valid_features'].append(feature_name)
    
    return results
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
