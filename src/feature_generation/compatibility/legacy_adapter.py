"""
Legacy Feature Adapter

This module provides backwards compatibility with existing feature generation code,
allowing seamless migration to the unified feature generation system.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Callable
import pandas as pd
import numpy as np

from ..core.feature_generator import FeatureGenerator, FeatureConfig, FeatureCategory, FeatureResult

logger = logging.getLogger(__name__)

class LegacyFeatureAdapter:
    """
    Adapter for legacy feature generation code.
    
    This class provides backwards compatibility with existing feature generation
    functions and classes, allowing them to work with the unified feature generation system.
    """
    
    def __init__(self):
        """Initialize the legacy feature adapter."""
        self.logger = logger.getChild('LegacyFeatureAdapter')
        self.legacy_functions: Dict[str, Callable] = {}
        self.legacy_classes: Dict[str, Any] = {}
        
        # Register known legacy functions
        self._register_legacy_functions()
        
        self.logger.info("✅ LegacyFeatureAdapter initialized")
    
    def _register_legacy_functions(self):
        """Register known legacy feature generation functions."""
        # Try to import and register legacy functions
        try:
            from ...feature_engineering.feature_generators import FeatureGenerators
            self.legacy_classes['FeatureGenerators'] = FeatureGenerators
            self.logger.info("✅ Registered legacy FeatureGenerators class")
        except ImportError:
            self.logger.warning("⚠️ Legacy FeatureGenerators not available")
        
        try:
            from ...analyst.feature_engineering_orchestrator import FeatureEngineeringOrchestrator
            self.legacy_classes['FeatureEngineeringOrchestrator'] = FeatureEngineeringOrchestrator
            self.logger.info("✅ Registered legacy FeatureEngineeringOrchestrator class")
        except ImportError:
            self.logger.warning("⚠️ Legacy FeatureEngineeringOrchestrator not available")
        
        # Register individual legacy functions
        self._register_legacy_technical_indicators()
        self._register_legacy_feature_functions()
    
    def _register_legacy_technical_indicators(self):
        """Register legacy technical indicator functions."""
        # These would be imported from the existing feature generation code
        legacy_indicators = {
            'sma': self._legacy_sma,
            'ema': self._legacy_ema,
            'rsi': self._legacy_rsi,
            'macd': self._legacy_macd,
            'bollinger_bands': self._legacy_bollinger_bands,
            'stochastic': self._legacy_stochastic,
            'atr': self._legacy_atr,
            'obv': self._legacy_obv,
            'vwap': self._legacy_vwap
        }
        
        self.legacy_functions.update(legacy_indicators)
        self.logger.info(f"✅ Registered {len(legacy_indicators)} legacy technical indicators")
    
    def _register_legacy_feature_functions(self):
        """Register legacy feature generation functions."""
        # These would be imported from existing feature engineering modules
        legacy_features = {
            'returns': self._legacy_returns,
            'volatility': self._legacy_volatility,
            'momentum': self._legacy_momentum,
            'volume_features': self._legacy_volume_features
        }
        
        self.legacy_functions.update(legacy_features)
        self.logger.info(f"✅ Registered {len(legacy_features)} legacy feature functions")
    
    def create_legacy_generator(self, 
                              function_name: str,
                              category: FeatureCategory,
                              description: str,
                              required_columns: List[str],
                              **kwargs) -> FeatureGenerator:
        """
        Create a feature generator from a legacy function.
        
        Args:
            function_name: Name of the legacy function
            category: Feature category
            description: Feature description
            required_columns: Required input columns
            **kwargs: Additional parameters
            
        Returns:
            Feature generator wrapping the legacy function
        """
        if function_name not in self.legacy_functions:
            raise ValueError(f"Legacy function not found: {function_name}")
        
        legacy_function = self.legacy_functions[function_name]
        
        # Create configuration
        config = FeatureConfig(
            name=function_name,
            category=category,
            description=description,
            required_columns=required_columns,
            parameters=kwargs
        )
        
        # Create wrapper generator
        return LegacyFunctionWrapper(config, legacy_function)
    
    def migrate_legacy_class(self, class_name: str, **kwargs) -> Any:
        """
        Migrate a legacy class to work with the unified system.
        
        Args:
            class_name: Name of the legacy class
            **kwargs: Initialization parameters
            
        Returns:
            Migrated class instance
        """
        if class_name not in self.legacy_classes:
            raise ValueError(f"Legacy class not found: {class_name}")
        
        legacy_class = self.legacy_classes[class_name]
        
        try:
            # Create instance with parameters
            instance = legacy_class(**kwargs)
            self.logger.info(f"✅ Migrated legacy class: {class_name}")
            return instance
        except Exception as e:
            self.logger.error(f"❌ Failed to migrate legacy class {class_name}: {e}")
            raise
    
    def list_available_legacy_functions(self) -> List[str]:
        """List all available legacy functions."""
        return list(self.legacy_functions.keys())
    
    def list_available_legacy_classes(self) -> List[str]:
        """List all available legacy classes."""
        return list(self.legacy_classes.keys())
    
    # Legacy function implementations (simplified versions)
    def _legacy_sma(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Legacy SMA implementation."""
        return data['close'].rolling(window=period).mean()
    
    def _legacy_ema(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Legacy EMA implementation."""
        return data['close'].ewm(span=period).mean()
    
    def _legacy_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Legacy RSI implementation."""
        close = data['close']
        delta = close.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gains = gains.ewm(alpha=1/period, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _legacy_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Legacy MACD implementation."""
        close = data['close']
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def _legacy_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """Legacy Bollinger Bands implementation."""
        close = data['close']
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        return upper_band
    
    def _legacy_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.Series:
        """Legacy Stochastic implementation."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        return k_percent
    
    def _legacy_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Legacy ATR implementation."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def _legacy_obv(self, data: pd.DataFrame) -> pd.Series:
        """Legacy OBV implementation."""
        close = data['close']
        volume = data['volume']
        
        price_change = close.diff()
        obv = np.where(price_change > 0, volume,
                      np.where(price_change < 0, -volume, 0))
        
        return pd.Series(obv, index=data.index).cumsum()
    
    def _legacy_vwap(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Legacy VWAP implementation."""
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        return vwap
    
    def _legacy_returns(self, data: pd.DataFrame, period: int = 1) -> pd.Series:
        """Legacy returns implementation."""
        return data['close'].pct_change(periods=period)
    
    def _legacy_volatility(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Legacy volatility implementation."""
        returns = data['close'].pct_change()
        return returns.rolling(window=period).std()
    
    def _legacy_momentum(self, data: pd.DataFrame, period: int = 10) -> pd.Series:
        """Legacy momentum implementation."""
        return data['close'] - data['close'].shift(period)
    
    def _legacy_volume_features(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Legacy volume features implementation."""
        return data['volume'].rolling(window=period).mean()

class LegacyFunctionWrapper(FeatureGenerator):
    """Wrapper for legacy functions to work with the unified system."""
    
    def __init__(self, config: FeatureConfig, legacy_function: Callable):
        """
        Initialize the legacy function wrapper.
        
        Args:
            config: Feature configuration
            legacy_function: Legacy function to wrap
        """
        super().__init__(config)
        self.legacy_function = legacy_function
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Generate feature using the legacy function.
        
        Args:
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            Generated feature
        """
        # Merge config parameters with kwargs
        all_params = {**self.config.parameters, **kwargs}
        
        # Call legacy function
        return self.legacy_function(data, **all_params)

def migrate_legacy_features(legacy_config: Dict[str, Any]) -> List[FeatureGenerator]:
    """
    Migrate legacy feature configuration to unified generators.
    
    Args:
        legacy_config: Legacy feature configuration
        
    Returns:
        List of migrated feature generators
    """
    adapter = get_legacy_adapter()
    generators = []
    
    for feature_name, feature_config in legacy_config.items():
        try:
            # Extract configuration
            category = FeatureCategory(feature_config.get('category', 'custom'))
            description = feature_config.get('description', f'Legacy {feature_name}')
            required_columns = feature_config.get('required_columns', ['close'])
            parameters = feature_config.get('parameters', {})
            
            # Create generator
            generator = adapter.create_legacy_generator(
                function_name=feature_name,
                category=category,
                description=description,
                required_columns=required_columns,
                **parameters
            )
            
            generators.append(generator)
            
        except Exception as e:
            logger.error(f"Failed to migrate legacy feature {feature_name}: {e}")
            continue
    
    return generators

# Global legacy adapter instance
_global_legacy_adapter: Optional[LegacyFeatureAdapter] = None

def get_legacy_adapter() -> LegacyFeatureAdapter:
    """
    Get the global legacy adapter instance.
    
    Returns:
        Legacy adapter instance
    """
    global _global_legacy_adapter
    if _global_legacy_adapter is None:
        _global_legacy_adapter = LegacyFeatureAdapter()
    return _global_legacy_adapter

def enable_legacy_compatibility(enable: bool = True) -> None:
    """
    Enable or disable legacy compatibility.
    
    Args:
        enable: Whether to enable legacy compatibility
    """
    global _global_legacy_adapter
    if enable:
        _global_legacy_adapter = LegacyFeatureAdapter()
    else:
        _global_legacy_adapter = None