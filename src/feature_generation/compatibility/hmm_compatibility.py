"""
HMM Compatibility Layer

This module provides compatibility for HMM processes that expect the old FeatureGenerators
interface, ensuring they can work with the new unified feature generation system.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union

# Conditional imports to handle missing dependencies gracefully
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from ..core.feature_generator import FeatureGenerator, FeatureConfig, FeatureCategory
    from ..core.feature_bank import FeatureBank
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    FeatureGenerator = None
    FeatureConfig = None
    FeatureCategory = None
    FeatureBank = None


# Try to use unified optimization system
try:
    from ..utils.optimization import get_feature_optimizer
    UNIFIED_OPTIMIZATION_AVAILABLE = True
except ImportError:
    UNIFIED_OPTIMIZATION_AVAILABLE = False

logger = logging.getLogger(__name__)

class HMMCompatibleFeatureGenerators:
    """
    Compatibility wrapper for HMM processes that expect the old FeatureGenerators interface.
    
    This class provides the same interface as the old FeatureGenerators class,
    but uses the new unified feature generation system under the hood.
    """
    
    def __init__(self):
        """Initialize the HMM-compatible feature generators."""
        self.logger = logger.getChild('HMMCompatibleFeatureGenerators')
        
        # Check dependencies
        if not PANDAS_AVAILABLE:
            self.logger.warning("⚠️ Pandas not available - limited functionality")
        if not NUMPY_AVAILABLE:
            self.logger.warning("⚠️ NumPy not available - limited functionality")
        if not CORE_AVAILABLE:
            self.logger.warning("⚠️ Core feature generation not available - using fallback")
        
        # Initialize the new feature bank
        if CORE_AVAILABLE:
            try:
                self.feature_bank = FeatureBank()
                self.logger.info("✅ Feature bank initialized successfully")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize feature bank: {e}")
                self.feature_bank = None
        else:
            self.feature_bank = None
        
        # Initialize legacy feature generators if available
        self._initialize_legacy_generators()
        
        self.logger.info("✅ HMM-compatible feature generators initialized")
    
    def _initialize_legacy_generators(self):
        """Initialize legacy feature generators for fallback."""
        # Skip legacy initialization to avoid circular imports
        # The circular import chain is:

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
        # hmm_compatibility -> utils.feature_generators -> feature_generators_compatibility -> feature_generation.__init__ -> hmm_compatibility
        self.legacy_generators = None
        self.legacy_available = False
        self.logger.info("ℹ️ Legacy feature generators disabled to avoid circular imports")
    
    def generate_features_for_hmm(self, data):
        """
        Generate focused feature set for HMM models training.
        
        This method creates a targeted feature set optimized for HMM models training, including:
        - Volume features (VWAP, volume patterns, volume-price relationships)
        - Volatility features (rolling volatility, GARCH-like features, volatility momentum)
        - Technical indicators (RSI, MACD, Bollinger Bands)
        - Momentum features (price momentum, volume momentum, momentum ratios)
        - Feature interactions (price-volume, volatility-momentum, RSI-MACD)
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with focused HMM-ready feature set
        """
        self.logger.info("🚀 Generating focused HMM-ready feature set...")
        
        # Check if pandas is available
        if not PANDAS_AVAILABLE:
            self.logger.error("❌ Pandas not available - cannot generate features")
            return None
        
        if data.empty:
            self.logger.warning("⚠️ Empty data provided to generate_features_for_hmm")
            return pd.DataFrame()
        
        # Try new feature generation system first
        if self.feature_bank is not None:
            try:
                return self._generate_features_with_new_system(data)
            except Exception as e:
                self.logger.warning(f"⚠️ New feature system failed: {e}, falling back to legacy")
        
        # Since legacy generators are disabled to avoid circular imports,
        # go directly to basic feature generation
        self.logger.info("🔄 Falling back to basic HMM feature generation")
        return self._generate_basic_hmm_features(data)
    
    def _generate_features_with_new_system(self, data):
        """Generate HMM features using the new unified feature generation system."""
        result_df = data.copy()
        start_time = time.time()
        
        # Convert any categorical columns to regular types
        for col in result_df.columns:
            if hasattr(result_df[col], 'cat'):
                try:
                    if result_df[col].dtype.name == 'category':
                        if len(result_df[col].cat.categories) > 0:
                            result_df[col] = result_df[col].astype(result_df[col].cat.categories.dtype)
                        else:
                            result_df[col] = result_df[col].astype('object')
                except Exception:
                    result_df[col] = result_df[col].astype('object')
        
        try:
            # Generate features by category using the new system
            feature_categories = [
                'volume', 'volatility', 'momentum', 'oscillator', 'trend'
            ]
            
            for category in feature_categories:
                try:
                    self.logger.info(f"📊 Adding {category} features...")
                    category_features = self.feature_bank.generate_features(
                        data=result_df,
                        categories=[category],
                        lookback_optimization=False  # Disable for HMM to get consistent features
                    )
                    
                    if not category_features.empty:
                        # Merge category features with result
                        for col in category_features.columns:
                            if col not in result_df.columns:
                                result_df[col] = category_features[col]
                        
                        self.logger.info(f"✅ Added {len(category_features.columns)} {category} features")
                    else:
                        self.logger.warning(f"⚠️ No {category} features generated")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to generate {category} features: {e}")
                    continue
            
            # Add specific HMM-optimized features
            self._add_hmm_specific_features(result_df, data)
            
            # Clean up the result
            result_df = self._clean_hmm_features(result_df)
            
            generation_time = time.time() - start_time
            self.logger.info(f"✅ HMM feature generation completed in {generation_time:.2f}s")
            self.logger.info(f"📊 Generated {len(result_df.columns)} total features")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"❌ New feature system failed: {e}")
            raise
    
    def _add_hmm_specific_features(self, result_df, original_data):
        """Add HMM-specific features that are optimized for regime detection."""
        try:
            # Volume features
            if 'volume' in original_data.columns:
                # Volume change and ratios
                result_df['volume_change'] = original_data['volume'].pct_change()
                result_df['volume_ma_ratio'] = original_data['volume'] / original_data['volume'].rolling(20).mean().replace(0, 1)
                
                # Volume patterns
                volume_ma_20 = original_data['volume'].rolling(20).mean()
                result_df['volume_spike'] = (original_data['volume'] > volume_ma_20 * 2).astype(int)
                result_df['volume_dry_up'] = (original_data['volume'] < volume_ma_20 * 0.5).astype(int)
                
                # Multiple timeframe volume features
                for window in [5, 10, 20, 50]:
                    result_df[f'volume_ma_{window}'] = original_data['volume'].rolling(window).mean()
                    result_df[f'volume_std_{window}'] = original_data['volume'].rolling(window).std()
                    volume_ma_safe = result_df[f'volume_ma_{window}'].replace(0, np.nan)
                    volume_ma_safe = volume_ma_safe.fillna(method='bfill').fillna(1.0)
                    result_df[f'volume_ratio_{window}'] = original_data['volume'] / volume_ma_safe
                    result_df[f'volume_ratio_{window}'] = result_df[f'volume_ratio_{window}'].clip(-100, 100)
            
            # Volatility features
            if 'close' in original_data.columns:
                # Rolling volatility
                returns = original_data['close'].pct_change()
                for window in [5, 10, 20, 50]:
                    result_df[f'volatility_{window}'] = returns.rolling(window).std()
                    result_df[f'volatility_ratio_{window}'] = result_df[f'volatility_{window}'] / result_df[f'volatility_{window}'].rolling(50).mean()
                
                # Volatility momentum
                result_df['volatility_momentum'] = result_df['volatility_20'].pct_change(5)
                
                # Price volatility relationship
                result_df['price_volatility_correlation'] = original_data['close'].rolling(20).corr(returns.rolling(20).std())
            
            # Technical indicators
            if 'close' in original_data.columns:
                # RSI
                delta = original_data['close'].diff()
                gains = delta.where(delta > 0, 0)
                losses = -delta.where(delta < 0, 0)
                avg_gains = gains.ewm(alpha=1/14, adjust=False).mean()
                avg_losses = losses.ewm(alpha=1/14, adjust=False).mean()
                rs = avg_gains / avg_losses
                result_df['rsi'] = 100 - (100 / (1 + rs))
                
                # MACD
                ema_12 = original_data['close'].ewm(span=12).mean()
                ema_26 = original_data['close'].ewm(span=26).mean()
                result_df['macd'] = ema_12 - ema_26
                result_df['macd_signal'] = result_df['macd'].ewm(span=9).mean()
                result_df['macd_histogram'] = result_df['macd'] - result_df['macd_signal']
                
                # Bollinger Bands
                sma_20 = original_data['close'].rolling(20).mean()
                std_20 = original_data['close'].rolling(20).std()
                result_df['bb_upper'] = sma_20 + (std_20 * 2)
                result_df['bb_lower'] = sma_20 - (std_20 * 2)
                result_df['bb_width'] = (result_df['bb_upper'] - result_df['bb_lower']) / sma_20
                result_df['bb_position'] = (original_data['close'] - result_df['bb_lower']) / (result_df['bb_upper'] - result_df['bb_lower'])
            
            # Momentum features
            if 'close' in original_data.columns:
                for period in [5, 10, 20]:
                    result_df[f'momentum_{period}'] = original_data['close'] - original_data['close'].shift(period)
                    result_df[f'momentum_ratio_{period}'] = result_df[f'momentum_{period}'] / original_data['close'].shift(period)
                
                # Momentum acceleration
                result_df['momentum_acceleration'] = result_df['momentum_5'].diff(5)
            
            # Feature interactions
            if 'close' in original_data.columns and 'volume' in original_data.columns:
                # Price-volume interactions
                result_df['price_volume_trend'] = (original_data['close'] - original_data['close'].shift(1)) * original_data['volume']
                result_df['price_volume_correlation'] = original_data['close'].rolling(20).corr(original_data['volume'])
                
                # Volume-weighted price features
                result_df['vwap'] = (original_data['close'] * original_data['volume']).rolling(20).sum() / original_data['volume'].rolling(20).sum()
                result_df['vwap_deviation'] = (original_data['close'] - result_df['vwap']) / result_df['vwap']
                result_df['vwap_deviation'] = result_df['vwap_deviation'].clip(-10, 10)
            
            self.logger.info("✅ Added HMM-specific features")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add HMM-specific features: {e}")
    
    def _clean_hmm_features(self, data):
        """Clean and validate HMM features."""
        try:
            # Remove infinite values
            data = data.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values with forward fill, then backward fill
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Remove columns that are all NaN
            data = data.dropna(axis=1, how='all')
            
            # Remove columns with constant values (no variance)
            constant_cols = []
            for col in data.columns:
                if data[col].nunique() <= 1:
                    constant_cols.append(col)
            
            if constant_cols:
                data = data.drop(columns=constant_cols)
                self.logger.info(f"📊 Removed {len(constant_cols)} constant columns")
            
            self.logger.info(f"✅ Cleaned features: {data.shape[1]} features, {data.shape[0]} samples")
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to clean features: {e}")
            return data
    
    def _generate_basic_hmm_features(self, data):
        """Generate basic HMM features as final fallback."""
        self.logger.warning("⚠️ Using basic HMM feature generation as fallback")
        
        result_df = data.copy()
        
        try:
            # Basic price features
            if 'close' in data.columns:
                result_df['returns'] = data['close'].pct_change()
                result_df['log_returns'] = np.log(data['close'] / data['close'].shift(1))
                result_df['volatility_20'] = result_df['returns'].rolling(20).std()
                
                # Simple momentum
                for period in [5, 10, 20]:
                    result_df[f'momentum_{period}'] = data['close'] - data['close'].shift(period)
            
            # Basic volume features
            if 'volume' in data.columns:
                result_df['volume_change'] = data['volume'].pct_change()
                result_df['volume_ma_20'] = data['volume'].rolling(20).mean()
                result_df['volume_ratio'] = data['volume'] / result_df['volume_ma_20']
            
            # Clean the result
            result_df = result_df.fillna(method='ffill').fillna(method='bfill')
            result_df = result_df.replace([np.inf, -np.inf], np.nan)
            result_df = result_df.dropna(axis=1, how='all')
            
            self.logger.info(f"✅ Generated {result_df.shape[1]} basic HMM features")
            return result_df
            
        except Exception as e:
            self.logger.error(f"❌ Basic feature generation failed: {e}")
            return data

# Create a global instance for compatibility
_global_hmm_generators: Optional[HMMCompatibleFeatureGenerators] = None

def get_hmm_compatible_generators() -> HMMCompatibleFeatureGenerators:
    """
    Get the global HMM-compatible feature generators instance.
    
    Returns:
        HMM-compatible feature generators instance
    """
    global _global_hmm_generators
    if _global_hmm_generators is None:
        _global_hmm_generators = HMMCompatibleFeatureGenerators()
    return _global_hmm_generators

# Compatibility alias
FeatureGenerators = HMMCompatibleFeatureGenerators
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
