"""
Enhanced Feature Engineering with Anti-Leakage Safeguards

This module implements comprehensive feature engineering with proper temporal handling
to avoid look-ahead bias and ensure robust regime detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings
from scipy import stats

# Import utilities
try:
    from src.utils.tprint import (
        tprint_info, tprint_success, tprint_warning, tprint_error
    )
except ImportError:
    def tprint_info(msg): print(f'ℹ️  {msg}')
    def tprint_success(msg): print(f'✅ {msg}')
    def tprint_warning(msg): print(f'⚠️  {msg}')
    def tprint_error(msg): print(f'❌ {msg}')


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # Return features
    include_raw_returns: bool = True
    include_log_returns: bool = True
    include_overnight_returns: bool = True
    
    # Volatility features
    include_realized_vol: bool = True
    vol_windows: List[int] = None
    
    # Rolling features
    include_rolling_features: bool = True
    rolling_windows: List[int] = None
    rolling_stats: List[str] = None
    
    # Factor exposures
    include_factor_exposures: bool = True
    factor_types: List[str] = None
    
    # Normalization
    enable_rank_normalization: bool = True
    normalization_method: str = 'rank'  # 'rank', 'zscore', 'minmax'
    
    # Anti-leakage
    shift_periods: int = 1
    enable_anti_leakage: bool = True
    
    def __post_init__(self):
        if self.vol_windows is None:
            self.vol_windows = [5, 10, 20]
        if self.rolling_windows is None:
            self.rolling_windows = [5, 10, 20]
        if self.rolling_stats is None:
            self.rolling_stats = ['mean', 'std', 'skew', 'kurtosis', 'zscore']
        if self.factor_types is None:
            self.factor_types = ['market', 'size', 'value', 'momentum']


class EnhancedFeatureEngineer:
    """
    Enhanced feature engineer with comprehensive feature types and anti-leakage safeguards.
    
    This class implements multiple feature types while ensuring no look-ahead bias
    through proper temporal handling and shifting.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize enhanced feature engineer.
        
        Args:
            config: Feature engineering configuration
        """
        self.config = config or FeatureConfig()
        self.logger = self._setup_logger()
        
        tprint_info("🔧 Initialized Enhanced Feature Engineer")
        
        # Initialize sub-components
        from .temporal_features import TemporalFeatureExtractor
        from .factor_exposures import FactorExposureCalculator
        from .rank_normalization import RankNormalizer
        
        self.temporal_extractor = TemporalFeatureExtractor(
            windows=self.config.rolling_windows,
            shift_periods=self.config.shift_periods
        )
        
        self.factor_calculator = FactorExposureCalculator(
            factor_types=self.config.factor_types
        )
        
        self.rank_normalizer = RankNormalizer(
            method=self.config.normalization_method
        )
    
    def _setup_logger(self):
        """Setup logger for feature engineering."""
        import logging
        return logging.getLogger(self.__class__.__name__)
    
    def extract_features(self, 
                      price_data: pd.DataFrame,
                      volume_data: Optional[pd.DataFrame] = None,
                      market_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Extract comprehensive features from price and volume data.
        
        Args:
            price_data: DataFrame with OHLC price data
            volume_data: Optional volume data
            market_data: Optional market data for factor calculations
            
        Returns:
            DataFrame with engineered features
        """
        tprint_info("🔍 Extracting enhanced features with anti-leakage safeguards")
        
        try:
            # Validate input data
            self._validate_input_data(price_data, volume_data, market_data)
            
            # Initialize features DataFrame
            features = pd.DataFrame(index=price_data.index)
            
            # 1. Return-based features
            if self.config.include_raw_returns:
                returns = self._calculate_returns(price_data)
                features = pd.concat([features, returns], axis=1)
                tprint_info("✅ Added raw returns features")
            
            if self.config.include_log_returns:
                log_returns = self._calculate_log_returns(price_data)
                features = pd.concat([features, log_returns], axis=1)
                tprint_info("✅ Added log returns features")
            
            if self.config.include_overnight_returns:
                overnight_returns = self._calculate_overnight_returns(price_data)
                features = pd.concat([features, overnight_returns], axis=1)
                tprint_info("✅ Added overnight returns features")
            
            # 2. Volatility features
            if self.config.include_realized_vol:
                vol_features = self._calculate_volatility_features(price_data)
                features = pd.concat([features, vol_features], axis=1)
                tprint_info("✅ Added volatility features")
            
            # 3. Rolling features with anti-leakage
            if self.config.include_rolling_features:
                rolling_features = self.temporal_extractor.extract_rolling_features(
                    price_data, volume_data
                )
                features = pd.concat([features, rolling_features], axis=1)
                tprint_info("✅ Added rolling features")
            
            # 4. Factor exposures
            if self.config.include_factor_exposures:
                factor_features = self.factor_calculator.calculate_factor_exposures(
                    price_data, volume_data, market_data
                )
                features = pd.concat([features, factor_features], axis=1)
                tprint_info("✅ Added factor exposure features")
            
            # 5. Regime predictive features
            regime_features = self._calculate_regime_predictive_features(
                price_data, volume_data
            )
            features = pd.concat([features, regime_features], axis=1)
            tprint_info("✅ Added regime predictive features")
            
            # 6. Apply normalization
            if self.config.enable_rank_normalization:
                features = self.rank_normalizer.normalize_features(features)
                tprint_info("✅ Applied rank normalization")
            
            # 7. Apply anti-leakage shifts
            if self.config.enable_anti_leakage:
                features = self._apply_anti_leakage_shifts(features)
                tprint_info("✅ Applied anti-leakage shifts")
            
            # 8. Clean and validate final features
            features = self._clean_features(features)
            
            tprint_success(f"✅ Enhanced feature extraction complete: {features.shape[1]} features")
            return features
            
        except Exception as e:
            tprint_error(f"❌ Feature extraction failed: {e}")
            self.logger.error(f"Feature extraction error: {e}", exc_info=True)
            raise
    
    def _validate_input_data(self, 
                           price_data: pd.DataFrame,
                           volume_data: Optional[pd.DataFrame],
                           market_data: Optional[pd.DataFrame]):
        """Validate input data for feature engineering."""
        if price_data.empty:
            raise ValueError("Price data cannot be empty")
        
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in price_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required price columns: {missing_columns}")
        
        if volume_data is not None and len(volume_data) != len(price_data):
            raise ValueError("Volume data length must match price data length")
        
        if market_data is not None and len(market_data) != len(price_data):
            raise ValueError("Market data length must match price data length")
    
    def _calculate_returns(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate raw returns with proper handling."""
        tprint_info("📈 Calculating raw returns")
        
        close_prices = price_data['close']
        returns = close_prices.pct_change()
        
        # Create return features
        tprint_info("📊 Creating multi-period return features")
        return_features = pd.DataFrame({
            'return_1d': returns,
            'return_3d': close_prices.pct_change(3),
            'return_5d': close_prices.pct_change(5),
            'return_10d': close_prices.pct_change(10),
            'return_20d': close_prices.pct_change(20)
        })
        
        tprint_success("✅ Raw returns calculated successfully")
        return return_features.add_prefix('raw_')
    
    def _calculate_log_returns(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate log returns with proper handling."""
        tprint_info("📈 Calculating log returns")
        
        close_prices = price_data['close']
        log_returns = np.log(close_prices / close_prices.shift(1))
        
        # Create log return features
        tprint_info("📊 Creating multi-period log return features")
        log_return_features = pd.DataFrame({
            'log_return_1d': log_returns,
            'log_return_3d': np.log(close_prices / close_prices.shift(3)),
            'log_return_5d': np.log(close_prices / close_prices.shift(5)),
            'log_return_10d': np.log(close_prices / close_prices.shift(10)),
            'log_return_20d': np.log(close_prices / close_prices.shift(20))
        })
        
        tprint_success("✅ Log returns calculated successfully")
        return log_return_features.add_prefix('log_')
    
    def _calculate_overnight_returns(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate overnight returns (close-to-open)."""
        tprint_info("🌙 Calculating overnight returns (close-to-open)")
        
        close_prices = price_data['close']
        open_prices = price_data['open']
        
        # Overnight return: (open_t - close_{t-1}) / close_{t-1}
        overnight_return = (open_prices - close_prices.shift(1)) / close_prices.shift(1)
        
        # Create overnight return features
        tprint_info("📊 Creating overnight return features")
        overnight_features = pd.DataFrame({
            'overnight_return': overnight_return,
            'overnight_return_3d_avg': overnight_return.rolling(3).mean(),
            'overnight_return_5d_avg': overnight_return.rolling(5).mean(),
            'overnight_return_vol': overnight_return.rolling(10).std()
        })
        
        tprint_success("✅ Overnight returns calculated successfully")
        return overnight_features.add_prefix('overnight_')
    
    def _calculate_volatility_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate realized volatility features."""
        tprint_info("📊 Calculating volatility features")
        
        close_prices = price_data['close']
        high_prices = price_data['high']
        low_prices = price_data['low']
        
        volatility_features = pd.DataFrame(index=price_data.index)
        
        tprint_info(f"📈 Using volatility windows: {self.config.vol_windows}")
        for window in self.config.vol_windows:
            tprint_info(f"🔄 Calculating {window}-day volatility features")
            
            # Realized volatility (standard deviation of returns)
            returns = close_prices.pct_change()
            realized_vol = returns.rolling(window).std() * np.sqrt(252)  # Annualized
            
            # Parkinson volatility (using high-low range)
            parkinson_vol = np.sqrt(
                0.361 * (np.log(high_prices / low_prices) ** 2).rolling(window).mean()
            ) * np.sqrt(252)
            
            # Garman-Klass volatility
            gk_vol = np.sqrt(
                0.5 * (np.log(high_prices / low_prices) ** 2).rolling(window).mean() -
                (2 * np.log(2) - 1) * (np.log(close_prices / close_prices.shift(1)) ** 2).rolling(window).mean()
            ) * np.sqrt(252)
            
            # Add to features
            volatility_features[f'realized_vol_{window}d'] = realized_vol
            volatility_features[f'parkinson_vol_{window}d'] = parkinson_vol
            volatility_features[f'gk_vol_{window}d'] = gk_vol
            
            # Volatility ratios
            if window > 5:
                short_vol = returns.rolling(5).std() * np.sqrt(252)
                vol_ratio = realized_vol / short_vol
                volatility_features[f'vol_ratio_{window}d_5d'] = vol_ratio
        
        tprint_success("✅ Volatility features calculated successfully")
        return volatility_features.add_prefix('vol_')
    
    def _calculate_regime_predictive_features(self,
                                         price_data: pd.DataFrame,
                                         volume_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate features that historically predict regime shifts."""
        tprint_info("🔍 Calculating regime predictive features")
        
        close_prices = price_data['close']
        high_prices = price_data['high']
        low_prices = price_data['low']
        
        regime_features = pd.DataFrame(index=price_data.index)
        
        # Short-term momentum reversal signals
        tprint_info("📈 Calculating momentum reversal signals")
        returns_1d = close_prices.pct_change()
        returns_3d = close_prices.pct_change(3)
        momentum_reversal = returns_1d * np.sign(returns_3d.shift(1))
        regime_features['momentum_reversal'] = momentum_reversal
        
        # Volume spikes (if volume data available)
        if volume_data is not None:
            tprint_info("📊 Calculating volume spike features")
            volume = volume_data['volume'] if 'volume' in volume_data.columns else volume_data.iloc[:, 0]
            volume_ma = volume.rolling(20).mean()
            volume_spike = volume / volume_ma
            regime_features['volume_spike'] = volume_spike
            regime_features['volume_spike_5d_avg'] = volume_spike.rolling(5).mean()
        
        # Spread widenings
        tprint_info("📈 Calculating spread widening features")
        spread = (high_prices - low_prices) / close_prices
        spread_ma = spread.rolling(20).mean()
        spread_widening = spread / spread_ma
        regime_features['spread_widening'] = spread_widening
        regime_features['spread_widening_5d_avg'] = spread_widening.rolling(5).mean()
        
        # Price acceleration
        tprint_info("📈 Calculating price acceleration features")
        price_change_1d = close_prices.diff()
        price_change_3d = close_prices.diff(3)
        price_acceleration = price_change_1d - price_change_3d.shift(2)
        regime_features['price_acceleration'] = price_acceleration
        
        # Volatility of volatility
        tprint_info("📊 Calculating volatility of volatility features")
        returns = close_prices.pct_change()
        vol_5d = returns.rolling(5).std()
        vol_20d = returns.rolling(20).std()
        vol_of_vol = vol_5d.rolling(10).std()
        regime_features['vol_of_vol'] = vol_of_vol
        regime_features['vol_ratio_5d_20d'] = vol_5d / vol_20d
        
        tprint_success("✅ Regime predictive features calculated successfully")
        return regime_features.add_prefix('regime_')
    
    def _apply_anti_leakage_shifts(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply shifts to prevent look-ahead bias."""
        tprint_info(f"🔄 Applying anti-leakage shifts ({self.config.shift_periods} periods)")
        
        if self.config.shift_periods <= 0:
            tprint_warning("⚠️ No shift periods specified, skipping anti-leakage")
            return features
        
        shifted_features = features.copy()
        
        # Apply shift to all features except those that should be current
        tprint_info("🔄 Shifting features to prevent look-ahead bias")
        shifted_count = 0
        for col in features.columns:
            # Don't shift features that are already lagged or are identifiers
            if not any(suffix in col.lower() for suffix in ['_lag', '_shift', '_id']):
                shifted_features[col] = features[col].shift(self.config.shift_periods)
                shifted_count += 1
        
        tprint_success(f"✅ Applied anti-leakage shifts to {shifted_count} features")
        return shifted_features
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate final features."""
        tprint_info("🧹 Cleaning and validating final features")
        
        # Remove infinite values
        tprint_info("🔄 Removing infinite values")
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Remove columns with too many NaN values
        tprint_info("🔍 Checking for columns with excessive NaN values")
        nan_ratio = features.isna().sum() / len(features)
        valid_columns = nan_ratio[nan_ratio < 0.5].index
        removed_cols = set(features.columns) - set(valid_columns)
        if removed_cols:
            tprint_warning(f"⚠️ Removing {len(removed_cols)} columns with excessive NaN values")
        features = features[valid_columns]
        
        # Forward fill remaining NaN values (limited)
        tprint_info("🔄 Applying forward fill (limit=3)")
        features = features.fillna(method='ffill', limit=3)
        
        # Backward fill remaining NaN values (limited)
        tprint_info("🔄 Applying backward fill (limit=1)")
        features = features.fillna(method='bfill', limit=1)
        
        # Drop any remaining NaN rows
        initial_rows = len(features)
        features = features.dropna()
        final_rows = len(features)
        if initial_rows != final_rows:
            tprint_info(f"🧹 Dropped {initial_rows - final_rows} rows with remaining NaN values")
        
        tprint_success(f"✅ Feature cleaning complete: {features.shape}")
        return features


def create_enhanced_feature_engineer(
    include_raw_returns: bool = True,
    include_log_returns: bool = True,
    include_realized_vol: bool = True,
    include_rolling_features: bool = True,
    include_factor_exposures: bool = True,
    rolling_windows: Optional[List[int]] = None,
    shift_periods: int = 1,
    enable_rank_normalization: bool = True
) -> EnhancedFeatureEngineer:
    """
    Factory function to create enhanced feature engineer.
    
    Args:
        include_raw_returns: Include raw return features
        include_log_returns: Include log return features
        include_realized_vol: Include volatility features
        include_rolling_features: Include rolling statistical features
        include_factor_exposures: Include factor exposure features
        rolling_windows: List of rolling windows
        shift_periods: Number of periods to shift for anti-leakage
        enable_rank_normalization: Enable rank normalization
        
    Returns:
        EnhancedFeatureEngineer instance
    """
    tprint_info("🏭 Creating Enhanced Feature Engineer with factory function")
    
    config = FeatureConfig(
        include_raw_returns=include_raw_returns,
        include_log_returns=include_log_returns,
        include_realized_vol=include_realized_vol,
        include_rolling_features=include_rolling_features,
        include_factor_exposures=include_factor_exposures,
        rolling_windows=rolling_windows,
        shift_periods=shift_periods,
        enable_rank_normalization=enable_rank_normalization
    )
    
    tprint_info(f"📊 Configuration: raw_returns={include_raw_returns}, log_returns={include_log_returns}, vol={include_realized_vol}")
    tprint_info(f"📊 Configuration: rolling_features={include_rolling_features}, factor_exposures={include_factor_exposures}")
    tprint_info(f"📊 Configuration: shift_periods={shift_periods}, rank_normalization={enable_rank_normalization}")
    
    engineer = EnhancedFeatureEngineer(config)
    tprint_success("✅ Enhanced Feature Engineer created successfully")
    return engineer