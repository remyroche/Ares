"""
Refactored cross-timeframe and interaction feature generation with reduced complexity.
This module breaks down the high-complexity feature generation methods into smaller,
focused functions with proper type annotations.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Set
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed


class TimeframeType(Enum):
    """Types of timeframes for analysis"""
    ULTRA_SHORT = [1, 2, 3]
    SHORT = [5, 10, 15]
    MEDIUM = [20, 30, 45]
    LONG = [60, 120, 240]


@dataclass
class CrossTimeframeConfig:
    """Configuration for cross-timeframe feature generation"""
    momentum_timeframes: List[int] = None
    volatility_timeframes: List[int] = None
    volume_timeframes: List[int] = None
    rsi_periods: List[int] = None
    macd_fast_periods: List[int] = None
    macd_slow_periods: List[int] = None
    bb_windows: List[int] = None
    bb_stds: List[float] = None
    min_data_points: int = 100
    variance_threshold: float = 1e-12
    parallel_processing: bool = True
    max_workers: int = 4
    
    def __post_init__(self):
        """Initialize default values"""
        if self.momentum_timeframes is None:
            self.momentum_timeframes = [1, 3, 5, 10, 15, 20]
        if self.volatility_timeframes is None:
            self.volatility_timeframes = [3, 5, 10, 15, 20, 30]
        if self.volume_timeframes is None:
            self.volume_timeframes = [5, 10, 15, 30]
        if self.rsi_periods is None:
            self.rsi_periods = [3, 5, 10, 14, 21]
        if self.macd_fast_periods is None:
            self.macd_fast_periods = [3, 5, 8, 12]
        if self.macd_slow_periods is None:
            self.macd_slow_periods = [10, 15, 20, 26]
        if self.bb_windows is None:
            self.bb_windows = [10, 15, 20]
        if self.bb_stds is None:
            self.bb_stds = [1.0, 1.5, 2.0]


@dataclass
class InteractionConfig:
    """Configuration for interaction feature generation"""
    max_interaction_depth: int = 2
    top_k_features: int = 50
    correlation_threshold: float = 0.95
    variance_threshold: float = 1e-12
    polynomial_degree: int = 2
    include_ratios: bool = True
    include_differences: bool = True
    include_products: bool = True
    parallel_processing: bool = True
    max_workers: int = 4


class CrossTimeframeFeatureGenerator:
    """Refactored cross-timeframe feature generator with reduced complexity"""
    
    def __init__(
        self,
        config: Optional[CrossTimeframeConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the generator.
        
        Args:
            config: Configuration for feature generation
            logger: Logger instance
        """
        self.config = config or CrossTimeframeConfig()
        self.logger = logger or logging.getLogger(__name__)
    
    def generate_cross_timeframe_features(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.Series]:
        """Generate cross-timeframe features with reduced complexity.
        
        Args:
            price_data: OHLCV price data
            volume_data: Volume data (optional)
            
        Returns:
            Dictionary of feature name to Series mappings
        """
        # Validate input data
        if not self._validate_input_data(price_data):
            return {}
        
        # Extract price components
        price_components = self._extract_price_components(price_data)
        if not price_components:
            return {}
        
        features = {}
        
        # Generate features in parallel if configured
        if self.config.parallel_processing:
            features = self._generate_features_parallel(price_components, volume_data)
        else:
            features = self._generate_features_sequential(price_components, volume_data)
        
        # Validate and filter features
        valid_features = self._validate_features(features)
        
        self.logger.info(f"✅ Generated {len(valid_features)} valid cross-timeframe features")
        return valid_features
    
    def _validate_input_data(self, price_data: pd.DataFrame) -> bool:
        """Validate input data meets requirements"""
        if price_data.empty or len(price_data) < self.config.min_data_points:
            self.logger.warning(
                f"⚠️ Insufficient data: {len(price_data)} rows, "
                f"need at least {self.config.min_data_points}"
            )
            return False
        
        required_cols = {'open', 'high', 'low', 'close'}
        if not required_cols.issubset(price_data.columns):
            self.logger.warning(f"⚠️ Missing required columns: {required_cols - set(price_data.columns)}")
            return False
        
        return True
    
    def _extract_price_components(self, price_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Extract and validate price components"""
        try:
            components = {
                'close': price_data['close'].astype(float),
                'high': price_data['high'].astype(float),
                'low': price_data['low'].astype(float),
                'open': price_data['open'].astype(float)
            }
            
            # Validate close prices
            if components['close'].isna().all() or components['close'].std() == 0:
                self.logger.warning("⚠️ Invalid close data")
                return {}
            
            return components
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting price components: {e}")
            return {}
    
    def _generate_features_parallel(
        self,
        price_components: Dict[str, pd.Series],
        volume_data: Optional[pd.DataFrame]
    ) -> Dict[str, pd.Series]:
        """Generate features using parallel processing"""
        features = {}
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            
            # Submit feature generation tasks
            futures.append(
                executor.submit(self._generate_momentum_features, price_components)
            )
            futures.append(
                executor.submit(self._generate_volatility_features, price_components)
            )
            futures.append(
                executor.submit(self._generate_range_features, price_components)
            )
            futures.append(
                executor.submit(self._generate_technical_indicator_features, price_components)
            )
            
            if volume_data is not None:
                futures.append(
                    executor.submit(self._generate_volume_features, price_components, volume_data)
                )
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    features.update(result)
                except Exception as e:
                    self.logger.error(f"❌ Feature generation task failed: {e}")
        
        return features
    
    def _generate_features_sequential(
        self,
        price_components: Dict[str, pd.Series],
        volume_data: Optional[pd.DataFrame]
    ) -> Dict[str, pd.Series]:
        """Generate features sequentially"""
        features = {}
        
        # Generate each feature category
        features.update(self._generate_momentum_features(price_components))
        features.update(self._generate_volatility_features(price_components))
        features.update(self._generate_range_features(price_components))
        features.update(self._generate_technical_indicator_features(price_components))
        
        if volume_data is not None:
            features.update(self._generate_volume_features(price_components, volume_data))
        
        return features
    
    def _generate_momentum_features(
        self,
        price_components: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """Generate momentum-based cross-timeframe features"""
        features = {}
        close = price_components['close']
        high = price_components['high']
        low = price_components['low']
        
        timeframes = self.config.momentum_timeframes[:4]  # Limit for safety
        
        for i, tf1 in enumerate(timeframes):
            for tf2 in timeframes[i + 1:]:
                if tf1 < len(close) and tf2 < len(close):
                    # Momentum difference
                    momentum_diff = close.pct_change(tf1) - close.pct_change(tf2)
                    if self._is_valid_feature(momentum_diff):
                        features[f"momentum_{tf1}m_{tf2}m"] = momentum_diff
                    
                    # Momentum ratio
                    momentum_ratio = close.pct_change(tf1) / (close.pct_change(tf2) + 1e-8)
                    if self._is_valid_feature(momentum_ratio):
                        features[f"momentum_ratio_{tf1}m_{tf2}m"] = momentum_ratio
                    
                    # High-Low momentum
                    if len(close) >= max(tf1, tf2) * 2:
                        hl_features = self._calculate_hl_momentum(high, low, close, tf1, tf2)
                        features.update(hl_features)
        
        return features
    
    def _calculate_hl_momentum(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        tf1: int,
        tf2: int
    ) -> Dict[str, pd.Series]:
        """Calculate high-low momentum features"""
        features = {}
        
        hl_momentum_1 = (
            high.rolling(tf1, min_periods=tf1 // 2).max() -
            low.rolling(tf1, min_periods=tf1 // 2).min()
        ) / (close.rolling(tf1, min_periods=tf1 // 2).mean() + 1e-8)
        
        hl_momentum_2 = (
            high.rolling(tf2, min_periods=tf2 // 2).max() -
            low.rolling(tf2, min_periods=tf2 // 2).min()
        ) / (close.rolling(tf2, min_periods=tf2 // 2).mean() + 1e-8)
        
        hl_diff = hl_momentum_1 - hl_momentum_2
        if self._is_valid_feature(hl_diff):
            features[f"hl_momentum_{tf1}m_{tf2}m"] = hl_diff
        
        return features
    
    def _generate_volatility_features(
        self,
        price_components: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """Generate volatility-based cross-timeframe features"""
        features = {}
        close = price_components['close']
        
        returns = close.pct_change().fillna(method='ffill').fillna(method='bfill').fillna(0)
        timeframes = self.config.volatility_timeframes[:3]  # Limit for safety
        
        for i, tf1 in enumerate(timeframes):
            for tf2 in timeframes[i + 1:]:
                if tf1 < len(close) and tf2 < len(close):
                    vol_features = self._calculate_volatility_pair(returns, tf1, tf2)
                    features.update(vol_features)
        
        return features
    
    def _calculate_volatility_pair(
        self,
        returns: pd.Series,
        tf1: int,
        tf2: int
    ) -> Dict[str, pd.Series]:
        """Calculate volatility features for a timeframe pair"""
        features = {}
        
        vol_1 = returns.rolling(tf1, min_periods=tf1 // 2).std()
        vol_2 = returns.rolling(tf2, min_periods=tf2 // 2).std()
        
        # Volatility ratio
        vol_ratio = vol_1 / (vol_2 + 1e-8)
        if self._is_valid_feature(vol_ratio):
            features[f"volatility_ratio_{tf1}m_{tf2}m"] = vol_ratio
        
        # Volatility difference
        vol_diff = vol_1 - vol_2
        if self._is_valid_feature(vol_diff):
            features[f"volatility_diff_{tf1}m_{tf2}m"] = vol_diff
        
        # Volatility standard deviation
        if len(returns) >= 20:
            vol_std = (vol_1 - vol_2).rolling(20, min_periods=10).std()
            if self._is_valid_feature(vol_std):
                features[f"volatility_std_{tf1}m_{tf2}m"] = vol_std
        
        return features
    
    def _generate_range_features(
        self,
        price_components: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """Generate price range cross-timeframe features"""
        features = {}
        high = price_components['high']
        low = price_components['low']
        close = price_components['close']
        
        timeframes = self.config.momentum_timeframes[:3]  # Limit for safety
        
        for i, tf1 in enumerate(timeframes):
            for tf2 in timeframes[i + 1:]:
                if tf1 < len(close) and tf2 < len(close):
                    range_features = self._calculate_range_pair(high, low, close, tf1, tf2)
                    features.update(range_features)
        
        return features
    
    def _calculate_range_pair(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        tf1: int,
        tf2: int
    ) -> Dict[str, pd.Series]:
        """Calculate range features for a timeframe pair"""
        features = {}
        
        range_1 = (
            high.rolling(tf1, min_periods=tf1 // 2).max() -
            low.rolling(tf1, min_periods=tf1 // 2).min()
        ) / (close.rolling(tf1, min_periods=tf1 // 2).mean() + 1e-8)
        
        range_2 = (
            high.rolling(tf2, min_periods=tf2 // 2).max() -
            low.rolling(tf2, min_periods=tf2 // 2).min()
        ) / (close.rolling(tf2, min_periods=tf2 // 2).mean() + 1e-8)
        
        # Range ratio
        range_ratio = range_1 / (range_2 + 1e-8)
        if self._is_valid_feature(range_ratio):
            features[f"price_range_ratio_{tf1}m_{tf2}m"] = range_ratio
        
        # Range difference
        range_diff = range_1 - range_2
        if self._is_valid_feature(range_diff):
            features[f"price_range_diff_{tf1}m_{tf2}m"] = range_diff
        
        return features
    
    def _generate_technical_indicator_features(
        self,
        price_components: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """Generate technical indicator cross-timeframe features"""
        features = {}
        
        # RSI features
        features.update(self._generate_rsi_features(price_components))
        
        # MACD features
        features.update(self._generate_macd_features(price_components))
        
        # Bollinger Bands features
        features.update(self._generate_bb_features(price_components))
        
        return features
    
    def _generate_rsi_features(
        self,
        price_components: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """Generate RSI cross-timeframe features"""
        features = {}
        close = price_components['close']
        
        for i, period1 in enumerate(self.config.rsi_periods[:-1]):
            for period2 in self.config.rsi_periods[i + 1:]:
                if period1 < len(close) and period2 < len(close):
                    rsi_1 = self._calculate_rsi(close, period1)
                    rsi_2 = self._calculate_rsi(close, period2)
                    
                    # RSI difference
                    rsi_diff = rsi_1 - rsi_2
                    if self._is_valid_feature(rsi_diff):
                        features[f"rsi_diff_{period1}_{period2}"] = rsi_diff
                    
                    # RSI ratio
                    rsi_ratio = rsi_1 / (rsi_2 + 1e-8)
                    if self._is_valid_feature(rsi_ratio):
                        features[f"rsi_ratio_{period1}_{period2}"] = rsi_ratio
        
        return features
    
    def _generate_macd_features(
        self,
        price_components: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """Generate MACD cross-timeframe features"""
        features = {}
        close = price_components['close']
        
        for fast in self.config.macd_fast_periods[:3]:
            for slow in self.config.macd_slow_periods[:3]:
                if fast < slow and slow < len(close):
                    macd_1 = self._calculate_macd(close, fast, slow)
                    macd_2 = self._calculate_macd(close, fast * 2, slow * 2)
                    
                    # MACD difference
                    macd_diff = macd_1 - macd_2
                    if self._is_valid_feature(macd_diff):
                        features[f"macd_diff_{fast}_{slow}"] = macd_diff
                    
                    # MACD ratio
                    macd_ratio = macd_1 / (macd_2 + 1e-8)
                    if self._is_valid_feature(macd_ratio):
                        features[f"macd_ratio_{fast}_{slow}"] = macd_ratio
        
        return features
    
    def _generate_bb_features(
        self,
        price_components: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """Generate Bollinger Bands cross-timeframe features"""
        features = {}
        close = price_components['close']
        
        for window in self.config.bb_windows:
            for std in self.config.bb_stds[:2]:  # Limit for safety
                if window < len(close):
                    bb_1 = self._calculate_bollinger_position(close, window, std)
                    bb_2 = self._calculate_bollinger_position(close, window * 2, std)
                    
                    if bb_1 is not None and bb_2 is not None:
                        bb_diff = bb_1 - bb_2
                        if self._is_valid_feature(bb_diff):
                            features[f"bb_position_diff_{window}_{std}"] = bb_diff
        
        return features
    
    def _generate_volume_features(
        self,
        price_components: Dict[str, pd.Series],
        volume_data: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """Generate volume-based cross-timeframe features"""
        features = {}
        
        if 'volume' not in volume_data.columns:
            return features
        
        volume = volume_data['volume'].astype(float)
        if volume.var() <= self.config.variance_threshold:
            return features
        
        timeframes = self.config.volume_timeframes[:3]  # Limit for safety
        
        for i, tf1 in enumerate(timeframes):
            for tf2 in timeframes[i + 1:]:
                if tf1 < len(volume) and tf2 < len(volume):
                    volume_features = self._calculate_volume_pair(volume, tf1, tf2)
                    features.update(volume_features)
        
        return features
    
    def _calculate_volume_pair(
        self,
        volume: pd.Series,
        tf1: int,
        tf2: int
    ) -> Dict[str, pd.Series]:
        """Calculate volume features for a timeframe pair"""
        features = {}
        
        vol_1 = volume.rolling(tf1, min_periods=tf1 // 2).mean()
        vol_2 = volume.rolling(tf2, min_periods=tf2 // 2).mean()
        
        # Volume ratio
        vol_ratio = vol_1 / (vol_2 + 1e-8)
        if self._is_valid_feature(vol_ratio):
            features[f"volume_ratio_{tf1}m_{tf2}m"] = vol_ratio
        
        # Volume difference
        vol_diff = vol_1 - vol_2
        if self._is_valid_feature(vol_diff):
            features[f"volume_diff_{tf1}m_{tf2}m"] = vol_diff
        
        # Volume momentum
        vol_momentum = volume.pct_change(tf1) - volume.pct_change(tf2)
        if self._is_valid_feature(vol_momentum):
            features[f"volume_momentum_{tf1}m_{tf2}m"] = vol_momentum
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast_period: int,
        slow_period: int
    ) -> pd.Series:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=fast_period, adjust=False).mean()
        exp2 = prices.ewm(span=slow_period, adjust=False).mean()
        return exp1 - exp2
    
    def _calculate_bollinger_position(
        self,
        prices: pd.Series,
        window: int,
        num_std: float
    ) -> Optional[pd.Series]:
        """Calculate position relative to Bollinger Bands"""
        try:
            sma = prices.rolling(window=window).mean()
            std = prices.rolling(window=window).std()
            upper_band = sma + (std * num_std)
            lower_band = sma - (std * num_std)
            
            # Position between bands (0 = lower, 1 = upper)
            position = (prices - lower_band) / (upper_band - lower_band + 1e-8)
            return position
        except Exception:
            return None
    
    def _is_valid_feature(self, feature: pd.Series) -> bool:
        """Check if a feature is valid"""
        if feature is None or feature.empty:
            return False
        
        # Check variance
        if feature.var() <= self.config.variance_threshold:
            return False
        
        # Check for all NaN
        if feature.isna().all():
            return False
        
        return True
    
    def _validate_features(
        self,
        features: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """Validate and filter features"""
        valid_features = {}
        
        for name, feature in features.items():
            if self._is_valid_feature(feature):
                valid_features[name] = feature
            else:
                self.logger.debug(f"⚠️ Skipping invalid feature: {name}")
        
        return valid_features


class InteractionFeatureGenerator:
    """Refactored interaction feature generator with reduced complexity"""
    
    def __init__(
        self,
        config: Optional[InteractionConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the generator.
        
        Args:
            config: Configuration for feature generation
            logger: Logger instance
        """
        self.config = config or InteractionConfig()
        self.logger = logger or logging.getLogger(__name__)
    
    def generate_interaction_features(
        self,
        features: pd.DataFrame,
        feature_categories: Optional[Dict[str, List[str]]] = None
    ) -> pd.DataFrame:
        """Generate interaction features with reduced complexity.
        
        Args:
            features: DataFrame containing base features
            feature_categories: Optional categorization of features
            
        Returns:
            DataFrame containing interaction features
        """
        if features.empty:
            self.logger.warning("⚠️ Empty features provided")
            return pd.DataFrame()
        
        # Select top features for interaction
        selected_features = self._select_top_features(features)
        if len(selected_features) < 2:
            self.logger.warning("⚠️ Not enough features for interactions")
            return pd.DataFrame()
        
        # Generate interactions
        if self.config.parallel_processing:
            interaction_features = self._generate_interactions_parallel(
                features[selected_features]
            )
        else:
            interaction_features = self._generate_interactions_sequential(
                features[selected_features]
            )
        
        # Remove highly correlated features
        final_features = self._remove_correlated_features(interaction_features)
        
        self.logger.info(f"✅ Generated {len(final_features.columns)} interaction features")
        return final_features
    
    def _select_top_features(self, features: pd.DataFrame) -> List[str]:
        """Select top features based on variance"""
        # Calculate variance for each feature
        variances = features.var()
        
        # Remove features with low variance
        valid_features = variances[variances > self.config.variance_threshold]
        
        # Sort by variance and select top k
        top_features = valid_features.nlargest(self.config.top_k_features).index.tolist()
        
        return top_features
    
    def _generate_interactions_parallel(
        self,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate interactions using parallel processing"""
        interaction_dfs = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            
            if self.config.include_ratios:
                futures.append(
                    executor.submit(self._generate_ratio_features, features)
                )
            
            if self.config.include_differences:
                futures.append(
                    executor.submit(self._generate_difference_features, features)
                )
            
            if self.config.include_products:
                futures.append(
                    executor.submit(self._generate_product_features, features)
                )
            
            if self.config.polynomial_degree > 1:
                futures.append(
                    executor.submit(self._generate_polynomial_features, features)
                )
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if not result.empty:
                        interaction_dfs.append(result)
                except Exception as e:
                    self.logger.error(f"❌ Interaction generation failed: {e}")
        
        if interaction_dfs:
            return pd.concat(interaction_dfs, axis=1)
        return pd.DataFrame()
    
    def _generate_interactions_sequential(
        self,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate interactions sequentially"""
        interaction_dfs = []
        
        if self.config.include_ratios:
            interaction_dfs.append(self._generate_ratio_features(features))
        
        if self.config.include_differences:
            interaction_dfs.append(self._generate_difference_features(features))
        
        if self.config.include_products:
            interaction_dfs.append(self._generate_product_features(features))
        
        if self.config.polynomial_degree > 1:
            interaction_dfs.append(self._generate_polynomial_features(features))
        
        if interaction_dfs:
            return pd.concat(interaction_dfs, axis=1)
        return pd.DataFrame()
    
    def _generate_ratio_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate ratio interaction features"""
        ratio_features = pd.DataFrame(index=features.index)
        feature_cols = features.columns.tolist()
        
        for i, col1 in enumerate(feature_cols):
            for col2 in feature_cols[i + 1:]:
                # Skip if same feature category (if identifiable)
                if self._same_category(col1, col2):
                    continue
                
                # Create ratio with safety check
                ratio = features[col1] / (features[col2] + 1e-8)
                
                if self._is_valid_interaction(ratio):
                    ratio_name = f"{col1}_ratio_{col2}"
                    ratio_features[ratio_name] = ratio
        
        return ratio_features
    
    def _generate_difference_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate difference interaction features"""
        diff_features = pd.DataFrame(index=features.index)
        feature_cols = features.columns.tolist()
        
        for i, col1 in enumerate(feature_cols):
            for col2 in feature_cols[i + 1:]:
                # Skip if same feature category
                if self._same_category(col1, col2):
                    continue
                
                # Create difference
                diff = features[col1] - features[col2]
                
                if self._is_valid_interaction(diff):
                    diff_name = f"{col1}_diff_{col2}"
                    diff_features[diff_name] = diff
        
        return diff_features
    
    def _generate_product_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate product interaction features"""
        product_features = pd.DataFrame(index=features.index)
        feature_cols = features.columns.tolist()
        
        for i, col1 in enumerate(feature_cols):
            for col2 in feature_cols[i + 1:]:
                # Skip if same feature category
                if self._same_category(col1, col2):
                    continue
                
                # Create product
                product = features[col1] * features[col2]
                
                if self._is_valid_interaction(product):
                    product_name = f"{col1}_x_{col2}"
                    product_features[product_name] = product
        
        return product_features
    
    def _generate_polynomial_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate polynomial interaction features"""
        poly_features = pd.DataFrame(index=features.index)
        
        for col in features.columns:
            for degree in range(2, self.config.polynomial_degree + 1):
                poly = features[col] ** degree
                
                if self._is_valid_interaction(poly):
                    poly_name = f"{col}_pow{degree}"
                    poly_features[poly_name] = poly
        
        return poly_features
    
    def _same_category(self, col1: str, col2: str) -> bool:
        """Check if two columns belong to the same category"""
        # Extract category prefixes
        cat1 = col1.split('_')[0]
        cat2 = col2.split('_')[0]
        
        # Common category patterns
        same_categories = {
            ('ma', 'ema', 'sma'),
            ('rsi', 'rsi'),
            ('macd', 'macd'),
            ('bb', 'bollinger'),
            ('volume', 'vol'),
        }
        
        for category_group in same_categories:
            if cat1 in category_group and cat2 in category_group:
                return True
        
        return False
    
    def _is_valid_interaction(self, feature: pd.Series) -> bool:
        """Check if an interaction feature is valid"""
        if feature.empty:
            return False
        
        # Check variance
        if feature.var() <= self.config.variance_threshold:
            return False
        
        # Check for all NaN or infinite values
        if feature.isna().all() or np.isinf(feature).any():
            return False
        
        return True
    
    def _remove_correlated_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features"""
        if features.empty:
            return features
        
        # Calculate correlation matrix
        corr_matrix = features.corr().abs()
        
        # Find features to remove
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [
            column for column in upper_triangle.columns
            if any(upper_triangle[column] > self.config.correlation_threshold)
        ]
        
        # Drop highly correlated features
        result = features.drop(columns=to_drop)
        
        self.logger.info(f"Removed {len(to_drop)} highly correlated features")
        return result