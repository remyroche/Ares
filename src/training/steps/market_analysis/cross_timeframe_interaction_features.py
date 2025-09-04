from __future__ import annotations
'\nRefactored cross-timeframe and interaction feature generation with reduced complexity.\nThis module breaks down the high-complexity feature generation methods into smaller,\nfocused functions with proper type annotations.\n'
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

class TimeframeType(Enum):
    """Types of timeframes for analysis"""
    ULTRA_SHORT = [1, 2, 3]
    SHORT = [5, 10, 15]
    MEDIUM = [20, 30, 45]
    LONG = [60, 120, 240]

@dataclass
class CrossTimeframeConfig:
    """Configuration for cross-timeframe feature generation"""
    momentum_timeframes: list[int] = None
    volatility_timeframes: list[int] = None
    volume_timeframes: list[int] = None
    rsi_periods: list[int] = None
    macd_fast_periods: list[int] = None
    macd_slow_periods: list[int] = None
    bb_windows: list[int] = None
    bb_stds: list[float] = None
    min_data_points: int = 100
    variance_threshold: float = 1e-12
    parallel_processing: bool = True
    max_workers: int = 4

    def __post_init__(self) -> None:
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

    def __init__(self, config: CrossTimeframeConfig | None=None, logger: logging.Logger | None=None) -> None:
        """Initialize the generator.

        Args:
            config: Configuration for feature generation
            logger: Logger instance
        """
        self.config = config or CrossTimeframeConfig()
        self.logger = logger or logging.getLogger(__name__)

    def generate_cross_timeframe_features(self, price_data: pd.DataFrame, volume_data: pd.DataFrame | None=None) -> dict[str, pd.Series]:
        """Generate cross-timeframe features with reduced complexity.

        Args:
            price_data: OHLCV price data
            volume_data: Volume data (optional)

        Returns:
            Dictionary of feature name to Series mappings
        """
        if not self._validate_input_data(price_data):
            return {}
        price_components = self._extract_price_components(price_data)
        if not price_components:
            return {}
        features = {}
        if self.config.parallel_processing:
            features = self._generate_features_parallel(price_components, volume_data)
        else:
            features = self._generate_features_sequential(price_components, volume_data)
        valid_features = self._validate_features(features)
        self.logger.info(f'✅ Generated {len(valid_features)} valid cross-timeframe features')
        return valid_features

    def _validate_input_data(self, price_data: pd.DataFrame) -> bool:
        """Validate input data meets requirements"""
        if price_data.empty or len(price_data) < self.config.min_data_points:
            self.logger.warning(f'⚠️ Insufficient data: {len(price_data)} rows, need at least {self.config.min_data_points}')
            return False
        required_cols = {'open', 'high', 'low', 'close'}
        if not required_cols.issubset(price_data.columns):
            self.logger.warning(f'⚠️ Missing required columns: {required_cols - set(price_data.columns)}')
            return False
        return True

    def _extract_price_components(self, price_data: pd.DataFrame) -> dict[str, pd.Series]:
        """Extract and validate price components"""
        try:
            components = {'close': price_data['close'].astype(float), 'high': price_data['high'].astype(float), 'low': price_data['low'].astype(float), 'open': price_data['open'].astype(float)}
            if components['close'].isna().all() or components['close'].std() == 0:
                self.logger.warning('⚠️ Invalid close data')
                return {}
    def _generate_features_parallel(self, price_components: dict[str, pd.Series], volume_data: pd.DataFrame | None) -> dict[str, pd.Series]:
        """Generate features using parallel processing"""
        features = {}
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            futures.append(executor.submit(self._generate_momentum_features, price_components))
            futures.append(executor.submit(self._generate_volatility_features, price_components))
            futures.append(executor.submit(self._generate_range_features, price_components))
            futures.append(executor.submit(self._generate_technical_indicator_features, price_components))
            if volume_data is not None:
                futures.append(executor.submit(self._generate_volume_features, price_components, volume_data))
            for future in as_completed(futures):
                try:
                    result = future.result()
                    features.update(result)
                except Exception as e:
                    self.logger.exception(f'❌ Feature generation task failed: {e}')
        return features
    def _is_valid_feature(self, feature: pd.Series) -> bool:
        """Check if a feature is valid"""
        if feature is None or feature.empty:
            return False
        if feature.var() <= self.config.variance_threshold:
            return False
        return not feature.isna().all()

    def _validate_features(self, features: dict[str, pd.Series]) -> dict[str, pd.Series]:
        """Validate and filter features"""
        valid_features = {}
        for name, feature in features.items():
            if self._is_valid_feature(feature):
                valid_features[name] = feature
            else:
                self.logger.debug(f'⚠️ Skipping invalid feature: {name}')
        return valid_features

class InteractionFeatureGenerator:
    """Refactored interaction feature generator with reduced complexity"""

    def __init__(self, config: InteractionConfig | None=None, logger: logging.Logger | None=None) -> None:
        """Initialize the generator.

        Args:
            config: Configuration for feature generation
            logger: Logger instance
        """
        self.config = config or InteractionConfig()
        self.logger = logger or logging.getLogger(__name__)

    def generate_interaction_features(self, features: pd.DataFrame, feature_categories: dict[str, list[str]] | None=None) -> pd.DataFrame:
        """Generate interaction features with reduced complexity.

        Args:
            features: DataFrame containing base features
            feature_categories: Optional categorization of features

        Returns:
            DataFrame containing interaction features
        """
        if features.empty:
            self.logger.warning('⚠️ Empty features provided')
            return pd.DataFrame()
        selected_features = self._select_top_features(features)
        if len(selected_features) < 2:
            self.logger.warning('⚠️ Not enough features for interactions')
            return pd.DataFrame()
        if self.config.parallel_processing:
            interaction_features = self._generate_interactions_parallel(features[selected_features])
        else:
            interaction_features = self._generate_interactions_sequential(features[selected_features])
        final_features = self._remove_correlated_features(interaction_features)
        self.logger.info(f'✅ Generated {len(final_features.columns)} interaction features')
        return final_features

    def _select_top_features(self, features: pd.DataFrame) -> list[str]:
        """Select top features based on variance"""
        variances = features.var()
        valid_features = variances[variances > self.config.variance_threshold]
        return valid_features.nlargest(self.config.top_k_features).index.tolist()

    def _generate_interactions_parallel(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate interactions using parallel processing"""
        interaction_dfs = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            if self.config.include_ratios:
                futures.append(executor.submit(self._generate_ratio_features, features))
            if self.config.include_differences:
                futures.append(executor.submit(self._generate_difference_features, features))
            if self.config.include_products:
                futures.append(executor.submit(self._generate_product_features, features))
            if self.config.polynomial_degree > 1:
                futures.append(executor.submit(self._generate_polynomial_features, features))
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if not result.empty:
                        interaction_dfs.append(result)
                except Exception as e:
                    self.logger.exception(f'❌ Interaction generation failed: {e}')
        if interaction_dfs:
            return pd.concat(interaction_dfs, axis=1)
        return pd.DataFrame()

    def _generate_interactions_sequential(self, features: pd.DataFrame) -> pd.DataFrame:
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
                if self._same_category(col1, col2):
                    continue
                ratio = features[col1] / (features[col2] + 1e-08)
                if self._is_valid_interaction(ratio):
                    ratio_name = f'{col1}_ratio_{col2}'
                    ratio_features[ratio_name] = ratio
        return ratio_features

    def _generate_difference_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate difference interaction features"""
        diff_features = pd.DataFrame(index=features.index)
        feature_cols = features.columns.tolist()
        for i, col1 in enumerate(feature_cols):
            for col2 in feature_cols[i + 1:]:
                if self._same_category(col1, col2):
                    continue
                diff = features[col1] - features[col2]
                if self._is_valid_interaction(diff):
                    diff_name = f'{col1}_diff_{col2}'
                    diff_features[diff_name] = diff
        return diff_features

    def _generate_product_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate product interaction features"""
        product_features = pd.DataFrame(index=features.index)
        feature_cols = features.columns.tolist()
        for i, col1 in enumerate(feature_cols):
            for col2 in feature_cols[i + 1:]:
                if self._same_category(col1, col2):
                    continue
                product = features[col1] * features[col2]
                if self._is_valid_interaction(product):
                    product_name = f'{col1}_x_{col2}'
                    product_features[product_name] = product
        return product_features

    def _generate_polynomial_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate polynomial interaction features"""
        poly_features = pd.DataFrame(index=features.index)
        for col in features.columns:
            for degree in range(2, self.config.polynomial_degree + 1):
                poly = features[col] ** degree
                if self._is_valid_interaction(poly):
                    poly_name = f'{col}_pow{degree}'
                    poly_features[poly_name] = poly
        return poly_features

    def _same_category(self, col1: str, col2: str) -> bool:
        """Check if two columns belong to the same category"""
        cat1 = col1.split('_')[0]
        cat2 = col2.split('_')[0]
        same_categories = {('ma', 'ema', 'sma'), ('rsi', 'rsi'), ('macd', 'macd'), ('bb', 'bollinger'), ('volume', 'vol')}
        for category_group in same_categories:
            if cat1 in category_group and cat2 in category_group:
                return True
        return False

    def _is_valid_interaction(self, feature: pd.Series) -> bool:
        """Check if an interaction feature is valid"""
        if feature.empty:
            return False
        if feature.var() <= self.config.variance_threshold:
            return False
        return not (feature.isna().all() or np.isinf(feature).any())

    def _remove_correlated_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features"""
        if features.empty:
            return features
        corr_matrix = features.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > self.config.correlation_threshold)]
        result = features.drop(columns=to_drop)
        self.logger.info(f'Removed {len(to_drop)} highly correlated features')
        return result