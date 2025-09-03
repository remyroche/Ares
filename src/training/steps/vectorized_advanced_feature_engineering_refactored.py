"""Refactored VectorizedAdvancedFeatureEngineering with reduced complexity and type
hints.

This refactored version breaks down the massive engineer_features method into smaller,
focused methods with proper type annotations.
"""

import asyncio
import datetime
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class FeatureCategory(Enum):
    """Enumeration of feature categories."""

    PRICE = "price"
    VOLUME = "volume"
    MICROSTRUCTURE = "microstructure"
    WAVELET = "wavelet"
    REGIME = "regime"
    TECHNICAL = "technical"
    ORDER_FLOW = "order_flow"
    CROSS_TIMEFRAME = "cross_timeframe"
    INTERACTION = "interaction"


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    enable_wavelet: bool = True
    enable_microstructure: bool = True
    enable_regime: bool = True
    enable_technical: bool = True
    enable_order_flow: bool = True
    enable_cross_timeframe: bool = True
    enable_interaction: bool = True
    max_lag: int = 20
    interaction_depth: int = 2


@dataclass
class PreprocessingResult:
    """Result of data preprocessing."""

    data: pd.DataFrame
    original_shape: Tuple[int, int]
    preprocessed_shape: Tuple[int, int]
    quality_improvement: float
    method: str


class VectorizedAdvancedFeatureEngineeringRefactored:
    """Refactored version with reduced complexity and type hints."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the feature engineering system.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self.is_initialized = False
        self.feature_config = FeatureConfig()
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all required components."""
        try:
            # Initialize feature extractors
            self._initialize_feature_extractors()
            # Initialize preprocessors
            self._initialize_preprocessors()
            # Initialize validators
            self._initialize_validators()
            self.is_initialized = True
            self.logger.info("✅ Feature engineering components initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize components: {e}")
            raise

    def _initialize_feature_extractors(self) -> None:
        """Initialize feature extraction components."""
        # Placeholder for actual initialization
        pass

    def _initialize_preprocessors(self) -> None:
        """Initialize data preprocessing components."""
        # Placeholder for actual initialization
        pass

    def _initialize_validators(self) -> None:
        """Initialize data validation components."""
        # Placeholder for actual initialization
        pass

    async def engineer_features(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: Optional[pd.DataFrame] = None,
        sr_levels: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Engineer advanced features with reduced complexity.

        This is the main refactored method that orchestrates feature engineering
        by delegating to specialized methods.

        Args:
            price_data: OHLCV price data
            volume_data: Volume and trade flow data
            order_flow_data: Order book and flow data (optional)
            sr_levels: Support/resistance levels (optional)

        Returns:
            Dictionary containing engineered features organized by category
        """
        # Step 1: Validate initialization
        if not self._validate_initialization():
            return {}

        # Step 2: Log input data information
        self._log_input_data_info(price_data, volume_data, order_flow_data)

        # Step 3: Preprocess data
        preprocessed_data = await self._preprocess_all_data(
            price_data, volume_data, order_flow_data
        )

        if not preprocessed_data:
            return {}

        # Step 4: Extract features in parallel
        feature_tasks = self._create_feature_extraction_tasks(
            preprocessed_data, sr_levels
        )

        # Step 5: Execute feature extraction
        features = await self._execute_feature_extraction(feature_tasks)

        # Step 6: Post-process features
        final_features = await self._post_process_features(features)

        # Step 7: Validate and return
        return self._validate_and_return_features(final_features)

    def _validate_initialization(self) -> bool:
        """Validate that the system is properly initialized."""
        if not self.is_initialized:
            self.logger.error("🚨 Feature engineering not initialized")
            return False
        return True

    def _log_input_data_info(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: Optional[pd.DataFrame],
    ) -> None:
        """Log information about input data."""
        self.logger.info(f"🔍 Price data shape: {price_data.shape}")
        self.logger.info(f"🔍 Volume data shape: {volume_data.shape}")

        if order_flow_data is not None:
            self.logger.info(f"🔍 Order flow data shape: {order_flow_data.shape}")
        else:
            self.logger.info("🔍 No order flow data provided")

    async def _preprocess_all_data(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: Optional[pd.DataFrame],
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Preprocess all input data."""
        try:
            # Preprocess price data
            price_result = await self._preprocess_price_data(price_data)

            # Preprocess volume data
            volume_result = await self._preprocess_volume_data(volume_data)

            # Preprocess order flow data if available
            order_flow_result = None
            if order_flow_data is not None:
                order_flow_result = await self._preprocess_order_flow_data(
                    order_flow_data
                )

            return {
                "price": price_result.data,
                "volume": volume_result.data,
                "order_flow": order_flow_result.data if order_flow_result else None,
            }

        except Exception as e:
            self.logger.error(f"❌ Preprocessing failed: {e}")
            return None

    async def _preprocess_price_data(self, data: pd.DataFrame) -> PreprocessingResult:
        """Preprocess price data with proper datetime handling."""
        original_shape = data.shape

        # Ensure datetime index
        processed_data = self._ensure_datetime_index(data, "price")

        # Handle irregular intervals
        processed_data = self._handle_irregular_intervals(processed_data)

        # Fill missing values
        processed_data = self._fill_missing_values(processed_data, "price")

        quality_improvement = self._calculate_quality_improvement(data, processed_data)

        return PreprocessingResult(
            data=processed_data,
            original_shape=original_shape,
            preprocessed_shape=processed_data.shape,
            quality_improvement=quality_improvement,
            method="enhanced",
        )

    async def _preprocess_volume_data(self, data: pd.DataFrame) -> PreprocessingResult:
        """Preprocess volume data."""
        original_shape = data.shape

        # Ensure datetime index
        processed_data = self._ensure_datetime_index(data, "volume")

        # Handle irregular intervals
        processed_data = self._handle_irregular_intervals(processed_data)

        # Fill missing values
        processed_data = self._fill_missing_values(processed_data, "volume")

        quality_improvement = self._calculate_quality_improvement(data, processed_data)

        return PreprocessingResult(
            data=processed_data,
            original_shape=original_shape,
            preprocessed_shape=processed_data.shape,
            quality_improvement=quality_improvement,
            method="enhanced",
        )

    async def _preprocess_order_flow_data(
        self, data: pd.DataFrame
    ) -> PreprocessingResult:
        """Preprocess order flow data."""
        # Similar to price/volume preprocessing
        return await self._preprocess_price_data(data)

    def _ensure_datetime_index(
        self, data: pd.DataFrame, data_type: str
    ) -> pd.DataFrame:
        """Ensure data has a proper datetime index."""
        if isinstance(data.index, pd.DatetimeIndex):
            return data

        self.logger.warning(f"⚠️ {data_type} data doesn't have DatetimeIndex, fixing...")

        # Try various methods to create datetime index
        if "timestamp" in data.columns:
            data = data.set_index("timestamp")
            data.index = pd.to_datetime(data.index)
        elif data.index.name == "timestamp":
            data.index = pd.to_datetime(data.index)
        else:
            # Create synthetic datetime index as last resort
            self.logger.warning(f"⚠️ Creating synthetic datetime index for {data_type}")
            start_time = pd.Timestamp("2024-01-01 00:00:00")
            interval = pd.Timedelta(minutes=1)
            timestamps = [start_time + i * interval for i in range(len(data))]
            data.index = timestamps

        return data

    def _handle_irregular_intervals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle irregular time intervals in data."""
        # Placeholder for actual implementation
        return data

    def _fill_missing_values(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Fill missing values based on data type."""
        if data_type == "price":
            # Forward fill for price data
            return data.fillna(method="ffill").fillna(method="bfill")
        elif data_type == "volume":
            # Fill volume with zeros
            return data.fillna(0)
        else:
            return data.fillna(method="ffill")

    def _calculate_quality_improvement(
        self, original: pd.DataFrame, processed: pd.DataFrame
    ) -> float:
        """Calculate quality improvement metric."""
        original_missing = original.isna().sum().sum()
        processed_missing = processed.isna().sum().sum()

        if original_missing == 0:
            return 0.0

        return (original_missing - processed_missing) / original_missing

    def _create_feature_extraction_tasks(
        self,
        preprocessed_data: Dict[str, pd.DataFrame],
        sr_levels: Optional[Dict[str, Any]],
    ) -> List[asyncio.Task]:
        """Create async tasks for parallel feature extraction."""
        tasks = []

        if self.feature_config.enable_technical:
            tasks.append(
                asyncio.create_task(self._extract_technical_features(preprocessed_data))
            )

        if self.feature_config.enable_microstructure:
            tasks.append(
                asyncio.create_task(
                    self._extract_microstructure_features(preprocessed_data)
                )
            )

        if self.feature_config.enable_wavelet:
            tasks.append(
                asyncio.create_task(self._extract_wavelet_features(preprocessed_data))
            )

        if self.feature_config.enable_regime and sr_levels:
            tasks.append(
                asyncio.create_task(
                    self._extract_regime_features(preprocessed_data, sr_levels)
                )
            )

        return tasks

    async def _execute_feature_extraction(
        self, tasks: List[asyncio.Task]
    ) -> Dict[str, pd.DataFrame]:
        """Execute feature extraction tasks in parallel."""
        results = await asyncio.gather(*tasks, return_exceptions=True)

        features = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"❌ Feature extraction task {i} failed: {result}")
            elif isinstance(result, dict):
                features.update(result)

        return features

    async def _extract_technical_features(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Extract technical indicator features."""
        price_data = data["price"]
        volume_data = data["volume"]

        features = {}

        # Moving averages
        features["ma"] = self._calculate_moving_averages(price_data)

        # RSI
        features["rsi"] = self._calculate_rsi(price_data)

        # MACD
        features["macd"] = self._calculate_macd(price_data)

        # Bollinger Bands
        features["bb"] = self._calculate_bollinger_bands(price_data)

        # Volume indicators
        features["volume_indicators"] = self._calculate_volume_indicators(
            price_data, volume_data
        )

        return {"technical": pd.concat(features.values(), axis=1)}

    async def _extract_microstructure_features(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Extract market microstructure features."""
        # Placeholder for actual implementation
        return {"microstructure": pd.DataFrame()}

    async def _extract_wavelet_features(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Extract wavelet transform features."""
        # Placeholder for actual implementation
        return {"wavelet": pd.DataFrame()}

    async def _extract_regime_features(
        self, data: Dict[str, pd.DataFrame], sr_levels: Dict[str, Any]
    ) -> Dict[str, pd.DataFrame]:
        """Extract regime-based features."""
        # Placeholder for actual implementation
        return {"regime": pd.DataFrame()}

    async def _post_process_features(
        self, features: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Post-process extracted features."""
        # Remove NaN values
        for category, feature_df in features.items():
            features[category] = self._handle_nan_values(feature_df, category)

        # Scale features if needed
        if self.config.get("scale_features", False):
            features = self._scale_features(features)

        # Add interaction features if enabled
        if self.feature_config.enable_interaction:
            interaction_features = await self._create_interaction_features(features)
            features["interaction"] = interaction_features

        return features

    def _handle_nan_values(self, df: pd.DataFrame, category: str) -> pd.DataFrame:
        """Handle NaN values in features."""
        # Strategy depends on feature category
        if category in ["technical", "microstructure"]:
            # Forward fill then backward fill
            return df.fillna(method="ffill").fillna(method="bfill")
        else:
            # Fill with zeros for other categories
            return df.fillna(0)

    def _scale_features(
        self, features: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Scale features to standard range."""
        # Placeholder for actual scaling implementation
        return features

    async def _create_interaction_features(
        self, features: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Create interaction features between different feature categories."""
        # Placeholder for actual implementation
        return pd.DataFrame()

    def _validate_and_return_features(
        self, features: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Validate and prepare final feature output."""
        # Validate each feature category
        validated_features = {}

        for category, feature_df in features.items():
            if self._validate_feature_category(feature_df, category):
                validated_features[category] = feature_df
            else:
                self.logger.warning(f"⚠️ Validation failed for {category} features")

        # Add metadata
        metadata = self._create_feature_metadata(validated_features)

        return {"features": validated_features, "metadata": metadata}

    def _validate_feature_category(self, df: pd.DataFrame, category: str) -> bool:
        """Validate a feature category."""
        if df.empty:
            return False

        # Check for excessive NaN values
        nan_ratio = df.isna().sum().sum() / (df.shape[0] * df.shape[1])
        if nan_ratio > 0.5:
            self.logger.warning(f"⚠️ {category} has {nan_ratio:.2%} NaN values")
            return False

        return True

    def _create_feature_metadata(
        self, features: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Create metadata for the generated features."""
        metadata = {
            "total_features": sum(df.shape[1] for df in features.values()),
            "categories": list(features.keys()),
            "feature_counts": {
                category: df.shape[1] for category, df in features.items()
            },
            "timestamp": pd.Timestamp.now(),
        }

        return metadata

    # Technical indicator calculation methods
    def _calculate_moving_averages(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate various moving averages."""
        ma_features = pd.DataFrame(index=price_data.index)

        for period in [5, 10, 20, 50, 100, 200]:
            if len(price_data) >= period:
                ma_features[f"ma_{period}"] = price_data["close"].rolling(period).mean()
                ma_features[f"ema_{period}"] = (
                    price_data["close"].ewm(span=period).mean()
                )

        return ma_features

    def _calculate_rsi(
        self, price_data: pd.DataFrame, period: int = 14
    ) -> pd.DataFrame:
        """Calculate Relative Strength Index."""
        delta = price_data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return pd.DataFrame({"rsi": rsi}, index=price_data.index)

    def _calculate_macd(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicators."""
        exp1 = price_data["close"].ewm(span=12, adjust=False).mean()
        exp2 = price_data["close"].ewm(span=26, adjust=False).mean()

        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal

        return pd.DataFrame(
            {"macd": macd, "macd_signal": signal, "macd_histogram": histogram},
            index=price_data.index,
        )

    def _calculate_bollinger_bands(
        self, price_data: pd.DataFrame, period: int = 20, std_dev: int = 2
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        sma = price_data["close"].rolling(window=period).mean()
        std = price_data["close"].rolling(window=period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        return pd.DataFrame(
            {
                "bb_upper": upper_band,
                "bb_middle": sma,
                "bb_lower": lower_band,
                "bb_width": upper_band - lower_band,
                "bb_percent": (price_data["close"] - lower_band)
                / (upper_band - lower_band),
            },
            index=price_data.index,
        )

    def _calculate_volume_indicators(
        self, price_data: pd.DataFrame, volume_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate volume-based indicators."""
        volume_features = pd.DataFrame(index=price_data.index)

        # On-Balance Volume (OBV)
        obv = (
            (np.sign(price_data["close"].diff()) * volume_data["volume"])
            .fillna(0)
            .cumsum()
        )
        volume_features["obv"] = obv

        # Volume Moving Average
        volume_features["volume_ma"] = volume_data["volume"].rolling(window=20).mean()

        # Volume Rate of Change
        volume_features["volume_roc"] = volume_data["volume"].pct_change(periods=10)

        return volume_features
