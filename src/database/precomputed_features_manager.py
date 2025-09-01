# src/database/precomputed_features_manager.py


from datetime import datetime
from typing import Any, Iterable
import json

import pandas as pd

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.utils.warning_symbols import error, failed, warning

try:
    from src.database.influxdb_manager import InfluxDBManager
    INFLUXDB_AVAILABLE = True
except Exception:
    InfluxDBManager = None  # type: ignore
    INFLUXDB_AVAILABLE = False


class PrecomputedFeaturesManager:
    """
    Manages precomputed features with standardized naming convention and database storage.

    Feature naming convention: {category}_{timeframe}_{name}
    Categories: candle, volatility, volume, momentum, technical, price, time,
                ml_enhanced, triple_barrier, autoencoder
    Timeframes: 1m, 5m, 15m, 30m

    Examples:
    - candle_1m_doji_present
    - volatility_5m_atr
    - momentum_15m_rsi
    - price_30m_change_pct
    - triple_barrier_1m_profit_take_hit
    - autoencoder_5m_reconstruction_error
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("PrecomputedFeaturesManager")

        # Initialize database manager (optional)
        if INFLUXDB_AVAILABLE:
            self.db_manager: InfluxDBManager | None = InfluxDBManager()
        else:
            self.db_manager = None
            self.logger.warning(
                "InfluxDB not available - features will be stored locally only",
            )

        # Feature categories with their descriptions
        self.feature_categories: dict[str, str] = {
            "candle": "Candlestick patterns and formations",
            "volatility": "Volatility-based indicators and regimes",
            "volume": "Volume-based analysis and flow",
            "momentum": "Momentum and oscillator indicators",
            "technical": "Technical analysis indicators",
            "price": "Price-based features and changes",
            "time": "Time-based and cyclical features",
            "ml_enhanced": "Machine learning enhanced features",
            "triple_barrier": "Triple barrier labeling results",
            "autoencoder": "Autoencoder-generated features",
        }

        # Standard timeframes
        self.timeframes: list[str] = ["1m", "5m", "15m", "30m"]

        # Features that should use price differences
        self.price_difference_features: set[str] = {
            "price_change",
            "price_momentum",
            "gap_size",
            "dist_to_resistance",
            "dist_to_support",
            "price_return",
            "log_return",
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="precomputed features manager initialization",
    )
    def parse_feature_name(self, feature_name: str) -> tuple[str, str, str]:
        """
        Parse standardized feature name into components.

        Args:
            feature_name: Standardized feature name

        Returns:
            Tuple of (category, timeframe, name)
        """
        parts = feature_name.split("_", 2)
        if len(parts) != 3:
            msg = f"Invalid feature name format: {feature_name}"
            raise ValueError(msg)

        category, timeframe, name = parts

        if category not in self.feature_categories:
            msg = f"Invalid category in feature name: {category}"
            raise ValueError(msg)

        if timeframe not in self.timeframes:
            msg = f"Invalid timeframe in feature name: {timeframe}"
            raise ValueError(msg)

        return category, timeframe, name

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="feature storage",
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.DataFrame(),
        context="feature retrieval",
    )
    def _ensure_price_differences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure price-based features use differences rather than absolute values.

        Args:
            df: Features DataFrame

        Returns:
            DataFrame with price differences applied
        """
        df_copy = df.copy()

        for col in df_copy.columns:
            # Parse feature name to check if it's price-related
            try:
                category, timeframe, name = self.parse_feature_name(col)
            except ValueError:
                # Not a standardized feature name — skip
                continue

            # Convert absolute prices to differences for price category features
            if category == "price" and any(
                price_feat in name for price_feat in self.price_difference_features
            ):
                # Already difference-based
                continue

            if category == "price" and any(
                abs_feat in name for abs_feat in ["open", "high", "low", "close"]
            ):
                # Convert absolute prices to percentage changes
                if name.endswith(("_close", "_open")):
                    df_copy[col] = df_copy[col].pct_change()
                elif name.endswith(("_high", "_low")):
                    # For high/low, calculate relative to close
                    close_col = col.replace(name.split("_")[-1], "close")
                    if close_col in df_copy.columns:
                        df_copy[col] = (df_copy[col] - df_copy[close_col]) / df_copy[close_col]
                    else:
                        df_copy[col] = df_copy[col].pct_change()

        # Fill NaN values
        return df_copy.fillna(0)

    def _apply_feature_filters(
        self,
        df: pd.DataFrame,
        category_filter: str | None = None,
        timeframe_filter: str | None = None,
    ) -> pd.DataFrame:
        """Apply category and timeframe filters to the DataFrame."""
        filtered_columns: list[str] = []

        for col in df.columns:
            try:
                category, timeframe, name = self.parse_feature_name(col)
            except ValueError:
                # Include non-standardized columns
                filtered_columns.append(col)
                continue

            # Apply filters
            if category_filter and category != category_filter:
                continue
            if timeframe_filter and timeframe != timeframe_filter:
                continue

            filtered_columns.append(col)

        return df[filtered_columns]

    async def _create_feature_metadata_tables(self) -> None:
        """Create tables for storing feature metadata."""
        # This would create metadata storage in the database
        # For InfluxDB, we can store metadata as a separate measurement
        self.logger.info("Feature metadata storage configured")

    async def _store_feature_metadata(
        self,
        feature_names: list[str],
        symbol: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store metadata about the features."""
        if self.db_manager is None:
            return

        metadata_records: list[dict[str, Any]] = []

        for feature_name in feature_names:
            try:
                category, timeframe, name = self.parse_feature_name(feature_name)
            except ValueError:
                # Skip non-standardized feature names
                continue

            record: dict[str, Any] = {
                "feature_name": feature_name,
                "category": category,
                "timeframe": timeframe,
                "name": name,
                "symbol": symbol,
                "created_at": datetime.now().isoformat(),
                "description": self.feature_categories.get(
                    category, "Unknown category",
                ),
            }

            if metadata:
                record.update(metadata)

            metadata_records.append(record)

        if not metadata_records:
            return

        # Store metadata as a separate measurement
        metadata_df = pd.DataFrame(metadata_records)
        metadata_df["timestamp"] = datetime.now()
        metadata_df = metadata_df.set_index("timestamp")

        self.db_manager.write_api.write(
            bucket=self.db_manager.bucket,
            record=metadata_df,
            data_frame_measurement_name="feature_metadata",
            data_frame_tag_columns=["symbol", "category", "timeframe"],
        )
