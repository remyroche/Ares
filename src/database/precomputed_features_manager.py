# src/database/precomputed_features_manager.py

import json
from datetime import datetime
from typing import Any

import pandas as pd

try:
    from src.database.influxdb_manager import InfluxDBManager

    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    InfluxDBManager = None
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    failed,
    warning,
)


class PrecomputedFeaturesManager:
    """
    Manages precomputed features with standardized naming convention and database storage.

    Feature naming convention: {category}_{timeframe}_{name}
    Categories: candle, volatility, volume, momentum, technical, price, time, ml_enhanced, triple_barrier, autoencoder
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
            self.db_manager = InfluxDBManager()
        else:
            self.db_manager = None
            self.logger.warning(
                "InfluxDB not available - features will be stored locally only",
            )

        # Feature categories with their descriptions
        self.feature_categories = {
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
        self.timeframes = ["1m", "5m", "15m", "30m"]

        # Features that should use price differences
        self.price_difference_features = {
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
    async def initialize(self) -> bool:
        """Initialize the precomputed features manager."""
        try:
            self.logger.info("🚀 Initializing PrecomputedFeaturesManager...")

            # Create feature metadata tables if needed
            await self._create_feature_metadata_tables()

            self.logger.info("✅ PrecomputedFeaturesManager initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Failed to initialize PrecomputedFeaturesManager: {e}",
            )
            return False

    def generate_feature_name(self, category: str, timeframe: str, name: str) -> str:
        """
        Generate standardized feature name.

        Args:
            category: Feature category (candle, volatility, etc.)
            timeframe: Timeframe (1m, 5m, 15m, 30m)
            name: Feature name

        Returns:
            Standardized feature name
        """
        if category not in self.feature_categories:
            msg = f"Invalid category: {category}. Valid categories: {list(self.feature_categories.keys())}"
            raise ValueError(
                msg,
            )

        if timeframe not in self.timeframes:
            msg = f"Invalid timeframe: {timeframe}. Valid timeframes: {self.timeframes}"
            raise ValueError(
                msg,
            )

        return f"{category}_{timeframe}_{name}"

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
        default_return=pd.DataFrame(),
        context="feature storage",
    )
    async def store_features(
        self,
        features_df: pd.DataFrame,
        symbol: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Store precomputed features in the database.

        Args:
            features_df: DataFrame with features using standardized naming
            symbol: Trading symbol
            metadata: Additional metadata about the features

        Returns:
            Success status
        """
        try:
            if features_df.empty:
                self.print(warning("Empty features DataFrame provided"))
                return False

            self.logger.info(
                f"Storing {len(features_df.columns)} features for {symbol}",
            )

            # Ensure price-based features use differences
            features_df = self._ensure_price_differences(features_df)

            # Add metadata columns
            features_df_copy = features_df.copy()
            features_df_copy["symbol"] = symbol
            features_df_copy["computation_timestamp"] = datetime.now().isoformat()

            if metadata:
                features_df_copy["metadata"] = json.dumps(metadata)

            # Store in InfluxDB
            self.db_manager.write_api.write(
                bucket=self.db_manager.bucket,
                record=features_df_copy,
                data_frame_measurement_name="precomputed_features",
                data_frame_tag_columns=["symbol"],
            )

            # Store feature metadata
            await self._store_feature_metadata(
                features_df.columns.tolist(),
                symbol,
                metadata,
            )

            self.logger.info(f"✅ Successfully stored features for {symbol}")
            return True

        except Exception:
            self.print(failed("❌ Failed to store features: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.DataFrame(),
        context="feature retrieval",
    )
    async def retrieve_features(
        self,
        symbol: str,
        feature_names: list[str] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        category_filter: str | None = None,
        timeframe_filter: str | None = None,
    ) -> pd.DataFrame:
        """
        Retrieve precomputed features from the database.

        Args:
            symbol: Trading symbol
            feature_names: Specific feature names to retrieve
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            category_filter: Filter by feature category
            timeframe_filter: Filter by timeframe

        Returns:
            DataFrame with requested features
        """
        try:
            # Build query based on filters
            query_filters = [f'r["symbol"] == "{symbol}"']

            if feature_names:
                field_filter = " or ".join(
                    [f'r["_field"] == "{name}"' for name in feature_names],
                )
                query_filters.append(f"({field_filter})")

            # Build time range
            time_range = ""
            if start_time and end_time:
                time_range = f"|> range(start: {start_time}, stop: {end_time})"
            elif start_time:
                time_range = f"|> range(start: {start_time})"
            elif end_time:
                time_range = f"|> range(stop: {end_time})"

            # Construct query
            query = f"""
            from(bucket: "{self.db_manager.bucket}")
              {time_range}
              |> filter(fn: (r) => r["_measurement"] == "precomputed_features")
              |> filter(fn: (r) => {" and ".join(query_filters)})
              |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            """

            df = self.db_manager.query_api.query_data_frame(
                query=query,
                org=self.db_manager.org,
            )

            if isinstance(df, list):
                if not df:
                    return pd.DataFrame()
                df = pd.concat(df, ignore_index=True)

            if df.empty:
                return pd.DataFrame()

            # Apply additional filters
            if category_filter or timeframe_filter:
                df = self._apply_feature_filters(df, category_filter, timeframe_filter)

            # Set timestamp as index
            if "_time" in df.columns:
                df["_time"] = pd.to_datetime(df["_time"])
                df = df.set_index("_time")

            self.logger.info(
                f"Retrieved {len(df)} rows with {len(df.columns)} features for {symbol}",
            )
            return df

        except Exception:
            self.print(failed("❌ Failed to retrieve features: {e}"))
            return pd.DataFrame()

    def _ensure_price_differences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure price-based features use differences rather than absolute values.

        Args:
            df: Features DataFrame

        Returns:
            DataFrame with price differences applied
        """
        try:
            df_copy = df.copy()

            for col in df_copy.columns:
                # Parse feature name to check if it's price-related
                try:
                    category, timeframe, name = self.parse_feature_name(col)

                    # Convert absolute prices to differences for price category features
                    if category == "price" and any(
                        price_feat in name
                        for price_feat in self.price_difference_features
                    ):
                        # These are already difference-based, keep as is
                        continue
                    if category == "price" and any(
                        abs_feat in name
                        for abs_feat in ["open", "high", "low", "close"]
                    ):
                        # Convert absolute prices to percentage changes
                        if name.endswith(("_close", "_open")):
                            df_copy[col] = df_copy[col].pct_change()
                        elif name.endswith(("_high", "_low")):
                            # For high/low, calculate relative to close
                            close_col = col.replace(name.split("_")[-1], "close")
                            if close_col in df_copy.columns:
                                df_copy[col] = (
                                    df_copy[col] - df_copy[close_col]
                                ) / df_copy[close_col]
                            else:
                                df_copy[col] = df_copy[col].pct_change()

                except ValueError:
                    # Not a standardized feature name, skip
                    continue

            # Fill NaN values
            return df_copy.fillna(0)

        except Exception:
            self.print(error("Error ensuring price differences: {e}"))
            return df

    def _apply_feature_filters(
        self,
        df: pd.DataFrame,
        category_filter: str | None,
        timeframe_filter: str | None,
    ) -> pd.DataFrame:
        """Apply category and timeframe filters to the DataFrame."""
        try:
            filtered_columns = []

            for col in df.columns:
                try:
                    category, timeframe, name = self.parse_feature_name(col)

                    # Apply filters
                    if category_filter and category != category_filter:
                        continue
                    if timeframe_filter and timeframe != timeframe_filter:
                        continue

                    filtered_columns.append(col)

                except ValueError:
                    # Include non-standardized columns
                    filtered_columns.append(col)

            return df[filtered_columns]

        except Exception:
            self.print(error("Error applying feature filters: {e}"))
            return df

    async def _create_feature_metadata_tables(self):
        """Create tables for storing feature metadata."""
        try:
            # This would create metadata storage in the database
            # For InfluxDB, we can store metadata as a separate measurement
            self.logger.info("Feature metadata storage configured")

        except Exception:
            self.print(error("Error creating feature metadata tables: {e}"))

    async def _store_feature_metadata(
        self,
        feature_names: list[str],
        symbol: str,
        metadata: dict[str, Any] | None,
    ):
        """Store metadata about the features."""
        try:
            metadata_records = []

            for feature_name in feature_names:
                try:
                    category, timeframe, name = self.parse_feature_name(feature_name)

                    record = {
                        "feature_name": feature_name,
                        "category": category,
                        "timeframe": timeframe,
                        "name": name,
                        "symbol": symbol,
                        "created_at": datetime.now().isoformat(),
                        "description": self.feature_categories.get(
                            category,
                            "Unknown category",
                        ),
                    }

                    if metadata:
                        record.update(metadata)

                    metadata_records.append(record)

                except ValueError:
                    # Skip non-standardized feature names
                    continue

            if metadata_records:
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

        except Exception:
            self.print(error("Error storing feature metadata: {e}"))

    def get_available_features(
        self,
        category: str | None = None,
        timeframe: str | None = None,
    ) -> list[str]:
        """
        Get list of available feature names based on filters.

        Args:
            category: Filter by category
            timeframe: Filter by timeframe

        Returns:
            List of available feature names
        """
        try:
            # This would query the metadata to get available features
            # For now, return example features based on the standardized naming

            categories = (
                [category] if category else list(self.feature_categories.keys())
            )
            timeframes = [timeframe] if timeframe else self.timeframes

            example_features = []

            for cat in categories:
                for tf in timeframes:
                    if cat == "candle":
                        example_features.extend(
                            [
                                f"{cat}_{tf}_doji_present",
                                f"{cat}_{tf}_hammer_present",
                                f"{cat}_{tf}_engulfing_bullish",
                            ],
                        )
                    elif cat == "volatility":
                        example_features.extend(
                            [
                                f"{cat}_{tf}_atr",
                                f"{cat}_{tf}_volatility_regime",
                                f"{cat}_{tf}_vol_ratio",
                            ],
                        )
                    elif cat == "momentum":
                        example_features.extend(
                            [
                                f"{cat}_{tf}_rsi",
                                f"{cat}_{tf}_macd_signal",
                                f"{cat}_{tf}_stoch_k",
                            ],
                        )
                    elif cat == "triple_barrier":
                        example_features.extend(
                            [
                                f"{cat}_{tf}_profit_take_hit",
                                f"{cat}_{tf}_stop_loss_hit",
                                f"{cat}_{tf}_time_barrier_hit",
                            ],
                        )
                    elif cat == "autoencoder":
                        example_features.extend(
                            [
                                f"{cat}_{tf}_reconstruction_error",
                                f"{cat}_{tf}_latent_feature_1",
                                f"{cat}_{tf}_latent_feature_2",
                            ],
                        )

            return example_features

        except Exception:
            self.print(error("Error getting available features: {e}"))
            return []

    def get_feature_statistics(self) -> dict[str, Any]:
        """Get statistics about stored features."""
        return {
            "categories": self.feature_categories,
            "timeframes": self.timeframes,
            "total_feature_types": len(self.feature_categories) * len(self.timeframes),
            "price_difference_features": list(self.price_difference_features),
        }
