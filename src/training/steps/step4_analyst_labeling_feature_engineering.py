# src/training/steps/step4_analyst_labeling_feature_engineering.py

import asyncio
import json
import os
import pickle
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.logger import system_logger


class DeprecatedAnalystLabelingFeatureEngineeringStep:
    """Step 4: Analyst Labeling and Feature Engineering (DEPRECATED)."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("DeprecatedAnalystLabelingFeatureEngineeringStep")

    async def initialize(self) -> None:
        """Initialize the analyst labeling and feature engineering step."""
        self.logger.info("Initializing Deprecated Analyst Labeling and Feature Engineering Step...")
        self.logger.info("Deprecated Analyst Labeling and Feature Engineering Step initialized successfully")

    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute analyst labeling and feature engineering step.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            dict: Updated pipeline state
        """
        self.logger.info("Executing deprecated analyst labeling and feature engineering step...")
        
        # This step is deprecated - return current state
        return pipeline_state

    async def _apply_triple_barrier_labeling(
        self,
        data: pd.DataFrame,
        regime_name: str,
    ) -> pd.DataFrame:
        """
        Apply Triple Barrier Method for labeling.

        Args:
            data: Market data for the regime
            regime_name: Name of the regime

        Returns:
            DataFrame with labels added
        """
        self.logger.info(
            f"Applying Triple Barrier Method for regime: {regime_name}",
        )

        # Ensure we have required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Calculate features for labeling
        data = self._calculate_features(data)

        # Apply Triple Barrier Method
        labeled_data = data.copy()

        # Define barrier parameters based on regime
        profit_take_multiplier = 0.002  # 0.2%
        stop_loss_multiplier = 0.001  # 0.1%
        time_barrier_minutes = 30  # 30 minutes
        
        # Apply triple barrier labeling
        labels = []
        for i in range(len(data)):
            if i >= len(data) - 1:  # Skip last point
                labels.append(0)
                continue

            entry_price = data.iloc[i]["close"]
            entry_time = data.index[i]

            # Calculate barriers
            profit_take_barrier = entry_price * (1 + profit_take_multiplier)
            stop_loss_barrier = entry_price * (1 - stop_loss_multiplier)
            time_barrier = entry_time + timedelta(minutes=time_barrier_minutes)

            # Check if any barrier is hit
            label = 0  # Neutral

            for j in range(
                i + 1,
                min(i + 100, len(data)),
            ):  # Look ahead up to 100 points
                current_time = data.index[j]
                current_price = data.iloc[j]["high"]  # Use high for profit take
                current_low = data.iloc[j]["low"]  # Use low for stop loss

                # Check time barrier
                if current_time > time_barrier:
                    label = 0  # Time barrier hit - neutral
                    break

                # Check profit take barrier
                if current_price >= profit_take_barrier:
                    label = 1  # Profit take hit - positive
                    break

                # Check stop loss barrier
                if current_low <= stop_loss_barrier:
                    label = -1  # Stop loss hit - negative
                    break

            labels.append(label)

        labeled_data["label"] = labels

        # Calculate label distribution
        label_counts = pd.Series(labels).value_counts()
        self.logger.info(
            f"Label distribution for {regime_name}: {dict(label_counts)}",
        )

        return labeled_data

    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical features for the data.

        Args:
            data: Market data

        Returns:
            DataFrame with features added
        """
        try:
            # Calculate RSI
            data["rsi"] = self._calculate_rsi(data["close"])

            # Calculate MACD
            macd, signal = self._calculate_macd(data["close"])
            data["macd"] = macd
            data["macd_signal"] = signal
            data["macd_histogram"] = macd - signal

            # Calculate Bollinger Bands
            bb_upper, bb_lower = self._calculate_bollinger_bands(data["close"])
            data["bb_upper"] = bb_upper
            data["bb_lower"] = bb_lower
            data["bb_width"] = bb_upper - bb_lower
            data["bb_position"] = (data["close"] - bb_lower) / (bb_upper - bb_lower)

            # Calculate ATR
            data["atr"] = self._calculate_atr(data)

            # Calculate price-based features
            data["price_change"] = data["close"].pct_change()
            data["price_change_abs"] = data["price_change"].abs()
            data["high_low_ratio"] = data["high"] / data["low"]
            data["volume_price_ratio"] = data["volume"] / data["close"]

            # Calculate moving averages
            data["sma_5"] = data["close"].rolling(window=5).mean()
            data["sma_10"] = data["close"].rolling(window=10).mean()
            data["sma_20"] = data["close"].rolling(window=20).mean()
            data["ema_12"] = data["close"].ewm(span=12).mean()
            data["ema_26"] = data["close"].ewm(span=26).mean()

            # Calculate momentum features
            data["momentum_5"] = data["close"] / data["close"].shift(5) - 1
            data["momentum_10"] = data["close"] / data["close"].shift(10) - 1
            data["momentum_20"] = data["close"] / data["close"].shift(20) - 1

            # Add candlestick pattern features using advanced feature engineering
            # Legacy S/R/Candle code removed - using simplified approach
            data = data  # Keep original data for now

            # Fill NaN values
            data = data.fillna(method="bfill").fillna(0)

            return data

        except Exception as e:
            self.logger.error(f"Error calculating features: {e}")
            raise

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line

    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: int = 2,
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data["high"] - data["low"]
        high_close = np.abs(data["high"] - data["close"].shift())
        low_close = np.abs(data["low"] - data["close"].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr

    async def _add_candlestick_pattern_features(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add candlestick pattern features using advanced feature engineering.

        Args:
            data: Market data with OHLCV

        Returns:
            DataFrame with candlestick pattern features added
        """
        try:
            self.logger.info("Adding candlestick pattern features...")

            # Import advanced feature engineering
            from src.analyst.advanced_feature_engineering import (
                AdvancedFeatureEngineering,
            )

            # Initialize advanced feature engineering
            config = {
                "advanced_features": {
                    # Legacy S/R/Candle code removed,
                    "enable_volatility_regime_modeling": True,
                    "enable_correlation_analysis": True,
                    "enable_momentum_analysis": True,
                    "enable_liquidity_analysis": True,
                },
                # Legacy S/R/Candle code removed
            }

            # Initialize advanced feature engineering
            advanced_fe = AdvancedFeatureEngineering(config)
            await advanced_fe.initialize()

            # Prepare data for feature engineering
            price_data = data[["open", "high", "low", "close"]].copy()
            volume_data = data[["volume"]].copy()

            # Get advanced features including candlestick patterns
            advanced_features = await advanced_fe.engineer_features(
                price_data=price_data,
                volume_data=volume_data,
                order_flow_data=None,
            )

            # Convert features to DataFrame and align with original data
            if advanced_features:
                features_df = pd.DataFrame([advanced_features])
                # Replicate features for all rows
                features_df = pd.concat([features_df] * len(data), ignore_index=True)
                features_df.index = data.index

                # Add candlestick pattern features to original data
                for col in features_df.columns:
                    if col not in data.columns:  # Avoid overwriting existing columns
                        data[col] = features_df[col]

                self.logger.info(
                    f"Added {len(features_df.columns)} candlestick pattern features",
                )
            else:
                self.logger.warning("No candlestick pattern features generated")

            return data

        except Exception as e:
            self.logger.error(f"Error adding candlestick pattern features: {e}")
            return data

    async def _perform_feature_engineering(
        self,
        data: pd.DataFrame,
        regime_name: str,
    ) -> pd.DataFrame:
        """
        Perform feature engineering for the regime.

        Args:
            data: Labeled data with features
            regime_name: Name of the regime

        Returns:
            DataFrame with engineered features
        """
        try:
            self.logger.info(
                f"Performing feature engineering for regime: {regime_name}",
            )

            # Select feature columns including candlestick patterns
            feature_columns = [
                "rsi",
                "macd",
                "macd_signal",
                "macd_histogram",
                "bb_upper",
                "bb_lower",
                "bb_width",
                "bb_position",
                "atr",
                "price_change",
                "price_change_abs",
                "high_low_ratio",
                "volume_price_ratio",
                "sma_5",
                "sma_10",
                "sma_20",
                "ema_12",
                "ema_26",
                "momentum_5",
                "momentum_10",
                "momentum_20",
            ]

            # Add candlestick pattern features
            # Legacy S/R/Candle code removed

            # Combine all feature columns
            # Legacy S/R/Candle code removed

            # Filter available features
            available_features = [col for col in feature_columns if col in data.columns]

            if not available_features:
                raise ValueError(f"No features available for regime {regime_name}")

            # Create feature matrix
            feature_data = data[available_features].copy()

            # Remove rows with NaN values
            feature_data = feature_data.dropna()

            # Standardize features
            scaler = StandardScaler()
            feature_data_scaled = scaler.fit_transform(feature_data)
            feature_data_scaled = pd.DataFrame(
                feature_data_scaled,
                columns=available_features,
                index=feature_data.index,
            )

            # Add label column
            feature_data_scaled["label"] = data.loc[feature_data.index, "label"]

            self.logger.info(
                f"Feature engineering completed for {regime_name}: {len(feature_data_scaled)} samples, {len(available_features)} features",
            )

            return feature_data_scaled

        except Exception as e:
            self.logger.error(f"Error in feature engineering for {regime_name}: {e}")
            raise

    async def _train_regime_encoders(
        self,
        data: pd.DataFrame,
        regime_name: str,
    ) -> dict[str, Any]:
        """
        Train regime-specific encoders.

        Args:
            data: Feature-engineered data
            regime_name: Name of the regime

        Returns:
            Dict containing trained encoders
        """
        try:
            self.logger.info(f"Training encoders for regime: {regime_name}")

            # Separate features and labels
            feature_columns = [col for col in data.columns if col != "label"]
            X = data[feature_columns]
            y = data["label"]

            # Train PCA encoder for dimensionality reduction
            pca = PCA(n_components=min(10, len(feature_columns)))
            X_pca = pca.fit_transform(X)

            # Train autoencoder for feature learning
            from sklearn.neural_network import MLPRegressor

            autoencoder = MLPRegressor(
                hidden_layer_sizes=(
                    len(feature_columns),
                    len(feature_columns) // 2,
                    len(feature_columns),
                ),
                max_iter=1000,
                random_state=42,
            )
            autoencoder.fit(X, X)

            # Store encoders
            encoders = {
                "pca": pca,
                "autoencoder": autoencoder,
                "feature_columns": feature_columns,
                "n_features": len(feature_columns),
                "n_samples": len(X),
            }

            self.logger.info(
                f"Encoders trained for {regime_name}: PCA components={pca.n_components_}, Autoencoder layers={autoencoder.n_layers_}",
            )

            return encoders

        except Exception as e:
            self.logger.error(f"Error training encoders for {regime_name}: {e}")
            raise


# For backward compatibility with existing step structure
async def deprecated_run_step(
    symbol: str,
    exchange_name: str = "BINANCE",
    data_dir: str = "data/training",
) -> bool:
    """
    DEPRECATED: Run analyst labeling and feature engineering step.
    
    This function is deprecated and should not be used in new training pipelines.
    """
    print("⚠️  WARNING: This step is deprecated and should not be used in new training pipelines.")
    return True


if __name__ == "__main__":
    # Test the step
    async def test():
        """Test the deprecated analyst labeling and feature engineering step."""
        result = await deprecated_run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Deprecated step test result: {result}")

    asyncio.run(test())
