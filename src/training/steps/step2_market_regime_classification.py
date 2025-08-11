# src/training/steps/step2_market_regime_classification.py

import asyncio
import json
import os
from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np

from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier
from src.utils.logger import system_logger
from src.training.steps.unified_data_loader import get_unified_data_loader


def convert_trade_data_to_ohlcv(
    trade_data: pd.DataFrame, timeframe: str = "1h"
) -> pd.DataFrame:
    """
    Convert trade data to OHLCV format.

    Args:
        trade_data: DataFrame with columns ['timestamp', 'price', 'quantity', 'is_buyer_maker', 'agg_trade_id']
        timeframe: Timeframe for resampling (e.g., '1h', '1m', '1d')

    Returns:
        DataFrame with OHLCV columns ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    """
    try:
        # Make a copy to avoid modifying original data
        df = trade_data.copy()

        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            # Check if timestamps are in milliseconds (large numbers)
            if df["timestamp"].iloc[0] > 1e12:  # Likely milliseconds since epoch
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Set timestamp as index for resampling
        df = df.set_index("timestamp")

        # Resample to the specified timeframe and calculate OHLCV
        ohlcv = df.resample(timeframe).agg(
            {"price": ["first", "max", "min", "last"], "quantity": "sum"}
        )

        # Flatten column names
        ohlcv.columns = ["open", "high", "low", "close", "volume"]

        # Reset index to create timestamp column
        ohlcv = ohlcv.reset_index()

        # Remove any rows with NaN values
        ohlcv = ohlcv.dropna()

        return ohlcv

    except Exception as e:
        system_logger.error(f"Error converting trade data to OHLCV: {e}")
        raise


class MarketRegimeClassificationStep:
    """Step 2: Market Regime Classification using UnifiedRegimeClassifier."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger
        self.regime_classifier = None

    async def initialize(self) -> None:
        """Initialize the market regime classification step."""
        self.logger.info("Initializing Market Regime Classification Step...")

        # Initialize the unified regime classifier (will be re-initialized with exchange/symbol in execute)
        self.regime_classifier = None

        self.logger.info(
            "Market Regime Classification Step initialized successfully",
        )

    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute market regime classification.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Dict containing regime classification results
        """
        self.logger.info("🔄 Executing Market Regime Classification...")

        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")
        timeframe = training_input.get("timeframe", "1m")

        # Initialize the unified regime classifier with exchange and symbol
        if self.regime_classifier is None:
            self.regime_classifier = UnifiedRegimeClassifier(
                self.config, exchange, symbol
            )
            await self.regime_classifier.initialize()

        # Use unified data loader to get data
        self.logger.info("🔄 Loading data using unified data loader...")
        data_loader = get_unified_data_loader(self.config)

        # Determine lookback period
        lookback_days = 180  # Default to 180 days for regime classification
        if os.environ.get("BLANK_TRAINING_MODE") == "1":
            lookback_days = 30  # Shorter period for blank mode

        # Load unified data with optimizations for ML training
        historical_data = await data_loader.load_unified_data(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            lookback_days=lookback_days,
            use_streaming=True,  # Enable streaming for large datasets
        )

        if historical_data is None or historical_data.empty:
            raise ValueError(f"No data found for {symbol} on {exchange}")

        # Log data information
        data_info = data_loader.get_data_info(historical_data)
        self.logger.info(f"✅ Loaded unified data: {data_info['rows']} rows")
        self.logger.info(
            f"   Date range: {data_info['date_range']['start']} to {data_info['date_range']['end']}"
        )
        self.logger.info(f"   Has aggtrades data: {data_info['has_aggtrades_data']}")
        self.logger.info(f"   Has futures data: {data_info['has_futures_data']}")

        # Ensure we have the required OHLCV columns
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_columns = [
            col for col in required_columns if col not in historical_data.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert to 1h timeframe if needed for regime classification
        if timeframe != "1h":
            self.logger.info(
                "🔄 Resampling data to 1h timeframe for regime classification..."
            )
            historical_data = self._resample_to_timeframe(historical_data, "1h")
            self.logger.info(f"✅ Resampled to 1h: {len(historical_data)} records")

        # Perform regime classification
        regime_results = await self._classify_market_regimes(
            historical_data,
            symbol,
            exchange,
        )

        # Save regime classification results
        regime_file_path = f"{data_dir}/{exchange}_{symbol}_regime_classification.json"
        os.makedirs(os.path.dirname(regime_file_path), exist_ok=True)

        with open(regime_file_path, "w") as f:
            json.dump(regime_results, f, indent=2)

        # Also save in parquet format with expected columns for validator
        parquet_file_path = (
            f"{data_dir}/{exchange}_{symbol}_regime_classification.parquet"
        )

        # Create DataFrame with expected columns
        if (
            "regime_sequence" in regime_results
            and "confidence_scores" in regime_results
        ):
            # Use timestamps from original data
            timestamps = (
                historical_data["timestamp"].tolist()
                if "timestamp" in historical_data.columns
                else list(range(len(regime_results["regime_sequence"])))
            )

            # Ensure all sequences have the same length
            min_length = min(
                len(timestamps),
                len(regime_results["regime_sequence"]),
                len(regime_results["confidence_scores"]),
            )

            parquet_df = pd.DataFrame(
                {
                    "timestamp": timestamps[:min_length],
                    "regime": regime_results["regime_sequence"][:min_length],
                    "confidence": regime_results["confidence_scores"][:min_length],
                }
            )

            # Save to parquet
            parquet_df.to_parquet(parquet_file_path, index=False)
            self.logger.info(
                f"✅ Saved regime classification results to parquet: {parquet_file_path}"
            )

        self.logger.info(
            f"✅ Market regime classification completed. Results saved to {regime_file_path}",
        )

        # Update pipeline state
        pipeline_state["regime_classification"] = regime_results
        pipeline_state["regime_file_path"] = regime_file_path

        return {
            "regime_classification": regime_results,
            "regime_file_path": regime_file_path,
            "duration": 0.0,  # Will be calculated in actual implementation
            "status": "SUCCESS",
        }

    def _resample_to_timeframe(
        self, df: pd.DataFrame, target_timeframe: str
    ) -> pd.DataFrame:
        """
        Resample data to a different timeframe.

        Args:
            df: DataFrame with OHLCV data
            target_timeframe: Target timeframe (e.g., '1h', '1d')

        Returns:
            Resampled DataFrame
        """
        try:
            # Make a copy to avoid modifying original data
            df_copy = df.copy()

            # Ensure timestamp is datetime and set as index
            if not pd.api.types.is_datetime64_any_dtype(df_copy["timestamp"]):
                df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])

            df_copy = df_copy.set_index("timestamp")

            # Resample to target timeframe
            resampled = df_copy.resample(target_timeframe).agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )

            # Reset index to get timestamp column back
            resampled = resampled.reset_index()

            # Remove any rows with NaN values
            resampled = resampled.dropna()

            return resampled

        except Exception as e:
            self.logger.error(f"Error resampling data: {e}")
            raise

    async def _classify_market_regimes(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str,
    ) -> dict[str, Any]:
        """
        Classify market regimes using the unified regime classifier.

        Args:
            data: Historical market data
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Dict containing regime classification results
        """
        try:
            self.logger.info(
                f"Classifying market regimes for {symbol} on {exchange}...",
            )

            # Prepare data for regime classification
            # The unified regime classifier expects specific column names
            required_columns = ["open", "high", "low", "close", "volume", "timestamp"]

            # Rename columns if needed
            column_mapping = {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "Timestamp": "timestamp",
            }

            for old_col, new_col in column_mapping.items():
                if old_col in data.columns and new_col not in data.columns:
                    data = data.rename(columns={old_col: new_col})

            # Ensure we have the required columns
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]
            if missing_columns:
                raise ValueError(
                    f"Missing required columns for regime classification: {missing_columns}",
                )

            # Sort by timestamp
            if "timestamp" in data.columns:
                data = data.sort_values("timestamp")

            # Perform regime classification
            regime_results = await self.regime_classifier.classify_regimes(data)

            # Process and format results
            formatted_results = {
                "symbol": symbol,
                "exchange": exchange,
                "classification_date": datetime.now().isoformat(),
                "total_records": len(data),
                "regime_distribution": {},
                "regime_sequence": [],
                "regime_transitions": [],
                "confidence_scores": {},
                "metadata": {
                    "classifier_version": "unified_regime_classifier",
                    "classification_method": "HMM_ensemble",
                },
            }

            # Extract regime information from classifier results
            if isinstance(regime_results, dict):
                # If the classifier returns a dict with regime information
                if "regimes" in regime_results:
                    regimes = regime_results["regimes"]
                    confidence_scores = regime_results.get("confidence_scores", [])

                    if isinstance(regimes, list):
                        # The regime sequence from the classifier may be shorter due to feature calculation
                        # We need to map it back to the original data length
                        original_data_length = len(data)
                        regime_sequence_length = len(regimes)

                        if regime_sequence_length < original_data_length:
                            self.logger.info(
                                f"Regime sequence length ({regime_sequence_length}) is shorter than original data length ({original_data_length}). "
                                f"Mapping regimes to original data length..."
                            )

                            # DATA EXTENSION LOGIC:
                            # The regime classifier processes data in chunks and may return fewer predictions than input rows.
                            # This happens because:
                            # 1. Feature calculation requires lookback periods (e.g., 20-period moving averages)
                            # 2. Some initial rows are dropped due to NaN values from feature calculation
                            # 3. The classifier may process data in batches
                            #
                            # SOLUTION: We map the shorter regime sequence back to the full data length using interpolation.
                            # For each original data point, we find the closest regime prediction and use that.
                            # This ensures we have regime classifications for all timestamps in the original dataset.

                            # Create a regime sequence that matches the original data length
                            # We'll use interpolation to map the regimes back to the full data
                            if regime_sequence_length > 0:
                                # Create indices for the regime sequence
                                regime_indices = np.linspace(
                                    0,
                                    original_data_length - 1,
                                    regime_sequence_length,
                                    dtype=int,
                                )

                                # Create a full-length regime sequence
                                full_regime_sequence = []
                                full_confidence_sequence = []
                                for i in range(original_data_length):
                                    # Find the closest regime index
                                    closest_idx = np.argmin(np.abs(regime_indices - i))
                                    full_regime_sequence.append(regimes[closest_idx])
                                    # Use corresponding confidence score or default
                                    if closest_idx < len(confidence_scores):
                                        full_confidence_sequence.append(
                                            confidence_scores[closest_idx]
                                        )
                                    else:
                                        full_confidence_sequence.append(
                                            0.8
                                        )  # Default confidence

                                formatted_results["regime_sequence"] = (
                                    full_regime_sequence
                                )
                                formatted_results["confidence_scores"] = (
                                    full_confidence_sequence
                                )
                                self.logger.info(
                                    f"Mapped regime sequence to original data length: {len(full_regime_sequence)}"
                                )
                            else:
                                # Fallback: use default regime
                                formatted_results["regime_sequence"] = [
                                    "SIDEWAYS"
                                ] * original_data_length
                                formatted_results["confidence_scores"] = [
                                    0.5
                                ] * original_data_length
                                self.logger.warning(
                                    "No regimes available, using default SIDEWAYS regime"
                                )
                        else:
                            # Regime sequence is long enough, truncate if needed
                            formatted_results["regime_sequence"] = regimes[
                                :original_data_length
                            ]
                            formatted_results["confidence_scores"] = confidence_scores[
                                :original_data_length
                            ]

                        # Calculate regime distribution
                        from collections import Counter

                        regime_counts = Counter(formatted_results["regime_sequence"])
                        formatted_results["regime_distribution"] = dict(regime_counts)

                        # Calculate regime transitions
                        transitions = []
                        for i in range(1, len(formatted_results["regime_sequence"])):
                            if (
                                formatted_results["regime_sequence"][i]
                                != formatted_results["regime_sequence"][i - 1]
                            ):
                                transitions.append(
                                    {
                                        "from_regime": formatted_results[
                                            "regime_sequence"
                                        ][i - 1],
                                        "to_regime": formatted_results[
                                            "regime_sequence"
                                        ][i],
                                        "transition_index": i,
                                    },
                                )
                        formatted_results["regime_transitions"] = transitions

                # Extract confidence scores if available
                if "confidence_scores" in regime_results:
                    formatted_results["confidence_scores"] = regime_results[
                        "confidence_scores"
                    ]

            self.logger.info(
                f"Regime classification completed. Found {len(formatted_results['regime_distribution'])} distinct regimes",
            )

            return formatted_results

        except Exception as e:
            self.logger.error(f"Error in regime classification: {e}")
            raise


# For backward compatibility with existing step structure
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    **kwargs,
) -> bool:
    """
    Run the market regime classification step.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory path
        **kwargs: Additional parameters

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create step instance
        config = {"symbol": symbol, "exchange": exchange, "data_dir": data_dir}
        step = MarketRegimeClassificationStep(config)
        await step.initialize()

        # Execute step
        training_input = {
            "symbol": symbol,
            "exchange": exchange,
            "data_dir": data_dir,
            **kwargs,
        }

        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)

        return result.get("status") == "SUCCESS"

    except Exception as e:
        print(f"❌ Market regime classification failed: {e}")
        return False


if __name__ == "__main__":
    # Test the step
    async def test():
        result = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Test result: {result}")

    asyncio.run(test())
