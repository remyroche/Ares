# src/training/steps/step3_regime_data_splitting.py

import asyncio
import json
import os
import pickle
from datetime import datetime
from typing import Any

import pandas as pd

from src.utils.logger import system_logger


class RegimeDataSplittingStep:
    """Step 3: Regime Data Splitting - Separate data by market regimes."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger

    async def initialize(self) -> None:
        """Initialize the regime data splitting step."""
        try:
            self.logger.info("Initializing Regime Data Splitting Step...")
            self.logger.info("Regime Data Splitting Step initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing Regime Data Splitting Step: {e}")
            raise

    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute regime data splitting.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Dict containing regime data splitting results
        """
        try:
            self.logger.info("🔄 Executing Regime Data Splitting...")

            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")

            # Load the historical data
            data_file_path = f"{data_dir}/{exchange}_{symbol}_historical_data.pkl"

            if not os.path.exists(data_file_path):
                raise FileNotFoundError(f"Data file not found: {data_file_path}")

            # Load data
            with open(data_file_path, "rb") as f:
                historical_data = pickle.load(f)

            # Convert to DataFrame if needed
            if not isinstance(historical_data, pd.DataFrame):
                historical_data = pd.DataFrame(historical_data)

            # Load regime classification results
            regime_file_path = (
                f"{data_dir}/{exchange}_{symbol}_regime_classification.json"
            )

            if not os.path.exists(regime_file_path):
                raise FileNotFoundError(
                    f"Regime classification file not found: {regime_file_path}",
                )

            with open(regime_file_path) as f:
                regime_results = json.load(f)

            # Split data by regimes
            regime_data = await self._split_data_by_regimes(
                historical_data,
                regime_results,
                symbol,
                exchange,
            )

            # Save regime-specific data
            regime_data_dir = f"{data_dir}/regime_data"
            os.makedirs(regime_data_dir, exist_ok=True)

            regime_data_paths = {}
            for regime, data in regime_data.items():
                regime_file = f"{regime_data_dir}/{exchange}_{symbol}_{regime}_data.pkl"
                with open(regime_file, "wb") as f:
                    pickle.dump(data, f)
                regime_data_paths[regime] = regime_file

            # Save regime splitting summary
            splitting_summary = {
                "symbol": symbol,
                "exchange": exchange,
                "splitting_date": datetime.now().isoformat(),
                "total_records": len(historical_data),
                "regime_data_paths": regime_data_paths,
                "regime_statistics": {},
                "data_splitting_config": {
                    "split_ratio": {"model_generation": 0.85, "validation": 0.15},
                    "time_series_split": True,
                    "maintain_chronological_order": True,
                },
            }

            # Calculate statistics for each regime
            for regime, data in regime_data.items():
                splitting_summary["regime_statistics"][regime] = {
                    "record_count": len(data),
                    "percentage": len(data) / len(historical_data) * 100,
                    "date_range": {
                        "start": data.index.min().isoformat()
                        if hasattr(data.index, "min")
                        else None,
                        "end": data.index.max().isoformat()
                        if hasattr(data.index, "max")
                        else None,
                    },
                }

            # Save splitting summary
            summary_file = (
                f"{data_dir}/{exchange}_{symbol}_regime_splitting_summary.json"
            )
            with open(summary_file, "w") as f:
                json.dump(splitting_summary, f, indent=2)

            self.logger.info(
                f"✅ Regime data splitting completed. Results saved to {regime_data_dir}",
            )

            # Update pipeline state
            pipeline_state["regime_data"] = regime_data
            pipeline_state["regime_data_paths"] = regime_data_paths
            pipeline_state["splitting_summary"] = splitting_summary

            return {
                "regime_data": regime_data,
                "regime_data_paths": regime_data_paths,
                "splitting_summary": splitting_summary,
                "duration": 0.0,  # Will be calculated in actual implementation
                "status": "SUCCESS",
            }

        except Exception as e:
            self.logger.error(f"❌ Error in Regime Data Splitting: {e}")
            return {"status": "FAILED", "error": str(e), "duration": 0.0}

    async def _split_data_by_regimes(
        self,
        data: pd.DataFrame,
        regime_results: dict[str, Any],
        symbol: str,
        exchange: str,
    ) -> dict[str, pd.DataFrame]:
        """
        Split data by market regimes.

        Args:
            data: Historical market data
            regime_results: Regime classification results
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Dict mapping regime names to their respective data
        """
        try:
            self.logger.info(f"Splitting data by regimes for {symbol} on {exchange}...")

            # Extract regime sequence from results
            regime_sequence = regime_results.get("regime_sequence", [])

            if not regime_sequence:
                raise ValueError(
                    "No regime sequence found in regime classification results",
                )

            # Ensure data and regime sequence have the same length
            if len(data) != len(regime_sequence):
                self.logger.warning(
                    f"Data length ({len(data)}) doesn't match regime sequence length ({len(regime_sequence)})",
                )
                # Truncate to the shorter length
                min_length = min(len(data), len(regime_sequence))
                data = data.iloc[:min_length]
                regime_sequence = regime_sequence[:min_length]

            # Create regime data dictionary
            regime_data = {}

            # Group data by regime
            for i, regime in enumerate(regime_sequence):
                if regime not in regime_data:
                    regime_data[regime] = []

                # Get the corresponding data row
                if i < len(data):
                    row_data = data.iloc[i].copy()
                    # Add regime information to the row
                    row_data["regime"] = regime
                    regime_data[regime].append(row_data)

            # Convert lists to DataFrames
            for regime in regime_data:
                regime_data[regime] = pd.DataFrame(regime_data[regime])

                # Set index if timestamp column exists
                if "timestamp" in regime_data[regime].columns:
                    regime_data[regime]["timestamp"] = pd.to_datetime(
                        regime_data[regime]["timestamp"],
                    )
                    regime_data[regime] = regime_data[regime].set_index("timestamp")

                self.logger.info(
                    f"Regime '{regime}': {len(regime_data[regime])} records",
                )

            # Handle high-data vs low-data regimes
            high_data_regimes = ["BULL", "BEAR", "SIDEWAYS"]
            low_data_regimes = [
                "SUPPORT_RESISTANCE",
                "CANDLES",
                "HUGE_CANDLE",
                "SR_ZONE_ACTION",
            ]

            # Categorize regimes
            regime_categories = {"high_data": {}, "low_data": {}, "other": {}}

            for regime, regime_df in regime_data.items():
                if regime.upper() in high_data_regimes:
                    regime_categories["high_data"][regime] = regime_df
                elif regime.upper() in low_data_regimes:
                    regime_categories["low_data"][regime] = regime_df
                else:
                    regime_categories["other"][regime] = regime_df

            # Log regime categorization
            self.logger.info(
                f"High-data regimes: {list(regime_categories['high_data'].keys())}",
            )
            self.logger.info(
                f"Low-data regimes: {list(regime_categories['low_data'].keys())}",
            )
            self.logger.info(
                f"Other regimes: {list(regime_categories['other'].keys())}",
            )

            return regime_data

        except Exception as e:
            self.logger.error(f"Error in regime data splitting: {e}")
            raise


# For backward compatibility with existing step structure
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    **kwargs,
) -> bool:
    """
    Run the regime data splitting step.

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
        step = RegimeDataSplittingStep(config)
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
        print(f"❌ Regime data splitting failed: {e}")
        return False


if __name__ == "__main__":
    # Test the step
    async def test():
        result = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Test result: {result}")

    asyncio.run(test())
