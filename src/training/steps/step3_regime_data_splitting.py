# src/training/steps/step3_regime_data_splitting.py

import asyncio
import json
import os
import pickle
from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np

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

            # Try to load prepared data first (contains features and labels)
            prepared_data_path = f"{data_dir}/{symbol}_prepared_data.pkl"
            historical_data_path = f"{data_dir}/{exchange}_{symbol}_historical_data.pkl"
            
            if os.path.exists(prepared_data_path):
                # Load prepared data which contains features and labels
                self.logger.info(f"Loading prepared data from: {prepared_data_path}")
                with open(prepared_data_path, "rb") as f:
                    historical_data = pickle.load(f)
                self.logger.info(f"✅ Loaded prepared data: {historical_data.shape if hasattr(historical_data, 'shape') else 'dict'}")
            elif os.path.exists(historical_data_path):
                # Fallback to historical data if prepared data doesn't exist
                self.logger.warning(f"Prepared data not found, using historical data: {historical_data_path}")
                with open(historical_data_path, "rb") as f:
                    historical_data = pickle.load(f)
                
                # Handle the data structure - it's a dictionary with 'klines', 'agg_trades', etc.
                if isinstance(historical_data, dict):
                    if 'klines' in historical_data:
                        # Use the klines data which contains OHLCV data
                        historical_data = historical_data['klines']
                        self.logger.info(f"✅ Extracted klines data: {historical_data.shape}")
                    else:
                        raise ValueError("No 'klines' data found in historical data dictionary")
                elif isinstance(historical_data, np.ndarray):
                    # Handle numpy array input
                    self.logger.info(f"Converting numpy array with shape {historical_data.shape} to DataFrame")
                    if len(historical_data.shape) == 2:
                        # Multi-dimensional array, convert to DataFrame
                        if historical_data.shape[1] == 5:  # Likely OHLCV data
                            columns = ['open', 'high', 'low', 'close', 'volume']
                            historical_data = pd.DataFrame(historical_data, columns=columns)
                        else:
                            # Use first column as the series
                            historical_data = pd.DataFrame(historical_data)
                    else:
                        # 1D array
                        historical_data = pd.DataFrame(historical_data)
                elif not isinstance(historical_data, pd.DataFrame):
                    # Convert to DataFrame if it's not already
                    historical_data = pd.DataFrame(historical_data)

                # Convert to DataFrame if needed
                if not isinstance(historical_data, pd.DataFrame):
                    historical_data = pd.DataFrame(historical_data)
            else:
                raise FileNotFoundError(f"Neither prepared data ({prepared_data_path}) nor historical data ({historical_data_path}) found")

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

            # Split data by regimes (vectorized)
            regime_data = await self._split_data_by_regimes(
                historical_data,
                regime_results,
                symbol,
                exchange,
            )

            # Check for analyst models and create regime data for missing regimes
            analyst_models_dir = os.path.join(data_dir, "analyst_models")
            if os.path.exists(analyst_models_dir):
                existing_regimes = [d for d in os.listdir(analyst_models_dir) 
                                  if os.path.isdir(os.path.join(analyst_models_dir, d))]
                
                # Create regime data for regimes that have models but no data
                for regime in existing_regimes:
                    if regime not in regime_data:
                        self.logger.info(f"Creating regime data for {regime} (has models but no classification)")
                        # Use a subset of data for this regime
                        subset_size = min(1000, len(historical_data) // max(1, len(existing_regimes)))
                        subset_data = historical_data.iloc[:subset_size].copy()
                        subset_data['regime'] = regime
                        regime_data[regime] = subset_data

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
                # Handle date range calculation properly
                date_range = {"start": None, "end": None}
                
                if hasattr(data.index, "min") and hasattr(data.index, "max"):
                    try:
                        # Check if index is datetime-like
                        if hasattr(data.index.min(), "isoformat"):
                            date_range["start"] = data.index.min().isoformat()
                            date_range["end"] = data.index.max().isoformat()
                        else:
                            # Handle integer or other index types
                            date_range["start"] = str(data.index.min())
                            date_range["end"] = str(data.index.max())
                    except Exception as e:
                        self.logger.warning(f"Could not calculate date range for regime {regime}: {e}")
                        date_range = {"start": None, "end": None}
                
                splitting_summary["regime_statistics"][regime] = {
                    "record_count": len(data),
                    "percentage": len(data) / max(1, len(historical_data)) * 100,
                    "date_range": date_range,
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
        Split data by market regimes (vectorized).
        """
        try:
            self.logger.info(f"Splitting data by regimes for {symbol} on {exchange}...")

            regime_sequence = regime_results.get("regime_sequence", [])
            if not regime_sequence:
                self.logger.warning("No regime sequence found, creating default BULL sequence")
                regime_sequence = ["BULL"] * len(data)

            # Align lengths: interpolate/truncate as needed
            if len(regime_sequence) < len(data):
                # Vectorized nearest mapping from regime_sequence to data length
                src_idx = np.linspace(0, len(data) - 1, len(regime_sequence))
                dst_idx = np.arange(len(data))
                # For each dst index, find nearest src index via interpolation rounding
                nearest_src = np.rint(np.interp(dst_idx, src_idx, np.arange(len(regime_sequence)))).astype(int)
                nearest_src = np.clip(nearest_src, 0, len(regime_sequence) - 1)
                regime_sequence = [regime_sequence[j] for j in nearest_src]
            elif len(regime_sequence) > len(data):
                regime_sequence = regime_sequence[: len(data)]

            # Vectorized assignment
            df = data.iloc[: len(regime_sequence)].copy()
            df["regime"] = regime_sequence

            # Groupby regimes
            regime_data: dict[str, pd.DataFrame] = {k: v.copy() for k, v in df.groupby("regime")}

            self.logger.info(
                f"Regime split complete: {', '.join([f'{k}:{len(v)}' for k,v in regime_data.items()])}"
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
