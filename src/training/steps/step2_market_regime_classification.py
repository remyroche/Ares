# src/training/steps/step2_market_regime_classification.py

import asyncio
import json
import os
from datetime import datetime
from typing import Any

import pandas as pd

from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier
from src.utils.logger import system_logger


class MarketRegimeClassificationStep:
    """Step 2: Market Regime Classification using UnifiedRegimeClassifier."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger
        self.regime_classifier = None

    async def initialize(self) -> None:
        """Initialize the market regime classification step."""
        self.logger.info("Initializing Market Regime Classification Step...")

        # Initialize the unified regime classifier
        self.regime_classifier = UnifiedRegimeClassifier(self.config)
        await self.regime_classifier.initialize()

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
        
        # Try to load pre-consolidated data first
        consolidated_file = f"data_cache/aggtrades_{exchange}_{symbol}_consolidated.parquet"
        data_file_path = f"{data_dir}/{exchange}_{symbol}_historical_data.pkl"
        
        historical_data = None
        
        # Try consolidated parquet file first
        if os.path.exists(consolidated_file):
            self.logger.info(f"📁 Loading pre-consolidated data from: {consolidated_file}")
            try:
                historical_data = pd.read_parquet(consolidated_file)
                self.logger.info(f"✅ Loaded consolidated data: {len(historical_data)} records")
            except Exception as e:
                self.logger.warning(f"⚠️  Failed to load consolidated data: {e}")
                historical_data = None
        
        # Fallback to pickle file if consolidated data not available
        if historical_data is None:
            if not os.path.exists(data_file_path):
                raise FileNotFoundError(f"Data file not found: {data_file_path}")
            
            self.logger.info(f"📁 Loading data from pickle file: {data_file_path}")
            import pickle
            
            with open(data_file_path, "rb") as f:
                historical_data = pickle.load(f)
            
            self.logger.info(f"✅ Loaded historical data: {len(historical_data)} records")
            
            # Convert to DataFrame if needed
            if not isinstance(historical_data, pd.DataFrame):
                historical_data = pd.DataFrame(historical_data)

        # Perform regime classification
        regime_results = await self._classify_market_regimes(
            historical_data,
            symbol,
            exchange,
        )

        # Save regime classification results
        regime_file_path = (
            f"{data_dir}/{exchange}_{symbol}_regime_classification.json"
        )
        os.makedirs(os.path.dirname(regime_file_path), exist_ok=True)

        with open(regime_file_path, "w") as f:
            json.dump(regime_results, f, indent=2)

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
                    if isinstance(regimes, list):
                        formatted_results["regime_sequence"] = regimes

                        # Calculate regime distribution
                        from collections import Counter

                        regime_counts = Counter(regimes)
                        formatted_results["regime_distribution"] = dict(regime_counts)

                        # Calculate regime transitions
                        transitions = []
                        for i in range(1, len(regimes)):
                            if regimes[i] != regimes[i - 1]:
                                transitions.append(
                                    {
                                        "from_regime": regimes[i - 1],
                                        "to_regime": regimes[i],
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
