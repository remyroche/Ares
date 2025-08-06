# src/training/steps/step8_tactician_labeling.py

import asyncio
import json
import os
import pickle
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from src.utils.logger import system_logger


class TacticianLabelingStep:
    """Step 8: Tactician Model Labeling using Analyst's model."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger

    async def initialize(self) -> None:
        """Initialize the tactician labeling step."""
        try:
            self.logger.info("Initializing Tactician Labeling Step...")
            self.logger.info("Tactician Labeling Step initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing Tactician Labeling Step: {e}")
            raise

    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute tactician model labeling.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Dict containing labeling results
        """
        try:
            self.logger.info("🔄 Executing Tactician Labeling...")

            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")

            # Load 1m data for tactician
            data_file_path = f"{data_dir}/{exchange}_{symbol}_historical_data.pkl"

            if not os.path.exists(data_file_path):
                raise FileNotFoundError(f"Data file not found: {data_file_path}")

            # Load data
            with open(data_file_path, "rb") as f:
                historical_data = pickle.load(f)

            # Convert to DataFrame if needed
            if not isinstance(historical_data, pd.DataFrame):
                historical_data = pd.DataFrame(historical_data)

            # Load analyst ensemble models
            analyst_ensembles_dir = f"{data_dir}/analyst_ensembles"
            analyst_ensembles = {}

            if os.path.exists(analyst_ensembles_dir):
                for ensemble_file in os.listdir(analyst_ensembles_dir):
                    if ensemble_file.endswith("_ensemble.pkl"):
                        regime_name = ensemble_file.replace("_ensemble.pkl", "")
                        ensemble_path = os.path.join(
                            analyst_ensembles_dir,
                            ensemble_file,
                        )

                        with open(ensemble_path, "rb") as f:
                            analyst_ensembles[regime_name] = pickle.load(f)

            # Generate strategic signals using analyst models
            strategic_signals = await self._generate_strategic_signals(
                historical_data,
                analyst_ensembles,
                symbol,
                exchange,
            )

            # Save labeled data
            labeled_data_dir = f"{data_dir}/tactician_labeled_data"
            os.makedirs(labeled_data_dir, exist_ok=True)

            labeled_file = (
                f"{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.pkl"
            )
            with open(labeled_file, "wb") as f:
                pickle.dump(labeled_data, f)

            # Save strategic signals
            signals_file = f"{data_dir}/{exchange}_{symbol}_strategic_signals.json"
            with open(signals_file, "w") as f:
                json.dump(strategic_signals, f, indent=2)

            self.logger.info(
                f"✅ Tactician labeling completed. Results saved to {labeled_data_dir}",
            )

            # Update pipeline state
            pipeline_state["tactician_labeled_data"] = labeled_data
            pipeline_state["strategic_signals"] = strategic_signals

            return {
                "tactician_labeled_data": labeled_data,
                "strategic_signals": strategic_signals,
                "labeled_file": labeled_file,
                "signals_file": signals_file,
                "duration": 0.0,  # Will be calculated in actual implementation
                "status": "SUCCESS",
            }

        except Exception as e:
            self.logger.error(f"❌ Error in Tactician Labeling: {e}")
            return {"status": "FAILED", "error": str(e), "duration": 0.0}

    async def _generate_strategic_signals(
        self,
        data: pd.DataFrame,
        analyst_ensembles: dict[str, Any],
        symbol: str,
        exchange: str,
    ) -> dict[str, Any]:
        """
        Generate strategic signals using analyst ensemble models.

        Args:
            data: Historical market data
            analyst_ensembles: Analyst ensemble models
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Dict containing strategic signals
        """
        try:
            self.logger.info(
                f"Generating strategic signals for {symbol} on {exchange}...",
            )

            # Calculate features for signal generation
            data_with_features = self._calculate_tactician_features(data)

            # Generate signals for each regime
            strategic_signals = {
                "symbol": symbol,
                "exchange": exchange,
                "signal_generation_date": datetime.now().isoformat(),
                "signals": [],
                "regime_signals": {},
            }

            # For each regime, generate signals using analyst ensemble
            for regime_name, ensemble_data in analyst_ensembles.items():
                self.logger.info(f"Generating signals for regime: {regime_name}")

                # Use the stacking ensemble for signal generation
                if "stacking_cv" in ensemble_data:
                    ensemble = ensemble_data["stacking_cv"]["ensemble"]

                    # Generate signals for this regime
                    regime_signals = await self._generate_regime_signals(
                        data_with_features,
                        ensemble,
                        regime_name,
                    )
                    strategic_signals["regime_signals"][regime_name] = regime_signals

                    # Add to overall signals
                    strategic_signals["signals"].extend(regime_signals)

            self.logger.info(
                f"Generated {len(strategic_signals['signals'])} strategic signals",
            )

            return strategic_signals

        except Exception as e:
            self.logger.error(f"Error generating strategic signals: {e}")
            raise

    def _calculate_tactician_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features for tactician signal generation.

        Args:
            data: Market data

        Returns:
            DataFrame with features added
        """
        try:
            # Calculate basic features
            data["price_change"] = data["close"].pct_change()
            data["volume_change"] = data["volume"].pct_change()
            data["high_low_ratio"] = data["high"] / data["low"]

            # Calculate moving averages
            data["sma_5"] = data["close"].rolling(window=5).mean()
            data["sma_10"] = data["close"].rolling(window=10).mean()
            data["sma_20"] = data["close"].rolling(window=20).mean()

            # Calculate momentum indicators
            data["momentum_5"] = data["close"] / data["close"].shift(5) - 1
            data["momentum_10"] = data["close"] / data["close"].shift(10) - 1

            # Calculate volatility
            data["volatility"] = data["price_change"].rolling(window=20).std()

            # Calculate RSI
            data["rsi"] = self._calculate_rsi(data["close"])

            # Fill NaN values
            data = data.fillna(method="bfill").fillna(0)

            return data

        except Exception as e:
            self.logger.error(f"Error calculating tactician features: {e}")
            raise

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    async def _generate_regime_signals(
        self,
        data: pd.DataFrame,
        ensemble: Any,
        regime_name: str,
    ) -> list[dict[str, Any]]:
        """
        Generate signals for a specific regime using analyst ensemble.

        Args:
            data: Market data with features
            ensemble: Analyst ensemble model
            regime_name: Name of the regime

        Returns:
            List of strategic signals
        """
        try:
            signals = []

            # Prepare feature columns
            feature_columns = [
                "price_change",
                "volume_change",
                "high_low_ratio",
                "sma_5",
                "sma_10",
                "sma_20",
                "momentum_5",
                "momentum_10",
                "volatility",
                "rsi",
            ]

            # Filter available features
            available_features = [col for col in feature_columns if col in data.columns]

            if not available_features:
                self.logger.warning(f"No features available for regime {regime_name}")
                return signals

            # Generate signals for each data point
            for i in range(len(data)):
                if i < 20:  # Skip first 20 points for feature calculation
                    continue

                # Get features for current point
                features = data[available_features].iloc[i]

                # Generate prediction using ensemble
                try:
                    # Reshape features for prediction
                    features_reshaped = features.values.reshape(1, -1)

                    # Get prediction probability
                    prediction_proba = ensemble.predict_proba(features_reshaped)[0]
                    prediction = ensemble.predict(features_reshaped)[0]

                    # Create signal based on prediction
                    signal_strength = max(prediction_proba)
                    signal_direction = (
                        "BUY"
                        if prediction == 1
                        else "SELL"
                        if prediction == -1
                        else "HOLD"
                    )

                    # Only create signal if confidence is high enough
                    if signal_strength > 0.6:
                        signal = {
                            "timestamp": data.index[i],
                            "regime": regime_name,
                            "signal_direction": signal_direction,
                            "signal_strength": signal_strength,
                            "prediction": prediction,
                            "prediction_proba": prediction_proba.tolist(),
                            "features": features.to_dict(),
                        }
                        signals.append(signal)

                except Exception as e:
                    self.logger.warning(f"Error generating signal for point {i}: {e}")
                    continue

            self.logger.info(
                f"Generated {len(signals)} signals for regime {regime_name}",
            )
            return signals

        except Exception as e:
            self.logger.error(f"Error generating regime signals for {regime_name}: {e}")
            raise

# For backward compatibility with existing step structure
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    **kwargs,
) -> bool:
    """
    Run the tactician labeling step.

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
        step = TacticianLabelingStep(config)
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
        print(f"❌ Tactician labeling failed: {e}")
        return False


if __name__ == "__main__":
    # Test the step
    async def test():
        result = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Test result: {result}")

    asyncio.run(test())
