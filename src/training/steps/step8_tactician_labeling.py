# src/training/steps/step8_tactician_labeling.py

import asyncio
import json
import os
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict

import pandas as pd
import numpy as np

from src.utils.logger import system_logger


# Preference order for selecting analyst ensembles
ENSEMBLE_PREFERENCE_ORDER = ("stacking_cv", "dynamic_weighting", "voting")


class TacticianLabelingStep:
    """Step 8: Tactician Model Labeling using Analyst's model."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger

    async def initialize(self) -> None:
        """Initialize the tactician labeling step."""
        self.logger.info("Initializing Tactician Labeling Step...")

    def _calculate_tactician_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features for tactician signal generation.

        Args:
            data: Market data

        Returns:
            DataFrame with features added
        """
        try:
            # Returns-based core features
            data["returns"] = data["close"].pct_change()
            data["log_returns"] = np.log(data["close"] / data["close"].shift(1))
            data["volume_change"] = data["volume"].pct_change()
            data["high_low_ratio"] = data["high"] / data["low"]

            # Moving averages on price and returns
            data["sma_5"] = data["close"].rolling(window=5).mean()
            data["sma_10"] = data["close"].rolling(window=10).mean()
            data["sma_20"] = data["close"].rolling(window=20).mean()
            data["ret_sma_5"] = data["returns"].rolling(window=5).mean()
            data["ret_sma_20"] = data["returns"].rolling(window=20).mean()

            # Momentum indicators based on returns
            data["momentum_5"] = data["close"] / data["close"].shift(5) - 1
            data["momentum_10"] = data["close"] / data["close"].shift(10) - 1

            # Volatility from returns
            data["volatility"] = data["returns"].rolling(window=20).std()

            # RSI on price
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
                "returns",
                "log_returns",
                "volume_change",
                "high_low_ratio",
                "sma_5",
                "sma_10",
                "sma_20",
                "ret_sma_5",
                "ret_sma_20",
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

    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute tactician model labeling."""
        try:
            self.logger.info("🔄 Executing Tactician Labeling...")

            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")

            # Load 1m data for tactician
            data_file_path = f"{data_dir}/{exchange}_{symbol}_historical_data.pkl"
            # Historical data is a pickled dict with keys like 'klines', 'agg_trades', 'futures'
            # Prefer 1m klines if available
            try:
                historical = pd.read_pickle(data_file_path)
            except Exception:
                import pickle as _pkl
                with open(data_file_path, "rb") as _f:
                    historical = _pkl.load(_f)
            if isinstance(historical, dict):
                data_1m = historical.get("klines")
                if data_1m is None:
                    # Fallback: try any DataFrame-like entry
                    for _k, _v in historical.items():
                        if isinstance(_v, pd.DataFrame) and not _v.empty:
                            data_1m = _v
                            break
                if data_1m is None:
                    raise ValueError(f"No DataFrame found in historical data at {data_file_path}")
            elif isinstance(historical, pd.DataFrame):
                data_1m = historical
            else:
                raise ValueError(f"Unsupported historical data type: {type(historical)} from {data_file_path}")

            # Load analyst ensemble models
            analyst_ensembles = self._load_analyst_ensembles(data_dir)

            # Generate strategic "setup" signals using analyst models
            data_with_features, strategic_signals = await self._generate_strategic_signals(
                data_1m, analyst_ensembles
            )

            # Apply the specialized Tactician Triple Barrier
            labeler = TacticianTripleBarrierLabeler(self.config)
            labeled_data = labeler.apply_labels(data_with_features, strategic_signals)

            # Save results
            labeled_file, signals_file = self._save_results(
                labeled_data, strategic_signals, data_dir, exchange, symbol
            )

            self.logger.info(f"✅ Tactician labeling completed. Labeled data saved to {labeled_file}")
            
            pipeline_state["tactician_labeled_data"] = labeled_data
            return {
                "status": "SUCCESS",
                "labeled_file": labeled_file,
                "signals_file": signals_file,
            }
        except Exception as e:
            self.logger.error(f"❌ Error in Tactician Labeling: {e}", exc_info=True)
            return {"status": "FAILED", "error": str(e)}

    def _load_analyst_ensembles(self, data_dir: str) -> Dict[str, Any]:
        """Loads all trained analyst ensemble models."""
        analyst_ensembles_dir = f"{data_dir}/analyst_ensembles"
        analyst_ensembles = {}
        if not os.path.exists(analyst_ensembles_dir):
            raise FileNotFoundError(f"Analyst ensembles directory not found: {analyst_ensembles_dir}")
            
        for ensemble_file in os.listdir(analyst_ensembles_dir):
            if ensemble_file.endswith("_ensemble.pkl"):
                regime_name = ensemble_file.replace("_ensemble.pkl", "")
                ensemble_path = os.path.join(analyst_ensembles_dir, ensemble_file)
                with open(ensemble_path, "rb") as f:
                    loaded = pickle.load(f)
                chosen_ensemble = None
                if isinstance(loaded, dict):
                    # Prefer stacking_cv, then dynamic_weighting, then voting
                    for key in ENSEMBLE_PREFERENCE_ORDER:
                        if key in loaded and isinstance(loaded[key], dict):
                            obj = loaded[key].get("ensemble")
                            if obj is not None:
                                chosen_ensemble = obj
                                break
                    if chosen_ensemble is None:
                        # Fallback if saved dict is a single-ensemble payload
                        chosen_ensemble = loaded.get("ensemble") if "ensemble" in loaded else None
                # Record whatever we found (could be None; upstream handles None)
                analyst_ensembles[regime_name] = chosen_ensemble
        return analyst_ensembles

    async def _generate_strategic_signals(
        self, data: pd.DataFrame, analyst_ensembles: dict[str, Any]
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Generate strategic signals using analyst ensemble models."""
        self.logger.info("Generating strategic 'setup' signals from Analyst models...")
        
        # Step 1: Calculate all features needed for any of the analyst models
        data_with_features = self._calculate_tactician_features(data)
        
        # Step 2: Determine the market regime for each data point
        # This is a placeholder for your regime detection logic (e.g., from step 4)
        # It is crucial that this logic is consistent with how regimes were defined during Analyst training.
        data_with_features['regime'] = self._get_market_regime(data_with_features)
        
        all_signals = pd.Series(0, index=data_with_features.index)

        # Step 3: Predict in a vectorized way for each regime
        for regime_name, ensemble in analyst_ensembles.items():
            if ensemble is None: continue
            
            regime_mask = data_with_features['regime'] == regime_name
            if not regime_mask.any(): continue
            
            # Ensure the model's expected features are present
            if hasattr(ensemble, 'feature_names_in_'):
                features_for_model = [f for f in ensemble.feature_names_in_ if f in data_with_features.columns]
                X_regime = data_with_features.loc[regime_mask, features_for_model]
            else:
                # Fallback if feature names are not stored in the model
                X_regime = data_with_features.loc[regime_mask].select_dtypes(include=np.number)

            if not X_regime.empty:
                predictions = ensemble.predict(X_regime)
                all_signals[regime_mask] = predictions

        self.logger.info(f"Generated strategic signals. Signal distribution:\n{all_signals.value_counts()}")
        return data_with_features, all_signals

    def _get_market_regime(self, data: pd.DataFrame) -> pd.Series:
        """
        Placeholder for your market regime detection logic.
        This should be consistent with the logic from step4_regime_specific_training.
        """
        # Example: Simple regime based on volatility percentile
        # NOTE: Volatility is calculated here because the Analyst models need it for regime detection.
        # It is NOT used by the Tactician's labeler.
        vol_percentile = data['volatility'].rank(pct=True)
        bins = [0, 0.33, 0.66, 1.0]
        labels = ['SIDEWAYS', 'BULL', 'BEAR']
        regimes = pd.cut(vol_percentile, bins=bins, labels=labels, right=False)
        return regimes.astype(str).fillna('SIDEWAYS')

    def _save_results(self, labeled_data, signals, data_dir, exchange, symbol):
        """Saves the labeled data and signals to disk."""
        labeled_data_dir = f"{data_dir}/tactician_labeled_data"
        os.makedirs(labeled_data_dir, exist_ok=True)
        
        labeled_file = f"{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.pkl"
        labeled_data.to_pickle(labeled_file)

        signals_file = f"{data_dir}/{exchange}_{symbol}_strategic_signals.pkl"
        signals.to_pickle(signals_file)
        
        return labeled_file, signals_file
        

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
