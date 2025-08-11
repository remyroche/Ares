# src/training/steps/step8_tactician_labeling.py

import asyncio
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.utils.warning_symbols import (
    error,
    failed,
)
from src.training.steps.unified_data_loader import get_unified_data_loader

# Preference order for selecting analyst ensembles
ENSEMBLE_PREFERENCE_ORDER = ("stacking_cv", "dynamic_weighting", "voting")

# Removing duplicate earlier TacticianLabelingStep definition to avoid conflicts


class TacticianTripleBarrierLabeler:
    """
    Applies a triple barrier to generate labels specifically for a short-term, high-leverage Tactician model.

    This labeler uses FIXED PERCENTAGE barriers and a short time horizon to reward
    models that can accurately predict immediate, favorable price action under strict risk parameters.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config.get("tactician_triple_barrier", {})
        self.logger = system_logger.getChild("TacticianTripleBarrierLabeler")

    def apply_labels(
        self,
        data: pd.DataFrame,
        strategic_signals: pd.Series,
    ) -> pd.DataFrame:
        """
        Vectorized application of the triple barrier method.

        Args:
            data: The 1-minute market data (must contain OHLC columns).
            strategic_signals: A Series with timestamps as index and signals (+1 for BUY, -1 for SELL)
                               as values, indicating when the Analyst has identified a setup.

        Returns:
            A DataFrame with the new 'tactician_label' column.
        """
        self.logger.info(
            "Applying specialized Tactician triple barrier labels using fixed percentages...",
        )

        # Get parameters from config, with defaults for a high-leverage, 1m timeframe
        pt_pct = self.config.get("profit_take_pct", 0.005)  # Target 0.5% profit
        sl_pct = self.config.get("stop_loss_pct", 0.0025)  # Stop out at 0.25% loss
        time_barrier = self.config.get(
            "time_barrier_periods",
            30,
        )  # 30-minute time horizon

        # Align signals with the data index
        entry_points = (
            strategic_signals[strategic_signals != 0].reindex(data.index).dropna()
        )
        if entry_points.empty:
            self.logger.warning(
                "No strategic signals found to label. Returning data without labels.",
            )
            data[
                "tactician_label"
            ] = -1  # Default to sell signal for binary classification
            return data

        entry_indices = data.index.get_indexer_for(entry_points.index)

        # Calculate fixed percentage barriers for each entry point
        entry_prices = data["open"].iloc[entry_indices + 1]

        profit_barriers = entry_prices * (1 + pt_pct * entry_points.values)
        stop_barriers = entry_prices * (1 - sl_pct * entry_points.values)

        labels = pd.Series(
            -1, index=data.index
        )  # Default to sell signal for binary classification

        # Vectorized barrier check
        for i, entry_idx in enumerate(entry_indices):
            if entry_idx >= len(data) - 1:
                continue

            signal = entry_points.iloc[i]
            pt = profit_barriers.iloc[i]
            sl = stop_barriers.iloc[i]

            path = data.iloc[entry_idx + 1 : entry_idx + 1 + time_barrier]
            if path.empty:
                continue

            # Check for hits
            pt_hit_mask = (path["high"] >= pt) if signal == 1 else (path["low"] <= pt)
            sl_hit_mask = (path["low"] <= sl) if signal == 1 else (path["high"] >= sl)

            pt_hit_time = path.index[pt_hit_mask].min()
            sl_hit_time = path.index[sl_hit_mask].min()

            # Determine label based on which barrier was hit first
            if pd.notna(pt_hit_time) and (
                pd.isna(sl_hit_time) or pt_hit_time <= sl_hit_time
            ):
                labels.iloc[entry_idx] = 1  # Profit take
            elif pd.notna(sl_hit_time):
                labels.iloc[entry_idx] = -1  # Stop loss

        data["tactician_label"] = labels
        self.logger.info(
            f"Tactician labeling complete. Label distribution:\n{labels.value_counts()}",
        )
        return data


class TacticianLabelingStep:
    """Step 8: Tactician Model Labeling using Analyst's model."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="tactician labeling step initialization",
    )
    async def initialize(self) -> None:
        """Initialize the tactician labeling step."""
        self.logger.info("Initializing Tactician Labeling Step...")

    @handle_errors(
        exceptions=(Exception,),
        default_return={"status": "FAILED", "error": "Execution failed"},
        context="tactician labeling step execution",
    )
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

            # Use unified data loader to get comprehensive data for tactician labeling
            self.logger.info("🔄 Loading unified data for tactician labeling...")
            data_loader = get_unified_data_loader(self.config)
            timeframe = training_input.get("timeframe", "1m")

            # Load unified data with optimizations for ML training (180 days for tactician labeling)
            data_1m = await data_loader.load_unified_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                lookback_days=180,
                use_streaming=True,  # Enable streaming for large datasets
            )

            if data_1m is None or data_1m.empty:
                self.logger.error(f"No unified data found for {symbol} on {exchange}")
                return {
                    "status": "FAILED",
                    "error": f"No unified data found for {symbol} on {exchange}",
                }

            # Log data information
            data_info = data_loader.get_data_info(data_1m)
            self.logger.info(f"✅ Loaded unified data: {data_info['rows']} rows")
            self.logger.info(
                f"   Date range: {data_info['date_range']['start']} to {data_info['date_range']['end']}"
            )
            self.logger.info(
                f"   Has aggtrades data: {data_info['has_aggtrades_data']}"
            )
            self.logger.info(f"   Has futures data: {data_info['has_futures_data']}")

            # Ensure we have the required OHLCV columns
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_columns = [
                col for col in required_columns if col not in data_1m.columns
            ]
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return {
                    "status": "FAILED",
                    "error": f"Missing required columns: {missing_columns}",
                }
            try:
                self.logger.info(
                    f"Loaded 1m data: shape={getattr(data_1m, 'shape', None)}, columns={list(getattr(data_1m, 'columns', [])[:10])}",
                )
            except Exception:
                pass

            # Load analyst ensemble models
            analyst_ensembles = self._load_analyst_ensembles(data_dir)

            # Generate strategic "setup" signals using analyst models
            (
                data_with_features,
                strategic_signals,
            ) = await self._generate_strategic_signals(data_1m, analyst_ensembles)

            # Apply the specialized Tactician Triple Barrier
            labeler = TacticianTripleBarrierLabeler(self.config)
            labeled_data = labeler.apply_labels(data_with_features, strategic_signals)
            try:
                self.logger.info(
                    f"Strategic signals summary: total={len(strategic_signals)}, nonzero={(strategic_signals != 0).sum()}",
                )
            except Exception:
                pass

            # Save results
            labeled_file, signals_file = self._save_results(
                labeled_data,
                strategic_signals,
                data_dir,
                exchange,
                symbol,
            )

            self.logger.info(
                f"✅ Tactician labeling completed. Labeled data saved to {labeled_file}",
            )

            pipeline_state["tactician_labeled_data"] = labeled_data
            return {
                "status": "SUCCESS",
                "labeled_file": labeled_file,
                "signals_file": signals_file,
            }
        except Exception as e:
            self.print(error("❌ Error in Tactician Labeling: {e}"))
            return {"status": "FAILED", "error": str(e)}

    def _load_analyst_ensembles(self, data_dir: str) -> dict[str, Any]:
        """Loads all trained analyst ensemble models."""
        analyst_ensembles_dir = f"{data_dir}/analyst_ensembles"
        analyst_ensembles = {}
        if not Path(analyst_ensembles_dir).exists():
            msg = f"Analyst ensembles directory not found: {analyst_ensembles_dir}"
            raise FileNotFoundError(
                msg,
            )

        for ensemble_file in os.listdir(analyst_ensembles_dir):
            if ensemble_file.endswith("_ensemble.pkl"):
                regime_name = ensemble_file.replace("_ensemble.pkl", "")
                ensemble_path = Path(analyst_ensembles_dir) / ensemble_file
                with ensemble_path.open("rb") as f:
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
                        chosen_ensemble = (
                            loaded.get("ensemble") if "ensemble" in loaded else None
                        )
                # Record whatever we found (could be None; upstream handles None)
                analyst_ensembles[regime_name] = chosen_ensemble
        return analyst_ensembles

    async def _generate_strategic_signals(
        self,
        data: pd.DataFrame,
        analyst_ensembles: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Generate strategic signals using analyst ensemble models."""
        self.logger.info("Generating strategic 'setup' signals from Analyst models...")

        # Step 1: Calculate all features needed for any of the analyst models
        data_with_features = self._calculate_features(data)

        # Step 2: Determine the market regime for each data point
        # This is a placeholder for your regime detection logic (e.g., from step 4)
        # It is crucial that this logic is consistent with how regimes were defined during Analyst training.
        data_with_features["regime"] = self._get_market_regime(data_with_features)

        all_signals = pd.Series(0, index=data_with_features.index)

        # Step 3: Predict in a vectorized way for each regime
        for regime_name, ensemble in analyst_ensembles.items():
            if ensemble is None:
                continue

            regime_mask = data_with_features["regime"] == regime_name
            if not regime_mask.any():
                continue

            # Ensure the model's expected features are present
            if hasattr(ensemble, "feature_names_in_"):
                features_for_model = [
                    f
                    for f in ensemble.feature_names_in_
                    if f in data_with_features.columns
                ]
                x_regime = data_with_features.loc[regime_mask, features_for_model]
            else:
                # Fallback if feature names are not stored in the model
                x_regime = data_with_features.loc[regime_mask].select_dtypes(
                    include=np.number,
                )

            if not x_regime.empty:
                predictions = ensemble.predict(x_regime)
                all_signals[regime_mask] = predictions

        self.logger.info(
            f"Generated strategic signals. Signal distribution:\n{all_signals.value_counts()}",
        )
        return data_with_features, all_signals

    def _get_market_regime(self, data: pd.DataFrame) -> pd.Series:
        """
        Placeholder for your market regime detection logic.
        This should be consistent with the logic from step4_regime_specific_training.
        """
        # Example: Simple regime based on volatility percentile
        # NOTE: Volatility is calculated here because the Analyst models need it for regime detection.
        # It is NOT used by the Tactician's labeler.
        vol_percentile = data["volatility"].rank(pct=True)
        bins = [0, 0.33, 0.66, 1.0]
        labels = ["SIDEWAYS", "BULL", "BEAR"]
        regimes = pd.cut(vol_percentile, bins=bins, labels=labels, right=False)
        return regimes.astype(str).fillna("SIDEWAYS")

    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all necessary features for both Analyst and Tactician."""
        data["returns"] = data["close"].pct_change()
        # Volatility is calculated here for the Analyst's regime detection, not for Tactician labeling.
        data["volatility"] = (
            data["returns"].rolling(window=60).std().bfill()
        )  # 1-hour volatility
        # ... Add all other features your Analyst models were trained on ...
        # e.g., RSI, MACD, Bollinger Bands, etc.
        return data.fillna(method="ffill").fillna(0)

    def _save_results(self, labeled_data, signals, data_dir, exchange, symbol):
        """Saves the labeled data and signals to disk."""
        labeled_data_dir = f"{data_dir}/tactician_labeled_data"
        Path(labeled_data_dir).mkdir(parents=True, exist_ok=True)

        # Prefer Parquet for DataFrame/Series persistence
        labeled_file_parquet = (
            f"{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.parquet"
        )
        try:
            try:
                from src.training.enhanced_training_manager_optimized import (
                    ParquetDatasetManager,
                )

                ParquetDatasetManager(logger=self.logger).write_flat_parquet(
                    labeled_data,
                    labeled_file_parquet,
                    schema_name="split",
                    compression="snappy",
                    use_dictionary=True,
                    row_group_size=128_000,
                )
            except Exception:
                from src.utils.logger import log_io_operation, log_dataframe_overview

                with log_io_operation(
                    self.logger,
                    "to_parquet",
                    labeled_file_parquet,
                    compression="snappy",
                ):
                    labeled_data.to_parquet(
                        labeled_file_parquet, compression="snappy", index=False
                    )
                try:
                    log_dataframe_overview(
                        self.logger, labeled_data, name="labeled_data"
                    )
                except Exception:
                    pass
        except Exception:
            # Fallback to Pickle for compatibility
            labeled_file_pickle = (
                f"{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.pkl"
            )
            labeled_data.to_pickle(labeled_file_pickle)
            labeled_file_parquet = labeled_file_pickle

        signals_file_parquet = (
            f"{data_dir}/{exchange}_{symbol}_strategic_signals.parquet"
        )
        try:
            # Save Series as Parquet by converting to DataFrame
            _signals_df = signals.to_frame(name="signal").reset_index()
            try:
                from src.training.enhanced_training_manager_optimized import (
                    ParquetDatasetManager,
                )

                ParquetDatasetManager(logger=self.logger).write_flat_parquet(
                    _signals_df,
                    signals_file_parquet,
                    schema_name="split",
                    compression="snappy",
                    use_dictionary=True,
                    row_group_size=128_000,
                )
            except Exception:
                from src.utils.logger import log_io_operation, log_dataframe_overview

                with log_io_operation(
                    self.logger,
                    "to_parquet",
                    signals_file_parquet,
                    compression="snappy",
                ):
                    _signals_df.to_parquet(
                        signals_file_parquet, compression="snappy", index=False
                    )
                try:
                    log_dataframe_overview(self.logger, _signals_df, name="signals_df")
                except Exception:
                    pass
        except Exception:
            signals_file_pickle = (
                f"{data_dir}/{exchange}_{symbol}_strategic_signals.pkl"
            )
            signals.to_pickle(signals_file_pickle)
            signals_file_parquet = signals_file_pickle

        return labeled_file_parquet, signals_file_parquet


# For backward compatibility with existing step structure
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    force_rerun: bool = False,
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
            "force_rerun": force_rerun,
            **kwargs,
        }

        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)

        return result.get("status") == "SUCCESS"

    except Exception:
        print(failed("Tactician labeling failed: {e}"))
        return False


if __name__ == "__main__":
    # Test the step
    async def test():
        result = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Test result: {result}")

    asyncio.run(test())
