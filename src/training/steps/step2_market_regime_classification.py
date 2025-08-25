# src/training/steps/step2_market_regime_classification.py

import asyncio
import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier
from src.training.steps.unified_data_loader import get_unified_data_loader
from src.utils.decorators import guard_dataframe_nulls, with_tracing_span
from src.utils.logger import system_logger


def convert_trade_data_to_ohlcv(
    trade_data: pd.DataFrame, timeframe: str = "1h",
) -> pd.DataFrame:
    """Convert trade data to OHLCV format.

    Args:
        trade_data: DataFrame with columns ['timestamp', 'price', 'quantity', 'is_buyer_maker', 'agg_trade_id']
        timeframe: Timeframe for resampling (e.g., '1h', '1m', '1d')

    Returns:
        DataFrame with OHLCV columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
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
            {"price": ["first", "max", "min", "last"], "quantity": "sum"},
        )

        # Flatten column names
        ohlcv.columns = ["open", "high", "low", "close", "volume"]

        # Reset index to create timestamp column
        ohlcv = ohlcv.reset_index()

        # Remove any rows with NaN values
        return ohlcv.dropna()

    except Exception as e:
        system_logger.error(f"🚨 Error converting trade data to OHLCV: {e}")
        raise


class MarketRegimeClassificationStep:
    """Step 2: Market Regime Classification using UnifiedRegimeClassifier."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger
        self.regime_classifier = None

    async def initialize(self) -> None:
        """Initialize the market regime classification step."""
        self.logger.info("🚀 Initializing Market Regime Classification Step...")

        # Using advanced HMM-based UnifiedRegimeClassifier
        self.regime_classifier = None  # Will be initialized per execution

        self.logger.info(
            "✅ Market Regime Classification Step initialized successfully (Advanced HMM)",
        )

    async def execute(
        self, training_input: dict[str, Any], pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute market regime classification.

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

        # Use unified data loader to get data
        self.logger.info("🔄 Loading data using unified data loader...")
        data_loader = get_unified_data_loader(self.config)

        # Determine lookback period: prefer training_input, fallback to config (default 180 days)
        from src.config.constants import (
            BLANK_TRAINING_LOOKBACK_DAYS,
        )

        # Use lookback_days from training_input (passed from enhanced training manager) or config
        lookback_days = training_input.get(
            "lookback_days",
            self.config.get("lookback_days", BLANK_TRAINING_LOOKBACK_DAYS),
        )

        # Load unified data with optimizations for ML training
        historical_data = await data_loader.load_unified_data(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            lookback_days=lookback_days,
            use_streaming=True,  # Enable streaming for large datasets
        )

        if historical_data is None or historical_data.empty:
            msg = f"No data found for {symbol} on {exchange}"
            raise ValueError(msg)

        # Log data information
        data_info = data_loader.get_data_info(historical_data)
        self.logger.info(f"✅ Loaded unified data: {data_info['rows']} rows")
        self.logger.info(
            f"   Date range: {data_info['date_range']['start']} to {data_info['date_range']['end']}",
        )
        self.logger.info(f"   Has aggtrades data: {data_info['has_aggtrades_data']}")
        self.logger.info(f"   Has futures data: {data_info['has_futures_data']}")

        # Ensure we have the required OHLCV columns
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_columns = [
            col for col in required_columns if col not in historical_data.columns
        ]
        if missing_columns:
            msg = f"Missing required columns: {missing_columns}"
            raise ValueError(msg)

        # Convert to 1h timeframe if needed for regime classification
        if timeframe != "1h":
            self.logger.info(
                "🔄 Resampling data to 1h timeframe for regime classification...",
            )
            historical_data = self._resample_to_timeframe(historical_data, "1h")
            self.logger.info(f"✅ Resampled to 1h: {len(historical_data)} records")

        # Perform regime classification
        regime_results = await self._classify_market_regimes(
            historical_data,
            symbol,
            exchange,
            training_input=training_input,
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
            # Use timestamps from original data/index
            if "timestamp" in historical_data.columns:
                timestamps = pd.to_datetime(historical_data["timestamp"]).tolist()
            elif isinstance(historical_data.index, pd.DatetimeIndex):
                timestamps = historical_data.index.to_list()
            else:
                # Fallback: generate hourly timestamps ending at current time
                try:
                    periods = len(regime_results["regime_sequence"]) 
                    timestamps = pd.date_range(
                        end=pd.Timestamp.utcnow(), periods=periods, freq="1H"
                    ).to_list()
                except Exception:
                    timestamps = list(range(len(regime_results["regime_sequence"])))

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
                },
            )

            # Save to parquet
            parquet_df.to_parquet(parquet_file_path, index=False)
            self.logger.info(
                f"✅ Saved regime classification results to parquet: {parquet_file_path}",
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

    @guard_dataframe_nulls(mode="warn", arg_index=1)
    @with_tracing_span("MarketRegime._resample_to_timeframe", log_args=False)
    def _resample_to_timeframe(
        self, df: pd.DataFrame, target_timeframe: str,
    ) -> pd.DataFrame:
        """Resample data to a different timeframe.

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
                },
            )

            # Reset index to get timestamp column back
            resampled = resampled.reset_index()

            # Remove any rows with NaN values
            return resampled.dropna()

        except Exception as e:
            self.logger.exception(f"🚨 Error resampling data: {e}")
            raise

    async def _classify_market_regimes(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str,
        *,
        training_input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Classify market regimes using the existing advanced HMM-based UnifiedRegimeClassifier.

        This uses the pre-existing classify_regimes method instead of rewriting the logic.
        """
        try:
            self.logger.info(
                f"Classifying market regimes (Advanced HMM) for {symbol} on {exchange}...",
            )

            # Ensure required columns exist and are sorted by timestamp
            required_columns = ["open", "high", "low", "close", "volume", "timestamp"]
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

            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]
            if missing_columns:
                msg = f"Missing required columns for regime classification: {missing_columns}"
                raise ValueError(
                    msg,
                )

            if "timestamp" in data.columns:
                data = data.sort_values("timestamp").reset_index(drop=True)

            df = data.copy()

            # Initialize the UnifiedRegimeClassifier
            self.logger.info("Initializing UnifiedRegimeClassifier...")
            regime_classifier = UnifiedRegimeClassifier(
                config=self.config,
                exchange=exchange,
                symbol=symbol,
            )

            # Use the existing classify_regimes method
            self.logger.info("Using existing advanced HMM classification system...")
            classification_results = await regime_classifier.classify_regimes(df)

            # Check for errors
            if "error" in classification_results:
                raise ValueError(f"Regime classification failed: {classification_results['error']}")

            # Extract results from the existing method
            regimes = classification_results.get("regimes", [])
            confidence_scores = classification_results.get("confidence_scores", [])
            regime_distribution = classification_results.get("regime_distribution", {})
            total_records = classification_results.get("total_records", len(df))

            # Build results in the expected format for step2
            formatted_results: dict[str, Any] = {
                "symbol": symbol,
                "exchange": exchange,
                "classification_date": datetime.utcnow().isoformat(),
                "total_records": total_records,
                "regime_distribution": regime_distribution,
                "regime_sequence": regimes,
                "regime_transitions": [],
                "confidence_scores": confidence_scores,
                "metadata": {
                    "classifier_version": "unified_hmm_v1",
                    "classification_method": "ADVANCED_HMM_CATEGORIZATION",
                    "timeframe": "1h",
                    "hmm_states": regime_classifier.n_states,
                    "hmm_iterations": regime_classifier.n_iter,
                    "system_status": regime_classifier.get_system_status(),
                },
            }

            # Calculate transitions
            if len(regimes) > 1:
                s_regimes = pd.Series(regimes)
                shifted = s_regimes.shift(1)
                mask = s_regimes != shifted
                transitions_df = pd.DataFrame(
                    {
                        "from_regime": shifted[mask].values,
                        "to_regime": s_regimes[mask].values,
                        "transition_index": s_regimes.index[mask].values,
                    },
                )
                formatted_results["regime_transitions"] = transitions_df.to_dict("records")

            self.logger.info(
                f"Advanced HMM regime classification completed. Found {len(regime_distribution)} distinct regimes",
            )

            return formatted_results

        except Exception as e:
            self.logger.exception(f"🚨 Error in regime classification: {e}")
            raise


# Import training pipeline decorators for comprehensive security and troubleshooting
from src.utils.training_pipeline_decorators import (
    circuit_breaker_protection,
    debug_training_step,
    memory_efficient,
    prevent_data_leakage,
    quality_gate,
    resource_monitor,
    secure_data_processing,
    validate_step_output,
    validate_step_prerequisites,
)


# For backward compatibility with existing step structure
@validate_step_prerequisites(
    required_directories=["data_cache", "data/training"],
    min_memory_gb=4.0,
    min_disk_gb=2.0,
    required_packages=["pandas", "numpy", "sklearn"],
    data_quality_checks={
        "min_rows": 1000,
        "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
    },
    context="Market Regime Classification",
)
@secure_data_processing(
    backup_before=True, integrity_checks=True, memory_cleanup=True, data_validation=True,
)
@prevent_data_leakage(
    temporal_validation=True,
    feature_leakage_detection=True,
    lookahead_bias_prevention=True,
)
@resource_monitor(
    memory_threshold_gb=8.0,
    cpu_threshold_percent=80.0,
    disk_threshold_gb=5.0,
    monitor_interval=30.0,
    auto_cleanup=True,
)
@memory_efficient(
    chunk_size=25000, streaming_processing=True, memory_pool=True, cleanup_frequency=50,
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=True,
    performance_profiling=True,
    error_context_preservation=True,
)
@circuit_breaker_protection(
    failure_threshold=3,
    recovery_timeout=120.0,
    expected_exception=Exception,
    monitor_interval=30.0,
)
@validate_step_output(
    required_files=["data/training/{exchange}_{symbol}_regime_classification.json"],
    data_quality_checks={"min_rows": 100, "required_columns": ["regime", "confidence"]},
    performance_thresholds={"classification_time_minutes": 30.0},
    format_validation=True,
)
@quality_gate(
    data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
    validation_score_requirements={"classification_accuracy": 0.7},
)
async def run_step(
    symbol: str, exchange: str = "BINANCE", data_dir: str = "data/training", **kwargs: Any,
) -> bool:
    """Run the market regime classification step.

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
        training_input: dict[str, Any] = {
            "symbol": symbol,
            "exchange": exchange,
            "data_dir": data_dir,
            **kwargs,
        }

        pipeline_state: dict[str, Any] = {}
        result = await step.execute(training_input, pipeline_state)

        return result.get("status") == "SUCCESS"

    except Exception:
        return False


if __name__ == "__main__":
    # Test the step
    async def test() -> None:
        await run_step("ETHUSDT", "BINANCE", "data/training")

    asyncio.run(test())