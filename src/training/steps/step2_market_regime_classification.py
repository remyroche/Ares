# src/training/steps/step2_market_regime_classification.py

import asyncio
import json
import os
from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np

from src.analyst.simple_regime_rules import classify_regime_series
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

        # No ML classifier; using deterministic EMA/ADX rules
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

        # Use unified data loader to get data
        self.logger.info("🔄 Loading data using unified data loader...")
        data_loader = get_unified_data_loader(self.config)

        # Determine lookback period: prefer training_input, fallback to config (default 180 days)
        lookback_days = training_input.get("lookback_days", self.config.get("lookback_days", 180))

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
                        timestamps = pd.date_range(end=pd.Timestamp.utcnow(), periods=periods, freq="1H").to_list()
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
        *,
        training_input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Classify market regimes using simplified EMA/ADX rules on raw OHLCV.

        Rules (1h timeframe):
        - Bull: EMA(21) > EMA(55) AND ADX > 25
        - Bear: EMA(21) < EMA(55) AND ADX > 25
        - Sideways: if neither Bull nor Bear OR ADX < 20
        """
        try:
            self.logger.info(
                f"Classifying market regimes (EMA/ADX) for {symbol} on {exchange}...",
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

            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(
                    f"Missing required columns for regime classification: {missing_columns}",
                )

            if "timestamp" in data.columns:
                data = data.sort_values("timestamp").reset_index(drop=True)

            df = data.copy()

            # Resolve parameter overrides (training_input > config > defaults)
            regime_cfg = (self.config or {}).get("regime_classification", {}) if isinstance(self.config, dict) else {}
            overrides = (training_input or {}).get("regime_params", {}) if isinstance(training_input, dict) else {}

            ema_fast = overrides.get("ema_fast", regime_cfg.get("ema_fast", 21))
            ema_slow = overrides.get("ema_slow", regime_cfg.get("ema_slow", 55))
            adx_period = overrides.get("adx_period", regime_cfg.get("adx_period", 14))
            adx_trend_threshold = overrides.get(
                "adx_trend_threshold", regime_cfg.get("adx_trend_threshold", 25.0)
            )
            adx_sideways_threshold = overrides.get(
                "adx_sideways_threshold", regime_cfg.get("adx_sideways_threshold", 20.0)
            )
            ema_sep_min_ratio = overrides.get(
                "ema_sep_min_ratio", regime_cfg.get("ema_sep_min_ratio", 0.0)
            )

            # Optional auto-calibration to hit target SIDEWAYS band
            target_range = overrides.get(
                "target_sideways_range", regime_cfg.get("target_sideways_range", [0.25, 0.35])
            )  # default 25–35%
            auto_calibrate = overrides.get(
                "auto_calibrate_sideways",
                regime_cfg.get("auto_calibrate_sideways", True),
            )
            max_calibration_iters = int(
                overrides.get(
                    "max_calibration_iters", regime_cfg.get("max_calibration_iters", 6)
                )
            )
            # Step sizes
            adx_trend_step = float(
                overrides.get("adx_trend_step", regime_cfg.get("adx_trend_step", 2.0))
            )
            adx_sideways_step = float(
                overrides.get(
                    "adx_sideways_step", regime_cfg.get("adx_sideways_step", 1.0)
                )
            )
            ema_sep_step = float(
                overrides.get("ema_sep_step", regime_cfg.get("ema_sep_step", 0.0005))
            )

            def classify_and_ratio(
                fast: int,
                slow: int,
                adx_p: int,
                adx_tr: float,
                adx_sw: float,
                ema_sep_min: float,
            ) -> tuple[list[str], list[float], float]:
                r, c = classify_regime_series(
                    df,
                    ema_fast=fast,
                    ema_slow=slow,
                    adx_period=adx_p,
                    adx_trend_threshold=adx_tr,
                    adx_sideways_threshold=adx_sw,
                    ema_sep_min_ratio=ema_sep_min,
                )
                if len(r) == 0:
                    return r, c, 0.0
                sideways_ratio = float(np.mean(np.array(r, dtype=object) == "SIDEWAYS"))
                return r, c, sideways_ratio

            regimes, confidences, sideways_ratio = classify_and_ratio(
                ema_fast,
                ema_slow,
                adx_period,
                adx_trend_threshold,
                adx_sideways_threshold,
                ema_sep_min_ratio,
            )

            calibrated_params = {
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "adx_period": adx_period,
                "adx_trend_threshold": adx_trend_threshold,
                "adx_sideways_threshold": adx_sideways_threshold,
                "ema_sep_min_ratio": ema_sep_min_ratio,
            }

            if auto_calibrate and target_range and isinstance(target_range, (list, tuple)) and len(target_range) == 2:
                target_low = float(target_range[0])
                target_high = float(target_range[1])
                it = 0
                while (sideways_ratio < target_low or sideways_ratio > target_high) and it < max_calibration_iters:
                    # Adjust thresholds to move ratio toward band
                    if sideways_ratio > target_high:
                        # Too much SIDEWAYS -> make trend easier
                        adx_trend_threshold = max(5.0, adx_trend_threshold - adx_trend_step)
                        adx_sideways_threshold = max(5.0, adx_sideways_threshold - adx_sideways_step)
                        ema_sep_min_ratio = max(0.0, ema_sep_min_ratio - ema_sep_step)
                    else:
                        # Too little SIDEWAYS -> make trend harder / expand sideways
                        adx_trend_threshold = min(60.0, adx_trend_threshold + adx_trend_step)
                        adx_sideways_threshold = min(
                            adx_trend_threshold - 1.0,
                            adx_sideways_threshold + adx_sideways_step,
                        )
                        ema_sep_min_ratio = min(0.02, ema_sep_min_ratio + ema_sep_step)

                    # Enforce relationship
                    adx_sideways_threshold = min(adx_sideways_threshold, adx_trend_threshold - 1.0)

                    regimes, confidences, sideways_ratio = classify_and_ratio(
                        ema_fast,
                        ema_slow,
                        adx_period,
                        adx_trend_threshold,
                        adx_sideways_threshold,
                        ema_sep_min_ratio,
                    )

                    it += 1

                calibrated_params = {
                    "ema_fast": ema_fast,
                    "ema_slow": ema_slow,
                    "adx_period": adx_period,
                    "adx_trend_threshold": adx_trend_threshold,
                    "adx_sideways_threshold": adx_sideways_threshold,
                    "ema_sep_min_ratio": ema_sep_min_ratio,
                    "target_sideways_range": [target_low, target_high],
                    "achieved_sideways_ratio": sideways_ratio,
                    "calibration_iters": it,
                }

            # Build results
            from collections import Counter
            regime_counts = Counter(regimes)

            formatted_results = {
                "symbol": symbol,
                "exchange": exchange,
                "classification_date": datetime.utcnow().isoformat(),
                "total_records": len(df),
                "regime_distribution": dict(regime_counts),
                "regime_sequence": regimes,
                "regime_transitions": [],
                "confidence_scores": confidences,
                "metadata": {
                    "classifier_version": "ema_adx_rules_v2",
                    "classification_method": "EMA_ADX_PARAMETERIZED",
                    "ema_periods": {"fast": ema_fast, "slow": ema_slow},
                    "adx": {
                        "period": adx_period,
                        "trend_threshold": adx_trend_threshold,
                        "sideways_threshold": adx_sideways_threshold,
                    },
                    "ema_sep_min_ratio": ema_sep_min_ratio,
                    "timeframe": "1h",
                    "calibrated_params": calibrated_params,
                },
            }

            # Transitions
            s_regimes = pd.Series(regimes)
            shifted = s_regimes.shift(1)
            mask = s_regimes != shifted
            transitions_df = pd.DataFrame({
                'from_regime': shifted[mask],
                'to_regime': s_regimes[mask],
                'transition_index': s_regimes.index[mask]
            })
            formatted_results["regime_transitions"] = transitions_df.to_dict('records')

            self.logger.info(
                f"Regime classification (EMA/ADX) completed. Found {len(regime_counts)} distinct regimes",
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
