# src/training/steps/step3_regime_data_splitting.py

import asyncio
import gc
import json
import os
from src.utils.data_optimizer import regime_columns
import pickle
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    execution_error,
    failed,
    initialization_error,
    warning,
)
from src.training.steps.unified_data_loader import get_unified_data_loader


class RegimeDataSplittingStep:
    """Step 3: Data Splitting for Training - separate data by regimes or meta-labels."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger

    async def initialize(self) -> None:
        """Initialize the regime data splitting step."""
        try:
            self.logger.info("Initializing Regime Data Splitting Step...")
            self.logger.info("Regime Data Splitting Step initialized successfully")

        except Exception as e:
            self.logger.exception(
                initialization_error(
                    f"Error initializing Regime Data Splitting Step: {e}",
                ),
            )
            raise

    async def execute(self) -> dict[str, Any]:
        """Execute the regime data splitting step."""
        try:
            self.logger.info("🔄 Loading unified data for regime data splitting...")

            # Load unified data
            data_loader = get_unified_data_loader(self.config)
            unified_data = await data_loader.load_unified_data(
                symbol=self.config.get("symbol", "ETHUSDT"),
                exchange=self.config.get("exchange", "BINANCE"),
                timeframe=self.config.get("timeframe", "1m"),
                lookback_days=self.config.get("lookback_days", 180),
            )

            self.logger.info(f"✅ Loaded unified data: {len(unified_data)} rows")
            self.logger.info(
                f"   Date range: {unified_data.index.min()} to {unified_data.index.max()}"
            )
            self.logger.info(
                f"   Has aggtrades data: {hasattr(unified_data, 'aggtrades') and unified_data.aggtrades is not None}"
            )
            self.logger.info(
                f"   Has futures data: {hasattr(unified_data, 'futures') and unified_data.futures is not None}"
            )

            regime_basis = str(self.config.get("regime_basis", "bull_bear_sideways")).lower()
            if regime_basis == "meta_labels":
                self.logger.info("Using meta-label columns from Step4 to form regimes (Method A)")
                # Load Step4 labeled data to access meta columns
                labeled_files = [
                    f"data/training/{self.config['exchange']}_{self.config['symbol']}_labeled_train.parquet",
                    f"data/training/{self.config['exchange']}_{self.config['symbol']}_labeled_validation.parquet",
                    f"data/training/{self.config['exchange']}_{self.config['symbol']}_labeled_test.parquet",
                ]
                labeled_frames = []
                for p in labeled_files:
                    if os.path.exists(p):
                        try:
                            labeled_frames.append(pd.read_parquet(p))
                        except Exception:
                            pass
                if not labeled_frames:
                    self.logger.error("No labeled parquet files found for meta-label regime splitting")
                    return {"success": False, "error": "Missing labeled data"}
                labeled_all = pd.concat(labeled_frames, axis=0, ignore_index=False)
                # Identify candidate meta-label columns (heuristic: sr_*, *_REGIME, *_FORMATION)
                meta_cols = [
                    c for c in labeled_all.columns
                    if c.lower().startswith("sr_")
                    or c.endswith("_REGIME")
                    or c.endswith("_FORMATION")
                ]
                if "timestamp" not in labeled_all.columns and isinstance(labeled_all.index, pd.DatetimeIndex):
                    labeled_all = labeled_all.reset_index().rename(columns={"index": "timestamp"})
                if "timestamp" not in unified_data.columns and isinstance(unified_data.index, pd.DatetimeIndex):
                    unified_data = unified_data.reset_index().rename(columns={"index": "timestamp"})
                # Merge minute-level raw with labeled columns via timestamp
                merged = unified_data.merge(
                    labeled_all[[c for c in meta_cols + ["timestamp"] if c in labeled_all.columns]],
                    on="timestamp",
                    how="left",
                )
                # One split per meta column where value is active (non-zero)
                regime_splits = {}
                for meta in meta_cols:
                    active = merged[merged[meta].fillna(0) != 0].copy()
                    if not active.empty:
                        regime_splits[meta] = active
                self.logger.info(f"✅ Built {len(regime_splits)} meta-label regime splits")
            else:
                # Load regime classification results (bull/bear/sideways)
                regime_file = f"data/training/{self.config['exchange']}_{self.config['symbol']}_regime_classification.parquet"
                self.logger.info(
                    f"📁 Loading regime classification results from: {regime_file}"
                )
                regime_data = self._load_regime_data_safely(regime_file)
                if regime_data is None:
                    self.logger.error(
                        f"❌ Failed to load regime classification data from {regime_file}"
                    )
                    return {
                        "success": False,
                        "error": "Failed to load regime classification data",
                    }
                self.logger.info(
                    f"✅ Loaded regime classification data: {len(regime_data)} rows"
                )
                self.logger.info(
                    f"   Regimes found: {regime_data['regime'].unique().tolist()}"
                )
                # Split data by regimes
                regime_splits = self._split_data_by_regimes(unified_data, regime_data)

            # Save regime splits
            self._save_regime_splits(regime_splits)

            # Create summary
            summary = self._create_regime_splitting_summary(regime_splits)

            self.logger.info("✅ Regime data splitting completed successfully")
            return {"success": True, "regime_splits": summary}

        except Exception as e:
            self.logger.error(f"❌ Regime data splitting failed: {e}")
            return {"success": False, "error": str(e)}

    def _load_regime_data_safely(self, regime_file: str) -> Optional[pd.DataFrame]:
        """Load regime data with multiple fallback strategies to avoid segmentation faults."""
        try:
            # Strategy 1: Direct pandas read with error handling
            try:
                self.logger.info("   Trying direct pandas read...")
                # Use a simple approach without the problematic parquet utilities
                regime_data = pd.read_parquet(regime_file)
                # Select only the columns we need
                regime_data = regime_data[["timestamp", "regime", "confidence"]]
                self.logger.info(
                    f"   ✅ Successfully loaded with direct pandas read: {regime_data.shape}"
                )
                return regime_data
            except Exception as e:
                self.logger.warning(f"   Direct pandas read failed: {e}")

            # Strategy 2: Try with pyarrow engine
            try:
                self.logger.info("   Trying pyarrow engine...")
                regime_data = pd.read_parquet(regime_file, engine="pyarrow")
                regime_data = regime_data[["timestamp", "regime", "confidence"]]
                self.logger.info(
                    f"   ✅ Successfully loaded with pyarrow engine: {regime_data.shape}"
                )
                return regime_data
            except Exception as e:
                self.logger.warning(f"   Pyarrow engine failed: {e}")

            # Strategy 3: Try with fastparquet engine
            try:
                self.logger.info("   Trying fastparquet engine...")
                regime_data = pd.read_parquet(regime_file, engine="fastparquet")
                regime_data = regime_data[["timestamp", "regime", "confidence"]]
                self.logger.info(
                    f"   ✅ Successfully loaded with fastparquet engine: {regime_data.shape}"
                )
                return regime_data
            except Exception as e:
                self.logger.warning(f"   Fastparquet engine failed: {e}")

            # Strategy 4: Try reading without column specification
            try:
                self.logger.info("   Trying without column specification...")
                regime_data = pd.read_parquet(regime_file)
                # Select only the columns we need
                regime_data = regime_data[["timestamp", "regime", "confidence"]]
                self.logger.info(
                    f"   ✅ Successfully loaded without column specification: {regime_data.shape}"
                )
                return regime_data
            except Exception as e:
                self.logger.warning(
                    f"   Reading without column specification failed: {e}"
                )

            # Strategy 5: Try with memory mapping
            try:
                self.logger.info("   Trying with memory mapping...")
                regime_data = pd.read_parquet(regime_file, memory_map=True)
                regime_data = regime_data[["timestamp", "regime", "confidence"]]
                self.logger.info(
                    f"   ✅ Successfully loaded with memory mapping: {regime_data.shape}"
                )
                return regime_data
            except Exception as e:
                self.logger.warning(f"   Memory mapping failed: {e}")

            self.logger.error("   ❌ All loading strategies failed")
            return None

        except Exception as e:
            self.logger.error(f"   ❌ Unexpected error in regime data loading: {e}")
            return None

    def _split_data_by_regimes(
        self, unified_data: pd.DataFrame, regime_data: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """Split unified data by market regimes."""
        try:
            # Ensure timestamp formats match
            if pd.api.types.is_datetime64_any_dtype(regime_data["timestamp"]):
                regime_data["timestamp"] = regime_data["timestamp"].dt.floor("1min")
            else:
                regime_data["timestamp"] = pd.to_datetime(
                    regime_data["timestamp"]
                ).dt.floor("1min")

            if pd.api.types.is_datetime64_any_dtype(unified_data["timestamp"]):
                unified_data["timestamp"] = unified_data["timestamp"].dt.floor("1min")
            else:
                unified_data["timestamp"] = pd.to_datetime(
                    unified_data["timestamp"]
                ).dt.floor("1min")

            # Merge regime data with unified data (align 1h regimes to 1m data by hour)
            ud = unified_data.copy()
            if pd.api.types.is_datetime64_any_dtype(ud["timestamp"]):
                ud["timestamp_hour"] = ud["timestamp"].dt.floor("1h")
            else:
                ud["timestamp_hour"] = pd.to_datetime(ud["timestamp"]).dt.floor("1h")

            rd = regime_data.copy()
            if pd.api.types.is_datetime64_any_dtype(rd["timestamp"]):
                rd["timestamp_hour"] = rd["timestamp"].dt.floor("1h")
            else:
                rd["timestamp_hour"] = pd.to_datetime(rd["timestamp"]).dt.floor("1h")

            merged_data = ud.merge(
                rd["timestamp_hour"].to_frame().join(rd[["regime", "confidence"]]),
                on="timestamp_hour",
                how="left",
            )
            # Cleanup helper key
            if "timestamp_hour" in merged_data.columns:
                merged_data = merged_data.drop(columns=["timestamp_hour"])  # keep original minute timestamp

            # Fill missing regimes with a default
            if merged_data["regime"].isna().any():
                missing_count = merged_data["regime"].isna().sum()
                self.logger.warning(
                    f"   {missing_count} rows have missing regime labels, filling with 'SIDEWAYS'"
                )
                merged_data["regime"] = merged_data["regime"].fillna("SIDEWAYS")
                merged_data["confidence"] = merged_data["confidence"].fillna(0.5)

            # Split by regimes
            regime_splits = {}
            for regime in merged_data["regime"].unique():
                regime_df = merged_data[merged_data["regime"] == regime].copy()
                regime_splits[regime] = regime_df

            # Soft rebalance: constrain SIDEWAYS to 20-40% by reassigning borderline rows
            try:
                total_rows = len(merged_data)
                if total_rows > 0:
                    sideways_rows = (merged_data["regime"] == "SIDEWAYS").sum()
                    sideways_ratio = sideways_rows / total_rows
                    # If SIDEWAYS > 40%, reassign a portion of borderline SIDEWAYS rows based on returns sign
                    if sideways_ratio > 0.40 and all(c in merged_data.columns for c in ["close"]):
                        self.logger.warning(f"⚠️ SIDEWAYS ratio {sideways_ratio:.1%} > 40%; rebalancing to cap at 40%")
                        df = merged_data.copy()
                        # Compute 1-step returns for sign signal
                        df["_ret"] = df["close"].pct_change().fillna(0)
                        # Select a small fraction of SIDEWAYS rows with non-trivial movement to flip
                        borderline = df[(df["regime"] == "SIDEWAYS") & (df["_ret"].abs() > 0.0002)]
                        # Compute allowable flips to keep SIDEWAYS within [20%,40%]
                        current_sideways = (df["regime"] == "SIDEWAYS").sum()
                        target_min_sideways = int(0.20 * total_rows)
                        target_max_sideways = int(0.40 * total_rows)
                        # Flips needed to reach cap
                        flips_needed = max(0, current_sideways - target_max_sideways)
                        # Hard-cap flips to 20% of total to avoid drastic changes
                        max_flips = min(flips_needed, int(0.20 * total_rows))
                        if len(borderline) > max_flips:
                            borderline = borderline.tail(max_flips)
                        # Apply reassignment
                        idx_pos = borderline.index[borderline["_ret"] > 0]
                        idx_neg = borderline.index[borderline["_ret"] < 0]
                        df.loc[idx_pos, "regime"] = "BULL"
                        df.loc[idx_neg, "regime"] = "BEAR"
                        df = df.drop(columns=["_ret"]) 
                        # Rebuild splits
                        regime_splits = {}
                        for regime in df["regime"].unique():
                            regime_splits[regime] = df[df["regime"] == regime].copy()
                        self.logger.info("✅ Applied soft rebalance to regime splits")

                # Final enforcement: ensure SIDEWAYS <= 40% via hard cap downsampling if needed
                try:
                    total_rows_final = sum(len(df) for df in regime_splits.values())
                    if total_rows_final > 0 and "SIDEWAYS" in regime_splits:
                        sideways_count_final = len(regime_splits["SIDEWAYS"])
                        sideways_ratio_final = sideways_count_final / total_rows_final
                        if sideways_ratio_final > 0.40:
                            target_max_sideways = int(0.40 * total_rows_final)
                            if target_max_sideways >= 1 and sideways_count_final > target_max_sideways:
                                regime_splits["SIDEWAYS"] = regime_splits["SIDEWAYS"].sample(
                                    n=target_max_sideways, random_state=42
                                )
                                self.logger.warning(
                                    f"⚠️ Applied hard cap: downsampled SIDEWAYS from {sideways_count_final} to {target_max_sideways} rows"
                                )
                except Exception:
                    pass
            except Exception:
                pass

            self.logger.info(
                f"✅ Successfully split data by regimes: {len(regime_splits)} regimes"
            )
            return regime_splits

        except Exception as e:
            self.logger.error(f"❌ Failed to split data by regimes: {e}")
            raise

    def _save_regime_splits(self, regime_splits: dict[str, pd.DataFrame]):
        """Save regime splits to parquet files."""
        data_dir = self.config.get("data_dir", "data/training")
        os.makedirs(data_dir, exist_ok=True)
        regime_data_dir = os.path.join(data_dir, "regime_data")
        os.makedirs(regime_data_dir, exist_ok=True)

        # Save regime-specific files
        for regime, regime_df in regime_splits.items():
            if not regime_df.empty:
                regime_file = os.path.join(regime_data_dir, f"{regime}.parquet")
                try:
                    regime_df.to_parquet(regime_file, index=False)
                    self.logger.info(
                        f"✅ Saved {regime} regime data: {len(regime_df)} rows -> {regime_file}"
                    )
                except Exception as e:
                    self.logger.error(f"❌ Failed to save {regime} regime data: {e}")
            else:
                self.logger.warning(f"⚠️ No data for {regime} regime")

        # Also create train/validation/test splits for validator compatibility
        self._create_train_validation_test_splits(regime_splits, data_dir)

    def _create_train_validation_test_splits(
        self, regime_splits: dict[str, pd.DataFrame], data_dir: str
    ):
        """Create train/validation/test splits for validator compatibility."""
        try:
            # Combine all regime data
            all_data = []
            for regime_df in regime_splits.values():
                if not regime_df.empty:
                    all_data.append(regime_df)

            if not all_data:
                self.logger.warning(
                    "⚠️ No data available for train/validation/test splits"
                )
                return

            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.sort_values("timestamp").reset_index(
                drop=True
            )

            # Ensure OHLCV columns are present for downstream quality checks
            try:
                required_ohlcv = ["open", "high", "low", "close", "volume"]
                missing = [c for c in required_ohlcv if c not in combined_data.columns]
                if missing:
                    self.logger.warning(f"⚠️ Missing OHLCV columns before split creation: {missing}")
                    # Try to backfill from avg_price and trade_volume if available
                    if "avg_price" in combined_data.columns:
                        for col in ("open","high","low","close"):
                            if col not in combined_data.columns:
                                combined_data[col] = combined_data["avg_price"]
                    if "trade_volume" in combined_data.columns and "volume" not in combined_data.columns:
                        combined_data["volume"] = combined_data["trade_volume"]
                    missing_after = [c for c in required_ohlcv if c not in combined_data.columns]
                    if missing_after:
                        self.logger.warning(f"⚠️ Still missing OHLCV after best-effort backfill: {missing_after}")
            except Exception:
                pass

            # Create time-based splits (70% train, 15% validation, 15% test)
            total_rows = len(combined_data)
            train_end = int(total_rows * 0.7)
            val_end = int(total_rows * 0.85)

            train_data = combined_data.iloc[:train_end]
            validation_data = combined_data.iloc[train_end:val_end]
            test_data = combined_data.iloc[val_end:]

            # Save the splits
            exchange = self.config.get("exchange", "BINANCE")
            symbol = self.config.get("symbol", "ETHUSDT")

            train_file = os.path.join(
                data_dir, f"{exchange}_{symbol}_train_data.parquet"
            )
            validation_file = os.path.join(
                data_dir, f"{exchange}_{symbol}_validation_data.parquet"
            )
            test_file = os.path.join(data_dir, f"{exchange}_{symbol}_test_data.parquet")

            train_data.to_parquet(train_file, index=False)
            validation_data.to_parquet(validation_file, index=False)
            test_data.to_parquet(test_file, index=False)

            self.logger.info(f"✅ Created train/validation/test splits:")
            self.logger.info(f"   Train: {len(train_data)} rows -> {train_file}")
            self.logger.info(
                f"   Validation: {len(validation_data)} rows -> {validation_file}"
            )
            self.logger.info(f"   Test: {len(test_data)} rows -> {test_file}")

        except Exception as e:
            self.logger.error(f"❌ Failed to create train/validation/test splits: {e}")

    def _create_regime_splitting_summary(
        self, regime_splits: dict[str, pd.DataFrame]
    ) -> dict[str, Any]:
        """Create a summary of the regime splitting results."""
        total_rows = sum(len(df) for df in regime_splits.values())
        regime_stats = {}
        for regime, regime_df in regime_splits.items():
            if not regime_df.empty:
                regime_stats[regime] = {
                    "rows": len(regime_df),
                    "date_range": {
                        "start": regime_df["timestamp"].min().isoformat(),
                        "end": regime_df["timestamp"].max().isoformat(),
                    },
                    "percentage": len(regime_df) / total_rows * 100,
                }
            else:
                regime_stats[regime] = {
                    "rows": 0,
                    "date_range": None,
                    "percentage": 0.0,
                }

        # Log regime statistics
        self.logger.info("📊 Regime Data Splitting Results:")
        for regime, stats in regime_stats.items():
            if stats["rows"] > 0:
                self.logger.info(
                    f"   {regime}: {stats['rows']} rows ({stats['percentage']:.1f}%)"
                )
            else:
                self.logger.info(f"   {regime}: {stats['rows']} rows (0.0%)")

        return regime_stats


# For backward compatibility with existing step structure
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    force_rerun: bool = False,
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
            "force_rerun": force_rerun,
            **kwargs,
        }

        pipeline_state = {}
        result = await step.execute()

        return result.get("success")

    except Exception as e:
        print(failed(f"Regime data splitting failed: {e}"))
        return False


if __name__ == "__main__":
    # Test the step
    async def test():
        result = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Test result: {result}")

    asyncio.run(test())
