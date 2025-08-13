# src/training/steps/step4_regime_data_splitting.py

import asyncio
import os
import json
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.utils.warning_symbols import failed
from src.training.steps.unified_data_loader import get_unified_data_loader


class RegimeDataSplittingStep:
    """Step 4: Data Splitting for Training - separate data by regimes or meta-labels."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("Step4.RegimeSplit")

    async def initialize(self) -> None:
        self.logger.info("Initializing Step 4: Regime Data Splitting...")
        print("Step4Split ▶ init")
        self.logger.info("Regime Data Splitting initialized successfully")

    async def execute(self) -> dict[str, Any]:
        try:
            self.logger.info("🔄 Loading unified data for regime data splitting...")
            print("Step4Split ▶ load_unified_data")
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

            regime_basis = str(self.config.get("regime_basis", "bull_bear_sideways")).lower()
            if regime_basis == "meta_labels":
                self.logger.info("Using meta-label columns from Step 2/3 to form regimes (Method A)")
                # Load labeled data to access meta columns
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
                    c
                    for c in labeled_all.columns
                    if c.lower().startswith("sr_")
                    or c.endswith("_REGIME")
                    or c.endswith("_FORMATION")
                ]
                if "timestamp" not in labeled_all.columns and isinstance(labeled_all.index, pd.DatetimeIndex):
                    labeled_all = labeled_all.reset_index().rename(columns={"index": "timestamp"})
                if "timestamp" not in unified_data.columns and isinstance(unified_data.index, pd.DatetimeIndex):
                    unified_data = unified_data.reset_index().rename(columns={"index": "timestamp"})
                merged = unified_data.merge(
                    labeled_all[[c for c in meta_cols + ["timestamp"] if c in labeled_all.columns]],
                    on="timestamp",
                    how="left",
                )
                regime_splits: dict[str, pd.DataFrame] = {}
                for meta in meta_cols:
                    active = merged[merged[meta].fillna(0) != 0].copy()
                    if not active.empty:
                        regime_splits[meta] = active
                self.logger.info(f"✅ Built {len(regime_splits)} meta-label regime splits")
                print(f"Step4Split ▶ meta_label_splits={len(regime_splits)}")

                # NEW: Emit gating matrix from meta label strengths if available
                try:
                    strengths_path = os.path.join(self.config.get("data_dir", "data/training"), f"{self.config['exchange']}_{self.config['symbol']}_meta_strengths.parquet")
                    gating_dir = os.path.join(self.config.get("data_dir", "data/training"), "gating")
                    os.makedirs(gating_dir, exist_ok=True)
                    if os.path.exists(strengths_path):
                        strengths_df = pd.read_parquet(strengths_path)
                        if "timestamp" in strengths_df.columns:
                            gmat = strengths_df.copy()
                            gmat = gmat.set_index("timestamp").sort_index()
                            # Normalize columns to [0,1] per timestamp row-wise
                            S = gmat.abs()
                            row_sum = S.sum(axis=1).replace(0, 1.0)
                            gnorm = (S.T / row_sum).T
                            gnorm = gnorm.clip(0.0, 1.0)
                            gnorm = gnorm.reset_index()
                            gm_path = os.path.join(
                                gating_dir,
                                f"{self.config['exchange']}_{self.config['symbol']}_gating.parquet",
                            )
                            gnorm.to_parquet(gm_path, index=False)
                            self.logger.info(f"✅ Wrote gating matrix to {gm_path}")
                except Exception as ge:
                    self.logger.warning(f"Gating matrix write skipped: {ge}")
            else:
                # Load regime classification results (bull/bear/sideways)
                regime_file = f"data/training/{self.config['exchange']}_{self.config['symbol']}_regime_classification.parquet"
                self.logger.info(f"📁 Loading regime classification results from: {regime_file}")
                regime_data = self._load_regime_data_safely(regime_file)
                if regime_data is None:
                    self.logger.error(
                        f"❌ Failed to load regime classification data from {regime_file}"
                    )
                    return {"success": False, "error": "Failed to load regime classification data"}
                self.logger.info(f"✅ Loaded regime classification data: {len(regime_data)} rows")
                self.logger.info(f"   Regimes found: {regime_data['regime'].unique().tolist()}")
                regime_splits = self._split_data_by_regimes(unified_data, regime_data)

                # NEW: Emit regime weights parquet derived from confidence
                try:
                    weights = regime_data.copy()
                    weights["timestamp"] = pd.to_datetime(weights["timestamp"]).dt.floor("1H")
                    weights = weights.rename(columns={"confidence": "sample_weight"})
                    weights_path = os.path.join(
                        self.config.get("data_dir", "data/training"),
                        f"{self.config['exchange']}_{self.config['symbol']}_regime_weights.parquet",
                    )
                    weights.to_parquet(weights_path, index=False)
                    self.logger.info(f"✅ Wrote regime weights to {weights_path}")
                except Exception as we:
                    self.logger.warning(f"Regime weights write skipped: {we}")

            # Save regime splits & summary
            self._save_regime_splits(regime_splits)
            summary = self._create_regime_splitting_summary(regime_splits)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"log/step4_regime_split_{ts}.json", "w") as f:
                json.dump(summary, f, indent=2)

            self.logger.info("✅ Regime data splitting completed successfully")
            print("Step4Split ▶ done")
            return {"success": True, "regime_splits": summary}
        except Exception as e:
            self.logger.error(f"❌ Regime data splitting failed: {e}")
            return {"success": False, "error": str(e)}

    def _load_regime_data_safely(self, regime_file: str) -> Optional[pd.DataFrame]:
        try:
            for engine in (None, "pyarrow", "fastparquet", None):
                try:
                    if engine:
                        regime_data = pd.read_parquet(regime_file, engine=engine)
                    else:
                        regime_data = pd.read_parquet(regime_file)
                    regime_data = regime_data[["timestamp", "regime", "confidence"]]
                    self.logger.info(
                        f"   ✅ Successfully loaded regime data ({engine or 'default'}): {regime_data.shape}"
                    )
                    return regime_data
                except Exception as e:
                    self.logger.warning(f"   Load with engine={engine} failed: {e}")
            self.logger.error("   ❌ All loading strategies failed")
            return None
        except Exception as e:
            self.logger.error(f"   ❌ Unexpected error in regime data loading: {e}")
            return None

    def _split_data_by_regimes(self, unified_data: pd.DataFrame, regime_data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        try:
            # Normalize timestamps
            def _floor_minute(s: pd.Series) -> pd.Series:
                return pd.to_datetime(s).dt.floor("1min")

            regime_data = regime_data.copy()
            unified_data = unified_data.copy()
            regime_data["timestamp"] = _floor_minute(regime_data["timestamp"])
            unified_data["timestamp"] = _floor_minute(unified_data["timestamp"])

            ud = unified_data.copy()
            ud["timestamp_hour"] = pd.to_datetime(ud["timestamp"]).dt.floor("1h")
            rd = regime_data.copy()
            rd["timestamp_hour"] = pd.to_datetime(rd["timestamp"]).dt.floor("1h")

            merged_data = ud.merge(
                rd[["timestamp_hour", "regime", "confidence"]],
                on="timestamp_hour",
                how="left",
            ).drop(columns=["timestamp_hour"], errors="ignore")

            # Fill missing
            if merged_data["regime"].isna().any():
                missing_count = merged_data["regime"].isna().sum()
                self.logger.warning(
                    f"   {missing_count} rows have missing regime labels, filling with 'SIDEWAYS'"
                )
                merged_data["regime"] = merged_data["regime"].fillna("SIDEWAYS")
                merged_data["confidence"] = merged_data["confidence"].fillna(0.5)

            splits: dict[str, pd.DataFrame] = {}
            for regime in merged_data["regime"].unique():
                splits[regime] = merged_data[merged_data["regime"] == regime].copy()

            self.logger.info(f"✅ Successfully split data by regimes: {list(splits.keys())}")
            print(f"Step4Split ▶ regimes={list(splits.keys())}")
            return splits
        except Exception as e:
            self.logger.error(f"❌ Failed to split data by regimes: {e}")
            raise

    def _save_regime_splits(self, regime_splits: dict[str, pd.DataFrame]):
        data_dir = self.config.get("data_dir", "data/training")
        os.makedirs(data_dir, exist_ok=True)
        regime_data_dir = os.path.join(data_dir, "regime_data")
        os.makedirs(regime_data_dir, exist_ok=True)
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

    def _create_regime_splitting_summary(self, regime_splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
        total_rows = sum(len(df) for df in regime_splits.values())
        regime_stats = {}
        for regime, regime_df in regime_splits.items():
            if not regime_df.empty:
                regime_stats[regime] = {
                    "rows": len(regime_df),
                    "date_range": {
                        "start": pd.to_datetime(regime_df["timestamp"]).min().isoformat(),
                        "end": pd.to_datetime(regime_df["timestamp"]).max().isoformat(),
                    },
                    "percentage": len(regime_df) / max(1, total_rows) * 100,
                }
            else:
                regime_stats[regime] = {
                    "rows": 0,
                    "date_range": None,
                    "percentage": 0.0,
                }
        self.logger.info("📊 Regime Data Splitting Results:")
        for regime, stats in regime_stats.items():
            self.logger.info(f"   {regime}: {stats['rows']} rows ({stats['percentage']:.1f}%)")
        return regime_stats


async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    force_rerun: bool = False,
    **kwargs,
) -> bool:
    try:
        config = {"symbol": symbol, "exchange": exchange, "data_dir": data_dir, **kwargs}
        step = RegimeDataSplittingStep(config)
        await step.initialize()
        result = await step.execute()
        return result.get("success")
    except Exception as e:
        print(failed(f"Regime data splitting failed: {e}"))
        return False


if __name__ == "__main__":
    async def test():
        result = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Step 4 split test result: {result}")
    asyncio.run(test())