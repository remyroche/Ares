"""Layer 2/3 feature builder integrating Hive predictions, OHLCV data,
VectorBT-based transforms, and ATR normalization.

This module provides a minimal, opinionated feature construction pipeline
for committee models (Layer 2 and Layer 3).

Responsibilities:
- Load committee training targets and base-layer predictions from Hive-partitioned
  storage ("specialists" / "base_models" / "meta_layer").
- Load OHLCV data via KlinesParquetManager.
- Construct a compact feature matrix (pandas.DataFrame) aligned with the
  target index, including:
  - price/volume context features;
  - ATR-normalized spatial features when available;
  - simple disagreement / meta-features from base predictions.

This is deliberately lightweight and focused: it does *not* attempt to
replicate the full Analyst/Tactician feature-generation stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.utils.hive_partitioned_predictions import HivePartitionedReader
from src.utils.data.klines_parquet import KlinesParquetManager
from src.utils.feature_common.atr_normalization import (
    atr_normalize,
    calculate_atr,
)


logger = system_logger.getChild("Layer2FeatureBuilder")


@dataclass
class Layer2FeatureBuilderConfig:
    """Configuration for Layer 2/3 feature builder.

    This is intentionally simple. Higher-level training code is expected to
    handle walk-forward splits, label construction, and any advanced
    feature-generation logic.
    """

    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "15m"

    # Hive layer that provides base predictions used as features
    base_layer_name: str = "specialists"

    # Which prediction columns to pull from Hive and use as raw features
    prediction_feature_cols: Tuple[str, ...] = ("prob_long", "prob_short")

    # ATR settings
    atr_window: int = 14


class Layer2FeatureBuilder:
    """Build compact features for Layer 2/3 committee training.

    High-level API:
        builder = Layer2FeatureBuilder(config)
        X, y = builder.build_features(start, end, labels_df)

    where `labels_df` is a DataFrame indexed by timestamp containing the
    committee training targets (e.g., meta labels or regime outcomes).
    """

    def __init__(self, config: Optional[Layer2FeatureBuilderConfig] = None) -> None:
        self.config = config or Layer2FeatureBuilderConfig()
        self._hive_reader = HivePartitionedReader(layer_name=self.config.base_layer_name)
        # KlinesParquetManager in src.utils.data.klines_parquet expects only a
        # base data directory and exchange; symbol and timeframe are passed at
        # read-time via read_data.
        self._klines_manager = KlinesParquetManager(
            data_dir="historical_data",
            exchange=self.config.exchange,
        )
        logger.info(
            f"Initialized Layer2FeatureBuilder for {self.config.symbol} "
            f"[{self.config.timeframe}] using layer='{self.config.base_layer_name}'"
        )

    def build_features(
        self,
        start: datetime,
        end: datetime,
        labels: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Construct feature matrix X and aligned target y.

        Args:
            start: inclusive start datetime for training window.
            end: inclusive end datetime for training window.
            labels: DataFrame with at least a `target` column, indexed by
                timestamp at the committee resolution.

        Returns:
            (X, y) where X and y are aligned on index and contain only
            finite values.
        """

        if "target" not in labels.columns:
            raise ValueError("labels DataFrame must contain a 'target' column")

        logger.info(
            f"Building Layer2/3 features from {start} to {end} "
            f"for {self.config.symbol} [{self.config.timeframe}]"
        )

        # 1) Load base-layer predictions from Hive
        preds = self._load_base_predictions(start, end)

        # 2) Load OHLCV data for context / ATR
        ohlcv = self._load_ohlcv(start, end)

        # 3) Join everything to label index and construct features
        X = self._assemble_features(labels.index, preds, ohlcv)

        # 4) Align and clean target
        y = labels["target"].copy()
        X, y = self._align_and_clean(X, y)

        return X, y

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_base_predictions(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Load base predictions from Hive as potential features."""
        df = self._hive_reader.load_recent_predictions(
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Keep only configured prediction feature columns if present
        keep_cols = [c for c in self.config.prediction_feature_cols if c in df.columns]
        if not keep_cols:
            logger.warning(
                "No configured prediction feature columns found in Hive data; "
                "committee will be trained without prediction-based features."
            )
            return pd.DataFrame(index=df.index)

        return df[keep_cols]

    def _load_ohlcv(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Load OHLCV data from parquet storage."""
        df = self._klines_manager.read_data(
            symbol=self.config.symbol,
            interval=self.config.timeframe,
            start_date=start,
            end_date=end,
            data_type="processed",
        )

        if df is None or df.empty:
            raise ValueError(
                f"No OHLCV data available for {self.config.symbol} "
                f"[{self.config.timeframe}] between {start} and {end}"
            )

        # Ensure a DatetimeIndex for downstream alignment
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"OHLCV data missing required columns: {missing}")

        return df

    def _assemble_features(
        self,
        label_index: pd.DatetimeIndex,
        preds: pd.DataFrame,
        ohlcv: pd.DataFrame,
    ) -> pd.DataFrame:
        """Join predictions + OHLCV and construct a compact feature set."""
        # Reindex both sources on union index and then align to labels
        combined = pd.concat(
            [
                preds.rename(columns=lambda c: f"pred_{c}"),
                self._build_price_volume_features(ohlcv),
            ],
            axis=1,
        )

        # ATR-normalized spatial features (e.g., true range)
        atr = calculate_atr(
            high=ohlcv["high"],
            low=ohlcv["low"],
            close=ohlcv["close"],
            window=self.config.atr_window,
        )
        spatial = (ohlcv["high"] - ohlcv["low"]).to_frame("range")
        spatial_norm = atr_normalize(
            spatial,
            high=ohlcv["high"],
            low=ohlcv["low"],
            close=ohlcv["close"],
            window=self.config.atr_window,
        )
        spatial_norm = spatial_norm.add_prefix("atr_norm_")

        combined = pd.concat([combined, spatial_norm], axis=1)

        # Align to label index
        combined = combined.reindex(label_index).sort_index()
        return combined

    def _build_price_volume_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Construct simple price/volume context features.

        This keeps things intentionally low-dimensional and numerically stable.
        """
        df = pd.DataFrame(index=ohlcv.index)

        close = ohlcv["close"]
        volume = ohlcv["volume"]

        df["ret_1"] = close.pct_change().fillna(0.0)
        df["ret_log_1"] = np.log1p(df["ret_1"]).replace([np.inf, -np.inf], 0.0)

        df["rolling_vol_20"] = df["ret_1"].rolling(20, min_periods=5).std().fillna(0.0)

        vol_rolling = volume.rolling(20, min_periods=5).mean()
        df["volume_rel_20"] = (volume / vol_rolling).replace([np.inf, -np.inf], 0.0).fillna(1.0)

        return df

    def _align_and_clean(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Align X/y on index and drop non-finite rows."""
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx].copy()
        y = y.loc[common_idx].copy()

        mask = np.isfinite(y.values) & np.all(np.isfinite(X.values), axis=1)
        X_clean = X[mask]
        y_clean = y[mask]

        if X_clean.empty:
            raise ValueError("No finite samples available after alignment and cleaning")

        return X_clean, y_clean
