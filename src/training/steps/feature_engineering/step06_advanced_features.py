from __future__ import annotations

"""Step 06: Advanced Feature Engineering (standard path for orchestrator).

This module provides a minimal-yet-functional implementation of the advanced
feature engineering step with proper async usage, BaseStep interface, logging,
and safe error handling. It consumes labeled data from step 05 and produces
engineered feature parquet files and summary metadata, matching step_config.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.training.base_step import BaseStep
from src.utils.decorators.errors import handles_errors


class AdvancedFeatureEngineeringStep(BaseStep):
    """Advanced feature engineering using the standardized BaseStep."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config, "06", "advanced_feature_engineering")

        self.feature_config: Dict[str, Any] = config.get(
            "feature_engineering",
            {
                "enable_wavelets": False,
                "enable_multi_timeframe": False,
                "timeframes": ["5m", "15m", "1h"],
                "chunk_size": 300_000,
            },
        )

    def _initialize_step(self) -> None:
        if self.logger:
            self.logger.info("✅ Step06 feature engineering initialized")

    def validate_inputs(
        self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if "labeled_data" not in pipeline_state:
            errors.append("Missing 'labeled_data' from previous step (05)")
        else:
            df = pipeline_state["labeled_data"]
            required = ["open", "high", "low", "close", "volume"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                errors.append(f"Missing required OHLCV columns: {missing}")
        return (len(errors) == 0, errors)

    @handles_errors(exceptions=(Exception,), default_return={"success": False}, context="advanced_feature_engineering")
    async def execute_logic(
        self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        labeled: pd.DataFrame = pipeline_state["labeled_data"]

        if self.logger:
            self.logger.info(
                f"🔧 Engineering features for labeled dataset: rows={len(labeled)} cols={len(labeled.columns)}"
            )

        features = self._build_basic_features(labeled)
        features = self._finalize_features(features, labeled)

        # Split features train/val by simple ratio if no index provided
        split_index = int(len(features) * 0.8)
        train_features = features.iloc[:split_index]
        val_features = features.iloc[split_index:]

        # Persist outputs to match step_config expectations
        exchange = training_input.get("exchange", "BINANCE")
        symbol = training_input.get("symbol", "ETHUSDT")
        base_timeframe = training_input.get("timeframe", "1m")
        data_dir = Path(training_input.get("data_dir", "data/training"))
        data_dir.mkdir(parents=True, exist_ok=True)

        train_path = data_dir / f"{exchange}_{symbol}_{base_timeframe}_features_train.parquet"
        val_path = data_dir / f"{exchange}_{symbol}_{base_timeframe}_features_val.parquet"
        train_features.to_parquet(train_path, compression="snappy")
        val_features.to_parquet(val_path, compression="snappy")

        if self.logger:
            self.logger.info(
                f"✅ Saved features | train={len(train_features)} val={len(val_features)} n_features={train_features.shape[1]}"
            )

        # Update pipeline_state with DataFrames for downstream steps and include file paths
        pipeline_state["engineered_data"] = {
            "train": train_features,
            "val": val_features,
        }
        pipeline_state["engineered_feature_paths"] = {
            "train": str(train_path),
            "val": str(val_path),
        }
        pipeline_state["feature_statistics"] = self._compute_feature_statistics(train_features)
        pipeline_state["selected_features"] = list(train_features.columns)
        pipeline_state["feature_reports"] = {"summary": f"features={train_features.shape[1]}"}

        return pipeline_state

    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if "engineered_data" not in pipeline_state:
            errors.append("engineered_data missing in pipeline_state")
            return (False, errors)
        info = pipeline_state["engineered_data"]
        for key in ("train", "val"):
            p = Path(info.get(key, ""))
            if not p.exists():
                errors.append(f"Missing features file: {key} -> {p}")
        return (len(errors) == 0, errors)

    def get_required_inputs(self) -> list:
        return ["labeled_data"]

    def get_produced_outputs(self) -> list:
        return [
            "engineered_data",
            "feature_statistics",
            "selected_features",
            "feature_reports",
        ]

    def get_dependencies(self) -> list:
        return ["05"]

    def _build_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=data.index)

        # Returns-based features
        features["ret_1"] = data["close"].pct_change()
        features["ret_5"] = data["close"].pct_change(5)
        features["ret_20"] = data["close"].pct_change(20)

        # Moving averages and ratios
        for period in (10, 20, 50):
            ma = data["close"].rolling(period).mean()
            features[f"sma_{period}"] = ma
            features[f"sma_{period}_ratio"] = data["close"] / (ma.replace(0, np.nan))

        # Volatility proxies
        features["vol_20"] = data["close"].pct_change().rolling(20).std()
        features["hl_spread"] = (data["high"] - data["low"]).astype(float)
        features["bb_width_20"] = (
            (data["close"].rolling(20).mean() + 2 * data["close"].rolling(20).std())
            - (data["close"].rolling(20).mean() - 2 * data["close"].rolling(20).std())
        )

        # Volume features
        if "volume" in data.columns:
            vma = data["volume"].rolling(20).mean()
            features["volume_sma_20"] = vma
            with np.errstate(divide="ignore", invalid="ignore"):
                features["volume_ratio_20"] = data["volume"] / vma.replace(0, np.nan)

        # Interactions
        if {"ret_1", "volume_ratio_20"}.issubset(features.columns):
            features["price_volume_int"] = features["ret_1"] * features["volume_ratio_20"]

        # Clean
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features.fillna(method="ffill", inplace=True)
        features.fillna(method="bfill", inplace=True)
        features.fillna(0, inplace=True)

        return features

    def _finalize_features(self, features: pd.DataFrame, labeled: pd.DataFrame) -> pd.DataFrame:
        # Retain label columns if present to keep alignment
        output = pd.concat([labeled[[c for c in labeled.columns if "label" in c.lower()]].copy() if any(
            "label" in c.lower() for c in labeled.columns
        ) else pd.DataFrame(index=labeled.index), features], axis=1)
        return output

    def _compute_feature_statistics(self, features: pd.DataFrame) -> Dict[str, Any]:
        numeric = features.select_dtypes(include=[np.number])
        return {
            "n_samples": int(len(features)),
            "n_features": int(numeric.shape[1]),
            "missing_values": {k: int(v) for k, v in numeric.isna().sum().to_dict().items()},
        }

