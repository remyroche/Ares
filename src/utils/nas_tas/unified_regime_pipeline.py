"""Unified Regime Pipeline combining Adaptive Regime NAS and TAS outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import logging
import numpy as np
import pandas as pd

from .nas.adaptive_regime_nas import AdaptiveRegimeNAS, AdaptiveRegimeNASConfig
from .tas.regime_trading_tree_nas import (
    RegimeTradingTreeNAS,
    RegimeTradingTreeNASConfig,
)


logger = logging.getLogger(__name__)


@dataclass
class UnifiedRegimePipelineConfig:
    """Configuration for the unified NAS + TAS regime pipeline."""

    adaptive_nas_config: Optional[AdaptiveRegimeNASConfig] = None
    trading_tree_config: Optional[RegimeTradingTreeNASConfig] = None
    enable_nas: bool = True
    enable_tas: bool = True
    prefer_regime_source: str = "nas"  # "nas", "tas", or "auto"
    store_intermediate_results: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class RegimeUnifiedPipeline:
    """Runs NAS and TAS regime pipelines in a single pass."""

    def __init__(self, config: UnifiedRegimePipelineConfig):
        self.config = config
        self.logger = logger.getChild("RegimeUnifiedPipeline")

        adaptive_config = config.adaptive_nas_config or AdaptiveRegimeNASConfig()
        trading_config = config.trading_tree_config or RegimeTradingTreeNASConfig()

        self.nas_pipeline = AdaptiveRegimeNAS(adaptive_config) if config.enable_nas else None
        self.tas_pipeline = RegimeTradingTreeNAS(trading_config) if config.enable_tas else None

        self.last_results: Optional[Dict[str, Any]] = None

        self.logger.info(
            "Initialized unified regime pipeline | NAS=%s TAS=%s",
            bool(self.nas_pipeline),
            bool(self.tas_pipeline),
        )

    def run(
        self,
        market_data: pd.DataFrame,
        *,
        target_variable: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        timestamps: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """Execute NAS and TAS pipelines using the supplied market data."""

        if market_data is None or market_data.empty:
            raise ValueError("Market data is required for unified regime pipeline")

        if feature_columns is None:
            feature_columns = [
                column
                for column in market_data.columns
                if column != target_variable
            ]

        feature_frame = market_data[feature_columns].select_dtypes(include=[np.number])
        dropped_features = set(feature_columns) - set(feature_frame.columns)
        if dropped_features:
            self.logger.warning(
                "Dropped %d non-numeric feature(s) before NAS search: %s",
                len(dropped_features),
                sorted(dropped_features),
            )

        X = feature_frame.to_numpy(dtype=float, copy=False)
        y: Optional[np.ndarray] = None
        if target_variable and target_variable in market_data.columns:
            y = market_data[target_variable].to_numpy()

        timestamps_array: Optional[np.ndarray]
        if timestamps is not None:
            timestamps_array = np.asarray(timestamps)
        elif isinstance(market_data.index, pd.DatetimeIndex):
            timestamps_array = market_data.index.to_numpy()
        else:
            timestamps_array = None

        nas_results: Optional[Dict[str, Any]] = None
        tas_results: Optional[Dict[str, Any]] = None

        if self.nas_pipeline:
            self.logger.info("Running Adaptive Regime NAS search on %s samples", len(feature_frame))
            nas_results = self.nas_pipeline.search(X, y)

        if self.tas_pipeline:
            self.logger.info("Running Regime Trading Tree NAS on %s samples", len(market_data))
            regime_data = self.tas_pipeline.detect_regimes(
                market_data=market_data,
                timestamps=timestamps_array if timestamps_array is not None else np.arange(len(market_data)),
            )
            trading_data = self.tas_pipeline.generate_trading_signals(
                market_data=market_data,
                regime_data=regime_data,
            )
            tas_results = self.tas_pipeline.get_combined_results()
            tas_results.setdefault("regime_detection", regime_data)
            tas_results.setdefault("trading_signals", trading_data)

        unified_regime_assignments = self._build_regime_assignments(
            nas_results=nas_results,
            tas_results=tas_results,
            market_data=market_data,
        )

        results = {
            "success": bool(unified_regime_assignments),
            "nas": nas_results,
            "tas": tas_results,
            "regime_assignments": unified_regime_assignments,
            "metadata": {
                "n_samples": len(market_data),
                "n_features": X.shape[1],
                "target_variable": target_variable,
                "dropped_features": sorted(dropped_features),
                "prefer_regime_source": self.config.prefer_regime_source,
                **self.config.metadata,
            },
        }

        if self.config.store_intermediate_results:
            self.last_results = results

        self.logger.info(
            "Unified regime pipeline completed | success=%s source=%s",
            results["success"],
            results["regime_assignments"].get("source") if unified_regime_assignments else None,
        )

        return results

    def _build_regime_assignments(
        self,
        *,
        nas_results: Optional[Dict[str, Any]],
        tas_results: Optional[Dict[str, Any]],
        market_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Derive regime assignments from NAS/TAS outputs."""

        choices: List[Dict[str, Any]] = []

        if nas_results and "regime_detection" in nas_results:
            detection = nas_results["regime_detection"]
            predictions = np.asarray(detection.get("regime_predictions"))
            if predictions.size:
                choices.append(
                    {
                        "source": "nas",
                        "regime_predictions": predictions,
                        "regime_probabilities": detection.get("regime_probabilities"),
                        "n_regimes": len(np.unique(predictions)),
                    }
                )

        if tas_results and "regime_detection" in tas_results:
            detection = tas_results["regime_detection"]
            predictions = np.asarray(detection.get("regime_predictions"))
            if predictions.size:
                choices.append(
                    {
                        "source": "tas",
                        "regime_predictions": predictions,
                        "regime_probabilities": detection.get("regime_probabilities"),
                        "n_regimes": len(np.unique(predictions)),
                    }
                )

        if not choices:
            self.logger.warning("No regime assignments available from NAS or TAS")
            return {}

        if self.config.prefer_regime_source == "nas":
            preferred = next((choice for choice in choices if choice["source"] == "nas"), choices[0])
        elif self.config.prefer_regime_source == "tas":
            preferred = next((choice for choice in choices if choice["source"] == "tas"), choices[0])
        else:
            # Auto: prefer NAS when available, otherwise TAS
            preferred = next((choice for choice in choices if choice["source"] == "nas"), choices[0])

        preferred = dict(preferred)
        preferred["n_samples"] = len(market_data)
        return preferred
