"""Hybrid regime detector for live trading.

This module provides a concrete RegimeDetector implementation that
consumes specialist regime outputs (risk, alpha, liquidity, breakout/
bounce, path, SMC, mean-reversion) via the existing
``load_live_regime_outputs`` helper and exposes a simple
``predict_regime`` API for the signal generation pipeline.

The detector intentionally uses a lightweight heuristic:

- Prefer ML Risk regime probabilities when available
  (``risk_regime_*_prob``).
- Fallback to liquidity regime probabilities
  (``liquidity_regime_*_prob``).
- If neither is available, use a simple 3-regime distribution with a
  dominant neutral regime.

The output format matches what ``SignalGenerationPipeline._detect_regime``
expects: a dict with ``primary_regime``, ``regime_probabilities``, and
optional diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.utils.artifact_router import ArtifactRouter
from src.utils.tprint import tprint_info, tprint_warning
from src.trading.integration.live_regime_outputs import load_live_regime_outputs


@dataclass
class RegimeDetectorConfig:
    symbol: str
    exchange: str
    direction: str = "long"
    base_timeframe: str = "15m"
    regime_timeframe: Optional[str] = None
    strict: bool = False


class HybridRegimeDetector:
    """Regime detector that aggregates specialist regime outputs.

    This detector is designed to be used by the live
    ``SignalGenerationPipeline``. It does **not** retrain any models; it
    only reads artifacts produced by the training pipeline and
    ``final_parameters_optimization``.
    """

    def __init__(
        self,
        symbol: str,
        exchange: str,
        direction: str = "long",
        base_timeframe: str = "15m",
        regime_timeframe: Optional[str] = None,
        artifact_router: Optional[ArtifactRouter] = None,
        strict: bool = False,
    ) -> None:
        self.config = RegimeDetectorConfig(
            symbol=symbol,
            exchange=exchange,
            direction=direction,
            base_timeframe=base_timeframe,
            regime_timeframe=regime_timeframe or base_timeframe,
            strict=strict,
        )
        self.router = artifact_router or ArtifactRouter()

        tprint_info(
            "Initialized HybridRegimeDetector for "
            f"{symbol}/{exchange} [{self.config.regime_timeframe}] {direction}"
        )

    async def predict_regime(
        self,
        market_data: pd.DataFrame,
        return_probabilities: bool = True,
    ) -> Dict[str, Any]:
        """Predict current regime and probabilities from specialist outputs.

        Args:
            market_data: Rolling market data on the base timeframe. Only
                the index is used for alignment; features are ignored
                here because specialist regime outputs already encode the
                regimes.
            return_probabilities: Unused flag kept for compatibility.

        Returns:
            Dict with at least:
            - 'primary_regime': int index (used by SignalGenerationPipeline)
            - 'regime_probabilities': mapping from 'regime_i' to float
            - Optional diagnostics: 'confidence', 'regime_strength',
              'transition_probability', 'features_used'
        """

        if not isinstance(market_data, pd.DataFrame) or market_data.empty:
            raise ValueError("market_data must be a non-empty DataFrame for regime detection")

        target_index = market_data.index
        if not isinstance(target_index, pd.DatetimeIndex):
            target_index = pd.to_datetime(target_index, errors="coerce")

        # Load all specialist regime outputs aligned to this index.
        regime_df = load_live_regime_outputs(
            symbol=self.config.symbol,
            exchange=self.config.exchange,
            direction=self.config.direction,
            base_timeframe=self.config.base_timeframe,
            regime_timeframe=self.config.regime_timeframe,
            target_index=target_index,
            artifact_router=self.router,
            strict=False,
        )

        if regime_df is None or regime_df.empty:
            tprint_warning(
                "HybridRegimeDetector: no specialist regime outputs available; "
                "falling back to neutral regime."
            )
            # 3 generic regimes with neutral dominance
            probs = np.array([0.2, 0.2, 0.6], dtype=float)
            primary_idx = int(np.argmax(probs))
            regime_probs_dict = {f"regime_{i}": float(p) for i, p in enumerate(probs)}
            return {
                "primary_regime": primary_idx,
                "regime_probabilities": regime_probs_dict,
                "confidence": float(probs[primary_idx]),
                "regime_strength": float(probs[primary_idx]),
                "transition_probability": 0.5,
                "features_used": {},
            }

        last = regime_df.iloc[-1]

        # Prefer ML Risk regime probabilities when available
        risk_cols = [
            c
            for c in last.index
            if c.startswith("risk_regime_") and c.endswith("_prob")
        ]

        probs: np.ndarray

        if risk_cols:
            risk_cols_sorted = sorted(risk_cols)
            raw = np.array([float(last[c]) for c in risk_cols_sorted], dtype=float)
            if not np.isfinite(raw).any() or raw.sum() <= 0:
                raw = np.full_like(raw, 1.0 / max(len(raw), 1))
            probs = raw / raw.sum()
        else:
            # Fallback to liquidity regime probabilities
            liq_cols = [
                c
                for c in last.index
                if c.startswith("liquidity_regime_") and c.endswith("_prob")
            ]
            if liq_cols:
                liq_cols_sorted = sorted(liq_cols)
                raw = np.array([float(last[c]) for c in liq_cols_sorted], dtype=float)
                if not np.isfinite(raw).any() or raw.sum() <= 0:
                    raw = np.full_like(raw, 1.0 / max(len(raw), 1))
                probs = raw / raw.sum()
            else:
                # As a final fallback, use a simple 3-regime distribution
                probs = np.array([0.3, 0.3, 0.4], dtype=float)

        n_regimes = int(len(probs))
        if n_regimes == 0:
            probs = np.array([1.0], dtype=float)
            n_regimes = 1

        primary_idx = int(np.argmax(probs))
        regime_probs_dict = {f"regime_{i}": float(p) for i, p in enumerate(probs)}

        # Confidence is the primary regime probability; strength identical for now.
        confidence = float(probs[primary_idx])

        # Capture a small subset of features used (for diagnostics only).
        features_used: Dict[str, Any] = {}
        try:
            numeric_sample = last.select_dtypes(include=["number"]) if isinstance(last, pd.Series) else last
            if isinstance(numeric_sample, pd.Series):
                for k in list(numeric_sample.index)[:10]:
                    v = numeric_sample[k]
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        features_used[str(k)] = float(v)
        except Exception:
            features_used = {}

        return {
            "primary_regime": primary_idx,
            "regime_probabilities": regime_probs_dict,
            "confidence": confidence,
            "regime_strength": confidence,
            "transition_probability": 0.5,
            "features_used": features_used,
        }
