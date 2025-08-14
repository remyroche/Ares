# src/transition/state_sequence_builder.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier
from src.utils.logger import system_logger


@dataclass
class StateBuilderConfig:
    hmm_n_states: int
    use_existing_urc_models: bool


class StateSequenceBuilder:
    """
    Build per-timestep state sequences for pre/post windows using the existing
    UnifiedRegimeClassifier (HMM) and its mapping to coarse regimes.
    """

    def __init__(self, config: dict[str, Any], exchange: str = "UNKNOWN", symbol: str = "UNKNOWN") -> None:
        self.config = config
        self.logger = system_logger.getChild("StateSequenceBuilder")
        tm_cfg = (config or {}).get("TRANSITION_MODELING", {})
        self.sb_cfg = StateBuilderConfig(
            hmm_n_states=int(tm_cfg.get("hmm_n_states", 5)),
            use_existing_urc_models=bool(tm_cfg.get("use_existing_urc_models", True)),
        )
        self.exchange = exchange
        self.symbol = symbol
        self.urc = UnifiedRegimeClassifier(config, exchange=exchange, symbol=symbol)

    async def initialize(self) -> bool:
        try:
            await self.urc.initialize()
            # If not trained or different n_states, we trigger training with current data later
            return True
        except Exception:
            return False

    def _ensure_trained(self, klines_df: pd.DataFrame) -> None:
        # Train URC if necessary or if state count differs
        try:
            desired_states = self.sb_cfg.hmm_n_states
            # Force n_states if available
            setattr(self.urc, "n_states", max(3, int(desired_states)))
            if not getattr(self.urc, "trained", False):
                # Minimal training using available history
                import asyncio
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.urc.train_complete_system(klines_df))
        except Exception as e:
            self.logger.warning(f"URC training fallback failed: {e}")

    def infer_states(self, klines_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame aligned to klines_df index with columns:
          - hmm_state_id (int)
          - regime (str: BULL/BEAR/SIDEWAYS)
        """
        if klines_df is None or klines_df.empty:
            return pd.DataFrame(index=pd.Index([], name=getattr(klines_df, 'index', None)))
        # Ensure trained
        self._ensure_trained(klines_df)
        try:
            # Reuse the URC feature pipeline to get HMM states
            features_df = self.urc._calculate_features(klines_df)
            if features_df.empty:
                return pd.DataFrame(index=klines_df.index)
            # Scale and predict HMM states
            X = features_df[[
                "log_returns","volatility_20","volume_ratio","rsi","macd","macd_signal",
                "macd_histogram","bb_position","bb_width","atr","volatility_regime","volatility_acceleration"
            ]].fillna(0)
            if self.urc.scaler is not None:
                X_scaled = self.urc.scaler.transform(X)
            else:
                from sklearn.preprocessing import StandardScaler
                self.urc.scaler = StandardScaler().fit(X)
                X_scaled = self.urc.scaler.transform(X)
            hmm_model = self.urc.hmm_model
            if hmm_model is None:
                # Train minimal HMM labeler if missing
                import asyncio
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.urc.train_hmm_labeler(klines_df))
                hmm_model = self.urc.hmm_model
            state_ids = hmm_model.predict(X_scaled)
            # Map to coarse regimes
            mapping = self.urc.state_to_regime_map or {}
            regimes = [mapping.get(int(s), "SIDEWAYS") for s in state_ids]
            out = pd.DataFrame({
                "hmm_state_id": state_ids.astype(int),
                "regime": regimes,
            }, index=klines_df.index)
            return out
        except Exception as e:
            self.logger.warning(f"State inference failed: {e}")
            return pd.DataFrame(index=klines_df.index)