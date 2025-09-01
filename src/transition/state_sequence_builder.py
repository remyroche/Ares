# src/transition/state_sequence_builder.py

from sklearn.preprocessing import StandardScaler
import asyncio
from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier
from src.utils.logger import system_logger
from typing import Any
import os
from dataclasses import dataclass
import pandas as pd

@dataclass
class StateBuilderConfig:
    hmm_n_states: int
    use_existing_urc_models: bool
    cache_dir: str | None

class StateSequenceBuilder:
    """
    Build per-timestep state sequences for pre/post windows using the existing
    UnifiedRegimeClassifier (HMM) and its mapping to coarse regimes.
    """

    def __init__(
        self,
        config: dict[str, Any],
        exchange: str = "UNKNOWN",
        symbol: str = "UNKNOWN",
    ) -> None:
        self.config = config
        self.logger = system_logger.getChild("StateSequenceBuilder")
        tm_cfg = (config or {}).get("TRANSITION_MODELING", {})
        self.sb_cfg = StateBuilderConfig(
            hmm_n_states=int(tm_cfg.get("hmm_n_states", 5)),
            use_existing_urc_models=bool(tm_cfg.get("use_existing_urc_models", True)),
            cache_dir=str(
                (tm_cfg.get("cache", {}) or {}).get(
                    "cache_dir",
                    "checkpoints/transition_cache",
                ),
            ),
        )
        self.exchange = exchange
        self.symbol = symbol
        self.urc = UnifiedRegimeClassifier(config, exchange=exchange, symbol=symbol)

    def _ensure_trained(self, klines_df: pd.DataFrame) -> None:
        # Train URC if necessary or if state count differs
        try:
            desired_states = self.sb_cfg.hmm_n_states
            # Force n_states if available
            self.urc.n_states = max(3, int(desired_states))
            if not getattr(self.urc, "trained", False):
                # Minimal training using available history
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.urc.train_complete_system(klines_df))
        except Exception as e:
            self.logger.warning(f"URC training fallback failed: {e}")
