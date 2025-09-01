# src/transition/event_window_dataset.py

from src.transition.state_sequence_builder import StateSequenceBuilder
from src.utils.logger import system_logger
from typing import Any
import json
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class WindowDatasetConfig:
    pre_window: int
    post_window: int
    max_events_per_label: int
    duplicate_similarity_threshold: float
    downsample_near_duplicates: bool


class EventWindowDatasetBuilder:
    """
    Creates a dataset of pre/post windows centered on event triggers.
    - Builds per-timestep HMM states and coarse regimes
    - Preserves secondary labels as a multi-hot vector at t , 0
    - Early pruning: drop incomplete windows; optional down-sampling of near-duplicate X_pre
    - Produces tensors and RF-friendly pooled features
    """

    def __init__(
        self,
        config: dict[str, Any],
        exchange: str = "UNKNOWN",
        symbol: str = "UNKNOWN",
    ) -> None:
        self.config = config
        self.logger = system_logger.getChild("EventWindowDatasetBuilder")
        tm_cfg = (config or {}).get("TRANSITION_MODELING", {})
        self.ds_cfg = WindowDatasetConfig(
            pre_window=int(tm_cfg.get("pre_window", 60)),
            post_window=int(tm_cfg.get("post_window", 20)),
            max_events_per_label=int(tm_cfg.get("max_events_per_label", 10000)),
            duplicate_similarity_threshold=float(
                tm_cfg.get("early_pruning", {}).get(
                    "duplicate_similarity_threshold",
                    0.98,
                ),
            ),
            downsample_near_duplicates=bool(
                tm_cfg.get("early_pruning", {}).get(
                    "downsample_near_duplicate_sequences",
                    True,
                ),
            ),
        )
        self.state_builder = StateSequenceBuilder(
            config, exchange=exchange,
            symbol=symbol,
        )
        self.cache_dir = str(
            (tm_cfg.get("cache", {}) or {}).get(
                "cache_dir",
                "checkpoints/transition_cache",
            ),
        )
        bcfg = tm_cfg.get("barriers", {}) or {}
        self.pt_mult = float(bcfg.get("profit_take_multiplier", 0.002))
        self.sl_mult = float(bcfg.get("stop_loss_multiplier", 0.001))
        self.ctx_cfg = tm_cfg.get("context_features", {}) or {}

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _rf_pooled_features(self, seq_df: pd.DataFrame) -> dict[str, float]:
        # Summaries for RandomForest: mean/std of key numeric features
        out: dict[str, float] = {}
        for col in [
            "log_returns",
            "volatility_20",
            "volume_ratio",
            "rsi",
            "macd",
            "macd_signal",
            "macd_histogram",
            "bb_position",
            "bb_width",
            "atr",
            "volatility_regime",
            "volatility_acceleration",
        ]:
            if col in seq_df.columns:
                s = pd.to_numeric(seq_df[col], errors="coerce")
                out[f"mean_{col}"] = float(np.nanmean(s.values))
                out[f"std_{col}"] = float(np.nanstd(s.values))
        return out
