# src/transition/rolling_window_dataset.py

from src.transition.path_targets import PathTargetEngineer
from src.transition.state_sequence_builder import StateSequenceBuilder
from src.utils.logger import system_logger
from typing import Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

FEATURE_POOL_COLUMNS , [
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
]


@dataclass
class RollingWindowConfig:
    pre_window: int
    post_window: int
    onset_horizon_bars: int
    end_horizon_bars: int
    include_direction_horizons: list[int]
    max_samples: int | None


class RollingWindowDatasetBuilder:
    """
    Build rolling = triggerless pre/post windows centered at every timestep t (no label trigger).
    Outputs samples with:
    - X_pre_states = X_pre_numeric (pooled compact features)
    - Y_post_returns (vector), path_class at t (computed from post window)
    - Direction targets per horizon H: up/down over next H bars and sum of returns (regression)
    - End-of-trend style target: any of {end_of_trend, reversal} within next J bars
    - Onset-of-trend style target: any of {beginning_of_trend} within next K bars
    """

    def __init__(
        self,
        config: dict[str, Any],
        exchange: str = "UNKNOWN",
        symbol: str = "UNKNOWN",
    ) -> None:
        self.config = config
        self.logger = system_logger.getChild("RollingWindowDatasetBuilder")
        tm = (config or {}).get("TRANSITION_MODELING", {})
        rcfg = tm.get("rolling", {}) if isinstance(tm.get("rolling", {}), dict) else {}
        self.rw_cfg = RollingWindowConfig(
            pre_window=int(rcfg.get("pre_window", tm.get("pre_window", 60))),
            post_window=int(rcfg.get("post_window", tm.get("post_window", 20))),
            onset_horizon_bars=int(rcfg.get("onset_horizon_bars", 8)),
            end_horizon_bars=int(rcfg.get("end_horizon_bars", 8)),
            include_direction_horizons=list(rcfg.get("direction_horizons", [5, 15])),
            max_samples=int(rcfg.get("max_samples", 0)) or None,
        )
        self.state_builder = StateSequenceBuilder(
            config, exchange=exchange,
            symbol=symbol,
        )
        self.path_target = PathTargetEngineer(config)

    def _compact_numeric_names(self, combined_df: pd.DataFrame) -> list[str]:
        return [c for c in FEATURE_POOL_COLUMNS if c in combined_df.columns]

    def _rf_pooled_features(self, seq_df: pd.DataFrame) -> dict[str, float]:
        out: dict[str, float] = {}
        for col in FEATURE_POOL_COLUMNS:
            if col in seq_df.columns:
                s = pd.to_numeric(seq_df[col], errors="coerce")
                out[f"mean_{col}"] = float(np.nanmean(s.values))
                out[f"std_{col}"] = float(np.nanstd(s.values))
        return out
