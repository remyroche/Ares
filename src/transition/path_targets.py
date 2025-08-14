# src/transition/path_targets.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import system_logger


@dataclass
class PathClassConfig:
    enable_beginning_of_trend: bool
    adx_sideways_threshold: float
    return_threshold: float
    onset_window_bars: int


class PathTargetEngineer:
    """
    Compute path-class targets from post-event sequences:
      - beginning_of_trend
      - continuation
      - reversal
      - end_of_trend
    Precedence: beginning_of_trend > continuation > reversal > end_of_trend
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.logger = system_logger.getChild("PathTargetEngineer")
        tm_cfg = (config or {}).get("TRANSITION_MODELING", {})
        pcfg = tm_cfg.get("path_class", {})
        self.cfg = PathClassConfig(
            enable_beginning_of_trend=bool(pcfg.get("enable_beginning_of_trend", True)),
            adx_sideways_threshold=float(pcfg.get("adx_sideways_threshold", 18)),
            return_threshold=float(pcfg.get("return_threshold", 0.001)),
            onset_window_bars=int(pcfg.get("onset_window_bars", 8)),
        )

    def compute_path_class(self, sample: dict[str, Any], klines_df: pd.DataFrame) -> str:
        # Inputs
        y_states: pd.DataFrame = sample["Y_post_states"]
        y_rets: np.ndarray = sample["Y_post_returns"]
        # Basic regime inference from states
        regimes = y_states["regime"].tolist() if "regime" in y_states.columns else []
        # Signed move proxy
        cumr = float(np.nansum(y_rets))
        # Detect early flip
        onset = min(self.cfg.onset_window_bars, len(regimes))
        first_regime = regimes[0] if regimes else "SIDEWAYS"
        any_flip_early = any(r != first_regime for r in regimes[:onset]) if regimes else False
        # Basic rules
        if self.cfg.enable_beginning_of_trend:
            # Flip early + decent cumulative move → beginning
            if any_flip_early and abs(cumr) >= self.cfg.return_threshold:
                return "beginning_of_trend"
        # continuation: no flip and decent same-direction persistence
        if regimes and all(r == first_regime for r in regimes) and abs(cumr) >= self.cfg.return_threshold:
            return "continuation"
        # reversal: flip within window with sufficient move
        if any(r != first_regime for r in regimes) and abs(cumr) >= self.cfg.return_threshold:
            return "reversal"
        # otherwise
        return "end_of_trend"