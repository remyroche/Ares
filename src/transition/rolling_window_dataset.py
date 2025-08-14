# src/transition/rolling_window_dataset.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.transition.state_sequence_builder import StateSequenceBuilder
from src.transition.path_targets import PathTargetEngineer
from src.utils.logger import system_logger


FEATURE_POOL_COLUMNS = [
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
    include_direction_horizons: List[int]
    max_samples: int | None


class RollingWindowDatasetBuilder:
    """
    Build rolling, triggerless pre/post windows centered at every timestep t (no label trigger).
    Outputs samples with:
    - X_pre_states, X_pre_numeric (pooled compact features)
    - Y_post_returns (vector), path_class at t (computed from post window)
    - Direction targets per horizon H: up/down over next H bars and sum of returns (regression)
    - End-of-trend style target: any of {end_of_trend, reversal} within next J bars
    - Onset-of-trend style target: any of {beginning_of_trend} within next K bars
    """

    def __init__(self, config: dict[str, Any], exchange: str = "UNKNOWN", symbol: str = "UNKNOWN") -> None:
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
        self.state_builder = StateSequenceBuilder(config, exchange=exchange, symbol=symbol)
        self.path_target = PathTargetEngineer(config)

    async def initialize(self) -> bool:
        return await self.state_builder.initialize()

    def _compact_numeric_names(self, combined_df: pd.DataFrame) -> List[str]:
        return [c for c in FEATURE_POOL_COLUMNS if c in combined_df.columns]

    def _rf_pooled_features(self, seq_df: pd.DataFrame) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for col in FEATURE_POOL_COLUMNS:
            if col in seq_df.columns:
                s = pd.to_numeric(seq_df[col], errors="coerce")
                out[f"mean_{col}"] = float(np.nanmean(s.values))
                out[f"std_{col}"] = float(np.nanstd(s.values))
        return out

    def build(self, klines_df: pd.DataFrame, combined_df: pd.DataFrame) -> dict[str, Any]:
        if klines_df is None or combined_df is None or len(klines_df) == 0:
            return {"samples": [], "numeric_feature_names": []}

        # Align and derive helper columns
        combined_df = combined_df.reindex(klines_df.index)
        numeric_cols = self._compact_numeric_names(combined_df)

        # States/regimes
        states_df = self.state_builder.infer_states(klines_df)
        if states_df.empty:
            return {"samples": [], "numeric_feature_names": numeric_cols}

        pre = self.rw_cfg.pre_window
        post = self.rw_cfg.post_window
        N = len(klines_df)
        start = pre
        end = N - post - 1
        if end <= start:
            return {"samples": [], "numeric_feature_names": numeric_cols}
        # If max_samples is set, prefer the most recent windows
        loop_start = start
        if self.rw_cfg.max_samples:
            recent_start = end - int(self.rw_cfg.max_samples) + 1
            loop_start = max(start, recent_start)

        close = pd.to_numeric(klines_df.get("close"), errors="coerce").values
        # Precompute per-t path class to enable onset/end targets later
        path_classes: List[str] = ["end_of_trend"] * (N)

        # First pass: compute core sample info and immediate path class
        samples: List[dict[str, Any]] = []
        for t in range(loop_start, end + 1):
            pre_slice = slice(t - pre, t)
            post_slice = slice(t + 1, t + 1 + post)
            X_states = states_df.iloc[pre_slice][["hmm_state_id","regime"]].copy()
            Y_states = states_df.iloc[post_slice][["hmm_state_id","regime"]].copy()
            seq_num = combined_df.iloc[pre_slice]
            rf_feats = self._rf_pooled_features(seq_num)
            # Post returns and simple horizons
            y_rets = (close[t + 1 : t + 1 + post] / close[t] - 1.0).astype(float)
            # Path class at t
            sample_tmp = {
                "X_pre_states": X_states,
                "Y_post_states": Y_states,
                "Y_post_returns": y_rets.copy(),
                "rf_features": rf_feats,
            }
            pc = self.path_target.compute_path_class(sample_tmp, klines_df)
            path_classes[t] = pc
            # Direction targets per configured horizons
            dir_targets: Dict[str, Any] = {}
            for H in self.rw_cfg.include_direction_horizons:
                if t + H < len(close):
                    R = float((close[t + H] / close[t]) - 1.0)
                    dir_targets[f"direction_up_{H}"] = int(R > 0)
                    dir_targets[f"return_{H}"] = R
                else:
                    dir_targets[f"direction_up_{H}"] = 0
                    dir_targets[f"return_{H}"] = 0.0
            # Assemble sample (onset/end filled in second pass)
            samples.append({
                "t_index": t,
                "t0_time": klines_df.index[t],
                "path_class": pc,
                "X_pre_states": X_states,
                "Y_post_states": Y_states,
                "Y_post_returns": y_rets.copy(),
                "rf_features": rf_feats,
                **dir_targets,
            })
            # No break; recent windowing handled by loop_start

        # Second pass: derive onset/end targets off path_classes
        K = self.rw_cfg.onset_horizon_bars
        J = self.rw_cfg.end_horizon_bars
        for s in samples:
            t = int(s.get("t_index", 0))
            # Onset of trend (beginning_of_trend within K bars)
            s["onset_beginning"] = int(any(pc == "beginning_of_trend" for pc in path_classes[t : min(N, t + K + 1)]))
            # End of trend (end_of_trend or reversal within J bars)
            s["end_trend"] = int(any(pc in ("end_of_trend", "reversal") for pc in path_classes[t : min(N, t + J + 1)]))

        return {"samples": samples, "numeric_feature_names": numeric_cols}