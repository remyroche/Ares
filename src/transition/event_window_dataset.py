# src/transition/event_window_dataset.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.transition.state_sequence_builder import StateSequenceBuilder
from src.utils.logger import system_logger


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
    - Preserves secondary labels as a multi-hot vector at t=0
    - Early pruning: drop incomplete windows; optional down-sampling of near-duplicate X_pre
    - Produces tensors and RF-friendly pooled features
    """

    def __init__(self, config: dict[str, Any], exchange: str = "UNKNOWN", symbol: str = "UNKNOWN") -> None:
        self.config = config
        self.logger = system_logger.getChild("EventWindowDatasetBuilder")
        tm_cfg = (config or {}).get("TRANSITION_MODELING", {})
        self.ds_cfg = WindowDatasetConfig(
            pre_window=int(tm_cfg.get("pre_window", 60)),
            post_window=int(tm_cfg.get("post_window", 20)),
            max_events_per_label=int(tm_cfg.get("max_events_per_label", 10000)),
            duplicate_similarity_threshold=float(tm_cfg.get("early_pruning", {}).get("duplicate_similarity_threshold", 0.98)),
            downsample_near_duplicates=bool(tm_cfg.get("early_pruning", {}).get("downsample_near_duplicate_sequences", True)),
        )
        self.state_builder = StateSequenceBuilder(config, exchange=exchange, symbol=symbol)

    async def initialize(self) -> bool:
        return await self.state_builder.initialize()

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
            "log_returns","volatility_20","volume_ratio","rsi","macd","macd_signal",
            "macd_histogram","bb_position","bb_width","atr","volatility_regime","volatility_acceleration"
        ]:
            if col in seq_df.columns:
                s = pd.to_numeric(seq_df[col], errors="coerce")
                out[f"mean_{col}"] = float(np.nanmean(s.values))
                out[f"std_{col}"] = float(np.nanstd(s.values))
        return out

    def build(
        self,
        klines_df: pd.DataFrame,
        combined_df: pd.DataFrame,
        event_index: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Returns a dict with:
          - samples: list of dicts with keys {event_label, t0_time, X_pre_states, Y_post_returns, Y_post_states, multi_hot_labels, rf_features}
          - tensors: optional stacked arrays for model consumption (can be large; we keep lightweight here)
        """
        if klines_df.empty or combined_df.empty or event_index.empty:
            return {"samples": []}

        # Infer states for the entire klines_df once
        states_df = self.state_builder.infer_states(klines_df)
        if states_df.empty:
            return {"samples": []}
        # Merge numeric features if present in combined_df for RF pooling
        # Align indices
        combined_num = combined_df.reindex(klines_df.index)

        pre = self.ds_cfg.pre_window
        post = self.ds_cfg.post_window
        samples: list[dict[str, Any]] = []

        # Prepare multi-hot vector template
        all_labels = sorted({lab for lab in event_index["event_label"].unique()}.union(*event_index.get("secondary_labels", pd.Series([[]]*len(event_index))).tolist()))
        label_to_idx = {l: i for i, l in enumerate(all_labels)}

        # Early pruning: drop events without full window
        valid_events = event_index[(event_index["row_index"] >= pre) & (event_index["row_index"] < len(klines_df) - post)]
        if valid_events.empty:
            return {"samples": []}

        # Optional cap per label
        capped = []
        for lab, grp in valid_events.groupby("event_label"):
            capped.append(grp.head(self.ds_cfg.max_events_per_label))
        valid_events = pd.concat(capped).sort_values("row_index")

        # Loop and build windows
        for _, ev in valid_events.iterrows():
            i0 = int(ev["row_index"])
            t0 = klines_df.index[i0]
            # Pre window
            pre_slice = slice(i0 - pre, i0)
            post_slice = slice(i0 + 1, i0 + 1 + post)
            X_states = states_df.iloc[pre_slice]
            Y_states = states_df.iloc[post_slice]
            # Targets: returns and next states
            close = pd.to_numeric(klines_df["close"], errors="coerce").values
            ret_seq = (close[i0 + 1 : i0 + 1 + post] / close[i0] - 1.0).astype(float)
            # Multi-hot labels at t0
            mh = np.zeros(len(all_labels), dtype=np.float32)
            # include anchor and secondaries
            mh[label_to_idx[ev["event_label"]]] = 1.0
            for s in (ev.get("secondary_labels") or []):
                if s in label_to_idx:
                    mh[label_to_idx[s]] = 1.0

            # RF pooled features over pre slice using combined numeric features (when available)
            seq_numeric = combined_num.iloc[pre_slice]
            rf_feats = self._rf_pooled_features(seq_numeric)

            samples.append({
                "event_label": ev["event_label"],
                "t0_time": t0,
                "X_pre_states": X_states[["hmm_state_id","regime"]].copy(),
                "Y_post_returns": ret_seq.copy(),
                "Y_post_states": Y_states[["hmm_state_id","regime"]].copy(),
                "multi_hot_labels": mh.copy(),
                "rf_features": rf_feats,
                "weighted_intensity": float(ev.get("weighted_intensity", ev.get("intensity", 0.0))),
            })

        # Optional down-sampling of near-duplicate pre sequences using cosine similarity on rf_features vector
        if self.ds_cfg.downsample_near_duplicates and len(samples) > 1:
            kept: list[dict[str, Any]] = []
            vectors: list[np.ndarray] = []
            for s in samples:
                v = np.array(list(s["rf_features"].values()), dtype=float)
                if v.size == 0:
                    kept.append(s)
                    vectors.append(v)
                    continue
                if not vectors:
                    kept.append(s)
                    vectors.append(v)
                else:
                    sims = [self._cosine_sim(v, u) for u in vectors if u.size == v.size]
                    if sims and max(sims) >= self.ds_cfg.duplicate_similarity_threshold:
                        # skip duplicate
                        continue
                    kept.append(s)
                    vectors.append(v)
            samples = kept

        return {"samples": samples, "label_index": all_labels}