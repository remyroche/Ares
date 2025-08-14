# src/transition/event_window_dataset.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.transition.state_sequence_builder import StateSequenceBuilder
from src.utils.logger import system_logger
import os
import json


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
        self.cache_dir = str((tm_cfg.get("cache", {}) or {}).get("cache_dir", "checkpoints/transition_cache"))
        bcfg = (tm_cfg.get("barriers", {}) or {})
        self.pt_mult = float(bcfg.get("profit_take_multiplier", 0.002))
        self.sl_mult = float(bcfg.get("stop_loss_multiplier", 0.001))
        self.ctx_cfg = (tm_cfg.get("context_features", {}) or {})

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
          - samples: list of dicts with keys {event_label, t0_time, X_pre_states, X_pre_numeric, Y_post_returns, Y_post_states, multi_hot_labels, rf_features}
          - tensors: optional stacked arrays for model consumption (can be large; we keep lightweight here)
        """
        if klines_df.empty or combined_df.empty or event_index.empty:
            return {"samples": []}

        # Try dataset cache
        try:
            if self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
                key = f"dataset_{hash(tuple(klines_df.index))}_{len(event_index)}.npz"
                p = os.path.join(self.cache_dir, key)
                meta_p = os.path.join(self.cache_dir, key + ".meta.json")
                if os.path.exists(p) and os.path.exists(meta_p):
                    with open(meta_p, "r") as f:
                        meta = json.load(f)
                    # Minimal load path: return meta only; samples remain to be regenerated if needed
                    if meta.get("label_index"):
                        return {"samples": [], "label_index": meta.get("label_index", []), "numeric_feature_names": meta.get("numeric_feature_names", [])}
        except Exception:
            pass

        # Infer states for the entire klines_df once
        states_df = self.state_builder.infer_states(klines_df)
        if states_df.empty:
            return {"samples": []}
        # Merge numeric features if present in combined_df for RF pooling
        # Align indices
        combined_num = combined_df.reindex(klines_df.index)
        # Define a compact numeric feature set if present
        candidate_numeric = [
            "close_returns","volatility_20","volume_ratio","rsi","macd","macd_signal",
            "macd_histogram","bb_position","bb_width","atr","volatility_regime","volatility_acceleration"
        ]
        present_numeric = [c for c in candidate_numeric if c in combined_num.columns]

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
            # Numeric sequence (compact set)
            if present_numeric:
                X_num = pd.to_numeric(combined_num[present_numeric].iloc[pre_slice], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            else:
                X_num = np.zeros((pre, 0), dtype=float)
            # Macro context at t0 (static across pre-window for simplicity)
            if bool(self.ctx_cfg.get("enable_macro_context", True)):
                try:
                    macro_cols = []
                    # 1h EMA50 and ATR pct if available
                    if bool(self.ctx_cfg.get("include_price_over_ema50", True)) and "1h_ema_50" in combined_num.columns:
                        ema = float(combined_num["1h_ema_50"].iloc[i0])
                        price = float(klines_df["close"].iloc[i0])
                        macro_cols.append([price / ema - 1.0 if ema else 0.0])
                    if bool(self.ctx_cfg.get("include_atr_pct", True)) and "1h_atr" in combined_num.columns:
                        atr = float(combined_num["1h_atr"].iloc[i0])
                        price = float(klines_df["close"].iloc[i0])
                        macro_cols.append([atr / max(price, 1e-12)])
                    if bool(self.ctx_cfg.get("include_macro_hmm_state", True)) and "1h_hmm_state" in combined_num.columns:
                        macro_cols.append([float(combined_num["1h_hmm_state"].iloc[i0])])
                    if bool(self.ctx_cfg.get("also_include_4h", False)) and "4h_hmm_state" in combined_num.columns:
                        macro_cols.append([float(combined_num["4h_hmm_state"].iloc[i0])])
                    if macro_cols:
                        macro_vec = np.concatenate(macro_cols, axis=0).astype(float)
                        # replicate across pre timesteps
                        rep = np.repeat(macro_vec.reshape(1, -1), repeats=pre, axis=0)
                        X_num = np.concatenate([X_num, rep], axis=1) if X_num.size else rep
                except Exception:
                    pass
            # Targets: returns and next states
            close = pd.to_numeric(klines_df["close"], errors="coerce").values
            ret_seq = (close[i0 + 1 : i0 + 1 + post] / close[i0] - 1.0).astype(float)
            # Time to PT (approx using close path)
            tt_pt = -1
            for t, r in enumerate(ret_seq, start=1):
                if r >= self.pt_mult:
                    tt_pt = t
                    break
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
                "X_pre_numeric": X_num,
                "Y_post_returns": ret_seq.copy(),
                "Y_post_states": Y_states[["hmm_state_id","regime"]].copy(),
                "Y_time_to_pt": int(tt_pt),
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

        # Save minimal meta cache
        try:
            if self.cache_dir:
                with open(os.path.join(self.cache_dir, key + ".meta.json"), "w") as f:
                    json.dump({"label_index": all_labels, "numeric_feature_names": present_numeric}, f)
        except Exception:
            pass
        return {"samples": samples, "label_index": all_labels, "numeric_feature_names": present_numeric}