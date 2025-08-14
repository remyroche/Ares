# src/transition/rolling_inference.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import os
import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.transition.multitask_rf import MultiTaskRandomForest


@dataclass
class RollingInferenceConfig:
    pre_window: int
    horizons: List[int]
    path_class_priority: List[str]


class RollingMTInference:
    """
    Runtime helper for the rolling MultiTask RF.
    - Loads per-head models, thresholds, and reliability
    - Builds a single-row feature vector for the latest pre-window
    - Produces entry/exit decisions and supporting probabilities
    """

    def __init__(self, config: dict[str, Any], models_dir: str, symbol: str, timeframe: str) -> None:
        self.logger = system_logger.getChild("RollingMTInference")
        tm = (config or {}).get("TRANSITION_MODELING", {})
        r = tm.get("rolling", {}) if isinstance(tm.get("rolling", {}), dict) else {}
        self.cfg = RollingInferenceConfig(
            pre_window=int(r.get("pre_window", tm.get("pre_window", 60))),
            horizons=list(r.get("direction_horizons", [5, 15])),
            path_class_priority=["beginning_of_trend", "continuation", "reversal", "end_of_trend"],
        )
        self.models_dir = models_dir
        self.prefix = f"{symbol}_{timeframe}_rolling_mtrf"
        self.models: Dict[str, Any] = {}
        self.thresholds: Dict[str, Any] = {}
        self.reliability: Dict[str, Any] = {}
        self.feature_names: List[str] = []

    def load(self) -> bool:
        try:
            models, meta, feat = MultiTaskRandomForest.load(self.models_dir, prefix=self.prefix)
            self.models = models
            self.thresholds = meta.get("thresholds", {})
            self.reliability = meta.get("reliability", {})
            self.feature_names = feat
            if not self.models:
                self.logger.warning("No models loaded for rolling inference")
                return False
            return True
        except Exception as e:
            self.logger.warning(f"Failed to load rolling models: {e}")
            return False

    def _rf_pooled_features(self, seq_df: pd.DataFrame) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for col in [
            "log_returns","volatility_20","volume_ratio","rsi","macd","macd_signal",
            "macd_histogram","bb_position","bb_width","atr","volatility_regime","volatility_acceleration",
        ]:
            if col in seq_df.columns:
                s = pd.to_numeric(seq_df[col], errors="coerce")
                out[f"mean_{col}"] = float(np.nanmean(s.values))
                out[f"std_{col}"] = float(np.nanstd(s.values))
        return out

    def _build_X_last(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        if combined_df is None or combined_df.empty:
            return pd.DataFrame(columns=self.feature_names)
        pre = self.cfg.pre_window
        if len(combined_df) < pre + 1:
            return pd.DataFrame(columns=self.feature_names)
        seq = combined_df.iloc[-pre:]
        rf = self._rf_pooled_features(seq)
        # Build DataFrame with known feature columns; fill missing
        x = {name: float(rf.get(name, 0.0)) for name in self.feature_names}
        return pd.DataFrame([x])

    def _apply_reliability(self, head: str, value: float, cls: str | None = None) -> float:
        try:
            if head == "path_class" and cls is not None:
                scale = float(self.reliability.get("path_class", {}).get(cls, 1.0))
                return float(np.clip(value * scale, 0.0, 1.0))
            else:
                scale = float(self.reliability.get(head, {}).get("positive_scale", 1.0))
                return float(np.clip(value * scale, 0.0, 1.0))
        except Exception:
            return float(np.clip(value, 0.0, 1.0))

    def _get_threshold(self, head: str, cls: str | None = None, default: float = 0.6) -> float:
        try:
            if head == "path_class" and cls is not None:
                return float(self.thresholds.get("path_class", {}).get(cls, default))
            return float(self.thresholds.get(head, default))
        except Exception:
            return float(default)

    def predict_latest(self, combined_df: pd.DataFrame) -> Dict[str, Any]:
        X = self._build_X_last(combined_df)
        if X.empty:
            return {"ready": False}
        out: Dict[str, Any] = {"ready": True}

        # Path class probabilities with reliability scaling
        pc = self.models.get("path_class")
        p_path: Dict[str, float] = {}
        if pc is not None:
            try:
                proba = pc.predict_proba(X)[0]
                classes = list(getattr(pc, "classes_", []))
                for i, c in enumerate(classes):
                    p_adj = self._apply_reliability("path_class", float(proba[i]), cls=str(c))
                    p_path[str(c)] = p_adj
                # Optionally normalize
                s = float(sum(p_path.values()))
                if s > 0:
                    p_path = {k: v / s for k, v in p_path.items()}
            except Exception:
                pass
        out["p_path_class"] = p_path

        # Heads: onset / end
        for head in ("onset_beginning", "end_trend"):
            mdl = self.models.get(head)
            if mdl is None:
                continue
            try:
                p = float(mdl.predict_proba(X)[0, 1])
                out[f"p_{head}"] = self._apply_reliability(head, p)
            except Exception:
                continue

        # Direction and returns per first horizon
        if self.cfg.horizons:
            H = int(self.cfg.horizons[0])
            head = f"direction_up_{H}"
            mdl = self.models.get(head)
            if mdl is not None:
                try:
                    p = float(mdl.predict_proba(X)[0, 1])
                    out[f"p_direction_up_{H}"] = self._apply_reliability(head, p)
                except Exception:
                    pass
            reg_head = f"return_{H}"
            rmdl = self.models.get(reg_head)
            if rmdl is not None:
                try:
                    out[f"return_{H}"] = float(rmdl.predict(X)[0])
                except Exception:
                    pass
            out["horizon"] = H

        # Next regime (multiclass)
        nr = self.models.get("next_regime")
        if nr is not None:
            try:
                proba = nr.predict_proba(X)[0]
                classes = list(getattr(nr, "classes_", []))
                p_nr = {}
                for i, c in enumerate(classes):
                    p_adj = self._apply_reliability("next_regime", float(proba[i]), cls=str(c))
                    p_nr[str(c)] = p_adj
                s = float(sum(p_nr.values()))
                if s > 0:
                    p_nr = {k: v / s for k, v in p_nr.items()}
                out["p_next_regime"] = p_nr
            except Exception:
                pass

        # Decisions
        # Entry: favor beginning/continuation crossings; fallback to onset
        allow = False
        trigger = None
        fav = 0.0
        fav_thr = 0.6
        for cls in ["beginning_of_trend", "continuation"]:
            p = float(p_path.get(cls, 0.0))
            thr = self._get_threshold("path_class", cls, default=0.6)
            if p >= thr and p > fav:
                allow, trigger, fav, fav_thr = True, cls, p, thr
        if not allow and "p_onset_beginning" in out:
            p_onset = float(out.get("p_onset_beginning", 0.0))
            thr_onset = self._get_threshold("onset_beginning", default=0.6)
            if p_onset >= thr_onset:
                allow, trigger, fav, fav_thr = True, "onset_beginning", p_onset, thr_onset
        out["allow_trade"] = allow
        out["trigger"] = trigger

        # Side and reinforcement
        side = None
        mult = 1.0
        H = out.get("horizon")
        if H is not None:
            p_up = float(out.get(f"p_direction_up_{H}", 0.0))
            thr_up = self._get_threshold(f"direction_up_{H}", default=0.6)
            side = "long" if p_up >= thr_up else "short"
            # reinforcement: scale between 0.5 and 2.0 based on how far above threshold fav is
            if allow and fav_thr < 1.0:
                mult = float(np.clip(0.5 + 1.5 * (fav - fav_thr) / max(1e-6, (1.0 - fav_thr)), 0.5, 2.0))
        out["side"] = side
        out["position_multiplier"] = mult

        # Exit logic
        p_rev = float(p_path.get("reversal", 0.0))
        thr_rev = self._get_threshold("path_class", "reversal", default=0.6)
        p_end = float(out.get("p_end_trend", 0.0))
        thr_end = self._get_threshold("end_trend", default=0.6)
        favorable = max(float(p_path.get("continuation", 0.0)), float(p_path.get("beginning_of_trend", 0.0)))
        exit_bias = float(p_rev - favorable)
        exit_flag = bool(p_rev >= thr_rev or p_end >= thr_end or exit_bias > 0)
        out["exit_flag"] = exit_flag
        out["exit_bias"] = exit_bias

        return out