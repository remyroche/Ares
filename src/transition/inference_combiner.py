# src/transition/inference_combiner.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import json
import os

from src.utils.logger import system_logger


@dataclass
class EnsembleConfig:
    weights: Dict[str, float]
    macro_thresholds: Dict[str, Dict[str, Dict[str, float]]]  # regime -> timeframe -> {class: thr}
    timeframe_thresholds: Dict[str, Dict[str, float]]        # timeframe -> {class: thr}
    reliability_path: str | None


class TransitionInferenceCombiner:
    """
    Combine per-timeframe path_class probabilities into a single, reliability-adjusted score,
    apply macro-regime thresholds for gating, and compute an exit bias with conservative rules.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.logger = system_logger.getChild("TransitionInferenceCombiner")
        tm = (config or {}).get("TRANSITION_MODELING", {})
        ens = (tm.get("timeframe_ensemble", {}) or {})
        inf = (tm.get("inference", {}) or {})
        seq = (tm.get("seq2seq", {}) or {})
        artifact_dir = str(seq.get("artifact_dir_models", "checkpoints/transition_models"))
        self.cfg = EnsembleConfig(
            weights=ens.get("weights", {"1m": 0.3, "5m": 0.3, "15m": 0.25, "30m": 0.15}),
            macro_thresholds=inf.get("macro_regime_thresholds", {}),
            timeframe_thresholds=inf.get("path_class_thresholds", {}),
            reliability_path=os.path.join(artifact_dir, "reliability.json"),
        )
        self.reliability: Dict[str, Dict[str, float]] = self._load_reliability(self.cfg.reliability_path)

    def _load_reliability(self, path: str | None) -> Dict[str, Dict[str, float]]:
        try:
            if path and os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                # Expecting {timeframe: {path_class: scale}}
                if isinstance(data, dict):
                    return {str(tf): {str(k): float(v) for k, v in d.items()} for tf, d in data.items() if isinstance(d, dict)}
        except Exception as e:
            self.logger.warning(f"Failed to load transition reliability: {e}")
        return {}

    def _apply_reliability(self, timeframe: str, cls: str, p: float) -> float:
        # Simple multiplicative scaling; can be replaced by calibrated curves later
        s = float(self.reliability.get(timeframe, {}).get(cls, 1.0))
        out = max(0.0, min(1.0, p * s))
        return out

    def combine_probs(
        self,
        path_probs_by_timeframe: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Weighted average of path_class probabilities across configured timeframes, after reliability scaling.
        path_probs_by_timeframe: {timeframe: {"continuation": p, "reversal": p, "beginning_of_trend": p, "end_of_trend": p}}
        """
        classes = ["continuation", "reversal", "beginning_of_trend", "end_of_trend"]
        combined: Dict[str, float] = {c: 0.0 for c in classes}
        weight_sum = 0.0
        for tf, probs in path_probs_by_timeframe.items():
            w = float(self.cfg.weights.get(tf, 0.0))
            if w <= 0.0:
                continue
            weight_sum += w
            for c in classes:
                p = float(probs.get(c, 0.0))
                p_adj = self._apply_reliability(tf, c, p)
                combined[c] += w * p_adj
        if weight_sum > 0:
            for c in combined:
                combined[c] /= weight_sum
        return combined

    def gate_decision(
        self,
        combined_probs: Dict[str, float],
        timeframe: str,
        macro_regime: str | None = None,
    ) -> Dict[str, Any]:
        """
        Decide if trade is allowed given thresholds. Neutral to long/short: we only check if favorable classes exceed thresholds.
        Returns a dict with gating flag and which class triggered it.
        """
        cont = float(combined_probs.get("continuation", 0.0))
        bot = float(combined_probs.get("beginning_of_trend", 0.0))
        thr_map = self.cfg.timeframe_thresholds.get(timeframe, {})
        if macro_regime and macro_regime in self.cfg.macro_thresholds:
            thr_map = self.cfg.macro_thresholds[macro_regime].get(timeframe, thr_map)
        thr_cont = float(thr_map.get("continuation", 0.75))
        thr_bot = float(thr_map.get("beginning_of_trend", 0.75))
        allow = False
        trigger = None
        if cont >= thr_cont:
            allow, trigger = True, "continuation"
        if bot >= thr_bot and bot >= cont:
            allow, trigger = True, "beginning_of_trend"
        return {"allow_trade": allow, "trigger": trigger, "thresholds": {"continuation": thr_cont, "beginning_of_trend": thr_bot}}

    def exit_bias(
        self,
        path_probs_1m: Dict[str, float],
        position_side: str = "long",
    ) -> Dict[str, Any]:
        """
        Conservative exit logic:
          - Compute exit_bias = P(reversal) - max(P(continuation), P(beginning_of_trend)) using 1m probabilities (reliability-adjusted)
          - If P(reversal) > 0.40, recommend exit immediately
          - exit_flag True if reversal>0.40 or exit_bias>0
        """
        # Reliability-adjusted 1m
        r_cont = self._apply_reliability("1m", "continuation", float(path_probs_1m.get("continuation", 0.0)))
        r_bot = self._apply_reliability("1m", "beginning_of_trend", float(path_probs_1m.get("beginning_of_trend", 0.0)))
        r_rev = self._apply_reliability("1m", "reversal", float(path_probs_1m.get("reversal", 0.0)))
        favorable = max(r_cont, r_bot)
        adverse = r_rev
        bias = adverse - favorable
        strong_reversal = adverse > 0.40
        exit_flag = bool(strong_reversal or bias > 0)
        return {
            "exit_bias": float(bias),
            "p_reversal": float(adverse),
            "p_favorable": float(favorable),
            "strong_reversal": bool(strong_reversal),
            "exit": exit_flag,
        }