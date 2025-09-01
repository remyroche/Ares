# src/transition/inference_combiner.py

from src.utils.logger import system_logger
from typing import Any
import json
import os
from dataclasses import dataclass


@dataclass
class EnsembleConfig:
    weights: dict[str, float]
    macro_thresholds: dict[
        str, dict[str, dict[str, float]],
    ]  # regime -> timeframe -> {class: thr}
    timeframe_thresholds: dict[str, dict[str, float]]  # timeframe -> {class: thr}
    reliability_path: str | None


class TransitionInferenceCombiner:
    """
    Combine per-timeframe path_class probabilities into a single, reliability-adjusted score,
    apply macro-regime thresholds for gating, and compute an exit bias with conservative rules.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.logger = system_logger.getChild("TransitionInferenceCombiner")
        tm = (config or {}).get("TRANSITION_MODELING", {})
        ens = tm.get("timeframe_ensemble", {}) or {}
        inf = tm.get("inference", {}) or {}
        seq = tm.get("seq2seq", {}) or {}
        artifact_dir = str(
            seq.get("artifact_dir_models", "checkpoints/transition_models"),
        )
        self.cfg = EnsembleConfig(
            weights=ens.get(
                "weights",
                {"1m": 0.3, "5m": 0.3, "15m": 0.25, "30m": 0.15},
            ),
            macro_thresholds=inf.get("macro_regime_thresholds", {}),
            timeframe_thresholds=inf.get("path_class_thresholds", {}),
            reliability_path=os.path.join(artifact_dir, "reliability.json"),
        )
        self.reliability: dict[str, dict[str, float]] = self._load_reliability(
            self.cfg.reliability_path,
        )

    def _load_reliability(self, path: str | None) -> dict[str, dict[str, float]]:
        try:
            if path and os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                # Expecting {timeframe: {path_class: scale}}
                if isinstance(data, dict):
                    return {
                        str(tf): {str(k): float(v) for k, v in d.items()}
                        for tf, d in data.items()
                        if isinstance(d, dict)
                    }
        except Exception as e:
            self.logger.warning(f"Failed to load transition reliability: {e}")
        return {}

    def _apply_reliability(self, timeframe: str, cls: str, p: float) -> float:
        # Simple multiplicative scaling; can be replaced by calibrated curves later
        s = float(self.reliability.get(timeframe, {}).get(cls, 1.0))
        return max(0.0, min(1.0, p * s))

    def exit_bias(
        self,
        path_probs_1m: dict[str, float],
        _position_side: str = "long",
    ) -> dict[str, Any]:
        """
        Conservative exit logic:
          - Compute exit_bias = P(reversal) - max(P(continuation), P(beginning_of_trend)) using 1m probabilities (reliability-adjusted)
          - If P(reversal) > 0.40, recommend exit immediately
          - exit_flag True if reversal>0.40 or exit_bias>0
        """
        # Reliability-adjusted 1m
        r_cont = self._apply_reliability(
            "1m",
            "continuation",
            float(path_probs_1m.get("continuation", 0.0)),
        )
        r_bot = self._apply_reliability(
            "1m",
            "beginning_of_trend",
            float(path_probs_1m.get("beginning_of_trend", 0.0)),
        )
        r_rev = self._apply_reliability(
            "1m",
            "reversal",
            float(path_probs_1m.get("reversal", 0.0)),
        )
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
