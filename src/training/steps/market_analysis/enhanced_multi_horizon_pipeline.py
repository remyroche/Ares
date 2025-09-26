"""High level helper that orchestrates multi horizon optimisation.

The original module attempted to run a complex pipeline that relied on many
unavailable submodules.  The refactored version focuses on combining the new
:mod:`automatic_timeframe_optimizer` results with a minimal contract that can be
used by higher level orchestration code.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .automatic_timeframe_optimizer import (
    AutomaticTimeframeOptimizer,
    ModelType,
    OptimizationError,
    OptimizationResult,
)

logger = logging.getLogger("EnhancedMultiHorizonPipeline")


@dataclass
class EnhancedPipelineConfig:
    """Runtime configuration for :class:`EnhancedMultiHorizonPipeline`."""

    enable_automatic_optimization: bool = True
    optimize_for_analyst: bool = True
    optimize_for_tactician: bool = True
    save_optimization_results: bool = True
    optimization_output_dir: Path = field(default_factory=lambda: Path("optimization_results"))


class EnhancedMultiHorizonPipeline:
    """Coordinate timeframe optimisation for Analyst and Tactician models."""

    def __init__(self, config: Optional[EnhancedPipelineConfig] = None) -> None:
        self.config = config or EnhancedPipelineConfig()
        self.optimizer = AutomaticTimeframeOptimizer()

    def execute(
        self, data: pd.DataFrame, model_type: str = "both"
    ) -> Dict[str, OptimizationResult]:
        if not isinstance(data, pd.DataFrame):
            raise OptimizationError("data must be a pandas DataFrame")
        if data.empty:
            raise OptimizationError("data must contain market records")

        if not self.config.enable_automatic_optimization:
            logger.info("Automatic optimisation disabled – returning defaults")
            return {}

        model_mapping = {
            "analyst": [ModelType.ANALYST],
            "tactician": [ModelType.TACTICIAN],
            "both": [ModelType.ANALYST, ModelType.TACTICIAN],
        }
        if model_type.lower() not in model_mapping:
            raise OptimizationError(f"Unknown model_type '{model_type}'")

        selected_models = [
            enum_model
            for enum_model in model_mapping[model_type.lower()]
            if (
                (enum_model is ModelType.ANALYST and self.config.optimize_for_analyst)
                or (enum_model is ModelType.TACTICIAN and self.config.optimize_for_tactician)
            )
        ]

        results: Dict[str, OptimizationResult] = {}
        for enum_model in selected_models:
            result = self.optimizer.optimize_for_model(enum_model, data)
            results[enum_model.value] = result

        if self.config.save_optimization_results:
            self._save_results(results)

        return results

    def execute_enhanced_labeling_step(
        self,
        data: pd.DataFrame,
        regime_labels: Optional[pd.Series] = None,
        config: Optional[Dict[str, object]] = None,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        timeframe: Optional[str] = None,
        mode: str = "full",
        features: Optional[Dict[str, object]] = None,
        model_type: str = "both",
        enable_optimization: Optional[bool] = None,
    ) -> Dict[str, Dict[str, object]]:
        """Compatibility wrapper that mirrors the historic API."""

        if enable_optimization is False:
            return {}

        results = self.execute(data, model_type=model_type)
        formatted: Dict[str, Dict[str, object]] = {}
        for name, result in results.items():
            formatted[name] = {
                "status": "success",
                "optimization_metadata": result.to_dict(),
                "artifacts": {},
            }
        if formatted:
            formatted["combined"] = {
                "status": "success",
                "optimization_metadata": {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "mode": mode,
                },
            }
        return formatted

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _save_results(self, results: Dict[str, OptimizationResult]) -> None:
        output_dir = Path(self.config.optimization_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        payload = {name: result.to_dict() for name, result in results.items()}
        output_path = output_dir / "timeframe_optimization.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Stored optimisation summary at %s", output_path)


def execute_enhanced_multi_horizon_labeling(
    data: pd.DataFrame,
    symbol: Optional[str] = None,
    exchange: Optional[str] = None,
    timeframe: Optional[str] = None,
    mode: str = "full",
    model_type: str = "both",
    enable_optimization: bool = True,
    config: Optional[EnhancedPipelineConfig] = None,
) -> Dict[str, Dict[str, object]]:
    """Convenience wrapper mirroring the legacy module level API."""

    pipeline = EnhancedMultiHorizonPipeline(config)
    if not enable_optimization:
        return {}
    return pipeline.execute_enhanced_labeling_step(
        data=data,
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        mode=mode,
        model_type=model_type,
        enable_optimization=enable_optimization,
    )


__all__ = [
    "EnhancedMultiHorizonPipeline",
    "EnhancedPipelineConfig",
    "execute_enhanced_multi_horizon_labeling",
]
