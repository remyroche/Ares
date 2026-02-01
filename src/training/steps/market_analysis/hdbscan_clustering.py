"""Legacy HDBSCAN regime discovery step.

This module exists to preserve backward compatibility with older
`ares_launcher` invocations that import
`src.training.steps.market_analysis.hdbscan_clustering`.
It simply forwards execution to the newer `RegimeClusteringStep` so
callers can keep running without modification.
"""

from __future__ import annotations

from typing import Any, Dict

from src.training.steps.base_step import BaseStep
from src.utils.tprint import tprint_info, tprint_warning

from .regime_clustering_step import RegimeClusteringStep


class HDBSCANRegimeDiscoveryStep(BaseStep):
    """Compatibility wrapper around ``RegimeClusteringStep``."""

    def __init__(self, step_name: str = "hdbscan_regime_discovery"):
        super().__init__(step_name)
        self._delegate = RegimeClusteringStep("regime_clustering")

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        tprint_info("🔄 HDBSCANRegimeDiscoveryStep delegating to RegimeClusteringStep")
        tprint_warning(
            "   ⚠️ hdbscan_regime_discovery is deprecated; please use "
            "regime_clustering instead."
        )
        return await self._delegate.execute(config)
