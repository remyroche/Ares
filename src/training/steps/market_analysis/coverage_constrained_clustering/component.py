from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.training.steps.market_analysis.components.base_component import (
	BaseMarketAnalysisComponent,
	ComponentConfig,
	ComponentResult,
)
from src.utils.logger import system_logger

from .config import CoverageClusteringConfig
from .clusterer import CoverageConstrainedClusterer
from .utils import load_latest_hmm_discovery_artifact

@dataclass
class CoverageClusteringComponentConfig(ComponentConfig):
	# Extend base config with clustering specifics if needed in future
	pass

class CoverageConstrainedClusteringComponent(BaseMarketAnalysisComponent):
	"""Aggregate 4D HMM regimes into ~20 macro-clusters with coverage/size constraints."""

	def __init__(self, config: CoverageClusteringComponentConfig | None = None) -> None:
		super().__init__(config)
		self.logger = system_logger.getChild("CoverageClustering")

	def get_required_artifacts(self) -> List[str]:
		return ["coverage_constrained_clustering_result"]

	async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
		try:
			if not isinstance(data, dict):
				raise ValueError("Input data must be a dict containing HMM discovery artifacts")
			cfg_dict: Dict[str, Any] = pipeline_state.get("coverage_clustering_config", {})
			cfg = CoverageClusteringConfig(**cfg_dict) if cfg_dict else CoverageClusteringConfig()
			clusterer = CoverageConstrainedClusterer(cfg)

			# Load HMM artifact natively if not provided in memory
			hmm_key = cfg.hmm_artifact_key
			if hmm_key in data and isinstance(data[hmm_key], dict):
				hmm_artifact = data[hmm_key]
			else:
				# Attempt to auto-load the latest artifact from artifacts/<session>/
				loaded = load_latest_hmm_discovery_artifact(
					base_dir="artifacts",
					symbol=getattr(self.config, "symbol", None),
					exchange=getattr(self.config, "exchange", None),
					timeframe=getattr(self.config, "timeframe", None),
				)
				if loaded is None:
					raise ValueError("Could not locate latest hmm_regime_discovery_result artifact")
				hmm_artifact = loaded

			outputs = clusterer.cluster(hmm_artifact)

			artifacts = {
				"coverage_constrained_clustering_result": {
					"cluster_labels": outputs.cluster_labels,
					"selected_regime_keys": outputs.selected_regime_keys,
					"metrics": outputs.metrics,
					"coverage_pct": outputs.coverage_pct,
					"noise_regime_keys": outputs.noise_regime_keys,
					"cluster_sizes": outputs.cluster_sizes,
					"cluster_size_pct": outputs.cluster_size_pct,
				}
			}

			metadata = {
				"symbol": getattr(self.config, "symbol", None),
				"timeframe": getattr(self.config, "timeframe", None),
				"target_clusters": cfg.target_num_clusters,
				"min_cluster_fraction": cfg.min_cluster_fraction,
				"max_cluster_fraction": cfg.max_cluster_fraction,
				"target_coverage": cfg.target_coverage,
			}

			return ComponentResult(success=True, artifacts=artifacts, metadata=metadata)
		except Exception as e:
			self.logger.exception(f"Coverage-constrained clustering failed: {e}")
			return ComponentResult(success=False, artifacts={}, error_message=str(e))
