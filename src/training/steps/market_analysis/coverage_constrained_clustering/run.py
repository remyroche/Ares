from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .config import CoverageClusteringConfig
from .clusterer import CoverageConstrainedClusterer


def load_hmm_artifact(input_path: str) -> Dict[str, Any]:
	p = Path(input_path)
	if not p.exists():
		raise FileNotFoundError(f"Input artifact not found: {input_path}")
	with open(p, "r") as f:
		data = json.load(f)
	return data


def main() -> None:
	parser = argparse.ArgumentParser(description="Coverage-Constrained Clustering for HMM regimes")
	parser.add_argument("--input", required=True, help="Path to JSON artifact containing hmm_regime_discovery_result")
	parser.add_argument("--output", required=True, help="Output JSON path for clustering results")
	parser.add_argument("--target_clusters", type=int, default=20)
	parser.add_argument("--min_clusters", type=int, default=15)
	parser.add_argument("--max_clusters", type=int, default=26)
	parser.add_argument("--min_coverage", type=float, default=0.90)
	parser.add_argument("--max_coverage", type=float, default=0.95)
	parser.add_argument("--min_frac", type=float, default=0.03)
	parser.add_argument("--max_frac", type=float, default=0.08)
	args = parser.parse_args()

	artifact = load_hmm_artifact(args.input)
	cfg = CoverageClusteringConfig(
		target_num_clusters=args.target_clusters,
		min_num_clusters=args.min_clusters,
		max_num_clusters=args.max_clusters,
		min_coverage=args.min_coverage,
		max_coverage=args.max_coverage,
		min_cluster_fraction=args.min_frac,
		max_cluster_fraction=args.max_frac,
	)
	clusterer = CoverageConstrainedClusterer(cfg)
	res = clusterer.cluster(artifact[cfg.hmm_artifact_key])

	output = {
		"cluster_labels": res.cluster_labels,
		"selected_regime_keys": res.selected_regime_keys,
		"metrics": res.metrics,
		"coverage_pct": res.coverage_pct,
		"noise_regime_keys": res.noise_regime_keys,
		"cluster_sizes": res.cluster_sizes,
		"cluster_size_pct": res.cluster_size_pct,
	}
	Path(args.output).parent.mkdir(parents=True, exist_ok=True)
	with open(args.output, "w") as f:
		json.dump(output, f, indent=2)

	print(json.dumps({"ok": True, "output": args.output}))


if __name__ == "__main__":
	main()

