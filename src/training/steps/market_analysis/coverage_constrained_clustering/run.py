from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .config import CoverageClusteringConfig
from .clusterer import CoverageConstrainedClusterer
from .utils import load_latest_hmm_discovery_artifact

def load_hmm_artifact(input_path: str | None) -> Dict[str, Any]:
    if input_path:
        p = Path(input_path)
        if not p.exists():
            raise FileNotFoundError(f"Input artifact not found: {input_path}")
        with open(p, "r") as f:
            data = json.load(f)
        return data
    # Fallback: load latest from artifacts
    latest = load_latest_hmm_discovery_artifact()
    if latest is None:
        raise FileNotFoundError("Could not auto-detect latest hmm_regime_discovery_result artifact in artifacts/")
    # Wrap to match expected shape
    return {"hmm_regime_discovery_result": latest}

def main() -> None:
    parser = argparse.ArgumentParser(description="Coverage-Constrained Clustering for HMM regimes")
    parser.add_argument("--input", required=False, default=None, help="Path to JSON artifact. If omitted, auto-detect latest in artifacts/")
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

    # Convert the unified artifact format to the expected format for clustering
    # Don't filter any regimes initially - use all available data
    if "regime_statistics" in artifact:
        # Convert from unified artifact format to clustering format
        regime_assignments = []
        regime_characteristics = {}

        # Extract regime assignments from regime_statistics
        regime_counts = artifact["regime_statistics"].get("regime_counts", {})
        for regime_id, count in regime_counts.items():
            regime_assignments.extend([int(regime_id)] * count)

        # Create regime characteristics from available data
        # Since we don't have detailed regime characteristics, create basic characteristics
        for regime_id, count in regime_counts.items():
            characteristics = {
                "sample_count": count,
                "percentage": (count / len(regime_assignments)) * 100 if regime_assignments else 0,
                # Add basic characteristics that will help with clustering
                "regime_id": regime_id,
                "total_samples": len(regime_assignments)
            }
            regime_characteristics[str(regime_id)] = characteristics

        # Create the expected artifact format
        clustering_artifact = {
            "regime_assignments": regime_assignments,
            "regime_characteristics": regime_characteristics
        }

        print(f"DEBUG: clustering_artifact keys: {list(clustering_artifact.keys())}")
        print(f"DEBUG: regime_assignments length: {len(regime_assignments)}")
        print(f"DEBUG: regime_characteristics keys: {list(regime_characteristics.keys())}")
    else:
        # Use the original logic if it's already in the expected format
        clustering_artifact = artifact

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

    # Debug: Check what artifact is being passed to clusterer
    print(f"DEBUG: About to call clusterer.cluster with artifact keys: {list(clustering_artifact.keys())}")
    if "regime_assignments" in clustering_artifact:
        print(f"DEBUG: regime_assignments length: {len(clustering_artifact['regime_assignments'])}")
    if "regime_characteristics" in clustering_artifact:
        print(f"DEBUG: regime_characteristics keys: {list(clustering_artifact['regime_characteristics'].keys())}")

    res = clusterer.cluster(clustering_artifact)

    # Calculate top 20 coverage and per-cluster coverage
    top_20_coverage = res.coverage_pct
    top_20_clusters_pct = {}

    if res.cluster_size_pct:
        # Sort clusters by size and take top 20
        sorted_clusters = sorted(res.cluster_size_pct.items(), key=lambda x: x[1], reverse=True)
        top_20_clusters = sorted_clusters[:20]

        # Calculate coverage per cluster in top 20
        for cluster_id, pct in top_20_clusters:
            top_20_clusters_pct[cluster_id] = pct

    output = {
        "cluster_labels": res.cluster_labels,
        "selected_regime_keys": res.selected_regime_keys,
        "metrics": res.metrics,
        "coverage_pct": res.coverage_pct,
        "top_20_coverage": top_20_coverage,
        "top_20_clusters_pct": top_20_clusters_pct,
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
