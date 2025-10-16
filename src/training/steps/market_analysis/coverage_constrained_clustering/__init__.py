"""Coverage-Constrained Clustering for HMM regime aggregation.

This package provides tools to aggregate many fine-grained 4D HMM regimes
into ~20 coherent macro-clusters that:
- Cover 90–95% of the sample distribution
- Each cluster represents ~3–8% of total samples
- Noise/outliers kept under 5%

Primary entrypoints:
- component.CoverageConstrainedClusteringComponent: pipeline component
- run.py: CLI to run clustering over HMM discovery output
"""

from .config import CoverageClusteringConfig
# from .component import CoverageConstrainedClusteringComponent  # Temporarily commented out due to import issue

__all__ = [
	"CoverageClusteringConfig",
	"CoverageConstrainedClusteringComponent",
]
