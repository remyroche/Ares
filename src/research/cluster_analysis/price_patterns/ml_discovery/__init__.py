"""Machine-learning driven price pattern discovery utilities.

This package aggregates the richer pattern discovery implementations that live
in the sibling modules and exposes concise, backwards compatible entry points.
Historically this module shipped placeholder classes so the import paths would
resolve during a staged migration.  The real implementations already exist in
the specialised modules – this package now simply re-exports those concrete
helpers so downstream code can rely on actual behaviour without changing
imports.
"""

from .lstm_discovery import LSTMPricePatternDiscovery
from .matrix_profile_discovery import MatrixProfilePriceDiscovery
from .clustering_discovery import PriceSequenceClusteringDiscovery
from .anomaly_discovery import AnomalyPatternDiscovery as _AnomalyPatternDiscovery


# Backwards compatible aliases -------------------------------------------------

# The legacy code expects these class names.  Re-exporting the concrete
# implementations keeps the public API intact while providing the fully fledged
# behaviour from the dedicated modules.
LSTMPatternDiscovery = LSTMPricePatternDiscovery
MatrixProfileDiscovery = MatrixProfilePriceDiscovery
ClusteringPatternDiscovery = PriceSequenceClusteringDiscovery
AnomalyPatternDiscovery = _AnomalyPatternDiscovery


__all__ = [
    "LSTMPatternDiscovery",
    "MatrixProfileDiscovery",
    "ClusteringPatternDiscovery",
    "AnomalyPatternDiscovery",
]