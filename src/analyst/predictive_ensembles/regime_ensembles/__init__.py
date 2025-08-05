# src/analyst/predictive_ensembles/regime_ensembles/__init__.py

# Import all specific ensemble classes here for easier access
from .base_ensemble import BaseEnsemble
from .bear_trend_ensemble import BearTrendEnsemble
from .bull_trend_ensemble import BullTrendEnsemble
from .sideways_range_ensemble import SidewaysRangeEnsemble
from .volatile_regime_ensemble import VolatileRegimeEnsemble

__all__ = [
    "BaseEnsemble",
    "BullTrendEnsemble",
    "BearTrendEnsemble",
    "SidewaysRangeEnsemble",
    "VolatileRegimeEnsemble",
]
