"""
Labeling Utilities

This module provides comprehensive labeling utilities that were previously
part of step06. These utilities can be used by any step in the pipeline that needs
advanced labeling capabilities.

Features include:
- Triple barrier labeling
- Meta-labeling
- Regime-aware labeling
- Fractional differentiation
- Profit-based feature engineering

Note: This module now uses the restored step06 functionality from utils.
"""

# Import the restored step06 utilities
from .step06_labeling_components import (
    OptimizedTripleBarrierLabeling,
    FractionalTripleBarrierLabeling,
    RegimeSpecificTripleBarrierOptimizer,
    ProfitBasedFeatureEngineering,
    RegimeAwareTripleBarrierLabeling
)

# Re-export for backward compatibility
TripleBarrierLabeling = OptimizedTripleBarrierLabeling
MetaLabeling = RegimeAwareTripleBarrierLabeling
RegimeAwareLabeling = RegimeAwareTripleBarrierLabeling
FractionalDifferentiation = FractionalTripleBarrierLabeling

# Convenience functions for easy access
def create_triple_barrier_labeling(profit_take_multiplier=0.004,
                                 stop_loss_multiplier=0.003,
                                 transaction_cost=0.0008,
                                 time_barrier_minutes=30):
    """Create a new instance of TripleBarrierLabeling."""
    return OptimizedTripleBarrierLabeling(profit_take_multiplier, stop_loss_multiplier, 
                                        transaction_cost, time_barrier_minutes)

def create_meta_labeling(confidence_threshold=0.6, min_samples_per_class=100):
    """Create a new instance of MetaLabeling."""
    return RegimeAwareTripleBarrierLabeling(confidence_threshold)

def create_regime_aware_labeling(regime_threshold=0.7, regime_specific_thresholds=None):
    """Create a new instance of RegimeAwareLabeling."""
    return RegimeAwareTripleBarrierLabeling(regime_threshold, regime_specific_thresholds)

def create_fractional_differentiation(d=0.5, threshold=0.01):
    """Create a new instance of FractionalDifferentiation."""
    return FractionalTripleBarrierLabeling(d, threshold)

def create_triple_barrier_labels(market_data, 
                               profit_take_multiplier=0.004,
                               stop_loss_multiplier=0.003,
                               transaction_cost=0.0008):
    """Convenience function to create triple barrier labels."""
    labeling = OptimizedTripleBarrierLabeling(profit_take_multiplier, stop_loss_multiplier, transaction_cost)
    return labeling.create_labels(market_data)