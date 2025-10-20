"""
Market Analysis Steps Module.

This module registers all market analysis steps for autonomous execution.
"""

from src.training.steps.base_step import step_registry
from .components.sr_clustering import SRClusteringComponent
from .components.sr_detection import SRDetectionComponent

# Import HDBSCAN regime discovery step
from .hdbscan_clustering import HDBSCANRegimeDiscoveryStep

# Import regime clustering step
from .components.regime_clustering import RegimeClusteringComponent

# Register market analysis steps
step_registry.register("sr_clustering", SRClusteringComponent)
step_registry.register("sr_detection", SRDetectionComponent)
step_registry.register("hdbscan_regime_discovery", HDBSCANRegimeDiscoveryStep)
step_registry.register("regime_clustering", RegimeClusteringComponent)