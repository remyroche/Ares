"""
NAS Regime Discovery Module

This module provides NAS-driven regime discovery that replaces the HMM regime discovery
with enhanced capabilities for short-term trading regime detection.
"""

from .components.nas_regime_discovery_component import NASRegimeDiscoveryComponent

__all__ = [
    'NASRegimeDiscoveryComponent'
]