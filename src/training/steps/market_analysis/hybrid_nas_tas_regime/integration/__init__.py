"""
Integration components for Hybrid NAS TAS Regime system.

Provides the main orchestrator that replaces HMM clustering functionality.
"""

from .hybrid_orchestrator import HybridOrchestrator

__all__ = [
    'HybridOrchestrator'
]