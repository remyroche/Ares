"""Minimal component stubs for hybrid NAS-TAS orchestrator tests."""

from .tas_integration import TASIntegrationComponent
from .nas_integration import NASIntegrationComponent

__all__ = [
    "TASIntegrationComponent",
    "NASIntegrationComponent",
]
