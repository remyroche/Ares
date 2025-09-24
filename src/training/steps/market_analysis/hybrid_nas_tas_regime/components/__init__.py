"""
Components for Hybrid NAS TAS Regime system.

Provides TAS and NAS integration components.
"""

from .tas_integration import TASIntegration
from .nas_integration import NASIntegration

__all__ = [
    'TASIntegration',
    'NASIntegration'
]