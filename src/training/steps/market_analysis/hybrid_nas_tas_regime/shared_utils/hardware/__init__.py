"""
Shared hardware optimization utilities for regime detection systems.

This module provides hardware optimization utilities that can be used by both
NAS and TAS regime detection systems.
"""

from .hardware_optimizer import HardwareOptimizer

__all__ = [
    'HardwareOptimizer'
]