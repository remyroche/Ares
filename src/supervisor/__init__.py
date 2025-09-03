from __future__ import annotations

"""Supervisor Package.

The supervisor package provides high-level orchestration and management
for the trading system, including portfolio management, performance monitoring,
model behavior tracking, and dynamic weighting of ensemble predictions.
"""

# src/supervisor/__init__.py
# This file makes the 'supervisor' directory a Python package.

from .ab_tester import ABTester
from .main import Supervisor
from .optimizer import Optimizer
from .performance_reporter import PerformanceReporter
from .risk_allocator import RiskAllocator

# Define __all__ to explicitly export these modules/classes
__all__ = [
    "ABTester",
    "Supervisor",
    "Optimizer",
    "PerformanceReporter",
    "RiskAllocator",
]
