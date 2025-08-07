# src/training/optimization/__init__.py

"""
Training Optimization Package

This package contains optimization components for training processes.
"""

from .rollback_manager import RollbackManager

__all__ = [
    "RollbackManager",
] 