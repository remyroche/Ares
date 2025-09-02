# src/supervisor/__init__.py
# This file makes the 'supervisor' directory a Python package.

from .main import Supervisor

# Define __all__ to explicitly export these modules/classes
__all__ = [
    "Supervisor",
]
