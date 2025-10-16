"""Registry interfaces shared across feature systems."""

# Import utility functions
try:
    from src.utils.tprint import tprint
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False

from .base_registry import BaseFeatureRegistry

__all__ = ['BaseFeatureRegistry']

if TPRINT_AVAILABLE:
    tprint("🔧 [registry] Module initialized with BaseFeatureRegistry", color="cyan")
else:
    print("🔧 [registry] Module initialized with BaseFeatureRegistry")
