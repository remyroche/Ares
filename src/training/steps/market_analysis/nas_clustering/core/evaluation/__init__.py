"""
NAS Clustering Evaluation Module

Evaluation components for NAS clustering.
"""

# Import evaluation components if they exist
try:
    from .multi_objective import MultiObjectiveEvaluator
    __all__ = ['MultiObjectiveEvaluator']
except ImportError:
    __all__ = []
