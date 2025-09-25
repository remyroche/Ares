"""
NAS Search Module

Neural Architecture Search components for clustering.
"""

# Import search components if they exist
try:
    from .evolutionary_search import EvolutionarySearch
    from .search_space import SearchSpace
    __all__ = ['EvolutionarySearch', 'SearchSpace']
except ImportError:
    __all__ = []
