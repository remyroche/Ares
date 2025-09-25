"""
Core modules for Tree Architecture Search (TAS) and Neural Architecture Search (NAS).

This package provides comprehensive implementations for:
- Tree Architecture Search with shared utilities integration
- Search Space management with Grid + TPE optimization
- Hardware optimization for M1 systems
- Integration with ML common utilities

Modules:
- tree_architecture_search: Tree-based model architecture search
- search_space: Comprehensive search space management
"""

__version__ = "1.0.0"
__author__ = "Tree Architecture Search Team"

# Import main classes for easy access
try:
    from .tree_architecture_search import (
        TreeArchitectureSearch,
        TreeArchitectureConfig,
        TreeArchitectureCandidate,
        search_tree_architecture
    )
    from .search_space import (
        SearchSpace,
        SearchSpaceConfig,
        ParameterRange,
        SearchSpaceType,
        OptimizationStrategy,
        create_default_nas_search_space,
        create_tree_search_space
    )
    
    __all__ = [
        'TreeArchitectureSearch',
        'TreeArchitectureConfig', 
        'TreeArchitectureCandidate',
        'search_tree_architecture',
        'SearchSpace',
        'SearchSpaceConfig',
        'ParameterRange',
        'SearchSpaceType',
        'OptimizationStrategy',
        'create_default_nas_search_space',
        'create_tree_search_space'
    ]
    
except ImportError as e:
    # Handle missing dependencies gracefully
    import warnings
    warnings.warn(f"Core modules require additional dependencies: {e}")
    __all__ = []