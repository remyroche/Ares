"""
Sticky Finite HMM Clustering Module

This module provides Sticky Finite HMM (K=5) with Variational Bayes inference
using Pyro + PyTorch as an alternative to the nonparametric HDP-HMM.

Key Features:
- Fixed K=5 states (not nonparametric - must choose K)
- Dirichlet priors on transition rows with stickiness parameter
- VB/SVI inference using Pyro + PyTorch
- Reuses existing infrastructure (feature generation, PCA, quality assessment, artifacts)

Usage:
    ```python
    from src.training.steps.market_analysis.sticky_finite_hmm_clustering import (
        StickyFiniteHMMClusterer,
        StickyFiniteHMMConfig,
        run_sticky_finite_hmm_step,
        run_sticky_finite_hmm_clustering
    )
    
    # Using the step interface
    config = {
        'symbol': 'BTCUSDT',
        'exchange': 'binance',
        'regime_timeframe': '1h',
        'sticky_finite_hmm_params': {
            'K': 5,
            'base_alpha': 0.5,
            'kappa': 10.0,
            'num_iters': 800,
            'lr': 1e-2
        }
    }
    results = await run_sticky_finite_hmm_step(config)
    
    # Using the standalone function
    results = run_sticky_finite_hmm_clustering(
        market_data=df,
        symbol="BTCUSDT",
        exchange="binance",
        timeframe="1h"
    )
    ```
"""

# Version info
__version__ = "1.0.0"
__author__ = "Ares Trading System"

# Check dependencies
try:
    import torch
    import pyro
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    import warnings
    warnings.warn(
        "Sticky Finite HMM requires pyro-ppl and torch. "
        "Install with: pip install pyro-ppl torch",
        ImportWarning
    )

# Import core components (only if dependencies available)
if DEPENDENCIES_AVAILABLE:
    from .sticky_finite_hmm_clusterer import (
        StickyFiniteHMMClusterer,
        StickyFiniteHMMConfig,
        StickyFiniteHMMResult,
        create_sticky_finite_hmm_clusterer
    )
    
from .sticky_finite_hmm_regime_discovery_step import (
    StickyFiniteHMMRegimeDiscoveryStep,
    run_sticky_finite_hmm_step
)

from .standalone_runner import (
    run_sticky_finite_hmm_clustering,
    run_sticky_finite_hmm_clustering_from_artifacts,
    load_market_data_for_clustering
)

# Auto-tuner (optional, requires optimization utilities)
try:
    from .sticky_finite_hmm_auto_tuner import (
        run_sticky_finite_hmm_auto_tuning,
        StickyFiniteHMMSearchSpace,
        create_default_search_space
    )
    AUTO_TUNER_AVAILABLE = True
except ImportError:
    # Auto-tuner not available (missing optimization dependencies)
    run_sticky_finite_hmm_auto_tuning = None
    StickyFiniteHMMSearchSpace = None
    create_default_search_space = None
    AUTO_TUNER_AVAILABLE = False
    
    __all__ = [
        # Core clusterer
        'StickyFiniteHMMClusterer',
        'StickyFiniteHMMConfig',
        'StickyFiniteHMMResult',
        'create_sticky_finite_hmm_clusterer',
        
        # Step interface
        'StickyFiniteHMMRegimeDiscoveryStep',
        'run_sticky_finite_hmm_step',
        
        # Standalone functions
        'run_sticky_finite_hmm_clustering',
        'run_sticky_finite_hmm_clustering_from_artifacts',
        'load_market_data_for_clustering',
        
        # Auto-tuner (if available)
        'run_sticky_finite_hmm_auto_tuning',
        'StickyFiniteHMMSearchSpace',
        'create_default_search_space',
        
        # Metadata
        'DEPENDENCIES_AVAILABLE',
        'AUTO_TUNER_AVAILABLE',
        '__version__'
    ]
else:
    __all__ = ['DEPENDENCIES_AVAILABLE', '__version__']

