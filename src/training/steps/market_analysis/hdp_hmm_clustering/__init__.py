"""
HDP-HMM Clustering Module

This module provides Hierarchical Dirichlet Process Hidden Markov Model
clustering for regime discovery with integrated quality assessment and
optimization goals.

ENHANCED with:
- Hierarchical hyperparameter optimization (3-5x faster)
- Unified vectorization manager (2-10x faster)
- Hardware optimization (M1/M2, GPU support)
- VectorBT integration (3-5x faster rolling ops)
- Memory management (handles large datasets)

Step Usage (Recommended - Uses BaseStep with artifact_manager):
    ```python
    from src.training.steps.market_analysis.hdp_hmm_clustering import (
        HDPHMMRegimeDiscoveryStep
    )
    
    step = HDPHMMRegimeDiscoveryStep()
    results = await step.execute({
        'symbol': 'BTCUSDT',
        'exchange': 'binance',
        'regime_timeframe': '1h',  # Default for regime detection
        'run_optimization': False,
        'hdp_hmm_params': {
            'alpha': 3.0,
            'kappa': 50.0,
            'n_iterations': 100
        }
    })
    ```

Standalone Usage (Direct):
    ```python
    from src.training.steps.market_analysis.hdp_hmm_clustering import run_hdp_hmm_clustering
    
    results = run_hdp_hmm_clustering(
        market_data=df,
        symbol="BTCUSDT",
        exchange="binance",
        timeframe="1h",  # Or "60m"
        alpha=3.0,
        kappa=50.0,
        n_iterations=100,
        # Enhancements (all enabled by default)
        enable_vectorization=True,
        enable_hardware_optimization=True,
        enable_memory_optimization=True,
        enable_vectorbt=True
    )
    ```
"""

from .hdp_hmm_clusterer import (
    HDPHMMClusterer,
    HDPHMMConfig,
    HDPHMMResult,
    create_hdp_hmm_clusterer,
    HMM_AVAILABLE,
    HMM_LIBRARY
)

from .standalone_runner import (
    run_hdp_hmm_clustering,
    run_hdp_hmm_clustering_from_artifacts,
    load_market_data_for_clustering
)

from .hdp_hmm_auto_tuner import (
    HDPHMMAutoTuner,
    HDPHMMSearchSpace,
    TuningResult,
    run_hdp_hmm_auto_tuning
)

# NEW: Step class for pipeline integration
from .hdp_hmm_regime_discovery_step import (
    HDPHMMRegimeDiscoveryStep,
    run_hdp_hmm_step
)

__all__ = [
    # Core classes
    'HDPHMMClusterer',
    'HDPHMMConfig',
    'HDPHMMResult',
    'create_hdp_hmm_clusterer',
    'HMM_AVAILABLE',
    'HMM_LIBRARY',
    # Standalone functions
    'run_hdp_hmm_clustering',
    'run_hdp_hmm_clustering_from_artifacts',
    'load_market_data_for_clustering',
    # Hyperparameter optimization
    'HDPHMMAutoTuner',
    'HDPHMMSearchSpace',
    'TuningResult',
    'run_hdp_hmm_auto_tuning',
    # NEW: Step class for pipeline integration
    'HDPHMMRegimeDiscoveryStep',
    'run_hdp_hmm_step'
]
