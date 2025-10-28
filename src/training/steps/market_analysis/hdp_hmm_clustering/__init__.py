"""
HDP-HMM Clustering Module

This module provides Hierarchical Dirichlet Process Hidden Markov Model
clustering for regime discovery with integrated quality assessment and
optimization goals.

Standalone Usage:
    ```python
    from src.training.steps.market_analysis.hdp_hmm_clustering import run_hdp_hmm_clustering
    
    results = run_hdp_hmm_clustering(
        market_data=df,
        symbol="BTCUSDT",
        exchange="binance",
        timeframe="30m",
        alpha=3.0,
        kappa=50.0,
        n_iterations=100
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

__all__ = [
    'HDPHMMClusterer',
    'HDPHMMConfig',
    'HDPHMMResult',
    'create_hdp_hmm_clusterer',
    'HMM_AVAILABLE',
    'HMM_LIBRARY',
    'run_hdp_hmm_clustering',
    'run_hdp_hmm_clustering_from_artifacts',
    'load_market_data_for_clustering'
]
