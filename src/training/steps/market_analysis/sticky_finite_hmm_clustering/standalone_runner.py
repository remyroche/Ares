"""
Standalone Sticky Finite HMM Clustering Runner

This module provides standalone functions to run Sticky Finite HMM clustering with:
- Cluster quality assessment
- Clustering optimization goals
- Artifact manager for data loading/saving

Usage Example:
    ```python
    # Basic usage with market data DataFrame
    from src.training.steps.market_analysis.sticky_finite_hmm_clustering.standalone_runner import (
        run_sticky_finite_hmm_clustering
    )
    
    results = run_sticky_finite_hmm_clustering(
        market_data=df,
        symbol="BTCUSDT",
        exchange="binance",
        timeframe="1h",
        K=5,
        base_alpha=0.5,
        kappa=10.0,
        num_iters=800
    )
    
    # Advanced usage loading data from artifacts
    results = run_sticky_finite_hmm_clustering_from_artifacts(
        artifact_name="market_data",
        step_name="data_collection",
        symbol="BTCUSDT",
        exchange="binance",
        timeframe="1h"
    )
    ```
"""

import pandas as pd
from typing import Dict, Any, Optional

from src.utils.tprint import (
    tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_structured, tprint_timer
)

# Import Sticky Finite HMM components
from .sticky_finite_hmm_clusterer import (
    DEPENDENCIES_AVAILABLE
)

# Import artifact manager (lazy import)
def _get_artifact_manager():
    """Lazy import of artifact manager."""
    from src.utils.artifact_manager import ArtifactManager
    return ArtifactManager

# Global cache for integration to avoid re-initialization
_cached_integration = None
_cached_feature_config = None

def run_sticky_finite_hmm_clustering(
    market_data: pd.DataFrame,
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    timeframe: str = "1h",
    min_features: int = 50,
    max_features: int = 100,
    K: int = 5,
    n_mixtures: int = 1,  # Number of Gaussian mixtures per state
    base_alpha: float = 0.5,
    kappa: float = 10.0,
    num_iters: int = 150,  # Reduced from 800 for faster training
    lr: float = 1e-2,
    enable_pca: bool = True,
    pca_components: int = 15,  # Default to 15 (can use up to 20)
    save_results: bool = True,
    output_dir: Optional[str] = None,
    compute_posteriors: bool = True  # Skip during auto-tuning for speed
) -> Dict[str, Any]:
    """
    Run Sticky Finite HMM clustering with feature generation integration.
    
    This is the main standalone function to perform Sticky Finite HMM regime discovery
    with integrated quality assessment, optimization goals, and feature generation.
    
    Args:
        market_data: DataFrame with OHLCV columns (open, high, low, close, volume)
        symbol: Trading symbol (e.g., "BTCUSDT")
        exchange: Exchange name (e.g., "binance")
        timeframe: Timeframe (e.g., "1h", "30m")
        min_features: Minimum number of features to use
        max_features: Maximum number of features to use
        K: Number of states (regimes) - fixed at 5 by default
        n_mixtures: Number of Gaussian mixtures per state (1-3, default: 1)
            - 1: Single Gaussian (fast, ~30-40s)
            - 2: Two-component mixture (moderate, ~50-70s, better fit)
            - 3: Three-component mixture (slow, ~80-120s, complex regimes)
        base_alpha: Concentration for off-diagonal transitions (0.1-1.0 typical)
        kappa: Stickiness parameter (higher = longer durations)
        num_iters: Number of SVI iterations
        lr: Learning rate for optimizer
        enable_pca: Enable PCA dimensionality reduction
        pca_components: Number of PCA components (default: 15, can use up to 20)
        save_results: Whether to save results to artifacts
        output_dir: Optional output directory (defaults to "artifacts")
        
    Note:
        Uses the SAME feature generation pipeline as HDP-HMM to ensure consistency.
        The only differences are: fixed K=5 states and VB/SVI inference.
        
    Returns:
        Dictionary containing:
            - cluster_labels: Regime labels for each time step
            - cluster_probabilities: Posterior probabilities
            - n_clusters: Number of regimes (always K)
            - transition_matrix: State transition matrix
            - emission_params: Emission distribution parameters
            - state_durations: Average duration for each state
            - quality_metrics: Comprehensive quality assessment
            - feature_names: Names of features used
            - feature_matrix: Feature matrix used for clustering
            - final_elbo: Final ELBO value
            - metadata: Additional metadata
    
    Example:
        ```python
        import pandas as pd
        from src.training.steps.market_analysis.sticky_finite_hmm_clustering.standalone_runner import (
            run_sticky_finite_hmm_clustering
        )
        
        # Load your market data
        df = pd.read_csv("market_data.csv")
        
        # Run Sticky Finite HMM clustering
        results = run_sticky_finite_hmm_clustering(
            market_data=df,
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="1h",
            K=5,
            base_alpha=0.5,
            kappa=10.0,
            num_iters=800
        )
        
        # Access results
        tprint_info(f"Discovered {results['n_clusters']} regimes")
        tprint_info(f"Quality score: {results['quality_metrics']['composite_score']:.3f}")
        tprint_info(f"Final ELBO: {results['final_elbo']:.2f}")
        ```
    """
    tprint_info("🚀 Starting Sticky Finite HMM Clustering Pipeline")
    
    if not DEPENDENCIES_AVAILABLE:
        tprint_error("❌ Pyro and PyTorch not available. Install: pip install pyro-ppl torch")
        raise ImportError("Pyro and PyTorch required for Sticky Finite HMM clustering")
    
    # Validate input data
    if market_data is None or market_data.empty:
        tprint_error("❌ market_data cannot be None or empty")
        raise ValueError("market_data cannot be None or empty")
    
    tprint_info(f"📊 Input validation passed: {len(market_data)} samples")
    
    tprint_structured({
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "data_shape": market_data.shape,
        "K": K,
        "n_mixtures": n_mixtures,
        "base_alpha": base_alpha,
        "kappa": kappa,
        "num_iters": num_iters,
        "lr": lr,
        "pca_components": pca_components,
        "min_features": min_features,
        "max_features": max_features
    }, level="INFO")
    
    # Initialize artifact manager if saving results (lazy import)
    artifact_manager = None
    if save_results:
        config = {
            "paths": {
                "data_dir": output_dir or "artifacts",
                "cache_dir": "data_cache",
                "reports_dir": "reports"
            }
        }
        ArtifactManager = _get_artifact_manager()
        artifact_manager = ArtifactManager(config)
        artifact_manager.set_context(
            step_name="sticky_finite_hmm_clustering",
            symbol=symbol,
            exchange=exchange,
            information="regime_discovery"
        )
        tprint_success("✅ Artifact manager initialized")
    
    with tprint_timer("Sticky Finite HMM Clustering", level="INFO"):
        # Import feature generation integration (lazy import to avoid circular dependency)
        tprint_info("📦 Loading feature generation integration...")
        from src.feature_generation.integration.enhanced_sticky_finite_hmm_clustering_integration import (
            EnhancedStickyFiniteHMMClusteringIntegration
        )
        
        # Create config hash for caching - separate feature config from HMM config
        feature_config = {
            'min_features': min_features,
            'max_features': max_features,
            'enable_comprehensive_features': True,
            'enable_pca_reduction': enable_pca,
            'pca_components': pca_components
        }
        
        # Keep full config for integration creation but use feature_config for caching
        current_config = {
            'min_features': min_features,
            'max_features': max_features,
            'enable_comprehensive_features': True,
            'enable_pca_reduction': enable_pca,
            'pca_components': pca_components,
            'K': K,
            'n_mixtures': n_mixtures,
            'base_alpha': base_alpha,
            'kappa': kappa,
            'num_iters': num_iters,
            'lr': lr
        }
        
        global _cached_integration, _cached_feature_config
        
        # Check if we can use cached integration (only based on feature config)
        if _cached_integration is not None and _cached_feature_config == feature_config:
            tprint_info("📋 Using cached integration (same feature configuration)")
            integration = _cached_integration
        else:
            # Initialize new integration
            tprint_info("🔧 Initializing Sticky Finite HMM clustering integration...")
            integration = EnhancedStickyFiniteHMMClusteringIntegration(
                min_features=min_features,
                max_features=max_features,
                enable_comprehensive_features=True,
                enable_pca_reduction=enable_pca,
                pca_components=pca_components,
                K=K,
                n_mixtures=n_mixtures,
                base_alpha=base_alpha,
                kappa=kappa,
                num_iters=num_iters,
                lr=lr
            )
            tprint_success("✅ Integration initialized")
            
            # Cache the integration for future use
            _cached_integration = integration
            _cached_feature_config = feature_config
            tprint_info("💾 Cached integration for future trials")
        
        # Run clustering
        if not compute_posteriors:
            tprint_info("🚀 Running Sticky Finite HMM clustering pipeline (FAST mode: posteriors skipped)...")
        else:
            tprint_info("🚀 Running Sticky Finite HMM clustering pipeline...")
        
        results = integration.cluster_with_sticky_finite_hmm(
            market_data,
            compute_posteriors=compute_posteriors
        )
        tprint_success("✅ Clustering complete")
        
        # Validate result structure
        if 'cluster_labels' not in results:
            tprint_error("❌ Missing 'cluster_labels' in clustering results")
            raise KeyError("Expected 'cluster_labels' in clustering results. Available keys: " +
                          str(list(results.keys())))
        
        cluster_labels = results['cluster_labels']
        
        # Validate length match
        if len(cluster_labels) != len(market_data):
            tprint_warning(
                f"⚠️ Length mismatch: market_data={len(market_data)}, "
                f"labels={len(cluster_labels)}"
            )
            # Truncate or align
            if len(cluster_labels) < len(market_data):
                market_data_aligned = market_data.iloc[:len(cluster_labels)]
                tprint_info(f"📊 Aligned market_data to {len(cluster_labels)} samples")
            else:
                # Truncate cluster labels to match data length
                cluster_labels = cluster_labels[:len(market_data)]
                market_data_aligned = market_data
                tprint_info(f"📊 Truncated cluster labels to {len(market_data)} samples")
        else:
            market_data_aligned = market_data
            tprint_success(f"✅ Data and labels aligned: {len(cluster_labels)} samples")
        
        # Save results if requested
        if save_results and artifact_manager:
            tprint_info("💾 Saving results to artifacts...")
            try:
                # Save cluster labels
                tprint_info("   Saving cluster labels...")
                artifact_manager.save(
                    data=pd.DataFrame({
                        'timestamp': market_data_aligned.index,
                        'cluster_label': cluster_labels
                    }),
                    artifact_name="cluster_labels",
                    artifact_type="data"
                )
                
                # Save transition matrix
                tprint_info("   Saving transition matrix...")
                artifact_manager.save(
                    data=pd.DataFrame(results['transition_matrix']),
                    artifact_name="transition_matrix",
                    artifact_type="data"
                )
                
                # Save quality metrics
                tprint_info("   Saving quality metrics...")
                artifact_manager.save(
                    data=results['quality_metrics'],
                    artifact_name="quality_metrics",
                    artifact_type="metadata"
                )
                
                # Save ELBO history
                tprint_info("   Saving ELBO history...")
                artifact_manager.save(
                    data={'elbo_history': results.get('elbo_history', [])},
                    artifact_name="elbo_history",
                    artifact_type="metadata"
                )
                
                tprint_success("✅ All results saved to artifacts")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to save results: {e}")
    
    # Print summary
    tprint_info("=" * 60)
    tprint_info("STICKY FINITE HMM CLUSTERING RESULTS")
    tprint_info("=" * 60)
    tprint_structured({
        "n_regimes": results['n_clusters'],
        "silhouette_score": results['quality_metrics']['silhouette_score'],
        "davies_bouldin_score": results['quality_metrics']['davies_bouldin_score'],
        "calinski_harabasz_score": results['quality_metrics']['calinski_harabasz_score'],
        "composite_score": results['quality_metrics'].get('composite_score', 0.0),
        "transition_persistence": results['quality_metrics']['transition_persistence'],
        "final_elbo": results.get('final_elbo', 0.0)
    }, level="INFO")
    tprint_info("=" * 60)
    
    return results


def run_sticky_finite_hmm_clustering_from_artifacts(
    artifact_name: str,
    step_name: str,
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    timeframe: str = "1h",
    artifact_dir: str = "artifacts",
    **clustering_kwargs
) -> Dict[str, Any]:
    """
    Run Sticky Finite HMM clustering on data loaded from artifacts.
    
    This function loads market data from a previously saved artifact and
    runs Sticky Finite HMM clustering on it.
    
    Args:
        artifact_name: Name of the artifact to load (e.g., "market_data")
        step_name: Name of the step that saved the artifact (e.g., "data_collection")
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        artifact_dir: Directory containing artifacts
        **clustering_kwargs: Additional arguments passed to run_sticky_finite_hmm_clustering
        
    Returns:
        Dictionary with clustering results (same as run_sticky_finite_hmm_clustering)
        
    Example:
        ```python
        from src.training.steps.market_analysis.sticky_finite_hmm_clustering.standalone_runner import (
            run_sticky_finite_hmm_clustering_from_artifacts
        )
        
        # Load data from artifacts and run clustering
        results = run_sticky_finite_hmm_clustering_from_artifacts(
            artifact_name="market_data",
            step_name="data_collection",
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="1h",
            K=5,
            base_alpha=0.5,
            kappa=10.0,
            num_iters=800
        )
        ```
    """
    tprint_info("📂 Loading market data from artifacts")
    tprint_info(f"   Artifact: {artifact_name}, Step: {step_name}")
    tprint_info(f"   Symbol: {symbol}, Exchange: {exchange}, Timeframe: {timeframe}")
    
    # Initialize artifact manager (lazy import)
    tprint_info("🔧 Initializing artifact manager...")
    config = {
        "paths": {
            "data_dir": artifact_dir,
            "cache_dir": "data_cache",
            "reports_dir": "reports"
        }
    }
    ArtifactManager = _get_artifact_manager()
    artifact_manager = ArtifactManager(config)
    artifact_manager.set_context(
        step_name=step_name,
        symbol=symbol,
        exchange=exchange
    )
    tprint_success("✅ Artifact manager initialized")
    
    # Load market data
    try:
        tprint_info(f"📥 Loading artifact: {artifact_name}...")
        market_data = artifact_manager.get_artifact(
            artifact_name=artifact_name,
            artifact_type="data"
        )
        
        if market_data is None:
            tprint_error(f"❌ Could not load artifact: {artifact_name}")
            raise ValueError(f"Could not load artifact: {artifact_name}")
        
        tprint_success(f"✅ Loaded market data: {market_data.shape}")
        
    except Exception as e:
        tprint_error(f"❌ Failed to load artifact: {e}")
        raise
    
    # Run clustering
    tprint_info("🚀 Starting clustering on loaded data...")
    return run_sticky_finite_hmm_clustering(
        market_data=market_data,
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        **clustering_kwargs
    )


def load_market_data_for_clustering(
    symbol: str,
    exchange: str,
    timeframe: str,
    artifact_name: str = "market_data",
    step_name: str = "data_collection",
    artifact_dir: str = "artifacts"
) -> pd.DataFrame:
    """
    Helper function to load market data from artifacts.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        artifact_name: Name of the artifact containing market data
        step_name: Step that saved the data
        artifact_dir: Artifacts directory
        
    Returns:
        DataFrame with market data
        
    Example:
        ```python
        from src.training.steps.market_analysis.sticky_finite_hmm_clustering.standalone_runner import (
            load_market_data_for_clustering
        )
        
        # Load market data
        df = load_market_data_for_clustering(
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="1h"
        )
        
        tprint_data_preview(df, "Market Data Preview", max_rows=5)
        ```
    """
    config = {
        "paths": {
            "data_dir": artifact_dir,
            "cache_dir": "data_cache",
            "reports_dir": "reports"
        }
    }
    ArtifactManager = _get_artifact_manager()
    artifact_manager = ArtifactManager(config)
    artifact_manager.set_context(
        step_name=step_name,
        symbol=symbol,
        exchange=exchange
    )
    
    market_data = artifact_manager.get_artifact(
        artifact_name=artifact_name,
        artifact_type="data"
    )
    
    if market_data is None:
        tprint_error(f"❌ Could not load market data artifact: {artifact_name}")
        raise ValueError(f"Could not load market data artifact: {artifact_name}")
    
    return market_data


__all__ = [
    'run_sticky_finite_hmm_clustering',
    'run_sticky_finite_hmm_clustering_from_artifacts',
    'load_market_data_for_clustering'
]

