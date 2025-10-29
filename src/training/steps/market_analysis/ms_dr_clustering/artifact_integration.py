"""
Artifact Integration for MS-DR Clustering

Provides convenience functions for using MS-DR clustering with artifact manager
for seamless data loading and result persistence.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.utils.artifact_manager import ArtifactManager
from src.utils.tprint import (
    tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_structured
)

# Import MS-DR components
from .ms_dr_clusterer import MSDRClusterer, MSDRConfig, MSDRResult
from .ms_dr_auto_tuner import MSDRAutoTuner, MSDRTuningConfig


def perform_ms_dr_clustering_with_artifact_manager(
    symbol: str,
    exchange: str = 'binance',
    timeframe: str = '60m',
    data_dir: str = 'historical_data',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    config: Optional[MSDRConfig] = None,
    artifact_manager: Optional[ArtifactManager] = None,
    save_artifacts: bool = True
) -> Dict[str, Any]:
    """
    Perform MS-DR clustering with automatic data loading and artifact saving.
    
    This is a convenience function that:
    1. Loads market data from klines_parquet (default: 60m/1h timeframe)
    2. Runs MS-DR clustering with enhancements
    3. Saves results to artifact manager
    
    Args:
        symbol: Trading symbol (e.g., 'ETHUSDT')
        exchange: Exchange name (default: 'binance')
        timeframe: Timeframe (default: '60m', alternatives: '1h', '15m', '5m')
        data_dir: Historical data directory
        start_date: Optional start date filter
        end_date: Optional end date filter
        config: Optional MSDRConfig (uses defaults if None)
        artifact_manager: Optional ArtifactManager (creates one if None)
        save_artifacts: Whether to save artifacts (default: True)
        
    Returns:
        Dictionary containing:
            - result: MSDRResult object
            - artifacts: List of saved artifact paths
            - market_data: Loaded market data DataFrame
            - metrics: Quality and performance metrics
            
    Example:
        >>> from src.training.steps.market_analysis.ms_dr_clustering import (
        ...     perform_ms_dr_clustering_with_artifact_manager
        ... )
        >>> 
        >>> result_dict = perform_ms_dr_clustering_with_artifact_manager(
        ...     symbol='ETHUSDT',
        ...     timeframe='60m',
        ...     save_artifacts=True
        ... )
        >>> 
        >>> msdr_result = result_dict['result']
        >>> print(f"Found {msdr_result.n_clusters} regimes")
        >>> print(f"Artifacts saved to: {result_dict['artifacts']}")
    """
    tprint_info(f"🚀 MS-DR Clustering for {symbol} ({timeframe})")
    
    # Normalize timeframe
    if timeframe == '1h':
        timeframe = '60m'
    
    # Load market data
    try:
        from src.utils.data.klines_parquet import get_klines_manager
        
        klines_manager = get_klines_manager(data_dir=data_dir)
        
        # Parse dates if provided
        start_dt = pd.to_datetime(start_date) if start_date else None
        end_dt = pd.to_datetime(end_date) if end_date else None
        
        tprint_info(f"📂 Loading market data from {data_dir}...")
        market_data = klines_manager.read_data(
            symbol=symbol,
            interval=timeframe,
            data_type="processed",
            start_date=start_dt,
            end_date=end_dt
        )
        
        if market_data is None or len(market_data) == 0:
            raise ValueError("No market data loaded")
        
        tprint_success(f"✅ Loaded {len(market_data)} rows of market data")
        
    except Exception as e:
        tprint_error(f"❌ Failed to load market data: {e}")
        raise
    
    # Create configuration if not provided
    if config is None:
        config = MSDRConfig(
            n_regimes=5,
            auto_select_regimes=True,
            use_safe_math=True,
            use_memory_optimization=True,
            use_hardware_acceleration=True,
            use_vectorbt_operations=True
        )
    
    # Initialize artifact manager if not provided
    if artifact_manager is None and save_artifacts:
        artifact_manager = ArtifactManager(config={})
        artifact_manager.set_context(
            step_name="ms_dr_clustering",
            datetime=datetime.now()
        )
    
    # Run clustering
    tprint_info("🔄 Running MS-DR clustering...")
    clusterer = MSDRClusterer(config)
    msdr_result = clusterer.fit_predict(market_data.values)
    
    if not msdr_result.success:
        tprint_error(f"❌ Clustering failed: {msdr_result.error_message}")
        raise RuntimeError(msdr_result.error_message)
    
    tprint_success(f"✅ Clustering complete: {msdr_result.n_clusters} regimes")
    
    # Save artifacts if enabled
    artifacts = []
    if save_artifacts and artifact_manager is not None:
        tprint_info("💾 Saving artifacts...")
        
        try:
            # Save regime labels
            regime_labels_df = pd.DataFrame({
                'timestamp': market_data.index,
                'regime_label': msdr_result.cluster_labels
            })
            artifact_path = artifact_manager.save(
                data=regime_labels_df,
                artifact_name=f"ms_dr_regime_labels_{symbol}_{timeframe}",
                artifact_type="data",
                compression="auto",
                metadata={
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'n_regimes': msdr_result.n_clusters
                }
            )
            artifacts.append(artifact_path)
            
            # Save comprehensive results
            results_dict = {
                'symbol': symbol,
                'timeframe': timeframe,
                'n_regimes': msdr_result.n_clusters,
                'regime_labels': msdr_result.cluster_labels.tolist(),
                'transition_matrix': msdr_result.transition_matrix.tolist() if msdr_result.transition_matrix is not None else None,
                'metrics': {
                    'silhouette_score': msdr_result.silhouette_score,
                    'aic': msdr_result.aic,
                    'bic': msdr_result.bic,
                    'transition_persistence': msdr_result.transition_persistence
                }
            }
            
            artifact_path = artifact_manager.save(
                data=results_dict,
                artifact_name=f"ms_dr_results_{symbol}_{timeframe}",
                artifact_type="metadata"
            )
            artifacts.append(artifact_path)
            
            tprint_success(f"✅ Saved {len(artifacts)} artifacts")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to save artifacts: {e}")
    
    # Generate metrics
    metrics = {
        'n_regimes': msdr_result.n_clusters,
        'silhouette_score': msdr_result.silhouette_score,
        'aic': msdr_result.aic,
        'bic': msdr_result.bic,
        'processing_time': msdr_result.processing_time,
        'memory_usage_mb': msdr_result.memory_usage_mb
    }
    
    return {
        'result': msdr_result,
        'artifacts': artifacts,
        'market_data': market_data,
        'metrics': metrics,
        'success': True
    }


def perform_enhanced_ms_dr_clustering(
    market_data: pd.DataFrame,
    symbol: str = 'UNKNOWN',
    timeframe: str = '60m',
    config: Optional[MSDRConfig] = None,
    enable_optimization: bool = False,
    use_hierarchical: bool = True,
    artifact_manager: Optional[ArtifactManager] = None,
    save_artifacts: bool = True
) -> Dict[str, Any]:
    """
    Perform enhanced MS-DR clustering with optional hyperparameter optimization.
    
    This function provides a high-level interface for MS-DR clustering with:
    - Automatic hyperparameter optimization (optional)
    - Hierarchical optimization for 50-70% speedup
    - Artifact persistence
    - Comprehensive quality assessment
    
    Args:
        market_data: Market data DataFrame (pre-loaded)
        symbol: Trading symbol for metadata
        timeframe: Timeframe for metadata
        config: Optional MSDRConfig
        enable_optimization: Enable hyperparameter optimization
        use_hierarchical: Use hierarchical optimization (faster)
        artifact_manager: Optional ArtifactManager
        save_artifacts: Whether to save artifacts
        
    Returns:
        Dictionary with results, artifacts, and metrics
        
    Example:
        >>> result = perform_enhanced_ms_dr_clustering(
        ...     market_data=df,
        ...     symbol='ETHUSDT',
        ...     enable_optimization=True,
        ...     use_hierarchical=True
        ... )
        >>> print(f"Found {result['result'].n_clusters} regimes")
    """
    tprint_info(f"🚀 Enhanced MS-DR Clustering for {symbol} ({timeframe})")
    
    # Initialize artifact manager if needed
    if artifact_manager is None and save_artifacts:
        artifact_manager = ArtifactManager(config={})
        artifact_manager.set_context(
            step_name="ms_dr_clustering",
            datetime=datetime.now()
        )
    
    artifacts = []
    best_params = None
    
    # Run with or without optimization
    if enable_optimization:
        tprint_info("🎯 Hyperparameter optimization enabled")
        
        # Create tuning config
        tuning_config = MSDRTuningConfig(
            n_trials=50 if not use_hierarchical else None,
            use_hierarchical=use_hierarchical,
            n_trials_per_group=20,
            timeout_minutes=30.0,
            random_state=42
        )
        
        # Run optimization
        tuner = MSDRAutoTuner(tuning_config)
        
        if use_hierarchical:
            tprint_info("⚡ Using hierarchical optimization (50-70% faster!)")
            opt_result = tuner.auto_tune_hierarchical(market_data)
        else:
            opt_result = tuner.auto_tune(market_data)
        
        best_params = opt_result['best_params']
        
        # Create clusterer with best params
        final_config = MSDRConfig(**best_params)
        final_config.use_safe_math = True
        final_config.use_memory_optimization = True
        final_config.use_hardware_acceleration = True
        
        clusterer = MSDRClusterer(final_config)
        msdr_result = clusterer.fit_predict(market_data.values)
        
        # Save optimization results
        if save_artifacts and artifact_manager is not None:
            opt_artifact = artifact_manager.save(
                data=opt_result,
                artifact_name=f"ms_dr_optimization_{symbol}_{timeframe}",
                artifact_type="metadata",
                metadata={'symbol': symbol, 'timeframe': timeframe}
            )
            artifacts.append(opt_artifact)
    
    else:
        # Run without optimization
        if config is None:
            config = MSDRConfig(
                use_safe_math=True,
                use_memory_optimization=True,
                use_hardware_acceleration=True
            )
        
        clusterer = MSDRClusterer(config)
        msdr_result = clusterer.fit_predict(market_data.values)
    
    if not msdr_result.success:
        tprint_error(f"❌ Clustering failed: {msdr_result.error_message}")
        raise RuntimeError(msdr_result.error_message)
    
    tprint_success(f"✅ Found {msdr_result.n_clusters} regimes")
    
    # Save artifacts
    if save_artifacts and artifact_manager is not None:
        tprint_info("💾 Saving clustering artifacts...")
        
        try:
            # Regime labels
            regime_labels_df = pd.DataFrame({
                'timestamp': market_data.index,
                'regime_label': msdr_result.cluster_labels
            })
            artifact_path = artifact_manager.save(
                data=regime_labels_df,
                artifact_name=f"ms_dr_regime_labels_{symbol}_{timeframe}",
                artifact_type="data",
                compression="auto"
            )
            artifacts.append(artifact_path)
            
            # Comprehensive results
            results_dict = {
                'symbol': symbol,
                'timeframe': timeframe,
                'n_regimes': msdr_result.n_clusters,
                'regime_labels': msdr_result.cluster_labels.tolist(),
                'transition_matrix': msdr_result.transition_matrix.tolist() if msdr_result.transition_matrix is not None else None,
                'best_params': best_params,
                'metrics': {
                    'silhouette_score': msdr_result.silhouette_score,
                    'aic': msdr_result.aic,
                    'bic': msdr_result.bic,
                    'processing_time': msdr_result.processing_time,
                    'memory_usage_mb': msdr_result.memory_usage_mb
                }
            }
            
            artifact_path = artifact_manager.save(
                data=results_dict,
                artifact_name=f"ms_dr_results_{symbol}_{timeframe}",
                artifact_type="metadata"
            )
            artifacts.append(artifact_path)
            
            tprint_success(f"✅ Saved {len(artifacts)} artifacts")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to save some artifacts: {e}")
    
    # Generate comprehensive metrics
    metrics = {
        'n_regimes': msdr_result.n_clusters,
        'noise_ratio': msdr_result.noise_ratio,
        'silhouette_score': msdr_result.silhouette_score,
        'calinski_harabasz_score': msdr_result.calinski_harabasz_score,
        'davies_bouldin_score': msdr_result.davies_bouldin_score,
        'aic': msdr_result.aic,
        'bic': msdr_result.bic,
        'hqic': msdr_result.hqic,
        'transition_persistence': msdr_result.transition_persistence,
        'processing_time': msdr_result.processing_time,
        'memory_usage_mb': msdr_result.memory_usage_mb,
        'best_params': best_params
    }
    
    tprint_structured(metrics, level="INFO")
    
    return {
        'result': msdr_result,
        'artifacts': artifacts,
        'market_data': market_data,
        'metrics': metrics,
        'best_params': best_params,
        'success': True
    }


def load_market_data_for_msdr(
    symbol: str,
    exchange: str = 'binance',
    timeframe: str = '60m',
    data_dir: str = 'historical_data',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    execution_mode: str = 'light'
) -> pd.DataFrame:
    """
    Load market data specifically for MS-DR clustering.
    
    Defaults to 60m/1h timeframe as MS-DR works best with hourly data.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe (default: '60m')
        data_dir: Data directory
        start_date: Optional start date
        end_date: Optional end date
        execution_mode: 'full', 'light', or 'blank'
        
    Returns:
        Market data DataFrame
    """
    from src.utils.data.klines_parquet import get_klines_manager
    
    # Normalize timeframe
    if timeframe == '1h':
        timeframe = '60m'
    
    tprint_info(f"📂 Loading market data for MS-DR clustering")
    tprint_info(f"   Symbol: {symbol}, Timeframe: {timeframe}")
    
    # Get klines manager
    klines_manager = get_klines_manager(data_dir=data_dir)
    
    # Auto date filtering based on execution mode
    if execution_mode == 'light' and not start_date and not end_date:
        end_dt = pd.Timestamp.now(tz='UTC').normalize()
        start_dt = end_dt - pd.Timedelta(days=30)
        tprint_info(f"📅 Light mode: Using last 30 days")
    elif execution_mode == 'blank' and not start_date and not end_date:
        end_dt = pd.Timestamp.now(tz='UTC').normalize()
        start_dt = end_dt - pd.Timedelta(days=90)
        tprint_info(f"📅 Blank mode: Using last 90 days")
    else:
        start_dt = pd.to_datetime(start_date) if start_date else None
        end_dt = pd.to_datetime(end_date) if end_date else None
    
    # Load data
    market_data = klines_manager.read_data(
        symbol=symbol,
        interval=timeframe,
        data_type="processed",
        start_date=start_dt,
        end_date=end_dt
    )
    
    if market_data is None or len(market_data) == 0:
        raise ValueError("No market data loaded")
    
    tprint_success(f"✅ Loaded {len(market_data)} rows")
    
    return market_data


__all__ = [
    'perform_ms_dr_clustering_with_artifact_manager',
    'perform_enhanced_ms_dr_clustering',
    'load_market_data_for_msdr'
]
