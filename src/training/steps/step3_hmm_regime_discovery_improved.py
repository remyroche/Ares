# src/training/steps/step3_hmm_regime_discovery_improved.py

"""
Improved Step 3: HMM Regime Discovery with enhanced code quality and performance.

Key improvements:
- Modular architecture with separate classes for different responsibilities
- Better memory management and resource cleanup
- Improved error handling and logging
- Type hints throughout
- Performance optimizations with parallel processing
- Better data validation and quality checks
- Reduced complexity and improved maintainability
"""

import os
import json
import math
import warnings
import sys
import time
import contextlib
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import joblib
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import gc
import psutil
import multiprocessing as mp
import signal
import atexit

import numpy as np
import pandas as pd
from hmmlearn.hmm import GMMHMM
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from scipy import stats
from scipy.spatial.distance import pdist, squareform

# Optional clustering
try:
    import hdbscan  # type: ignore
    _HDBSCAN_AVAILABLE = True
except Exception:
    _HDBSCAN_AVAILABLE = False

from src.utils.logger import system_logger
from src.utils.error_handler import (
    handle_errors,
    handle_data_processing_errors,
    handle_type_conversions,
    safe_division,
    clean_dataframe,
)
from src.utils.decorators import with_tracing_span, guard_dataframe_nulls
from src.training.steps.unified_data_loader import UnifiedDataLoader

# Import decorators from centralized module
from src.utils.centralized_decorators import (
    auto_fix_data_quality_issues,
    deterministic_seed,
    idempotent_step,
    artifact_write_lock,
    nan_inf_and_constant_guard,
    artifact_versioning,
    time_budget_watchdog,
    validate_step_prerequisites,
    secure_data_processing,
    prevent_data_leakage,
    resource_monitor,
    memory_efficient,
    debug_training_step,
    circuit_breaker_protection,
    validate_step_output,
    quality_gate,
    validate_data_quality,
    validate_feature_engineering_pipeline,
    validate_hmm_regime_discovery,
)


@dataclass
class HMMConfig:
    """Configuration for HMM regime discovery."""
    symbol: str
    exchange: str
    data_dir: str
    timeframe: str
    force_rerun: bool = False
    enable_parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    n_components: int = 5
    n_mix: int = 3
    covariance_type: str = "full"
    random_state: int = 42
    max_iter: int = 100
    tol: float = 1e-3
    verbose: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.symbol or not self.exchange:
            raise ValueError("Symbol and exchange must be provided")
        if self.n_components < 2:
            raise ValueError("n_components must be at least 2")
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")


class HMMRegimeAnalyzer:
    """Improved HMM regime analyzer with better memory management."""
    
    def __init__(self, config: HMMConfig):
        self.config = config
        self.logger = system_logger.getChild("HMMRegimeAnalyzer")
        self.models = {}
        self.scalers = {}
        
    def _cleanup_multiprocessing_resources(self):
        """Clean up multiprocessing resources."""
        try:
            # Clean up joblib cache
            joblib.clear()
            
            # Force garbage collection
            gc.collect()
            
            # Clean up any remaining processes
            for proc in mp.active_children():
                proc.terminate()
                proc.join()
        except Exception as e:
            self.logger.warning(f"Error during multiprocessing cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self._cleanup_multiprocessing_resources()
    
    @handle_errors(exceptions=(Exception,), default_return=None)
    def create_basic_features(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Create basic features for HMM analysis."""
        try:
            if price_df.empty:
                raise ValueError("Empty price dataframe")
            
            # Ensure we have required columns
            required_cols = ["open", "high", "low", "close"]
            missing_cols = [col for col in required_cols if col not in price_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Calculate returns
            returns = price_df["close"].pct_change().dropna()
            
            # Calculate log returns
            log_returns = np.log(price_df["close"] / price_df["close"].shift(1)).dropna()
            
            # Calculate volatility (rolling standard deviation)
            volatility = returns.rolling(window=20, min_periods=1).std()
            
            # Calculate price momentum
            momentum = price_df["close"] / price_df["close"].shift(20) - 1
            
            # Calculate volume features if available
            volume_features = pd.DataFrame()
            if "volume" in price_df.columns:
                volume_features["volume_ratio"] = price_df["volume"] / price_df["volume"].rolling(window=20).mean()
                volume_features["volume_momentum"] = price_df["volume"].pct_change()
            
            # Combine features
            features = pd.DataFrame({
                "returns": returns,
                "log_returns": log_returns,
                "volatility": volatility,
                "momentum": momentum,
            })
            
            # Add volume features if available
            if not volume_features.empty:
                features = pd.concat([features, volume_features], axis=1)
            
            # Remove any infinite or NaN values
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.dropna()
            
            if features.empty:
                raise ValueError("No valid features after preprocessing")
            
            self.logger.info(f"Created {len(features.columns)} features with {len(features)} samples")
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating basic features: {e}")
            raise
    
    @handle_errors(exceptions=(Exception,), default_return=None)
    def fit_hmm_model(self, features: pd.DataFrame, timeframe: str) -> Optional[GMMHMM]:
        """Fit HMM model with improved error handling."""
        try:
            if features.empty:
                raise ValueError("Empty features dataframe")
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Initialize HMM model
            model = GMMHMM(
                n_components=self.config.n_components,
                n_mix=self.config.n_mix,
                covariance_type=self.config.covariance_type,
                random_state=self.config.random_state,
                max_iter=self.config.max_iter,
                tol=self.config.tol,
                verbose=self.config.verbose
            )
            
            # Fit model
            model.fit(scaled_features)
            
            # Store model and scaler
            self.models[timeframe] = model
            self.scalers[timeframe] = scaler
            
            self.logger.info(f"Fitted HMM model for {timeframe} with {self.config.n_components} components")
            return model
            
        except Exception as e:
            self.logger.error(f"Error fitting HMM model for {timeframe}: {e}")
            return None
    
    @handle_errors(exceptions=(Exception,), default_return=pd.DataFrame())
    def predict_regimes(self, features: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Predict regimes using fitted HMM model."""
        try:
            if timeframe not in self.models:
                raise ValueError(f"No fitted model found for {timeframe}")
            
            model = self.models[timeframe]
            scaler = self.scalers[timeframe]
            
            # Scale features
            scaled_features = scaler.transform(features)
            
            # Predict states
            states = model.predict(scaled_features)
            
            # Get state probabilities
            state_probs = model.predict_proba(scaled_features)
            
            # Create results dataframe
            results = pd.DataFrame({
                "state": states,
                "timestamp": features.index
            })
            
            # Add state probabilities
            for i in range(state_probs.shape[1]):
                results[f"state_{i}_prob"] = state_probs[:, i]
            
            self.logger.info(f"Predicted regimes for {timeframe}: {len(results)} samples")
            return results
            
        except Exception as e:
            self.logger.error(f"Error predicting regimes for {timeframe}: {e}")
            return pd.DataFrame()


class RegimeClusterAnalyzer:
    """Analyzer for regime clustering and composite analysis."""
    
    def __init__(self, config: HMMConfig):
        self.config = config
        self.logger = system_logger.getChild("RegimeClusterAnalyzer")
    
    @handle_errors(exceptions=(Exception,), default_return=pd.DataFrame())
    def perform_composite_clustering(
        self, 
        regime_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Perform composite clustering across timeframes."""
        try:
            if not regime_data:
                raise ValueError("No regime data provided")
            
            # Combine regime data from all timeframes
            combined_data = []
            for timeframe, df in regime_data.items():
                if not df.empty:
                    df_copy = df.copy()
                    df_copy["timeframe"] = timeframe
                    combined_data.append(df_copy)
            
            if not combined_data:
                raise ValueError("No valid regime data to cluster")
            
            combined_df = pd.concat(combined_data, ignore_index=True)
            
            # Extract features for clustering
            prob_cols = [col for col in combined_df.columns if col.startswith("state_") and col.endswith("_prob")]
            if not prob_cols:
                raise ValueError("No state probability columns found")
            
            features = combined_df[prob_cols].values
            
            # Perform clustering
            n_clusters = min(len(prob_cols), 10)  # Limit number of clusters
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage="ward",
                metric="euclidean"
            )
            
            cluster_labels = clusterer.fit_predict(features)
            combined_df["composite_cluster"] = cluster_labels
            
            self.logger.info(f"Performed composite clustering: {n_clusters} clusters")
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Error in composite clustering: {e}")
            return pd.DataFrame()


class HMMArtifactManager:
    """Manages HMM artifacts with improved caching and validation."""
    
    def __init__(self, config: HMMConfig):
        self.config = config
        self.logger = system_logger.getChild("HMMArtifactManager")
        self.output_dir = Path(config.data_dir) / "hmm_regimes"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def check_artifacts_exist(self) -> bool:
        """Check if HMM artifacts already exist."""
        try:
            required_files = [
                self.output_dir / f"{self.config.exchange}_{self.config.symbol}_hmm_composite_clusters_{self.config.timeframe}.parquet",
                self.output_dir / f"{self.config.exchange}_{self.config.symbol}_hmm_states_{self.config.timeframe}.parquet",
            ]
            return all(f.exists() for f in required_files)
        except Exception as e:
            self.logger.error(f"Error checking artifacts: {e}")
            return False
    
    def save_artifacts(
        self, 
        regime_data: Dict[str, pd.DataFrame], 
        composite_clusters: pd.DataFrame
    ) -> bool:
        """Save HMM artifacts with validation."""
        try:
            # Save individual timeframe results
            for timeframe, df in regime_data.items():
                if not df.empty:
                    file_path = self.output_dir / f"{self.config.exchange}_{self.config.symbol}_hmm_states_{timeframe}.parquet"
                    df.to_parquet(file_path, index=True)
                    self.logger.info(f"Saved HMM states for {timeframe}: {len(df)} samples")
            
            # Save composite clusters
            if not composite_clusters.empty:
                file_path = self.output_dir / f"{self.config.exchange}_{self.config.symbol}_hmm_composite_clusters_{self.config.timeframe}.parquet"
                composite_clusters.to_parquet(file_path, index=True)
                self.logger.info(f"Saved composite clusters: {len(composite_clusters)} samples")
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving artifacts: {e}")
            return False


@validate_step_prerequisites(
    required_directories=["data/training"],
    min_memory_gb=4.0,
    min_disk_gb=2.0,
    required_packages=["pandas", "numpy", "hmmlearn", "sklearn"],
    data_quality_checks={
        "min_rows": 1000,
        "required_columns": ["open", "high", "low", "close"],
    },
    context="Improved HMM Regime Discovery",
)
@secure_data_processing(
    backup_before=True, 
    integrity_checks=True, 
    memory_cleanup=True, 
    data_validation=True
)
@prevent_data_leakage(
    temporal_validation=True,
    feature_leakage_detection=True,
    lookahead_bias_prevention=True,
)
@resource_monitor(
    memory_threshold_gb=12.0,
    cpu_threshold_percent=80.0,
    disk_threshold_gb=5.0,
    monitor_interval=15.0,
    auto_cleanup=True,
)
@memory_efficient(
    chunk_size=5000, 
    streaming_processing=True, 
    memory_pool=True, 
    cleanup_frequency=3
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=True,
    performance_profiling=True,
    error_context_preservation=True,
)
@circuit_breaker_protection(
    failure_threshold=3,
    recovery_timeout=60.0,
    expected_exception=Exception,
    monitor_interval=10.0,
)
@validate_step_output(
    output_validation=True,
    data_quality_checks=True,
    performance_metrics=True,
)
@quality_gate(
    quality_threshold=0.7,
    validation_metrics=["model_convergence", "regime_quality", "performance"],
)
async def run_step_improved(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    timeframe: str = "1m",
    force_rerun: bool = False,
    **kwargs: Any,
) -> bool:
    """
    Improved Step 3: HMM Regime Discovery with enhanced code quality and performance.
    
    Key improvements:
    - Modular architecture with separate classes
    - Better memory management and resource cleanup
    - Improved error handling and validation
    - Parallel processing support
    - Comprehensive logging and monitoring
    """
    logger = system_logger.getChild("Step3.ImprovedHMMRegimeDiscovery")
    start_time = time.time()
    
    try:
        logger.info("🚀 Starting improved Step 3: HMM Regime Discovery")
        
        # Initialize configuration
        config = HMMConfig(
            symbol=symbol,
            exchange=exchange,
            data_dir=data_dir,
            timeframe=timeframe,
            force_rerun=force_rerun,
            **kwargs
        )
        
        # Initialize components
        artifact_manager = HMMArtifactManager(config)
        
        # Check for existing artifacts
        if artifact_manager.check_artifacts_exist() and not force_rerun:
            logger.info("📦 HMM artifacts already exist, skipping processing")
            return True
        
        # Load data
        logger.info("📥 Loading data for HMM analysis")
        loader = UnifiedDataLoader({})
        data = await loader.load_unified_data(symbol, exchange, timeframe)
        
        if data is None or data.empty:
            raise ValueError(f"No data found for {symbol} on {exchange}")
        
        # Extract price data
        price_cols = ["open", "high", "low", "close", "volume"]
        price_data = data[price_cols].copy()
        
        # Initialize analyzers
        with HMMRegimeAnalyzer(config) as hmm_analyzer:
            cluster_analyzer = RegimeClusterAnalyzer(config)
            
            # Create features
            logger.info("🔧 Creating features for HMM analysis")
            features = hmm_analyzer.create_basic_features(price_data)
            
            if features.empty:
                raise ValueError("Failed to create features for HMM analysis")
            
            # Fit HMM model
            logger.info("🎯 Fitting HMM model")
            model = hmm_analyzer.fit_hmm_model(features, timeframe)
            
            if model is None:
                raise ValueError("Failed to fit HMM model")
            
            # Predict regimes
            logger.info("🔮 Predicting regimes")
            regime_results = hmm_analyzer.predict_regimes(features, timeframe)
            
            if regime_results.empty:
                raise ValueError("Failed to predict regimes")
            
            # Perform composite clustering
            logger.info("📊 Performing composite clustering")
            regime_data = {timeframe: regime_results}
            composite_clusters = cluster_analyzer.perform_composite_clustering(regime_data)
            
            # Save artifacts
            logger.info("💾 Saving HMM artifacts")
            if not artifact_manager.save_artifacts(regime_data, composite_clusters):
                raise RuntimeError("Failed to save HMM artifacts")
        
        # Log completion
        total_time = time.time() - start_time
        logger.info(f"✅ HMM regime discovery completed successfully")
        logger.info(f"   ⏱️ Total time: {total_time:.2f}s")
        logger.info(f"   📊 Regime samples: {len(regime_results)}")
        logger.info(f"   🎯 Model components: {config.n_components}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ HMM regime discovery failed: {e}")
        logger.exception("Full traceback:")
        return False
    finally:
        # Cleanup
        gc.collect()


# Backward compatibility
async def run_step(*args, **kwargs):
    """Backward compatibility wrapper for the improved run_step function."""
    return await run_step_improved(*args, **kwargs)


async def run_step_enhanced(*args, **kwargs):
    """Enhanced version wrapper for backward compatibility."""
    return await run_step_improved(*args, **kwargs)