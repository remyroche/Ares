#!/usr/bin/env python3
"""
Enhanced HMM Composite Manager with Consolidated Functionality

This enhanced manager consolidates functionality from multiple HMM clustering files:
- Memory management (using existing M1 utilities)
- Bayesian optimization (consolidated from 3 files)
- Feature engineering (consolidated from 3 files)
- Validation (consolidated from 3 files)

Centralized manager for HMM composite cluster files that can be used by:
- step3_hmm_regime_discovery (to create files)
- VectorizedAdvancedFeatureEngineering (to check if files exist)
- CompositeHMMRegimeSystem (to load files)

This ensures consistent behavior and prevents infinite loops.
"""

import contextlib
import json
import os
import time
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

# Import existing utilities
from .logger import system_logger
from ..core.decorators import handles_errors

# Import M1 optimization utilities (replacing memory management files)
try:
    from .m1_gpu_utils import get_m1_gpu_manager, M1GPUManager
    from .m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
    from .m1_cpu_optimizer import get_m1_cpu_optimizer, M1CPUOptimizer
    M1_UTILITIES_AVAILABLE = True
except ImportError:
    M1_UTILITIES_AVAILABLE = False

# Import optimization libraries
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from sklearn.feature_selection import (
        SelectKBest, SelectPercentile, RFE, SelectFromModel,
        mutual_info_regression, f_regression, chi2
    )
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.decomposition import PCA, FastICA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Module-level sets to avoid duplicate logs across multiple instances
_GLOBAL_LOGGED_LOADS: set[str] = set()
_GLOBAL_LOGGED_EVENTS: set[str] = set()

@dataclass
class BayesianOptimizationConfig:
    """Configuration for Bayesian optimization."""
    n_trials: int = 100
    timeout: Optional[int] = None
    n_jobs: int = 1
    study_name: str = "hmm_optimization"
    storage_url: Optional[str] = None
    load_if_exists: bool = True

@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering."""
    max_features: int = 100
    feature_selection_method: str = "mutual_info"
    scaling_method: str = "standard"
    dimensionality_reduction: bool = True
    n_components: int = 50

@dataclass
class ValidationConfig:
    """Configuration for validation."""
    min_regime_samples: int = 100
    max_regime_imbalance: float = 0.8
    min_silhouette_score: float = 0.3
    max_convergence_iterations: int = 100

class EnhancedHMMCompositeManager:
    """Enhanced HMM Composite Manager with consolidated functionality."""

    def __init__(self) -> None:
        self.logger = system_logger.getChild("EnhancedHMMCompositeManager")
        self._cache: dict[str, dict[str, Any]] = {}
        self._logged_loads = _GLOBAL_LOGGED_LOADS
        self._logged_events = _GLOBAL_LOGGED_EVENTS

        # Enhanced features
        self._file_metadata_cache: dict[str, dict[str, Any]] = {}
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600  # Cleanup cache every hour

        # Initialize M1 utilities for memory management
        self._initialize_m1_utilities()
        
        # Initialize optimization components
        self._initialize_optimization_components()
        
        # Initialize validation components
        self._initialize_validation_components()

    def _initialize_m1_utilities(self) -> None:
        """Initialize M1 optimization utilities (replacing memory management files)."""
        if M1_UTILITIES_AVAILABLE:
            try:
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
                self.logger.info("✅ M1 utilities initialized for memory management")
            except Exception as e:
                self.logger.warning(f"⚠️ M1 utilities initialization failed: {e}")
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
        else:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.logger.info("ℹ️ M1 utilities not available, using fallback implementations")

    def _initialize_optimization_components(self) -> None:
        """Initialize Bayesian optimization components."""
        self.bayesian_config = BayesianOptimizationConfig()
        self.feature_config = FeatureEngineeringConfig()
        
        if OPTUNA_AVAILABLE:
            self.logger.info("✅ Bayesian optimization components initialized")
        else:
            self.logger.warning("⚠️ Optuna not available, Bayesian optimization disabled")

    def _initialize_validation_components(self) -> None:
        """Initialize validation components."""
        self.validation_config = ValidationConfig()
        self.logger.info("✅ Validation components initialized")

    # Original HMM Composite Manager functionality
    def get_composite_cluster_file_path(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        base_path: str | None = None,
    ) -> str:
        """Get the file path for HMM composite cluster data."""
        if base_path is None:
            base_path = "data_cache"
        
        filename = f"hmm_composite_clusters_{exchange}_{symbol}_{timeframe}.parquet"
        return os.path.join(base_path, "hmm_clusters", filename)

    def file_exists(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        base_path: str | None = None,
    ) -> bool:
        """Check if HMM composite cluster file exists."""
        file_path = self.get_composite_cluster_file_path(exchange, symbol, timeframe, base_path)
        return os.path.exists(file_path)

    def load_composite_clusters(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        base_path: str | None = None,
    ) -> dict[str, Any] | None:
        """Load HMM composite cluster data."""
        file_path = self.get_composite_cluster_file_path(exchange, symbol, timeframe, base_path)
        
        if not os.path.exists(file_path):
            return None
        
        try:
            # Use memory optimizer if available
            if self.memory_optimizer:
                data = self.memory_optimizer.load_dataframe(file_path)
            else:
                data = pd.read_parquet(file_path)
            
            return {
                'data': data,
                'file_path': file_path,
                'metadata': self._get_file_metadata(file_path)
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to load composite clusters: {e}")
            return None

    def save_composite_clusters(
        self,
        data: pd.DataFrame,
        exchange: str,
        symbol: str,
        timeframe: str,
        base_path: str | None = None,
    ) -> bool:
        """Save HMM composite cluster data."""
        file_path = self.get_composite_cluster_file_path(exchange, symbol, timeframe, base_path)
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Use memory optimizer if available
            if self.memory_optimizer:
                self.memory_optimizer.save_dataframe(data, file_path)
            else:
                data.to_parquet(file_path, index=False)
            
            # Update metadata cache
            self._update_file_metadata(file_path, data)
            
            self.logger.info(f"✅ Saved composite clusters to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to save composite clusters: {e}")
            return False

    # Consolidated Bayesian Optimization functionality
    def optimize_hmm_parameters(
        self,
        data: pd.DataFrame,
        config: Optional[BayesianOptimizationConfig] = None
    ) -> Dict[str, Any]:
        """Optimize HMM parameters using Bayesian optimization."""
        if not OPTUNA_AVAILABLE:
            self.logger.warning("⚠️ Optuna not available, using default parameters")
            return self._get_default_hmm_parameters()
        
        config = config or self.bayesian_config
        
        def objective(trial):
            # Define parameter space
            n_components = trial.suggest_int('n_components', 2, 12)
            covariance_type = trial.suggest_categorical('covariance_type', 
                ['full', 'tied', 'diag', 'spherical'])
            n_iter = trial.suggest_int('n_iter', 50, 500)
            tol = trial.suggest_float('tol', 1e-8, 1e-1, log=True)
            
            try:
                # Create and fit HMM model
                from hmmlearn import hmm
                model = hmm.GaussianHMM(
                    n_components=n_components,
                    covariance_type=covariance_type,
                    n_iter=n_iter,
                    tol=tol,
                    random_state=42
                )
                
                # Prepare data
                X = data.select_dtypes(include=[np.number]).fillna(0)
                if len(X) < n_components:
                    return float('inf')
                
                model.fit(X)
                
                # Calculate score (negative log likelihood)
                score = model.score(X)
                return score
                
            except Exception as e:
                self.logger.warning(f"⚠️ Trial failed: {e}")
                return float('inf')
        
        try:
            study = optuna.create_study(
                direction='maximize',
                study_name=config.study_name,
                storage=config.storage_url,
                load_if_exists=config.load_if_exists
            )
            
            study.optimize(
                objective,
                n_trials=config.n_trials,
                timeout=config.timeout,
                n_jobs=config.n_jobs
            )
            
            best_params = study.best_params
            best_score = study.best_value
            
            self.logger.info(f"✅ Bayesian optimization completed. Best score: {best_score:.4f}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'study': study,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"❌ Bayesian optimization failed: {e}")
            return {'success': False, 'error': str(e)}

    # Consolidated Feature Engineering functionality
    def engineer_features(
        self,
        data: pd.DataFrame,
        config: Optional[FeatureEngineeringConfig] = None
    ) -> pd.DataFrame:
        """Engineer features for HMM regime discovery."""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("⚠️ Scikit-learn not available, returning original data")
            return data
        
        config = config or self.feature_config
        
        try:
            # Select numeric columns only
            numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
            
            if len(numeric_data.columns) == 0:
                self.logger.warning("⚠️ No numeric columns found")
                return data
            
            # Feature selection
            if config.feature_selection_method == "mutual_info":
                selector = SelectKBest(mutual_info_regression, k=min(config.max_features, len(numeric_data.columns)))
            elif config.feature_selection_method == "f_score":
                selector = SelectKBest(f_regression, k=min(config.max_features, len(numeric_data.columns)))
            else:
                selector = SelectKBest(f_regression, k=min(config.max_features, len(numeric_data.columns)))
            
            selected_features = selector.fit_transform(numeric_data, numeric_data.mean(axis=1))
            selected_columns = numeric_data.columns[selector.get_support()]
            
            # Feature scaling
            if config.scaling_method == "standard":
                scaler = StandardScaler()
            elif config.scaling_method == "minmax":
                scaler = MinMaxScaler()
            elif config.scaling_method == "robust":
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            scaled_features = scaler.fit_transform(selected_features)
            
            # Dimensionality reduction
            if config.dimensionality_reduction and len(selected_columns) > config.n_components:
                pca = PCA(n_components=config.n_components)
                reduced_features = pca.fit_transform(scaled_features)
                
                # Create feature names
                feature_names = [f"pca_{i}" for i in range(config.n_components)]
            else:
                reduced_features = scaled_features
                feature_names = selected_columns.tolist()
            
            # Create result DataFrame
            result = pd.DataFrame(reduced_features, columns=feature_names, index=data.index)
            
            self.logger.info(f"✅ Feature engineering completed. Shape: {result.shape}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Feature engineering failed: {e}")
            return data

    # Consolidated Validation functionality
    def validate_hmm_results(
        self,
        data: pd.DataFrame,
        regime_labels: np.ndarray,
        config: Optional[ValidationConfig] = None
    ) -> Dict[str, Any]:
        """Validate HMM regime discovery results."""
        config = config or self.validation_config
        
        try:
            validation_results = {
                'regime_counts': {},
                'regime_imbalance': 0.0,
                'silhouette_score': 0.0,
                'validation_passed': False,
                'warnings': [],
                'errors': []
            }
            
            # Check regime counts
            unique_regimes, counts = np.unique(regime_labels, return_counts=True)
            validation_results['regime_counts'] = dict(zip(unique_regimes, counts))
            
            # Check minimum regime samples
            min_count = min(counts)
            if min_count < config.min_regime_samples:
                validation_results['errors'].append(
                    f"Regime with {min_count} samples below minimum {config.min_regime_samples}"
                )
            
            # Check regime imbalance
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = min_count / max_count
            validation_results['regime_imbalance'] = imbalance_ratio
            
            if imbalance_ratio < config.max_regime_imbalance:
                validation_results['warnings'].append(
                    f"Regime imbalance {imbalance_ratio:.3f} below threshold {config.max_regime_imbalance}"
                )
            
            # Calculate silhouette score if possible
            if len(unique_regimes) > 1 and len(data) > len(unique_regimes):
                try:
                    from sklearn.metrics import silhouette_score
                    numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
                    if len(numeric_data.columns) > 0:
                        silhouette = silhouette_score(numeric_data, regime_labels)
                        validation_results['silhouette_score'] = silhouette
                        
                        if silhouette < config.min_silhouette_score:
                            validation_results['warnings'].append(
                                f"Silhouette score {silhouette:.3f} below threshold {config.min_silhouette_score}"
                            )
                except Exception as e:
                    validation_results['warnings'].append(f"Could not calculate silhouette score: {e}")
            
            # Overall validation
            validation_results['validation_passed'] = len(validation_results['errors']) == 0
            
            self.logger.info(f"✅ Validation completed. Passed: {validation_results['validation_passed']}")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"❌ Validation failed: {e}")
            return {
                'validation_passed': False,
                'errors': [str(e)],
                'warnings': []
            }

    def _get_default_hmm_parameters(self) -> Dict[str, Any]:
        """Get default HMM parameters when optimization is not available."""
        return {
            'n_components': 4,
            'covariance_type': 'full',
            'n_iter': 100,
            'tol': 1e-3,
            'success': True
        }

    def _get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get file metadata."""
        try:
            if file_path in self._file_metadata_cache:
                return self._file_metadata_cache[file_path]
            
            stat = os.stat(file_path)
            metadata = {
                'size_bytes': stat.st_size,
                'modified_time': stat.st_mtime,
                'created_time': stat.st_ctime
            }
            
            self._file_metadata_cache[file_path] = metadata
            return metadata
        except Exception:
            return {}

    def _update_file_metadata(self, file_path: str, data: pd.DataFrame) -> None:
        """Update file metadata cache."""
        metadata = {
            'size_bytes': len(data.to_parquet()),
            'modified_time': time.time(),
            'created_time': time.time(),
            'shape': data.shape,
            'columns': list(data.columns)
        }
        self._file_metadata_cache[file_path] = metadata

    def clear_cache(
        self,
        exchange: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> None:
        """Clear cache entries for specific or all files."""
        if exchange is None and symbol is None and timeframe is None:
            self._cache.clear()
            self.logger.info("🧹 Cleared all HMM composite manager cache")
        else:
            keys_to_remove: list[str] = []
            for key in list(self._cache.keys()):
                if exchange and exchange not in key:
                    continue
                if symbol and symbol not in key:
                    continue
                if timeframe and timeframe not in key:
                    continue
                keys_to_remove.append(key)

            for key in keys_to_remove:
                with contextlib.suppress(Exception):
                    del self._cache[key]

            self.logger.info(
                f"🧹 Cleared {len(keys_to_remove)} cache entries for {exchange}_{symbol}_{timeframe}",
            )

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self._cache)
        total_size_mb = sum(
            len(str(v.get("data", ""))) / (1024 * 1024)
            for v in self._cache.values()
            if isinstance(v, dict) and "data" in v
        )

        return {
            "total_entries": total_entries,
            "total_size_mb": total_size_mb,
            "metadata_entries": len(self._file_metadata_cache),
        }

# Global instance for backward compatibility
hmm_composite_manager = EnhancedHMMCompositeManager()

# Export for backward compatibility
HMMCompositeManager = EnhancedHMMCompositeManager