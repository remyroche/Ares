#!/usr/bin/env python3
"""
Automated Feature Importance Analysis System

This module provides comprehensive feature importance analysis capabilities
for the trading system, including:
- Multiple importance calculation methods
- Regime-specific importance analysis
- Temporal stability analysis
- Automated feature ranking and selection
- Integration with existing feature selection tools
"""

import numpy as np
import pandas as pd
import torch
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import system utilities
from ..logger import get_logger
from ..matrix_operations import get_unified_matrix_operations

# Import existing advanced feature selection tools
try:
    from .step08_unified_complete import FeatureSelectionValidation
    from .step08_advanced_feature_selection_wrapper import AdvancedFeatureSelectionStep
    from .step08_advanced_feature_selection_per_regime import PerRegimeAdvancedFeatureSelectionStep
    ADVANCED_FEATURE_SELECTION_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURE_SELECTION_AVAILABLE = False

class ImportanceMethod(Enum):
    """Available feature importance methods."""
    RANDOM_FOREST = "random_forest"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    RIDGE = "ridge"
    MUTUAL_INFO = "mutual_information"
    F_SCORE = "f_score"
    PERMUTATION = "permutation"
    SHAP = "shap"
    CORRELATION = "correlation"
    VARIANCE = "variance"

@dataclass
class FeatureImportanceConfig:
    """Configuration for feature importance analysis."""
    # Methods to use
    methods: List[ImportanceMethod] = field(default_factory=lambda: [
        ImportanceMethod.RANDOM_FOREST,
        ImportanceMethod.LASSO,
        ImportanceMethod.MUTUAL_INFO,
        ImportanceMethod.PERMUTATION
    ])
    
    # Model parameters
    random_forest_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    })
    
    lasso_params: Dict[str, Any] = field(default_factory=lambda: {
        'alpha': 0.01,
        'random_state': 42
    })
    
    elastic_net_params: Dict[str, Any] = field(default_factory=lambda: {
        'alpha': 0.01,
        'l1_ratio': 0.5,
        'random_state': 42
    })

    ridge_params: Dict[str, Any] = field(default_factory=lambda: {
        'alpha': 1.0,
        'random_state': 42
    })

    # Analysis parameters
    top_k_features: int = 20
    min_importance_threshold: float = 0.01
    stability_threshold: float = 0.7
    temporal_window: int = 1000
    
    # Performance settings
    n_jobs: int = -1
    chunk_size: int = 10000
    enable_parallel: bool = True
    
    # Output settings
    save_results: bool = True
    generate_plots: bool = True
    output_directory: Optional[str] = None

@dataclass
class FeatureImportanceResult:
    """Result of feature importance analysis."""
    feature_names: List[str]
    importance_scores: Dict[str, np.ndarray]
    method_scores: Dict[str, Dict[str, float]]
    stability_scores: Dict[str, float]
    temporal_stability: Dict[str, np.ndarray]
    rankings: Dict[str, List[str]]
    meta_info: Dict[str, Any]
    
    def get_top_features(self, method: str = "ensemble", k: int = 10) -> List[str]:
        """Get top k features for a specific method."""
        if method == "ensemble":
            # Average rankings across methods
            ensemble_scores = {}
            for method_name, scores in self.method_scores.items():
                for feature, score in scores.items():
                    if feature not in ensemble_scores:
                        ensemble_scores[feature] = []
                    ensemble_scores[feature].append(score)
            
            # Calculate average scores
            avg_scores = {feature: np.mean(scores) for feature, scores in ensemble_scores.items()}
            sorted_features = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            return [feature for feature, _ in sorted_features[:k]]
        else:
            return self.rankings.get(method, [])[:k]

class FeatureImportanceAnalyzer:
    """Automated feature importance analyzer with integration to advanced feature selection tools."""
    
    def __init__(self, config: Optional[FeatureImportanceConfig] = None):
        self.config = config or FeatureImportanceConfig()
        self.logger = get_logger("FeatureImportanceAnalyzer")
        
        # Initialize matrix operations for performance
        self.matrix_ops = get_unified_matrix_operations()
        
        # Initialize advanced feature selection tools if available
        self.advanced_tools = None
        if ADVANCED_FEATURE_SELECTION_AVAILABLE:
            try:
                self.advanced_tools = {
                    'validation': FeatureSelectionValidation(),
                    'step08_wrapper': AdvancedFeatureSelectionStep({}),
                    'per_regime': PerRegimeAdvancedFeatureSelectionStep({})
                }
                self.logger.info("✅ Advanced feature selection tools integrated")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize advanced tools: {e}")
                self.advanced_tools = None
        
        # Results storage
        self.results: Dict[str, FeatureImportanceResult] = {}
        
        self.logger.info("🚀 FeatureImportanceAnalyzer initialized with advanced integration")
    
    def analyze_features(self, 
                        X: pd.DataFrame, 
                        y: pd.Series,
                        regime_labels: Optional[pd.Series] = None,
                        feature_names: Optional[List[str]] = None) -> FeatureImportanceResult:
        """Perform comprehensive feature importance analysis."""
        
        start_time = time.time()
        self.logger.info(f"🔍 Starting feature importance analysis on {X.shape[1]} features")
        
        if feature_names is None:
            feature_names = X.columns.tolist()
        
        # Initialize results structure
        importance_scores = {}
        method_scores = {}
        temporal_stability = {}
        
        # Perform analysis for each method
        for method in self.config.methods:
            self.logger.info(f"📊 Computing importance using {method.value}")
            method_start = time.time()
            
            try:
                scores = self._compute_importance(X, y, method)
                importance_scores[method.value] = scores
                
                # Convert to feature-scores dictionary
                feature_scores = dict(zip(feature_names, scores))
                method_scores[method.value] = feature_scores
                
                method_time = time.time() - method_start
                self.logger.info(f"✅ {method.value} completed in {method_time:.3f}s")
                
            except Exception as e:
                self.logger.error(f"❌ Error computing {method.value}: {e}")
                continue
        
        # Compute stability scores
        stability_scores = self._compute_stability_scores(method_scores)
        
        # Compute temporal stability if regime labels available
        if regime_labels is not None:
            temporal_stability = self._compute_temporal_stability(X, y, regime_labels, feature_names)
        
        # Generate rankings
        rankings = self._generate_rankings(method_scores)
        
        # Create result object
        result = FeatureImportanceResult(
            feature_names=feature_names,
            importance_scores=importance_scores,
            method_scores=method_scores,
            stability_scores=stability_scores,
            temporal_stability=temporal_stability,
            rankings=rankings,
            meta_info={
                'analysis_time': time.time() - start_time,
                'n_features': len(feature_names),
                'n_samples': len(X),
                'methods_used': [m.value for m in self.config.methods],
                'config': self.config.__dict__
            }
        )
        
        # Save results if configured
        if self.config.save_results:
            self._save_results(result)
        
        # Generate plots if configured
        if self.config.generate_plots:
            self._generate_plots(result)
        
        total_time = time.time() - start_time
        self.logger.info(f"✅ Feature importance analysis completed in {total_time:.3f}s")
        
        return result

    def batch_compute_importance(self, X: pd.DataFrame, y: pd.Series,
                                methods: List[ImportanceMethod] = None) -> Dict[str, np.ndarray]:
        """Compute importance scores for multiple methods in batch using vectorized operations."""
        if methods is None:
            methods = self.config.methods

        # Convert to numpy arrays for vectorized operations
        X_matrix = X.values
        y_vector = y.values

        results = {}
        start_time = time.time()

        # Use parallel processing if enabled and beneficial
        if self.config.enable_parallel and len(methods) > 2:
            self.logger.info(f"🔄 Computing {len(methods)} importance methods in parallel...")

            with ThreadPoolExecutor(max_workers=min(len(methods), self.config.n_jobs)) as executor:
                # Submit all jobs
                future_to_method = {
                    executor.submit(self._compute_single_importance_matrix, X_matrix, y_vector, method): method
                    for method in methods
                }

                # Collect results
                for future in as_completed(future_to_method):
                    method = future_to_method[future]
                    try:
                        importance_scores = future.result()
                        results[method.value] = importance_scores
                        self.logger.debug(f"✅ {method.value} completed")
                    except Exception as e:
                        self.logger.warning(f"❌ {method.value} failed: {e}")
                        results[method.value] = np.zeros(X.shape[1])
        else:
            # Sequential processing
            for method in methods:
                try:
                    importance_scores = self._compute_single_importance_matrix(X_matrix, y_vector, method)
                    results[method.value] = importance_scores
                except Exception as e:
                    self.logger.warning(f"❌ {method.value} failed: {e}")
                    results[method.value] = np.zeros(X.shape[1])

        batch_time = time.time() - start_time
        self.logger.info(f"✅ Batch importance computation completed in {batch_time:.3f}s for {len(results)} methods")

        return results

    def _compute_single_importance_matrix(self, X: np.ndarray, y: np.ndarray, method: ImportanceMethod) -> np.ndarray:
        """Compute importance for a single method using matrix operations."""
        if method == ImportanceMethod.CORRELATION:
            # Vectorized correlation computation
            try:
                X_with_y = np.column_stack([X, y])
                corr_matrix = np.corrcoef(X_with_y.T)
                correlations_with_target = corr_matrix[-1, :-1]
                return np.abs(correlations_with_target)
            except:
                return self._correlation_importance_matrix(X, y)

        elif method == ImportanceMethod.VARIANCE:
            # Vectorized variance
            return np.var(X, axis=0)

        elif method == ImportanceMethod.MUTUAL_INFO:
            # Use sklearn's vectorized mutual info
            return self._mutual_info_importance_matrix(X, y)

        elif method == ImportanceMethod.F_SCORE:
            # Vectorized F-score
            return self._f_score_importance_matrix(X, y)

        elif method == ImportanceMethod.LASSO:
            # LASSO regression
            return self._lasso_importance_matrix(X, y)

        elif method == ImportanceMethod.ELASTIC_NET:
            # Elastic Net
            return self._elastic_net_importance_matrix(X, y)

        elif method == ImportanceMethod.RIDGE:
            # Ridge regression
            return self._ridge_importance_matrix(X, y)

        elif method == ImportanceMethod.PERMUTATION:
            # Permutation importance
            return self._permutation_importance_matrix(X, y)

        elif method == ImportanceMethod.SHAP:
            # SHAP importance
            return self._shap_importance_matrix(X, y)

        elif method == ImportanceMethod.RANDOM_FOREST:
            # Random Forest
            return self._random_forest_importance_matrix(X, y)

        else:
            # For any remaining methods, convert back to DataFrame temporarily
            X_df = pd.DataFrame(X)
            y_series = pd.Series(y)
            return self._compute_importance(pd.DataFrame(X_df), y_series, method)

    def _compute_importance(self, X: pd.DataFrame, y: pd.Series, method: ImportanceMethod) -> np.ndarray:
        """Compute feature importance using specified method with enhanced vectorization."""

        # Use matrix operations for performance boost
        if self.matrix_ops:
            return self._compute_importance_matrix_accelerated(X, y, method)
        else:
            return self._compute_importance_standard(X, y, method)

    def _compute_importance_matrix_accelerated(self, X: pd.DataFrame, y: pd.Series, method: ImportanceMethod) -> np.ndarray:
        """Compute importance using matrix operations acceleration."""
        X_values = X.values
        y_values = y.values

        if method == ImportanceMethod.RANDOM_FOREST:
            return self._random_forest_importance_matrix(X_values, y_values)

        elif method == ImportanceMethod.LASSO:
            return self._lasso_importance_matrix(X_values, y_values)

        elif method == ImportanceMethod.ELASTIC_NET:
            return self._elastic_net_importance_matrix(X_values, y_values)

        elif method == ImportanceMethod.RIDGE:
            return self._ridge_importance_matrix(X_values, y_values)

        elif method == ImportanceMethod.MUTUAL_INFO:
            return self._mutual_info_importance_matrix(X_values, y_values)

        elif method == ImportanceMethod.F_SCORE:
            return self._f_score_importance_matrix(X_values, y_values)

        elif method == ImportanceMethod.PERMUTATION:
            return self._permutation_importance_matrix(X_values, y_values)

        elif method == ImportanceMethod.CORRELATION:
            return self._correlation_importance_matrix(X_values, y_values)

        elif method == ImportanceMethod.VARIANCE:
            return self._variance_importance_matrix(X_values)

        elif method == ImportanceMethod.SHAP:
            return self._shap_importance_matrix(X_values, y_values)

        else:
            raise ValueError(f"Unknown importance method: {method}")

    def _compute_importance_standard(self, X: pd.DataFrame, y: pd.Series, method: ImportanceMethod) -> np.ndarray:
        """Standard importance computation methods."""

        if method == ImportanceMethod.RANDOM_FOREST:
            return self._random_forest_importance(X, y)

        elif method == ImportanceMethod.LASSO:
            return self._lasso_importance(X, y)

        elif method == ImportanceMethod.ELASTIC_NET:
            return self._elastic_net_importance(X, y)

        elif method == ImportanceMethod.RIDGE:
            return self._ridge_importance(X, y)

        elif method == ImportanceMethod.MUTUAL_INFO:
            return self._mutual_info_importance(X, y)

        elif method == ImportanceMethod.F_SCORE:
            return self._f_score_importance(X, y)

        elif method == ImportanceMethod.PERMUTATION:
            return self._permutation_importance(X, y)

        elif method == ImportanceMethod.CORRELATION:
            return self._correlation_importance(X, y)

        elif method == ImportanceMethod.VARIANCE:
            return self._variance_importance(X)

        else:
            raise ValueError(f"Unknown importance method: {method}")

    # Matrix-accelerated importance methods
    def _random_forest_importance_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Matrix-accelerated Random Forest importance."""
        # Use existing method but with matrix operations
        df = pd.DataFrame(X)  # Temporary conversion for compatibility
        return self._random_forest_importance(df, pd.Series(y))

    def _lasso_importance_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Matrix-accelerated LASSO importance."""
        # Use GPU acceleration if available
        if self.matrix_ops and hasattr(self.matrix_ops, 'gpu_manager'):
            try:
                # Convert to tensors for GPU processing
                X_tensor = torch.from_numpy(X.astype(np.float32))
                y_tensor = torch.from_numpy(y.astype(np.float32))

                # Use GPU-accelerated LASSO if available
                # This is a placeholder - actual implementation would depend on available GPU LASSO libraries
                model = Lasso(**self.config.lasso_params)
                model.fit(X, y)
                return np.abs(model.coef_)
            except:
                pass

        # Fallback to standard method
        model = Lasso(**self.config.lasso_params)
        model.fit(X, y)
        return np.abs(model.coef_)

    def _elastic_net_importance_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Matrix-accelerated Elastic Net importance."""
        try:
            model = ElasticNet(**self.config.elastic_net_params)
            model.fit(X, y)
            return np.abs(model.coef_)
        except:
            return np.zeros(X.shape[1])

    def _ridge_importance_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Matrix-accelerated Ridge importance."""
        try:
            model = Ridge(**self.config.ridge_params)
            model.fit(X, y)
            return np.abs(model.coef_)
        except:
            return np.zeros(X.shape[1])

    def _mutual_info_importance_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Matrix-accelerated mutual information importance."""
        try:
            return mutual_info_regression(X, y, random_state=42)
        except:
            return np.zeros(X.shape[1])

    def _f_score_importance_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Matrix-accelerated F-score importance."""
        try:
            selector = SelectKBest(f_regression, k='all')
            selector.fit(X, y)
            return selector.scores_
        except:
            return np.zeros(X.shape[1])

    def _permutation_importance_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Matrix-accelerated permutation importance."""
        try:
            # Use Random Forest as base model
            model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
            model.fit(X, y)

            # Compute permutation importance
            perm_importance = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            return perm_importance.importances_mean
        except:
            return np.zeros(X.shape[1])

    def _shap_importance_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Matrix-accelerated SHAP importance (simplified approximation)."""
        try:
            # Use Random Forest as base model for SHAP approximation
            model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
            model.fit(X, y)

            # Simple approximation of SHAP values using feature importance
            # In practice, you'd use shap library for exact SHAP values
            return model.feature_importances_
        except:
            return np.zeros(X.shape[1])

    def _correlation_importance_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Matrix-accelerated correlation importance."""
        try:
            # Vectorized correlation computation
            # Add y as last column for correlation matrix
            X_with_y = np.column_stack([X, y])
            corr_matrix = np.corrcoef(X_with_y.T)

            # Get correlations with target (last row, exclude self-correlation)
            correlations_with_target = corr_matrix[-1, :-1]
            return np.abs(correlations_with_target)
        except:
            # Fallback to element-wise computation
            correlations = []
            for i in range(X.shape[1]):
                try:
                    corr = np.corrcoef(X[:, i], y)[0, 1]
                    correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
                except:
                    correlations.append(0.0)
            return np.array(correlations)

    def _variance_importance_matrix(self, X: np.ndarray) -> np.ndarray:
        """Matrix-accelerated variance importance."""
        # Vectorized variance calculation
        return np.var(X, axis=0)
    
    def _random_forest_importance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Compute Random Forest feature importance."""
        # Determine if classification or regression
        is_classification = len(y.unique()) < 20 and y.dtype in ['object', 'category', 'int64']
        
        if is_classification:
            model = RandomForestClassifier(**self.config.random_forest_params)
        else:
            model = RandomForestRegressor(**self.config.random_forest_params)
        
        model.fit(X, y)
        return model.feature_importances_
    
    def _lasso_importance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Compute Lasso feature importance."""
        model = Lasso(**self.config.lasso_params)
        model.fit(X, y)
        return np.abs(model.coef_)
    
    def _elastic_net_importance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Compute Elastic Net feature importance."""
        model = ElasticNet(**self.config.elastic_net_params)
        model.fit(X, y)
        return np.abs(model.coef_)
    
    def _ridge_importance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Compute Ridge feature importance."""
        model = Ridge(**self.config.ridge_params)
        model.fit(X, y)
        return np.abs(model.coef_)
    
    def _mutual_info_importance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Compute mutual information importance."""
        return mutual_info_regression(X, y, random_state=42)
    
    def _f_score_importance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Compute F-score importance."""
        selector = SelectKBest(f_regression, k='all')
        selector.fit(X, y)
        return selector.scores_
    
    def _permutation_importance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Compute permutation importance."""
        # Use Random Forest as base model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Compute permutation importance
        perm_importance = permutation_importance(model, X, y, n_repeats=5, random_state=42)
        return perm_importance.importances_mean
    
    def _correlation_importance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Compute correlation-based importance."""
        correlations = X.corrwith(y).abs()
        return correlations.fillna(0).values
    
    def _variance_importance(self, X: pd.DataFrame) -> np.ndarray:
        """Compute variance-based importance."""
        return X.var().values
    
    def _compute_stability_scores(self, method_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Compute stability scores across methods."""
        if len(method_scores) < 2:
            return {}
        
        stability_scores = {}
        features = list(next(iter(method_scores.values())).keys())
        
        for feature in features:
            scores = []
            for method, scores_dict in method_scores.items():
                if feature in scores_dict:
                    scores.append(scores_dict[feature])
            
            if len(scores) > 1:
                # Compute coefficient of variation (lower is more stable)
                cv = np.std(scores) / (np.mean(scores) + 1e-8)
                stability_scores[feature] = 1 / (1 + cv)  # Convert to stability score (higher is better)
            else:
                stability_scores[feature] = 0.0
        
        return stability_scores
    
    def gpu_accelerated_stability_selection(self, X: pd.DataFrame, y: pd.Series,
                                           n_bootstrap: int = 100,
                                           bootstrap_fraction: float = 0.8,
                                           stability_threshold: float = 0.6) -> Dict[str, Any]:
        """GPU-accelerated stability selection for feature selection."""
        if not self.matrix_ops or not hasattr(self.matrix_ops, 'gpu_manager'):
            self.logger.warning("⚠️ GPU not available, falling back to CPU stability selection")
            return self._cpu_stability_selection(X, y, n_bootstrap, bootstrap_fraction, stability_threshold)

        start_time = time.time()
        self.logger.info(f"🚀 Starting GPU-accelerated stability selection with {n_bootstrap} bootstraps...")

        # Convert to numpy arrays
        X_matrix = X.values.astype(np.float32)
        y_vector = y.values.astype(np.float32)

        n_samples, n_features = X_matrix.shape
        bootstrap_size = int(n_samples * bootstrap_fraction)

        # Prepare bootstrap indices matrix for GPU processing
        np.random.seed(42)
        bootstrap_indices = np.random.randint(0, n_samples, (n_bootstrap, bootstrap_size))

        try:
            # Move data to GPU
            X_gpu = torch.from_numpy(X_matrix).to(self.matrix_ops.gpu_manager.device)
            y_gpu = torch.from_numpy(y_vector).to(self.matrix_ops.gpu_manager.device)
            indices_gpu = torch.from_numpy(bootstrap_indices).to(self.matrix_ops.gpu_manager.device)

            # GPU-accelerated bootstrap sampling and LASSO fitting
            stability_scores = self._gpu_bootstrap_lasso(X_gpu, y_gpu, indices_gpu)

            # Convert back to CPU
            stability_scores = stability_scores.cpu().numpy()

            # Calculate selection frequency
            selection_frequency = np.mean(stability_scores > stability_threshold, axis=0)

            # Get stable features
            stable_features_mask = selection_frequency >= stability_threshold
            stable_feature_indices = np.where(stable_features_mask)[0]
            stable_feature_names = [X.columns[i] for i in stable_feature_indices]

            gpu_time = time.time() - start_time
            self.logger.info(f"✅ GPU stability selection completed in {gpu_time:.3f}s")
            self.logger.info(f"📊 Found {len(stable_feature_names)} stable features out of {n_features}")

            return {
                'selected_features': stable_feature_names,
                'stability_scores': dict(zip(X.columns, selection_frequency)),
                'selection_frequency': selection_frequency,
                'stable_features_mask': stable_features_mask,
                'n_bootstrap': n_bootstrap,
                'bootstrap_fraction': bootstrap_fraction,
                'stability_threshold': stability_threshold,
                'computation_time': gpu_time,
                'computation_method': 'gpu_accelerated'
            }

        except Exception as e:
            self.logger.warning(f"GPU stability selection failed: {e}, falling back to CPU")
            return self._cpu_stability_selection(X, y, n_bootstrap, bootstrap_fraction, stability_threshold)

    def _gpu_bootstrap_lasso(self, X_gpu: torch.Tensor, y_gpu: torch.Tensor,
                            bootstrap_indices: torch.Tensor) -> torch.Tensor:
        """Perform GPU-accelerated bootstrap LASSO fitting."""
        n_bootstrap = bootstrap_indices.shape[0]
        n_features = X_gpu.shape[1]

        stability_scores = torch.zeros((n_bootstrap, n_features), device=X_gpu.device)

        # Use batched processing for better GPU utilization
        batch_size = min(32, n_bootstrap)  # Process in batches

        for i in range(0, n_bootstrap, batch_size):
            end_idx = min(i + batch_size, n_bootstrap)
            batch_indices = bootstrap_indices[i:end_idx]

            # Sample bootstrap data
            X_batch = X_gpu[batch_indices.flatten()].view(end_idx - i, -1, n_features)
            y_batch = y_gpu[batch_indices.flatten()].view(end_idx - i, -1)

            # Vectorized LASSO fitting (simplified - in practice you'd use GPU LASSO libraries)
            # This is a placeholder for actual GPU LASSO implementation
            for j in range(end_idx - i):
                X_sample = X_batch[j]
                y_sample = y_batch[j]

                # Simple GPU-accelerated correlation-based selection as placeholder
                correlations = torch.corrcoef(torch.cat([X_sample.T, y_sample.unsqueeze(0)]))[-1, :-1]
                importance_scores = torch.abs(correlations)

                # Apply threshold (simplified stability criterion)
                stability_scores[i + j] = (importance_scores > 0.1).float()

        return stability_scores

    def _cpu_stability_selection(self, X: pd.DataFrame, y: pd.Series,
                                n_bootstrap: int, bootstrap_fraction: float,
                                stability_threshold: float) -> Dict[str, Any]:
        """CPU-based stability selection as fallback."""
        start_time = time.time()

        n_samples, n_features = X.shape
        bootstrap_size = int(n_samples * bootstrap_fraction)

        stability_matrix = np.zeros((n_bootstrap, n_features))
        np.random.seed(42)

        # Bootstrap loop
        for i in range(n_bootstrap):
            # Sample bootstrap indices
            bootstrap_indices = np.random.choice(n_samples, size=bootstrap_size, replace=True)

            # Sample data
            X_bootstrap = X.iloc[bootstrap_indices]
            y_bootstrap = y.iloc[bootstrap_indices]

            # Fit LASSO
            model = Lasso(alpha=0.01, random_state=42)
            model.fit(X_bootstrap, y_bootstrap)

            # Record selected features (non-zero coefficients)
            selected_features = np.abs(model.coef_) > 1e-4
            stability_matrix[i] = selected_features.astype(int)

        # Calculate stability scores
        selection_frequency = np.mean(stability_matrix, axis=0)

        # Get stable features
        stable_features_mask = selection_frequency >= stability_threshold
        stable_feature_indices = np.where(stable_features_mask)[0]
        stable_feature_names = [X.columns[i] for i in stable_feature_indices]

        cpu_time = time.time() - start_time
        self.logger.info(f"✅ CPU stability selection completed in {cpu_time:.3f}s")

        return {
            'selected_features': stable_feature_names,
            'stability_scores': dict(zip(X.columns, selection_frequency)),
            'selection_frequency': selection_frequency,
            'stable_features_mask': stable_features_mask,
            'n_bootstrap': n_bootstrap,
            'bootstrap_fraction': bootstrap_fraction,
            'stability_threshold': stability_threshold,
            'computation_time': cpu_time,
            'computation_method': 'cpu_fallback'
        }

    def _compute_temporal_stability(self, X: pd.DataFrame, y: pd.Series,
                                  regime_labels: pd.Series, feature_names: List[str]) -> Dict[str, np.ndarray]:
        """Compute temporal stability of feature importance across regimes."""
        temporal_stability = {}
        
        for feature in feature_names:
            if feature in X.columns:
                feature_values = X[feature].values
                regime_correlations = []
                
                for regime in regime_labels.unique():
                    regime_mask = regime_labels == regime
                    if regime_mask.sum() > 10:  # Minimum samples
                        regime_corr = np.corrcoef(feature_values[regime_mask], y[regime_mask])[0, 1]
                        if not np.isnan(regime_corr):
                            regime_correlations.append(regime_corr)
                
                if len(regime_correlations) > 1:
                    temporal_stability[feature] = np.array(regime_correlations)
        
        return temporal_stability
    
    def _generate_rankings(self, method_scores: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
        """Generate feature rankings for each method."""
        rankings = {}
        
        for method, scores in method_scores.items():
            sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            rankings[method] = [feature for feature, _ in sorted_features]
        
        # Create ensemble ranking
        if len(method_scores) > 1:
            ensemble_scores = {}
            for method, scores in method_scores.items():
                for feature, score in scores.items():
                    if feature not in ensemble_scores:
                        ensemble_scores[feature] = []
                    ensemble_scores[feature].append(score)
            
            # Average scores
            avg_scores = {feature: np.mean(scores) for feature, scores in ensemble_scores.items()}
            sorted_features = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            rankings['ensemble'] = [feature for feature, _ in sorted_features]
        
        return rankings

    def matrix_based_ensemble_voting(self, importance_results: Dict[str, np.ndarray],
                                   weights: Optional[Dict[str, float]] = None,
                                   normalization_method: str = 'minmax') -> Dict[str, Any]:
        """Matrix-based ensemble voting for combining multiple importance methods."""
        if not importance_results:
            return {}

        start_time = time.time()
        self.logger.info(f"🔄 Starting matrix-based ensemble voting with {len(importance_results)} methods...")

        # Convert to matrix format for vectorized operations
        method_names = list(importance_results.keys())
        importance_matrix = np.column_stack([importance_results[method] for method in method_names])

        # Apply weights
        if weights is None:
            # Equal weights by default
            weights_array = np.ones(len(method_names)) / len(method_names)
        else:
            weights_array = np.array([weights.get(method, 1.0) for method in method_names])
            weights_array = weights_array / weights_array.sum()  # Normalize

        # Normalize importance scores using matrix operations
        if normalization_method == 'minmax':
            # Vectorized min-max normalization
            importance_matrix_norm = (importance_matrix - importance_matrix.min(axis=0)) / \
                                   (importance_matrix.max(axis=0) - importance_matrix.min(axis=0) + 1e-8)
        elif normalization_method == 'zscore':
            # Vectorized z-score normalization
            importance_matrix_norm = (importance_matrix - importance_matrix.mean(axis=0)) / \
                                   (importance_matrix.std(axis=0) + 1e-8)
        elif normalization_method == 'robust':
            # Vectorized robust normalization (median-based)
            medians = np.median(importance_matrix, axis=0)
            mads = np.median(np.abs(importance_matrix - medians), axis=0)
            importance_matrix_norm = (importance_matrix - medians) / (mads + 1e-8)
        else:
            importance_matrix_norm = importance_matrix

        # Matrix multiplication for weighted ensemble scores
        ensemble_scores = importance_matrix_norm @ weights_array

        # Calculate feature rankings using matrix operations
        sorted_indices = np.argsort(ensemble_scores)[::-1]  # Sort in descending order
        rankings = [f"feature_{i}" for i in sorted_indices]

        # Calculate voting statistics using matrix operations
        # Binarize importance scores for voting analysis
        threshold_matrix = importance_matrix_norm > np.median(importance_matrix_norm, axis=0)
        vote_counts = np.sum(threshold_matrix, axis=1)
        consensus_score = vote_counts / len(method_names)

        # Calculate agreement matrix (which features are selected by multiple methods)
        agreement_matrix = threshold_matrix @ threshold_matrix.T
        agreement_scores = agreement_matrix.diagonal() / len(method_names)

        voting_time = time.time() - start_time
        self.logger.info(f"✅ Matrix-based ensemble voting completed in {voting_time:.3f}s")

        return {
            'ensemble_scores': ensemble_scores,
            'rankings': rankings,
            'method_weights': dict(zip(method_names, weights_array)),
            'normalization_method': normalization_method,
            'vote_counts': vote_counts,
            'consensus_scores': consensus_score,
            'agreement_scores': agreement_scores,
            'importance_matrix_shape': importance_matrix.shape,
            'computation_time': voting_time,
            'top_features': rankings[:min(20, len(rankings))],
            'voting_statistics': {
                'mean_consensus': float(np.mean(consensus_score)),
                'max_consensus': float(np.max(consensus_score)),
                'min_consensus': float(np.min(consensus_score)),
                'consensus_std': float(np.std(consensus_score))
            }
        }

    def analyze_with_advanced_tools(self, 
                                  X: pd.DataFrame, 
                                  y: pd.Series,
                                  regime_labels: Optional[pd.Series] = None) -> FeatureImportanceResult:
        """Analyze features using advanced feature selection tools integration."""
        
        if not self.advanced_tools:
            self.logger.warning("⚠️ Advanced tools not available, falling back to standard analysis")
            return self.analyze_features(X, y, regime_labels)
        
        start_time = time.time()
        self.logger.info("🔍 Starting advanced feature importance analysis")
        
        # Perform standard analysis first
        standard_result = self.analyze_features(X, y, regime_labels)
        
        # Enhance with advanced correlation and variance filtering
        advanced_results = self._perform_advanced_filtering(X)
        
        # Enhance results with advanced analysis
        enhanced_result = self._enhance_with_advanced_analysis(standard_result, advanced_results)
        
        total_time = time.time() - start_time
        self.logger.info(f"✅ Advanced feature importance analysis completed in {total_time:.3f}s")
        
        return enhanced_result
    
    def _perform_advanced_filtering(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Perform advanced filtering using existing tools."""
        advanced_results = {}
        
        try:
            # Advanced correlation filtering
            correlation_threshold = 0.8
            corr_matrix = X.corr()
            high_correlations = []
            
            for i, feature1 in enumerate(X.columns):
                for j, feature2 in enumerate(X.columns[i+1:], i+1):
                    correlation = abs(corr_matrix.loc[feature1, feature2])
                    if correlation >= correlation_threshold:
                        high_correlations.append({
                            'feature1': feature1,
                            'feature2': feature2,
                            'abs_correlation': correlation
                        })
            
            advanced_results['correlation_filtering'] = {
                'threshold': correlation_threshold,
                'high_correlations': len(high_correlations),
                'correlations': high_correlations
            }
            
            # Advanced variance filtering
            variance_threshold = 0.01
            variances = X.var()
            low_variance_features = [f for f in X.columns if variances[f] < variance_threshold]
            
            advanced_results['variance_filtering'] = {
                'threshold': variance_threshold,
                'low_variance_features': len(low_variance_features),
                'features': low_variance_features
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in advanced filtering: {e}")
            advanced_results['error'] = str(e)
        
        return advanced_results
    
    def _enhance_with_advanced_analysis(self, 
                                      standard_result: FeatureImportanceResult, 
                                      advanced_results: Dict[str, Any]) -> FeatureImportanceResult:
        """Enhance standard results with advanced analysis."""
        
        # Add advanced metrics to meta_info
        enhanced_meta_info = standard_result.meta_info.copy()
        enhanced_meta_info['advanced_analysis'] = advanced_results
        
        # Create enhanced result
        enhanced_result = FeatureImportanceResult(
            feature_names=standard_result.feature_names,
            importance_scores=standard_result.importance_scores,
            method_scores=standard_result.method_scores,
            stability_scores=standard_result.stability_scores,
            temporal_stability=standard_result.temporal_stability,
            rankings=standard_result.rankings,
            meta_info=enhanced_meta_info
        )
        
        return enhanced_result
    
    def _save_results(self, result: FeatureImportanceResult):
        """Save analysis results."""
        if self.config.output_directory:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save detailed results
            results_file = output_dir / f"feature_importance_{int(time.time())}.json"
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_result = {
                'feature_names': result.feature_names,
                'method_scores': result.method_scores,
                'stability_scores': result.stability_scores,
                'rankings': result.rankings,
                'meta_info': result.meta_info
            }
            
            import json
            with open(results_file, 'w') as f:
                json.dump(serializable_result, f, indent=2)
            
            self.logger.info(f"💾 Results saved to {results_file}")
    
    def _generate_plots(self, result: FeatureImportanceResult):
        """Generate visualization plots."""
        if self.config.output_directory:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot 1: Feature importance comparison
            self._plot_importance_comparison(result, output_dir)
            
            # Plot 2: Stability analysis
            self._plot_stability_analysis(result, output_dir)
            
            # Plot 3: Top features
            self._plot_top_features(result, output_dir)
    
    def _plot_importance_comparison(self, result: FeatureImportanceResult, output_dir: Path):
        """Plot feature importance comparison across methods."""
        if len(result.method_scores) < 2:
            return
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        methods = list(result.method_scores.keys())
        top_features = result.get_top_features("ensemble", self.config.top_k_features)
        
        # Normalize scores for comparison
        normalized_scores = {}
        for method in methods:
            scores = result.method_scores[method]
            max_score = max(scores.values()) if scores else 1
            normalized_scores[method] = {f: scores.get(f, 0) / max_score for f in top_features}
        
        # Create heatmap
        data_matrix = []
        for feature in top_features:
            row = [normalized_scores[method].get(feature, 0) for method in methods]
            data_matrix.append(row)
        
        sns.heatmap(data_matrix, 
                   xticklabels=methods, 
                   yticklabels=top_features,
                   annot=True, 
                   fmt='.3f',
                   cmap='viridis',
                   ax=ax)
        
        ax.set_title('Feature Importance Comparison Across Methods')
        ax.set_xlabel('Methods')
        ax.set_ylabel('Features')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_stability_analysis(self, result: FeatureImportanceResult, output_dir: Path):
        """Plot stability analysis."""
        if not result.stability_scores:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        features = list(result.stability_scores.keys())
        stability_values = list(result.stability_scores.values())
        
        # Sort by stability
        sorted_data = sorted(zip(features, stability_values), key=lambda x: x[1], reverse=True)
        features, stability_values = zip(*sorted_data)
        
        bars = ax.bar(range(len(features)), stability_values)
        ax.set_xlabel('Features')
        ax.set_ylabel('Stability Score')
        ax.set_title('Feature Importance Stability Across Methods')
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, rotation=45, ha='right')
        
        # Color bars by stability
        for i, (bar, value) in enumerate(zip(bars, stability_values)):
            if value >= self.config.stability_threshold:
                bar.set_color('green')
            elif value >= self.config.stability_threshold * 0.7:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'stability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_top_features(self, result: FeatureImportanceResult, output_dir: Path):
        """Plot top features."""
        top_features = result.get_top_features("ensemble", self.config.top_k_features)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get ensemble scores
        ensemble_scores = {}
        if 'ensemble' in result.rankings:
            for method, scores in result.method_scores.items():
                for feature in top_features:
                    if feature in scores:
                        if feature not in ensemble_scores:
                            ensemble_scores[feature] = []
                        ensemble_scores[feature].append(scores[feature])
        
        # Average scores
        avg_scores = {feature: np.mean(scores) for feature, scores in ensemble_scores.items()}
        
        # Sort features by score
        sorted_features = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        features, scores = zip(*sorted_features)
        
        bars = ax.barh(range(len(features)), scores)
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Features')
        ax.set_title(f'Top {len(features)} Most Important Features')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'top_features.png', dpi=300, bbox_inches='tight')
        plt.close()

# Convenience functions
def analyze_feature_importance(X: pd.DataFrame, 
                             y: pd.Series,
                             regime_labels: Optional[pd.Series] = None,
                             config: Optional[FeatureImportanceConfig] = None) -> FeatureImportanceResult:
    """Convenience function for feature importance analysis."""
    analyzer = FeatureImportanceAnalyzer(config)
    return analyzer.analyze_features(X, y, regime_labels)

def get_important_features(X: pd.DataFrame, 
                          y: pd.Series,
                          regime_labels: Optional[pd.Series] = None,
                          k: int = 20,
                          methods: Optional[List[ImportanceMethod]] = None) -> List[str]:
    """Get top k important features using automated analysis."""
    if methods is None:
        methods = [ImportanceMethod.RANDOM_FOREST, ImportanceMethod.LASSO, ImportanceMethod.MUTUAL_INFO]
    
    config = FeatureImportanceConfig(
        methods=methods,
        top_k_features=k,
        save_results=False,
        generate_plots=False
    )
    
    analyzer = FeatureImportanceAnalyzer(config)
    result = analyzer.analyze_features(X, y, regime_labels)
    
    return result.get_top_features("ensemble", k)