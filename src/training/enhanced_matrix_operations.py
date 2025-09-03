# src/training/enhanced_matrix_operations.py

from src.core.decorators import (
from copy import copy

    cached,
    circuit_breaker,
    handles_errors,
    log_call,
    log_execution_time,
    validates
)

from src.core.domain import (
    prevent_data_leakage,
    quality_gate,
    secure_data_processing
)

"""Enhanced Matrix Operations Manager for advanced ML training processes."
Implements sophisticated matrix operations with security decorators and
performance optimizations for improved model training.
"""
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

try:
    import lightgbm as lgb
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False
    lgb = None
import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.decomposition import FactorAnalysis, FastICA, KernelPCA
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.utils.logger import system_logger

try:

    _DASK_AVAILABLE = True
except Exception:
    _DASK_AVAILABLE = False

# Import security and monitoring decorators

@dataclass
class MatrixOperationsConfig:
    """Configuration for enhanced matrix operations."""

    # Operation settings
    enable_gpu_acceleration: bool = False
    enable_sparse_optimizations: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True

    # Quality thresholds
    condition_number_threshold: float = 1e12
    min_eigenvalue_threshold: float = 1e-10
    correlation_threshold: float = 0.8
    memory_threshold_gb: float = 8.0

    # Performance settings
    batch_size: int = 1000
    max_iterations: int = 1000
    tolerance: float = 1e-6

    # Security settings
    enable_data_validation: bool = True
    enable_numerical_stability_checks: bool = True
    enable_quality_gates: bool = True

    # Feature selection settings
    target_features: int = 100
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    mutual_info_threshold: float = 0.01

class EnhancedMatrixOperations:
    """Enhanced matrix operations manager with security decorators and optimizations."

    Implements:
    - Advanced linear algebra optimizations
    - Sparse matrix operations
    - GPU acceleration
    - Memory-efficient operations
    - Tensor operations
    - Matrix completion
    - Advanced clustering
    - Optimization algorithms
    - Real-time updates
    - Quality assurance
    - Feature selection and reduction
    """
    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize enhanced matrix operations manager."""
        self.config = MatrixOperationsConfig(**config.get("matrix_operations", {}))
        self.logger = system_logger.getChild("EnhancedMatrixOperations")
        self.operation_results = {}
        
        # Feature selection configuration
        self.target_features = config.get("feature_reduction", {}).get("step2_target_features", 100)
        self.variance_threshold = config.get("feature_reduction", {}).get("variance_threshold", 0.01)
        self.correlation_threshold = config.get("feature_reduction", {}).get("correlation_threshold", 0.95)
        self.mutual_info_threshold = config.get("feature_reduction", {}).get("mutual_info_threshold", 0.01)

        # Feature importance cache
        self.feature_importance_cache = {}
        self.selection_metadata = {}

    @secure_data_processing(encryption_level="high", data_validation=True)
    @prevent_data_leakage(validate_inputs=True, sanitize_outputs=True)
    @log_execution_time(cpu_threshold_percent=90.0, memory_threshold_gb=16.0)
    @cached(chunk_size=5000, streaming_processing=True)
    @log_call(log_intermediate_results=True, save_debug_artifacts=True)
    @circuit_breaker(failure_threshold=3, recovery_timeout=300.0)
    @validates(required_files=[], data_quality_checks={"min_rows": 100})
    @quality_gate(
        model_performance_thresholds={},
        data_quality_metrics={"completeness": 0.9},
    )
    @handles_errors(exceptions=(ValueError, np.linalg.LinAlgError), default_return=None)
    def eigenvalue_based_feature_engineering(
        self,
        features_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Extract market regime features using eigenvalue decomposition."

        Args:
            features_df: Input features DataFrame

        Returns:
            Enhanced features DataFrame and metadata

        """
        try:
            start_time = time.time()
            self.logger.info("🔄 Applying eigenvalue-based feature engineering...")

            # Validate input
            if features_df.empty or features_df.isna().all().all():
                msg = "Input features are empty or all NaN"
                raise ValueError(msg)

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features_df)

            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(X_scaled.T)

            # Add regularization for numerical stability
            correlation_matrix += 1e-6 * np.eye(correlation_matrix.shape[0])

            # Eigenvalue decomposition
            eigenvalues, eigenvectors = la.eigh(correlation_matrix)

            # Sort by eigenvalue magnitude
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Select top components (explaining 95% variance)
            explained_variance = eigenvalues / eigenvalues.sum()
            cumulative_variance = np.cumsum(explained_variance)
            n_components = np.argmax(cumulative_variance >= 0.95) + 1
            n_components = min(n_components, 10)  # Limit to top 10

            # Create regime features
            regime_features = X_scaled @ eigenvectors[:, :n_components]
            regime_feature_names = [
                f"regime_component_{i+1}" for i in range(n_components)
            ]

            # Create DataFrame
            regime_df = pd.DataFrame(
                regime_features, columns=regime_feature_names, index=features_df.index,
            )

            # Combine with original features
            enhanced_df = pd.concat([features_df, regime_df], axis=1)

            # Metadata
            metadata = {
                "n_regime_components": n_components,
                "explained_variance": explained_variance[:n_components].tolist(),
                "cumulative_variance": cumulative_variance[n_components - 1],
                "eigenvalues": eigenvalues[:n_components].tolist(),
                "condition_number": np.linalg.cond(correlation_matrix),
                "processing_time": time.time() - start_time,
            }

            self.logger.info(
                f"✅ Eigenvalue-based features: {n_components} components = {metadata['cumulative_variance']:.3f} variance explained",
            )
            return enhanced_df, metadata

        except Exception as e:
            self.logger.exception(
                f"❌ Eigenvalue-based feature engineering failed: {e}",
            )
            return features_df, {"error": str(e)}

    @secure_data_processing(encryption_level="medium", data_validation=True)
    @cached(chunk_size=2000, streaming_processing=False)
    @log_call(log_intermediate_results=True)
    @quality_gate(data_quality_metrics={"completeness": 0.95})
    @handles_errors(exceptions=(ValueError, np.linalg.LinAlgError), default_return=None)
    def cholesky_covariance_estimation(
        self,
        features_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Stable covariance estimation using Cholesky decomposition."

        Args:
            features_df: Input features DataFrame

        Returns:
            Enhanced features DataFrame and metadata

        """
        try:
            start_time = time.time()
            self.logger.info("🔄 Applying Cholesky covariance estimation...")

            # Calculate sample covariance
            X = features_df.values
            n_samples, n_features = X.shape[0]

            # Center the data
            X_centered = X - np.mean(X, axis=0)

            # Calculate covariance matrix
            cov_matrix = X_centered.T @ X_centered / (n_samples - 1)

            # Add regularization for positive definiteness
            regularization = 1e-6 * np.eye(n_features)
            cov_matrix += regularization

            # Cholesky decomposition
            try:
                L = la.cholesky(cov_matrix, lower=True)
                cholesky_success = True
            except la.LinAlgError:
                # Fallback: use eigendecomposition
                eigenvals, eigenvecs = la.eigh(cov_matrix)
                eigenvals = np.maximum(eigenvals, 1e-6)  # Ensure positive
                L = eigenvecs @ np.diag(np.sqrt(eigenvals))
                cholesky_success = False

            # Create Cholesky-based features
            cholesky_features = X_centered @ L.T
            cholesky_feature_names = [
                f"cholesky_feature_{i+1}" for i in range(n_features)
            ]

            # Create DataFrame
            cholesky_df = pd.DataFrame(
                cholesky_features,
                columns=cholesky_feature_names,
                index=features_df.index,
            )

            # Combine with original features
            enhanced_df = pd.concat([features_df, cholesky_df], axis=1)

            # Metadata
            metadata = {
                "cholesky_success": cholesky_success,
                "condition_number": np.linalg.cond(cov_matrix),
                "min_eigenvalue": np.min(la.eigvals(cov_matrix)),
                "processing_time": time.time() - start_time,
            }

            self.logger.info(
                f"✅ Cholesky covariance estimation completed: success={cholesky_success}",
            )
            return enhanced_df, metadata

        except Exception as e:
            self.logger.exception(f"❌ Cholesky covariance estimation failed: {e}")
            return features_df, {"error": str(e)}

    @secure_data_processing(encryption_level="medium", data_validation=True)
    @cached(chunk_size=3000, streaming_processing=True)
    @log_call(log_intermediate_results=True)
    @quality_gate(data_quality_metrics={"completeness": 0.9})
    @handles_errors(exceptions=(ValueError, np.linalg.LinAlgError), default_return=None)
    def sparse_matrix_optimizations(
        self,
        features_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply sparse matrix optimizations for large-scale data."

        Args:
            features_df: Input features DataFrame

        Returns:
            Optimized features DataFrame and metadata

        """
        try:
            start_time = time.time()
            self.logger.info("🔄 Applying sparse matrix optimizations...")

            X = features_df.values

            # Calculate sparsity
            sparsity = 1.0 - np.count_nonzero(X) / X.size

            if sparsity > 0.3:  # If more than 30% zeros
                # Convert to sparse matrix
                X_sparse = sp.csr_matrix(X)

                # Apply sparse SVD
                U, s, Vt = sp.linalg.svds(X_sparse, k=min(50, *X.shape))

                # Create sparse features
                sparse_features = U * s
                sparse_feature_names = [
                    f"sparse_component_{i+1}" for i in range(sparse_features.shape[1])
                ]

                # Create DataFrame
                sparse_df = pd.DataFrame(
                    sparse_features,
                    columns=sparse_feature_names,
                    index=features_df.index,
                )

                enhanced_df = pd.concat([features_df, sparse_df], axis=1)

                metadata = {
                    "sparsity": sparsity,
                    "sparse_n_components": sparse_features.shape[1],
                    "memory_savings": f"{(1 - sparsity) * 100:.1f}%",
                    "processing_time": time.time() - start_time,
                }
            else:
                enhanced_df = features_df
                metadata = {
                    "sparsity": sparsity,
                    "sparse_optimization": "not_applied",
                    "reason": "low_sparsity",
                }

            self.logger.info(
                f"✅ Sparse matrix optimization: sparsity={metadata['sparsity']:.3f}",
            )
            return enhanced_df, metadata

        except Exception as e:
            self.logger.exception(f"❌ Sparse matrix optimization failed: {e}")
            return features_df, {"error": str(e)}

    @secure_data_processing(encryption_level="high", data_validation=True)
    @log_execution_time(cpu_threshold_percent=85.0, memory_threshold_gb=12.0)
    @cached(chunk_size=1000, streaming_processing=True)
    @log_call(log_intermediate_results=True)
    @quality_gate(data_quality_metrics={"completeness": 0.95})
    @handles_errors(exceptions=(ValueError, np.linalg.LinAlgError), default_return=None)
    def advanced_decomposition_techniques(
        self,
        features_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply advanced decomposition techniques (ICA = Factor Analysis, Kernel PCA)."

        Args:
            features_df: Input features DataFrame

        Returns:
            Enhanced features DataFrame and metadata

        """
        try:
            start_time = time.time()
            self.logger.info("🔄 Applying advanced decomposition techniques...")

            enhanced_df = features_df.copy()
            metadata = {}

            # 1. Independent Component Analysis (ICA)
            try:
                ica = FastICA(
                    n_components=min(20, features_df.shape[1]),
                    random_state=42,
                    max_iter=200,
                )
                ica_features = ica.fit_transform(features_df)
                ica_feature_names = [
                    f"ica_component_{i+1}" for i in range(ica_features.shape[1])
                ]
                ica_df = pd.DataFrame(
                    ica_features, columns=ica_feature_names, index=features_df.index,
                )
                enhanced_df = pd.concat([enhanced_df, ica_df], axis=1)
                metadata["ica"] = {
                    "n_components": ica_features.shape[1],
                    "convergence": ica.n_iter_,
                }
            except Exception as e:
                self.logger.warning(f"ICA failed: {e}")
                metadata["ica"] = {"error": str(e)}

            # 2. Factor Analysis
            try:
                fa = FactorAnalysis(
                    n_components=min(15, features_df.shape[1]),
                    random_state=42,
                    max_iter=200,
                )
                fa_features = fa.fit_transform(features_df)
                fa_feature_names = [
                    f"factor_component_{i+1}" for i in range(fa_features.shape[1])
                ]
                fa_df = pd.DataFrame(
                    fa_features, columns=fa_feature_names, index=features_df.index,
                )
                enhanced_df = pd.concat([enhanced_df, fa_df], axis=1)
                metadata["factor_analysis"] = {"n_components": fa_features.shape[1]}
            except Exception as e:
                self.logger.warning(f"Factor Analysis failed: {e}")
                metadata["factor_analysis"] = {"error": str(e)}

            # 3. Kernel PCA (for non-linear patterns)
            try:
                kpca = KernelPCA(
                    n_components=min(10, features_df.shape[1]),
                    kernel="rbf",
                    random_state=42,
                )
                kpca_features = kpca.fit_transform(features_df)
                kpca_feature_names = [
                    f"kpca_component_{i+1}" for i in range(kpca_features.shape[1])
                ]
                kpca_df = pd.DataFrame(
                    kpca_features, columns=kpca_feature_names, index=features_df.index,
                )
                enhanced_df = pd.concat([enhanced_df, kpca_df], axis=1)
                metadata["kernel_pca"] = {"n_components": kpca_features.shape[1]}
            except Exception as e:
                self.logger.warning(f"Kernel PCA failed: {e}")
                metadata["kernel_pca"] = {"error": str(e)}

            metadata["processing_time"] = time.time() - start_time
            metadata["total_enhancement"] = len(enhanced_df.columns) - len(
                features_df.columns,
            )

            self.logger.info(
                f"✅ Advanced decomposition: +{metadata['total_enhancement']} features",
            )
            return enhanced_df, metadata

        except Exception as e:
            self.logger.exception(f"❌ Advanced decomposition failed: {e}")
            return features_df, {"error": str(e)}

    @secure_data_processing(encryption_level="medium", data_validation=True)
    @cached(chunk_size=2000, streaming_processing=True)
    @log_call(log_intermediate_results=True)
    @quality_gate(data_quality_metrics={"completeness": 0.9})
    @handles_errors(exceptions=(ValueError, np.linalg.LinAlgError), default_return=None)
    def matrix_completion_techniques(
        self,
        features_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply matrix completion techniques for missing data."

        Args:
            features_df: Input features DataFrame

        Returns:
            Completed features DataFrame and metadata

        """
        try:
            start_time = time.time()
            self.logger.info("🔄 Applying matrix completion techniques...")

            # Check for missing values
            missing_count = features_df.isna().sum().sum()
            missing_percentage = missing_count / (
                features_df.shape[0] * features_df.shape[1]
            )

            if missing_percentage > 0.01:  # More than 1% missing
                # Use Iterative Imputer (MICE)
                imputer = IterativeImputer(
                    max_iter=10,
                    random_state=42,
                    skip_complete=True,
                )
                completed_features = imputer.fit_transform(features_df)

                # Create completed DataFrame
                completed_df = pd.DataFrame(
                    completed_features,
                    columns=features_df.columns,
                    index=features_df.index,
                )

                metadata = {
                    "missing_count": missing_count,
                    "missing_percentage": missing_percentage,
                    "imputation_method": "iterative_imputer",
                    "processing_time": time.time() - start_time,
                }
            else:
                completed_df = features_df
                metadata = {
                    "missing_count": missing_count,
                    "missing_percentage": missing_percentage,
                    "imputation_method": "none_needed",
                    "processing_time": time.time() - start_time,
                }

            self.logger.info(
                f"✅ Matrix completion: {missing_percentage:.3f} missing values handled",
            )
            return completed_df, metadata

        except Exception as e:
            self.logger.exception(f"❌ Matrix completion failed: {e}")
            return features_df, {"error": str(e)}

    @secure_data_processing(encryption_level="medium", data_validation=True)
    @cached(chunk_size=3000, streaming_processing=True)
    @log_call(log_intermediate_results=True)
    @quality_gate(data_quality_metrics={"completeness": 0.9})
    @handles_errors(exceptions=(ValueError, np.linalg.LinAlgError), default_return=None)
    def advanced_clustering_features(
        self,
        features_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply advanced clustering techniques for feature creation."

        Args:
            features_df: Input features DataFrame

        Returns:
            Enhanced features DataFrame and metadata

        """
        try:
            start_time = time.time()
            self.logger.info("🔄 Applying advanced clustering features...")

            enhanced_df = features_df.copy()
            metadata = {}

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features_df)

            # 1. Spectral Clustering
            try:
                spectral = SpectralClustering(
                    n_clusters=min(8, features_df.shape[1] // 2),
                    affinity="rbf",
                    random_state=42,
                )
                spectral_labels = spectral.fit_predict(X_scaled)

                # Create cluster features
                spectral_features = pd.get_dummies(
                    spectral_labels, prefix="spectral_cluster",
                )
                spectral_features.index = features_df.index
                enhanced_df = pd.concat([enhanced_df, spectral_features], axis=1)

                # Distance to cluster centroids

                kmeans = KMeans(
                    n_clusters=min(8, features_df.shape[1] // 2),
                    random_state=42,
                )
                kmeans.fit(X_scaled)
                distances = euclidean_distances(X_scaled, kmeans.cluster_centers_)

                distance_df = pd.DataFrame(
                    distances,
                    columns=[
                        f"distance_to_cluster_{i+1}" for i in range(distances.shape[1])
                    ],
                    index=features_df.index,
                )
                enhanced_df = pd.concat([enhanced_df, distance_df], axis=1)

                metadata["spectral_clustering"] = {
                    "n_clusters": len(np.unique(spectral_labels)),
                    "cluster_sizes": [
                        np.sum(spectral_labels == i)
                        for i in range(len(np.unique(spectral_labels)))
                    ],
                }
            except Exception as e:
                self.logger.warning(f"Spectral clustering failed: {e}")
                metadata["spectral_clustering"] = {"error": str(e)}

            # 2. DBSCAN for outlier detection
            try:
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                dbscan_labels = dbscan.fit_predict(X_scaled)

                # Create outlier features
                outlier_features = pd.DataFrame(
                    {
                        "is_outlier": (dbscan_labels == -1).astype(int),
                        "cluster_id": dbscan_labels,
                    },
                    index=features_df.index,
                )
                enhanced_df = pd.concat([enhanced_df, outlier_features], axis=1)

                metadata["dbscan"] = {
                    "n_clusters": len(np.unique(dbscan_labels[dbscan_labels != -1])),
                    "outlier_count": np.sum(dbscan_labels == -1),
                }
            except Exception as e:
                self.logger.warning(f"DBSCAN failed: {e}")
                metadata["dbscan"] = {"error": str(e)}

            metadata["processing_time"] = time.time() - start_time
            metadata["total_enhancement"] = len(enhanced_df.columns) - len(
                features_df.columns,
            )

            self.logger.info(
                f"✅ Advanced clustering: +{metadata['total_enhancement']} features",
            )
            return enhanced_df, metadata

        except Exception as e:
            self.logger.exception(f"❌ Advanced clustering failed: {e}")
            return features_df, {"error": str(e)}

    @secure_data_processing(encryption_level="high", data_validation=True)
    @log_execution_time(cpu_threshold_percent=80.0, memory_threshold_gb=10.0)
    @cached(chunk_size=2000, streaming_processing=True)
    @log_call(log_intermediate_results=True)
    @quality_gate(data_quality_metrics={"completeness": 0.95})
    @handles_errors(exceptions=(ValueError, np.linalg.LinAlgError), default_return=None)
    def optimization_algorithms(
        self,
        features_df: pd.DataFrame,
        target: pd.Series = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply optimization algorithms for feature selection and regularization."

        Args:
            features_df: Input features DataFrame
            target: Target variable

        Returns:
            Optimized features DataFrame and metadata

        """
        try:
            start_time = time.time()
            self.logger.info("🔄 Applying optimization algorithms...")

            enhanced_df = features_df.copy()
            metadata = {}

            # 1. Lasso for sparse feature selection
            try:
                lasso = Lasso(alpha=0.01, max_iter=1000, random_state=42)
                lasso.fit(features_df, target)

                # Select features with non-zero coefficients
                selected_features = features_df.columns[lasso.coef_ != 0]
                if len(selected_features) > 0:
                    lasso_df = features_df[selected_features].copy()
                    lasso_df.columns = [f"lasso_{col}" for col in selected_features]
                    enhanced_df = pd.concat([enhanced_df, lasso_df], axis=1)

                metadata["lasso"] = {
                    "selected_features": len(selected_features),
                    "sparsity": 1.0 - len(selected_features) / len(features_df.columns),
                }
            except Exception as e:
                self.logger.warning(f"Lasso failed: {e}")
                metadata["lasso"] = {"error": str(e)}

            # 2. Ridge regression for regularization
            try:
                ridge = Ridge(alpha=1.0, random_state=42)
                ridge.fit(features_df, target)

                # Create ridge features
                ridge_features = ridge.predict(features_df).reshape(-1, 1)
                ridge_df = pd.DataFrame(
                    ridge_features,
                    columns=["ridge_prediction"],
                    index=features_df.index,
                )
                enhanced_df = pd.concat([enhanced_df, ridge_df], axis=1)

                metadata["ridge"] = {
                    "r2_score": ridge.score(features_df, target),
                    "regularization_strength": ridge.alpha,
                }
            except Exception as e:
                self.logger.warning(f"Ridge failed: {e}")
                metadata["ridge"] = {"error": str(e)}

            metadata["processing_time"] = time.time() - start_time
            metadata["total_enhancement"] = len(enhanced_df.columns) - len(
                features_df.columns,
            )

            self.logger.info(
                f"✅ Optimization algorithms: +{metadata['total_enhancement']} features",
            )
            return enhanced_df, metadata

        except Exception as e:
            self.logger.exception(f"❌ Optimization algorithms failed: {e}")
            return features_df, {"error": str(e)}

    @secure_data_processing(encryption_level="medium", data_validation=True)
    @cached(chunk_size=1000, streaming_processing=True)
    @log_call(log_intermediate_results=True)
    @quality_gate(data_quality_metrics={"completeness": 0.9})
    @handles_errors(exceptions=(ValueError, np.linalg.LinAlgError), default_return=None)
    def advanced_feature_engineering(
        self,
        features_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply advanced feature engineering techniques."

        Args:
            features_df: Input features DataFrame

        Returns:
            Enhanced features DataFrame and metadata

        """
        try:
            start_time = time.time()
            self.logger.info("🔄 Applying advanced feature engineering...")

            enhanced_df = features_df.copy()
            metadata = {}

            # 1. Polynomial feature interactions
            try:
                poly = PolynomialFeatures(
                    degree=2,
                    interaction_only=True,
                    include_bias=False,
                )
                poly_features = poly.fit_transform(features_df)
                poly_feature_names = [
                    f"poly_interaction_{i+1}"
                    for i in range(poly_features.shape[1] - features_df.shape[1])
                ]

                # Select only interaction features (exclude original features)
                interaction_features = poly_features[:, features_df.shape[1] :]

                # Limit to top interactions to prevent explosion
                if interaction_features.shape[1] > 50:
                    # Select features with highest variance
                    variances = np.var(interaction_features, axis=0)
                    top_indices = np.argsort(variances)[-50:]
                    interaction_features = interaction_features[:, top_indices]
                    poly_feature_names = [poly_feature_names[i] for i in top_indices]

                poly_df = pd.DataFrame(
                    interaction_features,
                    columns=poly_feature_names,
                    index=features_df.index,
                )
                enhanced_df = pd.concat([enhanced_df, poly_df], axis=1)

                metadata["polynomial_features"] = {
                    "n_interactions": len(poly_feature_names),
                    "degree": 2,
                }
            except Exception as e:
                self.logger.warning(f"Polynomial features failed: {e}")
                metadata["polynomial_features"] = {"error": str(e)}

            # 2. Fourier transform features (for time series)
            try:
                # Apply FFT to each feature
                fft_features = []
                fft_feature_names = []

                for _i, col in enumerate(features_df.columns):
                    fft_vals = np.fft.fft(features_df[col].values)
                    # Take magnitude of first few components
                    n_components = min(5, len(fft_vals) // 2)
                    fft_magnitude = np.abs(fft_vals[:n_components])

                    fft_features.append(fft_magnitude)
                    fft_feature_names.extend(
                        [f"fft_{col}_comp_{j+1}" for j in range(n_components)],
                    )

                # Pad to same length
                max_len = max(len(f) for f in fft_features)
                padded_features = []
                for f in fft_features:
                    padded = np.pad(f, (0, max_len - len(f)), mode="constant")
                    padded_features.append(padded)

                fft_array = np.column_stack(padded_features)
                fft_df = pd.DataFrame(
                    fft_array,
                    columns=fft_feature_names[: fft_array.shape[1]],
                    index=features_df.index,
                )
                enhanced_df = pd.concat([enhanced_df, fft_df], axis=1)

                metadata["fourier_features"] = {
                    "n_fft_features": fft_array.shape[1],
                    "n_components_per_feature": 5,
                }
            except Exception as e:
                self.logger.warning(f"Fourier features failed: {e}")
                metadata["fourier_features"] = {"error": str(e)}

            metadata["processing_time"] = time.time() - start_time
            metadata["total_enhancement"] = len(enhanced_df.columns) - len(
                features_df.columns,
            )

            self.logger.info(
                f"✅ Advanced feature engineering: +{metadata['total_enhancement']} features",
            )
            return enhanced_df, metadata

        except Exception as e:
            self.logger.exception(f"❌ Advanced feature engineering failed: {e}")
            return features_df, {"error": str(e)}

    @secure_data_processing(encryption_level="high", data_validation=True)
    @cached(chunk_size=2000, streaming_processing=True)
    @log_call(log_intermediate_results=True)
    @quality_gate(data_quality_metrics={"completeness": 0.95})
    @handles_errors(exceptions=(ValueError, np.linalg.LinAlgError), default_return=None)
    def quality_assurance_checks(self, features_df: pd.DataFrame) -> dict[str, Any]:
        """Perform comprehensive quality assurance checks."

        Args:
            features_df: Input features DataFrame

        Returns:
            Quality assessment results

        """
        try:
            start_time = time.time()
            self.logger.info("🔍 Performing quality assurance checks...")

            quality_results = {"passed": True, "checks": {}, "recommendations": []}

            # 1. Numerical stability checks
            try:
                X = features_df.values
                condition_number = np.linalg.cond(X)
                eigenvals = la.eigvals(X.T @ X)
                min_eigenval = np.min(np.abs(eigenvals))

                quality_results["checks"]["numerical_stability"] = {
                    "condition_number": condition_number,
                    "min_eigenvalue": min_eigenval,
                    "passed": condition_number < self.config.condition_number_threshold
                    and min_eigenval > self.config.min_eigenvalue_threshold,
                }

                if condition_number > self.config.condition_number_threshold:
                    quality_results["recommendations"].append(
                        "High condition number detected - consider regularization",
                    )
                if min_eigenval < self.config.min_eigenvalue_threshold:
                    quality_results["recommendations"].append(
                        "Low minimum eigenvalue - consider feature selection",
                    )

            except Exception as e:
                quality_results["checks"]["numerical_stability"] = {
                    "error": str(e),
                    "passed": False,
                }

            # 2. Data quality checks
            try:
                nan_count = features_df.isna().sum().sum()
                nan_percentage = nan_count / (
                    features_df.shape[0] * features_df.shape[1]
                )

                inf_count = (
                    np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()
                )

                zero_var_features = features_df.var() == 0
                zero_var_count = zero_var_features.sum()

                quality_results["checks"]["data_quality"] = {
                    "nan_count": nan_count,
                    "nan_percentage": nan_percentage,
                    "inf_count": inf_count,
                    "zero_variance_features": zero_var_count,
                    "passed": nan_percentage < 0.1
                    and inf_count == 0
                    and zero_var_count < len(features_df.columns) * 0.1,
                }

                if nan_percentage > 0.1:
                    quality_results["recommendations"].append(
                        "High NaN percentage - consider imputation",
                    )
                if inf_count > 0:
                    quality_results["recommendations"].append(
                        "Infinite values detected - check data preprocessing",
                    )
                if zero_var_count > len(features_df.columns) * 0.1:
                    quality_results["recommendations"].append(
                        "Many zero-variance features - consider feature selection",
                    )

            except Exception as e:
                quality_results["checks"]["data_quality"] = {
                    "error": str(e),
                    "passed": False,
                }

            # 3. Correlation analysis
            try:
                corr_matrix = features_df.corr().abs()
                high_corr_pairs = np.where(
                    corr_matrix > self.config.correlation_threshold,
                )
                high_corr_count = len(high_corr_pairs[0]) - len(
                    features_df.columns,
                )  # Exclude diagonal

                quality_results["checks"]["correlation_analysis"] = {
                    "high_correlation_pairs": high_corr_count,
                    "max_correlation": corr_matrix.values[
                        np.triu_indices_from(corr_matrix.values, k=1)
                    ].max(),
                    "passed": high_corr_count < len(features_df.columns) * 0.1,
                }

                if high_corr_count > len(features_df.columns) * 0.1:
                    quality_results["recommendations"].append(
                        "High correlation detected - consider feature selection",
                    )

            except Exception as e:
                quality_results["checks"]["correlation_analysis"] = {
                    "error": str(e),
                    "passed": False,
                }

            # Overall assessment
            all_checks_passed = all(
                check.get("passed", False)
                for check in quality_results["checks"].values()
            )
            quality_results["passed"] = all_checks_passed
            quality_results["processing_time"] = time.time() - start_time

            self.logger.info(
                f"✅ Quality assurance: {'PASSED' if all_checks_passed else 'FAILED'}",
            )
            return quality_results

        except Exception as e:
            self.logger.exception(f"❌ Quality assurance failed: {e}")
            return {"error": str(e), "passed": False}

    @handles_errors(
        exceptions=(Exception,),
        default_return=(pd.DataFrame(), {}),
        context="feature selection step02",
    )
    def select_features_step2(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        symbol: str,
        exchange: str,
        data_dir: str,
        use_autoencoder_features: bool = True,
        use_regularization: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Multi-stage feature selection to reduce features to target count with autoencoder features and regularization."

        Args:
            features_df: Input features DataFrame
            target: Target variable series
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory for saving metadata
            use_autoencoder_features: Whether to include autoencoder features
            use_regularization: Whether to use regularization-aware selection

        Returns:
            Tuple of (selected_features_df, selection_metadata)

        """
        try:
            self.logger.info(f"🔍 Starting enhanced feature selection: {features_df.shape[1]} -> {self.target_features} features")
            
            # Stage 0: Add autoencoder features if enabled
            if use_autoencoder_features:
                features_df, stage0_metadata = self._stage0_autoencoder_features(features_df, target)
            else:
                stage0_metadata = {"autoencoder_features_added": 0}

            # Stage 1: Data quality filtering
            features_df, stage1_metadata = self._stage1_data_quality_filtering(features_df)

            # Stage 2: Variance-based filtering
            features_df, stage2_metadata = self._stage2_variance_filtering(features_df)

            # Stage 3: Correlation-based filtering
            features_df, stage3_metadata = self._stage3_correlation_filtering(features_df)

            # Stage 4: Mutual information ranking
            features_df, stage4_metadata = self._stage4_mutual_info_ranking(features_df, target)

            # Stage 5: Domain-specific selection
            features_df, stage5_metadata = self._stage5_domain_specific_selection(features_df, target)

            # Stage 6: Regularization-aware selection (if enabled)
            if use_regularization:
                features_df, stage6_metadata = self._stage6_regularization_aware_selection(features_df, target)
            else:
                stage6_metadata = {"regularization_applied": False}

            # Stage 7: Final ranking and selection
            features_df, stage7_metadata = self._stage7_final_selection(features_df, target)

            # Compile metadata
            selection_metadata = {
                "original_features": len(features_df.columns),
                "final_features": len(features_df.columns),
                "target_features": self.target_features,
                "stages": {
                    "stage0_autoencoder": stage0_metadata,
                    "stage1_data_quality": stage1_metadata,
                    "stage2_variance": stage2_metadata,
                    "stage3_correlation": stage3_metadata,
                    "stage4_mutual_info": stage4_metadata,
                    "stage5_domain_specific": stage5_metadata,
                    "stage6_regularization": stage6_metadata,
                    "stage7_final_selection": stage7_metadata,
                },
                "feature_categories": self._categorize_features(features_df.columns),
                "selection_timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "exchange": exchange,
            }

            # Save selection metadata
            self._save_selection_metadata(selection_metadata, symbol, exchange, data_dir)

            self.logger.info(f"✅ Feature selection completed: {len(features_df.columns)} features selected")
            return features_df, selection_metadata

        except Exception as e:
            self.logger.exception(f"❌ Feature selection failed: {e}")
            raise

    def _stage0_autoencoder_features(self, features_df: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Stage 0: Add autoencoder features from the autoencoder feature generator."""
        try:
            self.logger.info("🔧 Stage 0: Adding autoencoder features...")
            
            # Import autoencoder feature generator
            from src.analyst.autoencoder_feature_generator import AutoencoderFeatureGenerator
            
            # Create autoencoder generator
            autoencoder_generator = AutoencoderFeatureGenerator()
            
            # Generate autoencoder features
            autoencoder_features = autoencoder_generator.generate_features(
                features_df=features_df,
                regime_name="default",
                labels=target.values,
                enable_analysis=True
            )
            
            # If autoencoder features were generated, add them
            if not autoencoder_features.empty and len(autoencoder_features.columns) > 0:
                # Add autoencoder features with prefix
                autoencoder_features = autoencoder_features.add_prefix("ae_")
                features_df = pd.concat([features_df, autoencoder_features], axis=1)
                
                self.logger.info(f"✅ Added {len(autoencoder_features.columns)} autoencoder features")
                stage_metadata = {
                    "autoencoder_features_added": len(autoencoder_features.columns),
                    "total_features_after_ae": len(features_df.columns)
                }
            else:
                self.logger.info("📊 No autoencoder features generated, continuing with base features")
                stage_metadata = {"autoencoder_features_added": 0}
                
        except Exception as e:
            self.logger.warning(f"⚠️ Autoencoder feature generation failed: {e}")
            self.logger.info("📊 Continuing without autoencoder features")
            stage_metadata = {"autoencoder_features_added": 0, "error": str(e)}
        
        return features_df, stage_metadata

    def _stage1_data_quality_filtering(self, features_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Stage 1: Remove features with poor data quality."""
        original_count = len(features_df.columns)

        # Remove features with too many NaN values (>10%)
        nan_ratio = features_df.isna().sum() / len(features_df)
        high_nan_features = nan_ratio[nan_ratio > 0.1].index.tolist()
        features_df = features_df.drop(columns=high_nan_features)

        # Remove features with infinite values
        inf_features = []
        for col in features_df.columns:
            if np.isinf(features_df[col]).any():
                inf_features.append(col)
        features_df = features_df.drop(columns=inf_features)

        # Fill remaining NaN values with forward fill then backward fill
        features_df = features_df.fillna(method="ffill").fillna(method="bfill").fillna(0)

        metadata = {
            "removed_high_nan": len(high_nan_features),
            "removed_infinite": len(inf_features),
            "features_after_stage": len(features_df.columns),
        }

        self.logger.info(f"Stage 1: Removed {original_count - len(features_df.columns)} low-quality features")
        return features_df, metadata

    def _stage2_variance_filtering(self, features_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Stage 2: Remove low-variance features."""
        len(features_df.columns)

        # Calculate variance for each feature
        variances = features_df.var()

        # Remove features with variance below threshold
        low_variance_features = variances[variances < self.variance_threshold].index.tolist()
        features_df = features_df.drop(columns=low_variance_features)

        metadata = {
            "removed_low_variance": len(low_variance_features),
            "variance_threshold": self.variance_threshold,
            "features_after_stage": len(features_df.columns),
        }

        self.logger.info(f"Stage 2: Removed {len(low_variance_features)} low-variance features")
        return features_df, metadata

    def _stage3_correlation_filtering(self, features_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Stage 3: Remove highly correlated features."""
        len(features_df.columns)

        # Calculate correlation matrix
        corr_matrix = features_df.corr().abs()

        # Find highly correlated feature pairs
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = []

        for col in upper_tri.columns:
            high_corr_features = upper_tri[col][upper_tri[col] > self.correlation_threshold].index.tolist()
            for feature in high_corr_features:
                high_corr_pairs.append((col, feature))

        # Remove one feature from each highly correlated pair
        features_to_remove = set()
        for feat1, feat2 in high_corr_pairs:
            # Keep the feature with higher variance
            var1 = features_df[feat1].var()
            var2 = features_df[feat2].var()
            if var1 < var2:
                features_to_remove.add(feat1)
            else:
                features_to_remove.add(feat2)

        features_df = features_df.drop(columns=list(features_to_remove))

        metadata = {
            "removed_high_correlation": len(features_to_remove),
            "correlation_threshold": self.correlation_threshold,
            "high_corr_pairs": len(high_corr_pairs),
            "features_after_stage": len(features_df.columns),
        }

        self.logger.info(f"Stage 3: Removed {len(features_to_remove)} highly correlated features")
        return features_df, metadata

    def _stage4_mutual_info_ranking(self, features_df: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Stage 4: Rank features by mutual information."""
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(features_df, target, random_state=42)
        mi_ranking = pd.Series(mi_scores, index=features_df.columns).sort_values(ascending=False)

        # Store ranking for later use
        self.feature_importance_cache["mutual_info"] = mi_ranking

        metadata = {
            "top_10_mi_features": mi_ranking.head(10).index.tolist(),
            "mi_scores_range": (mi_ranking.min(), mi_ranking.max()),
            "features_after_stage": len(features_df.columns),
        }

        self.logger.info("Stage 4: Ranked features by mutual information")
        return features_df, metadata

    def _stage5_domain_specific_selection(self, features_df: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Stage 5: Domain-specific feature selection for financial data."""
        # Define feature categories and their importance weights
        # Note: Removed non-semantic categories (regime, lagged, normalized)
        feature_categories = {
            # Momentum/Trend indicators
            "momentum": [
                "momentum", "mom", "rsi", "macd", "cci", "roc", "willr", "stoch",
                "adx", "dmi", "kama", "tema", "dema", "hma", "wma", "vwma", "zlema",
                "ichimoku", "psar", "trix", "cmo", "tsi", "ppo", "pmo", "uo",
                "linreg", "lin_reg", "sma", "ema", "ma_", "moving_avg", "trend",
            ],
            # Volatility/range measures
            "volatility": [
                "volatility", "atr", "true_range", "truerange", "natr", "parkinson",
                "garman", "gk_vol", "garman_klass", "roll", "rvol", "realized_vol",
                "hv", "hist_vol", "historical_vol", "variance", "std", "bbands",
                "boll", "bollinger", "donch", "donchian", "keltner", "chop",
                "choppiness", "park_vol",
            ],
            # Liquidity/volume features
            "liquidity": [
                "liquidity", "volume", "tick_volume", "obv", "cmf", "mfi", "vwap",
                "pvi", "nvi", "efi", "delta_volume",
            ],
            # Microstructure/order book features
            "microstructure": [
                "microstructure", "order_flow", "orderflow", "ofi", "imbalance",
                "quote_imbalance", "spread", "bid_ask", "depth", "orderbook", "book",
                "microprice", "trade_count", "trade_frequency",
            ],
            # Wavelet/transform domain features
            "wavelet": ["wavelet", "dwt", "cwt", "wt_"],
            # Support/Resistance contextual features (sr_ prefix and related terms)
            "sr_distance": [
                "sr_", "sr_distance", "support", "resistance", "proximity",
                "breakout_probability", "rebounce_probability", "consolidation_probability",
                "sr_confidence", "multi_timeframe_sr_score",
            ],
            # Statistical descriptors
            "statistical": [
                "autocorr", "autocorrelation", "correl", "correlation", "entropy",
                "fractal", "hurst", "hjorth", "hj_", "kurtosis", "kurt", "skew",
                "skewness", "zscore", "z_score",
            ],
            # Candlestick pattern features
            "candlestick": [
                "cdl", "candlestick", "doji", "hammer", "engulf", "harami",
                "marubozu", "piercing", "shooting_star", "hanging_man",
                "three_black_crows", "three_white_soldiers", "morning_star", "evening_star",
                "dark_cloud",
            ],
            # Explicit interaction/composite features
            "interaction": ["_x_", "_div_", "_ratio_", "_over_", "_cross_", "interaction"],
        }

        # Calculate category importance scores
        category_scores = {}
        for category, keywords in feature_categories.items():
            category_features = [col for col in features_df.columns if any(keyword in col.lower() for keyword in keywords)]
            if category_features:
                mi_scores = self.feature_importance_cache["mutual_info"][category_features]
                category_scores[category] = mi_scores.mean()

        # Prioritize features from important categories
        prioritized_features = []
        for category, _score in sorted(category_scores.items(), key=lambda x: x[1], reverse=True):
            category_features = [col for col in features_df.columns if any(keyword in col.lower() for keyword in feature_categories[category])]
            prioritized_features.extend(category_features)

        # Ensure we don't exceed target features'
        if len(prioritized_features) > self.target_features:
            prioritized_features = prioritized_features[:self.target_features]

        features_df = features_df[prioritized_features]

        metadata = {
            "category_scores": category_scores,
            "prioritized_categories": list(category_scores.keys()),
            "features_after_stage": len(features_df.columns),
        }

        self.logger.info("Stage 5: Applied domain-specific selection")
        return features_df, metadata

    def _stage6_regularization_aware_selection(self, features_df: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Stage 6: Regularization-aware feature selection using pipeline regularization."""
        try:
            self.logger.info("🔧 Stage 6: Applying regularization-aware feature selection...")
            
            # Load regularization configuration from pipeline
            from src.training.regularization import RegularizationManager
            reg_manager = RegularizationManager()
            regularization_config = reg_manager.regularization_config
            
            # Apply regularization-aware feature selection
            if regularization_config:
                # Get regularization parameters
                l1_alpha = regularization_config.get('l1_alpha', 0.01)
                l2_alpha = regularization_config.get('l2_alpha', 0.001)
                
                # Calculate feature stability scores
                stability_scores = self._calculate_feature_stability(features_df, target)
                
                # Apply regularization penalty to feature importance
                if "mutual_info" in self.feature_importance_cache:
                    mi_scores = self.feature_importance_cache["mutual_info"]
                    
                    # Apply regularization penalty
                    regularization_penalty = 1.0 / (1.0 + l1_alpha + l2_alpha)
                    adjusted_scores = mi_scores * regularization_penalty
                    
                    # Select top features based on adjusted scores
                    top_features = adjusted_scores.nlargest(self.target_features).index.tolist()
                    features_df = features_df[top_features]
                    
                    stage_metadata = {
                        "regularization_applied": True,
                        "l1_alpha": l1_alpha,
                        "l2_alpha": l2_alpha,
                        "regularization_penalty": regularization_penalty,
                        "features_after_stage": len(features_df.columns)
                    }
                else:
                    stage_metadata = {"regularization_applied": False, "reason": "No mutual info scores available"}
            else:
                stage_metadata = {"regularization_applied": False, "reason": "No regularization config available"}
                
        except Exception as e:
            self.logger.warning(f"⚠️ Regularization-aware selection failed: {e}")
            stage_metadata = {"regularization_applied": False, "error": str(e)}
        
        return features_df, stage_metadata

    def _stage7_final_selection(self, features_df: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Stage 7: Final ranking and selection (renamed from stage6)."""
        # Use existing RFE-LightGBM selection logic
        return self._stage6_final_selection(features_df, target)

    def _stage6_final_selection(self, features_df: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Stage 6: Final feature selection using multiple methods (original method)."""
        if len(features_df.columns) <= self.target_features:
            # Already at or below target, return as is
            return features_df, {"final_selection": "no_change", "features_after_stage": len(features_df.columns)}

        # Use Recursive Feature Elimination with LightGBM if available
        if _LIGHTGBM_AVAILABLE and lgb is not None:
            try:
                estimator = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
                rfe = RFE(estimator=estimator, n_features_to_select=self.target_features, step=1)

                # Fit RFE
                rfe.fit(features_df, target)

                # Get selected features
                selected_features = features_df.columns[rfe.support_].tolist()
                features_df = features_df[selected_features]

                metadata = {
                    "final_selection": "rfe_lightgbm",
                    "rfe_ranking": rfe.ranking_.tolist(),
                    "features_after_stage": len(features_df.columns),
                }

                self.logger.info("Stage 6: Final selection using RFE-LightGBM")
                return features_df, metadata
            except Exception as e:
                self.logger.warning(f"⚠️ LightGBM RFE failed: {e}")
                # Fall back to mutual info selection
                return self._fallback_final_selection(features_df, target)
        else:
            self.logger.info("📊 LightGBM not available, using fallback selection")
            return self._fallback_final_selection(features_df, target)

    def _fallback_final_selection(self, features_df: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fallback final selection using mutual information scores."""
        if "mutual_info" in self.feature_importance_cache:
            mi_scores = self.feature_importance_cache["mutual_info"]
            top_features = mi_scores.nlargest(self.target_features).index.tolist()
            features_df = features_df[top_features]
            
            metadata = {
                "final_selection": "mutual_info_fallback",
                "features_after_stage": len(features_df.columns),
            }
            
            self.logger.info("Stage 6: Final selection using mutual info fallback")
            return features_df, metadata
        else:
            # If no mutual info scores, just take first N features
            if len(features_df.columns) > self.target_features:
                features_df = features_df.iloc[:, :self.target_features]
            
            metadata = {
                "final_selection": "simple_truncation",
                "features_after_stage": len(features_df.columns),
            }
            
            self.logger.info("Stage 6: Final selection using simple truncation")
            return features_df, metadata

    def _categorize_features(self, feature_names: list[str]) -> dict[str, list[str]]:
        """Categorize features by type."""
        categories = {
            "momentum": [],
            "volatility": [],
            "liquidity": [],
            "microstructure": [],
            "wavelet": [],
            "sr_distance": [],
            "statistical": [],
            "candlestick": [],
            "interaction": [],
            "transform": [],
            "other": [],
        }

        for feature in feature_names:
            feature_lower = feature.lower()
            categorized = False

            if any(keyword in feature_lower for keyword in [
                "momentum", "mom", "rsi", "macd", "cci", "roc", "willr", "stoch",
                "adx", "dmi", "kama", "tema", "dema", "hma", "wma", "vwma", "zlema",
                "ichimoku", "psar", "trix", "cmo", "tsi", "ppo", "pmo", "uo",
                "linreg", "lin_reg", "sma", "ema", "ma_", "moving_avg", "trend",
            ]):
                categories["momentum"].append(feature)
                categorized = True
            elif any(keyword in feature_lower for keyword in [
                "volatility", "atr", "true_range", "truerange", "natr", "parkinson",
                "garman", "gk_vol", "garman_klass", "roll", "rvol", "realized_vol",
                "hv", "hist_vol", "historical_vol", "variance", "std", "bbands",
                "boll", "bollinger", "donch", "donchian", "keltner", "chop",
                "choppiness", "park_vol",
            ]):
                categories["volatility"].append(feature)
                categorized = True
            elif any(keyword in feature_lower for keyword in [
                "liquidity", "volume", "tick_volume", "obv", "cmf", "mfi", "vwap",
                "pvi", "nvi", "efi", "delta_volume",
            ]):
                categories["liquidity"].append(feature)
                categorized = True
            elif any(keyword in feature_lower for keyword in [
                "microstructure", "order_flow", "orderflow", "ofi", "imbalance",
                "quote_imbalance", "spread", "bid_ask", "depth", "orderbook", "book",
                "microprice", "trade_count", "trade_frequency",
            ]):
                categories["microstructure"].append(feature)
                categorized = True
            elif any(keyword in feature_lower for keyword in ["wavelet", "dwt", "cwt", "wt_"]):
                categories["wavelet"].append(feature)
                categorized = True
            elif any(keyword in feature_lower for keyword in [
                "sr_", "sr_distance", "support", "resistance", "proximity",
                "breakout_probability", "rebounce_probability", "consolidation_probability",
                "sr_confidence", "multi_timeframe_sr_score",
            ]):
                categories["sr_distance"].append(feature)
                categorized = True
            elif any(keyword in feature_lower for keyword in ["cdl", "candlestick", "doji", "hammer", "engulf", "harami", "marubozu", "piercing", "shooting_star", "hanging_man", "three_black_crows", "three_white_soldiers", "morning_star", "evening_star", "dark_cloud"]):
                categories["candlestick"].append(feature)
                categorized = True
            elif any(keyword in feature_lower for keyword in [
                "autocorr", "autocorrelation", "correl", "correlation", "entropy",
                "fractal", "hurst", "hjorth", "hj_", "kurtosis", "kurt", "skew",
                "skewness", "zscore", "z_score",
            ]):
                categories["statistical"].append(feature)
                categorized = True
            elif any(keyword in feature_lower for keyword in ["_x_", "_div_", "_ratio_", "_over_", "_cross_", "interaction"]):
                categories["interaction"].append(feature)
                categorized = True
            elif any(keyword in feature_lower for keyword in ["fft", "fourier", "dct", "cosine", "sine", "transform_"]):
                categories["transform"].append(feature)
                categorized = True

            if not categorized:
                categories["other"].append(feature)

        return categories

    def _save_selection_metadata(self, metadata: dict[str, Any], symbol: str, exchange: str, data_dir: str) -> None:
        """Save feature selection metadata."""
        try:
            metadata_file = f"{data_dir}/{exchange}_{symbol}_feature_selection_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"💾 Feature selection metadata saved: {metadata_file}")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save feature selection metadata: {e}")

    def _calculate_feature_stability(self, features_df: pd.DataFrame, target: pd.Series) -> dict[str, float]:
        """Calculate feature stability scores using cross-validation."""
        stability_scores = {}
        
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.linear_model import LogisticRegression
        except Exception as e:
            pass  # TODO: Handle exception properly
            
        for feature in features_df.columns:
            try:
                # Use single feature for prediction
                X_single = features_df[[feature]]
                
                # Calculate cross-validation score
                cv_scores = cross_val_score(
                    LogisticRegression(random_state=42),
                    X_single,
                    target,
                    cv=3,
                    scoring='accuracy'
                )
                
                # Stability score is the mean CV score
                stability_scores[feature] = np.mean(cv_scores)
            except Exception:
                stability_scores[feature] = 0.0
        
        return stability_scores

    def comprehensive_matrix_enhancement(
        self,
        features_df: pd.DataFrame,
        target: pd.Series = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply comprehensive matrix enhancement pipeline."

        Args:
            features_df: Input features DataFrame
            target: Target variable (optional)

        Returns:
            Enhanced features DataFrame and comprehensive metadata

        """
        try:
            self.logger.info("🚀 Starting comprehensive matrix enhancement...")
            start_time = time.time()

            enhanced_df = features_df.copy()
            all_metadata = {}

            # 1. Quality assurance
            quality_check = self.quality_assurance_checks(enhanced_df)
            all_metadata["quality_assurance"] = quality_check

            if not quality_check.get("passed", False):
                self.logger.warning("⚠️ Quality checks failed = applying fixes...")
                # Apply basic fixes
                enhanced_df = enhanced_df.fillna(enhanced_df.mean())
                enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan).fillna(
                    enhanced_df.mean(),
                )

            # 2. Matrix completion
            enhanced_df, completion_metadata = self.matrix_completion_techniques(
                enhanced_df,
            )
            all_metadata["matrix_completion"] = completion_metadata

            # 3. Eigenvalue-based features
            enhanced_df, eigen_metadata = self.eigenvalue_based_feature_engineering(
                enhanced_df,
            )
            all_metadata["eigenvalue_features"] = eigen_metadata

            # 4. Cholesky covariance
            enhanced_df, cholesky_metadata = self.cholesky_covariance_estimation(
                enhanced_df,
            )
            all_metadata["cholesky_covariance"] = cholesky_metadata

            # 5. Sparse optimizations
            enhanced_df, sparse_metadata = self.sparse_matrix_optimizations(enhanced_df)
            all_metadata["sparse_optimizations"] = sparse_metadata

            # 6. Advanced decompositions
            enhanced_df, decomp_metadata = self.advanced_decomposition_techniques(
                enhanced_df,
            )
            all_metadata["advanced_decompositions"] = decomp_metadata

            # 7. Advanced clustering
            enhanced_df, cluster_metadata = self.advanced_clustering_features(
                enhanced_df,
            )
            all_metadata["advanced_clustering"] = cluster_metadata

            # 8. Advanced feature engineering
            enhanced_df, feature_metadata = self.advanced_feature_engineering(
                enhanced_df,
            )
            all_metadata["advanced_feature_engineering"] = feature_metadata

            # 9. Optimization algorithms (if target provided)
            if target is not None:
                enhanced_df, opt_metadata = self.optimization_algorithms(
                    enhanced_df,
                    target,
                )
                all_metadata["optimization_algorithms"] = opt_metadata

            # Final quality check
            final_quality = self.quality_assurance_checks(enhanced_df)
            all_metadata["final_quality_assurance"] = final_quality

            total_time = time.time() - start_time
            all_metadata["total_processing_time"] = total_time
            all_metadata["feature_count_increase"] = len(enhanced_df.columns) - len(
                features_df.columns,
            )

            self.logger.info(
                f"✅ Comprehensive matrix enhancement completed in {total_time:.2f}s",
            )
            self.logger.info(
                f"📊 Features: {len(features_df.columns)} -> {len(enhanced_df.columns)} (+{all_metadata['feature_count_increase']})",
            )

            return enhanced_df, all_metadata

        except Exception as e:
            self.logger.exception(f"❌ Comprehensive matrix enhancement failed: {e}")
            return features_df, {"error": str(e)}
