# src/training/enhanced_matrix_operations.py

"""
Enhanced Matrix Operations Manager for advanced ML training processes.
Implements sophisticated matrix operations with security decorators and
performance optimizations for improved model training.
"""

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
from sklearn.decomposition import PCA, NMF, FastICA, FactorAnalysis, KernelPCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.linear_model import Lasso, Ridge
from sklearn.impute import IterativeImputer
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster.hierarchy import linkage
from scipy.sparse.linalg import cg
from scipy.optimize import minimize
import dask.array as da
import warnings
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

# Import security and monitoring decorators
from src.utils.training_pipeline_decorators import (
    validate_step_prerequisites,
    secure_data_processing,
    prevent_data_leakage,
    resource_monitor,
    memory_efficient,
    debug_training_step,
    circuit_breaker_protection,
    validate_step_output,
    quality_gate,
)
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


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


class EnhancedMatrixOperations:
    """
    Enhanced matrix operations manager with security decorators and optimizations.
    
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
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced matrix operations manager."""
        self.config = MatrixOperationsConfig(**config.get("matrix_operations", {}))
        self.logger = system_logger.getChild("EnhancedMatrixOperations")
        self.operation_results = {}
        
    @secure_data_processing(encryption_level="high", data_validation=True)
    @prevent_data_leakage(validate_inputs=True, sanitize_outputs=True)
    @resource_monitor(cpu_threshold_percent=90.0, memory_threshold_gb=16.0)
    @memory_efficient(chunk_size=5000, streaming_processing=True)
    @debug_training_step(log_intermediate_results=True, save_debug_artifacts=True)
    @circuit_breaker_protection(failure_threshold=3, recovery_timeout=300.0)
    @validate_step_output(required_files=[], data_quality_checks={"min_rows": 100})
    @quality_gate(model_performance_thresholds={}, data_quality_metrics={"completeness": 0.9})
    @handle_errors(exceptions=(ValueError, np.linalg.LinAlgError), default_return=None)
    def eigenvalue_based_feature_engineering(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extract market regime features using eigenvalue decomposition.
        
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
                raise ValueError("Input features are empty or all NaN")
            
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
            regime_feature_names = [f"regime_component_{i+1}" for i in range(n_components)]
            
            # Create DataFrame
            regime_df = pd.DataFrame(
                regime_features,
                columns=regime_feature_names,
                index=features_df.index
            )
            
            # Combine with original features
            enhanced_df = pd.concat([features_df, regime_df], axis=1)
            
            # Metadata
            metadata = {
                "n_regime_components": n_components,
                "explained_variance": explained_variance[:n_components].tolist(),
                "cumulative_variance": cumulative_variance[n_components-1],
                "eigenvalues": eigenvalues[:n_components].tolist(),
                "condition_number": np.linalg.cond(correlation_matrix),
                "processing_time": time.time() - start_time
            }
            
            self.logger.info(f"✅ Eigenvalue-based features: {n_components} components, {metadata['cumulative_variance']:.3f} variance explained")
            return enhanced_df, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Eigenvalue-based feature engineering failed: {e}")
            return features_df, {"error": str(e)}
    
    @secure_data_processing(encryption_level="medium", data_validation=True)
    @memory_efficient(chunk_size=2000, streaming_processing=False)
    @debug_training_step(log_intermediate_results=True)
    @quality_gate(data_quality_metrics={"completeness": 0.95})
    @handle_errors(exceptions=(ValueError, np.linalg.LinAlgError), default_return=None)
    def cholesky_covariance_estimation(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Stable covariance estimation using Cholesky decomposition.
        
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
            n_samples, n_features = X.shape
            
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
            cholesky_feature_names = [f"cholesky_feature_{i+1}" for i in range(n_features)]
            
            # Create DataFrame
            cholesky_df = pd.DataFrame(
                cholesky_features,
                columns=cholesky_feature_names,
                index=features_df.index
            )
            
            # Combine with original features
            enhanced_df = pd.concat([features_df, cholesky_df], axis=1)
            
            # Metadata
            metadata = {
                "cholesky_success": cholesky_success,
                "condition_number": np.linalg.cond(cov_matrix),
                "min_eigenvalue": np.min(la.eigvals(cov_matrix)),
                "processing_time": time.time() - start_time
            }
            
            self.logger.info(f"✅ Cholesky covariance estimation completed: success={cholesky_success}")
            return enhanced_df, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Cholesky covariance estimation failed: {e}")
            return features_df, {"error": str(e)}
    
    @secure_data_processing(encryption_level="medium", data_validation=True)
    @memory_efficient(chunk_size=3000, streaming_processing=True)
    @debug_training_step(log_intermediate_results=True)
    @quality_gate(data_quality_metrics={"completeness": 0.9})
    @handle_errors(exceptions=(ValueError, np.linalg.LinAlgError), default_return=None)
    def sparse_matrix_optimizations(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply sparse matrix optimizations for large-scale data.
        
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
                U, s, Vt = sp.linalg.svds(X_sparse, k=min(50, min(X.shape)))
                
                # Create sparse features
                sparse_features = U * s
                sparse_feature_names = [f"sparse_component_{i+1}" for i in range(sparse_features.shape[1])]
                
                # Create DataFrame
                sparse_df = pd.DataFrame(
                    sparse_features,
                    columns=sparse_feature_names,
                    index=features_df.index
                )
                
                enhanced_df = pd.concat([features_df, sparse_df], axis=1)
                
                metadata = {
                    "sparsity": sparsity,
                    "sparse_n_components": sparse_features.shape[1],
                    "memory_savings": f"{(1 - sparsity) * 100:.1f}%",
                    "processing_time": time.time() - start_time
                }
            else:
                enhanced_df = features_df
                metadata = {
                    "sparsity": sparsity,
                    "sparse_optimization": "not_applied",
                    "reason": "low_sparsity"
                }
            
            self.logger.info(f"✅ Sparse matrix optimization: sparsity={metadata['sparsity']:.3f}")
            return enhanced_df, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Sparse matrix optimization failed: {e}")
            return features_df, {"error": str(e)}
    
    @secure_data_processing(encryption_level="high", data_validation=True)
    @resource_monitor(cpu_threshold_percent=85.0, memory_threshold_gb=12.0)
    @memory_efficient(chunk_size=1000, streaming_processing=True)
    @debug_training_step(log_intermediate_results=True)
    @quality_gate(data_quality_metrics={"completeness": 0.95})
    @handle_errors(exceptions=(ValueError, np.linalg.LinAlgError), default_return=None)
    def advanced_decomposition_techniques(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply advanced decomposition techniques (ICA, Factor Analysis, Kernel PCA).
        
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
                ica = FastICA(n_components=min(20, features_df.shape[1]), random_state=42, max_iter=200)
                ica_features = ica.fit_transform(features_df)
                ica_feature_names = [f"ica_component_{i+1}" for i in range(ica_features.shape[1])]
                ica_df = pd.DataFrame(ica_features, columns=ica_feature_names, index=features_df.index)
                enhanced_df = pd.concat([enhanced_df, ica_df], axis=1)
                metadata["ica"] = {"n_components": ica_features.shape[1], "convergence": ica.n_iter_}
            except Exception as e:
                self.logger.warning(f"ICA failed: {e}")
                metadata["ica"] = {"error": str(e)}
            
            # 2. Factor Analysis
            try:
                fa = FactorAnalysis(n_components=min(15, features_df.shape[1]), random_state=42, max_iter=200)
                fa_features = fa.fit_transform(features_df)
                fa_feature_names = [f"factor_component_{i+1}" for i in range(fa_features.shape[1])]
                fa_df = pd.DataFrame(fa_features, columns=fa_feature_names, index=features_df.index)
                enhanced_df = pd.concat([enhanced_df, fa_df], axis=1)
                metadata["factor_analysis"] = {"n_components": fa_features.shape[1]}
            except Exception as e:
                self.logger.warning(f"Factor Analysis failed: {e}")
                metadata["factor_analysis"] = {"error": str(e)}
            
            # 3. Kernel PCA (for non-linear patterns)
            try:
                kpca = KernelPCA(n_components=min(10, features_df.shape[1]), kernel='rbf', random_state=42)
                kpca_features = kpca.fit_transform(features_df)
                kpca_feature_names = [f"kpca_component_{i+1}" for i in range(kpca_features.shape[1])]
                kpca_df = pd.DataFrame(kpca_features, columns=kpca_feature_names, index=features_df.index)
                enhanced_df = pd.concat([enhanced_df, kpca_df], axis=1)
                metadata["kernel_pca"] = {"n_components": kpca_features.shape[1]}
            except Exception as e:
                self.logger.warning(f"Kernel PCA failed: {e}")
                metadata["kernel_pca"] = {"error": str(e)}
            
            metadata["processing_time"] = time.time() - start_time
            metadata["total_enhancement"] = len(enhanced_df.columns) - len(features_df.columns)
            
            self.logger.info(f"✅ Advanced decomposition: +{metadata['total_enhancement']} features")
            return enhanced_df, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Advanced decomposition failed: {e}")
            return features_df, {"error": str(e)}
    
    @secure_data_processing(encryption_level="medium", data_validation=True)
    @memory_efficient(chunk_size=2000, streaming_processing=True)
    @debug_training_step(log_intermediate_results=True)
    @quality_gate(data_quality_metrics={"completeness": 0.9})
    @handle_errors(exceptions=(ValueError, np.linalg.LinAlgError), default_return=None)
    def matrix_completion_techniques(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply matrix completion techniques for missing data.
        
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
            missing_percentage = missing_count / (features_df.shape[0] * features_df.shape[1])
            
            if missing_percentage > 0.01:  # More than 1% missing
                # Use Iterative Imputer (MICE)
                imputer = IterativeImputer(max_iter=10, random_state=42, skip_complete=True)
                completed_features = imputer.fit_transform(features_df)
                
                # Create completed DataFrame
                completed_df = pd.DataFrame(
                    completed_features,
                    columns=features_df.columns,
                    index=features_df.index
                )
                
                metadata = {
                    "missing_count": missing_count,
                    "missing_percentage": missing_percentage,
                    "imputation_method": "iterative_imputer",
                    "processing_time": time.time() - start_time
                }
            else:
                completed_df = features_df
                metadata = {
                    "missing_count": missing_count,
                    "missing_percentage": missing_percentage,
                    "imputation_method": "none_needed",
                    "processing_time": time.time() - start_time
                }
            
            self.logger.info(f"✅ Matrix completion: {missing_percentage:.3f} missing values handled")
            return completed_df, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Matrix completion failed: {e}")
            return features_df, {"error": str(e)}
    
    @secure_data_processing(encryption_level="medium", data_validation=True)
    @memory_efficient(chunk_size=3000, streaming_processing=True)
    @debug_training_step(log_intermediate_results=True)
    @quality_gate(data_quality_metrics={"completeness": 0.9})
    @handle_errors(exceptions=(ValueError, np.linalg.LinAlgError), default_return=None)
    def advanced_clustering_features(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply advanced clustering techniques for feature creation.
        
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
                    affinity='rbf',
                    random_state=42
                )
                spectral_labels = spectral.fit_predict(X_scaled)
                
                # Create cluster features
                spectral_features = pd.get_dummies(spectral_labels, prefix='spectral_cluster')
                spectral_features.index = features_df.index
                enhanced_df = pd.concat([enhanced_df, spectral_features], axis=1)
                
                # Distance to cluster centroids
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=min(8, features_df.shape[1] // 2), random_state=42)
                kmeans.fit(X_scaled)
                distances = euclidean_distances(X_scaled, kmeans.cluster_centers_)
                
                distance_df = pd.DataFrame(
                    distances,
                    columns=[f"distance_to_cluster_{i+1}" for i in range(distances.shape[1])],
                    index=features_df.index
                )
                enhanced_df = pd.concat([enhanced_df, distance_df], axis=1)
                
                metadata["spectral_clustering"] = {
                    "n_clusters": len(np.unique(spectral_labels)),
                    "cluster_sizes": [np.sum(spectral_labels == i) for i in range(len(np.unique(spectral_labels)))]
                }
            except Exception as e:
                self.logger.warning(f"Spectral clustering failed: {e}")
                metadata["spectral_clustering"] = {"error": str(e)}
            
            # 2. DBSCAN for outlier detection
            try:
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                dbscan_labels = dbscan.fit_predict(X_scaled)
                
                # Create outlier features
                outlier_features = pd.DataFrame({
                    'is_outlier': (dbscan_labels == -1).astype(int),
                    'cluster_id': dbscan_labels
                }, index=features_df.index)
                enhanced_df = pd.concat([enhanced_df, outlier_features], axis=1)
                
                metadata["dbscan"] = {
                    "n_clusters": len(np.unique(dbscan_labels[dbscan_labels != -1])),
                    "outlier_count": np.sum(dbscan_labels == -1)
                }
            except Exception as e:
                self.logger.warning(f"DBSCAN failed: {e}")
                metadata["dbscan"] = {"error": str(e)}
            
            metadata["processing_time"] = time.time() - start_time
            metadata["total_enhancement"] = len(enhanced_df.columns) - len(features_df.columns)
            
            self.logger.info(f"✅ Advanced clustering: +{metadata['total_enhancement']} features")
            return enhanced_df, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Advanced clustering failed: {e}")
            return features_df, {"error": str(e)}
    
    @secure_data_processing(encryption_level="high", data_validation=True)
    @resource_monitor(cpu_threshold_percent=80.0, memory_threshold_gb=10.0)
    @memory_efficient(chunk_size=2000, streaming_processing=True)
    @debug_training_step(log_intermediate_results=True)
    @quality_gate(data_quality_metrics={"completeness": 0.95})
    @handle_errors(exceptions=(ValueError, np.linalg.LinAlgError), default_return=None)
    def optimization_algorithms(self, features_df: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply optimization algorithms for feature selection and regularization.
        
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
                    "sparsity": 1.0 - len(selected_features) / len(features_df.columns)
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
                ridge_df = pd.DataFrame(ridge_features, columns=['ridge_prediction'], index=features_df.index)
                enhanced_df = pd.concat([enhanced_df, ridge_df], axis=1)
                
                metadata["ridge"] = {
                    "r2_score": ridge.score(features_df, target),
                    "regularization_strength": ridge.alpha
                }
            except Exception as e:
                self.logger.warning(f"Ridge failed: {e}")
                metadata["ridge"] = {"error": str(e)}
            
            metadata["processing_time"] = time.time() - start_time
            metadata["total_enhancement"] = len(enhanced_df.columns) - len(features_df.columns)
            
            self.logger.info(f"✅ Optimization algorithms: +{metadata['total_enhancement']} features")
            return enhanced_df, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Optimization algorithms failed: {e}")
            return features_df, {"error": str(e)}
    
    @secure_data_processing(encryption_level="medium", data_validation=True)
    @memory_efficient(chunk_size=1000, streaming_processing=True)
    @debug_training_step(log_intermediate_results=True)
    @quality_gate(data_quality_metrics={"completeness": 0.9})
    @handle_errors(exceptions=(ValueError, np.linalg.LinAlgError), default_return=None)
    def advanced_feature_engineering(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply advanced feature engineering techniques.
        
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
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                poly_features = poly.fit_transform(features_df)
                poly_feature_names = [f"poly_interaction_{i+1}" for i in range(poly_features.shape[1] - features_df.shape[1])]
                
                # Select only interaction features (exclude original features)
                interaction_features = poly_features[:, features_df.shape[1]:]
                
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
                    index=features_df.index
                )
                enhanced_df = pd.concat([enhanced_df, poly_df], axis=1)
                
                metadata["polynomial_features"] = {
                    "n_interactions": len(poly_feature_names),
                    "degree": 2
                }
            except Exception as e:
                self.logger.warning(f"Polynomial features failed: {e}")
                metadata["polynomial_features"] = {"error": str(e)}
            
            # 2. Fourier transform features (for time series)
            try:
                # Apply FFT to each feature
                fft_features = []
                fft_feature_names = []
                
                for i, col in enumerate(features_df.columns):
                    fft_vals = np.fft.fft(features_df[col].values)
                    # Take magnitude of first few components
                    n_components = min(5, len(fft_vals) // 2)
                    fft_magnitude = np.abs(fft_vals[:n_components])
                    
                    fft_features.append(fft_magnitude)
                    fft_feature_names.extend([f"fft_{col}_comp_{j+1}" for j in range(n_components)])
                
                # Pad to same length
                max_len = max(len(f) for f in fft_features)
                padded_features = []
                for f in fft_features:
                    padded = np.pad(f, (0, max_len - len(f)), mode='constant')
                    padded_features.append(padded)
                
                fft_array = np.column_stack(padded_features)
                fft_df = pd.DataFrame(
                    fft_array,
                    columns=fft_feature_names[:fft_array.shape[1]],
                    index=features_df.index
                )
                enhanced_df = pd.concat([enhanced_df, fft_df], axis=1)
                
                metadata["fourier_features"] = {
                    "n_fft_features": fft_array.shape[1],
                    "n_components_per_feature": 5
                }
            except Exception as e:
                self.logger.warning(f"Fourier features failed: {e}")
                metadata["fourier_features"] = {"error": str(e)}
            
            metadata["processing_time"] = time.time() - start_time
            metadata["total_enhancement"] = len(enhanced_df.columns) - len(features_df.columns)
            
            self.logger.info(f"✅ Advanced feature engineering: +{metadata['total_enhancement']} features")
            return enhanced_df, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Advanced feature engineering failed: {e}")
            return features_df, {"error": str(e)}
    
    @secure_data_processing(encryption_level="high", data_validation=True)
    @memory_efficient(chunk_size=2000, streaming_processing=True)
    @debug_training_step(log_intermediate_results=True)
    @quality_gate(data_quality_metrics={"completeness": 0.95})
    @handle_errors(exceptions=(ValueError, np.linalg.LinAlgError), default_return=None)
    def quality_assurance_checks(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive quality assurance checks.
        
        Args:
            features_df: Input features DataFrame
            
        Returns:
            Quality assessment results
        """
        try:
            start_time = time.time()
            self.logger.info("🔍 Performing quality assurance checks...")
            
            quality_results = {
                "passed": True,
                "checks": {},
                "recommendations": []
            }
            
            # 1. Numerical stability checks
            try:
                X = features_df.values
                condition_number = np.linalg.cond(X)
                eigenvals = la.eigvals(X.T @ X)
                min_eigenval = np.min(np.abs(eigenvals))
                
                quality_results["checks"]["numerical_stability"] = {
                    "condition_number": condition_number,
                    "min_eigenvalue": min_eigenval,
                    "passed": condition_number < self.config.condition_number_threshold and min_eigenval > self.config.min_eigenvalue_threshold
                }
                
                if condition_number > self.config.condition_number_threshold:
                    quality_results["recommendations"].append("High condition number detected - consider regularization")
                if min_eigenval < self.config.min_eigenvalue_threshold:
                    quality_results["recommendations"].append("Low minimum eigenvalue - consider feature selection")
                    
            except Exception as e:
                quality_results["checks"]["numerical_stability"] = {"error": str(e), "passed": False}
            
            # 2. Data quality checks
            try:
                nan_count = features_df.isna().sum().sum()
                nan_percentage = nan_count / (features_df.shape[0] * features_df.shape[1])
                
                inf_count = np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()
                
                zero_var_features = features_df.var() == 0
                zero_var_count = zero_var_features.sum()
                
                quality_results["checks"]["data_quality"] = {
                    "nan_count": nan_count,
                    "nan_percentage": nan_percentage,
                    "inf_count": inf_count,
                    "zero_variance_features": zero_var_count,
                    "passed": nan_percentage < 0.1 and inf_count == 0 and zero_var_count < len(features_df.columns) * 0.1
                }
                
                if nan_percentage > 0.1:
                    quality_results["recommendations"].append("High NaN percentage - consider imputation")
                if inf_count > 0:
                    quality_results["recommendations"].append("Infinite values detected - check data preprocessing")
                if zero_var_count > len(features_df.columns) * 0.1:
                    quality_results["recommendations"].append("Many zero-variance features - consider feature selection")
                    
            except Exception as e:
                quality_results["checks"]["data_quality"] = {"error": str(e), "passed": False}
            
            # 3. Correlation analysis
            try:
                corr_matrix = features_df.corr().abs()
                high_corr_pairs = np.where(corr_matrix > self.config.correlation_threshold)
                high_corr_count = len(high_corr_pairs[0]) - len(features_df.columns)  # Exclude diagonal
                
                quality_results["checks"]["correlation_analysis"] = {
                    "high_correlation_pairs": high_corr_count,
                    "max_correlation": corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max(),
                    "passed": high_corr_count < len(features_df.columns) * 0.1
                }
                
                if high_corr_count > len(features_df.columns) * 0.1:
                    quality_results["recommendations"].append("High correlation detected - consider feature selection")
                    
            except Exception as e:
                quality_results["checks"]["correlation_analysis"] = {"error": str(e), "passed": False}
            
            # Overall assessment
            all_checks_passed = all(check.get("passed", False) for check in quality_results["checks"].values())
            quality_results["passed"] = all_checks_passed
            quality_results["processing_time"] = time.time() - start_time
            
            self.logger.info(f"✅ Quality assurance: {'PASSED' if all_checks_passed else 'FAILED'}")
            return quality_results
            
        except Exception as e:
            self.logger.error(f"❌ Quality assurance failed: {e}")
            return {"error": str(e), "passed": False}
    
    def comprehensive_matrix_enhancement(self, features_df: pd.DataFrame, target: pd.Series = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply comprehensive matrix enhancement pipeline.
        
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
                self.logger.warning("⚠️ Quality checks failed, applying fixes...")
                # Apply basic fixes
                enhanced_df = enhanced_df.fillna(enhanced_df.mean())
                enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan).fillna(enhanced_df.mean())
            
            # 2. Matrix completion
            enhanced_df, completion_metadata = self.matrix_completion_techniques(enhanced_df)
            all_metadata["matrix_completion"] = completion_metadata
            
            # 3. Eigenvalue-based features
            enhanced_df, eigen_metadata = self.eigenvalue_based_feature_engineering(enhanced_df)
            all_metadata["eigenvalue_features"] = eigen_metadata
            
            # 4. Cholesky covariance
            enhanced_df, cholesky_metadata = self.cholesky_covariance_estimation(enhanced_df)
            all_metadata["cholesky_covariance"] = cholesky_metadata
            
            # 5. Sparse optimizations
            enhanced_df, sparse_metadata = self.sparse_matrix_optimizations(enhanced_df)
            all_metadata["sparse_optimizations"] = sparse_metadata
            
            # 6. Advanced decompositions
            enhanced_df, decomp_metadata = self.advanced_decomposition_techniques(enhanced_df)
            all_metadata["advanced_decompositions"] = decomp_metadata
            
            # 7. Advanced clustering
            enhanced_df, cluster_metadata = self.advanced_clustering_features(enhanced_df)
            all_metadata["advanced_clustering"] = cluster_metadata
            
            # 8. Advanced feature engineering
            enhanced_df, feature_metadata = self.advanced_feature_engineering(enhanced_df)
            all_metadata["advanced_feature_engineering"] = feature_metadata
            
            # 9. Optimization algorithms (if target provided)
            if target is not None:
                enhanced_df, opt_metadata = self.optimization_algorithms(enhanced_df, target)
                all_metadata["optimization_algorithms"] = opt_metadata
            
            # Final quality check
            final_quality = self.quality_assurance_checks(enhanced_df)
            all_metadata["final_quality_assurance"] = final_quality
            
            total_time = time.time() - start_time
            all_metadata["total_processing_time"] = total_time
            all_metadata["feature_count_increase"] = len(enhanced_df.columns) - len(features_df.columns)
            
            self.logger.info(f"✅ Comprehensive matrix enhancement completed in {total_time:.2f}s")
            self.logger.info(f"📊 Features: {len(features_df.columns)} -> {len(enhanced_df.columns)} (+{all_metadata['feature_count_increase']})")
            
            return enhanced_df, all_metadata
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive matrix enhancement failed: {e}")
            return features_df, {"error": str(e)}