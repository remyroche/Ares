from __future__ import annotations
# src/training/matrix_enhancement_manager.py

"""Matrix Enhancement Manager for advanced ML training processes.
Implements sophisticated matrix operations, factorizations, and vector optimizations
to enhance model performance and training efficiency.
"""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

from src.core.decorators import handles_errors
from src.utils.logger import system_logger


@dataclass
class MatrixEnhancementConfig:
    """Configuration for matrix enhancement operations."""

    # Matrix factorization settings
    enable_svd_enhancement: bool = True
    enable_nmf_enhancement: bool = True
    enable_tensor_decomposition: bool = True
    enable_spectral_clustering: bool = True

    # Dimensionality reduction
    svd_n_components: int = 50
    nmf_n_components: int = 30
    tsne_n_components: int = 2

    # Clustering settings
    n_clusters: int = 10
    spectral_n_clusters: int = 8

    # Performance settings
    enable_sparse_operations: bool = True
    enable_gpu_acceleration: bool = False
    batch_size: int = 1000

    # Quality thresholds
    min_explained_variance: float = 0.95
    correlation_threshold: float = 0.8
    condition_number_threshold: float = 1e12

class MatrixEnhancementManager:
    """Advanced matrix enhancement manager for ML training processes.

    Provides sophisticated matrix operations including:
    - SVD-based feature enhancement
    - Non-negative Matrix Factorization
    - Tensor decomposition
    - Spectral clustering
    - Advanced dimensionality reduction
    - Matrix condition number analysis
    - Sparse matrix optimizations
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize matrix enhancement manager."""
        self.config = MatrixEnhancementConfig(**config.get("matrix_enhancement", {}))
        self.logger = system_logger.getChild("MatrixEnhancementManager")
        self.enhancement_results = {}

    @handles_errors(fallback=None)
    def enhance_features_with_svd(
        self,
        features_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Enhance features using Singular Value Decomposition (SVD).

        Args:
            features_df: Input features DataFrame

        Returns:
            Enhanced features DataFrame and metadata

        """
        try:
            start_time = time.time()
            self.logger.info("🔄 Applying SVD-based feature enhancement...")

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features_df)

            # Perform SVD
            U, s, Vt = la.svd(X_scaled, full_matrices=False)

            # Calculate explained variance
            explained_variance = (s**2) / (s**2).sum()
            cumulative_variance = np.cumsum(explained_variance)

            # Select components explaining minimum variance
            n_components = (
                np.argmax(cumulative_variance >= self.config.min_explained_variance) + 1
            )
            n_components = min(n_components, self.config.svd_n_components)

            # Create SVD features
            svd_features = U[:, :n_components] * s[:n_components]
            svd_feature_names = [f"svd_component_{i+1}" for i in range(n_components)]

            # Create DataFrame
            svd_df = pd.DataFrame(
                svd_features, columns=svd_feature_names, index=features_df.index,
            )

            # Combine with original features
            enhanced_df = pd.concat([features_df, svd_df], axis=1)

            # Metadata
            metadata = {
                "svd_n_components": n_components,
                "explained_variance": explained_variance[:n_components].tolist(),
                "cumulative_variance": cumulative_variance[n_components - 1],
                "singular_values": s[:n_components].tolist(),
                "processing_time": time.time() - start_time,
            }

            self.logger.info(
                f"✅ SVD enhancement completed: {n_components} components = {metadata['cumulative_variance']:.3f} variance explained",
            )
            return enhanced_df, metadata

        except Exception as e:
            self.logger.exception(f"❌ SVD enhancement failed: {e}")
            return features_df, {"error": str(e)}

    @handles_errors(fallback=None)
    def enhance_features_with_nmf(
        self,
        features_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Enhance features using Non-negative Matrix Factorization (NMF).

        Args:
            features_df: Input features DataFrame

        Returns:
            Enhanced features DataFrame and metadata

        """
        try:
            start_time = time.time()
            self.logger.info("🔄 Applying NMF-based feature enhancement...")

            # Ensure non-negative data (shift if necessary)
            X = features_df.values
            X_min = np.min(X, axis=0)
            X_shifted = X - X_min if np.any(X_min < 0) else X

            # Apply NMF
            nmf = NMF(
                n_components=self.config.nmf_n_components,
                random_state=42,
                max_iter=200,
            )
            nmf_features = nmf.fit_transform(X_shifted)

            # Create feature names
            nmf_feature_names = [
                f"nmf_component_{i+1}" for i in range(self.config.nmf_n_components)
            ]

            # Create DataFrame
            nmf_df = pd.DataFrame(
                nmf_features, columns=nmf_feature_names, index=features_df.index,
            )

            # Combine with original features
            enhanced_df = pd.concat([features_df, nmf_df], axis=1)

            # Metadata
            metadata = {
                "nmf_n_components": self.config.nmf_n_components,
                "reconstruction_error": nmf.reconstruction_err_,
                "n_iterations": nmf.n_iter_,
                "processing_time": time.time() - start_time,
            }

            self.logger.info(
                f"✅ NMF enhancement completed: {self.config.nmf_n_components} components = reconstruction error: {metadata['reconstruction_error']:.6f}",
            )
            return enhanced_df, metadata

        except Exception as e:
            self.logger.exception(f"❌ NMF enhancement failed: {e}")
            return features_df, {"error": str(e)}

    @handles_errors(fallback=None)
    def apply_spectral_clustering_features(
        self,
        features_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply spectral clustering to create cluster-based features.

        Args:
            features_df: Input features DataFrame

        Returns:
            Enhanced features DataFrame and metadata

        """
        try:
            start_time = time.time()
            self.logger.info("🔄 Applying spectral clustering feature enhancement...")

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features_df)

            # Calculate similarity matrix (cosine similarity)
            X_norm = X_scaled / (np.linalg.norm(X_scaled, axis=1, keepdims=True) + 1e-8)
            similarity_matrix = X_norm @ X_norm.T

            # Apply spectral clustering

            spectral = SpectralClustering(
                n_clusters=self.config.spectral_n_clusters,
                affinity="precomputed",
                random_state=42,
            )
            cluster_labels = spectral.fit_predict(similarity_matrix)

            # Create cluster-based features
            cluster_features = pd.get_dummies(cluster_labels, prefix="spectral_cluster")
            cluster_features.index = features_df.index

            # Distance to cluster centroids

            centroids = []
            for i in range(self.config.spectral_n_clusters):
                mask, cluster_labels = i
                if np.any(mask):
                    centroid = np.mean(X_scaled[mask], axis=0)
                    centroids.append(centroid)
                else:
                    centroids.append(np.zeros(X_scaled.shape[1]))

            centroids = np.array(centroids)
            distances = euclidean_distances(X_scaled, centroids)

            # Create distance features
            distance_feature_names = [
                f"distance_to_cluster_{i+1}"
                for i in range(self.config.spectral_n_clusters)
            ]
            distance_df = pd.DataFrame(
                distances, columns=distance_feature_names, index=features_df.index,
            )

            # Combine all features
            enhanced_df = pd.concat(
                [features_df, cluster_features, distance_df],
                axis=1,
            )

            # Metadata
            metadata = {
                "n_clusters": self.config.spectral_n_clusters,
                "cluster_sizes": [
                    np.sum(cluster_labels == i)
                    for i in range(self.config.spectral_n_clusters)
                ],
                "processing_time": time.time() - start_time,
            }

            self.logger.info(
                f"✅ Spectral clustering enhancement completed: {self.config.spectral_n_clusters} clusters",
            )
            return enhanced_df, metadata

        except Exception as e:
            self.logger.exception(f"❌ Spectral clustering enhancement failed: {e}")
            return features_df, {"error": str(e)}

    @handles_errors(fallback=None)
    def apply_tensor_decomposition(
        self,
        features_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply tensor decomposition for multi-dimensional feature enhancement.

        Args:
            features_df: Input features DataFrame

        Returns:
            Enhanced features DataFrame and metadata

        """
        try:
            start_time = time.time()
            self.logger.info("🔄 Applying tensor decomposition enhancement...")

            # Reshape data into 3D tensor (samples = features, time_windows)
            # For simplicity, we'll create time windows from the data
            X = features_df.values
            n_samples = X.shape[0]
            n_features = X.shape[1]

            # Create time windows (e.g., rolling windows)
            window_size = min(10, n_samples // 10)  # Adaptive window size
            n_windows = n_samples - window_size + 1

            if n_windows > 0:
                # Create 3D tensor
                tensor = np.zeros((n_windows, n_features, window_size))
                for i in range(n_windows):
                    tensor[i, :, :] = X[i : i + window_size].T

                # Apply tensor decomposition (simplified version using SVD on unfolded tensor)
                # Unfold tensor along first mode
                unfolded = tensor.reshape(n_windows * n_features, -1)

                # Apply SVD to unfolded tensor
                U = s, Vt = la.svd(unfolded, full_matrices=False)

                # Select top components
                n_components = min(20, len(s))
                tensor_features = U[:, :n_components] * s[:n_components]

                # Create feature names
                tensor_feature_names = [
                    f"tensor_component_{i+1}" for i in range(n_components)
                ]

                # Create DataFrame (pad with zeros for samples that don't have full windows)
                tensor_df = pd.DataFrame(
                    np.vstack(
                        [
                            tensor_features,
                            np.zeros((n_samples - n_windows, n_components)),
                        ],
                    ),
                    columns=tensor_feature_names,
                    index=features_df.index,
                )

                # Combine with original features
                enhanced_df = pd.concat([features_df, tensor_df], axis=1)

                metadata = {
                    "tensor_n_components": n_components,
                    "window_size": window_size,
                    "n_windows": n_windows,
                    "singular_values": s[:n_components].tolist(),
                    "processing_time": time.time() - start_time,
                }
            else:
                enhanced_df = features_df
                metadata = {"error": "insufficient_samples_for_tensor"}

            self.logger.info("✅ Tensor decomposition enhancement completed")
            return enhanced_df, metadata

        except Exception as e:
            self.logger.exception(f"❌ Tensor decomposition enhancement failed: {e}")
            return features_df, {"error": str(e)}

    @handles_errors(fallback=None)
    def analyze_matrix_condition(self, features_df: pd.DataFrame) -> dict[str, Any]:
        """Analyze matrix condition number and numerical stability.

        Args:
            features_df: Input features DataFrame

        Returns:
            Analysis results dictionary

        """
        try:
            self.logger.info("🔍 Analyzing matrix condition...")

            X = features_df.values

            # Calculate condition number
            condition_number = np.linalg.cond(X)

            # Calculate rank
            rank = np.linalg.matrix_rank(X)

            # Calculate singular values
            singular_values = la.svd(X, compute_uv=False)

            # Calculate condition number ratio
            condition_ratio = singular_values[0] / singular_values[-1]

            # Check for multicollinearity
            correlation_matrix = np.corrcoef(X.T)
            eigenvals = la.eigvals(correlation_matrix)
            min_eigenval = np.min(np.abs(eigenvals))

            analysis = {
                "condition_number": condition_number,
                "condition_ratio": condition_ratio,
                "matrix_rank": rank,
                "full_rank": rank == X.shape[1],
                "min_singular_value": singular_values[-1],
                "max_singular_value": singular_values[0],
                "min_eigenvalue": min_eigenval,
                "numerically_stable": condition_number
                < self.config.condition_number_threshold,
                "well_conditioned": min_eigenval > 1e-6,
            }

            self.logger.info(
                f"📊 Matrix analysis: condition_number={condition_number:.2e}, rank={rank}/{X.shape[1]}",
            )
            return analysis

        except Exception as e:
            self.logger.exception(f"❌ Matrix condition analysis failed: {e}")
            return {"error": str(e)}

    @handles_errors(fallback=None)
    def apply_sparse_matrix_optimizations(
        self,
        features_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply sparse matrix optimizations for large-scale data.

        Args:
            features_df: Input features DataFrame

        Returns:
            Optimized features DataFrame and metadata

        """
        try:
            start_time = time.time()
            self.logger.info("🔄 Applying sparse matrix optimizations...")

            X = features_df.values

            # Convert to sparse matrix if beneficial
            sparsity = 1.0 - np.count_nonzero(X) / X.size

            if sparsity > 0.5:  # If more than 50% zeros
                X_sparse = sp.csr_matrix(X)

                # Apply sparse SVD
                U = s, Vt = sp.linalg.svds(X_sparse, k=min(50, *X.shape))

                # Create sparse features
                sparse_features = U * s
                sparse_feature_names = [
                    f"sparse_component_{i+1}" for i in range(sparse_features.shape[1])
                ]

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
                f"✅ Sparse matrix optimization completed: sparsity={metadata['sparsity']:.3f}",
            )
            return enhanced_df, metadata

        except Exception as e:
            self.logger.exception(f"❌ Sparse matrix optimization failed: {e}")
            return features_df, {"error": str(e)}

    def enhance_training_features(
        self,
        features_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply comprehensive matrix enhancement to training features.

        Args:
            features_df: Input features DataFrame

        Returns:
            Enhanced features DataFrame and comprehensive metadata

        """
        try:
            self.logger.info("🚀 Starting comprehensive matrix enhancement...")
            start_time = time.time()

            enhanced_df = features_df.copy()
            all_metadata = {}

            # 1. Matrix condition analysis
            condition_analysis = self.analyze_matrix_condition(enhanced_df)
            all_metadata["condition_analysis"] = condition_analysis

            # 2. SVD enhancement
            if self.config.enable_svd_enhancement:
                enhanced_df, svd_metadata = self.enhance_features_with_svd(enhanced_df)
                all_metadata["svd_enhancement"] = svd_metadata

            # 3. NMF enhancement
            if self.config.enable_nmf_enhancement:
                enhanced_df, nmf_metadata = self.enhance_features_with_nmf(enhanced_df)
                all_metadata["nmf_enhancement"] = nmf_metadata

            # 4. Spectral clustering
            if self.config.enable_spectral_clustering:
                enhanced_df, spectral_metadata = (
                    self.apply_spectral_clustering_features(enhanced_df)
                )
                all_metadata["spectral_clustering"] = spectral_metadata

            # 5. Tensor decomposition
            if self.config.enable_tensor_decomposition:
                enhanced_df, tensor_metadata = self.apply_tensor_decomposition(
                    enhanced_df,
                )
                all_metadata["tensor_decomposition"] = tensor_metadata

            # 6. Sparse optimizations
            if self.config.enable_sparse_operations:
                enhanced_df, sparse_metadata = self.apply_sparse_matrix_optimizations(
                    enhanced_df,
                )
                all_metadata["sparse_optimization"] = sparse_metadata

            # Final analysis
            final_analysis = self.analyze_matrix_condition(enhanced_df)
            all_metadata["final_analysis"] = final_analysis

            total_time = time.time() - start_time
            all_metadata["total_processing_time"] = total_time
            all_metadata["feature_count_increase"] = len(enhanced_df.columns) - len(
                features_df.columns,
            )

            self.logger.info(f"✅ Matrix enhancement completed in {total_time:.2f}s")
            self.logger.info(
                f"📊 Features: {len(features_df.columns)} -> {len(enhanced_df.columns)} (+{all_metadata['feature_count_increase']})",
            )

            return enhanced_df, all_metadata

        except Exception as e:
            self.logger.exception(f"❌ Comprehensive matrix enhancement failed: {e}")
            return features_df, {"error": str(e)}
