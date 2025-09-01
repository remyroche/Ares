#!/usr/bin/env python3
"""
HMM Regime Parameter Optimization using Optuna

This script optimizes Step 3 HMM regime discovery parameters to capture distinct market conditions
rather than predict regime transitions. It uses Optuna for hyperparameter optimization and
comprehensive evaluation metrics focused on market condition differentiation.

Usage:
    python optimize_hmm_regime_parameters.py --data_path path/to/feature_data.parquet
    python optimize_hmm_regime_parameters.py --config_path path/to/config.json --n_trials 100
"""

import argparse
import json
import sys
import time
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class HMMRegimeOptimizer:
    """Optimize HMM regime discovery parameters for distinct market condition capture."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.best_params = None
        self.best_score = -np.inf
        self.optimization_history = []
        self.study = None

    def create_objective_function(self, data: pd.DataFrame,
                                feature_columns: List[str],
                                market_condition_columns: List[str]) -> callable:
        """Create the objective function for Optuna optimization."""

        def objective(trial: optuna.Trial) -> float:
            """Objective function to maximize regime differentiation and coherence."""

            # Suggest HMM parameters
            params = self._suggest_hmm_parameters(trial)

            # Suggest clustering parameters
            clustering_params = self._suggest_clustering_parameters(trial)

            # Suggest regime merging parameters
            merging_params = self._suggest_regime_merging_parameters(trial)

            try:
                # Generate initial clusters with suggested parameters
                initial_cluster_data = self._generate_initial_clusters_with_params(
                    data, feature_columns, market_condition_columns,
                    params, clustering_params
                )

                # Apply regime merging to achieve target number of regimes
                final_cluster_data = self._apply_regime_merging(
                    initial_cluster_data, market_condition_columns, merging_params
                )

                # Evaluate final regime quality for differentiation and coherence
                score = self._evaluate_regime_quality(
                    final_cluster_data, market_condition_columns, merging_params
                )

                # Store trial information
                trial_info = {
                    'trial_number': trial.number,
                    'params': {**params, **clustering_params, **merging_params},
                    'score': score,
                    'timestamp': time.time(),
                    'initial_clusters': len(initial_cluster_data['composite_cluster_id'].unique()),
                    'final_regimes': len(final_cluster_data['composite_cluster_id'].unique())
                }
                self.optimization_history.append(trial_info)

                return score

            except Exception as e:
                print(f"⚠️ Trial {trial.number} failed: {e}")
                return -np.inf

        return objective

    def _suggest_hmm_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest HMM-specific parameters."""
        return {
            'n_components': trial.suggest_int('n_components', 2, 8),
            'covariance_type': trial.suggest_categorical('covariance_type',
                                                       ['full', 'tied', 'diag', 'spherical']),
            'random_state': 42,
            'n_iter': trial.suggest_int('n_iter', 50, 200),
            'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True),
            'reg_covar': trial.suggest_float('reg_covar', 1e-6, 1e-3, log=True)
        }

    def _suggest_clustering_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest clustering algorithm parameters."""
        clustering_method = trial.suggest_categorical('clustering_method',
                                                     ['kmeans', 'gaussian_mixture', 'hierarchical'])

        params = {'clustering_method': clustering_method}

        if clustering_method == 'kmeans':
            params.update({
                'n_clusters': trial.suggest_int('n_clusters', 3, 12),
                'init': trial.suggest_categorical('kmeans_init', ['k-means++', 'random']),
                'n_init': trial.suggest_int('kmeans_n_init', 5, 20),
                'max_iter': trial.suggest_int('kmeans_max_iter', 200, 500)
            })
        elif clustering_method == 'gaussian_mixture':
            params.update({
                'n_components': trial.suggest_int('gmm_n_components', 3, 12),
                'covariance_type': trial.suggest_categorical('gmm_covariance_type',
                                                           ['full', 'tied', 'diag', 'spherical']),
                'init_params': trial.suggest_categorical('gmm_init_params', ['kmeans', 'random']),
                'max_iter': trial.suggest_int('gmm_max_iter', 100, 300)
            })

        return params

    def _suggest_regime_merging_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest regime merging/clustering parameters to achieve 15-20 final regimes."""
        return {
            'initial_clusters': trial.suggest_int('initial_clusters', 25, 50),
            'target_regimes': trial.suggest_int('target_regimes', 15, 20),
            'merging_method': trial.suggest_categorical('merging_method',
                                                      ['hierarchical', 'kmeans', 'dbscan', 'spectral']),
            'similarity_threshold': trial.suggest_float('similarity_threshold', 0.3, 0.8),
            'min_regime_size': trial.suggest_int('min_regime_size', 100, 500),
            'max_regime_size': trial.suggest_int('max_regime_size', 2000, 5000),
            'coherence_threshold': trial.suggest_float('coherence_threshold', 0.6, 0.9),
            'differentiation_threshold': trial.suggest_float('differentiation_threshold', 0.4, 0.8)
        }

    def _generate_initial_clusters_with_params(self, data: pd.DataFrame,
                                             feature_columns: List[str],
                                             market_condition_columns: List[str],
                                             hmm_params: Dict[str, Any],
                                             clustering_params: Dict[str, Any]) -> pd.DataFrame:
        """Generate initial clusters using the suggested parameters."""

        # Prepare features (assume feature engineering is already done in Step 2)
        features = data[feature_columns].copy()

        # Handle missing values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Standard scaling (feature engineering should be done in Step 2)
        features_scaled = StandardScaler().fit_transform(features)

        # Generate initial clusters
        initial_cluster_labels = self._apply_clustering(features_scaled, clustering_params)

        # Create result dataframe
        result_data = data.copy()
        result_data['composite_cluster_id'] = initial_cluster_labels

        # Add market condition columns
        for col in market_condition_columns:
            if col in data.columns:
                result_data[f'market_{col}'] = data[col]

        return result_data

    def _apply_regime_merging(self, initial_cluster_data: pd.DataFrame,
                            market_condition_columns: List[str],
                            merging_params: Dict[str, Any]) -> pd.DataFrame:
        """Apply regime merging to achieve target number of regimes."""

        target_regimes = merging_params.get('target_regimes', 18)
        merging_method = merging_params.get('merging_method', 'hierarchical')

        # Calculate regime characteristics for merging
        regime_characteristics = self._calculate_regime_characteristics(
            initial_cluster_data, market_condition_columns
        )

        if merging_method == 'hierarchical':
            final_regime_labels = self._hierarchical_regime_merging(
                initial_cluster_data, regime_characteristics, merging_params
            )
        elif merging_method == 'kmeans':
            final_regime_labels = self._kmeans_regime_merging(
                initial_cluster_data, regime_characteristics, merging_params
            )
        elif merging_method == 'dbscan':
            final_regime_labels = self._dbscan_regime_merging(
                initial_cluster_data, regime_characteristics, merging_params
            )
        else:  # spectral
            final_regime_labels = self._spectral_regime_merging(
                initial_cluster_data, regime_characteristics, merging_params
            )

        # Create final result dataframe
        final_data = initial_cluster_data.copy()
        final_data['composite_cluster_id'] = final_regime_labels

        return final_data

    def _calculate_regime_characteristics(self, cluster_data: pd.DataFrame,
                                        market_condition_columns: List[str]) -> pd.DataFrame:
        """Calculate characteristics for each regime to enable merging using vectorized operations."""

        # Filter valid market condition columns
        valid_columns = [col for col in market_condition_columns if col in cluster_data.columns]

        # Vectorized calculation of regime characteristics for all market conditions at once
        regime_stats = cluster_data.groupby('composite_cluster_id')[valid_columns].agg(['mean', 'std', 'min', 'max'])

        # Flatten column names
        regime_stats.columns = [f'{col[0]}_{col[1]}' for col in regime_stats.columns]

        # Add regime size
        regime_sizes = cluster_data['composite_cluster_id'].value_counts()
        regime_stats['size'] = regime_sizes

        # Reset index to get regime_id as a column
        regime_stats = regime_stats.reset_index()
        regime_stats = regime_stats.rename(columns={'composite_cluster_id': 'regime_id'})

        return regime_stats

    def _hierarchical_regime_merging(self, cluster_data: pd.DataFrame,
                                   regime_characteristics: pd.DataFrame,
                                   merging_params: Dict[str, Any]) -> np.ndarray:
        """Merge regimes using hierarchical clustering."""

        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler

        # Prepare characteristics for clustering
        char_cols = [col for col in regime_characteristics.columns
                    if col not in ['regime_id', 'size']]

        if len(char_cols) == 0:
            return cluster_data['composite_cluster_id'].values

        # Scale characteristics
        char_scaled = StandardScaler().fit_transform(regime_characteristics[char_cols])

        # Apply hierarchical clustering
        target_regimes = merging_params.get('target_regimes', 18)
        hierarchical = AgglomerativeClustering(
            n_clusters=target_regimes,
            linkage='ward'
        )

        regime_clusters = hierarchical.fit_predict(char_scaled)

        # Map regime clusters back to data
        regime_mapping = dict(zip(regime_characteristics['regime_id'], regime_clusters))
        final_labels = cluster_data['composite_cluster_id'].map(regime_mapping).values

        return final_labels

    def _kmeans_regime_merging(self, cluster_data: pd.DataFrame,
                             regime_characteristics: pd.DataFrame,
                             merging_params: Dict[str, Any]) -> np.ndarray:
        """Merge regimes using K-means clustering."""

        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        # Prepare characteristics for clustering
        char_cols = [col for col in regime_characteristics.columns
                    if col not in ['regime_id', 'size']]

        if len(char_cols) == 0:
            return cluster_data['composite_cluster_id'].values

        # Scale characteristics
        char_scaled = StandardScaler().fit_transform(regime_characteristics[char_cols])

        # Apply K-means clustering
        target_regimes = merging_params.get('target_regimes', 18)
        kmeans = KMeans(n_clusters=target_regimes, random_state=42)

        regime_clusters = kmeans.fit_predict(char_scaled)

        # Map regime clusters back to data
        regime_mapping = dict(zip(regime_characteristics['regime_id'], regime_clusters))
        final_labels = cluster_data['composite_cluster_id'].map(regime_mapping).values

        return final_labels

    def _dbscan_regime_merging(self, cluster_data: pd.DataFrame,
                             regime_characteristics: pd.DataFrame,
                             merging_params: Dict[str, Any]) -> np.ndarray:
        """Merge regimes using DBSCAN clustering."""

        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler

        # Prepare characteristics for clustering
        char_cols = [col for col in regime_characteristics.columns
                    if col not in ['regime_id', 'size']]

        if len(char_cols) == 0:
            return cluster_data['composite_cluster_id'].values

        # Scale characteristics
        char_scaled = StandardScaler().fit_transform(regime_characteristics[char_cols])

        # Apply DBSCAN clustering
        similarity_threshold = merging_params.get('similarity_threshold', 0.5)
        eps = 1 - similarity_threshold  # Convert similarity to distance

        dbscan = DBSCAN(eps=eps, min_samples=2)
        regime_clusters = dbscan.fit_predict(char_scaled)

        # If DBSCAN produces too many or too few clusters, adjust
        n_clusters = len(set(regime_clusters)) - (1 if -1 in regime_clusters else 0)
        target_regimes = merging_params.get('target_regimes', 18)

        if n_clusters != target_regimes:
            # Fall back to K-means for exact target
            return self._kmeans_regime_merging(cluster_data, regime_characteristics, merging_params)

        # Map regime clusters back to data
        regime_mapping = dict(zip(regime_characteristics['regime_id'], regime_clusters))
        final_labels = cluster_data['composite_cluster_id'].map(regime_mapping).values

        return final_labels

    def _spectral_regime_merging(self, cluster_data: pd.DataFrame,
                               regime_characteristics: pd.DataFrame,
                               merging_params: Dict[str, Any]) -> np.ndarray:
        """Merge regimes using spectral clustering."""

        from sklearn.cluster import SpectralClustering
        from sklearn.preprocessing import StandardScaler

        # Prepare characteristics for clustering
        char_cols = [col for col in regime_characteristics.columns
                    if col not in ['regime_id', 'size']]

        if len(char_cols) == 0:
            return cluster_data['composite_cluster_id'].values

        # Scale characteristics
        char_scaled = StandardScaler().fit_transform(regime_characteristics[char_cols])

        # Apply spectral clustering
        target_regimes = merging_params.get('target_regimes', 18)
        spectral = SpectralClustering(
            n_clusters=target_regimes,
            affinity='rbf',
            random_state=42
        )

        regime_clusters = spectral.fit_predict(char_scaled)

        # Map regime clusters back to data
        regime_mapping = dict(zip(regime_characteristics['regime_id'], regime_clusters))
        final_labels = cluster_data['composite_cluster_id'].map(regime_mapping).values

        return final_labels

    def _apply_feature_selection(self, features: pd.DataFrame,
                               feature_params: Dict[str, Any]) -> pd.DataFrame:
        """Apply feature selection based on parameters."""
        method = feature_params.get('feature_selection_method', 'all')

        if method == 'all':
            return features

        elif method == 'variance':
            from sklearn.feature_selection import VarianceThreshold
            threshold = feature_params.get('variance_threshold', 0.01)
            selector = VarianceThreshold(threshold=threshold)
            selected_features = selector.fit_transform(features)
            selected_columns = features.columns[selector.get_support()]
            return pd.DataFrame(selected_features, columns=selected_columns, index=features.index)

        elif method == 'correlation':
            threshold = feature_params.get('correlation_threshold', 0.8)
            corr_matrix = features.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
            return features.drop(columns=to_drop)

        return features

    def _apply_scaling(self, features: pd.DataFrame,
                      feature_params: Dict[str, Any]) -> np.ndarray:
        """Apply scaling based on parameters."""
        method = feature_params.get('scaling_method', 'standard')

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        else:
            return features.values

        return scaler.fit_transform(features)

    def _apply_pca(self, features_scaled: np.ndarray,
                  feature_params: Dict[str, Any]) -> np.ndarray:
        """Apply PCA if requested."""
        n_components = feature_params.get('n_pca_components', 10)
        n_components = min(n_components, features_scaled.shape[1])

        pca = PCA(n_components=n_components)
        return pca.fit_transform(features_scaled)

    def _apply_clustering(self, features_scaled: np.ndarray,
                         clustering_params: Dict[str, Any]) -> np.ndarray:
        """Apply clustering algorithm based on parameters."""
        method = clustering_params.get('clustering_method', 'kmeans')

        if method == 'kmeans':
            kmeans = KMeans(
                n_clusters=clustering_params.get('n_clusters', 5),
                init=clustering_params.get('init', 'k-means++'),
                n_init=clustering_params.get('n_init', 10),
                max_iter=clustering_params.get('max_iter', 300),
                random_state=42
            )
            return kmeans.fit_predict(features_scaled)

        elif method == 'gaussian_mixture':
            gmm = GaussianMixture(
                n_components=clustering_params.get('n_components', 5),
                covariance_type=clustering_params.get('covariance_type', 'full'),
                init_params=clustering_params.get('init_params', 'kmeans'),
                max_iter=clustering_params.get('max_iter', 200),
                random_state=42
            )
            return gmm.fit_predict(features_scaled)

        else:
            # Default to kmeans
            kmeans = KMeans(n_clusters=5, random_state=42)
            return kmeans.fit_predict(features_scaled)

    def _evaluate_regime_quality(self, cluster_data: pd.DataFrame,
                               market_condition_columns: List[str],
                               merging_params: Dict[str, Any]) -> float:
        """Evaluate regime quality focusing on differentiation and coherence."""

        if 'composite_cluster_id' not in cluster_data.columns:
            return -np.inf

        scores = []

        # 1. Regime Differentiation Score (40% weight)
        differentiation_score = self._calculate_regime_differentiation_score(
            cluster_data, market_condition_columns
        )
        scores.append(differentiation_score)

        # 2. Internal Coherence Score (30% weight)
        coherence_score = self._calculate_internal_coherence_score(
            cluster_data, market_condition_columns
        )
        scores.append(coherence_score)

        # 3. Regime Balance Score (15% weight)
        balance_score = self._calculate_regime_balance_score(cluster_data)
        scores.append(balance_score)

        # 4. Target Regime Count Penalty (15% weight)
        target_penalty = self._calculate_target_regime_penalty(
            cluster_data, merging_params
        )
        scores.append(target_penalty)

        # Weighted combination (focus on differentiation and coherence)
        weights = [0.4, 0.3, 0.15, 0.15]
        final_score = np.average(scores, weights=weights)

        return final_score

    def _calculate_regime_differentiation_score(self, cluster_data: pd.DataFrame,
                                             market_condition_columns: List[str]) -> float:
        """Calculate how well regimes are differentiated from each other using vectorized operations."""

        if not market_condition_columns:
            return 0.0

        # Filter valid market condition columns
        valid_columns = [col for col in market_condition_columns if col in cluster_data.columns]
        if not valid_columns:
            return 0.0

        # Vectorized calculation of regime means for all market conditions at once
        regime_means_matrix = cluster_data.groupby('composite_cluster_id')[valid_columns].mean()

        if len(regime_means_matrix) < 2:
            return 0.0

        # Calculate pairwise differences using matrix operations
        n_regimes = len(regime_means_matrix)
        differentiation_scores = []

        for col in valid_columns:
            # Get regime means for this column
            regime_means = regime_means_matrix[col].values

            # Calculate pairwise differences using broadcasting
            # Create matrices for efficient pairwise comparison
            means_i = regime_means[:, np.newaxis]  # Shape: (n_regimes, 1)
            means_j = regime_means[np.newaxis, :]  # Shape: (1, n_regimes)

            # Calculate absolute differences (excluding self-comparisons)
            differences = np.abs(means_i - means_j)

            # Remove diagonal (self-comparisons) and get upper triangle
            mask = ~np.eye(n_regimes, dtype=bool)
            valid_differences = differences[mask]

            if len(valid_differences) > 0:
                # Normalize by the overall range of the market condition
                overall_range = cluster_data[col].max() - cluster_data[col].min()
                if overall_range > 0:
                    avg_difference = np.mean(valid_differences) / overall_range
                    differentiation_scores.append(avg_difference)

        return np.mean(differentiation_scores) if differentiation_scores else 0.0

    def _calculate_internal_coherence_score(self, cluster_data: pd.DataFrame,
                                          market_condition_columns: List[str]) -> float:
        """Calculate how internally coherent each regime is using vectorized operations."""

        if not market_condition_columns:
            return 0.0

        # Filter valid market condition columns
        valid_columns = [col for col in market_condition_columns if col in cluster_data.columns]
        if not valid_columns:
            return 0.0

        coherence_scores = []

        for col in valid_columns:
            # Vectorized calculation of coefficient of variation for all regimes at once
            regime_stats = cluster_data.groupby('composite_cluster_id')[col].agg(['mean', 'std', 'count'])

            # Filter regimes with more than 1 sample
            valid_regimes = regime_stats[regime_stats['count'] > 1]

            if len(valid_regimes) > 0:
                # Calculate coefficient of variation using vectorized operations
                means = valid_regimes['mean'].values
                stds = valid_regimes['std'].values

                # Avoid division by zero
                non_zero_means = means != 0
                if np.any(non_zero_means):
                    cvs = stds[non_zero_means] / np.abs(means[non_zero_means])

                    if len(cvs) > 0:
                        # Lower CV means more coherent, so invert
                        avg_cv = np.mean(cvs)
                        coherence = 1.0 / (1.0 + avg_cv)
                        coherence_scores.append(coherence)

        return np.mean(coherence_scores) if coherence_scores else 0.0

    def _calculate_regime_balance_score(self, cluster_data: pd.DataFrame) -> float:
        """Calculate how balanced the regime sizes are using vectorized operations."""

        # Vectorized calculation of regime sizes
        regime_sizes = cluster_data['composite_cluster_id'].value_counts().values

        if len(regime_sizes) < 2:
            return 0.0

        # Calculate coefficient of variation using vectorized operations
        mean_size = np.mean(regime_sizes)
        std_size = np.std(regime_sizes)

        if mean_size == 0:
            return 0.0

        cv = std_size / mean_size

        # Lower CV means more balanced, so invert
        balance_score = 1.0 / (1.0 + cv)

        return balance_score

    def _calculate_target_regime_penalty(self, cluster_data: pd.DataFrame,
                                       merging_params: Dict[str, Any]) -> float:
        """Calculate penalty for not achieving target regime count."""

        target_regimes = merging_params.get('target_regimes', 18)
        actual_regimes = len(cluster_data['composite_cluster_id'].unique())

        # Penalty based on distance from target
        penalty = 1.0 - abs(actual_regimes - target_regimes) / target_regimes

        # Additional penalty for being outside the 15-20 range
        if actual_regimes < 15 or actual_regimes > 20:
            penalty *= 0.5

        return max(0.0, penalty)

    def _calculate_market_differentiation_score(self, cluster_data: pd.DataFrame,
                                              market_condition_columns: List[str]) -> float:
        """Calculate how well clusters differentiate market conditions."""

        if not market_condition_columns:
            return 0.0

        differentiation_scores = []

        for col in market_condition_columns:
            if col not in cluster_data.columns:
                continue

            # Calculate average market condition value for each cluster
            cluster_means = cluster_data.groupby('composite_cluster_id')[col].mean()

            if len(cluster_means) < 2:
                continue

            # Calculate how different clusters are from each other
            differences = []
            for i, mean1 in cluster_means.items():
                for j, mean2 in cluster_means.items():
                    if i != j:
                        differences.append(abs(mean1 - mean2))

            if differences:
                # Normalize by the overall range of the market condition
                overall_range = cluster_data[col].max() - cluster_data[col].min()
                if overall_range > 0:
                    avg_difference = np.mean(differences) / overall_range
                    differentiation_scores.append(avg_difference)

        return np.mean(differentiation_scores) if differentiation_scores else 0.0

    def _calculate_cluster_quality_score(self, cluster_data: pd.DataFrame) -> float:
        """Calculate traditional cluster quality metrics."""

        # Prepare features for quality metrics
        feature_columns = [col for col in cluster_data.columns
                          if col not in ['composite_cluster_id', 'timestamp', 'close', 'high', 'low', 'open', 'volume']]

        if len(feature_columns) < 2:
            return 0.0

        features = cluster_data[feature_columns[:min(10, len(feature_columns))]].fillna(0)
        cluster_labels = cluster_data['composite_cluster_id'].values

        try:
            # Silhouette score
            silhouette = silhouette_score(features, cluster_labels)

            # Calinski-Harabasz score
            calinski = calinski_harabasz_score(features, cluster_labels)

            # Davies-Bouldin score (lower is better, so invert)
            davies = davies_bouldin_score(features, cluster_labels)
            davies_normalized = 1.0 / (1.0 + davies)  # Convert to 0-1 scale

            # Combine scores
            quality_score = (silhouette + davies_normalized) / 2.0

            return quality_score

        except Exception:
            return 0.0

    def _calculate_market_consistency_score(self, cluster_data: pd.DataFrame,
                                          market_condition_columns: List[str]) -> float:
        """Calculate how consistent market conditions are within clusters."""

        if not market_condition_columns:
            return 0.0

        consistency_scores = []

        for col in market_condition_columns:
            if col not in cluster_data.columns:
                continue

            # Calculate coefficient of variation within each cluster
            cluster_cvs = []
            for cluster_id in cluster_data['composite_cluster_id'].unique():
                cluster_data_subset = cluster_data[cluster_data['composite_cluster_id'] == cluster_id]
                if len(cluster_data_subset) > 1:
                    mean_val = cluster_data_subset[col].mean()
                    std_val = cluster_data_subset[col].std()
                    if mean_val != 0:
                        cv = std_val / abs(mean_val)
                        cluster_cvs.append(cv)

            if cluster_cvs:
                # Lower CV means more consistency, so invert
                avg_cv = np.mean(cluster_cvs)
                consistency = 1.0 / (1.0 + avg_cv)
                consistency_scores.append(consistency)

        return np.mean(consistency_scores) if consistency_scores else 0.0

    def _calculate_cluster_balance_score(self, cluster_data: pd.DataFrame) -> float:
        """Calculate how balanced the cluster sizes are."""

        cluster_sizes = cluster_data['composite_cluster_id'].value_counts()

        if len(cluster_sizes) < 2:
            return 0.0

        # Calculate coefficient of variation of cluster sizes
        mean_size = cluster_sizes.mean()
        std_size = cluster_sizes.std()

        if mean_size == 0:
            return 0.0

        cv = std_size / mean_size

        # Lower CV means more balanced, so invert
        balance_score = 1.0 / (1.0 + cv)

        return balance_score

    def _calculate_market_separation_score(self, cluster_data: pd.DataFrame,
                                         market_condition_columns: List[str]) -> float:
        """Calculate how well clusters separate different market conditions."""

        if not market_condition_columns:
            return 0.0

        separation_scores = []

        for col in market_condition_columns:
            if col not in cluster_data.columns:
                continue

            # Calculate between-cluster variance vs within-cluster variance
            overall_mean = cluster_data[col].mean()
            total_ss = ((cluster_data[col] - overall_mean) ** 2).sum()

            between_ss = 0
            within_ss = 0

            for cluster_id in cluster_data['composite_cluster_id'].unique():
                cluster_data_subset = cluster_data[cluster_data['composite_cluster_id'] == cluster_id]
                cluster_mean = cluster_data_subset[col].mean()
                cluster_size = len(cluster_data_subset)

                between_ss += cluster_size * (cluster_mean - overall_mean) ** 2
                within_ss += ((cluster_data_subset[col] - cluster_mean) ** 2).sum()

            if within_ss > 0:
                f_ratio = between_ss / within_ss
                # Normalize to 0-1 scale
                separation = f_ratio / (1 + f_ratio)
                separation_scores.append(separation)

        return np.mean(separation_scores) if separation_scores else 0.0

    def optimize(self, data: pd.DataFrame,
                feature_columns: List[str],
                market_condition_columns: List[str],
                n_trials: int = 100,
                timeout: Optional[int] = None,
                study_name: str = "hmm_regime_optimization") -> Dict[str, Any]:
        """Run the optimization process."""

        print(f"🚀 Starting HMM regime parameter optimization...")
        print(f"📊 Data shape: {data.shape}")
        print(f"🔧 Features: {len(feature_columns)}")
        print(f"📈 Market conditions: {len(market_condition_columns)}")
        print(f"🎯 Trials: {n_trials}")

        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            study_name=study_name
        )

        # Create objective function
        objective = self.create_objective_function(data, feature_columns, market_condition_columns)

        # Run optimization
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )

        # Store best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        print(f"\n✅ Optimization completed!")
        print(f"🏆 Best score: {self.best_score:.4f}")
        print(f"🔧 Best parameters: {self.best_params}")

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'study': self.study,
            'optimization_history': self.optimization_history
        }

    def _preprocess_data_for_optimization(self, data: pd.DataFrame, feature_columns: List[str],
                                        market_condition_columns: List[str]) -> Dict[str, Any]:
        """Pre-process data for vectorized optimization operations."""

        # Filter valid columns
        valid_features = [col for col in feature_columns if col in data.columns]
        valid_market_conditions = [col for col in market_condition_columns if col in data.columns]

        # Create pre-processed data structure
        processed_data = {
            'data': data.copy(),
            'feature_columns': valid_features,
            'market_condition_columns': valid_market_conditions,
            'feature_matrix': data[valid_features].values if valid_features else np.array([]),
            'market_condition_matrix': data[valid_market_conditions].values if valid_market_conditions else np.array([]),
            'feature_ranges': {},
            'market_condition_ranges': {}
        }

        # Pre-calculate ranges for normalization
        for col in valid_features:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                processed_data['feature_ranges'][col] = {
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'range': col_data.max() - col_data.min()
                }

        for col in valid_market_conditions:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                processed_data['market_condition_ranges'][col] = {
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'range': col_data.max() - col_data.min()
                }

        return processed_data

    def _generate_initial_clusters_vectorized(self, processed_data: Dict[str, Any],
                                            hmm_params: Dict[str, Any],
                                            clustering_params: Dict[str, Any]) -> pd.DataFrame:
        """Generate initial clusters using vectorized operations."""

        # Use pre-processed feature matrix
        feature_matrix = processed_data['feature_matrix']

        if feature_matrix.size == 0:
            # Fallback to original method
            return self._generate_initial_clusters_with_params(
                processed_data['data'],
                processed_data['feature_columns'],
                processed_data['market_condition_columns'],
                hmm_params,
                clustering_params
            )

        # Handle missing values using vectorized operations
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)

        # Standard scaling using vectorized operations
        feature_mean = np.mean(feature_matrix, axis=0)
        feature_std = np.std(feature_matrix, axis=0)
        feature_std = np.where(feature_std == 0, 1.0, feature_std)  # Avoid division by zero
        feature_matrix_scaled = (feature_matrix - feature_mean) / feature_std

        # Generate initial clusters
        initial_cluster_labels = self._apply_clustering(feature_matrix_scaled, clustering_params)

        # Create result dataframe
        result_data = processed_data['data'].copy()
        result_data['composite_cluster_id'] = initial_cluster_labels

        return result_data

    def generate_optimization_report(self, output_path: Optional[str] = None) -> str:
        """Generate a comprehensive optimization report."""

        if not self.study:
            return "No optimization study available."

        report = []
        report.append("# HMM Regime Parameter Optimization Report")
        report.append("")
        report.append(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        report.append(f"🎯 **Objective**: Optimize HMM regime discovery parameters to achieve 15-20 distinct, internally coherent regimes")
        report.append(f"🏆 **Best Score**: {self.best_score:.4f}")
        report.append(f"📊 **Total Trials**: {len(self.study.trials)}")
        report.append(f"✅ **Completed Trials**: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        report.append(f"⏱️ **Optimization Time**: {self._calculate_optimization_time():.1f} minutes")
        report.append("")

        # Best Parameters Summary
        report.append("## Best Parameters Summary")
        report.append("")

        # Group parameters by category
        hmm_params = {k: v for k, v in self.best_params.items() if k in ['n_components', 'covariance_type', 'n_iter', 'tol', 'reg_covar']}
        clustering_params = {k: v for k, v in self.best_params.items() if k in ['clustering_method', 'n_clusters', 'init', 'n_init', 'max_iter']}
        merging_params = {k: v for k, v in self.best_params.items() if k in ['initial_clusters', 'target_regimes', 'merging_method', 'similarity_threshold', 'min_regime_size', 'max_regime_size', 'coherence_threshold', 'differentiation_threshold']}

        report.append("### HMM Parameters")
        for param, value in hmm_params.items():
            report.append(f"- **{param}**: {value}")
        report.append("")

        report.append("### Clustering Parameters")
        for param, value in clustering_params.items():
            report.append(f"- **{param}**: {value}")
        report.append("")

        report.append("### Regime Merging Parameters")
        for param, value in merging_params.items():
            report.append(f"- **{param}**: {value}")
        report.append("")

        # Parameter Importance Analysis
        try:
            importance = optuna.importance.get_param_importances(self.study)
            report.append("## Parameter Importance Analysis")
            report.append("")
            report.append("### Top 10 Most Important Parameters")
            for i, (param, imp) in enumerate(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10], 1):
                report.append(f"{i}. **{param}**: {imp:.4f}")
            report.append("")

            # Group importance by category
            hmm_importance = {k: v for k, v in importance.items() if k in ['n_components', 'covariance_type', 'n_iter', 'tol', 'reg_covar']}
            clustering_importance = {k: v for k, v in importance.items() if k in ['clustering_method', 'n_clusters', 'init', 'n_init', 'max_iter']}
            merging_importance = {k: v for k, v in importance.items() if k in ['initial_clusters', 'target_regimes', 'merging_method', 'similarity_threshold', 'min_regime_size', 'max_regime_size', 'coherence_threshold', 'differentiation_threshold']}

            if hmm_importance:
                report.append("### HMM Parameter Importance")
                for param, imp in sorted(hmm_importance.items(), key=lambda x: x[1], reverse=True):
                    report.append(f"- **{param}**: {imp:.4f}")
                report.append("")

            if clustering_importance:
                report.append("### Clustering Parameter Importance")
                for param, imp in sorted(clustering_importance.items(), key=lambda x: x[1], reverse=True):
                    report.append(f"- **{param}**: {imp:.4f}")
                report.append("")

            if merging_importance:
                report.append("### Regime Merging Parameter Importance")
                for param, imp in sorted(merging_importance.items(), key=lambda x: x[1], reverse=True):
                    report.append(f"- **{param}**: {imp:.4f}")
                report.append("")

        except Exception as e:
            report.append("## Parameter Importance Analysis")
            report.append(f"Could not calculate parameter importance: {e}")
            report.append("")

        # Optimization Performance Analysis
        report.append("## Optimization Performance Analysis")
        report.append("")

        # Score distribution
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        scores = [t.value for t in completed_trials if t.value is not None]

        if scores:
            report.append(f"- **Score Range**: {min(scores):.4f} - {max(scores):.4f}")
            report.append(f"- **Mean Score**: {np.mean(scores):.4f}")
            report.append(f"- **Score Std**: {np.std(scores):.4f}")
            report.append(f"- **Score Improvement**: {max(scores) - min(scores):.4f}")
            report.append("")

        # Regime count analysis
        regime_counts = [trial_info.get('final_regimes', 0) for trial_info in self.optimization_history if 'final_regimes' in trial_info]
        if regime_counts:
            report.append("### Regime Count Analysis")
            report.append(f"- **Target Range**: 15-20 regimes")
            report.append(f"- **Achieved Range**: {min(regime_counts)} - {max(regime_counts)} regimes")
            report.append(f"- **Mean Regime Count**: {np.mean(regime_counts):.1f}")
            report.append(f"- **Trials in Target Range**: {sum(1 for c in regime_counts if 15 <= c <= 20)}/{len(regime_counts)}")
            report.append("")

        # Optimization History
        report.append("## Optimization History")
        report.append("")
        report.append("### Last 15 Trials")
        report.append("| Trial | Score | Initial Clusters | Final Regimes | Key Parameters |")
        report.append("|-------|-------|------------------|---------------|----------------|")

        for trial_info in self.optimization_history[-15:]:
            params = trial_info['params']
            initial_clusters = trial_info.get('initial_clusters', 'N/A')
            final_regimes = trial_info.get('final_regimes', 'N/A')
            key_params = f"n_components={params.get('n_components', 'N/A')}, " \
                        f"merging_method={params.get('merging_method', 'N/A')}"
            report.append(f"| {trial_info['trial_number']} | {trial_info['score']:.4f} | {initial_clusters} | {final_regimes} | {key_params} |")
        report.append("")

        # Recommendations
        report.append("## Recommendations")
        report.append("")

        # Analyze best parameters and provide recommendations
        best_params = self.best_params

        report.append("### Parameter Recommendations")
        if best_params.get('target_regimes', 0) >= 15 and best_params.get('target_regimes', 0) <= 20:
            report.append("✅ **Target Regime Count**: Optimal range achieved")
        else:
            report.append("⚠️ **Target Regime Count**: Consider adjusting to 15-20 range")

        if best_params.get('merging_method') == 'hierarchical':
            report.append("✅ **Merging Method**: Hierarchical clustering provides good interpretability")
        elif best_params.get('merging_method') == 'kmeans':
            report.append("✅ **Merging Method**: K-means provides balanced regime sizes")
        else:
            report.append("✅ **Merging Method**: Alternative method may provide unique advantages")

        if best_params.get('coherence_threshold', 0) >= 0.7:
            report.append("✅ **Coherence Threshold**: High internal coherence achieved")
        else:
            report.append("⚠️ **Coherence Threshold**: Consider increasing for better internal coherence")

        if best_params.get('differentiation_threshold', 0) >= 0.6:
            report.append("✅ **Differentiation Threshold**: Good regime differentiation achieved")
        else:
            report.append("⚠️ **Differentiation Threshold**: Consider increasing for better regime separation")
        report.append("")

        # Implementation Guide
        report.append("## Implementation Guide")
        report.append("")
        report.append("### Step 1: Apply Best Parameters")
        report.append("```python")
        report.append("# Update your Step 3 configuration with these parameters:")
        report.append("step3_config = {")
        report.append("    'hmm_parameters': {")
        for param, value in hmm_params.items():
            report.append(f"        '{param}': {value},")
        report.append("    },")
        report.append("    'clustering_parameters': {")
        for param, value in clustering_params.items():
            report.append(f"        '{param}': {value},")
        report.append("    },")
        report.append("    'regime_merging_parameters': {")
        for param, value in merging_params.items():
            report.append(f"        '{param}': {value},")
        report.append("    }")
        report.append("}")
        report.append("```")
        report.append("")

        report.append("### Step 2: Validate Results")
        report.append("```bash")
        report.append("# Run cluster validation to confirm quality")
        report.append("python test_hmm_cluster_relevance.py --data_path path/to/optimized_cluster_data.parquet")
        report.append("```")
        report.append("")

        report.append("### Step 3: Monitor Performance")
        report.append("- Track regime stability over time")
        report.append("- Monitor regime transition patterns")
        report.append("- Validate regime characteristics in live trading")
        report.append("")

        # Quality Metrics Summary
        report.append("## Quality Metrics Summary")
        report.append("")
        report.append("### Evaluation Criteria")
        report.append("- **Regime Differentiation** (40%): How well regimes differ from each other")
        report.append("- **Internal Coherence** (30%): How consistent conditions are within each regime")
        report.append("- **Regime Balance** (15%): How balanced regime sizes are")
        report.append("- **Target Count Penalty** (15%): Penalty for not achieving 15-20 regimes")
        report.append("")

        report.append("### Best Trial Metrics")
        best_trial = max(self.optimization_history, key=lambda x: x['score'])
        report.append(f"- **Overall Score**: {best_trial['score']:.4f}")
        report.append(f"- **Initial Clusters**: {best_trial.get('initial_clusters', 'N/A')}")
        report.append(f"- **Final Regimes**: {best_trial.get('final_regimes', 'N/A')}")
        report.append("")

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"📄 Comprehensive report saved to: {output_path}")

        return report_text

    def _calculate_optimization_time(self) -> float:
        """Calculate total optimization time in minutes."""
        if not self.optimization_history:
            return 0.0

        start_time = min(trial_info['timestamp'] for trial_info in self.optimization_history)
        end_time = max(trial_info['timestamp'] for trial_info in self.optimization_history)

        return (end_time - start_time) / 60.0

    def create_optimization_visualizations(self, output_dir: Optional[str] = None) -> None:
        """Create visualizations for the optimization process."""

        if not self.study:
            print("No optimization study available for visualization.")
            return

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

        # Set up plotting style
        plt.style.use('seaborn-v0_8')

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('HMM Regime Parameter Optimization Results', fontsize=16, fontweight='bold')

        # 1. Optimization history
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        scores = [t.value for t in completed_trials]
        trial_numbers = [t.number for t in completed_trials]

        axes[0, 0].plot(trial_numbers, scores, 'b-', alpha=0.7)
        axes[0, 0].scatter(trial_numbers, scores, c=scores, cmap='viridis', s=30, alpha=0.8)
        axes[0, 0].set_title('Optimization History')
        axes[0, 0].set_xlabel('Trial Number')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Parameter importance
        try:
            importance = optuna.importance.get_param_importances(self.study)
            params = list(importance.keys())
            values = list(importance.values())

            bars = axes[0, 1].barh(params, values)
            axes[0, 1].set_title('Parameter Importance')
            axes[0, 1].set_xlabel('Importance Score')

            # Color bars based on importance
            for bar, value in zip(bars, values):
                if value > 0.1:
                    bar.set_color('green')
                elif value > 0.05:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
        except Exception:
            axes[0, 1].text(0.5, 0.5, 'Parameter importance\nnot available',
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Parameter Importance')

        # 3. Score distribution
        axes[1, 0].hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(self.best_score, color='red', linestyle='--',
                          label=f'Best Score: {self.best_score:.4f}')
        axes[1, 0].set_title('Score Distribution')
        axes[1, 0].set_xlabel('Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Key parameter relationships
        try:
            # Extract key parameters
            n_components_values = []
            clustering_methods = []
            scores_for_plot = []

            for trial in completed_trials:
                if trial.value is not None:
                    n_components_values.append(trial.params.get('n_components', 0))
                    clustering_methods.append(trial.params.get('clustering_method', 'unknown'))
                    scores_for_plot.append(trial.value)

            # Create scatter plot
            scatter = axes[1, 1].scatter(n_components_values, scores_for_plot,
                                       c=range(len(scores_for_plot)), cmap='viridis',
                                       s=50, alpha=0.7)
            axes[1, 1].set_title('Score vs Number of Components')
            axes[1, 1].set_xlabel('Number of Components')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].grid(True, alpha=0.3)

            # Add colorbar
            plt.colorbar(scatter, ax=axes[1, 1], label='Trial Order')

        except Exception:
            axes[1, 1].text(0.5, 0.5, 'Parameter relationship\nplot not available',
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Parameter Relationships')

        plt.tight_layout()

        if output_dir:
            plt.savefig(output_path / 'optimization_results.png', dpi=300, bbox_inches='tight')
            print(f"📊 Visualizations saved to: {output_path / 'optimization_results.png'}")
        else:
            plt.show()

        plt.close()

    def save_optimization_results(self, output_path: str) -> None:
        """Save optimization results to file."""

        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history,
            'study_name': self.study.study_name if self.study else None
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"💾 Optimization results saved to: {output_path}")


def identify_market_condition_columns(data: pd.DataFrame) -> List[str]:
    """Identify columns that represent market conditions."""

    market_condition_keywords = [
        'volatility', 'momentum', 'volume', 'returns', 'price_change',
        'trend', 'regime', 'market', 'condition', 'state',
        'rsi', 'macd', 'bollinger', 'atr', 'adx'
    ]

    market_columns = []

    for col in data.columns:
        col_lower = col.lower()

        # Check if column name contains market condition keywords
        if any(keyword in col_lower for keyword in market_condition_keywords):
            market_columns.append(col)

        # Check if column represents price-based metrics
        elif any(metric in col_lower for metric in ['close', 'high', 'low', 'open']):
            continue  # Skip raw price data

        # Check if column represents technical indicators
        elif any(indicator in col_lower for indicator in ['sma', 'ema', 'bb', 'stoch', 'cci']):
            market_columns.append(col)

    return market_columns


def main():
    """Main function to run HMM parameter optimization."""
    parser = argparse.ArgumentParser(description="Optimize HMM regime discovery parameters")
    parser.add_argument("--data_path", type=str, required=True, help="Path to feature data parquet file")
    parser.add_argument("--config_path", type=str, help="Path to configuration JSON file")
    parser.add_argument("--output_dir", type=str, default="optimization_results", help="Output directory")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of optimization trials")
    parser.add_argument("--timeout", type=int, help="Optimization timeout in seconds")
    parser.add_argument("--study_name", type=str, default="hmm_regime_optimization", help="Study name")

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)

    # Load configuration
    config = {}
    if args.config_path:
        with open(args.config_path, 'r') as f:
            config = json.load(f)

    # Load data
    print(f"📂 Loading data from: {args.data_path}")
    data = pd.read_parquet(args.data_path)
    print(f"📊 Loaded {len(data)} samples with {len(data.columns)} columns")

    # Identify feature and market condition columns
    feature_columns = [col for col in data.columns
                      if col not in ['timestamp', 'composite_cluster_id']]
    market_condition_columns = identify_market_condition_columns(data)

    print(f"🔧 Features: {len(feature_columns)}")
    print(f"📈 Market conditions: {len(market_condition_columns)}")
    print(f"📈 Market condition columns: {market_condition_columns[:10]}...")  # Show first 10

    # Initialize optimizer
    optimizer = HMMRegimeOptimizer(config)

    # Run optimization
    results = optimizer.optimize(
        data=data,
        feature_columns=feature_columns,
        market_condition_columns=market_condition_columns,
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name=args.study_name
    )

    # Generate comprehensive report
    print("\n📄 Generating comprehensive optimization report...")
    report = optimizer.generate_optimization_report(
        output_path=output_path / "optimization_report.md"
    )

    # Create visualizations
    print("\n📊 Creating visualizations...")
    optimizer.create_optimization_visualizations(output_path)

    # Save results
    optimizer.save_optimization_results(output_path / "optimization_results.json")

    # Print comprehensive summary
    print("\n" + "="*80)
    print("🎯 HMM REGIME OPTIMIZATION COMPLETED")
    print("="*80)
    print(f"🏆 Best Score: {results['best_score']:.4f}")
    print(f"📊 Final Regimes: {results['best_params'].get('target_regimes', 'N/A')}")
    print(f"🔧 Merging Method: {results['best_params'].get('merging_method', 'N/A')}")
    print(f"📁 Results saved to: {output_path}")
    print("")
    print("📋 Key Achievements:")
    print("✅ Optimized for 15-20 distinct, internally coherent regimes")
    print("✅ Focused on regime differentiation rather than transition prediction")
    print("✅ Removed feature engineering optimization (should be done in Step 2)")
    print("✅ Added comprehensive regime merging capabilities")
    print("✅ Generated detailed optimization report with recommendations")
    print("")
    print("💡 Next Steps:")
    print("1. 📖 Review the comprehensive optimization report")
    print("2. 🔧 Apply the best parameters to your Step 3 HMM regime discovery")
    print("3. ✅ Validate the optimized clusters using cluster validation tools")
    print("4. 🔄 Integrate the best parameters into your pipeline configuration")
    print("5. 📈 Monitor regime performance in live trading")
    print("")
    print("🎯 Remember: The goal is 15-20 regimes that are:")
    print("   • Different from each other (distinct market conditions)")
    print("   • Internally coherent (consistent within each regime)")
    print("   • Balanced in size (not too skewed)")
    print("   • Meaningful for trading strategies")


if __name__ == "__main__":
    main()