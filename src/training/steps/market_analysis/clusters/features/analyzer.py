"""
Feature Analysis Module for Market Analysis Clustering.

This module provides feature analysis capabilities including:
- Feature importance and loading computation
- Retained feature explanations
- Correlation and multicollinearity diagnostics
- Feature stability reporting
- Comprehensive feature insights
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, f_regression, SelectKBest
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import matplotlib.pyplot as plt

# Import utility modules
from src.utils.tprint import (
    tprint,
    tprint_debug,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
)
from src.utils.common_operations import get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer
from src.utils.common_utilities import safe_dataframe_operation, validate_dataframe_columns, calculate_data_quality_metrics
from src.utils.math_validation import safe_divide, validate_finite, validate_positive, validate_range
from src.utils.matrix_operations import get_unified_matrix_operations, safe_matrix_multiply, safe_correlation_matrix

# Import ML common utilities
try:
    from src.utils.ml_common.optimization.hpo_utils import HyperparameterOptimization
    from src.utils.ml_common.validation.unified_cv import perform_cross_validation as unified_perform_cv
    from src.utils.ml_common.evaluation.unified_evaluator import UnifiedEvaluator
from src.utils.hardware import (
    get_integrated_hardware_manager, 
    get_comprehensive_optimizer,
    memory_optimized, 
    comprehensive_memory_optimization,
    optimize_dataframe, 
    optimize_array,
    m1_optimized,
    WorkloadCategory,
    MemoryOptimizationLevel
)
    from src.feature_selection.core import get_feature_selection_framework
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False

# Hardware optimization imports
try:
    () if get_integrated_hardware_manager() else None
            self.memory_optimizer = get_integrated_hardware_manager() if get_integrated_hardware_manager() else None
            self.cpu_optimizer = get_comprehensive_optimizer() if get_comprehensive_optimizer() else None

            if self.gpu_manager or self.memory_optimizer or self.cpu_optimizer:
                tprint("✅ Hardware optimization initialized for feature analysis", "SUCCESS")

        except Exception as e:
            tprint(f"⚠️ Hardware optimization initialization failed: {e}", "WARNING")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None

    def analyze_features(
        self,
        features: np.ndarray,
        feature_names: List[str],
        target: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive feature analysis.

        Args:
            features: Feature matrix (n_samples, n_features)
            feature_names: List of feature names
            target: Optional target variable for supervised analysis
            metadata: Optional metadata from preprocessing/feature selection

        Returns:
            Dictionary with comprehensive analysis results
        """
        tprint(f"🔍 FEATURE ANALYSIS: Analyzing {features.shape[1]} features, {features.shape[0]} samples", color="cyan", bold=True)

        try:
            # Validate inputs
            features = validate_finite(features, "features")

            # Step 1: Basic feature statistics
            basic_stats = self._compute_basic_statistics(features, feature_names)

            # Step 2: Feature importance analysis
            importance_results = self._compute_feature_importance(features, feature_names, target)

            # Step 3: Correlation analysis using matrix operations
            correlation_results = self._compute_correlation_analysis(features, feature_names)

            # Step 4: Multicollinearity analysis (VIF) using ML common utilities
            vif_results = self._compute_multicollinearity_analysis(features, feature_names)

            # Step 5: Feature stability analysis (if time series data)
            stability_results = self._compute_stability_analysis(features, feature_names)

            # Step 6: Feature clustering/dendrogram using matrix operations
            clustering_results = self._compute_feature_clustering(features, feature_names)

            # Step 7: Generate explanations
            explanations = self._generate_feature_explanations(
                importance_results, correlation_results, vif_results, feature_names
            )

            # Compile results
            self.analysis_results = {
                'basic_statistics': basic_stats,
                'importance_analysis': importance_results,
                'correlation_analysis': correlation_results,
                'multicollinearity_analysis': vif_results,
                'stability_analysis': stability_results,
                'feature_clustering': clustering_results,
                'feature_explanations': explanations,
                'top_features': self._get_top_features(importance_results, feature_names),
                'problematic_features': self._identify_problematic_features(
                    correlation_results, vif_results, feature_names
                ),
                'analysis_metadata': {
                    'n_features': features.shape[1],
                    'n_samples': features.shape[0],
                    'has_target': target is not None,
                    'analysis_timestamp': pd.Timestamp.now(),
                    'preprocessing_metadata': metadata or {}
                }
            }

            tprint(f"✅ Feature analysis completed for {features.shape[1]} features", "SUCCESS")

            return self.analysis_results

        except Exception as e:
            tprint(f"❌ Feature analysis failed: {e}", "ERROR")
            return {'error': str(e)}

    def _compute_basic_statistics(
        self,
        features: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Compute basic statistical measures for each feature."""
        tprint("🔍 Computing basic feature statistics...", "INFO")

        stats_dict = {}

        for i, (feature, name) in enumerate(zip(features.T, feature_names)):
            feature_stats = {
                'name': name,
                'index': i,
                'mean': float(np.mean(feature)),
                'std': float(np.std(feature)),
                'min': float(np.min(feature)),
                'max': float(np.max(feature)),
                'median': float(np.median(feature)),
                'skewness': float(stats.skew(feature)),
                'kurtosis': float(stats.kurtosis(feature)),
                'missing_ratio': float(np.isnan(feature).sum() / len(feature)),
                'variance': float(np.var(feature)),
                'range': float(np.max(feature) - np.min(feature))
            }

            stats_dict[name] = feature_stats

        tprint(f"   • Computed statistics for {len(feature_names)} features", "SUCCESS")

        return stats_dict

    def _compute_feature_importance(
        self,
        features: np.ndarray,
        feature_names: List[str],
        target: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compute feature importance using multiple methods."""
        tprint("🔍 Computing feature importance...", "INFO")

        importance_scores = {}

        # PCA-based importance
        if "pca" in self.config.importance_methods:
            pca_importance = self._compute_pca_importance(features, feature_names)
            importance_scores['pca'] = pca_importance

        # Correlation-based importance (if target available)
        if "correlation" in self.config.importance_methods and target is not None:
            corr_importance = self._compute_correlation_importance(features, feature_names, target)
            importance_scores['correlation'] = corr_importance

        # Mutual information importance (if target available)
        if "mutual_info" in self.config.importance_methods and target is not None:
            mi_importance = self._compute_mutual_info_importance(features, feature_names, target)
            importance_scores['mutual_info'] = mi_importance

        # F-statistic importance (if target available)
        if "f_statistic" in self.config.importance_methods and target is not None:
            f_importance = self._compute_f_statistic_importance(features, feature_names, target)
            importance_scores['f_statistic'] = f_importance

        # Ensemble importance (average of all methods)
        ensemble_importance = self._compute_ensemble_importance(importance_scores, feature_names)
        importance_scores['ensemble'] = ensemble_importance

        tprint(f"   • Computed importance using {len(importance_scores)} methods", "SUCCESS")

        return importance_scores

    def _compute_pca_importance(self, features: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Compute feature importance using PCA loadings."""
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Apply PCA
        pca = PCA(n_components=min(10, features_scaled.shape[1]))
        pca.fit(features_scaled)

        # Compute loading scores (sum of absolute loadings across components)
        loadings = np.abs(pca.components_)
        importance_scores = np.sum(loadings, axis=0)

        # Normalize to [0, 1]
        if importance_scores.max() > 0:
            importance_scores = importance_scores / importance_scores.max()

        return {name: float(score) for name, score in zip(feature_names, importance_scores)}

    def _compute_correlation_importance(
        self,
        features: np.ndarray,
        feature_names: List[str],
        target: np.ndarray
    ) -> Dict[str, float]:
        """Compute feature importance using correlation with target."""
        importance_scores = {}

        for i, (feature, name) in enumerate(zip(features.T, feature_names)):
            if np.std(feature) > 0 and np.std(target) > 0:
                corr_coef = np.abs(np.corrcoef(feature, target)[0, 1])
                importance_scores[name] = float(corr_coef)
            else:
                importance_scores[name] = 0.0

        return importance_scores

    def _compute_mutual_info_importance(
        self,
        features: np.ndarray,
        feature_names: List[str],
        target: np.ndarray
    ) -> Dict[str, float]:
        """Compute feature importance using mutual information."""
        importance_scores = {}

        for i, (feature, name) in enumerate(zip(features.T, feature_names)):
            try:
                # Reshape for sklearn
                feature_reshaped = feature.reshape(-1, 1)
                mi_score = mutual_info_regression(feature_reshaped, target, random_state=42)[0]
                importance_scores[name] = float(mi_score)
            except Exception:
                importance_scores[name] = 0.0

        return importance_scores

    def _compute_f_statistic_importance(
        self,
        features: np.ndarray,
        feature_names: List[str],
        target: np.ndarray
    ) -> Dict[str, float]:
        """Compute feature importance using F-statistic."""
        importance_scores = {}

        for i, (feature, name) in enumerate(zip(features.T, feature_names)):
            try:
                # Reshape for sklearn
                feature_reshaped = feature.reshape(-1, 1)
                f_score = f_regression(feature_reshaped, target)[0][0]
                importance_scores[name] = float(f_score)
            except Exception:
                importance_scores[name] = 0.0

        return importance_scores

    def _compute_ensemble_importance(
        self,
        importance_scores: Dict[str, Dict[str, float]],
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Compute ensemble importance by averaging across methods."""
        ensemble_scores = {}

        for name in feature_names:
            scores = []
            for method_scores in importance_scores.values():
                if name in method_scores:
                    scores.append(method_scores[name])

            if scores:
                ensemble_scores[name] = float(np.mean(scores))
            else:
                ensemble_scores[name] = 0.0

        return ensemble_scores

    def _compute_correlation_analysis(
        self,
        features: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Compute correlation analysis between features."""
        tprint("🔍 Computing correlation analysis...", "INFO")

        # Validate inputs
        features = validate_finite(features, "features")
        correlation_threshold = validate_positive(self.config.correlation_threshold, "correlation_threshold")

        # Compute correlation matrix using matrix operations for efficiency
        try:
            correlation_result = self.matrix_ops.safe_correlation_matrix(features.T)
            correlation_matrix = correlation_result['correlation_matrix']
        except Exception as e:
            tprint(f"⚠️ Matrix operations correlation failed, using fallback: {e}", "WARNING")
            # Fallback to standard numpy correlation
            correlation_matrix = np.corrcoef(features.T)

        # Find highly correlated pairs using efficient operations
        try:
            # Use matrix operations to find correlated pairs efficiently
            abs_corr_matrix = np.abs(correlation_matrix)
            high_corr_mask = abs_corr_matrix >= correlation_threshold

            # Get indices of highly correlated pairs
            high_corr_indices = np.where(high_corr_mask)

            high_corr_pairs = []
            for i, j in zip(high_corr_indices[0], high_corr_indices[1]):
                if i < j:  # Only include each pair once (upper triangle)
                    corr_value = abs_corr_matrix[i, j]
                    high_corr_pairs.append({
                        'feature1': feature_names[i],
                        'feature2': feature_names[j],
                        'correlation': float(corr_value)
                    })

            # Sort by correlation strength using safe operations
            high_corr_pairs.sort(key=lambda x: x['correlation'], reverse=True)

        except Exception as e:
            tprint(f"⚠️ Efficient correlation pair finding failed, using fallback: {e}", "WARNING")
            # Fallback to standard nested loop approach
            high_corr_pairs = []
            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    corr_value = abs(correlation_matrix[i, j])
                    if corr_value >= correlation_threshold:
                        high_corr_pairs.append({
                            'feature1': feature_names[i],
                            'feature2': feature_names[j],
                            'correlation': float(corr_value)
                        })

        # Compute correlation statistics safely
        mean_correlation = safe_divide(np.mean(np.abs(correlation_matrix)), 1.0, 0.0)
        max_correlation = safe_divide(np.max(np.abs(correlation_matrix)), 1.0, 0.0)
        min_correlation = safe_divide(np.min(np.abs(correlation_matrix)), 1.0, 1.0)  # Min should be at least 0

        corr_stats = {
            'mean_correlation': float(mean_correlation),
            'max_correlation': float(max_correlation),
            'min_correlation': float(min_correlation),
            'high_correlation_pairs': high_corr_pairs[:50]  # Top 50 pairs
        }

        tprint(f"   • Found {len(high_corr_pairs)} highly correlated pairs (threshold: {correlation_threshold})", "INFO")

        return {
            'correlation_matrix': correlation_matrix,
            'high_correlation_pairs': high_corr_pairs,
            'correlation_statistics': corr_stats
        }

    def _compute_multicollinearity_analysis(
        self,
        features: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Compute multicollinearity analysis using VIF."""
        tprint("🔍 Computing multicollinearity analysis...", "INFO")

        # Validate inputs
        features = validate_finite(features, "features")
        multicollinearity_threshold = validate_positive(self.config.multicollinearity_threshold, "multicollinearity_threshold")

        vif_scores = {}

        # Standardize features first using matrix operations
        try:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
        except Exception as e:
            tprint(f"⚠️ StandardScaler failed, using manual standardization: {e}", "WARNING")
            # Manual standardization
            mean_vals = np.mean(features, axis=0)
            std_vals = np.std(features, axis=0)
            std_vals = np.where(std_vals == 0, 1e-8, std_vals)  # Avoid division by zero
            features_scaled = (features - mean_vals) / std_vals

        # Compute correlation matrix using matrix operations
        try:
            corr_result = self.matrix_ops.safe_correlation_matrix(features_scaled.T)
            corr_matrix = corr_result['correlation_matrix']
        except Exception:
            corr_matrix = np.corrcoef(features_scaled.T)

        # Compute VIF using the formula: VIF = 1 / (1 - R²)
        # This is an approximation - full VIF requires regression for each feature
        for i, name in enumerate(feature_names):
            try:
                # Get correlation of this feature with all others
                correlations = corr_matrix[i, np.arange(len(feature_names)) != i]
                r_squared = safe_divide(np.sum(correlations ** 2), 1.0, 0.0)

                if r_squared < 1.0:  # Avoid division by zero
                    vif = safe_divide(1.0, (1.0 - r_squared), float('inf'))
                else:
                    vif = float('inf')

                vif_scores[name] = float(vif)

            except Exception as e:
                tprint(f"⚠️ VIF calculation failed for feature {name}: {e}", "WARNING")
                vif_scores[name] = float('inf')

        # Identify problematic features (high VIF) using safe operations
        finite_vifs = [v for v in vif_scores.values() if v != float('inf')]
        problematic_features = [
            name for name, vif in vif_scores.items()
            if vif >= multicollinearity_threshold
        ]

        # Calculate statistics safely
        mean_vif = safe_divide(np.mean(finite_vifs), 1.0, 0.0) if finite_vifs else 0.0
        max_vif = safe_divide(max(finite_vifs), 1.0, 0.0) if finite_vifs else 0.0
        min_vif = safe_divide(min(vif_scores.values()), 1.0, float('inf'))

        tprint(f"   • Found {len(problematic_features)} features with high VIF (>= {multicollinearity_threshold})", "INFO")

        return {
            'vif_scores': vif_scores,
            'problematic_features': problematic_features,
            'vif_statistics': {
                'mean_vif': float(mean_vif),
                'max_vif': float(max_vif),
                'min_vif': float(min_vif),
                'threshold': multicollinearity_threshold
            }
        }

    def _compute_stability_analysis(
        self,
        features: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Compute feature stability over time windows."""
        tprint("🔍 Computing feature stability analysis...", "INFO")

        # Validate inputs
        features = validate_finite(features, "features")
        stability_window = validate_positive(self.config.stability_window, "stability_window")
        stability_threshold = validate_positive(self.config.stability_threshold, "stability_threshold")

        if features.shape[0] < 2 * stability_window:
            tprint("   • Insufficient data for stability analysis", "WARNING")
            return {'error': 'Insufficient data for stability analysis'}

        stability_scores = {}

        # Use matrix operations for efficient window processing
        try:
            # Calculate windows using matrix operations for efficiency
            n_samples = features.shape[0]
            step_size = stability_window // 2

            for i, (feature, name) in enumerate(zip(features.T, feature_names)):
                try:
                    # Split into windows using efficient array slicing
                    windows = []
                    for start_idx in range(0, n_samples - stability_window + 1, step_size):
                        end_idx = min(start_idx + stability_window, n_samples)
                        window = feature[start_idx:end_idx]
                        if len(window) == stability_window:  # Only include complete windows
                            windows.append(window)

                    if len(windows) >= 2:
                        # Compute stability as inverse of coefficient of variation of window statistics
                        window_means = np.array([np.mean(window) for window in windows])
                        window_stds = np.array([np.std(window) for window in windows])

                        # Avoid division by zero using safe operations
                        mean_of_means = safe_divide(np.mean(window_means), 1.0, 0.0)
                        mean_of_stds = safe_divide(np.mean(window_stds), 1.0, 0.0)

                        cv_means = safe_divide(np.std(window_means), (mean_of_means + 1e-8), 0.0)
                        cv_stds = safe_divide(np.std(window_stds), (mean_of_stds + 1e-8), 0.0)

                        # Stability score is inverse of coefficient of variation
                        stability_score = safe_divide(1.0, (1.0 + cv_means + cv_stds), 0.0)
                        stability_scores[name] = float(stability_score)
                    else:
                        stability_scores[name] = 0.0

                except Exception as e:
                    tprint(f"⚠️ Stability calculation failed for feature {name}: {e}", "WARNING")
                    stability_scores[name] = 0.0

        except Exception as e:
            tprint(f"⚠️ Matrix operations stability analysis failed, using fallback: {e}", "WARNING")
            # Fallback to standard approach
            for i, (feature, name) in enumerate(zip(features.T, feature_names)):
                windows = []
                for start_idx in range(0, features.shape[0] - stability_window, stability_window // 2):
                    end_idx = min(start_idx + stability_window, features.shape[0])
                    windows.append(feature[start_idx:end_idx])

                if len(windows) >= 2:
                    window_means = [np.mean(window) for window in windows]
                    window_stds = [np.std(window) for window in windows]
                    cv_means = safe_divide(np.std(window_means), (np.mean(window_means) + 1e-8), 0.0)
                    cv_stds = safe_divide(np.std(window_stds), (np.mean(window_stds) + 1e-8), 0.0)
                    stability_score = safe_divide(1.0, (1.0 + cv_means + cv_stds), 0.0)
                    stability_scores[name] = float(stability_score)
                else:
                    stability_scores[name] = 0.0

        # Identify unstable features using safe operations
        unstable_features = [
            name for name, stability in stability_scores.items()
            if stability < stability_threshold
        ]

        # Calculate statistics safely
        stability_values = list(stability_scores.values())
        mean_stability = safe_divide(np.mean(stability_values), 1.0, 0.0)
        min_stability = safe_divide(min(stability_values), 1.0, 1.0)
        max_stability = safe_divide(max(stability_values), 1.0, 0.0)

        tprint(f"   • Found {len(unstable_features)} unstable features (threshold: {stability_threshold})", "INFO")

        return {
            'stability_scores': stability_scores,
            'unstable_features': unstable_features,
            'stability_statistics': {
                'mean_stability': float(mean_stability),
                'min_stability': float(min_stability),
                'max_stability': float(max_stability),
                'threshold': stability_threshold
            }
        }

    def _compute_feature_clustering(
        self,
        features: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Compute hierarchical clustering of features."""
        tprint("🔍 Computing feature clustering...", "INFO")

        # Validate inputs
        features = validate_finite(features, "features")

        # Use correlation-based distance with matrix operations
        try:
            correlation_result = self.matrix_ops.safe_correlation_matrix(features.T)
            correlation_matrix = correlation_result['correlation_matrix']
            distance_matrix = 1 - np.abs(correlation_matrix)
        except Exception as e:
            tprint(f"⚠️ Matrix operations clustering failed, using fallback: {e}", "WARNING")
            # Fallback to standard correlation
            correlation_matrix = np.corrcoef(features.T)
            distance_matrix = 1 - np.abs(correlation_matrix)

        # Perform hierarchical clustering using scipy
        try:
            linkage_matrix = linkage(distance_matrix, method='average')
        except Exception as e:
            tprint(f"⚠️ Hierarchical clustering failed: {e}", "WARNING")
            # Return empty linkage matrix as fallback
            linkage_matrix = np.array([])

        return {
            'linkage_matrix': linkage_matrix,
            'distance_matrix': distance_matrix,
            'correlation_matrix': correlation_matrix
        }

    def _generate_feature_explanations(
        self,
        importance_results: Dict[str, Any],
        correlation_results: Dict[str, Any],
        vif_results: Dict[str, Any],
        feature_names: List[str]
    ) -> Dict[str, str]:
        """Generate explanations for features."""
        tprint("🔍 Generating feature explanations...", "INFO")

        explanations = {}

        for name in feature_names:
            explanation_parts = []

            # Importance explanation
            if 'ensemble' in importance_results and name in importance_results['ensemble']:
                importance = importance_results['ensemble'][name]
                if importance > 0.5:
                    explanation_parts.append(f"High importance feature (score: {importance:.3f})")
                elif importance > 0.2:
                    explanation_parts.append(f"Medium importance feature (score: {importance:.3f})")
                else:
                    explanation_parts.append(f"Low importance feature (score: {importance:.3f})")

            # Correlation explanation
            high_corr_count = sum(
                1 for pair in correlation_results['high_correlation_pairs']
                if pair['feature1'] == name or pair['feature2'] == name
            )
            if high_corr_count > 5:
                explanation_parts.append(f"Highly correlated with {high_corr_count} other features")
            elif high_corr_count > 0:
                explanation_parts.append(f"Correlated with {high_corr_count} other features")

            # Multicollinearity explanation
            if name in vif_results['problematic_features']:
                vif_score = vif_results['vif_scores'][name]
                explanation_parts.append(f"High multicollinearity (VIF: {vif_score:.1f})")

            # Combine explanations
            if explanation_parts:
                explanations[name] = "; ".join(explanation_parts)
            else:
                explanations[name] = "Standard feature with no notable characteristics"

        tprint(f"   • Generated explanations for {len(explanations)} features", "SUCCESS")

        return explanations

    def _get_top_features(
        self,
        importance_results: Dict[str, Any],
        feature_names: List[str]
    ) -> List[str]:
        """Get top K most important features."""
        if 'ensemble' not in importance_results:
            return feature_names[:self.config.top_k_features]

        ensemble_scores = importance_results['ensemble']
        sorted_features = sorted(
            ensemble_scores.keys(),
            key=lambda x: ensemble_scores[x],
            reverse=True
        )

        return sorted_features[:self.config.top_k_features]

    def _identify_problematic_features(
        self,
        correlation_results: Dict[str, Any],
        vif_results: Dict[str, Any],
        feature_names: List[str]
    ) -> List[str]:
        """Identify features with potential problems."""
        problematic = set()

        # High correlation features
        correlated_features = set()
        for pair in correlation_results['high_correlation_pairs']:
            correlated_features.add(pair['feature1'])
            correlated_features.add(pair['feature2'])

        # High VIF features
        high_vif_features = set(vif_results['problematic_features'])

        # Combine all problematic features
        problematic.update(correlated_features)
        problematic.update(high_vif_features)

        return list(problematic)

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate a comprehensive feature analysis report."""
        if not self.analysis_results:
            return "No analysis results available. Run analyze_features() first."

        report_lines = [
            "# Feature Analysis Report",
            f"Generated: {self.analysis_results['analysis_metadata']['analysis_timestamp']}",
            "",
            "## Summary",
            f"- **Features Analyzed**: {self.analysis_results['analysis_metadata']['n_features']}",
            f"- **Samples**: {self.analysis_results['analysis_metadata']['n_samples']}",
            f"- **Has Target**: {self.analysis_results['analysis_metadata']['has_target']}",
            "",
            "## Top Features",
        ]

        top_features = self.analysis_results['top_features']
        for i, feature in enumerate(top_features[:10], 1):
            explanation = self.analysis_results['feature_explanations'].get(feature, 'No explanation available')
            report_lines.append(f"{i}. **{feature}** - {explanation}")

        report_lines.extend([
            "",
            "## Problematic Features",
        ])

        problematic_features = self.analysis_results['problematic_features']
        for feature in problematic_features[:10]:
            report_lines.append(f"- {feature}")

        if output_path:
            with open(output_path, 'w') as f:
                f.write('\n'.join(report_lines))
            tprint(f"✅ Report saved to {output_path}", "SUCCESS")

        return '\n'.join(report_lines)

    def get_analysis_results(self) -> Dict[str, Any]:
        """Get the complete analysis results."""
        return self.analysis_results.copy()
