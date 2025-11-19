"""
Integration wrapper for Improved mRMR with MI Proxy in Final Feature Selection

This module provides seamless integration of the improved_mrmr module with MI proxy
optimization into the final_feature_selection component.
"""

import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_debug, tprint_info
from src.feature_selection.advanced.improved_mrmr import ImprovedMRMR, create_improved_mrmr

logger = logging.getLogger(__name__)


class ImprovedMRMRIntegration:
    """
    Integration wrapper for improved_mrmr with MI proxy into final_feature_selection.

    Features:
    - Seamless integration with final_feature_selection component
    - MI proxy optimization for faster feature selection
    - Per-feature CSV export with detailed metrics
    - Cross-validation based MI stability analysis
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize improved_mrmr integration.

        Args:
            config: Configuration dictionary with:
                - use_mi_proxy: Enable MI proxy (default: True)
                - target_ratio: Ratio of features to select (default: 0.5)
                - export_per_feature_csv: Export CSV results (default: False)
                - output_dir: Directory for CSV exports (default: None)
        """
        self.config = config or {}
        self.logger = logger.getChild('ImprovedMRMRIntegration')

        # Initialize improved_mrmr selector
        mrmr_config = {
            'mi_weight': self.config.get('mi_weight', 0.7),
            'spearman_weight': self.config.get('spearman_weight', 0.3),
            'target_ratio': self.config.get('target_ratio', 0.5),
            'quantile_bins': self.config.get('quantile_bins', 10),
            'use_mi_proxy': self.config.get('use_mi_proxy', True),
            'cv_folds': self.config.get('cv_folds', 5),
            'export_per_feature_csv': self.config.get('export_per_feature_csv', False),
            'enable_hardware_optimization': self.config.get('enable_hardware_optimization', True),
            'verbose': True
        }

        self.mrmr = create_improved_mrmr(mrmr_config)
        self.output_dir = self.config.get('output_dir', None)

        # Results tracking
        self.last_result = None
        self.per_feature_scores = None

        tprint_success("✅ ImprovedMRMRIntegration initialized")

    def select_features_from_pool(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str],
        target_count: int,
        stability_analysis: Optional[Dict[str, Any]] = None,
        mi_analysis: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Select features using improved_mrmr with MI proxy and optional stability/MI analysis.

        Args:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
            target_count: Target number of features to select
            stability_analysis: Optional stability analysis results from final_feature_selection
            mi_analysis: Optional MI analysis results from final_feature_selection

        Returns:
            Tuple of (selected_features, detailed_results)
        """
        tprint_info(f"🎯 Selecting {target_count} features from {len(feature_names)} using improved_mrmr+MI proxy")

        start_time = time.time()

        try:
            # Filter X to only include feature_names
            X_filtered = X[feature_names].copy()

            # Convert to numpy
            X_np = X_filtered.values.astype(np.float32)
            y_np = y.values.astype(np.float32)

            # Handle NaN values
            nan_mask = np.isnan(X_np).any(axis=1) | np.isnan(y_np)
            if nan_mask.any():
                tprint_warning(f"⚠️ Removing {nan_mask.sum()} samples with NaN values")
                X_np = X_np[~nan_mask]
                y_np = y_np[~nan_mask]

            # Validate dimensions
            if X_np.shape[0] == 0:
                tprint_warning("⚠️ No valid samples after NaN removal")
                return feature_names[:target_count], {'success': False, 'error': 'No valid samples'}

            # Calculate target ratio based on desired feature count
            target_ratio = min(target_count / len(feature_names), 1.0)

            # Perform feature selection using improved_mrmr
            result = self.mrmr.select_features(
                X_np, y_np,
                feature_names=feature_names,
                target_ratio=target_ratio
            )

            if not result.get('success', False):
                tprint_warning(f"⚠️ mRMR selection failed: {result.get('error', 'Unknown error')}")
                return feature_names[:target_count], result

            # Get selected features
            selected_features = result.get('selected_features', [])

            # Ensure we have exactly target_count features (or close to it)
            if len(selected_features) > target_count:
                # Take top features by relevance score
                relevance = result.get('relevance_scores', {})
                sorted_features = sorted(
                    selected_features,
                    key=lambda f: relevance.get(f, 0.0),
                    reverse=True
                )
                selected_features = sorted_features[:target_count]

            # Enhance results with stability and MI analysis if available
            enhanced_result = {
                **result,
                'selected_features': selected_features,
                'n_selected': len(selected_features),
                'selection_time': time.time() - start_time,
                'method': 'improved_mrmr_with_mi_proxy'
            }

            # Add stability metrics if available
            if stability_analysis and isinstance(stability_analysis, dict):
                stable_features = stability_analysis.get('stable_features', [])
                # Filter selected features by stability
                stable_selected = [f for f in selected_features if f in stable_features]
                enhanced_result['stable_features_count'] = len(stable_selected)
                enhanced_result['stability_coverage'] = (
                    len(stable_selected) / len(selected_features)
                    if selected_features else 0.0
                )
                tprint_info(f"📊 Stability coverage: {enhanced_result['stability_coverage']:.1%}")

            # Add MI analysis if available
            if mi_analysis and isinstance(mi_analysis, dict):
                enhanced_result['mi_analysis'] = mi_analysis

            # Store for later retrieval
            self.last_result = enhanced_result
            self.per_feature_scores = result.get('relevance_scores', {})

            # Export per-feature CSV if requested
            if self.config.get('export_per_feature_csv', False):
                csv_path = self._export_per_feature_csv_detailed(
                    feature_names, result.get('relevance_scores', {}),
                    result.get('selected_indices', []),
                    stability_analysis, mi_analysis
                )
                if csv_path:
                    enhanced_result['per_feature_csv_path'] = csv_path

            tprint_success(f"✅ Selected {len(selected_features)} features using improved_mrmr in {time.time() - start_time:.3f}s")

            return selected_features, enhanced_result

        except Exception as e:
            self.logger.error(f"❌ Feature selection failed: {e}")
            import traceback
            traceback.print_exc()
            return feature_names[:target_count], {'success': False, 'error': str(e)}

    def calculate_mi_stability(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str],
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Calculate MI stability using cross-validation with MI proxy.

        Args:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
            cv_folds: Number of CV folds

        Returns:
            Dictionary with MI stability metrics
        """
        tprint_debug(f"🔧 Computing MI stability for {len(feature_names)} features")

        try:
            X_filtered = X[feature_names].copy()
            X_np = X_filtered.values.astype(np.float32)
            y_np = y.values.astype(np.float32)

            # Handle NaN values
            nan_mask = np.isnan(X_np).any(axis=1) | np.isnan(y_np)
            if nan_mask.any():
                X_np = X_np[~nan_mask]
                y_np = y_np[~nan_mask]

            if X_np.shape[0] < 10:
                tprint_warning("⚠️ Insufficient samples for MI stability analysis")
                return {'error': 'Insufficient samples'}

            # Use MI proxy for stable MI calculation
            mi_scores_folds = []
            fold_size = X_np.shape[0] // cv_folds

            for fold in range(cv_folds):
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size if fold < cv_folds - 1 else X_np.shape[0]

                X_fold = X_np[start_idx:end_idx]
                y_fold = y_np[start_idx:end_idx]

                # Use MI proxy
                mi_result = self.mrmr.mi_proxy.compute_mi_target(
                    X_fold, y_fold, feature_names
                ) if self.mrmr.mi_proxy else {}

                mi_scores_folds.append(mi_result)

            # Calculate MI statistics across folds
            mi_mean = {}
            mi_std = {}
            mi_cv = {}

            for feat in feature_names:
                scores = [mi_scores_folds[fold].get(feat, 0.0) for fold in range(cv_folds)]
                scores_arr = np.array(scores)
                mi_mean[feat] = float(np.mean(scores_arr))
                mi_std[feat] = float(np.std(scores_arr))
                mi_cv[feat] = float(np.std(scores_arr) / (np.mean(scores_arr) + 1e-8))

            return {
                'mi_mean': mi_mean,
                'mi_std': mi_std,
                'mi_cv': mi_cv,
                'cv_folds': cv_folds
            }

        except Exception as e:
            self.logger.error(f"MI stability calculation failed: {e}")
            return {'error': str(e)}

    def _export_per_feature_csv_detailed(
        self,
        feature_names: List[str],
        relevance_scores: Dict[str, float],
        selected_indices: List[int],
        stability_analysis: Optional[Dict[str, Any]] = None,
        mi_analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Export detailed per-feature analysis to CSV.

        Args:
            feature_names: List of feature names
            relevance_scores: Dictionary of relevance scores
            selected_indices: Indices of selected features
            stability_analysis: Optional stability metrics
            mi_analysis: Optional MI metrics

        Returns:
            Path to saved CSV file
        """
        try:
            # Create output directory
            output_dir = Path(self.output_dir or Path.cwd() / "mrmr_results")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Prepare data
            selected_set = set(selected_indices)
            data = []

            for i, feature_name in enumerate(feature_names):
                relevance_key = f"feature_{i}"
                row = {
                    'feature_index': i,
                    'feature_name': feature_name,
                    'relevance_score': float(relevance_scores.get(relevance_key, 0.0)),
                    'is_selected': i in selected_set
                }

                # Add stability metrics if available
                if stability_analysis and isinstance(stability_analysis, dict):
                    stable_features = stability_analysis.get('stable_features', [])
                    row['is_stable'] = feature_name in stable_features

                # Add MI metrics if available
                if mi_analysis and isinstance(mi_analysis, dict):
                    mi_mean = mi_analysis.get('mi_mean', {})
                    mi_cv = mi_analysis.get('mi_cv', {})
                    row['mi_mean'] = float(mi_mean.get(feature_name, 0.0))
                    row['mi_cv'] = float(mi_cv.get(feature_name, np.inf))

                data.append(row)

            # Create DataFrame and save
            df = pd.DataFrame(data)
            df = df.sort_values('relevance_score', ascending=False).reset_index(drop=True)

            csv_file = output_dir / "per_feature_improved_mrmr_scores.csv"
            df.to_csv(csv_file, index=False)

            tprint_info(f"📊 Per-feature analysis exported to: {csv_file}")

            return str(csv_file)

        except Exception as e:
            self.logger.error(f"Failed to export per-feature CSV: {e}")
            return ""

    def get_mi_proxy_stats(self) -> Dict[str, Any]:
        """Get MI proxy performance statistics."""
        if self.mrmr.mi_proxy:
            return self.mrmr.mi_proxy.get_performance_stats()
        return {}

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics."""
        return {
            **self.mrmr.get_performance_stats(),
            'mi_proxy_stats': self.get_mi_proxy_stats(),
            'last_result': self.last_result
        }


def create_improved_mrmr_integration(config: Optional[Dict[str, Any]] = None) -> ImprovedMRMRIntegration:
    """Factory function to create ImprovedMRMRIntegration instance."""
    return ImprovedMRMRIntegration(config)
