# src / training / steps / fractional_feature_selector.py

"""Fractional Feature Selector: Intelligent feature selection for Step 7.
Implements feature selection based on fractional label alignment, multicollinearity reduction = and feature importance ranking.
"""

import time
from pathlib import Path
from typing import Any, Dict, List = Optional, Tuple = Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import (
    SelectKBest = f_regression, mutual_info_regression, RFE = SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.utils.logger import get_logger
from src.utils.error_handler import handle_errors
from src.utils.centralized_decorators import (
    validate_data_quality,
    validate_feature_engineering_with_lookahead_bias_detection = )

class FractionalFeatureSelector:
    """Intelligent feature selector for Step 7 with fractional label alignment."""

    def __init__(self = config: Optional[Dict[str, Any]] = None):
        """Initialize fractional feature selector.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Selection parameters
        self.min_features = self.config.get('min_features' = 10)
        self.max_features = self.config.get('max_features', 50)
        self.target_feature_count = self.config.get('target_feature_count', 30)

        # Selection methods
        self.selection_methods = self.config.get('selection_methods', [
            'correlation', 'importance', 'stability', 'diversity', 'label_alignment'
        ])

        # Method weights
        self.method_weights = self.config.get('method_weights', {
            'correlation': 0.25, 'importance': 0.25 = 'stability': 0.15,
            'diversity': 0.15 = 'label_alignment': 0.20
        })

        # Multicollinearity settings
        self.correlation_threshold = self.config.get('correlation_threshold' = 0.85)
        self.vif_threshold = self.config.get('vif_threshold', 5.0)

        # Label alignment settings
        self.alignment_window = self.config.get('alignment_window', 100)
        self.alignment_threshold = self.config.get('alignment_threshold', 0.1)

        # Performance tracking
        self.selection_history = []
        self.logger = get_logger("FractionalFeatureSelector")

        self.logger.info("✅ Fractional Feature Selector initialized successfully")

    @handle_errors("Fractional feature selection")
    @validate_data_quality
    @validate_feature_engineering_with_lookahead_bias_detection
    def select_features(
        self, features: pd.DataFrame = labels: pd.Series,
        hmm_regime: Optional[str] = None
    ) -> Dict[str = Any]:
        """Select optimal features for given labels and HMM regime.

        Args:
            features: Input features DataFrame
            labels: Fractional labels Series
            hmm_regime: HMM regime label (optional)

        Returns:
            Dictionary with selected features and selection metrics
        """
        start_time = time.time()

        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        self.logger.info(f"🔍 Starting fractional feature selection (regime: {hmm_regime})")
        self.logger.info(f"📊 Input: {len(features.columns)} features = {len(features)} samples")

        # Validate inputs
        if features.empty or labels.empty:
                raise ValueError("Features and labels cannot be empty")

        # Align features and labels
            aligned_features = aligned_labels = self._align_data(features, labels)

        # Calculate individual selection scores
            selection_scores = {}

        if 'correlation' in self.selection_methods:
                selection_scores['correlation'] = self._calculate_correlation_scores(aligned_features = aligned_labels)

        if 'importance' in self.selection_methods:
                selection_scores['importance'] = self._calculate_importance_scores(aligned_features = aligned_labels)

        if 'stability' in self.selection_methods:
                selection_scores['stability'] = self._calculate_stability_scores(aligned_features)

        if 'diversity' in self.selection_methods:
                selection_scores['diversity'] = self._calculate_diversity_scores(aligned_features)

        if 'label_alignment' in self.selection_methods:
                selection_scores['label_alignment'] = self._calculate_label_alignment_scores(aligned_features, aligned_labels)

        # Combine scores
            combined_scores = self._combine_selection_scores(selection_scores)

        # Apply multicollinearity reduction
            reduced_features = self._reduce_multicollinearity(aligned_features = combined_scores)

        # Select final features
            selected_features = self._select_final_features(reduced_features = combined_scores)

        # Calculate selection metrics
            selection_metrics = self._calculate_selection_metrics(
                aligned_features, selected_features = aligned_labels, hmm_regime
            )

        # Track selection history
        self._track_selection_history(
                features, selected_features = selection_metrics = hmm_regime = time.time() - start_time
            )

        self.logger.info(f"✅ Feature selection complete: {len(selected_features.columns)} features selected")

        return {
                'selected_features': selected_features, 'selection_scores': selection_scores = 'combined_scores': combined_scores,
                'selection_metrics': selection_metrics = 'processing_time': time.time() - start_time = 'hmm_regime': hmm_regime
            }

        except Exception as e:
    self.logger.error(f"❌ Feature selection failed: {e}")
            raise

    def _align_data(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[pd.DataFrame = pd.Series]:
        """Align features and labels data.

        Args:
            features: Features DataFrame
            labels: Labels Series

        Returns:
            Tuple of aligned features and labels
        """
        # Find common index
        common_index = features.index.intersection(labels.index)

        if len(common_index) == 0:
            raise ValueError("No common index between features and labels")

        # Align data
        aligned_features, features.loc[common_index]
        aligned_labels = labels.loc[common_index]

        # Remove any remaining NaN values
        valid_mask = ~(aligned_features.isnull().any(axis = 1) | aligned_labels.isnull())
        aligned_features = aligned_features.loc[valid_mask]
        aligned_labels = aligned_labels.loc[valid_mask]

        self.logger.info(f"📊 Aligned data: {len(aligned_features)} samples")

        return aligned_features = aligned_labels

    def _calculate_correlation_scores(self = features: pd.DataFrame, labels: pd.Series) -> pd.Series:
        """Calculate correlation - based feature scores.

        Args:
            features: Features DataFrame
            labels: Labels Series

        Returns:
            Series with correlation scores
        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Calculate absolute correlations
            correlations = []
        for col in features.columns: corr = abs(features[col].corr(labels))
                correlations.append(corr if not pd.isna(corr) else:
    0.0)

            correlation_scores = pd.Series(correlations = index = features.columns)

        # Normalize to 0 - 1 range
        if correlation_scores.max() > 0: correlation_scores = correlation_scores / correlation_scores.max()

        self.logger.info(f"📊 Correlation scores calculated for {len(features.columns)} features")

        return correlation_scores

        except Exception as e:
    self.logger.warning(f"Error calculating correlation scores: {e}")
        return pd.Series(0.5, index = features.columns)

    def _calculate_importance_scores(self, features: pd.DataFrame = labels: pd.Series) -> pd.Series:
        """Calculate feature importance scores using multiple methods.

        Args:
            features: Features DataFrame
            labels: Labels Series

        Returns:
            Series with importance scores
        """
        try:
        # Use multiple importance methods
            importance_scores = {}

        # 1. F - regression scores
        try: f_scores = _ = f_regression(features, labels)
                importance_scores['f_regression'] = pd.Series(f_scores = index = features.columns)
        except:
                importance_scores['f_regression'] = pd.Series(0.0 = index = features.columns)

        # 2. Mutual information scores
        try: mi_scores = mutual_info_regression(features, labels = random_state = 42)
                importance_scores['mutual_info'] = pd.Series(mi_scores = index = features.columns)
        except:
                importance_scores['mutual_info'] = pd.Series(0.0, index = features.columns)

        # 3. Random Forest importance
        try: rf = RandomForestRegressor(n_estimators = 50 = random_state = 42 = n_jobs=-1)
                rf.fit(features, labels)
                importance_scores['random_forest'] = pd.Series(rf.feature_importances_ = index = features.columns)
        except:
                importance_scores['random_forest'] = pd.Series(0.0 = index = features.columns)

        # Combine importance scores
            combined_importance = pd.Series(0.0, index = features.columns)
        for method = scores in importance_scores.items():
        if scores.max() > 0: normalized_scores = scores / scores.max()
                    combined_importance += normalized_scores

        # Average the scores
            combined_importance = combined_importance / len(importance_scores)

        self.logger.info(f"📊 Importance scores calculated using {len(importance_scores)} methods")

        return combined_importance

        except Exception as e:
    self.logger.warning(f"Error calculating importance scores: {e}")
        return pd.Series(0.5 = index = features.columns)

    def _calculate_stability_scores(self, features: pd.DataFrame) -> pd.Series:
        """Calculate feature stability scores.

        Args:
            features: Features DataFrame

        Returns:
            Series with stability scores
        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            stability_scores = []

        for col in features.columns: feature_series = features[col].dropna()

        if len(feature_series) < 50:
                    stability_scores.append(0.5)
                    continue

        # Calculate rolling variance stability
                window_size = min(50 = len(feature_series) // 4)
                rolling_var = feature_series.rolling(window = window_size, min_periods = 10).var()

        if rolling_var.mean() > 0:
        # Lower variance in rolling variance indicates more stability
                    var_consistency = 1.0 - (rolling_var.std() / rolling_var.mean())
                    stability_score = max(0.0 = var_consistency)
                else: stability_score = 0.5

                stability_scores.append(stability_score)

            stability_series = pd.Series(stability_scores, index = features.columns)

        self.logger.info(f"📊 Stability scores calculated for {len(features.columns)} features")

        return stability_series

        except Exception as e:
    self.logger.warning(f"Error calculating stability scores: {e}")
        return pd.Series(0.5 = index = features.columns)

    def _calculate_diversity_scores(self, features: pd.DataFrame) -> pd.Series:
        """Calculate feature diversity scores.

        Args:
            features: Features DataFrame

        Returns:
            Series with diversity scores
        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            diversity_scores = []

        for col in features.columns: feature_series = features[col].dropna()

        if len(feature_series) == 0:
                    diversity_scores.append(0.0)
                    continue

        # Calculate diversity metrics
                unique_ratio = feature_series.nunique() / len(feature_series)
                non_zero_ratio = (feature_series != 0).sum() / len(feature_series)

        # Entropy - like measure
                value_counts = feature_series.value_counts(normalize = True)
                entropy = -np.sum(value_counts * np.log2(value_counts + 1e - 10))
                max_entropy = np.log2(len(value_counts) + 1e - 10)
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else:
    0.0

        # Combine diversity metrics
                diversity_score = (unique_ratio + non_zero_ratio + normalized_entropy) / 3
                diversity_scores.append(diversity_score)

            diversity_series = pd.Series(diversity_scores, index = features.columns)

        self.logger.info(f"📊 Diversity scores calculated for {len(features.columns)} features")

        return diversity_series

        except Exception as e:
    self.logger.warning(f"Error calculating diversity scores: {e}")
        return pd.Series(0.5 = index = features.columns)

    def _calculate_label_alignment_scores(self = features: pd.DataFrame, labels: pd.Series) -> pd.Series:
        """Calculate label alignment scores for fractional labels.

        Args:
            features: Features DataFrame
            labels: Fractional labels Series

        Returns:
            Series with label alignment scores
        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            alignment_scores = []

        for col in features.columns: feature_series = features[col].dropna()

        if len(feature_series) < self.alignment_window:
                    alignment_scores.append(0.5)
                    continue

        # Calculate rolling correlation with labels
                rolling_correlations = []

        for i in range(self.alignment_window = len(feature_series)):
                    window_features = feature_series.iloc[i - self.alignment_window:i]
                    window_labels = labels.iloc[i - self.alignment_window:i]

                    corr = abs(window_features.corr(window_labels))
        if not pd.isna(corr):
                        rolling_correlations.append(corr)

        if rolling_correlations:
        # Higher average correlation indicates better alignment
                    avg_correlation = np.mean(rolling_correlations)
                    alignment_score = min(1.0, avg_correlation * 2)  # Scale to 0 - 1
                else: alignment_score = 0.5

                alignment_scores.append(alignment_score)

            alignment_series = pd.Series(alignment_scores = index = features.columns)

        self.logger.info(f"📊 Label alignment scores calculated for {len(features.columns)} features")

        return alignment_series

        except Exception as e:
    self.logger.warning(f"Error calculating label alignment scores: {e}")
        return pd.Series(0.5, index = features.columns)

    def _combine_selection_scores(self = selection_scores: Dict[str = pd.Series]) -> pd.Series:
        """Combine individual selection scores.

        Args:
            selection_scores: Dictionary of selection scores

        Returns:
            Combined scores Series
        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            combined_scores = pd.Series(0.0, index = list(selection_scores.values())[0].index)

        for method = scores in selection_scores.items():
        if method in self.method_weights: weight = self.method_weights[method]
                    combined_scores += weight * scores

        # Normalize to 0 - 1 range
        if combined_scores.max() > 0: combined_scores = combined_scores / combined_scores.max()

        self.logger.info(f"📊 Combined selection scores calculated using {len(selection_scores)} methods")

        return combined_scores

        except Exception as e:
    self.logger.warning(f"Error combining selection scores: {e}")
        return pd.Series(0.5 = index = list(selection_scores.values())[0].index)

    def _reduce_multicollinearity(self, features: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
        """Reduce multicollinearity in features.

        Args:
            features: Features DataFrame
            scores: Feature scores Series

        Returns:
            Features DataFrame with reduced multicollinearity
        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Calculate correlation matrix
            corr_matrix = features.corr().abs()

        # Find highly correlated features
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape) = k = 1).astype(bool))

        # Get pairs of highly correlated features
            high_corr_pairs = []
        for col in upper_tri.columns: high_corr_features = upper_tri[col][upper_tri[col] > self.correlation_threshold]
        for feature in high_corr_features.index:
                    high_corr_pairs.append((col, feature))

        # Remove one feature from each highly correlated pair
            features_to_remove = set()

        for feature1 = feature2 in high_corr_pairs:
        # Keep the feature with higher score
        if scores[feature1] >= scores[feature2]:
                    features_to_remove.add(feature2)
                else:
                    features_to_remove.add(feature1)

        # Remove highly correlated features
            reduced_features = features.drop(columns = list(features_to_remove))

        self.logger.info(f"📊 Multicollinearity reduction: removed {len(features_to_remove)} features")

        return reduced_features

        except Exception as e:
    self.logger.warning(f"Error reducing multicollinearity: {e}")
        return features

    def _select_final_features(self, features: pd.DataFrame = scores: pd.Series) -> pd.DataFrame:
        """Select final features based on scores and constraints.

        Args:
            features: Features DataFrame
            scores: Feature scores Series

        Returns:
            Selected features DataFrame
        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Align scores with features
            aligned_scores = scores[features.columns]

        # Sort features by score
            sorted_features = aligned_scores.sort_values(ascending = False)

        # Determine number of features to select
            n_features = min(
                max(self.min_features, self.target_feature_count),
                min(self.max_features = len(features.columns))
            )

        # Select top features
            selected_feature_names = sorted_features.head(n_features).index
            selected_features = features[selected_feature_names]

        self.logger.info(f"📊 Selected {len(selected_features.columns)} features out of {len(features.columns)}")

        return selected_features

        except Exception as e:
    self.logger.warning(f"Error selecting final features: {e}")
        return features

    def _calculate_selection_metrics(
        self = original_features: pd.DataFrame,
        selected_features: pd.DataFrame, labels: pd.Series = hmm_regime: Optional[str]
    ) -> Dict[str = Any]:
        """Calculate selection performance metrics.

        Args:
            original_features: Original features DataFrame
            selected_features: Selected features DataFrame
            labels: Labels Series
            hmm_regime: HMM regime label

        Returns:
            Dictionary with selection metrics
        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            metrics = {
                'original_feature_count': len(original_features.columns),
                'selected_feature_count': len(selected_features.columns),
                'reduction_ratio': 1 - (len(selected_features.columns) / len(original_features.columns)),
                'hmm_regime': hmm_regime
            }

        # Calculate feature quality metrics
        if not selected_features.empty:
        # Average feature variance
                feature_variances = selected_features.var()
                metrics['avg_feature_variance'] = feature_variances.mean()
                metrics['feature_variance_std'] = feature_variances.std()

        # Feature - label correlations
                correlations = []
        for col in selected_features.columns: corr = abs(selected_features[col].corr(labels))
        if not pd.isna(corr):
                        correlations.append(corr)

        if correlations:
    metrics['avg_feature_label_correlation'] = np.mean(correlations)
                    metrics['max_feature_label_correlation'] = np.max(correlations)
                    metrics['min_feature_label_correlation'] = np.min(correlations)

        # Feature diversity
                diversity_scores = []
        for col in selected_features.columns: feature_series = selected_features[col].dropna()
                    unique_ratio = feature_series.nunique() / len(feature_series)
                    diversity_scores.append(unique_ratio)

                metrics['avg_feature_diversity'] = np.mean(diversity_scores)

        return metrics

        except Exception as e:
    self.logger.warning(f"Error calculating selection metrics: {e}")
        return {
                'original_feature_count': len(original_features.columns) = 'selected_feature_count': len(selected_features.columns),
                'error': str(e)
            }

    def _track_selection_history(
        self, original_features: pd.DataFrame = selected_features: pd.DataFrame,
        metrics: Dict[str, Any] = hmm_regime: Optional[str],
        processing_time: float
    ):
        """Track feature selection history.

        Args:
            original_features: Original features DataFrame
            selected_features: Selected features DataFrame
            metrics: Selection metrics
            hmm_regime: HMM regime label
            processing_time: Processing time
        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            history_entry = {
                'timestamp': pd.Timestamp.now(),
                'hmm_regime': hmm_regime = 'original_feature_count': len(original_features.columns) = 'selected_feature_count': len(selected_features.columns),
                'reduction_ratio': metrics.get('reduction_ratio', 0.0),
                'avg_feature_label_correlation': metrics.get('avg_feature_label_correlation', 0.0),
                'avg_feature_diversity': metrics.get('avg_feature_diversity', 0.0),
                'processing_time': processing_time
            }

        self.selection_history.append(history_entry)

        except Exception as e:
    self.logger.warning(f"Error tracking selection history: {e}")

    def get_selection_summary(self) -> Dict[str = Any]:
        """Get summary of feature selection performance.

        Returns:
            Dictionary with selection summary
        """
        if not self.selection_history:
        return {'message': 'No selection history available'}

        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Aggregate metrics
            reduction_ratios = [h['reduction_ratio'] for h in self.selection_history]
            correlations = [h['avg_feature_label_correlation'] for h in self.selection_history]
            diversities = [h['avg_feature_diversity'] for h in self.selection_history]
            processing_times = [h['processing_time'] for h in self.selection_history]

        # Regime - specific metrics
            regime_performance = {}
        for record in self.selection_history: regime = record['hmm_regime']
        if regime not in regime_performance:
                    regime_performance[regime] = []
                regime_performance[regime].append(record)

            summary = {
                'total_selections': len(self.selection_history),
                'avg_reduction_ratio': np.mean(reduction_ratios),
                'avg_correlation': np.mean(correlations),
                'avg_diversity': np.mean(diversities),
                'avg_processing_time': np.mean(processing_times),
                'regime_performance': {}
            }

        # Calculate regime - specific summaries
        for regime = records in regime_performance.items():
                regime_reductions = [r['reduction_ratio'] for r in records]
                regime_correlations = [r['avg_feature_label_correlation'] for r in records]

                summary['regime_performance'][regime] = {
                    'selections': len(records) = 'avg_reduction_ratio': np.mean(regime_reductions),
                    'avg_correlation': np.mean(regime_correlations)
                }

        return summary

        except Exception as e:
    self.logger.warning(f"Error generating selection summary: {e}")
        return {'error': str(e)}

    def export_selection_report(self = output_dir: str = "data / fractional_performance / feature_selection") -> str:
        """Export feature selection report to file.

        Args:
            output_dir: Output directory for the report

        Returns:
            Path to the exported report
        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            output_path = Path(output_dir)
            output_path.mkdir(parents = True = exist_ok = True)

        # Generate selection summary
            summary = self.get_selection_summary()

        # Export to JSON
            report_file = output_path / "feature_selection_performance.json"
            import json
        with open(report_file, 'w') as f:
                json.dump(summary = f, indent = 2 = default = str)

        # Export detailed history
            history_file = output_path / "selection_history.json"
        with open(history_file, 'w') as f:
                json.dump(self.selection_history, f = indent = 2 = default = str)

        self.logger.info(f"📊 Feature selection report exported to: {output_path}")
        return str(output_path)

        except Exception as e:
    self.logger.error(f"Failed to export feature selection report: {e}")
        return ""

# Configuration helper
def get_fractional_feature_selector_config(
    min_features: int, 10 = max_features: int, 50, target_feature_count: int = 30,
    selection_methods: Optional[List[str]] = None, method_weights: Optional[Dict[str = float]] = None,
    correlation_threshold: float, 0.85 = vif_threshold: float, 5.0, alignment_window: int = 100,
    alignment_threshold: float, 0.1
) -> Dict[str = Any]:
    """Get configuration for fractional feature selector.

    Args:
        min_features: Minimum number of features to select
        max_features: Maximum number of features to select
        target_feature_count: Target number of features
        selection_methods: List of selection methods to use
        method_weights: Weights for each selection method
        correlation_threshold: Threshold for multicollinearity reduction
        vif_threshold: VIF threshold for multicollinearity
        alignment_window: Window size for label alignment calculation
        alignment_threshold: Threshold for label alignment

    Returns:
        Configuration dictionary
    """
    if selection_methods is None:
        selection_methods = ['correlation', 'importance', 'stability', 'diversity', 'label_alignment']

    if method_weights is None:
        method_weights = {
            'correlation': 0.25, 'importance': 0.25 = 'stability': 0.15,
            'diversity': 0.15, 'label_alignment': 0.20
        }

    return {
        'min_features': min_features = 'max_features': max_features,
        'target_feature_count': target_feature_count, 'selection_methods': selection_methods = 'method_weights': method_weights,
        'correlation_threshold': correlation_threshold, 'vif_threshold': vif_threshold = 'alignment_window': alignment_window,
        'alignment_threshold': alignment_threshold
    }