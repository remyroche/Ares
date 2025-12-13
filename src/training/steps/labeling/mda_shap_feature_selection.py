"""
MDA/SHAP Feature Selection Module

Implements comprehensive feature selection using Mean Decrease Accuracy (MDA) and SHAP
with time-series cross-validation and feature clustering for robust selection.

    Phases:
1. Setup and Feature Preparation
2. Execution of Importance Methods (MDA/SHAP)
3. Analysis and Feature Selection (Elbow Method for Optimal Count)
"""

import warnings
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.tprint import tprint_info, tprint_warning, tprint_success, tprint_error

# Import required libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.feature_selection import f_classif, SelectKBest
    from sklearn.preprocessing import StandardScaler
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    import lightgbm as lgb
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def _safe_predict_proba_pos(model: Any, X: pd.DataFrame, pos_label: int = 1) -> Optional[np.ndarray]:
    try:
        proba = model.predict_proba(X)
        arr = np.asarray(proba)
        if arr.ndim == 1:
            return arr.astype(float)
        if arr.ndim != 2:
            return None

        n_classes = int(arr.shape[1])
        if n_classes <= 0:
            return None

        classes = getattr(model, "classes_", None)
        if classes is not None:
            try:
                classes = list(classes)
                if pos_label in classes:
                    idx = int(classes.index(pos_label))
                    return arr[:, idx].astype(float)
                if n_classes == 1:
                    return (np.zeros(arr.shape[0], dtype=float) if int(classes[0]) != int(pos_label) else arr[:, 0].astype(float))
            except Exception:
                pass

        if n_classes == 1:
            return arr[:, 0].astype(float)

        return arr[:, 1].astype(float)
    except Exception:
        return None


class MDA_SHAP_FeatureSelector:
    """
    Comprehensive feature selection using MDA and SHAP with time-series validation.

    Implements the 3-phase approach with extensive computational optimizations:
    1. Setup: Model training, data preparation, TSCV with subsampling
    2. Execution: Clustered MDA and SHAP analysis on subsampled data
    3. Selection: Combined analysis with Elbow method for optimal feature count

    Computational Optimizations:
    - Training data subsampling (default 80% with max 10k samples per fold)
    - Feature subsampling for correlation analysis (max 200 features)
    - Observation subsampling for correlation matrices (max 2000 samples)
    - SHAP evaluation budget control (max_evals parameter)
    - Stratified sampling to maintain class balance
    - Pre-filter correlation subsampling (max 150 features)
    - Hierarchical clustering on subsampled correlation matrix
    - MDA permutation testing on subsampled data
    """

    def __init__(
        self,
        model_type: str = "rf",  # "rf" or "lgbm"
        n_folds: int = 5,
        embargo_pct: float = 0.01,
        random_state: int = 42,
        verbose: bool = True,
        # Subsampling parameters for computational efficiency
        subsample_train_pct: float = 0.8,  # Fraction of training data to use
        max_train_samples: int = 10000,    # Maximum training samples per fold
        subsample_features_pct: float = 0.9,  # Fraction of features to consider
        shap_max_evals: int = 1000,         # SHAP computation budget
    ):
        """
        Initialize the feature selector.

        Args:
            model_type: Base model type ("rf" or "lgbm")
            n_folds: Number of time-series CV folds
            embargo_pct: Embargo percentage to prevent leakage
            random_state: Random state for reproducibility
            verbose: Whether to print progress
        """
        self.model_type = model_type
        self.n_folds = n_folds
        self.embargo_pct = embargo_pct
        self.random_state = random_state
        self.verbose = verbose
        self.enable_shap = bool(SHAP_AVAILABLE)

        # Subsampling parameters
        self.subsample_train_pct = subsample_train_pct
        self.max_train_samples = max_train_samples
        self.subsample_features_pct = subsample_features_pct
        self.shap_max_evals = shap_max_evals

        # Storage for results
        self.feature_clusters = {}
        self.mda_results = {}
        self.shap_results = {}
        self.selected_features = []
        self.importance_rankings = {}

        # Check dependencies
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for MDA/SHAP feature selection")
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP required for SHAP analysis. Install with: pip install shap")

    def _log(self, message: str, level: str = "info"):
        """Log message if verbose."""
        if not self.verbose:
            return

        if level == "info":
            tprint_info(message)
        elif level == "warning":
            tprint_warning(message)
        elif level == "success":
            tprint_success(message)
        elif level == "error":
            tprint_error(message)

    def _create_base_model(self):
        """Create the base model for importance analysis."""
        if self.model_type == "rf":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == "lgbm":
            return lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=self.random_state,
                verbosity=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _create_purged_tscv(self, n_samples: int) -> TimeSeriesSplit:
        """Create purged time-series cross-validation splits."""
        embargo_size = int(n_samples * self.embargo_pct)
        return TimeSeriesSplit(
            n_splits=self.n_folds,
            gap=embargo_size,
            test_size=None  # Use default test size
        )

    def _subsample_training_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        target_sample_weight: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """
        Subsample training data to limit computational demands.

        Args:
            X_train: Training features
            y_train: Training labels
            target_sample_weight: Training sample weights

        Returns:
            Tuple of (X_subsampled, y_subsampled, weights_subsampled)
        """
        n_samples = len(X_train)
        n_features = len(X_train.columns)

        # Ensure sample weights are aligned to X_train index so that any sampling
        # performed below keeps weights consistent with rows.
        if target_sample_weight is not None and not isinstance(target_sample_weight, pd.Series):
            try:
                arr = np.asarray(target_sample_weight).ravel()
                n = int(len(X_train))
                if arr.shape[0] == n:
                    target_sample_weight = pd.Series(arr, index=X_train.index)
                elif arr.shape[0] > n:
                    target_sample_weight = pd.Series(arr[:n], index=X_train.index)
                else:
                    padded = np.ones(n, dtype=float)
                    if arr.shape[0] > 0:
                        padded[: arr.shape[0]] = arr
                    target_sample_weight = pd.Series(padded, index=X_train.index)
            except Exception:
                try:
                    target_sample_weight = pd.Series(np.ones(int(len(X_train)), dtype=float), index=X_train.index)
                except Exception:
                    target_sample_weight = None

        # Determine target sample size
        target_samples = min(
            int(n_samples * self.subsample_train_pct),
            self.max_train_samples
        )

        # Subsample features if needed
        target_features = int(n_features * self.subsample_features_pct)
        if n_features > target_features:
            # Select most variable features for subsampling
            feature_variances = X_train.var()
            top_features = feature_variances.nlargest(target_features).index
            X_train = X_train[top_features]

        # Subsample observations if needed
        if n_samples > target_samples:
            # Stratified sampling by label to maintain class balance
            if y_train.nunique() > 1:
                # For binary/multiclass, sample proportionally
                samples_per_class = max(50, target_samples // y_train.nunique())

                sampled_indices = []
                for class_val in y_train.unique():
                    class_indices = y_train[y_train == class_val].index
                    n_class_samples = min(len(class_indices), samples_per_class)
                    sampled_class = np.random.RandomState(self.random_state).choice(
                        class_indices, size=n_class_samples, replace=False
                    )
                    sampled_indices.extend(sampled_class)

                # Fill remaining samples if needed
                remaining = target_samples - len(sampled_indices)
                if remaining > 0:
                    remaining_indices = [idx for idx in X_train.index if idx not in sampled_indices]
                    if remaining_indices:
                        additional = np.random.RandomState(self.random_state).choice(
                            remaining_indices, size=min(remaining, len(remaining_indices)), replace=False
                        )
                        sampled_indices.extend(additional)

                # Ensure we don't exceed target
                sampled_indices = sampled_indices[:target_samples]

            else:
                # Simple random sampling for single class
                sampled_indices = np.random.RandomState(self.random_state).choice(
                    X_train.index, size=target_samples, replace=False
                )

            try:
                sampled_index = pd.Index(sampled_indices)
                sampled_index = sampled_index.intersection(X_train.index)
                if len(sampled_index) < max(10, int(0.25 * target_samples)):
                    raise KeyError("too_few_samples_after_index_intersection")

                X_train = X_train.loc[sampled_index]
                y_train = y_train.reindex(sampled_index)
                if target_sample_weight is not None and hasattr(target_sample_weight, "reindex"):
                    target_sample_weight = target_sample_weight.reindex(sampled_index)
            except Exception:
                # Defensive fallback: sample by position to avoid index mismatches
                rs = np.random.RandomState(self.random_state)
                pos = rs.choice(np.arange(len(X_train)), size=min(target_samples, len(X_train)), replace=False)
                X_train = X_train.iloc[pos]
                y_train = y_train.reindex(X_train.index)
                if target_sample_weight is not None and hasattr(target_sample_weight, "reindex"):
                    target_sample_weight = target_sample_weight.reindex(X_train.index)

            if target_sample_weight is not None and hasattr(target_sample_weight, "fillna"):
                target_sample_weight = target_sample_weight.fillna(1.0)

        return X_train, y_train, target_sample_weight

    def _compute_fold_ic(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Compute Information Coefficient (IC) for each feature in a fold.

        IC = Spearman correlation between feature values and target on test set.

        Args:
            X_train: Training features (for model training)
            y_train: Training targets
            X_test: Test features (for IC calculation)
            y_test: Test targets

        Returns:
            Dict of feature ICs
        """
        ic_scores = {}

        # Train a simple model to get feature-target relationships
        # Use predictions on test set for more stable IC calculation
        model = self._create_base_model()
        model.fit(X_train, y_train)

        # Get model predictions on test set
        y_pred = None
        if hasattr(model, 'predict_proba'):
            y_pred = _safe_predict_proba_pos(model, X_test)
        if y_pred is None:
            try:
                y_pred = np.asarray(model.predict(X_test), dtype=float)
            except Exception:
                y_pred = np.zeros(int(len(X_test)), dtype=float)

        # Calculate IC as Spearman correlation between predictions and true labels
        # This gives a more stable measure than individual feature correlations
        for feature in X_test.columns:
            try:
                # Spearman correlation between feature and target
                corr = X_test[feature].corr(y_test, method='spearman')
                if not np.isnan(corr):
                    ic_scores[feature] = float(corr)
                else:
                    ic_scores[feature] = 0.0
            except Exception:
                ic_scores[feature] = 0.0

        return ic_scores

    def _compute_composite_scores(
        self,
        clusters: Dict[str, List[str]],
        mda_scores: Dict[str, float],
        shap_scores: Dict[str, float],
        ic_scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute composite scores combining MDA, SHAP, and IR.

        Formula: Composite Score = MDA_cluster_score * IR_rank

        Args:
            clusters: Feature clusters
            mda_scores: MDA scores per cluster
            shap_scores: SHAP scores per feature
            ic_scores: IC/IR scores per feature

        Returns:
            Dict with composite scores and rankings
        """
        composite_scores = {}

        # Get IR values for ranking
        ir_values = {}
        for feature, ic_data in ic_scores.items():
            ir_values[feature] = ic_data.get('ir', 0.0)

        # Rank features by IR (higher IR = more stable = better rank)
        sorted_by_ir = sorted(ir_values.items(), key=lambda x: x[1], reverse=True)
        ir_ranks = {feature: rank for rank, (feature, _) in enumerate(sorted_by_ir, 1)}

        # Compute composite scores for individual features
        for cluster_name, features in clusters.items():
            cluster_mda = mda_scores.get(cluster_name, 0.0)

            for feature in features:
                if feature in ir_ranks and feature in shap_scores:
                    # Composite Score = MDA_cluster * IR_rank (lower rank = better)
                    ir_rank = ir_ranks[feature]
                    composite_score = cluster_mda / ir_rank  # Higher MDA and lower rank = higher score

                    composite_scores[feature] = {
                        'composite_score': float(composite_score),
                        'mda_cluster_score': float(cluster_mda),
                        'ir_score': float(ir_values.get(feature, 0.0)),
                        'ir_rank': int(ir_rank),
                        'shap_score': float(shap_scores.get(feature, 0.0)),
                        'cluster': cluster_name
                    }

        return composite_scores

    def _find_elbow_point(self, scores: List[float], min_features: int = 10) -> int:
        """
        Find elbow point in the scores curve using the "elbow method".

        Args:
            scores: Sorted list of composite scores (descending)
            min_features: Minimum number of features to select

        Returns:
            Index of elbow point
        """
        if len(scores) <= 0:
            return 0

        if len(scores) <= min_features:
            return max(0, len(scores) - 1)

        # Normalize scores to 0-1 range
        scores = np.array(scores)
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        # Calculate the line from first to last point
        x = np.arange(len(scores_norm))
        line_start = scores_norm[0]
        line_end = scores_norm[-1]
        line = line_start + (line_end - line_start) * (x / (len(x) - 1))

        # Calculate perpendicular distances
        distances = np.abs(scores_norm - line)

        # Find the elbow (maximum distance point)
        elbow_idx = np.argmax(distances)

        # Ensure we select at least min_features
        elbow_idx = max(elbow_idx, min_features - 1)

        # Ensure elbow index is within bounds
        elbow_idx = min(elbow_idx, len(scores) - 1)

        return elbow_idx

    def _validate_feature_set_performance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_sample_weight: Optional[pd.Series],
        selected_features: List[str],
        n_validation_folds: int = 3
    ) -> Dict[str, Any]:
        """
        Validate that the selected feature set performs better than slightly smaller/larger sets.

        Args:
            X: Full feature matrix
            y: Target labels
            target_sample_weight: Sample weights
            selected_features: Currently selected features
            n_validation_folds: Number of CV folds for validation

        Returns:
            Validation results
        """
        n_selected = len(selected_features)

        # Test sets: selected, selected-25%, selected+25%
        test_sets = {
            'selected': selected_features,
            'smaller': selected_features[:max(5, int(n_selected * 0.75))],
            'larger': selected_features + [f for f in X.columns if f not in selected_features][:int(n_selected * 0.25)]
        }

        results = {}

        # Create TSCV for validation
        tscv = self._create_purged_tscv(len(X))
        fold_scores = {name: [] for name in test_sets.keys()}

        fold_count = 0
        for train_idx, test_idx in tscv.split(X):
            if fold_count >= n_validation_folds:
                break

            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
            y_test_fold = y.iloc[test_idx]

            # Subsample for speed
            X_train_fold, y_train_fold, _ = self._subsample_training_data(X_train_fold, y_train_fold, None)

            for set_name, features in test_sets.items():
                try:
                    # Train model
                    model = self._create_base_model()
                    X_train_subset = X_train_fold[features]
                    model.fit(X_train_subset, y_train_fold)

                    # Evaluate
                    X_test_subset = X_test_fold[features]
                    if hasattr(model, 'predict_proba'):
                        y_pred = _safe_predict_proba_pos(model, X_test_subset)
                        if y_pred is None:
                            y_pred = np.full(int(len(X_test_subset)), 0.5, dtype=float)
                        score = roc_auc_score(y_test_fold, y_pred) if y_test_fold.nunique() >= 2 else 0.5
                    else:
                        y_pred = model.predict(X_test_subset)
                        score = accuracy_score(y_test_fold, y_pred)

                    fold_scores[set_name].append(score)

                except Exception:
                    fold_scores[set_name].append(0.5)  # Default score

            fold_count += 1

        # Compute average scores
        for set_name, scores in fold_scores.items():
            results[set_name] = {
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores, ddof=1) if len(scores) > 1 else 0.0),
                'n_folds': len(scores),
                'n_features': len(test_sets[set_name])
            }

        # Determine if selected set is optimal
        selected_score = results['selected']['mean_score']
        smaller_score = results['smaller']['mean_score']
        larger_score = results['larger']['mean_score']

        is_optimal = bool(selected_score >= smaller_score and selected_score >= larger_score)
        recommendation = 'selected' if is_optimal else (
            'smaller' if smaller_score > selected_score else 'larger'
        )

        results['validation'] = {
            'selected_vs_smaller': float(selected_score - smaller_score),
            'selected_vs_larger': float(selected_score - larger_score),
            'is_optimal': bool(is_optimal),
            'recommendation': str(recommendation),
        }

        return results

    def plot_elbow_analysis(self, save_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Create and optionally save a plot of the composite scores with elbow point.

        Args:
            save_path: Path to save the plot (optional)

        Returns:
            Plot data dictionary
        """
        if not hasattr(self, 'sorted_features') or not self.sorted_features:
            return None

        try:
            import matplotlib.pyplot as plt

            # Extract scores
            scores = [score_data['composite_score'] for _, score_data in self.sorted_features]
            features = [feat for feat, _ in self.sorted_features]

            # Create plot
            fig, ax = plt.subplots(figsize=(12, 8))

            # Plot scores
            x = np.arange(len(scores))
            ax.plot(x, scores, 'b-', linewidth=2, label='Composite Scores')
            ax.scatter(x, scores, c='blue', s=30, alpha=0.7)

            # Mark elbow point
            if hasattr(self, 'elbow_idx'):
                elbow_x = int(self.elbow_idx)
                if 0 <= elbow_x < len(scores):
                    elbow_y = scores[elbow_x]
                    ax.scatter(elbow_x, elbow_y, c='red', s=100, marker='*',
                              label=f'Elbow Point ({elbow_x+1} features)')
                    ax.axvline(x=elbow_x, color='red', linestyle='--', alpha=0.7)

            # Formatting
            ax.set_xlabel('Feature Rank')
            ax.set_ylabel('Composite Score (MDA × IR Rank)')
            ax.set_title('Feature Selection: Elbow Method Analysis')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Add feature labels for top features
            for i in range(min(10, len(features))):
                ax.annotate(features[i][:20] + ('...' if len(features[i]) > 20 else ''),
                           (x[i], scores[i]), xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)

            plt.tight_layout()

            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                self._log(f"💾 Elbow plot saved to {save_path}")

            # Return plot data
            elbow_idx = getattr(self, 'elbow_idx', None)
            if elbow_idx is not None:
                elbow_idx = int(elbow_idx)
                if elbow_idx < 0 or elbow_idx >= len(scores):
                    elbow_idx = None

            plot_data = {
                'x': x.tolist(),
                'scores': scores,
                'features': features,
                'elbow_idx': elbow_idx,
                'elbow_score': scores[elbow_idx] if elbow_idx is not None else None
            }

            plt.close(fig)
            return plot_data

        except ImportError:
            self._log("⚠️ Matplotlib not available for plotting")
            return None
        except Exception as e:
            self._log(f"⚠️ Error creating elbow plot: {e}")
            return None

    def _cluster_features(self, X: pd.DataFrame, corr_threshold: float = 0.85) -> Dict[str, List[str]]:
        """
        Cluster highly correlated features to fix substitution effect.
        Uses subsampling for computational efficiency.

        Args:
            X: Feature matrix
            corr_threshold: Correlation threshold for clustering

        Returns:
            Dict mapping cluster names to feature lists
        """
        self._log("🔗 Clustering correlated features...")

        # Subsample features for correlation computation if needed
        n_features = len(X.columns)
        if n_features > 200:  # Only subsample if we have many features
            subsample_size = min(200, int(n_features * self.subsample_features_pct))
            # Select most variable features for correlation analysis
            feature_variances = X.var()
            top_features = feature_variances.nlargest(subsample_size).index
            X_for_corr = X[top_features]
            self._log(f"   📊 Subsampled {n_features} → {len(X_for_corr.columns)} features for correlation analysis")
        else:
            X_for_corr = X

        # Subsample observations for correlation computation if needed
        n_samples = len(X_for_corr)
        if n_samples > 2000:  # Subsample observations for speed
            sample_size = min(2000, int(n_samples * 0.7))  # 70% of samples or max 2000
            X_for_corr = X_for_corr.sample(n=sample_size, random_state=self.random_state)
            self._log(f"   📊 Subsampled {n_samples} → {len(X_for_corr)} observations for correlation analysis")

        # Calculate correlation matrix on subsampled data
        corr_matrix = X_for_corr.corr(method='spearman').fillna(0)
        try:
            np.fill_diagonal(corr_matrix.values, 1.0)
        except Exception:
            pass

        # Convert to distance matrix
        distance_matrix = np.sqrt(2 * (1 - np.abs(corr_matrix)))
        try:
            np.fill_diagonal(distance_matrix.values, 0.0)
        except Exception:
            pass

        # Hierarchical clustering on subsampled features
        linkage_matrix = linkage(squareform(distance_matrix.values, checks=False), method='ward')

        # Form clusters based on correlation threshold
        # Convert correlation threshold to distance
        max_distance = np.sqrt(2 * (1 - corr_threshold))

        cluster_labels = fcluster(linkage_matrix, t=max_distance, criterion='distance')

        # Map subsampled features to their clusters
        subsampled_features = list(X_for_corr.columns)
        feature_to_cluster = {}
        for i, feature in enumerate(subsampled_features):
            cluster_id = cluster_labels[i]
            feature_to_cluster[feature] = cluster_id

        # Now assign ALL original features to clusters based on similarity to subsampled features
        clusters = {}
        for feature in X.columns:
            if feature in feature_to_cluster:
                # Feature was in subsample
                cluster_id = feature_to_cluster[feature]
            else:
                # Feature not in subsample - find most correlated subsampled feature
                max_corr = -1
                best_cluster = 0

                # Compute correlation with subsampled features
                for sub_feature in subsampled_features:
                    try:
                        corr = X[feature].corr(X[sub_feature], method='spearman')
                        if not np.isnan(corr) and abs(corr) > max_corr:
                            max_corr = abs(corr)
                            best_cluster = feature_to_cluster[sub_feature]
                    except:
                        continue

                cluster_id = best_cluster if max_corr > 0.5 else len(cluster_labels) + 1  # New cluster for uncorrelated

            cluster_name = f"cluster_{cluster_id}"
            if cluster_name not in clusters:
                clusters[cluster_name] = []
            clusters[cluster_name].append(feature)

        # Remove singleton clusters (not useful for MDA)
        clusters = {k: v for k, v in clusters.items() if len(v) > 1}

        self._log(f"📊 Created {len(clusters)} feature clusters from {len(X.columns)} features")
        self.feature_clusters = clusters

        return clusters

    def _compute_clustered_mda(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        target_sample_weight: Optional[pd.Series],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        clusters: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Compute Mean Decrease Accuracy for feature clusters.

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            clusters: Feature clusters

        Returns:
            Dict of cluster importance scores
        """
        # Train baseline model with sample weights
        model = self._create_base_model()

        # Use sample weights if provided
        fit_kwargs = {}
        if target_sample_weight is not None:
            # Convert to numpy array and ensure proper shape
            weight_array = np.asarray(target_sample_weight).ravel()
            if self.model_type == "rf":
                fit_kwargs["sample_weight"] = weight_array
            elif self.model_type == "lgbm":
                fit_kwargs["sample_weight"] = weight_array

        model.fit(X_train, y_train, **fit_kwargs)

        # Calculate baseline performance
        if hasattr(y_test, 'nunique') and y_test.nunique() > 2:
            # Multi-class
            baseline_score = accuracy_score(y_test, model.predict(X_test))
        else:
            # Binary (use AUC if available)
            try:
                y_pred_proba = _safe_predict_proba_pos(model, X_test)
                if y_pred_proba is None:
                    raise ValueError("predict_proba_failed")
                baseline_score = roc_auc_score(y_test, y_pred_proba) if y_test.nunique() >= 2 else 0.5
            except:
                baseline_score = accuracy_score(y_test, model.predict(X_test))

        cluster_importance = {}

        for cluster_name, features in clusters.items():
            # Create shuffled version of test data for this cluster
            X_test_shuffled = X_test.copy()

            # Shuffle all features in this cluster
            for feature in features:
                if feature in X_test_shuffled.columns:
                    np.random.shuffle(X_test_shuffled[feature].values)

            # Calculate performance on shuffled data
            if hasattr(y_test, 'nunique') and y_test.nunique() > 2:
                shuffled_score = accuracy_score(y_test, model.predict(X_test_shuffled))
            else:
                try:
                    y_pred_proba_shuffled = _safe_predict_proba_pos(model, X_test_shuffled)
                    if y_pred_proba_shuffled is None:
                        raise ValueError("predict_proba_failed")
                    shuffled_score = roc_auc_score(y_test, y_pred_proba_shuffled) if y_test.nunique() >= 2 else 0.5
                except:
                    shuffled_score = accuracy_score(y_test, model.predict(X_test_shuffled))

            # Importance = baseline - shuffled (higher = more important)
            importance = baseline_score - shuffled_score
            cluster_importance[cluster_name] = max(0, importance)  # Ensure non-negative

        return cluster_importance

    def _compute_shap_importance(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        target_sample_weight: Optional[pd.Series],
        X_test: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute SHAP feature importance.

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features for SHAP calculation

        Returns:
            Dict of feature importance scores
        """
        if not self.enable_shap:
            return {}

        # Train model with sample weights
        model = self._create_base_model()

        # Use sample weights if provided
        fit_kwargs = {}
        if target_sample_weight is not None:
            # Convert to numpy array and ensure proper shape
            weight_array = np.asarray(target_sample_weight).ravel()
            if self.model_type == "rf":
                fit_kwargs["sample_weight"] = weight_array
            elif self.model_type == "lgbm":
                fit_kwargs["sample_weight"] = weight_array

        try:
            model.fit(X_train, y_train, **fit_kwargs)

            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)

            # Calculate SHAP values (sample for speed)
            n_samples = min(1000, len(X_test))
            X_sample = X_test.sample(n=n_samples, random_state=self.random_state)

            # Use max_evals to limit computational cost (if supported by installed shap)
            try:
                shap_values = explainer(X_sample, max_evals=self.shap_max_evals)
            except TypeError:
                shap_values = explainer(X_sample)

            shap_arr = None
            try:
                if hasattr(shap_values, "values"):
                    shap_arr = np.asarray(shap_values.values)
                else:
                    shap_arr = np.asarray(shap_values)
            except Exception:
                shap_arr = None

            if shap_arr is None:
                return {}

            # Normalize shapes:
            # - (n, f) => ok
            # - (n, f, c) => reduce across classes
            if shap_arr.ndim == 3:
                shap_arr = np.abs(shap_arr).mean(axis=2)
            feature_importance = np.abs(shap_arr).mean(axis=0)

            importance_dict = dict(zip(X_sample.columns, feature_importance))
            return importance_dict
        except Exception as e:
            self._log(f"⚠️ SHAP computation failed: {e}. Continuing without SHAP")
            return {}

    def _apply_pre_filters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_sample_weight: Optional[pd.Series],
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Apply pre-filters before MDA/SHAP analysis.

        Args:
            X: Feature matrix
            y: Target labels
            config: Pre-filter configuration

        Returns:
            Filtered feature matrix
        """
        self._log("🔍 Applying pre-filters...")

        try:
            self._prefilter_lgbm_mdi_features = []
        except Exception:
            pass

        initial_features = len(X.columns)
        filtered_features = list(X.columns)
        prefilter_counts: Dict[str, int] = {
            "initial": int(initial_features),
        }

        # 1. LGBM MDI filter
        if config.get("enable_lgbm_mdi_filter", True):
            self._log("   🌳 LGBM MDI filter...")
            before = int(len(filtered_features))
            # Use fewer estimators for pre-filtering to save computation
            lgbm_model = lgb.LGBMClassifier(
                n_estimators=100,  # Reduced from 500 for speed
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                verbosity=-1
            )

            # Use sample weights if provided
            fit_kwargs = {}
            if target_sample_weight is not None:
                # Convert to numpy array and ensure proper shape
                try:
                    if isinstance(target_sample_weight, pd.Series):
                        target_sample_weight = target_sample_weight.reindex(X.index).fillna(1.0)
                        weight_array = np.asarray(target_sample_weight.values).ravel()
                    else:
                        weight_array = np.asarray(target_sample_weight).ravel()
                    if int(weight_array.shape[0]) != int(len(X)):
                        if int(weight_array.shape[0]) > int(len(X)):
                            weight_array = weight_array[: int(len(X))]
                        else:
                            padded = np.ones(int(len(X)), dtype=float)
                            if int(weight_array.shape[0]) > 0:
                                padded[: int(weight_array.shape[0])] = weight_array
                            weight_array = padded
                except Exception:
                    weight_array = np.ones(int(len(X)), dtype=float)
                fit_kwargs["sample_weight"] = weight_array

            lgbm_model.fit(X, y, **fit_kwargs)

            # Get top 200 features by importance, or top 70% if fewer than 200 (pre-filter, Elbow method determines final count)
            mdi_scores = dict(zip(X.columns, lgbm_model.feature_importances_))
            n_keep = min(200, int(len(X.columns) * 0.7))
            top_features = sorted(mdi_scores.keys(), key=lambda x: mdi_scores[x], reverse=True)[:n_keep]

            filtered_features = [f for f in filtered_features if f in top_features]
            after = int(len(filtered_features))
            prefilter_counts["lgbm_mdi"] = after
            self._log(f"   📊 LGBM MDI: {before} → {after} features")

            try:
                self._prefilter_lgbm_mdi_features = list(filtered_features)
            except Exception:
                pass

        # 2. Correlation filter
        if config.get("enable_correlation_filter", True):
            self._log("   🔗 Correlation filter...")
            before = int(len(filtered_features))

            # Subsample for correlation computation if many features
            features_for_corr = filtered_features
            n_features_corr = len(features_for_corr)

            if n_features_corr > 150:  # Subsample if too many features
                subsample_corr_size = min(150, int(n_features_corr * 0.7))
                # Select most variable features
                X_subset = X[features_for_corr]
                feature_variances = X_subset.var()
                features_for_corr = feature_variances.nlargest(subsample_corr_size).index.tolist()
                self._log(f"   📊 Subsampled {n_features_corr} → {len(features_for_corr)} features for correlation filter")

            corr_matrix = X[features_for_corr].corr(method='spearman').abs()
            upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool))

            # Drop features with correlation > 0.95
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            filtered_features = [f for f in filtered_features if f not in to_drop]
            after = int(len(filtered_features))
            prefilter_counts["correlation"] = after
            self._log(f"   📊 Correlation filter: {before} → {after} features")

        # 3. Low variance filter
        if config.get("enable_variance_filter", True):
            self._log("   📈 Variance filter...")
            before = int(len(filtered_features))
            variances = X[filtered_features].var()
            variance_threshold = config.get("variance_threshold", 1e-9)
            high_variance_features = variances[variances > variance_threshold].index.tolist()
            filtered_features = [f for f in filtered_features if f in high_variance_features]
            after = int(len(filtered_features))
            prefilter_counts["variance"] = after
            self._log(f"   📊 Variance filter: {before} → {after} features")

        # 4. ANOVA F-test filter
        if config.get("enable_anova_filter", True):
            self._log("   📊 ANOVA F-test filter...")
            before = int(len(filtered_features))
            # Use only features that passed previous filters
            X_filtered = X[filtered_features]

            # Handle multi-class targets
            if y.nunique() > 2:
                # For multi-class, use f_classif which handles it
                selector = SelectKBest(score_func=f_classif, k='all')
                selector.fit(X_filtered, y)
                scores = selector.scores_
            else:
                # Binary case
                selector = SelectKBest(score_func=f_classif, k='all')
                selector.fit(X_filtered, y)
                scores = selector.scores_

            # Keep top 75th percentile
            percentile_threshold = np.percentile(scores, 25)  # Bottom 25% get removed
            keep_indices = scores >= percentile_threshold
            filtered_features = [f for f, keep in zip(filtered_features, keep_indices) if keep]
            after = int(len(filtered_features))
            prefilter_counts["anova"] = after
            self._log(f"   📊 ANOVA filter: {before} → {after} features")

        X_filtered = X[filtered_features]
        self._prefilter_counts = dict(prefilter_counts)
        try:
            self._prefilter_features = list(filtered_features)
        except Exception:
            self._prefilter_features = []
        self._log(f"🎯 Pre-filters complete: {initial_features} → {len(filtered_features)} features")

        return X_filtered

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_sample_weight: Optional[pd.Series] = None,
        regime_leaf_config: Optional[Dict[str, Any]] = None,
        pre_filter_config: Optional[Dict[str, Any]] = None,
        corr_threshold: float = 0.85,
        top_clusters: int = 5,
        shap_sample_size: int = 1000,
        enable_shap_interaction_features: bool = False,
        shap_interaction_config: Optional[Dict[str, Any]] = None,
        elbow_min_features: int = 10,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Main feature selection method implementing the 3-phase approach.

        Args:
            X: Feature matrix
            y: Target labels
            target_sample_weight: Optional target sample weights for training
            pre_filter_config: Configuration for pre-filters
            corr_threshold: Correlation threshold for clustering
            top_clusters: Number of top clusters to consider (Elbow method determines final count)
            shap_sample_size: Sample size for SHAP calculations

        Returns:
            Tuple of (selected_features, detailed_results)
        """
        self._log("🚀 Starting MDA/SHAP Feature Selection")
        self._log("=" * 50)

        # Defensive alignment: this selector internally samples by label index and then
        # indexes X via .loc[indices]. If X and y indices are not identical, this will
        # raise KeyError ("[...] not in index").
        try:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)

            if not isinstance(y, pd.Series):
                y = pd.Series(y, index=X.index)
            else:
                y = y.reindex(X.index)

            if target_sample_weight is not None:
                if isinstance(target_sample_weight, pd.Series):
                    target_sample_weight = target_sample_weight.reindex(X.index)
                else:
                    arr = np.asarray(target_sample_weight).ravel()
                    n = int(len(X))
                    if arr.shape[0] == n:
                        target_sample_weight = pd.Series(arr, index=X.index)
                    elif arr.shape[0] > n:
                        target_sample_weight = pd.Series(arr[:n], index=X.index)
                    else:
                        padded = np.ones(n, dtype=float)
                        if arr.shape[0] > 0:
                            padded[: arr.shape[0]] = arr
                        target_sample_weight = pd.Series(padded, index=X.index)

            valid_mask = y.notna()
            if hasattr(valid_mask, "values") and int(np.sum(valid_mask.values)) > 0:
                X = X.loc[valid_mask]
                y = y.loc[valid_mask]
                if target_sample_weight is not None:
                    target_sample_weight = target_sample_weight.loc[valid_mask].fillna(1.0)
        except Exception:
            pass

        # Phase 1: Setup and Feature Preparation
        self._log("📋 Phase 1: Setup and Feature Preparation")

        # Apply pre-filters
        if pre_filter_config:
            X_filtered = self._apply_pre_filters(X, y, target_sample_weight, pre_filter_config)
        else:
            X_filtered = X.copy()

        regime_leaf_features_df = pd.DataFrame(index=X_filtered.index)
        regime_leaf_feature_names: List[str] = []
        regime_leaf_info: Dict[str, Any] = {"enabled": False}

        try:
            cfg = dict(regime_leaf_config or {})
            enable_regime_leaves = bool(cfg.get("enabled", False))
            if enable_regime_leaves:
                market_data = cfg.get("market_data")
                if market_data is None:
                    raise ValueError("regime_leaf_config.market_data is required when enabled")

                # Regime leaf features are derived from OHLCV market dynamics.
                # Use X only for index alignment so that specialist/meta features
                # are never fed into the regime embedding models.
                X_for_regime = X

                from .regime_leaf_feature_extractor import extract_regime_leaf_onehot_features

                extractor_cfg = dict(cfg.get("extractor_config", {}))
                regime_leaf_features_df = extract_regime_leaf_onehot_features(
                    X=X_for_regime,
                    market_data=market_data,
                    config=extractor_cfg,
                    random_state=int(cfg.get("random_state", self.random_state)),
                    verbose=bool(cfg.get("verbose", self.verbose)),
                )

                if regime_leaf_features_df is None or regime_leaf_features_df.empty:
                    regime_leaf_features_df = pd.DataFrame(index=X_filtered.index)
                else:
                    regime_leaf_features_df = regime_leaf_features_df.reindex(X_filtered.index).fillna(0.0)

                regime_leaf_feature_names = list(regime_leaf_features_df.columns)
                regime_leaf_info = {
                    "enabled": True,
                    "n_features": int(len(regime_leaf_feature_names)),
                }
        except Exception as rl_exc:
            regime_leaf_features_df = pd.DataFrame(index=X_filtered.index)
            regime_leaf_feature_names = []
            regime_leaf_info = {"enabled": False, "error": str(rl_exc)}

        # Attach regime leaf features AFTER pre-filters so that they participate
        # in Phase 2 (MDA/SHAP + interaction mining) while bypassing pre-filters.
        if not regime_leaf_features_df.empty:
            try:
                X_filtered = pd.concat([X_filtered, regime_leaf_features_df], axis=1)
            except Exception:
                pass

        if len(X_filtered.columns) < 10:
            self._log("⚠️ Too few features after pre-filtering, using all features")
            X_filtered = X.copy()

        shap_interaction_defs: List[Dict[str, Any]] = []
        shap_interaction_info: Dict[str, Any] = {"enabled": False}
        if enable_shap_interaction_features:
            try:
                from .shap_interaction_feature_mining import (
                    apply_interaction_definitions,
                    mine_shap_interaction_feature_defs,
                )

                inter_cfg = dict(shap_interaction_config or {})
                try:
                    base_mt = str(self.model_type).lower()
                except Exception:
                    base_mt = ""
                inter_cfg.setdefault("model_type", "lgbm" if base_mt == "rf" else self.model_type)
                if "sample_size" not in inter_cfg:
                    try:
                        inter_cfg["sample_size"] = int(shap_sample_size)
                    except Exception:
                        pass

                candidate_cols: List[str] = []
                try:
                    stored = getattr(self, "_prefilter_lgbm_mdi_features", None)
                    if isinstance(stored, list):
                        candidate_cols = list(stored)
                except Exception:
                    candidate_cols = []

                if not candidate_cols:
                    try:
                        lgbm_model = lgb.LGBMClassifier(
                            n_estimators=100,
                            max_depth=5,
                            learning_rate=0.1,
                            random_state=self.random_state,
                            verbosity=-1,
                        )
                        fit_kwargs = {}
                        if target_sample_weight is not None:
                            fit_kwargs["sample_weight"] = np.asarray(target_sample_weight).ravel()
                        lgbm_model.fit(X_filtered, y, **fit_kwargs)
                        mdi_scores = dict(zip(X_filtered.columns, lgbm_model.feature_importances_))
                        n_keep = min(200, int(len(X_filtered.columns) * 0.7))
                        candidate_cols = sorted(
                            mdi_scores.keys(), key=lambda x: mdi_scores[x], reverse=True
                        )[:n_keep]
                    except Exception:
                        candidate_cols = []

                try:
                    seen = set()
                    kept: List[str] = []
                    for c in candidate_cols:
                        if c in X_filtered.columns and c not in seen:
                            seen.add(c)
                            kept.append(c)
                    candidate_cols = kept
                except Exception:
                    candidate_cols = []

                if len(candidate_cols) < 2:
                    shap_interaction_defs = []
                    shap_interaction_info = {"enabled": False, "reason": "lgbm_prefilter_unavailable"}
                    raise RuntimeError("lgbm_prefilter_unavailable")

                X_interaction_candidates = X_filtered[candidate_cols]

                shap_interaction_defs, shap_interaction_info = mine_shap_interaction_feature_defs(
                    X=X_interaction_candidates,
                    y=y,
                    target_sample_weight=target_sample_weight,
                    config=inter_cfg,
                    random_state=self.random_state,
                    embargo_pct=self.embargo_pct,
                    verbose=self.verbose,
                )

                if shap_interaction_defs:
                    inter_df = apply_interaction_definitions(
                        X_filtered,
                        shap_interaction_defs,
                        fillna_value=float(inter_cfg.get("fillna_value", 0.0)),
                    )
                    if inter_df is not None and not inter_df.empty:
                        X_filtered = pd.concat([X_filtered, inter_df], axis=1)
                        try:
                            shap_interaction_info["added_feature_names"] = list(inter_df.columns)
                        except Exception:
                            pass
            except Exception as inter_exc:
                shap_interaction_defs = []
                shap_interaction_info = {"enabled": False, "error": str(inter_exc)}

        # Create time-series CV splits
        tscv = self._create_purged_tscv(len(X_filtered))

        # Phase 2: Execution of Importance Methods
        self._log("⚙️ Phase 2: Execution of Importance Methods")

        # Cluster features
        clusters = self._cluster_features(X_filtered, corr_threshold)

        # Initialize result storage
        fold_mda_results = []
        fold_shap_results = []
        fold_ic_results = []  # Information Coefficient (IC) per fold

        fold_idx = 0
        for train_idx, test_idx in tscv.split(X_filtered):
            fold_idx += 1
            self._log(f"   📊 Processing fold {fold_idx}/{self.n_folds}")

            X_train_fold = X_filtered.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_test_fold = X_filtered.iloc[test_idx]
            y_test_fold = y.iloc[test_idx]

            # Split sample weights if provided
            target_sample_weight_train_fold = None
            if target_sample_weight is not None:
                # Ensure target_sample_weight is aligned with X_filtered index
                if hasattr(target_sample_weight, 'index'):
                    target_sample_weight_aligned = target_sample_weight.reindex(X_filtered.index).fillna(1.0)
                else:
                    target_sample_weight_aligned = pd.Series(target_sample_weight, index=X_filtered.index).fillna(1.0)

                # Use positional indexing to avoid index alignment issues
                target_sample_weight_train_fold = target_sample_weight_aligned.values[train_idx]

            # Subsample training data for computational efficiency
            X_train_fold, y_train_fold, target_sample_weight_train_fold = self._subsample_training_data(
                X_train_fold, y_train_fold, target_sample_weight_train_fold
            )

            # Ensure test fold uses the exact same feature columns as training fold.
            # This prevents LightGBM "unseen feature names" errors.
            try:
                kept_cols = list(X_train_fold.columns)
                X_test_fold = X_test_fold.reindex(columns=kept_cols)
            except Exception:
                kept_cols = list(X_test_fold.columns)

            clusters_fold = clusters
            try:
                if kept_cols and isinstance(clusters, dict):
                    kept_set = set(kept_cols)
                    clusters_fold = {
                        k: [f for f in v if f in kept_set]
                        for k, v in clusters.items()
                    }
                    clusters_fold = {k: v for k, v in clusters_fold.items() if v}
            except Exception:
                clusters_fold = clusters

            # Clustered MDA
            if clusters_fold:
                fold_mda = self._compute_clustered_mda(
                    X_train_fold, y_train_fold, target_sample_weight_train_fold,
                    X_test_fold, y_test_fold, clusters_fold
                )
                fold_mda_results.append(fold_mda)

            # SHAP
            fold_shap = self._compute_shap_importance(
                X_train_fold, y_train_fold, target_sample_weight_train_fold, X_test_fold
            )
            fold_shap_results.append(fold_shap)

            # Information Coefficient (IC) - Spearman correlation per fold
            fold_ic = self._compute_fold_ic(X_train_fold, y_train_fold, X_test_fold, y_test_fold)
            fold_ic_results.append(fold_ic)

        # Aggregate results across folds
        if fold_mda_results:
            # Average MDA scores across folds
            mda_scores = {}
            for cluster in clusters.keys():
                scores = [fold.get(cluster, 0) for fold in fold_mda_results]
                mda_scores[cluster] = np.mean(scores)

            self.mda_results = mda_scores

        # Compute Information Ratio (IR) for feature stability
        if fold_ic_results:
            # Aggregate IC scores across folds
            ic_scores = {}
            for feature in X_filtered.columns:
                scores = [fold.get(feature, 0) for fold in fold_ic_results if fold.get(feature, 0) != 0]
                if scores:
                    mean_ic = np.mean(scores)
                    std_ic = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
                    # Information Ratio = Mean(IC) / Std(IC)
                    ir = mean_ic / (std_ic + 1e-8) if std_ic > 1e-8 else mean_ic * 1000  # Large IR for zero variance
                    ic_scores[feature] = {
                        'mean_ic': float(mean_ic),
                        'std_ic': float(std_ic),
                        'ir': float(ir),
                        'n_folds': len(scores)
                    }

            self.ic_results = ic_scores

        if fold_shap_results:
            # Average SHAP scores across folds
            shap_scores = {}
            for feature in X_filtered.columns:
                scores = [fold.get(feature, 0) for fold in fold_shap_results]
                shap_scores[feature] = np.mean(scores)

            self.shap_results = shap_scores

        # Phase 3: Analysis and Feature Selection
        self._log("📊 Phase 3: Analysis and Feature Selection")

        selected_features = []

        if clusters and self.mda_results and hasattr(self, 'ic_results') and self.ic_results:
            # Compute composite scores (MDA * IR rank)
            composite_scores = self._compute_composite_scores(
                clusters, self.mda_results, self.shap_results, self.ic_results
            )

            if composite_scores:
                # Sort features by composite score (descending)
                sorted_features = sorted(
                    composite_scores.items(),
                    key=lambda x: x[1]['composite_score'],
                    reverse=True
                )

                # Extract scores for elbow method
                scores_only = [score_data['composite_score'] for _, score_data in sorted_features]

                # Find elbow point
                try:
                    elbow_min_features = int(elbow_min_features)
                except Exception:
                    elbow_min_features = 10

                elbow_idx = self._find_elbow_point(scores_only, min_features=elbow_min_features)

                # Select features up to elbow point
                selected_features = [feat for feat, _ in sorted_features[:elbow_idx + 1]]

                self._log(f"🎯 Elbow method selected {len(selected_features)} features at inflection point")

                # Validate selection with performance testing
                if len(selected_features) > 10:  # Only validate if we have enough features
                    self._log("🔍 Validating feature set performance...")
                    validation_results = self._validate_feature_set_performance(
                        X_filtered, y, target_sample_weight, selected_features,
                        n_validation_folds=min(3, self.n_folds)
                    )

                    # Check if we should adjust selection
                    if not validation_results.get('validation', {}).get('is_optimal', True):
                        recommendation = validation_results['validation']['recommendation']
                        if recommendation == 'smaller':
                            # Reduce selection by 10%
                            new_count = max(int(elbow_min_features), int(len(selected_features) * 0.9))
                            selected_features = selected_features[:new_count]
                            self._log(f"📉 Validation recommended smaller set: {len(selected_features)} features")
                        elif recommendation == 'larger':
                            # Increase selection by 10%
                            additional_count = int(len(selected_features) * 0.1)
                            available_extra = [f for f in X_filtered.columns if f not in selected_features]
                            selected_features.extend(available_extra[:additional_count])
                            self._log(f"📈 Validation recommended larger set: {len(selected_features)} features")

                    # Store validation results
                    self.validation_results = validation_results

                # Store composite scores for analysis
                self.composite_scores = composite_scores
                self.sorted_features = sorted_features
                self.elbow_idx = elbow_idx

        # Fallback selections if composite scoring fails
        if not selected_features:
            if clusters and self.mda_results:
                # Fallback to MDA-based selection
                sorted_clusters = sorted(self.mda_results.items(), key=lambda x: x[1], reverse=True)
                top_cluster_names = [cluster for cluster, _ in sorted_clusters[:top_clusters]]
                cluster_features = []
                for cluster_name in top_cluster_names:
                    cluster_features.extend(clusters[cluster_name])
                selected_features = cluster_features[:50]
                self._log("⚠️ Using MDA-based fallback selection")

            elif self.shap_results:
                # SHAP-only selection
                n_select = min(50, len(X_filtered.columns) // 2)
                sorted_features = sorted(self.shap_results.items(), key=lambda x: x[1], reverse=True)
                selected_features = [f for f, _ in sorted_features[:n_select]]
                self._log("⚠️ Using SHAP-only fallback selection")

            else:
                # Final fallback
                selected_features = list(X_filtered.columns[:30])
                self._log("⚠️ Using random fallback selection")

        # --------------------------------------------------------------
        # Phase 4: Append forced regime-leaf context features
        # --------------------------------------------------------------
        # Force-keep regime leaf features (optional bypass setup).
        forced_features: List[str] = []
        if regime_leaf_feature_names:
            forced_features = [f for f in regime_leaf_feature_names if f not in selected_features]
            selected_features = list(selected_features) + forced_features

        self.selected_features = selected_features

        # Create rankings
        self.importance_rankings = {
            'mda_clusters': dict(sorted(self.mda_results.items(), key=lambda x: x[1], reverse=True)),
            'shap_features': dict(sorted(self.shap_results.items(), key=lambda x: x[1], reverse=True))
        }

        # Summary
        self._log(f"✅ Feature selection complete: {len(X_filtered.columns)} → {len(selected_features)} features")
        self._log(f"   📊 Selected from {len(clusters)} clusters using MDA × IR composite scoring + Elbow method")

        results: Dict[str, Any] = {}

        # Generate elbow plot data
        plot_data = self.plot_elbow_analysis()
        if plot_data:
            self._log("📈 Elbow analysis plot generated")
            results['elbow_plot_data'] = plot_data

        # Create detailed results
        results.update({
            'selected_features': selected_features,
            'n_features_original': len(X.columns),
            'n_features_after_prefilters': len(X_filtered.columns),
            'n_features_selected': len(selected_features),
            'forced_features': forced_features,
            'prefilter_counts': getattr(self, '_prefilter_counts', {}),
            'prefilter_features': getattr(self, '_prefilter_features', []),
            'clusters': clusters,
            'mda_results': self.mda_results,
            'shap_results': self.shap_results,
            'ic_results': getattr(self, 'ic_results', {}),
            'composite_scores': getattr(self, 'composite_scores', {}),
            'elbow_idx': getattr(self, 'elbow_idx', None),
            'validation_results': getattr(self, 'validation_results', {}),
            'importance_rankings': self.importance_rankings,
            'regime_leaf_features': {
                'enabled': bool(regime_leaf_info.get('enabled', False)),
                'info': dict(regime_leaf_info) if isinstance(regime_leaf_info, dict) else {},
                'feature_names': list(regime_leaf_feature_names),
            },
            'shap_interaction_features': {
                'enabled': bool(enable_shap_interaction_features),
                'interaction_defs': list(shap_interaction_defs) if shap_interaction_defs else [],
                'info': dict(shap_interaction_info) if isinstance(shap_interaction_info, dict) else {},
            },
            'config': {
                'model_type': self.model_type,
                'n_folds': self.n_folds,
                'corr_threshold': corr_threshold,
                'top_clusters': top_clusters,
                'subsample_train_pct': self.subsample_train_pct,
                'max_train_samples': self.max_train_samples,
                'subsample_features_pct': self.subsample_features_pct,
                'shap_max_evals': self.shap_max_evals
            }
        })

        return selected_features, results


def run_mda_shap_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    target_sample_weight: Optional[pd.Series] = None,
    config: Optional[Dict[str, Any]] = None,
    artifact_router: Any = None,
    pipeline_context: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Convenience function to run MDA/SHAP feature selection.

    Args:
        X: Feature matrix
        y: Target labels
        config: Configuration dictionary

    Returns:
        Tuple of (selected_features, results)
    """
    try:
        enable_specialists = bool(config.get("enable_specialist_features_for_selection", True)) if config else True
    except Exception:
        enable_specialists = True

    n_specialist_added = 0
    if enable_specialists and artifact_router is not None and isinstance(X, pd.DataFrame):
        try:
            from src.utils.ml_common.get_specialist_models_outputs import get_specialist_models_outputs

            base_cols = list(X.columns)

            cfg_for_specialists: Dict[str, Any] = {}
            if isinstance(pipeline_context, dict):
                cfg_for_specialists.update(pipeline_context)
            if isinstance(config, dict):
                cfg_for_specialists.update(config)

            specialists = get_specialist_models_outputs(
                artifact_router=artifact_router,
                training_index=pd.DatetimeIndex(X.index),
                config=cfg_for_specialists,
                logger=None,
                strict=False,
            )

            if specialists is not None and not getattr(specialists, "empty", True):
                # Avoid overwriting existing columns; keep base feature matrix authoritative.
                overlap = [c for c in specialists.columns if c in X.columns]
                if overlap:
                    specialists = specialists.drop(columns=overlap, errors="ignore")

                if specialists is not None and not getattr(specialists, "empty", True):
                    X = pd.concat([X, specialists], axis=1)
                    try:
                        n_specialist_added = int(len([c for c in X.columns if c not in set(base_cols)]))
                    except Exception:
                        n_specialist_added = 0
        except Exception:
            pass

    selector = MDA_SHAP_FeatureSelector(
        model_type=config.get("model_type", "rf") if config else "rf",
        n_folds=config.get("n_folds", 5) if config else 5,
        embargo_pct=config.get("embargo_pct", 0.01) if config else 0.01,
        random_state=config.get("random_state", 42) if config else 42,
        verbose=config.get("verbose", True) if config else True,
        # Subsampling parameters for computational efficiency
        subsample_train_pct=config.get("subsample_train_pct", 0.8) if config else 0.8,
        max_train_samples=config.get("max_train_samples", 10000) if config else 10000,
        subsample_features_pct=config.get("subsample_features_pct", 0.9) if config else 0.9,
        shap_max_evals=config.get("shap_max_evals", 1000) if config else 1000
    )

    selected_features, results = selector.select_features(
        X=X,
        y=y,
        target_sample_weight=target_sample_weight,
        regime_leaf_config=(config.get("regime_leaf_config") if config and isinstance(config.get("regime_leaf_config"), dict) else None),
        pre_filter_config=config.get("pre_filters", {}) if config else {},
        corr_threshold=config.get("corr_threshold", 0.85) if config else 0.85,
        top_clusters=config.get("top_clusters", 8) if config else 8,  # Consider more clusters for Elbow method
        shap_sample_size=config.get("shap_sample_size", 1000) if config else 1000,
        enable_shap_interaction_features=bool(config.get("enable_shap_interaction_features", False)) if config else False,
        shap_interaction_config=(
            config.get("shap_interaction_config")
            if config and isinstance(config.get("shap_interaction_config"), dict)
            else {}
        ),
        elbow_min_features=int(config.get("elbow_min_features", 10)) if config else 10,
    )

    try:
        results["specialist_features"] = {
            "enabled": bool(enable_specialists and artifact_router is not None),
            "n_added": int(n_specialist_added),
        }
    except Exception:
        pass

    return selected_features, results


# Export functions

# Export functions
__all__ = [
    "MDA_SHAP_FeatureSelector",
    "run_mda_shap_feature_selection"
]





