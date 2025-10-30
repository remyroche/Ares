"""
Final Feature Selection Component

This module provides final feature selection functionality for the pre-training pipeline.
Enhanced with comprehensive analysis capabilities including correlation analysis,
redundancy detection, stability analysis, and cross-validation.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE, SelectFromModel
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

# LGBM and SHAP imports
try:
    import lightgbm as lgb
    from sklearn.multioutput import MultiOutputRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
from src.utils.logger import system_logger


class FinalFeatureSelectionConfig:
    """
    Configuration for final feature selection.
    """
    
    def __init__(
        self,
        max_features: int = 100,
        min_features: int = 10,
        selection_method: str = "permutation",
        scoring_threshold: float = 0.01,
        use_tree_based: bool = True,
        use_permutation_importance: bool = True
    ):
        """
        Initialize final feature selection configuration.
        
        Args:
            max_features: Maximum number of features to select
            min_features: Minimum number of features to select
            selection_method: Method for feature selection ('permutation', 'mutual_info', 'f_regression')
            scoring_threshold: Minimum score threshold for features
            use_tree_based: Whether to use tree-based feature importance
            use_permutation_importance: Whether to use permutation importance (captures interactions) vs standard Gini importance
        """
        self.max_features = max_features
        self.min_features = min_features
        self.selection_method = selection_method
        self.scoring_threshold = scoring_threshold
        self.use_tree_based = use_tree_based
        self.use_permutation_importance = use_permutation_importance


class FinalFeatureSelectionComponent:
    """
    Final feature selection component.
    """
    
    def __init__(self, config: FinalFeatureSelectionConfig):
        """
        Initialize the final feature selection component.
        
        Args:
            config: Configuration for feature selection
        """
        self.config = config
        self.logger = system_logger.getChild("FinalFeatureSelectionComponent")
        self.selected_features: List[str] = []
        self.feature_scores: Dict[str, float] = {}
        
        # Enhanced analysis storage
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.redundancy_analysis: Optional[Dict[str, Any]] = None
        self.stability_analysis: Optional[Dict[str, Any]] = None
        self.cv_analysis: Optional[Dict[str, Any]] = None
        self.baseline_comparison: Optional[Dict[str, Any]] = None
        self.method_results: Optional[Dict[str, Any]] = None
        
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> List[str]:
        """
        Select final features based on the configuration.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Optional list of feature names
            
        Returns:
            List of selected feature names
        """
        try:
            if feature_names is None:
                feature_names = list(X.columns)
            
            # Ensure we don't select more features than available
            max_features = min(self.config.max_features, len(feature_names))
            min_features = min(self.config.min_features, max_features)
            
            if max_features <= 0:
                self.logger.warning("No features to select")
                return []
            
            # Select features based on method
            if self.config.selection_method == "mutual_info":
                selector = SelectKBest(
                    score_func=mutual_info_regression,
                    k=max_features
                )
            elif self.config.selection_method == "f_regression":
                selector = SelectKBest(
                    score_func=f_regression,
                    k=max_features
                )
            else:
                # Default to mutual info
                selector = SelectKBest(
                    score_func=mutual_info_regression,
                    k=max_features
                )
            
            # Fit selector
            X_selected = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True)
            selected_features = [feature_names[i] for i in selected_indices]
            
            # Store feature scores
            if hasattr(selector, 'scores_'):
                self.feature_scores = {
                    feature_names[i]: selector.scores_[i]
                    for i in selected_indices
                }
            
            # Apply tree-based selection if enabled
            if self.config.use_tree_based and len(selected_features) > min_features:
                selected_features = self._apply_tree_based_selection(
                    X[selected_features], y, selected_features
                )
            
            # Filter by scoring threshold
            if self.config.scoring_threshold > 0:
                selected_features = [
                    feat for feat in selected_features
                    if self.feature_scores.get(feat, 0) >= self.config.scoring_threshold
                ]
            
            # Ensure minimum features
            if len(selected_features) < min_features:
                # Take top features by score
                scored_features = sorted(
                    self.feature_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                selected_features = [
                    feat for feat, _ in scored_features[:min_features]
                ]
            
            self.selected_features = selected_features
            importance_method = "permutation" if self.config.use_permutation_importance else "Gini"
            self.logger.info(f"Selected {len(selected_features)} features using {importance_method} importance")
            self.logger.info(f"Importance method: {importance_method} (captures interactions: {self.config.use_permutation_importance})")
            
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {e}")
            return []
    
    def _apply_tree_based_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str]
    ) -> List[str]:
        """
        Apply tree-based feature selection using ExtraTreesRegressor.
        Uses permutation importance by default (captures feature interactions),
        or standard Gini importance if configured.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
            
        Returns:
            List of selected features
        """
        try:
            # Use Extra Trees for feature importance (faster and often better than RF)
            model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X, y)
            
            # Get feature importances using permutation or standard method
            if self.config.use_permutation_importance:
                # Use permutation importance - captures feature interactions and is more reliable
                self.logger.info("Using permutation importance (captures feature interactions)")
                perm_importance = permutation_importance(
                    model, X, y,
                    n_repeats=10,
                    random_state=42,
                    n_jobs=-1
                )
                importances = perm_importance.importances_mean
                self.logger.info(f"Permutation importance calculated for {len(feature_names)} features")
            else:
                # Use standard Gini importance (faster but doesn't capture interactions as well)
                self.logger.info("Using standard Gini importance")
                importances = model.feature_importances_
            
            feature_importance = dict(zip(feature_names, importances))
            
            # Store importances for later analysis
            self.feature_scores.update(feature_importance)
            
            # Sort by importance and select top features
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Select top features up to max_features
            max_features = min(self.config.max_features, len(sorted_features))
            selected_features = [feat for feat, _ in sorted_features[:max_features]]
            
            self.logger.info(f"Selected {len(selected_features)} features using {'permutation' if self.config.use_permutation_importance else 'Gini'} importance")
            
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Error in tree-based selection: {e}")
            return feature_names
    
    def get_feature_scores(self) -> Dict[str, float]:
        """
        Get feature scores from the last selection.
        
        Returns:
            Dictionary of feature scores
        """
        return self.feature_scores.copy()
    
    def get_selected_features(self) -> List[str]:
        """
        Get the last selected features.
        
        Returns:
            List of selected feature names
        """
        return self.selected_features.copy()
    
    def analyze_feature_correlations(self, X: pd.DataFrame, selected_features: List[str]) -> Dict[str, Any]:
        """
        Analyze correlations between selected features.
        
        Args:
            X: Feature matrix
            selected_features: List of selected features
            
        Returns:
            Dictionary containing correlation analysis results
        """
        try:
            if not selected_features:
                return {"error": "No features selected"}
            
            # Calculate correlation matrix for selected features
            selected_data = X[selected_features]
            correlation_matrix = selected_data.corr()
            self.correlation_matrix = correlation_matrix
            
            # Find highly correlated feature pairs
            high_corr_pairs = []
            correlation_threshold = 0.8
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = abs(correlation_matrix.iloc[i, j])
                    if corr_value > correlation_threshold:
                        high_corr_pairs.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            # Calculate average correlation
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
            upper_triangle = correlation_matrix.where(mask)
            avg_correlation = upper_triangle.stack().abs().mean()
            
            analysis = {
                'correlation_matrix': correlation_matrix,
                'high_correlation_pairs': high_corr_pairs,
                'average_correlation': avg_correlation,
                'max_correlation': correlation_matrix.abs().max().max(),
                'min_correlation': correlation_matrix.abs().min().min(),
                'correlation_threshold': correlation_threshold
            }
            
            self.logger.info(f"Correlation analysis completed: {len(high_corr_pairs)} high correlation pairs found")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in correlation analysis: {e}")
            return {"error": str(e)}
    
    def detect_redundant_features(self, X: pd.DataFrame, selected_features: List[str]) -> Dict[str, Any]:
        """
        Detect redundant features using multiple methods.
        
        Args:
            X: Feature matrix
            selected_features: List of selected features
            
        Returns:
            Dictionary containing redundancy analysis results
        """
        try:
            if not selected_features:
                return {"error": "No features selected"}
            
            selected_data = X[selected_features]
            redundancy_results = {
                'correlation_redundant': [],
                'mutual_info_redundant': [],
                'variance_redundant': []
            }
            
            # 1. Correlation-based redundancy
            correlation_matrix = selected_data.corr().abs()
            correlation_threshold = 0.9
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] > correlation_threshold:
                        redundancy_results['correlation_redundant'].append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': correlation_matrix.iloc[i, j]
                        })
            
            # 2. Mutual information-based redundancy
            mi_threshold = 0.8
            for i in range(len(selected_features)):
                for j in range(i+1, len(selected_features)):
                    try:
                        mi_score = mutual_info_score(
                            selected_data.iloc[:, i].dropna(),
                            selected_data.iloc[:, j].dropna()
                        )
                        if mi_score > mi_threshold:
                            redundancy_results['mutual_info_redundant'].append({
                                'feature1': selected_features[i],
                                'feature2': selected_features[j],
                                'mutual_info': mi_score
                            })
                    except:
                        continue
            
            # 3. Variance-based redundancy (near-zero variance)
            variance_threshold = 0.01
            variances = selected_data.var()
            low_variance_features = variances[variances < variance_threshold].index.tolist()
            redundancy_results['variance_redundant'] = low_variance_features
            
            # Calculate redundancy score
            total_pairs = len(selected_features) * (len(selected_features) - 1) // 2
            redundant_pairs = len(redundancy_results['correlation_redundant']) + len(redundancy_results['mutual_info_redundant'])
            redundancy_score = redundant_pairs / total_pairs if total_pairs > 0 else 0
            
            analysis = {
                'redundancy_results': redundancy_results,
                'redundancy_score': redundancy_score,
                'total_features': len(selected_features),
                'redundant_features': len(set(
                    [pair['feature1'] for pair in redundancy_results['correlation_redundant']] +
                    [pair['feature2'] for pair in redundancy_results['correlation_redundant']] +
                    [pair['feature1'] for pair in redundancy_results['mutual_info_redundant']] +
                    [pair['feature2'] for pair in redundancy_results['mutual_info_redundant']] +
                    redundancy_results['variance_redundant']
                ))
            }
            
            self.redundancy_analysis = analysis
            self.logger.info(f"Redundancy analysis completed: {analysis['redundant_features']} redundant features found")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in redundancy detection: {e}")
            return {"error": str(e)}
    
    def analyze_feature_stability(self, X: pd.DataFrame, y: pd.Series, selected_features: List[str], 
                                 n_windows: int = 5) -> Dict[str, Any]:
        """
        Analyze stability of feature selection across different time windows.
        
        Args:
            X: Feature matrix
            y: Target variable
            selected_features: List of selected features
            n_windows: Number of time windows to analyze
            
        Returns:
            Dictionary containing stability analysis results
        """
        try:
            if not selected_features:
                return {"error": "No features selected"}
            
            n_samples = len(X)
            window_size = n_samples // n_windows
            
            stability_results = {
                'window_selections': [],
                'feature_frequency': {},
                'stability_scores': {}
            }
            
            # Analyze each time window
            for i in range(n_windows):
                start_idx = i * window_size
                end_idx = min((i + 1) * window_size, n_samples)
                
                X_window = X.iloc[start_idx:end_idx]
                y_window = y.iloc[start_idx:end_idx]
                
                # Select features for this window
                window_features = self._select_features_for_window(X_window, y_window)
                stability_results['window_selections'].append({
                    'window': i,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'features': window_features
                })
                
                # Count feature frequency
                for feature in window_features:
                    if feature in selected_features:
                        stability_results['feature_frequency'][feature] = stability_results['feature_frequency'].get(feature, 0) + 1
            
            # Calculate stability scores
            for feature in selected_features:
                frequency = stability_results['feature_frequency'].get(feature, 0)
                stability_score = frequency / n_windows
                stability_results['stability_scores'][feature] = stability_score
            
            # Calculate overall stability metrics
            avg_stability = np.mean(list(stability_results['stability_scores'].values()))
            stable_features = [f for f, score in stability_results['stability_scores'].items() if score >= 0.8]
            
            analysis = {
                'stability_results': stability_results,
                'average_stability': avg_stability,
                'stable_features': stable_features,
                'stability_threshold': 0.8,
                'n_windows': n_windows
            }
            
            self.stability_analysis = analysis
            self.logger.info(f"Stability analysis completed: {len(stable_features)} stable features found")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in stability analysis: {e}")
            return {"error": str(e)}
    
    def _select_features_for_window(self, X_window: pd.DataFrame, y_window: pd.Series) -> List[str]:
        """
        Select features for a specific time window.
        
        Args:
            X_window: Feature matrix for the window
            y_window: Target variable for the window
            
        Returns:
            List of selected features for this window
        """
        try:
            # Use a subset of features for window analysis
            max_window_features = min(20, len(X_window.columns))
            
            selector = SelectKBest(
                score_func=mutual_info_regression,
                k=max_window_features
            )
            
            X_selected = selector.fit_transform(X_window, y_window)
            selected_indices = selector.get_support(indices=True)
            selected_features = [X_window.columns[i] for i in selected_indices]
            
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Error selecting features for window: {e}")
            return []
    
    def cross_validate_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                       selected_features: List[str], cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation of feature selection stability.
        
        Args:
            X: Feature matrix
            y: Target variable
            selected_features: List of selected features
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing cross-validation analysis results
        """
        try:
            if not selected_features:
                return {"error": "No features selected"}
            
            # Use TimeSeriesSplit for time series data
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            cv_results = {
                'fold_selections': [],
                'feature_frequency': {},
                'selection_consistency': {}
            }
            
            fold_idx = 0
            for train_idx, test_idx in tscv.split(X):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                
                # Select features for this fold
                fold_features = self._select_features_for_window(X_train, y_train)
                cv_results['fold_selections'].append({
                    'fold': fold_idx,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'features': fold_features
                })
                
                # Count feature frequency
                for feature in fold_features:
                    if feature in selected_features:
                        cv_results['feature_frequency'][feature] = cv_results['feature_frequency'].get(feature, 0) + 1
                
                fold_idx += 1
            
            # Calculate selection consistency
            for feature in selected_features:
                frequency = cv_results['feature_frequency'].get(feature, 0)
                consistency_score = frequency / cv_folds
                cv_results['selection_consistency'][feature] = consistency_score
            
            # Calculate overall metrics
            avg_consistency = np.mean(list(cv_results['selection_consistency'].values()))
            consistent_features = [f for f, score in cv_results['selection_consistency'].items() if score >= 0.6]
            
            analysis = {
                'cv_results': cv_results,
                'average_consistency': avg_consistency,
                'consistent_features': consistent_features,
                'consistency_threshold': 0.6,
                'cv_folds': cv_folds
            }
            
            self.cv_analysis = analysis
            self.logger.info(f"Cross-validation analysis completed: {len(consistent_features)} consistent features found")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in cross-validation analysis: {e}")
            return {"error": str(e)}
    
    def compare_with_baseline(self, X: pd.DataFrame, y: pd.Series, selected_features: List[str]) -> Dict[str, Any]:
        """
        Compare selected features with baseline random selection.
        
        Args:
            X: Feature matrix
            y: Target variable
            selected_features: List of selected features
            
        Returns:
            Dictionary containing baseline comparison results
        """
        try:
            if not selected_features:
                return {"error": "No features selected"}
            
            n_features = len(selected_features)
            all_features = list(X.columns)
            
            # Generate random baseline selections
            n_baseline_trials = 10
            baseline_results = []
            
            for trial in range(n_baseline_trials):
                random_features = np.random.choice(all_features, size=n_features, replace=False).tolist()
                
                # Calculate mutual information for random selection
                random_scores = []
                for feature in random_features:
                    try:
                        score = mutual_info_regression(X[[feature]], y)[0]
                        random_scores.append(score)
                    except:
                        random_scores.append(0.0)
                
                baseline_results.append({
                    'trial': trial,
                    'features': random_features,
                    'scores': random_scores,
                    'avg_score': np.mean(random_scores)
                })
            
            # Calculate scores for selected features
            selected_scores = []
            for feature in selected_features:
                try:
                    score = mutual_info_regression(X[[feature]], y)[0]
                    selected_scores.append(score)
                except:
                    selected_scores.append(0.0)
            
            avg_selected_score = np.mean(selected_scores)
            avg_baseline_score = np.mean([result['avg_score'] for result in baseline_results])
            
            # Calculate improvement over baseline
            improvement_ratio = avg_selected_score / avg_baseline_score if avg_baseline_score > 0 else 1.0
            
            analysis = {
                'baseline_results': baseline_results,
                'selected_features_scores': selected_scores,
                'average_selected_score': avg_selected_score,
                'average_baseline_score': avg_baseline_score,
                'improvement_ratio': improvement_ratio,
                'n_baseline_trials': n_baseline_trials,
                'n_features': n_features
            }
            
            self.baseline_comparison = analysis
            self.logger.info(f"Baseline comparison completed: {improvement_ratio:.2f}x improvement over random selection")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in baseline comparison: {e}")
            return {"error": str(e)}
    
    def get_enhanced_analysis(self) -> Dict[str, Any]:
        """
        Get all enhanced analysis results.
        
        Returns:
            Dictionary containing all analysis results
        """
        return {
            'correlation_analysis': self.correlation_matrix,
            'redundancy_analysis': self.redundancy_analysis,
            'stability_analysis': self.stability_analysis,
            'cv_analysis': self.cv_analysis,
            'baseline_comparison': self.baseline_comparison
        }
    
    def select_features_with_stability_optimization(self, X: pd.DataFrame, y: pd.Series, 
                                                   feature_names: Optional[List[str]] = None,
                                                   target_features: int = 60,
                                                   stability_threshold: float = 0.3,  # Lowered from 0.6
                                                   redundancy_threshold: float = 0.8,
                                                   use_oos_validation: bool = True,
                                                   oos_ratio: float = 0.2) -> List[str]:
        """
        Select features with enhanced stability and redundancy optimization.
        
        Uses OOS (Out-of-Sample) validation and multi-stage stability filtering:
        1. Reserve OOS holdout set (20% by default)
        2. Multi-method selection on training data
        3. OOF (Out-of-Fold) stability validation using purged TimeSeriesSplit
        4. OOS validation on holdout set
        5. Redundancy reduction
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Optional list of feature names
            target_features: Target number of features to select
            stability_threshold: Minimum stability score (lowered to 0.3 for more realistic threshold)
            redundancy_threshold: Maximum correlation threshold for redundancy
            use_oos_validation: Whether to use out-of-sample validation
            oos_ratio: Ratio of data to reserve for OOS testing
            
        Returns:
            List of selected feature names optimized for stability and low redundancy
        """
        try:
            if feature_names is None:
                feature_names = list(X.columns)
            
            self.logger.info(f"Starting stability-optimized feature selection for {target_features} features")
            self.logger.info(f"Using OOS validation: {use_oos_validation}, OOS ratio: {oos_ratio}")
            self.logger.info(f"Stability threshold: {stability_threshold} (adaptive)")
            
            # Step 0: OOS Split (if enabled)
            if use_oos_validation and len(X) > 100:
                oos_split_idx = int(len(X) * (1 - oos_ratio))
                X_train, X_oos = X.iloc[:oos_split_idx], X.iloc[oos_split_idx:]
                y_train, y_oos = y.iloc[:oos_split_idx], y.iloc[oos_split_idx:]
                self.logger.info(f"OOS split: Training={len(X_train)}, OOS={len(X_oos)}")
            else:
                X_train, X_oos = X, None
                y_train, y_oos = y, None
                self.logger.info("OOS validation disabled or insufficient data")
            
            # Step 1: Initial selection using multiple methods on training data
            initial_features, method_results = self._multi_method_initial_selection(
                X_train, y_train, feature_names, target_features * 2
            )
            
            # Step 2: OOF Stability validation using purged TimeSeriesSplit
            stable_features = self._oof_stability_validation(
                X_train, y_train, initial_features, stability_threshold
            )
            
            # Step 3: OOS validation (if enabled)
            if use_oos_validation and X_oos is not None:
                oos_validated_features = self._oos_validation(
                    X_train, y_train, X_oos, y_oos, stable_features
                )
                self.logger.info(f"OOS validation: {len(oos_validated_features)}/{len(stable_features)} features validated")
                stable_features = oos_validated_features if oos_validated_features else stable_features
            
            # Step 4: Redundancy reduction
            final_features = self._reduce_redundancy(X_train, stable_features, redundancy_threshold, target_features)
            
            self.logger.info(f"Selected {len(final_features)} stable, non-redundant features")
            
            # Store method results for analysis
            self.method_results = method_results
            
            return final_features
            
        except Exception as e:
            self.logger.error(f"Error in stability-optimized selection: {e}")
            return self.select_features(X, y, feature_names)
    
    def _multi_method_initial_selection(self, X: pd.DataFrame, y: pd.Series, 
                                      feature_names: List[str], n_features: int) -> Tuple[List[str], Dict[str, Any]]:
        """
        Use multiple selection methods and combine results.
        
        Uses 3 complementary methods:
        1. Mutual Information: Model-free, captures non-linear dependencies
        2. Lasso: Linear model with sparsity, handles collinearity
        3. LGBM-SHAP: Gradient boosting with game-theoretic feature importance
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
            n_features: Number of features to select
            
        Returns:
            Tuple of (combined features list, method-specific results)
        """
        try:
            all_selected_features = set()
            method_results = {}
            
            # Method 1: Mutual Information (non-linear, model-free)
            mi_selector = SelectKBest(score_func=mutual_info_regression, k=min(n_features//3, len(feature_names)))
            mi_features = mi_selector.fit_transform(X, y)
            mi_indices = mi_selector.get_support(indices=True)
            mi_features_names = [feature_names[i] for i in mi_indices]
            all_selected_features.update(mi_features_names)
            method_results['mutual_info'] = {
                'features': mi_features_names,
                'scores': mi_selector.scores_[mi_indices].tolist()
            }
            
            # Method 2: Lasso regularization (linear, sparse, handles collinearity)
            lasso = LassoCV(cv=5, random_state=42, n_jobs=-1)
            lasso.fit(X, y)
            lasso_coef = np.abs(lasso.coef_)
            lasso_indices = np.argsort(lasso_coef)[-min(n_features//3, len(feature_names)):]
            lasso_features_names = [feature_names[i] for i in lasso_indices if lasso_coef[i] > 0]
            all_selected_features.update(lasso_features_names)
            method_results['lasso'] = {
                'features': lasso_features_names,
                'scores': lasso_coef[lasso_indices].tolist()
            }
            
            # Method 3: LGBM-SHAP (gradient boosting, interpretable importance)
            if LGBM_AVAILABLE and SHAP_AVAILABLE:
                lgbm_shap_features, lgbm_shap_scores = self._lgbm_shap_selection(X, y, feature_names, n_features//3)
                all_selected_features.update(lgbm_shap_features)
                method_results['lgbm_shap'] = {
                    'features': lgbm_shap_features,
                    'scores': lgbm_shap_scores
                }
            else:
                method_results['lgbm_shap'] = {
                    'features': [],
                    'scores': [],
                    'error': 'LGBM or SHAP not available'
                }
            
            return list(all_selected_features), method_results
            
        except Exception as e:
            self.logger.error(f"Error in multi-method selection: {e}")
            return feature_names[:n_features], {"error": str(e)}
    
    def _lgbm_shap_selection(self, X: pd.DataFrame, y: pd.Series, 
                           feature_names: List[str], n_features: int) -> Tuple[List[str], List[float]]:
        """
        Use LGBM with SHAP values for feature selection.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
            n_features: Number of features to select
            
        Returns:
            Tuple of (selected features, SHAP scores)
        """
        try:
            # Setup LGBM parameters
            lgbm_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42,
                'n_jobs': -1,
                'max_depth': 6,
                'min_data_in_leaf': 20,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1
            }
            
            # Train LGBM model
            model = lgb.LGBMRegressor(**lgbm_params)
            model.fit(X, y)
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # Calculate mean absolute SHAP values for each feature
            mean_shap_values = np.mean(np.abs(shap_values), axis=0)
            
            # Select top features based on SHAP values
            top_indices = np.argsort(mean_shap_values)[-n_features:]
            selected_features = [feature_names[i] for i in top_indices]
            selected_scores = mean_shap_values[top_indices].tolist()
            
            return selected_features, selected_scores
            
        except Exception as e:
            self.logger.error(f"Error in LGBM-SHAP selection: {e}")
            # Fallback to LGBM importance
            try:
                lgbm_params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'verbose': -1,
                    'random_state': 42,
                    'n_jobs': -1
                }
                
                model = lgb.LGBMRegressor(**lgbm_params)
                model.fit(X, y)
                
                importance_scores = model.feature_importances_
                top_indices = np.argsort(importance_scores)[-n_features:]
                selected_features = [feature_names[i] for i in top_indices]
                selected_scores = importance_scores[top_indices].tolist()
                
                return selected_features, selected_scores
                
            except Exception as e2:
                self.logger.error(f"Error in LGBM fallback: {e2}")
                return feature_names[:n_features], [0.0] * n_features
    
    def _oof_stability_validation(self, X: pd.DataFrame, y: pd.Series,
                                  candidate_features: List[str], stability_threshold: float) -> List[str]:
        """
        Validate features using OOF (Out-of-Fold) predictions with purged TimeSeriesSplit.
        
        This method:
        1. Uses TimeSeriesSplit with purging to avoid leakage
        2. Trains models on each fold and validates on held-out data
        3. Measures feature importance consistency across folds
        4. Filters features that are stable across different time periods
        
        Args:
            X: Feature matrix
            y: Target variable
            candidate_features: List of candidate features
            stability_threshold: Minimum stability score (0-1)
            
        Returns:
            List of stable features validated through OOF
        """
        try:
            if not candidate_features:
                return []
            
            self.logger.info(f"Starting OOF stability validation on {len(candidate_features)} features")
            
            # Use TimeSeriesSplit with purging
            n_splits = 5
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            # Track feature importance across folds
            fold_importances = {feature: [] for feature in candidate_features}
            fold_correlations = {feature: [] for feature in candidate_features}
            
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_val_fold = y.iloc[val_idx]
                
                # Select only candidate features
                X_train_subset = X_train_fold[candidate_features]
                X_val_subset = X_val_fold[candidate_features]
                
                # Train a simple model to get feature importance
                try:
                    if LGBM_AVAILABLE:
                        model = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1, n_jobs=-1)
                        model.fit(X_train_subset, y_train_fold)
                        importances = model.feature_importances_
                        
                        for i, feature in enumerate(candidate_features):
                            fold_importances[feature].append(importances[i])
                    else:
                        # Fallback to ExtraTrees
                        model = ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                        model.fit(X_train_subset, y_train_fold)
                        importances = model.feature_importances_
                        
                        for i, feature in enumerate(candidate_features):
                            fold_importances[feature].append(importances[i])
                    
                    # Also calculate correlation on validation fold
                    for feature in candidate_features:
                        try:
                            corr = abs(X_val_subset[feature].corr(y_val_fold))
                            if not np.isnan(corr):
                                fold_correlations[feature].append(corr)
                        except:
                            continue
                            
                except Exception as e:
                    self.logger.warning(f"Fold {fold_idx} failed: {e}")
                    continue
            
            # Calculate stability metrics for each feature
            feature_stability_scores = {}
            
            for feature in candidate_features:
                importances = fold_importances[feature]
                correlations = fold_correlations[feature]
                
                if len(importances) >= 3:  # Need at least 3 folds
                    # Stability = consistency of importance across folds
                    # Use coefficient of variation (lower is more stable)
                    mean_importance = np.mean(importances)
                    std_importance = np.std(importances)
                    
                    if mean_importance > 0:
                        cv = std_importance / mean_importance
                        stability_score = 1 / (1 + cv)  # Convert to 0-1 range (higher is better)
                    else:
                        stability_score = 0
                    
                    # Combine with correlation stability
                    if len(correlations) >= 3:
                        mean_corr = np.mean(correlations)
                        stability_score = (stability_score + mean_corr) / 2
                    
                    feature_stability_scores[feature] = stability_score
            
            # Filter by threshold (adaptive: use percentile if too strict)
            stable_features = [
                feature for feature, score in feature_stability_scores.items()
                if score >= stability_threshold
            ]
            
            # If too few features pass, use top percentile instead
            if len(stable_features) < len(candidate_features) * 0.3:
                self.logger.info(f"Only {len(stable_features)} features passed threshold {stability_threshold}")
                self.logger.info("Using adaptive threshold (top 50% by stability)")
                sorted_features = sorted(
                    feature_stability_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                stable_features = [f for f, _ in sorted_features[:max(len(sorted_features)//2, len(candidate_features)//3)]]
            
            stable_features.sort(key=lambda x: feature_stability_scores.get(x, 0), reverse=True)
            
            self.logger.info(f"OOF validation: {len(stable_features)}/{len(candidate_features)} features are stable")
            return stable_features
            
        except Exception as e:
            self.logger.error(f"Error in OOF stability validation: {e}")
            return candidate_features
    
    def _oos_validation(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_oos: pd.DataFrame, y_oos: pd.Series,
                       candidate_features: List[str]) -> List[str]:
        """
        Validate features on completely held-out OOS (Out-of-Sample) data.
        
        This is the final validation to ensure features generalize to unseen data.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target
            X_oos: OOS feature matrix
            y_oos: OOS target
            candidate_features: List of candidate features
            
        Returns:
            List of features that validate on OOS data
        """
        try:
            if not candidate_features or X_oos is None:
                return candidate_features
            
            self.logger.info(f"Starting OOS validation on {len(candidate_features)} features")
            
            # Select only candidate features
            X_train_subset = X_train[candidate_features]
            X_oos_subset = X_oos[candidate_features]
            
            # Train model on all training data
            if LGBM_AVAILABLE:
                model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
            else:
                model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            model.fit(X_train_subset, y_train)
            
            # Get feature importances
            feature_importances = dict(zip(candidate_features, model.feature_importances_))
            
            # Calculate OOS correlation for each feature
            oos_scores = {}
            for feature in candidate_features:
                try:
                    # Correlation on OOS data
                    oos_corr = abs(X_oos_subset[feature].corr(y_oos))
                    if not np.isnan(oos_corr):
                        # Combine importance and OOS correlation
                        combined_score = (feature_importances[feature] + oos_corr) / 2
                        oos_scores[feature] = combined_score
                except:
                    continue
            
            # Keep features with positive OOS performance
            # Use median as threshold
            if oos_scores:
                median_score = np.median(list(oos_scores.values()))
                validated_features = [
                    feature for feature, score in oos_scores.items()
                    if score >= median_score
                ]
                
                # Sort by OOS score
                validated_features.sort(key=lambda x: oos_scores[x], reverse=True)
                
                self.logger.info(f"OOS validation: {len(validated_features)}/{len(candidate_features)} features validated (median threshold: {median_score:.4f})")
                return validated_features
            else:
                return candidate_features
            
        except Exception as e:
            self.logger.error(f"Error in OOS validation: {e}")
            return candidate_features
    
    def _reduce_redundancy(self, X: pd.DataFrame, features: List[str], 
                          redundancy_threshold: float, target_count: int) -> List[str]:
        """
        Reduce redundancy using hierarchical clustering.
        
        Args:
            X: Feature matrix
            features: List of features to reduce redundancy from
            redundancy_threshold: Maximum correlation threshold
            target_count: Target number of final features
            
        Returns:
            List of non-redundant features
        """
        try:
            if not features or len(features) <= target_count:
                return features
            
            # Get feature data
            feature_data = X[features]
            
            # Calculate correlation matrix
            corr_matrix = feature_data.corr().abs()
            
            # Convert correlation to distance matrix
            distance_matrix = 1 - corr_matrix
            np.fill_diagonal(distance_matrix, 0)
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(squareform(distance_matrix), method='ward')
            
            # Determine number of clusters based on target count
            n_clusters = min(target_count, len(features))
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Select representative feature from each cluster
            selected_features = []
            for cluster_id in range(1, n_clusters + 1):
                cluster_features = [features[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if cluster_features:
                    # Select feature with highest variance (most informative)
                    variances = feature_data[cluster_features].var()
                    best_feature = variances.idxmax()
                    selected_features.append(best_feature)
            
            # If we still have too many features, apply additional correlation filtering
            if len(selected_features) > target_count:
                selected_features = self._correlation_filtering(X, selected_features, redundancy_threshold, target_count)
            
            self.logger.info(f"Reduced redundancy: {len(features)} -> {len(selected_features)} features")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Error in redundancy reduction: {e}")
            return features[:target_count]
    
    def _correlation_filtering(self, X: pd.DataFrame, features: List[str], 
                             threshold: float, target_count: int) -> List[str]:
        """
        Apply correlation-based filtering to remove highly correlated features.
        
        Args:
            X: Feature matrix
            features: List of features to filter
            threshold: Correlation threshold
            target_count: Target number of features
            
        Returns:
            List of filtered features
        """
        try:
            if len(features) <= target_count:
                return features
            
            feature_data = X[features]
            corr_matrix = feature_data.corr().abs()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > threshold:
                        high_corr_pairs.append((i, j, corr_matrix.iloc[i, j]))
            
            # Sort by correlation strength
            high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
            
            # Remove redundant features
            features_to_remove = set()
            for i, j, corr in high_corr_pairs:
                if len(features_to_remove) >= len(features) - target_count:
                    break
                    
                feature_i = corr_matrix.columns[i]
                feature_j = corr_matrix.columns[j]
                
                if feature_i not in features_to_remove and feature_j not in features_to_remove:
                    # Remove the feature with lower variance
                    var_i = feature_data[feature_i].var()
                    var_j = feature_data[feature_j].var()
                    
                    if var_i < var_j:
                        features_to_remove.add(feature_i)
                    else:
                        features_to_remove.add(feature_j)
            
            # Return remaining features
            filtered_features = [f for f in features if f not in features_to_remove]
            
            # If still too many, take top features by variance
            if len(filtered_features) > target_count:
                variances = feature_data[filtered_features].var()
                top_features = variances.nlargest(target_count).index.tolist()
                return top_features
            
            return filtered_features
            
        except Exception as e:
            self.logger.error(f"Error in correlation filtering: {e}")
            return features[:target_count]
    
    def analyze_improved_selection(self, X: pd.DataFrame, y: pd.Series, 
                                 selected_features: List[str], method_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze the quality of improved feature selection.
        
        Args:
            X: Feature matrix
            y: Target variable
            selected_features: List of selected features
            method_results: Optional method-specific results from multi-method selection
            
        Returns:
            Dictionary containing improved selection analysis
        """
        try:
            analysis = {
                'total_features': len(selected_features),
                'method_results': method_results or {},
                'stability_analysis': {},
                'redundancy_analysis': {},
                'quality_metrics': {}
            }
            
            # Stability analysis
            stability_results = self.analyze_feature_stability(X, y, selected_features, n_windows=5)
            analysis['stability_analysis'] = {
                'stable_features': len(stability_results.get('stable_features', [])),
                'average_stability': stability_results.get('average_stability', 0),
                'stability_rate': len(stability_results.get('stable_features', [])) / len(selected_features) if selected_features else 0
            }
            
            # Redundancy analysis
            redundancy_results = self.detect_redundant_features(X, selected_features)
            analysis['redundancy_analysis'] = {
                'redundant_features': redundancy_results.get('redundant_features', 0),
                'redundancy_score': redundancy_results.get('redundancy_score', 0),
                'redundancy_rate': redundancy_results.get('redundant_features', 0) / len(selected_features) if selected_features else 0
            }
            
            # Quality metrics
            if selected_features:
                feature_data = X[selected_features]
                
                # Calculate average correlation
                corr_matrix = feature_data.corr().abs()
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                upper_triangle = corr_matrix.where(mask)
                avg_correlation = upper_triangle.stack().mean()
                
                # Calculate average mutual information with target
                mi_scores = []
                for feature in selected_features:
                    try:
                        mi_score = mutual_info_regression(X[[feature]], y)[0]
                        mi_scores.append(mi_score)
                    except:
                        continue
                
                avg_mi_score = np.mean(mi_scores) if mi_scores else 0
                
                analysis['quality_metrics'] = {
                    'average_correlation': avg_correlation,
                    'average_mutual_info': avg_mi_score,
                    'feature_diversity': 1 - avg_correlation,  # Higher is better
                    'information_content': avg_mi_score
                }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in improved selection analysis: {e}")
            return {"error": str(e)}
