"""
Feature Scorecard - Detailed Metrics for Feature Assessment

This module provides comprehensive feature quality assessment with detailed
metrics across multiple dimensions.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from scipy.stats import spearmanr, kendalltau
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import warnings

logger = logging.getLogger(__name__)

class FeatureScorecard:
    """
    Comprehensive feature quality assessment with detailed metrics.
    """
    
    def __init__(self, 
                 stability_threshold: float = 0.4,
                 regime_delta_threshold: int = 10,
                 correlation_threshold: float = 0.9,
                 psi_threshold: float = 0.2,
                 n_bootstrap: int = 10):
        """
        Initialize feature scorecard.
        
        Args:
            stability_threshold: Minimum Spearman correlation for stability
            regime_delta_threshold: Maximum rank delta for regime stability
            correlation_threshold: Maximum correlation for uniqueness
            psi_threshold: Maximum PSI for drift detection
            n_bootstrap: Number of bootstrap samples
        """
        self.stability_threshold = stability_threshold
        self.regime_delta_threshold = regime_delta_threshold
        self.correlation_threshold = correlation_threshold
        self.psi_threshold = psi_threshold
        self.n_bootstrap = n_bootstrap
        
        # Results storage
        self.scorecard_results = {}
        self.feature_quality_scores = {}
    
    def compute_feature_scorecard(self, X: pd.DataFrame, y: pd.Series,
                                 feature_families: Optional[Dict[str, List[str]]] = None,
                                 time_splits: Optional[List[Tuple[int, int]]] = None) -> Dict[str, Any]:
        """
        Compute comprehensive feature scorecard.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_families: Dictionary mapping family names to feature lists
            time_splits: List of (start, end) tuples for temporal analysis
            
        Returns:
            Feature scorecard results
        """
        logger.info("Computing feature scorecard...")
        
        if feature_families is None:
            feature_families = self._auto_detect_families(X.columns)
        
        # A) Predictive signal metrics
        logger.info("Computing predictive signal metrics...")
        predictive_metrics = self._compute_predictive_signal_metrics(X, y, feature_families)
        
        # B) Stability & robustness metrics
        logger.info("Computing stability & robustness metrics...")
        stability_metrics = self._compute_stability_metrics(X, y, time_splits)
        
        # C) Uniqueness & redundancy metrics
        logger.info("Computing uniqueness & redundancy metrics...")
        uniqueness_metrics = self._compute_uniqueness_metrics(X, y)
        
        # D) Risk & hygiene metrics
        logger.info("Computing risk & hygiene metrics...")
        risk_metrics = self._compute_risk_metrics(X, y)
        
        # Compute Feature Quality Score (FQS)
        logger.info("Computing Feature Quality Scores...")
        fqs_results = self._compute_feature_quality_scores(
            predictive_metrics, stability_metrics, uniqueness_metrics, risk_metrics
        )
        
        # Compile results
        results = {
            'predictive_metrics': predictive_metrics,
            'stability_metrics': stability_metrics,
            'uniqueness_metrics': uniqueness_metrics,
            'risk_metrics': risk_metrics,
            'feature_quality_scores': fqs_results,
            'summary': self._generate_scorecard_summary(fqs_results)
        }
        
        logger.info("Feature scorecard computation completed")
        return results
    
    def _compute_predictive_signal_metrics(self, X: pd.DataFrame, y: pd.Series,
                                         feature_families: Dict[str, List[str]]) -> Dict[str, Any]:
        """Compute predictive signal metrics."""
        metrics = {}
        
        # Determine task type
        is_classification = y.dtype.name == 'category' or y.dtype == 'object'
        
        for family_name, family_features in feature_families.items():
            family_metrics = {}
            
            for feature in family_features:
                if feature not in X.columns:
                    continue
                
                feature_data = X[feature].dropna()
                common_idx = feature_data.index.intersection(y.index)
                if len(common_idx) == 0:
                    continue
                
                feature_aligned = feature_data.loc[common_idx]
                y_aligned = y.loc[common_idx]
                
                valid_mask = ~(feature_aligned.isna() | y_aligned.isna())
                if valid_mask.sum() < 10:
                    continue
                
                feature_clean = feature_aligned[valid_mask]
                y_clean = y_aligned[valid_mask]
                
                # kNN MI to target
                try:
                    if is_classification:
                        mi_score = mutual_info_classif(feature_clean.values.reshape(-1, 1), y_clean, random_state=42)[0]
                    else:
                        mi_score = mutual_info_regression(feature_clean.values.reshape(-1, 1), y_clean, random_state=42)[0]
                except:
                    mi_score = 0.0
                
                # Distance correlation
                try:
                    dist_corr = self._compute_distance_correlation(feature_clean, y_clean)
                except:
                    dist_corr = 0.0
                
                # Univariate CV score
                try:
                    cv_score = self._compute_univariate_cv_score(feature_clean, y_clean, is_classification)
                except:
                    cv_score = 0.0
                
                # Permutation importance (simplified)
                try:
                    perm_imp = self._compute_permutation_importance_single(feature_clean, y_clean, is_classification)
                except:
                    perm_imp = 0.0
                
                family_metrics[feature] = {
                    'mi_score': mi_score,
                    'distance_correlation': dist_corr,
                    'cv_score': cv_score,
                    'permutation_importance': perm_imp,
                    'family': family_name
                }
            
            metrics[family_name] = family_metrics
        
        return metrics
    
    def _compute_stability_metrics(self, X: pd.DataFrame, y: pd.Series,
                                  time_splits: Optional[List[Tuple[int, int]]] = None) -> Dict[str, Any]:
        """Compute stability & robustness metrics."""
        if time_splits is None:
            # Create default time splits
            n_samples = len(X)
            n_splits = 5
            split_size = n_samples // n_splits
            time_splits = [(i * split_size, (i + 1) * split_size) for i in range(n_splits)]
        
        metrics = {}
        
        for feature in X.columns:
            feature_data = X[feature].dropna()
            if len(feature_data) == 0:
                continue
            
            # Temporal stability
            temporal_ranks = []
            for start, end in time_splits:
                if start >= len(feature_data) or end > len(feature_data):
                    continue
                
                split_data = feature_data.iloc[start:end]
                split_y = y.iloc[start:end]
                
                common_idx = split_data.index.intersection(split_y.index)
                if len(common_idx) < 5:
                    continue
                
                split_aligned = split_data.loc[common_idx]
                split_y_aligned = split_y.loc[common_idx]
                
                valid_mask = ~(split_aligned.isna() | split_y_aligned.isna())
                if valid_mask.sum() < 5:
                    continue
                
                split_clean = split_aligned[valid_mask]
                split_y_clean = split_y_aligned[valid_mask]
                
                # Compute MI for this split
                try:
                    is_classification = split_y_clean.dtype.name == 'category' or split_y_clean.dtype == 'object'
                    if is_classification:
                        mi_score = mutual_info_classif(split_clean.values.reshape(-1, 1), split_y_clean, random_state=42)[0]
                    else:
                        mi_score = mutual_info_regression(split_clean.values.reshape(-1, 1), split_y_clean, random_state=42)[0]
                    temporal_ranks.append(mi_score)
                except:
                    continue
            
            if len(temporal_ranks) < 2:
                continue
            
            # Compute stability metrics
            rank_std = np.std(temporal_ranks)
            rank_mean = np.mean(temporal_ranks)
            rank_stability = 1 - (rank_std / (rank_mean + 1e-8))
            
            # Bootstrap stability
            bootstrap_scores = []
            for _ in range(self.n_bootstrap):
                try:
                    bootstrap_idx = np.random.choice(len(feature_data), size=min(100, len(feature_data)), replace=True)
                    bootstrap_data = feature_data.iloc[bootstrap_idx]
                    bootstrap_y = y.iloc[bootstrap_idx]
                    
                    common_idx = bootstrap_data.index.intersection(bootstrap_y.index)
                    if len(common_idx) < 5:
                        continue
                    
                    bootstrap_aligned = bootstrap_data.loc[common_idx]
                    bootstrap_y_aligned = bootstrap_y.loc[common_idx]
                    
                    valid_mask = ~(bootstrap_aligned.isna() | bootstrap_y_aligned.isna())
                    if valid_mask.sum() < 5:
                        continue
                    
                    bootstrap_clean = bootstrap_aligned[valid_mask]
                    bootstrap_y_clean = bootstrap_y_aligned[valid_mask]
                    
                    is_classification = bootstrap_y_clean.dtype.name == 'category' or bootstrap_y_clean.dtype == 'object'
                    if is_classification:
                        mi_score = mutual_info_classif(bootstrap_clean.values.reshape(-1, 1), bootstrap_y_clean, random_state=42)[0]
                    else:
                        mi_score = mutual_info_regression(bootstrap_clean.values.reshape(-1, 1), bootstrap_y_clean, random_state=42)[0]
                    bootstrap_scores.append(mi_score)
                except:
                    continue
            
            bootstrap_stability = 1 - (np.std(bootstrap_scores) / (np.mean(bootstrap_scores) + 1e-8)) if bootstrap_scores else 0.0
            
            # Scaling sensitivity
            scaling_sensitivity = self._compute_scaling_sensitivity(feature_data, y)
            
            metrics[feature] = {
                'temporal_stability': rank_stability,
                'bootstrap_stability': bootstrap_stability,
                'scaling_sensitivity': scaling_sensitivity,
                'temporal_ranks': temporal_ranks,
                'bootstrap_scores': bootstrap_scores
            }
        
        return metrics
    
    def _compute_uniqueness_metrics(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Compute uniqueness & redundancy metrics."""
        metrics = {}
        
        # Compute correlation matrix
        corr_matrix = X.corr().abs()
        
        for feature in X.columns:
            if feature not in corr_matrix.columns:
                continue
            
            # Max correlation to other features
            feature_corrs = corr_matrix[feature].drop(feature)
            max_correlation = feature_corrs.max() if len(feature_corrs) > 0 else 0.0
            
            # VIF (simplified)
            try:
                vif = self._compute_vif(X, feature)
            except:
                vif = 1.0
            
            # mRMR score (simplified)
            try:
                mrmr_score = self._compute_mrmr_score(feature, X, y)
            except:
                mrmr_score = 0.0
            
            # Conditional MI gain
            try:
                cond_mi_gain = self._compute_conditional_mi_gain(feature, X, y)
            except:
                cond_mi_gain = 0.0
            
            metrics[feature] = {
                'max_correlation': max_correlation,
                'vif': vif,
                'mrmr_score': mrmr_score,
                'conditional_mi_gain': cond_mi_gain
            }
        
        return metrics
    
    def _compute_risk_metrics(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Compute risk & hygiene metrics."""
        metrics = {}
        
        for feature in X.columns:
            feature_data = X[feature].dropna()
            if len(feature_data) == 0:
                continue
            
            # Leakage test
            leakage_score = self._compute_leakage_test(feature_data, y)
            
            # Outlier sensitivity
            outlier_sensitivity = self._compute_outlier_sensitivity(feature_data, y)
            
            # Monotonic expectation (simplified)
            monotonic_score = self._compute_monotonic_score(feature_data, y)
            
            # Drift indicators
            psi_score = self._compute_psi_score(feature_data)
            
            metrics[feature] = {
                'leakage_score': leakage_score,
                'outlier_sensitivity': outlier_sensitivity,
                'monotonic_score': monotonic_score,
                'psi_score': psi_score
            }
        
        return metrics
    
    def _compute_feature_quality_scores(self, predictive_metrics: Dict[str, Any],
                                       stability_metrics: Dict[str, Any],
                                       uniqueness_metrics: Dict[str, Any],
                                       risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compute Feature Quality Scores (FQS)."""
        # Collect all features
        all_features = set()
        for family_metrics in predictive_metrics.values():
            all_features.update(family_metrics.keys())
        
        fqs_results = {}
        
        for feature in all_features:
            # Get metrics for this feature
            pred_metrics = self._get_feature_metrics(feature, predictive_metrics)
            stability_metrics_feat = stability_metrics.get(feature, {})
            uniqueness_metrics_feat = uniqueness_metrics.get(feature, {})
            risk_metrics_feat = risk_metrics.get(feature, {})
            
            # Compute FQS components
            predictive_score = self._compute_predictive_score(pred_metrics)
            stability_score = self._compute_stability_score(stability_metrics_feat)
            uniqueness_score = self._compute_uniqueness_score(uniqueness_metrics_feat)
            risk_score = self._compute_risk_score(risk_metrics_feat)
            
            # Weighted FQS
            fqs = (0.5 * predictive_score + 
                   0.3 * stability_score + 
                   0.15 * uniqueness_score + 
                   0.05 * risk_score)
            
            # Flags
            flags = self._compute_flags(pred_metrics, stability_metrics_feat, 
                                      uniqueness_metrics_feat, risk_metrics_feat)
            
            fqs_results[feature] = {
                'fqs': fqs,
                'predictive_score': predictive_score,
                'stability_score': stability_score,
                'uniqueness_score': uniqueness_score,
                'risk_score': risk_score,
                'flags': flags,
                'family': pred_metrics.get('family', 'unknown')
            }
        
        return fqs_results
    
    def _compute_predictive_score(self, pred_metrics: Dict[str, Any]) -> float:
        """Compute predictive signal score."""
        if not pred_metrics:
            return 0.0
        
        # Normalize metrics to 0-1 scale
        mi_score = min(pred_metrics.get('mi_score', 0), 1.0)
        dist_corr = min(pred_metrics.get('distance_correlation', 0), 1.0)
        cv_score = max(0, min(pred_metrics.get('cv_score', 0), 1.0))
        perm_imp = min(pred_metrics.get('permutation_importance', 0), 1.0)
        
        # Weighted average
        return 0.4 * mi_score + 0.3 * dist_corr + 0.2 * cv_score + 0.1 * perm_imp
    
    def _compute_stability_score(self, stability_metrics: Dict[str, Any]) -> float:
        """Compute stability score."""
        if not stability_metrics:
            return 0.0
        
        temporal_stability = max(0, min(stability_metrics.get('temporal_stability', 0), 1.0))
        bootstrap_stability = max(0, min(stability_metrics.get('bootstrap_stability', 0), 1.0))
        scaling_sensitivity = max(0, 1 - stability_metrics.get('scaling_sensitivity', 1.0))
        
        return (temporal_stability + bootstrap_stability + scaling_sensitivity) / 3
    
    def _compute_uniqueness_score(self, uniqueness_metrics: Dict[str, Any]) -> float:
        """Compute uniqueness score."""
        if not uniqueness_metrics:
            return 0.0
        
        max_corr = uniqueness_metrics.get('max_correlation', 1.0)
        vif = uniqueness_metrics.get('vif', 10.0)
        mrmr_score = max(0, min(uniqueness_metrics.get('mrmr_score', 0), 1.0))
        cond_mi_gain = max(0, min(uniqueness_metrics.get('conditional_mi_gain', 0), 1.0))
        
        # Convert to uniqueness scores
        corr_score = max(0, 1 - max_corr)
        vif_score = max(0, 1 - (vif - 1) / 9)  # VIF 1-10 maps to 1-0
        
        return (corr_score + vif_score + mrmr_score + cond_mi_gain) / 4
    
    def _compute_risk_score(self, risk_metrics: Dict[str, Any]) -> float:
        """Compute risk score."""
        if not risk_metrics:
            return 0.0
        
        leakage_score = max(0, 1 - risk_metrics.get('leakage_score', 1.0))
        outlier_score = max(0, 1 - risk_metrics.get('outlier_sensitivity', 1.0))
        monotonic_score = max(0, risk_metrics.get('monotonic_score', 0))
        psi_score = max(0, 1 - risk_metrics.get('psi_score', 1.0))
        
        return (leakage_score + outlier_score + monotonic_score + psi_score) / 4
    
    def _compute_flags(self, pred_metrics: Dict[str, Any], stability_metrics: Dict[str, Any],
                      uniqueness_metrics: Dict[str, Any], risk_metrics: Dict[str, Any]) -> List[str]:
        """Compute feature flags."""
        flags = []
        
        # Predictive flags
        if pred_metrics.get('mi_score', 0) >= 0.1:
            flags.append('high_mi')
        if pred_metrics.get('cv_score', 0) >= 0.5:
            flags.append('good_cv')
        
        # Stability flags
        if stability_metrics.get('temporal_stability', 0) >= 0.6:
            flags.append('stable_temporal')
        if stability_metrics.get('bootstrap_stability', 0) >= 0.6:
            flags.append('stable_bootstrap')
        
        # Uniqueness flags
        if uniqueness_metrics.get('max_correlation', 1.0) <= 0.9:
            flags.append('unique')
        if uniqueness_metrics.get('vif', 10.0) <= 5.0:
            flags.append('low_vif')
        
        # Risk flags
        if risk_metrics.get('leakage_score', 1.0) <= 0.1:
            flags.append('no_leakage')
        if risk_metrics.get('psi_score', 1.0) <= 0.2:
            flags.append('no_drift')
        
        return flags
    
    def _get_feature_metrics(self, feature: str, predictive_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Get metrics for a specific feature."""
        for family_metrics in predictive_metrics.values():
            if feature in family_metrics:
                return family_metrics[feature]
        return {}
    
    def _compute_distance_correlation(self, x: pd.Series, y: pd.Series) -> float:
        """Compute distance correlation."""
        try:
            x_vals = x.values
            y_vals = y.values
            
            x_dist = pdist(x_vals.reshape(-1, 1))
            y_dist = pdist(y_vals.reshape(-1, 1))
            
            x_dist_centered = x_dist - np.mean(x_dist)
            y_dist_centered = y_dist - np.mean(y_dist)
            
            numerator = np.dot(x_dist_centered, y_dist_centered)
            denominator = np.sqrt(np.dot(x_dist_centered, x_dist_centered) * np.dot(y_dist_centered, y_dist_centered))
            
            return abs(numerator / denominator) if denominator != 0 else 0.0
        except:
            return 0.0
    
    def _compute_univariate_cv_score(self, feature_data: pd.Series, y: pd.Series, 
                                    is_classification: bool) -> float:
        """Compute univariate CV score."""
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import StandardScaler
            
            X_scaled = StandardScaler().fit_transform(feature_data.values.reshape(-1, 1))
            
            if is_classification:
                model = LogisticRegression(random_state=42, max_iter=1000)
                scoring = 'roc_auc'
            else:
                model = Ridge(alpha=1.0)
                scoring = 'r2'
            
            scores = cross_val_score(model, X_scaled, y, cv=3, scoring=scoring)
            return np.mean(scores)
        except:
            return 0.0
    
    def _compute_permutation_importance_single(self, feature_data: pd.Series, y: pd.Series, 
                                             is_classification: bool) -> float:
        """Compute permutation importance for a single feature."""
        try:
            from sklearn.inspection import permutation_importance
            from sklearn.preprocessing import StandardScaler
            
            X_scaled = StandardScaler().fit_transform(feature_data.values.reshape(-1, 1))
            
            if is_classification:
                model = LogisticRegression(random_state=42, max_iter=1000)
            else:
                model = Ridge(alpha=1.0)
            
            model.fit(X_scaled, y)
            perm_imp = permutation_importance(model, X_scaled, y, n_repeats=5, random_state=42)
            return perm_imp.importances_mean[0]
        except:
            return 0.0
    
    def _compute_scaling_sensitivity(self, feature_data: pd.Series, y: pd.Series) -> float:
        """Compute scaling sensitivity."""
        try:
            # Test different scaling methods
            scalers = [StandardScaler(), RobustScaler()]
            scores = []
            
            for scaler in scalers:
                X_scaled = scaler.fit_transform(feature_data.values.reshape(-1, 1))
                
                is_classification = y.dtype.name == 'category' or y.dtype == 'object'
                if is_classification:
                    model = LogisticRegression(random_state=42, max_iter=1000)
                    scoring = 'roc_auc'
                else:
                    model = Ridge(alpha=1.0)
                    scoring = 'r2'
                
                cv_scores = cross_val_score(model, X_scaled, y, cv=3, scoring=scoring)
                scores.append(np.mean(cv_scores))
            
            if len(scores) > 1:
                return np.std(scores) / (np.mean(scores) + 1e-8)
            else:
                return 0.0
        except:
            return 0.0
    
    def _compute_vif(self, X: pd.DataFrame, feature: str) -> float:
        """Compute VIF for a feature."""
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            
            feature_idx = X.columns.get_loc(feature)
            vif = variance_inflation_factor(X.values, feature_idx)
            return vif
        except:
            return 1.0
    
    def _compute_mrmr_score(self, feature: str, X: pd.DataFrame, y: pd.Series) -> float:
        """Compute mRMR score (simplified)."""
        try:
            # Relevance (MI)
            feature_data = X[feature].dropna()
            common_idx = feature_data.index.intersection(y.index)
            if len(common_idx) == 0:
                return 0.0
            
            feature_aligned = feature_data.loc[common_idx]
            y_aligned = y.loc[common_idx]
            
            valid_mask = ~(feature_aligned.isna() | y_aligned.isna())
            if valid_mask.sum() < 10:
                return 0.0
            
            feature_clean = feature_aligned[valid_mask]
            y_clean = y_aligned[valid_mask]
            
            is_classification = y_clean.dtype.name == 'category' or y_clean.dtype == 'object'
            if is_classification:
                relevance = mutual_info_classif(feature_clean.values.reshape(-1, 1), y_clean, random_state=42)[0]
            else:
                relevance = mutual_info_regression(feature_clean.values.reshape(-1, 1), y_clean, random_state=42)[0]
            
            # Redundancy (max correlation with other features)
            other_features = [f for f in X.columns if f != feature]
            if not other_features:
                return relevance
            
            correlations = []
            for other_feature in other_features:
                if other_feature in X.columns:
                    corr = X[feature].corr(X[other_feature])
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            redundancy = max(correlations) if correlations else 0.0
            
            # mRMR score
            return relevance - 0.5 * redundancy
        except:
            return 0.0
    
    def _compute_conditional_mi_gain(self, feature: str, X: pd.DataFrame, y: pd.Series) -> float:
        """Compute conditional MI gain (simplified)."""
        try:
            # Find most correlated feature
            other_features = [f for f in X.columns if f != feature]
            if not other_features:
                return 0.0
            
            correlations = []
            for other_feature in other_features:
                if other_feature in X.columns:
                    corr = X[feature].corr(X[other_feature])
                    if not np.isnan(corr):
                        correlations.append((other_feature, abs(corr)))
            
            if not correlations:
                return 0.0
            
            # Get most correlated feature
            most_corr_feature = max(correlations, key=lambda x: x[1])[0]
            
            # Compute MI gain
            feature_data = X[feature].dropna()
            other_data = X[most_corr_feature].dropna()
            common_idx = feature_data.index.intersection(other_data.index).intersection(y.index)
            
            if len(common_idx) < 10:
                return 0.0
            
            feature_aligned = feature_data.loc[common_idx]
            other_aligned = other_data.loc[common_idx]
            y_aligned = y.loc[common_idx]
            
            valid_mask = ~(feature_aligned.isna() | other_aligned.isna() | y_aligned.isna())
            if valid_mask.sum() < 10:
                return 0.0
            
            feature_clean = feature_aligned[valid_mask]
            other_clean = other_aligned[valid_mask]
            y_clean = y_aligned[valid_mask]
            
            # Compute MI with and without the other feature
            is_classification = y_clean.dtype.name == 'category' or y_clean.dtype == 'object'
            
            if is_classification:
                mi_with = mutual_info_classif(feature_clean.values.reshape(-1, 1), y_clean, random_state=42)[0]
                mi_without = mutual_info_classif(feature_clean.values.reshape(-1, 1), y_clean, random_state=42)[0]
            else:
                mi_with = mutual_info_regression(feature_clean.values.reshape(-1, 1), y_clean, random_state=42)[0]
                mi_without = mutual_info_regression(feature_clean.values.reshape(-1, 1), y_clean, random_state=42)[0]
            
            return max(0, mi_with - mi_without)
        except:
            return 0.0
    
    def _compute_leakage_test(self, feature_data: pd.Series, y: pd.Series) -> float:
        """Compute leakage test score."""
        try:
            # Original performance
            is_classification = y.dtype.name == 'category' or y.dtype == 'object'
            if is_classification:
                model = LogisticRegression(random_state=42, max_iter=1000)
                scoring = 'roc_auc'
            else:
                model = Ridge(alpha=1.0)
                scoring = 'r2'
            
            X_scaled = StandardScaler().fit_transform(feature_data.values.reshape(-1, 1))
            original_scores = cross_val_score(model, X_scaled, y, cv=3, scoring=scoring)
            original_score = np.mean(original_scores)
            
            # Shuffled performance
            y_shuffled = y.sample(frac=1, random_state=42).reset_index(drop=True)
            shuffled_scores = cross_val_score(model, X_scaled, y_shuffled, cv=3, scoring=scoring)
            shuffled_score = np.mean(shuffled_scores)
            
            # Leakage score (how much performance drops)
            return max(0, original_score - shuffled_score)
        except:
            return 0.0
    
    def _compute_outlier_sensitivity(self, feature_data: pd.Series, y: pd.Series) -> float:
        """Compute outlier sensitivity."""
        try:
            # Original performance
            is_classification = y.dtype.name == 'category' or y.dtype == 'object'
            if is_classification:
                model = LogisticRegression(random_state=42, max_iter=1000)
                scoring = 'roc_auc'
            else:
                model = Ridge(alpha=1.0)
                scoring = 'r2'
            
            X_scaled = StandardScaler().fit_transform(feature_data.values.reshape(-1, 1))
            original_scores = cross_val_score(model, X_scaled, y, cv=3, scoring=scoring)
            original_score = np.mean(original_scores)
            
            # Winsorized performance
            feature_winsorized = feature_data.clip(
                lower=feature_data.quantile(0.05),
                upper=feature_data.quantile(0.95)
            )
            X_winsorized = StandardScaler().fit_transform(feature_winsorized.values.reshape(-1, 1))
            winsorized_scores = cross_val_score(model, X_winsorized, y, cv=3, scoring=scoring)
            winsorized_score = np.mean(winsorized_scores)
            
            # Sensitivity score
            return abs(original_score - winsorized_score)
        except:
            return 0.0
    
    def _compute_monotonic_score(self, feature_data: pd.Series, y: pd.Series) -> float:
        """Compute monotonic score."""
        try:
            # Compute Kendall tau
            tau, p_value = kendalltau(feature_data, y)
            return abs(tau) if not np.isnan(tau) else 0.0
        except:
            return 0.0
    
    def _compute_psi_score(self, feature_data: pd.Series) -> float:
        """Compute PSI score (simplified)."""
        try:
            # Split data into train and recent
            n_samples = len(feature_data)
            train_data = feature_data.iloc[:n_samples//2]
            recent_data = feature_data.iloc[n_samples//2:]
            
            if len(train_data) == 0 or len(recent_data) == 0:
                return 0.0
            
            # Compute PSI
            train_hist, _ = np.histogram(train_data, bins=10, density=True)
            recent_hist, _ = np.histogram(recent_data, bins=10, density=True)
            
            # Normalize
            train_hist = train_hist / (train_hist.sum() + 1e-8)
            recent_hist = recent_hist / (recent_hist.sum() + 1e-8)
            
            # Compute PSI
            psi = np.sum((recent_hist - train_hist) * np.log((recent_hist + 1e-8) / (train_hist + 1e-8)))
            return psi
        except:
            return 0.0
    
    def _auto_detect_families(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Auto-detect feature families."""
        families = {
            'returns': [],
            'vwap': [],
            'volatility': [],
            'volume': [],
            'momentum': [],
            'technical': [],
            'other': []
        }
        
        for feature in feature_names:
            if any(pattern in feature.lower() for pattern in ['ret_', 'return']):
                families['returns'].append(feature)
            elif any(pattern in feature.lower() for pattern in ['vwap', 'vw_']):
                families['vwap'].append(feature)
            elif any(pattern in feature.lower() for pattern in ['vol_', 'volatility']):
                families['volatility'].append(feature)
            elif any(pattern in feature.lower() for pattern in ['volume', 'vol_']):
                families['volume'].append(feature)
            elif any(pattern in feature.lower() for pattern in ['mom_', 'momentum', 'acc_']):
                families['momentum'].append(feature)
            elif any(pattern in feature.lower() for pattern in ['rsi', 'macd', 'bollinger']):
                families['technical'].append(feature)
            else:
                families['other'].append(feature)
        
        return {k: v for k, v in families.items() if v}
    
    def _generate_scorecard_summary(self, fqs_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate scorecard summary."""
        if not fqs_results:
            return {}
        
        # Overall statistics
        fqs_scores = [result['fqs'] for result in fqs_results.values()]
        
        summary = {
            'total_features': len(fqs_results),
            'mean_fqs': np.mean(fqs_scores),
            'std_fqs': np.std(fqs_scores),
            'min_fqs': np.min(fqs_scores),
            'max_fqs': np.max(fqs_scores),
            'high_quality_features': len([fqs for fqs in fqs_scores if fqs >= 0.7]),
            'medium_quality_features': len([fqs for fqs in fqs_scores if 0.4 <= fqs < 0.7]),
            'low_quality_features': len([fqs for fqs in fqs_scores if fqs < 0.4])
        }
        
        # Family breakdown
        family_stats = {}
        for feature, result in fqs_results.items():
            family = result.get('family', 'unknown')
            if family not in family_stats:
                family_stats[family] = []
            family_stats[family].append(result['fqs'])
        
        for family, scores in family_stats.items():
            family_stats[family] = {
                'count': len(scores),
                'mean_fqs': np.mean(scores),
                'std_fqs': np.std(scores)
            }
        
        summary['family_breakdown'] = family_stats
        
        return summary
    
    def generate_scorecard_report(self, scorecard_results: Dict[str, Any]) -> pd.DataFrame:
        """Generate detailed scorecard report."""
        if 'feature_quality_scores' not in scorecard_results:
            return pd.DataFrame()
        
        fqs_results = scorecard_results['feature_quality_scores']
        
        # Create detailed report
        report_data = []
        for feature, result in fqs_results.items():
            report_data.append({
                'feature': feature,
                'fqs': result['fqs'],
                'predictive_score': result['predictive_score'],
                'stability_score': result['stability_score'],
                'uniqueness_score': result['uniqueness_score'],
                'risk_score': result['risk_score'],
                'family': result['family'],
                'flags': ', '.join(result['flags'])
            })
        
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('fqs', ascending=False)
        
        return report_df