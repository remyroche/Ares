"""
Pre-screening Pipeline for Feature Selection

This module implements a compute-aware pre-screening pipeline with matrix operations
for efficient feature selection and quality assessment.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, kendalltau
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import time
import warnings

logger = logging.getLogger(__name__)

class PreScreeningPipeline:
    """
    Compute-aware pre-screening pipeline for feature selection.
    """
    
    def __init__(self, 
                 top_k_per_bucket: int = 20,
                 mi_percentile: float = 0.6,
                 correlation_threshold: float = 0.95,
                 vif_threshold: float = 10.0,
                 mrmr_lambda: float = 0.5,
                 n_total_features: int = 100,
                 enable_matrix_ops: bool = True):
        """
        Initialize pre-screening pipeline.
        
        Args:
            top_k_per_bucket: Top K features to keep per family
            mi_percentile: MI percentile threshold (e.g., 0.6 for 60th percentile)
            correlation_threshold: Correlation threshold for redundancy pruning
            vif_threshold: VIF threshold for redundancy pruning
            mrmr_lambda: Lambda for mRMR (relevance - lambda * redundancy)
            n_total_features: Total number of features to keep
            enable_matrix_ops: Whether to enable matrix operations
        """
        self.top_k_per_bucket = top_k_per_bucket
        self.mi_percentile = mi_percentile
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold
        self.mrmr_lambda = mrmr_lambda
        self.n_total_features = n_total_features
        self.enable_matrix_ops = enable_matrix_ops
        
        # Initialize matrix operations if available
        if enable_matrix_ops:
            try:
                from src.utils.matrix_operations import get_unified_matrix_operations
                self.matrix_ops = get_unified_matrix_operations(enable_gpu=True, enable_parallel=True)
                self.matrix_available = True
            except ImportError:
                self.matrix_ops = None
                self.matrix_available = False
                logger.warning("Matrix operations not available, using standard operations")
        else:
            self.matrix_ops = None
            self.matrix_available = False
        
        # Results storage
        self.phase_results = {}
        self.feature_metadata = {}
        self.compute_profiles = {}
    
    def run_pre_screening(self, X: pd.DataFrame, y: pd.Series, 
                         feature_families: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Run complete pre-screening pipeline.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_families: Dictionary mapping family names to feature lists
            
        Returns:
            Pre-screening results
        """
        logger.info("Starting pre-screening pipeline...")
        
        # Phase A: Fast univariate signal screen
        logger.info("Phase A: Fast univariate signal screen...")
        phase_a_results = self._phase_a_univariate_screen(X, y, feature_families)
        
        # Phase B: Redundancy pruning
        logger.info("Phase B: Redundancy pruning...")
        phase_b_results = self._phase_b_redundancy_pruning(phase_a_results['selected_features'], X, y)
        
        # Phase C: Light model sanity check
        logger.info("Phase C: Light model sanity check...")
        phase_c_results = self._phase_c_light_model_sanity(phase_b_results['selected_features'], X, y)
        
        # Compile results
        results = {
            'phase_a': phase_a_results,
            'phase_b': phase_b_results,
            'phase_c': phase_c_results,
            'final_features': phase_c_results['selected_features'],
            'feature_metadata': self.feature_metadata,
            'compute_profiles': self.compute_profiles
        }
        
        logger.info(f"Pre-screening completed. Selected {len(results['final_features'])} features")
        return results
    
    def _phase_a_univariate_screen(self, X: pd.DataFrame, y: pd.Series, 
                                  feature_families: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """Phase A: Fast univariate signal screen."""
        if feature_families is None:
            feature_families = self._auto_detect_families(X.columns)
        
        selected_features = []
        family_results = {}
        
        for family_name, family_features in feature_families.items():
            logger.info(f"Processing family: {family_name}")
            
            # Filter features that exist in X
            existing_features = [f for f in family_features if f in X.columns]
            if not existing_features:
                continue
            
            family_X = X[existing_features]
            
            # Compute univariate metrics
            family_metrics = self._compute_univariate_metrics(family_X, y)
            
            # Apply selection criteria
            family_selected = self._select_family_features(family_metrics, family_name)
            selected_features.extend(family_selected)
            
            family_results[family_name] = {
                'total_features': len(existing_features),
                'selected_features': family_selected,
                'metrics': family_metrics
            }
        
        return {
            'selected_features': selected_features,
            'family_results': family_results
        }
    
    def _compute_univariate_metrics(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Compute univariate metrics for features."""
        metrics = []
        
        for feature in X.columns:
            feature_data = X[feature].dropna()
            if len(feature_data) == 0:
                continue
            
            # Align with target
            common_idx = feature_data.index.intersection(y.index)
            if len(common_idx) == 0:
                continue
            
            feature_aligned = feature_data.loc[common_idx]
            y_aligned = y.loc[common_idx]
            
            # Remove NaN values
            valid_mask = ~(feature_aligned.isna() | y_aligned.isna())
            if valid_mask.sum() < 10:  # Need minimum samples
                continue
            
            feature_clean = feature_aligned[valid_mask]
            y_clean = y_aligned[valid_mask]
            
            # Compute metrics
            feature_metrics = self._compute_single_feature_metrics(feature_clean, y_clean, feature)
            metrics.append(feature_metrics)
        
        return pd.DataFrame(metrics)
    
    def _compute_single_feature_metrics(self, feature_data: pd.Series, y: pd.Series, 
                                       feature_name: str) -> Dict[str, Any]:
        """Compute metrics for a single feature."""
        metrics = {'feature': feature_name}
        
        # Determine if classification or regression
        is_classification = y.dtype.name == 'category' or y.dtype == 'object'
        
        # Mutual Information
        try:
            if is_classification:
                mi_score = mutual_info_classif(feature_data.values.reshape(-1, 1), y, random_state=42)[0]
            else:
                mi_score = mutual_info_regression(feature_data.values.reshape(-1, 1), y, random_state=42)[0]
            metrics['mi_score'] = mi_score
        except:
            metrics['mi_score'] = 0.0
        
        # Distance correlation (simplified version)
        try:
            dist_corr = self._compute_distance_correlation(feature_data, y)
            metrics['distance_correlation'] = dist_corr
        except:
            metrics['distance_correlation'] = 0.0
        
        # Univariate CV score
        try:
            cv_score = self._compute_univariate_cv_score(feature_data, y, is_classification)
            metrics['cv_score'] = cv_score
        except:
            metrics['cv_score'] = 0.0
        
        # mRMR score (simplified - will be computed in context)
        metrics['mrmr_score'] = 0.0  # Placeholder
        
        # Compute cost profiling
        start_time = time.time()
        _ = feature_data.mean()  # Simple operation to profile
        metrics['compute_time_ms'] = (time.time() - start_time) * 1000
        
        return metrics
    
    def _compute_distance_correlation(self, x: pd.Series, y: pd.Series) -> float:
        """Compute distance correlation (simplified version)."""
        try:
            # Convert to numpy arrays
            x_vals = x.values
            y_vals = y.values
            
            # Compute pairwise distances
            x_dist = pdist(x_vals.reshape(-1, 1))
            y_dist = pdist(y_vals.reshape(-1, 1))
            
            # Compute distance correlation
            x_dist_centered = x_dist - np.mean(x_dist)
            y_dist_centered = y_dist - np.mean(y_dist)
            
            numerator = np.dot(x_dist_centered, y_dist_centered)
            denominator = np.sqrt(np.dot(x_dist_centered, x_dist_centered) * np.dot(y_dist_centered, y_dist_centered))
            
            if denominator == 0:
                return 0.0
            
            return abs(numerator / denominator)
        except:
            return 0.0
    
    def _compute_univariate_cv_score(self, feature_data: pd.Series, y: pd.Series, 
                                    is_classification: bool) -> float:
        """Compute univariate CV score."""
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data
            X_scaled = StandardScaler().fit_transform(feature_data.values.reshape(-1, 1))
            
            # Choose model
            if is_classification:
                model = LogisticRegression(random_state=42, max_iter=1000)
                scoring = 'roc_auc'
            else:
                model = Ridge(alpha=1.0)
                scoring = 'r2'
            
            # Cross-validation
            scores = cross_val_score(model, X_scaled, y, cv=3, scoring=scoring)
            return np.mean(scores)
        except:
            return 0.0
    
    def _select_family_features(self, metrics_df: pd.DataFrame, family_name: str) -> List[str]:
        """Select top features from a family based on metrics."""
        if len(metrics_df) == 0:
            return []
        
        # Compute MI percentile threshold
        mi_threshold = metrics_df['mi_score'].quantile(self.mi_percentile)
        
        # Filter by MI threshold
        qualified_features = metrics_df[metrics_df['mi_score'] >= mi_threshold]
        
        if len(qualified_features) == 0:
            return []
        
        # Sort by MI score and take top K
        top_features = qualified_features.nlargest(self.top_k_per_bucket, 'mi_score')
        
        return top_features['feature'].tolist()
    
    def _phase_b_redundancy_pruning(self, selected_features: List[str], 
                                   X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Phase B: Redundancy pruning."""
        if not selected_features:
            return {'selected_features': [], 'pruned_features': []}
        
        # Get feature data
        feature_data = X[selected_features].dropna()
        
        # Compute correlation matrix
        corr_matrix = feature_data.corr().abs()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > self.correlation_threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        # Prune redundant features
        features_to_remove = set()
        for feat1, feat2, corr in high_corr_pairs:
            if feat1 in features_to_remove or feat2 in features_to_remove:
                continue
            
            # Keep feature with higher MI or lower compute cost
            mi1 = self._get_feature_mi(feat1, feature_data, y)
            mi2 = self._get_feature_mi(feat2, feature_data, y)
            
            if mi1 >= mi2:
                features_to_remove.add(feat2)
            else:
                features_to_remove.add(feat1)
        
        # Remove redundant features
        final_features = [f for f in selected_features if f not in features_to_remove]
        
        return {
            'selected_features': final_features,
            'pruned_features': list(features_to_remove),
            'high_corr_pairs': high_corr_pairs
        }
    
    def _get_feature_mi(self, feature: str, X: pd.DataFrame, y: pd.Series) -> float:
        """Get MI score for a feature."""
        try:
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
                return mutual_info_classif(feature_clean.values.reshape(-1, 1), y_clean, random_state=42)[0]
            else:
                return mutual_info_regression(feature_clean.values.reshape(-1, 1), y_clean, random_state=42)[0]
        except:
            return 0.0
    
    def _phase_c_light_model_sanity(self, selected_features: List[str], 
                                   X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Phase C: Light model sanity check."""
        if not selected_features:
            return {'selected_features': [], 'model_results': {}}
        
        # Get feature data
        feature_data = X[selected_features].dropna()
        common_idx = feature_data.index.intersection(y.index)
        if len(common_idx) == 0:
            return {'selected_features': [], 'model_results': {}}
        
        X_aligned = feature_data.loc[common_idx]
        y_aligned = y.loc[common_idx]
        
        # Remove NaN values
        valid_mask = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
        X_clean = X_aligned[valid_mask]
        y_clean = y_aligned[valid_mask]
        
        if len(X_clean) == 0:
            return {'selected_features': [], 'model_results': {}}
        
        # Train shallow models
        model_results = self._train_shallow_models(X_clean, y_clean)
        
        # Compute permutation importance
        perm_importance = self._compute_permutation_importance(X_clean, y_clean, model_results)
        
        # Filter features based on permutation importance
        final_features = self._filter_by_permutation_importance(selected_features, perm_importance)
        
        return {
            'selected_features': final_features,
            'model_results': model_results,
            'permutation_importance': perm_importance
        }
    
    def _train_shallow_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train shallow models for sanity check."""
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine task type
        is_classification = y.dtype.name == 'category' or y.dtype == 'object'
        
        results = {}
        
        # Ridge regression
        try:
            ridge = Ridge(alpha=1.0)
            if is_classification:
                # Convert to binary for ridge
                y_binary = (y == y.mode()[0]).astype(int)
                ridge_scores = cross_val_score(ridge, X_scaled, y_binary, cv=3, scoring='r2')
            else:
                ridge_scores = cross_val_score(ridge, X_scaled, y, cv=3, scoring='r2')
            
            results['ridge'] = {
                'model': ridge,
                'scores': ridge_scores,
                'mean_score': np.mean(ridge_scores)
            }
        except Exception as e:
            logger.warning(f"Ridge training failed: {e}")
            results['ridge'] = {'error': str(e)}
        
        # LightGBM (shallow)
        try:
            import lightgbm as lgb
            
            lgb_params = {
                'objective': 'binary' if is_classification else 'regression',
                'num_leaves': 15,  # Shallow
                'max_depth': 4,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'random_state': 42,
                'verbosity': -1
            }
            
            lgb_model = lgb.LGBMClassifier(**lgb_params) if is_classification else lgb.LGBMRegressor(**lgb_params)
            lgb_model.fit(X_scaled, y)
            
            results['lgb'] = {
                'model': lgb_model,
                'feature_importance': lgb_model.feature_importances_
            }
        except Exception as e:
            logger.warning(f"LightGBM training failed: {e}")
            results['lgb'] = {'error': str(e)}
        
        return results
    
    def _compute_permutation_importance(self, X: pd.DataFrame, y: pd.Series, 
                                       model_results: Dict[str, Any]) -> pd.DataFrame:
        """Compute permutation importance."""
        from sklearn.inspection import permutation_importance
        
        importance_results = []
        
        for model_name, model_data in model_results.items():
            if 'error' in model_data or 'model' not in model_data:
                continue
            
            try:
                model = model_data['model']
                perm_imp = permutation_importance(
                    model, X, y, n_repeats=10, random_state=42, n_jobs=1
                )
                
                for i, feature in enumerate(X.columns):
                    importance_results.append({
                        'feature': feature,
                        'model': model_name,
                        'importance_mean': perm_imp.importances_mean[i],
                        'importance_std': perm_imp.importances_std[i],
                        'importance_ci_low': perm_imp.importances_mean[i] - 1.96 * perm_imp.importances_std[i],
                        'importance_ci_high': perm_imp.importances_mean[i] + 1.96 * perm_imp.importances_std[i]
                    })
            except Exception as e:
                logger.warning(f"Permutation importance failed for {model_name}: {e}")
        
        return pd.DataFrame(importance_results)
    
    def _filter_by_permutation_importance(self, features: List[str], 
                                         perm_importance: pd.DataFrame) -> List[str]:
        """Filter features based on permutation importance."""
        if len(perm_importance) == 0:
            return features
        
        # Group by feature and compute mean importance
        feature_importance = perm_importance.groupby('feature').agg({
            'importance_mean': 'mean',
            'importance_ci_low': 'mean'
        }).reset_index()
        
        # Filter features with positive importance and CI > 0
        qualified_features = feature_importance[
            (feature_importance['importance_mean'] > 0) & 
            (feature_importance['importance_ci_low'] > 0)
        ]
        
        # Sort by importance and take top N
        top_features = qualified_features.nlargest(self.n_total_features, 'importance_mean')
        
        return top_features['feature'].tolist()
    
    def _auto_detect_families(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Auto-detect feature families based on naming patterns."""
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
        
        # Remove empty families
        return {k: v for k, v in families.items() if v}
    
    def get_feature_metadata(self) -> pd.DataFrame:
        """Get feature metadata for analysis."""
        if not self.feature_metadata:
            return pd.DataFrame()
        
        return pd.DataFrame(self.feature_metadata).T