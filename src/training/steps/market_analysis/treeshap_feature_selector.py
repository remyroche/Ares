"""
TreeSHAP-based Feature Selector for Regime Analysis

This module provides a comprehensive TreeSHAP-based feature selection system that:
1. Uses LightGBM + TreeSHAP for accurate feature importance scoring
2. Handles feature diversity and redundancy through correlation filtering
3. Supports multi-target regime analysis
4. Integrates with the existing economic regime feature selection framework

Key Features:
- TreeSHAP importance scoring (more accurate than traditional methods)
- Correlation-based redundancy filtering
- Category diversity enforcement
- Multi-target support for regime analysis
- Hardware optimization and memory efficiency
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import logging
from dataclasses import dataclass
from pathlib import Path

# Import TreeSHAP and LightGBM
try:
    import lightgbm as lgb
    import shap
    from shap import TreeExplainer
    LIGHTGBM_AVAILABLE = True
    SHAP_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    SHAP_AVAILABLE = False
    lgb = None
    shap = None
    TreeExplainer = None

# Import existing utilities
try:
    from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
    from src.utils.common_operations import safe_correlation
except ImportError:
    # Fallback implementations
    def tprint_info(msg): print(f"INFO: {msg}")
    def tprint_warning(msg): print(f"WARNING: {msg}")
    def tprint_error(msg): print(f"ERROR: {msg}")
    def tprint_success(msg): print(f"SUCCESS: {msg}")
    def safe_correlation(x, y): return np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0

@dataclass
class TreeSHAPFeatureScore:
    """Feature score from TreeSHAP analysis."""
    feature_name: str
    treeshap_importance: float
    builtin_importance: float
    correlation_score: float
    diversity_score: float
    composite_score: float
    category: str = ""
    selected: bool = False

class TreeSHAPFeatureSelector:
    """
    TreeSHAP-based feature selector with diversity and redundancy handling.
    
    This selector addresses the limitations of pure TreeSHAP by:
    1. Using TreeSHAP for accurate importance scoring
    2. Adding correlation-based redundancy filtering
    3. Enforcing category diversity
    4. Supporting multi-target regime analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize TreeSHAP feature selector."""
        self.config = config or {}
        self.logger = logging.getLogger('TreeSHAPFeatureSelector')
        
        # TreeSHAP configuration
        self.n_estimators = self.config.get('n_estimators', 100)
        self.max_depth = self.config.get('max_depth', 8)
        self.learning_rate = self.config.get('learning_rate', 0.1)
        self.num_leaves = self.config.get('num_leaves', 31)
        self.min_child_samples = self.config.get('min_child_samples', 20)
        self.subsample = self.config.get('subsample', 0.8)
        self.col_sample_bytree = self.config.get('colsample_bytree', 0.8)
        self.reg_alpha = self.config.get('reg_alpha', 0.1)  # L1 regularization
        self.reg_lambda = self.config.get('reg_lambda', 0.1)  # L2 regularization
        self.random_state = self.config.get('random_state', 42)
        
        # Diversity and redundancy settings
        self.correlation_threshold = self.config.get('correlation_threshold', 0.85)
        self.diversity_weight = self.config.get('diversity_weight', 0.2)
        self.treeshap_weight = self.config.get('treeshap_weight', 0.6)
        self.correlation_weight = self.config.get('correlation_weight', 0.2)
        
        # Multi-target settings
        self.target_columns = self.config.get('target_columns', [])
        self.target_weights = self.config.get('target_weights', {})
        
        # Performance settings
        self.max_samples = self.config.get('max_samples', 250000)
        self.shap_sample_size = self.config.get('shap_sample_size', 1000)
        
        # Check dependencies
        if not (LIGHTGBM_AVAILABLE and SHAP_AVAILABLE):
            raise ImportError("LightGBM and SHAP are required. Install with: pip install lightgbm shap")
        
        tprint_info("🚀 TreeSHAPFeatureSelector initialized")
        tprint_info(f"⚙️ Correlation threshold: {self.correlation_threshold}")
        tprint_info(f"⚙️ Diversity weight: {self.diversity_weight}")
        tprint_info(f"⚙️ TreeSHAP weight: {self.treeshap_weight}")
    
    def select_features(
        self, 
        features_df: pd.DataFrame, 
        labels_df: pd.DataFrame,
        target_feature_count: int = 25
    ) -> Dict[str, Any]:
        """
        Select features using TreeSHAP with diversity and redundancy handling.
        
        Args:
            features_df: Feature matrix (samples x features)
            labels_df: Target labels (samples x targets)
            target_feature_count: Number of features to select
            
        Returns:
            Dictionary with selected features and scores
        """
        start_time = time.time()
        tprint_info(f"🔍 Starting TreeSHAP feature selection...")
        tprint_info(f"📊 Data shape: {features_df.shape}, Target features: {target_feature_count}")
        
        try:
            # Step 1: Preprocess data
            X, y, feature_names = self._preprocess_data(features_df, labels_df)
            
            # Step 2: Calculate TreeSHAP importances
            treeshap_scores = self._calculate_treeshap_importances(X, y, feature_names)
            
            # Step 3: Calculate correlation scores for redundancy handling
            correlation_scores = self._calculate_correlation_scores(X, y, feature_names)
            
            # Step 4: Calculate diversity scores
            diversity_scores = self._calculate_diversity_scores(feature_names)
            
            # Step 5: Combine scores
            feature_scores = self._combine_scores(
                feature_names, treeshap_scores, correlation_scores, diversity_scores
            )
            
            # Step 6: Apply redundancy filtering
            filtered_features = self._apply_redundancy_filtering(
                feature_scores, X, correlation_threshold=self.correlation_threshold
            )
            
            # Step 7: Enforce category diversity
            selected_features = self._enforce_category_diversity(
                filtered_features, target_feature_count
            )
            
            # Step 8: Prepare results
            execution_time = time.time() - start_time
            
            result = {
                'selected_features': selected_features,
                'feature_scores': {f.feature_name: f.composite_score for f in feature_scores},
                'treeshap_scores': treeshap_scores,
                'correlation_scores': correlation_scores,
                'diversity_scores': diversity_scores,
                'method': 'treeshap_with_diversity',
                'parameters': {
                    'correlation_threshold': self.correlation_threshold,
                    'diversity_weight': self.diversity_weight,
                    'treeshap_weight': self.treeshap_weight,
                    'target_feature_count': target_feature_count
                },
                'execution_time': execution_time,
                'success': True
            }
            
            tprint_success(f"✅ TreeSHAP selection completed in {execution_time:.3f}s")
            tprint_success(f"📊 Selected {len(selected_features)} features")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ TreeSHAP selection failed: {e}")
            return {
                'selected_features': [],
                'feature_scores': {},
                'method': 'treeshap_with_diversity',
                'error': str(e),
                'success': False
            }
    
    def _preprocess_data(
        self, 
        features_df: pd.DataFrame, 
        labels_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Preprocess data for TreeSHAP analysis."""
        # Align data
        common_index = features_df.index.intersection(labels_df.index)
        features_aligned = features_df.loc[common_index]
        labels_aligned = labels_df.loc[common_index]
        
        # Handle multi-target case
        if len(labels_aligned.shape) > 1 and labels_aligned.shape[1] > 1:
            # Multi-target: use first target or create composite target
            if self.target_columns:
                # Use weighted combination of targets
                y = np.zeros(len(labels_aligned))
                total_weight = 0
                for target_col in self.target_columns:
                    if target_col in labels_aligned.columns:
                        weight = self.target_weights.get(target_col, 1.0)
                        y += labels_aligned[target_col].values * weight
                        total_weight += weight
                if total_weight > 0:
                    y = y / total_weight
            else:
                # Use first target
                y = labels_aligned.iloc[:, 0].values
        else:
            # Single target
            y = labels_aligned.values.flatten()
        
        # Convert to numpy arrays
        X = features_aligned.values
        feature_names = list(features_aligned.columns)
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
        
        tprint_info(f"📊 Preprocessed data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, feature_names
    
    def _calculate_treeshap_importances(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Calculate TreeSHAP feature importances."""
        tprint_info("🔍 Calculating TreeSHAP importances...")
        
        # Sample data if too large
        if len(X) > self.max_samples:
            tprint_info(f"📊 Sampling {self.max_samples} rows from {len(X)} for efficiency")
            indices = np.random.choice(len(X), self.max_samples, replace=False)
            X_sample = X[indices]
            y_sample = y[indices]
        else:
            X_sample, y_sample = X, y
        
        # Create LightGBM model
        if len(np.unique(y_sample)) <= 10:  # Classification
            model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                subsample=self.subsample,
                colsample_bytree=self.col_sample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1
            )
        else:  # Regression
            model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                subsample=self.subsample,
                colsample_bytree=self.col_sample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1
            )
        
        # Train model
        model.fit(X_sample, y_sample)
        
        # Calculate TreeSHAP values
        try:
            explainer = TreeExplainer(model)
            shap_sample_size = min(self.shap_sample_size, len(X_sample))
            shap_indices = np.random.choice(len(X_sample), shap_sample_size, replace=False)
            X_shap = X_sample[shap_indices]
            
            shap_values = explainer.shap_values(X_shap)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                if len(shap_values) == 2:  # Binary classification
                    importance_scores = np.abs(shap_values[0]).mean(axis=0) + np.abs(shap_values[1]).mean(axis=0)
                else:  # Multi-class
                    all_importances = np.array([np.abs(sv).mean(axis=0) for sv in shap_values])
                    importance_scores = all_importances.mean(axis=0)
            else:  # Regression or single-output
                importance_scores = np.abs(shap_values).mean(axis=0)
            
            # Normalize scores
            total_importance = np.sum(importance_scores)
            if total_importance > 0:
                normalized_scores = importance_scores / total_importance
            else:
                normalized_scores = model.feature_importances_
            
            treeshap_scores = {feature_names[i]: float(normalized_scores[i]) for i in range(len(feature_names))}
            
            tprint_success(f"✅ TreeSHAP calculation completed using {shap_sample_size} samples")
            
        except Exception as e:
            tprint_warning(f"⚠️ TreeSHAP calculation failed: {e}, using built-in importances")
            treeshap_scores = {feature_names[i]: float(model.feature_importances_[i]) for i in range(len(feature_names))}
        
        return treeshap_scores
    
    def _calculate_correlation_scores(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Calculate correlation scores with target."""
        tprint_info("🔍 Calculating correlation scores...")
        
        correlation_scores = {}
        for i, feature_name in enumerate(feature_names):
            try:
                corr = abs(safe_correlation(X[:, i], y))
                correlation_scores[feature_name] = float(corr) if not np.isnan(corr) else 0.0
            except Exception:
                correlation_scores[feature_name] = 0.0
        
        return correlation_scores
    
    def _calculate_diversity_scores(self, feature_names: List[str]) -> Dict[str, float]:
        """Calculate diversity scores based on feature categories."""
        tprint_info("🔍 Calculating diversity scores...")
        
        # Simple category-based diversity scoring
        # In practice, this would be more sophisticated
        diversity_scores = {}
        category_counts = {}
        
        for feature_name in feature_names:
            # Extract category from feature name (simple heuristic)
            category = self._extract_category(feature_name)
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Assign diversity scores (inverse of category frequency)
        for feature_name in feature_names:
            category = self._extract_category(feature_name)
            category_count = category_counts[category]
            # Higher diversity score for less frequent categories
            diversity_scores[feature_name] = 1.0 / max(category_count, 1)
        
        return diversity_scores
    
    def _extract_category(self, feature_name: str) -> str:
        """Extract category from feature name."""
        # Simple heuristic - in practice, this would be more sophisticated
        if any(x in feature_name.lower() for x in ['return', 'price', 'close', 'open', 'high', 'low']):
            return 'price'
        elif any(x in feature_name.lower() for x in ['volume', 'vol']):
            return 'volume'
        elif any(x in feature_name.lower() for x in ['volatility', 'std', 'var']):
            return 'volatility'
        elif any(x in feature_name.lower() for x in ['momentum', 'rsi', 'macd']):
            return 'momentum'
        else:
            return 'other'
    
    def _combine_scores(
        self,
        feature_names: List[str],
        treeshap_scores: Dict[str, float],
        correlation_scores: Dict[str, float],
        diversity_scores: Dict[str, float]
    ) -> List[TreeSHAPFeatureScore]:
        """Combine different scoring methods."""
        tprint_info("🔍 Combining scores...")
        
        feature_scores = []
        for feature_name in feature_names:
            treeshap_score = treeshap_scores.get(feature_name, 0.0)
            correlation_score = correlation_scores.get(feature_name, 0.0)
            diversity_score = diversity_scores.get(feature_name, 0.0)
            
            # Weighted combination
            composite_score = (
                self.treeshap_weight * treeshap_score +
                self.correlation_weight * correlation_score +
                self.diversity_weight * diversity_score
            )
            
            feature_scores.append(TreeSHAPFeatureScore(
                feature_name=feature_name,
                treeshap_importance=treeshap_score,
                builtin_importance=0.0,  # Not used in this implementation
                correlation_score=correlation_score,
                diversity_score=diversity_score,
                composite_score=composite_score,
                category=self._extract_category(feature_name)
            ))
        
        # Sort by composite score
        feature_scores.sort(key=lambda x: x.composite_score, reverse=True)
        
        return feature_scores
    
    def _apply_redundancy_filtering(
        self,
        feature_scores: List[TreeSHAPFeatureScore],
        X: np.ndarray,
        correlation_threshold: float = 0.85
    ) -> List[TreeSHAPFeatureScore]:
        """Apply correlation-based redundancy filtering."""
        tprint_info(f"🔍 Applying redundancy filtering (threshold: {correlation_threshold})...")
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(X.T)
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(feature_scores)):
            for j in range(i + 1, len(feature_scores)):
                corr = abs(correlation_matrix[i, j])
                if corr > correlation_threshold:
                    high_corr_pairs.append((i, j, corr))
        
        # Remove redundant features
        features_to_remove = set()
        for i, j, corr in high_corr_pairs:
            # Keep the feature with higher composite score
            if feature_scores[i].composite_score >= feature_scores[j].composite_score:
                features_to_remove.add(j)
            else:
                features_to_remove.add(i)
        
        # Filter out redundant features
        filtered_features = [f for i, f in enumerate(feature_scores) if i not in features_to_remove]
        
        tprint_info(f"📊 Removed {len(features_to_remove)} redundant features")
        tprint_info(f"📊 Remaining features: {len(filtered_features)}")
        
        return filtered_features
    
    def _enforce_category_diversity(
        self,
        feature_scores: List[TreeSHAPFeatureScore],
        target_count: int
    ) -> List[str]:
        """Enforce category diversity in selected features."""
        tprint_info(f"🔍 Enforcing category diversity (target: {target_count} features)...")
        
        # Group features by category
        category_groups = {}
        for feature in feature_scores:
            category = feature.category
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(feature)
        
        # Sort each category by composite score
        for category in category_groups:
            category_groups[category].sort(key=lambda x: x.composite_score, reverse=True)
        
        # Select features ensuring diversity
        selected_features = []
        categories = list(category_groups.keys())
        category_indices = {cat: 0 for cat in categories}
        
        # Round-robin selection to ensure diversity
        while len(selected_features) < target_count and any(
            category_indices[cat] < len(category_groups[cat]) for cat in categories
        ):
            for category in categories:
                if len(selected_features) >= target_count:
                    break
                
                if category_indices[category] < len(category_groups[category]):
                    feature = category_groups[category][category_indices[category]]
                    selected_features.append(feature.feature_name)
                    category_indices[category] += 1
        
        tprint_success(f"✅ Selected {len(selected_features)} features with category diversity")
        
        # Log category distribution
        category_dist = {}
        for feature_name in selected_features:
            category = self._extract_category(feature_name)
            category_dist[category] = category_dist.get(category, 0) + 1
        
        for category, count in category_dist.items():
            tprint_info(f"📊 {category}: {count} features")
        
        return selected_features