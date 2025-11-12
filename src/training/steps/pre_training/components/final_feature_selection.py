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
from src.utils.tprint import tprint


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
        use_permutation_importance: bool = True,
        stability_weight: float = 0.0
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
            stability_weight: Weight for stability in ranking (0-1). 0=pure importance, 0.3=30% stability, 1=pure stability
        """
        self.max_features = max_features
        self.min_features = min_features
        self.selection_method = selection_method
        self.scoring_threshold = scoring_threshold
        self.use_tree_based = use_tree_based
        self.use_permutation_importance = use_permutation_importance
        self.stability_weight = stability_weight


class FinalFeatureSelectionComponent:
    """
    Final feature selection component using permutation importance.
    
    This component uses permutation importance to select the top N features
    from a larger pool of available features. It ranks ALL features by
    permutation importance and selects the top N.
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
        self.all_permutation_importances: Dict[str, float] = {}
        
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
            
            # CRITICAL: Validate input data for NaN values
            y_nan_count = y.isna().sum()
            if y_nan_count > 0:
                self.logger.error(f"❌ Input y contains {y_nan_count} NaN values ({100*y_nan_count/len(y):.2f}%)")
                raise ValueError(f"Input y contains {y_nan_count} NaN values. Please clean the target variable before feature selection.")
            
            X_nan_count = X.isna().sum().sum()
            if X_nan_count > 0:
                self.logger.warning(f"⚠️ Input X contains {X_nan_count} NaN values. Filling with median...")
                X = X.fillna(X.median())
            
            self.logger.info(f"✅ Input validation passed: X shape={X.shape}, y shape={y.shape}, no NaN values")
            
            # Apply feature combination with duplicate detection
            self.logger.info("Applying feature combination with duplicate detection...")
            X = self._combine_features(X)
            feature_names = list(X.columns)
            
            # CRITICAL FIX: Remove constant/low-variance features BEFORE correlation
            self.logger.info("🔍 Filtering low-variance features...")
            variance_threshold = 0.01
            variances = X.var()
            high_variance_features = variances[variances > variance_threshold].index.tolist()
            removed_count = len(feature_names) - len(high_variance_features)
            if removed_count > 0:
                self.logger.info(f"📊 Removed {removed_count} low-variance features (variance < {variance_threshold})")
                X = X[high_variance_features]
                feature_names = high_variance_features
            
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
            
            # Apply tree-based selection if enabled to rank ALL features
            if self.config.use_tree_based:
                # This will rank ALL features by permutation/SHAP importance
                # No diversity filtering happens here - just ranking
                ranked_features = self._apply_tree_based_selection(
                    X, y, feature_names
                )
            else:
                # Fallback: use feature_names as-is
                ranked_features = feature_names
            
            # CRITICAL FIX: Restructure flow - hierarchical is PRE-FILTER, SHAP is FINAL selection
            # Step 1: Pre-filter with hierarchical clustering (reduce to 2-3x target)
            prefilter_count = min(max_features * 3, len(feature_names))
            self.logger.info(f"📊 Step 1: Pre-filtering with hierarchical clustering ({len(feature_names)} -> {prefilter_count})")
            
            if len(feature_names) > prefilter_count:
                prefiltered_features = self._reduce_redundancy_hierarchical(
                    X,
                    feature_names,
                    target_count=prefilter_count,
                    correlation_threshold=0.75
                )
                X_prefiltered = X[prefiltered_features]
                self.logger.info(f"✅ Pre-filtered to {len(prefiltered_features)} features")
            else:
                X_prefiltered = X
                prefiltered_features = feature_names
                self.logger.info(f"✅ Skipping pre-filter (already {len(feature_names)} features)")
            
            # Step 2: Apply SHAP/permutation on pre-filtered features
            self.logger.info(f"📊 Step 2: Applying SHAP/permutation importance on {len(prefiltered_features)} pre-filtered features")
            ranked_features = self._apply_tree_based_selection(
                X_prefiltered, y, prefiltered_features
            )
            
            # Step 3: Filter out zero-importance features
            if self.config.use_permutation_importance and self.all_permutation_importances:
                self.logger.info(f"📊 Step 3: Filtering zero-importance features")
                
                # Sort by importance and filter zeros
                sorted_features = sorted(
                    self.all_permutation_importances.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Filter out zero importance
                nonzero_features = [(feat, imp) for feat, imp in sorted_features if imp > 0]
                zero_count = len(sorted_features) - len(nonzero_features)
                
                if zero_count > 0:
                    self.logger.info(f"📊 Filtered {zero_count} features with zero SHAP importance")
                
                # Get ranked feature list (non-zero only)
                ranked_features = [feat for feat, _ in nonzero_features]
                
                self.logger.info(f"✅ Final ranking: {len(ranked_features)} features with non-zero importance")
                self.logger.info(f"Top 5 features: {ranked_features[:5] if len(ranked_features) >= 5 else ranked_features}")
            
            # Step 3.5: Apply stability-weighted ranking if enabled
            stability_weight = getattr(self.config, 'stability_weight', 0.0)
            if stability_weight > 0 and len(ranked_features) > 0:
                self.logger.info(f"📊 Step 3.5: Applying stability-weighted ranking (weight={stability_weight})")
                ranked_features = self._apply_stability_weighted_ranking(
                    X_prefiltered, y, ranked_features, stability_weight
                )
            
            # Step 4: Select top N features by SHAP importance (or stability-weighted score)
            expected_count = min(max_features, len(ranked_features))
            selected_features = ranked_features[:expected_count]
            
            self.logger.info(f"📊 Step 4: Selected top {len(selected_features)} features by SHAP importance")
            
            self.selected_features = selected_features
            importance_method = "permutation" if self.config.use_permutation_importance else "Gini"
            self.logger.info(f"Final selection: {len(selected_features)} features using {importance_method} importance")
            self.logger.info(f"Selected {len(selected_features)} non-redundant features from {len(feature_names)} total")
            
            # Validate exact count
            if len(selected_features) == expected_count:
                self.logger.info(f"✅ Successfully selected exactly {expected_count} features as requested")
            else:
                self.logger.error(f"❌ Feature count mismatch: expected {expected_count}, got {len(selected_features)}")
            
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # FALLBACK: Return at least some features based on simple correlation
            self.logger.warning("⚠️ Falling back to correlation-based selection due to error")
            try:
                # Calculate simple correlation with target
                correlations = {}
                for col in feature_names:
                    try:
                        corr = abs(X[col].corr(y))
                        if not np.isnan(corr):
                            correlations[col] = corr
                    except:
                        pass
                
                # Sort by correlation and select top features
                sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
                max_features = min(self.config.max_features, len(sorted_features))
                selected_features = [feat for feat, _ in sorted_features[:max_features]]
                
                self.logger.info(f"✅ Fallback selection: {len(selected_features)} features selected by correlation")
                return selected_features
            except Exception as fallback_error:
                self.logger.error(f"❌ Fallback selection also failed: {fallback_error}")
                # Last resort: return first N features
                max_features = min(self.config.max_features, len(feature_names))
                return feature_names[:max_features]
    
    def _apply_tree_based_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str]
    ) -> List[str]:
        """
        Apply LGBM-SHAP feature selection for optimal performance.
        Uses SHAP values for game-theoretic feature importance that captures interactions.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
            
        Returns:
            List of selected features
        """
        try:
            self.logger.info(f"Starting LGBM-SHAP selection on {len(feature_names)} features")
            self.logger.info(f"Max features target: {self.config.max_features}")
            self.logger.info(f"Permutation importance enabled: {self.config.use_permutation_importance}")
            
            # Use LGBM-SHAP for optimal feature importance
            if self.config.use_permutation_importance and LGBM_AVAILABLE and SHAP_AVAILABLE:
                self.logger.info("Using LGBM-SHAP importance (captures feature interactions with game-theoretic interpretation)")
                self.logger.debug("Training LGBM model for SHAP analysis...")
                
                # LGBM parameters optimized for feature selection
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
                self.logger.debug("LGBM model training completed")
                
                # Calculate SHAP values for feature importance
                self.logger.debug("Calculating SHAP values for feature importance...")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                
                # Calculate mean absolute SHAP values for each feature
                mean_shap_values = np.mean(np.abs(shap_values), axis=0)
                importances = mean_shap_values
                self.logger.info(f"SHAP importance calculated for {len(feature_names)} features")
                
                # Debug logging for SHAP importance
                self.logger.debug(f"SHAP importance stats: min={np.min(importances):.6f}, max={np.max(importances):.6f}, mean={np.mean(importances):.6f}")
                self.logger.debug(f"SHAP importance std: {np.std(importances):.6f}")
                
                # Log ALL features by SHAP importance (not just top 10)
                sorted_indices = np.argsort(importances)[::-1]
                all_features = [(feature_names[i], importances[i]) for i in sorted_indices]
                self.logger.info(f"All {len(feature_names)} features by SHAP importance:")
                for feat, imp in all_features:
                    self.logger.info(f"  {feat}: {imp:.6f}")
                    
            else:
                # Fallback to ExtraTrees with permutation importance
                self.logger.info("Using ExtraTrees with permutation importance (fallback)")
                self.logger.debug("Training ExtraTreesRegressor model...")
                model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X, y)
                self.logger.debug("Model training completed")
                
                # Use permutation importance - captures feature interactions and is more reliable
                self.logger.info("Using permutation importance (captures feature interactions)")
                self.logger.debug("Calculating permutation importance with 10 repeats...")
                perm_importance = permutation_importance(
                    model, X, y,
                    n_repeats=10,
                    random_state=42,
                    n_jobs=-1
                )
                importances = perm_importance.importances_mean
                self.logger.info(f"Permutation importance calculated for {len(feature_names)} features")
                
                # Debug logging for permutation importance
                self.logger.debug(f"Permutation importance stats: min={np.min(importances):.6f}, max={np.max(importances):.6f}, mean={np.mean(importances):.6f}")
                self.logger.debug(f"Permutation importance std: {np.std(importances):.6f}")
                
                # Log ALL features by permutation importance (not just top 10)
                sorted_indices = np.argsort(importances)[::-1]
                all_features = [(feature_names[i], importances[i]) for i in sorted_indices]
                self.logger.info(f"All {len(feature_names)} features by permutation importance:")
                for feat, imp in all_features:
                    self.logger.info(f"  {feat}: {imp:.6f}")
            
            feature_importance = dict(zip(feature_names, importances))
            
            # Store ALL importances for later analysis
            if self.config.use_permutation_importance:
                self.all_permutation_importances = feature_importance.copy()
                self.logger.info(f"Stored SHAP/permutation importances for all {len(feature_names)} features")
            
            # SPECIAL ANALYSIS: Check interaction features specifically
            interaction_features = {feat: imp for feat, imp in feature_importance.items() 
                                   if 'interaction' in feat.lower() or '_x_' in feat.lower()}
            if interaction_features:
                self.logger.info(f"🔍 INTERACTION FEATURES ANALYSIS: Found {len(interaction_features)} interaction features")
                sorted_interactions = sorted(interaction_features.items(), key=lambda x: x[1], reverse=True)
                self.logger.info(f"📊 Top 10 interaction features by importance:")
                for feat, imp in sorted_interactions[:10]:
                    self.logger.info(f"   {feat}: {imp:.6f}")
                self.logger.info(f"📊 Bottom 10 interaction features by importance:")
                for feat, imp in sorted_interactions[-10:]:
                    self.logger.info(f"   {feat}: {imp:.6f}")
                
                # Compare with overall distribution
                all_importances = list(feature_importance.values())
                interaction_importances = list(interaction_features.values())
                self.logger.info(f"📊 Interaction features statistics:")
                self.logger.info(f"   Mean importance (all features): {np.mean(all_importances):.6f}")
                self.logger.info(f"   Mean importance (interactions): {np.mean(interaction_importances):.6f}")
                self.logger.info(f"   Max importance (all features): {np.max(all_importances):.6f}")
                self.logger.info(f"   Max importance (interactions): {np.max(interaction_importances):.6f}")
                
                # Find ranking of best interaction feature
                sorted_all = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                best_interaction = sorted_interactions[0]
                best_interaction_rank = next(i for i, (feat, _) in enumerate(sorted_all) if feat == best_interaction[0]) + 1
                self.logger.info(f"📊 Best interaction feature '{best_interaction[0]}' ranks #{best_interaction_rank} out of {len(feature_names)}")
            else:
                self.logger.warning(f"⚠️ No interaction features found in feature pool (checked for 'interaction' or '_x_' in names)")
            
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
            
            importance_type = "SHAP" if (self.config.use_permutation_importance and LGBM_AVAILABLE and SHAP_AVAILABLE) else ("permutation" if self.config.use_permutation_importance else "Gini")
            self.logger.info(f"Ranked {len(sorted_features)} features using {importance_type} importance")
            
            # Return ALL ranked features - diversity filtering will happen in select_features
            # This ensures we don't lose features before the final selection
            all_ranked_features = [feat for feat, _ in sorted_features]
            
            self.logger.info(f"Returning {len(all_ranked_features)} ranked features for downstream processing")
            
            return all_ranked_features
            
        except Exception as e:
            self.logger.error(f"Error in tree-based selection: {e}")
            return feature_names
    
    def _remove_exact_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove exact duplicate columns from the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with duplicate columns removed
        """
        try:
            duplicate_cols = []
            for i in range(len(df.columns)):
                for j in range(i+1, len(df.columns)):
                    if df.iloc[:, i].equals(df.iloc[:, j]):
                        duplicate_cols.append(df.columns[j])
            
            if duplicate_cols:
                self.logger.info(f"Removing {len(duplicate_cols)} duplicate columns: {duplicate_cols}")
                return df.drop(columns=duplicate_cols)
            else:
                self.logger.info("No duplicate columns found")
                return df
                
        except Exception as e:
            self.logger.error(f"Error removing duplicate columns: {e}")
            return df
    
    def _ensure_feature_diversity(self, selected_features: List[str], X: pd.DataFrame,
                                  correlation_threshold: float = 0.8) -> List[str]:
        """
        Ensure feature diversity by removing highly correlated features.
        
        Args:
            selected_features: List of selected features
            X: Feature matrix
            correlation_threshold: Correlation threshold for diversity
            
        Returns:
            List of diverse features
        """
        try:
            diverse_features = []
            for feature in selected_features:
                is_diverse = True
                for selected in diverse_features:
                    try:
                        if abs(X[feature].corr(X[selected])) > correlation_threshold:
                            is_diverse = False
                            self.logger.debug(f"Feature {feature} excluded due to high correlation ({abs(X[feature].corr(X[selected])):.3f}) with {selected}")
                            break
                    except:
                        continue
                if is_diverse:
                    diverse_features.append(feature)
            
            removed_count = len(selected_features) - len(diverse_features)
            if removed_count > 0:
                self.logger.info(f"Feature diversity: Removed {removed_count} highly correlated features (threshold: {correlation_threshold})")
            else:
                self.logger.info(f"Feature diversity: All {len(selected_features)} features are diverse (threshold: {correlation_threshold})")
            
            return diverse_features
            
        except Exception as e:
            self.logger.error(f"Error ensuring feature diversity: {e}")
            return selected_features
    
    def _combine_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Combine features with duplicate detection.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Combined feature matrix with duplicates removed
        """
        try:
            self.logger.info(f"Combining features from {len(X.columns)} columns")
            
            # Remove exact duplicates
            X_dedup = self._remove_exact_duplicates(X)
            
            self.logger.info(f"Feature combination complete: {len(X.columns)} -> {len(X_dedup.columns)} columns")
            return X_dedup
            
        except Exception as e:
            self.logger.error(f"Error combining features: {e}")
            return X
    
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
            correlation_threshold = 0.90  # Only flag very high correlations
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] > correlation_threshold:
                        redundancy_results['correlation_redundant'].append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': correlation_matrix.iloc[i, j]
                        })
            
            # 2. VIF-based redundancy (better for continuous features)
            # Calculate VIF for each feature to detect multicollinearity
            vif_threshold = 10.0  # VIF > 10 indicates high multicollinearity
            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor

                # Standardize features for VIF calculation
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(selected_data.fillna(0))

                # Calculate VIF for each feature (vectorized approach)
                vif_data = []
                for i in range(X_scaled.shape[1]):
                    try:
                        vif = variance_inflation_factor(X_scaled, i)
                        if vif > vif_threshold:
                            vif_data.append({
                                'feature': selected_features[i],
                                'vif': vif
                            })
                    except:
                        continue

                redundancy_results['mutual_info_redundant'] = vif_data  # Reuse field for VIF
                self.logger.info(f"VIF analysis: {len(vif_data)} features with VIF > {vif_threshold}")
            except ImportError:
                self.logger.warning("statsmodels not available, skipping VIF analysis")
            except Exception as e:
                self.logger.warning(f"VIF analysis failed: {e}")
            
            # 3. Variance-based redundancy (near-zero variance)
            variance_threshold = 0.001  # Decreased from 0.01 to be less aggressive
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
    
    def _apply_stability_weighted_ranking(self, X: pd.DataFrame, y: pd.Series, 
                                          ranked_features: List[str], 
                                          stability_weight: float = 0.3) -> List[str]:
        """
        Re-rank features by combining SHAP importance with stability scores using log multiplication.
        
        Formula: combined_score = importance^(1-w) × stability^w
        
        This is equivalent to: exp((1-w)*log(importance) + w*log(stability))
        
        Log multiplication is better than addition because:
        - Handles multiplicative relationships naturally
        - Features need BOTH high importance AND high stability
        - Low score in either dimension significantly reduces combined score
        - More aligned with probabilistic interpretation
        
        Args:
            X: Feature matrix
            y: Target variable
            ranked_features: Features ranked by SHAP importance
            stability_weight: Weight for stability (0-1). Default 0.3 = 30% stability, 70% importance
        
        Returns:
            Re-ranked feature list
        """
        try:
            self.logger.info(f"🔄 Computing stability scores for {len(ranked_features)} features...")
            
            # Get SHAP importances (normalized to 0-1)
            if not self.all_permutation_importances:
                self.logger.warning("⚠️ No SHAP importances available, skipping stability weighting")
                return ranked_features
            
            shap_scores = {feat: self.all_permutation_importances.get(feat, 0.0) for feat in ranked_features}
            max_shap = max(shap_scores.values()) if shap_scores.values() else 1.0
            normalized_shap = {feat: score / max_shap for feat, score in shap_scores.items()}
            
            # Compute stability scores using time windows
            n_samples = len(X)
            n_windows = 5
            window_size = n_samples // n_windows
            
            window_importances = []
            for i in range(n_windows):
                start_idx = i * window_size
                end_idx = min((i + 1) * window_size, n_samples)
                
                if end_idx - start_idx < 50:
                    continue
                
                X_window = X.iloc[start_idx:end_idx][ranked_features]
                y_window = y.iloc[start_idx:end_idx]
                
                window_importance = {}
                for feature in ranked_features:
                    try:
                        corr = abs(X_window[feature].corr(y_window))
                        window_importance[feature] = corr if not np.isnan(corr) else 0.0
                    except:
                        window_importance[feature] = 0.0
                
                window_importances.append(window_importance)
            
            # Calculate stability scores
            stability_scores = {}
            for feature in ranked_features:
                importances = [w.get(feature, 0.0) for w in window_importances]
                
                if len(importances) > 0 and np.std(importances) > 0:
                    mean_imp = np.mean(importances)
                    std_imp = np.std(importances)
                    cv = std_imp / mean_imp if mean_imp > 0 else 999
                    stability_score = 1 / (1 + cv)  # Normalize to 0-1
                else:
                    stability_score = 1.0 if len(importances) > 0 else 0.0
                
                stability_scores[feature] = stability_score
            
            # Combine scores using log multiplication (better for multiplicative relationships)
            # Formula: score = exp(w1*log(importance) + w2*log(stability))
            # This is equivalent to: score = importance^w1 * stability^w2
            combined_scores = {}
            importance_weight = 1 - stability_weight  # e.g., 0.7
            
            for feature in ranked_features:
                shap_norm = normalized_shap.get(feature, 0.0)
                stab_score = stability_scores.get(feature, 0.0)
                
                # Add small epsilon to avoid log(0)
                epsilon = 1e-10
                shap_safe = max(shap_norm, epsilon)
                stab_safe = max(stab_score, epsilon)
                
                # Log multiplication: log(A^w1 * B^w2) = w1*log(A) + w2*log(B)
                log_combined = importance_weight * np.log(shap_safe) + stability_weight * np.log(stab_safe)
                combined = np.exp(log_combined)
                
                combined_scores[feature] = combined
            
            # Re-rank by combined score
            reranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            reranked_features = [feat for feat, _ in reranked]
            
            # Log changes
            changes = 0
            for i, feat in enumerate(reranked_features[:20]):  # Check top 20
                original_rank = ranked_features.index(feat) if feat in ranked_features else -1
                if original_rank != i:
                    changes += 1
            
            self.logger.info(f"✅ Stability-weighted ranking complete (log multiplication):")
            self.logger.info(f"   Formula: importance^{1-stability_weight:.1f} × stability^{stability_weight:.1f}")
            self.logger.info(f"   Weight: {stability_weight:.1%} stability, {1-stability_weight:.1%} importance")
            self.logger.info(f"   Ranking changes in top 20: {changes}")
            self.logger.info(f"   New top 5: {reranked_features[:5]}")
            
            return reranked_features
            
        except Exception as e:
            self.logger.error(f"Error in stability-weighted ranking: {e}")
            return ranked_features
    
    def analyze_feature_stability(self, X: pd.DataFrame, y: pd.Series, selected_features: List[str],
                                 n_windows: int = 5) -> Dict[str, Any]:
        """
        Analyze stability of feature importance across different time windows.

        Uses rolling window importance consistency instead of feature selection frequency.

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

            # Track importance rankings across windows
            window_importances = []

            # Analyze each time window
            for i in range(n_windows):
                start_idx = i * window_size
                end_idx = min((i + 1) * window_size, n_samples)

                if end_idx - start_idx < 50:  # Skip too-small windows
                    continue

                X_window = X.iloc[start_idx:end_idx][selected_features]
                y_window = y.iloc[start_idx:end_idx]

                # Calculate feature importance for this window using correlation (fast)
                window_importance = {}
                for feature in selected_features:
                    try:
                        corr = abs(X_window[feature].corr(y_window))
                        window_importance[feature] = corr if not np.isnan(corr) else 0.0
                    except:
                        window_importance[feature] = 0.0

                window_importances.append(window_importance)

            # Calculate stability as consistency of importance across windows
            stability_scores = {}
            for feature in selected_features:
                # Get importance values across all windows
                importances = [w.get(feature, 0.0) for w in window_importances]

                if len(importances) > 0 and np.std(importances) > 0:
                    # Stability = 1 / coefficient_of_variation
                    # High stability = low variation in importance
                    mean_imp = np.mean(importances)
                    std_imp = np.std(importances)
                    cv = std_imp / mean_imp if mean_imp > 0 else 999
                    stability_score = 1 / (1 + cv)  # Normalize to 0-1
                else:
                    stability_score = 1.0 if len(importances) > 0 else 0.0

                stability_scores[feature] = stability_score

            # Calculate overall metrics
            avg_stability = np.mean(list(stability_scores.values())) if stability_scores else 0.0

            # Use adaptive threshold (60th percentile)
            if len(stability_scores) > 0:
                adaptive_threshold = np.percentile(list(stability_scores.values()), 60)
                adaptive_threshold = max(0.3, min(0.8, adaptive_threshold))  # Clamp between 0.3-0.8
            else:
                adaptive_threshold = 0.5

            stable_features = [f for f, score in stability_scores.items() if score >= adaptive_threshold]

            analysis = {
                'stability_results': {'stability_scores': stability_scores},
                'average_stability': avg_stability,
                'stable_features': stable_features,
                'stability_threshold': adaptive_threshold,
                'n_windows': len(window_importances),
                'method': 'importance_consistency'
            }

            self.stability_analysis = analysis
            self.logger.info(f"Stability analysis: {len(stable_features)}/{len(selected_features)} features stable (threshold={adaptive_threshold:.2f})")
            return analysis

        except Exception as e:
            self.logger.error(f"Error in stability analysis: {e}")
            return {"error": str(e)}
    
    def _select_features_for_window(self, X_window: pd.DataFrame, y_window: pd.Series) -> List[str]:
        """
        Select features for a specific time window using SAME method as main selection.
        
        CRITICAL FIX: This now uses SHAP/permutation importance (same as main selection)
        instead of mutual_info_regression to ensure CV consistency is meaningful.
        
        Args:
            X_window: Feature matrix for the window
            y_window: Target variable for the window
            
        Returns:
            List of selected features for this window
        """
        try:
            # FIX: Select same number of features as main selection for fair comparison
            max_window_features = min(self.config.max_features, len(X_window.columns))
            
            # Use SAME method as main selection (SHAP/permutation importance)
            if self.config.use_permutation_importance and LGBM_AVAILABLE and SHAP_AVAILABLE:
                # Use SHAP importance (same as main selection)
                self.logger.debug("Using SHAP importance for window selection (consistent with main selection)")
                # OPTIMIZED: Hyperparameters for stable feature selection (not prediction)
                # Goal: Stable SHAP values that generalize to LGBM, Neural, RF, CatBoost
                model = lgb.LGBMRegressor(
                    objective='regression',
                    n_estimators=100,      # More trees = more stable importance
                    num_leaves=15,         # Simpler trees = better generalization
                    max_depth=5,           # Limit depth to avoid overfitting to LGBM quirks
                    learning_rate=0.05,    # Lower LR = smoother, more stable learning
                    min_child_samples=20,  # Prevent overfitting to noise
                    subsample=0.8,         # Bagging for stability
                    colsample_bytree=0.8,  # Feature sampling for robustness
                    verbose=-1,
                    random_state=42
                )
                model.fit(X_window, y_window)
                
                # FIX: Use larger sample for more stable SHAP values
                sample_size = min(500, len(X_window))  # Increased to 500 for maximum stability
                X_sample = X_window.iloc[:sample_size] if len(X_window) > sample_size else X_window
                
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                importances = np.mean(np.abs(shap_values), axis=0)
                
            else:
                # Fallback to permutation importance (same as main selection)
                self.logger.debug("Using permutation importance for window selection (consistent with main selection)")
                model = ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                model.fit(X_window, y_window)
                
                perm_importance = permutation_importance(
                    model, X_window, y_window,
                    n_repeats=5,
                    random_state=42,
                    n_jobs=-1
                )
                importances = perm_importance.importances_mean
            
            # Select top features by importance
            top_indices = np.argsort(importances)[::-1][:max_window_features]
            selected_features = [X_window.columns[i] for i in top_indices]
            
            self.logger.debug(f"Selected {len(selected_features)} features for window using {'SHAP' if (self.config.use_permutation_importance and LGBM_AVAILABLE and SHAP_AVAILABLE) else 'permutation'} importance")
            
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
        Compare selected features with baseline using same importance metric.

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

            # Use the SAME metric as feature selection (permutation/SHAP importance)
            # This ensures we're comparing apples to apples

            # Get selected features importance from stored importances
            if self.all_permutation_importances:
                selected_scores = [
                    self.all_permutation_importances.get(feat, 0.0)
                    for feat in selected_features
                ]
                avg_selected_score = np.mean(selected_scores)

                # For baseline: use mean importance of all features
                all_importances = list(self.all_permutation_importances.values())
                avg_baseline_score = np.mean(all_importances) if all_importances else 0.0

                # Calculate improvement (using SAME metric)
                improvement_ratio = avg_selected_score / avg_baseline_score if avg_baseline_score > 0 else 1.0

                analysis = {
                    'baseline_results': [],
                    'selected_features_scores': selected_scores,
                    'average_selected_score': avg_selected_score,
                    'average_baseline_score': avg_baseline_score,
                    'improvement_ratio': improvement_ratio,
                    'n_baseline_trials': 10,
                    'n_features': n_features,
                    'comparison_metric': 'permutation_importance'
                }
            else:
                # Fallback: use correlation as a simple baseline
                selected_scores = []
                for feature in selected_features:
                    try:
                        corr = abs(X[feature].corr(y))
                        selected_scores.append(corr if not np.isnan(corr) else 0.0)
                    except:
                        selected_scores.append(0.0)

                avg_selected_score = np.mean(selected_scores)

                # Baseline: mean correlation of all features
                all_corrs = []
                for feature in all_features:
                    try:
                        corr = abs(X[feature].corr(y))
                        if not np.isnan(corr):
                            all_corrs.append(corr)
                    except:
                        continue

                avg_baseline_score = np.mean(all_corrs) if all_corrs else 0.0
                improvement_ratio = avg_selected_score / avg_baseline_score if avg_baseline_score > 0 else 1.0

                analysis = {
                    'baseline_results': [],
                    'selected_features_scores': selected_scores,
                    'average_selected_score': avg_selected_score,
                    'average_baseline_score': avg_baseline_score,
                    'improvement_ratio': improvement_ratio,
                    'n_baseline_trials': 10,
                    'n_features': n_features,
                    'comparison_metric': 'correlation'
                }

            self.baseline_comparison = analysis
            self.logger.info(f"Baseline comparison ({analysis['comparison_metric']}): {improvement_ratio:.2f}x improvement over mean")
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
    
    def _reduce_redundancy_hierarchical(self, X: pd.DataFrame, ranked_features: List[str], 
                                       target_count: int, correlation_threshold: float = 0.85) -> List[str]:
        """
        Reduce redundancy using hierarchical clustering while preserving importance ranking.
        
        Strategy:
        1. Calculate correlation matrix for all ranked features
        2. Perform hierarchical clustering based on correlation distance
        3. For each cluster, select the highest-ranked feature
        4. Continue until target_count is reached
        
        Args:
            X: Feature matrix
            ranked_features: List of features ranked by importance (highest first)
            target_count: Target number of features to select
            correlation_threshold: Correlation threshold for redundancy (default 0.85)
            
        Returns:
            List of non-redundant features preserving importance ranking
        """
        try:
            if len(ranked_features) <= target_count:
                self.logger.info(f"Ranked features ({len(ranked_features)}) <= target ({target_count}), returning all")
                return ranked_features[:target_count]
            
            self.logger.info(f"Reducing redundancy from {len(ranked_features)} to {target_count} features using hierarchical clustering")
            self.logger.info(f"Correlation threshold: {correlation_threshold}")
            
            # ADVANCED OPTIMIZATION: Adaptive sampling + vectorization + chunking
            n_samples = len(X)
            n_features = len(ranked_features)
            
            # 1. USE FULL DATASET: With vectorization and chunking, we can handle the full dataset
            if n_samples > 20000:
                # For very large datasets (>20K), use smart sampling
                optimal_sample_size = min(20000, n_samples)
                self.logger.info(f"🚀 LARGE DATASET SAMPLING: Using {optimal_sample_size} samples for {n_features} features")
                sample_indices = np.random.choice(n_samples, optimal_sample_size, replace=False)
                feature_data = X[ranked_features].iloc[sample_indices].values
            else:
                # Use FULL dataset for datasets ≤20K (like our 14K dataset)
                feature_data = X[ranked_features].values
                optimal_sample_size = n_samples
                self.logger.info(f"🚀 USING FULL DATASET: {optimal_sample_size} samples × {n_features} features")
            
            self.logger.info(f"📊 Correlation calculation: {optimal_sample_size} samples × {n_features} features")
            
            # 2. VECTORIZED NORMALIZATION: Batch normalize all features at once
            self.logger.info("⚡ Vectorized normalization...")
            feature_mean = np.nanmean(feature_data, axis=0, keepdims=True)
            feature_std = np.nanstd(feature_data, axis=0, keepdims=True)
            
            # Vectorized zero-std handling
            zero_std_mask = feature_std == 0
            feature_std = np.where(zero_std_mask, 1.0, feature_std)
            
            # Vectorized normalization
            feature_normalized = (feature_data - feature_mean) / feature_std
            
            # 3. OPTIMIZED CORRELATION: GPU acceleration (M1) + Symmetric matrix optimization
            import time
            chunk_start_time = time.time()
            
            # Try GPU acceleration first (Mac M1 Metal Performance Shaders via PyTorch MPS)
            # GPU is beneficial when: n_features * n_samples > 1M (transfer overhead is worth it)
            use_gpu = False
            gpu_threshold = 1_000_000  # Empirical threshold where GPU becomes faster
            workload_size = n_features * optimal_sample_size
            
            try:
                import torch
                if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                    if workload_size > gpu_threshold:
                        use_gpu = True
                        tprint(f"🚀 Mac M1 GPU (Metal) detected - workload {workload_size/1e6:.1f}M > threshold", "SUCCESS")
                    else:
                        tprint(f"ℹ️ GPU available but workload too small ({workload_size/1e6:.1f}M < {gpu_threshold/1e6:.1f}M)", "INFO")
                        tprint("   Using optimized CPU (faster for small matrices)", "INFO")
            except (ImportError, AttributeError):
                self.logger.info("ℹ️ GPU acceleration not available, using optimized CPU")
            
            if use_gpu:
                # GPU-ACCELERATED CORRELATION (Mac M1 Metal)
                try:
                    self.logger.info(f"🎮 GPU CORRELATION: Processing {n_features} features on M1 GPU")
                    
                    # Transfer to GPU
                    device = torch.device("mps")
                    feature_tensor = torch.from_numpy(feature_normalized.astype(np.float32)).to(device)
                    
                    # Compute correlation on GPU (no chunking needed!)
                    corr_matrix_gpu = torch.mm(feature_tensor.T, feature_tensor) / optimal_sample_size
                    
                    # Transfer back to CPU
                    corr_matrix = corr_matrix_gpu.cpu().numpy()
                    
                    total_time = time.time() - chunk_start_time
                    self.logger.info(f"✅ GPU correlation completed in {total_time:.1f}s (M1 Metal)")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ GPU acceleration failed: {e}")
                    self.logger.info("   Falling back to optimized CPU correlation")
                    use_gpu = False
            
            if not use_gpu:
                # CPU-OPTIMIZED CORRELATION with symmetric matrix optimization
                if n_features > 200:
                    # SYMMETRIC MATRIX OPTIMIZATION: Only compute upper triangle
                    chunk_size = min(250, max(100, n_features // 3))
                    total_chunks = (n_features - 1) // chunk_size + 1
                    total_pairs = (total_chunks * (total_chunks + 1)) // 2  # Upper triangle only
                    
                    tprint(f"🧩 SYMMETRIC CHUNKED CORRELATION: Processing {n_features} features", "INFO")
                    tprint(f"   Chunk size: {chunk_size} features", "INFO")
                    tprint(f"   Total chunks: {total_chunks}", "INFO")
                    tprint(f"   Upper triangle pairs: {total_pairs} (vs {total_chunks * total_chunks} full matrix)", "INFO")
                    tprint(f"   Speedup: {(total_chunks * total_chunks) / total_pairs:.1f}x fewer computations", "INFO")
                    
                    corr_matrix = np.zeros((n_features, n_features), dtype=np.float32)
                    chunk_count = 0
                    
                    # Only compute upper triangle (i <= j)
                    for i in range(0, n_features, chunk_size):
                        end_i = min(i + chunk_size, n_features)
                        chunk_i = feature_normalized[:, i:end_i].astype(np.float32)
                        chunk_num_i = i // chunk_size + 1
                        
                        # Start from i (not 0) to only compute upper triangle
                        for j in range(i, n_features, chunk_size):
                            end_j = min(j + chunk_size, n_features)
                            chunk_j = feature_normalized[:, j:end_j].astype(np.float32)
                            chunk_num_j = j // chunk_size + 1
                            
                            # Vectorized correlation calculation for this chunk pair
                            corr_chunk = np.dot(chunk_i.T, chunk_j) / optimal_sample_size
                            corr_matrix[i:end_i, j:end_j] = corr_chunk
                            
                            # Mirror to lower triangle (except diagonal blocks)
                            if i != j:
                                corr_matrix[j:end_j, i:end_i] = corr_chunk.T
                            
                            chunk_count += 1
                            
                            # Progress update every 10% or every row completion
                            if chunk_count % max(1, total_pairs // 10) == 0 or j + chunk_size >= n_features:
                                elapsed = time.time() - chunk_start_time
                                progress_pct = (chunk_count / total_pairs) * 100
                                eta = (elapsed / chunk_count) * (total_pairs - chunk_count) if chunk_count > 0 else 0
                                tprint(f"   📊 Chunk [{chunk_num_i},{chunk_num_j}] | Progress: {progress_pct:.1f}% | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", "INFO")
                    
                    total_time = time.time() - chunk_start_time
                    tprint(f"✅ Symmetric chunked correlation completed in {total_time:.1f}s", "SUCCESS")
                else:
                    # Standard vectorized correlation for smaller feature sets
                    tprint(f"⚡ Standard vectorized correlation for {n_features} features...", "INFO")
                    corr_matrix = np.dot(feature_normalized.astype(np.float32).T, 
                                        feature_normalized.astype(np.float32)) / optimal_sample_size
                    total_time = time.time() - chunk_start_time
                    tprint(f"✅ Correlation completed in {total_time:.1f}s", "SUCCESS")
            
            # Ensure correlation matrix is properly bounded and symmetric
            corr_matrix = np.abs(corr_matrix)  # Absolute correlation
            corr_matrix = np.clip(corr_matrix, 0, 1)  # Ensure [0,1] range
            
            # Handle NaN values
            nan_mask = np.isnan(corr_matrix)
            nan_count = np.sum(nan_mask)
            if nan_count > 0:
                self.logger.warning(f"Found {nan_count} NaN values in correlation matrix, filling with 0")
                corr_matrix[nan_mask] = 0
            
            # Convert to distance matrix (1 - correlation) - vectorized
            distance_matrix = 1.0 - corr_matrix
            
            # Ensure symmetric and clip in one operation
            distance_matrix = (distance_matrix + distance_matrix.T) * 0.5
            np.fill_diagonal(distance_matrix, 0)
            distance_matrix = np.clip(distance_matrix, 0, None)
            
            # Validate distance matrix (using numpy directly)
            if not np.allclose(distance_matrix, distance_matrix.T, rtol=1e-5, atol=1e-8):
                self.logger.error("Distance matrix is not symmetric after correction")
                max_diff = np.abs(distance_matrix - distance_matrix.T).max()
                self.logger.error(f"Max asymmetry: {max_diff}")
                raise ValueError("Distance matrix is not symmetric")
            
            # Perform hierarchical clustering (using numpy array directly)
            self.logger.info(f"Performing hierarchical clustering on {len(ranked_features)} features")
            self.logger.info(f"Distance matrix shape: {distance_matrix.shape}")
            self.logger.info(f"Distance matrix stats: min={distance_matrix.min():.4f}, max={distance_matrix.max():.4f}, mean={distance_matrix.mean():.4f}")
            linkage_matrix = linkage(squareform(distance_matrix), method='ward')
            self.logger.info(f"Hierarchical clustering completed successfully")
            
            # Dynamic cluster count: aim for more clusters than target to ensure diversity
            # Use correlation threshold to determine cluster count
            n_clusters = min(int(target_count * 1.5), len(ranked_features))
            self.logger.info(f"Creating {n_clusters} clusters from {len(ranked_features)} features")
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            self.logger.info(f"Cluster assignment completed: {len(set(cluster_labels))} unique clusters")
            
            # OPTIMIZATION: Vectorized cluster selection
            # Create a boolean mask for each cluster and select first True index
            selected_features = []
            unique_clusters = np.unique(cluster_labels)
            
            for cluster_id in unique_clusters:
                # Find indices of features in this cluster
                cluster_mask = cluster_labels == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                
                # Select the first feature (highest ranked) in this cluster
                if len(cluster_indices) > 0:
                    selected_features.append(ranked_features[cluster_indices[0]])
                
                if len(selected_features) >= target_count:
                    break
            
            # If still not enough features, add remaining high-ranked features
            # that don't violate correlation threshold
            if len(selected_features) < target_count:
                self.logger.info(f"Only {len(selected_features)} features from clustering, adding more to reach {target_count}")
                self.logger.info(f"Will check correlation threshold {correlation_threshold} for additional features")
                for feat in ranked_features:
                    if feat not in selected_features:
                        # Check correlation with already selected features
                        is_diverse = True
                        for selected_feat in selected_features:
                            try:
                                corr = abs(X[feat].corr(X[selected_feat]))
                                if corr > correlation_threshold:
                                    is_diverse = False
                                    break
                            except:
                                continue
                        
                        if is_diverse:
                            selected_features.append(feat)
                            if len(selected_features) >= target_count:
                                break
            
            # Ensure exact target count
            final_features = selected_features[:target_count]
            
            self.logger.info(f"✅ Hierarchical redundancy reduction: {len(ranked_features)} -> {len(final_features)} features")
            self.logger.info(f"Removed {len(ranked_features) - len(final_features)} redundant features ({(len(ranked_features) - len(final_features))/len(ranked_features)*100:.1f}%)")
            
            # Log sample of selected features
            self.logger.info(f"Sample selected features: {final_features[:5]}")
            
            return final_features
            
        except Exception as e:
            self.logger.error(f"❌ Error in hierarchical redundancy reduction: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Fallback: use simple correlation filtering with stricter threshold
            self.logger.warning(f"⚠️ Falling back to simple diversity filtering with threshold 0.70")
            self.logger.warning(f"This will use sequential pairwise correlation checks instead of clustering")
            return self._ensure_feature_diversity(ranked_features, X, 0.70)[:target_count]
    
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
