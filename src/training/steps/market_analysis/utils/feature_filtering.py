"""Feature Filtering Module for Step 7 Enhanced Matrix Operations.

This module provides regime-aware feature filtering with comprehensive
feature selection algorithms including MI, SHAP, and regime-specific analysis.
"""
from datetime import datetime
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

# Optional dependencies with fallback handling
try:
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    from sklearn.feature_selection import mutual_info_classif
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    mutual_info_classif = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    from scipy.stats import rankdata
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    rankdata = None


class FeatureFiltering:
    """Regime-aware feature filtering with comprehensive selection algorithms."""
    
    def __init__(self, logger, config: Dict[str, Any]):
        self.logger = logger
        self.config = config
        
        # Feature selection configuration
        self.target_features = config.get("target_features", 200)
        self.removal_fraction = config.get("removal_fraction", 0.33)
        self.enable_regime_selection = config.get("enable_regime_selection", True)
        self.enable_shap_filtering = config.get("enable_shap_filtering", True)
    
    def regime_aware_initial_filtering(
        self, 
        features_df: pd.DataFrame, 
        labels_df: pd.DataFrame,
        regime_labels: pd.Series = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Conservative feature filtering with per-regime awareness.
        Removes bottom 33% of features to arrive at ~200 features.
        
        Args:
            features_df: Feature dataframe
            labels_df: Labels dataframe with target column
            regime_labels: Optional regime labels for per-regime selection
            
        Returns:
            Filtered features and metadata
        """
        if not NUMPY_AVAILABLE or not PANDAS_AVAILABLE:
            return features_df, {'error': 'NumPy or Pandas not available'}
        
        try:
            self.logger.info(f"🎯 Starting regime-aware feature filtering: {features_df.shape[1]} features")
            
            # Extract target variable
            if 'target' in labels_df.columns:
                y = labels_df['target']
            elif 'direction' in labels_df.columns:
                y = labels_df['direction']
            else:
                raise ValueError("No target or direction column found in labels")
            
            # Ensure binary target
            if y.dtype != int:
                y = (y > 0).astype(int)
            
            # 1. Per-regime feature importance
            regime_importances = {}
            if self.enable_regime_selection and regime_labels is not None:
                self.logger.info("📊 Calculating per-regime feature importance...")
                for regime in np.unique(regime_labels):
                    regime_mask = regime_labels == regime
                    if regime_mask.sum() < 100:  # Skip small regimes
                        continue
                        
                    X_regime = features_df[regime_mask]
                    y_regime = y[regime_mask]
                    
                    # Fast MI calculation per regime
                    if SKLEARN_AVAILABLE:
                        mi_scores = mutual_info_classif(X_regime, y_regime, random_state=42)
                    else:
                        self.logger.warning("⚠️ sklearn not available, using variance-based importance")
                        mi_scores = X_regime.var().values
                    regime_importances[f'regime_{regime}'] = mi_scores
                
                # Aggregate importance across regimes (keep features important in ANY regime)
                aggregated_importance = np.max(
                    np.vstack(list(regime_importances.values())), 
                    axis=0
                )
            else:
                # Calculate MI for all data
                if SKLEARN_AVAILABLE:
                    aggregated_importance = mutual_info_classif(features_df, y, random_state=42)
                else:
                    self.logger.warning("⚠️ sklearn not available, using variance-based importance")
                    aggregated_importance = features_df.var().values
            
            # 2. Quick SHAP sampling (subsample for speed)
            shap_importance = None
            if self.enable_shap_filtering and LIGHTGBM_AVAILABLE:
                self.logger.info("🔮 Calculating SHAP-based importance (sampled)...")
                try:
                    # Subsample for efficiency
                    sample_size = min(5000, len(features_df))
                    if len(features_df) > sample_size:
                        sample_idx = np.random.choice(len(features_df), sample_size, replace=False)
                        X_sample = features_df.iloc[sample_idx]
                        y_sample = y.iloc[sample_idx]
                    else:
                        X_sample, y_sample = features_df, y
                    
                    # Train lightweight model
                    lgb_model = lgb.LGBMClassifier(
                        n_estimators=100, 
                        max_depth=5,
                        n_jobs=-1,
                        verbose=-1,
                        random_state=42
                    )
                    lgb_model.fit(X_sample, y_sample)
                    
                    # Get feature importance
                    shap_importance = lgb_model.feature_importances_
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ SHAP calculation failed, using MI only: {e}")
            
            # 3. Combined scoring
            if SCIPY_AVAILABLE:
                mi_rank = rankdata(aggregated_importance)
                
                if shap_importance is not None:
                    shap_rank = rankdata(shap_importance)
                    combined_rank = (mi_rank + shap_rank) / 2
                else:
                    combined_rank = mi_rank
            else:
                self.logger.warning("⚠️ scipy not available, using simple sorting")
                # Simple ranking without scipy
                sorted_indices = np.argsort(aggregated_importance)
                combined_rank = np.zeros_like(aggregated_importance)
                combined_rank[sorted_indices] = np.arange(len(aggregated_importance))
            
            # 4. Remove bottom features, ensure minimum target
            n_features_to_keep = max(self.target_features, int(len(combined_rank) * (1 - self.removal_fraction)))
            top_features_idx = np.argsort(combined_rank)[-n_features_to_keep:]
            
            # Get feature names
            selected_features = features_df.columns[top_features_idx].tolist()
            removed_features = [f for f in features_df.columns if f not in selected_features]
            
            # Create filtered dataframe
            filtered_df = features_df[selected_features]
            
            # 5. Generate metadata
            metadata = {
                'original_features': len(features_df.columns),
                'selected_features': len(selected_features),
                'removed_features': len(removed_features),
                'removal_fraction': len(removed_features) / len(features_df.columns),
                'regime_importances': regime_importances if regime_importances else None,
                'method': 'MI + SHAP ranking' if shap_importance is not None else 'MI ranking only',
                'removed_feature_names': removed_features[:50],  # Store first 50 for reference
                'top_features_by_mi': features_df.columns[np.argsort(aggregated_importance)[-20:]].tolist(),
                'selection_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Feature filtering complete: {len(features_df.columns)} → {len(selected_features)} features")
            self.logger.info(f"   Removed {len(removed_features)} features ({metadata['removal_fraction']:.1%})")
            
            return filtered_df, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Feature filtering failed: {e}")
            # Return original features if filtering fails
            return features_df, {
                'error': str(e),
                'original_features': len(features_df.columns),
                'selected_features': len(features_df.columns),
                'method': 'filtering_failed'
            }
    
    def apply_variance_filtering(self, features_df: pd.DataFrame, variance_threshold: float = 1e-6) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply variance-based filtering to remove low-variance features."""
        if not NUMPY_AVAILABLE or not PANDAS_AVAILABLE:
            return features_df, {'error': 'NumPy or Pandas not available'}
        
        try:
            self.logger.info(f"🔍 Applying variance filtering (threshold: {variance_threshold})")
            
            # Calculate variances
            variances = features_df.var()
            
            # Find features with sufficient variance
            high_variance_features = variances[variances >= variance_threshold].index.tolist()
            low_variance_features = variances[variances < variance_threshold].index.tolist()
            
            # Filter dataframe
            filtered_df = features_df[high_variance_features]
            
            metadata = {
                'original_features': len(features_df.columns),
                'filtered_features': len(high_variance_features),
                'removed_features': len(low_variance_features),
                'variance_threshold': variance_threshold,
                'removed_feature_names': low_variance_features[:50],  # Store first 50 for reference
                'method': 'variance_filtering'
            }
            
            self.logger.info(f"✅ Variance filtering complete: {len(features_df.columns)} → {len(high_variance_features)} features")
            self.logger.info(f"   Removed {len(low_variance_features)} low-variance features")
            
            return filtered_df, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Variance filtering failed: {e}")
            return features_df, {'error': str(e)}
    
    def apply_correlation_filtering(self, features_df: pd.DataFrame, correlation_threshold: float = 0.95) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply correlation-based filtering to remove highly correlated features."""
        if not NUMPY_AVAILABLE or not PANDAS_AVAILABLE:
            return features_df, {'error': 'NumPy or Pandas not available'}
        
        try:
            self.logger.info(f"🔗 Applying correlation filtering (threshold: {correlation_threshold})")
            
            # Calculate correlation matrix
            corr_matrix = features_df.corr().abs()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            features_to_remove = set()
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] >= correlation_threshold:
                        feature1 = corr_matrix.columns[i]
                        feature2 = corr_matrix.columns[j]
                        high_corr_pairs.append((feature1, feature2, corr_matrix.iloc[i, j]))
                        
                        # Remove the feature with lower variance
                        if features_df[feature1].var() < features_df[feature2].var():
                            features_to_remove.add(feature1)
                        else:
                            features_to_remove.add(feature2)
            
            # Filter dataframe
            remaining_features = [f for f in features_df.columns if f not in features_to_remove]
            filtered_df = features_df[remaining_features]
            
            metadata = {
                'original_features': len(features_df.columns),
                'filtered_features': len(remaining_features),
                'removed_features': len(features_to_remove),
                'correlation_threshold': correlation_threshold,
                'high_correlation_pairs': len(high_corr_pairs),
                'removed_feature_names': list(features_to_remove)[:50],  # Store first 50 for reference
                'method': 'correlation_filtering'
            }
            
            self.logger.info(f"✅ Correlation filtering complete: {len(features_df.columns)} → {len(remaining_features)} features")
            self.logger.info(f"   Removed {len(features_to_remove)} highly correlated features")
            
            return filtered_df, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Correlation filtering failed: {e}")
            return features_df, {'error': str(e)}
    
    def apply_combined_filtering(
        self, 
        features_df: pd.DataFrame, 
        labels_df: pd.DataFrame,
        regime_labels: pd.Series = None,
        variance_threshold: float = 1e-6,
        correlation_threshold: float = 0.95
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply combined filtering: variance -> correlation -> regime-aware selection."""
        if not NUMPY_AVAILABLE or not PANDAS_AVAILABLE:
            return features_df, {'error': 'NumPy or Pandas not available'}
        
        try:
            self.logger.info("🔄 Applying combined filtering pipeline...")
            
            # Step 1: Variance filtering
            df_variance, variance_metadata = self.apply_variance_filtering(features_df, variance_threshold)
            
            # Step 2: Correlation filtering
            df_correlation, correlation_metadata = self.apply_correlation_filtering(df_variance, correlation_threshold)
            
            # Step 3: Regime-aware filtering
            df_final, regime_metadata = self.regime_aware_initial_filtering(df_correlation, labels_df, regime_labels)
            
            # Combine metadata
            combined_metadata = {
                'pipeline_steps': ['variance_filtering', 'correlation_filtering', 'regime_aware_filtering'],
                'variance_filtering': variance_metadata,
                'correlation_filtering': correlation_metadata,
                'regime_aware_filtering': regime_metadata,
                'original_features': len(features_df.columns),
                'final_features': len(df_final.columns),
                'total_removed': len(features_df.columns) - len(df_final.columns),
                'removal_ratio': (len(features_df.columns) - len(df_final.columns)) / len(features_df.columns),
                'method': 'combined_filtering_pipeline'
            }
            
            self.logger.info(f"✅ Combined filtering complete: {len(features_df.columns)} → {len(df_final.columns)} features")
            self.logger.info(f"   Total removal ratio: {combined_metadata['removal_ratio']:.1%}")
            
            return df_final, combined_metadata
            
        except Exception as e:
            self.logger.error(f"❌ Combined filtering failed: {e}")
            return features_df, {'error': str(e)}


__all__ = ['FeatureFiltering']