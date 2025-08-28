#!/usr/bin/env python3
"""Enhanced Feature Selection Manager with Data-Driven Methods.

This module provides advanced feature selection using VIF, Mutual Information,
SHAP, RandomForest, and other data-driven methods instead of domain prioritization.
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.vif_calculator import calculate_vif_robust


class EnhancedFeatureSelectionManager:
    """Enhanced Feature Selection Manager using data-driven methods.
    
    This class provides comprehensive feature selection using:
    - VIF (Variance Inflation Factor) for multicollinearity
    - Mutual Information for feature relevance
    - SHAP for model-based importance
    - RandomForest for ensemble-based importance
    - Recursive Feature Elimination (RFE)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the enhanced feature selection manager.
        
        Args:
            config: Configuration dictionary with selection parameters
        """
        self.config = config
        self.logger = system_logger.getChild("EnhancedFeatureSelectionManager")
        
        # Feature selection configuration
        self.target_features = config.get("feature_reduction", {}).get("target_features", 100)
        self.vif_threshold = config.get("feature_reduction", {}).get("vif_threshold", 10.0)
        self.mi_threshold = config.get("feature_reduction", {}).get("mi_threshold", 0.01)
        self.correlation_threshold = config.get("feature_reduction", {}).get("correlation_threshold", 0.95)
        self.variance_threshold = config.get("feature_reduction", {}).get("variance_threshold", 0.01)
        
        # Method weights for ensemble selection
        self.method_weights = config.get("feature_reduction", {}).get("method_weights", {
            "vif": 0.2,
            "mutual_info": 0.25,
            "shap": 0.25,
            "random_forest": 0.2,
            "rfe": 0.1
        })
        
        # Feature importance cache
        self.feature_importance_cache = {}
        self.selection_metadata = {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=(pd.DataFrame(), {}),
        context="enhanced_feature_selection"
    )
    def select_features_enhanced(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        symbol: str,
        exchange: str,
        data_dir: str,
        task: str = "classification"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Enhanced feature selection using data-driven methods.
        
        Args:
            features_df: Input features DataFrame
            target: Target variable series
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory for saving metadata
            task: Task type ("classification" or "regression")
            
        Returns:
            Tuple of (selected_features_df, selection_metadata)
        """
        start_time = time.time()
        
        self.logger.info(f"🔍 Starting enhanced data-driven feature selection: {features_df.shape[1]} -> {self.target_features} features")
        
        # Stage 1: Data quality filtering
        features_df, stage1_metadata = self._stage1_data_quality_filtering(features_df)
        
        # Stage 2: Variance-based filtering
        features_df, stage2_metadata = self._stage2_variance_filtering(features_df)
        
        # Stage 3: VIF-based multicollinearity filtering
        features_df, stage3_metadata = self._stage3_vif_filtering(features_df)
        
        # Stage 4: Correlation-based filtering
        features_df, stage4_metadata = self._stage4_correlation_filtering(features_df)
        
        # Stage 5: Mutual Information ranking
        features_df, stage5_metadata = self._stage5_mutual_info_ranking(features_df, target, task)
        
        # Stage 6: SHAP-based importance
        features_df, stage6_metadata = self._stage6_shap_importance(features_df, target, task)
        
        # Stage 7: RandomForest importance
        features_df, stage7_metadata = self._stage7_random_forest_importance(features_df, target, task)
        
        # Stage 8: Ensemble selection
        features_df, stage8_metadata = self._stage8_ensemble_selection(features_df, target, task)
        
        # Stage 9: Final RFE selection
        features_df, stage9_metadata = self._stage9_final_rfe_selection(features_df, target, task)
        
        # Compile metadata
        processing_time = time.time() - start_time
        selection_metadata = {
            "original_features": len(features_df.columns),
            "final_features": len(features_df.columns),
            "target_features": self.target_features,
            "processing_time": processing_time,
            "stages": {
                "stage1_data_quality": stage1_metadata,
                "stage2_variance": stage2_metadata,
                "stage3_vif": stage3_metadata,
                "stage4_correlation": stage4_metadata,
                "stage5_mutual_info": stage5_metadata,
                "stage6_shap": stage6_metadata,
                "stage7_random_forest": stage7_metadata,
                "stage8_ensemble": stage8_metadata,
                "stage9_final_rfe": stage9_metadata,
            },
            "feature_importance_scores": self.feature_importance_cache,
            "selection_timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "exchange": exchange,
        }
        
        # Save selection metadata
        self._save_selection_metadata(selection_metadata, symbol, exchange, data_dir)
        
        self.logger.info(f"✅ Enhanced feature selection completed: {len(features_df.columns)} features selected")
        self.logger.info(f"   - Processing time: {processing_time:.2f}s")
        self.logger.info(f"   - VIF features removed: {stage3_metadata.get('removed_high_vif', 0)}")
        self.logger.info(f"   - MI features removed: {stage5_metadata.get('removed_low_mi', 0)}")
        self.logger.info(f"   - SHAP features removed: {stage6_metadata.get('removed_low_shap', 0)}")
        
        return features_df, selection_metadata

    def _stage1_data_quality_filtering(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Stage 1: Remove features with poor data quality."""
        original_count = len(features_df.columns)
        
        # Remove features with too many NaN values (>10%)
        nan_ratio = features_df.isna().sum() / len(features_df)
        high_nan_features = nan_ratio[nan_ratio > 0.1].index.tolist()
        features_df = features_df.drop(columns=high_nan_features)
        
        # Remove features with infinite values
        inf_features = []
        for col in features_df.columns:
            if np.isinf(features_df[col]).any():
                inf_features.append(col)
        features_df = features_df.drop(columns=inf_features)
        
        # Fill remaining NaN values with forward fill then backward fill
        features_df = features_df.fillna(method="ffill").fillna(method="bfill").fillna(0)
        
        metadata = {
            "removed_high_nan": len(high_nan_features),
            "removed_infinite": len(inf_features),
            "features_after_stage": len(features_df.columns),
        }
        
        self.logger.info(f"Stage 1: Removed {original_count - len(features_df.columns)} low-quality features")
        return features_df, metadata

    def _stage2_variance_filtering(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Stage 2: Remove low-variance features."""
        original_count = len(features_df.columns)
        
        # Calculate variance for each feature
        variances = features_df.var()
        
        # Remove features with variance below threshold
        low_variance_features = variances[variances < self.variance_threshold].index.tolist()
        features_df = features_df.drop(columns=low_variance_features)
        
        metadata = {
            "removed_low_variance": len(low_variance_features),
            "variance_threshold": self.variance_threshold,
            "features_after_stage": len(features_df.columns),
        }
        
        self.logger.info(f"Stage 2: Removed {len(low_variance_features)} low-variance features")
        return features_df, metadata

    def _stage3_vif_filtering(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Stage 3: Remove features with high VIF (multicollinearity)."""
        original_count = len(features_df.columns)
        
        # Calculate VIF scores
        vif_scores = calculate_vif_robust(features_df)
        
        # Remove features with VIF above threshold
        high_vif_features = vif_scores[vif_scores > self.vif_threshold].index.tolist()
        features_df = features_df.drop(columns=high_vif_features)
        
        # Store VIF scores in cache
        self.feature_importance_cache["vif"] = vif_scores.to_dict()
        
        metadata = {
            "removed_high_vif": len(high_vif_features),
            "vif_threshold": self.vif_threshold,
            "max_vif_remaining": vif_scores.max() if len(vif_scores) > 0 else 0,
            "features_after_stage": len(features_df.columns),
        }
        
        self.logger.info(f"Stage 3: Removed {len(high_vif_features)} high-VIF features")
        return features_df, metadata

    def _stage4_correlation_filtering(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Stage 4: Remove highly correlated features."""
        original_count = len(features_df.columns)
        
        # Calculate correlation matrix
        corr_matrix = features_df.corr().abs()
        
        # Find highly correlated feature pairs
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = []
        
        for col in upper_tri.columns:
            high_corr_features = upper_tri[col][upper_tri[col] > self.correlation_threshold].index.tolist()
            for feature in high_corr_features:
                high_corr_pairs.append((col, feature))
        
        # Remove one feature from each highly correlated pair
        features_to_remove = set()
        for feat1, feat2 in high_corr_pairs:
            # Keep the feature with higher variance
            var1 = features_df[feat1].var()
            var2 = features_df[feat2].var()
            if var1 < var2:
                features_to_remove.add(feat1)
            else:
                features_to_remove.add(feat2)
        
        features_df = features_df.drop(columns=list(features_to_remove))
        
        metadata = {
            "removed_high_correlation": len(features_to_remove),
            "correlation_threshold": self.correlation_threshold,
            "features_after_stage": len(features_df.columns),
        }
        
        self.logger.info(f"Stage 4: Removed {len(features_to_remove)} highly correlated features")
        return features_df, metadata

    def _stage5_mutual_info_ranking(self, features_df: pd.DataFrame, target: pd.Series, task: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Stage 5: Rank features by mutual information."""
        original_count = len(features_df.columns)
        
        # Calculate mutual information scores
        if task == "classification":
            mi_scores = mutual_info_classif(features_df, target, random_state=42)
        else:
            mi_scores = mutual_info_regression(features_df, target, random_state=42)
        
        mi_series = pd.Series(mi_scores, index=features_df.columns)
        
        # Store MI scores in cache
        self.feature_importance_cache["mutual_info"] = mi_series.to_dict()
        
        # Remove features with low mutual information
        low_mi_features = mi_series[mi_series < self.mi_threshold].index.tolist()
        features_df = features_df.drop(columns=low_mi_features)
        
        metadata = {
            "removed_low_mi": len(low_mi_features),
            "mi_threshold": self.mi_threshold,
            "max_mi_remaining": mi_series.max() if len(mi_series) > 0 else 0,
            "features_after_stage": len(features_df.columns),
        }
        
        self.logger.info(f"Stage 5: Removed {len(low_mi_features)} low-MI features")
        return features_df, metadata

    def _stage6_shap_importance(self, features_df: pd.DataFrame, target: pd.Series, task: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Stage 6: Calculate SHAP-based feature importance."""
        try:
            from src.analyst.meta_label_relevance import compute_shap_importance
            
            # Calculate SHAP importance
            shap_scores = compute_shap_importance(features_df, target, task=task)
            
            # Store SHAP scores in cache
            self.feature_importance_cache["shap"] = shap_scores
            
            # Remove features with low SHAP importance (bottom 20%)
            if len(shap_scores) > 0:
                shap_series = pd.Series(shap_scores)
                threshold = shap_series.quantile(0.2)  # Remove bottom 20%
                low_shap_features = shap_series[shap_series < threshold].index.tolist()
                features_df = features_df.drop(columns=low_shap_features)
                
                metadata = {
                    "removed_low_shap": len(low_shap_features),
                    "shap_threshold": threshold,
                    "max_shap_remaining": shap_series.max() if len(shap_series) > 0 else 0,
                    "features_after_stage": len(features_df.columns),
                }
            else:
                metadata = {
                    "removed_low_shap": 0,
                    "shap_threshold": 0,
                    "max_shap_remaining": 0,
                    "features_after_stage": len(features_df.columns),
                }
            
            self.logger.info(f"Stage 6: Removed {metadata['removed_low_shap']} low-SHAP features")
            
        except Exception as e:
            self.logger.warning(f"SHAP calculation failed: {e}")
            metadata = {
                "removed_low_shap": 0,
                "shap_threshold": 0,
                "max_shap_remaining": 0,
                "features_after_stage": len(features_df.columns),
                "error": str(e)
            }
        
        return features_df, metadata

    def _stage7_random_forest_importance(self, features_df: pd.DataFrame, target: pd.Series, task: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Stage 7: Calculate RandomForest-based feature importance."""
        try:
            # Train RandomForest for feature importance
            if task == "classification":
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            rf.fit(features_df, target)
            rf_importance = pd.Series(rf.feature_importances_, index=features_df.columns)
            
            # Store RF importance in cache
            self.feature_importance_cache["random_forest"] = rf_importance.to_dict()
            
            # Remove features with low RF importance (bottom 20%)
            threshold = rf_importance.quantile(0.2)  # Remove bottom 20%
            low_rf_features = rf_importance[rf_importance < threshold].index.tolist()
            features_df = features_df.drop(columns=low_rf_features)
            
            metadata = {
                "removed_low_rf": len(low_rf_features),
                "rf_threshold": threshold,
                "max_rf_remaining": rf_importance.max() if len(rf_importance) > 0 else 0,
                "features_after_stage": len(features_df.columns),
            }
            
            self.logger.info(f"Stage 7: Removed {len(low_rf_features)} low-RF features")
            
        except Exception as e:
            self.logger.warning(f"RandomForest importance calculation failed: {e}")
            metadata = {
                "removed_low_rf": 0,
                "rf_threshold": 0,
                "max_rf_remaining": 0,
                "features_after_stage": len(features_df.columns),
                "error": str(e)
            }
        
        return features_df, metadata

    def _stage8_ensemble_selection(self, features_df: pd.DataFrame, target: pd.Series, task: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Stage 8: Ensemble selection combining all methods."""
        original_count = len(features_df.columns)
        
        # Combine importance scores from all methods
        ensemble_scores = {}
        
        for feature in features_df.columns:
            score = 0.0
            weight_sum = 0.0
            
            # VIF score (inverse, lower is better)
            if "vif" in self.feature_importance_cache and feature in self.feature_importance_cache["vif"]:
                vif_score = self.feature_importance_cache["vif"][feature]
                if vif_score < np.inf:
                    normalized_vif = 1.0 / (1.0 + vif_score)  # Normalize to [0,1]
                    score += self.method_weights["vif"] * normalized_vif
                    weight_sum += self.method_weights["vif"]
            
            # Mutual Information score
            if "mutual_info" in self.feature_importance_cache and feature in self.feature_importance_cache["mutual_info"]:
                mi_score = self.feature_importance_cache["mutual_info"][feature]
                score += self.method_weights["mutual_info"] * mi_score
                weight_sum += self.method_weights["mutual_info"]
            
            # SHAP score
            if "shap" in self.feature_importance_cache and feature in self.feature_importance_cache["shap"]:
                shap_score = self.feature_importance_cache["shap"][feature]
                score += self.method_weights["shap"] * shap_score
                weight_sum += self.method_weights["shap"]
            
            # RandomForest score
            if "random_forest" in self.feature_importance_cache and feature in self.feature_importance_cache["random_forest"]:
                rf_score = self.feature_importance_cache["random_forest"][feature]
                score += self.method_weights["random_forest"] * rf_score
                weight_sum += self.method_weights["random_forest"]
            
            # Normalize by weight sum
            if weight_sum > 0:
                ensemble_scores[feature] = score / weight_sum
            else:
                ensemble_scores[feature] = 0.0
        
        # Store ensemble scores in cache
        self.feature_importance_cache["ensemble"] = ensemble_scores
        
        # Select top features based on ensemble scores
        ensemble_series = pd.Series(ensemble_scores)
        ensemble_series = ensemble_series.sort_values(ascending=False)
        
        # Keep top features (up to target_features * 1.5 to give RFE some choice)
        max_features = min(int(self.target_features * 1.5), len(ensemble_series))
        selected_features = ensemble_series.head(max_features).index.tolist()
        
        features_df = features_df[selected_features]
        
        metadata = {
            "ensemble_features_selected": len(selected_features),
            "max_ensemble_score": ensemble_series.max() if len(ensemble_series) > 0 else 0,
            "features_after_stage": len(features_df.columns),
        }
        
        self.logger.info(f"Stage 8: Selected {len(selected_features)} features via ensemble scoring")
        return features_df, metadata

    def _stage9_final_rfe_selection(self, features_df: pd.DataFrame, target: pd.Series, task: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Stage 9: Final selection using Recursive Feature Elimination."""
        if len(features_df.columns) <= self.target_features:
            # Already at or below target, return as is
            return features_df, {"final_selection": "no_change", "features_after_stage": len(features_df.columns)}
        
        try:
            # Use LightGBM for RFE
            if task == "classification":
                estimator = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
            else:
                estimator = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            
            rfe = RFE(estimator=estimator, n_features_to_select=self.target_features, step=1)
            rfe.fit(features_df, target)
            
            # Get selected features
            selected_features = features_df.columns[rfe.support_].tolist()
            features_df = features_df[selected_features]
            
            metadata = {
                "final_selection": "rfe_lightgbm",
                "rfe_ranking": rfe.ranking_.tolist(),
                "features_after_stage": len(features_df.columns),
            }
            
            self.logger.info(f"Stage 9: Final selection using RFE-LightGBM")
            
        except Exception as e:
            self.logger.warning(f"RFE failed: {e}, using top features by ensemble score")
            
            # Fallback: use top features by ensemble score
            ensemble_series = pd.Series(self.feature_importance_cache.get("ensemble", {}))
            if len(ensemble_series) > 0:
                selected_features = ensemble_series.sort_values(ascending=False).head(self.target_features).index.tolist()
                features_df = features_df[selected_features]
            
            metadata = {
                "final_selection": "ensemble_fallback",
                "features_after_stage": len(features_df.columns),
                "error": str(e)
            }
        
        return features_df, metadata

    def _save_selection_metadata(self, metadata: Dict[str, Any], symbol: str, exchange: str, data_dir: str) -> None:
        """Save feature selection metadata to disk."""
        try:
            import os
            os.makedirs(data_dir, exist_ok=True)
            
            filename = f"{data_dir}/feature_selection_metadata_{symbol}_{exchange}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"✅ Feature selection metadata saved to {filename}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save metadata: {e}")

    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """Get a summary of feature importance scores."""
        summary = {}
        
        for method, scores in self.feature_importance_cache.items():
            if isinstance(scores, dict) and len(scores) > 0:
                scores_series = pd.Series(scores)
                summary[method] = {
                    "mean": float(scores_series.mean()),
                    "std": float(scores_series.std()),
                    "min": float(scores_series.min()),
                    "max": float(scores_series.max()),
                    "top_10_features": scores_series.nlargest(10).to_dict()
                }
        
        return summary