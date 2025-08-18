# src/training/optimized_feature_selection_manager.py

import asyncio
import numpy as np
import pandas as pd
import json
import os
import time
from typing import Dict, List, Any, Tuple, Optional, Set
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
    f_classif,
    f_regression,
    SelectKBest,
    SelectFromModel,
    RFE,
    VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
import lightgbm as lgb
import xgboost as xgb
import shap

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors


class OptimizedFeatureSelectionManager:
    """
    Optimized Feature Selection Manager for ML Training Steps.
    
    Key improvements:
    1. Matrix-based VIF calculation (O(n²) instead of O(n³))
    2. RF+SHAP for feature importance assessment
    3. Balanced feature mix (50-100 features) across categories
    4. Model-specific optimization for different architectures
    5. Computational efficiency with vectorized operations
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("OptimizedFeatureSelection")
        
        # Load configuration
        self._load_config()
        
        # Feature importance cache
        self.feature_importance_cache = {}
        self.shap_values_cache = {}
        self.selection_metadata = {}
        
        # Performance tracking
        self.performance_metrics = {
            "vif_calculation_time": 0.0,
            "shap_calculation_time": 0.0,
            "correlation_analysis_time": 0.0,
            "total_selection_time": 0.0,
            "vectorized_operations_time": 0.0,
            "matrix_operations_time": 0.0
        }
        
    def _load_config(self):
        """Load and validate configuration."""
        # Default configuration
        default_config = {
            "target_features": {
                "neural_networks": 80,
                "linear_models": 60,
                "ensemble_models": 90,
                "step2_general": 100
            },
            "vif_threshold": 10.0,
            "correlation_threshold": 0.95,
            "mutual_info_threshold": 0.001,
            "variance_threshold": 0.01,
            "shap_threshold": 0.001,
            "max_removal_fraction": 0.5,
            "enable_shap_analysis": True,
            "enable_matrix_vif": True,
            "enable_balanced_selection": True,
            "feature_categories": {
                "momentum": 0.25,
                "volatility": 0.10,
                "liquidity": 0.10,
                "volume": 0.15,
                "microstructure": 0.10,
                "regime": 0.10,
                "sr_features": 0.10,
                "interaction": 0.10
            }
        }
        
        # Override with config if provided
        fs_config = self.config.get("feature_selection", {})
        for key, value in fs_config.items():
            if key in default_config:
                if isinstance(default_config[key], dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
        
        self.config = default_config
        
    @handle_errors(
        exceptions=(Exception,),
        default_return=(pd.DataFrame(), {}),
        context="optimized feature selection"
    )
    def select_features_optimized(
        self, 
        features_df: pd.DataFrame, 
        target: pd.Series,
        model_type: str = "general",
        step_name: str = "step2",
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Optimized feature selection with computational efficiency and balanced feature mix.
        
        Args:
            features_df: Input features DataFrame
            target: Target variable series
            model_type: Model type (neural_networks, linear_models, ensemble_models)
            step_name: Training step name
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (selected_features_df, selection_metadata)
        """
        start_time = time.time()
        self.logger.info(f"🚀 Starting optimized feature selection for {model_type} in {step_name}")
        
        try:
            # Get target feature count
            target_features = self._get_target_feature_count(model_type, step_name)
            self.logger.info(f"📊 Target features: {target_features} (from {len(features_df.columns)} original)")
            
            # Stage 1: Data quality filtering (fast)
            features_df, stage1_metadata = self._stage1_data_quality_filtering(features_df)
            
            # Stage 2: Matrix-based VIF calculation (optimized)
            features_df, stage2_metadata = self._stage2_matrix_vif_filtering(features_df)
            
            # Stage 3: Efficient correlation analysis
            features_df, stage3_metadata = self._stage3_efficient_correlation_filtering(features_df)
            
            # Stage 4: RF+SHAP importance assessment
            features_df, stage4_metadata = self._stage4_rf_shap_importance(features_df, target)
            
            # Stage 5: Balanced feature selection
            features_df, stage5_metadata = self._stage5_balanced_selection(features_df, target, target_features, model_type)
            
            # Stage 6: Model-specific optimization
            features_df, stage6_metadata = self._stage6_model_specific_optimization(features_df, target, model_type)
            
            # Compile metadata
            total_time = time.time() - start_time
            selection_metadata = {
                "original_features": len(features_df.columns),
                "final_features": len(features_df.columns),
                "target_features": target_features,
                "model_type": model_type,
                "step_name": step_name,
                "total_time": total_time,
                "performance_metrics": self.performance_metrics,
                "stages": {
                    "stage1_data_quality": stage1_metadata,
                    "stage2_matrix_vif": stage2_metadata,
                    "stage3_correlation": stage3_metadata,
                    "stage4_rf_shap": stage4_metadata,
                    "stage5_balanced": stage5_metadata,
                    "stage6_model_specific": stage6_metadata
                },
                "feature_categories": self._categorize_features(features_df.columns),
                "selection_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Optimized feature selection completed: {len(features_df.columns)} features in {total_time:.2f}s")
            return features_df, selection_metadata
            
        except Exception as e:
            self.logger.error(f"❌ Optimized feature selection failed: {e}")
            raise
    
    def _get_target_feature_count(self, model_type: str, step_name: str) -> int:
        """Get target feature count based on model type and step."""
        target_config = self.config["target_features"]
        
        if step_name == "step2":
            return target_config.get("step2_general", 100)
        elif model_type == "neural_networks":
            return target_config.get("neural_networks", 80)
        elif model_type == "linear_models":
            return target_config.get("linear_models", 60)
        elif model_type == "ensemble_models":
            return target_config.get("ensemble_models", 90)
        else:
            return target_config.get("step2_general", 100)
    
    def _stage1_data_quality_filtering(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Stage 1: Fast data quality filtering."""
        original_count = len(features_df.columns)
        
        # Remove features with too many NaN values (>10%)
        nan_ratio = features_df.isna().sum() / len(features_df)
        high_nan_features = nan_ratio[nan_ratio > 0.1].index.tolist()
        features_df = features_df.drop(columns=high_nan_features)
        
        # Remove features with infinite values
        inf_mask = np.isinf(features_df).any()
        inf_features = inf_mask[inf_mask].index.tolist()
        features_df = features_df.drop(columns=inf_features)
        
        # Remove zero variance features
        zero_var_features = features_df.columns[features_df.var() == 0].tolist()
        features_df = features_df.drop(columns=zero_var_features)
        
        # Fill remaining NaN values efficiently
        features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        metadata = {
            "removed_high_nan": len(high_nan_features),
            "removed_infinite": len(inf_features),
            "removed_zero_variance": len(zero_var_features),
            "features_after_stage": len(features_df.columns)
        }
        
        self.logger.info(f"Stage 1: Removed {original_count - len(features_df.columns)} low-quality features")
        return features_df, metadata
    
    def _stage2_matrix_vif_filtering(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Stage 2: Matrix-based VIF calculation (O(n²) instead of O(n³))."""
        if not self.config["enable_matrix_vif"]:
            return features_df, {"skipped": True, "reason": "matrix_vif_disabled"}
        
        start_time = time.time()
        original_count = len(features_df.columns)
        vif_threshold = self.config["vif_threshold"]
        
        try:
            # Use matrix-based VIF calculation
            vif_scores = self._calculate_matrix_vif(features_df)
            
            # Remove high VIF features
            high_vif_features = vif_scores[vif_scores > vif_threshold].index.tolist()
            features_df = features_df.drop(columns=high_vif_features)
            
            vif_time = time.time() - start_time
            self.performance_metrics["vif_calculation_time"] = vif_time
            
            metadata = {
                "removed_high_vif": len(high_vif_features),
                "vif_threshold": vif_threshold,
                "max_vif": float(vif_scores.max()) if not vif_scores.empty else 0.0,
                "calculation_time": vif_time,
                "features_after_stage": len(features_df.columns)
            }
            
            self.logger.info(f"Stage 2: Matrix VIF removed {len(high_vif_features)} features in {vif_time:.2f}s")
            return features_df, metadata
            
        except Exception as e:
            self.logger.warning(f"Stage 2: Matrix VIF failed, skipping: {e}")
            return features_df, {"error": str(e), "features_after_stage": len(features_df.columns)}
    
    def _calculate_matrix_vif(self, features_df: pd.DataFrame) -> pd.Series:
        """Calculate VIF using matrix operations (much faster than iterative approach)."""
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_df)
        
        try:
            # Use Ledoit-Wolf shrinkage for robust covariance estimation
            lw = LedoitWolf().fit(X_scaled)
            cov_matrix = lw.covariance_
            
            # Calculate correlation matrix
            std_vec = np.sqrt(np.diag(cov_matrix))
            std_vec[std_vec == 0.0] = 1.0
            corr_matrix = cov_matrix / np.outer(std_vec, std_vec)
            
            # Calculate VIF using matrix inverse
            try:
                corr_inv = np.linalg.pinv(corr_matrix)
                vif_scores = np.diag(corr_inv)
            except np.linalg.LinAlgError:
                # Fallback to iterative calculation for problematic matrices
                vif_scores = self._calculate_iterative_vif(features_df)
            
            return pd.Series(vif_scores, index=features_df.columns)
            
        except Exception:
            # Fallback to correlation-based approach
            corr_matrix = features_df.corr().values
            try:
                corr_inv = np.linalg.pinv(corr_matrix)
                vif_scores = np.diag(corr_inv)
            except np.linalg.LinAlgError:
                vif_scores = np.ones(len(features_df.columns))
            
            return pd.Series(vif_scores, index=features_df.columns)
    
    def _calculate_iterative_vif(self, features_df: pd.DataFrame) -> np.ndarray:
        """Fallback iterative VIF calculation for problematic matrices."""
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        vif_scores = []
        for i, col in enumerate(features_df.columns):
            try:
                vif = variance_inflation_factor(features_df.values, i)
                vif_scores.append(vif)
            except:
                vif_scores.append(1.0)
        
        return np.array(vif_scores)
    
    def _stage3_efficient_correlation_filtering(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Stage 3: Efficient correlation analysis using matrix operations."""
        start_time = time.time()
        original_count = len(features_df.columns)
        corr_threshold = self.config["correlation_threshold"]
        
        # Calculate correlation matrix efficiently
        corr_matrix = features_df.corr().abs()
        
        # Find highly correlated pairs using vectorized operations
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = np.where(upper_tri > corr_threshold)
        
        # Remove one feature from each highly correlated pair
        features_to_remove = set()
        for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
            feat1 = features_df.columns[i]
            feat2 = features_df.columns[j]
            
            # Keep the feature with higher variance
            var1 = features_df[feat1].var()
            var2 = features_df[feat2].var()
            if var1 < var2:
                features_to_remove.add(feat1)
            else:
                features_to_remove.add(feat2)
        
        features_df = features_df.drop(columns=list(features_to_remove))
        
        corr_time = time.time() - start_time
        self.performance_metrics["correlation_analysis_time"] = corr_time
        
        metadata = {
            "removed_high_correlation": len(features_to_remove),
            "correlation_threshold": corr_threshold,
            "high_corr_pairs": len(high_corr_pairs[0]),
            "calculation_time": corr_time,
            "features_after_stage": len(features_df.columns)
        }
        
        self.logger.info(f"Stage 3: Correlation filtering removed {len(features_to_remove)} features in {corr_time:.2f}s")
        return features_df, metadata
    
    def _stage4_rf_shap_importance(self, features_df: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Stage 4: RF+SHAP feature importance assessment."""
        if not self.config["enable_shap_analysis"]:
            return features_df, {"skipped": True, "reason": "shap_analysis_disabled"}
        
        start_time = time.time()
        
        try:
            # Train Random Forest for feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(features_df, target)
            
            # Calculate SHAP values (sample-based for efficiency)
            sample_size = min(1000, len(features_df))
            sample_indices = np.random.choice(len(features_df), sample_size, replace=False)
            X_sample = features_df.iloc[sample_indices]
            
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_sample)
            
            # Calculate mean absolute SHAP values
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values)
            mean_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Combine RF importance and SHAP importance
            rf_importance = pd.Series(rf.feature_importances_, index=features_df.columns)
            shap_importance = pd.Series(mean_shap, index=features_df.columns)
            
            # Normalize and combine
            rf_importance_norm = rf_importance / rf_importance.sum()
            shap_importance_norm = shap_importance / shap_importance.sum()
            combined_importance = (rf_importance_norm + shap_importance_norm) / 2
            
            # Store for later use
            self.feature_importance_cache['rf_shap'] = combined_importance
            
            shap_time = time.time() - start_time
            self.performance_metrics["shap_calculation_time"] = shap_time
            
            metadata = {
                "rf_importance_top_10": rf_importance.head(10).index.tolist(),
                "shap_importance_top_10": shap_importance.head(10).index.tolist(),
                "combined_importance_top_10": combined_importance.head(10).index.tolist(),
                "calculation_time": shap_time,
                "features_after_stage": len(features_df.columns)
            }
            
            self.logger.info(f"Stage 4: RF+SHAP importance calculated in {shap_time:.2f}s")
            return features_df, metadata
            
        except Exception as e:
            self.logger.warning(f"Stage 4: RF+SHAP failed, using RF only: {e}")
            # Fallback to RF importance only
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(features_df, target)
            self.feature_importance_cache['rf_shap'] = pd.Series(rf.feature_importances_, index=features_df.columns)
            
            return features_df, {"fallback": "rf_only", "features_after_stage": len(features_df.columns)}
    
    def _stage5_balanced_selection(self, features_df: pd.DataFrame, target: pd.Series, target_features: int, model_type: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Stage 5: Balanced feature selection across categories."""
        if not self.config["enable_balanced_selection"]:
            return features_df, {"skipped": True, "reason": "balanced_selection_disabled"}
        
        # Categorize features
        feature_categories = self._categorize_features(features_df.columns)
        category_weights = self.config["feature_categories"]
        
        # Calculate target features per category
        selected_features = []
        for category, weight in category_weights.items():
            if category in feature_categories:
                category_features = feature_categories[category]
                target_per_category = int(target_features * weight)
                
                if category_features:
                    # Rank features within category by importance
                    if 'rf_shap' in self.feature_importance_cache:
                        importance_scores = self.feature_importance_cache['rf_shap'][category_features]
                    else:
                        # Fallback to mutual information
                        mi_scores = mutual_info_classif(features_df[category_features], target, random_state=42)
                        importance_scores = pd.Series(mi_scores, index=category_features)
                    
                    # Select top features from category
                    top_features = importance_scores.nlargest(min(target_per_category, len(category_features))).index.tolist()
                    selected_features.extend(top_features)
        
        # If we don't have enough features, add from other categories
        if len(selected_features) < target_features:
            remaining_features = [f for f in features_df.columns if f not in selected_features]
            if remaining_features:
                if 'rf_shap' in self.feature_importance_cache:
                    importance_scores = self.feature_importance_cache['rf_shap'][remaining_features]
                else:
                    mi_scores = mutual_info_classif(features_df[remaining_features], target, random_state=42)
                    importance_scores = pd.Series(mi_scores, index=remaining_features)
                
                additional_features = importance_scores.nlargest(target_features - len(selected_features)).index.tolist()
                selected_features.extend(additional_features)
        
        # Ensure we don't exceed target
        selected_features = selected_features[:target_features]
        features_df = features_df[selected_features]
        
        metadata = {
            "selected_features": len(selected_features),
            "target_features": target_features,
            "category_distribution": {cat: len([f for f in selected_features if f in features]) 
                                    for cat, features in feature_categories.items()},
            "features_after_stage": len(features_df.columns)
        }
        
        self.logger.info(f"Stage 5: Balanced selection: {len(selected_features)} features across categories")
        return features_df, metadata
    
    def _stage6_model_specific_optimization(self, features_df: pd.DataFrame, target: pd.Series, model_type: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Stage 6: Model-specific optimization."""
        if model_type == "neural_networks":
            return self._optimize_for_neural_networks(features_df, target)
        elif model_type == "linear_models":
            return self._optimize_for_linear_models(features_df, target)
        elif model_type == "ensemble_models":
            return self._optimize_for_ensemble_models(features_df, target)
        else:
            return features_df, {"optimization": "none", "features_after_stage": len(features_df.columns)}
    
    def _optimize_for_neural_networks(self, features_df: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Optimize features for neural networks."""
        # Neural networks benefit from diverse, non-linear features
        # Keep interaction features, normalized features, and SR features
        interaction_features = [col for col in features_df.columns if "_x_" in col or "_div_" in col]
        normalized_features = [col for col in features_df.columns if "_norm" in col or "_z_score" in col]
        sr_features = [col for col in features_df.columns if any(keyword in col.lower() for keyword in [
            "sr_", "support", "resistance", "breakout", "proximity", "sr_distance"
        ])]
        
        preferred_features = list(set(interaction_features + normalized_features + sr_features))
        remaining_features = [col for col in features_df.columns if col not in preferred_features]
        
        # Add remaining features based on importance
        if 'rf_shap' in self.feature_importance_cache:
            importance_scores = self.feature_importance_cache['rf_shap'][remaining_features]
            additional_features = importance_scores.nlargest(len(features_df.columns) - len(preferred_features)).index.tolist()
        else:
            additional_features = remaining_features[:len(features_df.columns) - len(preferred_features)]
        
        final_features = preferred_features + additional_features
        features_df = features_df[final_features]
        
        metadata = {
            "optimization": "neural_networks",
            "interaction_features": len(interaction_features),
            "normalized_features": len(normalized_features),
            "sr_features": len(sr_features),
            "features_after_stage": len(features_df.columns)
        }
        
        return features_df, metadata
    
    def _optimize_for_linear_models(self, features_df: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Optimize features for linear models."""
        # Linear models benefit from uncorrelated, interpretable features
        # Remove interaction features
        linear_features = [col for col in features_df.columns if "_x_" not in col and "_div_" not in col]
        
        # Use Lasso for feature selection
        lasso = Lasso(alpha=0.01, random_state=42)
        lasso.fit(features_df[linear_features], target)
        
        selected_features = features_df.columns[lasso.coef_ != 0].tolist()
        if len(selected_features) > len(features_df.columns) * 0.8:  # If too many selected
            # Use top features by coefficient magnitude
            coef_ranking = pd.Series(lasso.coef_, index=linear_features).abs().sort_values(ascending=False)
            selected_features = coef_ranking.head(len(features_df.columns)).index.tolist()
        
        features_df = features_df[selected_features]
        
        metadata = {
            "optimization": "linear_models",
            "lasso_selected": len(selected_features),
            "features_after_stage": len(features_df.columns)
        }
        
        return features_df, metadata
    
    def _optimize_for_ensemble_models(self, features_df: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Optimize features for ensemble models."""
        # Ensemble models benefit from diverse feature set
        # Use multiple feature selection methods
        methods = [
            ("random_forest", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("lightgbm", lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)),
            ("mutual_info", None)
        ]
        
        feature_scores = {}
        for method_name, estimator in methods:
            if method_name == "mutual_info":
                scores = mutual_info_classif(features_df, target, random_state=42)
            else:
                estimator.fit(features_df, target)
                scores = estimator.feature_importances_
            
            feature_scores[method_name] = pd.Series(scores, index=features_df.columns)
        
        # Combine scores from different methods
        combined_scores = pd.DataFrame(feature_scores).mean(axis=1).sort_values(ascending=False)
        selected_features = combined_scores.head(len(features_df.columns)).index.tolist()
        
        features_df = features_df[selected_features]
        
        metadata = {
            "optimization": "ensemble_models",
            "methods_used": list(feature_scores.keys()),
            "features_after_stage": len(features_df.columns)
        }
        
        return features_df, metadata
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Categorize features by type."""
        categories = {
            "momentum": [],
            "volatility": [],
            "liquidity": [],
            "volume": [],
            "microstructure": [],
            "regime": [],
            "sr_features": [],
            "interaction": [],
            "other": []
        }
        
        for feature in feature_names:
            feature_lower = feature.lower()
            categorized = False
            
            # Interaction features (check first to avoid conflicts)
            if any(keyword in feature_lower for keyword in [
                "_x_", "_div_", "_ratio_", "_over_", "_cross_", "interaction",
                "momentum_x_", "volatility_x_", "volume_x_", "regime_x_",
                "momentum_div_", "volatility_div_", "volume_div_"
            ]):
                categories["interaction"].append(feature)
                categorized = True
            
            # Momentum indicators (including multi-timeframe and derivative forms)
            if not categorized:
                momentum_base_tokens = [
                    "momentum", "mom", "rsi", "macd", "cci", "roc", "willr", "stoch",
                    "adx", "dmi", "kama", "tema", "dema", "hma", "wma", "vwma", "zlema",
                    "ichimoku", "psar", "trix", "cmo", "tsi", "ppo", "pmo", "uo",
                    "linreg", "lin_reg", "sma", "ema", "ma_", "moving_avg", "trend",
                    "bb_position", "bb_upper", "bb_lower", "bb_width", "bb_percent"
                ]
                derivative_tokens = [
                    "_diff", "diff_", "_delta", "delta_", "_accel", "accel_",
                    "acceleration", "_slope", "slope_", "_change", "change_", "_norm", "norm_"
                ]
                has_momentum_base = any(token in feature_lower for token in momentum_base_tokens)
                has_derivative_with_anchor = (
                    any(token in feature_lower for token in derivative_tokens)
                    and any(anchor in feature_lower for anchor in [
                        "momentum", "roc", "rsi", "macd", "stoch", "cci", "willr", "trend", "bb"
                    ])
                )
                if has_momentum_base or has_derivative_with_anchor:
                    categories["momentum"].append(feature)
                    categorized = True
            
            # Volatility measures (including multi-timeframe and derivative forms)
            if not categorized:
                volatility_base_tokens = [
                    "volatility", "atr", "true_range", "truerange", "natr", "parkinson",
                    "garman", "gk_vol", "garman_klass", "roll", "rvol", "realized_vol",
                    "hv", "hist_vol", "historical_vol", "variance", "std", "bbands",
                    "boll", "bollinger", "donch", "donchian", "keltner", "chop",
                    "choppiness", "park_vol", "vol_", "volatility_"
                ]
                derivative_tokens = [
                    "_diff", "diff_", "_delta", "delta_", "_accel", "accel_",
                    "acceleration", "_slope", "slope_", "_change", "change_", "_norm", "norm_"
                ]
                has_volatility_base = any(token in feature_lower for token in volatility_base_tokens)
                has_derivative_with_anchor = (
                    any(token in feature_lower for token in derivative_tokens)
                    and any(anchor in feature_lower for anchor in [
                        "volatility", "atr", "true_range", "variance", "std", "bbands", "bollinger"
                    ])
                )
                if has_volatility_base or has_derivative_with_anchor:
                    categories["volatility"].append(feature)
                    categorized = True
            
            # Volume features (including multi-timeframe and derivative forms)
            if not categorized:
                volume_base_tokens = [
                    "volume", "tick_volume", "obv", "cmf", "mfi", "vwap",
                    "pvi", "nvi", "efi", "delta_volume", "volume_ratio", "volume_ma", 
                    "volume_change", "volume_sma", "volume_momentum", "volume_weighted",
                    "volume_velocity", "volume_acceleration", "volume_price", "volume_"
                ]
                derivative_tokens = [
                    "_diff", "diff_", "_delta", "delta_", "_accel", "accel_",
                    "acceleration", "_slope", "slope_", "_change", "change_", "_norm", "norm_"
                ]
                has_volume_base = any(token in feature_lower for token in volume_base_tokens)
                has_derivative_with_anchor = (
                    any(token in feature_lower for token in derivative_tokens)
                    and any(anchor in feature_lower for anchor in [
                        "volume", "obv", "cmf", "mfi", "vwap", "volume_ratio", "volume_ma"
                    ])
                )
                if has_volume_base or has_derivative_with_anchor:
                    categories["volume"].append(feature)
                    categorized = True
            
            # Liquidity features (including multi-timeframe and derivative forms)
            if not categorized:
                liquidity_base_tokens = [
                    "liquidity", "spread", "bid_ask", "bidask", "quote_imbalance",
                    "liquidity_", "spread_", "bid_", "ask_", "quote_"
                ]
                derivative_tokens = [
                    "_diff", "diff_", "_delta", "delta_", "_accel", "accel_",
                    "acceleration", "_slope", "slope_", "_change", "change_", "_norm", "norm_"
                ]
                has_liquidity_base = any(token in feature_lower for token in liquidity_base_tokens)
                has_derivative_with_anchor = (
                    any(token in feature_lower for token in derivative_tokens)
                    and any(anchor in feature_lower for anchor in [
                        "liquidity", "spread", "bid_ask", "quote_imbalance"
                    ])
                )
                if has_liquidity_base or has_derivative_with_anchor:
                    categories["liquidity"].append(feature)
                    categorized = True
            
            # Microstructure features (including multi-timeframe and derivative forms)
            if not categorized:
                microstructure_base_tokens = [
                    "microstructure", "order_flow", "orderflow", "ofi", "imbalance",
                    "quote_imbalance", "depth", "orderbook", "book", "microprice", 
                    "trade_count", "trade_frequency", "order_", "flow_", "imbalance_"
                ]
                derivative_tokens = [
                    "_diff", "diff_", "_delta", "delta_", "_accel", "accel_",
                    "acceleration", "_slope", "slope_", "_change", "change_", "_norm", "norm_"
                ]
                has_microstructure_base = any(token in feature_lower for token in microstructure_base_tokens)
                has_derivative_with_anchor = (
                    any(token in feature_lower for token in derivative_tokens)
                    and any(anchor in feature_lower for anchor in [
                        "order_flow", "imbalance", "microstructure", "trade_count"
                    ])
                )
                if has_microstructure_base or has_derivative_with_anchor:
                    categories["microstructure"].append(feature)
                    categorized = True
            
            # Regime features (including multi-timeframe and derivative forms)
            if not categorized:
                regime_base_tokens = [
                    "regime", "cluster", "state", "composite", "hmm", "regime_",
                    "cluster_", "state_", "hmm_", "composite_"
                ]
                derivative_tokens = [
                    "_diff", "diff_", "_delta", "delta_", "_accel", "accel_",
                    "acceleration", "_slope", "slope_", "_change", "change_", "_norm", "norm_"
                ]
                has_regime_base = any(token in feature_lower for token in regime_base_tokens)
                has_derivative_with_anchor = (
                    any(token in feature_lower for token in derivative_tokens)
                    and any(anchor in feature_lower for anchor in [
                        "regime", "cluster", "state", "hmm", "composite"
                    ])
                )
                if has_regime_base or has_derivative_with_anchor:
                    categories["regime"].append(feature)
                    categorized = True
            
            # Support/Resistance features (including multi-timeframe and derivative forms)
            if not categorized:
                sr_base_tokens = [
                    "sr_distance", "support_level", "resistance_level", "proximity",
                    "multi_timeframe_sr_score", "sr_proximity", "sr_outcome",
                    "normalized_distance", "sr_proximity_score", "strength_score", 
                    "clarity_factor", "directional_pressure", "sr_score", "delta_sr_score", 
                    "isolation_score", "sr_level", "sr_breakout", "sr_rebounce", 
                    "sr_consolidation", "sr_breakout_prob", "sr_rebounce_prob",
                    "sr_consolidation_prob", "sr_multi_timeframe", "sr_", "support_", "resistance_"
                ]
                derivative_tokens = [
                    "_diff", "diff_", "_delta", "delta_", "_accel", "accel_",
                    "acceleration", "_slope", "slope_", "_change", "change_", "_norm", "norm_"
                ]
                has_sr_base = any(token in feature_lower for token in sr_base_tokens)
                has_derivative_with_anchor = (
                    any(token in feature_lower for token in derivative_tokens)
                    and any(anchor in feature_lower for anchor in [
                        "sr_", "support", "resistance", "proximity", "distance"
                    ])
                )
                if has_sr_base or has_derivative_with_anchor:
                    categories["sr_features"].append(feature)
                    categorized = True
            

            
            if not categorized:
                categories["other"].append(feature)
        
        return categories
    
    def save_selection_metadata(self, metadata: Dict[str, Any], symbol: str, exchange: str, data_dir: str):
        """Save feature selection metadata."""
        try:
            metadata_file = f"{data_dir}/{exchange}_{symbol}_optimized_feature_selection_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"💾 Optimized feature selection metadata saved: {metadata_file}")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save feature selection metadata: {e}")
    
    def apply_vectorized_operations(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply vectorized operations for efficient feature processing.
        
        Args:
            features_df: Input features DataFrame
            
        Returns:
            Processed features DataFrame with vectorized operations
        """
        try:
            start_time = time.time()
            self.logger.info("🔄 Applying vectorized operations for feature processing...")
            
            # Create a copy to avoid modifying original
            processed_df = features_df.copy()
            
            # Vectorized operations for feature engineering
            # 1. Rolling statistics (vectorized)
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                # Rolling mean and std (vectorized)
                processed_df[f"{col}_rolling_mean_5"] = processed_df[col].rolling(window=5, min_periods=1).mean()
                processed_df[f"{col}_rolling_std_5"] = processed_df[col].rolling(window=5, min_periods=1).std()
                
                # Rolling mean and std (vectorized)
                processed_df[f"{col}_rolling_mean_10"] = processed_df[col].rolling(window=10, min_periods=1).mean()
                processed_df[f"{col}_rolling_std_10"] = processed_df[col].rolling(window=10, min_periods=1).std()
            
            # 2. Lag features (vectorized)
            for col in numeric_cols:
                processed_df[f"{col}_lag_1"] = processed_df[col].shift(1)
                processed_df[f"{col}_lag_5"] = processed_df[col].shift(5)
            
            # 3. Difference features (vectorized)
            for col in numeric_cols:
                processed_df[f"{col}_diff_1"] = processed_df[col].diff(1)
                processed_df[f"{col}_diff_5"] = processed_df[col].diff(5)
            
            # 4. Z-score normalization (vectorized)
            for col in numeric_cols:
                mean_val = processed_df[col].mean()
                std_val = processed_df[col].std()
                if std_val > 0:
                    processed_df[f"{col}_zscore"] = (processed_df[col] - mean_val) / std_val
            
            # 5. Percentile ranks (vectorized)
            for col in numeric_cols:
                processed_df[f"{col}_percentile_rank"] = processed_df[col].rank(pct=True)
            
            # 6. Interaction features (vectorized)
            if len(numeric_cols) >= 2:
                # Create interaction features between top correlated features
                corr_matrix = processed_df[numeric_cols].corr().abs()
                high_corr_pairs = []
                
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        if corr_matrix.iloc[i, j] > 0.7:  # High correlation threshold
                            high_corr_pairs.append((numeric_cols[i], numeric_cols[j]))
                
                # Create interaction features for highly correlated pairs
                for col1, col2 in high_corr_pairs[:10]:  # Limit to top 10 interactions
                    processed_df[f"{col1}_x_{col2}"] = processed_df[col1] * processed_df[col2]
                    processed_df[f"{col1}_div_{col2}"] = processed_df[col1] / (processed_df[col2] + 1e-8)
            
            # Fill NaN values with forward fill then backward fill
            processed_df = processed_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            vectorized_time = time.time() - start_time
            self.performance_metrics["vectorized_operations_time"] = vectorized_time
            
            self.logger.info(f"✅ Vectorized operations completed in {vectorized_time:.2f}s")
            self.logger.info(f"📊 Features: {len(features_df.columns)} -> {len(processed_df.columns)}")
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"❌ Vectorized operations failed: {e}")
            return features_df
    
    def apply_matrix_operations(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply efficient matrix operations for feature processing.
        
        Args:
            features_df: Input features DataFrame
            
        Returns:
            Processed features DataFrame with matrix operations
        """
        try:
            start_time = time.time()
            self.logger.info("🔄 Applying matrix operations for feature processing...")
            
            # Convert to numpy array for matrix operations
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return features_df
            
            # Extract numeric data
            X = features_df[numeric_cols].values
            
            # 1. Matrix-based correlation analysis
            corr_matrix = np.corrcoef(X.T)
            
            # 2. Matrix-based covariance analysis
            cov_matrix = np.cov(X.T)
            
            # 3. Matrix-based PCA for dimensionality reduction
            if X.shape[1] > 50:  # Only apply PCA if we have many features
                # Standardize the data
                X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
                
                # Compute covariance matrix
                cov_matrix = np.cov(X_std.T)
                
                # Compute eigenvalues and eigenvectors
                eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                
                # Sort eigenvalues and eigenvectors
                idx = eigenvals.argsort()[::-1]
                eigenvals = eigenvals[idx]
                eigenvecs = eigenvecs[:, idx]
                
                # Select top components explaining 95% of variance
                explained_var_ratio = eigenvals / np.sum(eigenvals)
                cumulative_var_ratio = np.cumsum(explained_var_ratio)
                n_components = np.argmax(cumulative_var_ratio >= 0.95) + 1
                
                # Project data onto principal components
                X_pca = X_std @ eigenvecs[:, :n_components]
                
                # Create new feature names
                pca_feature_names = [f"pca_component_{i+1}" for i in range(n_components)]
                
                # Create DataFrame with PCA features
                pca_df = pd.DataFrame(X_pca, columns=pca_feature_names, index=features_df.index)
                
                # Combine with original features
                result_df = pd.concat([features_df, pca_df], axis=1)
                
                self.logger.info(f"📊 PCA reduced features from {X.shape[1]} to {n_components} components")
            else:
                result_df = features_df
            
            # 4. Matrix-based feature scaling
            if len(numeric_cols) > 0:
                # Min-max scaling
                X_min = np.min(X, axis=0)
                X_max = np.max(X, axis=0)
                X_scaled = (X - X_min) / (X_max - X_min + 1e-8)
                
                # Create scaled features
                scaled_cols = [f"{col}_scaled" for col in numeric_cols]
                scaled_df = pd.DataFrame(X_scaled, columns=scaled_cols, index=features_df.index)
                
                # Combine with result
                result_df = pd.concat([result_df, scaled_df], axis=1)
            
            matrix_time = time.time() - start_time
            self.performance_metrics["matrix_operations_time"] = matrix_time
            
            self.logger.info(f"✅ Matrix operations completed in {matrix_time:.2f}s")
            self.logger.info(f"📊 Features: {len(features_df.columns)} -> {len(result_df.columns)}")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"❌ Matrix operations failed: {e}")
            return features_df