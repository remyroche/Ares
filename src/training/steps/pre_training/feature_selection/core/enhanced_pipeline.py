"""
Enhanced Multi-Stage Feature Selection Pipeline

This module implements the new multi-stage feature selection pipeline with:
1. Stage 1: mRMR + Spearman combination to skim top 50% above target
2. Stage 2: Progressive refinement using LGBM-SHAP and LASSO ensemble with RFE, CV, bootstrap stability

The pipeline maintains VectorBT optimizations and other performance enhancements.
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging

# Import required libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
import warnings

# Import VectorBT components
from src.feature_selection.vectorbt.vectorbt_mrmr_selector import VectorBTMRMRSelector
from src.feature_selection.vectorbt.vectorbt_config import VectorBTFeatureSelectionConfig

# Import utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
)
from src.utils.logger import get_logger

# Import configuration
from .config import FeatureSelectionConfig, FeatureSelectionResult

logger = get_logger("EnhancedPipeline")


class EnhancedMultiStageFeatureSelector:
    """
    Enhanced multi-stage feature selection with mRMR+Spearman and progressive refinement.
    
    This class implements:
    1. Stage 1: 70% mRMR + 30% Spearman to select top 50% above target
    2. Stage 2: Progressive refinement using LGBM-SHAP and LASSO ensemble
    """
    
    def __init__(self, config: Optional[FeatureSelectionConfig] = None):
        """Initialize the enhanced multi-stage feature selector."""
        self.config = config or FeatureSelectionConfig()
        self.logger = logger.getChild('EnhancedMultiStageFeatureSelector')
        
        # Initialize VectorBT mRMR selector
        vectorbt_config = VectorBTFeatureSelectionConfig(
            enable_vectorbt=self.config.enable_vectorbt_optimization,
            chunk_size=self.config.vectorbt_chunk_size,
            enable_parallel=self.config.vectorbt_enable_parallel
        )
        self.mrmr_selector = VectorBTMRMRSelector(vectorbt_config)
        
        # Check availability of required libraries
        self.lightgbm_available = LIGHTGBM_AVAILABLE
        self.shap_available = SHAP_AVAILABLE
        
        if not self.lightgbm_available:
            tprint_warning("⚠️ LightGBM not available - LGBM-SHAP methods will be disabled")
        if not self.shap_available:
            tprint_warning("⚠️ SHAP not available - SHAP-based methods will be disabled")
        
        tprint_success("🚀 EnhancedMultiStageFeatureSelector initialized")
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       symbol: str = "BTCUSDT", exchange: str = "binance", 
                       timeframe: str = "15m") -> FeatureSelectionResult:
        """
        Execute enhanced multi-stage feature selection.
        
        Args:
            X: Feature matrix
            y: Target variable
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            
        Returns:
            FeatureSelectionResult with selected features and metrics
        """
        start_time = time.time()
        tprint("🚀 Starting enhanced multi-stage feature selection")
        tprint_info(f"   📊 Input data shape: {X.shape}")
        tprint_info(f"   📊 Target shape: {y.shape}")
        tprint_info(f"   📊 Symbol: {symbol}, Exchange: {exchange}, Timeframe: {timeframe}")
        
        try:
            # Validate inputs
            self._validate_inputs(X, y)
            
            # Initialize result tracking
            stage_results = {}
            selected_features = X.columns.tolist()
            feature_importance = {}
            feature_scores = {}
            
            tprint_info(f"   📊 Starting with {len(selected_features)} features")
            tprint_info(f"   📊 Target features: {self.config.target_features}")
            
            # Stage 1: mRMR + Spearman combination
            tprint("📊 Stage 1: mRMR + Spearman combination (70% mRMR + 30% Spearman)")
            stage_1_result = self._stage_1_mrmr_spearman_combination(X, y, selected_features)
            selected_features = stage_1_result['selected_features']
            stage_results['stage_1'] = stage_1_result
            
            tprint_success(f"   ✅ Stage 1 completed: {len(selected_features)} features selected")
            
            # Stage 2: Progressive refinement
            tprint("📊 Stage 2: Progressive refinement using LGBM-SHAP and LASSO ensemble")
            stage_2_result = self._stage_2_progressive_refinement(X[selected_features], y, selected_features)
            selected_features = stage_2_result['selected_features']
            stage_results['stage_2'] = stage_2_result
            
            tprint_success(f"   ✅ Stage 2 completed: {len(selected_features)} features selected")
            
            # Calculate final metrics
            performance_metrics = self._calculate_performance_metrics(X[selected_features], y)
            
            # Create result
            execution_time = time.time() - start_time
            result = FeatureSelectionResult(
                selected_features=selected_features,
                feature_importance=feature_importance,
                feature_scores=feature_scores,
                performance_metrics=performance_metrics,
                validation_scores={},
                config_used=self.config,
                execution_time=execution_time,
                memory_usage={},
                stage_results=stage_results,
                success=True
            )
            
            tprint_success(f"✅ Enhanced feature selection completed successfully in {execution_time:.2f}s")
            tprint_info(f"   📊 Final result: {len(selected_features)} features selected from {len(X.columns)}")
            tprint_info(f"   📊 Reduction ratio: {len(selected_features)/len(X.columns):.1%}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Enhanced feature selection failed: {e}"
            tprint_error(f"❌ {error_msg}")
            
            return FeatureSelectionResult(
                selected_features=[],
                feature_importance={},
                feature_scores={},
                performance_metrics={},
                validation_scores={},
                config_used=self.config,
                execution_time=execution_time,
                memory_usage={},
                success=False,
                error_message=str(e)
            )
    
    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series):
        """Validate input data."""
        if X is None or X.empty:
            raise ValueError("Input feature matrix X is None or empty")
        
        if y is None or y.empty:
            raise ValueError("Target variable y is None or empty")
        
        if len(X) != len(y):
            raise ValueError(f"Feature matrix length ({len(X)}) doesn't match target length ({len(y)})")
        
        if X.shape[1] == 0:
            raise ValueError("Feature matrix has no columns")
    
    def _stage_1_mrmr_spearman_combination(self, X: pd.DataFrame, y: pd.Series, 
                                         initial_features: List[str]) -> Dict[str, Any]:
        """
        Stage 1: Combine mRMR and Spearman correlation to select top 50% above target.
        
        Uses 70% mRMR + 30% Spearman weighting.
        """
        tprint_debug("🔍 Stage 1: mRMR + Spearman combination")
        tprint_debug(f"   📊 Input features: {len(initial_features)}")
        tprint_debug(f"   📊 Data shape: {X.shape}")
        
        try:
            # Calculate target number of features (50% above final target)
            target_features = self.config.target_features
            stage1_target = int(target_features * (1 + self.config.stage1_target_ratio))
            stage1_target = min(stage1_target, len(initial_features))
            
            tprint_debug(f"   📊 Stage 1 target: {stage1_target} features")
            
            # Calculate mRMR scores using VectorBT
            tprint_debug("   📊 Calculating mRMR scores")
            mrmr_scores = self._calculate_mrmr_scores(X, y)
            
            # Calculate Spearman correlation scores
            tprint_debug("   📊 Calculating Spearman correlation scores")
            spearman_scores = self._calculate_spearman_scores(X, y)
            
            # Combine scores with weights
            tprint_debug("   📊 Combining scores with weights")
            combined_scores = (
                self.config.stage1_mrmr_weight * mrmr_scores + 
                self.config.stage1_spearman_weight * spearman_scores
            )
            
            # Select top features
            selected_features = self._select_top_features(
                initial_features, combined_scores, stage1_target
            )
            
            tprint_debug(f"   ✅ Stage 1 completed: {len(selected_features)} features selected")
            
            return {
                'selected_features': selected_features,
                'mrmr_scores': mrmr_scores.to_dict(),
                'spearman_scores': spearman_scores.to_dict(),
                'combined_scores': combined_scores.to_dict(),
                'target_count': stage1_target,
                'method': 'mrmr_spearman_combination',
                'weights': {
                    'mrmr': self.config.stage1_mrmr_weight,
                    'spearman': self.config.stage1_spearman_weight
                }
            }
            
        except Exception as e:
            error_msg = f"Stage 1 mRMR+Spearman combination failed: {e}"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def _stage_2_progressive_refinement(self, X: pd.DataFrame, y: pd.Series, 
                                      current_features: List[str]) -> Dict[str, Any]:
        """
        Stage 2: Progressive refinement using RFE with percentage-based step size.
        
        Uses RFE to recursively remove 10% of features above target in each round.
        Uses bootstrap stability and CV only when 40+ features away from target.
        """
        tprint_debug("🔍 Stage 2: Progressive refinement with RFE")
        tprint_debug(f"   📊 Input features: {len(current_features)}")
        tprint_debug(f"   📊 Data shape: {X.shape}")
        
        try:
            target_features = self.config.target_features
            current_features = current_features.copy()
            
            tprint_debug(f"   📊 Target features: {target_features}")
            tprint_debug(f"   📊 RFE step percentage: {self.config.rfe_step_size:.1%}")
            tprint_debug(f"   📊 Bootstrap/CV threshold: {self.config.stage2_bootstrap_cv_threshold} features")
            
            # Check if we should use bootstrap stability and CV
            features_above_target = len(current_features) - target_features
            use_bootstrap_cv = features_above_target >= self.config.stage2_bootstrap_cv_threshold
            tprint_debug(f"   📊 Use bootstrap stability and CV: {use_bootstrap_cv} (threshold: {self.config.stage2_bootstrap_cv_threshold})")
            
            # Use RFE with percentage-based step size
            selected_features = self._rfe_with_percentage_step(
                X[current_features], y, current_features, target_features, use_bootstrap_cv
            )
            
            tprint_debug(f"   ✅ Stage 2 completed: {len(selected_features)} features selected")
            
            return {
                'selected_features': selected_features,
                'target_count': target_features,
                'method': 'rfe_percentage_based',
                'use_bootstrap_cv': use_bootstrap_cv
            }
            
        except Exception as e:
            error_msg = f"Stage 2 progressive refinement failed: {e}"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def _rfe_with_percentage_step(self, X: pd.DataFrame, y: pd.Series, 
                                 feature_names: List[str], target_features: int,
                                 use_bootstrap_cv: bool = False) -> List[str]:
        """
        Recursive Feature Elimination with percentage-based step size.
        
        Removes 10% of features above target in each RFE round, recursively.
        """
        tprint_debug("🔍 Starting RFE with percentage-based step size")
        
        try:
            current_features = feature_names.copy()
            current_X = X.copy()
            rfe_rounds = []
            
            while len(current_features) > target_features:
                features_above_target = len(current_features) - target_features
                
                # Calculate step size as percentage of features above target
                step_size = max(1, int(features_above_target * self.config.rfe_step_size))
                tprint_debug(f"   📊 RFE Round: {len(current_features)} features, {features_above_target} above target")
                tprint_debug(f"   📊 Step size: {step_size} features (10% of {features_above_target})")
                
                # Calculate feature importance using ensemble methods
                feature_scores = self._calculate_ensemble_feature_scores(
                    current_X, y, use_bootstrap_cv=use_bootstrap_cv
                )
                
                # Select features to remove (lowest scores)
                features_to_remove = self._select_features_to_remove(
                    current_features, feature_scores, step_size
                )
                
                # Remove features
                current_features = [f for f in current_features if f not in features_to_remove]
                current_X = current_X.drop(columns=features_to_remove)
                
                tprint_debug(f"   📊 Removed {len(features_to_remove)} features: {features_to_remove}")
                
                rfe_rounds.append({
                    'round': len(rfe_rounds) + 1,
                    'features_remaining': len(current_features),
                    'features_removed': len(features_to_remove),
                    'step_size': step_size,
                    'features_above_target': features_above_target,
                    'features_removed_list': features_to_remove
                })
                
                # Safety check to prevent infinite loop
                if len(rfe_rounds) > 100:
                    tprint_warning("   ⚠️ Maximum RFE rounds reached, stopping")
                    break
            
            tprint_debug(f"   ✅ RFE completed: {len(current_features)} features selected in {len(rfe_rounds)} rounds")
            
            return current_features
            
        except Exception as e:
            tprint_warning(f"   ⚠️ RFE with percentage step failed: {e}")
            # Fallback to simple correlation-based selection
            return self._fallback_feature_selection(X, y, feature_names, target_features)
    
    def _fallback_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                   feature_names: List[str], target_features: int) -> List[str]:
        """Fallback feature selection using simple correlation."""
        try:
            tprint_debug("   📊 Using fallback correlation-based selection")
            
            # Calculate correlation scores
            correlations = []
            for col in X.columns:
                corr, _ = spearmanr(X[col], y)
                correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
            
            # Select top features
            feature_scores = pd.Series(correlations, index=X.columns)
            sorted_features = feature_scores.sort_values(ascending=False)
            
            selected_features = sorted_features.head(target_features).index.tolist()
            
            tprint_debug(f"   ✅ Fallback selection completed: {len(selected_features)} features selected")
            
            return selected_features
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Fallback selection failed: {e}")
            # Last resort: select first target_features
            return feature_names[:target_features]

    def _calculate_mrmr_scores(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calculate mRMR scores using VectorBT optimization."""
        try:
            # Use VectorBT mRMR selector
            result = self.mrmr_selector.select_features(
                X.values, y.values, 
                k=min(len(X.columns), 100),  # Use reasonable k for scoring
                feature_names=X.columns.tolist()
            )
            
            if result['success']:
                # Create scores based on selection order
                scores = pd.Series(0.0, index=X.columns)
                for i, feature in enumerate(result['selected_features']):
                    if feature in scores.index:
                        scores[feature] = 1.0 - (i / len(result['selected_features']))
                return scores
            else:
                # Fallback to uniform scores
                return pd.Series(0.5, index=X.columns)
                
        except Exception as e:
            tprint_warning(f"   ⚠️ mRMR calculation failed: {e}")
            return pd.Series(0.5, index=X.columns)
    
    def _calculate_spearman_scores(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Calculate Spearman correlation scores."""
        try:
            scores = []
            for col in X.columns:
                corr, _ = spearmanr(X[col], y)
                scores.append(abs(corr) if not np.isnan(corr) else 0.0)
            
            return pd.Series(scores, index=X.columns)
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Spearman calculation failed: {e}")
            return pd.Series(0.0, index=X.columns)
    
    def _calculate_ensemble_feature_scores(self, X: pd.DataFrame, y: pd.Series, 
                                         use_bootstrap_cv: bool = False) -> Dict[str, float]:
        """Calculate ensemble feature scores using multiple methods."""
        try:
            ensemble_scores = {}
            
            # LGBM-SHAP scores
            if self.lightgbm_available and self.shap_available:
                lgbm_scores = self._calculate_lgbm_shap_scores(X, y)
                ensemble_scores.update(lgbm_scores)
            
            # LASSO ensemble scores
            lasso_scores = self._calculate_lasso_ensemble_scores(X, y)
            ensemble_scores.update(lasso_scores)
            
            # RFE scores
            rfe_scores = self._calculate_rfe_scores(X, y)
            ensemble_scores.update(rfe_scores)
            
            # Bootstrap stability scores (only when threshold is met)
            if use_bootstrap_cv:
                tprint_debug("   📊 Using bootstrap stability and CV (40+ features away from target)")
                stability_scores = self._calculate_bootstrap_stability_scores(X, y)
                ensemble_scores.update(stability_scores)
            else:
                tprint_debug("   📊 Skipping bootstrap stability (within 40 features of target)")
            
            # Combine scores with ensemble weights
            final_scores = self._combine_ensemble_scores(ensemble_scores)
            
            return final_scores
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Ensemble scoring failed: {e}")
            # Fallback to simple correlation scores
            return self._calculate_spearman_scores(X, y).to_dict()
    
    def _calculate_lgbm_shap_scores(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate LGBM-SHAP feature scores."""
        try:
            if not self.lightgbm_available or not self.shap_available:
                return {}
            
            # Train LightGBM model
            lgb_model = lgb.LGBMRegressor(**self.config.lgbm_params)
            lgb_model.fit(X, y)
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(lgb_model)
            shap_values = explainer.shap_values(X)
            
            # Calculate feature importance as mean absolute SHAP values
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            
            scores = {}
            for i, feature in enumerate(X.columns):
                scores[f"lgbm_shap_{feature}"] = feature_importance[i]
            
            return scores
            
        except Exception as e:
            tprint_warning(f"   ⚠️ LGBM-SHAP calculation failed: {e}")
            return {}
    
    def _calculate_lasso_ensemble_scores(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate LASSO ensemble feature scores."""
        try:
            # Use LASSO with cross-validation
            lasso = LassoCV(
                alphas=np.logspace(
                    np.log10(self.config.lasso_alpha_range[0]),
                    np.log10(self.config.lasso_alpha_range[1]),
                    self.config.lasso_n_alphas
                ),
                cv=self.config.lasso_cv_folds,
                random_state=42
            )
            
            lasso.fit(X, y)
            
            # Calculate feature importance as absolute coefficients
            feature_importance = np.abs(lasso.coef_)
            
            scores = {}
            for i, feature in enumerate(X.columns):
                scores[f"lasso_{feature}"] = feature_importance[i]
            
            return scores
            
        except Exception as e:
            tprint_warning(f"   ⚠️ LASSO ensemble calculation failed: {e}")
            return {}
    
    def _calculate_rfe_scores(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate RFE feature scores."""
        try:
            # Use Random Forest for RFE
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            rfe = RFE(
                estimator, 
                n_features_to_select=max(1, len(X.columns) // 2),
                step=self.config.rfe_step_size
            )
            
            rfe.fit(X, y)
            
            # Calculate feature importance based on RFE ranking
            feature_ranking = rfe.ranking_
            max_ranking = np.max(feature_ranking)
            
            scores = {}
            for i, feature in enumerate(X.columns):
                # Lower ranking = higher importance
                scores[f"rfe_{feature}"] = 1.0 - (feature_ranking[i] - 1) / max_ranking
            
            return scores
            
        except Exception as e:
            tprint_warning(f"   ⚠️ RFE calculation failed: {e}")
            return {}
    
    def _calculate_bootstrap_stability_scores(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate bootstrap stability scores."""
        try:
            n_samples = len(X)
            bootstrap_size = int(n_samples * self.config.bootstrap_sample_ratio)
            
            feature_selection_counts = {feature: 0 for feature in X.columns}
            
            # Bootstrap sampling
            for _ in range(self.config.bootstrap_n_samples):
                # Sample with replacement
                bootstrap_indices = np.random.choice(n_samples, size=bootstrap_size, replace=True)
                X_bootstrap = X.iloc[bootstrap_indices]
                y_bootstrap = y.iloc[bootstrap_indices]
                
                # Simple feature selection using correlation
                correlations = []
                for col in X_bootstrap.columns:
                    corr, _ = spearmanr(X_bootstrap[col], y_bootstrap)
                    correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
                
                # Select top features
                top_indices = np.argsort(correlations)[-len(X.columns)//2:]
                for idx in top_indices:
                    feature_selection_counts[X.columns[idx]] += 1
            
            # Calculate stability scores
            scores = {}
            for feature in X.columns:
                stability = feature_selection_counts[feature] / self.config.bootstrap_n_samples
                scores[f"bootstrap_{feature}"] = stability
            
            return scores
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Bootstrap stability calculation failed: {e}")
            return {}
    
    def _combine_ensemble_scores(self, ensemble_scores: Dict[str, float]) -> Dict[str, float]:
        """Combine ensemble scores using configured weights."""
        try:
            # Group scores by feature
            feature_scores = {}
            for score_name, score_value in ensemble_scores.items():
                # Extract feature name and method
                if '_' in score_name:
                    method, feature = score_name.split('_', 1)
                    if feature not in feature_scores:
                        feature_scores[feature] = {}
                    feature_scores[feature][method] = score_value
            
            # Combine scores for each feature
            final_scores = {}
            for feature, method_scores in feature_scores.items():
                combined_score = 0.0
                total_weight = 0.0
                
                for method, score in method_scores.items():
                    weight = self.config.ensemble_weights.get(method, 0.0)
                    combined_score += weight * score
                    total_weight += weight
                
                if total_weight > 0:
                    final_scores[feature] = combined_score / total_weight
                else:
                    final_scores[feature] = 0.0
            
            return final_scores
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Score combination failed: {e}")
            return {}
    
    
    def _select_features_to_remove(self, features: List[str], scores: Dict[str, float], 
                                 batch_size: int) -> List[str]:
        """Select features to remove based on scores."""
        try:
            # Get scores for current features
            feature_scores = {f: scores.get(f, 0.0) for f in features}
            
            # Sort by score (ascending - remove lowest scores)
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1])
            
            # Select features to remove
            features_to_remove = [f for f, _ in sorted_features[:batch_size]]
            
            return features_to_remove
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Feature selection failed: {e}")
            # Fallback: remove first batch_size features
            return features[:batch_size]
    
    def _select_top_features(self, features: List[str], scores: pd.Series, 
                           target_count: int) -> List[str]:
        """Select top features based on scores."""
        try:
            # Sort features by score (descending)
            sorted_features = scores.sort_values(ascending=False)
            
            # Select top features
            selected_features = sorted_features.head(target_count).index.tolist()
            
            return selected_features
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Top feature selection failed: {e}")
            # Fallback: select first target_count features
            return features[:target_count]
    
    def _calculate_performance_metrics(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Calculate performance metrics for selected features."""
        try:
            metrics = {
                'n_features': len(X.columns),
                'n_samples': len(X),
                'feature_diversity': len(set([col.split('_')[0] for col in X.columns])),
                'data_quality': {
                    'missing_ratio': X.isnull().sum().sum() / (len(X) * len(X.columns)),
                    'variance_ratio': X.var().mean() / X.var().std() if X.var().std() > 0 else 0
                }
            }
            
            return metrics
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Performance metrics calculation failed: {e}")
            return {}