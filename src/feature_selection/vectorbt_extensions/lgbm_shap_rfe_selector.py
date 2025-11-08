"""
LGBM-SHAP with Recursive Feature Elimination (RFE)

This module implements a sophisticated feature selection method that combines
LGBM importance scores with SHAP values and uses RFE to iteratively remove
the least important features until reaching the target count.

Key Features:
- Removes 25% of features per iteration when above target
- Uses LGBM importance + SHAP values for feature ranking
- Comprehensive logging with tprint
- Detailed reporting with global and per-feature metrics
- Handles NaN values and data validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime
import os
import json
import warnings
from dataclasses import dataclass, field

# Import LGBM and SHAP
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    lgb = None

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

# Import project utilities
from src.utils.tprint import (
    tprint, tprint_success, tprint_warning, tprint_performance, tprint_debug,
    tprint_info, tprint_error, tprint_data_preview, tprint_data_format,
    tprint_feature_counts, tprint_structured, tprint_timer, tprint_progress
)
from src.training.utils.feature_selection.selection_methods import FeatureImportanceRanker

logger = logging.getLogger(__name__)

@dataclass
class LGBMSHAPRFEConfig:
    """Configuration for LGBM-SHAP RFE selector."""
    
    # Target settings
    target_features: int = 80  # Target 80 features for SHAP/LGBM filtering
    removal_percentage: float = 0.25  # Remove 25% of features per iteration
    
    # LGBM settings
    lgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    })
    
    # SHAP settings
    shap_explainer: str = 'tree'  # 'tree', 'linear', 'kernel'
    shap_sample_size: Optional[int] = None  # None for all samples
    
    # RFE settings
    min_features_to_keep: int = 10
    max_iterations: int = 20
    early_stopping_patience: int = 3
    
    # Validation settings
    cv_folds: int = 5
    validation_size: float = 0.2
    
    # Reporting settings
    enable_detailed_logging: bool = True
    save_intermediate_results: bool = True
    report_format: str = 'both'  # 'json', 'markdown', 'both'

class LGBMSHAPRFESelector:
    """
    LGBM-SHAP with Recursive Feature Elimination selector.
    
    This selector combines LGBM importance scores with SHAP values to rank features
    and uses RFE to iteratively remove the least important features.
    """
    
    def __init__(self, config: Optional[LGBMSHAPRFEConfig] = None):
        """Initialize the LGBM-SHAP RFE selector."""
        tprint_info("🚀 Initializing LGBM-SHAP RFE Selector")
        
        self.config = config or LGBMSHAPRFEConfig()
        self.logger = logger.getChild('LGBMSHAPRFESelector')
        
        # Log configuration
        config_info = {
            "target_features": self.config.target_features,
            "removal_percentage": self.config.removal_percentage,
            "max_iterations": self.config.max_iterations,
            "min_features_to_keep": self.config.min_features_to_keep,
            "shap_explainer": self.config.shap_explainer,
            "cv_folds": self.config.cv_folds,
            "validation_size": self.config.validation_size
        }
        tprint_structured(config_info, "LGBM-SHAP RFE Configuration")
        
        # Validate dependencies
        if not LGBM_AVAILABLE:
            tprint_error("❌ LightGBM is required but not available")
            raise ImportError("LightGBM is required but not available")
        if not SHAP_AVAILABLE:
            tprint_error("❌ SHAP is required but not available")
            raise ImportError("SHAP is required but not available")
        
        tprint_success("✅ Dependencies validated")
        
        # Initialize components
        tprint_info("🔧 Initializing feature ranker")
        self.feature_ranker = FeatureImportanceRanker()
        
        # Tracking variables
        self.selection_history = []
        self.feature_importance_history = []
        self.performance_history = []
        self.removed_features_history = []
        
        tprint_success("🔧 LGBM-SHAP RFE Selector initialized")
    
    def select_features(self, 
                       X: Union[np.ndarray, pd.DataFrame], 
                       y: Union[np.ndarray, pd.Series],
                       feature_names: Optional[List[str]] = None,
                       target_features: Optional[int] = None) -> Dict[str, Any]:
        """
        Select features using LGBM-SHAP with RFE.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
            target_features: Target number of features (overrides config)
            
        Returns:
            Dictionary containing selection results and metrics
        """
        tprint_info("🚀 Starting LGBM-SHAP RFE feature selection")
        
        # Log input data
        tprint_data_preview(X, "Input Feature Matrix", max_rows=3, max_cols=5)
        tprint_data_format(X, "Input Feature Matrix", check_compatibility=True)
        tprint_data_preview(y, "Input Target Variable", max_rows=5, max_cols=1)
        
        # Set target features
        if target_features is not None:
            tprint_info(f"🎯 Overriding target features: {self.config.target_features} -> {target_features}")
            self.config.target_features = target_features
        
        # Log selection parameters
        selection_params = {
            "target_features": self.config.target_features,
            "removal_percentage": self.config.removal_percentage,
            "max_iterations": self.config.max_iterations,
            "min_features_to_keep": self.config.min_features_to_keep
        }
        tprint_structured(selection_params, "Selection Parameters")
        
        # Prepare data
        tprint_info("🔧 Preparing input data")
        X_processed, y_processed, feature_names_processed = self._prepare_data(X, y, feature_names)
        
        # Initialize tracking
        current_features = list(range(X_processed.shape[1]))
        current_feature_names = feature_names_processed.copy()
        iteration = 0
        best_performance = -np.inf
        patience_counter = 0
        
        tprint(f"📊 Starting with {len(current_features)} features, target: {self.config.target_features}")
        
        # Main RFE loop - continue until we reach exactly the target features
        while len(current_features) > self.config.target_features and iteration < self.config.max_iterations:
            iteration += 1
            tprint(f"\n🔄 RFE Iteration {iteration}")
            tprint(f"📈 Current features: {len(current_features)}")
            
            # Check if we can remove features
            if len(current_features) <= self.config.min_features_to_keep:
                tprint_warning(f"⚠️ Reached minimum features limit: {self.config.min_features_to_keep}")
                break
            
            # Calculate number of features to remove
            features_to_remove = max(1, int(len(current_features) * self.config.removal_percentage))
            features_to_remove = min(features_to_remove, len(current_features) - self.config.target_features)
            
            tprint(f"🎯 Will remove {features_to_remove} features (25% of {len(current_features)})")
            
            # Get current feature subset
            X_current = X_processed[:, current_features]
            
            # Train LGBM model
            model, performance = self._train_lgbm_model(X_current, y_processed)
            
            # Calculate feature importance and SHAP values
            importance_scores, shap_values = self._calculate_importance_and_shap(
                model, X_current, y_processed, current_feature_names
            )
            
            # Combine importance and SHAP scores
            combined_scores = self._combine_scores(importance_scores, shap_values)
            
            # Select features to remove (lowest scores)
            features_to_remove_indices = self._select_features_to_remove(
                combined_scores, features_to_remove
            )
            
            # Get names of features to be removed
            removed_feature_names = [current_feature_names[i] for i in features_to_remove_indices]
            
            # Log removed features
            tprint(f"🗑️ Removing {len(removed_feature_names)} features:")
            for i, feature_name in enumerate(removed_feature_names):
                score = combined_scores[features_to_remove_indices[i]]
                tprint(f"   {i+1:2d}. {feature_name} (score: {score:.6f})")
            
            # Update feature lists
            remaining_indices = [i for i in range(len(current_features)) if i not in features_to_remove_indices]
            current_features = [current_features[i] for i in remaining_indices]
            current_feature_names = [current_feature_names[i] for i in remaining_indices]
            
            # Store iteration results
            iteration_result = {
                'iteration': iteration,
                'features_removed': removed_feature_names,
                'features_remaining': len(current_features),
                'performance': performance,
                'importance_scores': importance_scores.tolist(),
                'shap_values': shap_values.tolist() if shap_values is not None else None,
                'combined_scores': combined_scores.tolist()
            }
            self.selection_history.append(iteration_result)
            self.removed_features_history.extend(removed_feature_names)
            
            tprint(f"✅ Iteration {iteration} complete. Features remaining: {len(current_features)}")
            
            # Check for early stopping
            if performance > best_performance:
                best_performance = performance
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.early_stopping_patience:
                tprint_warning(f"⚠️ Early stopping triggered after {patience_counter} iterations without improvement")
                break
        
        # Final results
        final_features = current_features
        final_feature_names = current_feature_names
        
        # Ensure we have exactly the target number of features
        if len(final_features) > self.config.target_features:
            # If we still have too many features, remove the least important ones
            tprint(f"⚠️ Still have {len(final_features)} features, need exactly {self.config.target_features}")
            
            # Get final feature subset for final selection
            X_final = X_processed[:, final_features]
            model_final, _ = self._train_lgbm_model(X_final, y_processed)
            importance_scores, shap_values = self._calculate_importance_and_shap(
                model_final, X_final, y_processed, final_feature_names
            )
            combined_scores = self._combine_scores(importance_scores, shap_values)
            
            # Select exactly target_features
            top_indices = np.argsort(combined_scores)[-self.config.target_features:]
            final_features = [final_features[i] for i in top_indices]
            final_feature_names = [final_feature_names[i] for i in top_indices]
            
            tprint(f"✅ Final selection: {len(final_features)} features")
        
        tprint(f"\n🎉 RFE Complete!")
        tprint(f"📊 Final features: {len(final_features)} (target: {self.config.target_features})")
        tprint(f"📈 Total iterations: {iteration}")
        tprint(f"🗑️ Total features removed: {len(self.removed_features_history)}")
        
        # Generate detailed report
        report = self._generate_detailed_report(
            X_processed, y_processed, final_features, final_feature_names
        )
        
        # Save report
        if self.config.save_intermediate_results:
            self._save_report(report)
        
        return {
            'selected_features': final_features,
            'selected_feature_names': final_feature_names,
            'selection_history': self.selection_history,
            'removed_features': self.removed_features_history,
            'report': report,
            'success': True
        }
    
    def _prepare_data(self, X: Union[np.ndarray, pd.DataFrame], 
                     y: Union[np.ndarray, pd.Series],
                     feature_names: Optional[List[str]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare and validate input data."""
        tprint_info("🔧 Preparing input data")
        
        # Log input data types and shapes
        input_info = {
            "X_type": type(X).__name__,
            "y_type": type(y).__name__,
            "X_shape": X.shape if hasattr(X, 'shape') else len(X),
            "y_length": len(y),
            "feature_names_provided": feature_names is not None
        }
        tprint_structured(input_info, "Input Data Information")
        
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            tprint_info("🔄 Converting DataFrame to numpy array")
            X_array = X.values
            if feature_names is None:
                feature_names = X.columns.tolist()
                tprint_info(f"📝 Using DataFrame column names as feature names: {len(feature_names)} features")
        else:
            X_array = X
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]
                tprint_info(f"📝 Generated feature names: {len(feature_names)} features")
        
        if isinstance(y, pd.Series):
            tprint_info("🔄 Converting Series to numpy array")
            y_array = y.values
        else:
            y_array = y
        
        # Log data after conversion
        tprint_data_preview(X_array, "Processed Feature Matrix", max_rows=3, max_cols=5)
        tprint_data_format(X_array, "Processed Feature Matrix", check_compatibility=True)
        tprint_data_preview(y_array, "Processed Target Variable", max_rows=5, max_cols=1)
        
        # Handle NaN values
        tprint_info("🧹 Checking for NaN values")
        nan_mask = np.isnan(X_array).any(axis=1) | np.isnan(y_array)
        nan_count = np.sum(nan_mask)
        
        if nan_count > 0:
            tprint_warning(f"⚠️ Found {nan_count} rows with NaN values ({nan_count/len(nan_mask)*100:.1f}%)")
            tprint_info(f"🔧 Removing {nan_count} rows with NaN values")
            X_array = X_array[~nan_mask]
            y_array = y_array[~nan_mask]
            
            # Log data after NaN removal
            tprint_data_preview(X_array, "Data After NaN Removal", max_rows=3, max_cols=5)
        else:
            tprint_success("✅ No NaN values found")
        
        # Validate data
        if X_array.shape[0] == 0:
            tprint_error("❌ No valid data remaining after NaN removal")
            raise ValueError("No valid data remaining after NaN removal")
        
        # Log final data statistics
        final_info = {
            "final_samples": X_array.shape[0],
            "final_features": X_array.shape[1],
            "nan_removed": nan_count,
            "data_quality": "Good" if nan_count == 0 else "Cleaned"
        }
        tprint_structured(final_info, "Data Preparation Results")
        
        tprint_success(f"✅ Data prepared: {X_array.shape[0]} samples, {X_array.shape[1]} features")
        
        return X_array, y_array, feature_names
    
    def _train_lgbm_model(self, X: np.ndarray, y: np.ndarray) -> Tuple[Any, float]:
        """Train LGBM model and return performance."""
        tprint_info("🌲 Training LGBM model")
        
        # Log training parameters
        lgb_params_info = {
            "objective": self.config.lgb_params.get('objective', 'regression'),
            "boosting_type": self.config.lgb_params.get('boosting_type', 'gbdt'),
            "num_leaves": self.config.lgb_params.get('num_leaves', 31),
            "learning_rate": self.config.lgb_params.get('learning_rate', 0.05),
            "validation_size": self.config.validation_size
        }
        tprint_structured(lgb_params_info, "LGBM Training Parameters")
        
        # Split data for validation
        n_samples = X.shape[0]
        val_size = int(n_samples * self.config.validation_size)
        indices = np.random.permutation(n_samples)
        
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        # Log data split
        split_info = {
            "total_samples": n_samples,
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "val_percentage": self.config.validation_size
        }
        tprint_structured(split_info, "Data Split Information")
        
        # Log training data
        tprint_data_preview(X_train, "Training Features", max_rows=2, max_cols=3)
        tprint_data_preview(y_train, "Training Target", max_rows=5, max_cols=1)
        tprint_data_preview(X_val, "Validation Features", max_rows=2, max_cols=3)
        tprint_data_preview(y_val, "Validation Target", max_rows=5, max_cols=1)
        
        # Create datasets
        tprint_info("📦 Creating LGBM datasets")
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        tprint_info("🚀 Training LGBM model")
        with tprint_timer("LGBM Training", "PERFORMANCE"):
            model = lgb.train(
                self.config.lgb_params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=100,
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
        
        # Calculate performance
        tprint_info("📊 Calculating model performance")
        y_pred = model.predict(X_val)
        performance = -np.mean((y_val - y_pred) ** 2)  # Negative MSE for maximization
        
        # Log performance metrics
        performance_info = {
            "mse": -performance,  # Convert back to positive MSE
            "rmse": np.sqrt(-performance),
            "r2": 1 - (np.sum((y_val - y_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)),
            "mae": np.mean(np.abs(y_val - y_pred))
        }
        tprint_structured(performance_info, "Model Performance")
        
        tprint_success(f"✅ LGBM model trained - Performance: {performance:.6f}")
        
        return model, performance
    
    def _calculate_importance_and_shap(self, model: Any, X: np.ndarray, y: np.ndarray,
                                     feature_names: List[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Calculate LGBM importance and SHAP values."""
        tprint_info("📊 Calculating importance and SHAP values")
        
        # Get LGBM importance
        tprint_info("🌲 Getting LGBM feature importance")
        importance_scores = model.feature_importance(importance_type='gain')
        
        # Log importance statistics
        importance_stats = {
            "mean_importance": float(np.mean(importance_scores)),
            "std_importance": float(np.std(importance_scores)),
            "min_importance": float(np.min(importance_scores)),
            "max_importance": float(np.max(importance_scores)),
            "zero_importance_count": int(np.sum(importance_scores == 0))
        }
        tprint_structured(importance_stats, "LGBM Importance Statistics")
        
        # Calculate SHAP values
        tprint_info(f"🔍 Calculating SHAP values using {self.config.shap_explainer} explainer")
        shap_values = None
        try:
            with tprint_timer("SHAP Calculation", "PERFORMANCE"):
                if self.config.shap_explainer == 'tree':
                    tprint_info("🌳 Using TreeExplainer")
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X)
                elif self.config.shap_explainer == 'linear':
                    tprint_info("📏 Using LinearExplainer")
                    explainer = shap.LinearExplainer(model, X)
                    shap_values = explainer.shap_values(X)
                else:
                    tprint_info("🔧 Using KernelExplainer (fallback)")
                    # Use kernel explainer as fallback
                    explainer = shap.KernelExplainer(model.predict, X[:100])  # Sample for efficiency
                    shap_values = explainer.shap_values(X[:100])
            
            # Log SHAP statistics
            if shap_values is not None:
                shap_stats = {
                    "shap_shape": shap_values.shape,
                    "mean_abs_shap": float(np.mean(np.abs(shap_values))),
                    "std_abs_shap": float(np.std(np.abs(shap_values))),
                    "min_shap": float(np.min(shap_values)),
                    "max_shap": float(np.max(shap_values))
                }
                tprint_structured(shap_stats, "SHAP Statistics")
                tprint_success("✅ SHAP values calculated successfully")
            else:
                tprint_warning("⚠️ SHAP values are None")
                
        except Exception as e:
            tprint_error(f"❌ SHAP calculation failed: {e}")
            tprint_warning("⚠️ Continuing without SHAP values")
            shap_values = None
        
        return importance_scores, shap_values
    
    def _combine_scores(self, importance_scores: np.ndarray, 
                       shap_values: Optional[np.ndarray]) -> np.ndarray:
        """Combine LGBM importance and SHAP values."""
        tprint_info("🔗 Combining importance and SHAP scores")
        
        # Normalize importance scores
        tprint_info("📊 Normalizing LGBM importance scores")
        importance_normalized = importance_scores / (np.sum(importance_scores) + 1e-10)
        
        # Log normalization info
        norm_info = {
            "importance_sum": float(np.sum(importance_scores)),
            "importance_normalized_sum": float(np.sum(importance_normalized)),
            "shap_available": shap_values is not None
        }
        tprint_structured(norm_info, "Score Normalization")
        
        if shap_values is not None:
            tprint_info("📊 Normalizing SHAP values")
            # Calculate mean absolute SHAP values
            shap_mean_abs = np.mean(np.abs(shap_values), axis=0)
            shap_normalized = shap_mean_abs / (np.sum(shap_mean_abs) + 1e-10)
            
            # Log SHAP normalization
            shap_norm_info = {
                "shap_mean_abs_sum": float(np.sum(shap_mean_abs)),
                "shap_normalized_sum": float(np.sum(shap_normalized))
            }
            tprint_structured(shap_norm_info, "SHAP Normalization")
            
            # Combine with equal weights
            tprint_info("⚖️ Combining scores with equal weights (50% importance + 50% SHAP)")
            combined_scores = 0.5 * importance_normalized + 0.5 * shap_normalized
            
            # Log combination info
            combo_info = {
                "combination_method": "equal_weights",
                "importance_weight": 0.5,
                "shap_weight": 0.5,
                "combined_sum": float(np.sum(combined_scores))
            }
            tprint_structured(combo_info, "Score Combination")
        else:
            tprint_warning("⚠️ SHAP values not available - using only importance scores")
            # Use only importance scores
            combined_scores = importance_normalized
        
        # Log final combined scores statistics
        final_stats = {
            "mean_combined_score": float(np.mean(combined_scores)),
            "std_combined_score": float(np.std(combined_scores)),
            "min_combined_score": float(np.min(combined_scores)),
            "max_combined_score": float(np.max(combined_scores)),
            "zero_scores_count": int(np.sum(combined_scores == 0))
        }
        tprint_structured(final_stats, "Combined Scores Statistics")
        
        tprint_success("✅ Score combination completed")
        return combined_scores
    
    def _select_features_to_remove(self, scores: np.ndarray, 
                                  n_to_remove: int) -> List[int]:
        """Select features to remove based on scores."""
        tprint_debug(f"🎯 Selecting {n_to_remove} features to remove")
        
        # Get indices of features with lowest scores
        sorted_indices = np.argsort(scores)
        features_to_remove = sorted_indices[:n_to_remove].tolist()
        
        return features_to_remove
    
    def _generate_detailed_report(self, X: np.ndarray, y: np.ndarray,
                                selected_features: List[int], 
                                selected_feature_names: List[str]) -> Dict[str, Any]:
        """Generate detailed report with global and per-feature metrics."""
        tprint("📋 Generating detailed report")
        
        # Global metrics
        global_metrics = self._calculate_global_metrics(X, y, selected_features)
        
        # Per-feature metrics
        per_feature_metrics = self._calculate_per_feature_metrics(
            X, y, selected_features, selected_feature_names
        )
        
        # Selection summary
        selection_summary = {
            'total_iterations': len(self.selection_history),
            'total_features_removed': len(self.removed_features_history),
            'final_feature_count': len(selected_features),
            'target_feature_count': self.config.target_features,
            'removal_percentage': self.config.removal_percentage
        }
        
        # Performance over iterations
        performance_over_time = {
            'iterations': [h['iteration'] for h in self.selection_history],
            'features_remaining': [h['features_remaining'] for h in self.selection_history],
            'performance': [h['performance'] for h in self.selection_history]
        }
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'target_features': self.config.target_features,
                'removal_percentage': self.config.removal_percentage,
                'lgb_params': self.config.lgb_params,
                'shap_explainer': self.config.shap_explainer
            },
            'selection_summary': selection_summary,
            'global_metrics': global_metrics,
            'per_feature_metrics': per_feature_metrics,
            'performance_over_time': performance_over_time,
            'removed_features': self.removed_features_history,
            'selected_features': selected_feature_names
        }
        
        return report
    
    def _calculate_global_metrics(self, X: np.ndarray, y: np.ndarray,
                                selected_features: List[int]) -> Dict[str, Any]:
        """Calculate global performance metrics."""
        tprint_debug("📊 Calculating global metrics")
        
        # Get selected feature subset
        X_selected = X[:, selected_features]
        
        # Calculate basic statistics
        n_samples, n_features = X_selected.shape
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X_selected.T)
        avg_correlation = np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
        
        # Calculate variance
        feature_variances = np.var(X_selected, axis=0)
        avg_variance = np.mean(feature_variances)
        
        # Calculate mutual information (simplified)
        target_variance = np.var(y)
        
        return {
            'n_samples': int(n_samples),
            'n_features': int(n_features),
            'avg_correlation': float(avg_correlation),
            'avg_variance': float(avg_variance),
            'target_variance': float(target_variance),
            'feature_variance_std': float(np.std(feature_variances))
        }
    
    def _calculate_per_feature_metrics(self, X: np.ndarray, y: np.ndarray,
                                     selected_features: List[int],
                                     selected_feature_names: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Calculate per-feature metrics."""
        tprint_debug("📊 Calculating per-feature metrics")
        
        X_selected = X[:, selected_features]
        per_feature_metrics = []
        
        for i, feature_name in enumerate(selected_feature_names):
            feature_values = X_selected[:, i]
            
            # Basic statistics
            mean_val = np.mean(feature_values)
            std_val = np.std(feature_values)
            min_val = np.min(feature_values)
            max_val = np.max(feature_values)
            
            # Correlation with target
            correlation = np.corrcoef(feature_values, y)[0, 1] if len(feature_values) > 1 else 0
            
            # Variance
            variance = np.var(feature_values)
            
            # Skewness and kurtosis
            from scipy import stats
            skewness = stats.skew(feature_values)
            kurtosis = stats.kurtosis(feature_values)
            
            per_feature_metrics.append({
                'feature_name': feature_name,
                'feature_index': selected_features[i],
                'mean': float(mean_val),
                'std': float(std_val),
                'min': float(min_val),
                'max': float(max_val),
                'variance': float(variance),
                'correlation_with_target': float(correlation),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis)
            })
        
        return per_feature_metrics
    
    def _save_report(self, report: Dict[str, Any]) -> str:
        """Save detailed report to outcomes directory."""
        tprint("💾 Saving detailed report")
        
        # Create outcomes directory if it doesn't exist
        outcomes_dir = "/workspace/outcomes"
        os.makedirs(outcomes_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"lgbm_shap_rfe_report_{timestamp}"
        
        # Save JSON report
        if self.config.report_format in ['json', 'both']:
            json_filename = f"{base_filename}.json"
            json_path = os.path.join(outcomes_dir, json_filename)
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2)
            tprint_success(f"✅ JSON report saved: {json_path}")
        
        # Save Markdown report
        if self.config.report_format in ['markdown', 'both']:
            md_filename = f"{base_filename}.md"
            md_path = os.path.join(outcomes_dir, md_filename)
            self._save_markdown_report(report, md_path)
            tprint_success(f"✅ Markdown report saved: {md_path}")
        
        return base_filename
    
    def _save_markdown_report(self, report: Dict[str, Any], filepath: str) -> None:
        """Save report in Markdown format."""
        with open(filepath, 'w') as f:
            f.write(f"# LGBM-SHAP RFE Feature Selection Report\n\n")
            f.write(f"**Generated:** {report['timestamp']}\n\n")
            
            # Selection Summary
            f.write("## 📊 Selection Summary\n\n")
            summary = report['selection_summary']
            f.write(f"- **Total Iterations:** {summary['total_iterations']}\n")
            f.write(f"- **Features Removed:** {summary['total_features_removed']}\n")
            f.write(f"- **Final Features:** {summary['final_feature_count']}\n")
            f.write(f"- **Target Features:** {summary['target_feature_count']}\n")
            f.write(f"- **Removal Percentage:** {summary['removal_percentage']:.1%}\n\n")
            
            # Global Metrics
            f.write("## 🌍 Global Metrics\n\n")
            global_metrics = report['global_metrics']
            f.write(f"- **Samples:** {global_metrics['n_samples']:,}\n")
            f.write(f"- **Features:** {global_metrics['n_features']}\n")
            f.write(f"- **Avg Correlation:** {global_metrics['avg_correlation']:.4f}\n")
            f.write(f"- **Avg Variance:** {global_metrics['avg_variance']:.4f}\n")
            f.write(f"- **Target Variance:** {global_metrics['target_variance']:.4f}\n\n")
            
            # Selected Features
            f.write("## ✅ Selected Features\n\n")
            for i, feature_name in enumerate(report['selected_features'], 1):
                f.write(f"{i:2d}. {feature_name}\n")
            f.write("\n")
            
            # Removed Features
            f.write("## 🗑️ Removed Features\n\n")
            for i, feature_name in enumerate(report['removed_features'], 1):
                f.write(f"{i:2d}. {feature_name}\n")
            f.write("\n")
            
            # Per-Feature Metrics
            f.write("## 📈 Per-Feature Metrics\n\n")
            f.write("| Feature | Mean | Std | Min | Max | Variance | Correlation | Skewness | Kurtosis |\n")
            f.write("|---------|------|-----|-----|-----|----------|-------------|----------|----------|\n")
            
            for feature_metrics in report['per_feature_metrics']:
                f.write(f"| {feature_metrics['feature_name']} | "
                       f"{feature_metrics['mean']:.4f} | "
                       f"{feature_metrics['std']:.4f} | "
                       f"{feature_metrics['min']:.4f} | "
                       f"{feature_metrics['max']:.4f} | "
                       f"{feature_metrics['variance']:.4f} | "
                       f"{feature_metrics['correlation_with_target']:.4f} | "
                       f"{feature_metrics['skewness']:.4f} | "
                       f"{feature_metrics['kurtosis']:.4f} |\n")
            
            f.write("\n")
            
            # Performance Over Time
            f.write("## 📊 Performance Over Time\n\n")
            f.write("| Iteration | Features Remaining | Performance |\n")
            f.write("|-----------|-------------------|-------------|\n")
            
            perf_data = report['performance_over_time']
            for i in range(len(perf_data['iterations'])):
                f.write(f"| {perf_data['iterations'][i]} | "
                       f"{perf_data['features_remaining'][i]} | "
                       f"{perf_data['performance'][i]:.6f} |\n")
            
            f.write("\n")


def create_lgbm_shap_rfe_selector(config: Optional[LGBMSHAPRFEConfig] = None) -> LGBMSHAPRFESelector:
    """Create and return a LGBM-SHAP RFE selector instance."""
    return LGBMSHAPRFESelector(config)


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_samples, n_features = 1000, 200
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Create selector
    config = LGBMSHAPRFEConfig(target_features=80, removal_percentage=0.25)
    selector = create_lgbm_shap_rfe_selector(config)
    
    # Run selection
    result = selector.select_features(X, y, feature_names)
    
    print(f"Selected {len(result['selected_features'])} features")
    print(f"Removed {len(result['removed_features'])} features")