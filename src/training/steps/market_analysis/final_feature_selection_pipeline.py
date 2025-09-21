#!/usr/bin/env python3
"""
Final Feature Selection Pipeline

This module implements a comprehensive multi-stage feature selection pipeline
that runs at the end of the market analysis pipeline, progressively reducing
features from 120 → 100 → 80 → 60 using RandomForest and SHAP analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
import time
import json
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib

# Try to import SHAP, fallback if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

# Import system utilities
from src.utils.logger import get_logger
from src.utils.matrix_operations import get_unified_matrix_operations

@dataclass
class FeatureSelectionConfig:
    """Configuration for multi-stage feature selection."""
    # Stage targets
    initial_features: int = 120
    stage_1_target: int = 100
    stage_2_target: int = 80
    stage_3_target: int = 60
    
    # RandomForest parameters
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    rf_min_samples_split: int = 5
    rf_random_state: int = 42
    
    # SHAP parameters
    shap_sample_size: int = 1000
    shap_max_features: int = 200
    
    # Cross-validation
    cv_folds: int = 5
    cv_scoring: str = 'neg_mean_squared_error'
    
    # Quality thresholds
    min_feature_importance: float = 0.001
    min_correlation_threshold: float = 0.95
    min_variance_threshold: float = 0.01
    
    # Output settings
    save_models: bool = True
    save_analysis: bool = True
    output_directory: str = "outcomes"
    verbose: bool = True

@dataclass
class FeatureSelectionResult:
    """Result of feature selection analysis."""
    # Stage results
    stage_1_features: List[str] = field(default_factory=list)
    stage_2_features: List[str] = field(default_factory=list)
    stage_3_features: List[str] = field(default_factory=list)
    final_features: List[str] = field(default_factory=list)
    
    # Scores and metrics
    stage_1_scores: Dict[str, float] = field(default_factory=dict)
    stage_2_scores: Dict[str, float] = field(default_factory=dict)
    stage_3_scores: Dict[str, float] = field(default_factory=dict)
    final_scores: Dict[str, float] = field(default_factory=dict)
    
    # Feature importance
    rf_importance: Dict[str, float] = field(default_factory=dict)
    shap_importance: Dict[str, float] = field(default_factory=dict)
    combined_importance: Dict[str, float] = field(default_factory=dict)
    
    # Analysis metadata
    feature_counts: Dict[str, int] = field(default_factory=dict)
    selection_time: float = 0.0
    model_performance: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    correlation_analysis: Dict[str, Any] = field(default_factory=dict)
    variance_analysis: Dict[str, Any] = field(default_factory=dict)
    stability_scores: Dict[str, float] = field(default_factory=dict)

class MultiStageFeatureSelector:
    """Multi-stage feature selection using RandomForest and SHAP."""
    
    def __init__(self, config: Optional[FeatureSelectionConfig] = None):
        self.config = config or FeatureSelectionConfig()
        self.logger = get_logger("MultiStageFeatureSelector")
        self.matrix_ops = get_unified_matrix_operations()
        
        # Initialize results
        self.results = FeatureSelectionResult()
        
        self.logger.info("🚀 MultiStageFeatureSelector initialized")
    
    def select_features(self, 
                       X: pd.DataFrame, 
                       y: pd.Series,
                       feature_names: Optional[List[str]] = None) -> FeatureSelectionResult:
        """Perform multi-stage feature selection."""
        
        start_time = time.time()
        self.logger.info("🔍 Starting multi-stage feature selection")
        
        # Validate inputs
        if len(X.columns) < self.config.stage_3_target:
            self.logger.warning(f"⚠️ Input has only {len(X.columns)} features, less than target {self.config.stage_3_target}")
            return self._handle_insufficient_features(X, y)
        
        # Stage 0: Initial feature preparation
        self.logger.info("📊 Stage 0: Initial feature preparation")
        prepared_features = self._prepare_initial_features(X, y, feature_names)
        
        # Stage 1: 120 → 100 features
        self.logger.info("📊 Stage 1: Reducing to 100 features")
        stage_1_features, stage_1_scores = self._stage_1_selection(prepared_features, y)
        
        # Stage 2: 100 → 80 features
        self.logger.info("📊 Stage 2: Reducing to 80 features")
        stage_2_features, stage_2_scores = self._stage_2_selection(prepared_features[stage_1_features], y)
        
        # Stage 3: 80 → 60 features
        self.logger.info("📊 Stage 3: Reducing to 60 features")
        stage_3_features, stage_3_scores = self._stage_3_selection(prepared_features[stage_1_features][stage_2_features], y)
        
        # Compile final results
        self._compile_results(
            prepared_features, y, stage_1_features, stage_2_features, stage_3_features,
            stage_1_scores, stage_2_scores, stage_3_scores
        )
        
        # Save results
        if self.config.save_analysis:
            self._save_analysis()
        
        total_time = time.time() - start_time
        self.results.selection_time = total_time
        
        self.logger.info(f"✅ Multi-stage feature selection completed in {total_time:.3f}s")
        self.logger.info(f"📈 Final feature count: {len(self.results.final_features)}")
        
        return self.results
    
    def _prepare_initial_features(self, X: pd.DataFrame, y: pd.Series, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Prepare initial features for selection."""
        
        # Handle feature names
        if feature_names is not None:
            X = X[feature_names] if len(feature_names) <= len(X.columns) else X
        
        # Remove low variance features
        variance_threshold = self.config.min_variance_threshold
        low_variance_mask = X.var() < variance_threshold
        low_variance_features = X.columns[low_variance_mask].tolist()
        
        if low_variance_features:
            self.logger.info(f"🗑️ Removing {len(low_variance_features)} low variance features")
            X = X.drop(columns=low_variance_features)
        
        # Remove highly correlated features
        correlation_threshold = self.config.min_correlation_threshold
        high_corr_features = self._find_highly_correlated_features(X, correlation_threshold)
        
        if high_corr_features:
            self.logger.info(f"🗑️ Removing {len(high_corr_features)} highly correlated features")
            X = X.drop(columns=high_corr_features)
        
        # Select top features if we have too many
        if len(X.columns) > self.config.initial_features:
            self.logger.info(f"📊 Selecting top {self.config.initial_features} features initially")
            # Use simple variance-based selection for initial filtering
            feature_variance = X.var().sort_values(ascending=False)
            top_features = feature_variance.head(self.config.initial_features).index.tolist()
            X = X[top_features]
        
        self.logger.info(f"✅ Prepared {len(X.columns)} features for selection")
        return X
    
    def _find_highly_correlated_features(self, X: pd.DataFrame, threshold: float) -> List[str]:
        """Find and remove highly correlated features."""
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        
        return to_drop
    
    def _stage_1_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict[str, float]]:
        """Stage 1: 120 → 100 features using RandomForest."""
        
        # Train RandomForest
        rf_model = self._train_random_forest(X, y)
        
        # Get feature importance
        feature_importance = dict(zip(X.columns, rf_model.feature_importances_))
        
        # Select top features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features[:self.config.stage_1_target]]
        
        # Calculate scores
        scores = {
            'rf_importance_score': np.mean(list(feature_importance.values())),
            'feature_variance': X[selected_features].var().mean(),
            'selection_quality': len(selected_features) / len(X.columns)
        }
        
        self.logger.info(f"✅ Stage 1: Selected {len(selected_features)} features")
        return selected_features, scores
    
    def _stage_2_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict[str, float]]:
        """Stage 2: 100 → 80 features using SHAP (if available) or enhanced RandomForest."""
        
        if SHAP_AVAILABLE and len(X.columns) <= self.config.shap_max_features:
            # Use SHAP for more sophisticated selection
            selected_features, scores = self._shap_based_selection(X, y, self.config.stage_2_target)
            self.logger.info("✅ Stage 2: Used SHAP-based selection")
        else:
            # Fallback to enhanced RandomForest
            selected_features, scores = self._enhanced_rf_selection(X, y, self.config.stage_2_target)
            self.logger.info("✅ Stage 2: Used enhanced RandomForest selection")
        
        self.logger.info(f"✅ Stage 2: Selected {len(selected_features)} features")
        return selected_features, scores
    
    def _stage_3_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict[str, float]]:
        """Stage 3: 80 → 60 features using combined importance and cross-validation."""
        
        # Train multiple models for stability
        rf_model = self._train_random_forest(X, y)
        rf_importance = dict(zip(X.columns, rf_model.feature_importances_))
        
        # Cross-validation based selection
        cv_scores = self._cross_validate_feature_importance(X, y)
        
        # Combine importance scores
        combined_scores = {}
        for feature in X.columns:
            rf_score = rf_importance.get(feature, 0)
            cv_score = cv_scores.get(feature, 0)
            combined_scores[feature] = (rf_score + cv_score) / 2
        
        # Select top features
        sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features[:self.config.stage_3_target]]
        
        # Calculate scores
        scores = {
            'combined_importance_score': np.mean(list(combined_scores.values())),
            'rf_cv_agreement': self._calculate_agreement(rf_importance, cv_scores),
            'final_stability': np.std(list(combined_scores.values()))
        }
        
        self.logger.info(f"✅ Stage 3: Selected {len(selected_features)} features")
        return selected_features, scores
    
    def _shap_based_selection(self, X: pd.DataFrame, y: pd.Series, target_count: int) -> Tuple[List[str], Dict[str, float]]:
        """SHAP-based feature selection."""
        
        # Sample data for SHAP analysis
        sample_size = min(self.config.shap_sample_size, len(X))
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[sample_indices]
        y_sample = y.iloc[sample_indices]
        
        # Train model
        rf_model = self._train_random_forest(X_sample, y_sample)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_sample)
        
        # Calculate mean absolute SHAP values
        if isinstance(shap_values, list):  # Classification
            shap_importance = np.mean(np.abs(shap_values), axis=(0, 1))
        else:  # Regression
            shap_importance = np.mean(np.abs(shap_values), axis=0)
        
        # Create feature importance dictionary
        shap_importance_dict = dict(zip(X.columns, shap_importance))
        
        # Select top features
        sorted_features = sorted(shap_importance_dict.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features[:target_count]]
        
        # Calculate scores
        scores = {
            'shap_importance_score': np.mean(shap_importance),
            'shap_variance': np.var(shap_importance),
            'selection_confidence': len(selected_features) / len(X.columns)
        }
        
        return selected_features, scores
    
    def _enhanced_rf_selection(self, X: pd.DataFrame, y: pd.Series, target_count: int) -> Tuple[List[str], Dict[str, float]]:
        """Enhanced RandomForest selection with multiple criteria."""
        
        # Train multiple RandomForest models with different parameters
        models = []
        for n_est in [50, 100, 150]:
            model = RandomForestRegressor(
                n_estimators=n_est,
                max_depth=self.config.rf_max_depth,
                min_samples_split=self.config.rf_min_samples_split,
                random_state=self.config.rf_random_state + n_est
            )
            model.fit(X, y)
            models.append(model)
        
        # Average feature importance across models
        avg_importance = np.zeros(len(X.columns))
        for model in models:
            avg_importance += model.feature_importances_
        avg_importance /= len(models)
        
        # Create feature importance dictionary
        importance_dict = dict(zip(X.columns, avg_importance))
        
        # Select top features
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features[:target_count]]
        
        # Calculate scores
        scores = {
            'enhanced_rf_score': np.mean(avg_importance),
            'model_agreement': 1 - np.std(avg_importance) / np.mean(avg_importance),
            'selection_quality': len(selected_features) / len(X.columns)
        }
        
        return selected_features, scores
    
    def _cross_validate_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Cross-validate feature importance using multiple folds."""
        
        cv_scores = {feature: 0.0 for feature in X.columns}
        
        # Use StratifiedKFold for classification, regular KFold for regression
        if self._is_classification(y):
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.rf_random_state)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.rf_random_state)
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model on fold
            model = self._train_random_forest(X_train, y_train)
            
            # Get feature importance
            fold_importance = dict(zip(X.columns, model.feature_importances_))
            
            # Accumulate scores
            for feature, importance in fold_importance.items():
                cv_scores[feature] += importance
        
        # Average across folds
        for feature in cv_scores:
            cv_scores[feature] /= self.config.cv_folds
        
        return cv_scores
    
    def _train_random_forest(self, X: pd.DataFrame, y: pd.Series):
        """Train RandomForest model."""
        
        if self._is_classification(y):
            model = RandomForestClassifier(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                min_samples_split=self.config.rf_min_samples_split,
                random_state=self.config.rf_random_state
            )
        else:
            model = RandomForestRegressor(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                min_samples_split=self.config.rf_min_samples_split,
                random_state=self.config.rf_random_state
            )
        
        model.fit(X, y)
        return model
    
    def _is_classification(self, y: pd.Series) -> bool:
        """Determine if target is classification or regression."""
        # Simple heuristic: if target has few unique values, treat as classification
        unique_values = len(y.unique())
        return unique_values <= 10 or y.dtype == 'category'
    
    def _calculate_agreement(self, scores1: Dict[str, float], scores2: Dict[str, float]) -> float:
        """Calculate agreement between two scoring methods."""
        common_features = set(scores1.keys()) & set(scores2.keys())
        if not common_features:
            return 0.0
        
        # Calculate correlation between scores
        scores1_values = [scores1[f] for f in common_features]
        scores2_values = [scores2[f] for f in common_features]
        
        correlation = np.corrcoef(scores1_values, scores2_values)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def _compile_results(self, 
                        X: pd.DataFrame, 
                        y: pd.Series,
                        stage_1_features: List[str],
                        stage_2_features: List[str], 
                        stage_3_features: List[str],
                        stage_1_scores: Dict[str, float],
                        stage_2_scores: Dict[str, float],
                        stage_3_scores: Dict[str, float]):
        """Compile final results."""
        
        # Store feature lists
        self.results.stage_1_features = stage_1_features
        self.results.stage_2_features = stage_2_features
        self.results.stage_3_features = stage_3_features
        self.results.final_features = stage_3_features
        
        # Store scores
        self.results.stage_1_scores = stage_1_scores
        self.results.stage_2_scores = stage_2_scores
        self.results.stage_3_scores = stage_3_scores
        
        # Calculate final model performance
        final_model = self._train_random_forest(X[stage_3_features], y)
        cv_scores = cross_val_score(final_model, X[stage_3_features], y, cv=self.config.cv_folds, scoring=self.config.cv_scoring)
        
        self.results.final_scores = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model_score': final_model.score(X[stage_3_features], y)
        }
        
        # Store feature counts
        self.results.feature_counts = {
            'initial': len(X.columns),
            'stage_1': len(stage_1_features),
            'stage_2': len(stage_2_features),
            'stage_3': len(stage_3_features),
            'final': len(stage_3_features)
        }
        
        # Store model performance
        self.results.model_performance = {
            'final_model': final_model,
            'cv_scores': cv_scores.tolist(),
            'feature_importance': dict(zip(stage_3_features, final_model.feature_importances_))
        }
    
    def _handle_insufficient_features(self, X: pd.DataFrame, y: pd.Series) -> FeatureSelectionResult:
        """Handle case where we don't have enough features."""
        
        self.logger.warning("⚠️ Insufficient features for multi-stage selection")
        
        # Use all available features
        self.results.final_features = X.columns.tolist()
        self.results.stage_1_features = X.columns.tolist()
        self.results.stage_2_features = X.columns.tolist()
        self.results.stage_3_features = X.columns.tolist()
        
        # Train final model
        final_model = self._train_random_forest(X, y)
        
        self.results.final_scores = {
            'cv_mean': 0.0,
            'cv_std': 0.0,
            'model_score': final_model.score(X, y)
        }
        
        self.results.feature_counts = {
            'initial': len(X.columns),
            'stage_1': len(X.columns),
            'stage_2': len(X.columns),
            'stage_3': len(X.columns),
            'final': len(X.columns)
        }
        
        return self.results
    
    def _save_analysis(self):
        """Save analysis results."""
        try:
            from datetime import datetime

            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save results summary with proper outcomes naming convention
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = output_dir / f"market_analysis_feature_selection_outcome_{timestamp}.json"
            
            # Convert results to serializable format
            results_dict = {
                'feature_counts': self.results.feature_counts,
                'final_features': self.results.final_features,
                'scores': {
                    'stage_1': self.results.stage_1_scores,
                    'stage_2': self.results.stage_2_scores,
                    'stage_3': self.results.stage_3_scores,
                    'final': self.results.final_scores
                },
                'selection_time': self.results.selection_time,
                'config': {
                    'initial_features': self.config.initial_features,
                    'stage_1_target': self.config.stage_1_target,
                    'stage_2_target': self.config.stage_2_target,
                    'stage_3_target': self.config.stage_3_target,
                    'rf_n_estimators': self.config.rf_n_estimators,
                    'cv_folds': self.config.cv_folds
                }
            }
            
            with open(results_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            self.logger.info(f"💾 Analysis results saved to {results_file}")
            
            # Save final model if requested
            if self.config.save_models and hasattr(self.results, 'model_performance'):
                model_file = output_dir / f"market_analysis_feature_selection_model_{timestamp}.joblib"
                joblib.dump(self.results.model_performance['final_model'], model_file)
                self.logger.info(f"💾 Final model saved to {model_file}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save analysis: {e}")

# Convenience functions
def run_final_feature_selection(X: pd.DataFrame, 
                               y: pd.Series,
                               config: Optional[FeatureSelectionConfig] = None) -> FeatureSelectionResult:
    """Run final feature selection pipeline."""
    selector = MultiStageFeatureSelector(config)
    return selector.select_features(X, y)

def get_final_features(X: pd.DataFrame, 
                      y: pd.Series,
                      target_count: int = 60,
                      config: Optional[FeatureSelectionConfig] = None) -> List[str]:
    """Get final selected features."""
    if config is None:
        config = FeatureSelectionConfig()
        config.stage_3_target = target_count
    
    result = run_final_feature_selection(X, y, config)
    return result.final_features